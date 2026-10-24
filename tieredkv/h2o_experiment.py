"""Audited target-GPU full/H2O matrix. No compression dictionary is needed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch

from . import eval as ev, gsm8k
from .allocator import ScoreTracker, h2o_keep_idx
from .budget_audit import audit_rows
from .cache import TieredKVCache
from .h2o_baseline import h2o_generate
from .score_diagnostics import summarize_scores

BUDGETS = (.10, .15, .20, .25)


@torch.inference_mode()
def score_probe(model, tok, problems, shots, out):
    """Twenty untruncated target-model prefills, with raw sums and valid counts.

    No generated attention matrices are retained. Mean-mode share is explicitly
    separated from raw attention mass. This is diagnostic, not answer accuracy.
    """
    if len(problems) != 20:
        raise ValueError("score probe requires exactly twenty fixed prompts")
    out = Path(out)
    out.mkdir()
    rows, prompts = [], []
    with ScoreTracker(model, mode="mean", check=True) as tracker:
        for problem in problems:
            tracker.reset()
            prompt = gsm8k.build_prompt(problem["question"], shots)
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
            if ids.shape[-1] > model.config.max_position_embeddings:
                raise ValueError("score probe prompt exceeds model context")
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            del output
            layers = {layer: {"sums": tracker.acc[layer].detach().cpu(),
                              "counts": tracker.cnt[layer].detach().cpu(),
                              "scores": tracker.scores(layer).detach().cpu()}
                      for layer in tracker.acc}
            if len(layers) != model.config.num_hidden_layers:
                raise ValueError("score hooks did not observe every model layer")
            torch.save({"format_version": 2, "mode": "mean", "idx": problem["idx"],
                        "layers": layers}, out/f"prompt_{problem['idx']}.pt")
            prompts.append({"idx": problem["idx"], "input_tokens": ids.shape[-1],
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            for layer in layers:
                rows.append({"idx": problem["idx"], "layer": layer,
                             **summarize_scores(tracker.acc[layer], tracker.cnt[layer])})
            print(f"Target-model attention probe: {len(prompts)}/20", flush=True)
    summary = {"diagnostic_only": True, "scope": "prefill only, no truncation, no generation",
               "prompt_count": 20, "prompts": prompts, "rows": rows,
               "dataset_revision": gsm8k.DATASET_REVISION,
               "model_revision": getattr(model.config, "_commit_hash", None),
               "dtype": str(next(model.parameters()).dtype),
               "source_sha256": {name: ev._file_sha256(Path(__file__).parent/name)
                                 for name in ("allocator.py", "score_diagnostics.py", "h2o_experiment.py")},
               "interpretation": "raw_top_attention_mass_share measures raw mass; mean_top_score_share is a different quantity. Rounded top groups may exceed exactly 5%. This probe does not measure reconstruction error or accuracy."}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"path": "attention/summary.json", "prompt_count": 20, "layer_records": len(rows)}


@torch.inference_mode()
def generation_gates(model, tok, problems, shots):
    """Five target prompts: full=stock; shared H2O=independent reference at all budgets.

    Gates omit stop strings on BOTH paths. Accuracy runs use the shared stopping
    protocol. The reference prunes after attention; the shared cache prunes before
    the next update, so generated tokens (rather than final cache sizes) are compared.
    """
    checks = []
    for problem in problems[:5]:
        print(f"H2O gates: idx={problem['idx']}", flush=True)
        ids = tok(gsm8k.build_prompt(problem["question"], shots), return_tensors="pt")["input_ids"].to(model.device)
        if ids.shape[-1] + 7 > model.config.max_position_embeddings:
            raise ValueError("gate prompt plus generation exceeds model context")
        for budget in BUDGETS:
            h2o_keep_idx(torch.zeros(ids.shape[-1]), budget, 32, strict=True)
        kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=8,
                  do_sample=False, temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
        stock = model.generate(ids, **kw)
        full = model.generate(ids, past_key_values=TieredKVCache("full"), **kw)
        check = {"idx": problem["idx"], "full_equals_stock": torch.equal(stock, full), "h2o": {}}
        for budget in (1.0, *BUDGETS):
            with ScoreTracker(model, mode="sum") as tracker:
                shared = model.generate(ids, past_key_values=TieredKVCache(
                    "h2o", budget=budget, window=32, tracker=tracker), **kw)
            reference = h2o_generate(model, ids, budget=budget, window=32, max_new_tokens=8)
            matched = torch.equal(shared, reference.sequences)
            if budget == 1.0:
                matched = matched and torch.equal(shared, stock)
            check["h2o"][str(budget)] = matched
            print(f"  H2O {budget:.0%}: reference equality={matched}", flush=True)
        checks.append(check)
        if not check["full_equals_stock"] or not all(check["h2o"].values()):
            raise ValueError(f"target-model H2O generation gate failed: {check}")
    return checks


def analyze_runs(runs, root=None):
    """Re-score text and verify the complete, paired, same-protocol five-cell matrix."""
    if len(runs) != 5:
        raise ValueError("full plus exactly four H2O budgets are required")
    cohort = None
    cells, predictions = {}, {}

    def artifact(value):
        path = Path(value)
        if root is not None:
            base = Path(root).resolve()
            path = (base / path).resolve()
            if not path.is_relative_to(base):
                raise ValueError("artifact path escapes experiment directory")
        return path

    for run in runs:
        cfg = run["cfg"]
        manifest = json.loads(artifact(cfg["manifest"]).read_text(encoding="utf-8"))
        method, budget = manifest["method"], manifest["budget_target"]
        key = "full" if method == "full" and budget == 1.0 else f"h2o_b{budget:g}"
        if not ((method == "full" and budget == 1.0) or (method == "h2o" and budget in BUDGETS)) or key in cells:
            raise ValueError("unexpected or duplicate matrix cell")
        records = [json.loads(line) for line in artifact(cfg["preds"]).read_text(encoding="utf-8").splitlines()]
        n = manifest["n"]
        indices = [p["idx"] for p in records]
        if n < 5 or len(records) != n or len(set(indices)) != n or indices != manifest["test_indices"]:
            raise ValueError("incomplete, duplicated or reordered predictions")
        effective = manifest["effective_cache"]
        if (manifest["score_mode"] != ("sum" if method == "h2o" else None)
                or manifest["dtype"] not in ("torch.float16", "torch.bfloat16")
                or effective["window"] != 32 or effective["chunk"] != 8
                or manifest["dataset_revision"] != gsm8k.DATASET_REVISION
                or manifest["dictionary_sha256"] is not None
                or not re.fullmatch(r"[0-9a-f]{40}", manifest["model_revision"] or "")):
            raise ValueError("matrix differs from the pinned H2O protocol")
        identity = (manifest["model"], manifest["model_revision"], manifest["dataset_revision"],
                    manifest["seed"], n, manifest["max_new_tokens"], manifest["dtype"],
                    manifest["shot_indices"], manifest["source_sha256"],
                    [(p["idx"], p["target"], p["prompt_sha256"]) for p in records])
        if cohort is not None and identity != cohort:
            raise ValueError("matrix cells do not share model/data/prompts/source/settings")
        cohort = identity
        for record in records:
            rescored = gsm8k.is_correct(record["text"], record["target"])
            if any(type(record[k]) is not bool or record[k] != rescored[k] for k in ("strict", "flexible")):
                raise ValueError("saved correctness differs from independently rescored text")
        audits = audit_rows([{"method": method, "budget_target": budget,
                              "budget_measured": p["budget_measured"]} for p in records])
        budgets = [p["budget_measured"] for p in records]
        mean_budget = sum(budgets) / n
        aggregate = audit_rows([{"method": method, "budget_target": budget, "budget_measured": mean_budget}])[0]
        strict = sum(p["strict"] for p in records)
        flexible = sum(p["flexible"] for p in records)
        if (cfg["method"] != method or cfg["budget_target"] != budget
                or abs(cfg["acc"] - round(strict/n, 4)) > 1e-8
                or abs(cfg["acc_flex"] - round(flexible/n, 4)) > 1e-8
                or abs(cfg["budget_measured"] - round(mean_budget, 4)) > 1e-8):
            raise ValueError("result metrics disagree with raw predictions")
        cells[key] = {"method": method, "budget_target": budget, "n": n,
                      "strict_correct": strict, "flexible_correct": flexible,
                      "accuracy": strict/n, "accuracy_flexible": flexible/n,
                      "wilson95": gsm8k.wilson(strict, n), "mean_budget": mean_budget,
                      "min_budget": min(budgets), "max_budget": max(budgets),
                      "budget_failures": sum(not a["eligible"] for a in audits),
                      "eligible": bool(aggregate["eligible"] and all(a["eligible"] for a in audits)
                                       and manifest["eligible"] and cfg["eligible"]),
                      "run_id": manifest["run_id"]}
        predictions[key] = [p["strict"] for p in records]
    complete = all(c["eligible"] for c in cells.values())
    pairs = []
    if complete:
        for budget in BUDGETS:
            key = f"h2o_b{budget:g}"
            pair = gsm8k.mcnemar(predictions["full"], predictions[key])
            pairs.append({"a": "full", "b": key, **pair, "p_bonferroni": min(1, 4*pair["p"])})
    return {"status": "completed" if complete else "budget_audit_failed",
            "h2o_completed": complete, "cells": cells, "paired_mcnemar": pairs,
            "interpretation": "H2O uses raw summed attention. Budgets are final retained KV relative to full fp16 KV, within one percentage point. Full prefill, attention, metadata and model weights are not covered by this budget. This is not a peak-memory or speedup claim."}


def run_experiment(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("provide an exact 40-character model commit SHA")
    if not Path(args.model).is_dir():
        raise ValueError("--model must be a local pinned snapshot directory")
    if not torch.cuda.is_available():
        raise ValueError("target-model experiment requires a CUDA PyTorch GPU runtime")
    if args.n < 5 or args.max_new_tokens < 8:
        raise ValueError("use at least five problems and eight generated tokens")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "preflight", "h2o_completed": False, "model_id": args.model_id,
             "model_revision": args.model_revision, "n": args.n, "seed": args.seed,
             "max_new_tokens": args.max_new_tokens, "runs": [],
             "runner_sha256": ev._file_sha256(__file__)}

    def save():
        portable = json.loads(json.dumps(state))
        for run in portable["runs"]:
            for key in ("preds", "manifest"):
                run["cfg"][key] = Path(run["cfg"][key]).resolve().relative_to(out.resolve()).as_posix()
        (out/"experiment.json").write_text(json.dumps(portable, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        print(f"Loading pinned model: {args.model_id} @ {args.model_revision}", flush=True)
        model, tok = ev.load_model(args.model)
        if next(model.parameters()).dtype not in (torch.float16, torch.bfloat16) or model.device.type != "cuda":
            raise ValueError("require fp16/bf16 model weights on CUDA")
        if getattr(model.config, "_commit_hash", None) != args.model_revision:
            raise ValueError("loaded snapshot revision differs from requested model commit")
        problems, shots = gsm8k.load_subset(args.n, args.seed), gsm8k.fewshot(5)
        state["status"] = "attention_probe"
        save()
        state["attention_probe"] = score_probe(model, tok, gsm8k.load_subset(20, args.seed), shots, out/"attention")
        save()
        state["generation_gates"] = generation_gates(model, tok, problems, shots)
        state["status"] = "running"
        save()
        for method, budget in [("full", 1.0), *[("h2o", b) for b in BUDGETS]]:
            print(f"Starting {method} at {budget:.0%} retained-KV target", flush=True)
            result = ev.run(method, budget, "gsm8k", args.n, args.seed, model_name=args.model,
                            cfg={"window": 32, "chunk": 8}, results=str(out/"results.csv"),
                            max_new_tokens=args.max_new_tokens)
            state["runs"].append(result)
            save()
        report = analyze_runs(state["runs"])
        state.update(report)
        save()
        import plot
        rows, _ = plot.comparison_rows(plot.load(out/"results.csv"))
        if rows:
            plot.plot_main(rows, str(out))
            (out/"table.md").write_text(plot.table(rows)+"\n", encoding="utf-8")
        return 0 if report["h2o_completed"] else 1
    except Exception as exc:
        state.update(status="failed", h2o_completed=False, error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        shutil.make_archive(str(out), "zip", root_dir=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyze", help="Re-audit an extracted bundle without GPU/model")
    for key in ("model", "model-id", "model-revision", "out"):
        p.add_argument("--"+key)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    if args.analyze:
        state = json.loads((Path(args.analyze)/"experiment.json").read_text(encoding="utf-8"))
        report = analyze_runs(state["runs"], root=args.analyze)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["h2o_completed"] else 1
    if any(getattr(args, key) is None for key in ("model", "model_id", "model_revision", "out")):
        p.error("run requires --model, --model-id, --model-revision and --out")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
