"""HH-threshold allocation diagnostic; no compressed generation or accuracy claims.

Runs all three thresholds on the SAME real prefill scores. Projected storage uses
compress.py's fp16/CSR formula, not the float32 model's live memory.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from . import compress as C, gsm8k
from .allocator import ScoreTracker, allocate
from .h2o_reference_eval import load_inputs

VARIANTS = {"tiered_hh3": .03, "tiered": .05, "tiered_hh10": .10}


def analyze_scores(scores, *, budget=.20, window=32):
    """One layer, including recent scores; returns projected rows, never accuracy."""
    if (scores.ndim != 1 or not torch.isfinite(scores).all()
            or (scores < 0).any()):
        raise ValueError("scores must be finite nonnegative [tokens]")
    if not math.isfinite(budget) or not 0 < budget < 1:
        raise ValueError("budget must be a fraction strictly between zero and one")
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    n = scores.numel()
    avg_s = C.solve_s(budget, n, window)
    # Sparse storage is not useful at/above dense cost; reject incompatible shapes.
    if avg_s > 64 or C.bytes_per_token(avg_s) >= C.full_bytes_per_token():
        raise ValueError("uniform atom count exceeds this diagnostic's sparse range")
    old = scores[:n-window] if window else scores
    uniform = torch.full_like(old, avg_s, dtype=torch.long)
    equal = allocate(old, avg_s, ratios=(16, 16, 16), s_max=64)
    if not torch.equal(equal, uniform):
        raise RuntimeError("equal-ratio allocator differs from uniform")
    uniform_bytes = int(C.bytes_per_token(uniform).sum())
    denominator = n*C.full_bytes_per_token()
    uniform_projected = C.budget(uniform, window, n)
    rows = []
    for method, fraction in VARIANTS.items():
        atoms = allocate(old, avg_s, tiers=(fraction, .20), s_max=64)
        order = torch.argsort(old, descending=True, stable=True)
        ranked = atoms[order]
        heavy = int(fraction*old.numel()+.5)
        mid = min(int(.20*old.numel()+.5), old.numel()-heavy)
        actual_bytes = int(C.bytes_per_token(atoms).sum())
        delta = abs(actual_bytes-uniform_bytes)/uniform_bytes
        monotone = bool((ranked[:-1] >= ranked[1:]).all())
        if actual_bytes > uniform_bytes or delta > .005 or not monotone:
            raise RuntimeError("allocation fails matched-byte/monotonicity contract")
        projected = C.budget(atoms, window, n)
        gap = abs(projected-budget)*100
        values, counts = torch.unique(atoms, return_counts=True)
        def mean(a):
            return float(a.float().mean()) if a.numel() else None
        rows.append({
            "method": method, "target_fraction": budget,
            "projected_shared_formula_fraction": projected,
            "projected_uniform_fraction": uniform_projected,
            "target_gap_pp": gap, "within_target_one_pp": gap <= 1+1e-10,
            "old_byte_difference_fraction": delta,
            "uniform_old_bytes": uniform_bytes, "allocated_old_bytes": actual_bytes,
            "tokens": n, "recent_dense_tokens": window, "old_tokens": old.numel(),
            "uniform_s": avg_s, "heavy_tokens": heavy, "mid_tokens": mid,
            "light_tokens": old.numel()-heavy-mid,
            "realized_heavy_fraction": heavy/old.numel(),
            "heavy_mean_s": mean(ranked[:heavy]),
            "mid_mean_s": mean(ranked[heavy:heavy+mid]),
            "light_mean_s": mean(ranked[heavy+mid:]),
            "atom_histogram": {str(int(v)): int(c) for v,c in zip(values,counts)},
            "monotone": monotone, "equal_ratio_is_uniform": True,
            "kind": "allocation_projection_only",
        })
    return rows


def summarize(rows):
    """Keep missing/failed comparisons visible, including integer-s undershoot."""
    if not rows:
        raise ValueError("no allocation records")
    result = {}
    for method in VARIANTS:
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            raise ValueError(f"missing threshold {method}")
        result[method] = {
            "records": len(selected),
            "mean_projected_fraction": sum(r["projected_shared_formula_fraction"]
                                           for r in selected)/len(selected),
            "target_audit_passed": sum(r["within_target_one_pp"] for r in selected),
            "target_audit_flagged": sum(not r["within_target_one_pp"] for r in selected),
            "max_old_byte_difference_fraction": max(r["old_byte_difference_fraction"]
                                                   for r in selected),
            "max_target_gap_pp": max(r["target_gap_pp"] for r in selected),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Existing local HF directory")
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--data", required=True, help="Pinned GSM8K parquet directory")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if min(args.n,args.threads) < 1 or args.window < 0:
        p.error("invalid counts or window")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    problems, shots, dataset = load_inputs(args.data,args.n,args.seed)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation="eager").eval()
    C.configure(model.config)
    inputs, prompt_meta = [], []
    for problem in problems:
        text = gsm8k.build_prompt(problem["question"],shots)
        ids = tok(text,return_tensors="pt")["input_ids"]
        if ids.shape[-1] > model.config.max_position_embeddings:
            p.error("prompt exceeds model context; truncation is not allowed")
        C.solve_s(.20,ids.shape[-1],args.window)  # validate every prompt before scoring
        inputs.append(ids)
        prompt_meta.append({"idx":problem["idx"],"tokens":ids.shape[-1],
                            "sha256":hashlib.sha256(text.encode()).hexdigest()})
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    with torch.inference_mode(), ScoreTracker(model,mode="mean",check=True) as tracker:
        for i,(problem,ids) in enumerate(zip(problems,inputs)):
            tracker.reset()
            # Hook requests attention locally; no cache/generation or saved giant matrices.
            output = model(ids,use_cache=False)
            del output
            if len(tracker.acc) != model.config.num_hidden_layers:
                raise RuntimeError("missing layer scores")
            dump = {"mode":"mean","phase":"prefill_only","problem_idx":problem["idx"],
                    "prompt_sha256":prompt_meta[i]["sha256"],
                    "sums":tracker.acc,"counts":tracker.cnt}
            torch.save(dump,out/f"scores_{problem['idx']}.pt")
            for layer in sorted(tracker.acc):
                records = analyze_scores(tracker.scores(layer)[0],window=args.window)
                for record in records:
                    rows.append({"problem_idx":problem["idx"],"layer":layer,**record})
            print(f"Threshold diagnostic {i+1}/{args.n}: {ids.shape[-1]} tokens",flush=True)
    # No budget_measured/acc columns: these are projected allocations, not run rows.
    with (out/"allocation_diagnostics.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{**r,"atom_histogram":json.dumps(r["atom_histogram"],sort_keys=True)}
                             for r in rows])
    here = Path(__file__).parent
    sources = ("threshold_ablation.py","allocator.py","compress.py","gsm8k.py",
               "h2o_reference_eval.py")
    summary = {
        "kind":"allocation_projection_only", "task_accuracy_evaluated":False,
        "generation_cache_evaluated":False, "n_prompts":args.n,
        "model":args.model_id,"model_revision":args.model_revision,
        "model_dtype":"float32", "storage_projection":"shared fp16/CSR formula",
        "shape":{"kv_heads":C.N_KV_HEADS,"head_dim":C.HEAD_DIM},
        "dataset":dataset,"test_prompts":prompt_meta,
        "train_shot_indices":[s["idx"] for s in shots],
        "seed":args.seed,"window":args.window,"score_mode":"mean",
        "source_sha256":{s:hashlib.sha256((here/s).read_bytes()).hexdigest() for s in sources},
        "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "torch":torch.__version__,"transformers":transformers.__version__,
        "methods":summarize(rows),
        "limitations":[
            "Prefill-only mean scores; not scores from compressed generation.",
            "Projected formula storage, not measured SparseKV payload or process/GPU peak.",
            "Shared formula excludes two final int32 CSR pointers (8 bytes/layer).",
            "Integer uniform s can undershoot 20% by more than one percentage point.",
            "No accuracy, reconstruction-error, speedup or best-threshold conclusion.",
            "Target-model task ablation requires matching dictionary and shared compressed cache.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary["methods"],indent=2))


if __name__ == "__main__":
    main()
