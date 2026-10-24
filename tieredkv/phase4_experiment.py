"""Execute and audit the three Phase-4 HH thresholds on one pinned GPU setup."""
import argparse
import itertools
import json
from pathlib import Path
import re
import shutil

import torch

from . import eval as ev, gsm8k
from .allocator import ScoreTracker
from .budget_audit import audit_rows
from .cache import TieredKVCache
from .threshold_ablation import VARIANTS
from .notebook_support import dictionary_metadata


def generation_gates(model,tok,dictionary,problems,shots):
    """Actual target-model token equality, including a compressed encode path."""
    checks=[]
    kw=dict(max_new_tokens=8,do_sample=False,temperature=None,top_p=None,
            stop_strings=gsm8k.STOP,tokenizer=tok,pad_token_id=tok.eos_token_id)
    with torch.inference_mode():
        for problem in problems[:5]:
            print(f"Generation gates: problem {len(checks)+1}/{min(5,len(problems))}, idx={problem['idx']}",flush=True)
            ids=tok(gsm8k.build_prompt(problem["question"],shots),return_tensors="pt")["input_ids"].to(model.device)
            common={"attention_mask":torch.ones_like(ids),**kw}
            print("  Stock generation",flush=True)
            stock=model.generate(ids,**common)
            print("  Shared full cache",flush=True)
            full=model.generate(ids,past_key_values=TieredKVCache("full"),**common)
            with ScoreTracker(model,mode="sum") as tracker:
                print("  H2O at 100%",flush=True)
                h2o=model.generate(ids,past_key_values=TieredKVCache("h2o",tracker=tracker),**common)
            lex_cache=TieredKVCache("lexico",dictionary_path=dictionary,avg_s=8)
            print("  Uniform compression (real encoder)",flush=True)
            uniform=model.generate(ids,past_key_values=lex_cache,**common)
            with ScoreTracker(model,mode="mean") as tracker:
                print("  Equal-ratio tiered compression (real encoder)",flush=True)
                tier_cache=TieredKVCache("tiered",tracker=tracker,dictionary_path=dictionary,
                                        avg_s=8,ratios=(16,16,16))
                equal=model.generate(ids,past_key_values=tier_cache,**common)
            check={"idx":problem["idx"],"full_equals_stock":torch.equal(stock,full),
                   "h2o100_equals_stock":torch.equal(stock,h2o),
                   "equal_ratio_equals_uniform":torch.equal(uniform,equal),
                   "real_compression_exercised":any(s is not None for s in lex_cache.sparse)
                                               and any(s is not None for s in tier_cache.sparse)}
            checks.append(check)
            if not all(v for k,v in check.items() if k!="idx"):
                raise ValueError(f"generation equivalence gate failed: {check}")
            print("  All generation gates passed",flush=True)
    return checks


def analyze_runs(runs,root=None):
    """Independently re-audit predictions, prompts, metadata and paired accuracy."""
    if len(runs)!=3:
        raise ValueError("exactly three threshold runs are required")
    cells={}; cohort=None; predictions={}
    def artifact(value):
        path=Path(value)
        if root is not None:
            base=Path(root).resolve()
            path=(base/path).resolve()
            if not path.is_relative_to(base):
                raise ValueError("artifact path escapes experiment directory")
        return path
    for result in runs:
        cfg=result["cfg"]
        manifest=json.loads(artifact(cfg["manifest"]).read_text(encoding="utf-8"))
        method=manifest["method"]
        if method not in VARIANTS or method in cells:
            raise ValueError("missing, duplicated or unexpected threshold")
        records=[json.loads(line) for line in artifact(cfg["preds"]).read_text(encoding="utf-8").splitlines()]
        n=manifest["n"]
        indices=[p["idx"] for p in records]
        if n<1 or len(records)!=n or len(set(indices))!=n or indices!=manifest["test_indices"]:
            raise ValueError("incomplete, duplicated or reordered predictions")
        effective=manifest["effective_cache"]
        if (manifest["score_mode"]!="mean" or manifest["dtype"] not in ("torch.float16","torch.bfloat16")
                or effective.get("budget_policy")!="strict" or effective["evict_light"]
                or list(effective["tiers"])!=[VARIANTS[method],.20]):
            raise ValueError("threshold configuration/dtype differs from the Phase-4 protocol")
        identity=(manifest["model"],manifest["model_revision"],manifest["dictionary_sha256"],
                  manifest["dataset_revision"],manifest["seed"],n,manifest["max_new_tokens"],
                  manifest["dtype"],manifest["shot_indices"],manifest["source_sha256"],
                  effective["window"],effective["chunk"],effective["ratios"],
                  [(p["idx"],p["target"],p["prompt_sha256"]) for p in records])
        if cohort is not None and identity!=cohort:
            raise ValueError("runs do not share model/dictionary/data/prompts/source/settings")
        cohort=identity
        target=manifest["budget_target"]
        if target!=.20 or any(type(p["strict"]) is not bool or type(p["flexible"]) is not bool for p in records):
            raise ValueError("expected a 20% budget and boolean correctness records")
        for record in records:
            rescored=gsm8k.is_correct(record["text"],record["target"])
            if any(record[key]!=rescored[key] for key in ("strict","flexible")):
                raise ValueError("saved correctness differs from independently rescored text")
        audits=audit_rows([{"method":method,"budget_target":target,
                            "budget_measured":p["budget_measured"]} for p in records])
        for record,check in zip(records,audits):
            if record["budget_measured"]>target+1e-10:
                check.update(eligible=False,reason="strict ceiling exceeded")
        budgets=[p["budget_measured"] for p in records]
        aggregate=audit_rows([{"method":method,"budget_target":target,
                               "budget_measured":sum(budgets)/n}])[0]
        eligible=(aggregate["eligible"] and all(a["eligible"] for a in audits)
                  and manifest["eligible"] and cfg["eligible"])
        strict=sum(p["strict"] for p in records)
        flexible=sum(p["flexible"] for p in records)
        if abs(cfg["acc"]-round(strict/n,4))>1e-8 or abs(cfg["budget_measured"]-round(sum(budgets)/n,4))>1e-8:
            raise ValueError("CSV/result metrics disagree with raw predictions")
        cells[method]={"n":n,"strict_correct":strict,"flexible_correct":flexible,
                       "accuracy":strict/n,"accuracy_flexible":flexible/n,
                       "wilson95":gsm8k.wilson(strict,n),
                       "mean_budget":sum(budgets)/n,"min_budget":min(budgets),"max_budget":max(budgets),
                       "budget_failures":sum(not a["eligible"] for a in audits),
                       "eligible":bool(eligible),"run_id":manifest["run_id"]}
        predictions[method]=[p["strict"] for p in records]
    complete=all(c["eligible"] for c in cells.values())
    pairs=[]
    if complete:
        for a,b in itertools.combinations(VARIANTS,2):
            paired=gsm8k.mcnemar(predictions[a],predictions[b])
            pairs.append({"a":a,"b":b,**paired,"p_bonferroni":min(1,3*paired["p"])})
    return {"status":"completed" if complete else "budget_audit_failed",
            "phase4_completed":complete,"cells":cells,"paired_mcnemar":pairs,
            "interpretation":"Accuracy comparisons require all three eligible cells. Small n and nonsignificant results must be reported; no automatic superiority claim."}


def run_experiment(args):
    if not re.fullmatch(r"[0-9a-f]{40}",args.model_revision):
        raise ValueError("provide an exact 40-character model commit SHA")
    if not Path(args.model).is_dir():
        raise ValueError("--model must be the local downloaded pinned HF snapshot directory")
    metadata=dictionary_metadata(args.dictionary,args.model_id,args.dictionary_sha256)
    training_revision=metadata["training"].get("model_revision") or metadata["training"].get("revision")
    if training_revision and training_revision!=args.model_revision:
        raise ValueError("model revision differs from dictionary training metadata")
    if not torch.cuda.is_available():
        raise ValueError("target-model experiment requires a CUDA PyTorch GPU runtime")
    if args.n<5 or args.max_new_tokens<8:
        raise ValueError("use at least five problems and eight generated tokens")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)  # never overwrite a previous experiment
    state={"status":"preflight","phase4_completed":False,"model_id":args.model_id,
           "model_revision":args.model_revision,"dictionary":metadata,
           "n":args.n,"seed":args.seed,"max_new_tokens":args.max_new_tokens,"runs":[],
           "runner_sha256":ev._file_sha256(__file__)}
    state["dictionary_training_revision_recorded"]=bool(training_revision)
    def save():
        portable=json.loads(json.dumps(state))
        for run in portable["runs"]:
            for key in ("preds","manifest"):
                run["cfg"][key]=Path(run["cfg"][key]).resolve().relative_to(out.resolve()).as_posix()
        (out/"experiment.json").write_text(json.dumps(portable,indent=2,allow_nan=False),encoding="utf-8")
    save()
    try:
        print(f"Loading pinned model: {args.model_id} @ {args.model_revision}",flush=True)
        model,tok=ev.load_model(args.model)
        if next(model.parameters()).dtype not in (torch.float16,torch.bfloat16) or model.device.type!="cuda":
            raise ValueError("require fp16/bf16 model weights on CUDA")
        print("Checking dictionary shape and values",flush=True)
        D=torch.load(args.dictionary,map_location="cpu",weights_only=True)
        d=getattr(model.config,"head_dim",None) or model.config.hidden_size//model.config.num_attention_heads
        if D.ndim!=3 or D.shape[:2]!=(2*model.config.num_hidden_layers,d) or not 32<=D.shape[-1]<=32768 or not torch.isfinite(D).all():
            raise ValueError("dictionary shape/data is incompatible with the loaded model")
        del D
        print(f"Loading fixed GSM8K subset: n={args.n}, seed={args.seed}",flush=True)
        problems=gsm8k.load_subset(args.n,args.seed)
        shots=gsm8k.fewshot(5)
        state["generation_gates"]=generation_gates(model,tok,args.dictionary,problems,shots)
        state["status"]="running";save()
        descriptors=Path(__file__).resolve().parents[1]/"configs"
        for method,fraction in VARIANTS.items():
            print(f"Starting threshold run: {method} ({fraction:.0%} heavy), 20% KV target",flush=True)
            descriptor=json.loads((descriptors/f"{method}_b20.json").read_text(encoding="utf-8"))
            cfg={**descriptor["cfg"],"dictionary":args.dictionary,"score_mode":"mean"}
            if cfg.get("budget_policy")!="strict" or cfg["tiers"]!=[fraction,.20]:
                raise ValueError("Phase-4 config is not the expected strict threshold")
            result=ev.run(method,.20,"gsm8k",args.n,args.seed,model_name=args.model,
                          cfg=cfg,results=str(out/"results.csv"),max_new_tokens=args.max_new_tokens)
            state["runs"].append(result);save()
        report=analyze_runs(state["runs"])
        state.update(report);save()
        if report["phase4_completed"]:
            import plot
            rows,audits=plot.comparison_rows(plot.load(out/"results.csv"))
            plot.plot_main(rows,str(out))
            plot.plot_ablation(rows,str(out))
            (out/"table.md").write_text(plot.table(rows)+"\n",encoding="utf-8")
        return 0 if report["phase4_completed"] else 1
    except Exception as exc:
        state.update(status="failed",phase4_completed=False,error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        # Outside the output tree: do not zip an archive into itself.
        shutil.make_archive(str(out),"zip",root_dir=out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyze",help="Re-audit an extracted experiment directory without GPU/model")
    p.add_argument("--model")
    p.add_argument("--model-id")
    p.add_argument("--model-revision")
    p.add_argument("--dictionary")
    p.add_argument("--dictionary-sha256")
    p.add_argument("--out")
    p.add_argument("--n",type=int,default=20)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--max-new-tokens",type=int,default=256)
    args=p.parse_args()
    if args.analyze:
        state=json.loads((Path(args.analyze)/"experiment.json").read_text(encoding="utf-8"))
        report=analyze_runs(state["runs"],root=args.analyze)
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0 if report["phase4_completed"] else 1
    if any(getattr(args,key) is None for key in ("model","model_id","model_revision","dictionary","out")):
        p.error("run requires --model, --model-id, --model-revision, --dictionary and --out")
    return run_experiment(args)


if __name__=="__main__":
    raise SystemExit(main())
