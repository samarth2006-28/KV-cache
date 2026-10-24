"""Small, CPU-only full/H2O reference smoke matrix; never writes shared results.csv.

Requires a local Llama-family model and pinned GSM8K train/test parquet files.
This is independent of the unfinished shared compressed-generation cache.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch

from . import gsm8k
from .budget_audit import audit_rows
from .h2o_baseline import h2o_generate


def fixed_subset(records, n, seed):
    """Same shuffled-then-sorted index rule as gsm8k.load_subset."""
    if not 0 < n <= len(records):
        raise ValueError("subset size must be positive and fit the selected split")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    return [{"idx": i, **records[i], "target": gsm8k.target_of(records[i]["answer"])}
            for i in sorted(order[:n])]


def load_inputs(root, n, seed):
    import pyarrow.parquet as pq
    root = Path(root)
    provenance = json.loads((root / "source.json").read_text())
    splits = {}
    for name in ("train", "test"):
        path = root / f"{name}.parquet"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != provenance["files"][name]["sha256"]:
            raise ValueError(f"{name} parquet does not match its provenance hash")
        splits[name] = pq.read_table(path).to_pylist()
        if len(splits[name]) != provenance["files"][name]["rows"]:
            raise ValueError(f"{name} row count mismatch")
    return fixed_subset(splits["test"], n, seed), fixed_subset(splits["train"], 5, 1234), provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--budgets", nargs="+", type=float, default=[.10,.15,.20,.25])
    args = p.parse_args()
    if min(args.n,args.threads,args.max_new_tokens) < 1 or args.window < 0 or any(not 0 < b < 1 for b in args.budgets):
        p.error("invalid counts, window or fractional H2O budgets")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    problems, shots, dataset = load_inputs(args.data,args.n,args.seed)
    tok = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation="eager").eval()
    prompts = [gsm8k.build_prompt(problem["question"],shots) for problem in problems]
    inputs = [tok(prompt,return_tensors="pt")["input_ids"] for prompt in prompts]
    # Fail all impossible configurations before launching the matrix.
    from .allocator import h2o_keep_idx
    for ids in inputs:
        for b in args.budgets:
            h2o_keep_idx(torch.zeros(ids.shape[-1]),b,args.window,strict=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows, records, stock_checks = [], [], []
    for budget in [1.0,*args.budgets]:
        method="full_reference" if budget==1 else "h2o_reference"
        strict=flexible=0
        finals=[];times=[]
        for i,(problem,ids) in enumerate(zip(problems,inputs)):
            result=h2o_generate(model,ids,budget=budget,window=args.window,
                                max_new_tokens=args.max_new_tokens)
            if budget==1:
                with torch.inference_mode():
                    stock=model.generate(ids,attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_new_tokens,do_sample=False,
                        pad_token_id=tok.eos_token_id,temperature=None,top_p=None)
                matched=torch.equal(stock,result.sequences)
                stock_checks.append({"idx":problem["idx"],"token_identical":matched})
                if not matched:
                    raise RuntimeError(f"100% H2O reference differs from stock generation on problem {problem['idx']}")
            generated=tok.decode(result.sequences[0,ids.shape[-1]:],skip_special_tokens=True)
            scored=gsm8k.is_correct(generated,problem["target"])
            strict+=int(scored["strict"]);flexible+=int(scored["flexible"])
            final=result.samples[-1]
            if any(sample["kv_fraction"]>budget+1e-12 for sample in result.samples):
                raise RuntimeError("retained KV exceeded the byte ceiling")
            finals.append(final);times.append(result.seconds)
            records.append({"method":method,"budget_target":budget,"idx":problem["idx"],
                "prompt_sha256":hashlib.sha256(prompts[i].encode()).hexdigest(),
                "input_tokens":ids.shape[-1],"new_tokens":result.sequences.shape[-1]-ids.shape[-1],
                "text":generated,"target":problem["target"],**scored,"final":final,
                "max_retained_kv_fraction":max(s["kv_fraction"] for s in result.samples),
                "peak_materialized_kv_bytes":result.peak_materialized_kv_bytes,
                "seconds":result.seconds})
            print(f"{method} budget={budget:.0%} problem={i+1}/{args.n} "
                  f"retained_KV={final['kv_fraction']:.3%}",flush=True)
        rows.append({"method":method,"budget_target":budget,
            "budget_measured":sum(f["kv_fraction"] for f in finals)/args.n,
            "acc":strict/args.n,"acc_flex":flexible/args.n,"n":args.n,"seed":args.seed,
            "window":args.window,"mean_final_kv_bytes":sum(f["kv_bytes"] for f in finals)/args.n,
            "mean_final_auxiliary_bytes":sum(f["auxiliary_bytes"] for f in finals)/args.n,
            "mean_reference_seconds":sum(times)/args.n,"dtype":"float32","reference_only":True})
    audit = audit_rows(rows)
    per_problem_audit = audit_rows([{"method":r["method"],"budget_target":r["budget_target"],
                                   "budget_measured":r["final"]["kv_fraction"]} for r in records])
    if not all(r["eligible"] for r in [*audit,*per_problem_audit]):
        raise RuntimeError("reference matrix contains unmatched retained-KV budgets")
    try:
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):git_sha=None
    source_dir=Path(__file__).parent
    metadata={"reference_only":True,"model":args.model_id,"model_revision":args.model_revision,
        "dataset":dataset,"test_indices":[p["idx"] for p in problems],
        "train_shot_indices":[s["idx"] for s in shots],"n":args.n,"seed":args.seed,
        "window":args.window,"max_new_tokens":args.max_new_tokens,"threads":args.threads,
        "dtype":"float32","device":"cpu","torch":torch.__version__,"transformers":transformers.__version__,
        "git_sha":git_sha,"source_sha256":{name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
            for name in ["h2o_baseline.py","h2o_reference_eval.py","allocator.py","gsm8k.py"]},
        "full_budget_stock_checks":stock_checks,"rows":rows,"budget_audit":audit,
        "per_problem_budget_audit":per_problem_audit,
        "limitations":["Small-model, five-problem smoke matrix; not the team's target benchmark.",
            "Full uncompressed prefill is materialized. Retained KV is not peak GPU/process memory.",
            "KV budgets use the same float32 dtype in numerator/denominator; shared fp16/CSR results are not comparable.",
            "Score/count/position metadata is reported separately from KV payload.",
            "Greedy reference stops on EOS/token cap; scoring reuses shared stop-string truncation.",
            "No claim about TieredKV versus Lexico, throughput gains or statistical superiority."]}
    with (out/"reference_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
    (out/"reference_summary.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Saved {len(rows)} reference cells; all stock-equivalence and retained-KV budget checks passed.")


if __name__ == "__main__":main()
