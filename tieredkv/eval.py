"""Eval runner: one (method, budget, task) cell -> one row in results/results.csv.

Owners: Vishal (runner, profiling columns), Samarth (GSM8K scoring lives in gsm8k.py).

    run("full", 1.0, "gsm8k", n=50)        # -> {acc, mem_gb, tok_s, ms_tok, cfg}

    python -m tieredkv.eval --method full --budget 1.0 --task gsm8k --n 50
    python -m tieredkv.eval --method tiered --budget 0.2 --n 50 --dictionary dicts/llama.pt

Every run uses the same fixed GSM8K subset (gsm8k.load_subset(n, seed)) and the same 5-shot
prompt, greedy decoding, max_new_tokens 256, batch size 1. Per-problem predictions go to
results/runs/<tag>.jsonl so gsm8k.mcnemar can pair any two runs.

lexico and tiered need a dictionary (compress / train_dict format). h2o and tiered get a
ScoreTracker hooked on the model: mode="sum" for h2o (raw H2O), mode="mean" for tiered.
tok_s / ms_tok time decode steps only (profile.DecodeTimer); s_prob is wall time per problem.

Method names (what plot.py groups by): full, lexico, h2o, tiered, and ablation variants of tiered
built from "_"-joined suffixes, which parse_method turns into cache settings:
    tiered_hh3 / tiered_hh10     heavy-hitter tier = top 3% / 10%   (default 5% = plain "tiered")
    tiered_24-8-2                tier ratios (24, 8, 2)             (default 32-12-4)
    tiered_evict                 evict the light tier               (default keep)
    tiered_kv                    tier keys only; values uniform     (K/V-asymmetric)
Suffixes combine, e.g. tiered_hh3_evict.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import uuid
import math
from pathlib import Path

import torch

from . import gsm8k, profile
from .allocator import ScoreTracker
from .cache import CHUNK, METHODS, WINDOW, TieredKVCache
from .budget_audit import audit_rows

__all__ = ["run", "load_model", "parse_method", "COLUMNS"]

DEFAULT_MODEL = os.environ.get("TIEREDKV_MODEL", "unsloth/Llama-3.2-3B-Instruct")  # team model (fits 8 GB)
RESULTS = os.environ.get("TIEREDKV_RESULTS", "results/results.csv")
COLUMNS = ["method", "budget_target", "budget_measured", "acc", "acc_flex", "kv_mb", "peak_gb",
           "tok_s", "ms_tok", "s_prob", "n", "seed", "task", "model", "git_sha",
           "run_id", "eligible"]
MAX_NEW_TOKENS = 256
# Answer length the fixed budget is planned for. Llama-3.2-3B's 5-shot GSM8K answers are short
# (median ~55 tokens, 21-207 over 40 problems); planning for max_new_tokens put short answers
# >1 point over budget on ~640-token prompts. At 64 every answer length stays within 1 point.
EXPECTED_NEW_TOKENS = 64
N_SHOT = 5
TRACKER_MODE = {"h2o": "sum", "tiered": "mean"}

_loaded = {}


def load_model(name=DEFAULT_MODEL, dtype=torch.float16):
    """(model, tokenizer), loaded once per process. Eager attention for every method, so
    timings are comparable and the ScoreTracker can read attention weights."""
    if name not in _loaded:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else None).eval()
        _loaded[name] = (model, tok)
    return _loaded[name]


def parse_method(name):
    """'tiered_hh3_24-8-2' -> ('tiered', {'tiers': (0.03, 0.20), 'ratios': (24, 8, 2)})."""
    base, *suffixes = name.split("_")
    if base not in METHODS:
        raise ValueError(f"unknown method {name!r}: base must be one of {METHODS}")
    if suffixes and base != "tiered":
        raise ValueError(f"{name!r}: only tiered has ablation variants")
    cfg = {}
    for x in suffixes:
        if x.startswith("hh") and x[2:].isdigit():
            cfg["tiers"] = (int(x[2:]) / 100, 0.20)
        elif x.count("-") == 2 and x.replace("-", "").isdigit():
            cfg["ratios"] = tuple(int(r) for r in x.split("-"))
        elif x == "kv":
            cfg["value_policy"] = "uniform"
        elif x == "evict":
            cfg["evict_light"] = True
        else:
            raise ValueError(f"{name!r}: unknown variant {x!r} (hhN, A-B-C, evict, kv)")
    return base, cfg


def make_cache(method, budget, cfg, tracker=None, max_new_tokens=MAX_NEW_TOKENS):
    base, variant = parse_method(method)
    kw = {k: cfg[k] for k in ("tiers", "ratios", "evict_light", "avg_s", "budget_policy", "value_policy")
          if k in cfg}
    return TieredKVCache(base, budget=budget, window=cfg.get("window", WINDOW),
                         chunk=cfg.get("chunk", CHUNK), dictionary_path=cfg.get("dictionary"),
                         tracker=tracker, expected_new_tokens=cfg.get("expected_new_tokens",
                                                     min(EXPECTED_NEW_TOKENS, max_new_tokens)),
                         **{**variant, **kw})


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def generate(model, tok, prompt, cache, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy-decode one prompt through `cache`. Returns (completion text, DecodeTimer, seconds)."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    if ids["input_ids"].shape[-1]+max_new_tokens-1 > model.config.max_position_embeddings:
        raise ValueError("prompt plus generation exceeds model context; shorten settings explicitly")
    timer = profile.DecodeTimer()
    profile.sync()
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                         past_key_values=cache, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, streamer=timer,
                         stop_strings=gsm8k.STOP, tokenizer=tok, pad_token_id=tok.eos_token_id)
    profile.sync()
    dt = time.perf_counter() - t0
    new = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), timer, dt


def append_row(row, path=RESULTS):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {COLUMNS}; "
                             "write to a new file (--results) instead of mixing formats")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run(method: str, budget: float, task: str, n: int, seed: int = 0, *, model_name=DEFAULT_MODEL,
        cfg=None, results=RESULTS, max_new_tokens=MAX_NEW_TOKENS) -> dict:
    """Run one matrix cell; returns {acc, mem_gb, tok_s, ms_tok, cfg} and appends a CSV row."""
    if task != "gsm8k":
        raise NotImplementedError(f"task {task!r}: LongBench loader comes after GSM8K")
    cfg = dict(cfg or {})
    base, _ = parse_method(method)   # fail on a bad name before loading the model
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(budget, bool) or not math.isfinite(budget) or not 0 < budget <= 1:
        raise ValueError("budget must be a fraction in (0,1]")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if cfg.get("score_mode", TRACKER_MODE.get(base)) != TRACKER_MODE.get(base):
        raise ValueError(f"{base} requires score_mode={TRACKER_MODE.get(base)!r}")
    if base in ("lexico", "tiered") and not cfg.get("dictionary"):
        raise ValueError(f"{method} needs a dictionary: cfg['dictionary'] / --dictionary PATH")
    model, tok = load_model(model_name)
    probs = gsm8k.load_subset(n, seed=seed)
    if len(probs) != n:
        raise ValueError("loader returned a different subset size than requested")
    shots = gsm8k.fewshot(N_SHOT)

    profile.reset_peak()
    run_id = uuid.uuid4().hex
    tag = f"{task}_{method}_b{budget:g}_n{n}_s{seed}_{run_id}"
    os.makedirs(os.path.join(os.path.dirname(results) or ".", "runs"), exist_ok=True)
    preds_path = os.path.join(os.path.dirname(results) or ".", "runs", f"{tag}.jsonl")

    tracker = ScoreTracker(model, mode=TRACKER_MODE[base]) if base in TRACKER_MODE else None
    strict = flex = 0
    secs = dec_tokens = dec_secs = 0.0
    kv_bytes, budgets, avg_s, problem_audits = [], [], [], []
    try:
        with open(preds_path, "w") as f:
            for i, p in enumerate(probs):
                cache = make_cache(method, budget, cfg, tracker, max_new_tokens)
                text, timer, dt = generate(model, tok, gsm8k.build_prompt(p["question"], shots),
                                           cache, max_new_tokens)
                r = gsm8k.is_correct(text, p["target"])
                strict += r["strict"]
                flex += r["flexible"]
                secs += dt
                dec_tokens += timer.decode_tokens
                dec_secs += timer.decode_seconds
                kv_bytes.append(cache.kv_bytes())
                budgets.append(cache.measured_budget())
                problem_audits.append(audit_rows([{"method":method,"budget_target":budget,
                                                  "budget_measured":budgets[-1]}])[0])
                problem_audits[-1]["problem_idx"] = p["idx"]
                if cache.avg_s is not None:
                    avg_s.append(cache.avg_s)
                f.write(json.dumps({"idx": p["idx"], "strict": r["strict"], "flexible": r["flexible"],
                                    "pred": r["pred"], "target": p["target"], "text": text,
                                    "budget_measured": budgets[-1], "avg_s": cache.avg_s,
                                    "new_tokens": timer.new_tokens, "kv_bytes": kv_bytes[-1],
                                    "decoded_bytes": cache.decoded_bytes(),
                                    "prompt_sha256":hashlib.sha256(gsm8k.build_prompt(p["question"],shots).encode()).hexdigest(),
                                    "budget_eligible":problem_audits[-1]["eligible"]}) + "\n")
                f.flush()   # an interrupted run keeps every finished problem (compare.py reads them)
                print(f"[{tag}] {i + 1}/{n} strict={strict / (i + 1):.3f} flex={flex / (i + 1):.3f} "
                      f"budget={budgets[-1]:.3f} {dt:.1f}s", flush=True)
    finally:
        if tracker is not None:
            tracker.remove()

    row = {"method": method, "budget_target": budget,
           "budget_measured": round(sum(budgets) / len(budgets), 4),
           "acc": round(strict / n, 4), "acc_flex": round(flex / n, 4),
           "kv_mb": round(sum(kv_bytes) / len(kv_bytes) / 2**20, 2),
           "peak_gb": round(profile.peak_gb(), 3),
           "tok_s": round(dec_tokens / dec_secs, 2) if dec_secs else 0.0,
           "ms_tok": round(1000 * dec_secs / dec_tokens, 2) if dec_tokens else 0.0,
           "s_prob": round(secs / n, 2), "n": n, "seed": seed, "task": task, "model": model_name,
           "git_sha": _git_sha(), "run_id":run_id}
    aggregate_audit = audit_rows([row])[0]
    row["eligible"] = aggregate_audit["eligible"] and all(p["eligible"] for p in problem_audits)
    manifest = {"run_id":run_id,"method":method,"cfg":cfg,"model":model_name,
                "model_revision":getattr(model.config,"_commit_hash",None),
                "dtype":str(next(model.parameters()).dtype),"seed":seed,"n":n,
                "max_new_tokens":max_new_tokens,"score_mode":TRACKER_MODE.get(base),
                "test_indices":[p["idx"] for p in probs],
                "shot_indices":[p["idx"] for p in shots],
                "budget_target":budget,"aggregate_audit":aggregate_audit,
                "problem_audits":problem_audits,"eligible":row["eligible"],
                "git_sha":row["git_sha"],"torch":torch.__version__,
                "dictionary_note":"Dictionary must match this model; path is in cfg.",
                "memory_denominator":"full fp16 KV, not the model dtype",
                "dataset_revision":gsm8k.DATASET_REVISION,
                "effective_cache":{"window":cache.window,"chunk":cache.chunk,
                                   "tiers":cache.tiers,"ratios":cache.ratios,
                                   "evict_light":cache.evict_light,"budget_policy":cache.budget_policy,
                                   "expected_new_tokens":cache.expected_new_tokens,
                                   "cache_decoded":cache.cache_decoded,
                                   "value_policy":cache.value_policy},
                "source_sha256":{name:_file_sha256(Path(__file__).parent/name) for name in
                    ("eval.py","cache.py","allocator.py","compress.py","gsm8k.py","profile.py")},
                "dictionary_sha256":_file_sha256(cfg["dictionary"]) if cfg.get("dictionary") else None,
                "limitations":"Budget audit is not a statistical accuracy or speedup test."}
    manifest_path = os.path.splitext(preds_path)[0]+".manifest.json"
    with open(manifest_path,"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,allow_nan=False)
    if not row["eligible"]:
        print("BUDGET AUDIT FAILED: retained for inspection, excluded from matched-budget comparisons.",flush=True)
    append_row(row, results)
    s_note = f" avg_s={sum(avg_s) / len(avg_s):.1f}" if avg_s else ""
    print(f"[{tag}] acc={row['acc']:.3f} budget={row['budget_measured']:.3f}{s_note} "
          f"{row['s_prob']} s/problem (n for ~20 runs = GPU-seconds / (20 * {row['s_prob']}))",
          flush=True)
    return {"acc": row["acc"], "mem_gb": row["kv_mb"] / 1024, "tok_s": row["tok_s"],
            "ms_tok": row["ms_tok"], "s_prob": row["s_prob"],
            "cfg": {**cfg, **row, "preds": preds_path, "manifest":manifest_path}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON run descriptor (CLI arguments override it)")
    ap.add_argument("--method")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--task")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dictionary", default=None, help="dictionary .pt (lexico / tiered)")
    ap.add_argument("--window", type=int)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--expected-new-tokens", type=int,
                    help=f"answer length the budget is planned for (default {EXPECTED_NEW_TOKENS}, "
                         "capped at --max-new-tokens); the full-cache run's mean new_tokens is a good value")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args()
    descriptor = {}
    if a.config:
        with open(a.config,encoding="utf-8") as f:
            descriptor = json.load(f)
    cfg = dict(descriptor.get("cfg",{}))
    if "score_mode" in descriptor:
        cfg["score_mode"] = descriptor["score_mode"]
    for key in ("dictionary","window","chunk","expected_new_tokens"):
        if getattr(a,key) is not None:
            cfg[key] = getattr(a,key)
    def setting(key, default):
        value = getattr(a,key)
        return descriptor.get(key,default) if value is None else value
    out = run(setting("method","full"), setting("budget",1.0), setting("task","gsm8k"),
              setting("n",50), setting("seed",0), model_name=a.model, cfg=cfg,
              results=a.results, max_new_tokens=a.max_new_tokens)
    print(json.dumps({k: v for k, v in out.items() if k != "cfg"}))
    return 0 if out["cfg"]["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
