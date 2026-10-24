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
Suffixes combine, e.g. tiered_hh3_evict.
"""
import argparse
import csv
import json
import os
import subprocess
import time

import torch

from . import gsm8k, profile
from .allocator import ScoreTracker
from .cache import CHUNK, METHODS, WINDOW, TieredKVCache

__all__ = ["run", "load_model", "parse_method", "COLUMNS"]

DEFAULT_MODEL = os.environ.get("TIEREDKV_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
RESULTS = os.environ.get("TIEREDKV_RESULTS", "results/results.csv")
COLUMNS = ["method", "budget_target", "budget_measured", "acc", "acc_flex", "kv_mb", "peak_gb",
           "tok_s", "ms_tok", "s_prob", "n", "seed", "task", "model", "git_sha"]
MAX_NEW_TOKENS = 256
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
        elif x == "evict":
            cfg["evict_light"] = True
        else:
            raise ValueError(f"{name!r}: unknown variant {x!r} (hhN, A-B-C, evict)")
    return base, cfg


def make_cache(method, budget, cfg, tracker=None, max_new_tokens=MAX_NEW_TOKENS):
    base, variant = parse_method(method)
    kw = {k: cfg[k] for k in ("tiers", "ratios", "evict_light", "avg_s") if k in cfg}
    return TieredKVCache(base, budget=budget, window=cfg.get("window", WINDOW),
                         chunk=cfg.get("chunk", CHUNK), dictionary_path=cfg.get("dictionary"),
                         tracker=tracker, expected_new_tokens=max_new_tokens, **{**variant, **kw})


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


@torch.no_grad()
def generate(model, tok, prompt, cache, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy-decode one prompt through `cache`. Returns (completion text, DecodeTimer, seconds)."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
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
    if base in ("lexico", "tiered") and not cfg.get("dictionary"):
        raise ValueError(f"{method} needs a dictionary: cfg['dictionary'] / --dictionary PATH")
    model, tok = load_model(model_name)
    probs = gsm8k.load_subset(n, seed=seed)
    shots = gsm8k.fewshot(N_SHOT)

    profile.reset_peak()
    tag = f"{task}_{method}_b{budget:g}_n{n}_s{seed}"
    os.makedirs(os.path.join(os.path.dirname(results) or ".", "runs"), exist_ok=True)
    preds_path = os.path.join(os.path.dirname(results) or ".", "runs", f"{tag}.jsonl")

    tracker = ScoreTracker(model, mode=TRACKER_MODE[base]) if base in TRACKER_MODE else None
    strict = flex = 0
    secs = dec_tokens = dec_secs = 0.0
    kv_bytes, budgets, avg_s = [], [], []
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
                if cache.avg_s is not None:
                    avg_s.append(cache.avg_s)
                f.write(json.dumps({"idx": p["idx"], "strict": r["strict"], "flexible": r["flexible"],
                                    "pred": r["pred"], "target": p["target"], "text": text,
                                    "budget_measured": budgets[-1], "avg_s": cache.avg_s}) + "\n")
                print(f"[{tag}] {i + 1}/{n} strict={strict / (i + 1):.3f} "
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
           "git_sha": _git_sha()}
    append_row(row, results)
    s_note = f" avg_s={sum(avg_s) / len(avg_s):.1f}" if avg_s else ""
    print(f"[{tag}] acc={row['acc']:.3f} budget={row['budget_measured']:.3f}{s_note} "
          f"{row['s_prob']} s/problem (n for ~20 runs = GPU-seconds / (20 * {row['s_prob']}))",
          flush=True)
    return {"acc": row["acc"], "mem_gb": row["kv_mb"] / 1024, "tok_s": row["tok_s"],
            "ms_tok": row["ms_tok"], "s_prob": row["s_prob"], "cfg": {**cfg, **row, "preds": preds_path}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="full")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dictionary", default=None, help="dictionary .pt (lexico / tiered)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args()
    cfg = {"dictionary": a.dictionary} if a.dictionary else {}
    out = run(a.method, a.budget, a.task, a.n, a.seed, model_name=a.model, cfg=cfg,
              results=a.results, max_new_tokens=a.max_new_tokens)
    print(json.dumps({k: v for k, v in out.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
