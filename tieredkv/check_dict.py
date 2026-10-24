"""Sanity check a trained dictionary on real KV vectors, before any accuracy runs.

Answers, on 5-shot GSM8K prompts (the same prompts eval.py uses):
  1. error_by_s       does reconstruction error fall as s grows? (K and V separately)
  2. timing           how long encode_kv / decode_kv take per prompt (all layers, K+V)
  3. spearman         are high-attention tokens harder to compress? (plan correction 3)
  4. heavy_vs_light   mean error of the top-5% vs bottom-75% tokens at the same s
  5. weighted_error   the idea in miniature: attention-weighted error of uniform s vs
                      allocate()'s tiered s at the SAME bytes. Lower = more of the
                      attention mass lands on well-reconstructed tokens.

    python -m tieredkv.check_dict --model unsloth/Llama-3.2-3B-Instruct \
        --dict dicts/llama32_3b_N4096_s16.pt --n 5

Writes results/sanity/check_dict.json and error_by_s.png. Diagnostic only, not accuracy.
"""
import argparse
import json
import os
import statistics as st
import time

import torch

from . import compress as C
from .allocator import ScoreTracker, allocate
from .sanity import spearman


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def token_error(kv, D, s):
    """Relative L2 reconstruction error per token (mean over heads). kv [T, h, d]."""
    rec = C.decode_kv(C.encode_kv(kv, D, s), D, torch.float32)
    x = kv.float()
    return ((rec - x).norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-6)).mean(dim=1)


def _mean(xs):
    return st.fmean(xs) if xs else None


@torch.no_grad()
def check(model, prompts, dict_path, s_values=(2, 4, 8, 16, 32), probe_s=8, avg_s=8,
          tiers=(0.05, 0.20), ratios=(32, 12, 4)):
    """prompts: list of [1, T] input_ids. Returns the summary dict."""
    C.configure(model.config)
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    dicts = [C.load_dictionary(dict_path, l, device=dev) for l in range(n_layers)]
    s_values = sorted(set(s_values) | {probe_s, avg_s})
    parts = ("key", "value")
    err = {s: {p: [] for p in parts} for s in s_values}
    rho = {p: [] for p in parts}
    heavy, light = {p: [] for p in parts}, {p: [] for p in parts}
    w_uni, w_tier = {p: [] for p in parts}, {p: [] for p in parts}
    t_enc, t_dec, n_tok = [], [], []

    with ScoreTracker(model, mode="sum") as tr:
        for ids in prompts:
            tr.reset()
            past = model(ids.to(dev), use_cache=True).past_key_values
            enc = dec = 0.0
            for l in range(n_layers):
                mass = tr.acc[l][0]                                   # attention received
                mean = mass / tr.cnt[l][0].clamp(min=1)               # opportunity-normalized importance
                order = mean.argsort(descending=True)
                T = mass.numel()
                top, bottom = order[:max(1, round(tiers[0] * T))], order[round((tiers[0] + tiers[1]) * T):]
                w = mass / mass.sum()
                s_tier = allocate(mean, avg_s, tiers, ratios)
                for p, x, D in zip(parts, past[l], dicts[l]):
                    kv = x[0].transpose(0, 1)                         # [h, T, d] -> [T, h, d]
                    for s in s_values:
                        e = token_error(kv, D, s)
                        err[s][p].append(e.mean().item())
                        if s == probe_s:
                            r = spearman(mean.float().cpu().numpy(), e.cpu().numpy())
                            if r is not None:
                                rho[p].append(r)
                            heavy[p].append(e[top].mean().item())
                            if bottom.numel():
                                light[p].append(e[bottom].mean().item())
                        if s == avg_s:
                            w_uni[p].append((w * e).sum().item())
                    w_tier[p].append((w * token_error(kv, D, s_tier)).sum().item())
                    _sync(); t0 = time.perf_counter()
                    sp = C.encode_kv(kv, D, avg_s)
                    _sync(); t1 = time.perf_counter()
                    C.decode_kv(sp, D)
                    _sync(); t2 = time.perf_counter()
                    enc, dec = enc + t1 - t0, dec + t2 - t1
            t_enc.append(enc)
            t_dec.append(dec)
            n_tok.append(ids.shape[-1])

    out = {
        "n_prompts": len(prompts), "avg_tokens": _mean(n_tok), "probe_s": probe_s, "avg_s": avg_s,
        "error_by_s": {s: {p: _mean(err[s][p]) for p in parts} for s in s_values},
        "timing_s_per_prompt": {"encode": _mean(t_enc), "decode": _mean(t_dec), "at_s": avg_s},
        "spearman_mean_score_vs_error": {
            p: {"median": st.median(rho[p]) if rho[p] else None,
                "frac_positive": _mean([float(r > 0) for r in rho[p]]), "n": len(rho[p])} for p in parts},
        "heavy_vs_light_error": {p: {"top5": _mean(heavy[p]), "bottom75": _mean(light[p])} for p in parts},
        "attention_weighted_error": {p: {"uniform": _mean(w_uni[p]), "tiered": _mean(w_tier[p])} for p in parts},
    }
    return out


def plot_error_by_s(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = sorted(summary["error_by_s"])
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for p, c in (("key", "#2a78d6"), ("value", "#eb6834")):
        y = [summary["error_by_s"][k][p] for k in s]
        ax.plot(s, y, color=c, linewidth=2, marker="o", markersize=8, markeredgecolor="#fcfcfb",
                markeredgewidth=2, label=p.capitalize() + "s")
        ax.annotate(p.capitalize() + "s", (s[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(s, [str(k) for k in s])
    ax.set_xlabel("atoms per vector (s)", color="#52514e")
    ax.set_ylabel("relative reconstruction error", color="#52514e")
    ax.set_title("Dictionary reconstruction error vs sparsity", color="#0b0b0b", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")
    ax.legend(frameon=False)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--n", type=int, default=5, help="GSM8K test prompts (seed 0, 5-shot)")
    p.add_argument("--s", default="2,4,8,16,32")
    p.add_argument("--probe_s", type=int, default=8)
    p.add_argument("--avg_s", type=int, default=8)
    p.add_argument("--out", default="results/sanity")
    a = p.parse_args()

    from . import gsm8k
    from .eval import load_model
    model, tok = load_model(a.model)
    shots = gsm8k.fewshot(5)
    prompts = [tok(gsm8k.build_prompt(q["question"], shots), return_tensors="pt")["input_ids"]
               for q in gsm8k.load_subset(a.n, seed=0)]
    summary = check(model, prompts, a.dict, tuple(int(x) for x in a.s.split(",")), a.probe_s, a.avg_s)
    summary.update(model=a.model, dictionary=a.dict)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "check_dict.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_error_by_s(summary, os.path.join(a.out, "error_by_s.png"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
