"""Heavy-hitter hit rate: do the tokens judged important from the prompt stay important?

TieredKV (and H2O) rank the prompt's tokens ONCE, from prefill attention, when they compress
them. This measures how well that ranking predicts the attention the same tokens actually get
while the answer is generated. Full cache (nothing compressed), greedy, the eval.py prompt.

Per layer, over the prompt tokens that get compressed (all but the last W):
    hh_precision   of the top 5% by prefill score, the share that are also top 5% by decode
                   attention (precision@k; random = 5%)
    hh_coverage    share of decode attention landing on the predicted top 5% (random = 5%)
    tiered_coverage                    ... on the predicted heavy + mid tiers (random = 25%)
    h2o_lost@b     share of decode attention on prompt tokens H2O at budget b evicts at its first
                   eviction (right after prefill). Evicted tokens never return, so this is a lower
                   bound on the attention H2O loses over the whole answer

Prefill score = opportunity-normalised mean attention (ScoreTracker mode="mean", as tiered uses);
H2O's kept set uses the raw sum (mode="sum", as H2O uses). Decode attention = raw attention the
token receives from generated queries only.

    python -m tieredkv.hit_rate --model unsloth/Llama-3.2-3B-Instruct --n 20

Writes results/hit_rate.json. Diagnostic, not accuracy.
"""
import argparse
import json
import os
import statistics as st

import torch

from . import gsm8k
from .allocator import ScoreTracker, h2o_keep_idx
from .cache import WINDOW, TieredKVCache

__all__ = ["layer_hit_rates", "collect", "summarize"]

BUDGETS = (0.10, 0.15, 0.20, 0.25)


def _top(x, frac):
    k = max(1, int(frac * x.numel() + 0.5))
    return torch.argsort(x, descending=True, stable=True)[:k]


def layer_hit_rates(pre_sum, pre_cnt, decode_mass, *, window=WINDOW, tiers=(0.05, 0.20),
                    budgets=BUDGETS, seen=None):
    """One layer. pre_sum / pre_cnt: [T] prefill attention sums and query counts for the prompt's
    T tokens; decode_mass: [T] attention those tokens received from generated queries.
    seen: tokens in context when H2O would first evict (default T, i.e. right after prefill)."""
    T = pre_sum.numel()
    n_old = T - window
    if n_old < 1:
        raise ValueError(f"prompt of {T} tokens has nothing older than the {window}-token window")
    old = slice(0, n_old)
    score = (pre_sum / pre_cnt.clamp(min=1))[old]
    mass = decode_mass[old].double()
    total = mass.sum().item()
    pred_hh, pred_tier = _top(score, tiers[0]), _top(score, tiers[0] + tiers[1])
    actual_hh = _top(mass, tiers[0])
    row = {"old_tokens": n_old, "decode_mass_on_old": total,
           "hh_precision": len(set(pred_hh.tolist()) & set(actual_hh.tolist())) / pred_hh.numel(),
           "hh_coverage": mass[pred_hh].sum().item() / total if total else None,
           "tiered_coverage": mass[pred_tier].sum().item() / total if total else None,
           "random_hh": pred_hh.numel() / n_old, "random_tiered": pred_tier.numel() / n_old}
    for b in budgets:
        keep = h2o_keep_idx(pre_sum.float(), b, window, original_length=seen or T, strict=False)
        lost = torch.ones(T, dtype=torch.bool)
        lost[keep.cpu()] = False
        row[f"h2o_lost@{b:g}"] = (decode_mass.double()[lost].sum().item() / total) if total else None
    return row


@torch.no_grad()
def collect(model, ids, max_new_tokens=192, stop=None):
    """Greedy decode with the full cache. Returns ({layer: (pre_sum, pre_cnt, decode_mass)}, n_new).

    stop(text_so_far) -> bool ends generation early (the eval stop strings)."""
    T = ids.shape[-1]
    with ScoreTracker(model, mode="mean") as tr:
        cache = TieredKVCache("full")
        out = model(ids, past_key_values=cache, use_cache=True)
        pre = {l: (tr.acc[l][0, :T].clone(), tr.cnt[l][0, :T].clone()) for l in tr.acc}
        new = []
        nxt = out.logits[:, -1:].argmax(-1)
        for step in range(max_new_tokens):       # like generate(): the last token is never fed back
            new.append(nxt)
            if step + 1 == max_new_tokens or (stop is not None and stop(torch.cat(new, -1))):
                break
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
        layers = {l: (s, c, (tr.acc[l][0, :T] - s)) for l, (s, c) in pre.items()}
    return layers, len(new)


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("problem", "layer", "old_tokens", "decode_mass_on_old")]
    return {k: st.fmean(r[k] for r in rows if r[k] is not None) for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--out", default="results/hit_rate.json")
    a = p.parse_args()

    from .eval import load_model
    model, tok = load_model(a.model)
    shots = gsm8k.fewshot(5)
    rows, lengths = [], []
    for i, prob in enumerate(gsm8k.load_subset(a.n, seed=a.seed)):
        ids = tok(gsm8k.build_prompt(prob["question"], shots), return_tensors="pt")["input_ids"].to(model.device)
        stop = lambda new: any(s in tok.decode(new[0]) for s in gsm8k.STOP)
        layers, n_new = collect(model, ids, a.max_new_tokens, stop)
        lengths.append(n_new)
        for l, (s, c, m) in layers.items():
            rows.append({"problem": prob["idx"], "layer": l, **layer_hit_rates(s.cpu(), c.cpu(), m.cpu())})
        print(f"hit rate {i + 1}/{a.n}: {ids.shape[-1]} prompt tokens, {n_new} generated", flush=True)

    summary = summarize(rows)
    by_layer = {l: summarize([r for r in rows if r["layer"] == l]) for l in sorted({r["layer"] for r in rows})}
    out = {"model": a.model, "n": a.n, "seed": a.seed, "window": WINDOW, "mean_new_tokens": st.fmean(lengths),
           "summary": summary, "by_layer": by_layer, "rows": rows,
           "note": "Full cache, greedy. Prefill-ranked prompt tokens vs attention from generated tokens."}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmean over {a.n} problems x {len(by_layer)} layers:")
    for k, v in summary.items():
        print(f"  {k:18s} {v:.3f}")


if __name__ == "__main__":
    main()
