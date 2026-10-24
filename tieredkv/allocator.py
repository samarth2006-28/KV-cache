"""TieredKV allocator: H2O attention scores -> per-token dictionary-atom counts.

Owner: Sampreeth. Contract (see plan section 7):
    h2o_scores(attn)                     [b, H, q, k] -> [b, k]
    ScoreTracker(model)                  hooks every model.model.layers[i].self_attn
        .scores(layer, idx) / .keep(layer, idx) / .reset()
    allocate(scores, avg_s, ...)         -> LongTensor s_per_token (0 = evicted), byte-matched
    h2o_keep_idx(scores, budget, W)      -> LongTensor of kept cache indices (sorted)

Works with transformers < 4.48 (weights only returned when output_attentions=True;
a pre-hook forces it per attention module) and >= 4.48 (eager always returns weights).
The model MUST be loaded with attn_implementation="eager".
"""
from functools import partial
import math
import warnings

import torch
import torch.nn.functional as F

__all__ = ["h2o_scores", "ScoreTracker", "allocate", "h2o_keep_idx", "check_attn"]


# --------------------------------------------------------------------------- scores

def h2o_scores(attn):
    """attn [b, n_heads, q, k] -> [b, k]: sum over queries, mean over heads (fp32)."""
    if attn.dim() != 4 or any(d == 0 for d in attn.shape[:3]):
        raise ValueError("attention must be [batch, heads, queries, keys] with nonempty first axes")
    return attn.detach().sum(dim=2, dtype=torch.float32).mean(dim=1)


def check_attn(attn, atol=1e-2):
    """Assert attn is [b, heads, q, k], non-negative, rows summing to 1."""
    assert attn.dim() == 4, f"expected [b, heads, q, k], got {tuple(attn.shape)}"
    rows = attn.sum(dim=-1, dtype=torch.float32)
    err = (rows - 1).abs().max().item()
    assert err < atol, f"attention rows don't sum to 1 (max err {err:.3g}); hooked pre-softmax?"
    assert attn.min().item() >= 0, "negative attention weights; hooked pre-softmax?"


def _force_attn_weights(module, args, kwargs):
    # transformers < 4.48: LlamaAttention drops weights unless output_attentions=True.
    # Forcing it here (not on model.generate) avoids generate() stacking every step's attentions.
    kwargs["output_attentions"] = True
    return args, kwargs


def _take(t, idx):
    if torch.is_tensor(idx) and idx.dim() == 2:
        return t.gather(1, idx.to(t.device))
    return t[:, idx.to(t.device) if torch.is_tensor(idx) else idx]


class ScoreTracker:
    """Accumulates H2O scores per layer: acc[layer] is [b, k_len], aligned with the KV cache.

    mode="sum":  raw accumulated attention (faithful H2O; use for the H2O baseline).
    mode="mean": accumulated attention / number of queries that could see the key.
                 Corrects unequal causal observation counts (key j at prefill is
                 seen by N-j queries). Does not guarantee position-independent scores.

    Automatic hooks assume unpadded causal decoder attention. For padding, sliding
    windows or custom visibility, call add(..., valid_mask=mask) explicitly, with
    the hook disabled. True means a query can see a key, including zero-weight
    attention entries. Raw numerators remain in acc, opportunity counts in cnt.
    """

    def __init__(self, model, mode="sum", check=False):
        if mode not in ("sum", "mean"):
            raise ValueError(f"mode must be 'sum' or 'mean', got {mode!r}")
        self.mode, self.check = mode, check
        self.enabled = True
        self.acc, self.cnt = {}, {}
        self._handles = []
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            self._handles.append(attn.register_forward_pre_hook(_force_attn_weights, with_kwargs=True))
            self._handles.append(attn.register_forward_hook(partial(self._hook, i)))

    def _hook(self, layer, module, args, output):
        if not self.enabled:
            return
        w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if w is None:
            raise RuntimeError(
                f"layer {layer}: self_attn returned no attention weights. "
                "Load the model with attn_implementation='eager'.")
        with torch.no_grad():
            self.add(layer, w.detach())

    @torch.no_grad()
    def add(self, layer, attn, valid_mask=None):
        """Accumulate one attention call [b, H, q, k] (called by the hook; public for tests)."""
        if self.check:
            check_attn(attn)
        b, _, q, k = attn.shape
        if q > k:
            raise ValueError("decoder queries must be a suffix of the physical cache")
        s = h2o_scores(attn)
        if valid_mask is None:
            counts = (k - torch.arange(k, device=s.device)).clamp(max=q).to(s.dtype)
        else:
            mask = torch.as_tensor(valid_mask, device=attn.device)
            if mask.dtype != torch.bool:
                raise ValueError("valid_mask must be boolean; True means visible")
            if mask.dim() == 3:
                mask = mask[:, None]
            if (mask.dim() != 4 or mask.shape[0] != b or
                    mask.shape[1] not in (1, attn.shape[1]) or tuple(mask.shape[2:]) != (q, k)):
                raise ValueError("valid_mask must be [b,q,k], [b,1,q,k] or [b,heads,q,k]")
            s = h2o_scores(attn * mask)
            counts = h2o_scores(mask.float())
        acc = self.acc.get(layer, s.new_zeros(b, 0))
        cnt = self.cnt.get(layer, s.new_zeros(b, 0))
        if acc.shape[0] != b or acc.device != s.device:
            raise ValueError("batch or device changed; reset the tracker for a new run")
        n = acc.shape[1]
        if n > k:
            raise RuntimeError(
                f"layer {layer}: tracker holds {n} keys but attention saw {k}; "
                "call tracker.keep(layer, idx) whenever the cache evicts")
        if n < k:
            acc, cnt = F.pad(acc, (0, k - n)), F.pad(cnt, (0, k - n))
        # Queries in this call that can see key j: the last q of k positions are queries,
        # so key j is visible to min(q, k - j). Decode (q=1) -> 1 for every key, which stays
        # correct after eviction since every cached key precedes the new query.
        self.acc[layer] = acc + s
        self.cnt[layer] = cnt + counts

    def scores(self, layer, idx=None):
        """[b, len(idx)] scores for cache positions idx (all positions if idx is None)."""
        acc = self.acc[layer]
        if self.mode == "mean":
            count = self.cnt[layer]
            acc = acc / count.masked_fill(count == 0, 1)
        return acc.clone() if idx is None else _take(acc, idx)

    def keep(self, layer, idx):
        """Re-align after eviction: keep only cache positions idx (in their new order)."""
        self.acc[layer] = _take(self.acc[layer], idx)
        self.cnt[layer] = _take(self.cnt[layer], idx)

    def reset(self):
        self.acc.clear()
        self.cnt.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# --------------------------------------------------------------------------- allocation

def _default_bytes_fn():
    try:
        from .compress import bytes_per_token
        return bytes_per_token
    except ImportError:
        # Counting atoms == counting bytes whenever bytes_per_token is linear in s.
        return lambda s: s


def _bytes(fn, s):
    """Per-token bytes for integer tensor s; s == 0 (evicted) always costs 0."""
    try:
        b = torch.as_tensor(fn(s), dtype=torch.float64)
        if b.shape != s.shape:
            raise TypeError
    except Exception:
        b = torch.tensor([float(fn(int(x))) for x in s.tolist()], dtype=torch.float64)
    return torch.where(s > 0, b, torch.zeros_like(b))


def allocate(scores, avg_s, tiers=(0.05, 0.20), ratios=(32, 12, 4), evict_light=False,
             *, bytes_fn=None, s_min=1, s_max=64):
    """Per-token atom counts from importance scores, at the byte budget of uniform avg_s.

    scores: [k] or [b, k]. Tokens are ranked by score; the top tiers[0] fraction is heavy,
    the next tiers[1] mid, the rest light. ratios are RELATIVE atom counts per tier: they are
    rescaled by one factor alpha so that sum(bytes_fn(s_i)) == k * bytes_fn(avg_s)
    (to within one atom). evict_light gives light tokens s=0 and hands their bytes to the rest.
    s_max caps atoms per token (set it to where sparse costs as much as dense, or the dict size).

    Returns LongTensor shaped like scores. Guarantees: s is non-increasing in score;
    equal ratios + integer avg_s -> every token gets exactly avg_s.
    """
    if not math.isfinite(avg_s) or avg_s <= 0:
        raise ValueError("avg_s must be finite and positive")
    if (len(tiers) != 2 or any(not math.isfinite(x) or x < 0 for x in tiers)
            or sum(tiers) > 1):
        raise ValueError("tiers must be two nonnegative fractions with sum <= 1")
    if len(ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("ratios must contain three finite positive values")
    if (isinstance(s_min, bool) or not isinstance(s_min, int) or s_min < 1 or
            (s_max is not None and (isinstance(s_max, bool) or
             not isinstance(s_max, int) or s_max < s_min))):
        raise ValueError("atom bounds must be positive integers with s_max >= s_min")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if scores.dim() == 2:
        if scores.shape[0] == 0:
            return torch.zeros_like(scores, dtype=torch.long)
        return torch.stack([allocate(r, avg_s, tiers, ratios, evict_light,
                                     bytes_fn=bytes_fn, s_min=s_min, s_max=s_max) for r in scores])
    if scores.dim() != 1:
        raise ValueError(f"scores must be [k] or [b, k], got {tuple(scores.shape)}")
    n = scores.numel()
    out = torch.zeros(n, dtype=torch.long, device=scores.device)
    if n == 0:
        return out
    fn = bytes_fn or _default_bytes_fn()
    cap = float("inf") if s_max is None else float(s_max)

    order = torch.sort(scores.detach().float().cpu(), descending=True, stable=True).indices
    n_h = int(tiers[0] * n + 0.5)
    n_m = min(int(tiers[1] * n + 0.5), n - n_h)
    r = torch.full((n,), float(ratios[2]), dtype=torch.float64)  # in score order
    r[:n_h], r[n_h:n_h + n_m] = float(ratios[0]), float(ratios[1])
    if evict_light:
        r[n_h + n_m:] = 0.0
    live = r > 0
    target = n * float(fn(avg_s))

    def s_at(alpha):
        s = torch.floor(alpha * r).clamp(s_min, cap)
        return torch.where(live, s, torch.zeros_like(s)).long()

    def cost(s):
        return _bytes(fn, s).sum().item()

    if not live.any():
        warnings.warn("allocate: every token evicted (no heavy/mid tokens)")
        return out
    if cost(s_at(0.0)) > target:
        raise ValueError(f"avg_s={avg_s} too small: all kept tokens at s_min={s_min} exceed the budget")

    # Largest alpha with cost <= target (cost is monotone in alpha).
    lo, hi = 0.0, 1.0
    while cost(s_at(hi)) <= target:
        if (s_at(hi)[live] >= cap).all() or hi > 1e9:
            warnings.warn("allocate: s_max reached before the budget; allocation is under budget")
            break
        lo, hi = hi, hi * 2
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cost(s_at(mid)) <= target else (lo, mid)
    s = s_at(lo)

    # Spend the leftover (< one atom per token) on the highest-scored tokens: keeps
    # monotonicity and makes equal ratios exactly uniform.
    elig = live & (s < cap)
    delta = (_bytes(fn, s + 1) - _bytes(fn, s)) * elig
    s = s + (elig & (torch.cumsum(delta, 0) <= target - cost(s) + 1e-9)).long()

    out[order.to(out.device)] = s.to(out.device)
    return out


def h2o_keep_idx(scores, budget, W, *, original_length=None, strict=False):
    """H2O baseline: indices to KEEP (ascending), = last W tokens + top heavy hitters.

    scores: [k] or [b, k]. budget: number of tokens to keep (int, includes the W recent), or a
    fraction of original_length (defaults to k) if float <= 1. Pass the number
    of tokens seen before eviction as original_length on repeated calls, rather
    than applying the fraction to an already-pruned cache. Fractional budgets
    round DOWN. strict=True rejects a budget smaller than the recent window;
    legacy callers keep that window and receive an overspending warning.
    """
    if scores.dim() == 2:
        return torch.stack([h2o_keep_idx(r, budget, W, original_length=original_length,
                                         strict=strict) for r in scores])
    if scores.dim() != 1 or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite and shaped [k] or [batch,k]")
    if isinstance(W, bool) or not isinstance(W, int) or W < 0:
        raise ValueError("W must be a nonnegative integer")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a nonnegative finite token count or fraction")
    k = scores.numel()
    original_length = k if original_length is None else original_length
    if isinstance(original_length, bool) or not isinstance(original_length, int) or original_length < k:
        raise ValueError("original_length must be an integer >= physical cache length")
    n_keep = math.floor(budget * original_length + 1e-12) if isinstance(budget, float) and budget <= 1.0 else int(budget)
    w = min(W, k)
    if n_keep < w:
        if strict:
            raise ValueError("budget cannot hold the protected recent window")
        warnings.warn("H2O recent window exceeds the requested budget; use strict=True for matched-budget runs")
    n_keep = min(k, max(n_keep, w))
    dev = scores.device
    recent = torch.arange(k - w, k, device=dev)
    n_hh = n_keep - w
    if n_hh <= 0:
        return recent
    hh = torch.argsort(scores[: k - w].float(), descending=True, stable=True)[:n_hh].sort().values
    return torch.cat([hh.to(dev), recent])
