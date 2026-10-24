"""TieredKVCache: the cache generate() talks to every step.

Owner: Vishal. Contract (see plan section 7):
    TieredKVCache(method, ...)   method in {full, lexico, h2o, tiered}
        .update(k, v, layer_idx, cache_kwargs) -> (K, V) for attention   [b, kv_heads, seq, d]
        .get_seq_length(layer_idx)             tokens seen (positions), not tokens stored
        .kv_bytes() / .measured_budget()       what the cache actually holds vs full fp16

Per layer the cache stores `sparse` (compressed old tokens, original order) + `recent`
(dense, last tokens). At the START of update(), before appending the new token: if recent
reaches W + C tokens, everything but the last W is compressed before appending (prefill: the whole
prompt at once; decode: C tokens every C steps). Starting there guarantees the tracker has
scores for every stored token, since its hook writes them after attention.

    full     store everything dense (== DynamicCache)
    lexico   every compressed token gets floor(avg_s) or floor(avg_s) + 1 (uniform)
    tiered   s per token from allocator.allocate on H2O scores (same bytes as lexico)
    h2o      nothing compressed; keep recent W + top-score tokens, evict the rest

Attention gets cat(decode_kv(sparse), recent). With cache_decoded (default) each chunk is decoded
once when it is compressed and the dense copy is reused every step: decoding everything per step
costs ~25 s per GSM8K problem on 3B (Samarth's check_dict timing; on GPU lexico ran 17x and tiered
4x faster with identical outputs). kv_bytes() counts only the compressed storage, as Lexico
reports; decoded_bytes() reports the decoded copy separately (a fused sparse-attention kernel
would remove it). allocate_fn / encode_fn / decode_fn default to
the real allocator.allocate / compress.encode_kv / compress.decode_kv and can be swapped for
stubs: with a constant allocator and an identity encoder, output must equal method="full".

Pinned to transformers==4.49.0 (same as Lexico). Batch size 1 for every method but full.

    cache = TieredKVCache("full")
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=256, do_sample=False)
"""
from typing import Any, Dict, List, Optional, Tuple
import math
from pathlib import Path
import warnings

import torch
from transformers.cache_utils import Cache

from . import allocator, compress

__all__ = ["TieredKVCache", "METHODS", "WINDOW", "CHUNK"]

METHODS = ("full", "lexico", "h2o", "tiered")

# Dense recent window W, one constant for every method. 128 (Lexico's default) makes 10% and
# 15% infeasible on ~1000-token 5-shot GSM8K prompts: the window alone is ~13%. At W=32,
# compress.solve_s gives s = 4 / 9 / 13 / 17 at 10 / 15 / 20 / 25% for N=1000.
WINDOW = 32
# Tokens compressed together during decode. Recent swings between W and W + C, so C adds up to
# C/N to the measured budget; 8 keeps that under 1 point (the audit tolerance) at N ~ 1000.
CHUNK = 8

_DICTS = {}


def _load_dictionary(path):
    """Whole Lexico dictionary file [2L, d, N], loaded once per process (caches are per problem)."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    if key not in _DICTS:
        # A dictionary may be retrained at the same path in an interactive run.
        for stale in [k for k in _DICTS if k[0] == str(resolved)]:
            del _DICTS[stale]
        _DICTS[key] = torch.load(resolved, map_location="cpu", weights_only=True)
    return _DICTS[key]


def _nbytes(x):
    return x.nbytes   # SparseKV.nbytes, or tensor.nbytes for stub encoders


class TieredKVCache(Cache):
    def __init__(self, method: str = "full", *, window: int = WINDOW, chunk: int = CHUNK,
                 budget: float = 1.0, tracker=None, dictionary=None,
                 dictionary_path: Optional[str] = None, avg_s: Optional[int] = None,
                 expected_new_tokens: int = 256, tiers=(0.05, 0.20), ratios=(32, 12, 4),
                 evict_light: bool = False, cache_decoded: bool = True,
                 allocate_fn=None, encode_fn=None, decode_fn=None, budget_policy: str = "fixed"):
        """window W: dense recent tokens (shared by all methods). chunk C: decode tokens
        compressed at once. budget: target memory fraction of full fp16 KV.
        tracker: allocator.ScoreTracker on the same model (mode="mean" for tiered, "sum" for h2o).
        dictionary: [2L, d, N] tensor (D[2l] keys, D[2l+1] values), or dictionary_path.
        avg_s: atoms per compressed token for lexico (mean for tiered); None -> solved from
        budget at the first compression, with N = prompt + expected_new_tokens.
        tiers / ratios / evict_light: passed to allocate (tiered only).
        cache_decoded: decode each compressed chunk once, when it is encoded, and keep the
        dense copy for attention. Outputs are identical either way (decode is deterministic);
        False re-decodes every chunk on every step, so the GPU really holds only the CSR
        buffers but each step costs O(compressed tokens). kv_bytes() counts the compressed
        storage in both modes; decoded_bytes() reports the extra dense copy.
        budget_policy: fixed preserves uniform-byte matching; strict is the HH-only
        fp16/bf16 adaptive CSR quota, with the dense buffer reserved in advance."""
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if budget_policy not in ("fixed", "strict"):
            raise ValueError("budget_policy must be fixed or strict")
        if budget_policy == "strict" and (method != "tiered" or evict_light or avg_s is not None):
            raise ValueError("strict policy requires tiered, keep-light and automatic atom allocation")
        if isinstance(budget, bool) or not math.isfinite(budget) or not 0 < budget <= 1:
            raise ValueError("budget must be a fraction in (0,1]")
        for name, value, minimum in (("window", window, 0), ("chunk", chunk, 1),
                                     ("expected_new_tokens", expected_new_tokens, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if avg_s is not None and (isinstance(avg_s, bool) or not isinstance(avg_s, int) or avg_s < 1):
            raise ValueError("avg_s must be a positive integer")
        if method in ("h2o", "tiered") and tracker is None:
            raise ValueError(f"method={method!r} needs a ScoreTracker (tracker=...)")
        if (method in ("lexico", "tiered") and dictionary is None and dictionary_path is None
                and encode_fn is None):
            raise ValueError(f"method={method!r} needs a dictionary (dictionary= or dictionary_path=)")
        if tracker is not None and method in ("h2o", "tiered"):
            want = "sum" if method == "h2o" else "mean"
            if tracker.mode != want:
                warnings.warn(f"{method} is specified with ScoreTracker(mode={want!r}), got {tracker.mode!r}")
        self.method, self.window, self.chunk, self.budget = method, window, chunk, budget
        self.budget_policy = budget_policy
        self.tracker, self.dictionary_path = tracker, dictionary_path
        self.dictionary = dictionary
        self.avg_s, self.expected_new_tokens = avg_s, expected_new_tokens
        self.tiers, self.ratios, self.evict_light = tuple(tiers), tuple(ratios), evict_light
        self.cache_decoded = cache_decoded
        self.allocate_fn = allocate_fn or allocator.allocate
        self.encode_fn = encode_fn or compress.encode_kv
        self.decode_fn = decode_fn or compress.decode_kv
        self.key_cache: List[torch.Tensor] = []     # recent K per layer
        self.value_cache: List[torch.Tensor] = []   # recent V per layer
        # per layer: None, or {"k": [chunks], "v": [chunks], "s": [LongTensor], "n": tokens}
        self.sparse: List[Optional[Dict[str, Any]]] = []
        # per layer: None, or (K, V) [1, h, n_sparse, d] decoded once per chunk
        self._decoded: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        self._layer_dicts: Dict[int, Tuple[Any, Any]] = {}
        self._atom_carry: Dict[int, float] = {}     # lexico: fractional atoms owed per layer
        self._seen_tokens = 0

    # ------------------------------------------------------------------ Cache API

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            if not self.key_cache and self.tracker is not None:
                self.tracker.reset()   # new sequence: scores from a previous prompt don't apply
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.sparse.append(None)
            self._decoded.append(None)
            return key_states, value_states
        if len(self.key_cache) < layer_idx:
            raise ValueError(f"layer {layer_idx} updated before layer {len(self.key_cache)}")

        if self.method == "h2o":
            self._evict(layer_idx, seen_before=self._seen_tokens - key_states.shape[-2])
        elif self.method in ("lexico", "tiered"):
            self._compress(layer_idx, seen_before=self._seen_tokens - key_states.shape[-2],
                           incoming_tokens=key_states.shape[-2])
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        k, v = self.key_cache[layer_idx], self.value_cache[layer_idx]
        if self.sparse[layer_idx] is None:
            return k, v
        dk, dv = self._decode(layer_idx, k.dtype)
        return torch.cat([dk, k], dim=-2), torch.cat([dv, v], dim=-2)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= (layer_idx or 0):
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:   # deprecated name, still called in 4.49
        return None

    def __len__(self):
        return len(self.key_cache)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("TieredKVCache supports greedy decoding only")

    # ------------------------------------------------------------------ compression

    def _layer_dictionary(self, layer_idx, device):
        """(key_dict, value_dict) [d, N] for one layer on device, or (None, None) for stubs."""
        if layer_idx not in self._layer_dicts:
            D = self.dictionary
            if D is None and self.dictionary_path is not None:
                D = _load_dictionary(self.dictionary_path)
            if D is None:
                self._layer_dicts[layer_idx] = (None, None)
            else:
                if (D.ndim != 3 or D.shape[0] < 2*(layer_idx+1) or D.shape[0] % 2 or
                        D.shape[1] != self.key_cache[layer_idx].shape[-1] or
                        not 1 <= D.shape[2] <= 32768):
                    raise ValueError("dictionary must be [2*layers, head_dim, atoms<=32768] for this model")
                self._layer_dicts[layer_idx] = (D[2 * layer_idx].to(device, torch.float16),
                                                D[2 * layer_idx + 1].to(device, torch.float16))
        return self._layer_dicts[layer_idx]

    def _resolve_avg_s(self, recent, seen_before):
        """Average atoms per token that hits the budget, set once and shared by lexico and tiered.

        Fractional, unlike compress.solve_s (same formula, without the floor): an integer s
        undershoots the budget by up to one atom's bytes (~1.2 points at 8 heads x 128 dims).
        lexico realises it as a mix of floor(s) and floor(s) + 1; allocate matches its bytes.
        """
        if self.avg_s is None:
            compress.configure(n_kv_heads=recent.shape[1], head_dim=recent.shape[-1])
            n_total = seen_before + self.expected_new_tokens
            # After each update recent holds W+1 .. W+C dense tokens: (C+1)/2 extra on average.
            dense = self.window + (self.chunk + 1) / 2
            if n_total <= dense:
                raise ValueError(f"N={n_total} <= dense window {dense}: nothing to compress")
            full = compress.full_bytes_per_token()
            per_token = (self.budget * n_total - dense) * full / (n_total - dense)
            s = ((per_token / (2 * recent.shape[1]) - compress.INDPTR_BYTES)
                 / (compress.IDX_BYTES + compress.COEF_BYTES))
            if s < 1:
                raise ValueError(f"budget {self.budget:.1%} infeasible at N={n_total}: the dense "
                                 f"window alone is {dense / n_total:.1%}")
            self.avg_s = s
        return self.avg_s

    def _uniform_s(self, layer_idx, n, avg_s, device):
        """lexico: n tokens at avg_s on average, as floor / floor + 1. The fractional remainder
        carries over to the next chunk, so the layer's total atoms track avg_s * tokens."""
        ideal = self._atom_carry.get(layer_idx, 0.0) + avg_s * n
        total = int(ideal + 1e-9)
        self._atom_carry[layer_idx] = ideal - total
        s = torch.full((n,), total // n, dtype=torch.long, device=device)
        s[:total % n] += 1
        return s

    def _check_tracker(self, layer_idx, n_stored):
        n = self.tracker.acc[layer_idx].shape[-1] if layer_idx in self.tracker.acc else 0
        if n != n_stored:
            raise RuntimeError(f"layer {layer_idx}: tracker has {n} scores for {n_stored} cached "
                               "tokens; is the tracker hooked on this model and enabled?")

    def _strict_quota(self, recent, sp, seen_before, incoming_tokens):
        """Reserve the worst dense occupancy before the next compression.

        Fractional avg_s expresses a byte quota; stored atom counts remain integers.
        Terminal K/V CSR pointers and existing actual CSR bytes are included.
        This policy is for the HH-only ablation, not uniform-baseline equivalence.
        """
        if recent.dtype not in (torch.float16, torch.bfloat16) or incoming_tokens != 1:
            raise ValueError("strict policy requires fp16/bf16 KV and single-token decode")
        compress.configure(n_kv_heads=recent.shape[1], head_dim=recent.shape[-1])
        full = compress.full_bytes_per_token()
        future_seen = seen_before + incoming_tokens + self.chunk - 1
        future_dense = self.window + incoming_tokens + self.chunk - 1
        old_bytes = sum(_nbytes(c) for c in sp["k"]+sp["v"])
        sparse_ceiling = self.budget*future_seen*full-future_dense*full
        n_old = recent.shape[-2]-self.window
        per_token = (sparse_ceiling-old_bytes-2*compress.INDPTR_BYTES)/n_old
        avg_s = (per_token/(2*recent.shape[1])-compress.INDPTR_BYTES)/(compress.IDX_BYTES+compress.COEF_BYTES)
        if avg_s < 1:
            raise ValueError("strict budget infeasible: dense reserve and minimum sparse payload exceed capacity")
        return avg_s, sparse_ceiling

    def _compress(self, layer_idx, seen_before, incoming_tokens=1):
        """Encode all but W recent tokens before the incoming token exceeds W+C."""
        recent_k, recent_v = self.key_cache[layer_idx], self.value_cache[layer_idx]
        n_rec = recent_k.shape[-2]
        # Compression runs before the incoming token is appended. Trigger at
        # the boundary so the post-update recent buffer never grows to W+C+1.
        if n_rec < self.window + self.chunk:
            return
        if recent_k.shape[0] != 1:
            raise NotImplementedError("compression supports batch size 1")
        n_old = n_rec - self.window
        sp = self.sparse[layer_idx] or {"k": [], "v": [], "s": [], "n": 0}
        first = sp["n"]                      # physical index of the first token in recent
        strict = self.budget_policy == "strict"
        if strict:
            if recent_v.dtype != recent_k.dtype:
                raise ValueError("strict policy requires matching K/V dtypes")
            avg_s, sparse_ceiling = self._strict_quota(recent_k,sp,seen_before,incoming_tokens)
            self.avg_s = avg_s
        else:
            avg_s = self._resolve_avg_s(recent_k, seen_before)

        if self.method == "lexico":
            s = self._uniform_s(layer_idx, n_old, avg_s, recent_k.device)
        else:
            self._check_tracker(layer_idx, first + n_rec)
            scores = self.tracker.scores(layer_idx)[0, first:first + n_old]
            Dk, _ = self._layer_dictionary(layer_idx, recent_k.device)
            max_atoms = min(64, Dk.shape[-1]) if Dk is not None else 64
            if avg_s > max_atoms:
                raise ValueError("uniform avg_s exceeds the dictionary/allocator atom limit")
            s = self.allocate_fn(scores, avg_s, self.tiers, self.ratios, self.evict_light,
                                 bytes_fn=compress.bytes_per_token, s_max=max_atoms).to(recent_k.device)

        live = s > 0                         # s = 0: evicted light token, stored nowhere
        old_k = recent_k[0, :, :n_old].transpose(0, 1)[live]     # [n_live, h, d]
        old_v = recent_v[0, :, :n_old].transpose(0, 1)[live]
        if live.any():
            Dk, Dv = self._layer_dictionary(layer_idx, recent_k.device)
            ck, cv = self.encode_fn(old_k, Dk, s[live]), self.encode_fn(old_v, Dv, s[live])
            if strict:
                total = sum(_nbytes(c) for c in sp["k"]+sp["v"])+_nbytes(ck)+_nbytes(cv)
                if total > sparse_ceiling+1e-6:
                    raise RuntimeError("encoded payload exceeds strict byte quota; check storage dtypes")
            sp["k"].append(ck)
            sp["v"].append(cv)
            sp["s"].append(s[live])
            sp["n"] += int(live.sum())
            if self.cache_decoded:
                dk = self.decode_fn(ck, Dk, dtype=recent_k.dtype).transpose(0, 1).unsqueeze(0)
                dv = self.decode_fn(cv, Dv, dtype=recent_v.dtype).transpose(0, 1).unsqueeze(0)
                prev = self._decoded[layer_idx]
                self._decoded[layer_idx] = ((dk, dv) if prev is None else
                                            (torch.cat([prev[0], dk], dim=-2),
                                             torch.cat([prev[1], dv], dim=-2)))
        if not live.all():
            dev = live.device
            keep = torch.cat([torch.arange(first, device=dev), first + live.nonzero().flatten(),
                              torch.arange(first + n_old, first + n_rec, device=dev)])
            self.tracker.keep(layer_idx, keep)
        self.sparse[layer_idx] = sp if sp["n"] else None
        self.key_cache[layer_idx] = recent_k[:, :, n_old:]
        self.value_cache[layer_idx] = recent_v[:, :, n_old:]

    def _evict(self, layer_idx, seen_before):
        """h2o: keep the last W + top-scoring tokens dense, at floor(budget * seen) tokens."""
        k, v = self.key_cache[layer_idx], self.value_cache[layer_idx]
        n = k.shape[-2]
        if n <= int(self.budget * seen_before + 1e-12):
            return
        if k.shape[0] != 1:
            raise NotImplementedError("eviction supports batch size 1")
        self._check_tracker(layer_idx, n)
        idx = allocator.h2o_keep_idx(self.tracker.scores(layer_idx)[0], float(self.budget),
                                     self.window, original_length=seen_before, strict=True)
        idx = idx.to(k.device)
        self.key_cache[layer_idx] = k[:, :, idx]
        self.value_cache[layer_idx] = v[:, :, idx]
        self.tracker.keep(layer_idx, idx)

    def _decode(self, layer_idx, dtype):
        """Compressed tokens back to dense [1, h, n_sparse, d] for attention."""
        if self.cache_decoded and self._decoded[layer_idx] is not None:
            return self._decoded[layer_idx]
        sp = self.sparse[layer_idx]
        Dk, Dv = self._layer_dictionary(layer_idx, self.key_cache[layer_idx].device)
        dk = torch.cat([self.decode_fn(c, Dk, dtype=dtype) for c in sp["k"]])
        dv = torch.cat([self.decode_fn(c, Dv, dtype=dtype) for c in sp["v"]])
        return dk.transpose(0, 1).unsqueeze(0), dv.transpose(0, 1).unsqueeze(0)

    # ------------------------------------------------------------------ accounting

    def stored_tokens(self, layer_idx: int = 0) -> int:
        """Tokens actually held for a layer (dense + compressed, excluding evicted)."""
        n = self.key_cache[layer_idx].shape[-2] if len(self.key_cache) > layer_idx else 0
        sp = self.sparse[layer_idx] if len(self.sparse) > layer_idx else None
        return n + (sp["n"] if sp is not None else 0)

    def token_s(self, layer_idx: int = 0) -> torch.Tensor:
        """Atoms per compressed token for a layer, in order (for the demo's tier colouring)."""
        sp = self.sparse[layer_idx] if len(self.sparse) > layer_idx else None
        return torch.cat(sp["s"]) if sp is not None else torch.zeros(0, dtype=torch.long)

    def kv_bytes(self) -> int:
        """Bytes of KV storage across all layers and the batch (dense tensors + CSR buffers)."""
        total = 0
        for k, v, sp in zip(self.key_cache, self.value_cache, self.sparse):
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
            if sp is not None:
                total += sum(_nbytes(c) for c in sp["k"] + sp["v"])
        return total

    def decoded_bytes(self) -> int:
        """Bytes of the dense decoded copies kept when cache_decoded=True (0 otherwise)."""
        return sum(k.nbytes + v.nbytes for kv in self._decoded if kv is not None for k, v in [kv])

    def full_kv_bytes(self) -> int:
        """Bytes a full fp16 cache would use for every token seen so far."""
        if not self.key_cache:
            return 0
        b, h, _, d = self.key_cache[0].shape
        return len(self.key_cache) * b * self._seen_tokens * 2 * h * d * compress.DENSE_BYTES

    def measured_budget(self) -> float:
        """kv_bytes / full_kv_bytes: the memory fraction this run actually used (1.0 for full)."""
        full = self.full_kv_bytes()
        return self.kv_bytes() / full if full else 0.0
