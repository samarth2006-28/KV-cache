"""TieredKVCache: the cache generate() talks to every step.

Owner: Vishal. Contract (see plan section 7):
    TieredKVCache(method, ...)   method in {full, lexico, h2o, tiered}
        .update(k, v, layer_idx, cache_kwargs) -> (K, V) for attention   [b, kv_heads, seq, d]
        .get_seq_length(layer_idx)             tokens seen (positions), not tokens stored
        .kv_bytes() / .measured_budget()       what the cache actually holds vs full fp16

Per layer the cache stores `sparse` (compressed old tokens, original order) + `recent`
(dense, last W tokens). Phase 1: only method="full" is wired; it keeps everything in
`recent`, so it is exactly DynamicCache. The compression path (compress on overflow at the
start of update(), decode_kv + recent on return) lands in phase 2.

Pinned to transformers==4.49.0 (same as Lexico). Pass it to generate() like any Cache:

    cache = TieredKVCache("full")
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=256, do_sample=False)
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from . import compress

__all__ = ["TieredKVCache", "METHODS"]

METHODS = ("full", "lexico", "h2o", "tiered")

# Dense recent window W, one constant for every method. 128 (Lexico's default) makes 10% and
# 15% infeasible on ~1000-token 5-shot GSM8K prompts: the window alone is ~13%. At W=32,
# compress.solve_s gives s = 4 / 9 / 13 / 17 at 10 / 15 / 20 / 25% for N=1000.
WINDOW = 32
CHUNK = 32


class TieredKVCache(Cache):
    def __init__(self, method: str = "full", *, window: int = WINDOW, chunk: int = CHUNK,
                 budget: float = 1.0, tracker=None, dictionary_path: Optional[str] = None,
                 tiers=(0.05, 0.20), ratios=(32, 12, 4), evict_light: bool = False):
        """window W: dense recent tokens (shared by all methods). chunk C: tokens compressed
        at once when recent exceeds W + C. budget: target memory fraction of full fp16 KV.
        tracker: allocator.ScoreTracker on the same model (h2o / tiered).
        tiers / ratios / evict_light: passed to allocator.allocate (tiered only)."""
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if method != "full":
            raise NotImplementedError(f"method={method!r} lands in phase 2; only 'full' is wired")
        self.method, self.window, self.chunk, self.budget = method, window, chunk, budget
        self.tracker, self.dictionary_path = tracker, dictionary_path
        self.tiers, self.ratios, self.evict_light = tuple(tiers), tuple(ratios), evict_light
        self.key_cache: List[torch.Tensor] = []     # recent K per layer
        self.value_cache: List[torch.Tensor] = []   # recent V per layer
        self.sparse: List[Optional[Tuple[Any, Any]]] = []   # (SparseKV K, SparseKV V) per layer
        self._seen_tokens = 0

    # ------------------------------------------------------------------ Cache API

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.sparse.append(None)
        elif len(self.key_cache) > layer_idx:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            raise ValueError(f"layer {layer_idx} updated before layer {len(self.key_cache)}")
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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

    # ------------------------------------------------------------------ accounting

    def stored_tokens(self, layer_idx: int = 0) -> int:
        """Tokens actually held for a layer (dense + compressed, excluding evicted)."""
        n = self.key_cache[layer_idx].shape[-2] if len(self.key_cache) > layer_idx else 0
        sp = self.sparse[layer_idx] if len(self.sparse) > layer_idx else None
        return n + (sp[0].n_tokens if sp is not None else 0)

    def kv_bytes(self) -> int:
        """Bytes of KV storage across all layers and the batch (dense tensors + CSR buffers)."""
        total = 0
        for k, v, sp in zip(self.key_cache, self.value_cache, self.sparse):
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
            if sp is not None:
                total += sp[0].nbytes + sp[1].nbytes
        return total

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
