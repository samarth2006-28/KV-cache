"""H2O selection plus an independent greedy reference path for Llama/RoPE models.

The shared cache.py is unchanged. This correctness reference supports one
unpadded prompt, causal prefill, single-token decode and no logits processors.
It materializes full prefill KV and eager attention; it is not a peak-memory or
production-speed optimization. Selection remains canonical in allocator.py.
"""
from dataclasses import dataclass
import math
import time

import torch
from transformers.cache_utils import DynamicCache

from .allocator import h2o_keep_idx
from .allocator import ScoreTracker

__all__ = ["h2o_keep_idx", "H2OReferenceCache", "h2o_generate", "H2OGeneration"]


class H2OReferenceCache(DynamicCache):
    """Prune only AFTER the observer has scored the current attention call.

    Cached keys already carry RoPE at their original positions. We never reapply
    RoPE to survivors. The caller supplies absolute query position_ids and
    cache_position; physical cache length is deliberately not used as token age.
    """
    def __init__(self, tracker, budget=1.0, window=32):
        super().__init__()
        if not math.isfinite(budget) or not 0 < budget <= 1:
            raise ValueError("budget must be in (0,1]")
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError("window must be a nonnegative integer")
        self.tracker, self.budget, self.window = tracker, budget, window
        self.total_seen = 0
        self.positions = []
        self.peak_materialized_kv_bytes = 0

    def kv_bytes(self):
        return sum(t.numel()*t.element_size() for pair in zip(self.key_cache, self.value_cache) for t in pair)

    def full_bytes(self):
        return sum(t.shape[0]*t.shape[1]*self.total_seen*t.shape[3]*t.element_size()
                   for pair in zip(self.key_cache, self.value_cache) for t in pair)

    def auxiliary_bytes(self):
        arrays = list(self.tracker.acc.values())+list(self.tracker.cnt.values())+self.positions
        return sum(t.numel()*t.element_size() for t in arrays)

    def prune(self, total_seen):
        if total_seen <= self.total_seen:
            raise ValueError("total_seen must increase after each new attention call")
        self.peak_materialized_kv_bytes = max(self.peak_materialized_kv_bytes, self.kv_bytes())
        # Validate all layers before mutating any cache or tracker arrays.
        plans = []
        for layer, keys in enumerate(self.key_cache):
            scores = self.tracker.scores(layer)
            if keys.shape[0] != 1 or scores.shape != (1, keys.shape[-2]):
                raise ValueError("reference supports batch=1 and scores aligned to physical K/V rows")
            old_pos = self.positions[layer] if layer < len(self.positions) else torch.empty(0, dtype=torch.long, device=keys.device)
            positions = torch.cat([old_pos, torch.arange(self.total_seen, total_seen, device=keys.device)])
            if positions.numel() != keys.shape[-2]:
                raise ValueError("logical token history does not match cache growth")
            idx = h2o_keep_idx(scores[0], self.budget, self.window,
                               original_length=total_seen, strict=True)
            plans.append((idx, positions))
        for layer, (idx, positions) in enumerate(plans):
            self.key_cache[layer] = self.key_cache[layer].index_select(-2, idx)
            self.value_cache[layer] = self.value_cache[layer].index_select(-2, idx)
            self.tracker.keep(layer, idx)
            kept_pos = positions.index_select(0, idx)
            if layer == len(self.positions):
                self.positions.append(kept_pos)
            else:
                self.positions[layer] = kept_pos
        self.total_seen = total_seen
        full = self.full_bytes()
        return {"tokens_seen": total_seen, "tokens_stored": self.get_seq_length(),
                "kv_bytes": self.kv_bytes(), "full_kv_bytes": full,
                "kv_fraction": self.kv_bytes()/full if full else 0.0,
                "auxiliary_bytes": self.auxiliary_bytes()}


@dataclass
class H2OGeneration:
    sequences: torch.Tensor
    samples: list
    seconds: float
    peak_materialized_kv_bytes: int


@torch.no_grad()
def h2o_generate(model, input_ids, *, budget=1.0, window=32, max_new_tokens=64):
    """Greedy reference: score -> prune -> feed next token at its original position.

    budget controls retained dense KV bytes relative to full KV at the SAME
    dtype. Metadata is reported separately. The uncompressed prefill transient
    is reported, so retained KV savings are not presented as peak-memory savings.
    """
    if model.config.model_type != "llama" or getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("reference requires a Llama-family model with eager attention")
    if model.training:
        raise ValueError("call model.eval() before reference generation")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be one nonempty unpadded prompt")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    prompt_len = input_ids.shape[1]
    if prompt_len+max_new_tokens-1 > model.config.max_position_embeddings:
        raise ValueError("prompt plus generation exceeds configured context length")
    # Reject infeasible protected-window settings before an expensive forward.
    h2o_keep_idx(torch.zeros(prompt_len, device=input_ids.device), budget, window,
                 original_length=prompt_len, strict=True)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    samples, generated = [], []
    sync = torch.cuda.synchronize if input_ids.device.type == "cuda" else lambda: None
    sync(); start = time.perf_counter()
    with ScoreTracker(model, mode="sum") as tracker:
        cache = H2OReferenceCache(tracker, budget, window)
        pos = torch.arange(prompt_len, device=input_ids.device)
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
        seen = prompt_len
        for step in range(max_new_tokens):
            token = output.logits[:, -1:].argmax(-1)
            generated.append(token)
            del output
            samples.append(cache.prune(seen))
            if int(token.item()) in eos_ids or step+1 == max_new_tokens:
                break
            # Every surviving key precedes this single query. The explicit 4D
            # additive mask prevents compact row indices becoming fake positions.
            mask = torch.zeros((1,1,1,cache.get_seq_length()+1), device=input_ids.device,
                               dtype=next(model.parameters()).dtype)
            position = torch.tensor([seen], device=input_ids.device)
            output = model(input_ids=token, attention_mask=mask, position_ids=position[None],
                           cache_position=position, past_key_values=cache, use_cache=True)
            seen += 1
    sync()
    return H2OGeneration(torch.cat([input_ids, *generated], dim=1), samples,
                         time.perf_counter()-start, cache.peak_materialized_kv_bytes)
