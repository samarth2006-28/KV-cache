"""Profiling: KV storage, peak GPU memory, decode throughput and latency.

Owner: Vishal. Plan step 4:
    KV storage   cache.kv_bytes(): dense tensors + CSR buffers (the primary memory number)
    peak_gb      torch.cuda.max_memory_allocated (secondary; includes the temporary decode)
    tok_s/ms_tok timed over DECODE steps only, with torch.cuda.synchronize()

DecodeTimer is a generate() streamer: generate() calls put() once with the prompt before
prefill, then once per generated token. So stamps[1] is the end of prefill (first token) and
stamps[1:] time the decode steps.

    timer = DecodeTimer()
    model.generate(..., streamer=timer)
    timer.decode_tokens, timer.decode_seconds, timer.prefill_seconds
"""
import time

import torch
from transformers.generation.streamers import BaseStreamer

__all__ = ["DecodeTimer", "sync", "reset_peak", "peak_gb"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


class DecodeTimer(BaseStreamer):
    """Timestamps every generate() step. Batch size 1 (generate's streamer requirement)."""

    def __init__(self):
        self.stamps = []

    def put(self, value):
        sync()
        self.stamps.append(time.perf_counter())

    def end(self):
        pass

    @property
    def new_tokens(self):
        return max(len(self.stamps) - 1, 0)

    @property
    def prefill_seconds(self):
        return self.stamps[1] - self.stamps[0] if len(self.stamps) > 1 else 0.0

    @property
    def decode_tokens(self):
        """Tokens produced by decode steps (the first token comes out of prefill)."""
        return max(len(self.stamps) - 2, 0)

    @property
    def decode_seconds(self):
        return self.stamps[-1] - self.stamps[1] if len(self.stamps) > 2 else 0.0
