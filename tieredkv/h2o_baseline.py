"""Canonical H2O policy re-export; implementation stays in allocator.py.

This module provides the work-plan import path without creating a second policy.
Use strict=True and original_length=total_tokens_seen for matched-budget runs.
"""
from .allocator import h2o_keep_idx

__all__ = ["h2o_keep_idx"]
