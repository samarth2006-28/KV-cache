# TieredKV

**Attention-guided sparsity allocation for extreme KV cache compression.** HashCode 14.0, Track 01 (LLM Inference).

Lexico gives every token the same number of dictionary atoms `s`. TieredKV uses H₂O's accumulated attention score to decide how many atoms each token gets, while keeping total memory in bytes the same.

## Layout

| File | Owner | What it does |
|---|---|---|
| `tieredkv/allocator.py` | Sampreeth | Attention hooks, `ScoreTracker`, `allocate()` (byte-matched tiers), `h2o_keep_idx()` |
| `tieredkv/compress.py` | Samarth | Batched OMP, `encode_kv` / `decode_kv` with per-token `s`, CSR storage like Lexico's, memory math (`bytes_per_token`, `budget`, `solve_s`) |
| `tieredkv/train_dict.py` | Samarth | Trains Lexico-format dictionaries within a time limit (Lexico doesn't release any) |
| `tieredkv/gsm8k.py` | Samarth | Fixed subset, 5-shot prompt (same format as lm-eval), strict/flexible scoring, Wilson CI, McNemar |
| `plot.py` | Samarth | `results/results.csv` → accuracy-vs-memory plot, ablation plot, markdown table |
| `tieredkv/cache.py`, `eval.py` | Vishal | `TieredKVCache`, runner |

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

## Train a dictionary (needed before any Lexico or tiered run)

```bash
python -m tieredkv.train_dict --model meta-llama/Llama-3.1-8B-Instruct \
    --out dicts/llama31_8b_N4096_s16.pt --minutes 45
```

This writes `[2*n_layers, head_dim, 4096]` fp16 tensors: `D[2l]` holds keys, `D[2l+1]` values. Load them with `compress.load_dictionary(path, layer)`. The script prints the relative reconstruction error before and after training. Share the `.pt` file through HF Hub or Drive, not git.

## Memory accounting (the single definition)

Each compressed token-head is stored as int16 atom ids + fp8 coefficients + one int32 CSR pointer, with no padding:

- `bytes_per_token(s) = 2 · n_kv_heads · (3s + 4)`
- `budget = (Σ bytes(s_i) + W · full) / (N · full)`

For the Lexico baseline, `solve_s(budget, N, W)` gives the uniform `s`. Tiered runs use the same `s` as `avg_s`, so both methods use the same number of bytes.
