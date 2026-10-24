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
| `tieredkv/cache.py` | Vishal | `TieredKVCache(method)` for `generate()`: dense recent window + compressed old tokens, byte accounting (`kv_bytes`, `measured_budget`). `full` / `lexico` / `tiered` / `h2o`: compresses (or evicts) everything but the last W tokens once recent exceeds W + C |
| `tieredkv/profile.py` | Vishal | `DecodeTimer` (decode-only tok/s, ms/token), peak GPU memory |
| `notebooks/colab_gpu_runs.ipynb` | Vishal | Colab GPU run book: plan test 1, full baseline, dictionary, lexico / tiered / h2o at 20%, audit, plot |
| `tieredkv/eval.py` | Vishal | `run(method, budget, task, n, seed)`: fixed GSM8K subset, greedy 256 tokens, appends one row to `results/results.csv`, per-problem preds to `results/runs/*.jsonl` |

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

## Run one matrix cell

```bash
# full-cache upper bound (plan test 1 on the real model first)
TIEREDKV_MODEL=meta-llama/Llama-3.1-8B-Instruct python -m pytest -q tests/test_integration.py
python -m tieredkv.eval --method full --budget 1.0 --task gsm8k --n 50
python -m tieredkv.eval --method tiered --budget 0.2 --n 50 --dictionary dicts/llama31_8b_N4096_s16.pt
```

The run prints seconds per problem (also the `s_prob` column). Size n with `n = GPU-seconds / (20 × s_prob)`.

Method names: `full`, `lexico`, `h2o`, `tiered`, plus tiered ablations `tiered_hh3`, `tiered_hh10`, `tiered_24-8-2`, `tiered_evict` (suffixes combine, e.g. `tiered_hh3_evict`). `plot.py` treats anything except the four main names as an ablation.

Dense recent window: `W = 32` for every method (`cache.WINDOW`). Lexico's default of 128 makes the 10% and 15% budgets impossible on ~1000-token prompts. Decode chunk `C = 8` (`cache.CHUNK`): recent swings between W and W + C dense tokens, so C = 32 would overshoot the measured budget by ~2.5 points.

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

## Integrated validation and threshold runs

For the current Phase-4 GPU runner, use
[notebooks/phase4_thresholds_colab.ipynb](notebooks/phase4_thresholds_colab.ipynb).
See [docs/phase4_colab.md](docs/phase4_colab.md) for its strict byte policy,
dictionary requirements and result audits. It prepares and executes the experiment;
local code tests are not completed target-model accuracy results.

The integration-review branch combines c-eval with the allocator phases and latest
shared main. See [docs/integration_review.md](docs/integration_review.md) for tested
behavior and remaining target-model gates. Score normalization corrects observation
counts; it does not guarantee position-independent scores or better accuracy.

Run all threshold descriptors with identical model/dictionary/subset settings:

```bash
python -m tieredkv.eval --config configs/tiered_hh3_b20.json --model LOCAL_MODEL_OR_HF_ID --dictionary MATCHING_DICT.pt --n 50 --results results/review_thresholds.csv
python -m tieredkv.eval --config configs/tiered_b20.json --model LOCAL_MODEL_OR_HF_ID --dictionary MATCHING_DICT.pt --n 50 --results results/review_thresholds.csv
python -m tieredkv.eval --config configs/tiered_hh10_b20.json --model LOCAL_MODEL_OR_HF_ID --dictionary MATCHING_DICT.pt --n 50 --results results/review_thresholds.csv
python plot.py --csv results/review_thresholds.csv --out results/review_plots
```

Each run gets unique prediction and manifest files. Manifests record effective
cache settings, model dtype/revision when available, pinned dataset revision,
prompt hashes, dictionary/source hashes and per-problem budget audits.
The CSV now includes run_id and eligible: use a new CSV rather than appending to
the old header. Budget failures remain recorded with eligible=False and CLI exit
code 1; plots exclude them and save budget_audit.json. Plots reject mixed
model/task/seed/subset sizes. Fraction units are the default; legacy percent CSVs
require --units percent. No eligible rows means no new comparison plot.

Memory is actual retained dense+CSR payload divided by full **fp16** KV. A float32
full cache therefore measures 200%, not 100%; CPU compatibility tests are not
matched-memory benchmark rows. Dictionary weights, tracker metadata and temporary
decoded tensors are not included in retained KV; peak GPU memory is separate.
Each CSR K/V chunk also has final pointer bytes beyond the per-token formula.
Integer s, dense-window/chunk occupancy, early EOS and expected output length can
cause target undershoot or overshoot. Audit actual results; do not relabel them.
