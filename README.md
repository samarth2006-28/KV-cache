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
| `scripts/run_matrix.sh` | Vishal | Whole GSM8K matrix into one CSV (resumes: finished cells are skipped), then audit, tiered-vs-lexico/h2o McNemar and plot |
| `tieredkv/hit_rate.py` | Vishal | Heavy-hitter hit rate: do tokens ranked important from prompt attention get the attention during generation? Also attention H2O loses by evicting |
| `tieredkv/compare.py` | Samarth | Paired per-problem McNemar between two methods, strict and flexible |
| `tieredkv/check_dict.py` | Samarth | Dictionary sanity check: error vs s, timing, attention-vs-error correlation, attention-weighted error |
| `tieredkv/budget_audit.py` | Sampreeth | Flags any run (or problem) more than 1 point from its target memory |
| `tieredkv/h2o_baseline.py`, `h2o_experiment.py` | Sampreeth | Independent H2O reference and its audited GPU matrix |
| `tieredkv/threshold_ablation.py`, `phase4_experiment.py` | Sampreeth | Heavy-hitter threshold (3/5/10%) ablation |

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

## Run one matrix cell

```bash
# full-cache upper bound (plan test 1 on the real model first)
TIEREDKV_MODEL=unsloth/Llama-3.2-3B-Instruct python -m pytest -q tests/test_integration.py
python -m tieredkv.eval --method full --budget 1.0 --task gsm8k --n 50
python -m tieredkv.eval --method tiered --budget 0.2 --n 50 --dictionary dicts/llama32_3b_N4096_s16.pt
```

Whole matrix (resumable), and the hit-rate diagnostic (~5 s per problem on 3B):

```bash
scripts/run_matrix.sh dicts/llama32_3b_N4096_s16.pt 40 results/main/results.csv
python -m tieredkv.hit_rate --model unsloth/Llama-3.2-3B-Instruct --n 20
```

Headline accuracy for Llama-3.2-3B is flexible match (`plot.py --metric acc_flex`): it often answers "The answer is N" instead of "#### N".

The run prints seconds per problem (also the `s_prob` column). Size n with `n = GPU-seconds / (20 × s_prob)`.

Method names: `full`, `lexico`, `h2o`, `tiered`, plus tiered ablations `tiered_hh3`, `tiered_hh10`, `tiered_24-8-2`, `tiered_evict` (suffixes combine, e.g. `tiered_hh3_evict`). `plot.py` treats anything except the four main names as an ablation.

Dense recent window: `W = 32` for every method (`cache.WINDOW`). Lexico's default of 128 makes the 10% and 15% budgets impossible on ~1000-token prompts. Decode chunk `C = 8` (`cache.CHUNK`): recent swings between W and W + C dense tokens, so C = 32 would overshoot the measured budget by ~2.5 points.

## Train a dictionary (needed before any Lexico or tiered run)

```bash
python -m tieredkv.train_dict --model unsloth/Llama-3.2-3B-Instruct \
    --out dicts/llama32_3b_N4096_s16.pt --minutes 45
```

This writes `[2*n_layers, head_dim, 4096]` fp16 tensors: `D[2l]` holds keys, `D[2l+1]` values. Load them with `compress.load_dictionary(path, layer)`. The script prints the relative reconstruction error before and after training. Share the `.pt` file through HF Hub or Drive, not git.

## Memory accounting (the single definition)

Each compressed token-head is stored as int16 atom ids + fp8 coefficients + one int32 CSR pointer, with no padding:

- `bytes_per_token(s) = 2 · n_kv_heads · (3s + 4)`
- `budget = (Σ bytes(s_i) + W · full) / (N · full)`

`solve_s(budget, N, W)` gives the largest whole uniform `s`. The cache uses the same formula without the floor (and with the average dense chunk, W + (C+1)/2): a fractional `avg_s` shared by both methods. Lexico stores floor(avg_s) or floor(avg_s) + 1 atoms per token; tiered's `allocate` matches the same bytes. Both plan for `--expected-new-tokens` (default 64) generated tokens.

## Integrated validation and threshold runs

For Sampreeth's four H2O budgets plus full baseline, open
[notebooks/h2o_matrix_colab.ipynb](notebooks/h2o_matrix_colab.ipynb).
It needs a GPU and pinned model, **no dictionary**. It also saves a twenty-prompt
target-model attention probe. See [docs/h2o_colab.md](docs/h2o_colab.md).
Your remaining deliverables and deck order are tracked in
[docs/sampreeth_completion.md](docs/sampreeth_completion.md), with slide text and
speaker notes in [docs/sampreeth_slides.md](docs/sampreeth_slides.md).

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
matched-memory benchmark rows. Dictionary weights, tracker metadata and the
decoded copies are not included in retained KV: for speed the cache keeps each compressed chunk
decoded (`cache_decoded=True`), reported per problem as `decoded_bytes`. Peak GPU memory is separate.
Each CSR K/V chunk also has final pointer bytes beyond the per-token formula.
Dense-window/chunk occupancy, early EOS and the planned output length
(`--expected-new-tokens`) can cause undershoot or overshoot. Audit actual results; do not relabel them.
