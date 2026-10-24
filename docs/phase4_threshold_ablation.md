# Phase 4: heavy-hitter threshold ablation

Status: configs and real-score allocation diagnostics complete. **The task-accuracy
ablation is blocked, not completed.** Sampreeth owns the 3/5/10% threshold experiment
at a nominal 20% KV budget (team plan, allocator step 7).

## Contribution

- Three run descriptors: configs/tiered_hh3_b20.json, tiered_b20.json and
  tiered_hh10_b20.json. Same budget, seed, window, chunk, ratios, keep-light policy
  and mean score mode; only the heavy percentile changes.
- tieredkv.threshold_ablation runs those thresholds on the SAME normalized scores
  from real, uncompressed prefills. It reserves the dense recent window, ranks
  only old tokens, uses canonical allocate and compress memory helpers, checks
  monotonicity and equal-ratio uniform equivalence, and records atom histograms.
- Tests cover rounding, recency independence, feasible/infeasible budgets, exact
  matching to uniform bytes, config/parser agreement and visible audit failures.
- No changes to cache.py, eval.py, compress.py, dictionary training or answer scoring.

## Executed diagnostic

Model: HuggingFaceTB/SmolLM2-135M-Instruct, revision
12fd25f77366fa6b3b4b768ec3050bf629380bac. CPU float32 / eager attention,
Torch 2.5.1+cpu, Transformers 4.49.0, four threads, no generated tokens.
Twenty fixed GSM8K test prompts, seed 0, with five train-only examples at seed
1234. Data revision 740312add88f781978c0658806c59bc2815b9866; parquet hashes
and row counts checked by the existing loader. No prompt truncation.

All 20 prompts have 30 scored layers: 600 prompt/layer pairs and 1,800 allocation
records across three thresholds. Every variant uses W=32, ratios 32/12/4, next
20% mid, rest light. Percentile sizes round to the nearest token using the
canonical allocator rule; realized fractions are saved.

| Heavy fraction | Mean projected KV | Exact old-byte match to uniform | Within 1 point of 20% |
| --- | ---: | ---: | ---: |
| 3% | 18.48355% | 600/600 | 0/600 |
| 5% | 18.48355% | 600/600 | 0/600 |
| 10% | 18.48355% | 600/600 | 0/600 |

The small model has head_dim=64. The largest affordable uniform integer count
is s=5; rounding down produces an undershoot of up to 1.70251 percentage points.
All target-audit failures are retained in the CSV and summary. Do not include
these as passed 20% benchmark cells. Changing only tiered to spend more would
break its byte match to the uniform comparator.

Projection uses the shared fp16/CSR formula, even though the model runs fp32.
It is not measured SparseKV storage, GPU memory, or peak process memory.
The shared formula excludes the two terminal int32 CSR pointers (8 bytes/layer).
No encoding, reconstruction-error evaluation, compressed generation, or
accuracy measurement occurred. This experiment cannot identify the best threshold.

The result bundle contains the 20 score dumps (raw sums + counts), all allocation
rows, prompt/dataset/source provenance, local model-file hashes, independent
byte/count validation and shared-cache construction probes. Compact observations
are in phase4_threshold_observations.json; raw runtime data stays outside git.

## Checks

141 passed, 1 skipped across allocator, scoring diagnostics, H2O reference,
threshold diagnostics and shared integration tests. The skipped target-model
integration test needs external weights. Separately reopened all 20 score dumps,
checked causal counts and attention mass for 600 layers, independently recomputed
all 1,800 histogram byte totals, and verified the executed source hashes.

## Remaining end-to-end gate

At main 9350a8f, all THREE configs were passed to eval.make_cache. Each raises
NotImplementedError because shared TieredKVCache supports full only.
Shared eval.run also does not yet construct/attach a mean-mode ScoreTracker.
Config score_mode is a handoff requirement, not a switch consumed by that runner.

Before running task accuracy on shared compute:
1. Wire compressed generation and mean ScoreTracker in the shared cache/runner.
   Preserve original positions/masks and score/KV alignment.
2. Use the same capable model, matching dictionary, fixed subset n/seed, five-shot
   examples, generation settings and memory denominator for all three thresholds.
3. Pass the uniform-vs-equal-ratio generation equivalence gate, not just the
   allocator-level equality checked here.
4. Audit every problem and aggregate against the target; retain and flag failures.
5. Compare paired answer accuracy only after those gates pass. Do not choose a
   threshold using these allocation projections.

The JSON files are run descriptors; eval.py's CLI does not currently load them.
For the integration owner, the existing Python entry point is:

```python
import json
from tieredkv.eval import run

descriptor = json.load(open("configs/tiered_hh3_b20.json"))
# Only after the compressed cache, tracker and equivalence gates above pass:
result = run(descriptor["method"], descriptor["budget"], descriptor["task"],
             n=AGREED_N, seed=descriptor["seed"],
             model_name=AGREED_MODEL,
             cfg={**descriptor["cfg"], "dictionary":MATCHING_DICTIONARY},
             results=SEPARATE_ABLATION_CSV)
```

Use each descriptor unchanged except for shared model/dictionary/n/output choices.
The integration must honor descriptor["score_mode"] by constructing the tracker;
passing it as a stray cfg field does not do that in the current runner.

## Reproduce the completed diagnostic

```bash
python -m tieredkv.threshold_ablation --model LOCAL_MODEL_DIR --model-id HuggingFaceTB/SmolLM2-135M-Instruct --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --data PINNED_GSM8K_DIR --out OUTPUT_DIR --n 20 --threads 4
```

Outputs are allocation_diagnostics.csv, summary.json and scores_IDX.pt.
The CSV deliberately has no acc or budget_measured columns, so allocation
projections cannot silently enter the shared task-results table.
