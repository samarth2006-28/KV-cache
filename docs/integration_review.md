# Integrated phase review

Status: ready for target-model validation; not a completed accuracy/performance benchmark.

This branch combines Sampreeth's allocator phases through 02cf6a7, Vishal's c-eval
through 81c0b96, and shared main through 658ebdd. The original branches and
historical experiment reports remain intact. No main-branch push was made.

## Reproduced defects and fixes

| Area | Finding | Result |
| --- | --- | --- |
| Sparse coefficient selection | Torch 2.5 CPU cannot advanced-index FP8 tensors | Select raw bytes; preserve exact FP8 bits, including reordered/repeated/empty selections |
| Atom counts | Fractional/negative/nonfinite or oversized s could be truncated or enter OMP | Reject invalid requests; cap tiered allocation to dictionary size |
| Dictionary compatibility | Wrong head dimension produced low-level failures; retraining at the same path could reuse stale cached weights | Validate layout and invalidate by resolved file path/size/mtime |
| Allocator/cache settings | Invalid tiers, atom bounds, budgets, windows or chunk sizes lacked clear guards | Reject before allocation/cache use; empty batches supported |
| Dense decode buffer | Compression triggered one update late, allowing W+C+1 recent tokens | Trigger before the incoming token would exceed W+C; decode chunks now contain exactly C tokens |
| Ablation descriptors | Chunk=32 disagreed with the integrated cache's C=8; CLI could not consume descriptors | Align all configs to 8; --config now loads thresholds and mean scoring |
| Experiment storage | Repeated cells could overwrite prediction files | Unique run IDs, prediction and manifest paths |
| Budget qualification | Aggregate rows did not represent per-problem failures | Both per-problem and aggregate audits; eligible=False remains recorded; CLI exit 1 |
| Plotting | Units were guessed, failed budgets were included, mixed cohorts could be combined, x positions shifted | Explicit units; exclude flagged comparisons; reject mixed model/task/seed/n; use actual measured x |
| Reproducibility | Dataset revision and effective settings were not fully recorded | Pin GSM8K revision; record prompt/test/shot IDs, source/dictionary hashes, dtype and cache settings |
| Presentation | Unrecorded concentration/position percentages and stronger downstream/novelty claims | Replace with documented diagnostic evidence and limits |

The append-only CSV schema now adds run_id and eligible. Existing old-header files
are refused rather than silently rewritten; choose a fresh output path.
Plots write a separate budget_audit.json. Historical results are not modified or
relabelled. The original 18.48% allocation projections remain failed 20% cells.

## Validation

Environment: Python 3.10, Torch 2.5.1+cpu, Transformers 4.49.0.

```bash
python -m pytest -q tests/
```

**227 passed, 1 skipped.** The skipped check is the opt-in real target-model
integration test requiring TIEREDKV_MODEL, model access and the dataset dependency.
The warnings exercise legacy H2O recency overspending and a tiny eviction batch
whose rounded heavy/mid tiers are empty. Strict audits keep failed budgets out
of comparisons.

The suite covers all phase modules plus the merged cache, compressor, dictionary
trainer/sanity checker, answer scoring, plotting and runner. Integration checks
include identity-encoder plumbing, real-encoder equal-ratio TieredKV vs uniform
generation, H2O full-budget equality, repeated eviction, original rotary positions,
tracker cleanup/alignment, invalid settings, FP8 CPU selection, small dictionaries,
non-overwritten runs, CLI config handling and plot/audit eligibility.

A separate real pretrained CPU check used the pinned SmolLM2-135M-Instruct weights
and five pinned GSM8K prompts (728-753 input tokens), with eight generated tokens:
- Shared full cache equals stock HF: 5/5.
- Shared H2O at 100% equals stock HF: 5/5.
- Shared H2O at 20% equals independent H2O reference: 5/5.
- Scores align to surviving cache rows after eviction: 5/5.

Settings and source fingerprints are in integration_smoke_observations.json.
This small-model test did not score answers, encode with the target dictionary,
measure GPU performance, or validate a target-model memory/accuracy advantage.

## Phase status after review

| Phase | Verified locally | Still required on agreed model/compute |
| --- | --- | --- |
| Scoring and allocation | Hook, causal opportunity counts, sum/mean modes, byte matching, real 20-prompt diagnostics | Target-model score diagnostics if used in the final claims |
| H2O baseline | Independent reference and shared implementation agree; full-budget generation equality | Same-dtype/denominator main matrix at four audited budgets |
| Shared integration | Combined code passes all offline integration gates | Target-model full/identity/equal-ratio gates and reproduced uniform baseline |
| Threshold ablation | Three configs load correctly; 1,800 historical projections audited | Actual 3/5/10% accuracy comparison at one eligible budget |
| Sanity/presentation | Latest committed 3B reconstruction probe included and slide wording checked | Reproduce with the shared dictionary as needed; finish accuracy/result slides |

## Target-model run sequence

1. Push and use this integrated review branch on shared compute.
2. Install requirements, obtain the agreed model and matching dictionary.
3. Pass the target-model full-cache gate, then uniform/equal-ratio generation checks.
4. Run a small pilot before the full matrix. Confirm actual retained bytes and
   per-problem eligibility; early EOS, integer s and chunk occupancy affect budgets.
5. Run full, uniform, H2O and TieredKV on the same fixed subset and seed. Then
   launch the three --config threshold descriptors with identical settings.
6. Plot only eligible comparisons. Report observed accuracy and decode profiling;
   keep any negative result visible. Finish the demo and evidence-backed deck.

An untrained/random dictionary in offline tests proves wiring only. The committed
3B sanity JSON reports reconstruction results on five prompts; it is not a task
accuracy result or a reproduced published-Lexico comparison. Final claims must not
equate attention-weighted reconstruction error with downstream accuracy.

## Memory boundary

The shared denominator is full fp16 KV. Float32 full-cache tests therefore report
2x that denominator. Actual CSR bytes include terminal pointers per chunk, while
the per-token planning formula omits them. Retained KV excludes dictionary weights,
score/count metadata and temporary decoded tensors. Peak GPU memory is separate.
Do not retain dense decoded tensors and continue claiming only sparse storage.

Current uniform s is chosen using expected generation length. Actual early stopping
or dense-chunk occupancy can miss the requested budget; audits flag that rather
than claiming target compliance. A passed budget audit alone does not establish
equal accuracy, a throughput improvement, or a statistically significant result.

Tier percentiles are computed within each compression batch. At C=8, 3% and 5%
heavy groups can round to zero on decode batches; 10% rounds to one. The prefill
batch is much larger. This implementation does not continuously re-rank and
re-encode all previously compressed tokens. State that granularity when presenting
the threshold experiment; do not claim a global evolving top-5% cache policy.
