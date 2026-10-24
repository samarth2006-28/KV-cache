# Phase 3: H2O reference baseline and smoke matrix

This contribution stays in Sampreeth's modules. Vishal's `cache.py` and `eval.py`
are not modified. The reference is not a production inference engine and is not
the team's target-model benchmark.

## Working reference path

`H2OReferenceCache` uses the canonical `h2o_keep_idx` selection function with
strict budgets and the original number of tokens seen. `h2o_generate` runs an
unpadded Llama-family prompt with eager attention, accumulates per-layer raw
attention (mean over heads), and prunes only after scoring each attention call.
It retains the recent W tokens plus the highest-scoring older tokens.

Survivor K vectors keep their existing RoPE rotations. Query position_ids and
cache_position use original absolute positions, not compact cache-row indices.
For single-token decode every surviving key precedes the query; an explicit
additive visibility mask allows those keys. Score totals/counts and original
position records are gathered with exactly the same survivor indices as K/V.

Scope: batch size 1, ordinary causal prefill, one-token greedy decode, EOS/token
cap, no beam search or logits processors. The implementation rejects unsupported
models, impossible recent windows, training-mode models and context overflow.
Metadata and full-prefill transient KV are exposed separately from retained KV.

## Completed small-model smoke run

- Model: HuggingFaceTB/SmolLM2-135M-Instruct.
- Model revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Data: openai/gsm8k, main train/test parquet, revision
  `740312add88f781978c0658806c59bc2815b9866`.
- Dataset sizes: 7,473 training and 1,319 test rows; file hashes verified.
- Test indices: 120, 192, 474, 991, 1101 (n=5, seed=0).
- Five TRAIN exemplars: 797, 3169, 5780, 6203, 6526 (seed=1234).
- Shared `gsm8k.build_prompt`, `target_of` and `is_correct` are reused. The local
  parquet subset uses the same shuffle-then-sort index rule as `load_subset`.
- CPU float32, Torch 2.5.1+cpu, Transformers 4.49.0, four threads, W=32, at most
  64 generated tokens; no prompt truncation. Inputs were 728-753 tokens.

| Reference method | Target retained KV | Mean final retained KV | Strict accuracy |
| --- | ---: | ---: | ---: |
| Full | 100% | 100.000% | 0/5 |
| H2O-style | 10% | 9.934% | 0/5 |
| H2O-style | 15% | 14.929% | 0/5 |
| H2O-style | 20% | 19.955% | 0/5 |
| H2O-style | 25% | 24.962% | 0/5 |

Flexible accuracy was also 0/5 in every cell. This tiny, capped run gives no
useful evidence of an accuracy advantage. It validates the evaluation path,
not model capability, research superiority or the full TieredKV proposal.

The 100% reference was token-identical to stock HF generation on all five test
prompts. All 25 per-problem retained-KV budgets and all five cell averages passed
the one-percentage-point audit. Every retained-KV sample stayed below its byte
ceiling. Token-count rounding explains the slight undershoot.

## Memory and timing caveats

Numerator and denominator use the SAME float32 dtype in this reference. The
shared compression baseline uses fp16/CSR accounting, so these rows must NOT be
mixed into the shared accuracy-vs-memory comparison. The reference writes its
own `reference_results.csv`, never shared `results/results.csv`.

Score/count/position arrays are reported separately from KV payload. Full KV is
materialized during prefill; retained-cache savings do NOT demonstrate reduced
peak GPU/process memory. Timings include reference bookkeeping, and development
tests ran concurrently during part of the run. Do not report these timings as a
controlled throughput benchmark or a production speedup.

Greedy generation stops at EOS or the 64-token cap. Shared stop-string truncation
is applied for answer scoring; generation-level stop-string stopping from the
shared runner is not implemented here. Therefore runtime comparisons to that
runner are also inappropriate.

## Validation and provenance

Affected scoring, allocator, reference and integration tests: **127 passed,
1 skipped** (the shared real-target-model integration test was not selected).
Tests cover full-budget equivalence, absolute positions after pruning, exact
survivor rotated K vectors, recency, logical-history budgets, mask visibility,
EOS stopping, context/batch guards and dataset hashes/train-only exemplars.

The output manifest contains model/data revisions, file hashes, source hashes,
test/exemplar indices, prompt hashes, dtype, settings, predictions and audit
results. Source fingerprints were checked against the executed working files;
the run preceded its source commit.

## Reproduce

Prepare local model weights and the two pinned parquet files outside git. The
data directory must contain train.parquet, test.parquet and source.json with
the dataset revision plus per-split row counts and SHA256 hashes. Then run:

```bash
python -m tieredkv.h2o_reference_eval --model LOCAL_MODEL_DIR --model-id HuggingFaceTB/SmolLM2-135M-Instruct --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --data LOCAL_DATA_DIR --out LOCAL_OUTPUT_DIR --n 5 --window 32 --max-new-tokens 64 --threads 4
python -m tieredkv.budget_audit LOCAL_OUTPUT_DIR/reference_results.csv
```

Next: validate this baseline on the agreed capable target model and fixed larger
subset, then compare uniform/tiered compression once the shared generation cache
is wired. The 3/5/10% threshold ablation still needs that compressed path and a
matched dictionary. No Lexico/TieredKV accuracy rows are fabricated here.
