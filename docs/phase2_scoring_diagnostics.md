# Phase 2: pretrained-model scoring diagnostics

This is a scoring diagnostic, not a GSM8K/LongBench benchmark, compressed-cache
generation run, or result for the team's target 1B/8B model.

## Run

- Model: HuggingFaceTB/SmolLM2-135M-Instruct, pretrained instruction weights.
- Revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Official model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Corpus: the 20 fixed diagnostic strings in `configs/diagnostic_prompts.json`.
- Model-native chat template; 37-54 input tokens; at most 8 new tokens per prompt.
- Python 3.10, Torch 2.5.1+cpu, Transformers 4.49.0, CPU float32, four threads,
  seed 0, eager attention, unpadded single-prompt calls.
- 20 score dumps, 30 layers each: 600 prompt-layer records. These are repeated
  measurements within 20 prompts, not 600 independent prompts.

The diagnostic manifest records source fingerprints, model revision, prompt
hash, dtype, versions, generation lengths and whether the code was uncommitted
at execution time. The run occurred before this source commit; source hashes
identify the actual scoring implementation used.

## Observed descriptive statistics

| Quantity | Mean across prompt-layer records |
| --- | ---: |
| Raw attention mass held by the raw-score top group | 64.64% |
| Normalized score held by the normalized-score top group | 48.74% |
| Overlap of raw-score and normalized-score top groups | 47.40% |
| Spearman correlation: position versus raw score | -0.3923 |
| Spearman correlation: position versus normalized score | +0.4646 |

The top group uses ceil(5% * token count). On these short prompts, that is
actually 5.00%-6.82% of tokens. The second row is normalized SCORE share, not
raw attention probability mass, and must not be compared with row 1 as if it
were the same quantity. All chat-template/special tokens are included.

Normalization materially changes priority assignments. However, the position
association switches from favoring earlier positions to favoring later ones;
this does NOT show that mean scores are free of position effects or better
predict downstream importance. The small recent-token exposure counts can also
make averages sensitive to individual queries. A controlled sum-versus-mean
allocation experiment is warranted before claiming a quality benefit.

## Checks

All 20 dumps were reopened with weights_only=True. All 600 normalized arrays
equal raw sum / opportunity count. Each layer's total accumulated attention
matches its number of scored causal query positions. Source fingerprints match
the recorded implementation. The figure was rendered and visually checked.

Regression suite for the affected scoring/allocator/integration paths:
**119 passed, 1 skipped** (the target-model integration test requires an actual
model selection). No benchmark accuracy, speedup or compression gain is claimed.

## Reproduce

Download the official model revision into a local directory first, keeping
weights outside git. Then, from the repository root:

```bash
python -m tieredkv.score_diagnostics --model LOCAL_MODEL_DIR --model-id HuggingFaceTB/SmolLM2-135M-Instruct --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --prompts configs/diagnostic_prompts.json --out LOCAL_OUTPUT_DIR --mode mean --n 20 --device cpu --threads 4 --chat-template
python -m tieredkv.score_report LOCAL_OUTPUT_DIR/summary.json --out LOCAL_OUTPUT_DIR
```

The collector saves raw sums and counts alongside selected scores, allowing
paired sum/mean comparisons from one generation run. The figure plots per-layer
mean position correlations and top-group overlap. It is descriptive; no
confidence interval or causal interpretation is implied.

## What blocks the real matrix

At the inspected main revision, TieredKVCache rejects lexico/h2o/tiered with
NotImplementedError. Only the full-cache path is wired. The shared compressed
generation path and matched model dictionaries must be available before H2O
and tiered task-accuracy comparisons can run.

These diagnostic prompts are deliberately short; W=32 would itself exceed the
10%-25% budgets on many of them. They must not be reused as a supposedly
memory-matched accuracy matrix. Use the shared fixed GSM8K subset instead.

Next handoff: run these scoring checks on the agreed target model, pair scores
with Samarth's reconstruction errors, then run the H2O budget matrix and the
3/5/10% threshold ablation through the integrated cache. Add a sum/mean scoring
control in light of the observed positional shift. Raw score tensors, model
weights and the generated plots are kept out of this source commit.
