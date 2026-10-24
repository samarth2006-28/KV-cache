# Sampreeth's allocator handoff

## Canonical implementation

Keep the existing `tieredkv/allocator.py`; no parallel allocator is introduced.
Its public `ScoreTracker(model, mode=...)`, `add`, `scores`, `keep`, `reset`,
`remove`, `allocate(..., bytes_fn=...)` and `h2o_keep_idx` APIs are preserved.
The standalone starter is not copied wholesale into this repository.

TieredKV uses `ScoreTracker(model, mode="mean")`; pure H2O uses `mode="sum"`.
The shared requirements already pin Transformers 4.49.0. This contribution does
not modify the encoder, dictionary trainer, answer scorer, plotting code, cache
or evaluation runner.

## Scoring contract

Mean mode divides each token's raw attention sum by valid causal query
opportunities, not nonzero weights. Both sums and counts survive `keep()` in
the same physical order as K/V. Reset per prompt; do not access `acc` when you
intend normalized scores: use `scores()`.

The automatic hooks assume one unpadded causal prompt, with queries occupying
the last Q of K physical cache rows. For padding, sliding windows or custom
visibility, disable automatic accumulation (`tracker.enabled=False`) and call
`add(layer, attention, valid_mask=boolean_mask)` from the appropriate adapter.
True means visible. Masks may be [B,Q,K], [B,1,Q,K] or [B,H,Q,K]. Allowed zero
weights still count. Explicitly masked queries contribute neither sums nor
counts. `ScoreTracker` now supports `with` to remove hooks even on errors.

## H2O selection and budgets

`tieredkv.h2o_baseline.h2o_keep_idx` re-exports the canonical function rather
than duplicating it. Existing three-argument calls remain supported. For
matched-memory runs use:

```python
idx = h2o_keep_idx(scores, fraction, W,
                   original_length=total_tokens_seen, strict=True)
tracker.keep(layer, idx)
```

Use original token count after prior eviction, not surviving cache length.
Fractional token capacities round down. Strict mode rejects a budget unable to
hold W; legacy mode retains W and warns if this exceeds the budget. For byte
ceilings with fixed overhead, reserve that storage first and pass the resulting
integer dense-token capacity instead. Cache position IDs/masks remain the
integration owner's responsibility.

The inspected c-eval branch uses W=32 and C=32, addressing the screenshot's
W=128/10% conflict. `configs/sampreeth.json` therefore retains four H2O targets
(10/15/20/25%) at W=32. This supersedes the standalone starter's provisional
decision to defer 10% with W=128. Actual feasibility must be checked per prompt.
The threshold variants match c-eval's parser and plot grouping: `tiered_hh3`,
`tiered` (5%) and `tiered_hh10`, all at 20%. No other branch is merged here.

## Diagnostics and evaluation support

```bash
python -m tieredkv.score_diagnostics --model LOCAL_MODEL_DIR --prompts prompts.json --mode mean --n 20 --device cuda
python -m tieredkv.budget_audit results/results.csv --units fraction
```

Diagnostics require a JSON list of at least 20 real prompt strings and existing
local HF weights/tokenizer. Outputs are per-layer score/count tensors and a JSON
summary, not benchmark accuracy. Use mode=sum for raw accumulated attention mass;
mean-mode score share must not be labeled raw attention mass. No model weights,
downloads, score dumps or benchmark results are committed.

Audit exit codes: 0 = recorded rows within one percentage point; 1 = flagged
comparisons; 2 = bad/missing input. Empty CSVs do not pass. The original results
file is never modified. `tieredkv.sanity.spearman` supports aligned attention
scores / reconstruction errors, with tied ranks and undefined-correlation handling.
Compute within prompt/layer before pooling; a correlation is not proof of higher
intrinsic dimensionality. Samarth supplies reconstruction errors.

## Validation and boundaries

Reference checks use Python 3.10, Torch 2.5.1+cpu and Transformers 4.49.0. The
existing allocator suite is retained. Added regressions cover masked mean scores,
eviction/count alignment, chunked prefill, detached/defensive scores, hook cleanup,
strict and repeated H2O selection, budget-audit CLI behavior and real tiny Llama
greedy-generation equivalence in sum and mean modes.

Tiny Llama tests use random weights and check compatibility only. Scores on the
team's target model, H2O matrix runs, threshold ablations and CUDA/memory performance
remain pending dictionary availability and the shared cache/runner integration.

Phase 2 adds a completed small pretrained-model scoring diagnostic; see
`phase2_scoring_diagnostics.md` for the exact model/corpus, findings and caveats.
Target-model scores and compressed-generation task benchmarks remain pending.
