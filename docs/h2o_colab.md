# Sampreeth's target-model H2O runs

Status: runner and local correctness checks implemented. No target-model results
are claimed. This closes the missing launch/audit workflow, not the GPU experiment.

Open `notebooks/h2o_matrix_colab.ipynb` on branch `b-allocator-phase4-final`.
Select a GPU runtime. The default 3B workflow requires at least 10 GiB VRAM.
**No dictionary is needed**, so H2O can run before the compression assets arrive.

1. Run setup and mount Drive.
2. Set the agreed exact model commit, sample size, seed and token cap. Use the
   same values in the Phase 4 notebook; do not choose them after seeing answers.
3. Run download, then the experiment cell.
4. Inspect the report and download its zip. Failed runs retain their traceback,
   available predictions and checkpoint. Retry into a new directory.

The runner first records a twenty-prompt target-model prefill attention probe
using the seeded fixed GSM8K selection and the same train-only five-shot examples.
It saves raw sums, valid query counts, normalized scores and prompt/source/model
provenance. Raw top-group attention mass and normalized score share stay separate.
This probe uses no truncation or generation; it does not measure reconstruction
error or task accuracy. Its twenty prompts do not depend on the matrix's N.

The runner loads the model once and runs full plus H2O at 10/15/20/25%, with
W=32, C=8, raw sum-mode attention and shared GSM8K scoring/stopping. Five target
prompts first check full against stock HF and shared H2O against the independent
reference at 100% and each of the four budgets. Gate decoding has no stop strings
on either path; accuracy runs use the shared stop strings and token cap.

Each cell writes unique raw predictions, a source/model/data manifest and a CSV
row. The independent analyzer re-scores text, checks exact cohort/settings
identity and audits each question plus the aggregate within one percentage point
of the nominal budget. All five cells must pass before `h2o_completed=true`.
Wilson intervals and four paired full-vs-H2O exact McNemar tests (Bonferroni
adjusted) are provided only for a fully eligible matrix. Small samples can remain
inconclusive. No superiority statement is generated.

The shared cache keeps the current token after pruning prior history; it may
temporarily exceed the nominal H2O fraction by a token. This runner follows the
plan's one-percentage-point final-budget audit, not the Phase 4 strict ceiling.
It does not claim that full prefill or attention temporary buffers fit that KV
budget. Dictionary/tracker/model/allocator overhead is not included in KV bytes;
reported peak VRAM is a separate quantity. The independent reference's final
pruning point differs, so gates compare token output rather than cache size.

CLI after downloading a pinned snapshot:

```bash
python -m tieredkv.h2o_experiment --model LOCAL_SNAPSHOT --model-id unsloth/Llama-3.2-3B-Instruct --model-revision EXACT_40_CHARACTER_SHA --out NEW_OUTPUT_DIRECTORY --n 20 --seed 0 --max-new-tokens 256
```

Portable re-audit without a GPU or weights (Python dependencies still required):

```bash
python -m tieredkv.h2o_experiment --analyze EXTRACTED_OUTPUT_DIRECTORY
```

N=20 is a small default, not the team's final sample-size decision. The notebook
does not fix Lexico's speed/budget issues, provide their baseline, or complete
the whole-team matrix. Those remain separate integration requirements.
