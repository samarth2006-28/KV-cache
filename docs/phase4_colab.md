# Phase 4 GPU experiment runner

Status: implementation and local validation completed; the target-model accuracy
experiment has NOT run here. Sampreeth selected preparation of the Colab runner
because the trained 3B .pt dictionary and remote GPU runtime were not available.

Open notebooks/phase4_thresholds_colab.ipynb after pushing b-allocator-phase4-final.
The notebook uses unsloth/Llama-3.2-3B-Instruct to match the committed training
metadata. Supply the team's existing .pt and adjacent training .json on Drive.

## What is improved

Validation: **256 tests passed, 1 target-model test skipped** on Torch 2.5.1+cpu
and Transformers 4.49.0. All eight notebook code cells parse successfully, with
no prefilled benchmark outputs. Controller tests mock GPU work while exercising
real checkpointing, independent analysis, plotting, portable paths and archives.
They do not substitute for the target-model run.

The original fixed-uniform diagnostic reached 18.48% projected KV, failing the
nominal 20% target. That historical result is unchanged. The Phase-4 descriptors
now explicitly select budget_policy=strict for ALL THREE thresholds.

Strict policy:
1. Counts existing actual compressed K/V buffers, including terminal CSR pointers.
2. Reserves the dense recent buffer's worst occupancy until the next compression.
3. Converts the remaining byte quota into a fractional avg_s request to the
   canonical allocator. Each stored token still has an integer atom count.
4. Checks actual encoded payload against the quota before retaining it.
5. Requires fp16/bf16 KV, single-token decode, keep-light and tiered allocation.
   Unsupported or infeasible settings fail rather than silently overspend.

At matching sequence lengths, the tested thresholds had exactly identical stored
dense+CSR bytes. Simulated payload tests used real int16/FP8/int32 SparseKV buffers
at prompt lengths 641, 728 and 789, both 64- and 128-dimensional heads, through
multiple decode chunks. All post-compression samples stayed within 19-20%.
A random tiny FP16 model also passed actual encode/decode/generation at that range.
These are correctness checks, not a completed GSM8K accuracy ablation.

The default fixed policy remains unchanged for shared uniform-baseline equivalence.
Do not compare a new strict-policy threshold against an old fixed-policy 5% row:
run the three strict descriptors together. Strict allocation can vary its quota
by compression batch, so this is a separate, explicit budget schedule.

## Notebook execution

The notebook setup explicitly adds the checkout to the kernel import path;
changing folders alone is insufficient in a notebook kernel. Asset checks use a
lightweight module so they do not load stale Torch/Transformers kernel imports.
The experiment runs in an unbuffered child process, streams progress, saves a
complete .log next to its output directory and raises a visible error containing
the original child traceback on failure. Reopen the GitHub notebook after
updating: an existing Drive copy does not automatically receive these changes.

The notebook:
- checks CUDA and requires at least 10 GiB VRAM for this 3B workflow;
- verifies the dictionary training model and optional trusted SHA256;
- resolves one exact model commit and downloads that snapshot;
- uses the pinned GSM8K revision and the same train-only five-shot exemplars;
- runs five real-model full/stock, H2O100/stock and uniform/equal-ratio gates;
- requires the real compression path to be exercised;
- executes 3%, 5% and 10% thresholds with identical settings;
- rescans predictions, rescoring answers and checking prompt/subset/source identity;
- independently audits every problem and each aggregate, including the strict
  upper ceiling even when an overshoot would fit the old one-point tolerance;
- reports strict/flexible accuracy, Wilson intervals and all three paired exact
  McNemar tests with Bonferroni adjustment;
- saves progress, raw predictions, manifests, plots, tables and an archive on Drive.

The dictionary's old training metadata does not record an exact model commit.
The notebook reports that limitation rather than asserting training-revision
identity. When a training commit IS recorded, the run must use that commit.

N defaults to 20. That may produce wide intervals and inconclusive comparisons.
No automatic winner or accuracy-improvement claim is generated. Choose the sample
size before seeing test outcomes. Do not filter failing questions or relabel budgets.

## CLI

Download a pinned model snapshot and place the trained dictionary and training
metadata together, then:

```bash
python -m tieredkv.phase4_experiment --model LOCAL_SNAPSHOT --model-id unsloth/Llama-3.2-3B-Instruct --model-revision EXACT_40_CHARACTER_SHA --dictionary SHARED_DICTIONARY.pt --out NEW_OUTPUT_DIRECTORY --n 20 --seed 0
```

Use a NEW output directory. The program refuses to overwrite prior experiments.
It writes experiment.json while running. Failed gates or mismatched budgets never
set phase4_completed=true. Exit 0 means all three cells passed; exit 1 means
budget failure. Other failures retain their error and available artifacts.

A completed/extracted archive can be independently re-audited without model
weights or a GPU (the repository's Python dependencies are still required):

```bash
python -m tieredkv.phase4_experiment --analyze EXTRACTED_OUTPUT_DIRECTORY
```

Prediction/manifest references in experiment.json are relative, so the archive is
portable. Artifact paths cannot escape the experiment directory.

## Limits that remain

Full prefill and eager attention still materialize full storage temporarily.
The strict budget is retained KV relative to full fp16 KV; it does not include
dictionary weights or tracker metadata, and does not prove lower peak GPU memory.
Tier fractions remain local to each compression batch, not a global continuously
updated top-percentile policy. At C=8, rounded 3%/5% heavy decode groups can be empty.

If generation ends before the first compression, retained KV can still be full;
the final actual-budget audit flags that run. The notebook does not conceal it.
No new target-model accuracy numbers have been fabricated or inferred from tests.
