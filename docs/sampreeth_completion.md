# Sampreeth completion checklist and deck assembly

Current status: implementation and local evidence exist; target-model experiment
evidence and final assembled PowerPoint are pending. Do not mark all phases done.

Latest local verification: 274 tests passed, one optional target-model integration
test skipped, using Torch 2.5.1+cpu and Transformers 4.49.0. The added tests cover
the complete H2O matrix analyzer, mixed-cohort/tampered/failed-budget rejection,
portable bundles, interruption checkpoints and archives, actual tiny Llama
shared-versus-reference generation at all budgets, twenty real tiny-model
attention dumps, and the H2O notebook's executable setup/import path. GPU work
in controller tests is mocked. None of these tests supplies 3B benchmark data.

| Assigned deliverable | Evidence / next action |
|---|---|
| Attention hooks and score tracking | `tieredkv/allocator.py`; local tests and 20-prompt 135M diagnostic |
| Tier assignment and byte rescaling | Canonical allocator; fixed-policy matching and explicit strict HH policy |
| H2O baseline | Canonical selection, independent generation reference, small-model matrix |
| Target-model H2O x four budgets | Launch `notebooks/h2o_matrix_colab.ipynb`; no dictionary needed; real run pending |
| 20-prompt target-model score dump | Built into H2O notebook with raw/mean/count dumps; actual target-model run remains pending |
| Shared reconstruction sanity study | Committed 3B probe is five prompts; the plan's 20-prompt target-model reconstruction study remains pending with Samarth |
| HH threshold 3/5/10% at 20% | `notebooks/phase4_thresholds_colab.ipynb`; requires trained .pt plus JSON and GPU; real run pending |
| Cross-method budget audit | Audit implementation exists; whole-team result CSV not yet available |
| Novelty and intuition slides | `docs/sampreeth_slides.md` provides slide text, speaker notes and primary-source related-work positioning; this is not an exhaustive priority search |
| Final deck assembly | Order below; final PPT requires teammates' final slides/plots and actual eligible results |

## Execution order

1. Push this branch, then reopen each notebook from GitHub (old Drive copies
   do not update). Use GPU runtime, not the laptop's CPU Torch installation.
2. Agree model SHA, N, seed, max-new-tokens before seeing outcomes. H2O can run
   immediately without the dictionary or compressed-cache speed fix.
3. Run the H2O notebook; preserve its Drive zip and log. Require all five cells,
   target-model equality gates and per-problem/aggregate audits to pass.
4. Copy the team's matching .pt and .json to Drive. Enter the same model SHA in
   Phase 4. Resolve the actual failure from its saved traceback, then run all
   three strict thresholds. Require `phase4_completed=true`.
5. Coordinate Samarth's target-model 20-prompt score/reconstruction study. Keep
   raw sum attention mass separate from mean-normalized score share.
6. Give Vishal the shared cache/eval diff for review before merging. The new
   H2O controller does not rewrite his cache or the shared eval runner.
7. Build final plots only from compatible, budget-eligible runs; fill final deck.

## Proposed final deck order

| Slide | Content | Source / owner |
|---|---|---|
| 1 | Problem, task and team | Original proposal; shared |
| 2 | KV-cache memory bottleneck | Samarth method/memory slides |
| 3 | Unequal attention motivates unequal atom counts | Sampreeth slides 1-2 |
| 4 | Architecture and data flow | Cache integration, Vishal |
| 5 | Sparse dictionary representation and byte accounting | Samarth slides 1 and 3 |
| 6 | H2O vs uniform vs TieredKV; related-work positioning | Sampreeth slide 3 |
| 7 | Reconstruction diagnostic, including caveats | Sampreeth slide 4 / Samarth slide 4 |
| 8 | Fixed experimental protocol and equality gates | Sampreeth slide 5 |
| 9 | Main accuracy versus measured-memory plot | Actual eligible whole-team matrix; pending |
| 10 | HH threshold and tier-ratio ablations | Actual eligible ablation plots; pending |
| 11 | Timing, peak memory and live demo | Vishal measured results/demo; pending |
| 12 | Findings, limitations and next steps | Actual observed results; shared |

Do not use old proposal numbers as measured results. The 135M CPU reference
cannot fill a 3B GPU result row. The strict HH schedule is distinct from the
fixed uniform-baseline policy; label it explicitly. A reference equality gate
does not establish reproduction of the external Lexico repository.

## Actual external dependencies

The current environment has CPU Torch and a 6 GiB laptop GPU. The default 3B
notebooks require a CUDA runtime with at least 10 GiB VRAM; this session has no
connected Colab runtime. The trained 3B dictionary file is not in the checkout.
Code readiness cannot substitute for either asset or for successful experiment
artifacts. Main and c-eval have not incorporated these branch changes.
