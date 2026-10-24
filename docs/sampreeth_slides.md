# Sampreeth: intuition, contribution and evidence

Slide-ready content and speaker notes. Target-model accuracy remains pending;
do not turn pending result slots into measured claims. This is presentation
source, not an assembled final PowerPoint.

## Slide 1 - Spend memory where it matters

- Every old token consumes KV-cache memory, but tokens receive unequal attention.
- Uniform sparse coding gives every token the same number of dictionary atoms.
- TieredKV uses attention scores to distribute a fixed atom/byte allowance.
- More atoms means a richer reconstruction and **less sparsity**.

Speaker: "My part decides how much memory each token receives. Think of a fixed
study budget: important topics get more time, but the total time stays fixed.
Here, important tokens get more dictionary atoms. The encoder then uses those
atom counts to compress the key and value vectors."

## Slide 2 - From attention to integer atom counts

1. Observe eager-attention weights in each layer.
2. Sum over queries and average over attention heads.
3. For TieredKV, divide by valid causal query opportunities; preserve raw sums
   separately for the H2O baseline.
4. Rank tokens in each compression batch: default heavy 5%, next 20%, remainder
   light. Scale the 32/12/4 ratios to the byte allowance and assign integer counts.
5. Preserve token order and keep score/count arrays aligned when tokens are evicted.

Speaker: "An older token has had more chances to receive attention. Our mean mode
accounts for the number of visible queries. It changes the ranking; it does not
prove that all position bias disappears. The output is one integer atom count
per token, per layer, shared across KV heads."

## Slide 3 - What changes relative to our baselines?

| Method | Decision for old tokens |
|---|---|
| Full | Keep all key/value vectors in full precision |
| H2O | Keep recent and high-scoring tokens; delete the rest |
| Uniform sparse coding | Use the same atom count for every old token |
| TieredKV keep-light | Keep old tokens, with atom counts selected by importance |

Speaker: "H2O changes which tokens survive. Our keep-light variant changes the
representation budget of surviving history. Importance-aware KV compression is
an existing research direction. Our implemented contribution is attention-guided
allocation of dictionary atom counts, integrated with a byte audit. We do not
claim that importance-aware precision itself is new."

Primary-source related-work comparison:

| Work | What its authors describe | Position of our implementation |
|---|---|---|
| [MiKV, Yang et al., 2024](https://arxiv.org/abs/2402.18096) | Lower precision for less important KV pairs, higher precision for important pairs | Preserving less important history at lower fidelity is established prior work |
| [ZipCache, He et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html) | Normalized attention accounting for the lower-triangular matrix, with token saliency guiding quantization bit widths | Attention normalization and importance-based precision are established prior work |
| [Lexico, Kim et al., ICML 2025](https://proceedings.mlr.press/v267/kim25ae.html) | Sparse coding over universal dictionaries with OMP and sparsity control | We build on this representation and vary per-token atom counts with our tracker/allocator |

Speaker: "ZipCache is especially relevant because it already normalizes attention
for token saliency. We should credit that connection. Our prototype combines an
attention-based allocation policy with dictionary sparse coding and byte audits.
These comparisons describe the implementation difference, not proof of a
literature-wide first."

## Slide 4 - Evidence we actually have

- Small 135M diagnostic: 20 prompts; rounded top group (5-6.8% of tokens) holds
  64.6% of raw attention mass on average. This is **not a 3B measurement**.
- Team's committed 3B reconstruction probe: 5 prompts, 28 layers, 140 pairs.
  At average s=8, attention-weighted relative error was about 0.38 to 0.20 for
  keys and 0.53 to 0.27 for values, uniform versus tiered.
- Key attention/error Spearman median was about +0.53; value median about +0.03.
- The same prompt's attention chooses and weights the allocation. These are
  reconstruction diagnostics, **not GSM8K accuracy or generation-speed results**.

Speaker: "The reconstruction probe supports testing our allocation idea. It does
not establish that the model answers more questions correctly. That requires the
separate fixed-subset generation experiment. The weak value correlation also
stops us from claiming every important vector is intrinsically harder to compress."

Sources: `scoring_diagnostic_observations.json`,
`../results/sanity/check_dict.json`, `samarth_slides.md`.

## Slide 5 - Fair tests and completion conditions

- Same model revision, question subset, train-only five-shot examples and decoding.
- H2O: 10/15/20/25% memory; full/stock and independent-reference generation gates.
- Threshold ablation: 3/5/10% heavy at a 20% target; identical strict byte policy.
- Actual retained dense+CSR buffers determine budget usage, including CSR pointers.
- Exclude failed budgets. Report accuracy with uncertainty, negative outcomes,
  decode time and separate peak VRAM.
- Target-model H2O and threshold results: **pending successful GPU execution**.

Speaker: "Our code checks the experiment rather than assuming it is fair. Equal
ratios must reproduce the shared uniform path. We audit every question, not just
the average. Unit tests establish implementation behaviour; the target-model
results establish what happened in the experiment."

## Judge questions - short answers

**Why normalize scores?** Older tokens have more causal observations. Mean mode
divides by visible-query count; pure H2O deliberately retains raw accumulation.

**Does 32/12/4 mean those exact counts always?** No. They are relative weights;
the byte allowance rescales them and storage uses integer counts.

**Are the tiers global?** No. They are ranked within each compression batch.
At C=8, integer rounding can leave the 3%/5% heavy groups empty. Prefill batches
are larger. Explain that limitation when interpreting the threshold experiment.

**Is 20% KV the same as 80% less GPU memory?** No. Model weights, attention,
dictionary and tracking metadata, plus transient decoded vectors, also use memory.

**What if accuracy does not improve?** Report the negative result and uncertainty;
use the ablation to investigate the allocation choice. Do not select a winner from
a tiny or budget-ineligible sample.

**What exactly did you build?** Attention tracking and normalization, tier/atom
allocation, H2O selection and its independent reference, diagnostics and budget
audits, and the H2O/threshold experiment controllers. Shared integration fixes
are on the review branch; teammates retain ownership of their core modules.
