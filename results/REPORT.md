# TieredKV results (Llama-3.2-3B-Instruct, GSM8K)

Data: `results/final/` (pushed from the RTX 4060 runs). Figure: `results/final/plots/acc_bars.png`
(flexible) and `acc_bars_strict.png`. Paired tests: `results/final/mcnemar.md`.

## Setup

| | |
|---|---|
| Model | `unsloth/Llama-3.2-3B-Instruct` (rev 006f5dcd), fp16, eager attention |
| Benchmark | GSM8K test, fixed subset n = 40 (seed 0), 5-shot (lm-eval format), greedy, max 192 new tokens |
| Memory | measured retained KV bytes (dense + CSR) / full fp16 KV; target 20% |
| Cache | recent window W = 32 dense, decode chunk C = 8, dictionary N = 4096 atoms per layer (K and V), trained 45 min on WikiText |
| Allocation | tiers by opportunity-normalised H2O score: top 5% / next 20% / rest, ratios 32/12/4, bytes matched to Lexico |
| Metric | flexible match (last number) is the headline: Llama-3.2-3B often answers "The answer is N" instead of "#### N" |

All four runs share the same 40 problems, model revision, dictionary file (sha bbe103e6) and
`cache.py` (sha 212f29cf). Re-scoring all 160 saved answers reproduces every recorded score.

## Main result: 20% KV memory, n = 40

| Method | Memory | Flexible | 95% CI | Strict | s / problem |
|---|---|---|---|---|---|
| Full KV cache | 100% | **67.5%** (27/40) | 52–80% | 57.5% | 5.5 |
| **TieredKV** | 20.5% | **35.0%** (14/40) | 22–51% | 0.0% | 48.9 |
| Lexico (uniform) | 20.6% | **32.5%** (13/40) | 20–48% | 20.0% | 25.1 |
| H₂O (eviction) | 20.0% | **10.0%** (4/40) | 4–23% | 5.0% | 10.0 |

Paired exact McNemar, flexible:

| Comparison | Only A right | Only B right | p |
|---|---|---|---|
| TieredKV vs Lexico | 9 | 8 | 1.00 |
| TieredKV vs H₂O | 10 | 0 | **0.002** |
| Lexico vs H₂O | 12 | 3 | **0.035** |
| TieredKV vs full | 1 | 14 | **0.001** |

**Findings**

1. **Compression beats eviction.** Keeping every token as a few dictionary atoms (Lexico,
   TieredKV) beats deleting 80% of them (H₂O) by 22–25 points at the same memory, significant
   for both. Lexico's paper shows the same for SnapKV/PyramidKV on 3B.
2. **TieredKV ≈ Lexico at n = 40.** +2.5 points, not significant (p = 1.0). The two methods
   fail on different problems (9 vs 8 discordant; of the 27 the full cache solves, TieredKV
   keeps 13, Lexico 11, both only 5), so the allocation changes *which* problems break more
   than *how many*.
3. **Attention-guided allocation starves formatting tokens.** TieredKV writes the "#### N"
   answer format in 2/40 answers (full: 35/40, Lexico: 19/40), hence strict 0%. The few-shot
   examples' "####" markers get little attention, so they land in the light tier (~7 atoms vs
   Lexico's ~13) and the model stops copying the format. Attention predicts what the next
   token needs to *read*, not which structural tokens keep the output well-formed.
4. **Speed.** In this prototype compression is slow: Lexico 4.5x and TieredKV 8.8x the full
   cache's time per problem (batched OMP on 8-token chunks, plus per-chunk allocation).

## Against the papers

Lexico (ICML 2025), Llama-3.2-3B-Instruct, GSM8K 5-shot, read from its Figure 1 (±1 point):

| KV memory | Lexico paper | This work |
|---|---|---|
| 100% (full) | ~66 | 67.5 (flexible) |
| ~21% | ~64.5 | Lexico 32.5, TieredKV 35.0 |
| ~19% | ~63 | |

The full cache matches the paper, so the setup is sound. Our Lexico is far below theirs at the
same memory: the paper's 64.5 is outside our 95% CI (20–48%). The likely causes, in order:
our dictionary trained 45 minutes on a laptop (reconstruction error at s = 8: keys 0.40,
values 0.59); a 32-token dense window instead of their 128; n = 40 instead of 1,319. Both of
our compressed methods use the same dictionary, window and problems, so their comparison is fair.

H₂O (NeurIPS 2023) reports no GSM8K numbers (OPT/LLaMA-1 on COPA, PiQA, XSUM, ...). Our 10%
at 20% memory is new data; Lexico's paper shows eviction collapsing on GSM8K for 3B as well.

## Budget audit

Per problem, memory may differ from the target by at most 1 point. 8/40 Lexico and 3/40 TieredKV
problems were slightly over (mean memory 20.5–20.6%): these runs planned for 192 generated
tokens while answers average ~55. Current code plans for 64 (`--expected-new-tokens`), which
keeps every answer length within 1 point in simulation. TieredKV used slightly *less* memory than
Lexico on every problem, so its result is not bought with extra bytes.

## Hit rate: does prompt attention predict answer-time attention?

`results/final/hit_rate.json`: full cache, 20 problems, 28 layers, ~80 generated tokens each.
Prompt tokens are ranked once from prefill attention (as TieredKV does), then compared with the
attention they actually receive while the answer is generated.

| Measure | Value | Random |
|---|---|---|
| Predicted top-5% that are actually top-5% during the answer (precision) | **56%** | 5% |
| Answer-time attention landing on the predicted top 5% | **86%** | 5% |
| ...on the predicted top 25% (heavy + mid tiers) | **92%** | 25% |
| Answer-time attention on prompt tokens H₂O evicts first, at 10 / 15 / 20 / 25% | 17 / 15 / **14** / 12% | |

Coverage is high in every layer (72–83% for the top 5%), and precision ranges 44–68% across
problems. **Implication:** TieredKV puts its extra atoms where the attention goes, so *picking*
tokens is not the bottleneck. Yet it only ties Lexico, which points to reconstruction quality
(our 45-minute dictionary) and the starved light tier (finding 3) as the limits. For H₂O, losing
only ~14% of attention costs 57 accuracy points: the low-attention tokens still carry facts.
This matches ReFreeKV (arXiv 2502.16886), which reports H₂O keeping ~3% of full-cache GSM8K
performance at a 20% budget.

## tiered_kv (K/V-asymmetric)

Keys are tiered by attention; values get Lexico's uniform atoms. This was motivated by the
dictionary check: key error rises with attention (Spearman +0.53, positive in 140/140 cases), but
value error does not (+0.03). Same model, problems, dictionary and 20% target as above. One
difference: it ran with `expected_new_tokens` 64 (current code), while the four cells above used
192.

| Method | Memory | Flexible | 95% CI | Strict | Eligible | s / problem |
|---|---|---|---|---|---|---|
| tiered_kv | 19.9% | **20.0%** (8/40) | 10.5–34.8% | 2.5% | True (0/40 problems over) | 127.9* |

\* Includes about one hour when the laptop went to sleep mid-problem (one problem logged
3,547 s). Without that problem it is about 40 s per problem, close to tiered's 48.9.

Paired exact McNemar (`results/final/tiered_kv_vs_*.md`):

| tiered_kv vs lexico | n | tiered_kv % | lexico % | only tiered_kv | only lexico | p |
|---|---|---|---|---|---|---|
| flexible | 40 | 20.0 | 32.5 | 5 | 10 | 0.302 |
| strict | 40 | 2.5 | 20.0 | 1 | 8 | 0.0391 |

| tiered_kv vs tiered | n | tiered_kv % | tiered % | only tiered_kv | only tiered | p |
|---|---|---|---|---|---|---|
| flexible | 40 | 20.0 | 35.0 | 1 | 7 | 0.0703 |
| strict | 40 | 2.5 | 0.0 | 1 | 0 | 1 |

**Format tokens:** answers containing "####" were tiered_kv 6/40, tiered 2/40, Lexico 19/40 and
full 35/40. Giving values uniform atoms restores a little of the format, but not much.

**Did it help? No.** tiered_kv is 15 points below tiered and 12.5 below Lexico on flexible accuracy
(neither difference is significant at n = 40: p = 0.07 and 0.30). It is also the lowest of the
three compressed methods. Tiering the values as well as the keys (plain TieredKV) is the better
of our two variants.

`tiered_evict` (evict the light tier instead of compressing it) was not run, for lack of time.
