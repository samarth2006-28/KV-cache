# Review notes: what to show and how to explain it

Open these four things: `results/REPORT.md`, `results/final/plots/acc_bars.png`,
`results/final/mcnemar.md`, `results/final/hit_rate.json`. The live demo is `demo/app.py`.

## 1. The idea (30 seconds)

- **The problem:** an LLM stores a key and a value vector for every past token (the KV cache).
  That is what fills GPU memory during long generation.
- **Lexico (ICML 2025):** rebuilds each stored vector from a few "atoms" of a learned dictionary.
  Every token gets the same number of atoms, s.
- **H₂O (NeurIPS 2023):** a few tokens get almost all the attention. It keeps those tokens and
  deletes the rest.
- **Our idea:** keep every token like Lexico, but give the tokens that get attention more atoms
  and the rest fewer, at **exactly the same total bytes**.

## 2. Setup (say it in one breath)

Llama-3.2-3B-Instruct on an 8 GB laptop GPU. 40 fixed GSM8K maths problems, 5-shot, greedy.
Every method runs at 20% of full KV memory, **measured** from the real stored bytes. We trained
our own dictionary (Lexico releases none). All methods run in one codebase: same model, same
problems, same dictionary.

## 3. Results (`acc_bars.png`)

| Method (20% memory) | Accuracy (flexible) | Share of full kept |
|---|---|---|
| Full KV (100%) | 67.5% | 100% |
| **TieredKV (ours)** | **35.0%** | 52% |
| TieredKV-KV (keys tiered, values uniform)* | 20.0% (CI 10.5–34.8%, strict 2.5%, 19.9% memory) | 30% |
| Lexico | 32.5% | 48% |
| H₂O | 10.0% | 15% |

\* tiered_kv ran with expected_new_tokens 64; the others used 192. It is worse than TieredKV
(p = 0.07) and Lexico (p = 0.30), so **tiering keys and values together is our better variant**.
It writes `####` in 6/40 answers (TieredKV 2, Lexico 19, full 35).

- **Compression beats deletion.** TieredKV and Lexico both beat H₂O by more than 3× at the same
  memory. This is statistically significant: McNemar p = 0.002 and p = 0.035.
- **TieredKV vs Lexico: +2.5 points, not significant (p = 1.0)** on 40 problems. Say it
  honestly: "a tie at this sample size". The two methods fail on *different* problems, so the
  allocation changes which problems break.
- **Against the papers:** the full cache matches the Lexico paper's 3B number (~66%), so the
  setup is sound. Our Lexico is lower than theirs at 20% because our dictionary trained for
  45 minutes rather than days. It is the same dictionary for both compressed methods, so the
  comparison is fair. H₂O's paper has no GSM8K numbers; a later paper (ReFreeKV) also finds H₂O
  collapses on GSM8K at 20%.

## 4. Why we believe the mechanism (`hit_rate.json`, 20 problems, full cache)

The key question: if we pick important tokens from the **prompt**, are they still the ones the
model looks at while **writing the answer**?

| Measure | Value | Random would be |
|---|---|---|
| Predicted top-5% tokens that are really top-5% during the answer | **56%** | 5% |
| Share of answer-time attention landing on the predicted top 5% | **86%** | 5% |
| ...on the predicted top 25% (heavy + mid tiers) | **92%** | 25% |
| Answer-time attention on tokens H₂O deletes at 20% | **14%** | — |

**How to say it:** "Attention is very predictable: 5% of the prompt gets 86% of the attention
while answering. That is why spending atoms there makes sense. It also explains H₂O's collapse:
the tokens it deletes would still have received about 14% of the attention."

## 5. The honest finding (good to volunteer)

TieredKV almost never writes the `#### N` answer format, so its strict accuracy is 0%. The
`####` markers in the examples get little attention, so they land in the light tier with about
7 atoms, versus Lexico's 13, and the model stops copying the format.
**Lesson:** attention shows what the next token needs to *read*, not which structural tokens
keep the output well-formed. The next step is to protect format tokens.

## 6. Engineering points (if asked)

- **Memory is measured, not estimated:** CSR storage like Lexico's (int16 ids, fp8 coefficients),
  and a per-problem budget audit that flags any problem more than 1 point off.
- **Fairness test:** TieredKV with equal tiers is bit-identical to Lexico. Any difference comes
  only from the allocation.
- **Speed:** decoding each chunk once made Lexico 17× faster with identical outputs. TieredKV is
  still about 2× slower than Lexico because of attention tracking. This is a prototype; the
  claim is memory vs accuracy, not speed.

## Likely questions

- **"Why not 8B?"** 8B needs 16 GB in fp16; we had an 8 GB GPU. Lexico's own example uses 3B.
- **"Only 40 problems?"** It's one laptop GPU in 24 hours. We report 95% CIs and paired tests,
  and don't claim more than the data shows.
- **"Is this just H₂O + Lexico?"** No. H₂O deletes tokens; we keep all of them and use attention
  to set each token's precision.
