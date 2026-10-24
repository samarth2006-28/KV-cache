# Slides: method + memory math (Samarth)

Draft content for the deck. Numbers marked **[TBD]** are filled from `results/sanity/check_dict.json` and `results/results.csv` once those runs finish.

---

## Slide 1: How we compress one token (Lexico's engine)

- An LLM stores a **key** and a **value** vector (128 numbers each) for every past token, in every layer and every KV head. That is the KV cache, and it's what fills GPU memory.
- We learn a **dictionary** of 4,096 "atom" vectors per layer, once, offline.
- Each stored vector becomes a sum of a few atoms: `k ≈ c₁·atom₁₇ + c₂·atom₉₀₂ + … (s atoms)`.
- We store only **s pairs (atom id, coefficient)** instead of 128 numbers. Atoms are chosen by Orthogonal Matching Pursuit (greedy: pick the best atom, refit, repeat).
- **More atoms = more accurate and more bytes.** Measured relative error on Llama-3.2-3B (5 GSM8K prompts, about 640 tokens each), plotted in `results/sanity/error_by_s.png`:

| s | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Keys | 0.57 | 0.49 | 0.40 | 0.27 | 0.13 |
| Values | 0.81 | 0.73 | 0.59 | 0.39 | 0.16 |

- Values are harder to compress than keys at every s.
- Our dictionary trained for 45 min. This is our reconstruction probe, not a reproduced comparison against published Lexico results.

Visual: one vector → a stacked bar of 8 coloured atoms.

---

## Slide 2: What we change: atoms follow attention

- **Lexico:** every token gets the same s.
- **Attention concentration:** the separate 135M diagnostic averaged 64.6% raw attention mass in the rounded top group (5-6.8% of tokens). The committed 3B reconstruction report does not record a 70% concentration measurement; do not present one.
- **TieredKV:** rank tokens by attention, then spend atoms where attention goes.

| Tier | Share of tokens | Atoms (example, 15% budget) |
|---|---|---|
| Heavy | top 5% | 43 |
| Mid | next 20% | 15–16 |
| Light | bottom 75% | 5 |
| **Average** | | **9, the same as Lexico's uniform s = 9** |

- **We normalize by causal query opportunities.** Older tokens receive more prefill observations. Mean mode divides accumulated attention by the number of visible queries. It changes rankings but does not guarantee position-independent scores or better task accuracy.

---

## Slide 3: Memory math (why the comparison is fair)

Everything is measured in **bytes**, not atoms:

- Dense token, per layer: `2 (K,V) × 8 heads × 128 dims × 2 bytes = 4,096 B`
- Compressed token at s atoms: `2 × 8 × (s × 3 B + 4 B)`. Each atom costs 3 B (int16 id + fp8 coefficient), plus a 4 B row pointer. At s = 8 that's **448 B, or 11% of dense**.
- Memory % for a whole run: **`(Σ bytes(sᵢ) + W × dense) / (N × dense)`**. This is one formula, used by every method.
- The most recent W = 32 tokens stay uncompressed. With Lexico's default W = 128 this would make 10% impossible, because on GSM8K prompts the window alone is 13% of memory.

Uniform s that fits each budget (N = 1,000 tokens, W = 32):

| Budget | 10% | 15% | 20% | 25% |
|---|---|---|---|---|
| s | 4 | 9 | 13 | 17 |
| Measured | 9.3% | 14.9% | 19.5% | 24.0% |

- **Fairness check, in code:** TieredKV with equal tiers (16/16/16) produces Lexico's s = 16 output, bit-identical. Any gain comes only from **where** the atoms go.
- We report **measured** memory from the real GPU tensors, which are stored without padding. A run more than 1 point off its target is rejected by the budget audit.

---

## Slide 4: Does the idea hold before any accuracy run? (sanity check)

From `python -m tieredkv.check_dict`: Llama-3.2-3B, 5 GSM8K prompts, s = 8.

- **Attention-weighted reconstruction error.** This measures how much of the attention lands on badly reconstructed tokens. Same bytes, average s = 8:

| | Uniform (Lexico) | Tiered (ours) | Change |
|---|---|---|---|
| Keys | 0.38 | 0.20 | **−49%** |
| Values | 0.53 | 0.27 | **−48%** |

  In this small same-prompt diagnostic, the attention-weighted reconstruction metric fell by about half. This does not measure downstream attention/output error or answer accuracy.
  Caveat: this uses the same prompt attention for both choosing and weighting. Generation accuracy is the real test.
- **Are important tokens harder to compress?**
  - **Keys in this probe:** Spearman ρ between attention and error has a median of +0.53, and it's positive in 100% of the 140 (prompt × layer) pairs. Top-5% keys have error 0.50 vs 0.38 for the bottom 75% at the same s.
  - **Values in this probe:** ρ ≈ 0.03, and the error is the same for heavy and light tokens (0.59 vs 0.59).
  - The key result supports the motivation in this probe; the value result is weak. Neither establishes that allocation improves task accuracy.
- **Diagnostic timing:** the committed reconstruction probe reports 1.5 s encoding and 0.10 s decoding per prompt. These are not generation-throughput results. Measure the full decode loop; retaining decoded dense tensors would change the memory cost.

---

## Slide 5: Results (filled from `plot.py`)

- `results/plots/acc_vs_budget.png`: accuracy vs measured memory for Full, Lexico, H₂O and TieredKV, with 95% CIs.
- **Significance:** paired McNemar test on the same 100 problems, tiered vs Lexico at each budget: **[TBD]**.
- **Honest costs:** compression and decoding make generation slower. Tokens/s and ms per token are in the table.

---

## Likely judge questions

- **Q: Isn't this just H₂O followed by Lexico?**
  A: No. H₂O deletes tokens, and Lexico treats all tokens the same. We keep every token and use attention to decide each token's precision. Importance-aware precision already exists for quantization (MiKV, ZipCache), Our implemented change is to allocate sparse-coding atom counts by normalized attention; avoid claiming literature-wide novelty without a documented review.
- **Q: Where did the dictionaries come from?**
  A: Lexico doesn't release any. We trained our own with Lexico's objective, on WikiText (no GSM8K test data), in 45 minutes on a laptop GPU.
- **Q: Why 3B, not 8B?**
  A: 8B needs 16 GB in fp16, and we had an 8 GB GPU. 3B is the model in Lexico's own example code.
