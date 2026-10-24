# Slides: method + memory math (Samarth)

Draft content for the deck. Numbers marked **[TBD]** are filled from `results/sanity/check_dict.json` and `results/results.csv` once those runs finish.

---

## Slide 1: How we compress one token (Lexico's engine)

- An LLM stores a **key** and a **value** vector (128 numbers each) for every past token, in every layer and every KV head. That is the KV cache, and it's what fills GPU memory.
- We learn a **dictionary** of 4,096 "atom" vectors per layer, once, offline.
- Each stored vector becomes a sum of a few atoms: `k ≈ c₁·atom₁₇ + c₂·atom₉₀₂ + … (s atoms)`.
- We store only **s pairs (atom id, coefficient)** instead of 128 numbers. Atoms are chosen by Orthogonal Matching Pursuit (greedy: pick the best atom, refit, repeat).
- **More atoms = more accurate and more bytes.** Our measured error for s = 2 / 4 / 8 / 16 / 32 is **[TBD]**, shown in the `error_by_s.png` plot.

Visual: one vector → a stacked bar of 8 coloured atoms.

---

## Slide 2: What we change: atoms follow attention

- **Lexico:** every token gets the same s.
- **H₂O:** under 5% of tokens receive most of the attention. We measured this on Llama-3.2: **the top 5% of tokens hold about 70% of the attention.**
- **TieredKV:** rank tokens by attention, then spend atoms where attention goes.

| Tier | Share of tokens | Atoms (example, 15% budget) |
|---|---|---|
| Heavy | top 5% | 43 |
| Mid | next 20% | 15–16 |
| Light | bottom 75% | 5 |
| **Average** | | **9, the same as Lexico's uniform s = 9** |

- **We use mean attention, not summed.** Summed attention mostly picks the earliest tokens: in our measurement, 50% of the "important" tokens were in the first 10% of positions. Mean attention cuts that to 31%.

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

- **Attention-weighted reconstruction error.** This measures how much of the attention lands on badly reconstructed tokens.
  - Uniform s = 8: **[TBD]**
  - Tiered (same bytes): **[TBD]**
  - If tiered is lower, the model "sees" less error, even before measuring accuracy.
- **Are important tokens harder to compress?** Spearman correlation between attention and error: **[TBD]**.
  - If it's around 0, the reason to tier is that their errors matter more, not that they're harder to compress. Be honest about which it is.

---

## Slide 5: Results (filled from `plot.py`)

- `results/plots/acc_vs_budget.png`: accuracy vs measured memory for Full, Lexico, H₂O and TieredKV, with 95% CIs.
- **Significance:** paired McNemar test on the same 100 problems, tiered vs Lexico at each budget: **[TBD]**.
- **Honest costs:** compression and decoding make generation slower. Tokens/s and ms per token are in the table.

---

## Likely judge questions

- **Q: Isn't this just H₂O followed by Lexico?**
  A: No. H₂O deletes tokens, and Lexico treats all tokens the same. We keep every token and use attention to decide each token's precision. Importance-aware precision already exists for quantization (MiKV, ZipCache), but not for dictionary atoms.
- **Q: Where did the dictionaries come from?**
  A: Lexico doesn't release any. We trained our own with Lexico's objective, on WikiText (no GSM8K test data), in 45 minutes on a laptop GPU.
- **Q: Why 3B, not 8B?**
  A: 8B needs 16 GB in fp16, and we had an 8 GB GPU. 3B is the model in Lexico's own example code.
