# Experiment results (Samarth's RTX 4060 laptop, 8 GB)

All runs: GSM8K test subset `load_subset(n, seed=0)`, 5-shot, greedy, W = 32, C = 8, batch 1,
transformers 4.49.0. Models are the ungated `unsloth/` copies of Meta's Llama-3.2 Instruct weights.
"strict" = `#### N` match; "flexible" = last number in the answer (team headline metric).
Memory = retained dense + CSR bytes / full fp16 KV. `eligible=False` = runner's per-problem
or aggregate budget audit failed (any problem more than 1 point from the target).

## 3B, the main model (`unsloth/Llama-3.2-3B-Instruct`, revision 006f5dcd13)

| Folder | Code | Cell | n | max_new | Strict | Flexible | Measured | s/problem | tok/s | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| `final/` | fd96000 (= c-eval 8345d31 code + plot/compare only) | lexico 20% | 40 | 192 | 20.0% | **32.5%** | 20.55% | 25.06 | 3.84 | False (8/40 problems over) |
| `final/` | fd96000 | tiered 20% | 40 | 192 | — | — | — | — | — | **Crashed: stopped by hand at 34/40, no output** |
| `3b_timing/` | 8345d31 | lexico 20% | 5 | 256 | 0% | 0% | 21.07% | 20.36 | 3.86 | False (2/5 over) |
| `3b_timing/` | 8345d31 | tiered 20% | 5 | 256 | 0% | **80%** | 20.61% | 36.16 | 2.31 | True |
| `smoke_Llama-3.2-3B-Instruct.csv` | 69ae372 | full | 10 | 256 | 50% | 60% | 100% | 5.33 | 13.11 | n/a (pre-audit) |

Full (n = 40), h2o 20% and the 25% cells were queued after tiered 20% and never ran.

## 1B checks (`unsloth/Llama-3.2-1B-Instruct`)

| Folder | Code | Cell | n | Strict | Flexible | Measured | s/problem | tok/s | Eligible |
|---|---|---|---|---|---|---|---|---|---|
| `combined/` | 8345d31 | lexico 20% | 5 | 0% | 0% | 20.85% | 16.47 | 7.25 | False |
| `combined/` | 8345d31 | tiered 20% | 5 | 0% | 0% | 20.57% | 25.40 | 6.27 | False |
| `combined/` | 8345d31 | h2o 20% | 5 | 0% | 0% | 20.07% | 12.31 | 18.05 | True |
| `speed/` | 81c0b96 + decode-once | lexico 20% | 5 | 0% | 0% | 18.46% | 15.31 | 15.76 | n/a |
| `speed/` | 81c0b96 + decode-once | tiered 20% | 5 | 0% | 0% | 18.83% | 17.67 | 8.30 | n/a |
| `results.csv` | 81c0b96 | full | 20 | 40% | 40% | 100% | 3.84 | 30.84 | n/a |
| `results.csv` | 81c0b96 | lexico 20% | 5 | 0% | 0% | 18.46% | 259.53 | 0.84 | n/a (pre decode-once) |
| `results.csv` | 81c0b96 | tiered 20% | 5 | 0% | 0% | 18.83% | 77.80 | 1.73 | n/a (pre decode-once) |
| `results.csv` | 81c0b96 | h2o 20% | 5 | 0% | 0% | 20.07% | 15.46 | 14.27 | n/a |
| `smoke_*1B*.csv` | b838513 / 69ae372 | full | 10 | 70% | 70% | 100% | 8.28 / 3.22 | 7.83 / 20.15 | n/a |

`speed/` vs `results.csv` lexico/tiered: identical answers (10/10), only the decode-once speed differs.
`runs/gsm8k_full_b1_n10_s0.jsonl` holds the **3B** smoke run's answers (the 1B n=10 file was
overwritten by it). `plots/` was made from `results.csv` (1B, mixed n) and is not a final figure.
