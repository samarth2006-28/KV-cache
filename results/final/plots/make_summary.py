"""One-image summary for the review: accuracy at 20% memory + the two live-demo questions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, GRID, SURFACE = "#14213D", "#4A5568", "#E4E3DF", "#FBFBF8"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8A93A3"

fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
fig.text(0.04, 0.93, "TieredKV: accuracy at 20% KV memory", fontsize=26, fontweight="bold", color=INK)
fig.text(0.04, 0.885, "Llama-3.2-3B-Instruct · GSM8K, 40 fixed problems · memory measured from stored bytes",
         fontsize=14, color=INK2)

# Left: bar chart
ax = fig.add_axes([0.05, 0.12, 0.47, 0.68], facecolor=SURFACE)
names = ["TieredKV\n(ours)", "Lexico\n(uniform)", "TieredKV-KV\n(keys only)", "H₂O\n(eviction)"]
vals = [35.0, 32.5, 20.0, 10.0]
lo = [22.1, 20.1, 10.5, 4.0]
hi = [50.5, 48.0, 34.8, 23.1]
cols = [BLUE, ORANGE, BLUE, GREEN]
bars = ax.bar(range(4), vals, color=cols, width=0.6, zorder=3)
bars[2].set_hatch("//"); bars[2].set_edgecolor(SURFACE)
ax.errorbar(range(4), vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]],
            fmt="none", ecolor=INK2, elinewidth=1.4, zorder=4)
for i, v in enumerate(vals):
    ax.text(i, hi[i] + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=17, fontweight="bold", color=INK,
            zorder=5)
ax.axhline(67.5, color=INK2, linestyle=(0, (5, 4)), linewidth=1.6, zorder=2)
ax.text(3.45, 68.5, "Full KV cache 67.5% (100% memory)", ha="right", va="bottom", fontsize=13, color=INK2)
ax.set_xticks(range(4), names, fontsize=14, color=INK)
ax.set_ylim(0, 75)
ax.set_ylabel("Accuracy, final answer correct (%)", fontsize=13, color=INK2)
ax.tick_params(axis="y", colors=INK2, labelsize=12)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.text(0.05, 0.035, "Error bars: 95% CI.  TieredKV vs H₂O p = 0.002 (significant) · TieredKV vs Lexico p = 1.0 (tie).",
         fontsize=12.5, color=INK2)

# Right: demo cards
fig.text(0.58, 0.80, "Live demo (same 20% memory)", fontsize=19, fontweight="bold", color=INK)
demos = [("Amber ran 8 mi, Micah 3.5× that; 52 mi total.\nHow far did Ahito run?", "16",
          [("Full KV", "16", True, "75.8 MB"), ("TieredKV (ours)", "16", True, "15.4 MB"), ("Lexico", "36", False, "15.2 MB")]),
         ("A candle melts 2 cm per hour. How much\nshorter from 1:00 PM to 5:00 PM?", "8",
          [("Full KV", "8", True, "73.2 MB"), ("TieredKV (ours)", "8", True, "14.8 MB"), ("Lexico", "4", False, "14.4 MB")])]
y0 = 0.73
for q, ans, rows in demos:
    box = FancyBboxPatch((0.58, y0 - 0.30), 0.38, 0.30, boxstyle="round,pad=0.008,rounding_size=0.015",
                         transform=fig.transFigure, fc="#FFFFFF", ec=GRID, lw=1.2)
    fig.patches.append(box)
    fig.text(0.60, y0 - 0.025, q, fontsize=13.5, color=INK, va="top", linespacing=1.35)
    fig.text(0.60, y0 - 0.115, f"Correct answer: {ans}", fontsize=12.5, color=INK2, va="top")
    for j, (m, a, ok, mb) in enumerate(rows):
        yy = y0 - 0.165 - j * 0.042
        fig.text(0.60, yy, m, fontsize=14, color=INK, va="center",
                 fontweight="bold" if "ours" in m else "normal")
        fig.text(0.80, yy, f"{a}  {'✓ right' if ok else '✗ wrong'}", fontsize=14, va="center",
                 color="#167a55" if ok else "#c0392b", fontweight="bold")
        fig.text(0.955, yy, mb, fontsize=13, color=INK2, va="center", ha="right")
    y0 -= 0.34

fig.savefig("results/final/plots/summary.png", dpi=150, facecolor=SURFACE)
print("wrote results/final/plots/summary.png")
