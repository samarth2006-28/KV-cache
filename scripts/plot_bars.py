"""Bar chart for slides: flexible accuracy per method at one memory budget, with 95% Wilson CIs.

    python scripts/plot_bars.py --csv results/final/results.csv --budget 0.2 --out results/final/plots/acc_bars

Full cache is drawn as a dashed reference line, not a bar. Ablations of TieredKV (tiered_kv,
tiered_evict, ...) at the same budget get TieredKV's colour with hatching. Rows that failed
the budget audit keep their bar and are marked with an asterisk.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tieredkv.gsm8k import wilson  # noqa: E402

COLOR = {"tiered": "#2a78d6", "lexico": "#eb6834", "h2o": "#1baf7a"}     # same as plot.py
LABEL = {"tiered": "TieredKV\n(ours)", "lexico": "Lexico\n(uniform)", "h2o": "H₂O\n(eviction)",
         "tiered_kv": "TieredKV-KV\n(keys only)", "tiered_evict": "TieredKV\nevict light"}
ORDER = ["tiered", "tiered_kv", "tiered_evict", "lexico", "h2o"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path, budget, metric):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if abs(float(r["budget_target"]) - budget) < 1e-9 or r["method"] == "full":
                rows[r["method"]] = {"acc": float(r[metric]), "n": int(float(r["n"])),
                                     "measured": float(r["budget_measured"]),
                                     "ok": str(r.get("eligible", "True")).lower() in ("true", "1")}
    return rows


def plot(rows, budget, metric, out):
    methods = [m for m in ORDER if m in rows] + sorted(m for m in rows if m not in ORDER + ["full"])
    fig, ax = plt.subplots(figsize=(1.55 * len(methods) + 2.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=10, length=0)

    for i, m in enumerate(methods):
        r = rows[m]
        k = round(r["acc"] * r["n"])
        lo, hi = wilson(k, r["n"])
        base = m.split("_")[0]
        ax.bar(i, 100 * r["acc"], width=0.62, color=COLOR.get(base, COLOR["tiered"]),
               hatch="///" if m != base else None, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(i, 100 * r["acc"], yerr=[[100 * (r["acc"] - lo)], [100 * (hi - r["acc"])]],
                    color=INK2, elinewidth=1.2, capsize=0, zorder=3)
        star = "" if r["ok"] else "*"
        ax.annotate(f"{100 * r['acc']:.1f}%{star}", (i, 100 * hi), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.annotate(f"{100 * r['measured']:.1f}% mem", (i, 0), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=SURFACE, fontsize=8)
    if "full" in rows:
        f = rows["full"]
        ax.axhline(100 * f["acc"], color=INK2, linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
        ax.annotate(f"Full KV cache {100 * f['acc']:.1f}%", (len(methods) - 0.5, 100 * f["acc"]),
                    xytext=(0, 4), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.set_xticks(range(len(methods)), [LABEL.get(m, m) for m in methods], color=INK)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"GSM8K accuracy, {'flexible' if metric == 'acc_flex' else 'strict'} (%)", color=INK2)
    n = sorted({r["n"] for r in rows.values()})
    ax.set_title(f"Llama-3.2-3B at {100 * budget:.0f}% KV memory  (n={'/'.join(map(str, n))}, 95% CI)",
                 color=INK, fontsize=12, loc="left")
    if any(not r["ok"] for r in rows.values()):
        fig.text(0.01, 0.01, "* some problems more than 1 point over the memory target (budget audit)",
                 color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default="results/final/results.csv")
    p.add_argument("--budget", type=float, default=0.2)
    p.add_argument("--metric", choices=("acc_flex", "acc"), default="acc_flex")
    p.add_argument("--out", default="results/final/plots/acc_bars")
    a = p.parse_args()
    plot(load(a.csv, a.budget, a.metric), a.budget, a.metric, a.out)
    print(f"wrote {a.out}.png / .svg")


if __name__ == "__main__":
    main()
