"""Plots + tables from results/results.csv (columns per eval.run contract).

    python plot.py                       # -> results/plots/{acc_vs_budget,ablation}.{png,svg}, table.md
    python plot.py --csv other.csv --out somewhere/

acc and budget columns may be fractions (0.42) or percents (42); both are handled.
Main methods: full, lexico, h2o, tiered. Any other method name (e.g. tiered_hh3,
tiered_24-8-2, tiered_evict) is treated as an ablation variant.
If a (method, budget_target) pair appears more than once, the last row wins.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tieredkv.gsm8k import wilson

MAIN = ["tiered", "lexico", "h2o"]                        # fixed order = fixed colour
COLOR = {"tiered": "#2a78d6", "lexico": "#eb6834", "h2o": "#1baf7a"}
LABEL = {"tiered": "TieredKV (ours)", "lexico": "Lexico (uniform s)", "h2o": "H₂O (eviction)",
         "full": "Full KV"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _pct(x):
    x = float(x)
    return x * 100 if x <= 1.0 else x


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["acc"], r["n"] = _pct(r["acc"]), int(float(r["n"]))
            r["budget_measured"] = _pct(r["budget_measured"])
            r["budget_target"] = _pct(r["budget_target"])
            rows[(r["method"], round(r["budget_target"], 3))] = r
    return list(rows.values())


def _ci(r):
    lo, hi = wilson(round(r["acc"] / 100 * r["n"]), r["n"])
    return r["acc"] - lo * 100, hi * 100 - r["acc"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_main(rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    full = [r for r in rows if r["method"] == "full"]
    if full:
        f = full[-1]
        ax.axhline(f["acc"], color=INK2, linestyle=(0, (4, 3)), linewidth=1.5)
        ax.annotate(f"{LABEL['full']} {f['acc']:.1f}%", (0.0, f["acc"]), xycoords=("axes fraction", "data"),
                    xytext=(4, 4), textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=9)
    for k, m in enumerate(MAIN):
        pts = sorted((r for r in rows if r["method"] == m), key=lambda r: r["budget_measured"])
        if not pts:
            continue
        dodge = (k - 1) * 0.25   # display-only x shift so CIs at the same budget don't overlap
        x, y = [r["budget_measured"] + dodge for r in pts], [r["acc"] for r in pts]
        err = list(zip(*[_ci(r) for r in pts]))
        ax.errorbar(x, y, yerr=err, color=COLOR[m], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=0, elinewidth=1.2,
                    label=LABEL[m], zorder=3)
        ax.annotate(LABEL[m], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xlabel("KV cache memory (% of full, measured)", color=INK2, fontsize=10)
    ax.set_ylabel("GSM8K accuracy (%)", color=INK2, fontsize=10)
    n = sorted({r["n"] for r in rows})
    ax.set_title(f"Accuracy at matched memory  (n={'/'.join(map(str, n))} problems, 95% CI)",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.margins(x=0.12)
    _save(fig, out, "acc_vs_budget")


def plot_ablation(rows, out):
    abl = [r for r in rows if r["method"] not in MAIN + ["full"]]
    if not abl:
        return
    target = abl[0]["budget_target"]
    ref = [r for r in rows if r["method"] in ("tiered", "lexico") and abs(r["budget_target"] - target) < 1e-6]
    items = sorted(abl + ref, key=lambda r: r["acc"])
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(items) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for i, r in enumerate(items):
        c = COLOR.get(r["method"], COLOR["tiered"])
        lo, hi = _ci(r)
        ax.errorbar(r["acc"], i, xerr=[[lo], [hi]], color=c, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=0)
        ax.annotate(f"{r['acc']:.1f}%", (r["acc"] + hi, i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(items)), [LABEL.get(r["method"], r["method"]) for r in items], color=INK)
    ax.set_xlabel("GSM8K accuracy (%)", color=INK2, fontsize=10)
    ax.set_title(f"Ablations at {target:.0f}% memory (95% CI)", color=INK, fontsize=12, loc="left")
    ax.margins(x=0.15)
    _save(fig, out, "ablation")


def table(rows):
    head = "| method | target % | measured % | acc % | 95% CI | KV MB | peak GB | tok/s | ms/tok | n |"
    lines = [head, "|" + "---|" * 10]
    for r in sorted(rows, key=lambda r: (r["method"], r["budget_target"])):
        lo, hi = _ci(r)
        lines.append(f"| {r['method']} | {r['budget_target']:.0f} | {r['budget_measured']:.1f} | "
                     f"{r['acc']:.1f} | {r['acc'] - lo:.1f}–{r['acc'] + hi:.1f} | {r.get('kv_mb', '')} | "
                     f"{r.get('peak_gb', '')} | {r.get('tok_s', '')} | {r.get('ms_tok', '')} | {r['n']} |")
    return "\n".join(lines)


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/results.csv")
    p.add_argument("--out", default="results/plots")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    plot_main(rows, a.out)
    plot_ablation(rows, a.out)
    md = table(rows)
    with open(os.path.join(a.out, "table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
