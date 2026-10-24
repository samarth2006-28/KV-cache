"""Paired comparison of two methods on the same GSM8K problems (exact McNemar test).

Reads the per-problem prediction files eval.run writes (results/<dir>/runs/*.jsonl), pairs
the two methods by problem idx at each budget, and reports both metrics.

    python -m tieredkv.compare --runs results/final/runs --a tiered --b lexico
    python -m tieredkv.compare --runs results/final/runs --a tiered --b h2o --budgets 0.2

p < 0.05 means the two methods really differ on these problems. a_only / b_only are the
discordant pairs: problems only A solved / only B solved.
"""
import argparse
import glob
import json
import os
import re

from .gsm8k import mcnemar

FIELDS = {"flexible": "flexible", "strict": "strict"}
_TAG = re.compile(r"^(?P<task>[^_]+)_(?P<method>.+)_b(?P<budget>[0-9.]+)_n(?P<n>\d+)_s(?P<seed>\d+)")


def find_runs(runs_dir, method, task="gsm8k"):
    """{budget: latest jsonl path} for one method (exact method name, not its ablations)."""
    out = {}
    for path in glob.glob(os.path.join(runs_dir, f"{task}_{method}_b*.jsonl")):
        m = _TAG.match(os.path.basename(path))
        if not m or m["method"] != method:
            continue
        b = float(m["budget"])
        if b not in out or os.path.getmtime(path) > os.path.getmtime(out[b]):
            out[b] = path
    return out


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, f)}


def compare(a_preds, b_preds, field):
    """Pair on shared problems. Returns counts, accuracies and the McNemar p-value."""
    shared = sorted(set(a_preds) & set(b_preds))
    a = [bool(a_preds[i][field]) for i in shared]
    b = [bool(b_preds[i][field]) for i in shared]
    r = mcnemar(a, b)
    return {"n": len(shared), "acc_a": sum(a) / len(shared) if shared else None,
            "acc_b": sum(b) / len(shared) if shared else None, **r}


def report(runs_dir, a, b, budgets=None, metrics=("flexible", "strict")):
    ra, rb = find_runs(runs_dir, a), find_runs(runs_dir, b)
    rows = []
    for budget in sorted(budgets or set(ra) & set(rb)):
        if budget not in ra or budget not in rb:
            raise ValueError(f"no {a} or {b} run at budget {budget:g} in {runs_dir}")
        pa, pb = load_preds(ra[budget]), load_preds(rb[budget])
        for metric in metrics:
            rows.append({"budget": budget, "metric": metric, **compare(pa, pb, FIELDS[metric])})
    return rows


def markdown(rows, a, b):
    lines = [f"| budget | metric | n | {a} % | {b} % | only {a} | only {b} | McNemar p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['budget'] * 100:g}% | {r['metric']} | {r['n']} | {r['acc_a'] * 100:.1f} | "
                     f"{r['acc_b'] * 100:.1f} | {r['a_only']} | {r['b_only']} | {r['p']:.3g} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="results/runs", help="directory of per-problem jsonl files")
    p.add_argument("--a", default="tiered")
    p.add_argument("--b", default="lexico")
    p.add_argument("--budgets", type=float, nargs="*", help="default: every budget both have")
    p.add_argument("--json", help="also write the rows here")
    args = p.parse_args()
    rows = report(args.runs, args.a, args.b, args.budgets)
    print(markdown(rows, args.a, args.b))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
