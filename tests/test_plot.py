import csv
import os
import sys

import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import plot  # noqa: E402

COLS = ["method", "budget_target", "budget_measured", "acc", "kv_mb", "peak_gb", "tok_s", "ms_tok",
        "n", "seed", "git_sha"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def test_plot_end_to_end(tmp_path):
    rows = [["full", 1.0, 1.0, 0.78, 120, 17.1, 20, 50, 200, 0, "abc"]]
    for m, base in (("lexico", 0.62), ("h2o", 0.55), ("tiered", 0.66)):
        for i, b in enumerate((0.10, 0.15, 0.20, 0.25)):
            rows.append([m, b, b + 0.004, base + 0.03 * i, 30 * b, 16.5, 12, 80, 200, 0, "abc"])
    rows += [["tiered_hh3", 0.20, 0.203, 0.70, 6, 16.5, 12, 80, 200, 0, "abc"],
             ["tiered_hh10", 0.20, 0.203, 0.69, 6, 16.5, 12, 80, 200, 0, "abc"],
             ["lexico", 0.20, 0.201, 0.69, 6, 16.5, 12, 80, 200, 0, "def"]]   # rerun: last row wins
    csv_path, out = tmp_path / "r.csv", tmp_path / "plots"
    _write(csv_path, rows)
    sys.argv = ["plot.py", "--csv", str(csv_path), "--out", str(out)]
    plot.main()
    for f in ("acc_vs_budget.png", "acc_vs_budget.svg", "ablation.png", "table.md"):
        assert (out / f).stat().st_size > 0
    md = (out / "table.md").read_text(encoding="utf-8")
    assert md.count("| lexico | 20 |") == 1 and "| lexico | 20 | 20.1 | 69.0 |" in md
