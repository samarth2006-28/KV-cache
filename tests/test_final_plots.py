"""--metric / --include-flagged / --allow-mixed-n in plot.py, and tieredkv.compare."""
import csv
import json
import os
import sys

import pytest

from tieredkv import compare as cmp

pytest.importorskip("matplotlib")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import plot  # noqa: E402

COLS = ["method", "budget_target", "budget_measured", "acc", "acc_flex", "kv_mb", "peak_gb", "tok_s",
        "ms_tok", "s_prob", "n", "seed", "task", "model", "git_sha", "run_id", "eligible"]


def _row(method, b, measured, acc, flex, n, eligible="True"):
    return [method, b, measured, acc, flex, 5, 6.3, 3, 300, 20, n, 0, "gsm8k", "m", "x", "r", eligible]


def _csv(path):
    rows = [_row("full", 1.0, 1.0, 0.5, 0.6, 40),
            _row("lexico", 0.2, 0.211, 0.0, 0.1, 40, "False"),
            _row("tiered", 0.2, 0.206, 0.1, 0.5, 40),
            _row("h2o", 0.2, 0.201, 0.0, 0.05, 40),
            _row("lexico", 0.25, 0.258, 0.1, 0.3, 30),
            _row("tiered", 0.25, 0.255, 0.2, 0.55, 30)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def test_mixed_n_needs_flag(tmp_path):
    _csv(tmp_path / "r.csv")
    with pytest.raises(ValueError, match="mixed"):
        plot.load(tmp_path / "r.csv")
    assert len(plot.load(tmp_path / "r.csv", allow_mixed_n=True)) == 6


def test_flex_metric_and_flagged_rows(tmp_path):
    _csv(tmp_path / "r.csv")
    rows = plot.load(tmp_path / "r.csv", metric="acc_flex", allow_mixed_n=True)
    kept, report = plot.comparison_rows(rows)
    assert "lexico" not in {r["method"] for r in kept if r["budget_target"] == 20}
    kept, report = plot.comparison_rows(rows, include_flagged=True)
    lex = next(r for r in kept if r["method"] == "lexico" and r["budget_target"] == 20)
    assert lex["flagged"] and lex["acc"] == pytest.approx(10.0) and lex["acc_strict"] == 0.0
    md = plot.table(kept)
    assert "| lexico | 20 | 21.1 | 10.0 |" in md and "| no |" in md


def test_plot_cli_end_to_end(tmp_path):
    _csv(tmp_path / "r.csv")
    out = tmp_path / "plots"
    sys.argv = ["plot.py", "--csv", str(tmp_path / "r.csv"), "--out", str(out), "--metric", "acc_flex",
                "--include-flagged", "--allow-mixed-n"]
    plot.main()
    assert (out / "acc_vs_budget.png").stat().st_size > 0
    assert json.loads((out / "budget_audit.json").read_text())[1]["eligible"] is False


def _jsonl(path, results):
    with open(path, "w", encoding="utf-8") as f:
        for idx, (strict, flex) in results.items():
            f.write(json.dumps({"idx": idx, "strict": strict, "flexible": flex}) + "\n")


def test_compare_pairs_by_problem(tmp_path):
    _jsonl(tmp_path / "gsm8k_tiered_b0.2_n5_s0_aaa.jsonl",
           {1: (0, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0, 0)})
    _jsonl(tmp_path / "gsm8k_lexico_b0.2_n5_s0_bbb.jsonl",
           {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (0, 0)})
    _jsonl(tmp_path / "gsm8k_tiered_hh3_b0.2_n5_s0_ccc.jsonl", {1: (1, 1)})   # ablation: ignored
    assert set(cmp.find_runs(tmp_path, "tiered")) == {0.2}
    rows = cmp.report(tmp_path, "tiered", "lexico")
    flex = next(r for r in rows if r["metric"] == "flexible")
    assert (flex["n"], flex["a_only"], flex["b_only"]) == (5, 4, 0)
    assert flex["acc_a"] == 0.8 and flex["p"] == pytest.approx(0.125)
    assert next(r for r in rows if r["metric"] == "strict")["p"] == 1.0
    assert "| 20% | flexible | 5 | 80.0 | 0.0 | 4 | 0 | 0.125 |" in cmp.markdown(rows, "tiered", "lexico")
