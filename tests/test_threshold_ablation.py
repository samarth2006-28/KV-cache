"""Allocation-ablation contracts, including honest target-budget failures."""
import json
from pathlib import Path

import pytest
import torch

from tieredkv import compress as C
from tieredkv.eval import parse_method
from tieredkv.threshold_ablation import VARIANTS, analyze_scores, summarize


@pytest.fixture(autouse=True)
def model_shape(monkeypatch):
    monkeypatch.setattr(C,"N_KV_HEADS",3)
    monkeypatch.setattr(C,"HEAD_DIM",64)


def test_same_scores_match_uniform_bytes_and_realized_tiers():
    scores = torch.arange(728,dtype=torch.float32)
    rows = analyze_scores(scores)
    assert [r["method"] for r in rows] == list(VARIANTS)
    assert len({r["allocated_old_bytes"] for r in rows}) == 1
    for row, fraction in zip(rows,VARIANTS.values()):
        assert row["heavy_tokens"] == int(696*fraction+.5)
        assert row["mid_tokens"] == int(696*.20+.5)
        assert row["old_tokens"]+row["recent_dense_tokens"] == 728
        assert sum(row["atom_histogram"].values()) == 696
        assert row["monotone"] and row["equal_ratio_is_uniform"]
        assert row["allocated_old_bytes"] == row["uniform_old_bytes"]
        assert row["heavy_mean_s"] >= row["mid_mean_s"] >= row["light_mean_s"]
        # Integer s=5 at head_dim=64 is below the allowed 19% lower bound.
        assert not row["within_target_one_pp"]
        assert row["target_gap_pp"] > 1
        assert "acc" not in row and "budget_measured" not in row


def test_recent_scores_do_not_change_old_allocation():
    scores = torch.linspace(0,1,1000)
    rows = analyze_scores(scores)
    scores[-32:] = 1e10
    assert rows == analyze_scores(scores)


def test_feasible_target_and_zero_window():
    C.configure(n_kv_heads=8,head_dim=128)
    for window in (0,32):
        rows = analyze_scores(torch.ones(1000),window=window)
        assert all(r["within_target_one_pp"] for r in rows)
        assert all(r["old_byte_difference_fraction"] <= .005 for r in rows)


@pytest.mark.parametrize("scores,budget,window",[
    (torch.ones(32),.20,32), (torch.ones(100),.20,32),
    (torch.ones(1000),0,32), (torch.ones(1000),float("nan"),32),
    (torch.ones(1000),.20,-1), (torch.ones(1000),.20,True),
    (torch.ones(1,1000),.20,32), (torch.tensor([float("nan")]),.20,0),
    (-torch.ones(1000),.20,32),
])
def test_invalid_or_infeasible_inputs_fail(scores,budget,window):
    with pytest.raises(ValueError):
        analyze_scores(scores,budget=budget,window=window)


def test_summary_does_not_hide_flagged_cells():
    rows = analyze_scores(torch.ones(728))
    report = summarize(rows)
    assert all(r["target_audit_flagged"] == 1 for r in report.values())
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize(rows[:1])


def test_configs_match_shared_parser_and_only_change_threshold():
    root = Path(__file__).resolve().parents[1]/"configs"
    configs = [json.loads((root/f"{method}_b20.json").read_text()) for method in VARIANTS]
    for item,fraction in zip(configs,VARIANTS.values()):
        base,variant = parse_method(item["method"])
        assert base == "tiered"
        assert variant.get("tiers",(.05,.20)) == (fraction,.20)
        assert item["budget"] == .20 and item["score_mode"] == "mean"
        assert item["cfg"]["tiers"] == [fraction,.20]
    comparable = [{k:v for k,v in c.items() if k not in ("method","cfg")} for c in configs]
    assert comparable[0] == comparable[1] == comparable[2]
    cfgs = [{k:v for k,v in c["cfg"].items() if k != "tiers"} for c in configs]
    assert cfgs[0] == cfgs[1] == cfgs[2]

