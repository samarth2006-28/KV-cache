"""Regression tests for Sampreeth's additions to the canonical allocator."""
import csv
import json
import subprocess
import sys

import pytest
import torch

from tieredkv.allocator import ScoreTracker, h2o_keep_idx, h2o_scores
from tieredkv.budget_audit import audit_rows, read_audit
from tieredkv.h2o_baseline import h2o_keep_idx as baseline_keep
from tieredkv.sanity import spearman


class FakeModel:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Identity()


def test_mean_eviction_preserves_numerators_and_counts():
    with ScoreTracker(FakeModel(), mode="mean") as tr:
        tr.add(0, torch.tril(torch.ones(3, 3))[None, None] * .25)
        torch.testing.assert_close(tr.cnt[0], torch.tensor([[3., 2., 1.]]))
        torch.testing.assert_close(tr.scores(0), torch.full((1, 3), .25))
        tr.keep(0, torch.tensor([0, 2]))
        tr.add(0, torch.tensor([[[[.5, .25, .25]]]]))
        torch.testing.assert_close(tr.scores(0), torch.tensor([[1.25/4, .25, .25]]))
        torch.testing.assert_close(tr.cnt[0], torch.tensor([[4., 2., 1.]]))
        tr.reset()
        assert tr.acc == tr.cnt == {}


def test_mean_chunked_prefill_equals_whole():
    a = torch.tril(torch.rand(4, 4))[None, None]
    with ScoreTracker(FakeModel(), mode="mean") as full, ScoreTracker(FakeModel(), mode="mean") as chunks:
        full.add(0, a)
        chunks.add(0, a[:, :, :2, :2]); chunks.add(0, a[:, :, 2:, :])
        torch.testing.assert_close(full.scores(0), chunks.scores(0))
        torch.testing.assert_close(full.cnt[0], chunks.cnt[0])


def test_explicit_padding_and_head_masks():
    mask = torch.tensor([[[True, False, False], [True, True, False], [False, False, False]]])
    with ScoreTracker(FakeModel(), mode="mean") as tr:
        tr.add(0, torch.ones(1, 2, 3, 3), valid_mask=mask)
        torch.testing.assert_close(tr.cnt[0], torch.tensor([[2., 1., 0.]]))
        torch.testing.assert_close(tr.scores(0), torch.tensor([[1., 1., 0.]]))
        tr.reset()
        tr.add(0, torch.tensor([[[[.2, 0]], [[0, .4]]]]),
               valid_mask=torch.tensor([[[[True, False]], [[False, True]]]]))
        torch.testing.assert_close(tr.scores(0), torch.tensor([[.2, .4]]))


def test_visible_zero_attention_counts_as_opportunity():
    with ScoreTracker(FakeModel(), mode="mean") as tr:
        tr.add(0, torch.tensor([[[[1., 0], [1., 0]]]]))
        torch.testing.assert_close(tr.cnt[0], torch.tensor([[2., 1.]]))


def test_detached_scores_and_defensive_read():
    a = torch.softmax(torch.randn(1, 2, 2, 2, requires_grad=True), -1)
    assert not h2o_scores(a).requires_grad
    with ScoreTracker(FakeModel()) as tr:
        tr.add(0, a)
        tr.scores(0).zero_()
        assert tr.scores(0).sum() > 0
        with pytest.raises(ValueError, match="batch"):
            tr.add(0, a.expand(2, -1, -1, -1))


def test_bad_visibility_and_hook_cleanup_on_error():
    model = FakeModel()
    with pytest.raises(ValueError):
        with ScoreTracker(model) as tr:
            tr.add(0, torch.ones(1, 1, 1, 2), valid_mask=torch.ones(1, 1, 2))
    attn = model.model.layers[0].self_attn
    assert not attn._forward_hooks and not attn._forward_pre_hooks


def test_h2o_original_denominator_floor_and_strict_window():
    assert baseline_keep is h2o_keep_idx
    assert h2o_keep_idx(torch.ones(4), .2, 2, original_length=20, strict=True).tolist() == [0,1,2,3]
    assert h2o_keep_idx(torch.ones(10), .15, 0, strict=True).numel() == 1
    with pytest.raises(ValueError, match="recent"):
        h2o_keep_idx(torch.ones(10), .1, 2, strict=True)
    with pytest.warns(UserWarning, match="recent"):
        assert h2o_keep_idx(torch.ones(10), .1, 2).numel() == 2


def test_h2o_tied_scores_are_stable_and_chronological():
    assert h2o_keep_idx(torch.ones(10), 4, 2, strict=True).tolist() == [0,1,8,9]


def test_budget_audit_flags_unfair_and_invalid_rows(tmp_path):
    rows = [{"method":"tiered", "budget_target":.2, "budget_measured":.209},
            {"method":"h2o", "budget_target":.2, "budget_measured":.22},
            {"method":"lexico", "budget_target":.2, "budget_measured":"nan"}]
    assert [r["eligible"] for r in audit_rows(rows)] == [True, False, False]
    p = tmp_path / "results.csv"
    with p.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert len(read_audit(p)) == 3
    result = subprocess.run([sys.executable, "-m", "tieredkv.budget_audit", str(p)], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)[1]["difference_pp"] == pytest.approx(2)


def test_empty_audit_and_spearman_ties(tmp_path):
    p=tmp_path/"empty.csv";p.write_text("method,budget_target,budget_measured\n")
    with pytest.raises(ValueError, match="no runs"): read_audit(p)
    assert spearman([1,2,2,4], [4,2,2,1]) == pytest.approx(-1)
    assert spearman([1,1], [2,3]) is None


@pytest.mark.parametrize("mode", ["sum", "mean"])
@torch.no_grad()
def test_real_llama_generation_unmodified_by_tracker(mode):
    tf=pytest.importorskip("transformers")
    cfg=tf.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        eos_token_id=None, pad_token_id=0)
    cfg._attn_implementation="eager"
    torch.manual_seed(0); model=tf.LlamaForCausalLM(cfg).eval()
    ids=torch.tensor([[1,7,13,22]]); mask=torch.ones_like(ids)
    ref=model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    with ScoreTracker(model, mode=mode, check=True) as tr:
        out=model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
        assert torch.equal(ref,out)
        for layer in range(2):
            torch.testing.assert_close(tr.cnt[layer],torch.tensor([[6.,5.,4.,3.,2.,1.]]))
            expected=tr.acc[layer] if mode=="sum" else tr.acc[layer]/tr.cnt[layer]
            torch.testing.assert_close(tr.scores(layer),expected)
    assert all(not l.self_attn._forward_hooks and not l.self_attn._forward_pre_hooks for l in model.model.layers)
