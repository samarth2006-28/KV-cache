import pytest
import torch

from tieredkv.score_diagnostics import summarize_scores


def test_sum_and_mean_are_distinct_with_real_causal_example():
    result = summarize_scores(torch.tensor([[1.8,.7,.5]]), torch.tensor([[3.,2.,1.]]), fraction=1/3)
    assert result["raw_top_attention_mass_share"] == pytest.approx(.6)
    assert result["mean_top_score_share"] == pytest.approx(.6/(.6+.35+.5))
    assert result["top_count"] == 1
    assert result["raw_position_spearman"] == pytest.approx(-1)
    assert result["mean_position_spearman"] == pytest.approx(-.5)


def test_ties_are_stable_and_short_prompt_fraction_is_reported():
    result = summarize_scores(torch.ones(3), torch.ones(3))
    assert result["realized_top_fraction"] == pytest.approx(1/3)
    assert result["top_overlap_fraction"] == 1
    assert result["raw_position_spearman"] is None


def test_padding_and_zero_scores():
    result = summarize_scores(torch.tensor([0.,0.,0.]), torch.tensor([0.,1.,1.]))
    assert result["observed_tokens"] == 2
    assert result["raw_top_attention_mass_share"] == 0
    result = summarize_scores(torch.empty(0), torch.empty(0))
    assert result["top_count"] == 0
    assert result["top_overlap_fraction"] is None


@pytest.mark.parametrize("sums,counts", [
    (torch.ones(2,3), torch.ones(2,3)),
    (torch.tensor([float('nan')]), torch.ones(1)),
    (torch.ones(1), torch.zeros(1)),
    (torch.ones(2), torch.ones(3)),
])
def test_invalid_pairs_rejected(sums, counts):
    with pytest.raises(ValueError): summarize_scores(sums, counts)
