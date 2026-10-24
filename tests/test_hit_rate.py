import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tieredkv.hit_rate import collect, layer_hit_rates, summarize


def test_perfect_and_inverted_prediction():
    T, W = 132, 32
    pre = torch.arange(T, 0, -1).float()          # prefill ranks token 0 highest
    cnt = torch.ones(T)
    good = layer_hit_rates(pre, cnt, pre.clone(), window=W)
    assert good["hh_precision"] == 1.0
    assert good["hh_coverage"] > good["random_hh"] and good["tiered_coverage"] > good["random_tiered"]
    bad = layer_hit_rates(pre, cnt, torch.arange(T).float(), window=W)    # decode prefers the opposite
    assert bad["hh_precision"] == 0.0
    assert bad["hh_coverage"] < bad["random_hh"]


def test_h2o_lost_mass_shrinks_with_budget():
    T, W = 200, 32
    g = torch.Generator().manual_seed(0)
    pre, mass = torch.rand(T, generator=g), torch.rand(T, generator=g)
    r = layer_hit_rates(pre, torch.ones(T), mass, window=W)
    lost = [r[f"h2o_lost@{b:g}"] for b in (0.10, 0.15, 0.20, 0.25)]
    assert all(0 <= x <= 1 for x in lost) and lost == sorted(lost, reverse=True)


def test_window_too_large():
    with pytest.raises(ValueError):
        layer_hit_rates(torch.ones(10), torch.ones(10), torch.ones(10), window=32)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=None)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


def test_collect_on_tiny_llama(tiny):
    T, NEW = 60, 12
    ids = torch.randint(1, 256, (1, T), generator=torch.Generator().manual_seed(1))
    layers, n_new = collect(tiny, ids, NEW)
    assert len(layers) == 2 and n_new == NEW
    for s, c, m in layers.values():
        assert s.shape == c.shape == m.shape == (T,)
        torch.testing.assert_close(c, (T - torch.arange(T)).float())    # prefill: key j seen by T - j queries
        # n_new - 1 decode queries, each attention row sums to 1 (part of it on generated tokens)
        assert (m >= -1e-6).all() and 0 < m.sum().item() <= (n_new - 1) + 1e-3
    rows = [{"problem": 0, "layer": l, **layer_hit_rates(s, c, m, window=8)} for l, (s, c, m) in layers.items()]
    out = summarize(rows)
    assert 0 <= out["hh_precision"] <= 1 and 0 <= out["tiered_coverage"] <= 1


def test_collect_stops_early(tiny):
    ids = torch.randint(1, 256, (1, 40), generator=torch.Generator().manual_seed(2))
    _, n_new = collect(tiny, ids, 50, stop=lambda new: new.shape[-1] >= 3)
    assert n_new == 3
