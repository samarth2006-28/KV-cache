"""Phase 2: compression path inside TieredKVCache, on a tiny random Llama (CPU, no download).

Small W and C so the prompt (prefill) and several decode chunks all get compressed.
1. Stubs (constant allocate, identity encode): lexico / tiered must equal full exactly.
2. Plan test 5: h2o at 100% budget equals full; h2o / evict_light below 100% stay aligned.
3. Real compress.encode_kv: plan test 3 (tiered with equal ratios == lexico) and measured budget.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tieredkv import compress as C
from tieredkv.allocator import ScoreTracker
from tieredkv.cache import TieredKVCache

W, CH, PROMPT, NEW = 8, 4, 40, 30


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=None)   # never stop early: every run is PROMPT + NEW tokens
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(autouse=True)
def _restore_shape():
    yield
    C.configure(n_kv_heads=8, head_dim=128)


def _ids(seed):
    return torch.randint(1, 256, (1, PROMPT), generator=torch.Generator().manual_seed(seed))


def _gen(model, ids, cache):
    return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=NEW,
                          do_sample=False, pad_token_id=0, past_key_values=cache)


def const_allocate(scores, avg_s, *args, **kwargs):
    return torch.full(scores.shape, int(avg_s), dtype=torch.long)


def identity_encode(kv, dictionary, s):
    return kv.clone()


def identity_decode(sparse, dictionary, dtype=None):
    return sparse.to(dtype)


STUBS = dict(encode_fn=identity_encode, decode_fn=identity_decode, window=W, chunk=CH, avg_s=4)


def _full(model, ids):
    return _gen(model, ids, TieredKVCache("full"))


def _check_compressed(cache):
    for layer in range(len(cache)):
        assert cache.sparse[layer] is not None and cache.sparse[layer]["n"] > 0
        assert W <= cache.key_cache[layer].shape[-2] <= W + CH + 1
        assert cache.get_seq_length() == PROMPT + NEW - 1


# --------------------------------------------------------------------------- 1. plumbing

@pytest.mark.parametrize("seed", range(3))
def test_lexico_stub_equals_full(tiny, seed):
    cache = TieredKVCache("lexico", **STUBS)
    assert torch.equal(_gen(tiny, _ids(seed), cache), _full(tiny, _ids(seed)))
    _check_compressed(cache)
    assert cache.stored_tokens() == PROMPT + NEW - 1          # nothing lost


@pytest.mark.parametrize("seed", range(3))
def test_tiered_stub_equals_full(tiny, seed):
    with ScoreTracker(tiny, mode="mean") as tr:
        cache = TieredKVCache("tiered", tracker=tr, allocate_fn=const_allocate, **STUBS)
        out = _gen(tiny, _ids(seed), cache)
        for layer in range(len(cache)):
            assert tr.acc[layer].shape[-1] == cache.stored_tokens(layer)
    assert torch.equal(out, _full(tiny, _ids(seed)))
    _check_compressed(cache)


def test_tiered_real_allocate_identity_encode_equals_full(tiny):
    """Real allocate() picks varying s; with an identity encoder output still equals full."""
    with ScoreTracker(tiny, mode="mean") as tr:
        cache = TieredKVCache("tiered", tracker=tr, **STUBS)
        out = _gen(tiny, _ids(7), cache)
    assert torch.equal(out, _full(tiny, _ids(7)))
    assert cache.token_s(0).unique().numel() > 1              # tiers really differ


def test_tracker_reset_between_prompts(tiny):
    with ScoreTracker(tiny, mode="mean") as tr:
        for seed in (1, 2):
            cache = TieredKVCache("tiered", tracker=tr, allocate_fn=const_allocate, **STUBS)
            _gen(tiny, _ids(seed), cache)
            assert tr.acc[0].shape[-1] == cache.stored_tokens(0)


# --------------------------------------------------------------------------- 2. eviction

def test_h2o_full_budget_equals_full(tiny):
    """Plan test 5: h2o at 100% budget evicts nothing and matches full."""
    with ScoreTracker(tiny, mode="sum") as tr:
        cache = TieredKVCache("h2o", tracker=tr, budget=1.0, window=W, chunk=CH)
        out = _gen(tiny, _ids(3), cache)
    assert torch.equal(out, _full(tiny, _ids(3)))
    assert cache.stored_tokens() == PROMPT + NEW - 1


@pytest.mark.parametrize("budget", [0.5, 0.3])
def test_h2o_evicts_to_budget_and_stays_aligned(tiny, budget):
    with ScoreTracker(tiny, mode="sum") as tr:
        cache = TieredKVCache("h2o", tracker=tr, budget=budget, window=W, chunk=CH)
        out = _gen(tiny, _ids(4), cache)
        seen = cache.get_seq_length()
        for layer in range(len(cache)):
            assert cache.stored_tokens(layer) <= int(budget * seen) + 1
            assert tr.acc[layer].shape[-1] == cache.stored_tokens(layer)
    assert out.shape[1] == PROMPT + NEW
    assert cache.measured_budget() == pytest.approx(2 * budget, abs=0.1)   # fp32 tiny model: x2


def test_tiered_evict_light_stays_aligned(tiny):
    with ScoreTracker(tiny, mode="mean") as tr:
        cache = TieredKVCache("tiered", tracker=tr, evict_light=True, **STUBS)
        out = _gen(tiny, _ids(5), cache)
        for layer in range(len(cache)):
            assert cache.stored_tokens(layer) < PROMPT + NEW - 1     # light tier dropped
            assert tr.acc[layer].shape[-1] == cache.stored_tokens(layer)
    assert out.shape[1] == PROMPT + NEW


# --------------------------------------------------------------------------- 3. real encoder

@pytest.fixture(scope="module")
def dictionary(tiny):
    """Random unit-norm atoms, Lexico layout [2L, head_dim, N]."""
    g = torch.Generator().manual_seed(0)
    D = torch.randn(2 * tiny.config.num_hidden_layers, 16, 64, generator=g)
    return D / D.norm(dim=1, keepdim=True)


def test_tiered_equal_ratios_equals_lexico(tiny, dictionary):
    """Plan test 3: tiered with ratios (s, s, s) == lexico at s, through the real encoder."""
    lex = _gen(tiny, _ids(6), TieredKVCache("lexico", dictionary=dictionary, avg_s=6, window=W, chunk=CH))
    with ScoreTracker(tiny, mode="mean") as tr:
        cache = TieredKVCache("tiered", tracker=tr, dictionary=dictionary, avg_s=6,
                              ratios=(6, 6, 6), window=W, chunk=CH)
        tie = _gen(tiny, _ids(6), cache)
    assert torch.equal(lex, tie)
    assert (cache.token_s(0) == 6).all()


@pytest.mark.parametrize("method", ["lexico", "tiered"])
def test_real_encoder_hits_budget(tiny, dictionary, method):
    """avg_s solved from the budget; measured bytes within ~1 point of the target."""
    budget = 0.6
    with ScoreTracker(tiny, mode="mean") as tr:
        cache = TieredKVCache(method, tracker=tr if method == "tiered" else None,
                              dictionary=dictionary, budget=budget, expected_new_tokens=NEW - 1,
                              window=W, chunk=CH)
        _gen(tiny, _ids(8), cache)
    # tiny model is fp32, so dense tokens cost 2x the fp16 accounting; compare compressed parts
    n_seen, n_dense = cache.get_seq_length(), cache.key_cache[0].shape[-2]
    sparse = sum(c.nbytes for sp in cache.sparse for c in sp["k"] + sp["v"])
    full = C.full_bytes_per_token() * len(cache) * n_seen
    measured = (sparse + C.full_bytes_per_token() * len(cache) * n_dense) / full
    assert measured == pytest.approx(budget, abs=0.03)
    assert cache.avg_s == C.solve_s(budget, PROMPT + NEW - 1, W + CH // 2)


# --------------------------------------------------------------------------- 4. decode once

@pytest.mark.parametrize("method,kw", [("lexico", {}), ("tiered", {}), ("tiered", {"evict_light": True})])
def test_decode_once_equals_decode_every_step(tiny, dictionary, method, kw):
    """cache_decoded=True keeps one dense copy per chunk; tokens and bytes match re-decoding."""
    outs, caches = [], []
    for cached in (False, True):
        with ScoreTracker(tiny, mode="mean") as tr:
            cache = TieredKVCache(method, tracker=tr if method == "tiered" else None,
                                  dictionary=dictionary, avg_s=6, window=W, chunk=CH,
                                  cache_decoded=cached, **kw)
            outs.append(_gen(tiny, _ids(9), cache))
            caches.append(cache)
    assert torch.equal(outs[0], outs[1])
    off, on = caches
    assert off.kv_bytes() == on.kv_bytes() and off.decoded_bytes() == 0
    for layer in range(len(on)):
        dk, dv = on._decoded[layer]
        assert dk.shape[-2] == on.sparse[layer]["n"]
        ok, ov = off._decode(layer, dk.dtype)
        assert torch.equal(dk, ok) and torch.equal(dv, ov)
    assert on.decoded_bytes() > 0
