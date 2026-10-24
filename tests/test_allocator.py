import pytest
import torch

from tieredkv.allocator import ScoreTracker, allocate, check_attn, h2o_keep_idx, h2o_scores

linear = lambda s: 2 * 8 * s * (2 + 2)  # 2(K,V) * 8 kv heads * s * (int16 idx + fp16 coef)
affine = lambda s: 3 * s + 5            # per-token overhead (e.g. storing s / offsets)


def total(s, fn):
    s = torch.as_tensor(s)
    return sum(fn(int(x)) for x in s.flatten().tolist() if x > 0)


# --------------------------------------------------------------------------- h2o_scores

def test_h2o_scores_shape_and_values():
    attn = torch.softmax(torch.randn(2, 4, 5, 7), -1)
    s = h2o_scores(attn)
    assert s.shape == (2, 7) and s.dtype == torch.float32
    torch.testing.assert_close(s, attn.sum(2).mean(1))
    torch.testing.assert_close(s.sum(-1), torch.full((2,), 5.0))  # total mass = n queries


def test_check_attn_rejects_logits():
    check_attn(torch.softmax(torch.randn(1, 2, 3, 3), -1))
    with pytest.raises(AssertionError):
        check_attn(torch.randn(1, 2, 3, 3))


# --------------------------------------------------------------------------- allocate

@pytest.mark.parametrize("fn", [linear, affine])
@pytest.mark.parametrize("avg_s", [4, 8, 16, 24, 10.5])
@pytest.mark.parametrize("evict", [False, True])
@pytest.mark.parametrize("n", [200, 1000, 4097])
def test_budget_matched_in_bytes(fn, avg_s, evict, n):
    g = torch.Generator().manual_seed(n)
    scores = torch.rand(n, generator=g) ** 4
    s = allocate(scores, avg_s, evict_light=evict, bytes_fn=fn, s_max=128)
    target = n * fn(avg_s)
    got = total(s, fn)
    assert got <= target + 1e-6
    assert abs(got - target) / target < 0.005


@pytest.mark.parametrize("avg_s", [4, 16, 24])
def test_equal_ratios_is_uniform(avg_s):
    scores = torch.rand(777)
    s = allocate(scores, avg_s, ratios=(16, 16, 16), bytes_fn=linear)
    assert (s == avg_s).all()


@pytest.mark.parametrize("evict", [False, True])
def test_monotone_in_score(evict):
    scores = torch.randn(1000)
    s = allocate(scores, 12, evict_light=evict, bytes_fn=linear)
    s_sorted = s[torch.argsort(scores, descending=True)]
    assert (s_sorted[:-1] >= s_sorted[1:]).all()


def test_tiers_and_eviction():
    n = 1000
    scores = torch.arange(n, dtype=torch.float32)  # last tokens most important
    s = allocate(scores, 8, bytes_fn=linear)
    heavy, mid, light = s[-50:], s[-250:-50], s[:-250]
    assert heavy.min() > mid.max() >= mid.min() > light.max() > 0
    s = allocate(scores, 8, evict_light=True, bytes_fn=linear)
    assert (s[:-250] == 0).all() and (s[-250:] > 0).all()


def test_s_max_respected_and_spills_down():
    s = allocate(torch.rand(500), 16, bytes_fn=linear, s_max=20)
    assert s.max() <= 20
    assert abs(total(s, linear) - 500 * linear(16)) / (500 * linear(16)) < 0.005


def test_batched_and_device_dtype():
    scores = torch.rand(3, 300, dtype=torch.float16)
    s = allocate(scores, 8, bytes_fn=linear)
    assert s.shape == (3, 300) and s.dtype == torch.long
    for b in range(3):
        torch.testing.assert_close(s[b], allocate(scores[b], 8, bytes_fn=linear))


def test_budget_too_small_raises():
    with pytest.raises(ValueError):
        allocate(torch.rand(100), 0.5, bytes_fn=linear)  # s_min=1 for every kept token > 0.5


def test_default_bytes_fn_is_compress_bytes_per_token():
    from tieredkv.compress import bytes_per_token
    s = allocate(torch.rand(400), 10)
    assert abs(total(s, bytes_per_token) - 400 * bytes_per_token(10)) <= bytes_per_token(1)


# --------------------------------------------------------------------------- h2o_keep_idx

def test_h2o_keep_idx():
    scores = torch.tensor([9., 0., 5., 0., 0., 7., 0., 0., 0., 0.])
    idx = h2o_keep_idx(scores, 5, W=2)
    assert idx.tolist() == [0, 2, 5, 8, 9]
    assert h2o_keep_idx(scores, 0.5, W=2).tolist() == [0, 2, 5, 8, 9]
    assert h2o_keep_idx(scores, 1.0, W=2).tolist() == list(range(10))
    assert h2o_keep_idx(scores, 1, W=3).tolist() == [7, 8, 9]  # never below W
    assert h2o_keep_idx(scores.repeat(2, 1), 5, W=2).shape == (2, 5)


# --------------------------------------------------------------------------- ScoreTracker (unit)

class _FakeModel:  # just enough structure for hook registration
    def __init__(self, n):
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Module() for _ in range(n)])
        for l in self.model.layers:
            l.self_attn = torch.nn.Identity()


def _causal_attn(b, h, q, k):
    logits = torch.randn(b, h, q, k)
    mask = torch.ones(q, k, dtype=torch.bool).tril(k - q)
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)


def test_tracker_accumulates_pads_and_realigns():
    tr = ScoreTracker(_FakeModel(1), check=True)
    a0 = _causal_attn(1, 4, 6, 6)          # prefill
    a1 = _causal_attn(1, 4, 1, 7)          # decode step
    tr.add(0, a0); tr.add(0, a1)
    exp = torch.nn.functional.pad(h2o_scores(a0), (0, 1)) + h2o_scores(a1)
    torch.testing.assert_close(tr.scores(0), exp)
    torch.testing.assert_close(tr.scores(0, torch.tensor([1, 3])), exp[:, [1, 3]])
    tr.keep(0, torch.tensor([0, 2, 6]))
    torch.testing.assert_close(tr.scores(0), exp[:, [0, 2, 6]])
    tr.add(0, _causal_attn(1, 4, 1, 4))    # next step sees 3 kept + 1 new
    assert tr.scores(0).shape == (1, 4)
    with pytest.raises(RuntimeError):      # forgot keep() after eviction
        tr.add(0, _causal_attn(1, 4, 1, 2))
    tr.reset()
    assert tr.acc == {}


def test_tracker_mean_mode_removes_causal_bias():
    tr = ScoreTracker(_FakeModel(1), mode="mean")
    k = 8
    a = _causal_attn(1, 2, k, k)
    tr.add(0, a)
    cnt = torch.arange(k, 0, -1, dtype=torch.float32)  # key j seen by k - j queries
    torch.testing.assert_close(tr.scores(0), h2o_scores(a) / cnt)


# --------------------------------------------------------------------------- ScoreTracker (real HF Llama)

def _tiny_llama():
    tf = pytest.importorskip("transformers")
    cfg = tf.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return tf.LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def test_tracker_on_llama_matches_output_attentions():
    model = _tiny_llama()
    ids = torch.randint(0, 97, (1, 12))
    ref = model(ids, output_attentions=True).attentions
    for a in ref:
        assert a.shape == (1, 4, 12, 12)
        check_attn(a)

    tr = ScoreTracker(model, check=True)
    out = model(ids, use_cache=True)                # no output_attentions: pre-hook forces it
    for i, a in enumerate(ref):
        torch.testing.assert_close(tr.scores(i), h2o_scores(a))

    nxt = out.logits[:, -1:].argmax(-1)
    model(nxt, past_key_values=out.past_key_values, use_cache=True)
    assert all(tr.scores(i).shape == (1, 13) for i in range(2))
    tr.remove()
    model(ids)
    assert tr.scores(0).shape == (1, 13)            # hooks gone, nothing added
