import pytest
import torch

from tieredkv import compress as C
from tieredkv.allocator import allocate
from tieredkv.compress import (SparseKV, budget, bytes_per_token, decode_kv, encode_kv,
                               full_bytes_per_token, load_dictionary, omp, solve_s)


def _dict(d=64, N=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(d, N, generator=g)
    return D / D.norm(dim=0, keepdim=True)


def _sparse_signals(D, B, s, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(D.shape[1], generator=g)[:s] for _ in range(B)])
    coef = (torch.rand(B, s, generator=g) + 0.5) * torch.sign(torch.randn(B, s, generator=g))
    return (D.T[idx] * coef[..., None]).sum(1), idx, coef


def _rel_err(x, y):
    return ((x.float() - y.float()).norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-8)).mean().item()


@pytest.fixture(autouse=True)
def _mem_cfg():
    C.configure(n_kv_heads=8, head_dim=128)
    yield
    C.configure(n_kv_heads=8, head_dim=128)


# --------------------------------------------------------------------------- OMP

def test_omp_recovers_exact_sparse_signals():
    D = _dict()
    Y, idx, coef = _sparse_signals(D, 50, 4)
    got_idx, got_coef = omp(D, Y, 4)
    for b in range(50):
        assert set(got_idx[b].tolist()) == set(idx[b].tolist())
    recon = (D.T[got_idx] * got_coef[..., None]).sum(1)
    torch.testing.assert_close(recon, Y, atol=1e-4, rtol=1e-4)


def test_omp_error_decreases_with_s_and_is_greedy_prefix():
    D = _dict()
    Y = torch.randn(40, 64)
    errs, prev = [], None
    for s in (1, 2, 4, 8, 16, 32):
        i, c = omp(D, Y, s)
        errs.append(_rel_err(Y, (D.T[i] * c[..., None]).sum(1)))
        if prev is not None:
            assert torch.equal(i[:, :prev.shape[1]], prev)
        prev = i
    assert all(a > b for a, b in zip(errs, errs[1:]))


# --------------------------------------------------------------------------- encode / decode

def test_roundtrip_fp16_exact_sparse():
    D = _dict(d=64)
    Y, _, _ = _sparse_signals(D, 30, 6)
    kv = Y.view(10, 3, 64)
    sp = encode_kv(kv, D, 6, coef_dtype=torch.float16)
    assert sp.n_tokens == 10 and (sp.s_per_token() == 6).all()
    assert _rel_err(kv.view(-1, 64), decode_kv(sp, D, torch.float32).view(-1, 64)) < 2e-3


def test_fp8_close_and_never_nan():
    D = _dict(d=64)
    Y, _, _ = _sparse_signals(D, 30, 6)
    sp = encode_kv(Y.view(10, 3, 64), D, 6)
    assert sp.data.dtype == torch.float8_e4m3fn
    assert _rel_err(Y, decode_kv(sp, D, torch.float32).view(-1, 64)) < 0.05
    huge = encode_kv(torch.full((1, 1, 64), 1e4), D, 4)
    assert not torch.isnan(decode_kv(huge, D, torch.float32)).any()


def test_variable_s_storage_is_unpadded_and_matches_bytes():
    D = _dict(d=128, N=512)
    n, h = 200, 8
    kv = torch.randn(n, h, 128, dtype=torch.float16)
    s = allocate(torch.rand(n), 8, s_max=32)
    K, V = encode_kv(kv, D, s), encode_kv(kv, D, s)
    assert torch.equal(K.s_per_token(), s)
    assert K.indices.numel() == h * s.sum().item()                 # no padding to s_max
    assert K.nbytes + V.nbytes == bytes_per_token(s).sum().item() + 2 * 4  # +1 indptr each
    errs = (kv.float() - decode_kv(K, D).float()).norm(dim=-1).mean(1) / kv.float().norm(dim=-1).mean(1)
    hi, lo = s >= s.float().quantile(0.9), s <= s.float().quantile(0.1)
    assert errs[hi].mean() < errs[lo].mean()                          # more atoms, less error


def test_uniform_tensor_s_equals_int_s():
    """Integration test 3 at module level: tiered with equal ratios == lexico."""
    D = _dict(d=128, N=512)
    kv = torch.randn(64, 8, 128)
    s = allocate(torch.rand(64), 16, ratios=(16, 16, 16))
    a, b = encode_kv(kv, D, s), encode_kv(kv, D, 16)
    for x, y in ((a.indptr, b.indptr), (a.indices, b.indices), (a.data, b.data)):
        assert torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
                           y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y)
    assert torch.equal(decode_kv(a, D), decode_kv(b, D))


def test_evicted_tokens_are_empty_rows():
    D = _dict(d=64)
    kv = torch.randn(5, 2, 64)
    sp = encode_kv(kv, D, torch.tensor([4, 0, 2, 0, 1]))
    assert sp.indptr.diff().tolist() == [4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    out = decode_kv(sp, D, torch.float32)
    assert (out[[1, 3]] == 0).all() and (out[[0, 2, 4]] != 0).any()


def test_cat_and_select():
    D = _dict(d=64)
    kv = torch.randn(12, 2, 64)
    s = torch.tensor([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1])
    whole = encode_kv(kv, D, s)
    parts = SparseKV.cat([encode_kv(kv[:5], D, s[:5]), encode_kv(kv[5:], D, s[5:])])
    torch.testing.assert_close(decode_kv(parts, D), decode_kv(whole, D))
    keep = torch.tensor([0, 3, 4, 10])
    torch.testing.assert_close(decode_kv(whole.select(keep), D), decode_kv(whole, D)[keep])
    assert torch.equal(whole.select(keep).s_per_token(), s[keep])
    empty = SparseKV.cat([SparseKV.empty(2), encode_kv(kv[:3], D, s[:3])])
    torch.testing.assert_close(decode_kv(empty, D), decode_kv(whole, D)[:3])


def test_load_dictionary(tmp_path):
    D = torch.randn(2 * 3, 64, 256)
    torch.save(D, tmp_path / "d.pt")
    k, v = load_dictionary(tmp_path / "d.pt", layer=1, dtype=torch.float32)
    assert torch.equal(k, D[2]) and torch.equal(v, D[3])


# --------------------------------------------------------------------------- memory math

def test_bytes_formula():
    assert full_bytes_per_token() == 2 * 8 * 128 * 2
    assert bytes_per_token(16) == 2 * 8 * (16 * 3 + 4)
    assert bytes_per_token(0) == 0
    torch.testing.assert_close(bytes_per_token(torch.tensor([0, 16])), torch.tensor([0, 832]))


def test_budget_and_solve_s_are_inverse():
    N, W = 2000, 128
    for b in (0.10, 0.15, 0.20, 0.25):
        s = solve_s(b, N, W)
        assert budget([s] * (N - W), W, N) <= b < budget([s + 1] * (N - W), W, N)
    assert budget([[4] * 10, [8] * 10], 0, 10) == budget([6] * 10, 0, 10)  # layers averaged


def test_infeasible_budget_raises():
    with pytest.raises(ValueError, match="dense window"):
        solve_s(0.10, 1000, 128)   # W alone is 12.8% of memory


def test_configure_from_hf_config():
    class Cfg:
        num_key_value_heads, num_attention_heads, hidden_size = 2, 4, 256
    C.configure(Cfg())
    assert (C.N_KV_HEADS, C.HEAD_DIM) == (2, 64)


def test_allocate_with_real_bytes_is_matched_to_lexico():
    N, W = 1200, 128
    s_uniform = solve_s(0.20, N, W)
    s = allocate(torch.rand(N - W), s_uniform, s_max=64)   # default bytes_fn = compress's
    assert abs(budget(s, W, N) - budget([s_uniform] * (N - W), W, N)) < 1e-3
