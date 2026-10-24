import pytest
import torch

from tieredkv.compress import omp
from tieredkv.train_dict import collect_kv, init_dict, rel_error, train


def test_batched_omp_equals_per_dictionary():
    D = torch.randn(3, 32, 100)
    D /= D.norm(dim=1, keepdim=True)
    Y = torch.randn(3, 20, 32)
    idx, coef = omp(D, Y, 5)
    for l in range(3):
        i, c = omp(D[l], Y[l], 5)
        assert torch.equal(idx[l], i)
        torch.testing.assert_close(coef[l], c)


def test_train_reduces_error_and_keeps_format():
    g = torch.Generator().manual_seed(0)
    atoms = torch.randn(4, 16, 64, generator=g)
    atoms /= atoms.norm(dim=1, keepdim=True)
    codes = torch.zeros(4, 3000, 64)
    codes.scatter_(-1, torch.randint(0, 64, (4, 3000, 3), generator=g), torch.randn(4, 3000, 3, generator=g))
    buf = (codes @ atoms.transpose(1, 2)).half()                  # [L2=4, M, d=16], 3-sparse
    D, log = train(buf, N=64, s=3, minutes=1, max_steps=150, batch=256, lr=3e-3, n_eval=500, log_every=1000)
    assert D.shape == (4, 16, 64) and D.dtype == torch.float16
    torch.testing.assert_close(D.float().norm(dim=1), torch.ones(4, 64), atol=1e-2, rtol=0)
    assert sum(log["final_rel_err"]) < sum(log["init_rel_err"])


def test_collect_kv_on_tiny_llama():
    tf = pytest.importorskip("transformers")
    cfg = tf.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = tf.LlamaForCausalLM(cfg).eval()
    batches = (torch.randint(0, 97, (2, 10)) for _ in range(5))
    buf = collect_kv(model, batches, max_tokens=40)
    assert buf.shape == (2 * 2, 2 * 10 * 2 * 2, 16)               # 2 batches * 20 tokens * 2 heads
    D = init_dict(buf, 8)
    assert D.shape == (4, 16, 8)
    assert rel_error(D, buf[:, :10].float(), 2).shape == (4,)
