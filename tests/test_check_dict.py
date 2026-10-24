import pytest
import torch

from tieredkv import compress as C
from tieredkv.check_dict import check, plot_error_by_s, token_error


@pytest.fixture(autouse=True)
def _restore_mem_cfg():
    yield
    C.configure(n_kv_heads=8, head_dim=128)


def test_token_error_shape_and_monotone():
    D = torch.nn.functional.normalize(torch.randn(16, 64), dim=0)
    kv = torch.randn(10, 2, 16)
    e2, e8 = token_error(kv, D, 2), token_error(kv, D, 8)
    assert e2.shape == (10,) and (e8 < e2).all()


def test_check_on_tiny_llama(tmp_path):
    tf = pytest.importorskip("transformers")
    cfg = tf.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = tf.LlamaForCausalLM(cfg).eval()
    D = torch.nn.functional.normalize(torch.randn(4, 16, 64), dim=1)
    torch.save(D, tmp_path / "d.pt")
    prompts = [torch.randint(0, 97, (1, 40)) for _ in range(2)]
    out = check(model, prompts, tmp_path / "d.pt", s_values=(2, 4, 8), probe_s=4, avg_s=4)

    errs = [out["error_by_s"][s]["key"] for s in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]
    assert out["n_prompts"] == 2 and out["avg_tokens"] == 40
    assert out["timing_s_per_prompt"]["encode"] > 0
    for p in ("key", "value"):
        assert out["spearman_mean_score_vs_error"][p]["n"] == 2 * 2      # prompts x layers
        assert out["attention_weighted_error"][p]["uniform"] > 0
        assert out["attention_weighted_error"][p]["tiered"] > 0
        assert out["heavy_vs_light_error"][p]["top5"] is not None
    plot_error_by_s(out, tmp_path / "e.png")
    assert (tmp_path / "e.png").stat().st_size > 0
