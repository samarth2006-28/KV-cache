"""demo/app.py core path on a tiny random Llama with a character tokenizer (no GPU, no gradio)."""
import importlib.util
import os
import string

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from tieredkv import compress as C

_spec = importlib.util.spec_from_file_location(
    "demo_app", os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "app.py"))
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=None)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(string.printable[:255])}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer, t.decoder = pre_tokenizers.Split("", "isolated"), decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="<unk>")
    path = tmp_path_factory.mktemp("d") / "dict.pt"
    torch.save(torch.nn.functional.normalize(torch.randn(6, 16, 64), dim=1), path)
    shots = [{"question": "What is 2 plus 2?", "answer": "2+2=4\n#### 4"}]
    yield model, tok, str(path), shots
    C.configure(n_kv_heads=8, head_dim=128)


def test_compare_runs_every_method(setup):
    model, tok, path, shots = setup
    q = "Tom has 3 apples and buys 5 more. How many apples does he have now?"
    rows, shade = app.compare(model, tok, q, 0.5, path, shots=shots, max_new_tokens=16)
    assert [r["method"] for r in rows] == ["full", "tiered", "lexico"]
    full, tiered, lexico = rows
    assert full["budget"] == pytest.approx(2.0)                 # fp32 tiny model vs fp16 accounting
    assert tiered["kv_mb"] < full["kv_mb"] and lexico["kv_mb"] < full["kv_mb"]
    assert tiered["cache"].sparse[0] is not None               # compression really happened
    # with 16 new tokens the question's last tokens are still in the dense recent window
    assert "atoms" in shade and "recent window" in shade
    tail = q + chr(10) + "Answer:"                             # the prompt ends "<question>\nAnswer:"
    assert shade.count("<span") == len(tok(tail)["input_ids"])  # each token shown once
    assert all(isinstance(app.summary_md(r), str) for r in rows)


def test_token_atoms_none_without_compression():
    from tieredkv.cache import TieredKVCache
    assert app.token_atoms(TieredKVCache("full")) is None
