"""Integration tests (plan: "what it works means"). Phase 1 has plan test 1, plus checks that
the tracker, allocator, compressor, runner and plot.py fit together.

Test 1 runs twice: on a tiny random Llama (always, CPU, no download) and on the real model with
5 fixed GSM8K prompts when TIEREDKV_MODEL is set, e.g.
    TIEREDKV_MODEL=meta-llama/Llama-3.1-8B-Instruct python -m pytest -q tests/test_integration.py
"""
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tieredkv.cache import TieredKVCache


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


def _greedy(model, ids, cache=None, n=40):
    kw = {"past_key_values": cache} if cache is not None else {}
    return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                          do_sample=False, pad_token_id=0, **kw)


@pytest.mark.parametrize("seed", range(5))
def test_full_matches_stock_tiny(tiny, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 256, (1, 20 + 7 * seed), generator=g)
    cache = TieredKVCache("full")
    out = _greedy(tiny, ids, cache)
    assert torch.equal(_greedy(tiny, ids), out)
    n = out.shape[1] - 1                    # the last token is sampled, never cached
    assert cache.get_seq_length() == n and cache.stored_tokens() == n
    assert cache.measured_budget() == pytest.approx(2.0)   # fp32 model vs fp16 accounting


def test_full_logits_match_stock_tiny(tiny):
    """Prefill + 10 manual decode steps: logits identical, not just argmax."""
    from transformers import DynamicCache
    ids = torch.randint(1, 256, (1, 16), generator=torch.Generator().manual_seed(9))
    a, b = DynamicCache(), TieredKVCache("full")
    la = tiny(ids, past_key_values=a, use_cache=True).logits
    lb = tiny(ids, past_key_values=b, use_cache=True).logits
    for _ in range(10):
        assert torch.equal(la, lb)
        nxt = la[:, -1:].argmax(-1)
        la = tiny(nxt, past_key_values=a, use_cache=True).logits
        lb = tiny(nxt, past_key_values=b, use_cache=True).logits
    assert torch.equal(la, lb)


def test_methods_need_their_inputs():
    with pytest.raises(ValueError):
        TieredKVCache("bogus")
    for m in ("lexico", "h2o", "tiered"):      # tracker and/or dictionary missing
        with pytest.raises(ValueError):
            TieredKVCache(m)


@pytest.mark.skipif(not os.environ.get("TIEREDKV_MODEL"), reason="set TIEREDKV_MODEL to run on the real model")
def test_full_matches_stock_real_model():
    """Plan test 1: method=full vs stock HF, token-identical greedy output on 5 GSM8K prompts."""
    from tieredkv import gsm8k
    from tieredkv.eval import load_model
    model, tok = load_model(os.environ["TIEREDKV_MODEL"])
    shots = gsm8k.fewshot(5)
    for p in gsm8k.load_subset(5, seed=0):
        ids = tok(gsm8k.build_prompt(p["question"], shots), return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=64, do_sample=False, temperature=None, top_p=None,
                  pad_token_id=tok.eos_token_id)
        stock = model.generate(**ids, **kw)
        ours = model.generate(**ids, past_key_values=TieredKVCache("full"), **kw)
        assert torch.equal(stock, ours), f"problem {p['idx']} diverged"


# --------------------------------------------------------------------------- modules together

def test_tracker_allocate_encode_on_cache(tiny):
    """ScoreTracker stays aligned with the cache through generate(); allocate -> encode_kv ->
    decode_kv run on the cache's own tensors. This is the phase-2 compression path, unplugged."""
    from tieredkv import compress as C
    from tieredkv.allocator import ScoreTracker, allocate
    tr = ScoreTracker(tiny, mode="mean", check=True)
    try:
        cache = TieredKVCache("full")
        ids = torch.randint(1, 256, (1, 30), generator=torch.Generator().manual_seed(3))
        _greedy(tiny, ids, cache, n=15)
        for layer in range(len(cache)):
            assert tr.acc[layer].shape[1] == cache.stored_tokens(layer)
        C.configure(tiny.config)
        s = allocate(tr.scores(0)[0], avg_s=8, bytes_fn=C.bytes_per_token)
        assert C.bytes_per_token(s).sum() <= len(s) * C.bytes_per_token(8)
        D = torch.nn.functional.normalize(torch.randn(16, 256, generator=torch.Generator().manual_seed(0)), dim=0)
        kv = cache.key_cache[0][0].transpose(0, 1)            # [b, h, seq, d] -> [seq, h, d]
        sp = C.encode_kv(kv, D, s)
        assert torch.equal(sp.s_per_token(), s)
        assert C.decode_kv(sp, D, dtype=kv.dtype).shape == kv.shape
    finally:
        tr.remove()
        C.configure(n_kv_heads=8, head_dim=128)


@pytest.mark.parametrize("budget,s", [(0.10, 4), (0.15, 9), (0.20, 13), (0.25, 17)])
def test_every_budget_feasible_at_default_window(budget, s):
    """~1000-token 5-shot GSM8K prompts: all 4 matrix budgets must leave room for s >= 1."""
    from tieredkv import compress as C
    from tieredkv.cache import WINDOW
    assert C.solve_s(budget, 1000, WINDOW) == s


@pytest.mark.parametrize("name,base,cfg", [
    ("full", "full", {}), ("tiered", "tiered", {}),
    ("tiered_hh3", "tiered", {"tiers": (0.03, 0.20)}),
    ("tiered_hh10", "tiered", {"tiers": (0.10, 0.20)}),
    ("tiered_24-8-2", "tiered", {"ratios": (24, 8, 2)}),
    ("tiered_evict", "tiered", {"evict_light": True}),
    ("tiered_hh3_evict", "tiered", {"tiers": (0.03, 0.20), "evict_light": True}),
    ("tiered_kv", "tiered", {"value_policy": "uniform"}),
])
def test_parse_method(name, base, cfg):
    from tieredkv.eval import parse_method
    assert parse_method(name) == (base, cfg)


@pytest.mark.parametrize("bad", ["bogus", "lexico_hh3", "tiered_foo", "tiered_24-8"])
def test_parse_method_rejects(bad):
    from tieredkv.eval import parse_method
    with pytest.raises(ValueError):
        parse_method(bad)


def test_runner_row_readable_by_plot(tiny, tmp_path, monkeypatch):
    """eval.run end to end offline (tiny model, char tokenizer, fake problems) -> plot.load."""
    import string
    import sys
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from tieredkv import eval as ev, gsm8k
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(string.printable[:255])}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer, t.decoder = pre_tokenizers.Split("", "isolated"), decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="<unk>")
    probs = [{"idx": i, "question": f"{i}+1?", "answer": f"#### {i + 1}", "target": str(i + 1)} for i in range(3)]
    monkeypatch.setitem(ev._loaded, "tiny", (tiny, tok))
    monkeypatch.setattr(gsm8k, "load_subset", lambda n, seed=0, split="test": probs[:n])
    monkeypatch.setattr(gsm8k, "fewshot", lambda k=5, seed=1234: probs[:1])
    csv_path = str(tmp_path / "results.csv")
    out = ev.run("full", 1.0, "gsm8k", 3, model_name="tiny", results=csv_path, max_new_tokens=8)
    assert set(out) >= {"acc", "mem_gb", "tok_s", "ms_tok", "cfg"} and out["s_prob"] > 0
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import plot
    (row,) = plot.load(csv_path)
    assert row["method"] == "full" and row["n"] == 3
    assert sum(1 for _ in open(out["cfg"]["preds"])) == 3


@pytest.mark.parametrize("method,budget", [("lexico", 0.5), ("tiered", 0.5), ("h2o", 0.5),
                                           ("tiered_hh3_evict", 0.5), ("tiered_24-8-2", 0.5),
                                           ("tiered_kv", 0.5)])
def test_runner_every_method(tiny, tmp_path, monkeypatch, method, budget):
    """eval.run with tracker + dictionary wiring, offline. Small W/C so short prompts compress."""
    import string
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from tieredkv import compress as C, eval as ev, gsm8k
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(string.printable[:255])}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer, t.decoder = pre_tokenizers.Split("", "isolated"), decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="<unk>")
    probs = [{"idx": i, "question": f"What is {i} plus one?", "answer": f"#### {i + 1}",
              "target": str(i + 1)} for i in range(2)]
    monkeypatch.setitem(ev._loaded, "tiny", (tiny, tok))
    monkeypatch.setattr(gsm8k, "load_subset", lambda n, seed=0, split="test": probs[:n])
    monkeypatch.setattr(gsm8k, "fewshot", lambda k=5, seed=1234: probs[:1])
    D = torch.nn.functional.normalize(torch.randn(6, 16, 64, generator=torch.Generator().manual_seed(0)), dim=1)
    torch.save(D, tmp_path / "dict.pt")
    cfg = {"dictionary": str(tmp_path / "dict.pt"), "window": 4, "chunk": 2, "expected_new_tokens": 10}
    try:
        out = ev.run(method, budget, "gsm8k", 2, model_name="tiny", cfg=cfg,
                     results=str(tmp_path / "r.csv"), max_new_tokens=12)
    finally:
        C.configure(n_kv_heads=8, head_dim=128)
    row = out["cfg"]
    assert row["method"] == method and 0 < row["budget_measured"] < 2.0   # fp32 tiny: full = 2.0
    assert row["ms_tok"] > 0 and row["tok_s"] > 0
    assert not tiny.model.layers[0].self_attn._forward_hooks              # tracker removed
    import json
    rec = json.loads(open(row["preds"]).readline())
    assert rec["new_tokens"] >= 1 and rec["kv_bytes"] > 0
    assert (rec["decoded_bytes"] > 0) == (method != "h2o")     # dense copy only for compressed methods


def test_runner_needs_dictionary():
    from tieredkv import eval as ev
    with pytest.raises(ValueError, match="dictionary"):
        ev.run("lexico", 0.2, "gsm8k", 1)


def test_results_header_mismatch_refused(tmp_path):
    from tieredkv import eval as ev
    p = tmp_path / "old.csv"
    p.write_text("method,acc\nfull,0.5\n")
    with pytest.raises(ValueError, match="columns"):
        ev.append_row({"method": "full"}, str(p))
