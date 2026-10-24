import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tieredkv.allocator import ScoreTracker
from tieredkv.h2o_baseline import H2OReferenceCache, h2o_generate


@pytest.fixture
def model():
    cfg=LlamaConfig(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=256,eos_token_id=None,pad_token_id=0)
    cfg._attn_implementation="eager"
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def test_full_budget_matches_stock_greedy(model):
    ids=torch.tensor([[1,7,13,22,31,17,5,6]])
    stock=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=12,
                         do_sample=False,pad_token_id=0)
    ours=h2o_generate(model,ids,budget=1.0,window=2,max_new_tokens=12)
    assert torch.equal(stock,ours.sequences)
    assert all(s["kv_fraction"] == 1 for s in ours.samples)
    assert all(not l.self_attn._forward_hooks for l in model.model.layers)


@torch.no_grad()
def test_pruned_generation_uses_absolute_positions_and_budget(model):
    positions=[]
    def capture(module,args,kwargs):positions.append(kwargs["position_ids"].clone())
    handle=model.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        out=h2o_generate(model,torch.arange(1,17)[None],budget=.25,window=2,max_new_tokens=8)
    finally:handle.remove()
    assert out.sequences.shape == (1,24)
    assert positions[0].tolist() == [list(range(16))]
    assert [int(p.item()) for p in positions[1:]] == list(range(16,23))
    for sample in out.samples:
        assert sample["tokens_stored"] == int(sample["tokens_seen"]*.25)
        assert sample["kv_fraction"] <= .25
        assert sample["auxiliary_bytes"] > 0
    assert out.peak_materialized_kv_bytes > out.samples[0]["kv_bytes"]


@torch.no_grad()
def test_pruning_preserves_rotated_keys_and_recent_original_indices(model):
    ids=torch.arange(1,17)[None]
    with ScoreTracker(model,mode="sum") as tr:
        cache=H2OReferenceCache(tr,budget=.25,window=2)
        model(ids,past_key_values=cache,use_cache=True)
        keys=[x.clone() for x in cache.key_cache]
        cache.prune(16)
        for layer,pos in enumerate(cache.positions):
            assert pos[-2:].tolist() == [14,15]
            assert (pos[1:] > pos[:-1]).all()
            torch.testing.assert_close(cache.key_cache[layer],keys[layer].index_select(-2,pos))
            assert tr.scores(layer).shape[-1] == cache.get_seq_length(layer)


def test_impossible_window_and_training_mode_fail_before_forward(model):
    with pytest.raises(ValueError,match="recent"):
        h2o_generate(model,torch.ones(1,10,dtype=torch.long),budget=.1,window=2)
    model.train()
    with pytest.raises(ValueError,match="eval"):
        h2o_generate(model,torch.ones(1,10,dtype=torch.long),window=0)


def test_context_and_batch_guards(model):
    with pytest.raises(ValueError,match="context"):
        h2o_generate(model,torch.ones(1,250,dtype=torch.long),window=0,max_new_tokens=10)
    with pytest.raises(ValueError,match="one"):
        h2o_generate(model,torch.ones(2,10,dtype=torch.long),window=0)


@torch.no_grad()
def test_eos_stops_without_feeding_an_extra_token(model):
    ids=torch.arange(1,9)[None]
    eos=int(model(ids).logits[:,-1].argmax())
    model.generation_config.eos_token_id=eos
    out=h2o_generate(model,ids,window=2,max_new_tokens=12)
    assert out.sequences.shape[-1] == 9
    assert len(out.samples) == 1 and out.samples[0]["tokens_seen"] == 8
