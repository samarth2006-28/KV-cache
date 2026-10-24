"""Actual CSR byte audits for the optional Phase-4 budget scheduler."""
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM

from tieredkv import compress as C
from tieredkv.allocator import ScoreTracker
from tieredkv.cache import TieredKVCache


class Tracker:
    mode="mean"
    def __init__(self): self.acc={}
    def reset(self): self.acc.clear()
    def scores(self,layer): return self.acc[layer].clone()


def compact_payload(kv,D,s):
    h=kv.shape[1]
    counts=s.repeat_interleave(h)
    ptr=torch.zeros(counts.numel()+1,dtype=torch.int32)
    ptr[1:]=counts.cumsum(0)
    nnz=int(ptr[-1])
    return C.SparseKV(ptr,torch.zeros(nnz,dtype=torch.int16),
                      torch.zeros(nnz,dtype=torch.float8_e4m3fn),h)


def zero_decode(sp,D,dtype):
    return torch.zeros(sp.n_tokens,sp.n_heads,D.shape[0],dtype=dtype)


@pytest.fixture(autouse=True)
def restore_shape():
    yield
    C.configure(n_kv_heads=8,head_dim=128)


@pytest.mark.parametrize("prompt,dim,heads",[(641,128,8),(728,64,3),(789,64,3)])
def test_all_thresholds_match_actual_csr_bytes_and_ceiling(prompt,dim,heads):
    trajectories=[]
    for fraction in (.03,.05,.10):
        tracker=Tracker()
        D=torch.ones(2,dim,64)
        cache=TieredKVCache("tiered",tracker=tracker,dictionary=D,budget=.20,
                            window=32,chunk=8,tiers=(fraction,.20),budget_policy="strict",
                            encode_fn=compact_payload,decode_fn=zero_decode)
        prefill=torch.zeros(1,heads,prompt,dim,dtype=torch.float16)
        cache.update(prefill,prefill,0)
        tracker.acc[0]=torch.arange(prompt,dtype=torch.float32)[None]
        samples=[]
        for _ in range(40):
            token=torch.zeros(1,heads,1,dim,dtype=torch.float16)
            cache.update(token,token,0)
            tracker.acc[0]=torch.arange(cache.stored_tokens(),dtype=torch.float32)[None]
            fraction_used=cache.measured_budget()
            assert .19 <= fraction_used <= .20+1e-12
            samples.append(cache.kv_bytes())
        trajectories.append(samples)
    # This checks actual dense+FP8/int16/int32 buffers, not the planning formula.
    assert trajectories[0]==trajectories[1]==trajectories[2]


def test_strict_real_encoder_generate():
    torch.manual_seed(2)
    cfg=LlamaConfig(vocab_size=128,hidden_size=128,intermediate_size=192,
                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                    max_position_embeddings=512,eos_token_id=None)
    cfg._attn_implementation="eager"
    model=LlamaForCausalLM(cfg).half().eval()
    D=torch.nn.functional.normalize(torch.randn(2,64,64),dim=1)
    ids=torch.randint(1,128,(1,100))
    with ScoreTracker(model,mode="mean") as tracker:
        cache=TieredKVCache("tiered",tracker=tracker,dictionary=D,budget=.2,
                            window=4,chunk=2,budget_policy="strict")
        out=model.generate(ids,past_key_values=cache,attention_mask=torch.ones_like(ids),
                           max_new_tokens=8,do_sample=False,pad_token_id=0)
        assert tracker.acc[0].shape[-1]==cache.stored_tokens()
    assert out.shape[-1]==108
    assert .19<=cache.measured_budget()<=.20
    assert cache.sparse[0]["k"][0].data.dtype==torch.float8_e4m3fn


@pytest.mark.parametrize("kwargs",[
    {"method":"lexico"},{"evict_light":True},{"avg_s":4},{"budget_policy":"guess"},
])
def test_unsupported_strict_contract_rejected(kwargs):
    kw={"method":"tiered","tracker":Tracker(),"dictionary":torch.ones(2,64,64),
        "budget_policy":"strict",**kwargs}
    with pytest.raises(ValueError):
        TieredKVCache(**kw)


def test_strict_rejects_fp32_denominator_mismatch():
    tracker=Tracker()
    cache=TieredKVCache("tiered",tracker=tracker,dictionary=torch.ones(2,64,64),
                        budget=.2,budget_policy="strict")
    prefill=torch.zeros(1,1,728,64)
    cache.update(prefill,prefill,0)
    tracker.acc[0]=torch.ones(1,728)
    with pytest.raises(ValueError,match="fp16"):
        cache.update(torch.zeros(1,1,1,64),torch.zeros(1,1,1,64),0)


def test_strict_rejects_encoder_that_retains_dense_payload():
    tracker=Tracker()
    cache=TieredKVCache("tiered",tracker=tracker,dictionary=torch.ones(2,64,64),
                        budget=.2,budget_policy="strict",encode_fn=lambda kv,D,s:kv.clone())
    prefill=torch.zeros(1,1,728,64,dtype=torch.float16)
    cache.update(prefill,prefill,0)
    tracker.acc[0]=torch.ones(1,728)
    with pytest.raises(RuntimeError,match="payload exceeds"):
        token=torch.zeros(1,1,1,64,dtype=torch.float16)
        cache.update(token,token,0)
