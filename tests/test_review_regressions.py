"""Regressions found while joining the allocator and compressed-cache branches."""
import json
import string
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tieredkv import compress as C, eval as ev, gsm8k
from tieredkv.allocator import ScoreTracker, allocate
from tieredkv.cache import TieredKVCache, CHUNK
from tieredkv.h2o_baseline import h2o_generate


@pytest.fixture
def tiny():
    torch.manual_seed(42)
    cfg=LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
                    num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                    max_position_embeddings=1024,eos_token_id=None)
    cfg._attn_implementation="eager"
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(autouse=True)
def restore_shape():
    yield
    C.configure(n_kv_heads=8,head_dim=128)


@pytest.mark.parametrize("kwargs",[
    {"avg_s":float("nan")},{"avg_s":-1},{"tiers":(-.1,.2)},
    {"tiers":(.9,.2)},{"ratios":(0,12,4)},{"s_max":0},
])
def test_bad_allocation_settings_rejected(kwargs):
    kw={"avg_s":8,**kwargs}
    with pytest.raises(ValueError):
        allocate(torch.ones(100),**kw)


def test_nonfinite_scores_and_empty_batches():
    with pytest.raises(ValueError,match="finite"):
        allocate(torch.tensor([1.,float("nan")]),8)
    assert allocate(torch.ones(0,100),8).shape==(0,100)


@pytest.mark.parametrize("counts",[-1,1.5,float("nan"),9])
def test_bad_atom_counts_rejected(counts):
    with pytest.raises(ValueError,match="integer counts"):
        C.encode_kv(torch.ones(2,1,4),torch.eye(4).repeat(1,2),counts)


def test_fp8_select_keeps_exact_bits_and_empty_selection():
    D=torch.eye(4)
    kv=torch.randn(3,1,4)
    sp=C.encode_kv(kv,D,2)
    kept=sp.select(torch.tensor([2,0,2]))
    expected=sp.data.view(torch.uint8).reshape(3,2)[[2,0,2]].flatten()
    assert kept.data.dtype==sp.data.dtype
    assert torch.equal(kept.data.view(torch.uint8),expected)
    empty=sp.select(torch.tensor([],dtype=torch.long))
    assert empty.n_tokens==0 and empty.data.numel()==0


@pytest.mark.parametrize("kwargs",[
    {"budget":0},{"budget":float("nan")},{"window":-1},{"chunk":0},
    {"expected_new_tokens":-1},{"avg_s":1.5},
])
def test_bad_cache_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        TieredKVCache("full",**kwargs)


@pytest.mark.parametrize("seed",[0,1])
def test_shared_h2o_matches_independent_reference_after_eviction(tiny,seed):
    ids=torch.randint(1,128,(1,40),generator=torch.Generator().manual_seed(seed))
    ref=h2o_generate(tiny,ids,budget=.5,window=4,max_new_tokens=12)
    with ScoreTracker(tiny,mode="sum") as tracker:
        cache=TieredKVCache("h2o",tracker=tracker,budget=.5,window=4)
        positions=[]
        handle=tiny.model.rotary_emb.register_forward_pre_hook(
            lambda module,args: positions.append(args[1].detach().clone()))
        try:
            actual=tiny.generate(ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
                                 max_new_tokens=12,do_sample=False,pad_token_id=0)
        finally:
            handle.remove()
        assert tracker.acc[0].shape[-1]==cache.stored_tokens()
    assert torch.equal(actual,ref.sequences)
    assert [int(p[0,-1]) for p in positions]==list(range(39,51))
    assert cache.stored_tokens()<cache.get_seq_length()


def test_small_dictionary_caps_heavy_atoms(tiny):
    D=torch.nn.functional.normalize(torch.randn(4,16,4),dim=1)
    ids=torch.randint(1,128,(1,40))
    with ScoreTracker(tiny,mode="mean") as tracker:
        cache=TieredKVCache("tiered",tracker=tracker,dictionary=D,avg_s=3,window=4,chunk=2)
        tiny.generate(ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
                      max_new_tokens=4,do_sample=False,pad_token_id=0)
    assert int(cache.token_s().max())<=4


def test_config_chunk_matches_integrated_cache():
    root=Path(__file__).resolve().parents[1]/"configs"
    for path in root.glob("*_b20.json"):
        assert json.loads(path.read_text())["cfg"]["chunk"]==CHUNK


@pytest.mark.parametrize("kwargs",[{"n":0},{"budget":0},{"max_new_tokens":0}])
def test_runner_invalid_settings_fail_before_loading(monkeypatch,kwargs):
    def forbidden(*args,**kw):
        pytest.fail("invalid settings reached model loading")
    monkeypatch.setattr(ev,"load_model",forbidden)
    with pytest.raises(ValueError):
        ev.run("full",kwargs.get("budget",1),"gsm8k",kwargs.get("n",1),
               max_new_tokens=kwargs.get("max_new_tokens",1))


def test_run_files_are_unique_and_failed_budgets_stay_flagged(tiny,tmp_path,monkeypatch):
    vocab={"<unk>":0,**{c:i+1 for i,c in enumerate(string.printable)}}
    tokenizer=Tokenizer(models.WordLevel(vocab,unk_token="<unk>"))
    tokenizer.pre_tokenizer=pre_tokenizers.Split("","isolated")
    tokenizer.decoder=decoders.Fuse()
    tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token="<unk>",eos_token="<unk>")
    probs=[{"idx":120,"question":"1+1?","answer":"#### 2","target":"2"}]
    monkeypatch.setitem(ev._loaded,"review-tiny",(tiny,tok))
    monkeypatch.setattr(gsm8k,"load_subset",lambda *a,**kw:probs)
    monkeypatch.setattr(gsm8k,"fewshot",lambda *a,**kw:[])
    path=str(tmp_path/"results.csv")
    a=ev.run("full",1,"gsm8k",1,model_name="review-tiny",results=path,max_new_tokens=2)
    old=Path(a["cfg"]["preds"]).read_bytes()
    b=ev.run("full",1,"gsm8k",1,model_name="review-tiny",results=path,max_new_tokens=2)
    assert a["cfg"]["preds"]!=b["cfg"]["preds"]
    assert Path(a["cfg"]["preds"]).read_bytes()==old
    manifest=json.loads(Path(b["cfg"]["manifest"]).read_text())
    assert manifest["test_indices"]==[120]
    # CPU float32 full cache is 2x the explicit fp16 denominator.
    assert not manifest["eligible"] and not b["cfg"]["eligible"]
    assert manifest["problem_audits"][0]["difference_pp"]==100


def test_config_cli_passes_threshold_and_honors_overrides(tmp_path,monkeypatch):
    p=tmp_path/"config.json"
    p.write_text(json.dumps({"method":"tiered_hh3","budget":.2,"seed":7,
                            "score_mode":"mean","cfg":{"chunk":8,"tiers":[.03,.2]}}))
    captured={}
    def fake_run(*args,**kwargs):
        captured.update({"args":args,**kwargs})
        return {"acc":0,"cfg":{"eligible":True}}
    monkeypatch.setattr(ev,"run",fake_run)
    monkeypatch.setattr("sys.argv",["eval","--config",str(p),"--n","3","--chunk","4",
                                   "--dictionary","dict.pt"])
    assert ev.main()==0
    assert captured["args"]==("tiered_hh3",.2,"gsm8k",3,7)
    assert captured["cfg"]["chunk"]==4 and captured["cfg"]["score_mode"]=="mean"
    assert captured["cfg"]["tiers"]==[.03,.2]


def test_plot_rejects_mixed_cohorts_and_flags_budget_outliers(tmp_path):
    import plot
    path=tmp_path/"mixed.csv"
    path.write_text("method,budget_target,budget_measured,acc,n,seed,model\n"
                    "tiered,.2,.2,.5,5,0,a\nlexico,.2,.2,.5,5,0,b\n")
    with pytest.raises(ValueError,match="mixed"):
        plot.load(path)
    path.write_text("method,budget_target,budget_measured,acc,n,seed,model,eligible\n"
                    "tiered,.2,.1848,.5,5,0,a,False\n"
                    "lexico,.2,.2,.5,5,0,a,False\n"
                    "h2o,.2,.2,.5,5,0,a,True\n")
    rows,report=plot.comparison_rows(plot.load(path))
    assert [r["method"] for r in rows]==["h2o"]
    assert sum(not r["eligible"] for r in report)==2
    # Fractions above 1 must not be guessed to be percentages.
    assert plot._pct(2)==200
    assert plot._pct(.5,"percent")==.5


def test_retraining_same_dictionary_path_invalidates_cache(tmp_path,monkeypatch):
    from tieredkv import cache as module
    monkeypatch.setattr(module,"_DICTS",{})
    path=tmp_path/"dictionary.pt"
    torch.save(torch.ones(4,16,4),path)
    first=module._load_dictionary(path)
    torch.save(torch.zeros(4,16,5),path)
    second=module._load_dictionary(path)
    assert first.shape[-1]==4 and second.shape[-1]==5
    assert len(module._DICTS)==1 and second.count_nonzero()==0


def test_gsm8k_subset_size_and_pinned_revision(monkeypatch):
    import sys
    import types
    seen=[]
    def load_dataset(*args,**kwargs):
        seen.append(kwargs)
        return [{"question":"one?","answer":"#### 1"}]*3
    monkeypatch.setitem(sys.modules,"datasets",types.SimpleNamespace(load_dataset=load_dataset))
    result=gsm8k.load_subset(2)
    assert len(result)==2 and seen[-1]["revision"]==gsm8k.DATASET_REVISION
    for n in (0,-1,4,1.5):
        with pytest.raises(ValueError):
            gsm8k.load_subset(n)


def test_wrong_dictionary_head_dimension_fails_clearly(tiny):
    ids=torch.ones(1,40,dtype=torch.long)
    with ScoreTracker(tiny,mode="mean") as tracker:
        cache=TieredKVCache("tiered",tracker=tracker,dictionary=torch.ones(4,17,8),
                           avg_s=3,window=4,chunk=2)
        with pytest.raises(ValueError,match="dictionary must be"):
            tiny.generate(ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
                          max_new_tokens=2,do_sample=False,pad_token_id=0)


def test_decode_chunks_respect_dense_window_boundary():
    cache=TieredKVCache("lexico",window=4,chunk=2,avg_s=2,
                       encode_fn=lambda kv,D,s:kv.clone(),
                       decode_fn=lambda kv,D,dtype:kv.to(dtype))
    prefill=torch.zeros(1,1,40,16)
    cache.update(prefill,prefill,0)
    for _ in range(10):
        token=torch.zeros(1,1,1,16)
        cache.update(token,token,0)
        assert cache.key_cache[0].shape[-2]<=6
    assert all(part.shape[0]==2 for part in cache.sparse[0]["k"][1:])
