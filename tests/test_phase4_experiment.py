"""Protocol and result-integrity tests for the GPU Phase-4 experiment runner."""
import json
import types
import zipfile
from pathlib import Path

import pytest
import torch

from tieredkv import gsm8k
from tieredkv import phase4_experiment as experiment
from tieredkv.phase4_experiment import analyze_runs,dictionary_metadata
from tieredkv.threshold_ablation import VARIANTS


def make_runs(root):
    runs=[]
    for number,(method,fraction) in enumerate(VARIANTS.items()):
        folder=root/method
        folder.mkdir()
        records=[]
        for i in range(5):
            ok=i<number+2
            records.append({"idx":i,"target":"1","text":"#### 1" if ok else "#### 2",
                            "prompt_sha256":str(i)*64,"strict":ok,"flexible":ok,
                            "budget_measured":.198})
        pred=folder/"pred.jsonl"
        pred.write_text("\n".join(json.dumps(p) for p in records),encoding="utf-8")
        manifest={"run_id":method,"method":method,"n":5,"test_indices":list(range(5)),
                  "model":"snapshot","model_revision":"a"*40,"dictionary_sha256":"d"*64,
                  "dataset_revision":gsm8k.DATASET_REVISION,"seed":0,"max_new_tokens":256,
                  "dtype":"torch.float16","shot_indices":[10,11,12,13,14],"source_sha256":{"cache.py":"c"*64},
                  "score_mode":"mean","eligible":True,"budget_target":.2,
                  "effective_cache":{"budget_policy":"strict","evict_light":False,
                                     "tiers":[fraction,.2],"window":32,"chunk":8,"ratios":[32,12,4]}}
        path=folder/"manifest.json"
        path.write_text(json.dumps(manifest),encoding="utf-8")
        runs.append({"cfg":{"manifest":str(path),"preds":str(pred),"eligible":True,
                            "acc":(number+2)/5,"budget_measured":.198}})
    return runs


def rewrite(path,change):
    p=Path(path); data=json.loads(p.read_text(encoding="utf-8"))
    change(data)
    p.write_text(json.dumps(data),encoding="utf-8")


def test_complete_runs_rescore_answers_and_pair_all_thresholds(tmp_path):
    report=analyze_runs(make_runs(tmp_path))
    assert report["phase4_completed"] and report["status"]=="completed"
    assert [report["cells"][m]["strict_correct"] for m in VARIANTS]==[2,3,4]
    assert len(report["paired_mcnemar"])==3
    assert all(pair["p_bonferroni"]>=pair["p"] for pair in report["paired_mcnemar"])


@pytest.mark.parametrize("field,value",[
    ("dictionary_sha256","different"),("seed",1),("model","different"),
    ("max_new_tokens",128),("dataset_revision","changed"),
])
def test_mixed_experiments_rejected(tmp_path,field,value):
    runs=make_runs(tmp_path)
    rewrite(runs[1]["cfg"]["manifest"],lambda m:m.update({field:value}))
    with pytest.raises(ValueError,match="share"):
        analyze_runs(runs)


def test_budget_failures_do_not_produce_completed_phase_or_pairwise_claims(tmp_path):
    runs=make_runs(tmp_path)
    runs[0]["cfg"]["eligible"]=False
    rewrite(runs[0]["cfg"]["manifest"],lambda m:m.update(eligible=False))
    report=analyze_runs(runs)
    assert not report["phase4_completed"] and report["status"]=="budget_audit_failed"
    assert report["paired_mcnemar"]==[]


def test_edited_correctness_rejected(tmp_path):
    runs=make_runs(tmp_path)
    p=Path(runs[0]["cfg"]["preds"])
    lines=p.read_text().splitlines(); record=json.loads(lines[0])
    record["text"]="#### 2";lines[0]=json.dumps(record);p.write_text("\n".join(lines))
    with pytest.raises(ValueError,match="rescored"):
        analyze_runs(runs)


def test_missing_predictions_and_reordered_prompts_rejected(tmp_path):
    runs=make_runs(tmp_path)
    p=Path(runs[0]["cfg"]["preds"]); lines=p.read_text().splitlines()
    p.write_text("\n".join(lines[::-1]))
    with pytest.raises(ValueError,match="reordered"):
        analyze_runs(runs)


def test_dictionary_training_model_and_hash_checked(tmp_path):
    path=tmp_path/"dict.pt"
    torch.save(torch.ones(2,64,64),path)
    path.with_suffix(".json").write_text(json.dumps({"model":"expected","N":64}))
    result=dictionary_metadata(path,"expected")
    assert len(result["sha256"])==64
    with pytest.raises(ValueError,match="model"):
        dictionary_metadata(path,"other")
    with pytest.raises(ValueError,match="SHA256"):
        dictionary_metadata(path,"expected","f"*64)


def test_portable_bundle_can_be_reanalyzed_and_paths_cannot_escape(tmp_path):
    runs=make_runs(tmp_path)
    for run in runs:
        for key in ("preds","manifest"):
            run["cfg"][key]=Path(run["cfg"][key]).relative_to(tmp_path).as_posix()
    assert analyze_runs(runs,root=tmp_path)["phase4_completed"]
    runs[0]["cfg"]["manifest"]="../outside.json"
    with pytest.raises(ValueError,match="escapes"):
        analyze_runs(runs,root=tmp_path)


def test_strict_ceiling_is_independently_enforced_even_inside_one_pp(tmp_path):
    runs=make_runs(tmp_path)
    p=Path(runs[0]["cfg"]["preds"])
    records=[json.loads(line) for line in p.read_text().splitlines()]
    for r in records:r["budget_measured"]=.205
    p.write_text("\n".join(json.dumps(r) for r in records))
    runs[0]["cfg"]["budget_measured"]=.205
    report=analyze_runs(runs)
    assert not report["phase4_completed"]
    assert report["cells"]["tiered_hh3"]["budget_failures"]==5


@pytest.mark.parametrize("interrupt",[False,True])
def test_controller_checkpoints_portable_artifacts_and_archives_failures(tmp_path,monkeypatch,interrupt):
    # Mock GPU/model work only. Exercise the real controller, JSON/CSV journal,
    # independent analyzer, plot path and archive construction.
    model_dir=tmp_path/"snapshot";model_dir.mkdir()
    dictionary=tmp_path/"dictionary.pt"
    torch.save(torch.ones(2,64,64),dictionary)
    dictionary.with_suffix(".json").write_text(json.dumps({"model":"target"}))
    fake=types.SimpleNamespace(config=types.SimpleNamespace(head_dim=64,num_hidden_layers=1,_commit_hash="a"*40),
                               device=torch.device("cuda"),
                               parameters=lambda:iter([torch.ones(1,dtype=torch.float16)]))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(experiment.ev,"load_model",lambda name:(fake,None))
    monkeypatch.setattr(gsm8k,"load_subset",lambda *a,**kw:[{"idx":i} for i in range(5)])
    monkeypatch.setattr(gsm8k,"fewshot",lambda *a,**kw:[])
    monkeypatch.setattr(experiment,"generation_gates",lambda *a,**kw:[{"idx":i,"passed":True} for i in range(5)])
    prepared=[];calls=[]
    def fake_run(method,budget,task,n,seed,**kwargs):
        calls.append(method)
        if interrupt and len(calls)==2:
            raise RuntimeError("simulated GPU interruption")
        if not prepared:
            prepared.extend(make_runs(Path(kwargs["results"]).parent))
        result=prepared[list(VARIANTS).index(method)]
        row={"method":method,"budget_target":.2,"budget_measured":.198,
             "acc":result["cfg"]["acc"],"n":5,"seed":0,"model":"snapshot","task":"gsm8k",
             "run_id":method,"eligible":True,"kv_mb":1,"peak_gb":0,"tok_s":1,"ms_tok":1}
        experiment.ev.append_row(row,kwargs["results"])
        return result
    monkeypatch.setattr(experiment.ev,"run",fake_run)
    out=tmp_path/"output"
    args=types.SimpleNamespace(model=str(model_dir),model_id="target",model_revision="a"*40,
        dictionary=str(dictionary),dictionary_sha256=None,n=5,seed=0,max_new_tokens=8,out=str(out))
    if interrupt:
        with pytest.raises(RuntimeError,match="interruption"):
            experiment.run_experiment(args)
    else:
        assert experiment.run_experiment(args)==0
    saved=json.loads((out/"experiment.json").read_text())
    assert saved["phase4_completed"] is (not interrupt)
    assert len(saved["runs"])==(1 if interrupt else 3)
    for run in saved["runs"]:
        assert not Path(run["cfg"]["preds"]).is_absolute()
        assert (out/run["cfg"]["preds"]).is_file()
    with zipfile.ZipFile(str(out)+".zip") as archive:
        assert archive.testzip() is None and "experiment.json" in archive.namelist()
    if not interrupt:
        assert analyze_runs(saved["runs"],root=out)["phase4_completed"]
