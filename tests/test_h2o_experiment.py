"""Matrix integrity, failure recovery, and real independent H2O generation gates."""
import json
from pathlib import Path
import types
import zipfile

import pytest
import torch

from tieredkv import gsm8k, h2o_experiment as experiment


def make_runs(root):
    runs = []
    for number, (method, budget) in enumerate([("full", 1.0), *[("h2o", b) for b in experiment.BUDGETS]]):
        folder = root / str(number)
        folder.mkdir()
        records = [{"idx": i, "target": "1", "text": "#### 1" if i < number else "#### 2",
                    "prompt_sha256": str(i)*64, "strict": i < number, "flexible": i < number,
                    "budget_measured": budget-.001} for i in range(5)]
        pred = folder/"predictions.jsonl"
        pred.write_text("\n".join(json.dumps(p) for p in records), encoding="utf-8")
        manifest = {"run_id": str(number), "method": method, "budget_target": budget,
                    "n": 5, "test_indices": list(range(5)), "score_mode": "sum" if method == "h2o" else None,
                    "dtype": "torch.float16", "model": "snapshot", "model_revision": "a"*40,
                    "dictionary_sha256": None, "dataset_revision": gsm8k.DATASET_REVISION,
                    "seed": 0, "max_new_tokens": 256, "shot_indices": [10, 11, 12, 13, 14],
                    "source_sha256": {"cache.py": "c"*64}, "eligible": True,
                    "effective_cache": {"window": 32, "chunk": 8}}
        path = folder/"manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        runs.append({"cfg": {"method": method, "budget_target": budget, "manifest": str(path),
                             "preds": str(pred), "acc": number/5, "acc_flex": number/5,
                             "budget_measured": budget-.001, "eligible": True}})
    return runs


def rewrite(path, change):
    path = Path(path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_matrix_requires_all_budgets_and_pairs_full_with_each(tmp_path):
    runs = make_runs(tmp_path)
    report = experiment.analyze_runs(runs)
    assert report["h2o_completed"] and len(report["cells"]) == 5
    assert len(report["paired_mcnemar"]) == 4
    assert all(p["p_bonferroni"] >= p["p"] for p in report["paired_mcnemar"])
    with pytest.raises(ValueError, match="four"):
        experiment.analyze_runs(runs[:-1])
    runs[-1] = runs[-2]
    with pytest.raises(ValueError, match="duplicate"):
        experiment.analyze_runs(runs)


@pytest.mark.parametrize("field,value", [("model_revision", "b"*40), ("seed", 1),
                                         ("shot_indices", [1, 2, 3, 4, 5]), ("source_sha256", {})])
def test_mixed_cohorts_rejected(tmp_path, field, value):
    runs = make_runs(tmp_path)
    rewrite(runs[-1]["cfg"]["manifest"], lambda m: m.update({field: value}))
    with pytest.raises(ValueError, match="share"):
        experiment.analyze_runs(runs)


@pytest.mark.parametrize("field,value", [("score_mode", "mean"), ("dtype", "torch.float32"),
                                         ("dataset_revision", "changed"), ("model_revision", None)])
def test_invalid_protocol_rejected(tmp_path, field, value):
    runs = make_runs(tmp_path)
    rewrite(runs[-1]["cfg"]["manifest"], lambda m: m.update({field: value}))
    with pytest.raises(ValueError, match="protocol"):
        experiment.analyze_runs(runs)


def test_per_problem_budget_failure_blocks_matrix_and_statistics(tmp_path):
    runs = make_runs(tmp_path)
    pred = Path(runs[1]["cfg"]["preds"])
    records = [json.loads(line) for line in pred.read_text().splitlines()]
    records[0]["budget_measured"] = .07
    mean = sum(p["budget_measured"] for p in records)/5
    runs[1]["cfg"]["budget_measured"] = round(mean, 4)
    pred.write_text("\n".join(json.dumps(p) for p in records))
    report = experiment.analyze_runs(runs)
    assert not report["h2o_completed"]
    assert report["cells"]["h2o_b0.1"]["budget_failures"] == 1
    assert report["paired_mcnemar"] == []


def test_raw_text_and_csv_metrics_checked_independently(tmp_path):
    runs = make_runs(tmp_path)
    runs[0]["cfg"]["acc"] = .2
    with pytest.raises(ValueError, match="metrics"):
        experiment.analyze_runs(runs)
    runs[0]["cfg"]["acc"] = 0
    pred = Path(runs[0]["cfg"]["preds"])
    pred.write_text(pred.read_text().replace('#### 2', '#### 1'))
    with pytest.raises(ValueError, match="rescored"):
        experiment.analyze_runs(runs)


def test_portable_bundle_and_escape_guard(tmp_path):
    runs = make_runs(tmp_path)
    for run in runs:
        for key in ("preds", "manifest"):
            run["cfg"][key] = Path(run["cfg"][key]).relative_to(tmp_path).as_posix()
    assert experiment.analyze_runs(runs, root=tmp_path)["h2o_completed"]
    runs[0]["cfg"]["manifest"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes"):
        experiment.analyze_runs(runs, root=tmp_path)


@pytest.mark.parametrize("interrupt", [False, True])
def test_controller_preserves_partial_runs_and_archive(tmp_path, monkeypatch, interrupt):
    model_dir = tmp_path/"snapshot"
    model_dir.mkdir()
    fake = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="a"*40),
                                 device=torch.device("cuda"),
                                 parameters=lambda: iter([torch.ones(1, dtype=torch.float16)]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiment.ev, "load_model", lambda name: (fake, None))
    monkeypatch.setattr(gsm8k, "load_subset", lambda *a, **kw: [{"idx": i} for i in range(5)])
    monkeypatch.setattr(gsm8k, "fewshot", lambda *a, **kw: [])
    monkeypatch.setattr(experiment, "score_probe", lambda *a: {"prompt_count": 20})
    monkeypatch.setattr(experiment, "generation_gates", lambda *a: [{"passed": True}])
    prepared, calls = [], []

    def fake_run(method, budget, task, n, seed, **kw):
        calls.append((method, budget))
        if interrupt and len(calls) == 3:
            raise RuntimeError("simulated GPU interruption")
        if not prepared:
            prepared.extend(make_runs(Path(kw["results"]).parent))
        result = prepared[len(calls)-1]
        experiment.ev.append_row({**result["cfg"], "n": 5, "seed": 0, "model": "snapshot",
                                  "task": "gsm8k", "run_id": str(len(calls)), "kv_mb": 1,
                                  "peak_gb": 0, "tok_s": 1, "ms_tok": 1}, kw["results"])
        return result

    monkeypatch.setattr(experiment.ev, "run", fake_run)
    out = tmp_path/"output"
    args = types.SimpleNamespace(model=str(model_dir), model_id="target", model_revision="a"*40,
                                 out=str(out), n=5, seed=0, max_new_tokens=8)
    if interrupt:
        with pytest.raises(RuntimeError, match="interruption"):
            experiment.run_experiment(args)
    else:
        assert experiment.run_experiment(args) == 0
    saved = json.loads((out/"experiment.json").read_text())
    assert saved["h2o_completed"] is (not interrupt)
    assert len(saved["runs"]) == (2 if interrupt else 5)
    assert all(not Path(r["cfg"]["preds"]).is_absolute() for r in saved["runs"])
    with zipfile.ZipFile(str(out)+".zip") as archive:
        assert archive.testzip() is None and "experiment.json" in archive.namelist()
    if not interrupt:
        assert experiment.analyze_runs(saved["runs"], root=out)["h2o_completed"]


def test_real_tiny_llama_gates_use_shared_and_independent_cache(monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=512, eos_token_id=None, pad_token_id=0,
                         attn_implementation="eager")
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(1, 32, (1, 400))
    tok = types.SimpleNamespace(eos_token_id=0)
    class Tokenizer:
        eos_token_id = tok.eos_token_id
        def __call__(self, *a, **kw):
            return {"input_ids": ids}
    monkeypatch.setattr(gsm8k, "build_prompt", lambda *a: "prompt")
    checks = experiment.generation_gates(model, Tokenizer(), [{"idx": 1, "question": "q"}], [])
    assert checks[0]["full_equals_stock"] and all(checks[0]["h2o"].values())


def test_attention_probe_saves_twenty_aligned_sum_count_dumps(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              attn_implementation="eager")).eval()
    class Tokenizer:
        def __call__(self, *a, **kw):
            return {"input_ids": torch.arange(1, 21).reshape(1, 20)}
    monkeypatch.setattr(gsm8k, "build_prompt", lambda question, shots: question)
    out = tmp_path/"attention"
    result = experiment.score_probe(model, Tokenizer(), [{"idx": i, "question": str(i)} for i in range(20)], [], out)
    assert result["layer_records"] == 20 and len(list(out.glob("*.pt"))) == 20
    dump = torch.load(out/"prompt_0.pt", weights_only=True)["layers"][0]
    torch.testing.assert_close(dump["scores"], dump["sums"]/dump["counts"])
    summary = json.loads((out/"summary.json").read_text())
    assert summary["diagnostic_only"] and len(summary["prompts"]) == 20
    assert all("raw_top_attention_mass_share" in row and "mean_top_score_share" in row for row in summary["rows"])
