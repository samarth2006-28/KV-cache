import hashlib
import json
import random

import pytest

from tieredkv.h2o_reference_eval import fixed_subset, load_inputs


def test_subset_matches_shared_index_rule_and_target_extraction():
    data=[{"question":f"problem {i}","answer":f"reasoning\n#### {i}"} for i in range(20)]
    order=list(range(20));random.Random(0).shuffle(order)
    selected=fixed_subset(data,5,0)
    assert [r["idx"] for r in selected] == sorted(order[:5])
    assert all(r["target"] == str(r["idx"]) for r in selected)
    with pytest.raises(ValueError):fixed_subset(data,21,0)


def test_pinned_local_parquet_inputs_and_training_only_shots(tmp_path):
    pa=pytest.importorskip("pyarrow");pq=pytest.importorskip("pyarrow.parquet")
    source={"dataset":"unit-test-only","revision":"test","files":{}}
    for split,n in [("train",8),("test",3)]:
        path=tmp_path/f"{split}.parquet"
        rows=[{"question":f"{split} {i}","answer":f"#### {i}"} for i in range(n)]
        pq.write_table(pa.Table.from_pylist(rows),path)
        source["files"][split]={"rows":n,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/"source.json").write_text(json.dumps(source))
    probs,shots,meta=load_inputs(tmp_path,2,0)
    assert len(probs)==2 and len(shots)==5
    assert all(p["question"].startswith("test") for p in probs)
    assert all(s["question"].startswith("train") for s in shots)
    assert meta["revision"] == "test"
    (tmp_path/"test.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError,match="provenance"):load_inputs(tmp_path,2,0)
