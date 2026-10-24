"""Execute notebook startup and real child-process success/failure paths."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from tieredkv.notebook_support import dictionary_metadata,run_logged

REPO=Path(__file__).resolve().parents[1]


def notebook_cell(cell_id):
    notebook=json.loads((REPO/"notebooks/phase4_thresholds_colab.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in notebook["cells"] if c["id"]==cell_id)["source"])


def test_setup_makes_project_importable_in_kernel_style_sys_path(tmp_path,monkeypatch):
    project=tmp_path/"checkout";project.mkdir()
    probe=project/"_phase4_notebook_probe.py";probe.write_text("READY=True\n")
    monkeypatch.chdir(tmp_path)
    paths=[str(tmp_path),*[p for p in sys.path if p and Path(p).resolve()!=REPO]]
    monkeypatch.setattr(sys,"path",paths)
    monkeypatch.setenv("TIEREDKV_PROJECT",str(project))
    def output(command,**kwargs):
        if command[1]=="branch": return "b-allocator-phase4-final\n"
        if command[1]=="status": return ""
        return "test-commit\n"
    monkeypatch.setattr(subprocess,"check_output",output)
    monkeypatch.setattr(subprocess,"run",lambda *a,**kw:types.SimpleNamespace(returncode=0))
    assert importlib.util.find_spec("_phase4_notebook_probe") is None
    exec(compile(notebook_cell("phase4-01"),"setup-cell","exec"),{})
    spec=importlib.util.find_spec("_phase4_notebook_probe")
    assert Path(spec.origin)==probe


def test_logged_subprocess_streams_and_saves_output(tmp_path,capsys):
    log=tmp_path/"run.log"
    assert run_logged([sys.executable,"-u","-c","print('stage one'); print('stage two')"],log)==0
    assert "stage two" in capsys.readouterr().out
    assert log.read_text()=="stage one\nstage two\n"


def test_logged_subprocess_surfaces_original_error_and_checkpoint(tmp_path):
    result=tmp_path/"result";result.mkdir()
    (result/"experiment.json").write_text(json.dumps({"status":"failed","error":"ValueError: saved detail"}))
    log=tmp_path/"failure.log"
    command=[sys.executable,"-u","-c","import sys; print('loading model'); print('ORIGINAL_TRACEBACK',file=sys.stderr); sys.exit(7)"]
    with pytest.raises(RuntimeError) as exc:
        run_logged(command,log,result_dir=result)
    message=str(exc.value)
    assert "exit 7" in message and "ORIGINAL_TRACEBACK" in message and "saved detail" in message
    assert "loading model" in log.read_text()


def test_asset_error_names_missing_file(tmp_path):
    path=tmp_path/"missing.pt"
    with pytest.raises(ValueError,match="missing.pt"):
        dictionary_metadata(path,"model")
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError,match="missing.json"):
        dictionary_metadata(path,"model")


def test_notebook_launch_uses_logged_unbuffered_child():
    source=notebook_cell("phase4-09")
    assert '"-u"' in source and "run_logged(command,LOG" in source
    assert "subprocess.run(command)" not in source
    assert "notebook_support import dictionary_metadata" in notebook_cell("phase4-05")
