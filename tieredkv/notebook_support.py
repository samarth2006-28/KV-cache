"""Lightweight notebook asset checks and streamed subprocess diagnostics.

No Torch/Transformers import: setup can run without loading stale kernel modules.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess


def dictionary_metadata(path,model_id,expected_sha256=None):
    path=Path(path)
    if not path.is_file():
        raise ValueError(f"Trained dictionary not found: {path}. Set DICT_SOURCE to the existing .pt file.")
    meta_path=path.with_suffix(".json")
    if not meta_path.is_file():
        raise ValueError(f"Dictionary training metadata not found: {meta_path}. Copy the matching training .json beside the .pt.")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta,dict) or meta.get("model")!=model_id:
        actual=meta.get("model") if isinstance(meta,dict) else None
        raise ValueError(f"Dictionary training model {actual!r} does not match --model-id {model_id!r}.")
    def sha256(file):
        digest=hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):
                digest.update(block)
        return digest.hexdigest()
    digest=sha256(path)
    if expected_sha256 and digest!=expected_sha256.strip().lower():
        raise ValueError("Dictionary SHA256 differs from expected value")
    return {"sha256":digest,"metadata_sha256":sha256(meta_path),"training":meta}


def run_logged(command,log_path,*,result_dir=None,cwd=None):
    """Stream combined output, save it, and raise a useful error on child failure."""
    log_path=Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    tail=deque(maxlen=20)
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line);log.flush()
                tail.append(line.rstrip())
            code=process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        status="process_failed";detail=""
        if result_dir is not None:
            try:
                state=json.loads((Path(result_dir)/"experiment.json").read_text(encoding="utf-8"))
                status=state.get("status",status)
                detail=state.get("error","Inspect per-problem budget audits." if status=="budget_audit_failed" else "")
            except (OSError,ValueError):
                pass
        raise RuntimeError(f"Phase 4 stopped: {status} (exit {code}). {detail}\n"
                           f"Full log: {log_path}\nLast output:\n"+"\n".join(tail))
    return code
