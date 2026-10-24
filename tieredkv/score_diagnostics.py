"""Dump per-layer scores for local-model prompts; not a task-accuracy benchmark.

python -m tieredkv.score_diagnostics --model LOCAL_DIR --prompts prompts.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from .allocator import ScoreTracker
from .sanity import spearman


def summarize_scores(sums, counts, fraction=.05):
    """Paired SUM/MEAN diagnostics for one prompt/layer, not accuracy metrics."""
    if sums.dim() == 2 and sums.shape[0] == 1:
        sums, counts = sums[0], counts[0]
    if sums.dim() != 1 or sums.shape != counts.shape or not 0 < fraction <= 1:
        raise ValueError("expected aligned [tokens] or [1,tokens] and fraction in (0,1]")
    if not torch.isfinite(sums).all() or not torch.isfinite(counts).all() or (sums < 0).any() or (counts < 0).any():
        raise ValueError("scores/counts must be finite and nonnegative")
    if ((counts == 0) & (sums != 0)).any():
        raise ValueError("unobserved tokens cannot have accumulated attention")
    valid = counts > 0
    positions = torch.arange(sums.numel(), device=sums.device)[valid]
    raw, count = sums[valid].float(), counts[valid].float()
    mean = raw/count
    n = raw.numel()
    top_n = math.ceil(fraction*n)
    raw_idx = torch.argsort(raw, descending=True, stable=True)[:top_n]
    mean_idx = torch.argsort(mean, descending=True, stable=True)[:top_n]
    def share(values, idx):
        total = float(values.sum())
        return float(values[idx].sum())/total if total else 0.0
    return {
        "tokens": sums.numel(), "observed_tokens": n, "top_count": top_n,
        "realized_top_fraction": top_n/n if n else 0.0,
        "raw_top_attention_mass_share": share(raw, raw_idx),
        "mean_top_score_share": share(mean, mean_idx),
        "top_overlap_fraction": len(set(raw_idx.tolist()) & set(mean_idx.tolist()))/top_n if top_n else None,
        "raw_position_spearman": spearman(positions.cpu().numpy(), raw.cpu().numpy()),
        "mean_position_spearman": spearman(positions.cpu().numpy(), mean.cpu().numpy()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Existing local HF model/tokenizer directory")
    p.add_argument("--prompts", required=True, help="JSON list of diagnostic prompt strings")
    p.add_argument("--out", default="results/attention")
    p.add_argument("--mode", choices=["mean", "sum"], default="mean")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--chat-template", action="store_true", help="Use the tokenizer's instruction template")
    p.add_argument("--model-id", help="Public model ID for provenance; loading still uses --model locally")
    p.add_argument("--revision", help="Revision of the already-downloaded local model")
    args = p.parse_args()
    if min(args.n, args.max_new_tokens, args.max_input_tokens, args.threads) < 1:
        p.error("prompt count and token limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable in this Torch build")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) < args.n or any(not isinstance(x, str) or not x.strip() for x in prompts[:args.n]):
        p.error("prompts must contain at least n nonempty strings")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    generations = []
    started = time.perf_counter()
    with torch.inference_mode(), ScoreTracker(model, mode=args.mode, check=True) as tracker:
        for i, prompt in enumerate(prompts[:args.n]):
            tracker.reset()
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True) if args.chat_template else prompt
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_input_tokens).to(args.device)
            # Tracker's pre-hook requests weights locally; generate does not retain
            # all attention matrices. One unpadded prompt per call.
            generated = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            scores = {layer: tracker.scores(layer).detach().cpu() for layer in tracker.acc}
            torch.save({"format_version": 2, "scores": scores, "mode": args.mode,
                        "sums": {layer: value.cpu() for layer, value in tracker.acc.items()},
                        "counts": {layer: value.cpu() for layer, value in tracker.cnt.items()}},
                       out / f"prompt_{i:02d}.pt")
            for layer, values in scores.items():
                metrics = summarize_scores(tracker.acc[layer], tracker.cnt[layer])
                selected = metrics["mean_top_score_share"] if args.mode == "mean" else metrics["raw_top_attention_mass_share"]
                rows.append({"prompt": i, "layer": layer, **metrics, "top_5pct_score_share": selected})
            generations.append({"prompt": i, "input_tokens": ids["input_ids"].shape[-1],
                                "new_tokens": generated.shape[-1]-ids["input_ids"].shape[-1]})
            print(f"Scored prompt {i+1}/{args.n}: {len(scores)} layers", flush=True)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                                 stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None
    source_dir = Path(__file__).parent
    source_sha256 = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("allocator.py", "score_diagnostics.py", "sanity.py")}
    metadata = {"model": args.model_id or args.model, "revision": args.revision,
        "mode": args.mode, "diagnostic_only": True, "prompt_count": args.n,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts[:args.n], ensure_ascii=False).encode()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": args.device, "dtype": str(next(model.parameters()).dtype),
        "seed": 0, "threads": args.threads, "chat_template": args.chat_template,
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
        "git_sha": git_sha, "git_dirty": git_dirty, "source_sha256": source_sha256,
        "seconds": time.perf_counter()-started,
        "timing_scope": "diagnostic loop, excluding model loading; not an inference benchmark",
        "generations": generations, "rows": rows}
    (out / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.n} real-model diagnostic score dumps to {out}")


if __name__ == "__main__":
    main()
