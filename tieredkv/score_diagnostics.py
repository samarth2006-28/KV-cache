"""Dump per-layer scores for local-model prompts; not a task-accuracy benchmark.

python -m tieredkv.score_diagnostics --model LOCAL_DIR --prompts prompts.json
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .allocator import ScoreTracker


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
    args = p.parse_args()
    if min(args.n, args.max_new_tokens, args.max_input_tokens) < 1:
        p.error("prompt count and token limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable in this Torch build")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) < args.n or any(not isinstance(x, str) or not x.strip() for x in prompts[:args.n]):
        p.error("prompts must contain at least n nonempty strings")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode(), ScoreTracker(model, mode=args.mode, check=True) as tracker:
        for i, prompt in enumerate(prompts[:args.n]):
            tracker.reset()
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=args.max_input_tokens).to(args.device)
            # Tracker's pre-hook requests weights locally; generate does not retain
            # all attention matrices. One unpadded prompt per call.
            model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
            scores = {layer: tracker.scores(layer).detach().cpu() for layer in tracker.acc}
            torch.save({"scores": scores, "mode": args.mode,
                        "counts": {layer: value.cpu() for layer, value in tracker.cnt.items()}},
                       out / f"prompt_{i:02d}.pt")
            for layer, values in scores.items():
                k = values.shape[-1]
                total = float(values.sum())
                top = float(values.topk(math.ceil(.05*k), dim=-1).values.sum())
                rows.append({"prompt": i, "layer": layer, "tokens": k,
                             "top_5pct_score_share": top/total if total else 0.0})
    (out / "summary.json").write_text(json.dumps({"model": args.model, "mode": args.mode,
        "diagnostic_only": True, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved {args.n} real-model diagnostic score dumps to {out}")


if __name__ == "__main__":
    main()
