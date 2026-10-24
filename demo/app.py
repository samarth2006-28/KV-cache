"""Live demo: one GSM8K-style question, full KV cache vs TieredKV (and Lexico) side by side.

Shows each method's answer, KV memory (MB and % of full), time, and the question's tokens
coloured by how many dictionary atoms TieredKV gave them (averaged over layers).

    pip install gradio
    python demo/app.py --dictionary dicts/llama32_3b_N4096_s16.pt            # opens a local web page
    python demo/app.py --dictionary dicts/llama32_3b_N4096_s16.pt --cli "Tom has 3 apples..."

Needs the GPU the experiments use (Llama-3.2-3B in fp16, eager attention).
"""
import argparse
import html
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tieredkv import gsm8k, profile  # noqa: E402
from tieredkv.allocator import ScoreTracker  # noqa: E402
from tieredkv.cache import TieredKVCache  # noqa: E402

METHODS = ("full", "tiered", "lexico")
LABEL = {"full": "Full KV cache", "tiered": "TieredKV (ours)", "lexico": "Lexico (uniform)"}


@torch.no_grad()
def answer(model, tok, prompt, method, budget, dictionary, max_new_tokens=192):
    """Generate with one cache. Returns a dict with the answer, memory, time and the cache."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    tracker = ScoreTracker(model, mode="mean") if method == "tiered" else None
    try:
        kw = {} if method == "full" else {"budget": budget, "dictionary_path": dictionary,
                                          "expected_new_tokens": min(64, max_new_tokens)}
        cache = TieredKVCache(method, tracker=tracker, **kw)
        profile.sync()
        t0 = time.perf_counter()
        out = model.generate(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                             past_key_values=cache, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, stop_strings=gsm8k.STOP, tokenizer=tok,
                             pad_token_id=tok.eos_token_id)
        profile.sync()
        secs = time.perf_counter() - t0
    finally:
        if tracker is not None:
            tracker.remove()
    text = gsm8k.truncate_at_stop(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return {"method": method, "text": text.strip(), "pred": gsm8k.extract(text)[1],
            "kv_mb": cache.kv_bytes() / 2**20, "budget": cache.measured_budget(),
            "seconds": secs, "new_tokens": out.shape[1] - ids["input_ids"].shape[1], "cache": cache}


def token_atoms(cache):
    """Mean atoms per compressed token across layers ([n_sparse]); None if nothing compressed."""
    per_layer = [cache.token_s(l).float() for l in range(len(cache)) if cache.sparse[l] is not None]
    if not per_layer:
        return None
    n = min(len(s) for s in per_layer)
    return torch.stack([s[:n] for s in per_layer]).mean(0)


def tier_html(tok, prompt, question, atoms):
    """The question's tokens, shaded by TieredKV's atoms (darker = more atoms = judged important)."""
    ids = tok(prompt)["input_ids"]
    q_start = len(tok(prompt[:prompt.rindex(question)])["input_ids"])
    lo, hi = (float(atoms.min()), float(atoms.max())) if atoms is not None and len(atoms) else (0.0, 1.0)
    spans = []
    for i in range(q_start, len(ids)):
        piece = html.escape(tok.decode([ids[i]]))
        if atoms is not None and i < len(atoms):
            a = float(atoms[i])
            w = 0.08 + 0.85 * (a - lo) / max(hi - lo, 1e-6)
            spans.append(f'<span title="{a:.1f} atoms" style="background:rgba(42,120,214,{w:.2f});'
                         f'border-radius:3px">{piece}</span>')
        else:   # still in the dense recent window: kept exactly
            spans.append(f'<span title="recent window (full precision)" '
                         f'style="outline:1px dashed #888;border-radius:3px">{piece}</span>')
    legend = (f"<p style='font-size:12px;color:#666'>Darker = more atoms (range {lo:.1f}-{hi:.1f}). "
              "Dashed = recent tokens kept at full precision.</p>")
    return "<div style='line-height:1.9'>" + "".join(spans) + "</div>" + legend


def compare(model, tok, question, budget, dictionary, methods=METHODS, shots=None, max_new_tokens=192):
    shots = gsm8k.fewshot(5) if shots is None else shots
    prompt = gsm8k.build_prompt(question.strip(), shots)
    rows = [answer(model, tok, prompt, m, budget, dictionary, max_new_tokens) for m in methods]
    tiered = next((r for r in rows if r["method"] == "tiered"), None)
    shade = tier_html(tok, prompt, question.strip(), token_atoms(tiered["cache"]) if tiered else None)
    return rows, shade


def summary_md(r):
    return (f"### {LABEL[r['method']]}\n**Answer: {r['pred']}**\n\n"
            f"KV memory: **{r['kv_mb']:.1f} MB** ({100 * r['budget']:.1f}% of full)  \n"
            f"Time: {r['seconds']:.1f} s for {r['new_tokens']} tokens\n\n```\n{r['text'][:900]}\n```")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--dictionary", required=True)
    p.add_argument("--budget", type=float, default=0.2)
    p.add_argument("--cli", help="answer this question in the terminal instead of opening the web page")
    a = p.parse_args()

    from tieredkv.eval import load_model
    model, tok = load_model(a.model)
    shots = gsm8k.fewshot(5)
    if a.cli:
        rows, _ = compare(model, tok, a.cli, a.budget, a.dictionary, shots=shots)
        for r in rows:
            print(f"{LABEL[r['method']]:18s} answer={r['pred']!s:8s} {r['kv_mb']:6.1f} MB "
                  f"({100 * r['budget']:5.1f}%)  {r['seconds']:5.1f}s")
        return

    import gradio as gr
    example = gsm8k.load_subset(1, seed=7)[0]["question"]

    def run(question, budget):
        rows, shade = compare(model, tok, question, budget, a.dictionary, shots=shots)
        return [summary_md(r) for r in rows] + [shade]

    with gr.Blocks(title="TieredKV demo") as ui:
        gr.Markdown("# TieredKV: attention-guided KV cache compression\n"
                    "Same question, same model. Full cache vs compressed caches at the same memory budget.")
        with gr.Row():
            q = gr.Textbox(value=example, label="Question", lines=4, scale=4)
            b = gr.Slider(0.10, 0.30, value=a.budget, step=0.05, label="KV memory budget", scale=1)
        go = gr.Button("Run", variant="primary")
        with gr.Row():
            outs = [gr.Markdown() for _ in METHODS]
        gr.Markdown("### Which tokens TieredKV judged important")
        shade = gr.HTML()
        go.click(run, [q, b], outs + [shade])
    ui.launch()


if __name__ == "__main__":
    main()
