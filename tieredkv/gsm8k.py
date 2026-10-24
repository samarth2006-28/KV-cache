"""GSM8K: fixed subset, few-shot prompt, answer extraction, scoring, paired stats.

Mirrors lm-eval-harness's `gsm8k` task (what Lexico reports): "Question: ...\\nAnswer:" format,
5-shot from the TRAIN split, strict "#### N" match plus flexible last-number match.
One difference: the same exemplars for every problem (fixed seed), so prompts are identical
across methods and every comparison is paired.

    probs = load_subset(200, seed=0)
    prompt = build_prompt(probs[0]["question"], fewshot(5))
    text = truncate_at_stop(generated_text)
    ok = is_correct(text, probs[0]["target"])["strict"]
"""
import math
import random
import re

STOP = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]
_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def target_of(answer):
    return answer.split("####")[-1].strip()


def load_subset(n, seed=0, split="test"):
    """Fixed random subset of n problems: [{"idx", "question", "answer", "target"}]."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    return [{"idx": i, "question": ds[i]["question"], "answer": ds[i]["answer"],
             "target": target_of(ds[i]["answer"])} for i in sorted(order[:n])]


def fewshot(k=5, seed=1234):
    """k exemplars from the train split (never test)."""
    return load_subset(k, seed=seed, split="train") if k else []


def build_prompt(question, shots):
    ctx = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    return f"{ctx}Question: {question}\nAnswer:"


def truncate_at_stop(text, stop=STOP):
    cut = min((i for i in (text.find(s) for s in stop) if i >= 0), default=len(text))
    return text[:cut]


def _norm(x):
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def extract(text):
    """(strict, flexible) answers as strings, None if absent."""
    m = _STRICT.search(text)
    strict = m.group(1) if m else None
    flex = [a or b for a, b in _FLEX.findall(text)]
    flex = [f for f in flex if _norm(f) is not None]
    return strict, (flex[-1] if flex else None)


def is_correct(text, target):
    t = _norm(target)
    strict, flex = extract(truncate_at_stop(text))
    return {"strict": strict is not None and _norm(strict) == t,
            "flexible": flex is not None and _norm(flex) == t,
            "pred": strict if strict is not None else flex}


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def mcnemar(a, b):
    """Exact McNemar test on paired correctness lists (same problems, same order).

    Returns {"a_only", "b_only", "p"}: p is two-sided; < 0.05 means the methods really differ.
    """
    if len(a) != len(b):
        raise ValueError("paired lists must have equal length (same problems)")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = a_only + b_only, min(a_only, b_only)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": a_only, "b_only": b_only, "p": p}
