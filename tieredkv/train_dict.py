"""Train Lexico-format KV dictionaries on a time budget (Lexico ships none).

Same objective as Lexico's dictionary_learning/train.py: OMP-encode KV vectors with the current
dictionary (no grad), minimise ||D y - kv||^2 over D, keep atoms unit-norm. Differences, to fit a
hackathon: atoms start as sampled real KV vectors (usable at step 0), all 2L dictionaries train
in one batched step, one fixed KV buffer is collected once and the model is freed, and training
stops after --minutes.

Output: tensor [2*n_layers, head_dim, N] fp16, D[2l] = layer l keys, D[2l+1] = values
(post-RoPE keys, as stored in the cache), loadable by Lexico and compress.load_dictionary.

    python -m tieredkv.train_dict --model meta-llama/Llama-3.1-8B-Instruct \
        --out dicts/llama31_8b_N4096_s16.pt --minutes 45
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .compress import omp


@torch.no_grad()
def collect_kv(model, batches, max_tokens):
    """Run the model, return KV vectors [2L, M, head_dim] fp16 on CPU (M = tokens * kv heads)."""
    per_layer, seen = None, 0
    for ids in batches:
        past = model(ids.to(model.device), use_cache=True).past_key_values
        kv = []
        for l in range(len(past)):
            k, v = past[l]
            kv += [k.transpose(1, 2).reshape(-1, k.shape[-1]), v.transpose(1, 2).reshape(-1, v.shape[-1])]
        per_layer = [[] for _ in kv] if per_layer is None else per_layer
        for buf, x in zip(per_layer, kv):
            buf.append(x.half().cpu())
        seen += ids.numel()
        if seen >= max_tokens:
            break
    return torch.stack([torch.cat(b) for b in per_layer])


def init_dict(buf, N, seed=0):
    """Atoms = N random real KV vectors per dictionary, unit-normalised. buf [L2, M, d]."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(buf.shape[1], generator=g)[:N]
    D = buf[:, pick].float().transpose(1, 2)                     # [L2, d, N]
    return D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def rel_error(D, X, s):
    """Mean relative L2 reconstruction error per dictionary. D [L2, d, N], X [L2, B, d]."""
    idx, coef = omp(D, X, s)
    li = torch.arange(D.shape[0], device=D.device)[:, None, None]
    rec = (D.transpose(1, 2)[li, idx] * coef[..., None]).sum(-2)
    return ((rec - X.float()).norm(dim=-1) / X.float().norm(dim=-1).clamp(min=1e-8)).mean(1)


def train(buf, N=4096, s=16, minutes=30.0, max_steps=None, batch=512, lr=1e-3,
          n_eval=2048, device="cpu", seed=0, log_every=50):
    """buf [L2, M, d] -> (D [L2, d, N] fp16 on CPU, log dict)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(buf.shape[1], generator=g)
    ev, tr = buf[:, perm[:n_eval]].to(device), buf[:, perm[n_eval:]].to(device)
    D = torch.nn.Parameter(init_dict(tr.cpu(), N, seed).to(device))
    opt = torch.optim.Adam([D], lr=lr)
    li = torch.arange(D.shape[0], device=device)[:, None, None]
    log = {"init_rel_err": rel_error(D.detach(), ev, s).tolist(), "loss": []}
    t0, step = time.time(), 0
    while (time.time() - t0) < minutes * 60 and (max_steps is None or step < max_steps):
        x = tr[:, torch.randint(tr.shape[1], (batch,), device=device)].float()
        idx, coef = omp(D.detach(), x, s)
        rec = (D.transpose(1, 2)[li, idx] * coef[..., None]).sum(-2)
        loss = F.mse_loss(rec, x)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():                                    # gradient tangent to unit sphere
            Dn = D / D.norm(dim=1, keepdim=True)
            D.grad -= (D.grad * Dn).sum(1, keepdim=True) * Dn
        opt.step()
        with torch.no_grad():
            D /= D.norm(dim=1, keepdim=True)
        if step % log_every == 0:
            log["loss"].append((step, loss.item()))
            print(f"step {step:6d}  {time.time() - t0:7.0f}s  loss {loss.item():.4f}", flush=True)
        step += 1
    log["final_rel_err"] = rel_error(D.detach(), ev, s).tolist()
    log["steps"] = step
    return D.detach().half().cpu(), log


def _token_batches(tokenizer, texts, seq_len, bs):
    """Pack texts into [bs, seq_len] blocks (no padding)."""
    buf = []
    for t in texts:
        if t.strip():
            buf.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= seq_len * bs:
            yield torch.tensor(buf[:seq_len * bs]).view(bs, seq_len)
            buf = buf[seq_len * bs:]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=4096, help="atoms per dictionary")
    p.add_argument("--s", type=int, default=16, help="sparsity used while training")
    p.add_argument("--tokens", type=int, default=16384, help="tokens to collect KV from")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dataset", default="wikitext", choices=["wikitext", "gsm8k_train"])
    a = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda").eval()
    if a.dataset == "wikitext":
        texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    else:  # never the test split: that's what we evaluate on
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [q + "\n" + ans for q, ans in zip(ds["question"], ds["answer"])]
    buf = collect_kv(model, _token_batches(tok, texts, a.seq_len, 4), a.tokens)
    del model
    torch.cuda.empty_cache()
    print(f"collected KV buffer {tuple(buf.shape)}")

    D, log = train(buf, a.N, a.s, a.minutes, batch=a.batch, lr=a.lr, device="cuda")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(D, a.out)
    log.update(vars(a))
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(log, f)
    L = len(log["final_rel_err"])
    for name, sl in (("keys", slice(0, L, 2)), ("values", slice(1, L, 2))):
        i, e = log["init_rel_err"][sl], log["final_rel_err"][sl]
        print(f"{name}: rel err at s={a.s}  init {sum(i) / len(i):.3f} -> final {sum(e) / len(e):.3f}")
    print(f"saved {tuple(D.shape)} to {a.out}")


if __name__ == "__main__":
    main()
