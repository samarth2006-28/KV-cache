"""TieredKV compression: sparse coding of K/V vectors over a Lexico dictionary, per-token s.

Owner: Samarth. Contract (see plan section 7):
    encode_kv(kv, dictionary, s_per_token) -> SparseKV     kv [n, h, d]
    decode_kv(sparse, dictionary)          -> Tensor [n, h, d]
    bytes_per_token(s)        bytes for one token, one layer, K+V, all kv heads, at s atoms
    full_bytes_per_token()    same for a dense fp16 token
    budget(s_list, W, N)      (sum bytes(s_i) + W*full) / (N*full)  <- the ONE memory-% definition
    solve_s(budget, N, W)     largest uniform s that fits (Lexico baseline; also tiered's avg_s)

Storage matches Lexico's cache exactly: CSR rows (one row per token x kv-head), int32 indptr,
int16 atom indices, float8_e4m3fn coefficients. CSR stores variable s natively, so memory is
NOT padded to s_max: SparseKV.nbytes is what actually sits on the GPU.

Dictionary layout (Lexico's dictionary_learning output): D [2*n_layers, head_dim, N], unit-norm
columns; D[2*l] is layer l's KEY dictionary, D[2*l + 1] its VALUE dictionary. See load_dictionary.
"""
from dataclasses import dataclass
import math

import torch

__all__ = ["SparseKV", "omp", "encode_kv", "decode_kv", "load_dictionary", "configure",
           "bytes_per_token", "full_bytes_per_token", "budget", "solve_s"]

# --------------------------------------------------------------------------- memory math

IDX_BYTES = 2      # int16 atom index (dictionary size <= 32768)
COEF_BYTES = 1     # float8_e4m3fn coefficient
INDPTR_BYTES = 4   # int32 CSR row pointer, one per (token, kv head)
DENSE_BYTES = 2    # fp16 / bf16 dense KV element
N_KV_HEADS = 8     # Llama-3.1-8B / Llama-3.2-3B; call configure() for other models
HEAD_DIM = 128


def configure(config=None, *, n_kv_heads=None, head_dim=None):
    """Set model shape for the memory math, from an HF config or explicit values."""
    global N_KV_HEADS, HEAD_DIM
    if config is not None:
        N_KV_HEADS = config.num_key_value_heads
        HEAD_DIM = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    N_KV_HEADS = n_kv_heads or N_KV_HEADS
    HEAD_DIM = head_dim or HEAD_DIM


def bytes_per_token(s):
    """Bytes for one token in one layer (K and V, all kv heads) at s atoms; s = 0 (evicted) -> 0.

    Accepts an int or a tensor of s values. Affine in s because of the CSR row pointer;
    allocate() matches bytes, not atoms, so this is handled.
    """
    per = 2 * N_KV_HEADS * (s * (IDX_BYTES + COEF_BYTES) + INDPTR_BYTES)
    if torch.is_tensor(s):
        return torch.where(s > 0, per, torch.zeros_like(per))
    return per if s > 0 else 0


def full_bytes_per_token():
    return 2 * N_KV_HEADS * HEAD_DIM * DENSE_BYTES


def budget(s_list, W, N):
    """Memory as a fraction of full KV: (sum bytes(s_i) + W*full) / (N*full).

    s_list: s for every non-recent token (0 = evicted), [tokens] or [layers, tokens]
    (averaged over layers). W: dense recent tokens. N: total tokens seen (incl. evicted).
    """
    s = torch.as_tensor(s_list, dtype=torch.long)
    n_layers = s.shape[0] if s.dim() == 2 else 1
    sparse = bytes_per_token(s).sum().item() / n_layers
    return (sparse + W * full_bytes_per_token()) / (N * full_bytes_per_token())


def solve_s(budget, N, W):
    """Largest integer s such that compressing all N - W old tokens at s fits the budget."""
    full = full_bytes_per_token()
    if N <= W:
        raise ValueError(f"N={N} <= W={W}: nothing to compress")
    per_token = (budget * N * full - W * full) / (N - W)
    s = math.floor((per_token / (2 * N_KV_HEADS) - INDPTR_BYTES) / (IDX_BYTES + COEF_BYTES) + 1e-9)
    if s < 1:
        raise ValueError(
            f"budget {budget:.1%} infeasible at N={N}, W={W}: the dense window alone is "
            f"{W / N:.1%} and s=1 needs {budget_at(1, N, W):.1%}")
    return s


def budget_at(s, N, W):
    return budget([s] * (N - W), W, N)


# --------------------------------------------------------------------------- storage

@dataclass
class SparseKV:
    """CSR-compressed K or V for one layer. Row r = token r // n_heads, kv head r % n_heads."""
    indptr: torch.Tensor   # [n_tokens * n_heads + 1] int32
    indices: torch.Tensor  # [nnz] int16
    data: torch.Tensor     # [nnz] float8_e4m3fn (or fp16)
    n_heads: int

    @property
    def n_tokens(self):
        return (self.indptr.numel() - 1) // self.n_heads

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.indptr, self.indices, self.data))

    def s_per_token(self):
        return self.indptr.diff().view(self.n_tokens, self.n_heads)[:, 0].long()

    @staticmethod
    def empty(n_heads, device=None, coef_dtype=torch.float8_e4m3fn):
        return SparseKV(torch.zeros(1, dtype=torch.int32, device=device),
                        torch.zeros(0, dtype=torch.int16, device=device),
                        torch.zeros(0, dtype=coef_dtype, device=device), n_heads)

    @staticmethod
    def cat(parts):
        """Concatenate along tokens (e.g. append a newly compressed chunk)."""
        parts = list(parts)
        offs, ptrs = 0, [parts[0].indptr[:1]]
        for p in parts:
            ptrs.append(p.indptr[1:] + offs)
            offs += int(p.indptr[-1])
        return SparseKV(torch.cat(ptrs), torch.cat([p.indices for p in parts]),
                        torch.cat([p.data for p in parts]), parts[0].n_heads)

    def select(self, token_idx):
        """Keep only tokens token_idx (in that order), e.g. after evicting compressed tokens."""
        token_idx = torch.as_tensor(token_idx, device=self.indptr.device).long()
        h = self.n_heads
        rows = (token_idx[:, None] * h + torch.arange(h, device=token_idx.device)).flatten()
        start, lens = self.indptr[:-1].long()[rows], self.indptr.diff().long()[rows]
        pos = _ragged_arange(lens) + start.repeat_interleave(lens)
        indptr = torch.zeros(rows.numel() + 1, dtype=torch.int32, device=rows.device)
        indptr[1:] = lens.cumsum(0)
        return SparseKV(indptr, self.indices[pos], self.data[pos], h)


def _ragged_arange(lens):
    """[0..lens[0]), [0..lens[1]), ... concatenated."""
    total = int(lens.sum())
    starts = torch.repeat_interleave(lens.cumsum(0) - lens, lens)
    return torch.arange(total, device=lens.device) - starts


def load_dictionary(path, layer, device=None, dtype=torch.float16):
    """(key_dict, value_dict) for one layer from a Lexico dictionary file, each [head_dim, N]."""
    D = torch.load(path, map_location="cpu")
    return D[2 * layer].to(device, dtype), D[2 * layer + 1].to(device, dtype)


# --------------------------------------------------------------------------- OMP

@torch.no_grad()
def omp(D, Y, s, eps=1e-6):
    """Batched Orthogonal Matching Pursuit.

    D [d, N] dictionary (unit-norm columns), Y [B, d] signals, s atoms each.
    Returns (idx [B, s] long, coef [B, s] fp32). Also batches over L dictionaries:
    D [L, d, N], Y [L, B, d] -> [L, B, s]. Greedy: the first k atoms chosen with s atoms
    are exactly the atoms chosen with k atoms.
    """
    single = D.dim() == 2
    if single:
        D, Y = D[None], Y[None]
    D, Y = D.float(), Y.float()
    L, B, N = Y.shape[0], Y.shape[1], D.shape[-1]
    DT = D.transpose(1, 2).contiguous()                          # [L, N, d]
    li = torch.arange(L, device=Y.device)[:, None, None]
    idx = torch.zeros(L, B, s, dtype=torch.long, device=Y.device)
    chosen = torch.zeros(L, B, N, dtype=torch.bool, device=Y.device)
    r, coef = Y, Y.new_zeros(L, B, 0)
    for k in range(s):
        j = (r @ D).abs().masked_fill_(chosen, -1.0).argmax(-1)  # [L, B]
        idx[..., k] = j
        chosen.scatter_(-1, j[..., None], True)
        A = DT[li, idx[..., :k + 1]]                             # [L, B, k+1, d]
        G = A @ A.transpose(-1, -2)
        G.diagonal(dim1=-2, dim2=-1).add_(eps)
        coef = torch.linalg.solve(G, A @ Y.unsqueeze(-1))        # [L, B, k+1, 1]
        r = Y - (A.transpose(-1, -2) @ coef).squeeze(-1)
        coef = coef.squeeze(-1)
    return (idx[0], coef[0]) if single else (idx, coef)


@torch.no_grad()
def encode_kv(kv, dictionary, s_per_token, coef_dtype=torch.float8_e4m3fn, chunk=20000):
    """Compress kv [n, h, d] with s_per_token[i] atoms for token i (same s for all heads).

    s_per_token: int or [n] LongTensor (e.g. allocate() output). Tokens with s = 0 get empty
    rows and decode to zero vectors: the cache should drop evicted tokens, not attend to zeros.
    Tokens are grouped by s and each group runs one batched OMP.
    """
    n, h, d = kv.shape
    if dictionary.shape[0] != d:
        raise ValueError(f"dictionary is {tuple(dictionary.shape)}, expected [{d}, N]")
    if dictionary.shape[1] > 32768:
        raise ValueError("int16 indices need dictionary size <= 32768")
    dev = kv.device
    s_tok = torch.as_tensor(s_per_token, device=dev).long().flatten()
    if s_tok.numel() == 1:
        s_tok = s_tok.expand(n)
    if s_tok.numel() != n:
        raise ValueError(f"s_per_token has {s_tok.numel()} entries for {n} tokens")
    s_row = s_tok.repeat_interleave(h)
    Y = kv.reshape(n * h, d)
    D = dictionary.to(dev, torch.float32)

    s_max = int(s_row.max()) if n else 0
    idx_pad = torch.zeros(n * h, s_max, dtype=torch.long, device=dev)
    coef_pad = torch.zeros(n * h, s_max, dtype=torch.float32, device=dev)
    for s in s_row.unique().tolist():
        if s <= 0:
            continue
        for rows in (s_row == s).nonzero().flatten().split(chunk):
            i, c = omp(D, Y[rows], s)
            idx_pad[rows, :s], coef_pad[rows, :s] = i, c

    keep = torch.arange(s_max, device=dev) < s_row[:, None]
    data = coef_pad[keep]
    if coef_dtype == torch.float8_e4m3fn:
        data = data.clamp(-448.0, 448.0)   # e4m3fn overflows to NaN, not inf
    indptr = torch.zeros(n * h + 1, dtype=torch.int32, device=dev)
    indptr[1:] = s_row.cumsum(0)
    return SparseKV(indptr, idx_pad[keep].to(torch.int16), data.to(coef_dtype), h)


@torch.no_grad()
def decode_kv(sparse, dictionary, dtype=torch.float16, chunk=20000):
    """Reconstruct [n, h, d]. Deterministic (no atomics), so equal inputs give equal outputs."""
    D = dictionary.float()
    DT = D.T.contiguous()
    rows = sparse.indptr.numel() - 1
    out = torch.zeros(rows, D.shape[0], dtype=torch.float32, device=D.device)
    lens = sparse.indptr.diff().long().to(D.device)
    if rows and int(lens.max()) > 0:
        s_max = int(lens.max())
        row_of = torch.repeat_interleave(torch.arange(rows, device=D.device), lens)
        pos = _ragged_arange(lens)
        idx_pad = torch.zeros(rows, s_max, dtype=torch.long, device=D.device)
        coef_pad = torch.zeros(rows, s_max, dtype=torch.float32, device=D.device)
        idx_pad[row_of, pos] = sparse.indices.to(D.device).long()
        coef_pad[row_of, pos] = sparse.data.to(D.device).float()
        for r in torch.arange(rows, device=D.device).split(chunk):
            out[r] = (coef_pad[r].unsqueeze(1) @ DT[idx_pad[r]]).squeeze(1)
    return out.view(-1, sparse.n_heads, D.shape[0]).to(dtype)
