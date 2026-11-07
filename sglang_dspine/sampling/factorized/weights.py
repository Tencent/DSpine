"""Cached factorized proposal probabilities and exact cancellation fallback."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_base(
    X, Head, Norm, Cand, Out, K: tl.constexpr, H: tl.constexpr, BH: tl.constexpr
):
    row = tl.program_id(0)
    n = row // K
    h = tl.arange(0, BH)
    cid = tl.load(Cand + row)
    x = tl.load(X + n * H + h, h < H, 0).to(tl.float64)
    head = tl.load(Head + cid * H + h, h < H, 0).to(tl.float64)
    norm = tl.load(Norm + h, h < H, 0).to(tl.float64)
    tl.store(Out + row, tl.sum(x * head * norm, axis=0))


def build_cache(
    codebook, pred, prev, bias, W, norm, head, res_emb=None, res_proj=None, chunk=4096
):
    if (res_emb is None) != (res_proj is None):
        raise ValueError("residual pair")
    (v, c) = codebook.shape
    r = pred.shape[0]
    h = W.shape[0]
    assert (
        pred.shape == prev.shape == (r, c)
        and W.shape == (h, r)
        and (head.shape == (v, h))
        and (norm.shape == (h,))
    )
    (wd, pd, vd, nd, bd) = [z.double() for z in (W, pred, prev, norm, bias)]
    cache = dict(W=wd, norm=nd, bias=bd, Gram=wd.T @ wd, head=head)
    for key in ("G", "V", "TG"):
        cache[key] = torch.empty((v, r), device=W.device, dtype=torch.float64)
    rp = None if res_proj is None else res_proj.double()
    for begin in range(0, v, chunk):
        end = min(v, begin + chunk)
        e = math.sqrt(c) * codebook[begin:end].double()
        cache["G"][begin:end] = e @ pd.T
        value = e @ vd.T
        if rp is not None:
            value = value + res_emb[begin:end].double() @ rp.T
        cache["V"][begin:end] = value
        cache["TG"][begin:end] = head[begin:end].double() * nd @ wd
    return cache
