"""Cached factorized proposal probabilities and exact cancellation fallback."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from . import weights as F

build_cache = F.build_cache


@triton.jit
def _unsafe_y(
    X,
    A,
    Message,
    W,
    Safe,
    Y,
    H: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
):
    nj = tl.program_id(0)
    if not tl.load(Safe + nj):
        n = nj // K
        rs = tl.arange(0, BR)
        message = tl.load(Message + nj * R + rs, rs < R, 0).to(tl.float64)
        a = tl.load(A + n).to(tl.float64)
        for start in range(0, H, BH):
            hs = start + tl.arange(0, BH)
            weights = tl.load(
                W + hs[:, None] * R + rs[None, :],
                (hs[:, None] < H) & (rs[None, :] < R),
                0,
            ).to(tl.float64)
            delta = tl.sum(weights * message[None, :], axis=1)
            x = tl.load(X + n * H + hs, hs < H, 0).to(tl.float64)
            y = x + a * delta
            tl.store(Y + nj * H + hs, y, hs < H)


@triton.jit
def _unsafe_logits(
    Y,
    Norm,
    Head,
    Cand,
    Safe,
    Out,
    H: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BH: tl.constexpr,
):
    nj = tl.program_id(0)
    if not tl.load(Safe + nj):
        n = nj // K
        hs = tl.arange(0, BH)
        y = tl.load(Y + nj * H + hs, hs < H, 0).to(tl.float64)
        mean_square = tl.sum(y * y, axis=0) / H
        denom = libdevice.sqrt_rn(mean_square + EPS)
        norm = tl.load(Norm + hs, hs < H, 0).to(tl.float64)
        z = y / denom * norm
        for c in range(0, K):
            cid = tl.load(Cand + n * K + c)
            head = tl.load(Head + cid * H + hs, hs < H, 0).to(tl.float64)
            value = tl.sum(z * head, axis=0)
            tl.store(Out + nj * K + c, value.to(tl.float32))


@dataclass
class Workspace:
    y: torch.Tensor

    @classmethod
    def allocate(cls, n, k, h, device):
        return cls(torch.empty((n, k, h), device=device, dtype=torch.float64))

    def validate(self, n, k, h, device):
        assert self.y.shape == (n, k, h) and self.y.dtype == torch.float64
        assert self.y.device == torch.device(device) and self.y.is_contiguous()


def apply_direct_fallback(x, a, message, cand, cache, safe, out, workspace, eps):
    (n, k) = cand.shape
    h = x.shape[-1]
    r = message.shape[-1]
    workspace.validate(n, k, h, x.device)
    assert all(
        (
            z.is_contiguous()
            for z in (x, a, message, cand, cache["W"], cache["norm"], cache["head"])
        )
    )
    assert safe.shape == (n, k) and safe.dtype == torch.bool and safe.is_contiguous()
    assert out.shape == (n, k, k) and out.dtype == torch.float32 and out.is_contiguous()
    if x.device.type == "cuda":
        _unsafe_y[n * k,](
            x,
            a,
            message,
            cache["W"],
            safe,
            workspace.y,
            H=h,
            R=r,
            K=k,
            BH=16,
            BR=triton.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )
        _unsafe_logits[n * k,](
            workspace.y,
            cache["norm"],
            cache["head"],
            cand,
            safe,
            out,
            H=h,
            K=k,
            EPS=float(eps),
            BH=triton.next_power_of_2(h),
            num_warps=8,
            enable_fp_fusion=False,
        )
    else:
        for ni, ji in (~safe).nonzero().tolist():
            y = x[ni] + a[ni] * (cache["W"] @ message[ni, ji])
            workspace.y[ni, ji].copy_(y)
            z = y / torch.sqrt(y.square().mean() + eps) * cache["norm"]
            out[ni, ji].copy_((cache["head"][cand[ni]].double() @ z).float())
    return out
