"""Request-seeded random draws keyed by absolute block position and draw purpose."""

import secrets
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.sampling.murmur_hash import (
    fmix32,
    murmur3_mix,
)

DRAFT, ACCEPT, BONUS, PREFILL = range(4)


def _cpu_uniforms(seeds, positions, width, stream):
    mask = 0xFFFFFFFF
    h = torch.zeros((seeds.numel(), width), dtype=torch.int64)
    blocks = (
        seeds[:, None] & mask,
        (seeds[:, None] >> 32) & mask,
        positions.long()[:, None] & mask,
        torch.arange(width)[None, :] + (stream << 16),
    )
    for block in blocks:
        k = (block * 0xCC9E2D51) & mask
        k = ((k << 15) | (k >> 17)) & mask
        h ^= (k * 0x1B873593) & mask
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    h ^= h >> 16
    return (h >> 8).float() * (2.0**-24)


@triton.jit
def _uniforms(
    Seeds,
    Positions,
    Out,
    WIDTH: tl.constexpr,
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK).to(tl.uint32) + (STREAM << 16)
    seed = tl.load(Seeds + row).to(tl.uint64)
    position = tl.load(Positions + row).to(tl.uint32)
    h = tl.full((BLOCK,), 0, tl.uint32)
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, (seed >> 32).to(tl.uint32))
    h = murmur3_mix(h, position)
    h = murmur3_mix(h, col)
    bits = fmix32(h ^ 16)
    value = (bits >> 8).to(tl.float32) * (2.0**-24)
    tl.store(
        Out + row * WIDTH + tl.arange(0, BLOCK), value, tl.arange(0, BLOCK) < WIDTH
    )


def seeded_uniforms(seeds, positions, width, stream):
    if seeds.ndim != 1 or positions.shape != seeds.shape:
        raise ValueError("Seeds and positions must be matching request vectors")
    if seeds.dtype != torch.int64 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected int64 seeds and integer positions")
    if (
        seeds.device != positions.device
        or not 0 < width < 65536
        or stream not in range(4)
    ):
        raise ValueError("Invalid seeded uniform layout")
    seeds, positions = seeds.contiguous(), positions.contiguous()
    if seeds.device.type == "cpu":
        return _cpu_uniforms(seeds, positions, width, stream)
    output = torch.empty(
        (seeds.numel(), width), device=seeds.device, dtype=torch.float32
    )
    _uniforms[(seeds.numel(),)](
        seeds, positions, output, width, stream, triton.next_power_of_2(width)
    )
    return output


class RequestRNG:
    def __init__(self):
        self.seeds = self.positions = None

    def stage(self, seeds, positions):
        self.seeds = seeds
        self.positions = positions.clone() if seeds is not None else None

    def draw(self, width, stream):
        return seeded_uniforms(self.seeds, self.positions, width, stream)


def verification_uniforms(bs, width, device, random_state=None):
    if random_state is None or random_state.seeds is None:
        return (
            torch.rand((bs, width - 1), device=device, dtype=torch.float32),
            torch.rand((bs,), device=device, dtype=torch.float32),
        )
    if random_state.seeds.numel() != bs:
        raise RuntimeError("Seeded verification request order was not staged")
    return random_state.draw(width - 1, ACCEPT), random_state.draw(1, BONUS).view(bs)


@contextmanager
def seeded_batch(batch):
    """Scope request seeds to the worker call, leaving scheduler merge state intact."""
    info = getattr(batch, "sampling_info", None)
    if info is None:
        yield
        return
    original = getattr(info, "sampling_seed", None)
    reqs = getattr(batch, "reqs", ())
    if original is None and any(
        r.sampling_params.sampling_seed is not None for r in reqs
    ):
        seeds = []
        for req in reqs:
            seed = req.sampling_params.sampling_seed
            if seed is None:
                seed = getattr(req, "_dspine_random_seed", None)
                if seed is None:
                    seed = secrets.randbits(63)
                    req._dspine_random_seed = seed
            seeds.append(seed)
        host_seeds = torch.tensor(
            seeds, dtype=torch.int64, pin_memory=info.temperatures.is_cuda
        )
        info.sampling_seed = host_seeds.to(info.temperatures.device, non_blocking=True)
    try:
        yield
    finally:
        info.sampling_seed = original


def sample_prefill(probs, info, positions):
    from sglang.srt.layers.sampler import _SamplingMaskCapture
    from sglang.srt.runtime_context import get_exec

    from sglang_dspine.runtime.worker import filter_target_probs

    bs, vocab = probs.shape
    weights = filter_target_probs(
        probs, info, 1, bs, get_exec().kernel.sampling_backend, normalize=False
    ).view(bs, vocab)
    cdf = weights.cumsum(-1, dtype=torch.float64)
    coin = seeded_uniforms(info.sampling_seed, positions.reshape(bs), 1, PREFILL)
    tokens = (
        (cdf > coin.double() * cdf[:, -1:]).to(torch.int32).argmax(-1).to(torch.int32)
    )
    capture = None
    indices = getattr(info, "sampling_mask_batch_indices", None)
    if indices is not None:
        selected = weights[indices]
        capture = _SamplingMaskCapture(
            weights=selected,
            token_ids=None,
            selected_weight=selected.gather(1, tokens[indices].long()[:, None]).view(
                -1
            ),
            batch_rows=indices,
        )
    return tokens, capture
