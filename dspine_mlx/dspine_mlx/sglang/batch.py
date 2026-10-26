"""Batched draft-backbone context.

The draft backbone is a 5-layer Qwen3-style transformer with a per-request
context K/V cache.  A decode round drafts the same block size ``T`` for every
request, so the five layers can run on ``[B, T, H]`` if each layer's context is
padded to a common prefix length.  This module owns the per-round metadata
(prefix lengths, padding, attention mask) and is consumed by
``DraftAttention._forward_block_batched``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class DraftBatchContext:
    prefix_lens: list[int]
    caches: list[list[Any]]          # [B][layer] mlx-lm KVCache
    num_tokens: int                  # T
    offsets: mx.array = field(init=False)   # [B] int32
    max_prefix: int = field(init=False)
    needs_padding: bool = field(init=False)
    mask: mx.array | None = field(init=False)  # [B,1,T,max_prefix+T] bool

    def __post_init__(self) -> None:
        if not self.prefix_lens:
            raise ValueError("DraftBatchContext needs at least one request")
        self.offsets = mx.array(self.prefix_lens, dtype=mx.int32)
        self.max_prefix = max(self.prefix_lens)
        self.needs_padding = min(self.prefix_lens) < self.max_prefix
        self.mask = _build_mask(
            self.offsets, self.max_prefix, int(self.num_tokens)
        )

    @classmethod
    def from_caches(cls, caches: list[list[Any]], block_size: int):
        """Build from ``[B][layer]`` per-request draft caches."""
        if not caches:
            raise ValueError("no draft caches")
        layers = len(caches[0])
        for req, req_caches in enumerate(caches):
            if len(req_caches) != layers:
                raise ValueError(
                    f"request {req} has {len(req_caches)} draft caches, "
                    f"expected {layers}"
                )
        # The context cache is appended in lockstep across layers; any layer
        # reports the absolute offset.
        prefix_lens = [int(caches[b][0].offset) for b in range(len(caches))]
        return cls(
            prefix_lens=prefix_lens,
            caches=caches,
            num_tokens=int(block_size),
        )

    def layer_caches(self, layer_index: int) -> list[Any]:
        return [self.caches[b][layer_index] for b in range(len(self.caches))]


def _build_mask(
    offsets: mx.array, max_prefix: int, num_tokens: int
) -> mx.array | None:
    """Boolean keep-mask ``[B, 1, T, max_prefix+T]`` for one draft forward."""
    block = int(num_tokens)
    if max_prefix == 0:
        causal = mx.arange(block)[None, :, None] >= mx.arange(block)[
            None, None, :
        ]
        return mx.broadcast_to(
            causal[None], (offsets.shape[0], 1, block, block)
        )
    idx = mx.arange(max_prefix)[None, :]
    prefix_valid = idx < offsets[:, None]  # [B, max_prefix]
    prefix = mx.broadcast_to(
        prefix_valid[:, None, None, :],
        (offsets.shape[0], 1, block, max_prefix),
    )
    causal = mx.arange(block)[None, :, None] >= mx.arange(block)[None, None, :]
    block_mask = mx.broadcast_to(
        causal[None], (offsets.shape[0], 1, block, block)
    )
    return mx.concatenate([prefix, block_mask], axis=-1)
