from __future__ import annotations

import math
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from .config import DraftConfig

# torch.finfo(torch.float32).tiny, used as the clamp floor of the bridge read RMS.
_F32_TINY = 1.1754943508222875e-38


class DSpineCodeBridge(nn.Module):
    """The DSpine code bridge, one write after every draft layer.

    Parameter names and shapes match the HF export exactly under
    ``selected_prefix_bridge.`` so the checkpoint loads without a name map.
    The forward mirrors ``sglang_dspine/models/dspine.py:_forward_eager``
    rounding point by rounding point.
    """

    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.code_size = int(cfg.bridge.code_size)
        self.rank = int(cfg.bridge.rank)
        self.num_layers = int(cfg.num_hidden_layers)
        self.block_size = int(cfg.block_size)
        self.eps = float(cfg.bridge.eps)
        self.active_layers = frozenset(cfg.feature_layers)
        self.terminal_residual_rank = 0

        self.prev = nn.Linear(self.code_size, self.rank, bias=False)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True)
        self.predecessor_gate = nn.Linear(self.code_size, self.rank, bias=False)
        self.out = nn.Linear(self.rank, self.code_size, bias=False)

        self.read_weight = mx.zeros(
            (self.code_size, self.hidden_size), dtype=mx.bfloat16
        )
        self.early_read_weights = mx.zeros(
            (self.num_layers - 1, self.code_size, self.hidden_size),
            dtype=mx.bfloat16,
        )
        self.feature_write_weights = mx.zeros(
            (self.num_layers, self.hidden_size, self.rank), dtype=mx.bfloat16
        )
        self.codebook = mx.zeros(
            (int(cfg.vocab_size), self.code_size), dtype=mx.bfloat16
        )
        self.independent_write_version = mx.array(0, dtype=mx.int64)
        self.terminal_disabled_version = mx.array(0, dtype=mx.int64)
        # Derived [3R, C] = [G_q; G_p; A]; underscore keeps it out of parameters().
        self._packed_weight: mx.array | None = None

    def layer_read_weight(self, layer: int) -> mx.array:
        if layer == self.num_layers - 1:
            return self.read_weight
        return self.early_read_weights[layer]

    def prepare(self) -> None:
        if int(self.independent_write_version.item()) != 1:
            raise ValueError("Unsupported independent_write_version")
        if int(self.terminal_disabled_version.item()) != 1:
            raise ValueError("Unsupported terminal_disabled_version")
        if bool(mx.any(self.out.weight != 0).item()):
            raise ValueError(
                "Terminal-disabled checkpoint contains a nonzero terminal projection"
            )
        self._packed_weight = mx.contiguous(
            mx.concatenate(
                [self.gate.weight, self.predecessor_gate.weight, self.prev.weight],
                axis=0,
            )
        )
        mx.eval(self._packed_weight)

    @property
    def packed_weight(self) -> mx.array:
        if self._packed_weight is None:
            raise RuntimeError("DSpine code-bridge weights were not prepared")
        return self._packed_weight

    def set_block_size(self, block_size: int) -> None:
        self.block_size = int(block_size)

    def __call__(
        self,
        hidden: mx.array,
        residual: mx.array | None,
        input_ids: mx.array,
        layer: int,
        next_norm: nn.RMSNorm,
        stash: dict | None = None,
    ) -> tuple[mx.array, mx.array]:
        """One write after ``layer``.

        ``hidden`` is the MLP output, ``residual`` the pre-MLP stream (or None
        when the caller already summed the layer). Returns
        ``(next_norm(out), out)``; ``stash`` receives the pre-write state.
        """
        B, T, H = hidden.shape
        if hidden.dtype != mx.bfloat16:
            raise ValueError("The DSpine code bridge requires BF16 activations")
        m = B * T
        sqrt_c = math.sqrt(self.code_size)

        raw = hidden if residual is None else (hidden + residual)
        raw = raw.astype(mx.bfloat16)
        f = raw.astype(mx.float32)
        rms = mx.sqrt(mx.maximum(mx.mean(f * f, axis=-1), _F32_TINY))  # [B,T]
        normalized = (f / (rms[..., None] + self.eps)).astype(mx.bfloat16)

        projected = mx.matmul(
            normalized, self.layer_read_weight(layer).T
        ).astype(mx.float32)
        proj_norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
        codes = (
            sqrt_c * projected / (proj_norm + self.eps)
        ).astype(mx.bfloat16)

        anchors = input_ids[:, 0]
        anchor_codes = (
            sqrt_c * self.codebook[anchors].astype(mx.float32)
        ).astype(mx.bfloat16)
        codes = mx.concatenate([anchor_codes[:, None, :], codes[:, 1:, :]], axis=1)

        packed = mx.matmul(codes, self.packed_weight.T)  # [B,T,3R] bf16
        if stash is not None:
            stash.update(
                raw=raw.reshape(m, H),
                rms=rms.reshape(m),
                packed=packed.reshape(m, 3 * self.rank),
            )

        q, s, v = mx.split(
            packed.astype(mx.float32),
            [self.rank, 2 * self.rank],
            axis=-1,
        )
        gate = mx.sigmoid(q[:, 1:, :] + s[:, :-1, :] + self.gate.bias.astype(mx.float32))
        message = mx.concatenate(
            [mx.zeros_like(q[:, :1, :]), gate * v[:, :-1, :]], axis=1
        ).astype(mx.bfloat16)
        delta = mx.matmul(
            message, self.feature_write_weights[layer].T
        ).astype(mx.float32)

        not_anchor = (
            (mx.arange(m).reshape(B, T) % T != 0).astype(mx.float32)[..., None]
        )
        out = (f + (rms[..., None] * delta) * not_anchor).astype(mx.bfloat16)

        x = out.astype(mx.float32)
        y = (
            x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + next_norm.eps)
        ).astype(mx.bfloat16)
        normed = (
            y.astype(mx.float32) * next_norm.weight.astype(mx.float32)
        ).astype(mx.bfloat16)
        return normed, out


def build_serial_tables(
    *,
    head_rows: Callable[[int, int], mx.array],
    vocab_size: int,
    norm_w: mx.array,
    write_w: mx.array,
    pv_w: mx.array,
    codebook: mx.array,
    chunk: int = 8192,
) -> tuple[mx.array, mx.array]:
    """Vocabulary-wide greedy refinement tables.

    ``head_rows(start, end)`` returns dense BF16 target rows [end-start, H].
    Returns ``TG [V,R]`` and ``TSV [V,2R]`` in BF16, exactly as
    ``sglang_dspine/kernels.py:build_serial_tables``.
    """
    V, H = int(vocab_size), int(norm_w.shape[0])
    R = int(write_w.shape[1])
    C = int(codebook.shape[1])
    if write_w.shape != (H, R) or pv_w.shape != (2 * R, C) or codebook.shape != (V, C):
        raise ValueError("build_serial_tables: shape contract violated")
    nwf = norm_w.astype(mx.float32)
    wb = write_w.astype(mx.bfloat16)
    pvb = pv_w.astype(mx.bfloat16)
    sqrt_c = math.sqrt(C)
    tg_parts: list[mx.array] = []
    tsv_parts: list[mx.array] = []
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        rows = head_rows(start, end)
        if rows.shape != (end - start, H):
            raise ValueError(
                f"head_rows({start},{end}) returned {rows.shape}, expected "
                f"{(end - start, H)}"
            )
        tg_parts.append(
            mx.matmul(
                (rows.astype(mx.float32) * nwf).astype(mx.bfloat16), wb
            )
        )
        tsv_parts.append(
            mx.matmul(
                (sqrt_c * codebook[start:end].astype(mx.float32)).astype(
                    mx.bfloat16
                ),
                pvb.T,
            )
        )
    tg = mx.concatenate(tg_parts, axis=0) if len(tg_parts) > 1 else tg_parts[0]
    tsv = (
        mx.concatenate(tsv_parts, axis=0) if len(tsv_parts) > 1 else tsv_parts[0]
    )
    mx.eval(tg, tsv)
    return tg, tsv
