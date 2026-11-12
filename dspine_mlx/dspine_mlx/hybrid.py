"""Hybrid targets: exact rollback for GatedDeltaNet (linear-attention) layers.

mlx-lm keeps a GatedDeltaNet layer's state in an ``ArraysCache``
(``[conv_state, ssm_state]``) that cannot be trimmed like a KV cache. After a
verify pass over T tokens of which only ``n`` are accepted, the state must be
the state after exactly ``n`` tokens.

During a verify pass each linear layer records its pre-verify state and the
recurrence inputs (conv window, q, k, v, a, b). ``rollback`` then rebuilds the
state for the accepted prefix:

* conv state = rows ``n .. n+K-2`` of the recorded conv input (a slice);
* ssm state  = ``gated_delta_update`` replayed over the first ``n`` recorded
  steps from the pre-verify state, the same recurrence the verify pass ran.

No projections or attention are recomputed; full-attention layers keep using
``KVCache.trim``. This is what lets Qwen3.8-27B (``qwen3_5``: 48 GatedDeltaNet
+ 16 full-attention layers) be a DSpine target.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

_BASE_GDN = qwen3_5.GatedDeltaNet


class _Record:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _RecordBox:
    def __init__(self):
        self.recording = False
        self.record: Optional[_Record] = None


class RecordingGatedDeltaNet(_BASE_GDN):
    """``mlx_lm.models.qwen3_5.GatedDeltaNet.__call__`` plus recording.

    The compute path is a line-for-line copy of the mlx-lm 0.32 method; the
    only additions are the ``_record`` writes. ``install_recording`` checks on
    the real layer that the copy is bit-identical to the installed mlx-lm.
    """

    # State lives in a plain holder: mlx Modules treat dict/array attributes
    # as children, which must not happen for recorded activations.
    _box: "_RecordBox"

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            raise NotImplementedError("sharded targets are not supported")

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

        if self._box.recording and cache is not None:
            if mask is not None or cache.lengths is not None:
                raise NotImplementedError("rollback supports unpadded batch-1 decode")
            self._box.record = _Record(
                conv_input=conv_input, ssm0=state, q=q, k=k, v=v, a=a, b=b, steps=S
            )

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        return out

    def rollback(self, cache: ArraysCache, accepted: int) -> None:
        """Set ``cache`` to the state after ``accepted`` of the recorded steps."""
        rec = self._box.record
        if rec is None:
            raise RuntimeError("no recorded verify pass to roll back")
        steps = int(rec.steps)
        if not 1 <= accepted <= steps:
            raise ValueError(f"accepted={accepted} outside [1,{steps}]")
        if accepted < steps:
            n_keep = self.conv_kernel_size - 1
            cache[0] = mx.contiguous(
                rec.conv_input[:, accepted : accepted + n_keep, :]
            )
            sl = slice(0, accepted)
            _, state = gated_delta_update(
                rec.q[:, sl],
                rec.k[:, sl],
                rec.v[:, sl],
                rec.a[:, sl],
                rec.b[:, sl],
                self.A_log,
                self.dt_bias,
                rec.ssm0,
                None,
                use_kernel=not self.training,
            )
            cache[1] = state
        self._box.record = None


def _self_check(mod) -> None:
    """Fail loudly if the copied forward drifted from the installed mlx-lm."""
    dtype = mod.conv1d.weight.dtype
    if dtype not in (mx.float16, mx.bfloat16, mx.float32):
        dtype = mx.bfloat16
    x = (mx.random.normal((1, 3, mod.hidden_size), key=mx.random.key(0)) * 0.5).astype(dtype)
    c1, c2 = ArraysCache(size=2), ArraysCache(size=2)
    y1 = _BASE_GDN.__call__(mod, x, None, c1)
    y2 = RecordingGatedDeltaNet.__call__(mod, x, None, c2)
    same = all(
        bool(mx.array_equal(a, b).item()) for a, b in ((y1, y2), (c1[0], c2[0]), (c1[1], c2[1]))
    )
    if not same:
        raise RuntimeError(
            "mlx-lm's qwen3_5.GatedDeltaNet changed; dspine_mlx.hybrid needs updating "
            "for this mlx-lm version"
        )


def install_recording(layers) -> list:
    """Swap every qwen3_5 GatedDeltaNet in ``layers`` for the recording subclass.

    Returns one entry per layer (= per cache slot): the linear module, or None
    for a full-attention layer. Returns ``[]`` when there is no linear layer.
    """
    linear = []
    for layer in layers:
        layer = getattr(layer, "_layer", layer)  # unwrap a hidden-state hook
        mod = getattr(layer, "linear_attn", None)
        if mod is None:
            linear.append(None)
            continue
        if type(mod) not in (_BASE_GDN, RecordingGatedDeltaNet):
            raise TypeError(
                f"unsupported linear-attention module {type(mod).__name__}; "
                "only mlx_lm.models.qwen3_5.GatedDeltaNet is supported"
            )
        mod.__class__ = RecordingGatedDeltaNet
        if not isinstance(getattr(mod, "_box", None), _RecordBox):
            object.__setattr__(mod, "_box", _RecordBox())
        linear.append(mod)
    mods = [m for m in linear if m is not None]
    if not mods:
        return []
    _self_check(mods[0])
    return linear
