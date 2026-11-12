from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from .hybrid import install_recording

# Prompt tokens per prefill forward (mlx-lm generate uses 2048 too).
PREFILL_CHUNK = 2048


class HeadAccess:
    """Uniform dense / quantized access to the target LM head.

    ``rows`` and ``rows_range`` return dense BF16 rows, used by the DSpine
    refinement tail; ``matmul`` runs the pass-1 logits with the head's native
    quantized kernel.
    """

    def __init__(self, lm_head: nn.Module):
        self.module = lm_head
        self.quantized = hasattr(lm_head, "scales")
        self.weight = lm_head.weight
        if self.quantized:
            self.scales = lm_head.scales
            self.biases = getattr(lm_head, "biases", None)
            self.group_size = int(lm_head.group_size)
            self.bits = int(lm_head.bits)
            mode = getattr(lm_head, "mode", "affine")
            self.mode = mode if isinstance(mode, str) else str(mode)
            self.vocab_size = int(self.weight.shape[0])
            self.hidden_size = int(self.scales.shape[1]) * self.group_size
        else:
            self.vocab_size, self.hidden_size = (int(x) for x in self.weight.shape)
            if self.weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
                raise ValueError(
                    f"unsupported target lm_head dtype {self.weight.dtype}"
                )

    def rows(self, ids: mx.array) -> mx.array:
        if not self.quantized:
            return self.weight[ids]
        biases = None if self.biases is None else self.biases[ids]
        return mx.dequantize(
            self.weight[ids],
            self.scales[ids],
            biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=mx.bfloat16,
        )

    def rows_range(self, start: int, end: int) -> mx.array:
        if not self.quantized:
            return self.weight[start:end]
        biases = None if self.biases is None else self.biases[start:end]
        return mx.dequantize(
            self.weight[start:end],
            self.scales[start:end],
            biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=mx.bfloat16,
        )

    def matmul(self, x: mx.array) -> mx.array:
        if not self.quantized:
            return mx.matmul(x, self.weight.T)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


class _LayerHook:
    def __init__(self, layer, idx: int, storage: list):
        self._layer = layer
        self._idx = idx
        self._storage = storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _find_embed(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "embed_tokens")
    ):
        return model.language_model.model.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    raise AttributeError(f"Cannot find embed_tokens in {type(model).__name__}")


def _find_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    raise AttributeError(f"Cannot find lm_head in {type(model).__name__}")


class TargetAdapter:
    """An mlx-lm target plus hidden-state hooks.

    Pure-attention targets (Qwen3-8B) run unmodified. Hybrid ``qwen3_5``
    targets (Qwen3.8-27B) get recording GatedDeltaNet layers so that ``trim``
    can roll their recurrent state back exactly (see ``hybrid.py``).
    """

    def __init__(self, model, tokenizer, target_layer_ids):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = _get_layers(model)
        self.target_layer_ids = tuple(int(x) for x in target_layer_ids)
        for lid in self.target_layer_ids:
            if not 0 <= lid < len(self.layers):
                raise ValueError(
                    f"target_layer_id {lid} outside [0,{len(self.layers)})"
                )
        self._storage: list = [None] * len(self.target_layer_ids)
        for i, lid in enumerate(self.target_layer_ids):
            self.layers[lid] = _LayerHook(self.layers[lid], i, self._storage)
        self.embed_tokens = _find_embed(model)
        self.lm_head = _find_lm_head(model)
        self.head = HeadAccess(self.lm_head)
        self.cache = None
        # One entry per cache slot: a recording GatedDeltaNet or None ([] if pure attention).
        self._linear = install_recording(self.layers)
        self._verify_steps = 0

    @classmethod
    def load(cls, path: str | Path, target_layer_ids) -> TargetAdapter:
        model, tokenizer = mlx_lm.load(str(path))
        return cls(model, tokenizer, target_layer_ids)

    def reset(self) -> None:
        self._storage = [None] * len(self.target_layer_ids)
        for hook in (self.layers[lid] for lid in self.target_layer_ids):
            hook._storage = self._storage
        self.cache = make_prompt_cache(self.model)

    def _forward(self, tokens: mx.array, capture: bool = True):
        self._storage = [None] * len(self.target_layer_ids)
        for hook in (self.layers[lid] for lid in self.target_layer_ids):
            hook._storage = self._storage
        logits = self.model(tokens, self.cache)
        hidden = None
        if capture:
            if any(h is None for h in self._storage):
                raise RuntimeError("a target_layer_ids layer did not run")
            hidden = mx.concatenate(self._storage, axis=-1)
        return logits, hidden

    def prefill(self, tokens: mx.array, capture: bool = True):
        """Chunked prefill; returns logits for the last position only.

        A one-shot prefill materialises logits for every prompt position
        (about 16 GB for 32k tokens and a 248k vocabulary). Chunks whose
        logits are never evaluated skip the LM head. Hidden features are kept
        for every position, since the draft needs them.
        """
        self.cache = make_prompt_cache(self.model)
        n = int(tokens.shape[1])
        hs = []
        logits = hidden = None
        for s in range(0, n, PREFILL_CHUNK):
            logits, hidden = self._forward(tokens[:, s : s + PREFILL_CHUNK], capture=capture)
            if capture:
                hs.append(hidden)
            if s + PREFILL_CHUNK < n:
                mx.eval([c.state for c in self.cache] + hs[-1:])
        if capture and len(hs) > 1:
            hidden = mx.concatenate(hs, axis=1)
        return logits[:, -1:], hidden

    def _set_recording(self, on: bool) -> None:
        for mod in self._linear:
            if mod is not None:
                mod._box.recording = on

    def verify(self, tokens: mx.array, capture: bool = True):
        if not self._linear:
            return self._forward(tokens, capture=capture)
        self._set_recording(True)
        try:
            out = self._forward(tokens, capture=capture)
        finally:
            self._set_recording(False)
        self._verify_steps = int(tokens.shape[1])
        return out

    def step(self, token: int) -> mx.array:
        logits, _ = self._forward(
            mx.array([[int(token)]], dtype=mx.int32), capture=False
        )
        return logits

    def trim(self, n: int) -> None:
        """Drop the last ``n`` tokens of the preceding verify pass."""
        if self._linear:
            return self._trim_hybrid(n)
        if n <= 0:
            return
        for c in self.cache:
            if not c.is_trimmable():
                raise RuntimeError(
                    f"target cache {type(c).__name__} is not trimmable; "
                    "only KV caches and qwen3_5 GatedDeltaNet layers are supported"
                )
            c.trim(n)

    def _trim_hybrid(self, n: int) -> None:
        if n <= 0:
            for mod in self._linear:
                if mod is not None:
                    mod._box.record = None
            return
        steps = self._verify_steps
        if steps <= 0 or n >= steps:
            raise RuntimeError(f"trim({n}) needs a preceding verify of more than {n} tokens")
        if len(self.cache) != len(self._linear):
            raise RuntimeError("cache / layer count mismatch")
        accepted = steps - n
        for c, mod in zip(self.cache, self._linear):
            if mod is None:
                c.trim(n)
            else:
                mod.rollback(c, accepted)
        self._verify_steps = 0

    @property
    def vocab_size(self) -> int:
        return self.head.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.head.hidden_size
