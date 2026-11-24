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
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(
        model.language_model.model, "layers"
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
        self.prompt_cache = False
        self._ctx_rows = None
        self._drop_session()

    @classmethod
    def load(cls, path: str | Path, target_layer_ids) -> "TargetAdapter":
        model, tokenizer = mlx_lm.load(str(path))
        return cls(model, tokenizer, target_layer_ids)

    def reset(self) -> None:
        if self.prompt_cache and self.cache is not None and self._seq is not None:
            return  # the next prefill decides what to reuse
        self._storage = [None] * len(self.target_layer_ids)
        for hook in (self.layers[lid] for lid in self.target_layer_ids):
            hook._storage = self._storage
        self.cache = make_prompt_cache(self.model)
        self._drop_session()

    # ---- prompt cache across turns ------------------------------------------
    #
    # With the prompt cache on, ``reset()`` keeps the target cache and
    # ``prefill(prompt)`` reuses the longest part of it that ``prompt`` starts
    # with:
    #   * everything the cache holds (the previous prompt and the answer
    #     committed after it), when the new prompt continues it, as in a chat
    #     or agent loop that sends the whole history each turn;
    #   * otherwise the previous prompt, from a snapshot taken at the end of
    #     its prefill (a GatedDeltaNet state cannot be rewound, so the restore
    #     point is saved); this covers a history whose last answer was edited
    #     or had its thinking stripped;
    #   * otherwise a mark: a restore point inside an earlier prompt. One is
    #     saved where a prompt stopped matching the one before it (the end of
    #     a shared system prompt), and one where the caller says a stable
    #     prefix ends (``mark_hint``), so a new conversation with the same long
    #     system prompt reads only what follows it;
    #   * otherwise nothing: a full prefill.
    # The tokens a reused cache gives are the ones a full prefill gives: the
    # cache holds exactly the states a full prefill of the same tokens leaves.
    # The draft gets the hidden rows of the reused part too (``ctx_rows`` of
    # them: the draft's sliding window, or all of them without a window), so a
    # fresh draft cache sees the context it sees after a full prefill.

    mark_min = 256  # shortest shared prefix worth a mark, in tokens
    mark_max = 3  # marks kept (each holds the GatedDeltaNet states and draft rows)
    mark_hint = None  # set before prefill: a prefix length to mark in this prompt

    def enable_prompt_cache(self, on: bool = True, ctx_rows: int | None = None) -> None:
        """Keep the cache between ``generate`` calls. ``ctx_rows``: hidden rows
        kept for the draft (its sliding window; None keeps all of them)."""
        self.prompt_cache = bool(on)
        self._ctx_rows = None if ctx_rows is None else int(ctx_rows)
        self._drop_session()

    def _drop_session(self) -> None:
        self._seq = None       # token ids the cache holds, committed
        self._hid = []         # hidden chunks of the committed positions
        self._pending = None   # (token ids, hidden) of a verify not yet trimmed
        self._snap = None      # (n_tokens, layer states, hidden rows, logits) after the last prefill
        self._marks = []       # (n_tokens, layer states, hidden rows, None) inside earlier prompts
        self._shared = 0       # tokens the last prompt shared with the cache before it
        self._ahead = False    # the cache holds tokens the next prompt may not have
        self.last_reused = 0   # prompt tokens the last prefill took from the cache

    def _hid_rows(self):
        if not self._hid:
            return None
        h = self._hid[0] if len(self._hid) == 1 else mx.concatenate(self._hid, axis=1)
        if self._ctx_rows is not None:
            h = h[:, -self._ctx_rows :]
        self._hid = [h]
        return h

    def _commit(self, ids: list, hidden) -> None:
        self._seq.extend(ids)
        if hidden is not None:
            self._hid.append(hidden)
            if len(self._hid) > 64:
                self._hid_rows()

    def _slots(self):
        """(cache entry, recording GatedDeltaNet or None) per layer."""
        return zip(self.cache, self._linear or [None] * len(self.cache))

    def _layer_state(self) -> list:
        return [int(c.offset) if mod is None else (c[0], c[1]) for c, mod in self._slots()]

    def _restore(self, snap) -> None:
        n, states, hid, _ = snap
        for (c, mod), st in zip(self._slots(), states):
            if mod is None:
                c.trim(int(c.offset) - st)
            else:
                c[0], c[1] = st
                mod._box.record = None
        self._seq = self._seq[:n]
        self._marks = [m for m in self._marks if m[0] <= n]
        self._hid = [hid] if hid is not None else []
        self._pending = None
        self._verify_steps = 0

    def _resume(self, ids: list) -> int:
        """Bring the cache to a prefix of ``ids``; returns its length (0: start over)."""
        self._shared = 0
        if self.cache is None or self._seq is None:
            return 0
        if self._pending is not None:
            # The last verify of a generation is not trimmed: keep the part of
            # it that the new prompt repeats.
            toks = [int(t) for t in self._pending[0].tolist()]
            base, m = len(self._seq), 0
            while m < len(toks) and base + m < len(ids) and ids[base + m] == toks[m]:
                m += 1
            if m == 0:
                self._ahead, self._pending = True, None
            else:
                self.trim(len(toks) - m)
        seq = self._seq
        c, lim = 0, min(len(seq), len(ids))
        while c < lim and seq[c] == ids[c]:
            c += 1
        self._shared = c
        if c == len(seq) and c < len(ids) and not self._ahead:
            return c
        best = None
        for snap in [self._snap] + self._marks:
            # a mark has no logits, so it cannot stand for the whole prompt
            if snap is None or snap[0] > c or (snap[3] is None and snap[0] >= len(ids)):
                continue
            if best is None or snap[0] > best[0]:
                best = snap
        if best is not None:
            self._restore(best)
            self._ahead = False
            return best[0]
        return 0

    def _mark_points(self, start: int, n: int) -> list:
        """Where to save marks while prefilling positions start..n."""
        hint = int(self.mark_hint or 0)
        pts = {m[0] for m in self._marks}
        if self._shared >= self.mark_min:
            pts.add(self._shared)
        if hint >= self.mark_min:
            pts.add(hint)
        top = max(self._shared, hint)
        return sorted(p for p in pts if start < p < n and p <= top)

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

        With the prompt cache on, only the part of ``tokens`` that the cache
        does not hold yet is fed (see ``enable_prompt_cache``).
        """
        session = self.prompt_cache and capture
        start, points = 0, []
        if session:
            ids = [int(t) for t in tokens[0].tolist()]
            start = self._resume(ids)
            self.last_reused = start
            if start == len(ids):  # the same prompt again: the snapshot has its logits
                self.mark_hint = None
                return self._snap[3], self._snap[2]
            points = self._mark_points(start, len(ids))
            self.mark_hint = None
        if start == 0:
            self.cache = make_prompt_cache(self.model)
            self._drop_session()
            if session:
                self._seq = []
        n = int(tokens.shape[1])
        hs = []
        if start:
            prev = self._hid_rows()
            if prev is not None:
                hs.append(prev)
        logits = hidden = None
        pos = start
        for end in points + [n]:
            first = len(hs)
            for s in range(pos, end, PREFILL_CHUNK):
                e = min(s + PREFILL_CHUNK, end)
                logits, hidden = self._forward(tokens[:, s:e], capture=capture)
                if capture:
                    hs.append(hidden)
                if e < n:
                    mx.eval([c.state for c in self.cache] + hs[-1:])
            if session:
                new = hs[first:]
                self._commit(ids[pos:end], new[0] if len(new) == 1 else mx.concatenate(new, axis=1))
                if end < n:  # a mark
                    rows = self._hid_rows()
                    mx.eval(rows, [c.state for c in self.cache])
                    self._marks.append((end, self._layer_state(), rows, None))
            pos = end
        if capture and len(hs) > 1:
            hidden = mx.concatenate(hs, axis=1)
        logits = logits[:, -1:]
        if session:
            self._marks = self._marks[-self.mark_max :]
            rows = self._hid_rows()
            mx.eval(logits, rows, [c.state for c in self.cache])
            self._snap = (len(ids), self._layer_state(), rows, logits)
            self._ahead = False
        return logits, hidden

    def _set_recording(self, on: bool) -> None:
        for mod in self._linear:
            if mod is not None:
                mod._box.recording = on

    def verify(self, tokens: mx.array, capture: bool = True):
        if not self._linear:
            out = self._forward(tokens, capture=capture)
        else:
            self._set_recording(True)
            try:
                out = self._forward(tokens, capture=capture)
            finally:
                self._set_recording(False)
            self._verify_steps = int(tokens.shape[1])
        if self._seq is not None:
            self._pending = (tokens[0], out[1])
        return out

    def step(self, token: int) -> mx.array:
        self._seq = None  # plain decoding bypasses the prompt-cache bookkeeping
        logits, _ = self._forward(
            mx.array([[int(token)]], dtype=mx.int32), capture=False
        )
        return logits

    def trim(self, n: int) -> None:
        """Drop the last ``n`` tokens of the preceding verify pass."""
        if self._pending is not None:
            toks, hidden = self._pending
            keep = int(toks.shape[0]) - max(int(n), 0)
            self._pending = None
            self._commit(
                [int(t) for t in toks[:keep].tolist()],
                None if hidden is None else hidden[:, :keep],
            )
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
