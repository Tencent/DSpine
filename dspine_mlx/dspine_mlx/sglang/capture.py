"""Target hidden-state capture for DSpine drafting.

The target model is an unmodified mlx-lm Qwen3.  DSpine needs the outputs of
five transformer layers at every position.  We wrap those layer modules with a
lightweight ``nn.Module`` that records the (lazy) output into a thread-local
capture session.  Nothing is materialised until the caller evals the graph.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn

_session = threading.local()


def get_active_session() -> "CaptureSession | None":
    return getattr(_session, "active", None)


class CaptureSession:
    """Collects one record (one model forward) per expected call.

    A record is a ``dict[layer_id, mx.array]`` keyed by the *target layer id*
    from the draft config.  Records are split when a layer id repeats, which is
    how the per-request extend loop is separated from the single batched
    verify forward.
    """

    def __init__(self, expected_layer_ids):
        self.expected_layer_ids = tuple(int(x) for x in expected_layer_ids)
        self._expected = 0
        self._records: list[dict[int, mx.array]] = []
        self._current: dict[int, mx.array] = {}
        self._active = False

    def begin(self, expected_calls: int) -> None:
        if self._active:
            raise RuntimeError("capture session already active")
        self._expected = int(expected_calls)
        self._records = []
        self._current = {}
        self._active = True
        previous = getattr(_session, "active", None)
        if previous is not None and previous is not self:
            raise RuntimeError("another capture session is already active")
        _session.active = self

    def record(self, layer_id: int, out: mx.array) -> None:
        if not self._active:
            return
        if layer_id in self._current:
            self._records.append(self._current)
            self._current = {}
        self._current[layer_id] = out

    def end(self) -> list[dict[int, mx.array]]:
        if not self._active:
            raise RuntimeError("capture session is not active")
        if self._current:
            self._records.append(self._current)
        self._current = {}
        self._active = False
        _session.active = None
        for i, record in enumerate(self._records):
            if set(record) != set(self.expected_layer_ids):
                raise RuntimeError(
                    f"capture record {i} has layers {sorted(record)}, "
                    f"expected {sorted(self.expected_layer_ids)}"
                )
        if len(self._records) != self._expected:
            raise RuntimeError(
                f"captured {len(self._records)} records, expected {self._expected}"
            )
        return self._records

    def abort_if_active(self) -> None:
        if self._active:
            self._active = False
            self._records = []
            self._current = {}
            _session.active = None


class LayerCapture(nn.Module):
    """Transparent wrapper that reports a transformer block's output."""

    def __init__(self, inner: nn.Module, layer_id: int):
        super().__init__()
        self._inner = inner
        self._layer_id = int(layer_id)

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        session = get_active_session()
        if session is not None:
            session.record(self._layer_id, out)
        return out


def install_capture_layers(
    model,
    target_layer_ids,
) -> tuple[list[int], list[LayerCapture]]:
    """Wrap ``model.model.layers[lid]`` for every configured target layer.

    Returns ``(installed_ids, wrappers)``.  Idempotent: already-wrapped layers
    are returned unchanged.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")

    installed: list[int] = []
    wrappers: list[LayerCapture] = []
    for lid in (int(x) for x in target_layer_ids):
        if not 0 <= lid < len(layers):
            raise ValueError(f"target layer id {lid} outside [0,{len(layers)})")
        layer = layers[lid]
        if isinstance(layer, LayerCapture):
            installed.append(lid)
            wrappers.append(layer)
            continue
        capture = LayerCapture(layer, lid)
        layers[lid] = capture
        installed.append(lid)
        wrappers.append(capture)
    return installed, wrappers
