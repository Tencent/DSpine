"""Runtime dispatch between MSL kernels and the pure-MLX reference path."""

from __future__ import annotations

import logging

import mlx.core as mx

from . import reference, serial
from .config import KernelConfig, from_env

logger = logging.getLogger(__name__)

_config: KernelConfig = from_env()
_msl_ok = True
_fallback_reason: str | None = None


def get_config() -> KernelConfig:
    return _config


def configure(cfg: KernelConfig) -> None:
    """Override the configuration (tests / programmatic setup)."""
    global _config
    _config = cfg


def reset_fallback() -> None:
    global _msl_ok, _fallback_reason
    _msl_ok = True
    _fallback_reason = None


def msl_available() -> bool:
    return _msl_ok


def _disable(name: str, exc: BaseException) -> None:
    global _msl_ok, _fallback_reason
    if _msl_ok:
        logger.warning(
            "DSpine MSL %s disabled after failure (%r); falling back to MLX.",
            name,
            exc,
        )
    _msl_ok = False
    _fallback_reason = f"{name}: {exc!r}"


def chain_enabled() -> bool:
    return _msl_ok and _config.chain


def filter_fastpath_enabled() -> bool:
    return _config.filter_fastpath


def tail_batch_enabled() -> bool:
    return _config.tail_batch


def draft_batch_enabled() -> bool:
    return _config.draft_batch


def serial_sample_chain(
    table: mx.array,
    cand: mx.array,
    *,
    inv_temp,
    top_k,
    top_p,
    min_p,
    uniforms: mx.array,
    block_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    if chain_enabled():
        try:
            return serial.serial_sample_chain(
                table,
                cand,
                inv_temp=inv_temp,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                uniforms=uniforms,
                block_size=block_size,
            )
        except Exception as exc:  # pragma: no cover - hardware/driver failure
            _disable("serial_sample_chain", exc)
    if isinstance(inv_temp, mx.array) or isinstance(top_k, mx.array):
        # The reference implementation is per-request; disallow batched calls.
        if block_size is not None and int(block_size) > 2:
            raise ValueError(
                "batched serial_sample_chain requires the MSL kernel"
            )
    return reference.serial_sample_chain(
        table,
        cand,
        inv_temp=_scalar(inv_temp),
        top_k=int(_scalar(top_k)),
        top_p=_scalar(top_p),
        min_p=_scalar(min_p),
        uniforms=uniforms,
    )


def greedy_chain(
    idx: mx.array,
    cand: mx.array,
    *,
    block_size: int | None = None,
) -> mx.array:
    if chain_enabled():
        try:
            return serial.greedy_chain(idx, cand, block_size=block_size)
        except Exception as exc:  # pragma: no cover - hardware/driver failure
            _disable("greedy_chain", exc)
    if block_size is not None and int(block_size) > 2:
        raise ValueError("batched greedy_chain requires the MSL kernel")
    return reference.chain(idx, cand)


def bridge_call(*args, **kwargs):
    from .bridge import bridge_call as _msl_bridge

    if _msl_ok and _config.bridge:
        try:
            return _msl_bridge(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - hardware/driver failure
            _disable("bridge_call", exc)
    raise RuntimeError("MSL bridge is disabled")


def warmup(*, block_size: int, k: int) -> None:
    """Compile the enabled kernels once, at worker startup."""
    if not chain_enabled():
        return
    try:
        serial.warmup(k=k, block_size=block_size)
    except Exception as exc:  # pragma: no cover - hardware/driver failure
        _disable("warmup", exc)


def fallback_reason() -> str | None:
    return _fallback_reason


def _scalar(value):
    if isinstance(value, mx.array):
        return float(value.reshape(-1)[0])
    return value
