"""Hand-written Metal kernels and dispatch for the MLX DSpine path."""

from __future__ import annotations

from . import bridge, config, dispatch, reference, serial
from .config import KernelConfig, from_env

__all__ = [
    "KernelConfig",
    "bridge",
    "config",
    "configure",
    "dispatch",
    "from_env",
    "get_config",
    "reference",
    "serial",
    "warmup",
]

configure = dispatch.configure
get_config = dispatch.get_config
warmup = dispatch.warmup
