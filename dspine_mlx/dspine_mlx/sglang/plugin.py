"""SGLang plugin entry point for DSpine on the MLX backend.

Loaded through the ``sglang.srt.plugins`` entry point.  The plugin is inert
unless ``SGLANG_USE_MLX=1``; the CUDA plugin in ``sglang_dspine/`` must not be installed
in the same environment.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_registered = False
_active_worker = None

_ATTENTION_TARGET = (
    "sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper."
    "MLXAttentionWrapper.__call__"
)
_VALIDATE_TARGET = "sglang.srt.speculative.dflash_utils.validate_dflash_request"
_RELEASE_TARGET = "sglang.srt.mem_cache.common.release_kv_cache"


def set_active_worker(worker) -> None:
    global _active_worker
    _active_worker = worker


def get_active_worker():
    return _active_worker


def _around_attention(original, wrapper, x, mask=None, cache=None):
    from .verify import around_attention_call

    return around_attention_call(original, wrapper, x, mask=mask, cache=cache)


def _around_validate(original, req, enable_overlap, *args, **kwargs):
    error = original(req, enable_overlap, *args, **kwargs)
    if error is not None:
        return error
    worker = _active_worker
    if worker is None:
        return None
    from .sampling import validate_request

    return validate_request(req, head_topk=worker.draft.head_topk)


def _after_release(result, req, *args, **kwargs):
    worker = _active_worker
    if worker is not None:
        worker.on_request_released(getattr(req, "rid", None))
    return result


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    from sglang.srt.utils.tensor_bridge import use_mlx

    if not use_mlx():
        if os.environ.get("SGLANG_USE_MLX"):
            logger.warning(
                "dspine-mlx: SGLANG_USE_MLX is set but MLX is not "
                "importable; plugin left inert."
            )
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import get_spec

    from .spec import (
        ALGORITHM,
        DSpineMlxSpecAlgo,
        worker_factory,
    )

    existing = get_spec(ALGORITHM)
    if existing is not None and not isinstance(existing, DSpineMlxSpecAlgo):
        raise RuntimeError(
            "DSPINE is already registered by another plugin. Install either the "
            "sglang-dspine (CUDA) plugin or the dspine-mlx (MLX) plugin in this "
            "environment, not both."
        )

    HookRegistry.register(
        _ATTENTION_TARGET, _around_attention, HookType.AROUND
    )
    HookRegistry.register(_VALIDATE_TARGET, _around_validate, HookType.AROUND)
    HookRegistry.register(_RELEASE_TARGET, _after_release, HookType.AFTER)

    if existing is None:
        SpeculativeAlgorithm.register(
            ALGORITHM,
            supports_overlap=False,
            spec_class=DSpineMlxSpecAlgo,
        )(worker_factory)
        logger.info("dspine-mlx: registered DSPINE on the MLX backend")
    else:
        logger.info("dspine-mlx: DSPINE already registered; hooks ready")
