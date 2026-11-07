"""DSpine serving on SGLang 0.5.22."""

import logging

import torch

TARGET = "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.process_weights_after_loading"
_count = 0


def after_load(result, method, layer, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    from sglang_dspine.spec import is_dspine

    if not is_dspine(get_spec().speculative_algorithm or ""):
        return result
    weight = getattr(layer, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or tuple(weight.shape) not in ((34816, 5120), (5120, 17408))
    ):
        return result
    if weight.dtype != torch.bfloat16 or not weight.is_cuda or weight.requires_grad:
        raise ValueError("DSPine MLP layout requires inference-only CUDA BF16 weights")
    if not weight.is_contiguous():
        raise ValueError("MLP weight already has an unrecognized storage layout")
    packed = weight.T.contiguous().T
    if not torch.equal(packed, weight):
        raise RuntimeError("MLP storage transformation changed checkpoint values")
    weight.data = packed
    global _count
    _count += 1
    if _count <= 2 or _count % 32 == 0:
        logging.getLogger(__name__).info(
            "DSPine BF16 MLP layout: transformed=%d shape=%s stride=%s values=exact",
            _count,
            tuple(weight.shape),
            weight.stride(),
        )
    return result


def register_bf16_layout_hook():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, after_load, HookType.AFTER)
