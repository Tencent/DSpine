"""Capture the empty-history ReplaySSM kernel and guard live cursors before replay."""

import importlib
import inspect
import types
from contextvars import ContextVar

import torch

from sglang_dspine.runtime.replayssm import guard as replay_guard
from sglang_dspine.runtime.replayssm import kernels as replay_kernel
from sglang_dspine.runtime.replayssm.launch import (
    LAUNCH_KEYS,
    native_defaults,
    select_launch,
)

_active = ContextVar("dspine_replay_forward", default=False)
_capture = ContextVar("dspine_replay_capture", default=False)
_helper = None
_public = None
_registered = False


def _pool(runner):
    from sglang_dspine.spec import is_dspine

    if runner.is_draft_worker or not is_dspine(runner.spec_algorithm):
        return None
    pool = getattr(runner.req_to_token_pool, "mamba_pool", None)
    if (
        pool is None
        or not pool.enable_linear_replayssm_spec
        or getattr(pool, "replayssm_is_kda", False)
        or getattr(pool, "replayssm_spec_fold", False)
    ):
        return None
    return pool


def around_capture(original, backend, key, forward_fn, *args, **kwargs):
    runner = backend._cuda_graph_runner.model_runner
    if _pool(runner) is None:
        return original(backend, key, forward_fn, *args, **kwargs)
    token = _capture.set(True)
    try:
        return original(backend, key, forward_fn, *args, **kwargs)
    finally:
        _capture.reset(token)


def around_forward(original, runner, batch, *args, **kwargs):
    pool = _pool(runner)
    if (
        pool is None
        or not batch.forward_mode.is_target_verify()
        or torch.cuda.is_current_stream_capturing()
    ):
        return original(runner, batch, *args, **kwargs)
    bs = int(batch.batch_size)
    if bs < 1:
        return original(runner, batch, *args, **kwargs)
    req_pool = runner.req_to_token_pool
    state = req_pool.get_speculative_mamba2_params_all_layers()
    if state.temporal.dtype != torch.float32 or state.replayssm_beta is not None:
        return original(runner, batch, *args, **kwargs)
    replay_guard.launch(
        torch,
        batch.req_pool_indices,
        req_pool.req_index_to_mamba_index_mapping,
        pool.replayssm_spec_write_pos,
        pool.replayssm_is_flush,
        pool.replayssm_spec_write_pos,
        bs=bs,
        state_slots=state.temporal.shape[1],
        count=False,
    )
    token = _active.set(True)
    try:
        return original(runner, batch, *args, **kwargs)
    finally:
        _active.reset(token)


def around_kernel(original, *args, **kwargs):
    if args or not (_active.get() or _capture.get()):
        return original(*args, **kwargs)
    defaults = native_defaults(_public)
    values = {**defaults, **kwargs}
    launch = select_launch("DSPINE", values, defaults)
    if launch is None:
        return original(**kwargs)
    values.update(zip(LAUNCH_KEYS, launch))
    values["is_flush_kernel"] = False
    if values.get("scale") is None:
        values["scale"] = values["q"].shape[-1] ** -0.5
    parameters = inspect.signature(_helper).parameters
    _helper(**{name: values[name] for name in parameters})
    return values["out"]


def register():
    global _helper, _public, _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    native = importlib.import_module(
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode"
    )
    _public = native.gdn_replayssm_spec_decode
    original = native._launch_gdn_spec
    scope = dict(
        original.__globals__,
        gdn_replayssm_spec_circular_kernel=replay_kernel.empty_history_normal_kernel,
    )
    _helper = types.FunctionType(
        original.__code__, scope, "dspine_replay_launch", original.__defaults__
    )
    for target, hook in (
        (
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode.gdn_replayssm_spec_decode",
            around_kernel,
        ),
        ("sglang.srt.model_executor.model_runner.ModelRunner.forward", around_forward),
        (
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_one",
            around_capture,
        ),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
    _registered = True
