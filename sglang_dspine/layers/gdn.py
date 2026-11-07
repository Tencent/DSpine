"""DSpine serving on SGLang 0.5.22."""

import triton
import triton.language as tl


@triton.jit
def _to_dense(
    X,
    Q,
    Y,
    D: tl.constexpr,
    T: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = tl.arange(0, T)
    start = tl.load(Q + req)
    end = tl.load(Q + req + 1)
    value = tl.load(
        X + (start + t[:, None]) * S0 + f[None, :] * S1,
        (t[:, None] < end - start) & (f[None, :] < D),
        0,
    )
    tl.store(Y + (req * T + t[:, None]) * D + f[None, :], value, f[None, :] < D)


@triton.jit
def _from_dense(
    X,
    Q,
    Y,
    D: tl.constexpr,
    T: tl.constexpr,
    BS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.full((), 0, tl.int32)
    for row in tl.static_range(BS):
        req = tl.where(token >= tl.load(Q + row + 1), row + 1, req)
    start = tl.load(Q + tl.minimum(req, BS - 1))
    within = token - start
    value = tl.load(
        X + req * S0 + f * S1 + within * S2, (req < BS) & (within < T) & (f < D), 0
    )
    tl.store(Y + token * D + f, value, f < D)


def dense_conv_input(x, indptr, steps):
    if steps != 16:
        raise ValueError("Ragged GDN adapter currently requires draft16")
    bs = indptr.numel() - 1
    dim = x.shape[-1]
    dense = x.new_empty((bs * steps, dim))
    _to_dense[bs, triton.cdiv(dim, 128)](x, indptr, dense, dim, steps, *x.stride(), 128)
    return dense.view(bs, steps, dim).transpose(1, 2)


def packed_conv_output(x, indptr, tokens):
    (bs, dim, steps) = x.shape
    result = x.new_empty((tokens, dim))
    _from_dense[tokens, triton.cdiv(dim, 256)](
        x, indptr, result, dim, steps, bs, *x.stride(), 256
    )
    return result


def after_backend_init(result, backend, model_runner, *args, **kwargs):
    from sglang_dspine.spec import is_dspine

    if is_dspine(model_runner.spec_algorithm):
        backend.supports_ragged_verify_graph = True
    return result


def around_extend(original, backend, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    layout = getattr(
        getattr(forward_batch, "spec_info", None), "ragged_verify_layout", None
    )
    if not forward_batch.forward_mode.is_target_verify() or layout is None:
        return original(backend, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool

    if kwargs.get("linear_attn_output") is not None:
        raise ValueError("Ragged GDN adapter does not accept a caller-owned output")
    fm = backend.forward_metadata
    if fm.retrieve_parent_token is not None or backend.mis_metadata is not None:
        raise ValueError("Ragged GDN supports linear chains without MIS")
    state = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    if not isinstance(state, MambaPool.SpeculativeState):
        raise ValueError("Ragged GDN requires the speculative state pool")
    indptr = fm.query_start_loc
    bs = indptr.numel() - 1
    steps = int(forward_batch.spec_info.draft_token_num)
    slots = fm.mamba_cache_indices
    scratch_indices = select_verify_intermediate_state_indices(
        backend.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        slots[:bs] >= 0,
        backend.req_to_token_pool.size,
    )
    from sglang_dspine.layers.convolution import packed_conv

    packed = packed_conv(
        mixed_qkv,
        indptr,
        state.conv[0],
        layer.conv_weights,
        layer.bias,
        layer.activation,
        slots[:bs],
        scratch_indices[:bs],
        state.intermediate_conv_window[0],
        steps,
    )
    (q, k, v) = packed.split((layer.q_dim, layer.k_dim, layer.v_dim), dim=-1)
    q = q.view(1, -1, layer.num_q_heads, layer.head_q_dim)
    k = k.view(1, -1, layer.num_k_heads, layer.head_k_dim)
    v = v.view(1, -1, layer.num_v_heads, layer.head_v_dim)
    pool = backend.req_to_token_pool.mamba_pool
    if state.replayssm_d is not None:
        if getattr(pool, "replayssm_spec_fold", False) or getattr(
            pool, "replayssm_is_kda", False
        ):
            raise ValueError("Ragged GDN requires the native compact circular ring")
        return backend._replayssm_target_verify(
            layer=layer,
            query=q,
            key=k,
            value=v,
            a=a,
            b=b,
            mamba_pool=pool,
            layer_cache=state,
            cache_indices=slots,
            replay_indices=forward_batch.req_pool_indices,
            query_start_loc=indptr,
            draft_token_num=steps,
        )
    if state.intermediate_ssm is None:
        raise ValueError("Missing native GDN snapshots")
    return backend.kernel_dispatcher.target_verify(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        ssm_states=state.temporal,
        cache_indices=slots,
        query_start_loc=indptr,
        intermediate_states_buffer=state.intermediate_ssm,
        intermediate_state_indices=scratch_indices,
        cache_steps=steps,
        retrieve_parent_token=None,
        stable_rows=False,
    )


TARGETS = (
    "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.__init__",
    "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_extend",
)


def register_ragged_gdn_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGETS[0], after_backend_init, HookType.AFTER)
    HookRegistry.register(TARGETS[1], around_extend, HookType.AROUND)
