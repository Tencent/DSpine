"""ReplaySSM preserves native geometry and empty-history math at dynamic sizes."""

import importlib
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.runtime.replayssm.launch import select_launch


def _metadata(bs, width=16, heads=2, value_heads=6, key_dim=64, value_dim=64):
    device = torch.device("cuda")
    def tensor(shape, dtype="torch.bfloat16"):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
                               device=device, numel=lambda: __import__("math").prod(shape))
    n, slots = bs * width, bs + 2
    ring = 1 << (width - 1).bit_length()
    return dict(
        q=tensor((n, heads, key_dim)), k=tensor((n, heads, key_dim)),
        v=tensor((n, value_heads, value_dim)), out=tensor((n, value_heads, value_dim)),
        a=tensor((n, value_heads)), b=tensor((n, value_heads)),
        A_log=tensor((value_heads,), "torch.float32"), dt_bias=tensor((value_heads,)),
        checkpoint_state=tensor((slots, value_heads, value_dim, key_dim), "torch.float32"),
        d_cache=tensor((slots, value_heads, ring, value_dim)),
        k_cache=tensor((slots, heads, ring, key_dim)),
        g_cache=tensor((slots, value_heads, ring), "torch.float32"),
        rawv_cache=tensor((slots, value_heads, ring, value_dim)),
        rawk_cache=tensor((slots, heads, ring, key_dim)), beta_cache=None,
        query_start_loc=tensor((bs + 1,), "torch.int32"),
        replay_indices=tensor((bs,), "torch.int64"),
        ssm_state_indices=tensor((bs,), "torch.int32"),
        write_pos=tensor((slots,), "torch.int32"),
        cache_base=tensor((slots,), "torch.int32"), is_flush=tensor((slots,), "torch.int8"),
        max_cache_len=ring, max_spec_len=width, launch_mode="verify", null_block_id=-1,
    )


@pytest.mark.parametrize("bs", [1, 3, 8, 9, 17, 32, 129, 257])
@pytest.mark.parametrize("width", [4, 12, 16, 32])
def test_replay_dispatch_inherits_native_geometry_for_all_batch_sizes(bs, width):
    kwargs = _metadata(bs, width)
    defaults = dict(block_v=64, num_warps=1, num_stages=2, nk=2)
    assert select_launch("DSPINE", kwargs, defaults) == (64, 1, 2, 2)
    kwargs.update(block_v=32, num_warps=4, nk=1)
    assert select_launch("DSPINE", kwargs, defaults) == (32, 4, 2, 1)
    kwargs["beta_cache"] = object()
    assert select_launch("DSPINE", kwargs, defaults) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs,width", [(1, 16), (9, 16), (17, 12), (33, 16)])
def test_empty_history_matches_native_replay_at_dynamic_sizes(bs, width):
    native = importlib.import_module("sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode")
    from sglang_dspine.runtime.replayssm.kernels import empty_history_normal_kernel
    torch.manual_seed(73)
    h, hv, k, v, ring = 2, 6, 64, 64, 16
    lens = (torch.arange(bs, device="cuda", dtype=torch.int32) % width + 1)
    indptr = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), lens.cumsum(0).int()))
    n = int(indptr[-1])
    rand = lambda *shape: (torch.randn(shape, device="cuda") * .1).bfloat16()
    kwargs = dict(
        q=rand(n, h, k), k=rand(n, h, k), v=rand(n, hv, v),
        a=rand(n, hv), b=rand(n, hv), A_log=torch.zeros(hv, device="cuda"),
        dt_bias=rand(hv), out=torch.empty((n, hv, v), device="cuda", dtype=torch.bfloat16),
        checkpoint_state=torch.randn((bs, hv, v, k), device="cuda") * .01,
        d_cache=rand(bs, hv, ring, v), k_cache=rand(bs, h, ring, k),
        g_cache=torch.zeros((bs, hv, ring), device="cuda"),
        rawv_cache=rand(bs, hv, ring, v), rawk_cache=rand(bs, h, ring, k),
        beta_cache=None, query_start_loc=indptr,
        ssm_state_indices=torch.arange(bs, device="cuda", dtype=torch.int32),
        replay_indices=torch.arange(bs, device="cuda", dtype=torch.int64),
        write_pos=torch.zeros(bs, device="cuda", dtype=torch.int32),
        cache_base=torch.arange(bs, device="cuda", dtype=torch.int32) % ring,
        is_flush=torch.zeros(bs, device="cuda", dtype=torch.int8),
        scale=k ** -.5, max_cache_len=ring, max_spec_len=width,
        use_qk_l2norm_in_kernel=True, is_flush_kernel=False,
        block_v=32, num_warps=4, num_stages=2, nk=2, bs_min=4,
        null_block_id=-1, dot_precision="tf32",
    )
    actual = {name: value.clone() if isinstance(value, torch.Tensor) else value
              for name, value in kwargs.items()}
    helper = native._launch_gdn_spec
    scope = dict(helper.__globals__, gdn_replayssm_spec_circular_kernel=empty_history_normal_kernel)
    specialized = types.FunctionType(helper.__code__, scope, "test_empty_replay", helper.__defaults__)
    helper(**kwargs)
    specialized(**actual)
    for name in ("out", "checkpoint_state", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache"):
        torch.testing.assert_close(actual[name], kwargs[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs", [9, 17, 129, 257])
def test_live_cursor_guard_checks_all_rows(bs):
    from sglang_dspine.runtime.replayssm.guard import launch
    requests = torch.arange(bs, device="cuda", dtype=torch.int64)
    mapping = requests.clone()
    wp = torch.zeros(bs, device="cuda", dtype=torch.int32)
    flags = torch.zeros(bs, device="cuda", dtype=torch.int8)
    counts = torch.zeros(bs + 1, device="cuda", dtype=torch.int32)
    launch(torch, requests, mapping, wp, flags, counts, bs=bs, state_slots=bs)
    torch.cuda.synchronize()
    assert counts[bs].item() == 1
