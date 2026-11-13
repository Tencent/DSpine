"""Validate live ReplaySSM request cursors on device before graph replay."""

import triton
import triton.language as tl


@triton.jit
def assert_active_empty(
    REQ,
    MAP,
    WP,
    FLAGS,
    COUNTS,
    N: tl.constexpr,
    REQ_SLOTS: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < N
    request = tl.load(REQ + row, mask=active, other=-1).to(tl.int64)
    in_req = active & (request >= 0) & (request < REQ_SLOTS)
    state = tl.load(MAP + request, mask=in_req, other=-1).to(tl.int64)
    null = in_req & (state <= -1)
    nonnull = in_req & ~null
    in_ring = nonnull & (request < RING_SLOTS)
    valid = in_ring & (state < STATE_SLOTS)
    wp = tl.load(WP + request, mask=valid, other=0)
    flag = tl.load(FLAGS + request, mask=valid, other=0)
    bad = active & ~in_req | nonnull & (~valid | ((wp != 0) | (flag != 0)))
    tl.device_assert(
        tl.sum(bad.to(tl.int32), 0) == 0,
        "DSPine WP0 invariant: active request/state/ring index or cursor/flags invalid",
    )
    if COUNT and tl.program_id(0) == 0:
        tl.atomic_add(COUNTS + N, 1, sem="relaxed")


def launch(torch, requests, mapping, wp, flags, counts, *, bs, state_slots, count=True):
    assert bs >= 1
    assert requests.ndim == mapping.ndim == wp.ndim == flags.ndim == 1
    assert wp.shape == flags.shape and requests.shape[0] >= bs
    if count:
        assert counts.numel() > bs
    assert_active_empty[triton.cdiv(bs, 128),](
        requests,
        mapping,
        wp,
        flags,
        counts,
        N=bs,
        REQ_SLOTS=mapping.shape[0],
        STATE_SLOTS=state_slots,
        RING_SLOTS=wp.shape[0],
        COUNT=count,
        BLOCK=128,
        num_warps=1,
        debug=True,
    )
