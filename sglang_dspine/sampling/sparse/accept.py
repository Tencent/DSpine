"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl


@triton.jit
def _compress_tiles(
    P,
    Ids,
    Values,
    Counts,
    Bad,
    V: tl.constexpr,
    CAP: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(P + row * V + i, i < V, 0.0)
    keep = (i < V) & (x > 0.0)
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    count = tl.sum(keep.to(tl.int32), 0)
    tl.store(Counts + row * TILES + tile, count)
    tl.store(
        Bad + row * TILES + tile,
        tl.sum(
            ((i < V) & ((x < 0.0) | (x != x) | (tl.abs(x) == float("inf")))).to(
                tl.int32
            ),
            0,
        ),
    )
    dst = (row * TILES + tile) * CAP + rank
    tl.store(Ids + dst, i, keep & (rank < CAP))
    tl.store(Values + dst, x, keep & (rank < CAP))


@triton.jit
def _pack_rows(
    TileIds,
    TileValues,
    Counts,
    Bad,
    Ids,
    Values,
    Lengths,
    Overflow,
    CAP: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BT)
    n = tl.load(Counts + row * TILES + tile, tile < TILES, 0)
    before = tl.cumsum(n, 0) - n
    total = tl.sum(n, 0)
    bad = tl.sum(tl.load(Bad + row * TILES + tile, tile < TILES, 0), 0)
    p = tl.arange(0, CAP)
    tl.store(Ids + row * CAP + p, -1)
    tl.store(Values + row * CAP + p, 0.0)
    tl.debug_barrier()
    j = tl.arange(0, B)
    t = j // CAP
    k = j % CAP
    count = tl.gather(n, t, 0)
    prefix = tl.gather(before, t, 0)
    valid = (t < TILES) & (k < count) & (prefix + k < CAP)
    src = (row * TILES + t) * CAP + k
    ids = tl.load(TileIds + src, valid, 0)
    vals = tl.load(TileValues + src, valid, 0.0)
    tl.store(Ids + row * CAP + prefix + k, ids, valid)
    tl.store(Values + row * CAP + prefix + k, vals, valid)
    tl.store(Lengths + row, tl.minimum(total, CAP))
    tl.store(Overflow + row, (total > CAP) | (bad > 0))


@triton.jit
def _empty_p_tile(PIds, PLen, row, start, CAP: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, CAP)
    length = tl.load(PLen + row)
    ids = tl.load(PIds + row * CAP + lane, lane < length, -1)
    return (
        tl.sum(
            ((lane < length) & (ids >= start) & (ids < start + BLOCK)).to(tl.int32), 0
        )
        == 0
    )


@triton.jit
def _tile_values(
    DenseP,
    PIds,
    PVals,
    PLen,
    QIds,
    QVals,
    Scratch,
    row,
    qi,
    b,
    start,
    pure,
    dense,
    Denom,
    V: tl.constexpr,
    CAP: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    RAW: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    offset = start + lane
    tl.store(Scratch + b * BLOCK + lane, 0.0)
    tl.debug_barrier()
    qlane = tl.arange(0, K)
    if dense:
        if pure:
            result = tl.load(DenseP + row * V + offset, offset < V, 0.0)
            if RAW:
                den = tl.load(Denom + row)
                result = tl.inline_asm_elementwise(
                    "div.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    args=[result, den],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        else:
            qid = tl.load(QIds + qi * K + qlane)
            q = tl.load(QVals + qi * K + qlane)
            q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
            tl.store(
                Scratch + b * BLOCK + qid - start,
                q,
                (qid >= start) & (qid < start + BLOCK) & (qid < V),
            )
            tl.debug_barrier()
            qvec = tl.load(Scratch + b * BLOCK + lane)
            pvec = tl.load(DenseP + row * V + offset, offset < V, 0.0)
            if RAW:
                den = tl.load(Denom + row)
                pvec = tl.inline_asm_elementwise(
                    "div.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    args=[pvec, den],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
            result = tl.maximum(pvec - qvec, 0.0)
    else:
        p = tl.arange(0, CAP)
        length = tl.load(PLen + row)
        pid = tl.load(PIds + row * CAP + p, p < length, -1)
        value = tl.load(PVals + row * CAP + p, p < length, 0.0)
        if not pure:
            qid = tl.load(QIds + qi * K + qlane)
            q = tl.load(QVals + qi * K + qlane)
            q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
            qp = tl.sum(tl.where(pid[:, None] == qid[None, :], q[None, :], 0.0), 1)
            value = tl.maximum(value - qp, 0.0)
        tl.store(
            Scratch + b * BLOCK + pid - start,
            value,
            (p < length) & (pid >= start) & (pid < start + BLOCK),
        )
        tl.debug_barrier()
        result = tl.load(Scratch + b * BLOCK + lane)
    return result


@triton.jit
def _accept_sparse(
    Candidates,
    DenseP,
    PIds,
    PVals,
    PLen,
    Overflow,
    QIds,
    QVals,
    Caps,
    U,
    UF,
    Prefix,
    Scratch,
    Denom,
    Accept,
    Commit,
    Bonus,
    Output,
    NewSeq,
    T: tl.constexpr,
    V: tl.constexpr,
    CAP: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    RAW: tl.constexpr,
):
    b = tl.program_id(0)
    cap = tl.load(Caps + b)
    accepted = 0
    step = 1
    continuing = 1
    pl = tl.arange(0, CAP)
    ql = tl.arange(0, K)
    while (step < cap) & (continuing == 1):
        row = b * T + step - 1
        qi = b * (T - 1) + step - 1
        token = tl.load(Candidates + b * T + step)
        if tl.load(Overflow + row):
            p = tl.load(DenseP + row * V + token)
            if RAW:
                den = tl.load(Denom + row)
                p = tl.inline_asm_elementwise(
                    "div.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    args=[p, den],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        else:
            length = tl.load(PLen + row)
            ids = tl.load(PIds + row * CAP + pl, pl < length, -1)
            vals = tl.load(PVals + row * CAP + pl, pl < length, 0.0)
            p = tl.sum(tl.where(ids == token, vals, 0.0), 0)
        qid = tl.load(QIds + qi * K + ql)
        qv = tl.load(QVals + qi * K + ql)
        q = tl.sum(tl.where(qid == token, qv, 0.0), 0)
        coin = tl.load(U + qi)
        if (q > 0.0) & (q <= 1.0) & (coin * q < p):
            accepted += 1
            step += 1
        else:
            continuing = 0
    all_accepted = accepted == cap - 1
    row = b * T + accepted
    qi = b * (T - 1) + accepted
    dense = tl.load(Overflow + row)
    mass = 0.0
    for start in range(0, V, BLOCK):
        empty = False
        if dense == 0:
            empty = _empty_p_tile(PIds, PLen, row, start, CAP, BLOCK)
        if not empty:
            value = _tile_values(
                DenseP,
                PIds,
                PVals,
                PLen,
                QIds,
                QVals,
                Scratch,
                row,
                qi,
                b,
                start,
                all_accepted,
                dense,
                Denom,
                V,
                CAP,
                K,
                BLOCK,
                RAW,
            )
            mass += tl.sum(value, 0)
    pure = all_accepted | (mass <= 0.0)
    if (mass <= 0.0) & (all_accepted == 0):
        mass = 0.0
        for start in range(0, V, BLOCK):
            empty = False
            if dense == 0:
                empty = _empty_p_tile(PIds, PLen, row, start, CAP, BLOCK)
            if not empty:
                value = _tile_values(
                    DenseP,
                    PIds,
                    PVals,
                    PLen,
                    QIds,
                    QVals,
                    Scratch,
                    row,
                    qi,
                    b,
                    start,
                    True,
                    dense,
                    Denom,
                    V,
                    CAP,
                    K,
                    BLOCK,
                    RAW,
                )
                mass += tl.sum(value, 0)
    threshold = tl.load(UF + b) * mass
    before = 0.0
    bonus = V - 1
    found = 0
    for start in range(0, V, BLOCK):
        if found == 0:
            empty = False
            if dense == 0:
                empty = _empty_p_tile(PIds, PLen, row, start, CAP, BLOCK)
            if empty:
                if (before > threshold) & (start < V):
                    bonus = start
                    found = 1
            else:
                value = _tile_values(
                    DenseP,
                    PIds,
                    PVals,
                    PLen,
                    QIds,
                    QVals,
                    Scratch,
                    row,
                    qi,
                    b,
                    start,
                    pure,
                    dense,
                    Denom,
                    V,
                    CAP,
                    K,
                    BLOCK,
                    RAW,
                )
                cdf = before + tl.cumsum(value, 0)
                offset = start + tl.arange(0, BLOCK)
                hit = (cdf > threshold) & (offset < V)
                if tl.max(hit.to(tl.int32), 0) != 0:
                    bonus = start + tl.argmax(hit.to(tl.int32), 0)
                    found = 1
                before += tl.sum(value, 0)
    tl.store(Accept + b, accepted)
    tl.store(Commit + b, accepted + 1)
    tl.store(Bonus + b, bonus)
    i = tl.arange(0, T)
    proposed = tl.load(Candidates + b * T + i + 1, i < T - 1, 0)
    emitted = tl.where(i < accepted, proposed, tl.where(i == accepted, bonus, 0))
    tl.store(Output + b * T + i, emitted)
    tl.store(NewSeq + b, tl.load(Prefix + b) + accepted + 1)


def compress_probs(torch, p, capacity=64):
    (rows, vocab) = p.shape
    tiles = triton.cdiv(vocab, 4096)
    ids = torch.empty((rows, capacity), device=p.device, dtype=torch.int32)
    values = torch.empty((rows, capacity), device=p.device, dtype=torch.float32)
    lengths = torch.empty(rows, device=p.device, dtype=torch.int32)
    overflow = torch.empty(rows, device=p.device, dtype=torch.int32)
    ti = torch.empty((rows, tiles, capacity), device=p.device, dtype=torch.int32)
    tv = torch.empty((rows, tiles, capacity), device=p.device, dtype=torch.float32)
    count = torch.empty((rows, tiles), device=p.device, dtype=torch.int32)
    bad = torch.empty_like(count)
    _compress_tiles[rows, tiles](
        p,
        ti,
        tv,
        count,
        bad,
        V=vocab,
        CAP=capacity,
        TILES=tiles,
        BLOCK=4096,
        num_warps=4,
    )
    _pack_rows[rows,](
        ti,
        tv,
        count,
        bad,
        ids,
        values,
        lengths,
        overflow,
        CAP=capacity,
        TILES=tiles,
        BT=triton.next_power_of_2(tiles),
        B=triton.next_power_of_2(tiles * capacity),
        num_warps=4,
    )
    return (ids, values, lengths, overflow)


def accept_sparse(
    torch,
    *,
    candidates,
    dense_p,
    packed,
    q_ids,
    q_values,
    verify_lens,
    uniforms,
    uniform_final,
    prefix_lens,
    outputs,
    scratch=None,
    denominator=None,
    raw_fallback=False,
):
    (bs, t) = candidates.shape
    vocab = dense_p.shape[-1]
    (ids, vals, lens, overflow) = packed
    if scratch is None:
        scratch = torch.empty((bs, 4096), device=dense_p.device, dtype=torch.float32)
    if denominator is None:
        denominator = torch.ones((bs * t,), device=dense_p.device, dtype=torch.float32)
    assert t == 16 and q_ids.shape == q_values.shape == (bs, 15, 16)
    _accept_sparse[bs,](
        candidates,
        dense_p,
        ids,
        vals,
        lens,
        overflow,
        q_ids,
        q_values,
        verify_lens,
        uniforms,
        uniform_final,
        prefix_lens,
        scratch,
        denominator,
        *outputs,
        T=t,
        V=vocab,
        CAP=ids.shape[-1],
        K=16,
        BLOCK=4096,
        RAW=raw_fallback,
        num_warps=4,
        enable_fp_fusion=True,
        enable_reflect_ftz=True,
    )
    return outputs
