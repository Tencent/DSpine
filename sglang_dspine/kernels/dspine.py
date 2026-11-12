"""DSpine serving on SGLang 0.5.22."""

import torch
import triton
import triton.language as tl

TOPK_TILE = 4096
TOPK_CHUNK = 64


@triton.jit
def prepare_read(
    X,
    Residual,
    Raw,
    Normalized,
    RMS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def normalize_codes(
    Projected,
    Codebook,
    Anchors,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    ANCHOR_ROW: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if ANCHOR_ROW and row % T == 0:
        token = tl.load(Anchors + row // T)
        z = tl.load(Codebook + token * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = tl.sqrt(float(C)) * z / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def feature_message(
    Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid(query + sender + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def feature_finish(
    Raw,
    Delta,
    RMS,
    Valid,
    Weight,
    Output,
    Normed,
    H: tl.constexpr,
    T: tl.constexpr,
    WRITE: tl.constexpr,
    NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    previous = tl.maximum(row - 1, 0)
    edge = (row % T != 0) & tl.load(Valid + row) & tl.load(Valid + previous)
    update = rms * delta * tl.cast(edge & WRITE, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    if NORM:
        x = result.to(tl.float32)
        variance = tl.sum(x * x, 0) / H
        y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
        weight = tl.load(Weight + k, k < H, 0)
        y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
        tl.store(Normed + row * H + k, y, k < H)


@triton.jit
def _chunk_max(X, stride_x, V, CM, NCH, BLOCK: tl.constexpr, CH: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + offs, mask=offs < V, other=float("-inf")).to(
        tl.float32
    )
    m = tl.max(tl.reshape(x, (BLOCK // CH, CH)), axis=1)
    j = blk * (BLOCK // CH) + tl.arange(0, BLOCK // CH)
    tl.store(CM + row * NCH + j, m, mask=j < NCH)


@triton.jit
def _select_topk(
    X,
    stride_x,
    V,
    CM,
    NCH,
    OUT,
    K: tl.constexpr,
    CH: tl.constexpr,
    NCHPAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NCHPAD)
    cm = tl.load(CM + row * NCH + offs, mask=offs < NCH, other=float("-inf"))
    kk = tl.arange(0, K)
    chunks = tl.zeros([K], dtype=tl.int32)
    for j in tl.static_range(K):
        (m, i) = tl.max(
            cm, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        chunks = tl.where(kk == j, i, chunks)
        cm = tl.where(offs == i, float("-inf"), cm)
    chunks = tl.sort(chunks)
    idx = tl.reshape(chunks[:, None] * CH + tl.arange(0, CH)[None, :], (K * CH,))
    vals = tl.load(X + row * stride_x + idx, mask=idx < V, other=float("-inf")).to(
        tl.float32
    )
    lane = tl.arange(0, K * CH)
    res = tl.zeros([K], dtype=tl.int32)
    for j in tl.static_range(K):
        (m, i) = tl.max(
            vals, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        sel = tl.sum(tl.where(lane == i, idx, 0), axis=0)
        res = tl.where(kk == j, sel, res)
        vals = tl.where(lane == i, float("-inf"), vals)
    res = tl.sort(res)
    tl.store(OUT + row * K + kk, res.to(tl.int64))


def topk_ids(logits: torch.Tensor, k: int, out: torch.Tensor = None) -> torch.Tensor:
    """logits [N, V] (contiguous rows) -> int64 [N, k] ids of the k largest values, ascending by id.

    Two stages: the max of every 64-wide chunk, then per row the k chunks with the largest maxima
    and the top-k of their k*64 elements.  The k largest values of a row always lie in the k
    chunks with the largest maxima, so the selected value multiset equals torch.topk's; ties
    straddling the k-th place go to lower ids within the selected chunks."""
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("topk_ids expects a row-contiguous [N, V] tensor")
    if k & k - 1 or not 1 <= k <= 64:
        raise ValueError(f"topk_ids needs a power-of-two k in [1, 64], got {k}")
    (n, v) = logits.shape
    ch = TOPK_CHUNK
    nch = triton.cdiv(v, ch)
    if nch < k:
        raise ValueError(f"topk_ids needs at least k={k} chunks of {ch}, vocab={v}")
    if out is None:
        out = torch.empty((n, k), dtype=torch.int64, device=logits.device)
    elif (
        tuple(out.shape) != (n, k)
        or out.dtype != torch.int64
        or (not out.is_contiguous())
    ):
        raise ValueError("topk_ids: out must be a contiguous int64 [N, k] tensor")
    if n == 0:
        return out
    cm = torch.empty((n, nch), dtype=torch.float32, device=logits.device)
    _chunk_max[n, triton.cdiv(v, TOPK_TILE)](
        logits, logits.stride(0), v, cm, nch, BLOCK=TOPK_TILE, CH=ch, num_warps=4
    )
    _select_topk[n,](
        logits,
        logits.stride(0),
        v,
        cm,
        nch,
        out,
        K=k,
        CH=ch,
        NCHPAD=triton.next_power_of_2(nch),
        num_warps=8,
    )
    return out


@triton.jit
def _serial_table(
    Raw,
    stride_raw,
    NormW,
    Cand,
    InputIds,
    LmW,
    stride_w,
    TG,
    TSV,
    Query,
    stride_q,
    Rms,
    GateB,
    Out,
    Confidence,
    H: tl.constexpr,
    RK: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TM1: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    n = tl.program_id(0)
    b = n // TM1
    r = n % TM1
    row = b * (TM1 + 1) + 1 + r
    kp = tl.arange(0, KP)
    live = kp < K
    ids = tl.load(Cand + n * K + kp, mask=live, other=0)
    prev = tl.load(Cand + tl.maximum(n - 1, 0) * K + kp, mask=live, other=0)
    anchor = tl.load(InputIds + b * (TM1 + 1)).to(tl.int64)
    sid = tl.where(r == 0, anchor, prev)
    base = tl.zeros([KP], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BH):
        off = h0 + tl.arange(0, BH)
        u = tl.load(Raw + row * stride_raw + off).to(tl.float32) * tl.load(
            NormW + off
        ).to(tl.float32)
        w = tl.load(
            LmW + ids[:, None] * stride_w + off[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        base += tl.sum(w * u[None, :], axis=1)
    s = tl.zeros([KP, KP], dtype=tl.float32)
    for r0 in tl.static_range(0, RK, BR):
        rr = r0 + tl.arange(0, BR)
        gate = tl.load(
            TSV + sid[:, None] * (2 * RK) + rr[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        val = tl.load(
            TSV + sid[:, None] * (2 * RK) + RK + rr[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(Query + row * stride_q + rr).to(tl.float32)
        gb = tl.load(GateB + rr).to(tl.float32)
        m = tl.sigmoid(q[None, :] + gate + gb[None, :]) * val
        g = tl.load(
            TG + ids[:, None] * RK + rr[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        s += tl.dot(m, tl.trans(g), input_precision="ieee")
    base = tl.where(live, base, -1e30)
    s = base[None, :] + tl.load(Rms + row).to(tl.float32) * s
    (best_value, best) = tl.max(
        s, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(Out + n * K + kp, best, mask=live)
    if WITH_CONFIDENCE:
        scale = tl.maximum(tl.load(Rms + row).to(tl.float32), 1e-06)
        mass = tl.sum(tl.exp((s - best_value[:, None]) / scale), 1)
        tl.store(Confidence + n * K + kp, 1.0 / mass, mask=live)


@triton.jit
def _serial_chain(
    Idx,
    Cand,
    Out,
    Confidence,
    ConfidenceOut,
    K: tl.constexpr,
    TM1: tl.constexpr,
    RPAD: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    b = tl.program_id(0)
    rr = tl.arange(0, RPAD)
    kk = tl.arange(0, K)
    inside = rr[:, None] < TM1
    at = (b * TM1 + rr[:, None]) * K + kk[None, :]
    tab = tl.load(Idx + at, mask=inside, other=0)
    cnd = tl.load(Cand + at, mask=inside, other=0)
    if WITH_CONFIDENCE:
        probs = tl.load(Confidence + at, mask=inside, other=0.0)
    j = tl.sum(tl.where((rr[:, None] == 0) & (kk[None, :] == 0), tab, 0))
    previous = tl.full((), 0, tl.int64)
    for r in tl.static_range(TM1):
        if WITH_CONFIDENCE:
            probability = tl.sum(
                tl.where((rr[:, None] == r) & (kk[None, :] == previous), probs, 0.0)
            )
            tl.store(ConfidenceOut + b * TM1 + r, probability)
        tl.store(
            Out + b * TM1 + r,
            tl.sum(tl.where((rr[:, None] == r) & (kk[None, :] == j), cnd, 0)),
        )
        previous = j
        j = tl.sum(tl.where((rr[:, None] == r + 1) & (kk[None, :] == j), tab, 0))


def build_serial_tables(lm_head_w, norm_w, write_w, pv_w, codebook, chunk: int = 8192):
    """(TG [V, R] bf16, TSV [V, 2R] bf16).  lm_head_w [V, H], norm_w [H], write_w [H, R] (the last
    layer's write), pv_w [2R, C] = [G_p; A], codebook [V, C].
      TG[v]  = bf16(LmHead[v] * norm_w) @ W_last
      TSV[v] = [G_p; A] @ bf16(sqrt(C) * Codebook[v])"""
    (v, h) = lm_head_w.shape
    r = write_w.shape[1]
    c = codebook.shape[1]
    if (
        codebook.shape[0] != v
        or tuple(write_w.shape) != (h, r)
        or tuple(pv_w.shape) != (2 * r, c)
        or (norm_w.shape != (h,))
    ):
        raise ValueError("build_serial_tables: shape contract violated")
    tg = torch.empty((v, r), device=lm_head_w.device, dtype=torch.bfloat16)
    tsv = torch.empty((v, 2 * r), device=lm_head_w.device, dtype=torch.bfloat16)
    (nwf, wb, scale) = (norm_w.float(), write_w.to(torch.bfloat16), float(c) ** 0.5)
    for s in range(0, v, chunk):
        e = min(s + chunk, v)
        tg[s:e] = torch.matmul((lm_head_w[s:e].float() * nwf).to(torch.bfloat16), wb)
        tsv[s:e] = torch.nn.functional.linear(
            (scale * codebook[s:e].float()).to(torch.bfloat16), pv_w
        )
    return (tg, tsv)


def serial_table(
    raw,
    norm_w,
    cand,
    input_ids,
    lm_head_w,
    tg,
    tsv,
    packed,
    rms,
    gate_b,
    block_size: int,
    confidence=None,
):
    """raw [bs*T, H] bf16 pre-write state of the last write; cand int64 [bs*(T-1), K] ascending;
    input_ids [bs*T]; packed [bs*T, >=R] bf16 with the receiver query in columns :R; rms fp32
    [bs*T]  ->  idx int64 [bs*(T-1), K]: idx[n, j] = the best candidate lane of row n when the
    previous row chose lane j (all j equal on block row 1, whose sender is the anchor)."""
    t = int(block_size)
    (m, h) = raw.shape
    (n, k) = cand.shape
    rk = tg.shape[1]
    if m % t or n != m // t * (t - 1) or input_ids.numel() != m or (rms.numel() != m):
        raise ValueError(
            f"serial_table row mismatch: raw {raw.shape}, cand {cand.shape}, T={t}"
        )
    if (
        raw.dtype != torch.bfloat16
        or raw.stride(1) != 1
        or packed.dtype != torch.bfloat16
        or (packed.stride(1) != 1)
    ):
        raise ValueError("serial_table expects row-contiguous BF16 state and query")
    if (
        packed.shape[0] != m
        or packed.shape[1] < rk
        or rms.dtype != torch.float32
        or (not rms.is_contiguous())
    ):
        raise ValueError("serial_table query/rms contract violated")
    if (
        not cand.is_contiguous()
        or cand.dtype != torch.int64
        or input_ids.dtype != torch.int64
        or (not input_ids.is_contiguous())
    ):
        raise ValueError(
            "serial_table expects contiguous int64 candidates and input ids"
        )
    if (
        lm_head_w.stride(1) != 1
        or tuple(tsv.shape) != (tg.shape[0], 2 * rk)
        or (not (tg.is_contiguous() and tsv.is_contiguous()))
    ):
        raise ValueError("serial_table table layout mismatch")
    if (
        h % 512
        or rk % 128
        or k & k - 1
        or (not 2 <= k <= 64)
        or (gate_b.numel() != rk)
        or (norm_w.numel() != h)
    ):
        raise ValueError("serial_table size contract violated")
    idx = torch.empty((n, k), dtype=torch.int64, device=raw.device)
    if confidence is not None and (
        tuple(confidence.shape) != (n, k)
        or confidence.dtype != torch.float32
        or (not confidence.is_contiguous())
    ):
        raise ValueError("Confidence output must be contiguous float32 [N, K]")
    if n:
        _serial_table[n,](
            raw,
            raw.stride(0),
            norm_w,
            cand,
            input_ids,
            lm_head_w,
            lm_head_w.stride(0),
            tg,
            tsv,
            packed,
            packed.stride(0),
            rms,
            gate_b,
            idx,
            confidence if confidence is not None else idx,
            H=h,
            RK=rk,
            K=k,
            KP=max(k, 16),
            TM1=t - 1,
            BH=512,
            BR=128,
            WITH_CONFIDENCE=confidence is not None,
            num_warps=8,
            num_stages=2,
        )
    return idx


def serial_chain(idx, cand, out, block_size: int, confidence=None, confidence_out=None):
    """Walk every request's greedy table: out[:bs*(T-1)] written in place (int64)."""
    t = int(block_size)
    (n, k) = cand.shape
    if (
        n % (t - 1)
        or tuple(idx.shape) != (n, k)
        or (not (idx.is_contiguous() and cand.is_contiguous()))
    ):
        raise ValueError("serial_chain layout mismatch")
    if out.dtype != torch.int64 or out.numel() < n or idx.dtype != torch.int64:
        raise ValueError("serial_chain dtype/capacity contract violated")
    if confidence is not None and (
        confidence_out is None
        or confidence_out.dtype != torch.float32
        or confidence_out.numel() < n
        or (tuple(confidence.shape) != (n, k))
    ):
        raise ValueError(
            "Serial confidence needs a float32 [N, K] table and [N] output"
        )
    if n:
        _serial_chain[n // (t - 1),](
            idx,
            cand,
            out,
            confidence if confidence is not None else idx,
            confidence_out if confidence_out is not None else out,
            K=k,
            TM1=t - 1,
            RPAD=triton.next_power_of_2(t - 1),
            WITH_CONFIDENCE=confidence is not None,
            num_warps=1,
        )
    return out


def serial_rs_table(
    raw,
    rms,
    query,
    anchors,
    cand,
    codebook,
    pred_gate_w,
    prev_w,
    gate_b,
    write_w,
    norm_w,
    lm_head_w,
    eps: float,
    block_size: int,
    out: torch.Tensor = None,
    chunk_rows: int = 256,
    res_emb: torch.Tensor = None,
    res_proj: torch.Tensor = None,
):
    """fp32 [n, K, K] sampling table of the serial last-write refinement (module docstring).

    raw [bs*T, H] bf16 (pre-write state of the last write), rms [bs*T] fp32 (its write scale),
    query [bs*T, >=R] (the receiver query columns of the stash), anchors [bs] int64, cand [n, K]
    int64 with n = bs*(T-1).  pred_gate_w [R, C], prev_w [R, C], gate_b [R], write_w [H, R],
    norm_w [H] must already be fp32 (the caller converts once); lm_head_w [V, H] and codebook
    [V, C] are gathered by row and converted per chunk.  Rows are processed ``chunk_rows`` at a
    time so the fp32 intermediates stay bounded (~chunk_rows * K * H * 4 bytes each).  The caller
    guarantees TF32 is off, so the table is computed in true fp32.  With the terminal token
    residual, res_emb [V, r] (gathered by sender id, converted per chunk) and res_proj [R, r] (fp32)
    add res_emb[s] @ res_proj^T to each sender's value (the engine's terminal_residual_values).
    """
    t = int(block_size)
    (m, h) = raw.shape
    bs = m // t
    (n, k) = cand.shape
    r = pred_gate_w.shape[0]
    if (
        m % t
        or n != bs * (t - 1)
        or rms.shape != (m,)
        or (query.shape[0] != m)
        or (query.shape[1] < r)
        or (anchors.shape != (bs,))
    ):
        raise ValueError("serial_rs_table layout mismatch")
    f32 = torch.float32
    for name, w in (
        ("pred_gate_w", pred_gate_w),
        ("prev_w", prev_w),
        ("gate_b", gate_b),
        ("write_w", write_w),
        ("norm_w", norm_w),
    ):
        if w.dtype != f32:
            raise ValueError(f"serial_rs_table: {name} must be fp32")
    if (res_emb is None) != (res_proj is None):
        raise ValueError("serial_rs_table: res_emb and res_proj go together")
    if res_proj is not None and (
        res_proj.dtype != f32
        or res_proj.shape[0] != r
        or res_emb.shape != (codebook.shape[0], res_proj.shape[1])
    ):
        raise ValueError("serial_rs_table: terminal residual layout mismatch")
    if out is None:
        out = torch.empty((n, k, k), dtype=f32, device=raw.device)
    elif tuple(out.shape) != (n, k, k) or out.dtype != f32 or (not out.is_contiguous()):
        raise ValueError(
            "serial_rs_table: out must be a contiguous fp32 [n, K, K] tensor"
        )
    if n == 0:
        return out
    x_all = raw.view(bs, t, h)[:, 1:, :].reshape(n, h)
    scale_all = rms.view(bs, t)[:, 1:].reshape(n)
    q_all = query.view(bs, t, -1)[:, 1:, :r].reshape(n, r)
    sid_all = torch.cat(
        (anchors.view(bs, 1, 1).expand(bs, 1, k), cand.view(bs, t - 1, k)[:, :-1, :]),
        dim=1,
    ).reshape(n, k)
    sqrt_c = float(codebook.shape[1]) ** 0.5
    for s in range(0, n, int(chunk_rows)):
        e = min(n, s + int(chunk_rows))
        x = x_all[s:e].to(f32)
        scale = scale_all[s:e].to(f32)
        q = q_all[s:e].to(f32)
        exact = sqrt_c * codebook[sid_all[s:e]].to(f32)
        gate = torch.sigmoid(q[:, None, :] + exact @ pred_gate_w.T + gate_b)
        value = exact @ prev_w.T
        if res_proj is not None:
            value = value + res_emb[sid_all[s:e]].to(f32) @ res_proj.T
        message = gate * value
        refined = x[:, None, :] + scale[:, None, None] * (message @ write_w.T)
        normalized = (
            refined / torch.sqrt(refined.square().mean(-1, keepdim=True) + eps) * norm_w
        )
        head = lm_head_w[cand[s:e]].to(f32)
        out[s:e] = torch.einsum("njh,nch->njc", normalized, head)
    return out


@triton.jit
def _serial_sample_chain(
    Table,
    InvT,
    TopK,
    TopP,
    MinP,
    Greedy,
    U,
    Cand,
    GreedyTok,
    OutTok,
    OutQ,
    Confidence,
    K: tl.constexpr,
    TM1: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    b = tl.program_id(0)
    lane = tl.arange(0, K)
    inv_t = tl.load(InvT + b)
    top_k = tl.load(TopK + b)
    top_p = tl.load(TopP + b)
    min_p = tl.load(MinP + b)
    greedy = tl.load(Greedy + b) != 0
    sender = tl.full((), 0, tl.int32)
    for row in tl.static_range(TM1):
        n = b * TM1 + row
        table = tl.load(Table + n * K * K + lane[:, None] * K + lane[None, :])
        scores = tl.sum(tl.where(lane[:, None] == sender, table, 0.0), axis=0) * inv_t
        mass = tl.exp(scores - tl.max(scores, axis=0))
        p = mass / tl.sum(mass, axis=0)
        if WITH_CONFIDENCE:
            if not greedy:
                # Actual sampled prefix, before filtering or drawing this token.
                tl.store(Confidence + n, tl.max(p, axis=0))
        p_c = p[:, None]
        p_d = p[None, :]
        ahead = (p_d > p_c) | (p_d == p_c) & (lane[None, :] < lane[:, None])
        rank = tl.sum(ahead.to(tl.int32), axis=1)
        before = tl.sum(tl.where(ahead, p_d, 0.0), axis=1)
        keep = (rank < top_k) & (before <= top_p)
        keep = keep & (p >= min_p * tl.max(p, axis=0))
        w = tl.where(keep, p, 0.0)
        q = w / tl.sum(w, axis=0)
        cdf = tl.cumsum(q, axis=0)
        total = tl.sum(tl.where(lane == K - 1, cdf, 0.0), axis=0)
        threshold = tl.load(U + n) * total
        choice = tl.sum((cdf <= threshold).to(tl.int32), axis=0)
        last = tl.max(tl.where(keep, lane, 0), axis=0)
        choice = tl.minimum(choice, last)
        cand = tl.load(Cand + n * K + lane)
        sampled = tl.sum(tl.where(lane == choice, cand, 0), axis=0)
        gtok = tl.load(GreedyTok + n)
        tl.store(OutTok + n, tl.where(greedy, gtok, sampled))
        tl.store(
            OutQ + n * K + lane, tl.where(greedy, (cand == gtok).to(tl.float32), q)
        )
        sender = choice


def serial_sample_chain(
    table,
    inv_temp,
    top_k,
    top_p,
    min_p,
    greedy,
    u,
    cand,
    greedy_tok,
    out_tok,
    out_q,
    block_size: int,
    *,
    confidence_out=None,
):
    """Sample every request's chain (module docstring); fills out_tok[:n] / out_q[:n] in place.

    table fp32 [n, K, K]; inv_temp fp32 [>=bs]; top_k int32 [>=bs] (>= 1; >= K disables it);
    top_p fp32 [>=bs] in (0, 1]; min_p fp32 [>=bs] in [0, 1]; greedy int32 [>=bs]; u fp32 [>=n]
    in [0, 1); cand int64 [n, K]; greedy_tok int64 [>=n]; out_tok int64 [>=n]; out_q fp32 [>=n, K].

    Optional confidence_out fp32 [>=n] stores the maximum temperature-softmax
    probability conditioned on the actual sampled prefix x_<i, before top-k,
    top-p, and min-p filtering or drawing x_i. Greedy rows retain their existing
    greedy confidence; tokens, q, and random draws are unchanged.
    """
    t = int(block_size)
    (n, k) = cand.shape
    if n % (t - 1) or tuple(table.shape) != (n, k, k) or (not table.is_contiguous()):
        raise ValueError("serial_sample_chain table layout mismatch")
    bs = n // (t - 1)
    staged = (inv_temp, top_k, top_p, min_p, greedy)
    if any((x.numel() < bs for x in staged)) or u.numel() < n or greedy_tok.numel() < n:
        raise ValueError("serial_sample_chain staging buffers too small")
    if out_tok.numel() < n or out_q.shape[0] < n or out_q.shape[1] != k:
        raise ValueError("serial_sample_chain output buffers too small")
    if not (cand.is_contiguous() and out_q.is_contiguous()):
        raise ValueError("serial_sample_chain expects contiguous candidates and q")
    if confidence_out is not None and (
        confidence_out.dtype != torch.float32
        or confidence_out.numel() < n
        or not confidence_out.is_contiguous()
        or confidence_out.device != table.device
    ):
        raise ValueError(
            "serial_sample_chain confidence needs contiguous float32 storage on the table device"
        )
    dtypes = (
        table.dtype,
        inv_temp.dtype,
        top_k.dtype,
        top_p.dtype,
        min_p.dtype,
        greedy.dtype,
        u.dtype,
        cand.dtype,
        greedy_tok.dtype,
        out_tok.dtype,
        out_q.dtype,
    )
    expected = (
        torch.float32,
        torch.float32,
        torch.int32,
        torch.float32,
        torch.float32,
        torch.int32,
        torch.float32,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.float32,
    )
    if dtypes != expected:
        raise ValueError(f"serial_sample_chain dtype contract violated: {dtypes}")
    if bs:
        _serial_sample_chain[bs,](
            table,
            inv_temp,
            top_k,
            top_p,
            min_p,
            greedy,
            u,
            cand,
            greedy_tok,
            out_tok,
            out_q,
            confidence_out if confidence_out is not None else out_q,
            K=k,
            TM1=t - 1,
            WITH_CONFIDENCE=confidence_out is not None,
            num_warps=1,
        )
    return (out_tok, out_q)
