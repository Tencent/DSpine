"""Empty-history GDN ReplaySSM verify kernel; operation order matches SGLang."""

import triton
import triton.language as tl


@triton.jit
def empty_history_normal_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    d_cache,
    k_cache,
    g_cache,
    rawv_cache,
    rawk_cache,
    beta_cache,
    query_start_loc,
    ssm_state_indices,
    replay_indices,
    write_pos,
    cache_base,
    is_flush_flags,
    scale,
    stride_q_t: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_qsl: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_replay_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    STATE_FP32: tl.constexpr,
    STORE_EXACT_RAW: tl.constexpr,
    STORE_D_RESIDUAL: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    IS_FLUSH: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V
    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos
    mask_s = o_s < spec_len
    o_s_safe = tl.where(mask_s, o_s, 0)
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    replay_idx = tl.load(replay_indices + i_n * stride_replay_indices).to(tl.int64)
    p_o = o + (bos + o_s_safe[:, None]) * stride_o_t + i_hv * V + o_v[None, :]
    if IS_FLUSH:
        if state_idx <= NULL_BLOCK_ID:
            return
        b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
        if not b_is_flush:
            return
    else:
        if state_idx <= NULL_BLOCK_ID:
            full_mask = (o_s < spec_len)[:, None] & mask_v[None, :]
            tl.store(
                p_o,
                tl.zeros([BS, BV], dtype=tl.float32).to(p_o.dtype.element_ty),
                mask=full_mask,
            )
            return
        b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
        if b_is_flush:
            return
    b_write_pos = tl.load(write_pos + replay_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + replay_idx).to(tl.int32)
    out_mask = mask_s[:, None] & mask_v[None, :]
    b_wp_i = b_write_pos.to(tl.int32)
    cache_valid = o_c < b_write_pos
    phys_c = b_cache_base + o_c & MAX_CACHE_LEN - 1
    phys_spec = b_cache_base + b_wp_i + o_s & MAX_CACHE_LEN - 1
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    a_s = tl.load(a + (bos + o_s_safe) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s_safe) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)
    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            ld = mask_s[:, None] & mask_kt[None, :]
            qn = tl.load(
                q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            knn = tl.load(
                k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            qnorm_acc += tl.sum(qn * qn, axis=1)
            knorm_acc += tl.sum(knn * knn, axis=1)
        q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-06), 0.0)
        k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-06), 0.0)
    else:
        q_rnorm = tl.where(mask_s, 1.0, 0.0)
        k_rnorm = tl.where(mask_s, 1.0, 0.0)
    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)
    write_k = i_v == 0 and i_hv == i_h * (HV // H)
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        q_tile = tl.load(
            q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_raw_tile = k_tile
        q_precise = q_tile * (q_rnorm * scale)[:, None]
        k_precise = k_tile * k_rnorm[:, None]
        k_store_tile = k_precise.to(k_cache.dtype.element_ty)
        k_residual_tile = (k_precise - k_store_tile.to(tl.float32)).to(
            k.dtype.element_ty
        )
        if STATE_FP32:
            q_tile = q_precise
            k_tile = k_precise
        else:
            q_tile = q_precise.to(q.dtype.element_ty)
            k_tile = k_store_tile
        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0)
        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT, input_precision=DOT_PRECISION)
        kq_mat += tl.dot(k_tile, qT, input_precision=DOT_PRECISION)
        if STORE_K_RESIDUAL and (not STATE_FP32):
            k_residual_T = tl.trans(k_residual_tile)
            kk_mat += tl.dot(k_tile, k_residual_T, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(k_residual_tile, kT, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(
                k_residual_tile, k_residual_T, input_precision=DOT_PRECISION
            )
            kq_mat += tl.dot(k_residual_tile, qT, input_precision=DOT_PRECISION)
        if STATE_FP32:
            sc_tile = sc_tile.to(tl.float32)
            hw_q += tl.dot(sc_tile, qT, input_precision=DOT_PRECISION)
            hw_k += tl.dot(sc_tile, kT, input_precision=DOT_PRECISION)
        else:
            sc_tile = sc_tile.to(k_tile.dtype)
            hw_q += tl.dot(sc_tile, qT)
            hw_k += tl.dot(sc_tile, kT)
            if STORE_K_RESIDUAL:
                hw_k += tl.dot(sc_tile, k_residual_T)
        if write_k:
            spec_kt_mask = (
                mask_s[:, None]
                & mask_kt[None, :]
                & (b_write_pos + o_s[:, None] < MAX_CACHE_LEN)
            )
            if STORE_EXACT_RAW or STORE_K_RESIDUAL:
                p_cur_rawk = rawk_cache + (
                    replay_idx * stride_rawk_slot
                    + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                    + o_kt[None, :]
                )
                if STORE_EXACT_RAW:
                    tl.store(
                        p_cur_rawk,
                        k_raw_tile.to(p_cur_rawk.dtype.element_ty),
                        mask=spec_kt_mask,
                    )
                else:
                    tl.store(p_cur_rawk, k_residual_tile, mask=spec_kt_mask)
            p_cur_k = k_cache + (
                replay_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(p_cur_k, k_store_tile, mask=spec_kt_mask)
    if not IS_FLUSH:
        hw_q = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, 0f00000000;",
            constraints="=f,f",
            args=[hw_q],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        hw_k = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, 0f00000000;",
            constraints="=f,f",
            args=[hw_k],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)
    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)
    p_v = v + (bos + o_s_safe[None, :]) * stride_v_t + i_hv * V + o_v[:, None]
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(
        tl.float32
    )
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF
    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)
    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    p_cur_d = d_cache + (
        replay_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    d_hi = D_spec.to(p_cur_d.dtype.element_ty)
    tl.store(p_cur_d, d_hi, mask=mask_v[:, None] & spec_store_mask[None, :])
    if STORE_EXACT_RAW or STORE_D_RESIDUAL:
        p_cur_v = rawv_cache + (
            replay_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
            + o_v[:, None]
        )
        if STORE_EXACT_RAW:
            tl.store(
                p_cur_v,
                v_tile.to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
        else:
            tl.store(
                p_cur_v,
                (D_spec - d_hi.to(tl.float32)).to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
    if i_v == 0:
        p_cur_g = (
            g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        )
        tl.store(p_cur_g, g_s, mask=spec_store_mask)
        if STORE_EXACT_RAW:
            p_cur_beta = (
                beta_cache
                + replay_idx * stride_beta_slot
                + i_hv * MAX_CACHE_LEN
                + phys_spec
            )
            tl.store(p_cur_beta, beta_s, mask=spec_store_mask)
