"""MSL serial refinement-chain kernels (greedy and sampled).

Both kernels process one request per threadgroup: request ``b`` owns rows
``[b*TM1, (b+1)*TM1)`` of the flattened ``[B*(T-1), ...]`` tensors.  The sampled
kernel reads its sampling parameters from per-request arrays, so a batch of
requests with heterogeneous ``temperature`` / ``top_k`` / ``top_p`` / ``min_p``
can be served in one dispatch.

Semantics mirror ``dspine_mlx.propose`` exactly:

* softmax with max subtraction,
* strict total order ``(p_d > p_c) || (p_d == p_c && d < c)``,
* joint top-k / top-p judged on the unfiltered ``p``, min-p against ``max(p)``,
* first lane whose cumulative mass exceeds ``u * total``,
* ``choice = min(choice, max(keep ? lane : 0))``,
* ``sender`` becomes the chosen lane for the next row.
"""

from __future__ import annotations

import mlx.core as mx

from .common import as_f32, as_i32, make_kernel

_SOURCE = r"""
uint b = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
if (lane != 0) return;

const float invt = inv_t[b];
const int topk = top_k[b];
const float topp = top_p[b];
const float minp = min_p[b];

int sender = 0;
const int n0 = (int)b * TM1;

for (int r = 0; r < TM1; ++r) {
    const int n = n0 + r;
    const device float* row =
        table + (size_t)n * (size_t)K * (size_t)K + (size_t)sender * (size_t)K;

    // softmax(row * inv_t) with max subtraction
    float p[K];
    float vmax = -INFINITY;
    for (int k = 0; k < K; ++k) {
        p[k] = row[k] * invt;
        vmax = fmax(vmax, p[k]);
    }
    float denom = 0.0f;
    for (int k = 0; k < K; ++k) {
        p[k] = exp(p[k] - vmax);
        denom += p[k];
    }
    for (int k = 0; k < K; ++k) p[k] /= denom;

    float pmax = -INFINITY;
    for (int k = 0; k < K; ++k) pmax = fmax(pmax, p[k]);

    // joint top-k / top-p + min-p
    float w[K];
    float wsum = 0.0f;
    int last = 0;
    for (int c = 0; c < K; ++c) {
        int rank = 0;
        float before = 0.0f;
        for (int d = 0; d < K; ++d) {
            const bool ahead = (p[d] > p[c]) || (p[d] == p[c] && d < c);
            if (ahead) {
                rank += 1;
                before += p[d];
            }
        }
        bool keep = (rank < topk) && (before <= topp);
        if (minp > 0.0f) keep = keep && (p[c] >= minp * pmax);
        w[c] = keep ? p[c] : 0.0f;
        wsum += w[c];
        if (keep) last = c;
    }

    for (int k = 0; k < K; ++k) qout[n * K + k] = w[k] / wsum;

    // inverse-CDF sample
    float acc = 0.0f;
    float total = 0.0f;
    for (int k = 0; k < K; ++k) total += w[k] / wsum;
    const float threshold = u[n] * total;
    int choice = 0;
    for (int k = 0; k < K; ++k) {
        acc += w[k] / wsum;
        if (acc <= threshold) choice += 1;
    }
    if (choice > last) choice = last;
    tok[n] = cand[n * K + choice];
    sender = choice;
}
"""

_GREEDY_SOURCE = r"""
uint b = threadgroup_position_in_grid.x;
if (thread_position_in_threadgroup.x != 0) return;

int j = idx[(b * TM1) * K];
for (int r = 0; r < TM1; ++r) {
    tok[b * TM1 + r] = cand[(b * TM1 + r) * K + j];
    if (r + 1 < TM1) j = idx[(b * TM1 + r + 1) * K + j];
}
"""

_KERNELS: dict[str, object] = {}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    if name == "sampled":
        kernel = make_kernel(
            "dspine_serial_sample_chain",
            ["table", "cand", "inv_t", "top_k", "top_p", "min_p", "u"],
            ["tok", "qout"],
            _SOURCE,
        )
    elif name == "greedy":
        kernel = make_kernel(
            "dspine_greedy_chain",
            ["idx", "cand"],
            ["tok"],
            _GREEDY_SOURCE,
        )
    else:
        raise KeyError(name)
    _KERNELS[name] = kernel
    return kernel


def _batch_shape(n: int, block_size: int | None) -> tuple[int, int]:
    """Return ``(requests, rows_per_request)`` for a flattened row count."""
    if block_size is None:
        return 1, n
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    tm1 = block_size - 1
    if n % tm1:
        raise ValueError(f"row count {n} is not divisible by block_size-1={tm1}")
    return n // tm1, tm1


def _param(value, batch: int, dtype) -> mx.array:
    """Broadcast a scalar parameter to ``[batch]``; accept an existing array."""
    if isinstance(value, mx.array):
        if value.ndim == 0:
            return mx.full((batch,), float(value), dtype=dtype)
        if value.size == 1:
            return mx.full((batch,), float(value.reshape(-1)[0]), dtype=dtype)
        if value.shape != (batch,):
            raise ValueError(f"parameter shape {value.shape} != ({batch},)")
        return value.astype(dtype)
    return mx.full((batch,), float(value), dtype=dtype)


def serial_sample_chain(
    table: mx.array,
    cand: mx.array,
    *,
    inv_temp,
    top_k,
    top_p,
    min_p,
    uniforms: mx.array,
    block_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Sample each request's chain; parameters may be scalars or ``[B]`` arrays.

    ``table`` is fp32 ``[n, K, K]``, ``cand`` int32/int64 ``[n, K]``, and the
    chain is grouped by ``block_size`` (default ``n + 1``, i.e. one request).
    """
    if table.ndim != 3 or table.shape[1] != table.shape[2]:
        raise ValueError(f"table must be [n, K, K], got {table.shape}")
    n, k, _ = table.shape
    batch, tm1 = _batch_shape(n, block_size)
    if tuple(cand.shape) != (n, k):
        raise ValueError(f"cand must be {(n, k)}, got {cand.shape}")
    uniforms = uniforms.astype(mx.float32)
    if uniforms.shape != (n,):
        raise ValueError(f"uniforms must be [{n}], got {uniforms.shape}")

    tok, qout = _kernel("sampled")(
        inputs=[
            table.astype(mx.float32),
            cand.astype(mx.int32),
            _param(inv_temp, batch, mx.float32),
            _param(top_k, batch, mx.int32),
            _param(top_p, batch, mx.float32),
            _param(min_p, batch, mx.float32),
            uniforms,
        ],
        template=[("K", int(k)), ("TM1", int(tm1))],
        grid=(batch * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,), (n, k)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return tok, qout


def greedy_chain(
    idx: mx.array,
    cand: mx.array,
    *,
    block_size: int | None = None,
) -> mx.array:
    """Walk the K-state greedy table; ``block_size`` groups rows per request."""
    n, k = cand.shape
    batch, tm1 = _batch_shape(n, block_size)
    (tok,) = _kernel("greedy")(
        inputs=[idx.astype(mx.int32), cand.astype(mx.int32)],
        template=[("K", int(k)), ("TM1", int(tm1))],
        grid=(batch * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.int32],
    )
    return tok


def warmup(k: int = 16, block_size: int = 16) -> None:
    """Compile both kernels on tiny inputs (call once at worker startup)."""
    tm1 = max(1, int(block_size) - 1)
    n = tm1
    table = mx.zeros((n, k, k), dtype=mx.float32)
    cand = mx.zeros((n, k), dtype=mx.int32)
    idx = mx.zeros((n, k), dtype=mx.int32)
    uniforms = mx.zeros((n,), dtype=mx.float32)
    tok, qout = serial_sample_chain(
        table,
        cand,
        inv_temp=1.0,
        top_k=k,
        top_p=1.0,
        min_p=0.0,
        uniforms=uniforms,
        block_size=block_size,
    )
    greedy = greedy_chain(idx, cand, block_size=block_size)
    mx.eval(tok, qout, greedy)
