"""Full-vocabulary target sampling and factorized-q numerical checks."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_factorized_q_matches_direct_float64_refinement():
    from sglang_dspine.sampling.factorized.kernels import Workspace, factor_table
    from sglang_dspine.sampling.factorized.weights import build_cache

    torch.manual_seed(42)
    bs, t, k, h, r, c, v = 2, 16, 16, 64, 8, 16, 64
    n = bs * (t - 1)
    rand = lambda *shape: torch.randn(shape, device="cuda") * 0.1
    raw = rand(bs * t, h).bfloat16()
    rms = torch.ones(bs * t, device="cuda")
    query = rand(bs * t, r).bfloat16()
    anchors = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    candidates = torch.arange(k, device="cuda").expand(n, k).contiguous()
    codebook, head = rand(v, c).bfloat16(), rand(v, h).bfloat16()
    pred, prev, bias, write, norm = (
        rand(r, c),
        rand(r, c),
        rand(r),
        rand(h, r),
        torch.ones(h, device="cuda"),
    )
    cache = build_cache(codebook, pred, prev, bias, write, norm, head)
    out = torch.empty((n, k, k), device="cuda")
    base = torch.empty((n, k), device="cuda", dtype=torch.float64)
    factor_table(
        raw,
        rms,
        query,
        anchors,
        candidates,
        cache,
        1e-6,
        t,
        out,
        base,
        Workspace.allocate(n, k, h, r, "cuda"),
    )
    source = (
        torch.arange(n, device="cuda") // 15 * 16
        + torch.arange(n, device="cuda") % 15
        + 1
    )
    for row in range(n):
        predecessor = (
            anchors[row // 15].expand(k) if row % 15 == 0 else candidates[row - 1]
        )
        message = (
            torch.sigmoid(
                query[source[row]].double() + cache["G"][predecessor] + cache["bias"]
            )
            * cache["V"][predecessor]
        )
        state = raw[source[row]].double()[None, :] + message @ cache["W"].T
        normalized = state / torch.sqrt(state.square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized * cache["norm"] @ head[candidates[row]].double().T
        torch.testing.assert_close(out[row], expected.float(), rtol=2e-5, atol=2e-6)
