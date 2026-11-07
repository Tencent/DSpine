"""Full-vocabulary target sampling and factorized-q numerical checks."""

import pytest
import torch
from ref_utils import sampling_info

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _info(bs):
    info = sampling_info(bs, temperature=1.0, top_k=20, top_p=0.95)
    for key, value in vars(info).items():
        if isinstance(value, torch.Tensor):
            setattr(info, key, value.cuda())
    return info


@cuda
@pytest.mark.parametrize("bs", [1, 4, 8])
def test_sparse_target_matches_native_joint_distribution_bitwise(bs):
    from sglang_dspine.runtime.worker import build_dspine_target_probs
    from sglang_dspine.sampling.sparse import materializer, producer

    torch.manual_seed(37)
    logits = torch.randn((bs * 16, 248320), device="cuda", dtype=torch.float32)
    logits[::3, :16] += 15
    logits[1::3].fill_(float("-inf"))
    logits[1::3, 0] = 0
    info = _info(bs)
    producer.initialize()
    bundle = producer.build_sparse_joint(
        logits=logits, info=info, bs=bs, width=16, native=build_dspine_target_probs
    )
    actual = materializer.materialize(torch, bundle)
    expected = build_dspine_target_probs(
        next_token_logits=logits,
        sampling_info=info,
        block_size=16,
        bs=bs,
        sampling_backend="flashinfer",
    ).reshape_as(actual)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@cuda
@pytest.mark.parametrize("bs", [4, 8])
def test_sparse_rejection_consumes_the_same_q_and_coins(bs):
    from sglang_dspine.runtime.worker import build_dspine_target_probs
    from sglang_dspine.sampling.rejection import sample_capped_chain_from_probs
    from sglang_dspine.sampling.sparse import accept, materializer, producer

    torch.manual_seed(19)
    logits = torch.randn((bs * 16, 248320), device="cuda")
    logits[:, :16] += 15
    info = _info(bs)
    producer.initialize()
    b = producer.build_sparse_joint(
        logits=logits, info=info, bs=bs, width=16, native=build_dspine_target_probs
    )
    p = materializer.materialize(torch, b).view(bs, 16, -1)
    ids = torch.arange(16, device="cuda").expand(bs, 15, 16).contiguous()
    q = torch.softmax(torch.randn((bs, 15, 16), device="cuda"), dim=-1)
    draws = torch.multinomial(q.reshape(-1, 16), 1).view(bs, 15)
    candidates = torch.cat(
        (torch.zeros((bs, 1), device="cuda", dtype=torch.int64), draws), 1
    )
    lens = torch.tensor([12] * bs, device="cuda", dtype=torch.int32)
    if bs == 8:
        lens = torch.tensor(
            [1, 5, 9, 12, 16, 15, 14, 16], device="cuda", dtype=torch.int32
        )
    coins = torch.rand((bs, 15), device="cuda")
    final = torch.rand(bs, device="cuda")
    prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 8190

    def buffers():
        return (
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int64),
            torch.empty((bs, 16), device="cuda", dtype=torch.int64),
            torch.empty(bs, device="cuda", dtype=torch.int32),
        )

    actual, expected = buffers(), buffers()
    dense_q = torch.zeros((bs, 15, 248320), device="cuda")
    dense_q.scatter_(-1, ids, q)
    sample_capped_chain_from_probs(
        candidates=candidates,
        target_probs=p,
        draft_probs=dense_q,
        verify_lens=lens,
        uniforms=coins,
        uniform_final=final,
        prefix_lens=prefix,
        outputs=expected,
    )
    accept.accept_sparse(
        torch,
        candidates=candidates,
        dense_p=b["raw_masked"],
        packed=(b["ids"], b["probs"], b["lengths"], b["overflow"]),
        q_ids=ids,
        q_values=q,
        verify_lens=lens,
        uniforms=coins,
        uniform_final=final,
        prefix_lens=prefix,
        outputs=actual,
        scratch=torch.empty((bs, 4096), device="cuda"),
        denominator=b["denominator"],
        raw_fallback=True,
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))


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
