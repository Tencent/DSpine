"""Dynamic proposal storage and exact rejection across sampling configurations."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang_dspine.sampling.factorized.graph import (
    ProposalGraphs,
    _Buffers,
    _capture_sizes,
)
from sglang_dspine.sampling.target import TargetDistribution


def test_proposal_capture_storage_grows_linearly_with_configured_capacity():
    for capacity in (1, 8, 17, 128, 1024):
        configured = list(range(1, capacity + 1))
        sizes = _capture_sizes(configured, capacity)
        assert sizes[-1] == capacity
        assert all(bs in configured for bs in sizes)
        assert sum(sizes) <= 2 * capacity
    assert _capture_sizes([], 128) == []
    assert _capture_sizes([0, 1, 9, 17, 128], 17) == [1, 17]


def _sampler(bs, device="cpu"):
    t, k, h, r, c, v = 16, 16, 64, 8, 16, 64
    rand = lambda *shape: torch.randn(shape, device=device) * 0.1
    bridge = SimpleNamespace(codebook=rand(v, c).bfloat16())
    return SimpleNamespace(
        draft_model=SimpleNamespace(selected_prefix_bridge=bridge),
        raw=rand(bs * t, h).bfloat16(),
        rms=torch.ones(bs * t, device=device),
        query=rand(bs * t, r).bfloat16(),
        anchors=torch.ones(bs, device=device, dtype=torch.int64),
        candidate_out=torch.arange(k, device=device).expand(bs, t - 1, k).contiguous(),
        pred_gate_f32=rand(r, c),
        prev_f32=rand(r, c),
        gate_b_f32=rand(r),
        write_f32=rand(h, r),
        norm_f32=torch.ones(h, device=device),
        lm_head_weight=rand(v, h).bfloat16(),
        norm_eps=1e-6,
        max_bs=bs,
        block_size=t,
        k=k,
        capture_batch_sizes=[],
    )


def test_factorized_workspace_views_share_capacity_without_stale_rows():
    sampler = _sampler(33)
    buffers = _Buffers(33, sampler)
    for bs in (1, 9, 33, 3, 17):
        n = bs * 15
        out, base, workspace = buffers.view(n)
        assert out.shape == (n, 16, 16)
        assert base.shape == (n, 16)
        workspace.validate(n, 16, 64, 8, "cpu")
        assert workspace.x.data_ptr() == buffers.workspace.x.data_ptr()
        assert (
            workspace.fallback.y.data_ptr() == buffers.workspace.fallback.y.data_ptr()
        )


def test_target_passes_mixed_sampling_parameters_to_native_builder(monkeypatch):
    seen = {}
    expected = torch.ones(17, 16, 3) / 3

    def build(**kwargs):
        seen.update(kwargs)
        return expected

    context = SimpleNamespace(
        get_exec=lambda: SimpleNamespace(
            kernel=SimpleNamespace(sampling_backend="pytorch")
        )
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", context)
    monkeypatch.setitem(
        sys.modules,
        "sglang_dspine.runtime.worker",
        SimpleNamespace(build_dspine_target_probs=build),
    )
    info = SimpleNamespace(
        temperatures=torch.linspace(0.2, 2.0, 17).view(-1, 1),
        top_ks=torch.tensor([1, 7, 1 << 30] * 6)[:17],
        top_ps=torch.linspace(0.4, 1.0, 17),
        min_ps=torch.linspace(0, 0.3, 17),
        is_any_greedy=True,
    )
    logits = torch.randn(17 * 16, 3)
    distribution = TargetDistribution(SimpleNamespace(block_size=16))
    assert distribution.dense(logits, info, 17) is expected
    assert seen["sampling_info"] is info
    assert seen["next_token_logits"] is logits
    assert seen["bs"] == 17 and seen["sampling_backend"] == "pytorch"


def test_proposal_uses_next_captured_bucket_and_clears_padding():
    sampler = _sampler(33)
    cache = object.__new__(ProposalGraphs)
    cache.sampler = sampler
    cache.source_tensors = cache.source_signature = cache.weight_signature = ()
    cache.weights = {}
    cache.stream = None
    cache.eager_buffers = None
    cache.graph_buffers = _Buffers(32, sampler)
    cache.graph_buffers.output.fill_(16)
    cache.entries = {size: Mock() for size in (1, 8, 16, 32)}
    sampler.raw[10 * 16 : 16 * 16].fill_(float("nan"))
    sampler.anchors[10:16].fill_(999)
    sampler.candidate_out[10:16].fill_(999)
    with patch.object(
        torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=42)
    ):
        actual = cache.lookup(10)
    cache.entries[16].replay.assert_called_once()
    cache.entries[32].replay.assert_not_called()
    assert actual.shape == (10 * 15, 16, 16)
    assert bool((actual == 16).all())
    assert torch.isfinite(sampler.raw[10 * 16 : 16 * 16]).all()
    assert bool((sampler.anchors[10:16] == 0).all())
    assert bool((sampler.candidate_out[10:16] == 0).all())
    assert cache.eager_buffers is None


@pytest.mark.parametrize("filtered", [False, True])
def test_unfiltered_target_skips_support_compression(filtered, monkeypatch):
    bs, width, vocab = 9, 16, 97
    p = torch.full((bs, width, vocab), 1 / vocab)
    sampler = SimpleNamespace(
        round_sampled=True, round_bs=bs, k=16, random=None,
        candidate_out=torch.zeros((bs, width - 1, 16), dtype=torch.int64),
        q_out=torch.ones((bs, width - 1, 16)) / 16,
    )
    outputs = object()
    worker = SimpleNamespace(
        block_size=width, _dspine_sampler=sampler,
        _next_accept_bonus_buffers=lambda size: outputs,
    )
    distribution = TargetDistribution(worker)
    info = SimpleNamespace(
        need_top_k_sampling=filtered,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    packed = object()
    monkeypatch.setitem(
        sys.modules,
        "sglang_dspine.sampling.random",
        SimpleNamespace(verification_uniforms=lambda *args: (
            torch.zeros(bs, width - 1), torch.zeros(bs)
        )),
    )
    with (
        patch.object(distribution, "dense", return_value=p),
        patch(
            "sglang_dspine.sampling.target.accept.compress_probs", return_value=packed
        ) as compress,
        patch("sglang_dspine.sampling.target.accept.accept_sparse") as reject,
    ):
        result = distribution.sample_prefix(
            candidates=torch.zeros((bs, width), dtype=torch.int64),
            next_token_logits=torch.zeros((bs * width, vocab)),
            sampling_info=info,
            prefix_lens=torch.zeros(bs, dtype=torch.int32),
            bs=bs,
            verify_lens=torch.full((bs,), width, dtype=torch.int32),
        )
    assert result is outputs
    assert reject.call_args.kwargs["packed"] is (packed if filtered else None)
    assert reject.call_args.kwargs["dense_p"].data_ptr() == p.data_ptr()
    assert compress.call_count == int(filtered)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_factorized_cache_replays_and_accepts_uncaptured_batch_sizes():
    from sglang_dspine.sampling.factorized.kernels import Workspace, factor_table
    from sglang_dspine.sampling.proposal import prepare

    torch.manual_seed(61)
    sampler = _sampler(33, "cuda")
    sampler.capture_batch_sizes = [1, 2, 4, 8, 16, 32]
    # Preparation must restore live draft state as well as RNG state.
    saved = sampler.raw.clone(), sampler.candidate_out.clone()
    cache = prepare(sampler)
    assert torch.equal(sampler.raw, saved[0])
    assert torch.equal(sampler.candidate_out, saved[1])
    for bs in (1, 9, 32, 17, 33, 2):
        sampler.raw.normal_(mean=0, std=0.1)
        n = bs * 15
        actual = cache.lookup(bs).clone()
        expected = torch.empty_like(actual)
        factor_table(
            sampler.raw[: bs * 16],
            sampler.rms[: bs * 16],
            sampler.query[: bs * 16],
            sampler.anchors[:bs],
            sampler.candidate_out[:bs].view(n, 16),
            cache.weights,
            sampler.norm_eps,
            16,
            expected,
            torch.empty((n, 16), device="cuda", dtype=torch.float64),
            Workspace.allocate(n, 16, 64, 8, "cuda"),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.eager_buffers.bs <= sampler.max_bs


@cuda
@pytest.mark.parametrize("bs", [1, 9, 17, 33])
@pytest.mark.parametrize("backend", ["flashinfer", "pytorch"])
@pytest.mark.parametrize("kind", ["unfiltered", "mixed", "joint"])
def test_generic_sparse_rejection_matches_dense_native_p_and_same_coins(
    bs, backend, kind
):
    from ref_utils import sampling_info
    from sglang_dspine.runtime.worker import build_dspine_target_probs
    from sglang_dspine.sampling.rejection import sample_capped_chain_from_probs

    torch.manual_seed(83)
    t, k, vocab = 16, 16, 97
    info = sampling_info(bs, temperature=0.7, device="cuda")
    if kind != "unfiltered":
        info.need_top_k_sampling = info.need_top_p_sampling = True
        info.temperatures = torch.linspace(0.4, 1.8, bs, device="cuda").view(bs, 1)
        info.top_ks = torch.tensor(
            [1, 7, 20, 1 << 30], device="cuda", dtype=torch.int32
        ).repeat(bs)[:bs]
        info.top_ps = torch.linspace(0.5, 1, bs, device="cuda")
        info.is_any_greedy = True
        if kind == "mixed":
            info.need_min_p_sampling = True
            info.min_ps = torch.linspace(0, 0.3, bs, device="cuda")
    logits = torch.randn((bs * t, vocab), device="cuda") * 2
    ids = torch.arange(k, device="cuda").expand(bs, t - 1, k).contiguous()
    q = torch.softmax(torch.randn((bs, t - 1, k), device="cuda"), -1)
    draws = torch.multinomial(q.reshape(-1, k), 1).view(bs, t - 1)
    candidates = torch.cat(
        (torch.zeros((bs, 1), device="cuda", dtype=torch.int64), draws), 1
    )
    caps = 1 + torch.arange(bs, device="cuda", dtype=torch.int32) % t
    prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 8190
    uniforms = torch.rand((bs, t - 1), device="cuda")
    final = torch.rand(bs, device="cuda")

    def buffers():
        return (
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int64),
            torch.empty((bs, t), device="cuda", dtype=torch.int64),
            torch.empty(bs, device="cuda", dtype=torch.int32),
        )

    actual, expected = buffers(), buffers()
    sampler = SimpleNamespace(
        round_sampled=True, round_bs=bs, k=k,
        candidate_out=ids, q_out=q, random=None,
    )
    worker = SimpleNamespace(
        block_size=t, _dspine_sampler=sampler,
        _next_accept_bonus_buffers=lambda size: actual,
    )
    distribution = TargetDistribution(worker)
    with (
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(kernel=SimpleNamespace(sampling_backend=backend)),
        ),
        patch(
            "sglang_dspine.sampling.random.verification_uniforms",
            return_value=(uniforms, final),
        ),
    ):
        assert distribution.sample_prefix(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=info,
            prefix_lens=prefix,
            bs=bs,
            verify_lens=caps,
        ) is actual
    p = build_dspine_target_probs(
        next_token_logits=logits, sampling_info=info,
        block_size=t, bs=bs, sampling_backend=backend,
    )
    dense_q = torch.zeros((bs, t - 1, vocab), device="cuda")
    dense_q.scatter_(-1, ids, q)
    sample_capped_chain_from_probs(
        candidates=candidates, target_probs=p,
        draft_probs=dense_q, verify_lens=caps, uniforms=uniforms,
        uniform_final=final, prefix_lens=prefix, outputs=expected,
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))


@cuda
@pytest.mark.parametrize("backend", ["flashinfer", "pytorch"])
def test_mixed_greedy_rows_keep_argmax_tie_breaking(backend):
    from ref_utils import ref_target_row, sampling_info
    from sglang_dspine.runtime.worker import build_dspine_target_probs

    torch.manual_seed(113)
    bs, width, vocab = 9, 16, 97
    info = sampling_info(
        bs, temperature=0.7, top_k=10, top_p=0.8, device="cuda"
    )
    info.is_any_greedy = True
    greedy = [0, 3, 8]
    info.top_ks[greedy] = 1
    logits = torch.randn((bs, width, vocab), device="cuda")
    logits[greedy] = 0  # Every token ties; native greedy picks the first ID.
    p = build_dspine_target_probs(
        next_token_logits=logits.reshape(-1, vocab),
        sampling_info=info,
        block_size=width,
        bs=bs,
        sampling_backend=backend,
    )
    onehot = torch.zeros_like(p[greedy])
    onehot[..., 0] = 1
    assert torch.equal(p[greedy], onehot)
    for b in (1, 2, 4, 5, 6, 7):
        for row in (0, width - 1):
            expected = ref_target_row(logits[b, row].cpu(), 0.7, 10, 0.8, 0, backend)
            assert torch.equal(p[b, row].cpu() > 0, expected > 0)
            torch.testing.assert_close(
                p[b, row].double().cpu(), expected, rtol=1e-5, atol=1e-7
            )
