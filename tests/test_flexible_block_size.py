"""Flexible served draft block sizes (2..16), including non-power-of-2 widths."""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton

from sglang_dspine.verification import caps, policy as _policy, sps
from sglang_dspine.verification.calibration import (
    FORMAT,
    load_calibration_models,
    load_calibration_tables,
)
from sglang_dspine.verification.policy import SpsPolicy, execution_identity
from sglang_dspine.verification.ragged import RaggedVerifyCaps, _accept

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _cpu_rejection_reference(p, candidates, qids, qvals, verify_lens, u, uf, prefix):
    width = len(candidates[0])
    answer = []
    offset = 0
    for b, length in enumerate(verify_lens):
        a = 0
        while a < length - 1:
            token = candidates[b][a + 1]
            q = sum(v for k, v in zip(qids[b][a], qvals[b][a]) if k == token)
            if q > 0 and u[b][a] * q < p[offset + a][token]:
                a += 1
            else:
                break
        row = p[offset + a]
        weights = list(row)
        if a != length - 1:
            q = dict(zip(qids[b][a], qvals[b][a]))
            weights = [max(v - q.get(k, 0), 0) for k, v in enumerate(row)]
            if sum(weights) <= 0:
                weights = list(row)
        threshold = uf[b] * sum(weights)
        cumulative = 0
        bonus = len(row) - 1
        for k, value in enumerate(weights):
            cumulative += value
            if cumulative > threshold:
                bonus = k
                break
        emitted = candidates[b][1 : a + 1] + [bonus] + [0] * (width - 1 - a)
        answer.append((a, a + 1, bonus, emitted, prefix[b] + a + 1))
        offset += length
    return answer


@pytest.mark.parametrize("block_size", [2, 6, 8, 12, 16])
def test_packaged_calibration_slices_causal_prefix(block_size):
    packaged = Path(sps.__file__).with_name("calibration.json")
    full = load_calibration_tables(packaged, gamma=15)
    gamma = block_size - 1
    models = load_calibration_models(packaged, gamma=gamma)
    for mode in ("greedy", "sampled"):
        cal = models[mode]
        assert cal.gamma == gamma
        assert cal.probabilities.shape == (gamma, full[mode].shape[1])
        torch.testing.assert_close(cal.probabilities, full[mode][:gamma])
        raw = torch.linspace(0.05, 0.95, 3 * gamma).view(3, gamma)
        out = cal(raw)
        assert out.shape == (3, gamma)


def test_execution_identity_tracks_non_default_block_size(tmp_path):
    assert execution_identity(2, 2, "fa3", "fa3") == {
        "tp_size": 2,
        "draft_tp_size": 2,
        "attention_backend": "fa3",
        "draft_attention_backend": "fa3",
    }
    assert execution_identity(2, 2, "fa3", "fa3", block_size=16) == execution_identity(
        2, 2, "fa3", "fa3"
    )
    exec8 = execution_identity(2, 2, "fa3", "fa3", block_size=8)
    assert exec8["block_size"] == 8

    path16 = tmp_path / "sps16.json"
    path16.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "execution": execution_identity(2, 2, "fa3", "fa3"),
                "policy": {"token_budgets": {"3": 24}},
            }
        )
    )
    with pytest.raises(ValueError, match="execution layout differs"):
        SpsPolicy.load(path16, {}, exec8)

    path8 = tmp_path / "sps8.json"
    path8.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "execution": exec8,
                "policy": {"token_budgets": {"3": 16}},
            }
        )
    )
    assert SpsPolicy.load(path8, {}, exec8).budget(3, full=8) == 16
    with pytest.raises(ValueError, match="execution layout differs"):
        SpsPolicy.load(path8, {}, execution_identity(2, 2, "fa3", "fa3"))

    # Legacy profiles without an execution block imply block_size=16 only.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "policy": {"token_budgets": {"3": 24}},
            }
        )
    )
    assert SpsPolicy.load(legacy, {}, execution_identity(1, 1)).budget(3) == 24
    with pytest.raises(ValueError, match="execution layout differs"):
        SpsPolicy.load(legacy, {}, execution_identity(1, 1, block_size=8))

    # Profiles cannot specify widths or token budgets beyond their block_size.
    oversized = tmp_path / "oversized8.json"
    oversized.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "execution": exec8,
                "policy": {"verify_widths": {"1": 12}},
            }
        )
    )
    with pytest.raises(ValueError, match="exceed draft block size 8"):
        SpsPolicy.load(oversized, {}, exec8)

    oversized_budget = tmp_path / "oversized_budget8.json"
    oversized_budget.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "execution": exec8,
                "policy": {"token_budgets": {"4": 36}},
            }
        )
    )
    with pytest.raises(ValueError, match="exceed draft block size 8"):
        SpsPolicy.load(oversized_budget, {}, exec8)


@pytest.mark.parametrize("block_size", [2, 6, 8, 12])
def test_caps_and_ragged_support_valid_small_block_profiles(
    monkeypatch, tmp_path, block_size
):
    path = tmp_path / "sps.json"
    narrow = max(1, block_size // 2)
    path.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": {},
                "execution": execution_identity(1, 1, "fa3", "fa3", block_size=block_size),
                "policy": {
                    "verify_widths": {"1": narrow},
                    "token_budgets": {"3": 3 * block_size - 1},
                },
            }
        )
    )
    captured = []

    class Graph:
        def __init__(self, runner=None, **kwargs):
            if runner is not None:
                captured.append(kwargs["speculative_num_draft_tokens"])
            self.capture_num_tokens = [block_size * bs for bs in range(1, 5)]

        def _ragged_capture_slots(self, tier):
            return min(tier, 32)

    backend = SimpleNamespace(needs_cpu_seq_lens=False, supports_ragged_verify_graph=True)
    runner = SimpleNamespace(
        attn_backend=backend,
        decode_cuda_graph_runner=Graph(),
        init_new_workspace=None,
        _get_attention_backend=lambda **kwargs: backend,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_config",
        SimpleNamespace(Backend=SimpleNamespace(DISABLED="disabled")),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        SimpleNamespace(DecodeCudaGraphRunner=Graph),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_spec=lambda: SimpleNamespace(speculative_dspark_sps_table_path=path),
            get_parallel=lambda: SimpleNamespace(attn_dp_enabled=False, pp_size=1),
            get_exec=lambda: SimpleNamespace(
                kernel=SimpleNamespace(attention_backend="fa3"),
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend="full")
                    )
                ),
            ),
        ),
    )
    monkeypatch.setattr(caps, "spec_width", lambda width: contextlib.nullcontext())
    monkeypatch.setattr(_policy, "hardware_identity", lambda device: {})
    worker = SimpleNamespace(
        block_size=block_size,
        device="cpu",
        _dspine_tp=SimpleNamespace(group=SimpleNamespace(world_size=1)),
        draft_model_runner=SimpleNamespace(draft_attention_backend="fa3"),
        target_worker=SimpleNamespace(model_runner=runner),
    )
    control = RaggedVerifyCaps(worker)
    assert control.widths == sorted({narrow, block_size})
    assert control.ragged_backend is not None
    assert all(w <= block_size for w in captured)


def test_tuner_cli_supports_non16_block_size(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/tune_sps.py"
    spec = importlib.util.spec_from_file_location("dspine_flex_tuner_test", source)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)

    def write_rows(name, tasks):
        path = tmp_path / name
        rows = [
            {
                "task_id": task,
                "input_ids": [task],
                "sampling_params": {"temperature": 0},
            }
            for task in tasks
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    training = write_rows("tune.jsonl", range(1, 7))
    holdout = write_rows("holdout.jsonl", range(20, 23))
    destination = tmp_path / "sps8.json"
    expected_exec = execution_identity(1, 1, "fa3", "fa3", block_size=8)

    @contextlib.contextmanager
    def server(_command, profile, *_args):
        assert json.loads(profile.read_text())["execution"] == expected_exec
        yield "http://cpu-fixture"

    def request(*_args, **_kwargs):
        return {
            "speculative_algorithm": "DSPINE",
            "speculative_num_draft_tokens": 8,
            "tp_size": 1,
            "attention_backend": "fa3",
            "speculative_draft_attention_backend": "fa3",
            "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 3, 4]}},
        }

    def trial(_command, document, _rows, bs, repeat, count, *_args, **_kwargs):
        budget = tuner.current_budget(document, "greedy", bs, block=8)
        rate = 120.0 if budget == 16 else 100.0
        return {
            "status": "PASS",
            "repeat": repeat,
            "requests": count,
            "wall_seconds": 2.0,
            "output_tokens": int(2 * rate),
            "tokens_per_second": rate,
            "verify_calls": count,
            "jobs_sha256": f"{repeat}-{count}",
        }

    monkeypatch.setattr(tuner, "hardware_identity", lambda _device: {})
    monkeypatch.setattr(tuner, "server", server)
    monkeypatch.setattr(tuner, "request", request)
    monkeypatch.setattr(tuner, "run_profile_trials", trial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--requests",
            str(training),
            "--holdout-requests",
            str(holdout),
            "--output",
            str(destination),
            "--batch-sizes",
            "3",
            "--repeats",
            "2",
            "--min-seconds",
            "1",
            "--min-requests",
            "4",
            "--",
            "sglang",
            "serve",
            "--speculative-algorithm",
            "DSPINE",
            "--speculative-num-draft-tokens",
            "8",
            "--attention-backend",
            "fa3",
            "--speculative-draft-attention-backend",
            "fa3",
        ],
    )
    tuner.main()
    result = json.loads(destination.read_text())
    assert result["execution"] == expected_exec
    assert result["tuning"]["selection_status"] == "VALIDATED"
    assert tuner.current_budget(result, "greedy", 3, block=8) == 16


@cuda
@pytest.mark.parametrize("block_size", [2, 6, 8, 12, 16])
def test_fp8_padding_mask_uses_served_gamma(block_size):
    from sglang_dspine.layers.fp8 import padding

    gamma = block_size - 1
    graph_bs, actual_bs, hidden = 4, 2, 64
    rows = torch.arange(
        graph_bs * gamma * hidden, device="cuda", dtype=torch.bfloat16
    ).view(graph_bs * gamma, hidden) + 1
    out = torch.full_like(rows, -7.0)
    bs_tensor = torch.tensor([actual_bs], device="cuda", dtype=torch.int32)
    padding.mask(rows, bs_tensor, out, gamma=gamma)
    torch.testing.assert_close(out[: actual_bs * gamma], rows[: actual_bs * gamma])
    assert torch.count_nonzero(out[actual_bs * gamma :]).item() == 0


@cuda
@pytest.mark.parametrize("block_size", [2, 6, 8, 12, 16])
def test_packed_conv_and_dense_conv_input_support_flexible_steps(block_size):
    from sglang_dspine.layers.convolution import packed_conv
    from sglang_dspine.layers.gdn import dense_conv_input

    torch.manual_seed(17 + block_size)
    bs, dim = 3, 32
    lens = [min(block_size, i + 1) for i in range(bs)]
    lens[-1] = block_size
    indptr = torch.tensor(
        [0, lens[0], lens[0] + lens[1], sum(lens)], device="cuda", dtype=torch.int32
    )
    total = int(indptr[-1])
    x = torch.randn((total, dim), device="cuda", dtype=torch.bfloat16)
    state = torch.randn((bs, dim, 3), device="cuda", dtype=torch.bfloat16)
    state_ref = state.clone()
    weight = torch.randn((dim, 4), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(bs, device="cuda", dtype=torch.int32)
    windows = torch.zeros((bs, block_size, dim, 3), device="cuda", dtype=torch.bfloat16)

    out = packed_conv(
        x, indptr, state, weight, bias, "silu", slots, slots, windows, steps=block_size
    )

    # Pure PyTorch reference
    expected_rows = []
    for req in range(bs):
        c0 = state_ref[req, :, 0].float()
        c1 = state_ref[req, :, 1].float()
        c2 = state_ref[req, :, 2].float()
        start, end = int(indptr[req]), int(indptr[req + 1])
        for t in range(block_size):
            xt = (
                x[start + t].float()
                if t < end - start
                else torch.zeros(dim, device="cuda", dtype=torch.float32)
            )
            acc = (
                bias.float()
                + c0 * weight[:, 0].float()
                + c1 * weight[:, 1].float()
                + c2 * weight[:, 2].float()
                + xt * weight[:, 3].float()
            )
            c0, c1, c2 = c1, c2, xt
            if t < end - start:
                expected_rows.append(torch.nn.functional.silu(acc).bfloat16())
                torch.testing.assert_close(windows[req, t, :, 0], c0.bfloat16())
                torch.testing.assert_close(windows[req, t, :, 1], c1.bfloat16())
                torch.testing.assert_close(windows[req, t, :, 2], c2.bfloat16())
        torch.testing.assert_close(state[req, :, 0], c0.bfloat16())
        torch.testing.assert_close(state[req, :, 1], c1.bfloat16())
        torch.testing.assert_close(state[req, :, 2], c2.bfloat16())
    torch.testing.assert_close(out, torch.stack(expected_rows), rtol=2e-2, atol=2e-2)

    dense = dense_conv_input(x, indptr, block_size)
    assert dense.shape == (bs, dim, block_size)
    for req in range(bs):
        start, end = int(indptr[req]), int(indptr[req + 1])
        torch.testing.assert_close(dense[req, :, : end - start].T, x[start:end])


@cuda
@pytest.mark.parametrize("block_size", [2, 6, 8, 12, 16])
def test_sparse_and_capped_rejection_and_ragged_greedy_accept_flexible_block_sizes(
    block_size,
):
    from sglang_dspine.sampling.rejection import sample_capped_chain_from_probs
    from sglang_dspine.sampling.sparse.accept import accept_sparse, compress_probs

    torch.manual_seed(91 + block_size)
    bs, vocab, gamma = 4, 128, block_size - 1
    verify_lens = (torch.arange(bs, device="cuda", dtype=torch.int32) % block_size) + 1
    verify_lens[0] = block_size
    offsets = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int64), verify_lens.long().cumsum(0))
    )[:-1]

    # Greedy ragged accept kernel (tests non-power-of-2 block_size)
    candidates = torch.randint(0, vocab, (bs, block_size), device="cuda", dtype=torch.int64)
    top1 = candidates.clone()
    top1[:, :-1] = candidates[:, 1:]  # force matches up to cap-1 on row 0
    top1[1, 0] = (candidates[1, 1] + 1) % vocab
    prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 10
    g_accept = torch.empty(bs, device="cuda", dtype=torch.int32)
    g_commit = torch.empty(bs, device="cuda", dtype=torch.int32)
    g_bonus = torch.empty(bs, device="cuda", dtype=torch.int64)
    g_output = torch.full((bs, block_size), -1, device="cuda", dtype=torch.int64)
    g_new_seq = torch.empty(bs, device="cuda", dtype=torch.int32)
    _accept[bs,](
        candidates,
        top1,
        verify_lens,
        g_accept,
        g_commit,
        g_bonus,
        g_output,
        prefix,
        g_new_seq,
        T=block_size,
        WIDTH=triton.next_power_of_2(block_size),
    )
    assert g_accept[0].item() == block_size - 1
    assert g_accept[1].item() == 0
    assert g_bonus[1].item() == top1[1, 0].item()
    assert g_output[1, 0].item() == top1[1, 0].item()

    # Sampled sparse + dense capped rejection
    qids = (
        torch.arange(16, device="cuda", dtype=torch.int32)
        .expand(bs, gamma, 16)
        .contiguous()
    )
    qvals = torch.softmax(torch.randn((bs, gamma, 16), device="cuda"), -1)
    candidates[:, 1:] = torch.multinomial(qvals.reshape(-1, 16), 1).reshape(bs, gamma)
    u = torch.full((bs, gamma), 0.371, device="cuda")
    u[0] = 0.0
    uf = torch.linspace(0.11, 0.89, bs, device="cuda")

    dense_p = torch.zeros((bs, block_size, vocab), device="cuda")
    support = torch.tensor([0, 2, 5, 9, 13, 15], device="cuda")
    dense_p[:, :, support] = torch.softmax(
        torch.randn((bs, block_size, len(support)), device="cuda"), -1
    )
    compact_p = torch.cat(
        [dense_p[b, : int(verify_lens[b])] for b in range(bs)], dim=0
    ).contiguous()
    packed = compress_probs(torch, compact_p, capacity=32)

    expected = _cpu_rejection_reference(
        compact_p.cpu().tolist(),
        candidates.cpu().tolist(),
        qids.cpu().tolist(),
        qvals.cpu().tolist(),
        verify_lens.cpu().tolist(),
        u.cpu().tolist(),
        uf.cpu().tolist(),
        prefix.cpu().tolist(),
    )

    def alloc():
        return (
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int64),
            torch.empty((bs, block_size), device="cuda", dtype=torch.int64),
            torch.empty(bs, device="cuda", dtype=torch.int32),
        )

    sparse_out = alloc()
    accept_sparse(
        torch,
        candidates=candidates,
        dense_p=compact_p,
        packed=packed,
        q_ids=qids,
        q_values=qvals,
        verify_lens=verify_lens,
        uniforms=u,
        uniform_final=uf,
        prefix_lens=prefix,
        outputs=sparse_out,
        row_offsets=offsets,
        vocab_size=vocab,
    )
    actual_sparse = [x.cpu().tolist() for x in sparse_out]
    for b, row in enumerate(expected):
        assert tuple(col[b] for col in actual_sparse) == row

    dense_q = torch.zeros((bs, gamma, vocab), device="cuda", dtype=torch.float32)
    dense_q.scatter_(-1, qids.long(), qvals)
    capped_out = alloc()
    sample_capped_chain_from_probs(
        candidates=candidates,
        target_probs=dense_p,
        draft_probs=dense_q,
        verify_lens=verify_lens,
        uniforms=u,
        uniform_final=uf,
        prefix_lens=prefix,
        outputs=capped_out,
    )
    actual_capped = [x.cpu().tolist() for x in capped_out]
    for b, row in enumerate(expected):
        assert tuple(col[b] for col in actual_capped) == row


@cuda
@pytest.mark.parametrize("block_size", [2, 6, 8, 12])
def test_sps_device_plan_and_target_distribution_support_flexible_block_sizes(
    monkeypatch, block_size
):
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )

    from sglang_dspine.sampling.target import TargetDistribution
    from sglang_dspine.verification.sps import device_plan

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_exec=lambda: SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="flashinfer")
            )
        ),
    )

    gamma = block_size - 1
    bs = 3
    packaged = Path(sps.__file__).with_name("calibration.json")
    models = load_calibration_models(packaged, gamma=gamma)
    cal = models["greedy"].to("cuda")
    raw = torch.linspace(0.1, 0.9, bs * gamma, device="cuda").view(bs, gamma)
    depth = torch.arange(gamma, device="cuda")
    cfg = DSparkScheduleConfig(
        gamma=gamma, min_verify_len=1, max_verify_len=block_size, survival_eps=0.0
    )
    cfg.validate()
    budget = max(bs, bs * block_size - 1)
    layout, index = device_plan(
        raw, cal.probabilities, depth, budget, cfg, budget, cal.boundaries
    )
    lens = layout.verify_lens.cpu().tolist()
    assert sum(lens) == budget
    assert all(1 <= n <= block_size for n in lens)
    assert index.cpu().tolist() == [
        b * block_size + i for b, n in enumerate(lens) for i in range(n)
    ]

    # TargetDistribution.sample_prefix should stay on the sparse fast path (not return None)
    vocab = 128
    qids = (
        torch.arange(16, device="cuda", dtype=torch.int64)
        .expand(bs, gamma, 16)
        .contiguous()
    )
    qvals = torch.full((bs, gamma, 16), 1.0 / 16.0, device="cuda", dtype=torch.float32)
    sampler = SimpleNamespace(
        round_sampled=True,
        round_bs=bs,
        k=16,
        random=None,
        candidate_out=qids,
        q_out=qvals,
    )
    worker = SimpleNamespace(
        block_size=block_size,
        _dspine_sampler=sampler,
        _next_accept_bonus_buffers=lambda size: (
            torch.empty(size, device="cuda", dtype=torch.int32),
            torch.empty(size, device="cuda", dtype=torch.int32),
            torch.empty(size, device="cuda", dtype=torch.int64),
            torch.empty((size, block_size), device="cuda", dtype=torch.int64),
            torch.empty(size, device="cuda", dtype=torch.int32),
        ),
    )
    dist = TargetDistribution(worker)
    logits = torch.randn((bs * block_size, vocab), device="cuda", dtype=torch.bfloat16)
    candidates = torch.zeros((bs, block_size), device="cuda", dtype=torch.int64)
    candidates[:, 1:] = 1
    info = SimpleNamespace(
        temperatures=torch.ones(bs, device="cuda", dtype=torch.float32),
        top_ks=torch.full((bs,), 20, device="cuda", dtype=torch.int32),
        top_ps=torch.full((bs,), 0.95, device="cuda", dtype=torch.float32),
        min_ps=torch.zeros(bs, device="cuda", dtype=torch.float32),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=False,
        is_any_greedy=False,
        max_top_k=20,
    )
    out = dist.sample_prefix(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=info,
        prefix_lens=torch.zeros(bs, device="cuda", dtype=torch.int32),
        bs=bs,
        verify_lens=torch.full((bs,), block_size, device="cuda", dtype=torch.int32),
    )
    assert out is not None
    assert out[3].shape == (bs, block_size)


@pytest.mark.parametrize(
    "block_size,expected",
    [
        (2, [1, 2]),
        (3, [1, 3]),
        (4, [2, 4]),
        (6, [4, 6]),
        (8, [4, 8]),
        (12, [4, 8, 12]),
        (16, [4, 8, 12, 16]),
    ],
)
def test_tuner_default_widths_cover_narrow_candidates_for_all_block_sizes(
    block_size, expected
):
    source = Path(__file__).resolve().parents[1] / "scripts/tune_sps.py"
    spec = importlib.util.spec_from_file_location("dspine_flex_tuner_widths", source)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    assert tuner.default_widths(block_size) == expected


@pytest.mark.parametrize(
    "block_size,ring,expected",
    [(2, 16, False), (6, 16, False), (7, 16, False), (8, 16, True), (16, 16, True)],
)
def test_replayssm_empty_history_hook_only_activates_when_wp0_invariant_holds(
    monkeypatch, block_size, ring, expected
):
    from sglang_dspine.runtime import worker as worker_module
    from sglang_dspine.runtime.replayssm.hooks import _pool

    monkeypatch.setattr(worker_module, "_active_worker", None)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_spec=lambda: SimpleNamespace(speculative_num_draft_tokens=block_size)
        ),
    )
    mamba_pool = SimpleNamespace(
        enable_linear_replayssm_spec=True,
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
    )
    d_cache = SimpleNamespace(shape=(4, 6, ring, 64))
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool,
        get_speculative_mamba2_params_all_layers=lambda: SimpleNamespace(
            replayssm_d=d_cache
        ),
    )
    # Raw server_args may stay None when CLI omitted --speculative-num-draft-tokens
    # and SGLang resolved the default into runtime_context.get_spec().
    runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm="DSPINE",
        req_to_token_pool=req_pool,
        server_args=SimpleNamespace(speculative_num_draft_tokens=None),
    )
    assert (_pool(runner) is mamba_pool) is expected


def test_replayssm_hook_uses_active_worker_block_size_under_spec_width_override(
    monkeypatch,
):
    from sglang_dspine.runtime import worker as worker_module
    from sglang_dspine.runtime.replayssm.hooks import _pool

    # Served block is 8 (WP0 holds), while VerifyCaps temporarily overrides
    # get_spec().speculative_num_draft_tokens to 4 during narrow-width capture.
    monkeypatch.setattr(
        worker_module, "_active_worker", SimpleNamespace(block_size=8)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_spec=lambda: SimpleNamespace(speculative_num_draft_tokens=4)
        ),
    )
    mamba_pool = SimpleNamespace(
        enable_linear_replayssm_spec=True,
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
    )
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool,
        get_speculative_mamba2_params_all_layers=lambda: SimpleNamespace(
            replayssm_d=SimpleNamespace(shape=(4, 6, 16, 64))
        ),
    )
    runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm="DSPINE",
        req_to_token_pool=req_pool,
        server_args=SimpleNamespace(speculative_num_draft_tokens=None),
    )
    assert _pool(runner) is mamba_pool


