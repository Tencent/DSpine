"""CUDA validation on small and serving shapes (vocab 151936, K = T = 16).
Compare independent references and bf16 bridge rounding; test Monte Carlo sampled
verification and CUDA graph replay with re-staged sampling parameters."""

import math
import os
import types
from unittest import mock

import pytest
import torch
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

from sglang_dspine import kernels as K
from sglang_dspine.graph_runner import DSpineDraftCudaGraphRunner
from sglang_dspine.models.dspine import DSpineCodeBridge
from sglang_dspine.worker import (
    DSpineDraftSampler,
    build_dspine_target_probs,
    dspine_accept_sampling,
)

from ref_utils import (
    assert_follows,
    draft_candidates_overlapping,
    ref_refine_table,
    ref_sample_chain,
    ref_target_row,
    sampling_info,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpine Triton kernels require CUDA"
)

DEV = "cuda"


# ---------------------------------------------------------------------------
# code bridge
# ---------------------------------------------------------------------------


def _bridge(implementation, seed=0, H=512, C=64, R=64, T=4, L=2, V=300):
    cfg = types.SimpleNamespace(
        hidden_size=H,
        num_hidden_layers=L,
        block_size=T,
        vocab_size=V,
        dflash_config={
            "selected_prefix_terminal": False,
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "draft_block_causal": True,
            "selected_prefix_feature_layers": list(range(1, L + 1)),
            "selected_prefix_code_bridge": {"code_size": C, "rank": R, "eps": 1e-6},
        },
    )
    with (
        mock.patch.dict(os.environ, {"SGLANG_DSPINE_BRIDGE_IMPL": implementation}),
        torch.device(DEV),
    ):
        bridge = DSpineCodeBridge(cfg).to(torch.bfloat16)
    assert bridge.implementation == implementation
    g = torch.Generator().manual_seed(seed)
    for name, t in list(bridge.named_parameters()) + list(bridge.named_buffers()):
        if name == "packed_weight":
            continue
        if name.endswith("_version"):
            value = torch.ones((), dtype=torch.int64)
        elif name == "out.weight":
            value = torch.zeros(t.shape)
        else:
            value = torch.randn(t.shape, generator=g) / math.sqrt(t.shape[-1])
        bridge.load_weight(name, value.to(t.dtype).to(DEV))
    bridge.prepare()
    return bridge


def test_bridge_fused_matches_eager():
    H, T, V = 512, 4, 300
    fused, eager = _bridge("fused"), _bridge("eager")
    g = torch.Generator().manual_seed(1)
    m = 4 * T
    hidden = torch.randn(m, H, generator=g).to(torch.bfloat16).to(DEV)
    residual = torch.randn(m, H, generator=g).to(torch.bfloat16).to(DEV)
    input_ids = torch.randint(0, V, (m,), generator=g).to(DEV)
    norm = types.SimpleNamespace(
        weight=(1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16).to(DEV),
        variance_epsilon=1e-6,
    )
    for layer in (0, 1):
        sf, se = {}, {}
        nf, of = fused(hidden, residual, input_ids, layer, norm, sf)
        ne, oe = eager(hidden, residual, input_ids, layer, norm, se)
        assert torch.equal(
            sf["raw"], se["raw"]
        )  # both round (hidden + residual) once, RNE
        torch.testing.assert_close(sf["rms"], se["rms"], rtol=1e-5, atol=1e-6)
        for a, b in ((sf["packed"], se["packed"]), (of, oe), (nf, ne)):
            diff = (a.float() - b.float()).abs()
            # reduction order may move a bf16 intermediate by one step on rare elements
            assert float((diff > 2e-2).float().mean()) < 2e-3, float(diff.max())
            assert float(diff.mean()) < 1e-3, float(diff.mean())
        assert torch.equal(
            of.view(-1, T, H)[:, 0], (hidden + residual).view(-1, T, H)[:, 0]
        )


# ---------------------------------------------------------------------------
# candidate top-k
# ---------------------------------------------------------------------------


def test_topk_ids_matches_torch_topk():
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(5, 2048, generator=g).to(dtype).to(DEV)
        ids = K.topk_ids(logits, 16)
        ref = torch.topk(logits.float(), 16, dim=-1)
        assert torch.equal(ids, ids.sort(-1).values), "ids must be ascending"
        # The contract is the value multiset; bf16 ties at the k-th place may pick other ids.
        picked = torch.gather(logits.float(), 1, ids).sort(-1).values
        assert torch.equal(picked, ref.values.sort(-1).values)
        if dtype == torch.float32:  # no ties: the ids themselves must agree
            assert torch.equal(ids, ref.indices.sort(-1).values)


def test_topk_ids_full_vocab():
    g = torch.Generator(device=DEV).manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(240, 151936, generator=g, device=DEV).to(dtype)
        ids = K.topk_ids(logits, 16)
        ref = torch.topk(logits.float(), 16, dim=-1)
        picked = torch.gather(logits.float(), 1, ids).sort(-1).values
        assert torch.equal(picked, ref.values.sort(-1).values)
        assert torch.equal(ids, ids.sort(-1).values)


# ---------------------------------------------------------------------------
# sampled serial chain
# ---------------------------------------------------------------------------


def _chain_inputs(bs, T, k, seed):
    g = torch.Generator().manual_seed(seed)
    n = bs * (T - 1)
    table = torch.randn(n, k, k, generator=g) * 3.0
    cand = torch.stack(
        [torch.randperm(1000, generator=g)[:k].sort().values for _ in range(n)]
    )
    greedy_tok = cand[torch.arange(n), torch.randint(0, k, (n,), generator=g)]
    u = torch.rand(n, generator=g)
    return (
        table.contiguous(),
        cand.contiguous(),
        greedy_tok.contiguous(),
        u.contiguous(),
    )


def _run_sample_chain(
    table, inv_t, top_k, top_p, min_p, greedy, u, cand, greedy_tok, T
):
    """The kernel on the GPU from CPU inputs; returns (out_tok, out_q) on the CPU."""
    n, k = cand.shape
    out_tok = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    out_q = torch.full((n, k), -1.0, dtype=torch.float32, device=DEV)
    staged = (table, inv_t, top_k, top_p, min_p, greedy, u, cand, greedy_tok)
    K.serial_sample_chain(*[x.to(DEV) for x in staged], out_tok, out_q, T)
    return out_tok.cpu(), out_q.cpu()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_chain_matches_reference(seed):
    T, k = 5, 8
    params = [  # (temperature, top_k, top_p, min_p, greedy)
        (1.0, k, 1.0, 0.0, 0),
        (0.7, 3, 0.8, 0.1, 0),
        (1.5, k, 0.5, 0.0, 0),
        (1.0, k, 1.0, 0.3, 0),
        (0.3, 1, 1.0, 0.0, 0),
        (1.0, 1, 1.0, 0.0, 1),
    ]
    bs = len(params)
    table, cand, greedy_tok, u = _chain_inputs(bs, T, k, seed)
    staged = (
        torch.tensor([1.0 / p[0] for p in params], dtype=torch.float32),
        torch.tensor([p[1] for p in params], dtype=torch.int32),
        torch.tensor([p[2] for p in params], dtype=torch.float32),
        torch.tensor([p[3] for p in params], dtype=torch.float32),
        torch.tensor([p[4] for p in params], dtype=torch.int32),
    )
    n = bs * (T - 1)
    out_tok, out_q = _run_sample_chain(table, *staged, u, cand, greedy_tok, T)
    ref_tok, ref_q = ref_sample_chain(table, *staged, u, cand, greedy_tok, T)
    assert torch.equal(out_tok, ref_tok)
    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_q.sum(-1), torch.ones(n), rtol=0, atol=1e-5)
    # the drawn token always has positive mass under the q handed to verification
    lane = (cand == out_tok[:, None]).float().argmax(-1)
    assert bool((out_q[torch.arange(n), lane] > 0).all())
    # top_k = 1 without the greedy flag is deterministic: q is one-hot.  One lane is nonzero;
    # its mass is w / w, which Triton's approximate fp32 division on the GPU rounds to within
    # 2 ulp of 1 (2 ulp just above 1.0 = 2 * 2**-23 ~ 2.4e-7).
    row4 = slice(4 * (T - 1), 5 * (T - 1))
    assert ((out_q[row4] != 0).sum(-1) == 1).all()
    torch.testing.assert_close(
        out_q[row4].max(-1).values, torch.ones(T - 1), rtol=0, atol=2.5e-7
    )


def test_sample_chain_ties_break_to_lower_lane():
    T, k = 3, 8
    n = T - 1
    table = torch.zeros(n, k, k)
    table[:, :, 2] = 5.0
    table[:, :, 6] = 5.0  # exact tie between lanes 2 and 6
    cand = torch.arange(k, dtype=torch.int64).repeat(n, 1) + 100
    greedy_tok = cand[:, 0].clone()
    staged = (
        torch.ones(1),
        torch.tensor([1], dtype=torch.int32),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int32),
    )
    out_tok, out_q = _run_sample_chain(
        table, *staged, torch.zeros(n), cand, greedy_tok, T
    )
    assert out_tok.tolist() == [102, 102]
    assert out_q[:, 2].tolist() == [1.0, 1.0]


def test_sample_chain_matches_reference_realistic():
    T, k, bs = 16, 16, 8
    n = bs * (T - 1)
    g = torch.Generator().manual_seed(3)
    table = (torch.randn(n, k, k, generator=g) * 3).contiguous()
    cand = torch.stack(
        [torch.randperm(151936, generator=g)[:k].sort().values for _ in range(n)]
    )
    greedy_tok = cand[torch.arange(n), torch.randint(0, k, (n,), generator=g)]
    u = torch.rand(n, generator=g)
    temps = [1.0, 0.6, 1.3, 1.0, 0.8, 1.0, 2.0, 1.0]
    top_ks = [k, 4, k, 2, 16, 1, 8, 1]
    top_ps = [1.0, 0.9, 0.7, 1.0, 0.95, 1.0, 0.5, 1.0]
    min_ps = [0.0, 0.05, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0]
    greedy = [0, 0, 0, 0, 0, 0, 0, 1]
    staged = (
        torch.tensor([1 / t for t in temps]),
        torch.tensor(top_ks, dtype=torch.int32),
        torch.tensor(top_ps),
        torch.tensor(min_ps),
        torch.tensor(greedy, dtype=torch.int32),
    )
    out_tok, out_q = _run_sample_chain(table, *staged, u, cand, greedy_tok, T)
    ref_tok, ref_q = ref_sample_chain(table, *staged, u, cand, greedy_tok, T)
    assert torch.equal(out_tok, ref_tok)
    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# serial refinement tables
# ---------------------------------------------------------------------------


def _refine_inputs(seed, bs=2, T=4, H=512, R=128, C=64, V=1024, k=8):
    g = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    m, n = bs * T, bs * (T - 1)
    d = dict(
        raw=(torch.randn(m, H, generator=g)).to(bf),
        rms=(torch.rand(m, generator=g) + 0.5).float(),
        packed=(torch.randn(m, 3 * R, generator=g)).to(bf),
        codebook=(torch.randn(V, C, generator=g) / math.sqrt(C)).to(bf),
        pg=(torch.randn(R, C, generator=g) / math.sqrt(C)).to(bf),
        pv=(torch.randn(R, C, generator=g) / math.sqrt(C)).to(bf),
        gb=(torch.randn(R, generator=g) * 0.1).to(bf),
        w_last=(torch.randn(H, R, generator=g) / math.sqrt(R)).to(bf),
        norm_w=(1.0 + 0.1 * torch.randn(H, generator=g)).to(bf),
        lm=(torch.randn(V, H, generator=g) / math.sqrt(H)).to(bf),
    )
    d["cand"] = torch.stack(
        [torch.randperm(V, generator=g)[:k].sort().values for _ in range(n)]
    ).contiguous()
    input_ids = torch.randint(0, V, (m,), generator=g)
    d["input_ids"] = input_ids.contiguous()
    d["anchors"] = input_ids.view(bs, T)[:, 0].contiguous()
    gpu = {name: t.to(DEV) for name, t in d.items()}
    d.update(bs=bs, T=T, H=H, R=R, C=C, V=V, k=k, eps=1e-6)
    return d, gpu


def _ref_table(d):
    return ref_refine_table(
        d["raw"],
        d["rms"],
        d["packed"][:, : d["R"]],
        d["anchors"],
        d["cand"],
        d["codebook"],
        d["pg"],
        d["pv"],
        d["gb"],
        d["w_last"],
        d["norm_w"],
        d["lm"],
        d["eps"],
        d["T"],
    )


def _assert_greedy_chain_is_reference_argmax(out, cand, ref, bs, T):
    for b in range(bs):
        j = 0
        for i in range(T - 1):
            nn = b * (T - 1) + i
            lane = int((cand[nn] == out[nn]).nonzero().item())
            row = ref[nn, j]
            # bf16 vocabulary tables: the pick must be a reference argmax up to rounding
            assert float(row[lane]) >= float(row.max()) - 2e-2 * float(
                row.abs().max()
            ), (
                b,
                i,
            )
            j = lane


@pytest.mark.parametrize("seed", [0, 1])
def test_rs_table_matches_fp64_reference(seed):
    d, g = _refine_inputs(seed)
    f32 = torch.float32
    table = K.serial_rs_table(
        g["raw"],
        g["rms"],
        g["packed"],
        g["anchors"],
        g["cand"],
        g["codebook"],
        g["pg"].to(f32),
        g["pv"].to(f32),
        g["gb"].to(f32),
        g["w_last"].to(f32),
        g["norm_w"].to(f32),
        g["lm"],
        d["eps"],
        d["T"],
        chunk_rows=2,
    ).cpu()
    torch.testing.assert_close(table.double(), _ref_table(d), rtol=1e-4, atol=1e-4)
    # block row 1: every sender option is the anchor, so all K table rows agree
    for b in range(d["bs"]):
        first = table[b * (d["T"] - 1)]
        torch.testing.assert_close(first, first[:1].expand_as(first))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_chain_follows_reference_argmax(seed):
    d, g = _refine_inputs(seed)
    T = d["T"]
    tg, tsv = K.build_serial_tables(
        g["lm"],
        g["norm_w"],
        g["w_last"],
        torch.cat((g["pg"], g["pv"]), 0),
        g["codebook"],
    )
    idx = K.serial_table(
        g["raw"],
        g["norm_w"],
        g["cand"],
        g["input_ids"],
        g["lm"],
        tg,
        tsv,
        g["packed"],
        g["rms"],
        g["gb"],
        T,
    )
    out = torch.full((d["bs"] * (T - 1),), -1, dtype=torch.int64, device=DEV)
    K.serial_chain(idx, g["cand"], out, T)
    _assert_greedy_chain_is_reference_argmax(
        out.cpu(), d["cand"], _ref_table(d), d["bs"], T
    )


def test_refine_tables_realistic_shapes():
    bs, T, H, R, C, V, k = 2, 16, 4096, 512, 1024, 4096, 16
    m, n = bs * T, bs * (T - 1)
    g = torch.Generator().manual_seed(5)
    bf, f32 = torch.bfloat16, torch.float32

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(bf).to(DEV)

    raw, packed = rnd(m, H), rnd(m, 3 * R)
    rms = (torch.rand(m, generator=g) + 0.5).to(DEV)
    codebook = rnd(V, C, scale=1 / math.sqrt(C))
    pg, pv = rnd(R, C, scale=1 / math.sqrt(C)), rnd(R, C, scale=1 / math.sqrt(C))
    gb, w_last = rnd(R, scale=0.1), rnd(H, R, scale=1 / math.sqrt(R))
    norm_w = (1 + 0.1 * torch.randn(H, generator=g)).to(bf).to(DEV)
    lm = rnd(V, H, scale=1 / math.sqrt(H))
    cand = torch.stack(
        [torch.randperm(V, generator=g)[:k].sort().values for _ in range(n)]
    ).to(DEV)
    input_ids = torch.randint(0, V, (m,), generator=g).to(DEV)
    anchors = input_ids.view(bs, T)[:, 0].contiguous()
    table = K.serial_rs_table(
        raw,
        rms,
        packed,
        anchors,
        cand,
        codebook,
        pg.to(f32),
        pv.to(f32),
        gb.to(f32),
        w_last.to(f32),
        norm_w.to(f32),
        lm,
        1e-6,
        T,
        chunk_rows=7,
    )
    ref = ref_refine_table(
        raw.cpu(),
        rms.cpu(),
        packed[:, :R].cpu(),
        anchors.cpu(),
        cand.cpu(),
        codebook.cpu(),
        pg.cpu(),
        pv.cpu(),
        gb.cpu(),
        w_last.cpu(),
        norm_w.cpu(),
        lm.cpu(),
        1e-6,
        T,
    )
    torch.testing.assert_close(table.double().cpu(), ref, rtol=1e-4, atol=1e-4)
    tg, tsv = K.build_serial_tables(
        lm, norm_w, w_last, torch.cat((pg, pv), 0), codebook
    )
    idx = K.serial_table(raw, norm_w, cand, input_ids, lm, tg, tsv, packed, rms, gb, T)
    out = torch.empty(n, dtype=torch.int64, device=DEV)
    K.serial_chain(idx, cand, out, T)
    _assert_greedy_chain_is_reference_argmax(out.cpu(), cand.cpu(), ref, bs, T)


# ---------------------------------------------------------------------------
# sampled verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "temperature,top_k,top_p,min_p",
    [
        (1.0, None, None, None),
        (0.7, 20, None, None),
        (1.2, None, 0.8, None),
        (0.6, 20, 0.95, None),
        (1.0, 5, 0.5, None),
        (0.9, 30, 0.9, 0.05),
    ],
)
def test_sampled_verify_is_lossless_cuda(temperature, top_k, top_p, min_p):
    """flashinfer primitives (the default CUDA sampling backend) against the independent
    reference of ordinary SGLang sampling, then Monte Carlo through the rejection sampler."""
    torch.manual_seed(0)
    trials, T, V, k = 40000, 4, 256, 8
    gamma = T - 1
    filters = dict(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
    base_logits = (torch.randn(T, V) * 2).to(DEV)
    built = build_dspine_target_probs(
        next_token_logits=base_logits,
        sampling_info=sampling_info(1, **filters, device=DEV),
        block_size=T,
        bs=1,
        sampling_backend="flashinfer",
    )[0].cpu()
    p = torch.stack(
        [
            ref_target_row(
                base_logits[i].cpu(),
                temperature,
                top_k or (1 << 30),
                top_p or 1.0,
                min_p or 0.0,
                "flashinfer",
            )
            for i in range(T)
        ]
    ).float()
    assert torch.equal(built > 0, p > 0), (
        "target support differs from ordinary sampling"
    )
    torch.testing.assert_close(built, p, rtol=1e-4, atol=1e-6)
    cand_rows = draft_candidates_overlapping(p[:gamma], k, 4, V)
    # q = half the target restricted to the candidates, half a random distribution: overlapping
    # enough that the last position is reached often, different enough that rejections happen.
    on_target = torch.gather(p[:gamma], 1, cand_rows)
    q_rows = 0.5 * on_target / on_target.sum(-1, keepdim=True) + 0.5 * torch.softmax(
        torch.randn(gamma, k) * 2, -1
    )
    lanes = torch.stack(
        [torch.multinomial(q_rows[i], trials, replacement=True) for i in range(gamma)],
        1,
    )
    drafts = cand_rows[torch.arange(gamma)[None, :], lanes]
    candidates = torch.cat((torch.zeros(trials, 1, dtype=torch.int64), drafts), 1).to(
        DEV
    )
    dense_q = torch.zeros(trials, gamma, V, device=DEV)
    accept_len, bonus = dspine_accept_sampling(
        candidates=candidates.contiguous(),
        target_logits=base_logits.repeat(trials, 1).contiguous(),
        candidate_ids=cand_rows[None].expand(trials, gamma, k).contiguous().to(DEV),
        q_rows=q_rows[None].expand(trials, gamma, k).contiguous().to(DEV),
        sampling_info=sampling_info(trials, **filters, device=DEV),
        sampling_backend="flashinfer",
        dense_q=dense_q,
    )
    assert bool((dense_q == 0).all())
    accept_len, bonus, candidates = accept_len.cpu(), bonus.cpu(), candidates.cpu()
    committed = torch.cat(
        (candidates[:, 1:], torch.zeros(trials, 1, dtype=torch.int64)), 1
    )
    committed.scatter_(1, accept_len.long()[:, None], bonus[:, None])
    for pos in range(T):
        # position pos is committed this round iff the drafts before it were all accepted
        assert_follows(committed[accept_len >= pos, pos], p[pos], 2000, f"pos {pos}")


# ---------------------------------------------------------------------------
# CUDA-graph replay of the refinement tail
# ---------------------------------------------------------------------------


def test_sampler_tail_replays_in_cuda_graph():
    """The capture-safe part of the tail replays correctly from a CUDA graph (candidates, the
    greedy chain proposed in `out`, state copies).  The draft graph runner's post-replay hook
    samples only on a round with a sampled request, and the sampled tokens and q match an
    independent recomputation on the replayed state under re-staged sampling params."""
    H, T, V, L, k = 512, 4, 2048, 2, 8
    bridge = _bridge("fused", H=H, C=64, R=128, T=T, L=L, V=V)
    g = torch.Generator().manual_seed(2)
    norm = types.SimpleNamespace(
        weight=(1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16).to(DEV),
        variance_epsilon=1e-6,
    )
    draft_model = types.SimpleNamespace(
        selected_prefix_bridge=bridge,
        head_topk=k,
        layers=[None] * L,
        norm=norm,
        last_write_stash=None,
        dspine_sampler=None,
    )
    lm = (torch.randn(V, H, generator=g) / math.sqrt(H)).to(torch.bfloat16).to(DEV)
    bs = 3
    m, n = bs * T, bs * (T - 1)
    hidden = torch.randn(m, H, generator=g).to(torch.bfloat16).to(DEV)
    residual = torch.randn(m, H, generator=g).to(torch.bfloat16).to(DEV)
    input_ids = torch.randint(0, V, (m,), generator=g).to(DEV)
    stash = {}
    final, _ = bridge(hidden, residual, input_ids, L - 1, norm, stash)
    draft_model.last_write_stash = stash
    sampler = DSpineDraftSampler(
        draft_model=draft_model, lm_head_weight=lm, block_size=T, max_bs=4, device=DEV
    )
    draft_model.dspine_sampler = sampler
    # The runner's execute() replays through DecodeCudaGraphRunner.execute (here: this test's
    # graph) and then runs the sampled step; it reads the sampler from model_runner.model.
    runner = object.__new__(DSpineDraftCudaGraphRunner)
    runner.model_runner = types.SimpleNamespace(model=draft_model)
    replayed_batch = types.SimpleNamespace(batch_size=bs)
    replay_output = object()

    def base_execute(self, forward_batch, pp_proxy_tensors=None):
        graph.replay()
        return replay_output

    def replay_round(info):
        sampler.stage_sampling_params(bs=bs, sampling_info=info)
        with mock.patch.object(DecodeCudaGraphRunner, "execute", base_execute):
            assert runner.execute(replayed_batch) is replay_output
        torch.cuda.synchronize()

    sampler(final, input_ids)  # compile every kernel before capture
    eager_greedy = sampler.greedy_out[:n].clone()
    eager_cand = sampler.candidate_out[:bs].clone()
    for buf in (
        sampler.greedy_out,
        sampler.out,
        sampler.candidate_out,
        sampler.raw,
        sampler.query,
    ):
        buf.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler(final, input_ids)
    # an all-greedy round proposes the greedy chain and never samples
    replay_round(None)
    assert not sampler.round_sampled
    assert torch.equal(sampler.greedy_out[:n], eager_greedy)
    assert torch.equal(sampler.out[:n], eager_greedy)
    assert torch.equal(sampler.candidate_out[:bs], eager_cand)
    assert torch.equal(sampler.raw[:m], stash["raw"])
    assert torch.equal(sampler.query[:m], stash["packed"][:, :128])

    sampled = sampling_info(
        bs, temperature=0.8, top_k=5, top_p=0.9, min_p=0.02, device=DEV
    )
    for _ in range(3):
        replay_round(sampled)
        assert sampler.round_sampled
        table = K.serial_rs_table(
            stash["raw"],
            stash["rms"],
            stash["packed"],
            input_ids.view(bs, T)[:, 0].contiguous(),
            eager_cand.reshape(n, k),
            bridge.codebook,
            sampler.pred_gate_f32,
            sampler.prev_f32,
            sampler.gate_b_f32,
            sampler.write_f32,
            sampler.norm_f32,
            lm,
            1e-6,
            T,
        )
        ref_tok, ref_q = ref_sample_chain(
            table.cpu(),
            sampler.inv_temp[:bs].cpu(),
            sampler.top_k[:bs].cpu(),
            sampler.top_p[:bs].cpu(),
            sampler.min_p[:bs].cpu(),
            sampler.greedy[:bs].cpu(),
            sampler.uniforms[:n].cpu(),
            eager_cand.reshape(n, k).cpu(),
            eager_greedy.cpu(),
            T,
        )
        assert torch.equal(sampler.out[:n].cpu(), ref_tok)
        torch.testing.assert_close(
            sampler.q_out[:bs].reshape(n, k).cpu(), ref_q, rtol=1e-5, atol=1e-6
        )
    # re-stage as greedy (top_k = 1) in a sampling round: the sampled step emits the greedy
    # chain with a one-hot q
    greedy_info = types.SimpleNamespace(**vars(sampled))
    greedy_info.top_ks = torch.ones(bs, dtype=torch.int32, device=DEV)
    replay_round(greedy_info)
    assert sampler.round_sampled
    assert torch.equal(sampler.out[:n], sampler.greedy_out[:n])
    assert torch.equal(
        sampler.q_out[:bs].reshape(n, k).max(-1).values, torch.ones(n, device=DEV)
    )
