"""Independent same-state/coin checks for compact and packed chain rejection."""
import hashlib
import importlib.util
import sys

import pytest
import torch

from sglang_dspine.sampling.sparse.accept import accept_sparse, compress_probs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def buffers(bs):
    return (torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int32),
            torch.empty(bs, device="cuda", dtype=torch.int64),
            torch.empty((bs, 16), device="cuda", dtype=torch.int64),
            torch.empty(bs, device="cuda", dtype=torch.int32))


def reference(p, candidates, qids, qvals, caps, u, uf, prefix):
    """Direct dense CPU rejection, independent of either Triton implementation."""
    answer = []
    offset = 0
    for b, length in enumerate(caps):
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
        emitted = candidates[b][1:a + 1] + [bonus] + [0] * (15 - a)
        answer.append((a, a + 1, bonus, emitted, prefix[b] + a + 1))
        offset += length
    return answer


@pytest.mark.parametrize("bs", [1, 9, 32, 65])
@pytest.mark.parametrize("kind", ["compact", "overflow", "pure_packed"])
def test_compact_packed_same_coins(bs, kind):
    torch.manual_seed(713)
    vocab = 257
    caps = (torch.arange(bs, device="cuda", dtype=torch.int32) % 16 + 1)
    caps[0] = 16
    offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), caps.long().cumsum(0)))[:-1]
    rows = int(caps.sum().item())
    p = torch.zeros((rows, vocab), device="cuda")
    support = torch.tensor([0, 2, 3, 7, 11, 13, 15], device="cuda")
    p[:, support] = torch.softmax(torch.randn(rows, len(support), device="cuda"), -1)
    qids = torch.arange(16, device="cuda", dtype=torch.int32).expand(bs, 15, 16).contiguous()
    qvals = torch.softmax(torch.randn((bs, 15, 16), device="cuda"), -1)
    candidates = torch.zeros((bs, 16), device="cuda", dtype=torch.int64)
    candidates[:, 1:] = torch.multinomial(qvals.reshape(-1, 16), 1).reshape(bs, 15)
    u = torch.full((bs, 15), .371, device="cuda")
    uf = torch.linspace(.071, .917, bs, device="cuda")
    prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 8191
    candidates[0, 1:] = 0
    u[0] = 0  # all accepted, including the cap16 bonus q-index boundary
    if bs > 1:
        # A rejection with zero residual must sample p, not invalid zero mass.
        b = 1
        lo = int(offsets[b].item())
        hi = lo + int(caps[b].item())
        same = torch.softmax(torch.arange(16, device="cuda", dtype=torch.float32), 0)
        p[lo:hi] = 0
        p[lo:hi, :16] = same
        qvals[b] = same
        u[b] = 1
    if kind == "overflow":
        p[::11] = torch.softmax(torch.randn((len(p[::11]), vocab), device="cuda"), -1)
    packed = compress_probs(torch, p, capacity=32)
    ids, values, lengths, overflow = packed
    # Support order is deliberately reversed; CDF must retain vocab-ID order.
    lane = torch.arange(32, device="cuda")[None, :]
    permutation = (lengths[:, None] - 1 - lane).clamp_min(0).long()
    packed = (ids.gather(1, permutation).where(lane < lengths[:, None], -1),
              values.gather(1, permutation).where(lane < lengths[:, None], 0), lengths, overflow)
    expected = reference(p.cpu().tolist(), candidates.cpu().tolist(), qids.cpu().tolist(),
                         qvals.cpu().tolist(), caps.cpu().tolist(), u.cpu().tolist(),
                         uf.cpu().tolist(), prefix.cpu().tolist())
    out = buffers(bs)
    accept_sparse(torch, candidates=candidates, dense_p=None if kind == "pure_packed" else p,
                  packed=packed, q_ids=qids, q_values=qvals, verify_lens=caps,
                  uniforms=u, uniform_final=uf, prefix_lens=prefix, outputs=out,
                  row_offsets=offsets, vocab_size=vocab)
    actual = [x.cpu().tolist() for x in out]
    for b, result in enumerate(expected):
        assert tuple(column[b] for column in actual) == result


@pytest.mark.parametrize("bs", [1, 17, 65])
def test_dense_without_packing_compact_rows(bs):
    torch.manual_seed(79)
    caps = torch.arange(bs, device="cuda", dtype=torch.int32) % 16 + 1
    offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), caps.long().cumsum(0)))[:-1]
    p = torch.softmax(torch.randn((int(caps.sum().item()), 97), device="cuda"), -1)
    qids = torch.arange(16, device="cuda", dtype=torch.int32).expand(bs, 15, 16).contiguous()
    qvals = torch.full((bs, 15, 16), 1 / 16, device="cuda")
    candidates = torch.arange(16, device="cuda").expand(bs, 16).contiguous()
    u = torch.full((bs, 15), .413, device="cuda")
    uf = torch.full((bs,), .273, device="cuda")
    prefix = torch.zeros(bs, device="cuda", dtype=torch.int32)
    expected = reference(p.cpu().tolist(), candidates.cpu().tolist(), qids.cpu().tolist(),
                         qvals.cpu().tolist(), caps.cpu().tolist(), u.cpu().tolist(),
                         uf.cpu().tolist(), prefix.cpu().tolist())
    out = buffers(bs)
    accept_sparse(torch, candidates=candidates, dense_p=p, packed=None, q_ids=qids, q_values=qvals,
                  verify_lens=caps, uniforms=u, uniform_final=uf, prefix_lens=prefix,
                  outputs=out, row_offsets=offsets)
    actual = [x.cpu().tolist() for x in out]
    for b, result in enumerate(expected):
        assert tuple(column[b] for column in actual) == result
