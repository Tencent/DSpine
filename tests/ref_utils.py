"""Independent numerical references for DSpine tests.
Derive results directly with explicit loops, sorts, and float64 where rounding
matters, without depending on SGLang or DSpine implementations."""

import math
import types

import torch


def ref_sample_chain(table, inv_t, top_k, top_p, min_p, greedy, u, cand, greedy_tok, T):
    """(out_tok [n], out_q [n, K]) of the sampled serial chain, per the contract of
    sglang_dspine.kernels.serial_sample_chain."""
    n, k, _ = table.shape
    tm1 = T - 1
    out_tok = torch.empty(n, dtype=torch.int64)
    out_q = torch.empty(n, k, dtype=torch.float32)
    for b in range(n // tm1):
        sender = 0
        for i in range(tm1):
            row = b * tm1 + i
            scores = table[row, sender].float() * float(inv_t[b])
            mass = torch.exp(scores - scores.max())
            p = mass / mass.sum()
            order = sorted(range(k), key=lambda c: (-p[c].item(), c))
            rank = {c: r for r, c in enumerate(order)}
            before = [float(sum(p[d] for d in order[: rank[c]])) for c in range(k)]
            keep = torch.tensor(
                [
                    rank[c] < int(top_k[b])
                    and before[c] <= float(top_p[b])
                    and float(p[c]) >= float(min_p[b]) * float(p.max())
                    for c in range(k)
                ]
            )
            w = torch.where(keep, p, torch.zeros_like(p))
            q = w / w.sum()
            cdf = torch.cumsum(q, 0)
            threshold = float(u[row]) * float(cdf[-1])
            choice = int((cdf <= threshold).sum())
            choice = min(choice, max(c for c in range(k) if keep[c]))
            if int(greedy[b]):
                out_tok[row] = greedy_tok[row]
                out_q[row] = (cand[row] == greedy_tok[row]).float()
            else:
                out_tok[row] = cand[row, choice]
                out_q[row] = q
            sender = choice
    return out_tok, out_q


def ref_refine_table(
    raw, rms, query, anchors, cand, codebook, pg, pv, gb, w_last, norm_w, lm, eps, T
):
    """float64 Table[n, j, c] of the redone last write, per the contract of
    sglang_dspine.kernels.serial_rs_table."""
    m, h = raw.shape
    n, k = cand.shape
    c_size = codebook.shape[1]
    out = torch.zeros(n, k, k, dtype=torch.float64)
    for b in range(m // T):
        for i in range(T - 1):
            row, nn = b * T + 1 + i, b * (T - 1) + i
            senders = [int(anchors[b])] * k if i == 0 else cand[nn - 1].tolist()
            x = raw[row].double()
            for j, s in enumerate(senders):
                e = math.sqrt(c_size) * codebook[s].double()
                gate = torch.sigmoid(
                    query[row].double() + pg.double() @ e + gb.double()
                )
                msg = gate * (pv.double() @ e)
                xr = x + rms[row].double() * (w_last.double() @ msg)
                normalized = xr / torch.sqrt(xr.square().mean() + eps) * norm_w.double()
                out[nn, j] = lm[cand[nn]].double() @ normalized
    return out


def ref_target_row(logits_row, temperature, top_k, top_p, min_p, backend):
    """One row of the distribution SGLang's ordinary Sampler draws from, rebuilt from its
    definition (srt/layers/sampler.py) in float64 with an explicit sort:
      flashinfer, min_p == 0: keep rank < top_k AND (mass ranked ahead) <= top_p, both on the
          unfiltered probs (filter_apply_order="joint");
      flashinfer, min_p > 0: renormalise over the top-k, apply top-p to THAT, keep >= min_p * max;
      pytorch: the joint top-k / top-p support, then keep >= min_p * max."""
    p = torch.softmax(logits_row.double() / temperature, -1)
    order = torch.argsort(-p, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(p.numel())
    in_k = rank < top_k

    def mass_ahead(w):
        ws = w[order]
        ahead_sorted = torch.cumsum(ws, 0) - ws
        out = torch.empty_like(w)
        out[order] = ahead_sorted
        return out

    if backend == "flashinfer" and min_p > 0:
        w = torch.where(in_k, p, torch.zeros_like(p))
        w = w / w.sum()
        w = torch.where(mass_ahead(w) <= top_p, w, torch.zeros_like(w))
        w = w / w.sum()
        w = torch.where(w >= min_p * w.max(), w, torch.zeros_like(w))
    else:
        keep = in_k & (mass_ahead(p) <= top_p) & (p >= min_p * p.max())
        w = torch.where(keep, p, torch.zeros_like(p))
    return w / w.sum()


def assert_follows(toks, p_row, min_samples, tag):
    """Committed tokens at one position must be distributed as the target row p_row: no token
    outside its support, and a chi-square fit (sparse tail pooled so every bin expects >= 5).
    A single-token target is fully checked by the support assertion (chi-square has no degrees
    of freedom there); too few samples is a test-design error, raised rather than skipped."""
    from scipy.stats import chisquare

    n = toks.numel()
    assert n >= min_samples, f"{tag}: the test design reached only {n} samples"
    counts = torch.bincount(toks, minlength=p_row.numel()).double()
    live = p_row > 0
    assert float(counts[~live].sum()) == 0, (tag, "sampled a token the target excludes")
    expected = p_row.double()[live]
    expected = expected / expected.sum() * n
    keep = expected >= 5
    obs = torch.cat((counts[live][keep], counts[live][~keep].sum().view(1)))
    exp = torch.cat((expected[keep], expected[~keep].sum().view(1)))
    if float(exp[-1]) == 0:
        obs, exp = obs[:-1], exp[:-1]
    if obs.numel() < 2:
        return
    stat = chisquare(obs.numpy(), exp.numpy())
    assert stat.pvalue > 1e-4, (tag, stat)


def draft_candidates_overlapping(p, k, n_top, V, generator=None):
    """Per draft row i: the n_top most likely target tokens of p[i] plus k - n_top random others,
    ascending -- a draft that overlaps the target (so deep positions are reached) while its q
    (chosen by the caller) still disagrees with p."""
    rows = []
    for i in range(p.shape[0]):
        top = torch.topk(p[i], n_top).indices.tolist()
        rest = [
            t for t in torch.randperm(V, generator=generator).tolist() if t not in top
        ][: k - n_top]
        rows.append(torch.tensor(top + rest).sort().values)
    return torch.stack(rows)


def sampling_info(bs, *, temperature, top_k=None, top_p=None, min_p=None, device="cpu"):
    """Stand-in for SamplingBatchInfo with the fields DSpine's sampler and verify read; a filter
    left as None is off (SGLang encodes "no top-k" as a huge top_k)."""
    return types.SimpleNamespace(
        temperatures=torch.full((bs, 1), temperature, device=device),
        top_ks=torch.full((bs,), top_k or (1 << 30), dtype=torch.int32, device=device),
        top_ps=torch.full((bs,), top_p or 1.0, device=device),
        min_ps=torch.full((bs,), min_p or 0.0, device=device),
        need_top_k_sampling=top_k is not None,
        need_top_p_sampling=top_p is not None,
        need_min_p_sampling=min_p is not None,
        is_all_greedy=False,
    )
