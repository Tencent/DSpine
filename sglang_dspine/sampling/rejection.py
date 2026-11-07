"""DSpine serving on SGLang 0.5.22."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _sample_capped_chain(
    Candidates,
    TargetProbs,
    DraftProbs,
    VerifyLens,
    Uniforms,
    UniformFinal,
    Prefix,
    Accept,
    Commit,
    Bonus,
    Output,
    NewSeq,
    T: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One request per program; classical token-wise speculative rejection.

    The dense target/draft envelopes are full16/full15.  Only the verified
    prefix is dereferenced.  This follows upstream's classic rejection kernel
    and changes its terminal condition from the static width to VerifyLens[b].
    """
    b = tl.program_id(0)
    cap = tl.load(VerifyLens + b)
    accepted = 0
    step = 1
    continuing = 1
    while (step < cap) & (continuing == 1):
        token = tl.load(Candidates + b * T + step)
        p = tl.load(TargetProbs + (b * T + step - 1) * V + token)
        q = tl.load(DraftProbs + (b * (T - 1) + step - 1) * V + token)
        coin = tl.load(Uniforms + b * (T - 1) + step - 1)
        if (q > 0.0) & (q <= 1.0) & (coin * q < p):
            accepted += 1
            step += 1
        else:
            continuing = 0
    all_accepted = accepted == cap - 1
    p_base = TargetProbs + (b * T + accepted) * V
    q_base = DraftProbs + (b * (T - 1) + accepted) * V
    mass = 0.0
    for start in range(0, V, BLOCK_V):
        offset = start + tl.arange(0, BLOCK_V)
        p_vec = tl.load(p_base + offset, offset < V, 0.0)
        if all_accepted:
            value = p_vec
        else:
            q_vec = tl.load(q_base + offset, offset < V, 0.0)
            q_vec = tl.where((q_vec >= 0.0) & (q_vec <= 1.0), q_vec, 0.0)
            value = tl.maximum(p_vec - q_vec, 0.0)
        mass += tl.sum(value)
    pure_target = all_accepted | (mass <= 0.0)
    if (mass <= 0.0) & (all_accepted == 0):
        mass = 0.0
        for start in range(0, V, BLOCK_V):
            offset = start + tl.arange(0, BLOCK_V)
            mass += tl.sum(tl.load(p_base + offset, offset < V, 0.0))
    threshold = tl.load(UniformFinal + b) * mass
    before = 0.0
    bonus = V - 1
    found = 0
    for start in range(0, V, BLOCK_V):
        if found == 0:
            offset = start + tl.arange(0, BLOCK_V)
            p_vec = tl.load(p_base + offset, offset < V, 0.0)
            if pure_target:
                value = p_vec
            else:
                q_vec = tl.load(q_base + offset, offset < V, 0.0)
                q_vec = tl.where((q_vec >= 0.0) & (q_vec <= 1.0), q_vec, 0.0)
                value = tl.maximum(p_vec - q_vec, 0.0)
            cdf = before + tl.cumsum(value, 0)
            hit = (cdf > threshold) & (offset < V)
            if tl.max(hit.to(tl.int32), 0) != 0:
                bonus = start + tl.argmax(hit.to(tl.int32), 0)
                found = 1
            before += tl.sum(value)
    tl.store(Accept + b, accepted)
    tl.store(Commit + b, accepted + 1)
    tl.store(Bonus + b, bonus)
    i = tl.arange(0, T)
    proposed = tl.load(Candidates + b * T + i + 1, i < T - 1, 0)
    emitted = tl.where(i < accepted, proposed, tl.where(i == accepted, bonus, 0))
    tl.store(Output + b * T + i, emitted)
    tl.store(NewSeq + b, tl.load(Prefix + b) + accepted + 1)


def sample_capped_chain_from_probs(
    *,
    candidates,
    target_probs,
    draft_probs,
    verify_lens,
    uniforms,
    uniform_final,
    prefix_lens,
    outputs,
):
    """Launch with explicit coins, shared by the runtime and numerical tests."""
    (bs, width) = candidates.shape
    vocab = target_probs.shape[-1]
    if width != 16 or target_probs.shape != (bs, width, vocab):
        raise ValueError("Sampled ragged adapter requires target [B,16,V]")
    if draft_probs.shape != (bs, width - 1, vocab):
        raise ValueError("Sampled ragged adapter requires draft q [B,15,V]")
    if verify_lens.shape != (bs,) or uniforms.shape != (bs, width - 1):
        raise ValueError("Invalid cap or uniform layout")
    if uniform_final.shape != (bs,) or prefix_lens.shape != (bs,):
        raise ValueError("Invalid final uniform or prefix layout")
    if len(outputs) != 5:
        raise ValueError("Expected accept/commit/bonus/output/new_seq buffers")
    tensors = (
        candidates,
        target_probs,
        draft_probs,
        verify_lens,
        uniforms,
        uniform_final,
        prefix_lens,
        *outputs,
    )
    if any((not t.is_contiguous() for t in tensors)):
        raise ValueError("The sampled capped kernel requires contiguous tensors")
    if any((t.device != candidates.device for t in tensors)):
        raise ValueError("The sampled capped kernel requires one device")
    _sample_capped_chain[bs,](
        candidates,
        target_probs,
        draft_probs,
        verify_lens,
        uniforms,
        uniform_final,
        prefix_lens,
        *outputs,
        T=width,
        V=vocab,
        BLOCK_V=4096,
        num_warps=4,
    )
    return outputs


def accept_sampled_prefix(
    controller,
    *,
    candidates,
    next_token_logits,
    sampling_info,
    prefix_lens,
    bs,
    verify_lens,
):
    """Verify one shortened target call, using this round's sampled chain q."""
    from sglang.srt.runtime_context import get_exec

    from sglang_dspine.runtime.worker import build_dspine_target_probs

    worker = controller.worker
    if worker.target_distribution is not None:
        result = worker.target_distribution.sample_prefix(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            prefix_lens=prefix_lens,
            bs=bs,
            verify_lens=verify_lens,
        )
        if result is not None:
            controller.sampled_executions += 1
            return result
    sampler = worker._dspine_sampler
    if not sampler.round_sampled or sampler.round_bs != bs:
        raise RuntimeError("Sampled caps require the current sampled proposal and q")
    width = int(worker.block_size)
    if width != 16 or candidates.shape != (bs, width):
        raise ValueError("Sampled caps require full16 proposal storage")
    if next_token_logits.shape[0] != bs * width:
        raise ValueError("Target output must retain the full16 mapped row envelope")
    target_probs = build_dspine_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        block_size=width,
        bs=bs,
        sampling_backend=get_exec().kernel.sampling_backend,
    )
    vocab = target_probs.shape[-1]
    dense_q = worker._dense_q_buffer(bs, vocab)[:bs]
    ids = sampler.candidate_out[:bs]
    q_rows = sampler.q_out[:bs]
    outputs = worker._next_accept_bonus_buffers(bs)
    from sglang_dspine.sampling.random import verification_uniforms

    uniforms, final = verification_uniforms(
        bs, width, candidates.device, sampler.random
    )
    try:
        dense_q.scatter_(-1, ids, q_rows.to(torch.float32))
        sample_capped_chain_from_probs(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=dense_q,
            verify_lens=verify_lens,
            uniforms=uniforms,
            uniform_final=final,
            prefix_lens=prefix_lens,
            outputs=outputs,
        )
    finally:
        dense_q.scatter_(-1, ids, 0.0)
    controller.sampled_executions += 1
    if controller.sampled_executions == 1:
        logger.info(
            "DSPine sampled capped rejection executed: batch=%d full=%d target_policy=AR-joint lengths=%s",
            bs,
            width,
            "device-ragged"
            if getattr(controller, "ragged_round", False)
            else controller.active,
        )
    return outputs
