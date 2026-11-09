"""Hybrid (qwen3_5) targets: GatedDeltaNet rollback and chunked prefill, on a tiny random model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, make_prompt_cache

from dspine_mlx import target as target_mod
from dspine_mlx.generate import DSpineGenerator, baseline_generate
from dspine_mlx.hybrid import RecordingGatedDeltaNet
from dspine_mlx.target import TargetAdapter
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


def _eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _states(cache):
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append(("ssm", c[0], c[1]))
        else:
            out.append(("kv", c.keys[:, :, : c.offset], c.values[:, :, : c.offset]))
    return out


def test_recording_layer_is_bit_identical():
    ref = make_target(0)
    rec = make_target(0)
    adapter = TargetAdapter(rec, FakeTokenizer(), TARGET_IDS)
    toks = mx.random.randint(0, VOCAB, (1, 23))
    c1 = make_prompt_cache(ref)
    l1 = ref(toks, c1)
    adapter.reset()
    l2, _ = adapter.prefill(toks)
    assert _eq(l1[:, -1:], l2)
    nxt = mx.random.randint(0, VOCAB, (1, 9))
    assert _eq(ref(nxt, c1), adapter.verify(nxt)[0])
    n_lin = sum(isinstance(m, RecordingGatedDeltaNet) for m in adapter._linear)
    assert n_lin == 6


@pytest.mark.parametrize("accepted", [1, 2, 5, 8, 9])
def test_rollback_matches_direct_prefix(accepted):
    """verify(T) + trim(T-n) leaves the cache exactly as feeding the n accepted tokens."""
    adapter = TargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    prompt = mx.random.randint(0, VOCAB, (1, 17))
    block = mx.random.randint(0, VOCAB, (1, 9))

    adapter.reset()
    adapter.prefill(prompt)
    adapter.verify(block)
    adapter.trim(9 - accepted)
    rolled = _states(adapter.cache)
    l_rolled, _ = adapter.verify(mx.array([[7]]))

    adapter.reset()
    adapter.prefill(prompt)
    adapter.verify(block[:, :accepted])
    adapter.trim(0)
    direct = _states(adapter.cache)
    l_direct, _ = adapter.verify(mx.array([[7]]))

    for (kind, a0, a1), (_, b0, b1) in zip(rolled, direct):
        assert _eq(a0, b0) and _eq(a1, b1), kind
    assert _eq(l_direct, l_rolled)


def test_trim_needs_verify():
    adapter = TargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    adapter.reset()
    adapter.prefill(mx.array([[1, 2, 3]]))
    with pytest.raises(RuntimeError):
        adapter.trim(1)


def test_chunked_prefill_matches_one_shot(monkeypatch):
    adapter = TargetAdapter(make_target(7), FakeTokenizer(), TARGET_IDS)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 37), key=mx.random.key(7))
    adapter.reset()
    ref_logits, ref_hidden = adapter.prefill(prompt)  # one chunk
    ref_states = _states(adapter.cache)
    mx.eval(ref_logits, ref_hidden, ref_states)
    monkeypatch.setattr(target_mod, "PREFILL_CHUNK", 8)
    logits, hidden = adapter.prefill(prompt)
    assert logits.shape == ref_logits.shape == (1, 1, VOCAB)
    assert hidden.shape == ref_hidden.shape

    def close(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        return mx.max(mx.abs(a - b)).item() <= 3e-2 * max(mx.max(mx.abs(b)).item(), 1.0)

    assert close(logits, ref_logits) and close(hidden, ref_hidden)
    for (k1, a1, b1), (k2, a2, b2) in zip(_states(adapter.cache), ref_states):
        assert k1 == k2 and close(a1, a2) and close(b1, b2)


class _Fp32HiddenAdapter(TargetAdapter):
    """fp32 target whose captured features reach the bf16 draft as bf16."""

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


def _divergence(adapter, prompt, toks, base):
    if toks == base:
        return None
    i = next(j for j in range(min(len(toks), len(base))) if toks[j] != base[j])
    adapter.reset()
    logits, _ = adapter.prefill(mx.array([prompt + base[:i]]), capture=False)
    row = logits[0, -1].astype(mx.float32)
    margin = abs(float(row[toks[i]].item()) - float(row[base[i]].item()))
    return margin, float((mx.max(row) - mx.min(row)).item())


def _noise_floor(adapter, prompt, seq):
    """Max |logit| gap between one len(seq)-token pass and len(seq) one-token passes."""
    adapter.reset()
    adapter.prefill(mx.array([prompt]), capture=False)
    block, _ = adapter._forward(mx.array([seq]), capture=False)
    adapter.reset()
    adapter.prefill(mx.array([prompt]), capture=False)
    single = mx.concatenate([adapter._forward(mx.array([[t]]), capture=False)[0] for t in seq], axis=1)
    return float(mx.max(mx.abs(block.astype(mx.float32) - single.astype(mx.float32))).item())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dspine_greedy_matches_plain_greedy(seed):
    """fp32 hybrid target: DSpine greedy = plain greedy (on Metal, up to block-vs-single rounding at near-ties)."""
    model = make_target(seed)
    model.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                        for k, v in tree_flatten(model.parameters())])
    adapter = _Fp32HiddenAdapter(model, FakeTokenizer(), TARGET_IDS)
    draft, _ = make_draft(seed + 10)
    draft.bind(model)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 12)).tolist()[0]
    toks, stats = DSpineGenerator(adapter, draft).generate(prompt, max_tokens=80)
    base, _ = baseline_generate(adapter, prompt, max_tokens=80)
    assert stats.rounds > 5
    d = _divergence(adapter, prompt, toks, base)
    if d is None:
        return
    assert mx.default_device() != mx.cpu, "CPU must be token-exact"
    margin, spread = d
    assert margin <= max(4 * _noise_floor(adapter, prompt, base), 1e-4 * spread)
