"""Prompt cache across turns: a reused cache must give what a full prefill gives."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from dspine_mlx.generate import DSpineGenerator, baseline_generate
from dspine_mlx.target import TargetAdapter
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


def _fp32_target(seed):
    model = make_target(seed)
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
            for k, v in tree_flatten(model.parameters())
        ]
    )
    return model


class Adapter(TargetAdapter):
    """fp32 target whose captured features reach the bf16 draft as bf16."""

    def __init__(self, model):
        super().__init__(model, FakeTokenizer(), TARGET_IDS)

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


def _states(cache):
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append((c[0], c[1]))
        else:
            out.append((c.keys[:, :, : c.offset], c.values[:, :, : c.offset]))
    return out


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    if a.shape != b.shape:
        return float("inf")
    return mx.max(mx.abs(a - b)).item() / max(mx.max(mx.abs(b)).item(), 1.0)


def _diffs(logits, hidden, cache, rl, rh, rcache):
    n = int(hidden.shape[1])
    st = max(
        max(_rel(a0, b0), _rel(a1, b1))
        for (a0, a1), (b0, b1) in zip(_states(cache), _states(rcache))
    )
    return _rel(logits[0, -1], rl[0, -1]), _rel(hidden, rh[:, -n:]), st


def _same_as_fresh(seed, adapter, ids, logits, hidden, blocked_from=None):
    """Logits, every cache state and the hidden rows equal a fresh full prefill.

    blocked_from: the reused cache was built by verify blocks after this many
    tokens. Metal rounds block passes differently from a one-shot prefill, so
    the tolerance grows to 8x the difference a plain block-wise pass shows on
    this device; a wrong state is off by far more. On CPU that difference is ~0.
    """
    ref = Adapter(_fp32_target(seed))
    ref.reset()
    rl, rh = ref.prefill(mx.array([ids]))
    got = _diffs(logits, hidden, adapter.cache, rl, rh, ref.cache)
    tol = [2e-4] * 3
    if blocked_from is not None:
        blk = Adapter(_fp32_target(seed))
        blk.reset()
        _, h0 = blk.prefill(mx.array([ids[:blocked_from]]))
        hs = [h0]
        end = len(ids) - 1
        for s in range(blocked_from, end, 5):
            _, h = blk.verify(mx.array([ids[s : min(s + 5, end)]]))
            blk.trim(0)
            hs.append(h)
        bl, h = blk.verify(mx.array([ids[end:]]))
        hs.append(h)
        rows = mx.concatenate(hs, axis=1)[:, -int(hidden.shape[1]) :]
        noise = _diffs(bl, rows, blk.cache, rl, rh, ref.cache)
        tol = [max(t, 8 * x) for t, x in zip(tol, noise)]
    for name, g, t in zip(("logits", "hidden", "states"), got, tol):
        assert g <= t, f"{name}: {g:.2e} > {t:.2e}"


def _setup(seed, window=None, cache=True):
    model = _fp32_target(seed)
    adapter = Adapter(model)
    draft, _ = make_draft(seed + 10, window=window)
    draft.bind(model)
    if cache:
        adapter.enable_prompt_cache(True, ctx_rows=draft.config.sliding_window)
    gen = DSpineGenerator(adapter, draft)
    mx.random.seed(seed)
    p1 = mx.random.randint(0, VOCAB - 4, (40,)).tolist()
    return adapter, gen, p1


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("max_tokens", [30, 31, 37])
def test_continued_chat_reuses_everything(seed, max_tokens):
    """Turn 2 = turn 1 prompt + answer + new text: only the new part is prefilled."""
    adapter, gen, p1 = _setup(seed)
    toks, _ = gen.generate(p1, max_tokens=max_tokens)
    assert adapter.last_reused == 0
    user = mx.random.randint(0, VOCAB - 4, (11,)).tolist()
    p2 = p1 + toks + user
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([p2]))
    # The last generated token was never fed, so at least that one is new.
    assert len(p1) + len(toks) - 1 - 16 <= adapter.last_reused <= len(p1) + len(toks) - 1
    assert adapter.last_reused > len(p1)
    assert hidden.shape[1] == len(p2)
    _same_as_fresh(seed, adapter, p2, logits, hidden, blocked_from=len(p1))


@pytest.mark.parametrize("seed", [0, 1])
def test_edited_history_restores_prompt_snapshot(seed):
    """Turn 2 changes the answer (e.g. stripped thinking): back to the prompt snapshot."""
    adapter, gen, p1 = _setup(seed)
    toks, _ = gen.generate(p1, max_tokens=24)
    p2 = p1 + toks[:3] + [VOCAB - 2] + toks[5:9]
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([p2]))
    assert adapter.last_reused == len(p1)
    _same_as_fresh(seed, adapter, p2, logits, hidden)


def test_same_prompt_again_and_new_prompt():
    adapter, gen, p1 = _setup(3)
    a, _ = gen.generate(p1, max_tokens=20)
    b, _ = gen.generate(p1, max_tokens=20)
    assert adapter.last_reused == len(p1) and a == b
    other = [(t + 1) % (VOCAB - 4) for t in p1]
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([other]))
    assert adapter.last_reused == 0
    _same_as_fresh(3, adapter, other, logits, hidden)


@pytest.mark.parametrize("window", [None, 24])
def test_multi_turn_greedy_matches_fresh_runs(window):
    """Three turns with the cache give the same tokens as three fresh runs."""
    adapter, gen, p1 = _setup(5, window)
    _, fresh, _ = _setup(5, window, cache=False)
    hist = list(p1)
    for turn in range(3):
        toks, _ = gen.generate(hist, max_tokens=25)
        ref, _ = fresh.generate(hist, max_tokens=25)
        assert toks == ref, f"turn {turn}"
        if turn:
            assert adapter.last_reused > len(p1)
        hist = hist + toks + mx.random.randint(0, VOCAB - 4, (7,)).tolist()


def test_stop_token_ends_a_turn_and_the_next_one_reuses_it():
    adapter, gen, p1 = _setup(4)
    _, fresh, _ = _setup(4, cache=False)
    full, _ = gen.generate(p1, max_tokens=30)
    stop = full[12]
    adapter.enable_prompt_cache(True)
    toks, _ = gen.generate(p1, max_tokens=30, stop_ids=[stop])
    assert toks[-1] == stop and stop not in toks[:-1]
    p2 = p1 + toks + [3, 4, 5]
    a, _ = gen.generate(p2, max_tokens=15)
    assert adapter.last_reused >= len(p1)
    b, _ = fresh.generate(p2, max_tokens=15)
    assert a == b


def test_off_by_default_and_plain_decoding_drops_the_session():
    assert Adapter(_fp32_target(2)).prompt_cache is False
    adapter, gen, p1 = _setup(2)
    toks, _ = gen.generate(p1, max_tokens=12)
    base, _ = baseline_generate(adapter, p1, max_tokens=12)  # capture=False + step()
    assert base == toks
    adapter.reset()
    adapter.prefill(mx.array([p1 + [1, 2]]))
    assert adapter.last_reused == 0
    off, gen_off, _ = _setup(2, cache=False)
    gen_off.generate(p1, max_tokens=12)
    gen_off.generate(p1 + toks + [1], max_tokens=12)
    assert off.last_reused == 0 and off._seq is None


def _attention_target():
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs.from_dict(
        dict(
            model_type="qwen3",
            hidden_size=128,
            num_hidden_layers=8,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
    )
    mx.random.seed(0)
    model = qwen3.Model(args)
    model.set_dtype(mx.float32)
    model.eval()
    return TargetAdapter(model, FakeTokenizer(), TARGET_IDS)


def test_pure_attention_target():
    """Qwen3-style targets (no GatedDeltaNet layers) reuse the cache the same way."""
    adapter, ref = _attention_target(), _attention_target()
    adapter.enable_prompt_cache(True)
    p1 = mx.random.randint(0, VOCAB, (30,)).tolist()
    blk = mx.random.randint(0, VOCAB, (9,)).tolist()
    adapter.reset()
    adapter.prefill(mx.array([p1]))
    adapter.verify(mx.array([blk]))
    adapter.trim(4)
    adapter.verify(mx.array([[7, 8, 9]]))  # the last round of a turn is not trimmed
    cont = p1 + blk[:5] + [7, 8, 1, 2]
    for prompt, reused in ((cont, 37), (cont, 39), (cont + [3], 39), (p1 + [5, 6], 0)):
        adapter.reset()
        logits, hidden = adapter.prefill(mx.array([prompt]))
        assert adapter.last_reused == reused
        ref.reset()
        rl, rh = ref.prefill(mx.array([prompt]))
        # Metal rounds a continued cache slightly differently from a one-shot pass
        tol = 2e-4 if mx.default_device() == mx.cpu else 5e-3
        assert _rel(logits, rl) <= tol and _rel(hidden, rh) <= tol


@pytest.mark.parametrize("seed", [0, 1])
def test_new_conversation_with_the_same_system_prompt_uses_a_mark(seed):
    """Same long start, different question: the third prompt restarts from the shared start."""
    adapter, gen, p1 = _setup(seed)
    adapter.mark_min = 8
    system = p1[:30]
    q = [mx.random.randint(0, VOCAB - 4, (9,)).tolist() for _ in range(4)]
    gen.generate(system + q[0], max_tokens=20)
    assert adapter.last_reused == 0
    gen.generate(system + q[1], max_tokens=20)  # learns where the two prompts part
    assert adapter.last_reused == 0 and [m[0] for m in adapter._marks] == [30]
    toks, _ = gen.generate(system + q[2], max_tokens=20)
    assert adapter.last_reused == 30
    _, fresh, _ = _setup(seed, cache=False)
    ref, _ = fresh.generate(system + q[2], max_tokens=20)
    assert toks == ref
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system + q[3]]))
    assert adapter.last_reused == 30 and hidden.shape[1] == 39
    _same_as_fresh(seed, adapter, system + q[3], logits, hidden)


def test_mark_hint_helps_from_the_second_prompt():
    adapter, gen, p1 = _setup(2)
    adapter.mark_min = 8
    system = p1[:25]
    q = [mx.random.randint(0, VOCAB - 4, (7,)).tolist() for _ in range(3)]
    adapter.mark_hint = 25
    gen.generate(system + q[0], max_tokens=12)
    assert adapter.mark_hint is None and [m[0] for m in adapter._marks] == [25]
    adapter.mark_hint = 25
    gen.generate(system + q[1], max_tokens=12)
    assert adapter.last_reused == 25
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system + q[2]]))
    assert adapter.last_reused == 25
    _same_as_fresh(2, adapter, system + q[2], logits, hidden)
    # a prompt that is exactly the marked start has no stored logits: it is read again
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system]))
    _same_as_fresh(2, adapter, system, logits, hidden)


def test_marks_with_a_draft_window():
    adapter, gen, p1 = _setup(6, window=16)
    _, fresh, _ = _setup(6, window=16, cache=False)
    adapter.mark_min = 8
    system = p1[:30]
    q = [mx.random.randint(0, VOCAB - 4, (9,)).tolist() for _ in range(3)]
    for i in range(3):
        a, _ = gen.generate(system + q[i], max_tokens=20)
        b, _ = fresh.generate(system + q[i], max_tokens=20)
        assert a == b
    assert adapter.last_reused == 30
