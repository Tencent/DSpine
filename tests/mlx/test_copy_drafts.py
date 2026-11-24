"""Copy drafts: proposals copied from the context, checked by the target like any draft."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dspine_mlx.generate import CopyIndex, DSpineGenerator, baseline_generate
from dspine_mlx.target import TargetAdapter
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


class Adapter(TargetAdapter):
    def __init__(self, model):
        super().__init__(model, FakeTokenizer(), TARGET_IDS)

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


def _gen(seed, **kw):
    model = make_target(seed)
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
            for k, v in tree_flatten(model.parameters())
        ]
    )
    adapter = Adapter(model)
    draft, _ = make_draft(seed + 10)
    draft.bind(model)
    return adapter, DSpineGenerator(adapter, draft, **kw)


def test_copy_index():
    idx = CopyIndex(3)
    idx.extend([1, 2, 3, 4, 5, 9, 1, 2, 3])
    assert idx.lookup(4) == [4, 5, 9, 1]
    idx.extend([7])
    assert idx.lookup(4) == []
    idx = CopyIndex(2)
    idx.extend([5, 6, 1, 5, 6, 2, 5, 6])
    assert idx.lookup(3) == [2, 5, 6]  # the latest earlier occurrence wins


def _patch_lookup(monkeypatch, prompt, propose):
    """Make every copy lookup propose ``propose(position in the answer, max_len)``."""

    def lookup(self, max_len):
        return propose(len(self.ids) - len(prompt), max_len)

    monkeypatch.setattr(CopyIndex, "lookup", lookup)


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("right", [True, False])
def test_greedy_copy_rounds_keep_the_output(monkeypatch, seed, right):
    """A copy the target agrees with is taken whole, a wrong one is rejected; same tokens either way."""
    adapter, plain = _gen(seed)
    _, gen = _gen(seed, copy=True, copy_min=4)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (30,)).tolist()
    ref, _ = plain.generate(prompt, max_tokens=60)
    if right:
        _patch_lookup(monkeypatch, prompt, lambda at, n: ref[at : at + n])
    else:
        _patch_lookup(monkeypatch, prompt, lambda at, n: [(t + 1) % VOCAB for t in ref[at : at + n]])
    toks, stats = gen.generate(prompt, max_tokens=60)
    assert toks == ref
    if mx.default_device() == mx.cpu:
        base, _ = baseline_generate(adapter, prompt, max_tokens=60)
        assert toks == base
    assert stats.copy_rounds > 0
    if right:
        assert stats.tokens_per_round > 6  # copies double up to 15 tokens while they are accepted
    else:
        assert stats.rounds > stats.copy_rounds  # a missed copy hands the next round to the draft


def test_sampled_copy_rounds_run(monkeypatch):
    _, gen = _gen(1, copy=True, copy_min=4)
    mx.random.seed(1)
    prompt = mx.random.randint(0, VOCAB - 4, (20,)).tolist()
    _patch_lookup(monkeypatch, prompt, lambda at, n: [7] * n)
    toks, stats = gen.generate(prompt, max_tokens=40, temperature=0.3)
    assert len(toks) == 40 and stats.copy_rounds > 0


def test_copy_off_is_the_old_generator():
    _, a = _gen(2)
    _, b = _gen(2, copy=True, copy_min=10_000)  # copy on, never matches
    mx.random.seed(2)
    p = mx.random.randint(0, VOCAB - 4, (30,)).tolist()
    mx.random.seed(5)
    x, _ = a.generate(p, max_tokens=40, temperature=0.7)
    mx.random.seed(5)
    y, _ = b.generate(p, max_tokens=40, temperature=0.7)
    assert x == y
