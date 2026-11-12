"""The draft's sliding window (Qwen3.8-27B-DSpine-b16: 8192), on a tiny random model."""

import dataclasses

import mlx.core as mx
import pytest

from dspine_mlx.config import DraftConfig
from tiny import HIDDEN, TARGET_IDS, draft_config, make_draft, make_target


def _hidden(n, seed):
    return mx.random.normal((1, n, HIDDEN * len(TARGET_IDS)), key=mx.random.key(seed)).astype(mx.bfloat16)


def _run(draft, ctx_len, rounds=3, seed=0):
    """First block with ctx_len context rows, then rounds that add 5 rows each."""
    caches = draft.make_cache()
    hid, outs = _hidden(ctx_len, seed), []
    for r in range(rounds):
        block = mx.random.randint(0, 400, (1, 16), key=mx.random.key(100 + r))
        out, _ = draft(hid, block, caches)
        outs.append(out)
        hid = _hidden(5, seed + 1 + r)
    return outs


def test_config_parses_window():
    assert DraftConfig.from_dict(draft_config(2)).sliding_window is None
    assert DraftConfig.from_dict(draft_config(2, window=8192)).sliding_window == 8192
    raw = draft_config(2, window=8192)
    raw["layer_types"] = ["sliding_attention", "full_attention"]
    with pytest.raises(Exception):
        DraftConfig.from_dict(raw)
    raw = draft_config(2, window=8192)
    raw["max_window_layers"] = 1
    with pytest.raises(Exception):
        DraftConfig.from_dict(raw)


def test_window_wider_than_context_is_full_attention():
    target = make_target(0)
    a_d, _ = make_draft(3, window=4096)
    b_d, _ = make_draft(3)
    a_d.bind(target), b_d.bind(target)
    for x, y in zip(_run(a_d, 30), _run(b_d, 30)):
        assert bool(mx.array_equal(x, y).item())


def test_window_mask_semantics():
    """Each block query at absolute position p sees exactly keys p-W+1..p."""
    W = 8
    d, _ = make_draft(4, window=W)
    d.bind(make_target(0))
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, scale, mask):
        out = real(q, k, v, scale=scale, mask=mask)
        if isinstance(mask, str):
            return out
        L = q.shape[2]
        assert k.shape[2] == L + W - 1
        kk = mx.repeat(k, q.shape[1] // k.shape[1], axis=1).astype(mx.float32)
        vv = mx.repeat(v, q.shape[1] // v.shape[1], axis=1).astype(mx.float32)
        qq = q.astype(mx.float32)
        rows = []
        for i in range(L):
            s = (qq[:, :, i : i + 1] @ kk[:, :, i : i + W].transpose(0, 1, 3, 2)) * scale
            rows.append(mx.softmax(s, axis=-1) @ vv[:, :, i : i + W])
        seen.append(float(mx.max(mx.abs(mx.concatenate(rows, axis=2) - out.astype(mx.float32))).item()))
        return out

    mx.fast.scaled_dot_product_attention = spy
    try:
        _run(d, 40, rounds=2)
    finally:
        mx.fast.scaled_dot_product_attention = real
    assert seen and max(seen) < 2e-2  # bf16 inputs, fp32 reference


def test_first_call_crop_matches_full_context():
    """A fresh cache keeps only the last W context rows; outputs match feeding it every row."""
    W = 16
    d, cfg = make_draft(5, window=W)
    d.bind(make_target(0))
    a = _run(d, 50, rounds=4, seed=3)
    d.config = dataclasses.replace(cfg, sliding_window=None)  # layers keep the window, no crop
    b = _run(d, 50, rounds=4, seed=3)
    d.config = cfg
    for x, y in zip(a, b):
        assert x.shape == y.shape
        diff = mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()
        assert diff <= 2e-2 * max(mx.max(mx.abs(y.astype(mx.float32))).item(), 1.0), diff
