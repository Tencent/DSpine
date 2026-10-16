"""CPU tests for sampled-verify target distributions and config parsing.
Check both sampling backends against hand-computed cases and independent float64
references; CUDA-free renormalization uses the PyTorch fallbacks."""

import types

import pytest
import torch

from sglang_dspine.models.dspine import resolve_dspine_head_topk
from sglang_dspine.worker import build_dspine_target_probs

from ref_utils import ref_target_row, sampling_info


def _build(logits, T, info, backend):
    bs = logits.shape[0] // T
    return build_dspine_target_probs(
        next_token_logits=logits,
        sampling_info=info,
        block_size=T,
        bs=bs,
        sampling_backend=backend,
    )


def test_target_probs_hand_computed():
    # unfiltered p = [.4, .3, .2, .1] (logits log 4, log 3, log 2, log 1; temperature 1)
    logits = torch.log(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    f64 = torch.float64
    # joint (flashinfer, min-p off), top_k=2, top_p=.5: top-2 = {0, 1}; mass ahead on the
    # unfiltered p = [0, .4, .7, .9] -> nucleus {0, 1}; support {0, 1} -> [.4, .3] / .7 = [4/7, 3/7]
    got = _build(
        logits, 1, sampling_info(1, temperature=1.0, top_k=2, top_p=0.5), "flashinfer"
    )
    expected = torch.tensor([4 / 7, 3 / 7, 0, 0], dtype=f64)
    torch.testing.assert_close(got[0, 0].double(), expected, rtol=0, atol=1e-6)
    # sequential min-p branch (flashinfer, min-p on), top_k=3, top_p=.8, min_p=.6: top-3 renorm
    # [4/9, 3/9, 2/9, 0]; mass ahead [0, .444, .778] <= .8 keeps all three; min-p .6 * 4/9 = .267
    # drops 2/9 (.222) -> [4/7, 3/7, 0, 0]
    info = sampling_info(1, temperature=1.0, top_k=3, top_p=0.8, min_p=0.6)
    got = _build(logits, 1, info, "flashinfer")
    expected = torch.tensor([4 / 7, 3 / 7, 0, 0], dtype=f64)
    torch.testing.assert_close(got[0, 0].double(), expected, rtol=0, atol=1e-6)
    # pytorch backend, top-p only (.75): mass ahead [0, .4, .7, .9] keeps {0, 1, 2} -> [4/9, 3/9, 2/9, 0]
    got = _build(logits, 1, sampling_info(1, temperature=1.0, top_p=0.75), "pytorch")
    expected = torch.tensor([4 / 9, 3 / 9, 2 / 9, 0], dtype=f64)
    torch.testing.assert_close(got[0, 0].double(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend", ["flashinfer", "pytorch"])
@pytest.mark.parametrize(
    "temperature,top_k,top_p,min_p",
    [
        (0.8, None, None, 0.2),
        (0.6, 20, 0.95, None),
        (1.0, 7, None, None),
        (1.2, None, 0.6, None),
        (0.9, 30, 0.9, 0.05),
    ],
)
def test_target_probs_match_ordinary_sampler(backend, temperature, top_k, top_p, min_p):
    bs, T, V = 3, 4, 200
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(bs * T, V, generator=g) * 3
    info = sampling_info(
        bs, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p
    )
    got = _build(logits, T, info, backend).view(bs * T, V)
    for row in range(bs * T):
        ref = ref_target_row(
            logits[row],
            temperature,
            top_k or (1 << 30),
            top_p or 1.0,
            min_p or 0.0,
            backend,
        )
        assert torch.equal(got[row] > 0, ref > 0), (row, "support differs")
        torch.testing.assert_close(got[row].double(), ref, rtol=1e-5, atol=1e-7)


# A 5-layer draft: serial_table redoes the last write, so first_exact must be 5 - 1 = 4.
SERIAL = {"mode": "serial_table", "first_exact": 4, "head_topk": 16}
REFINE = {"kind": "exact_sender", "head_topk": 16}


def _draft_config(second_pass=None, refine=None):
    dflash_config = {}
    if refine is not None:
        dflash_config["selected_prefix_last_write_refine"] = refine
    config = types.SimpleNamespace(dflash_config=dflash_config)
    if second_pass is not None:
        config.code_bridge_second_pass = second_pass
    return config


def test_head_topk_from_either_source_or_both_agreeing():
    assert resolve_dspine_head_topk(_draft_config(second_pass=SERIAL), 5) == 16
    assert resolve_dspine_head_topk(_draft_config(refine=REFINE), 5) == 16
    both = _draft_config(second_pass=SERIAL, refine=REFINE)
    assert resolve_dspine_head_topk(both, 5) == 16


@pytest.mark.parametrize(
    "second_pass,refine",
    [
        (None, None),  # no width anywhere
        (SERIAL, {**REFINE, "head_topk": 8}),  # the two sources disagree
        ({**SERIAL, "mode": "restricted"}, None),  # another second-pass mode
        ({**SERIAL, "first_exact": 3}, None),  # not the last write
        ({**SERIAL, "extra": 1}, None),  # unknown key
        (None, {**REFINE, "kind": "restricted"}),  # another refinement kind
        (None, {**REFINE, "head_topk": 12}),  # not a power of two
        (None, {**REFINE, "head_topk": 128}),  # wider than the kernels' 64 lanes
    ],
)
def test_head_topk_refuses_missing_ambiguous_or_unsupported(second_pass, refine):
    with pytest.raises(ValueError):
        resolve_dspine_head_topk(_draft_config(second_pass, refine), 5)
