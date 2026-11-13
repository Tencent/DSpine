"""Sampled-prefix confidence is predictable and leaves proposal sampling intact."""

from unittest.mock import patch

import pytest
import torch

from sglang_dspine.kernels import dspine as K


def _inputs(bs, *, width=16, k=16, device="cpu", temperature=1.0):
    n = bs * (width - 1)
    table = torch.randn((n, k, k), device=device) * 2
    candidates = torch.arange(k, device=device).expand(n, k).contiguous()
    greedy_tokens = candidates[:, 0].clone()
    greedy = torch.zeros(bs, device=device, dtype=torch.int32)
    if bs:
        greedy[-1] = 1
    return (
        table,
        torch.full((bs,), 1 / temperature, device=device),
        torch.tensor([3, k, 7], device=device, dtype=torch.int32).repeat(bs)[:bs],
        torch.tensor([0.65, 0.95, 1.0], device=device).repeat(bs)[:bs],
        torch.tensor([0.05, 0.0, 0.2], device=device).repeat(bs)[:bs],
        greedy,
        torch.rand(n, device=device),
        candidates,
        greedy_tokens,
        torch.empty(n, device=device, dtype=torch.int64),
        torch.empty((n, k), device=device),
        width,
    )


class _Launch:
    def __init__(self):
        self.kwargs = None

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.kwargs = kwargs
        return run


def test_optional_confidence_keeps_the_existing_call_contract():
    args = _inputs(3)
    launch = _Launch()
    state = torch.get_rng_state().clone()
    with patch.object(K, "_serial_sample_chain", launch):
        result = K.serial_sample_chain(*args)
        assert result == (args[-3], args[-2])
        assert launch.kwargs["WITH_CONFIDENCE"] is False
        confidence = torch.full((3 * 15,), -0.25)
        K.serial_sample_chain(*args, confidence_out=confidence)
        assert launch.kwargs["WITH_CONFIDENCE"] is True
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("invalid", ["dtype", "capacity", "strided", "device"])
def test_confidence_storage_contract_is_checked_before_launch(invalid):
    args = _inputs(3)
    n = 3 * 15
    confidence = {
        "dtype": lambda: torch.empty(n, dtype=torch.float64),
        "capacity": lambda: torch.empty(n - 1),
        "strided": lambda: torch.empty(n * 2)[::2],
        "device": lambda: torch.empty(n, device="meta"),
    }[invalid]()
    with pytest.raises(ValueError, match="confidence needs contiguous float32"):
        K.serial_sample_chain(*args, confidence_out=confidence)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("bs", [3, 9, 17, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.4])
def test_confidence_matches_actual_prefix_softmax_without_changing_sampling(
    bs, temperature
):
    torch.manual_seed(107)
    args = _inputs(bs, device="cuda", temperature=temperature)
    n, k = args[7].shape
    baseline_tokens = torch.empty_like(args[-3])
    baseline_q = torch.empty_like(args[-2])
    baseline_args = (*args[:-3], baseline_tokens, baseline_q, args[-1])
    confidence = torch.rand(n + 7, device="cuda")
    initial_confidence = confidence.clone()
    uniforms = args[6].clone()
    state = torch.cuda.get_rng_state().clone()
    K.serial_sample_chain(*baseline_args)
    K.serial_sample_chain(*args, confidence_out=confidence)
    assert torch.equal(args[-3], baseline_tokens)
    assert torch.equal(args[-2].view(torch.int32), baseline_q.view(torch.int32))
    assert torch.equal(args[6], uniforms)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    sampled = (args[5] == 0).repeat_interleave(args[-1] - 1)
    # Reconstruct each row using its actual sampled predecessor, independently
    # of the filtered q that verification receives.
    table, inv_temperature = args[0].cpu(), args[1].cpu()
    candidates, tokens = args[7].cpu(), args[-3].cpu()
    expected = torch.empty(n)
    depth = args[-1] - 1
    for b in range(bs):
        if args[5][b]:
            continue
        sender = 0
        for i in range(depth):
            row = b * depth + i
            expected[row] = torch.softmax(
                table[row, sender] * inv_temperature[b], dim=0
            ).max()
            sender = int((candidates[row] == tokens[row]).nonzero().item())
    torch.testing.assert_close(
        confidence[:n][sampled].cpu(), expected[sampled.cpu()], rtol=1e-5, atol=1e-6
    )
    assert bool((confidence[:n][sampled] <= args[-2].amax(-1)[sampled] + 1e-6).all())
    assert torch.equal(confidence[:n][~sampled], initial_confidence[:n][~sampled])
    assert torch.equal(confidence[n:], initial_confidence[n:])


@cuda
def test_confidence_depends_on_sampled_ancestors_not_the_current_draw():
    width, k = 4, 4
    distributions = torch.tensor(
        [[0.899, 0.099, 0.001, 0.001],
         [0.4, 0.3, 0.2, 0.1],
         [0.6, 0.2, 0.1, 0.1],
         [0.25, 0.25, 0.25, 0.25]], device="cuda"
    )
    table = distributions.log().expand(width - 1, k, k).clone()
    table[0] = distributions[1].log().expand(k, k)
    candidates = torch.arange(10, 10 + k, device="cuda").expand(width - 1, k).contiguous()

    def run(draws):
        tokens = torch.empty(width - 1, device="cuda", dtype=torch.int64)
        q = torch.empty((width - 1, k), device="cuda")
        confidence = torch.full((width - 1,), -1.0, device="cuda")
        K.serial_sample_chain(
            table,
            torch.ones(1, device="cuda"),
            torch.full((1,), k, device="cuda", dtype=torch.int32),
            torch.ones(1, device="cuda"),
            torch.zeros(1, device="cuda"),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor(draws, device="cuda"),
            candidates,
            candidates[:, 0].clone(),
            tokens,
            q,
            width,
            confidence_out=confidence,
        )
        return tokens, q, confidence

    first = run([0.05, 0.2, 0.4])
    changed_first_draw = run([0.55, 0.2, 0.4])
    changed_second_draw = run([0.05, 0.95, 0.4])
    assert first[0][0] != changed_first_draw[0][0]
    assert first[2][0] == changed_first_draw[2][0]
    assert first[2][1] != changed_first_draw[2][1]
    assert first[0][1] != changed_second_draw[0][1]
    assert first[2][1] == changed_second_draw[2][1]
    assert first[2][2] != changed_second_draw[2][2]


@cuda
def test_prefilter_confidence_retains_information_when_filters_make_q_one_hot():
    p = torch.tensor(
        [[0.85, 0.14, 0.009, 0.001], [0.99, 0.008, 0.001, 0.001]],
        device="cuda",
    )
    k, width = p.shape[1], 2
    temperature = 0.7
    table = (p.log() * temperature).unsqueeze(1).expand(2, k, k).contiguous()
    confidence = torch.empty(2, device="cuda")
    q = torch.empty((2, k), device="cuda")
    K.serial_sample_chain(
        table,
        torch.full((2,), 1 / temperature, device="cuda"),
        torch.full((2,), 5, device="cuda", dtype=torch.int32),
        torch.full((2,), 0.8, device="cuda"),
        torch.full((2,), 0.05, device="cuda"),
        torch.zeros(2, device="cuda", dtype=torch.int32),
        torch.tensor([0.2, 0.8], device="cuda"),
        torch.arange(k, device="cuda").expand(2, k).contiguous(),
        torch.zeros(2, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        q,
        width,
        confidence_out=confidence,
    )
    torch.testing.assert_close(q.amax(-1), torch.ones(2, device="cuda"))
    torch.testing.assert_close(confidence, p.amax(-1), rtol=1e-5, atol=1e-6)
    assert confidence[0] < confidence[1]
