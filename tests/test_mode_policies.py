"""Mode selection happens before proposal draws and preserves legacy defaults."""

import contextlib
import json
import sys
from types import SimpleNamespace

import pytest

from sglang_dspine.verification import caps
from sglang_dspine.verification.policy import SpsPolicy
from sglang_dspine.verification.ragged import RaggedVerifyCaps


def controller(monkeypatch, tmp_path, modes=None, fallback=None, cls=RaggedVerifyCaps):
    fallback = fallback or SpsPolicy()
    path = tmp_path / "sps.json"
    data = {"format": "dspine-sps-v2", "hardware": {}, "policy": fallback.to_dict()}
    if modes is not None:
        data["mode_policies"] = {mode: policy.to_dict() for mode, policy in modes.items()}
    path.write_text(json.dumps(data))
    captured = []

    class Graph:
        def __init__(self, runner=None, **kwargs):
            if runner is not None:
                captured.append(kwargs["speculative_num_draft_tokens"])
            self.capture_num_tokens = [16, 32, 48, 64, 96, 128]

        def _ragged_capture_slots(self, tier):
            return min(tier, 32)

    backend = SimpleNamespace(needs_cpu_seq_lens=False, supports_ragged_verify_graph=True)
    runner = SimpleNamespace(attn_backend=backend, decode_cuda_graph_runner=Graph(), init_new_workspace=None, _get_attention_backend=lambda **kwargs: backend)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.cuda_graph_config", SimpleNamespace(Backend=SimpleNamespace(DISABLED="disabled")))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.runner.decode_cuda_graph_runner", SimpleNamespace(DecodeCudaGraphRunner=Graph))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(
        get_spec=lambda: SimpleNamespace(speculative_dspark_sps_table_path=path),
        get_parallel=lambda: SimpleNamespace(attn_dp_enabled=False, pp_size=1),
        get_exec=lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")))),
    ))
    monkeypatch.setattr(caps, "spec_width", lambda width: contextlib.nullcontext())
    monkeypatch.setattr("sglang_dspine.verification.policy.hardware_identity", lambda device: {})
    worker = SimpleNamespace(block_size=16, device="cpu", target_worker=SimpleNamespace(model_runner=runner))
    return cls(worker), captured, path


def batch(greedy, bs=3, seeds=None):
    return SimpleNamespace(
        batch_size=lambda: bs,
        sampling_info=SimpleNamespace(is_all_greedy=greedy, is_any_greedy=greedy),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        has_grammar=False, return_logprob=False,
        reqs=[SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed)) for seed in (seeds or [])],
    )


def test_same_batch_switches_between_full_greedy_and_sampled_budget(monkeypatch, tmp_path):
    modes = {"greedy": SpsPolicy(), "sampled": SpsPolicy(token_budgets={3: 16})}
    control, captured, _ = controller(monkeypatch, tmp_path, modes)
    assert captured == [16] and control.ragged_backend is not None
    for greedy, expected in ((True, 48), (False, 16), (True, 48)):
        control.begin(batch(greedy))
        assert control.tokens == expected and control.in_round
        assert control.ragged_round == (not greedy)
        assert control.policy is control.mode_policies["greedy" if greedy else "sampled"]
    control.begin(batch(False, seeds=[7, 7, 8]))
    assert not control.in_round and not control.ragged_round and control.active == 16
    mixed = batch(False)
    mixed.sampling_info.is_any_greedy = True
    control.begin(mixed)
    assert not control.in_round and control.active == 16


def test_mode_specific_shortened_budgets_switch_on_the_same_batch(monkeypatch, tmp_path):
    modes = {"greedy": SpsPolicy(token_budgets={3: 32}), "sampled": SpsPolicy(token_budgets={3: 16})}
    control, _, _ = controller(monkeypatch, tmp_path, modes)
    for greedy, expected in ((True, 32), (False, 16), (True, 32)):
        control.begin(batch(greedy))
        assert control.tokens == expected and control.ragged_round
    control.begin(batch(False, bs=7))
    assert control.tokens == 112 and not control.ragged_round


def test_missing_mode_uses_full_fallback_and_legacy_policy_is_shared(monkeypatch, tmp_path):
    control, _, path = controller(monkeypatch, tmp_path, {"sampled": SpsPolicy(token_budgets={3: 16})})
    control.begin(batch(True))
    assert control.tokens == 48 and not control.ragged_round
    fallback, modes = SpsPolicy.load_with_modes(path, {})
    assert fallback == SpsPolicy() and set(modes) == {"sampled"}
    legacy, _, _ = controller(monkeypatch, tmp_path, fallback=SpsPolicy(token_budgets={3: 16}))
    assert not legacy.mode_policies
    for greedy in (True, False):
        legacy.begin(batch(greedy))
        assert legacy.tokens == 16 and legacy.ragged_round


def test_target_width_capture_is_union_of_live_mode_policies(monkeypatch, tmp_path):
    modes = {"greedy": SpsPolicy(verify_widths={3: 8}), "sampled": SpsPolicy(verify_widths={3: 12})}
    control, captured, _ = controller(monkeypatch, tmp_path, modes, cls=caps.VerifyCaps)
    assert control.widths == [8, 12, 16] and set(control.states) == {8, 12, 16}
    assert captured == [8, 12]
    for greedy, width in ((True, 8), (False, 12), (True, 8)):
        control.begin(batch(greedy))
        assert control.active == width


def test_mode_profile_retains_explicit_root_fallback(monkeypatch, tmp_path):
    fallback = SpsPolicy(token_budgets={3: 32})
    control, _, _ = controller(monkeypatch, tmp_path, {"sampled": SpsPolicy(token_budgets={3: 16})}, fallback)
    for greedy, expected in ((True, 32), (False, 16), (True, 32)):
        control.begin(batch(greedy))
        assert control.tokens == expected


def test_profile_rejects_unknown_mode(tmp_path):
    path = tmp_path / "sps.json"
    path.write_text(json.dumps({"format": "dspine-sps-v2", "hardware": {}, "policy": {}, "mode_policies": {"temperature_0.7": {}}}))
    with pytest.raises(ValueError, match="mode_policies"):
        SpsPolicy.load_with_modes(path, {})
