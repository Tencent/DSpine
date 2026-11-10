# TP1 large-batch validation

This record summarizes the regression coverage for TP1 batches above eight.

## Behavior

The request-capacity restriction of eight is removed for TP1. Small sampled
batches keep their captured proposal graphs. Larger sampled batches compute the
same factorized conditional logits eagerly in chunks of at most eight requests,
reusing the existing workspace. No workspace or graph is retained for every
possible large batch size. Cache mutation and inference-stream checks remain.

Above batch eight, verification uses the complete B16 envelope. The existing
target-sampling and ReplaySSM eligibility checks select their native fallbacks.
TP greater than one remains unsupported.

## CUDA Graph boundaries

The large-batch fallback does not disable CUDA Graphs for the entire decoder:

| Stage | Batch above eight |
| --- | --- |
| Draft-model forward | Can replay configured model graphs, subject to backend and shape eligibility |
| Sampled proposal probability table | Eager factorized calculation in chunks of at most eight; no large-batch proposal-table graph |
| Full B16 target verification | Can replay configured model graphs, subject to backend and shape eligibility |
| SPS shortened/ragged verification | Not enabled for these larger batches |

Using native ReplaySSM and target-sampling fallbacks does not by itself force
model forwards into eager mode. Sampling work outside model graphs is a separate
stage.

The serving log recorded target and draft model graph capture for batch sizes
`[1, 2, 4, 8, 16, 32]`. Decode traces for actual batches 9, 16 and 17 reported
`cuda graph: True`. The separate `batch=32 chunk=8` witness establishes an actual
batch-32 sampled proposal using the eager table path. Per-stage graph replay
counters for batch 32 were not collected; these observations should not be
described as an end-to-end graph-only batch-32 execution.

## Numerical and routing tests

The full test suite passed: **125 passed, 6 skipped**. The skipped cases were the
two opt-in server modules (run separately where applicable) and four
Triton-interpreter-only cases. The dependency-free hosted-CI selection also
passed: **10 passed, 7 deselected**. All repository pre-commit checks passed.

`tests/test_large_batch.py` compares captured and eager proposal tables against
direct FP64 refinement, including live-input changes after capture, batch 9, 16,
17 and 32, and transitions back to small batches. It also covers sampler capacity
below eight, full-width verification, sparse-target bypass, native ReplaySSM
dispatch and proposal-cache mutation rejection.

The suite also corrects a pre-existing seeded-sampler integration test that sent
CPU tensors through FlashInfer's CUDA filters; that CPU test now selects the
PyTorch backend.

## Real serving test

`tests/test_large_batch_e2e.py` passed with CUDA graphs enabled. For each of greedy,
sampled and mixed sampling, it submitted batches of 8, 9, 16, 17, 32 and 8:
18 cases, 270 requests and 17,280 generated tokens. Every request completed its
64-token budget and reported speculative verification rounds. The sampled cases
included joint top-k/top-p; mixed cases additionally exercised min-p.

The log confirmed `DSpine eager proposal table: batch=32 chunk=8`, demonstrating
an actual batch-32 sampled proposal. Request completion also exercised shrinking
decode batches. The recurrent-state pool must cover the requested concurrency,
including prefix-cache and speculative state. The request limit alone is
insufficient to size this pool.

Run the test with the command in [CONTRIBUTING.md](../CONTRIBUTING.md).

Suggested issue update:

> The TP1 large-batch fix has passed regression tests, including actual batch-32
> decoding across greedy, sampled, and mixed workloads. For batches above eight,
> sampled proposal probability tables use an eager chunked fallback; model
> forwards can still use CUDA Graphs. Large-batch SPS optimization and TP support
> remain separate follow-ups.

## Limits

This is a correctness and serving smoke regression, not a throughput comparison
against AR or the previous batch-eight configuration. It does not establish
byte-for-byte agreement across different greedy batch shapes, statistical
equivalence of complete output distributions, long-context capacity, TP support,
or portability across runtime configurations. Small-batch SPS policy tuning is
unchanged; larger-batch SPS tuning is a separate performance milestone.
