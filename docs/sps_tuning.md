# SPS tuning

Tune verification budgets on your deployment GPU using representative conversation
prefixes. The draft block size stays at 16. Tuning selects a total token budget
for each measured batch size, where
[DSpark](https://arxiv.org/abs/2607.05147) allocates verification by draft confidence.

## Prepare requests

Save tokenized prefixes and sampling settings as JSONL:

```json
{"task_id": "chat-001", "input_ids": [151644, 8948, 198], "sampling_params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}
```

Cover your deployment context lengths, tasks, reasoning levels and sampling
settings. Include prefixes from the beginning, middle and later parts of long
responses. Keep the same `task_id` for all turns and prefixes of one task.
The tuner separates requests into greedy (near-zero temperature or `top_k=1`) and
sampled families and measures them independently. A row without
`sampling_params` defaults to greedy; an existing `sampling_params` object
without `temperature` follows the serving default of 1. Each family's prefixes
cycle to fill the requested native batch size, so a family can have fewer
prefixes than the largest batch. More representative distinct prefixes improve
the workload coverage. By default, one third of tasks are reserved for validation.
Use `--holdout-requests holdout.jsonl` to provide a separate validation file.
Tasks and identical input prefixes must be disjoint across the two files.

Each row may set `sampling_params.max_new_tokens` for a shorter continuation;
the CLI's `--max-new-tokens` is the upper bound. For example, tune on short
continuations at varied decode positions and validate full responses from
independent task prompts.

To fit a confidence map, collect full-block rounds with `confidence`, `accepted`,
`conversation_id` and `mode` (`greedy` or `sampled`). `accepted` excludes the anchor
and bonus token. The fitter uses NumPy and SciPy:

```bash
python scripts/fit_calibration.py rounds.jsonl \
  --output calibration.json --report calibration_metrics.json
```

Pass `--confidence-calibration calibration.json` to the tuner to embed the map in
its profiles. Fitting and selection use conversation-disjoint calibration and
validation rounds; reserve test requests for evaluating the frozen profile.

## Tune

Use the same serving settings for tuning and deployment. Define your launch
command once:

```bash
serve=(
  sglang serve
  --model-path /path/to/Qwen3.8-27B
  --speculative-algorithm DSPINE
  --speculative-draft-model-path tencent/Qwen3.8-27B-DSpine-b16
  --speculative-num-draft-tokens 16
  --tp-size 1 --dtype bfloat16
  --attention-backend fa3 --speculative-draft-attention-backend fa3
  --linear-attn-backend triton --enable-linear-replayssm-spec
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer
  --max-mamba-cache-size 256 --mem-fraction-static 0.80
  --context-length 131072 --max-running-requests 32
  --cuda-graph-max-bs-decode 32 --min-free-slots-delay 1
  --sampling-backend flashinfer --reasoning-parser qwen3
  --tool-call-parser qwen3_coder --trust-remote-code
)

CUDA_VISIBLE_DEVICES=0 python scripts/tune_sps.py \
  --requests /path/to/prefixes.jsonl --output /path/to/sps.json \
  --batch-sizes 1 8 16 32 \
  -- "${serve[@]}"
```

The tuner supplies the host, port and SPS profile options.

The sweep tests average widths 4, 8, 12 and 16, snapped to the server's captured
token tiers. It changes one actual-batch entry at a time while keeping the rest
of the table fixed. Greedy and sampled policies are tuned independently.
Use `--baseline-profile current-sps.json` to refine an existing table; otherwise
the baseline verifies full blocks. Unmeasured entries retain their baseline.

Candidates use paired requests and seeds, three repetitions in alternating
order, and identical cache-reset and warmup steps. The default minimum is
128 requests and 30 seconds per timed pass. Short passes cause the same queue
to grow for every candidate. Periodic native decode logs confirm that the
requested actual batch size was observed.

Scores are **pooled output tokens / pooled queue wall time**. A change needs a
material gain with consistent direction across repeats; approximately tied
candidates prefer the wider budget. Independent validation then checks adjacent
budgets and the complete table against the baseline. Failed validation retains
the baseline. The output records `VALIDATED`, `BASELINE_RETAINED` or
`INSUFFICIENT_EVIDENCE` along with all measurements.

- `sps.json`: the deployable policy and tuning status.
- `sps.runs/`: task partitions, per-trial profiles, native outputs, timings,
  cache hits, acceptance counters and the complete `RESULTS.json` report.

Throughput includes prefill and request overhead. Acceptance length is pooled
accepted draft tokens / verify calls + 1. Use `--help` to adjust the sweep and
response length (default: 4096 tokens). Generation requests have no client timeout
by default; set `--request-timeout SECONDS` when a limit is needed.

## Serve

Launch the same command with the generated profile:

```bash
CUDA_VISIBLE_DEVICES=0 "${serve[@]}" \
  --speculative-dspark-sps-table-path /path/to/sps.json
```

The file uses DSpine's v2 policy format and records the GPU, policies and tuning
status. Input digests and launch records are saved in `sps.runs/`.
Its `mode_policies.greedy` and `mode_policies.sampled` entries contain the
independently measured policies; the root `policy` retains the baseline fallback:

```json
{
  "policy": {"verify_widths": {}, "token_budgets": {}},
  "mode_policies": {
    "greedy": {"verify_widths": {}, "token_budgets": {"8": 64}},
    "sampled": {"verify_widths": {}, "token_budgets": {"8": 96}}
  }
}
```

Existing profiles without `mode_policies` retain their root-policy behavior.
Retune when changing the GPU, model, runtime or workload. Without a profile,
verification uses the full 16-token block. Only the specified batch sizes are
tuned; other batch sizes and missing request families retain the baseline,
which defaults to the full block. Mixed greedy/temperature batches use full-block verification;
temperature, top-k, top-p and min-p remain per-request sampling settings.
Ragged budgets use CUDA Graphs when they exactly match an existing token tier;
other budgets run eagerly without padding the verification layout.

Profiles with `confidence_calibration` use separate greedy and sampled maps.
The selected lookup tables load onto the GPU at startup, and planner graphs are
cached separately for each mode. Temperature, top-k, top-p and min-p continue to
apply to individual requests; calibration evidence records the sampled settings
without restricting runtime to those settings. Profiles without this field use
the packaged calibration artifact with its greedy and sampled maps.
