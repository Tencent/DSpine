# SPS tuning

Tune verification budgets for your hardware and workload using the same serving
settings as deployment.

## 1. Prepare requests

Save representative tokenized prefixes in `prefixes.jsonl`:

```json
{"task_id": "chat-001", "input_ids": [151644, 8948, 198], "sampling_params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}
```

Use at least three independent tasks covering your context lengths and sampling
settings. Keep the same `task_id` across a task's turns. To tune both greedy and
temperature sampling, include requests for both modes (`temperature=0` for greedy).
The tuner reserves one third of tasks for validation.

## 2. Tune

Define your deployment command without host, port or SPS profile options; the
tuner supplies those:

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
  --requests prefixes.jsonl --output sps.json \
  --batch-sizes 1 8 16 32 \
  -- "${serve[@]}"
```

Set `--batch-sizes` to cover the batches you expect, including smaller batches
as requests finish. Keep the serving capacity at least as large as the maximum.
Defaults are three repetitions and an output limit of 4096 tokens; see `--help`
for other options.

The tuner saves the policy to `sps.json` and measurements to `sps.runs/`.
Failed validation retains the baseline policy.

## 3. Serve

```bash
CUDA_VISIBLE_DEVICES=0 "${serve[@]}" \
  --speculative-dspark-sps-table-path sps.json
```

Unmeasured batches retain the baseline, which defaults to full-block verification.
Retune when changing hardware, models, serving settings or workload.
