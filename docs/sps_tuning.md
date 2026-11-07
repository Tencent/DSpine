# SPS tuning

Tune verification budgets on your deployment GPU using representative conversation
prefixes. The draft block size stays at 16. Tuning selects a uniform width for
batch sizes 1–7 and a total budget for batch size 8, where
[DSpark](https://arxiv.org/abs/2607.05147) allocates verification by draft confidence.

## Prepare requests

Save tokenized prefixes and sampling settings as JSONL:

```json
{"input_ids": [151644, 8948, 198], "sampling_params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}
```

Use at least eight distinct prefixes covering your context lengths, tasks and
reasoning levels. Tune greedy (`temperature=0`) and temperature sampling separately.

## Tune

Run on one idle GPU. Pass your serving command after `--`; the tuner supplies the
host, port and SPS profile options.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tune_sps.py \
  --requests /path/to/prefixes.jsonl --output /path/to/sps.json \
  -- \
  sglang serve \
  --model-path /path/to/Qwen3.8-27B \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path /path/to/Qwen3.8-27B-DSpine-b16 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 --dtype bfloat16 \
  --attention-backend fa3 --speculative-draft-attention-backend fa3 \
  --linear-attn-backend triton --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 64 --mem-fraction-static 0.80 \
  --context-length 131072 --max-running-requests 8 \
  --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1 \
  --sampling-backend flashinfer --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --trust-remote-code
```

The default sweep tests widths 4, 8, 12 and 16 at batch sizes 1–8. Each candidate
warms the prefixes, runs three timed repetitions, and is scored by median
**total output tokens / total request wall time**. The selected combination is
replayed once before saving.

- `sps.json`: selected policy and measurements.
- `sps.runs/`: server logs and per-request outputs, timings, cache hits and acceptance counters.

Throughput includes prefill and request overhead. Acceptance length is pooled
accepted draft tokens / verify calls + 1. Use `--help` to adjust the sweep and
response length (default: 512 tokens).

## Serve

Add the generated profile to your serving command:

```bash
--speculative-dspark-sps-table-path /path/to/sps.json
```

The file uses DSpine's policy format and records the GPU, launch command and input
digest. Retune when changing the GPU, model, runtime or workload. Without a profile,
verification uses the full 16-token block.
