# DSpine for Apple Silicon (MLX)

[DSpine](https://arxiv.org/abs/2609.36173) speculative decoding for Qwen3-8B on
Apple Silicon: a pure-MLX single-stream generator plus an optional SGLang-MLX
serving plugin, using the public
[tencent/Qwen3-8B-DSpine-b16](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16)
draft with an unmodified `mlx-lm` target.

## Installation

Requirements:

- Apple Silicon Mac (M1+), macOS 14+, Python 3.11+
- `mlx >= 0.32`, `mlx-lm >= 0.32`
- about 21 GB of unified memory for a bf16 Qwen3-8B target
- for serving: SGLang v0.5.18 built from source with the MPS extras

Single-stream generation, from the repository root:

```bash
pip install -e dspine_mlx
```

SGLang serving. PyPI has no macOS wheels for `sglang==0.5.18`, so build SGLang
from source first:

```bash
git clone --branch v0.5.18 https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
python3.12 -m venv .venv && source .venv/bin/activate
SGLANG_BUILD_RUST_EXTS=none pip install -e "python[srt_mps]"

# Back in the DSpine repository root
pip install -e "dspine_mlx[sglang]"
```

If the source checkout reports a version other than 0.5.18, use
`pip install -e dspine_mlx --no-deps`. The CUDA plugin in `sglang_dspine/`
(SGLang 0.5.20) needs a separate environment.

## Usage

Single-stream generation:

```bash
dspine-generate \
  --target Qwen/Qwen3-8B \
  --draft tencent/Qwen3-8B-DSpine-b16 \
  --block-size 8 --max-tokens 256 \
  --prompt "..."
```

Common options: `--block-size`, `--temperature/--top-k/--top-p/--min-p`,
`--raw` (no chat template), `--profile`.

SGLang serving:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path tencent/Qwen3-8B-DSpine-b16 \
  --speculative-num-draft-tokens 8 \
  --max-running-requests 8 \
  --host 127.0.0.1 --port 30000
```

`--speculative-num-draft-tokens` is the verify block size (default: the draft's
trained block size). The SGLang plugin supports full-attention Qwen3 targets only; overlap
scheduling, CUDA graphs and the radix cache are disabled.

## Qwen3.8-27B (hybrid target)

[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)
runs with any MLX build of Qwen3.8-27B (`qwen3_5`: 48 GatedDeltaNet + 16
full-attention layers). Two pieces make this work:

- **Exact rollback for GatedDeltaNet layers** (`dspine_mlx/hybrid.py`). Their
  recurrent state lives in an `ArraysCache` that cannot be trimmed. During a
  verify pass each linear layer records its starting state and recurrence
  inputs; when only part of the block is accepted, `TargetAdapter.trim`
  rebuilds the conv state by slicing and replays `gated_delta_update` over the
  accepted steps. Nothing else is recomputed, and the state equals a plain
  decode of the accepted prefix.
- **The draft's 8192-token sliding window.** A fresh draft cache is fed only
  the last 8192 context rows, which is all the window can see.

Prefill runs in 2048-token chunks and keeps logits for the last position only,
so long prompts don't hold a full-vocabulary logit tensor in memory.

```bash
dspine-generate \
  --target mlx-community/Qwen3.8-27B-4bit \
  --draft tencent/Qwen3.8-27B-DSpine-b16 \
  --block-size 5 --quantize-draft --draft-bits 4 \
  --max-tokens 512 --baseline --prompt "..."
```

On an Apple GPU a quantized 27B forward costs about the same for 1 to 5 tokens
and then steps up (4-bit: 34 ms for 1 token, 42 ms for 5, 89 ms for 16), so a
verify block of 5 is fastest on quantized targets; bf16 keeps 16. A 4-bit
draft proposes almost the same tokens as bf16 and costs less every round.

MacBook Pro M5 Max (128 GB), greedy, 512 new tokens, 10 prompts (GSM8K,
MATH-500, HumanEval, MBPP, MT-Bench), thinking on, decode tok/s:

| Target | mlx-lm AR | DSpine | Speedup | Block |
|---|---:|---:|---:|---:|
| `mlx-community/Qwen3.8-27B-4bit` | 31.9 | **64.7** | **2.03x** | 5 |
| `mlx-community/Qwen3.8-27B-8bit` | 17.9 | **52.1** | **2.91x** | 5 |
| bf16 | 9.1 | **39.1** | **4.31x** | 16 |

With a 33k-token prompt (thinking off, 256 new tokens) the 4-bit target goes
from 28.3 to 57.3 tok/s (2.02x) and 8-bit from 16.7 to 47.7 tok/s (2.85x).
`tests/mlx/` checks the rollback against a direct decode and DSpine greedy against
plain greedy on a tiny random `qwen3_5` model (`pytest tests/mlx`).

## Results

**Single-stream MLX.** Qwen3-8B bf16, Apple M4 Pro (16-core GPU, 64 GB), block
size 16. Cells show decode-only tok/s (speedup over `mlx-lm` AR / accept length).

| Temp. | Method | MATH-500 | HumanEval | MBPP | LCB | MT-Bench | Arena-Hard | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T=0 | mlx-lm AR | 15.94 | 15.75 | 15.17 | 15.05 | 15.15 | 15.13 | 15.37 |
| T=0 | DFlash<sup>†</sup> | 39.90 (2.50x / 6.45) | 37.90 (2.41x / 6.18) | 32.19 (2.12x / 5.50) | 31.29 (2.08x / 5.52) | 18.62 (1.23x / 3.19) | 18.93 (1.25x / 3.23) | 29.81 (1.93x / 5.01) |
| T=0 | **DSpine** | **53.26 (3.34x / 8.74)** | **48.60 (3.09x / 8.02)** | **41.11 (2.71x / 7.07)** | **37.78 (2.51x / 6.69)** | **24.64 (1.63x / 4.23)** | **25.73 (1.70x / 4.41)** | **38.52 (2.50x / 6.53)** |
| T=1 | mlx-lm AR | 15.11 | 15.10 | 15.16 | 15.03 | 15.12 | 15.11 | 15.11 |
| T=1 | DFlash<sup>†</sup> | 24.73 (1.64x / 4.21) | 25.50 (1.69x / 4.35) | 23.14 (1.53x / 3.95) | 24.44 (1.63x / 4.33) | 14.03 (0.93x / 2.41) | 13.37 (0.88x / 2.29) | 20.87 (1.38x / 3.59) |
| T=1 | **DSpine** | **42.93 (2.84x / 7.71)** | **40.43 (2.68x / 7.24)** | **37.57 (2.48x / 6.71)** | **35.31 (2.35x / 6.54)** | **23.00 (1.52x / 4.12)** | **22.82 (1.51x / 4.08)** | **33.68 (2.23x / 6.07)** |

<sup>†</sup> [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16).

**SGLang MLX serving.** SGLang v0.5.18 MLX backend, Qwen3-8B bf16, Apple M4 Pro
(64 GB), greedy decoding, block size 8.

| Conc. | Method | MATH-500 | HumanEval | MBPP | Avg. |
|---:|---|---:|---:|---:|---:|
| 1 | AR baseline (tok/s) | 14.72 | 14.70 | 14.90 | 14.77 |
| 1 | DSpine (tok/s) | 64.87 | 58.28 | 58.22 | 60.46 |
| 1 | **Speedup** | **4.41x** | **3.96x** | **3.91x** | **4.09x** |
| 4 | AR baseline (tok/s) | 47.18 | 48.47 | 50.01 | 48.55 |
| 4 | DSpine (tok/s) | 104.47 | 95.16 | 94.50 | 98.04 |
| 4 | **Speedup** | **2.21x** | **1.96x** | **1.89x** | **2.02x** |
