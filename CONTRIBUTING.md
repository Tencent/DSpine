# Contributing

Open an issue with a minimal reproduction, or link the relevant issue in your PR.
Keep unrelated fixes and performance changes in separate commits. Use `Refs #N`
for partial work; use `Fixes #N` only when the issue is fully resolved.

## Local checks

The repository already provides shared Git hooks:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks check formatting, imports, undefined names, spelling, YAML and TOML.
GitHub Actions runs the same checks on PRs, so checks also run for contributors
who have not installed the local hooks.

## Tests

Install the plugin into a Python 3.11+ CUDA environment using the exact dependency
versions in `pyproject.toml`:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

Without CUDA, GPU tests skip. The lightweight hosted CI job runs only the
dependency-free SPS policy tests; its success does not establish GPU correctness.
Changes to kernels, sampling, CUDA graphs or recurrent state must include GPU
validation. Report the GPU, dependency versions, command, results and limitations
in the PR. Do not run unreviewed fork code automatically on private GPU runners.

The Qwen3.8 large-batch serving test is opt-in:

```bash
DSPINE_LARGE_BATCH_E2E=1 \
DSPINE_TARGET_MODEL=/path/to/Qwen3.8-27B \
DSPINE_DRAFT_MODEL=/path/to/Qwen3.8-27B-DSpine-b16 \
python -m pytest tests/test_large_batch_e2e.py -v -s
```

It starts and stops its own local TP1 server and requires a GPU with enough memory
for the BF16 target, draft and batch-32 caches.

## Maintainer merge settings

After the Checks workflow has run, repository administrators can create a `main`
ruleset requiring PRs, one approving review, resolved conversations and the
`Formatting and static checks` and `SPS policy tests (CPU)` status checks. Block
force pushes and branch deletion. These settings are configured on GitHub; adding
this file does not enable them. Keep GPU results as an explicit review requirement
until a suitable trusted GPU CI runner is available.
