"""MLX tests for dspine_mlx (Apple Silicon or CPU MLX). Skipped where mlx / mlx-lm are not installed."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

collect_ignore_glob = (
    [] if importlib.util.find_spec("mlx") and importlib.util.find_spec("mlx_lm") else ["test_*.py"]
)
