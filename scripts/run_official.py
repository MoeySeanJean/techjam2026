"""Run the organizer's benchmark against our submission, without editing it.

`torch_transformer_benchmark.py` stays byte-for-byte as delivered. We import it,
swap the `UserOptimizedTransformer` symbol for ours, and hand argv straight
through, so every flag the organizer documents works unchanged:

    python scripts/run_official.py --batch-size 8 --seq-len 128 --dtype float32
    python scripts/run_official.py --seq-len 2048 --causal --padding-ratio 0.4
    python scripts/run_official.py --dtype float16 --layers 12
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch_transformer_benchmark as bench  # noqa: E402
from submission import UserOptimizedTransformer  # noqa: E402

bench.UserOptimizedTransformer = UserOptimizedTransformer

if __name__ == "__main__":
    os.environ.setdefault("KERNELFORGE_VERBOSE", "1")
    raise SystemExit(bench.main())
