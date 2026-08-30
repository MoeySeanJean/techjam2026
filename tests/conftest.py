"""Shared fixtures.

GPU-dependent tests are skipped rather than failed when no CUDA device is
present, so the suite is still useful on a machine without one (the numerics
and dispatch-logic tests are pure Python and always run).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import torch_transformer_benchmark as B  # noqa: E402

def _has_gpu() -> bool:
    """True only if a usable CUDA device is actually present.

    `is_available()` alone is not enough: a machine can report True with
    `device_count() == 0` (for example under CUDA_VISIBLE_DEVICES=""), and the
    GPU tests then fail instead of skipping. A reviewer without a GPU should see
    skips, not a wall of red.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


HAS_GPU = _has_gpu()

requires_cuda = pytest.mark.skipif(not HAS_GPU, reason="needs a CUDA device")
requires_triton = pytest.mark.skipif(not HAS_GPU, reason="needs a CUDA device")


@pytest.fixture(scope="session", autouse=True)
def _match_benchmark_settings():
    """Reproduce the organizer's global torch settings exactly.

    The script sets TF32 on and matmul precision 'high' before doing anything.
    Testing under different settings would measure a different reference.
    """
    if HAS_GPU:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if HAS_GPU else "cpu")


@pytest.fixture(scope="session")
def smem_kb():
    if not HAS_GPU:
        return 99.0
    from kernelforge.hw import probe
    return probe(measure=False).shared_mem_per_block_kb


def make_case(device, dtype=torch.float32, batch=4, seq=64, d=256, heads=4,
              ffn=512, layers=2, causal=False, padding=0.0, scale=1.0, seed=1234):
    """A baseline model plus one input drawn exactly as the benchmark draws it."""
    cfg = B.TransformerConfig(batch, seq, d, heads, ffn, layers, causal)
    model = B.BaselineTransformer(cfg).to(device, dtype).eval()
    x, mask = B.generate_random_case(cfg, device, dtype, seed, padding, scale)
    return cfg, model, x, mask
