"""Streaming the batch must not change the answer.

Official test shape 14 needs two independent things to be runnable: attention
that never forms `[B,H,S,S]`, and a forward that never holds the whole batch's
activations at once. The first is the flash kernel. The second is the batch
slicing in `FusedTransformer._chunked_forward`, and this file is the argument
that slicing is an execution-order change rather than an approximation.

The claim being tested is specific: for every chunk size, the streamed result
must sit inside the organizer's own tolerance of the unstreamed one, on the
masked and causal cases where a batch-order bug would actually show up.
"""
from __future__ import annotations

import pytest
import torch

import torch_transformer_benchmark as B
from kernelforge import shapes
from kernelforge.dispatch import SAFE
from kernelforge.numerics import check
from kernelforge.optimized import FusedTransformer, build_shared

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="needs a CUDA GPU",
)

SPECS = [
    "B8-S64-d128-H4-F128-L2-causal",
    "B8-S64-d128-H4-F128-L2-causal-pad0.4",
    "B8-S64-d128-H4-F128-L2-pad0.4",
    "B6-S32-d64-H8-F128-L3",
]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_streamed_batch_matches_whole_batch(spec, chunk):
    case = shapes.resolve([spec])[0]
    cfg = case.to_config()
    device = torch.device("cuda")
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    ours = build_shared(cfg, SAFE, base)
    x, mask = B.generate_random_case(cfg, device, case.torch_dtype, 7,
                                     case.padding_ratio, case.input_scale)
    with torch.inference_mode():
        whole = ours._eager_forward(x, mask)
        streamed = ours._chunked_forward(x, mask, chunk)
    res = check(whole, streamed)
    assert res.passed, (f"chunk={chunk} drifts from the whole batch: "
                        f"envelope {res.envelope_utilization:.3f}")


@pytest.mark.parametrize("spec", SPECS[:2])
def test_streamed_batch_still_matches_the_reference(spec):
    """The gate that actually matters: streamed output vs the organizer's model."""
    case = shapes.resolve([spec])[0]
    cfg = case.to_config()
    device = torch.device("cuda")
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    ours = build_shared(cfg, SAFE, base)
    x, mask = B.generate_random_case(cfg, device, case.torch_dtype, 11,
                                     case.padding_ratio, case.input_scale)
    with torch.inference_mode():
        res = check(base(x, mask), ours._chunked_forward(x, mask, 2))
    assert res.passed, f"envelope {res.envelope_utilization:.3f}"


# (shape, GPU memory free, expected: whole batch or a slice)
BUDGETS = [
    # An ordinary official shape on any real card: never chunked, because
    # chunking would serialize the batch for nothing.
    ("B64-S128-d128-H4-F128-L4-causal", 8, "whole"),
    ("B128-S128-d128-H4-F128-L4-causal", 8, "whole"),
    ("B64-S1024-d128-H4-F128-L4-causal", 8, "whole"),
    # 10,000 sequences, but each one tiny: fits whole on a real card, and is
    # *correctly* sliced on a small one rather than failing outright.
    ("B10000-S128-d128-H4-F128-L4-causal", 79, "whole"),
    ("B10000-S128-d128-H4-F128-L4-causal", 6, "slice"),
    # Official shape 14: must be sliced even on the largest card we have.
    ("B32-S100000-d1024-H16-F1024-L2-causal", 79, "slice"),
    ("B32-S100000-d1024-H16-F1024-L2-causal", 93, "slice"),
]


@pytest.mark.parametrize("spec,free_gb,expect", BUDGETS)
def test_chunk_decision(spec, free_gb, expect):
    """The estimator's decisions, checked without needing the card in question.

    A chunking heuristic that fires on normal shapes would serialize the batch
    and cost far more than it saves; one that fails to fire on shape 14 runs out
    of memory. Both directions are pinned here.
    """
    cfg = shapes.resolve([spec])[0].to_config()
    m = FusedTransformer(cfg, SAFE)
    chunk = m._chunk_for_budget(cfg.batch_size, cfg.seq_len, torch.float32,
                                free_gb * 2**30)
    if expect == "whole":
        assert chunk == cfg.batch_size, f"{spec} was sliced into {chunk}"
    else:
        assert 1 <= chunk < cfg.batch_size, f"{spec} was not sliced ({chunk})"


def test_shape14_needs_slicing_because_the_batch_does_not_fit():
    """The arithmetic behind shape 14, checked without allocating it."""
    cfg = shapes.resolve(["B32-S100000-d1024-H16-F1024-L2-causal"])[0].to_config()
    m = FusedTransformer(cfg, SAFE)
    whole = m._activation_bytes(cfg.batch_size, cfg.seq_len, torch.float32)
    per_sample = m._activation_bytes(1, cfg.seq_len, torch.float32)
    assert whole > 300 * 2**30, f"whole batch {whole / 2**30:.0f} GB"
    assert per_sample * 4 < whole, "slicing has to actually reduce the footprint"
