"""End-to-end equivalence and the safety properties the submission relies on.

The central claim of docs/EQUIVALENCE.md is that our structural rewrite is
*bit-identical* to the baseline on every dtype. That is what makes a safe fast
path possible at float16/bfloat16, so it is asserted here rather than argued.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from conftest import make_case, requires_cuda
from kernelforge.numerics import check
from kernelforge.optimized import build_shared
from kernelforge.search import SAFE

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal,padding", [(False, 0.0), (True, 0.0),
                                            (False, 0.4), (True, 0.4)])
def test_structural_rewrite_is_bit_exact(device, dtype, causal, padding):
    """0.000 envelope, every dtype, including causal crossed with padding."""
    cfg, base, x, mask = make_case(device, dtype, causal=causal, padding=padding)
    opt = build_shared(cfg, SAFE, base)
    with torch.inference_mode():
        res = check(base(x, mask), opt(x, mask))
    assert res.passed
    assert res.max_abs_error == 0.0, "structural rewrite must be bit-identical"


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_bit_exact_under_large_input_scale(device, dtype):
    """LayerNorm is scale-invariant, so --input-scale must not move the answer.
    This is also the fp16 overflow probe for the residual stream."""
    cfg, base, x, mask = make_case(device, dtype, scale=64.0)
    opt = build_shared(cfg, SAFE, base)
    with torch.inference_mode():
        res = check(base(x, mask), opt(x, mask))
    assert res.passed and res.max_abs_error == 0.0


@requires_cuda
def test_single_token_specialization_is_exact(device):
    """At S=1 softmax over one key is 1.0 bit-for-bit, so attention is the
    identity on V. Skipping it must change nothing."""
    for dtype in DTYPES:
        for causal in (False, True):
            cfg, base, x, mask = make_case(device, dtype, batch=8, seq=1,
                                           causal=causal, padding=0.4)
            opt = build_shared(cfg, SAFE, base)
            with torch.inference_mode():
                res = check(base(x, mask), opt(x, mask))
            assert res.max_abs_error == 0.0, f"{dtype} causal={causal}"


@requires_cuda
def test_padded_rows_are_zero_in_the_output(device):
    """The baseline zeroes invalid token rows after final_norm; so must we."""
    cfg, base, x, mask = make_case(device, torch.float32, padding=0.5)
    opt = build_shared(cfg, SAFE, base)
    with torch.inference_mode():
        out = opt(x, mask)
    invalid = ~mask.bool()
    assert torch.count_nonzero(out[invalid]) == 0


@requires_cuda
def test_weight_sharing_does_not_perturb_the_reference(device):
    """build_shared rebinds parameters instead of copying. Candidates must
    share one weight set without any of them mutating it."""
    cfg, base, x, mask = make_case(device, torch.float32)
    with torch.inference_mode():
        before = base(x, mask).clone()

    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    models = [build_shared(cfg, SAFE, base) for _ in range(6)]
    growth_mb = (torch.cuda.memory_allocated() - start) / 2**20

    with torch.inference_mode():
        for m in models:
            m(x, mask)
        after = base(x, mask)

    assert check(before, after).max_abs_error == 0.0, "reference was mutated"
    assert growth_mb < 1.0, f"weights were copied, not shared ({growth_mb:.1f} MB)"


@requires_cuda
def test_cuda_graph_matches_eager(device):
    """Graph capture changes no arithmetic; it must be numerically a no-op."""
    cfg, base, x, mask = make_case(device, torch.float32)
    eager = build_shared(cfg, SAFE, base)
    graphed = build_shared(cfg, dataclasses.replace(SAFE, cuda_graph=True), base)
    with torch.inference_mode():
        for _ in range(3):
            graphed(x, mask)
        assert check(eager(x, mask), graphed(x, mask)).max_abs_error == 0.0


@requires_cuda
def test_graph_output_survives_a_later_replay(device):
    """The runner clones its static buffer; a held result must not be clobbered
    by the next call, which is exactly what the accuracy check does."""
    cfg, base, x, mask = make_case(device, torch.float32)
    graphed = build_shared(cfg, dataclasses.replace(SAFE, cuda_graph=True), base)
    with torch.inference_mode():
        first = graphed(x, mask)
        held = first.clone()
        graphed(x + 1.0, mask)
        assert check(held, first).max_abs_error == 0.0


@requires_cuda
def test_submission_entry_point_runs_and_passes(device):
    """The actual competition entry, through its real dispatch path."""
    from submission import UserOptimizedTransformer
    cfg, base, x, mask = make_case(device, torch.float32, layers=6)
    opt = UserOptimizedTransformer(cfg)
    opt.load_state_dict(base.state_dict(), strict=True)
    opt = opt.to(device, torch.float32).eval()
    with torch.inference_mode():
        assert check(base(x, mask), opt(x, mask)).passed


@requires_cuda
def test_submission_falls_back_rather_than_raising(device, monkeypatch):
    """If dispatch fails for any reason the entry point must still produce a
    correct answer -- accuracy failure is a hard `return 2` in the script."""
    import submission
    from kernelforge import dispatch as dispatch_mod

    def boom(*a, **k):
        raise RuntimeError("simulated dispatch failure")

    monkeypatch.setattr(dispatch_mod.DispatchTable, "load", staticmethod(boom))
    cfg, base, x, mask = make_case(device, torch.float32)
    opt = submission.UserOptimizedTransformer(cfg)
    opt.load_state_dict(base.state_dict(), strict=True)
    opt = opt.to(device, torch.float32).eval()
    with torch.inference_mode():
        assert check(base(x, mask), opt(x, mask)).max_abs_error == 0.0
