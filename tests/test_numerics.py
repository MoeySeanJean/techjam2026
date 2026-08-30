"""The accuracy gate must replicate the organizer's rule exactly.

If this drifts from `compare_outputs` in `torch_transformer_benchmark.py`, every
downstream decision in the project is made against the wrong target -- so it is
tested against the script's own implementation, not against a restatement of it.
"""
from __future__ import annotations

import pytest
import torch

import torch_transformer_benchmark as B
from kernelforge.numerics import ATOL, RTOL, check


def test_defaults_match_the_script():
    """Our constants must track the script's argparse defaults, not the PDF's.

    The PDF quotes the looser 2e-3 / 2e-2; the script is the authority, and it
    is read here directly so this test fails if the organizers change it.
    """
    import inspect
    import re
    src = inspect.getsource(B.parse_args)
    atol = float(re.search(r'"--atol",\s*type=float,\s*default=([0-9.e-]+)', src).group(1))
    rtol = float(re.search(r'"--rtol",\s*type=float,\s*default=([0-9.e-]+)', src).group(1))
    assert (ATOL, RTOL) == (atol, rtol)


def test_agrees_with_the_script_on_random_data():
    torch.manual_seed(0)
    for _ in range(50):
        ref = torch.randn(64, 32) * torch.randint(1, 100, (1,)).item()
        opt = ref + torch.randn_like(ref) * 10 ** torch.randint(-5, -1, (1,)).item()
        ours = check(ref, opt)
        theirs = B.compare_outputs(ref, opt, rtol=RTOL, atol=ATOL)
        assert ours.passed == theirs.passed
        assert ours.failed_elements == theirs.failed_elements
        assert abs(ours.max_abs_error - theirs.max_abs_error) < 1e-12


def test_rule_is_or_not_and():
    """A large value inside 1% relative passes even though abs error >> atol."""
    ref = torch.tensor([100.0])
    opt = ref + 100.0 * (RTOL / 2)       # abs >> atol, but inside rtol
    assert check(ref, opt).passed
    # And a tiny value inside atol passes even though relative error is huge.
    ref = torch.tensor([1e-6])
    opt = ref + ATOL / 2                 # enormous relative error, inside atol
    assert check(ref, opt).passed


def test_stricter_than_isclose():
    """The script says it deliberately does not use torch.isclose; verify that
    matters, i.e. there exist points isclose accepts and the rule rejects."""
    ref = torch.tensor([0.05])
    opt = ref + (ATOL + RTOL * ref.abs()) * 0.9   # inside atol + rtol*|ref|
    assert torch.isclose(opt, ref, rtol=RTOL, atol=ATOL).item()
    assert not check(ref, opt).passed


def test_envelope_utilization_is_the_headroom_signal():
    """Derived from ATOL/RTOL, not hardcoded -- the organizer has already
    changed them once (1e-3/1e-2 -> 2e-3/2e-2 on 27 Aug), and a test that
    pins the old values fails for the wrong reason."""
    ref = torch.tensor([1.0])
    allowance = max(ATOL, RTOL * 1.0)
    assert check(ref, ref + allowance / 2).envelope_utilization ==         pytest.approx(0.5, rel=1e-4)
    assert check(ref, ref + allowance).envelope_utilization ==         pytest.approx(1.0, rel=1e-4)
    assert not check(ref, ref + allowance * 2).passed


def test_non_finite_output_fails():
    ref = torch.ones(4)
    for bad in (float("nan"), float("inf")):
        opt = ref.clone()
        opt[2] = bad
        res = check(ref, opt)
        assert not res.passed
        assert "non-finite" in res.note


def test_shape_mismatch_is_a_failure_not_an_exception():
    res = check(torch.zeros(4, 8), torch.zeros(4, 9))
    assert not res.passed
    assert "shape mismatch" in res.note
