"""The correctness gate.

This is an exact replica of `compare_outputs` in the organizer's benchmark script.
It exists as a standalone module because the agent loop must reject numerically
wrong kernels *before* they are ever benchmarked -- LLM-written kernels are wrong
far more often than they are slow, and a silently wrong kernel that benchmarks
fast is the single most dangerous failure mode in this project.

The rule (script line ~317) is an OR, not an AND:

    passed = (abs_err <= atol) | (abs_err <= rtol * |ref|)

so the per-element envelope is `max(atol, rtol*|ref|)`. Note this is *stricter*
than `torch.isclose`, which uses `atol + rtol*|ref|`; the script says so explicitly
and deliberately does not use isclose. We match the script, not isclose.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

def _script_defaults() -> tuple:
    """Read atol/rtol straight out of the organizer's script.

    These have already changed once: the 27 Aug update loosened them from
    1e-3/1e-2 to 2e-3/2e-2. Hardcoding them means the gate silently drifts from
    the thing it is supposed to replicate, so they are parsed from the source of
    truth and only fall back to literals if that ever fails.
    """
    import inspect
    import re
    try:
        import torch_transformer_benchmark as _b
        src = inspect.getsource(_b.parse_args)
        atol = float(re.search(r'"--atol",\s*type=float,\s*default=([0-9.eE+-]+)',
                               src).group(1))
        rtol = float(re.search(r'"--rtol",\s*type=float,\s*default=([0-9.eE+-]+)',
                               src).group(1))
        return atol, rtol
    except Exception:
        return 2e-3, 2e-2


ATOL, RTOL = _script_defaults()

# Envelope utilization is not perfectly reproducible: cuBLAS selects different
# kernels for the same call between processes, and the same (case, plan) pair
# moves with it. Measured by re-evaluating three shipped entries in six fresh
# processes each -- the largest spread was 0.141 (per case: 0.141, 0.089, 0.061).
#
# The two margins below are derived from that number rather than chosen:
#
#   ADMISSION (0.80)  what may enter the table. An entry observed at 0.80 has a
#                     true value that could be ~0.14 higher, i.e. ~0.94 -- still
#                     inside the gate, which is the property that matters.
#   DEMOTION          what counts as drift on re-verification. It has to sit
#                     above the largest observation a legitimately-admitted entry
#                     can produce, or `verify --demote` becomes a ratchet that
#                     re-rolls the noise and demotes good plans on every pass.
#                     ADMISSION + SPREAD is that bound.
#
# Re-measure if the hardware changes: run `cli verify` over the frozen table
# twice in separate processes and take the largest envelope difference.
ENVELOPE_SPREAD = 0.141
ADMISSION_MARGIN = 0.80
DEMOTION_MARGIN = round(ADMISSION_MARGIN + ENVELOPE_SPREAD, 3)   # 0.941


@dataclasses.dataclass
class CheckResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    failed_elements: int
    total_elements: int
    worst_index: Optional[Tuple[int, ...]]
    reference_at_worst: float
    optimized_at_worst: float
    # How much of the allowed envelope we actually consumed. <1.0 means we pass;
    # this is the number that tells us whether a precision change has headroom left.
    envelope_utilization: float
    note: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status} | max_abs={self.max_abs_error:.3e} "
            f"max_rel={self.max_rel_error:.3e} "
            f"envelope={self.envelope_utilization:.3f} "
            f"failed={self.failed_elements}/{self.total_elements}"
            + (f" | {self.note}" if self.note else "")
        )


def check(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> CheckResult:
    if reference.shape != optimized.shape:
        return CheckResult(
            False, float("inf"), float("inf"), float("inf"),
            reference.numel(), reference.numel(), None, 0.0, 0.0, float("inf"),
            note=f"shape mismatch {tuple(reference.shape)} vs {tuple(optimized.shape)}",
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite = torch.isfinite(ref) & torch.isfinite(opt)
    abs_err = (opt - ref).abs()

    abs_ok = abs_err <= atol
    rel_ok = abs_err <= rtol * ref.abs()
    passed_mask = finite & (abs_ok | rel_ok)
    failed_mask = ~passed_mask
    failed = int(failed_mask.sum().item())

    # Envelope utilisation: worst-case ratio of the error to what was permitted.
    # This is the headroom signal the precision-budget tool reports on.
    allowed = torch.maximum(torch.full_like(ref, atol), rtol * ref.abs())
    utilization = float((abs_err / allowed).max().item())

    flat_worst = int(abs_err.reshape(-1).argmax().item())
    idx, remaining = [], flat_worst
    for size in reversed(ref.shape):
        idx.append(remaining % size)
        remaining //= size
    worst = tuple(reversed(idx))

    note = ""
    if not bool(finite.all()):
        n_bad = int((~finite).sum().item())
        note = f"{n_bad} non-finite element(s) -- NaN/Inf in output"

    return CheckResult(
        passed=(failed == 0),
        max_abs_error=float(abs_err.max().item()),
        max_rel_error=float((abs_err / ref.abs().clamp_min(1e-12)).max().item()),
        mean_abs_error=float(abs_err.mean().item()),
        failed_elements=failed,
        total_elements=ref.numel(),
        worst_index=worst,
        reference_at_worst=float(ref[worst].item()),
        optimized_at_worst=float(opt[worst].item()),
        envelope_utilization=utilization,
        note=note,
    )
