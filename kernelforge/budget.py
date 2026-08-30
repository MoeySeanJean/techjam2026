"""The precision error budget.

The organizer's tolerance is an OR: an element passes if `abs_err <= atol` OR
`abs_err <= rtol*|ref|`, so the permitted envelope is `max(atol, rtol*|ref|)`
and "how close are we to failing" is a single number:

    envelope utilization = max over elements of  abs_err / max(atol, rtol*|ref|)

Utilization < 1.0 passes. This module measures it *per pipeline stage*, holding
every other stage at full precision, which turns precision selection from a
guessing game into an attribution problem: if fp16 in `ffn2` alone consumes 0.9
of the envelope, that is where the budget went, and no amount of tuning
elsewhere will buy it back.

Running this before writing a fast kernel is the difference between "we tried
fp16 and it failed" and "stage X costs 0.9 of budget, stage Y costs 0.02, so we
keep X wide and everything else narrow."
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch

import torch_transformer_benchmark as B

from .numerics import ATOL, RTOL, check
from .optimized import Plan, build_shared
from .shapes import Case

STAGES = ("attn", "out_proj", "ffn1", "ffn2")
NARROW = ("float16", "bfloat16")


@dataclasses.dataclass
class BudgetRow:
    case: str
    stage: str
    dtype: str
    utilization: float
    max_abs: float
    passed: bool
    error: str = ""

    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if not self.passed:
            return "FAIL"
        if self.utilization > 0.5:
            return "tight"
        return "ok"


def _reference(case: Case, device, seed: int = 1234):
    cfg = case.to_config()
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    x, vm = B.generate_random_case(cfg, device, case.torch_dtype, seed,
                                   case.padding_ratio, case.input_scale)
    with torch.inference_mode():
        ref = base(x, vm)
    return base, x, vm, ref


def evaluate(
    case: Case,
    plan: Plan,
    device: torch.device,
    trials: int = 3,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    """Worst-case check result for a plan across several random inputs.

    Multiple seeds matter: a kernel that passes on seed 0 and fails on seed 7 is
    the failure mode the agent exists to catch, and a single-trial gate would
    wave it through.
    """
    cfg = case.to_config()
    worst = None
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    opt = build_shared(cfg, plan, base)
    for t in range(trials):
        x, vm = B.generate_random_case(cfg, device, case.torch_dtype, 1234 + t,
                                       case.padding_ratio, case.input_scale)
        with torch.inference_mode():
            ref = base(x, vm)
            out = opt(x, vm)
        res = check(ref, out, rtol=rtol, atol=atol)
        if worst is None or res.envelope_utilization > worst.envelope_utilization:
            worst = res
    del base, opt
    torch.cuda.empty_cache()
    return worst


def stage_budget(
    case: Case,
    device: torch.device,
    base_plan: Optional[Plan] = None,
    trials: int = 3,
) -> List[BudgetRow]:
    """Attribute envelope consumption to individual stages.

    The control plan runs every stage at the I/O dtype (so it should consume
    almost nothing); each subsequent row narrows exactly one stage.
    """
    rows: List[BudgetRow] = []
    control = base_plan or Plan(
        name="control", compute_dtype="auto", residual_dtype="float32",
        fuse_qkv=True, attention="sdpa", fused_norm=True)

    res = evaluate(case, control, device, trials)
    rows.append(BudgetRow(case.name, "(control: all wide)", "auto",
                          res.envelope_utilization, res.max_abs_error, res.passed))

    for stage in STAGES:
        for dt in NARROW:
            plan = control.with_override(stage, dt)
            if stage == "attn":
                plan = dataclasses.replace(plan, attention="flash")
            try:
                res = evaluate(case, plan, device, trials)
                rows.append(BudgetRow(case.name, stage, dt,
                                      res.envelope_utilization, res.max_abs_error,
                                      res.passed))
            except Exception as e:
                rows.append(BudgetRow(case.name, stage, dt, float("nan"),
                                      float("nan"), False,
                                      f"{type(e).__name__}: {str(e)[:80]}"))

    # And everything narrow at once, to expose compounding across stages.
    for dt in NARROW:
        plan = control
        for stage in STAGES:
            plan = plan.with_override(stage, dt)
        plan = dataclasses.replace(plan, attention="flash", name=f"all:{dt}")
        try:
            res = evaluate(case, plan, device, trials)
            rows.append(BudgetRow(case.name, "(all stages)", dt,
                                  res.envelope_utilization, res.max_abs_error,
                                  res.passed))
        except Exception as e:
            rows.append(BudgetRow(case.name, "(all stages)", dt, float("nan"),
                                  float("nan"), False,
                                  f"{type(e).__name__}: {str(e)[:80]}"))

    # The residual stream is called out separately: it accumulates across
    # 2*num_layers sublayers, which is the one place narrow storage compounds
    # rather than merely rounding.
    for dt in NARROW:
        plan = control
        for stage in STAGES:
            plan = plan.with_override(stage, dt)
        plan = dataclasses.replace(plan, attention="flash", residual_dtype=dt,
                                   name=f"all+res:{dt}")
        try:
            res = evaluate(case, plan, device, trials)
            rows.append(BudgetRow(case.name, "(all + residual)", dt,
                                  res.envelope_utilization, res.max_abs_error,
                                  res.passed))
        except Exception as e:
            rows.append(BudgetRow(case.name, "(all + residual)", dt, float("nan"),
                                  float("nan"), False,
                                  f"{type(e).__name__}: {str(e)[:80]}"))
    return rows


def format_rows(rows: List[BudgetRow]) -> str:
    out = [f"{'case':<20} {'stage':<20} {'dtype':<10} {'envelope':>9}  "
           f"{'max_abs':>10}  verdict",
           "-" * 82]
    for r in rows:
        util = "  nan" if r.utilization != r.utilization else f"{r.utilization:9.3f}"
        mab = "  nan" if r.max_abs != r.max_abs else f"{r.max_abs:10.3e}"
        out.append(f"{r.case:<20} {r.stage:<20} {r.dtype:<10} {util}  {mab}  "
                   f"{r.verdict()}{('  ' + r.error) if r.error else ''}")
    return "\n".join(out)
