"""Budget-guided plan search.

The naive way to pick precision is to try fp16 everywhere and back off when the
gate fails. That wastes the whole search on configurations that were never
going to pass, and it gives no explanation for *why* one failed.

Instead we spend a handful of measurements up front attributing envelope
consumption to individual stages (`budget.stage_budget`), then narrow stages in
increasing order of measured error cost, stopping when the remaining margin runs
out. On the default shape that ordering comes out as attention (essentially
free) < out_proj < ffn2 < ffn1, which is not the order anyone would guess: the
FFN GEMMs have the longest reduction dimensions and dominate the budget even
though the attention path looks scarier.

Safety margin
-------------
The gate is `utilization <= 1.0`, but we require `<= margin` (default 0.80)
across several seeds. A plan that passes at 0.99 on our seeds is not a plan that
passes on the organizer's seed, and the accuracy check is a hard failure that
skips benchmarking entirely.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

import torch_transformer_benchmark as B

from . import bench
from .budget import evaluate
from .numerics import check
from .optimized import FusedTransformer, Plan, build_shared
from .shapes import Case

STAGES = ("attn", "out_proj", "ffn1", "ffn2")

# Structurally safe: measured bit-identical to the baseline on every dtype, so
# it is the fallback whenever the aggressive path cannot clear the gate.
SAFE = Plan(name="safe", compute_dtype="auto", residual_dtype="auto",
            fuse_qkv=False, attention="exact", fused_norm=False)


@dataclasses.dataclass
class Candidate:
    plan: Plan
    utilization: float
    passed: bool
    median_ms: float = float("nan")
    speedup: float = float("nan")
    speedup_vs_compile: float = float("nan")
    error: str = ""

    def row(self) -> str:
        status = "pass" if self.passed else "FAIL"
        return (f"{self.plan.name:<38} env={self.utilization:6.3f} {status}  "
                f"{self.median_ms:9.4f} ms  {self.speedup:6.3f}x")


@dataclasses.dataclass
class SearchResult:
    case: Case
    best: Optional[Candidate]
    candidates: List[Candidate]
    baseline_ms: float
    compile_ms: float
    stage_cost: Dict[str, float]
    seconds: float
    compile_modes: Dict[str, float] = dataclasses.field(default_factory=dict)
    compile_passed: Dict[str, bool] = dataclasses.field(default_factory=dict)
    compile_envelope: Dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def compile_admissible(self) -> bool:
        """Did any torch.compile mode actually clear the accuracy gate?"""
        return any(self.compile_passed.values())

    def summary(self) -> str:
        lines = [f"### {self.case.name}: {self.case.label()}",
                 f"    baseline {self.baseline_ms:.4f} ms | " +
                 " | ".join(
                     f"{k} {v:.4f} ms"
                     f"[{'PASS' if self.compile_passed.get(k) else 'GATE-FAIL'}"
                     f" env={self.compile_envelope.get(k, float('nan')):.2f}]"
                     for k, v in self.compile_modes.items()),
                 "    stage error cost: " +
                 ", ".join(f"{k}={v:+.3f}" for k, v in
                           sorted(self.stage_cost.items(), key=lambda kv: kv[1]))]
        for c in self.candidates:
            lines.append("      " + c.row())
        if self.best:
            lines.append(f"    -> BEST {self.best.plan.name} "
                         f"{self.best.speedup:.3f}x vs baseline, "
                         f"{self.best.speedup_vs_compile:.3f}x vs torch.compile"
                         + ("" if self.compile_admissible else
                            "  [torch.compile is NOT admissible here: it fails "
                            "the accuracy gate]"))
        return "\n".join(lines)


def _make(cfg, plan, base, device, dtype) -> FusedTransformer:
    """Candidates share the baseline's weight tensors -- see `build_shared`."""
    return build_shared(cfg, plan, base)


def measure_stage_cost(
    case: Case, device: torch.device, control: Plan, trials: int = 2,
) -> Tuple[float, Dict[str, float]]:
    """Marginal envelope cost of narrowing each stage to fp16, one at a time."""
    base_util = evaluate(case, control, device, trials).envelope_utilization
    cost: Dict[str, float] = {}
    for stage in STAGES:
        plan = control.with_override(stage, "float16")
        try:
            u = evaluate(case, plan, device, trials).envelope_utilization
            cost[stage] = u - base_util
        except Exception:
            cost[stage] = float("inf")
    return base_util, cost


# How much faster our own kernel may be and still be preferred over a
# `torch.compile` candidate. Set from the measured gap: compile plans came in at
# a mean 0.891x under the organizer's harness against our kernels' 1.017x, so a
# compile plan needs roughly 15% of headroom here to be genuinely ahead there.
COMPILE_MARGIN = 1.15


def search(
    case: Case,
    device: torch.device,
    margin: float = 0.80,
    trials: int = 3,
    smem_kb: float = 99.0,
    time_budget_s: float = 240.0,
    verbose: bool = True,
) -> SearchResult:
    started = time.time()
    cfg = case.to_config()
    dtype = case.torch_dtype

    base = B.BaselineTransformer(cfg).to(device, dtype).eval()
    x, vm = B.generate_random_case(cfg, device, dtype, 1234,
                                   case.padding_ratio, case.input_scale)

    compiled = B.BaselineTransformer(cfg).to(device, dtype).eval()
    compiled.load_state_dict(base.state_dict())
    compiled = torch.compile(compiled, mode="max-autotune")

    # `max-autotune` does NOT apply CUDA graphs; `reduce-overhead` does. Since a
    # large part of our win on small shapes is graph capture, comparing only
    # against max-autotune would flatter us. Both are timed.
    compiled_ro = None
    try:
        _ro = B.BaselineTransformer(cfg).to(device, dtype).eval()
        _ro.load_state_dict(base.state_dict())
        compiled_ro = torch.compile(_ro, mode="reduce-overhead")
    except Exception:
        compiled_ro = None

    # Hold the library baselines to the same gate we hold ourselves to. At
    # float16/bfloat16 `torch.compile` does not pass it (see docs/PRECISION.md),
    # and a candidate that is "slower than torch.compile" there is not actually
    # losing to anything admissible. Reporting the speed ratio without this flag
    # would misrepresent the comparison in both directions.
    compile_passed: Dict[str, bool] = {}
    compile_envelope: Dict[str, float] = {}
    with torch.inference_mode():
        ref_out = base(x, vm)
        for label, mod in (("torch.compile", compiled),
                           ("torch.compile[ro]", compiled_ro)):
            if mod is None:
                continue
            try:
                r = check(ref_out, mod(x, vm))
                compile_passed[label] = r.passed
                compile_envelope[label] = r.envelope_utilization
            except Exception:
                compile_passed[label] = False
                compile_envelope[label] = float("nan")
    del ref_out
    torch.cuda.empty_cache()

    candidates: List[Candidate] = []

    def gate(plan: Plan) -> Candidate:
        try:
            r = evaluate(case, plan, device, trials)
            return Candidate(plan, r.envelope_utilization,
                             r.passed and r.envelope_utilization <= margin)
        except Exception as e:
            return Candidate(plan, float("nan"), False,
                             error=f"{type(e).__name__}: {str(e)[:100]}")

    # 1. The structurally safe plan is always in the running: it is bit-exact,
    #    so it is the only thing that can be shipped for fp16/bf16 inputs.
    safe = dataclasses.replace(SAFE, smem_kb=smem_kb, name="safe(exact)")
    safe_graph = dataclasses.replace(safe, cuda_graph=True, name="safe(exact)+graph")

    # 2. All-wide structural rewrite: fused norm + flash, no precision change.
    wide = Plan(name="wide", compute_dtype="auto", residual_dtype="float32",
                fuse_qkv=True, attention="flash", fused_norm=True, smem_kb=smem_kb)
    wide_sdpa = dataclasses.replace(wide, attention="sdpa", name="wide(sdpa)")
    # The fused Triton norm is the largest single structural error source (it
    # differs from torch's LayerNorm only by reduction order, but that gets
    # amplified ~1000x through the stack). Keeping torch's norm costs a kernel
    # launch and buys back envelope, which can pay for an extra fp16 stage --
    # so both sides of that trade are searched rather than assumed.
    wide_tn = dataclasses.replace(wide, fused_norm=False, name="wide(torch-norm)")

    # Pick whichever structural base leaves more headroom, then narrow from it.
    util_fused, cost_fused = measure_stage_cost(case, device, wide, trials=2)
    util_tn, cost_tn = measure_stage_cost(case, device, wide_tn, trials=2)
    if util_tn < util_fused - 0.05:
        control, base_util, stage_cost = wide_tn, util_tn, cost_tn
    else:
        control, base_util, stage_cost = wide, util_fused, cost_fused
    if verbose:
        print(f"  [{case.name}] structural envelope fused-norm {util_fused:.3f} / "
              f"torch-norm {util_tn:.3f} -> base '{control.name}'; stage cost " +
              ", ".join(f"{k}{v:+.3f}" for k, v in stage_cost.items()))

    # torch.compile competes as an optimization, not just as a yardstick: the
    # organizer's script lists it as a suggested direction. It is offered only on
    # Triton-free plans (our Triton kernels graph-break inductor), and it is held
    # to exactly the same gate -- which is why it is never selected at
    # float16/bfloat16 when applied to the baseline, where it cannot pass.
    #
    # Compiling our *structural rewrite* is a different proposition from
    # compiling the baseline: the rewrite is bit-exact in eager, and inductor
    # preserves that on shapes where compiling the baseline does not.
    compile_plans: List[Plan] = []
    for mode in ("max-autotune", "reduce-overhead"):
        short = "ma" if mode == "max-autotune" else "ro"
        compile_plans.append(dataclasses.replace(
            SAFE, smem_kb=smem_kb, torch_compile=mode,
            name=f"exact+compile[{short}]"))
        compile_plans.append(dataclasses.replace(
            SAFE, smem_kb=smem_kb, torch_compile=mode, attention="baseline",
            name=f"compile[{short}]"))

    plans: List[Plan] = [safe, safe_graph, wide_sdpa, wide, wide_tn] + compile_plans

    # 3. Narrow stages cheapest-error-first while the margin holds.
    order = sorted(STAGES, key=lambda s: stage_cost.get(s, float("inf")))
    running = control
    suffix = "" if control.fused_norm else "/tn"
    for stage in order:
        if stage_cost.get(stage, float("inf")) == float("inf"):
            continue
        running = running.with_override(stage, "float16")
        plans.append(dataclasses.replace(
            running, name="fp16[" + ",".join(
                s for s, _ in running.overrides) + "]" + suffix))

    seen = set()
    for plan in plans:
        if plan.name in seen:
            continue
        seen.add(plan.name)
        candidates.append(gate(plan))
        if time.time() - started > time_budget_s:
            break

    # 4. CUDA-graph the fastest gate-passing precision plan. Graph capture does
    #    not change arithmetic, so it inherits the utilization of its base plan.
    passing = [c for c in candidates if c.passed]
    for c in list(passing):
        if not c.plan.cuda_graph and not c.plan.torch_compile:
            g = dataclasses.replace(c.plan, cuda_graph=True,
                                    name=c.plan.name + "+graph")
            if g.name not in seen:
                seen.add(g.name)
                candidates.append(Candidate(g, c.utilization, True))

    # 5. Benchmark everything that cleared the gate, all in one interleaved run.
    runnable: Dict[str, Callable] = {
        "baseline": lambda: base(x, vm),
        "torch.compile": lambda: compiled(x, vm),
    }
    if compiled_ro is not None:
        runnable["torch.compile[ro]"] = lambda: compiled_ro(x, vm)
    models = {}
    for c in candidates:
        if not c.passed:
            continue
        try:
            m = _make(cfg, c.plan, base, device, dtype)
            models[c.plan.name] = m
            runnable[c.plan.name] = (lambda mm=m: mm(x, vm))
        except Exception as e:
            c.error = f"build: {type(e).__name__}: {str(e)[:80]}"
            c.passed = False

    timings = bench.compare(runnable, warmup=20, repeats=30, rounds=4)
    baseline_ms = timings["baseline"].median_ms
    # Compare against whichever torch.compile mode is faster -- the strongest
    # available library baseline, not the most flattering one.
    compile_ms = min(t.median_ms for k, t in timings.items()
                     if k.startswith("torch.compile"))
    compile_modes = {k: t.median_ms for k, t in timings.items()
                     if k.startswith("torch.compile")}
    for c in candidates:
        t = timings.get(c.plan.name)
        if t is None:
            continue
        c.median_ms = t.median_ms
        c.speedup = baseline_ms / t.median_ms
        c.speedup_vs_compile = compile_ms / t.median_ms

    # A `torch.compile` candidate has to win by a real margin, not by a hair.
    #
    # This harness times candidates interleaved, so a compiled model is already
    # warm and its CUDA graph captured when we measure it. The organizer's
    # script does not work that way -- fresh process, warmup 20, repeats 100 --
    # and measured across 103 shape pairs on nine GPUs, plans that delegate to
    # torch.compile came in at a mean 0.891x of what our own kernels achieved
    # there, regressing on 52% of shapes, while our own kernels averaged 1.017x
    # and regressed on 10% (`results/tuned_vs_untuned_fleet.json`).
    #
    # So a tie here is a loss there. Requiring a clear margin keeps the
    # compiler where it genuinely wins and stops it taking shapes it only
    # appears to win because of how we measure. This is a correction to our
    # selection, not a claim that torch.compile is slow.
    ranked = [c for c in candidates if c.passed and c.median_ms == c.median_ms]
    ranked.sort(key=lambda c: c.median_ms)

    def _delegates(cand) -> bool:
        return bool(getattr(cand.plan, "torch_compile", None))

    best = ranked[0] if ranked else None
    if best is not None and _delegates(best):
        mine = next((c for c in ranked if not _delegates(c)), None)
        if mine is not None and mine.median_ms <= best.median_ms * COMPILE_MARGIN:
            best = mine

    # Free the reference models before returning: a sweep builds one of these
    # per case and the GPU memory matters more than the microsecond.
    #
    # A linter will flag the lambdas above as closing over names that get
    # deleted here. They do, and it is safe: `bench.compare` is the only caller
    # and it has already returned, after which `runnable` is unreachable. If you
    # ever add a second timing pass, move it above this line.
    del base, compiled, compiled_ro, models
    torch.cuda.empty_cache()

    return SearchResult(case, best, candidates, baseline_ms, compile_ms,
                        stage_cost, time.time() - started, compile_modes,
                        compile_passed, compile_envelope)
