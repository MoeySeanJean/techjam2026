"""The optimization loop.

    profile -> propose -> compile -> GATE -> measure -> feed back -> freeze

The gate is the load-bearing component. A proposer -- heuristic or LLM -- is
allowed to emit configurations that do not compile, do not fit shared memory, or
are silently wrong. None of those reach a benchmark, and a wrong-but-fast
configuration can never be promoted into the dispatch table, because
correctness is checked before latency is ever measured.

Everything the loop learns is written to `results/genealogy_<arch>.json`:
what was proposed, why it was rejected, and what it measured. That file is the
evidence behind the dispatch table, and the source of the failure taxonomy in
the report.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch

import torch_transformer_benchmark as B

from .. import bench
from ..budget import evaluate
from ..dispatch import DispatchTable, Entry, shape_signature
from ..hw import GPUSpec, device_slug, probe
from ..optimized import Plan, build_shared
from ..search import measure_stage_cost
from ..shapes import Case
from . import proposers
from .profile import profile_case

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "results")


def classify_exception(e: BaseException) -> str:
    """Turn a raw failure into a taxonomy label.

    These labels are the report's contribution on *how AI fails at kernel
    engineering*: they are collected across every rejected proposal and counted.
    """
    text = f"{type(e).__name__}: {e}".lower()
    if "out of resource" in text or "shared memory" in text or "smem" in text:
        return "shared_memory_overflow"
    if "compilationerror" in text or "compile" in text:
        return "compile_error"
    if "out of memory" in text or "cuda oom" in text:
        return "device_oom"
    if "not supported" in text or "unsupported" in text or "assert" in text:
        return "unsupported_config"
    if "illegal memory access" in text or "misaligned" in text:
        return "illegal_access"
    return "runtime_error"


def run_case(
    case: Case,
    device: torch.device,
    spec: GPUSpec,
    proposer: proposers.Proposer,
    iterations: int = 12,
    margin: float = 0.80,
    trials: int = 3,
) -> Dict:
    print(f"\n=== {case.name}: {case.label()} ===")
    prof = profile_case(case, device)
    print(f"  regime: {prof.regime} | {prof.total_cuda_ms:.3f} ms GPU, "
          f"{prof.total_cpu_ms:.3f} ms CPU, {prof.launches} launches "
          f"@ {prof.launch_us_each:.1f} us")

    control = Plan(name="wide", compute_dtype="auto", residual_dtype="float32",
                   fuse_qkv=True, attention="flash", fused_norm=True,
                   smem_kb=spec.shared_mem_per_block_kb)
    if case.dtype == "float32":
        _, stage_cost = measure_stage_cost(case, device, control, trials=2)
    else:
        stage_cost = {}

    history: List[proposers.Attempt] = []
    for it in range(1, iterations + 1):
        plan = proposer.propose(case, spec, prof, history, stage_cost, margin)
        if plan is None:
            print(f"  [{it}] proposer exhausted its search space")
            break
        att = proposers.Attempt(it, plan, proposer=proposer.name)
        try:
            r = evaluate(case, plan, device, trials)
            att.utilization = r.envelope_utilization
            att.passed = r.passed and r.envelope_utilization <= margin
            att.status = "ok" if att.passed else "numeric_fail"
            if not att.passed:
                att.detail = (f"envelope {r.envelope_utilization:.3f} > "
                              f"margin {margin}")
        except Exception as e:
            att.status = classify_exception(e)
            att.detail = f"{type(e).__name__}: {str(e)[:160]}"
        history.append(att)
        print(f"  [{it}] {att.line()}")
        torch.cuda.empty_cache()

    # Benchmark the survivors together, interleaved, in a single run.
    passing = [a for a in history if a.passed]
    if passing:
        cfg = case.to_config()
        base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
        x, vm = B.generate_random_case(cfg, device, case.torch_dtype, 1234,
                                       case.padding_ratio, case.input_scale)
        compiled = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
        compiled.load_state_dict(base.state_dict())
        compiled = torch.compile(compiled, mode="max-autotune")
        # max-autotune does not apply CUDA graphs; reduce-overhead does. Time
        # both so our graph-capture win is measured against a baseline that has
        # the same advantage available to it.
        compiled_ro = None
        try:
            _ro = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
            _ro.load_state_dict(base.state_dict())
            compiled_ro = torch.compile(_ro, mode="reduce-overhead")
        except Exception:
            compiled_ro = None

        runnable = {"baseline": lambda: base(x, vm),
                    "torch.compile": lambda: compiled(x, vm)}
        if compiled_ro is not None:
            runnable["torch.compile[ro]"] = lambda: compiled_ro(x, vm)
        keep = []
        for a in passing:
            try:
                m = build_shared(cfg, a.plan, base)
                runnable[a.plan.name] = (lambda mm=m: mm(x, vm))
                keep.append((a, m))
            except Exception as e:
                a.status = classify_exception(e)
                a.passed = False
                a.detail = f"{type(e).__name__}: {str(e)[:160]}"
        timings = bench.compare(runnable, warmup=20, repeats=30, rounds=4)
        base_ms = timings["baseline"].median_ms
        comp_ms = min(t.median_ms for k, t in timings.items()
                      if k.startswith("torch.compile"))
        for a, _ in keep:
            t = timings[a.plan.name]
            a.median_ms = t.median_ms
            a.speedup = base_ms / t.median_ms
        del base, compiled, compiled_ro, keep
        torch.cuda.empty_cache()
    else:
        base_ms = comp_ms = float("nan")

    ranked = sorted((a for a in history if a.passed and a.median_ms == a.median_ms),
                    key=lambda a: a.median_ms)
    best = ranked[0] if ranked else None
    if best:
        print(f"  -> best {best.plan.name}: {best.speedup:.3f}x vs baseline, "
              f"{comp_ms / best.median_ms:.3f}x vs torch.compile "
              f"(envelope {best.utilization:.3f})")
    else:
        print("  -> nothing cleared the gate; dispatch keeps the bit-exact plan")

    return {
        "case": case.name, "label": case.label(), "regime": case.regime,
        "profile": {"regime": prof.regime, "cuda_ms": prof.total_cuda_ms,
                    "cpu_ms": prof.total_cpu_ms, "launches": prof.launches,
                    "launch_us": prof.launch_us_each, "buckets": prof.buckets},
        "stage_cost": stage_cost,
        "baseline_ms": base_ms, "compile_ms": comp_ms,
        "attempts": [dataclasses.asdict(a) | {"plan": a.plan.name,
                                              "plan_detail": a.plan.describe()}
                     for a in history],
        "best": None if not best else {
            "plan": best.plan.name, "describe": best.plan.describe(),
            "median_ms": best.median_ms, "speedup": best.speedup,
            "speedup_vs_compile": comp_ms / best.median_ms,
            "utilization": best.utilization},
        "_best_plan": best.plan if best else None,
        "_best_compile": comp_ms,
    }


# How much a proposal must beat the frozen plan by before it takes the slot.
#
# A strict `>` promoted a 3.21x over a 3.20x on the laptop -- 0.3%, which is
# inside run-to-run drift on a card that cannot lock its clocks. Displacing a
# measured plan on that basis is fitting noise, and it would let the agent churn
# the table on every run without improving anything. Three percent is a
# judgement call rather than a measured constant, chosen to sit well above the
# drift we have observed and well below the gaps that represent real wins.
PROMOTE_MARGIN = 1.03


def run_agent(
    device: torch.device,
    cases: Sequence[Case],
    iterations: int = 12,
    margin: float = 0.80,
    provider: str = "auto",
    tag: Optional[str] = None,
) -> int:
    spec = probe()
    proposer = proposers.build(provider)
    suffix = f"_{tag}" if tag else f"_{proposer.name}"
    print(f"# {spec.summary()}")
    print(f"# proposer: {proposer.name} | {len(cases)} cases | "
          f"{iterations} iterations each | margin {margin}")

    table = DispatchTable.load_for(spec)
    records, started = [], time.time()
    taxonomy = Counter()

    for case in cases:
        try:
            rec = run_case(case, device, spec, proposer, iterations, margin)
        except Exception as e:
            print(f"  case failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        plan = rec.pop("_best_plan")
        rec.pop("_best_compile")      # popped to keep it out of the artifact
        for a in rec["attempts"]:
            taxonomy[a["status"]] += 1
        if plan is not None:
            # Only displace an existing entry if this proposal is actually
            # faster. `DispatchTable.add` supersedes a same-case row outright,
            # which is right for a re-measurement but wrong here: the agent runs
            # after `tune`, so an LLM proposal that merely passed the gate would
            # replace a better plan the search had already frozen. The proposer
            # earns its way into the table; it is not handed the slot.
            entry = Entry(
                signature=shape_signature(case), dtype=case.dtype,
                arch=spec.arch, regime=case.regime, case=case.name, plan=plan,
                utilization=rec["best"]["utilization"],
                speedup=rec["best"]["speedup"],
                speedup_vs_compile=rec["best"]["speedup_vs_compile"])
            prior = table._index.get((entry.arch, entry.dtype, entry.signature))
            improves = (prior is None
                        or entry.speedup > prior.speedup * PROMOTE_MARGIN)
            if improves:
                table.add(entry)
                table.save(spec.arch, device_slug(spec))
                rec["promoted"] = True
                print(f"    -> promoted into the table "
                      f"({entry.speedup:.2f}x"
                      + (f" vs {prior.speedup:.2f}x held" if prior else "")
                      + ")")
            else:
                rec["promoted"] = False
                print(f"    -> not promoted: {entry.speedup:.2f}x does not beat "
                      f"the {prior.speedup:.2f}x already frozen by the "
                      f"{PROMOTE_MARGIN:.0%} margin")
        records.append(rec)

        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f"genealogy_{device_slug(spec)}{suffix}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"gpu": spec.summary(), "arch": spec.arch,
                       "device": device_slug(spec),
                       "proposer": proposer.name,
                       "proposer_model": getattr(proposer, "model", None),
                       "proposer_failures": getattr(proposer, "failures", 0),
                       "elapsed_s": time.time() - started,
                       "taxonomy": dict(taxonomy),
                       "records": records}, f, indent=2, default=str)

    total = sum(taxonomy.values())
    print(f"\n=== genealogy: {total} proposals across {len(records)} shapes ===")
    for status, n in taxonomy.most_common():
        print(f"  {status:<24} {n:4d}  ({100.0 * n / max(total, 1):5.1f}%)")
    if getattr(proposer, "transcript", None):
        with open(os.path.join(RESULTS, f"llm_transcript_{device_slug(spec)}{suffix}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(proposer.transcript, f, indent=2)
    return 0
