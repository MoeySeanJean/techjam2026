"""Command line entry point.

    python -m kernelforge.cli doctor      # environment + hardware probe
    python -m kernelforge.cli budget      # per-stage precision error budget
    python -m kernelforge.cli verify      # numeric gate over the shape matrix
    python -m kernelforge.cli sweep       # search + three-way benchmark
    python -m kernelforge.cli table       # show the frozen dispatch table
    python -m kernelforge.cli agent       # autonomous optimization loop
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time

import torch

from . import bench, budget, search, shapes
from .dispatch import DispatchTable, Entry, shape_signature, summarize
from . import env as envmod
from .hw import device_slug, host_summary, probe

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def _setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def _cases(args):
    """Resolve the requested cases.

    A name that is not in the built-in matrix is parsed as a shape spec, so
    `--cases B4-S777-d640-H10-F2560-L9` works exactly like `--cases default`.
    That is what lets the official shape list be tuned the day it is published,
    without touching the code.
    """
    spec = probe(measure=False)
    if getattr(args, "shapes_file", None):
        cs = shapes.load_spec_file(args.shapes_file)
    elif args.cases:
        cs = shapes.resolve(args.cases.split(","))
    else:
        regimes = args.regimes.split(",") if args.regimes else None
        cs = shapes.select(regimes=regimes,
                           include_variants=not args.no_variants)
    kept = [c for c in cs if shapes.fits_on(spec, c)]
    dropped = [c.name for c in cs if c not in kept]
    if dropped:
        print(f"[skipped, would not fit {spec.total_mem_gb:.0f} GB: "
              f"{', '.join(dropped)}]")
    return kept


# --- commands --------------------------------------------------------------

def cmd_doctor(args):
    print(envmod.format_report())
    print()
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    spec = probe()
    print(spec.summary())
    print()
    print(spec.as_prompt_block())
    print()
    from .ops.flash import HAS_TRITON, legal_blocks
    print(f"triton available: {HAS_TRITON}")
    if HAS_TRITON:
        for dh in (64, 128):
            print(f"  legal flash blocks head_dim={dh}: "
                  f"{legal_blocks(2048, dh, spec.shared_mem_per_block_kb)[:3]}")
    locked = bench.lock_clocks()
    print(f"clock lock available: {locked} "
          f"({'stable timing' if locked else 'timings will carry thermal drift'})")
    if locked:
        bench.reset_clocks()

    # What can actually be run on THIS machine. A reviewer should not have to
    # discover a missing dependency by hitting a traceback halfway through.
    from . import secrets as _secrets
    from .dispatch import DispatchTable, _arch_major
    _secrets.load()
    have_llm = bool(os.environ.get("SOCLAAS_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("ANTHROPIC_API_KEY"))
    tuned = len(DispatchTable.load(spec.arch).entries)

    print("\ncapabilities on this machine")
    print("-" * 62)
    rows = [
        ("run the organizer's benchmark", True, "scripts/run_official.py"),
        ("verify correctness", True, "cli verify"),
        ("tune new shapes / full sweep", HAS_TRITON,
         "cli tune / cli sweep" if HAS_TRITON else "needs Triton"),
        ("our Triton kernels", HAS_TRITON,
         "in use" if HAS_TRITON else "falls back to SDPA + torch LayerNorm"),
        ("LLM proposer / codegen", have_llm,
         "cli agent --provider llm" if have_llm
         else "no API key -> heuristic proposer is used instead"),
        ("pre-tuned plans for this GPU", tuned > 0,
         f"{tuned} entries for {spec.arch}" if tuned
         else f"none for {spec.arch} -> safe default; run `cli sweep` to tune"),
    ]
    for label, ok, note_ in rows:
        print(f"  [{'x' if ok else ' '}] {label:<32} {note_}")

    if _arch_major(spec.arch) < 8:
        print("\n  NOTE: pre-Ampere GPU. There is no TF32 here, so"
              " our fp16 defaults do not apply; dispatch falls back to"
              " the bit-exact plan. Run `cli sweep` to tune it.")
    if not tuned:
        print(f"\n  NOTE: no dispatch table for {spec.arch}. The"
              f" submission still runs correctly on a safe default;"
              f" `cli sweep` tunes this GPU (~50 min).")
    return 0


def cmd_budget(args):
    dev = _setup()
    rows = []
    for case in _cases(args):
        rows += budget.stage_budget(case, dev, trials=args.trials)
    text = budget.format_rows(rows)
    print(text)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "error_budget.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nwritten: {path}")
    return 0


def cmd_verify(args):
    """Re-measure the frozen table.

    This is not a formality. Envelope utilization is not perfectly reproducible
    run to run -- cuBLAS kernel selection varies with device state, and we have
    observed the same (case, plan) pair move by ~0.1 between runs. A plan
    admitted at 0.79 during the sweep can therefore re-measure at 0.89.

    Two different thresholds are at work, and conflating them is a mistake we
    made and had to undo:

      * the **admission margin** (0.80, used by the sweep) decides what may enter
        the table in the first place;
      * the **demotion threshold** (0.90, here) decides what has drifted into
        genuinely risky territory and must be replaced by the bit-exact plan.

    They must differ by more than the measurement variance. Setting the demotion
    threshold at or near the admission margin makes `--demote` a ratchet: every
    pass re-rolls the noise, demotes another entry that was actually fine, and a
    few passes reduce the whole table to bit-exact for no safety gain.
    """
    dev = _setup()
    spec = probe(measure=False)
    table = DispatchTable.load(spec.arch)
    failures, demoted = [], []
    print(f"{'case':<26} {'plan':<32} {'envelope':>9} verdict")
    print("-" * 78)
    for case in _cases(args):
        cfg = case.to_config()
        plan, source = table.lookup(spec.arch, case.torch_dtype, cfg,
                                    spec.shared_mem_per_block_kb)
        try:
            r = budget.evaluate(case, plan, dev, trials=args.trials)
            over = r.envelope_utilization > args.margin
            ok = "PASS" if r.passed else "FAIL"
            if over and r.passed:
                ok = f"PASS but > margin {args.margin}"
            print(f"{case.name:<26} {plan.name:<32} {r.envelope_utilization:9.3f} "
                  f"{ok} ({source})")
            if (not r.passed or over) and args.demote:
                import dataclasses as _dc
                from .search import SAFE
                safe = _dc.replace(SAFE, cuda_graph=True,
                                   smem_kb=spec.shared_mem_per_block_kb,
                                   name="safe(exact)+graph")
                table.add(Entry(
                    signature=shape_signature(case), dtype=case.dtype,
                    arch=spec.arch, regime=case.regime, plan=safe,
                    utilization=0.0, speedup=float("nan"),
                    speedup_vs_compile=float("nan"), case=case.name))
                demoted.append((case.name, plan.name, r.envelope_utilization))
            if not r.passed:
                failures.append((case.name, plan.name, str(r)))
        except Exception as e:
            print(f"{case.name:<26} {plan.name:<32} {'ERR':>9} "
                  f"{type(e).__name__}: {str(e)[:60]}")
            failures.append((case.name, plan.name, f"{type(e).__name__}: {e}"))
        torch.cuda.empty_cache()
    print()
    if demoted:
        table.save(spec.arch)
        print(f"demoted {len(demoted)} entr(y/ies) to the bit-exact plan:")
        for c, p, u in demoted:
            print(f"  {c:<26} {p:<32} re-measured {u:.3f}")
        print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for c, p, m in failures:
            print(f"  {c:<26} {p:<32} {m[:80]}")
        return 1
    print("all cases pass the organizer's tolerance")
    return 0


def _plan_json(plan) -> dict:
    d = dataclasses.asdict(plan)
    d["overrides"] = [list(o) for o in plan.overrides]
    d["flash_block"] = list(plan.flash_block) if plan.flash_block else None
    return d


def _plan_from_json(d: dict):
    from .optimized import Plan
    p = dict(d)
    p["overrides"] = tuple(tuple(o) for o in p.get("overrides") or [])
    p["flash_block"] = tuple(p["flash_block"]) if p.get("flash_block") else None
    return Plan(**p)


def _merge_records(slug: str, fresh: list) -> list:
    """Fold new records into whatever this device already measured.

    A sweep is incremental: `tune` runs one or two shapes and must not erase the
    other twelve. Replacing the file wholesale silently destroyed a full laptop
    sweep once, so records are keyed by case and merged, newest winning.
    """
    path = os.path.join(RESULTS, f"sweep_{slug}.json")
    by_case = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                for rec in json.load(f).get("records", []):
                    by_case[rec["case"]] = rec
        except (json.JSONDecodeError, KeyError):
            pass
    for rec in fresh:
        by_case[rec["case"]] = rec
    return list(by_case.values())


def _record_for(case, r):
    return {
        "case": case.name, "label": case.label(), "regime": case.regime,
        "cli": case.cli(),
        "baseline_ms": r.baseline_ms, "compile_ms": r.compile_ms,
        "compile_modes": r.compile_modes,
        "compile_passed": r.compile_passed,
        "compile_envelope": r.compile_envelope,
        "compile_admissible": r.compile_admissible,
        "stage_cost": r.stage_cost,
        "best": None if not r.best else {
            "plan": r.best.plan.name, "describe": r.best.plan.describe(),
            "median_ms": r.best.median_ms, "speedup": r.best.speedup,
            "speedup_vs_compile": r.best.speedup_vs_compile,
            "utilization": r.best.utilization,
            # Serialized so `cli rebuild-table` can reconstruct dispatch from
            # artifacts alone, without re-running a multi-hour sweep.
            "plan_spec": _plan_json(r.best.plan)},
        "candidates": [{"plan": c.plan.name, "utilization": c.utilization,
                        "passed": c.passed, "median_ms": c.median_ms,
                        "speedup": c.speedup, "error": c.error}
                       for c in r.candidates],
    }


def cmd_sweep_one(args):
    """Search a single case and write its record. Invoked as a subprocess.

    Isolation is not a nicety here. A sweep touches torch.compile in two modes
    plus our own CUDA graph capture, and neither inductor's compiled artifacts
    nor a cudagraph memory pool is reclaimed when a case finishes. Run
    back-to-back in one process they accumulate until the device runs out --
    which killed two full sweeps on an 8 GB card, silently and without a
    traceback. One process per case bounds that, and turns a crash into one lost
    row instead of a lost run.
    """
    dev = _setup()
    spec = probe(measure=False)
    # resolve() parses anything not in the built-in matrix, so a subprocess can
    # be handed an arbitrary published shape spec, not just a known case name.
    case = shapes.resolve([args.case])[0]
    r = search.search(case, dev, margin=args.margin, trials=args.trials,
                      smem_kb=spec.shared_mem_per_block_kb,
                      time_budget_s=args.case_budget)
    print(r.summary(), flush=True)
    payload = _record_for(case, r)
    if r.best:
        payload["_plan"] = _plan_json(r.best.plan)
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return 0


def cmd_sweep(args):
    import subprocess
    import tempfile
    from .optimized import Plan

    spec = probe()
    table = DispatchTable.load(spec.arch)
    cases = _cases(args)
    print(f"# {spec.summary()}")
    print(f"# {host_summary()}")
    print(f"# {len(cases)} cases, margin={args.margin}, "
          f"{'isolated' if not args.in_process else 'in-process'}\n", flush=True)

    records = []
    started = time.time()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.name}", flush=True)
        rec = None
        if args.in_process:
            dev = _setup()
            try:
                r = search.search(case, dev, margin=args.margin,
                                  trials=args.trials,
                                  smem_kb=spec.shared_mem_per_block_kb,
                                  time_budget_s=args.case_budget)
                print(r.summary())
                rec = _record_for(case, r)
                if r.best:
                    rec["_plan_obj"] = r.best.plan
            except Exception as e:
                print(f"    ERROR {type(e).__name__}: {str(e)[:120]}")
        else:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                out_path = tf.name
            cmd = [sys.executable, "-u", "-m", "kernelforge.cli", "sweep-one",
                   "--case", case.name, "--margin", str(args.margin),
                   "--trials", str(args.trials),
                   "--case-budget", str(args.case_budget), "--out", out_path]
            proc = subprocess.run(cmd, cwd=os.path.dirname(RESULTS),
                                  capture_output=True, text=True,
                                  timeout=args.case_budget * 6)
            sys.stdout.write(proc.stdout)
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-4:]
                print(f"    case FAILED (exit {proc.returncode}) — skipped\n"
                      f"    {' | '.join(tail)[:400]}", flush=True)
            elif os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as f:
                    rec = json.load(f)
                if rec.get("_plan"):
                    p = dict(rec.pop("_plan"))
                    p["overrides"] = tuple(tuple(o) for o in p.get("overrides") or [])
                    p["flash_block"] = (tuple(p["flash_block"])
                                        if p.get("flash_block") else None)
                    rec["_plan_obj"] = Plan(**p)
            try:
                os.unlink(out_path)
            except OSError:
                pass
        if rec is None:
            continue
        r = None
        best_plan = rec.pop("_plan_obj", None)
        if best_plan is not None and rec.get("best"):
            b = rec["best"]
            table.add(Entry(
                signature=shape_signature(case),
                dtype=case.dtype, arch=spec.arch, regime=case.regime,
                case=case.name,
                plan=best_plan, utilization=b["utilization"],
                speedup=b["speedup"],
                speedup_vs_compile=b["speedup_vs_compile"]))
            table.save(spec.arch)
        records.append(rec)
        os.makedirs(RESULTS, exist_ok=True)
        records = _merge_records(device_slug(spec), records)
        with open(os.path.join(RESULTS, f"sweep_{device_slug(spec)}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"gpu": spec.summary(), "arch": spec.arch,
                       "device": device_slug(spec),
                       # Host identity matters: the GPU is exclusive but the CPU
                       # is shared, and the baseline issues ~105 kernel launches
                       # per forward. A slower host inflates the baseline far
                       # more than our graph-captured plans, so two runs on
                       # different nodes are not directly comparable.
                       "node": os.environ.get("SLURMD_NODENAME")
                               or platform.node(),
                       "slurm_job": os.environ.get("SLURM_JOB_ID"),
                       "host": host_summary(),
                       "elapsed_s": time.time() - started,
                       "records": records}, f, indent=2)
    print(f"\ndispatch table -> {table.path_for(spec.arch)}")
    print(summarize(table))
    return 0


def cmd_rebuild_table(args):
    """Rebuild dispatch tables from the sweep artifacts already on disk.

    Useful after a change to the collision-resolution policy: the winning plan
    for every case is recorded in `sweep_*.json`, so the table is derivable
    without spending another multi-hour sweep on the GPU.

    WARNING: this rebuilds from *sweep winners*, which discards any demotion a
    previous `verify --demote` applied. A sweep only measures the cases it was
    given, so a plan that won `default_causal_pad` lands on the shared causal
    signature and may fail the unpadded variant -- which is exactly what
    happened here (envelope 1.149). **Always follow a rebuild with
    `verify --demote`.**
    """
    import glob
    from .shapes import all_cases, resolve
    by_name = {c.name: c for c in all_cases()}
    tables = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_*.json"))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        arch = blob.get("arch")
        if not arch:
            continue
        table = tables.setdefault(arch, DispatchTable())
        for rec in blob.get("records", []):
            best = rec.get("best") or {}
            spec = best.get("plan_spec")
            case = by_name.get(rec["case"])
            if case is None:
                try:                       # a tuned-in shape spec, not a named case
                    case = resolve([rec["case"]])[0]
                except Exception:
                    case = None
            if not spec or case is None:
                continue
            table.add(Entry(
                signature=shape_signature(case), dtype=case.dtype, arch=arch,
                regime=case.regime, plan=_plan_from_json(spec),
                utilization=best["utilization"], speedup=best["speedup"],
                speedup_vs_compile=best.get("speedup_vs_compile", float("nan")),
                case=case.name))
    for arch, table in tables.items():
        print(f"{arch}: {len(table.entries)} entries -> {table.save(arch)}")
    if tables:
        print("\nNOTE: rebuilding restores sweep winners and discards any "
              "demotion a previous verify applied.\n"
              "      Run `python -m kernelforge.cli verify --demote` "
              "before trusting this table.")
    if not tables:
        print("no rebuildable sweep artifacts found (records lack plan_spec)")
    return 0


def cmd_env(args):
    """Dump the full machine description the tech report requires."""
    payload = envmod.describe()
    print(envmod.format_report(payload))
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"environment_{device_slug(probe(measure=False))}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten: {path}")
    return 0


def cmd_table(args):
    spec = probe(measure=False)
    print(summarize(DispatchTable.load(spec.arch)))
    return 0


def cmd_tune(args):
    """Tune shapes we have never seen, then prove the result.

    The problem statement says the official input-shape combinations will be
    told to the participants. Until they are, our matrix is an informed guess --
    so the system has to absorb a published list without a code change. This is
    that path end to end: search each shape, freeze the winners, then re-verify
    the whole table and demote anything that drifted.

        python -m kernelforge.cli tune --shapes B4-S777-d640-H10-F2560-L9
        python -m kernelforge.cli tune --shapes-file official_shapes.txt
    """
    if not args.cases and not args.shapes_file:
        print("give --shapes <spec,spec> or --shapes-file <path>")
        return 1
    cases = _cases(args)
    if not cases:
        print("nothing to tune")
        return 1
    spec = probe(measure=False)
    print(f"# tuning {len(cases)} shape(s) on {spec.arch}")
    for c in cases:
        print(f"#   {c.label():<52} regime={c.regime}")
    print()
    rc = cmd_sweep(args)
    if rc:
        return rc
    print("\n=== re-verifying the whole table (new entries included) ===")
    verify_args = argparse.Namespace(
        cases=None, regimes=None, no_variants=False, trials=args.trials,
        margin=0.90, demote=True, shapes_file=None)
    return cmd_verify(verify_args)


def cmd_codegen(args):
    """Have the LLM write actual Triton source, then gate it.

    This is the track's "AI-based code generation" in scope, as opposed to
    AI-selected configuration. Generated sources land in results/generated/ for
    review; none of them ship without a human reading them first.
    """
    from .agent import codegen, proposers
    dev = _setup()
    spec = probe()          # measure bandwidth: the model is told the real number
    proposer = proposers.build(args.provider)
    if not hasattr(proposer, "raw_completion"):
        print("codegen needs an LLM proposer; set SOCLAAS_API_KEY in .env")
        return 1
    targets = args.targets.split(",") if args.targets else list(codegen.TARGETS)
    unknown = [t for t in targets if t not in codegen.TARGETS]
    if unknown:
        print(f"unknown target(s) {unknown}; available: {list(codegen.TARGETS)}")
        return 1
    print(f"# {spec.summary()}")
    print(f"# proposer: {proposer.name} ({getattr(proposer, 'model', '?')}), "
          f"{args.iterations} attempts per target")
    codegen.run_codegen(dev, targets, args.iterations, proposer,
                        spec.as_prompt_block(), repair_budget=args.repair)
    return 0


def cmd_agent(args):
    from .agent.loop import run_agent
    dev = _setup()
    return run_agent(dev, _cases(args), iterations=args.iterations,
                     margin=args.margin, provider=args.provider, tag=args.tag)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kernelforge")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cases", default=None,
                        help="comma separated case names or shape specs")
        # Every command that resolves cases can take the official list from a
        # file. `tune` had this first; not having it on `verify` meant checking
        # the frozen table against the official shapes required pasting twelve
        # specs onto the command line, which is exactly how a shape gets left
        # out of a verification run.
        sp.add_argument("--shapes-file", default=None,
                        help="file of shape specs, one per line, # comments "
                             "allowed (e.g. official_shapes.txt)")
        sp.add_argument("--regimes", default=None,
                        help="latency,gemm,attention,bandwidth")
        sp.add_argument("--no-variants", action="store_true")
        sp.add_argument("--trials", type=int, default=3)
        return sp

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    common(sub.add_parser("budget")).set_defaults(func=cmd_budget)
    sp = common(sub.add_parser("verify"))
    sp.add_argument("--margin", type=float, default=0.90,
                    help="demotion threshold; deliberately looser than the "
                         "sweep's 0.80 admission margin (see cmd_verify)")
    sp.add_argument("--demote", action="store_true",
                    help="replace any entry that re-measures above the "
                         "demotion threshold with the bit-exact plan")
    sp.set_defaults(func=cmd_verify)

    sp = common(sub.add_parser("sweep"))
    sp.add_argument("--margin", type=float, default=0.80)
    sp.add_argument("--case-budget", type=float, default=300.0)
    sp.add_argument("--in-process", action="store_true",
                    help="run every case in this process (accumulates CUDA "
                         "graph pools and inductor caches; not recommended)")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("sweep-one")
    sp.add_argument("--case", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--margin", type=float, default=0.80)
    sp.add_argument("--trials", type=int, default=3)
    sp.add_argument("--case-budget", type=float, default=300.0)
    sp.set_defaults(func=cmd_sweep_one)

    sub.add_parser("table").set_defaults(func=cmd_table)
    sub.add_parser("env").set_defaults(func=cmd_env)
    sub.add_parser("rebuild-table").set_defaults(func=cmd_rebuild_table)

    sp = common(sub.add_parser("tune"))
    sp.add_argument("--shapes", dest="cases", default=None,
                    help="comma separated shape specs, e.g. "
                         "B8-S128-d512-H8-F2048-L6-causal")
    sp.add_argument("--margin", type=float, default=0.80)
    sp.add_argument("--case-budget", type=float, default=420.0)
    sp.add_argument("--in-process", action="store_true")
    sp.set_defaults(func=cmd_tune)

    sp = sub.add_parser("codegen")
    sp.add_argument("--targets", default=None,
                    help="comma separated: layernorm,gelu")
    sp.add_argument("--iterations", type=int, default=6)
    sp.add_argument("--provider", default="llm")
    sp.add_argument("--repair", type=int, default=0, metavar="N",
                    help="consecutive repair attempts allowed per failing "
                         "kernel before starting fresh (0 = pure resampling)")
    sp.set_defaults(func=cmd_codegen)

    sp = common(sub.add_parser("agent"))
    sp.add_argument("--iterations", type=int, default=12)
    sp.add_argument("--margin", type=float, default=0.80)
    sp.add_argument("--provider", default="auto",
                    help="auto | heuristic | llm | anthropic. 'auto' uses an "
                         "LLM when credentials are configured, else heuristic.")
    sp.add_argument("--tag", default=None,
                    help="suffix for the genealogy artifact, e.g. 'llm' vs "
                         "'heuristic', so two proposers can be compared")
    sp.set_defaults(func=cmd_agent)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
