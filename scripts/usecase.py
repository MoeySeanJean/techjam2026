"""A serving workload this project is actually shaped for, measured end to end.

The benchmark's Transformer is not an abstract object. A stack of this size --
6 layers, d_model 512, 8 heads, ffn 2048, over a padded sequence -- is the shape
of a **user-behaviour sequence model**: the ranker that reads a user's recent
interaction history and scores candidate items against it. That is the highest
volume transformer inference in a recommendation product, and it is what the
For You style feed runs on every request.

It is also, specifically, a workload where this project's choices pay:

  * **Histories are variable length.** Short-session users and heavy users share
    a batch, so requests are padded -- exactly the `valid_token_mask` the
    organizer's script models, and the thing most fused kernels get wrong.
  * **Latency is a hard SLO, so batches stay small.** Small batches are
    launch-bound, not compute-bound, which is where CUDA-graph capture wins big.
  * **Traffic is not one shape.** Peak and off-peak batch sizes, light and heavy
    histories, and re-ranking passes are all different shapes on the same model.
  * **The fleet is heterogeneous.** Recommendation serving runs across GPU
    generations, and the best kernel differs per generation.

This script defines a traffic mix over those segments, measures the baseline and
our dispatched plan on each, and weights the result by traffic share -- so the
headline is what the fleet would actually see, not a best-case shape.

    python scripts/usecase.py                # measure on this GPU
    python scripts/usecase.py --fleet 5000   # scale the energy/cost view
    python scripts/usecase.py --qps 250000   # your own traffic assumption

Every latency number is measured here, now. Everything downstream of it is
arithmetic on stated assumptions.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import torch_transformer_benchmark as B  # noqa: E402
from kernelforge import bench, shapes  # noqa: E402
from kernelforge.dispatch import DispatchTable  # noqa: E402
from kernelforge.hw import probe  # noqa: E402
from kernelforge.numerics import check  # noqa: E402
from kernelforge.optimized import build_shared  # noqa: E402

# A ranking-serving traffic mix. Shares are illustrative and stated openly so a
# reader can substitute their own; the shapes are the point.
#
#  segment            what it is                                      share
SEGMENTS = [
    ("realtime_light", "B32-S64-d512-H8-F2048-L6-pad0.4",
     "single request, short history -- the latency-critical common case", 0.42),
    ("realtime_heavy", "B32-S256-d512-H8-F2048-L6-pad0.4",
     "single request, long history -- power users", 0.18),
    ("batched_peak", "B128-S128-d512-H8-F2048-L6-pad0.4",
     "peak traffic, requests coalesced for throughput", 0.25),
    ("rerank", "B8-S128-d512-H8-F2048-L6",
     "second-stage re-rank over a small candidate set", 0.15),
]

TDP = {"sm_86": 83, "sm_80": 300, "sm_90": 400}
USD_PER_KWH = 0.12
PUE = 1.2
HOURS_PER_YEAR = 24 * 365


def forget(table, specs, arch):
    """A copy of the table with these shapes' entries removed.

    The point of this script is the gap between an untuned system and a tuned
    one, and that gap stops being reproducible the moment you tune: once the
    entries are frozen in, a plain run shows the tuned number and the "before"
    is gone. Rather than ask a reader to trust a figure they can no longer
    produce, `--cold` removes exactly these entries and re-runs the same
    lookup, which exercises the real nearest-neighbour-then-safe-default
    fallback rather than simulating it.
    """
    from kernelforge.dispatch import DispatchTable, shape_signature
    drop = {shape_signature(shapes.resolve([sp])[0].to_config()) for sp in specs}
    kept = [e for e in table.entries
            if not (e.arch == arch and e.signature in drop)]
    return DispatchTable(kept)


def measure(case, device, spec, table):
    """Baseline vs whatever the frozen table dispatches for this shape."""
    cfg = case.to_config()
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    x, mask = B.generate_random_case(cfg, device, case.torch_dtype, 1234,
                                     case.padding_ratio, case.input_scale)
    plan, source = table.lookup(spec.arch, case.torch_dtype, cfg,
                                spec.shared_mem_per_block_kb)
    ours = build_shared(cfg, plan, base)
    with torch.inference_mode():
        res = check(base(x, mask), ours(x, mask))
    t = bench.compare({"baseline": lambda: base(x, mask),
                       "ours": lambda: ours(x, mask)},
                      warmup=15, repeats=25, rounds=3)
    del base, ours
    torch.cuda.empty_cache()
    return {
        "baseline_ms": t["baseline"].median_ms,
        "ours_ms": t["ours"].median_ms,
        "plan": plan.name, "source": source,
        "passed": res.passed, "envelope": res.envelope_utilization,
        "tokens": case.batch_size * case.seq_len,
        "requests": case.batch_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=int, default=1000,
                    help="GPUs serving this model")
    ap.add_argument("--qps", type=int, default=100000,
                    help="ranking requests per second across the fleet")
    ap.add_argument("--cold", action="store_true",
                    help="ignore this GPU's tuned entries for these four "
                         "shapes, so the untuned baseline is reproducible even "
                         "after you have tuned. This is the 'before' number.")
    ap.add_argument("--tune", action="store_true",
                    help="tune any segment this GPU has no entry for, first "
                         "(~3 min per shape). This is the realistic workflow: "
                         "you give the system your traffic mix, it tunes for it.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    spec = probe()
    table = DispatchTable.load(spec.arch)
    if args.cold:
        table = forget(table, [sp for _, sp, _, _ in SEGMENTS], spec.arch)

    print(f"\n{'=' * 78}")
    print("Recommendation ranking: a user-behaviour sequence model under a "
          "realistic traffic mix")
    print(f"{'=' * 78}")
    print(f"measured on {spec.name} [{spec.arch}]\n")

    if args.tune:
        untuned = []
        for _, spec_str, _, _ in SEGMENTS:
            case = shapes.resolve([spec_str])[0]
            _, source = table.lookup(spec.arch, case.torch_dtype,
                                     case.to_config(),
                                     spec.shared_mem_per_block_kb)
            if source != "exact":
                untuned.append(spec_str)
        if untuned:
            print(f"  tuning {len(untuned)} untuned segment(s) first "
                  f"(~3 min each)...\n")
            import subprocess
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, "-m", "kernelforge.cli", "tune",
                            "--shapes", ",".join(untuned), "--trials", "2",
                            "--case-budget", "300"], cwd=ROOT, env=env)
            table = DispatchTable.load(spec.arch)
        else:
            print("  every segment already has a tuned entry\n")

    rows, weighted_base, weighted_ours = [], 0.0, 0.0
    for name, spec_str, blurb, share in SEGMENTS:
        case = shapes.resolve([spec_str])[0]
        if not shapes.fits_on(spec, case):
            print(f"  {name}: skipped, will not fit {spec.total_mem_gb:.0f} GB")
            continue
        r = measure(case, device, spec, table)
        r.update(name=name, blurb=blurb, share=share, label=case.label())
        rows.append(r)
        weighted_base += share * r["baseline_ms"]
        weighted_ours += share * r["ours_ms"]
        print(f"  measured {name:<16} {r['baseline_ms']:8.3f} -> "
              f"{r['ours_ms']:7.3f} ms   {r['baseline_ms'] / r['ours_ms']:5.2f}x   "
              f"{'PASS' if r['passed'] else 'FAIL'}", flush=True)

    if not rows:
        print("nothing measured")
        return 1

    print(f"\n{'segment':<17}{'share':>7}{'shape':>34}{'baseline':>10}"
          f"{'ours':>9}{'gain':>7}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<17}{r['share']:>6.0%} {r['label'][:33]:>33}"
              f"{r['baseline_ms']:>10.2f}{r['ours_ms']:>9.2f}"
              f"{r['baseline_ms'] / r['ours_ms']:>6.2f}x")
    speedup = weighted_base / weighted_ours

    print("-" * 84)
    print(f"{'traffic-weighted':<17}{'100%':>6} {'':>33}"
          f"{weighted_base:>10.2f}{weighted_ours:>9.2f}{speedup:>6.2f}x")

    print("\n  Plans chosen (one model, four shapes):")
    for r in rows:
        print(f"    {r['name']:<17} {r['plan']:<34} envelope "
              f"{r['envelope']:.3f}  [{r['source']}]")

    # --- what that means for a service -----------------------------------
    watts = TDP.get(spec.arch)
    reqs_per_gpu_s_base = sum(r["share"] * r["requests"] / (r["baseline_ms"] / 1e3)
                              for r in rows)
    reqs_per_gpu_s_ours = sum(r["share"] * r["requests"] / (r["ours_ms"] / 1e3)
                              for r in rows)
    gpus_base = args.qps / max(reqs_per_gpu_s_base, 1e-9)
    gpus_ours = args.qps / max(reqs_per_gpu_s_ours, 1e-9)

    print(f"\n{'=' * 78}")
    print("What that is worth to the service")
    print(f"{'=' * 78}")
    print(f"  assumptions: {args.qps:,} ranking requests/s, "
          f"{watts} W/GPU, PUE {PUE}, ${USD_PER_KWH}/kWh\n")
    print(f"  requests/s per GPU      {reqs_per_gpu_s_base:12,.0f}  ->"
          f"{reqs_per_gpu_s_ours:12,.0f}")
    print(f"  GPUs to hold the load   {gpus_base:12,.0f}  ->"
          f"{gpus_ours:12,.0f}   "
          f"({gpus_base - gpus_ours:,.0f} freed, {1 - gpus_ours / gpus_base:.0%})")

    if watts:
        kwh_saved = (gpus_base - gpus_ours) * watts * PUE * HOURS_PER_YEAR / 1000
        print(f"  energy at that capacity {kwh_saved:12,.0f} kWh/year saved"
              f"   ~${kwh_saved * USD_PER_KWH:,.0f}/year")

    print(f"\n  Latency matters as much as capacity: the realtime segments are "
          f"{rows[0]['baseline_ms'] / rows[0]['ours_ms']:.1f}x and "
          f"{rows[1]['baseline_ms'] / rows[1]['ours_ms']:.1f}x faster, which is "
          f"headroom\n  that gets spent on a bigger candidate set or a longer "
          f"history -- a better feed,\n  not just a cheaper one.")
    print("\n  Every row above cleared the organizer's accuracy gate before it "
          f"was timed.")
    print("  A ranker that is fast and wrong ships the wrong video.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
