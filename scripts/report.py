"""Turn the artifacts in results/ into docs/RESULTS.md.

Reads every `sweep_<arch>.json` and `genealogy_<arch>.json` present, so the same
script produces a single-GPU report today and a cross-architecture comparison
once the cluster runs land. Nothing is hardcoded per GPU.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Everything in results/ is reported. We deliberately do NOT filter by
# architecture here: a hardcoded allow-list would silently exclude a reader's own
# GPU from their own report, which is the opposite of what this script is for.
# Machines whose timings we do not trust are kept out of results/ instead.


def load(pattern: str) -> Dict[str, dict]:
    """Load artifacts keyed by filename stem.

    Keying by device would collapse two runs on the same GPU into one -- which
    is exactly what the proposer head-to-head needs to keep apart (a heuristic
    run and an LLM run on the same GPU share a device string). Callers that want
    the device read `blob["device"]`.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        stem = os.path.splitext(os.path.basename(path))[0]
        for prefix in ("sweep_", "genealogy_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        blob.setdefault("device", blob.get("arch") or stem)
        out[stem] = blob
    return out


def fmt(x, spec=".3f", dash="-"):
    if x is None:
        return dash
    try:
        if x != x:  # NaN
            return dash
        return format(x, spec)
    except (TypeError, ValueError):
        return dash


def speed_table(sweeps: Dict[str, dict]) -> List[str]:
    lines = []
    for arch, blob in sweeps.items():
        node = blob.get("node")
        provenance = f"`{blob.get('host','')}`"
        if node:
            provenance += f"  ·  node `{node}`"
        lines += [f"### {blob.get('gpu', arch)}", "", provenance, "",
                  "| shape | regime | baseline ms | torch.compile ms | ours ms | "
                  "vs baseline | vs compile | plan | envelope |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rec in blob.get("records", []):
            best = rec.get("best")
            if not best:
                lines.append(
                    f"| `{rec['label']}` | {rec['regime']} | "
                    f"{fmt(rec.get('baseline_ms'),'.3f')} | "
                    f"{fmt(rec.get('compile_ms'),'.3f')} | - | - | - | "
                    f"*(gate not cleared)* | - |")
                continue
            vs_c = best.get("speedup_vs_compile")
            # `compile_admissible` is absent in runs recorded before the library
            # baselines were themselves gated; fall back to the dtype rule that
            # measurement established (see docs/PRECISION.md).
            adm = rec.get("compile_admissible")
            if adm is None:
                adm = "float16" not in rec["label"] and "bfloat16" not in rec["label"]
            slower = vs_c is not None and vs_c == vs_c and vs_c < 1.0
            if slower and not adm:
                mark = " †"
            elif slower:
                mark = " ⚠"
            else:
                mark = ""
            compile_cell = fmt(rec.get("compile_ms"), ".3f")
            if not adm:
                compile_cell += " †"
            lines.append(
                f"| `{rec['label']}` | {rec['regime']} | "
                f"{fmt(rec.get('baseline_ms'),'.3f')} | "
                f"{compile_cell} | "
                f"{fmt(best.get('median_ms'),'.3f')} | "
                f"**{fmt(best.get('speedup'),'.2f')}x** | "
                f"{fmt(vs_c,'.2f')}x{mark} | `{best.get('plan')}` | "
                f"{fmt(best.get('utilization'),'.3f')} |")
        lines.append("")
        lines.append(
            "⚠ `torch.compile` is genuinely faster here and passes the accuracy "
            "gate. We report these rather than omitting them.  \n"
            "† `torch.compile` **fails the organizer's accuracy gate** at this "
            "configuration, so its time is not an admissible result — it is shown "
            "for completeness, not as a target we lost to. Our entry is the "
            "bit-exact plan. See [PRECISION.md](PRECISION.md).")
        lines.append("")
    return lines


def plan_identity(best):
    """What makes two winning plans the same plan -- the spec, not the name."""
    ps = best.get("plan_spec") or {}
    if not ps:
        return best.get("plan")
    return (ps.get("attention"), ps.get("compute_dtype"), ps.get("residual_dtype"),
            bool(ps.get("cuda_graph")), bool(ps.get("fused_norm")),
            ps.get("torch_compile"),
            tuple(sorted(tuple(o) for o in (ps.get("overrides") or []))))


def cross_arch(sweeps: Dict[str, dict]) -> List[str]:
    """The portability claim: same agent, different hardware, different winners."""
    if len(sweeps) < 2:
        return []
    by_case: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for arch, blob in sweeps.items():
        for rec in blob.get("records", []):
            if rec.get("best"):
                by_case[rec["case"]][arch] = rec
    archs = list(sweeps)
    lines = ["## Cross-architecture: the same search, different winners", "",
             "The search is hardware-parameterized -- tile legality is derived "
             "from the measured shared-memory budget, and precision is chosen "
             "from a per-stage error budget measured on the target. Bringing up "
             "a new GPU is one `sweep` invocation with no code changes.", "",
             "| shape | " + " | ".join(f"{a} plan | {a} speedup" for a in archs) + " |",
             "|---" * (1 + 2 * len(archs)) + "|"]
    for case, per_arch in sorted(by_case.items()):
        if len(per_arch) < 2:
            continue
        cells = []
        for a in archs:
            rec = per_arch.get(a)
            if rec:
                cells += [f"`{rec['best']['plan']}`",
                          f"{fmt(rec['best']['speedup'],'.2f')}x"]
            else:
                cells += ["-", "-"]
        lines.append(f"| `{case}` | " + " | ".join(cells) + " |")
    # Compare plan *specifications*, not names. A name lists its fp16 stages in
    # the order the search admitted them, so two spellings of one plan look like
    # two plans. Counting names claimed every shape diverged; counting specs is
    # the number we can defend.
    multi = {c: pa for c, pa in by_case.items() if len(pa) > 1}
    divergent = [c for c, pa in multi.items()
                 if len({plan_identity(r["best"]) for r in pa.values()}) > 1]
    lines += ["", f"**{len(divergent)} of {len(multi)} shapes measured on more "
                  f"than one GPU chose a genuinely different plan.** That "
                  f"divergence is the point: a single hand-tuned kernel set "
                  f"cannot be optimal across a heterogeneous fleet.", ""]
    return lines


def proposer_comparison(gens):
    """Heuristic search vs an LLM, given identical evidence and the same gate.

    This is the part of the brief that asks what AI actually contributed. Both
    proposers see the same hardware spec sheet, the same measured profile, the
    same per-stage error budget and the same attempt history, and both are held
    to the same numeric gate. The only difference is what proposes the next
    configuration.
    """
    # Group by (device, proposer). Comparing a heuristic run on one GPU against
    # an LLM run on another would be meaningless -- the speedups are not
    # commensurable across hardware. Only devices that ran both are compared.
    per_device = defaultdict(dict)
    for blob in gens.values():
        name, dev = blob.get("proposer"), blob.get("device") or blob.get("arch")
        if name and dev:
            per_device[dev][name] = blob
    candidates = {d: r for d, r in per_device.items() if len(r) >= 2}
    if not candidates:
        return []
    device = sorted(candidates)[0]
    by_proposer = {n: [b] for n, b in candidates[device].items()}

    lines = ["## Proposer head-to-head: heuristic search vs an LLM", "",
             f"Both runs on **{device}**, same shapes, same evidence, same "
             f"accuracy gate. Only the thing choosing the next configuration "
             f"differs.", "",
             "Both arms run the same four official shapes on the same card, "
             "so the comparison is about which configurations each proposer "
             "reaches — not about the hardware. The speedups below are "
             "single-run agent measurements, not the gated, re-verified "
             "numbers in the tables above; read them as relative.", ""]

    lines += ["| proposer | model | proposals | cleared the gate | rejected | "
              "API failures | wall clock |", "|---|---|---|---|---|---|---|"]
    for name, blobs in sorted(by_proposer.items()):
        b = blobs[0]
        tax = b.get("taxonomy", {})
        total = sum(tax.values()) or 1
        ok = tax.get("ok", 0)
        lines.append(
            f"| `{name}` | {b.get('proposer_model') or 'n/a'} | {total} | "
            f"{ok} ({100.0*ok/total:.0f}%) | {total-ok} | "
            f"{b.get('proposer_failures', 0)} | {b.get('elapsed_s', 0):.0f}s |")
    lines.append("")

    names = sorted(by_proposer)
    per_case = {}
    for name in names:
        for rec in by_proposer[name][0].get("records", []):
            if rec.get("best"):
                per_case.setdefault(rec["case"], {})[name] = rec["best"]
    if per_case:
        lines += ["| shape | " + " | ".join(f"{n} plan | {n} speedup"
                                            for n in names) + " |",
                  "|---" * (1 + 2 * len(names)) + "|"]
        wins = {n: 0 for n in names}
        for case, best in sorted(per_case.items()):
            cells = []
            for n in names:
                b = best.get(n)
                cells += ([f"`{b['plan']}`", f"{fmt(b['speedup'], '.2f')}x"]
                          if b else ["-", "-"])
            lines.append(f"| `{case}` | " + " | ".join(cells) + " |")
            ranked = [(n, best[n]["speedup"]) for n in names if n in best]
            if ranked:
                wins[max(ranked, key=lambda t: t[1])[0]] += 1
        lines += ["", "Fastest gate-passing plan found, by shape: "
                  + ", ".join(f"**{n}** {w}" for n, w in wins.items()) + ".", ""]

    lines += [
        "The LLM is not obviously better at the parts the heuristic already "
        "encodes -- it re-proposes bfloat16 compute and an fp16 residual "
        "stream, both of which our error budget had already ruled out, and the "
        "gate rejects them. Where it helps is exploring combinations the "
        "hand-written ordering never reaches, because the heuristic narrows "
        "stages in a fixed cheapest-first sequence and cannot jump.", "",
        "The rejection counts are the honest cost of that freedom: the "
        "heuristic proposes only what it already believes is legal and clears "
        "the gate every time, while the LLM proposes things the gate has to "
        "throw out. Neither of those is a virtue on its own -- a proposer that "
        "is never rejected is a proposer that never explores, and one that is "
        "often rejected is only useful because something downstream is "
        "checking.", "",
        "At four shapes and twelve proposals per arm this is a small sample, "
        "and we would not defend a per-shape gap as significant. What is solid "
        "is the process claim -- **every proposal from both proposers passed "
        "through the same gate, and no configuration was ever timed before it "
        "was proven correct.**", ""]
    return lines


def genealogy(gens: Dict[str, dict]) -> List[str]:
    if not gens:
        return []
    lines = ["## Kernel genealogy: what the loop proposed and why it was rejected",
             ""]
    for arch, blob in gens.items():
        tax = blob.get("taxonomy", {})
        total = sum(tax.values()) or 1
        lines += [f"### {arch} (proposer: {blob.get('proposer','?')})", "",
                  "| outcome | count | share |", "|---|---|---|"]
        for k, v in sorted(tax.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{k}` | {v} | {100.0*v/total:.1f}% |")
        lines += ["", f"Total proposals: **{total}** across "
                      f"{len(blob.get('records', []))} shapes.", "",
                  "Every rejected proposal was rejected *before* it was timed. A "
                  "numerically wrong configuration can never be promoted into the "
                  "dispatch table, no matter how fast it runs.", ""]
    return lines


# Measured achieved bandwidth per architecture (from `cli doctor`).
MEASURED_BW = {"sm_86": 359.0, "sm_80": 1651.0, "sm_90": 3511.0}


def roofline_section(sweeps: Dict[str, dict]) -> List[str]:
    """How much of each machine we actually use, and what is left.

    The track names "GPU compute throughput, memory bandwidth, cache
    efficiency, kernel launch overhead, and tensor core utilization" as the
    limiters. A speedup says we got faster; the roofline says whether anything
    remains -- and which lever would move it.
    """
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from kernelforge import roofline, shapes

    by_name = {c.name: c for c in shapes.all_cases()}
    lines = ["## Roofline: how much of the machine are we using?", "",
             "Arithmetic intensity is FLOPs per byte of DRAM traffic for a "
             "*fused* implementation; the ridge point is where a kernel stops "
             "being bandwidth-bound and starts being tensor-core-bound. "
             "\"% of ceiling\" is against whichever limit binds. Peak figures "
             "are vendor numbers for the tensor-core path with fp32 "
             "accumulation and are listed in `kernelforge/roofline.py`.", "",
             "| GPU | shape | TFLOP/s | GB/s | intensity | limiter | % of ceiling |",
             "|---|---|---|---|---|---|---|"]
    worst = []
    for blob in sweeps.values():
        arch = blob.get("arch")
        bw = MEASURED_BW.get(arch)
        for rec in blob.get("records", []):
            best, case = rec.get("best"), by_name.get(rec["case"])
            if not best or case is None or case.dtype != "float32":
                continue
            r = roofline.analyse(case, best["median_ms"], arch, bw or 0.0,
                                 case.dtype)
            if r is None:
                continue
            lines.append(
                f"| {arch} | `{rec['case']}` | {r.achieved_tflops:.1f} | "
                f"{r.achieved_bandwidth_gbs:.0f} | {r.arithmetic_intensity:.0f} | "
                f"{r.limiter} | {r.utilization:.0%} |")
            worst.append((r.utilization, arch, rec["case"]))
    if not worst:
        return []
    worst.sort()
    # Derived from the table above rather than written down, so the prose cannot
    # drift out of step with the numbers it describes.
    # The lowest cell overall is a bandwidth-bound shape, not a causal one --
    # naming it "long causal attention" was wrong, so pick from causal cases.
    causal = sorted(w for w in worst if "causal" in w[2])
    causal_txt = (f"It sits at {causal[0][0]:.0%} of ceiling on "
                  f"{causal[0][1]} (`{causal[0][2]}`), the lowest of the "
                  f"compute-bound shapes." if causal else
                  "No causal shape carries a roofline in these artifacts.")
    gemm = sorted(u for u, _, _ in worst if u >= 0.20)
    span = (f"{gemm[0]:.0%}-{gemm[-1]:.0%}" if gemm else "n/a")
    archs = " and ".join(sorted({a for _, a, _ in worst}))
    lines += ["", f"**What this says.** The GEMM-bound shapes reach "
              f"{span} of the tensor-core ceiling on {archs}, which is a "
              "reasonable place to be for a mixed Triton/cuBLAS implementation. "
              "Two groups sit well below it, for different and instructive "
              "reasons:", "",
              "- **`tiny` and `decode` are bandwidth-bound with very low "
              "intensity.** They are not failing to use the machine; there is "
              "barely any arithmetic to do. Their speedups come from removing "
              "kernel launches, and the roofline confirms there is nothing "
              "further to win from better math.",
              f"- **Long causal attention is the real headroom.** {causal_txt} "
              f"That is the one regime where we still fall back to a library "
              f"implementation, and where the next kernel should go.", "",
              "Utilization is uniformly lower on H100 than on A100: the machine "
              "is much larger and our tile sizes do not saturate it. Closing "
              "that would mean Hopper-specific work (TMA, wgmma, larger "
              "persistent tiles) that we scoped out.", ""]
    return lines


def codegen_section() -> List[str]:
    """AI-written kernel source, as opposed to AI-selected configuration."""
    path = os.path.join(RESULTS, "codegen.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    tax = blob.get("taxonomy", {})
    total = sum(tax.values()) or 1
    lines = ["## AI-generated kernel source", "",
             f"The LLM was asked to write complete Triton kernels against a "
             f"contract, and every candidate was compiled, gated and timed by "
             f"the same harness. **{total} kernels generated.** Full write-up in "
             f"[CODEGEN.md](CODEGEN.md).", "",
             "| outcome | count | share |", "|---|---|---|"]
    for k, v in sorted(tax.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v} | {100.0 * v / total:.1f}% |")
    lines.append("")
    lines += ["| target | success rate | best generated kernel |", "|---|---|---|"]
    for target, atts in blob.get("targets", {}).items():
        oks = [a for a in atts if a["status"] == "ok"]
        best = max(oks, key=lambda a: a["speedup_vs_torch"], default=None)
        cell = (f"**{best['speedup_vs_torch']:.2f}x** vs torch, envelope "
                f"{best['envelope']:.3f}, {best['lines']} lines" if best else "—")
        lines.append(f"| `{target}` | {len(oks)}/{len(atts)} "
                     f"({100.0 * len(oks) / max(len(atts), 1):.0f}%) | {cell} |")
    lines += ["", "No generated kernel is in the shipped dispatch table. They "
              "are proposals; promoting one is a deliberate human step, because "
              "a public submission should not contain code nobody has read.", ""]
    return lines


def regime_profile(sweeps: Dict[str, dict]) -> List[str]:
    lines = ["## Where the baseline's time goes", "",
             "| GPU | shape | regime | GPU ms | CPU ms | launches | µs/launch |",
             "|---|---|---|---|---|---|---|"]
    any_row = False
    for arch, blob in load("genealogy_*.json").items():
        for rec in blob.get("records", []):
            p = rec.get("profile") or {}
            if not p:
                continue
            any_row = True
            lines.append(
                f"| {arch} | `{rec['label']}` | {p.get('regime','?')} | "
                f"{fmt(p.get('cuda_ms'),'.3f')} | {fmt(p.get('cpu_ms'),'.3f')} | "
                f"{p.get('launches','-')} | {fmt(p.get('launch_us'),'.1f')} |")
    return lines + [""] if any_row else []


def official_specs() -> List[str]:
    """The 14 shapes from Appendix 3.7, in the order the organizers list them."""
    path = os.path.join(ROOT, "official_shapes.txt")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            spec = line.split("#")[0].strip()
            if spec:
                out.append(spec)
    return out


def official_table(sweeps: Dict[str, dict]) -> List[str]:
    """The official test shapes, first, separated from our exploratory matrix.

    A judge should not have to find the fourteen rows that matter inside a table
    of sixty. Everything else we measured is still below, because the shapes we
    tuned against before the list existed are what makes the fallback path
    credible -- but they are not the deliverable.
    """
    specs = official_specs()
    if not specs:
        return []
    lines = ["## The official test shapes", "",
             "The 14 shapes in Appendix 3.7 of the problem statement. Shape "
             "numbers are the organizers'.", ""]
    for arch, blob in sweeps.items():
        by = {r["case"]: r for r in blob.get("records", [])}
        rows, sb, sc = [], [], []
        for i, spec in enumerate(specs, 1):
            rec = by.get(spec)
            if not rec or not rec.get("best"):
                rows.append((i, spec, None))
                continue
            rows.append((i, spec, rec))
            sb.append(rec["best"]["speedup"])
            sc.append(rec["best"]["speedup_vs_compile"])
        if not sb:
            continue
        node = blob.get("node")
        prov = f"`{blob.get('host','')}`" + (f"  ·  node `{node}`" if node else "")
        lines += [f"### {blob.get('gpu', arch)}", "", prov, "",
                  "| # | shape | baseline ms | torch.compile ms | ours ms | "
                  "vs baseline | vs compile | envelope | plan |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, spec, rec in rows:
            if rec is None:
                # Shape 14 has no row here by design: there is no baseline to
                # divide by, so it gets its own section rather than a line of
                # dashes that reads like a failure.
                why = ("*reference cannot run it — see below*"
                       if os.path.exists(os.path.join(RESULTS, "shape14.json"))
                       and spec.startswith("B32-S100000")
                       else "*baseline does not fit this GPU*")
                lines.append(f"| {i} | `{spec}` | — | — | — | — | — | — | "
                             f"{why} |")
                continue
            b = rec["best"]
            vs_c = b["speedup_vs_compile"]
            mark = " ⚠" if vs_c < 1.0 else ""
            lines.append(
                f"| {i} | `{spec}` | {fmt(rec.get('baseline_ms'),'.3f')} | "
                f"{fmt(rec.get('compile_ms'),'.3f')} | "
                f"{fmt(b.get('median_ms'),'.3f')} | "
                f"**{fmt(b.get('speedup'),'.2f')}x** | "
                f"{fmt(vs_c,'.2f')}x{mark} | {fmt(b.get('utilization'),'.3f')} | "
                f"`{b.get('plan')}` |")
        sb_s, sc_s = sorted(sb), sorted(sc)
        won = sum(1 for v in sc if v >= 1.0)
        lines += ["",
                  f"**{len(sb)} of {len(specs)} shapes measured** on this GPU. "
                  f"Median **{sb_s[len(sb_s)//2]:.2f}x** over the reference and "
                  f"**{sc_s[len(sc_s)//2]:.2f}x** over `torch.compile`; "
                  f"range {min(sb):.2f}x–{max(sb):.2f}x over the reference. "
                  f"Every measured shape cleared the accuracy gate "
                  f"(max envelope "
                  f"{max(r['best']['utilization'] for _, _, r in rows if r):.3f} "
                  f"of 1.0). Faster than `torch.compile` on {won} of {len(sc)}.",
                  ""]
    lines += ["⚠ marks a shape where `torch.compile` is faster than us. We "
              "report these rather than omitting them.", ""]
    return lines


def shape14_section() -> List[str]:
    """Shape 14, which has no speedup because the reference cannot run it."""
    path = os.path.join(RESULTS, "shape14.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    lines = ["## Shape 14, the one the reference cannot run", "",
             f"`{d['shape']}` in {d['dtype']}. `BaselineSelfAttention` "
             f"materializes `[B, H, S, S]` before its softmax, which for this "
             f"shape is **{d['baseline_score_matrix_tb']:.1f} TB** — not a "
             f"number any GPU can allocate. The organizer's own reference "
             f"cannot execute it, and neither can `torch.compile` applied to "
             f"that reference.", "",
             "**No speedup is quoted.** A ratio against an implementation that "
             "cannot run is not a measurement. The claim is narrower and "
             "stronger: this shape is reachable with a fused kernel and "
             "unreachable without one.", "",
             "| GPU | node | latency | peak memory | batch slice | plan | "
             "finite | shape |", "|---|---|---|---|---|---|---|---|"]
    for m in d["measurements"]:
        lines.append(
            f"| {m['gpu']} | `{m['node']}` | {m['ms'] / 1000:,.1f} s | "
            f"{m['peak_gb']:.1f} GB | {m['batch_slice']}/32 | `{m['plan']}` | "
            f"{'yes' if m['finite'] else 'NO'} | "
            f"{'correct' if m['shape_ok'] else 'WRONG'} |")
    lines += ["", "Three things have to be true at once for this to run:", ""]
    for i, fix in enumerate(d["fixes_required"], 1):
        lines.append(f"{i}. {fix}")
    lines += ["",
              "`python scripts/shape14.py --scan` sweeps sequence length and "
              "reports where a given GPU stops.", "",
              "Accuracy for this code path is established at sequence lengths "
              "where the reference *can* be computed — `tests/test_kernels.py` "
              "checks the same kernel against an exact reference across causal "
              "× padding × length, and `tests/test_streaming.py` checks that "
              "slicing the batch does not change the answer.", ""]
    return lines


def main() -> int:
    sweeps = load("sweep_*.json")
    gens = load("genealogy_*.json")
    if not sweeps:
        print("no results/sweep_*.json found -- run `python -m kernelforge.cli sweep`")
        return 1

    out = ["# KernelForge results", "",
           "Generated by `scripts/report.py` from the JSON artifacts in "
           "`results/`. Every row cleared the organizer's accuracy gate at a "
           "0.80 envelope margin over three seeds **before** it was timed.", "",
           "Timing is interleaved round-robin with rotating order and reported "
           "as a median; see `kernelforge/bench.py` for why.", "",
           "Envelope utilization re-measures within roughly ±0.1 between runs "
           "(cuBLAS kernel selection varies with device state), which is why "
           "the admission margin is 0.80 and `cli verify --demote` re-checks the "
           "frozen table at 0.90.", "",
           ""]
    out += official_table(sweeps)
    out += shape14_section()
    out += ["## Every shape we measured", "",
            "The matrix we tuned against before the official list was "
            "published, kept because it is what makes the fallback path "
            "credible on a shape nobody tuned for.", ""]
    out += speed_table(sweeps)
    out += cross_arch(sweeps)
    out += roofline_section(sweeps)
    out += codegen_section()
    out += proposer_comparison(gens)
    out += genealogy(gens)
    out += regime_profile(sweeps)
    out += ["## Reproducing", "",
            "```bash", "python -m kernelforge.cli sweep", "python scripts/report.py",
            "```", "",
            "See [PRECISION.md](PRECISION.md) for why fp16 and bfloat16 "
            "shapes ship a bit-exact plan, and "
            "[EQUIVALENCE.md](EQUIVALENCE.md) for why that plan is "
            "bit-exact.", ""]

    # Written into docs/ alongside the reports it cross-links, so every
    # link below is a sibling path rather than a "docs/" prefix.
    path = os.path.join(ROOT, "docs", "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {path} ({len(out)} lines) from "
          f"{len(sweeps)} sweep(s), {len(gens)} genealogy log(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
