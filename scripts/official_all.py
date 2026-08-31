"""Run the organizer's benchmark over every official shape on this GPU.

Reads the frozen tables, never writes them. Dispatch resolves in two tiers: this
card's own device table if it has one, falling back to the architecture table.
`plan_source` records which tier answered for each shape, so a run says plainly
whether it was measured on tuned plans or inherited ones.

Each shape runs in its own subprocess: a shape that exhausts memory on a small
card then costs one row rather than the whole run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from kernelforge import shapes  # noqa: E402
from kernelforge.hw import probe  # noqa: E402


def flags(spec: str):
    c = shapes.resolve([spec])[0].to_config()
    out = ["--batch-size", str(c.batch_size), "--seq-len", str(c.seq_len),
           "--d-model", str(c.d_model), "--heads", str(c.num_heads),
           "--ffn-dim", str(c.ffn_dim), "--layers", str(c.num_layers)]
    if c.causal:
        out.append("--causal")
    return out


def parse(text: str):
    got = {}
    m = re.search(r"^summary:\s*(\w+)\s*\|\s*max_abs=([0-9.eE+-]+)", text, re.M)
    if m:
        got["passed"] = m.group(1) == "PASS"
        got["max_abs"] = float(m.group(2))
    for key, tag in (("baseline_ms", "baseline"), ("ours_ms", "optimized")):
        m = re.search(rf"^{tag}\s*:\s*median=([0-9.]+)\s*ms", text, re.M)
        if m:
            got[key] = float(m.group(1))
    m = re.search(r"^speedup\s*:\s*([0-9.]+)x", text, re.M)
    if m:
        got["speedup"] = float(m.group(1))
    # `[kernelforge] sm_90 torch.float32 -> plan-name (source): ...`
    # `source` is the dispatch tier: `exact` means this card had its own entry.
    m = re.search(r"\[kernelforge\][^\r\n]*->\s*(.+?)\s*\(([a-z]+)\)", text)
    if m:
        got["plan"], got["plan_source"] = m.group(1), m.group(2)
    return got


def main() -> int:
    spec = probe(measure=False)
    name = spec.name.replace(" ", "-").replace("/", "-")
    with open(os.path.join(ROOT, "official_shapes.txt"), encoding="utf-8") as f:
        specs = [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"{spec.name} [{spec.arch}] {spec.sm_count} SMs, {total:.1f} GB, "
          f"smem/block {spec.shared_mem_per_block_kb:.0f} KB")
    print(f"{len(specs)} official shapes, organizer's script, frozen table\n")

    records = []
    for i, sp in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {sp}", flush=True)
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_official.py")]
            + flags(sp),
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, "KERNELFORGE_VERBOSE": "1"})
        got = parse(p.stdout)
        if got.get("speedup"):
            print(f"        {got['speedup']:.3f}x  [{got.get('plan_source','?')}] "
                  f"{got.get('plan','?')[:34]}  "
                  f"{'PASS' if got.get('passed') else 'FAIL'}  "
                  f"baseline {got.get('baseline_ms', float('nan')):.3f} ms  "
                  f"ours {got.get('ours_ms', float('nan')):.3f} ms", flush=True)
        else:
            tail = (p.stdout + p.stderr).strip().splitlines()
            reason = next((l for l in reversed(tail)
                           if "Error" in l or "error" in l), "did not complete")
            got = {"ran": False, "error": reason[:160]}
            print(f"        {reason[:110]}", flush=True)
        got["case"] = sp
        records.append(got)

    ran = [r for r in records if r.get("speedup")]
    exact = sum(1 for r in ran if r.get("plan_source") == "exact")
    passed = [r for r in ran if r.get("passed")]
    out = {
        "gpu": spec.name, "arch": spec.arch, "sm_count": spec.sm_count,
        "total_gib": round(total, 1),
        "smem_per_block_kb": spec.shared_mem_per_block_kb,
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "method": ("The organizer's torch_transformer_benchmark.py, unmodified, "
                   "with UserOptimizedTransformer swapped for ours. Plans are "
                   "read from the frozen tables -- this card's device table "
                   "where it has one, the architecture table otherwise. Nothing "
                   "is tuned here; see `plan_source` per shape for which tier "
                   "answered."),
        "shapes_run": len(ran), "shapes_total": len(specs),
        "shapes_passed": len(passed),
        "shapes_from_this_cards_own_table": exact,
        "per_device_table_present": bool(os.environ.get("KF_TUNED")),
        "table_note": ("`plan_source` is the dispatch tier that answered: "
                       "`exact` means this card's own device table had an entry "
                       "for the shape, `nearest`/`default` mean it fell through "
                       "to the architecture table."),
        "records": records,
    }
    if ran:
        s = sorted(r["speedup"] for r in ran)
        out["median_speedup"] = round(s[len(s) // 2], 3)
        out["min_speedup"] = round(s[0], 3)
        out["max_speedup"] = round(s[-1], 3)
        print(f"\n{len(passed)}/{len(ran)} pass, median {out['median_speedup']}x "
              f"(range {out['min_speedup']}-{out['max_speedup']}x)")

    # The untuned pass and the per-device-tuned pass must both survive: the
    # comparison between them is the point of running twice.
    prefix = "tuned_official" if os.environ.get("KF_TUNED") else "official"
    path = os.path.join(ROOT, "results", f"{prefix}_{name}_{spec.arch}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
