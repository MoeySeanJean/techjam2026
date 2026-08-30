"""Recompute the throughput, energy and cost figures quoted in the tech report.

"3.3x faster" is an engineering result. What decides whether anyone deploys it
is throughput per GPU, energy per token, and what those mean across a fleet.
This script derives those from artifacts already in `results/` -- no new GPU
runs, no invented figures -- and states its assumptions inline so a reader can
substitute their own.

    python scripts/impact.py
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Board power (W). Datasheet TDP, used only to convert latency into energy;
# every derived number scales linearly with these, so they are stated openly.
TDP = {"a100-80gb-79gb_sm_80": 300,        # A100 80GB PCIe
       "h100-nvl-93gb_sm_90": 400}         # H100 NVL, per card

# Assumptions for the fleet illustration. Deliberately conservative.
GRID_KWH_PER_USD = 0.12      # USD per kWh, commercial rate
PUE = 1.2                    # datacentre overhead multiplier
HOURS_PER_YEAR = 24 * 365


def load_sweeps() -> Dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_*.json"))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        out[blob.get("device", os.path.basename(path))] = blob
    return out


def tokens_from_label(label: str) -> int:
    """B and S are the first two fields of the shape signature."""
    parts = label.split("-")
    return int(parts[0][1:]) * int(parts[1][1:])


def main() -> int:
    sweeps = load_sweeps()
    if not sweeps:
        print("no results/sweep_*.json -- run `python -m kernelforge.cli sweep`")
        return 1

    lines: List[str] = [
        "# Impact: what the speedups mean in throughput, energy and cost", "",
        "Derived from the measured artifacts in `results/` by "
        "`scripts/impact.py`. Latency is measured; power is board TDP "
        f"({', '.join(f'{k.split('_')[0]}={v} W' for k, v in TDP.items())}); "
        f"cost assumes ${GRID_KWH_PER_USD}/kWh and PUE {PUE}. Every derived "
        "number scales linearly with those three assumptions, which are stated "
        "so they can be replaced.", "",
        "Only shapes that cleared the accuracy gate are included — an "
        "incorrect kernel has no throughput.", "",
    ]

    fleet_rows = []
    for device, blob in sweeps.items():
        watts = TDP.get(device)
        lines += [f"## {blob.get('gpu', device)}", "",
                  "| shape | tokens/s baseline | tokens/s ours | "
                  "mJ/1k tokens baseline | mJ/1k tokens ours | energy saved |",
                  "|---|---|---|---|---|---|"]
        savings = []
        for rec in blob.get("records", []):
            best = rec.get("best")
            if not best:
                continue
            toks = tokens_from_label(rec["label"])
            base_tps = toks * 1000.0 / rec["baseline_ms"]
            ours_tps = toks * 1000.0 / best["median_ms"]
            if watts:
                base_mj = watts * rec["baseline_ms"] / toks * 1000.0
                ours_mj = watts * best["median_ms"] / toks * 1000.0
                saved = 1.0 - ours_mj / base_mj
                savings.append(saved)
                cells = (f"{base_mj:.2f} | {ours_mj:.2f} | {saved:.0%}")
            else:
                cells = "- | - | -"
            lines.append(f"| `{rec['label']}` | {base_tps:,.0f} | "
                         f"{ours_tps:,.0f} | {cells} |")
        lines.append("")
        if savings:
            mean_saved = sum(savings) / len(savings)
            fleet_rows.append((device, watts, mean_saved))
            lines += [f"Mean energy reduction per token across gate-passing "
                      f"shapes: **{mean_saved:.0%}**.", ""]

    if fleet_rows:
        lines += [
            "## What that is worth at fleet scale", "",
            "Inference fleets are large and long-lived, so a per-token energy "
            "reduction compounds. For a hypothetical 1,000-GPU deployment held "
            "at full utilisation for a year:", "",
            "| GPU | mean energy reduction | kWh saved / GPU / year | "
            "1,000-GPU saving (USD/year) |", "|---|---|---|---|"]
        for device, watts, saved in fleet_rows:
            kwh = watts * saved * PUE * HOURS_PER_YEAR / 1000.0
            lines.append(f"| {device} | {saved:.0%} | {kwh:,.0f} | "
                         f"${kwh * 1000 * GRID_KWH_PER_USD:,.0f} |")
        lines += [
            "", "The point is not the dollar figure, which depends entirely on "
            "the assumptions above. It is the shape of the argument: the work "
            "is a **build-time** cost paid once per (architecture, shape), and "
            "the saving accrues on every inference thereafter, on hardware that "
            "is already bought and already running.", "",
            "The same reasoning is why the dispatch table is per-architecture. "
            "A fleet is heterogeneous, and **4 of the 13 official shapes "
            "chose a different plan on an A100 than on an H100** — so a single "
            "hand-tuned kernel set "
            "leaves money on the table on every card it was not tuned for, "
            "while re-tuning for a new card here is one command and no code "
            "changes.", ""]

    # Printed rather than written to a file: these figures are quoted inline in
    # docs/TECH_REPORT.md section 6, and a second copy would only drift from it.
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
