"""Build the KernelForge dashboard: a self-contained HTML decision explorer.

The results tables answer "how fast". They do not convey the thing the project
is actually about -- that the *same shape on a different GPU picks a different
kernel*, and that every candidate had to clear a correctness gate before it was
allowed to post a time. Those are interactions, not rows.

So this generates one page driven by a single gesture: choose a GPU and a shape,
and watch the system decide. Everything the decision rested on is shown beside
it -- the candidates that were tried, which cleared the gate, the per-stage
error budget that drove the precision choice, and the measured profile.

Everything is inlined: the page is a single file with no network dependency, so
it opens from the repo, attaches to a submission, or drives a demo recording.

    python scripts/dashboard.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dashboard_template.html")
OUT = os.path.join(ROOT, "docs", "dashboard.html")

# Board power (W), by device: the two sm_75 cards differ by 4x, so this cannot
# be keyed on architecture. Bandwidth is not listed here -- each sweep records
# what the probe measured on the node, and `_bandwidth` reads it back.
TDP = {
    "a100-80gb-79gb_sm_80": 300,
    "h100-nvl-93gb_sm_90": 400,
    "titan-23gb_sm_75": 280,
    "titan-v-12gb_sm_70": 250,
    "tesla-t4-15gb_sm_75": 70,
}

NICE = {
    "a100-80gb-79gb_sm_80": "A100-80 PCIe",
    "h100-nvl-93gb_sm_90": "H100 NVL",
    "titan-23gb_sm_75": "TITAN RTX",
    "titan-v-12gb_sm_70": "TITAN V",
    "tesla-t4-15gb_sm_75": "Tesla T4",
}
# Newest and largest first; the two tuned datacentre parts lead.
ORDER = ["h100-nvl-93gb_sm_90",
         "a100-80gb-79gb_sm_80",
         "titan-23gb_sm_75",
         "tesla-t4-15gb_sm_75",
         "titan-v-12gb_sm_70"]


def _bandwidth(blob: dict):
    """Measured GB/s, as the sweep's probe recorded it in the `gpu` string."""
    m = re.search(r"~(\d+(?:\.\d+)?) GB/s", blob.get("gpu", ""))
    return float(m.group(1)) if m else None


def load(pattern: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def build_payload() -> dict:
    sweeps = {b.get("device"): b for b in load("sweep_*.json")}
    profiles: Dict[str, Dict[str, dict]] = {}
    for blob in load("genealogy_*.json"):
        dev = blob.get("device")
        for rec in blob.get("records", []):
            if rec.get("profile"):
                profiles.setdefault(dev, {})[rec["case"]] = rec["profile"]

    devices, cases = [], {}
    for dev in ORDER:
        blob = sweeps.get(dev)
        if not blob:
            continue
        arch = blob.get("arch")
        devices.append({
            "id": dev, "name": NICE.get(dev, dev), "arch": arch,
            "gpu": blob.get("gpu", ""), "node": blob.get("node"),
            "tdp": TDP.get(dev), "bandwidth": _bandwidth(blob),
        })
        for rec in blob.get("records", []):
            best = rec.get("best")
            if not best:
                continue
            entry = cases.setdefault(rec["case"], {
                "case": rec["case"], "label": rec["label"],
                "regime": rec["regime"], "cli": rec.get("cli", ""), "per": {},
            })
            cands = sorted(rec.get("candidates", []),
                           key=lambda c: (not c["passed"],
                                          c.get("median_ms") or 1e9))
            entry["per"][dev] = {
                "baseline_ms": rec["baseline_ms"],
                "compile_ms": rec.get("compile_ms"),
                "compile_admissible": rec.get("compile_admissible", True),
                "compile_envelope": rec.get("compile_envelope", {}),
                "best": best,
                "stage_cost": rec.get("stage_cost") or {},
                "profile": (profiles.get(dev) or {}).get(rec["case"]),
                "candidates": [{
                    "plan": c["plan"], "envelope": c["utilization"],
                    "passed": bool(c["passed"]),
                    "speedup": c.get("speedup"),
                    "median_ms": c.get("median_ms"),
                } for c in cands],
            }

    # How often the winning plan differs across GPUs -- the portability claim.
    #
    # Compare plan *specifications*, not plan names. A name lists its fp16 stages
    # in the order the search admitted them, so `fp16[attn,ffn1,ffn2,out_proj]`
    # and `fp16[attn,ffn2,ffn1,out_proj]` are the same plan spelled two ways.
    # Counting names said every shape diverged between the A100 and the H100;
    # counting specs says 4 of 13. The smaller number is the true one, and
    # quoting the larger one would have been a claim we could not defend.
    def plan_identity(best):
        ps = best.get("plan_spec") or {}
        if not ps:
            return best.get("plan")
        return (ps.get("attention"), ps.get("compute_dtype"),
                ps.get("residual_dtype"), bool(ps.get("cuda_graph")),
                bool(ps.get("fused_norm")), ps.get("torch_compile"),
                tuple(sorted(tuple(o) for o in (ps.get("overrides") or []))))

    # Counted over the *official* shapes only. The explorer still shows every
    # shape we measured, but the headline count has to match the one quoted in
    # docs/RESULTS.md and the tech report -- two different numbers for one claim
    # is worse than either number alone.
    official = set()
    ospath = os.path.join(ROOT, "official_shapes.txt")
    if os.path.exists(ospath):
        with open(ospath, encoding="utf-8") as f:
            official = {ln.split("#")[0].strip() for ln in f
                        if ln.split("#")[0].strip()}

    divergent = counted = 0
    for name, entry in cases.items():
        if len(entry["per"]) < 2:
            entry["divergent"] = False
            continue
        specs = {plan_identity(p["best"]) for p in entry["per"].values()}
        entry["divergent"] = len(specs) > 1
        if official and name not in official:
            continue
        counted += 1
        divergent += len(specs) > 1

    codegen = None
    cg_path = os.path.join(RESULTS, "codegen.json")
    if os.path.exists(cg_path):
        with open(cg_path, encoding="utf-8") as f:
            blob = json.load(f)
        per_target = {}
        for target, atts in blob.get("targets", {}).items():
            oks = [a for a in atts if a["status"] == "ok"]
            best = max(oks, key=lambda a: a["speedup_vs_torch"], default=None)
            per_target[target] = {
                "attempts": len(atts), "ok": len(oks),
                "best": round(best["speedup_vs_torch"], 2) if best else None,
                "envelope": round(best["envelope"], 3) if best else None,
            }
        codegen = {"taxonomy": blob.get("taxonomy", {}), "targets": per_target}

    # Present shapes in the order the problem statement lists them, which is
    # the order a judge will be looking for.
    official = os.path.join(ROOT, "official_shapes.txt")
    listed = []
    if os.path.exists(official):
        with open(official, encoding="utf-8") as fh:
            listed = [ln.split("#")[0].strip() for ln in fh if ln.split("#")[0].strip()]
    order = [c for c in listed if c in cases]
    order += [c for c in cases if c not in order]

    return {
        "devices": devices,
        "cases": [cases[c] for c in order],
        "divergent": divergent,
        "divergent_of": counted,
        "total_cases": len(cases),
        "codegen": codegen,
    }


def main() -> int:
    payload = build_payload()
    if not payload["devices"]:
        print("no results/sweep_*.json found")
        return 1
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null",
                        json.dumps(payload, separators=(",", ":")))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size:.0f} KB, self-contained)")
    print(f"  {len(payload['devices'])} GPUs, {payload['total_cases']} shapes, "
          f"{payload['divergent']} divergent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
