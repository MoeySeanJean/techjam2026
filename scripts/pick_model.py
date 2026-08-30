"""Score the models an LLM key can reach on proposal quality.

The score here measures whether a model can emit a valid *plan* as JSON. On our
gateway that was ANTI-predictive of whether the same model could write a correct
Triton kernel, so use it to eliminate models that ignore explicit constraints,
not to pick a winner -- `docs/CODEGEN.md` has the comparison that should decide.

Picking a model by reputation is guesswork. What this loop actually needs is
narrow and testable, so we measure it directly on every model the key can reach:

  1. **Parse rate** -- does it return ONE JSON object matching the plan schema?
     A reply we cannot parse costs a whole iteration.
  2. **Validity** -- are the fields legal (known dtypes, a `flash_block` that
     fits this GPU's shared memory)? An illegal tile is the classic failure of
     a model trained on A100 kernels.
  3. **Constraint compliance** -- at float16/bfloat16 I/O our error budget says
     only the bit-exact plan can pass. Does the model respect what the prompt
     tells it, or does it cheerfully propose fp16 GEMMs anyway?
  4. **Novelty** -- does it avoid repeating configurations already in the
     history? Repeats waste iterations without exploring anything.
  5. **Latency** -- the loop is GPU-bound, but a slow model still adds up over
     hundreds of proposals.

No GPU is required: this scores the *proposer*, not the kernels. Correctness of
any accepted plan is still decided later by `kernelforge.numerics`.

    python scripts/pick_model.py            # score every visible model
    python scripts/pick_model.py --rounds 5 # more samples per model
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelforge import secrets  # noqa: E402
from kernelforge.agent.proposers import SYSTEM_PROMPT, parse_plan  # noqa: E402
from kernelforge.ops.flash import legal_blocks, smem_bytes  # noqa: E402

DTYPES = {"auto", "float16", "bfloat16", "float32"}
ATTENTION = {"flash", "sdpa", "exact"}

# A realistic prompt: this is the shape of thing the agent actually sends. The
# card described here is deliberately a *constrained* one -- a model that ignores
# a tight shared-memory budget is exactly the failure this benchmark should
# catch, and a generous spec sheet would let that pass unnoticed.
SPEC_BLOCK = """GPU: NVIDIA A100 80GB PCIe
Compute capability: sm_80 (major=8, minor=0)
SM count: 108
Shared memory per block (opt-in): 163 KB   <-- HARD LIMIT for BLOCK_M*BLOCK_N*num_stages tiling
Registers per block: 65536
L2 cache: 40 MB
Device memory: 79.2 GB
Achieved DRAM bandwidth (measured): ~1651 GB/s
BF16 supported: True
TMA (Hopper async copy) available: False
FP8 available: False
NOTE: pre-Hopper. Do NOT emit TMA descriptors, wgmma, or FP8 paths."""

PROFILE_BLOCK = """Shape: B8-S128-d512-H8-F2048-L6-float32
Bottleneck regime (measured): gemm-bound
GPU time per forward: 5.389 ms
CPU time per forward: 2.412 ms
Kernel launches per forward: 105 (~13.1 us CPU each)
Time by category:
  gemm            71.2%
  elementwise     20.1%
  copy             4.0%
  softmax          2.2%"""

HISTORY = """- wide: compute=auto residual=float32 attn=flash fused-qkv fused-norm -> ok, envelope=0.627, latency=5.5793ms, speedup=1.623x
- fp16[attn]: compute=auto residual=float32 attn=flash fused-qkv fused-norm [attn=float16] -> ok, envelope=0.592, latency=5.0586ms, speedup=1.790x
- fp16[attn,out_proj]: -> ok, envelope=0.707, latency=4.9239ms, speedup=1.839x
- fp16[attn,out_proj,ffn2]: -> numeric_fail, envelope=0.866, latency=nanms, speedup=nanx"""

STAGE_COST = """  attn: -0.030
  out_proj: +0.031
  ffn2: +0.221
  ffn1: +0.326"""


def build_prompt(io_dtype: str, legal) -> str:
    constraint = ""
    if io_dtype in ("float16", "bfloat16"):
        constraint = (
            "\n\nIMPORTANT: at this I/O dtype the measured error budget shows "
            "that NO reassociating optimization can pass the gate -- even "
            "torch.compile fails it. Only a bit-exact plan is admissible: "
            "attention must be \"exact\", fuse_qkv false, fused_norm false, "
            "compute_dtype \"auto\", residual_dtype \"auto\". You may still set "
            "cuda_graph true.")
    return "\n\n".join([
        "## Hardware", SPEC_BLOCK,
        "## Measured profile", PROFILE_BLOCK,
        "## Workload",
        f"batch=8 seq_len=128 d_model=512 heads=8 head_dim=64 ffn=2048 "
        f"layers=6 causal=False padding_ratio=0.0 io_dtype={io_dtype}",
        "## Measured per-stage error cost", STAGE_COST,
        f"## Legal flash_block configurations on this GPU\n{legal}",
        "## Correctness margin\nenvelope must be <= 0.8",
        "## History", HISTORY,
        "Propose the next configuration." + constraint,
    ])


def call(base_url: str, api_key: str, model: str, prompt: str,
         timeout: float) -> Dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 900,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return {"text": data["choices"][0]["message"]["content"] or "",
            "usage": data.get("usage", {}),
            "seconds": time.time() - started}


def score_plan(plan, io_dtype: str, legal, tried) -> List[str]:
    """Return a list of problems with this plan. Empty means clean."""
    problems: List[str] = []
    if plan.compute_dtype not in DTYPES or plan.residual_dtype not in DTYPES:
        problems.append("bad-dtype")
    if plan.attention not in ATTENTION:
        problems.append("bad-attention")
    if plan.flash_block is not None:
        blk = tuple(plan.flash_block)
        if len(blk) != 4:
            problems.append("bad-block-shape")
        elif blk not in {tuple(b) for b in legal}:
            bm, bn, _, stages = blk
            need = smem_bytes(bm, bn, 64, stages) / 1024
            problems.append(f"illegal-tile({need:.0f}KB>99KB)"
                            if need > 99 else "illegal-tile")
    for stage, dt in plan.overrides:
        if stage not in ("attn", "out_proj", "ffn1", "ffn2"):
            problems.append("bad-stage")
        if dt not in DTYPES:
            problems.append("bad-override-dtype")
    if plan.name in tried:
        problems.append("repeat")
    if io_dtype in ("float16", "bfloat16"):
        # The prompt states plainly that only a bit-exact plan is admissible.
        bit_exact = (plan.attention == "exact" and not plan.fuse_qkv
                     and not plan.fused_norm
                     and plan.compute_dtype == "auto"
                     and plan.residual_dtype == "auto"
                     and not plan.overrides)
        if not bit_exact:
            problems.append("ignored-dtype-constraint")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3,
                    help="prompts per model per dtype")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--models", default=None, help="comma separated subset")
    args = ap.parse_args()

    secrets.load()
    base_url = (os.environ.get("SOCLAAS_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("SOCLAAS_API_KEY")
    if not base_url or not api_key:
        print("set SOCLAAS_BASE_URL and SOCLAAS_API_KEY in .env")
        return 1

    if args.models:
        models = args.models.split(",")
    else:
        req = urllib.request.Request(
            base_url + "/models",
            headers={"Authorization": "Bearer " + api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            catalog = json.load(resp)
        models = [m["id"] for m in catalog.get("data", [])]

    # Embedding / vision / probe models cannot do this job; skip them.
    skip = ("bge", "embed", "vision", "-vl", "test")
    models = [m for m in models if not any(s in m.lower() for s in skip)]

    legal = legal_blocks(128, 64, 99.0)
    tried = {"wide", "fp16[attn]", "fp16[attn,out_proj]", "fp16[attn,out_proj,ffn2]"}

    rows = []
    for model in models:
        calls = parsed = clean = 0
        problems: Dict[str, int] = {}
        lat: List[float] = []
        toks = 0
        errors: Dict[str, int] = {}
        for io_dtype in ("float32", "float16"):
            prompt = build_prompt(io_dtype, legal)
            for _ in range(args.rounds):
                calls += 1
                try:
                    out = call(base_url, api_key, model, prompt, args.timeout)
                except urllib.error.HTTPError as e:
                    errors[f"http{e.code}"] = errors.get(f"http{e.code}", 0) + 1
                    continue
                except Exception as e:
                    key = type(e).__name__
                    errors[key] = errors.get(key, 0) + 1
                    continue
                lat.append(out["seconds"])
                toks += out["usage"].get("total_tokens", 0) or 0
                plan = parse_plan(out["text"], 99.0)
                if plan is None:
                    problems["unparseable"] = problems.get("unparseable", 0) + 1
                    continue
                parsed += 1
                bad = score_plan(plan, io_dtype, legal, tried)
                for b in bad:
                    problems[b] = problems.get(b, 0) + 1
                if not bad:
                    clean += 1
        rows.append({
            "model": model, "calls": calls, "parsed": parsed, "clean": clean,
            "parse_rate": parsed / calls if calls else 0.0,
            "clean_rate": clean / calls if calls else 0.0,
            "median_s": statistics.median(lat) if lat else float("nan"),
            "tokens": toks, "problems": problems, "errors": errors,
        })
        r = rows[-1]
        print(f"  {model:<22} parse {r['parse_rate']:5.0%}  usable "
              f"{r['clean_rate']:5.0%}  {r['median_s']:5.2f}s  "
              f"{dict(sorted(problems.items())) or ''}"
              f"{' ERR ' + str(errors) if errors else ''}", flush=True)

    rows.sort(key=lambda r: (-r["clean_rate"], -r["parse_rate"], r["median_s"]))
    print("\n" + "=" * 78)
    print(f"{'model':<22} {'parse':>6} {'usable':>7} {'median':>8} {'tokens':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<22} {r['parse_rate']:6.0%} {r['clean_rate']:7.0%} "
              f"{r['median_s']:7.2f}s {r['tokens']:8d}")
    if rows:
        best = rows[0]
        print(f"\nBest on this benchmark: {best['model']} "
              f"(usable {best['clean_rate']:.0%}, median {best['median_s']:.2f}s)")
        # Printing the caveat next to the number is the point. On our gateway six
        # of ten models tied at 100% usable, so the ranking fell to latency --
        # and the model it put first went on to write 3 correct Triton kernels
        # out of 40, while the model it ranked last on format wrote 48 of 60. A
        # reader who took this as a recommendation would repeat that mistake.
        print("\n  NOTE: this scores plan-JSON quality only, and on our gateway"
              "\n  it was ANTI-predictive of kernel-writing ability. Use it to"
              "\n  eliminate models that ignore explicit constraints, not to"
              "\n  pick a winner. To choose a model, run the real task --"
              "\n      python -m kernelforge.cli codegen --targets layernorm,gelu"
              "\n  once per candidate, and compare correct-kernel counts."
              "\n  docs/CODEGEN.md has the measured comparison.")

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "model_selection.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rounds": args.rounds, "rows": rows}, f, indent=2)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
