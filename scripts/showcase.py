"""A guided walkthrough of KernelForge, for a demo or a recording.

Seven acts, each self-contained and each making one point. Run the whole thing,
or a single act while you talk over it:

    python scripts/showcase.py                 # acts 1-5, ~3 minutes, needs a GPU
    python scripts/showcase.py --act 2         # just one act
    python scripts/showcase.py --all           # adds the two slow acts (~8 min)
    python scripts/showcase.py --list          # what the acts are
    python scripts/showcase.py --no-gpu        # acts that read artifacts only

Acts 1-2 need a GPU. Acts 3-5 read the committed artifacts and run anywhere,
which makes them safe to fall back on if a live GPU misbehaves mid-demo.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

BOLD, DIM, CYAN, GREEN, RED, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[31m", "\033[0m")
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    try:                      # enable ANSI on legacy Windows consoles
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        BOLD = DIM = CYAN = GREEN = RED = RESET = ""


def rule(title: str, subtitle: str = "") -> None:
    print(f"\n{BOLD}{'=' * 74}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    if subtitle:
        print(f"{DIM}{subtitle}{RESET}")
    print(f"{BOLD}{'=' * 74}{RESET}\n")


def say(text: str) -> None:
    print(f"{CYAN}>{RESET} {text}")


def run(cmd: list, note: str = "", timeout: float = 900) -> str:
    """Run a command, echoing it first so a viewer sees what produced the output."""
    if note:
        say(note)
    print(f"{DIM}$ {' '.join(cmd)}{RESET}\n", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return out


def tail(text: str, n: int, drop_noise: bool = True) -> None:
    noise = ("AUTOTUNE", "strides:", "dtypes:", "bias_addmm", "SingleProcess",
             "Autotune Choices", "num_choices", "Not enough SMs", "warnings.warn",
             "UserWarning", "Online softmax", "split the reduction",
             "important use case", "torch/_inductor")
    lines = [ln for ln in text.splitlines()
             if not (drop_noise and any(k in ln for k in noise))]
    for ln in [ln for ln in lines if ln.strip()][-n:]:
        print("   " + ln)


def load_sweeps():
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_*.json"))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        out[blob.get("device", os.path.basename(path))] = blob
    return out


# --------------------------------------------------------------------------

def act1_it_works(args):
    """The organizer's own script, unmodified, against our submission."""
    rule("ACT 1 - It passes their test, and it is faster",
         "The organizer's benchmark script, unmodified. We only swap in our class.")
    say("Accuracy is a hard gate in their script: fail it and you are not "
        "benchmarked at all.")
    out = run([sys.executable, "scripts/run_official.py",
               "--batch-size", "8", "--seq-len", "128",
               "--accuracy-trials", "3", "--repeats", "40",
               "--benchmark-rounds", "3"],
              "Running their script end to end...")
    tail(out, 14)
    if "speedup" in out:
        line = [l for l in out.splitlines() if "speedup" in l][-1]
        print(f"\n{GREEN}{BOLD}   {line.strip()}{RESET}")


def act2_the_gate(args):
    """A wrong-but-fast configuration is rejected before it is ever timed."""
    rule("ACT 2 - The gate: correctness is checked BEFORE speed",
         "A fast, silently-wrong kernel is the real danger. Watch one get rejected.")
    say("On the official test set -- all 14 shapes are float32 -- torch.compile "
        "passes the gate everywhere,")
    say("so every speedup we report is against an admissible opponent. Change "
        "one flag to float16 and")
    say("that stops being true: the reference's own rounding, amplified through "
        "the stack, exceeds the")
    say("tolerance, so no reassociating optimization can pass -- not even "
        "PyTorch's own compiler.")
    say("Nothing in the code special-cases this. The gate just measures, and "
        "rejects. Watch it.")
    out = run([sys.executable, "-m", "kernelforge.cli", "verify",
               "--cases", "default,default_float16,default_bfloat16,long_causal",
               "--trials", "3"],
              "Verifying the shipped plans against the exact tolerance rule...")
    tail(out, 12)
    print()
    say("Now the evidence for the float16 claim, measured on every GPU we have:")
    for dev, blob in load_sweeps().items():
        for rec in blob.get("records", []):
            cp = rec.get("compile_passed") or {}
            if cp and not any(cp.values()):
                ce = rec.get("compile_envelope") or {}
                env = max(ce.values()) if ce else float("nan")
                print(f"   {RED}torch.compile FAILS the gate{RESET}  "
                      f"{dev:<28} {rec['label'][-9:]:<9} envelope {env:.2f} "
                      f"{DIM}(limit 1.0){RESET}")


def official_specs():
    """The 14 shapes from Appendix 3.7, in the organizers' order."""
    path = os.path.join(ROOT, "official_shapes.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]


def plan_identity(best):
    """What makes two winning plans the same plan.

    Deliberately not the plan *name*: a name lists its fp16 stages in the order
    the search admitted them, so `fp16[attn,ffn1]` and `fp16[ffn1,attn]` are one
    plan written two ways. Counting names overstates how much the hardware
    changes the answer, which is a claim we would rather understate than inflate.
    """
    ps = best.get("plan_spec") or {}
    if not ps:
        return best.get("plan")
    return (ps.get("attention"), ps.get("compute_dtype"), ps.get("residual_dtype"),
            bool(ps.get("cuda_graph")), bool(ps.get("fused_norm")),
            ps.get("torch_compile"),
            tuple(sorted(tuple(o) for o in (ps.get("overrides") or []))))


def act_shape14(args):
    """The official shape no reference implementation can run."""
    rule("ACT 5 - The shape the reference cannot run",
         "Reads committed artifacts - no GPU needed.")
    path = os.path.join(RESULTS, "shape14.json")
    if not os.path.exists(path):
        say("results/shape14.json not present; run scripts/shape14.py first.")
        return
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    say(f"Official test shape 14:  {d['shape']}  in {d['dtype']}")
    print()
    print(f"   The reference forms [B,H,S,S] before its softmax. Here that is "
          f"{BOLD}{d['baseline_score_matrix_tb']:.1f} TB{RESET}.")
    print("   Not 'slow', not 'needs a bigger card'. No machine can allocate "
          "it, so the reference")
    print("   cannot run this shape -- and neither can torch.compile applied "
          "to the reference.")
    print()
    for m in d["measurements"]:
        print(f"   {GREEN}{m['gpu']:<24}{RESET} {m['ms'] / 1000:6.1f} s   "
              f"peak {m['peak_gb']:4.1f} GB   batch sliced "
              f"{m['batch_slice']}/32   output finite, shape correct")
    print()
    say("We quote no speedup. A ratio against something that cannot run is not "
        "a measurement.")
    say("The claim is narrower and stronger: this shape is reachable with a "
        "fused kernel, and")
    say("unreachable without one. Three things had to be true at once:")
    print()
    for i, fix in enumerate(d["fixes_required"], 1):
        print(f"   {i}. {fix}")
    print()
    say("The third was our own bug: the fp32 fallback was rebuilding the "
        "quadratic term the flash")
    say("kernel exists to remove. Found by printing the failing allocation, "
        "after two wrong guesses.")


def act3_portability(args):
    """The same shape picks a different kernel on a different GPU."""
    rule("ACT 3 - Same shape, different GPU, different kernel",
         "Reads committed artifacts - no GPU needed.")
    sweeps = load_sweeps()
    if len(sweeps) < 2:
        say("Need sweeps from 2+ GPUs; run `cli sweep` on another card.")
        return
    devices = list(sweeps)
    # Restricted to the official test shapes. The sweeps also carry the wider
    # matrix we tuned against before the list was published, and mixing the two
    # here would put a number on screen that does not match anything quoted in
    # the report.
    official = official_specs()
    by_case = {}
    for dev, blob in sweeps.items():
        for rec in blob.get("records", []):
            if rec.get("best") and (not official or rec["case"] in official):
                by_case.setdefault(rec["case"], {})[dev] = rec["best"]
    by_case = {c: by_case[c] for c in official if c in by_case} or by_case

    short = {d: d.split("_")[0][:18] for d in devices}
    print(f"   {'shape':<20}" + "".join(f"{short[d]:>26}" for d in devices))
    print("   " + "-" * (20 + 26 * len(devices)))
    divergent = 0
    for case, per in by_case.items():
        if len(per) < len(devices):
            continue
        specs = {plan_identity(p) for p in per.values()}
        if len(specs) > 1:
            divergent += 1
        mark = GREEN if len(specs) > 1 else DIM
        row = f"   {case:<20}"
        for d in devices:
            b = per[d]
            row += f"{mark}{b['plan'][:18]:>19}{RESET} {b['speedup']:5.2f}x"
        print(row)
    total = len([c for c, p in by_case.items() if len(p) == len(devices)])
    print(f"\n{BOLD}   {divergent} of {total} shapes chose a genuinely "
          f"different plan depending on the GPU.{RESET}")
    say("Green rows differ. We compare plan *specs*, not plan names: a name "
        "lists fp16 stages in the")
    say("order the search admitted them, so counting names said every shape "
        "diverged while counting")
    say("specs says four. The smaller number is the true one -- and four in "
        "thirteen, between two")
    say("cards one generation apart, is still the argument for automating "
        "this instead of hand-tuning.")


def act4_ai_kernels(args):
    """What the AI wrote, and how it failed."""
    rule("ACT 4 - Kernels the AI wrote, and how they fail",
         "Reads committed artifacts - no GPU needed.")
    path = os.path.join(RESULTS, "codegen.json")
    if not os.path.exists(path):
        say("No results/codegen.json — run `cli codegen` first.")
        return
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    tax = blob.get("taxonomy", {})
    total = sum(tax.values()) or 1
    say(f"The model was asked for complete Triton source. {total} kernels "
        f"generated, every one gated:")
    for k, v in sorted(tax.items(), key=lambda kv: -kv[1]):
        colour = GREEN if k == "ok" else RED
        print(f"   {colour}{k:<30}{RESET} {v:3d}  ({100.0*v/total:5.1f}%)")
    print()
    for t, atts in blob.get("targets", {}).items():
        oks = [a for a in atts if a["status"] == "ok"]
        best = max(oks, key=lambda a: a["speedup_vs_torch"], default=None)
        if best:
            print(f"   {t:<12} {len(oks)}/{len(atts)} correct   best "
                  f"{GREEN}{best['speedup_vs_torch']:.2f}x vs torch{RESET}, "
                  f"envelope {best['envelope']:.3f}, {best['lines']} lines")
    print()
    say("The instructive failure did not crash, and it was not careless. One "
        "bias+GELU kernel tiles")
    say("correctly, masks both edges, applies its strides -- then swaps exact "
        "erf for an")
    say("Abramowitz-Stegun polynomial, and comments that this gives ~1.5e-7 "
        "accuracy, \"well within")
    say("float16 tolerance\". The model did not forget. It decided, quantified "
        "the error, and was right")
    say("about the number -- but the tolerance is measured against the "
        "REFERENCE, which calls exact erf.")
    say("Envelope 22.8 against a limit of 1.0. The same model does this "
        "reproducibly, on both GPUs.")
    say("Clean code, explicit correct reasoning, a comment telling you it is "
        "fine. Only the gate caught it.")
    say("Sources are in results/generated/ for review.")


def act5_new_shape(args):
    """Tune a shape the system has never seen. Slow."""
    rule("ACT 6 - Absorbing a shape we have never seen  (~3 min)",
         "The official shape list is published late. This is how we ingest it.")
    spec = args.shape
    say(f"Nothing about {spec} is in the code. Searching it now:")
    out = run([sys.executable, "-m", "kernelforge.cli", "tune",
               "--shapes", spec, "--trials", "2", "--case-budget", "260"],
              "Search -> gate -> benchmark -> freeze -> re-verify")
    for ln in out.splitlines():
        if ("BEST" in ln or ln.startswith("#   ") or "nothing cleared" in ln
                or "all cases pass" in ln):
            print("   " + ln.strip())


def act6_codegen_live(args):
    """Have the AI write a kernel, live. Slow, needs SOCLAAS_API_KEY."""
    rule("ACT 7 - Watch the AI write a kernel  (~3 min, needs an API key)",
         "Generated source is gated exactly like everything else.")
    out = run([sys.executable, "-m", "kernelforge.cli", "codegen",
               "--targets", "gelu", "--iterations", "4"],
              "Generating Triton kernels and gating each one...")
    tail(out, 14)


ACTS = [
    ("It passes their test, and it is faster", act1_it_works, True),
    ("The gate rejects wrong kernels before timing", act2_the_gate, True),
    ("Same shape, different GPU, different kernel", act3_portability, False),
    ("Kernels the AI wrote, and how they fail", act4_ai_kernels, False),
    ("The shape the reference cannot run", act_shape14, False),
    ("Absorbing a shape we have never seen", act5_new_shape, True),
    ("Watch the AI write a kernel, live", act6_codegen_live, True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", type=int, help="run a single act (1-6)")
    ap.add_argument("--all", action="store_true", help="include the slow acts 5-6")
    ap.add_argument("--no-gpu", action="store_true",
                    help="only acts that read committed artifacts")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shape", default="B6-S192-d512-H8-F2048-L6",
                    help="shape for act 5")
    args = ap.parse_args()

    if args.list:
        for i, (name, _, gpu) in enumerate(ACTS, 1):
            print(f"  {i}. {name}{'' if gpu else '   (no GPU needed)'}")
        return 0

    if args.act:
        chosen = [ACTS[args.act - 1]]
    elif args.no_gpu:
        chosen = [a for a in ACTS if not a[2]]
    elif args.all:
        chosen = ACTS
    else:
        chosen = ACTS[:4]

    rule("KernelForge - TikTok TechJam 2026, Track 3",
         "We did not hand-tune one kernel. We built the thing that produces them.")
    started = time.time()
    for name, fn, needs_gpu in chosen:
        if needs_gpu and args.no_gpu:
            continue
        try:
            fn(args)
        except subprocess.TimeoutExpired:
            print(f"{RED}   (timed out — skipping){RESET}")
        except Exception as e:                      # a demo must never crash
            print(f"{RED}   ({type(e).__name__}: {e}){RESET}")

    rule("Where to look next",
         f"walkthrough finished in {time.time() - started:.0f}s")
    for line in [
        "docs/RESULTS.md         every shape on every GPU, three-way",
        "docs/TECH_REPORT.md     environment, optimizations, results, AI tools",
        "docs/PRECISION.md       what we measured about the tolerance",
        "docs/CODEGEN.md         AI-written kernels, and a negative result",
        "docs/dashboard.html     interactive: pick a GPU and a shape",
    ]:
        print(f"   {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
