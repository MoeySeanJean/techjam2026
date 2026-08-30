"""AI-generated kernel *source*, not just AI-selected configuration.

The track puts "AI-based code generation" in scope and asks how AI can "generate
more efficient implementations for specific GPU hardware". Selecting a
configuration from a fixed plan space does not answer that question -- it is
hyperparameter search with a language model attached. This module closes the
gap: the LLM writes complete Triton source, and the harness decides whether it
is worth anything.

Nothing about that is safe on its own. Generated code fails constantly and in
ways that matter: it does not compile, it requests more shared memory than the
GPU has, it indexes out of bounds, or -- worst -- it runs fine and returns
subtly wrong numbers. The existing gate already handles the last case, and this
module adds the rest, classifying every failure so the report can say *how* AI
fails at kernel engineering rather than merely that it does.

Trust model
-----------
Generated source is `exec`'d. That is not sandboxed and this module does not
pretend otherwise. Three properties keep it defensible:

  1. It runs at **build time only**, on the developer's own machine, never in
     the submitted inference path.
  2. Every candidate is written to `results/generated/` before execution, so
     there is a reviewable artifact of exactly what ran.
  3. A generated kernel is only ever *proposed*. Promoting one into the shipped
     dispatch table is a deliberate, human step (`--promote`), because a public
     submission should not contain code no person has read.

Run it with `python -m kernelforge.cli codegen`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..numerics import check

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "results")
GENERATED = os.path.join(RESULTS, "generated")


# --------------------------------------------------------------------------
# What we ask for


@dataclasses.dataclass
class KernelSpec:
    """A precise, testable contract for a kernel the LLM must write."""

    name: str
    entry_point: str                 # python function the source must define
    contract: str                    # prose + signature handed to the model
    make_inputs: Callable            # () -> dict of kwargs
    reference: Callable              # (**kwargs) -> reference output tensor(s)
    flops: Optional[Callable] = None

    def prompt(self, spec_block: str, history: str) -> str:
        return "\n\n".join([
            "## Target GPU", spec_block,
            "## What to write", self.contract,
            "## Previous attempts on this target", history or "(none yet)",
            "Write the complete Python module now.",
        ])


SYSTEM_PROMPT = """You write GPU kernels in Triton for an inference workload.

Output a COMPLETE, RUNNABLE Python module and NOTHING else -- no prose, no
explanation outside the code. Wrap it in one ```python fence.

The module may import only: torch, triton, triton.language as tl, math.
It must define exactly the entry-point function named in the contract, with the
exact signature given. Do not read files, use the network, or call exit().

Hard rules, each of which has failed real generated kernels before:
- Triton kernels can only read module-level globals declared `tl.constexpr`.
  A plain global float (e.g. NEG_INF = float("-inf")) raises NameError at trace
  time. Declare them as `tl.constexpr` or inline the literal.
- Shared memory per block is a HARD limit on this GPU; it is in the spec sheet
  above. A tiling that exceeds it will not launch. Configurations copied from
  A100-targeted code routinely overflow smaller cards.
- Every `tl.load` and `tl.store` on a dimension that is not a multiple of the
  block size needs a mask; out-of-bounds access is a hard crash, not a warning.
- Reductions that must be numerically stable (mean, variance, softmax) should
  accumulate in float32 even when the data is float16.
- Match the reference's arithmetic, including where it rounds. Being *more*
  precise than the reference can still fail an exact-tolerance comparison."""


# --------------------------------------------------------------------------
# Targets


def _layernorm_inputs(device="cuda", M=517, d=512, dtype=torch.float16):
    torch.manual_seed(0)
    return {
        "x": torch.randn(M, d, device=device, dtype=dtype),
        "y": torch.randn(M, d, device=device, dtype=dtype),
        "keep": (torch.rand(M, device=device) > 0.3).to(dtype),
        "weight": torch.randn(d, device=device, dtype=dtype),
        "bias": torch.randn(d, device=device, dtype=dtype),
        "eps": 1e-5,
    }


def _layernorm_reference(x, y, keep, weight, bias, eps):
    s = (x.float() + y.float()) * keep[:, None].float()
    h = F.layer_norm(s, (s.shape[-1],), weight.float(), bias.float(), eps)
    return s, h.to(x.dtype)


LAYERNORM = KernelSpec(
    name="add_mask_layernorm",
    entry_point="fused_add_mask_layernorm",
    contract=textwrap.dedent("""\
        A single fused kernel for one Transformer block boundary.

            def fused_add_mask_layernorm(x, y, keep, weight, bias, eps):
                '''
                x      : [M, d] residual stream          (float16 or float32)
                y      : [M, d] sublayer output          (same dtype as x)
                keep   : [M]    per-row 0/1 mask         (same dtype as x)
                weight : [d]    LayerNorm gamma
                bias   : [d]    LayerNorm beta
                eps    : python float

                Returns (s, h):
                  s = (x + y) * keep[:, None]          accumulated in float32,
                                                       returned as float32 [M, d]
                  h = LayerNorm(s, weight, bias, eps)  returned in x.dtype [M, d]

                LayerNorm here is exactly nn.LayerNorm: biased variance (divide
                by d, not d-1) and eps INSIDE the sqrt.
                '''

        It must be one Triton kernel doing the whole thing -- one pass over x
        and y, not a sequence of torch ops. d is a runtime value and is not
        necessarily a power of two; M is not necessarily a multiple of anything.
        Handle both with masking. Speed matters: this kernel is memory-bound,
        so minimise the number of passes over memory.

        CRITICAL -- do not split the d axis. mean and variance are per-row
        statistics over ALL d elements, so a single program must reduce a whole
        row. Assign one program per row (grid = (M,)) with BLOCK_D =
        next_power_of_2(d) and mask the tail. If you tile d across programs and
        call tl.sum over the tile, you compute per-tile statistics instead of
        per-row ones: the kernel will compile, run, and return wrong numbers."""),
    make_inputs=_layernorm_inputs,
    reference=_layernorm_reference,
)


def _gelu_inputs(device="cuda", M=4096, n=2048, dtype=torch.float16):
    torch.manual_seed(0)
    return {"x": torch.randn(M, n, device=device, dtype=dtype),
            "bias": torch.randn(n, device=device, dtype=dtype)}


def _gelu_reference(x, bias):
    return F.gelu(x.float() + bias.float(), approximate="none").to(x.dtype)


GELU_BIAS = KernelSpec(
    name="bias_gelu",
    entry_point="fused_bias_gelu",
    contract=textwrap.dedent("""\
        The FFN activation, fused with its bias add.

            def fused_bias_gelu(x, bias):
                '''
                x    : [M, n] pre-activation  (float16 or float32)
                bias : [n]    broadcast bias
                returns [M, n] in x.dtype
                '''

        The activation is the EXACT erf GELU that
        `torch.nn.functional.gelu(..., approximate="none")` computes:

            gelu(v) = 0.5 * v * (1 + erf(v / sqrt(2)))

        Do NOT substitute the tanh approximation -- it differs by enough to fail
        an exact-tolerance comparison. Accumulate in float32. This kernel is
        purely memory-bound, so it should read x once and write once."""),
    make_inputs=_gelu_inputs,
    reference=_gelu_reference,
)

TARGETS: Dict[str, KernelSpec] = {
    "layernorm": LAYERNORM,
    "gelu": GELU_BIAS,
}



REPAIR_PROMPT = """The kernel you wrote does not work. Fix it.

Below is the exact source you produced and exactly what went wrong. Return the
COMPLETE corrected module in one ```python fence -- not a diff, not a
description, the whole file. Keep the same entry-point name and signature.

Think about the specific failure before rewriting. If the diagnosis says the
error is constant within each row, the row statistics are being computed over
the wrong extent. If it says a compile error, read the line it points at. Do not
change things unrelated to the failure."""


def repair_prompt(spec: KernelSpec, source: str, attempt: "Attempt") -> str:
    parts = [
        "## The kernel you wrote",
        "```python\n" + source + "\n```",
        "## What went wrong",
        f"Failure category: {attempt.status}",
        f"Error: {attempt.detail}",
    ]
    if attempt.diagnosis:
        parts.append("Diagnosis of the numerical error:\n" + attempt.diagnosis)
    parts += ["## The contract it must satisfy", spec.contract,
              "Return the corrected module."]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Validation


@dataclasses.dataclass
class Attempt:
    iteration: int
    target: str
    status: str                       # ok | <failure category>
    detail: str = ""
    source_path: str = ""
    envelope: float = float("nan")
    median_ms: float = float("nan")
    speedup_vs_torch: float = float("nan")
    lines: int = 0
    diagnosis: str = ""
    repair_of: int = 0        # iteration this attempt tried to fix, 0 = fresh
    status_note: str = ""

    def as_history(self) -> str:
        base = (f"- attempt {self.iteration}"
                + (f" (repair of #{self.repair_of})" if self.repair_of else "")
                + f": {self.status}")
        if self.status == "ok":
            return (base + f", envelope={self.envelope:.3f}, "
                    f"{self.median_ms:.4f} ms, {self.speedup_vs_torch:.2f}x vs torch")
        return base + f" -- {self.detail[:200]}"


def extract_code(text: str) -> Optional[str]:
    """Pull the python module out of a model reply."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if fenced:
        return max(fenced, key=len).strip()
    if "import" in text and "def " in text:
        return text.strip()
    return None


def error_signature(attempt: "Attempt") -> str:
    """A stable fingerprint of *what* went wrong, ignoring line numbers.

    Used to detect a repair loop that is not converging: if a fix reproduces the
    same failure signature, the model has not understood the error and further
    repairs on that lineage are wasted budget.
    """
    text = re.sub(r"\d+", "#", attempt.detail or "")
    return f"{attempt.status}|{text[:120]}"


def classify(exc: BaseException) -> str:
    """Failure taxonomy. These labels are the report's contribution on *how*
    AI-generated kernels fail, so they are derived from real observed errors."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of resource" in text or "shared memory" in text or "smem" in text:
        return "shared_memory_overflow"
    if "nameerror" in text and "constexpr" in text:
        return "triton_global_not_constexpr"
    if "compilationerror" in text or "triton" in text and "compile" in text:
        return "compile_error"
    if isinstance(exc, SyntaxError) or "syntaxerror" in text:
        return "syntax_error"
    if "illegal memory access" in text or "misaligned" in text:
        return "illegal_memory_access"
    if "out of memory" in text:
        return "device_oom"
    if "not defined" in text or "has no attribute" in text:
        return "bad_api_usage"
    if "shape" in text or "size mismatch" in text or "dimension" in text:
        return "shape_error"
    return "runtime_error"



def diagnose(want, got) -> str:
    """Describe *how* a wrong kernel is wrong, in terms a fix can act on.

    "envelope 4525" tells the model nothing it can use. The shape of the error
    does: an error confined to the tail of each row is a masking bug; an error
    that grows along the row is an indexing bug; a per-row-constant error is a
    reduction computed over the wrong axis -- which is exactly the failure this
    loop hit most often.
    """
    import torch as _t
    if want is None or got is None or want.shape != got.shape:
        return "output shape did not match the reference"
    err = (got.detach().float() - want.detach().float()).abs()
    notes = []

    if not _t.isfinite(got).all():
        n = int((~_t.isfinite(got)).sum())
        notes.append(f"{n} non-finite values (NaN/Inf) in the output")

    if err.dim() == 2:
        rows, cols = err.shape
        per_row = err.amax(dim=1)
        bad_rows = int((per_row > 0).sum())
        notes.append(f"{bad_rows}/{rows} rows differ")

        # Where along the row does the error live?
        half = cols // 2
        first, second = err[:, :half].mean().item(), err[:, half:].mean().item()
        if second > 4 * max(first, 1e-12):
            notes.append("error is concentrated in the SECOND half of each row "
                         "-- suspect a missing mask on the column tail, or a "
                         "block size that does not cover d")
        elif first > 4 * max(second, 1e-12):
            notes.append("error is concentrated at the START of each row")

        # A per-row-constant error means the row statistics are wrong.
        row_spread = (err.amax(dim=1) - err.amin(dim=1)).mean().item()
        row_level = err.mean().item()
        if row_level > 0 and row_spread < 0.25 * row_level:
            notes.append("the error is nearly CONSTANT within each row, which "
                         "means the per-row statistics (mean/variance) are "
                         "wrong -- most likely reduced over a tile of the "
                         "feature axis instead of the whole row")

        if rows > 1:
            head = err[0].mean().item()
            tail = err[-1].mean().item()
            if tail > 4 * max(head, 1e-12):
                notes.append("later rows are much worse than the first row -- "
                             "suspect a row-stride or program-id indexing bug")
    return "; ".join(notes) or "values differ with no obvious structure"


def _as_tuple(v):
    return v if isinstance(v, (tuple, list)) else (v,)


def load_module(path: str):
    """Import a generated module from the file we already wrote.

    Triton refuses to trace a `@triton.jit` function that did not come from a
    real file -- it reads the source back off disk to build the AST -- so an
    `exec` of the string fails with "@jit functions should be defined in a
    Python file". Importing the artifact we saved is both what Triton needs and
    the more auditable choice: the exact bytes that ran are on disk.
    """
    import importlib.util
    name = "kf_generated_" + os.path.splitext(os.path.basename(path))[0]
    spec_ = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec_)
    import sys as _sys
    _sys.modules[name] = module           # Triton resolves globals via sys.modules
    spec_.loader.exec_module(module)
    return module


def validate_inprocess(path: str, spec: KernelSpec, device: torch.device,
                       trials: int = 3) -> Attempt:
    """Import, run and gate one generated module in THIS process.

    Only safe inside the throwaway worker: an out-of-bounds generated kernel
    corrupts the CUDA context, and no amount of exception handling here can
    recover it. `validate()` is the isolated entry point callers should use.
    """
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    attempt = Attempt(0, spec.name, "ok", lines=len(source.splitlines()))

    try:
        module = load_module(path)
    except BaseException as e:            # noqa: BLE001 - classify everything
        attempt.status = classify(e)
        attempt.detail = f"{type(e).__name__}: {e}"
        return attempt

    fn = getattr(module, spec.entry_point, None)
    if not callable(fn):
        attempt.status = "missing_entry_point"
        attempt.detail = f"module does not define {spec.entry_point}()"
        return attempt

    worst = None
    worst_pair = (None, None)
    for t in range(trials):
        torch.manual_seed(1234 + t)
        kwargs = spec.make_inputs()
        try:
            with torch.inference_mode():
                got = _as_tuple(fn(**{k: v for k, v in kwargs.items()}))
                want = _as_tuple(spec.reference(**kwargs))
        except BaseException as e:        # noqa: BLE001
            attempt.status = classify(e)
            attempt.detail = f"{type(e).__name__}: {e}"
            return attempt

        if len(got) != len(want):
            attempt.status = "wrong_return_arity"
            attempt.detail = f"returned {len(got)} tensors, expected {len(want)}"
            return attempt

        for g, w in zip(got, want):
            if not isinstance(g, torch.Tensor):
                attempt.status = "wrong_return_type"
                attempt.detail = f"returned {type(g).__name__}, expected Tensor"
                return attempt
            try:
                # An illegal access inside the generated kernel is asynchronous:
                # the launch returns cleanly and the error lands here, at the
                # next CUDA call. Compare inside the guard for that reason.
                res = check(w, g)
            except BaseException as e:    # noqa: BLE001
                attempt.status = classify(e)
                attempt.detail = f"{type(e).__name__}: {e}"
                return attempt
            if worst is None or res.envelope_utilization > worst.envelope_utilization:
                worst = res
                worst_pair = (w, g)

    attempt.envelope = worst.envelope_utilization
    if not worst.passed:
        attempt.status = "numeric_fail"
        attempt.detail = str(worst)
        attempt.diagnosis = diagnose(*worst_pair)
        return attempt

    # It is correct. Now: is it actually faster than the torch it replaces?
    from .. import bench
    kwargs = spec.make_inputs()
    try:
        timings = bench.compare(
            {"torch": lambda: spec.reference(**kwargs),
             "generated": lambda: fn(**kwargs)},
            warmup=15, repeats=30, rounds=3)
        attempt.median_ms = timings["generated"].median_ms
        attempt.speedup_vs_torch = (timings["torch"].median_ms /
                                    timings["generated"].median_ms)
    except BaseException as e:            # noqa: BLE001
        attempt.status = classify(e)
        attempt.detail = f"during benchmark: {type(e).__name__}: {e}"
    return attempt


def validate(path: str, spec: KernelSpec, device: torch.device,
             trials: int = 3, timeout: float = 300.0) -> Attempt:
    """Validate a generated kernel in a subprocess, so a crash costs one
    candidate rather than the run."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = tf.name
    cmd = [sys.executable, "-m", "kernelforge.agent.codegen_worker",
           "--path", path, "--target", _target_key(spec),
           "--out", out_path, "--trials", str(trials)]
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.dirname(RESULTS) + os.pathsep
                         + env.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=os.path.dirname(RESULTS),
                              env=env)
    except subprocess.TimeoutExpired:
        _unlink(out_path)
        return Attempt(0, spec.name, "timeout",
                       detail=f"exceeded {timeout:.0f}s -- likely a hang or an "
                              f"unbounded loop in the generated kernel")

    if os.path.exists(out_path) and os.path.getsize(out_path):
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        _unlink(out_path)
        return Attempt(**data)

    _unlink(out_path)
    # No verdict written: the worker died before it could report. That is the
    # CUDA-context corruption case, and it is why this runs out of process.
    tail = (proc.stderr or "").strip().splitlines()[-3:]
    joined = " | ".join(tail)
    status = ("illegal_memory_access"
              if "illegal memory access" in joined.lower()
              else "worker_crashed")
    return Attempt(0, spec.name, status,
                   detail=f"exit {proc.returncode}: {joined[:220]}")


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _target_key(spec: KernelSpec) -> str:
    for key, value in TARGETS.items():
        if value is spec:
            return key
    raise KeyError(spec.name)


# --------------------------------------------------------------------------
# The loop


def run_codegen(device: torch.device, targets: List[str], iterations: int,
                proposer, spec_block: str, verbose: bool = True,
                repair_budget: int = 0) -> Dict:
    """Generate kernels, optionally repairing failures instead of resampling.

    With `repair_budget > 0` a failed candidate is sent back with its compiler
    diagnostic and a structural diagnosis of the numerical error, and the model
    is asked to fix that specific kernel. Independent resampling wastes
    everything a failure taught; this is the difference between an AI that
    generates code and an AI that helps a developer get code working.
    """
    os.makedirs(GENERATED, exist_ok=True)
    records: Dict[str, List[Attempt]] = {}
    taxonomy: Dict[str, int] = {}
    stalls: Dict[str, int] = {}

    for target_name in targets:
        spec = TARGETS[target_name]
        history: List[Attempt] = []
        if verbose:
            print(f"\n=== generating {spec.name} "
                  f"({spec.entry_point}) ===", flush=True)

        # `pending` is the last failed candidate. While one exists and repair
        # budget remains, the next call fixes it rather than starting over.
        pending = None          # (source, Attempt)
        repairs_left = repair_budget

        for it in range(1, iterations + 1):
            repairing = pending is not None and repairs_left > 0
            if repairing:
                src, att = pending
                system, prompt = REPAIR_PROMPT, repair_prompt(spec, src, att)
            else:
                system = SYSTEM_PROMPT
                prompt = spec.prompt(
                    spec_block, "\n".join(a.as_history() for a in history[-6:]))
            try:
                text = proposer.raw_completion(system, prompt)
            except Exception as e:
                # Record it. An arm where every request was rate-limited used to
                # produce an empty taxonomy, which reads identically to "the
                # model generated nothing usable" -- and in a model comparison
                # that turns our infrastructure problem into the model's score.
                # `api_error` is not a verdict on the model; it means we never
                # got an answer to judge.
                taxonomy["api_error"] = taxonomy.get("api_error", 0) + 1
                if verbose:
                    print(f"  [{it}] API error: {type(e).__name__}: {e}")
                continue

            source = extract_code(text)
            if source is None:
                att = Attempt(it, spec.name, "no_code_in_reply",
                              detail=text.strip()[:200])
                history.append(att)
                taxonomy[att.status] = taxonomy.get(att.status, 0) + 1
                if verbose:
                    print(f"  [{it}] {att.status}")
                continue

            digest = hashlib.sha1(source.encode()).hexdigest()[:10]
            path = os.path.join(GENERATED, f"{spec.name}_{digest}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Generated by the KernelForge codegen loop.\n"
                        f"# Target: {spec.name}  entry: {spec.entry_point}\n"
                        f"# Reviewed: NO -- generated code is not shipped until\n"
                        f"# a human has read it. See docs/CODEGEN.md.\n\n"
                        + source + "\n")

            att = validate(path, spec, device)
            att.iteration, att.source_path = it, os.path.relpath(path, RESULTS)
            if repairing:
                att.repair_of = pending[1].iteration
                repairs_left -= 1
            history.append(att)
            taxonomy[att.status] = taxonomy.get(att.status, 0) + 1

            stalled = False
            if att.status == "ok":
                pending = None
                repairs_left = repair_budget      # reset for the next lineage
            else:
                if repairing and error_signature(att) == error_signature(pending[1]):
                    # The repair reproduced the same failure: the model has not
                    # understood it, and further repairs on this lineage are
                    # wasted. Abandon it and sample fresh.
                    stalled = True
                    att.status_note = "repair stalled"
                    pending, repairs_left = None, repair_budget
                    stalls[spec.name] = stalls.get(spec.name, 0) + 1
                else:
                    pending = (source, att)

            if verbose:
                tag = f"repair of #{att.repair_of}" if att.repair_of else "fresh"
                extra = ""
                if att.status == "ok":
                    extra = (f" env={att.envelope:.3f} {att.median_ms:.4f}ms "
                             f"{att.speedup_vs_torch:.2f}x vs torch")
                if stalled:
                    tag += " STALLED"
                print(f"  [{it}] {tag:<22} {att.status:<24}{extra}"
                      f"{'' if att.status == 'ok' else '  ' + att.detail[:70]}",
                      flush=True)
            torch.cuda.empty_cache()

        records[target_name] = history

    total = sum(taxonomy.values()) or 1
    if verbose:
        print(f"\n=== codegen taxonomy: {total} generated kernels ===")
        for status, n in sorted(taxonomy.items(), key=lambda kv: -kv[1]):
            print(f"  {status:<30} {n:3d}  ({100.0 * n / total:5.1f}%)")

    lineage = {"fresh": 0, "repair": 0, "fresh_ok": 0, "repair_ok": 0}
    for atts in records.values():
        for a in atts:
            key = "repair" if a.repair_of else "fresh"
            lineage[key] += 1
            if a.status == "ok":
                lineage[key + "_ok"] += 1
    if verbose and lineage["repair"]:
        fr = 100.0 * lineage["fresh_ok"] / max(lineage["fresh"], 1)
        rr = 100.0 * lineage["repair_ok"] / max(lineage["repair"], 1)
        print(f"\n  fresh attempts   {lineage['fresh_ok']:2d}/{lineage['fresh']:2d} "
              f"correct ({fr:.0f}%)")
        print(f"  repair attempts  {lineage['repair_ok']:2d}/{lineage['repair']:2d} "
              f"correct ({rr:.0f}%)")

    payload = {
        # The id we asked for and the id(s) the gateway actually served.
        # They differ on aliased names, and attributing a score to the
        # requested name would credit the wrong model.
        "model_requested": getattr(proposer, "model", None),
        "model_served": sorted(getattr(proposer, "served_models", []) or []),
        "taxonomy": taxonomy,
        "lineage": lineage,
        "stalls": stalls,
        "repair_budget": repair_budget,
        "targets": {k: [dataclasses.asdict(a) for a in v]
                    for k, v in records.items()},
    }
    out = os.path.join(RESULTS, "codegen.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if verbose:
        print(f"\nwritten: {out}\ngenerated sources: {GENERATED}")
    return payload
