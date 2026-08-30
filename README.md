# KernelForge

**TikTok TechJam 2026 — Track 3: Implement a GPU Kernel for a Transformer Layer**

---

## Project overview

The track asks for GPU kernels that implement a fixed Transformer layer, stay
within `relative error < 0.02, abs error < 0.002` of the reference, and run as
fast as possible across a published list of input shapes — noting explicitly that
*"participants can choose different implementations for different shapes by
adding shape checks."*

We took that literally. Rather than hand-tuning one kernel, we built the loop
that produces and selects them:

1. **Profile** the workload and compute a per-stage precision error budget.
2. **Propose** candidate configurations — from a hand-written heuristic, or from
   an LLM given the same hardware spec sheet, measured profile and error budget.
3. **Prove** each candidate correct against the organizer's own tolerance rule,
   over multiple seeds, *before* it is allowed to be timed.
4. **Time** it honestly — interleaved round-robin, rotating order, medians.
5. **Freeze** the winner into a per-(architecture, shape) dispatch table, then
   re-verify that table end to end and demote anything that drifts.

At run time `submission.py` does none of that: it looks up the current GPU
architecture, dtype and shape in the frozen table, and runs the plan it names. No
LLM call, no autotuning stall, no nondeterminism.

### What we wrote

- **Triton FlashAttention** handling causal **and** key-padding masking
  in-register. PyTorch's SDPA takes `is_causal` *or* an `attn_mask`, not both
  cheaply, and the benchmark's generator produces exactly that combination.
  Supports `head_dim` ∈ {8, 16, 32, 64, 128, 256}; `head_dim` 8 is handled by
  exact zero-padding, since Triton's `tl.dot` needs a contraction of ≥16.
- **Fused add + mask + LayerNorm**, fused QKV projection, and CUDA-graph capture
  that removes ~105 kernel launches per forward on the 6-layer default stack.
- **An accuracy gate** built from the organizer's own comparison code, collapsed
  to one number — *envelope utilization* — so "how close to failing" is
  measurable rather than a yes/no.

### Results

Measured on two nodes of the NUS SoC Slurm cluster. Full tables in
[RESULTS.md](docs/RESULTS.md); every figure is regenerated from the JSON artifacts in
`results/`.

| | A100-80 PCIe (sm_80) | H100 NVL (sm_90) |
|---|---|---|
| official shapes measured | 13 of 14 | 13 of 14 |
| median vs the reference | 5.39x | **7.35x** |
| range vs the reference | 2.32x – 15.25x | 2.34x – 13.29x |
| median vs `torch.compile` | 1.53x | 1.69x |
| **faster than both references** | **13 of 13** | **13 of 13** |
| passed the accuracy gate | **all** | **all** |
| demoted on re-verification | 0 | 0 |

**Every official shape with a runnable reference beats both the naive baseline
and `torch.compile(max-autotune)` — 26 of 26 measurements.** The narrowest
margins are 2.32x over the reference and 1.02x over `torch.compile`; that last
one is parity rather than a win. `torch.compile` clears the accuracy gate on all
14 official shapes, so every comparison above is against an admissible opponent.

**Shape 14 is the exception.** `B32-S100000-d1024-H16-F1024-L2` would require the
reference to allocate an 18.6 TB attention score matrix, so the reference cannot
run it and neither can `torch.compile` applied to the reference. We run it in
**77.7 s on the A100** and **54.5 s on the H100**, 45.9 GB peak, finite output of
the correct shape. We quote no speedup — a ratio against something that cannot
run is not a measurement.

### Where to look

| | |
|---|---|
| the full technical report | [docs/TECH_REPORT.md](docs/TECH_REPORT.md) |
| every number, per shape, per GPU | [RESULTS.md](docs/RESULTS.md) or `docs/dashboard.html` |
| AI-written kernels and the model bake-off | [docs/CODEGEN.md](docs/CODEGEN.md) |
| what we measured about the tolerance | [docs/PRECISION.md](docs/PRECISION.md) |
| why the rewrite is bit-exact | [docs/EQUIVALENCE.md](docs/EQUIVALENCE.md) |
| what the speedups are worth to a real workload | [docs/USE_CASE.md](docs/USE_CASE.md) |

---

## Setup and installation

Requires a CUDA-capable GPU. Verified on PyTorch 2.11–2.13, CUDA 12.6 and 13.0.

```bash
pip install -r requirements.txt
python -m kernelforge.cli doctor
```

`doctor` prints a capability table for the machine it runs on — what can be run
here, and what it will fall back to — so a missing dependency shows up before a
benchmark rather than halfway through one.

Install `torch` matched to your CUDA version first if pip picks the wrong build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Triton supplies our two hand-written kernels: `pip install triton` on Linux,
`pip install triton-windows` on Windows (needs MSVC build tools). Without it the
code falls back to `F.scaled_dot_product_attention` and `torch.layer_norm`
automatically — slower, still correct.

### Running on your GPU

Nothing here requires our hardware. Dispatch resolves in this order: an exact
entry for your `(architecture, dtype, shape)`; the conservative bit-exact plan if
we measured your architecture but not that shape; then an architecture-appropriate
default. We ship tuned tables for **sm_80** (A100), **sm_90** (H100) and
**sm_86** (RTX 30-series).

On **Ampere or newer** the fp32 default uses fp16 for attention and `out_proj`,
justified because the reference itself runs its matmuls at TF32 — see
[docs/PRECISION.md](docs/PRECISION.md). On **Turing or Volta** there is no TF32,
that reasoning does not hold, and dispatch falls back to the bit-exact plan
automatically. `doctor` tells you which applies.

To tune your own card properly, see *Steps to reproduce* below.

### Using your own LLM endpoint

The LLM is a **build-time** proposer. It is never on the inference path, and
everything it produces passes the same accuracy gate as everything else. With no
credentials configured, a deterministic heuristic proposer runs instead and the
whole loop still works.

Any OpenAI-compatible `/v1/chat/completions` endpoint works. Copy `.env.example`
to `.env` (gitignored) and set two variables — OpenAI, a local Ollama, vLLM, LM
Studio, or your own gateway. `ANTHROPIC_API_KEY` is supported directly.

The client is plain `urllib`, so there is no SDK to install and no vendor
lock-in.

### With no GPU at all

```bash
python -m pytest tests/ -q            # 41 of 133 pass, 92 skip cleanly
python scripts/showcase.py --no-gpu   # three narrated acts from committed data
python scripts/report.py              # regenerate docs/RESULTS.md
```

Every number in `docs/RESULTS.md` and `docs/dashboard.html` comes from committed
JSON, so the results are inspectable without reproducing them.

---

## Steps to reproduce your results

**The headline claim** — every official shape beats both the naive baseline and
`torch.compile`, on whatever GPU you have:

```bash
python -m kernelforge.cli tune --shapes-file official_shapes.txt
python -m kernelforge.cli verify --shapes-file official_shapes.txt --demote
python scripts/report.py
```

`official_shapes.txt` holds the 14 shapes from Appendix 3.7 of the problem
statement. The search spends up to `--case-budget` seconds per shape (default
300), so wall-clock scales with how many shapes fit your GPU. Any other shape is
tuned the same way by writing its signature:

```bash
python -m kernelforge.cli tune --shapes B4-S777-d640-H10-F2560-L9
```

**Through the organizer's script directly.** This imports
`torch_transformer_benchmark.py` *unmodified* and substitutes
`UserOptimizedTransformer` from `submission.py`:

```bash
python scripts/run_official.py --batch-size 64 --seq-len 128 \
    --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --causal
```

**Shape 14.** `--scan` sweeps sequence length and reports where your GPU stops,
so it runs on any card; the full shape peaks at 45.9 GB and needs ~80 GB with the
input and output resident:

```bash
python scripts/shape14.py --scan     # how far this GPU gets
python scripts/shape14.py            # the full shape
```

**The AI-assisted half.** Needs an LLM endpoint; skip it and the heuristic arm
still runs:

```bash
python scripts/pick_model.py                                    # score your models
python -m kernelforge.cli agent --provider heuristic --tag heuristic
python -m kernelforge.cli agent --provider llm       --tag llm
python -m kernelforge.cli codegen                               # the model writes Triton
```

**Everything else:**

```bash
python -m kernelforge.cli budget      # per-stage precision error budget
python -m kernelforge.cli sweep       # the full matrix, not just official shapes
python scripts/usecase.py --cold      # the use case untuned; drop --cold once tuned
python scripts/dashboard.py           # regenerate docs/dashboard.html
```

`scripts/README.md` describes each script. The Slurm job files that produced our
cluster numbers are not in this repository — they run against one specific
university cluster, so they would be dead weight here; every result they produced
is committed as JSON, and `scripts/README.md` gives the portable equivalent of
each.

---

## Limitations, and what we would improve given more time

- **Shape 14's correctness cannot be proven at full size.** Nothing can compute a
  reference output at `S=100000`, so there is no envelope to measure on the one
  official shape that matters most. We verify the same code path against an exact
  reference at every length that *does* fit, and separately verify that slicing
  the batch does not change the answer. That is weaker than a measured envelope.
- **Shape 14 spends 97.7% of its GPU time in the fp32 attention fallback.**
  Triton's `tl.dot` needs a narrow float type, so an fp32 attention stage falls
  through to `F.scaled_dot_product_attention`. An fp32 flash kernel, or a bf16
  path with an error budget verified at that length, is the only optimization
  that would matter for this shape.
- **Our flash kernel does not win every causal shape.** It wins the official ones
  — shape 13 (`B64-S1024`) is 12.14x over the reference and 4.26x over
  `torch.compile` — but at small batch and long sequence the search selects
  `torch.compile` instead, because skipping the tiles above the diagonal halves
  the work without shrinking the launch grid: 17% of tensor-core ceiling on the
  A100 against 45–50% for the same shape uncausal. A persistent-tile kernel that
  packs the triangular work is the fix.
- **The accuracy margin rests on a threshold narrower than the noise.** The same
  (case, plan) pair re-measures up to 0.141 apart across fresh processes, as
  cuBLAS picks different kernels for the same call; admission is gated at 0.80
  and re-verification permits 0.90, a gap of 0.10. The asymmetry errs toward the
  bit-exact plan and nothing has been demoted on any GPU, but the gap should be
  set from that measurement rather than by judgement.

---

## Team member contributions

Solo entry. All design, implementation, measurement and writing by the
submitting author, using Claude Code (Claude Opus 5) as the development
environment throughout.
