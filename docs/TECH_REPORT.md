# KernelForge — Technical Report

**TikTok TechJam 2026, Track 3: Implement a GPU Kernel for a Transformer Layer**

This report covers what §3.2 asks for: the environment, the optimizations, the
final test results, and the AI tools and skills used to produce them.

---

## 1. Environment

Regenerate with `python -m kernelforge.cli env` on your own machine.

Every number in this report was measured on one of two nodes of the NUS SoC
Slurm cluster. There is no third machine and no mixed reporting.

### Measurement machines

| | A100 node (`xgph1`) | H100 node (`xgpi10`) |
|---|---|---|
| GPU | NVIDIA A100 80GB PCIe — sm_80, 108 SMs, 163 KB shared/block, ~1651 GB/s | NVIDIA H100 NVL — sm_90, 132 SMs, 227 KB shared/block, ~3511 GB/s |
| CPU | Intel Xeon Gold 6326 @ 2.90 GHz, 64 cores | AMD EPYC 9334, 128 cores |
| Memory | 256 GB | 1030 GB |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, Linux 6.8 | same |
| Disk | NFS home (server-side quota ≈5.5 GB) + **3.1 TB node-local NVMe scratch** | same |

The CPU rows are not filler. The nodes differ by vendor and by a factor of two
in core count, and jobs are allocated 8 cores of them — which matters because,
as the next two sections show, kernel *submission* rate is part of what we are
measuring on the small shapes, and it is a property of the host, not the GPU.

Two environment details had real engineering consequences, which is why they are
reported rather than assumed:

- **The disk decided the cluster job design.** A PyTorch install plus Inductor
  and Triton caches does not fit the NFS home quota. Jobs build their entire
  environment on node-local NVMe and copy back only
  small JSON artifacts.
- **The CPU matters because small shapes are launch-bound.** A 6-layer forward
  issues ~105 kernel launches, at ~13 µs of CPU each on Windows WDDM versus
  ~4–5 µs on the Linux cluster nodes. On the smallest shapes the CPU, not the
  GPU, is the bottleneck — which is what makes CUDA-graph capture the dominant
  optimization there.

### Host CPU affects the measured speedup

A cluster job gets an exclusive GPU but *shares* the host CPU. The baseline
issues ~105 kernel launches per forward while our fastest plans capture a CUDA
graph, so a slower host inflates the baseline far more than it inflates us. Two
A100-80 runs, same GPU model, same code:

| node | µs per kernel launch | GPU time, `default` | baseline, `default` | our reported speedup |
|---|---|---|---|---|
| `xgph1` | 4.6 – 5.6 | 3.02 ms | 2.76 ms | 2.76x |
| `xgpj0` | **9.9 – 11.8** | 3.02 ms | **7.31 ms** | **10.89x** |

Identical GPU times; `xgpj0`'s host is ~2.2x slower at submitting kernels, which
alone moves the headline number by 4x. A100 figures in this report are from
`xgph`-class nodes, and `SLURMD_NODENAME` plus measured launch overhead are
recorded in every sweep artifact so two runs can be checked for comparability.

Both nodes hold stable clocks under sustained load. We report no figure measured
on throttling hardware — the same shape there measured 16.9x, 36.5x and 45.6x
across three runs.

---

## 2. What the script defines

Three properties of `torch_transformer_benchmark.py` shaped every decision.

**It evaluates a 6-layer stack, not a single layer.** Any perturbation is
amplified as it propagates — measured at roughly 10³ end to end.

**The accuracy rule is an OR, not an AND:**

```python
passed_mask = finite_mask & (abs_ok | rel_ok)
```

so the per-element allowance is `max(atol, rtol·|ref|)`. We collapse that to one
number, *envelope utilization* = `max(abs_err / allowance)`, where <1.0 passes.
Script defaults are `atol=2e-3, rtol=2e-2`. We read both out of `parse_args` at
import time rather than hardcoding them, so the gate tracks the script — they
were half this before the 27 August 2026 revision, and most of our table was
searched under that tighter gate.

**In float32 the reference already runs TF32.** `--allow-tf32` defaults true and
`matmul_precision='high'`. TF32 carries 10 mantissa bits; fp16 carries 11.
Computing in fp16 is therefore not a precision regression against *this*
reference — a fact the error budget later confirmed quantitatively.

Accuracy is a hard gate (`return 2` skips benchmarking entirely), so correctness
is not a constraint to satisfy at the end. It is the thing the whole design is
organized around.

---

## 3. What we built

A measurement-driven optimization loop, plus the kernels it produced.

```
profile ─► propose ─► compile ─► GATE ─► measure ─► feed back ─► freeze
                                  │
                    numerically wrong candidates stop here,
                    before they can ever post a fast time
```

| stage | module | what it does |
|---|---|---|
| profile | `agent/profile.py` | classifies a shape into launch- / gemm- / attention- / bandwidth-bound from measured kernel time, CPU submission time and launch count |
| propose | `agent/proposers.py` | emits a `Plan`; heuristic or LLM, same interface |
| gate | `numerics.py` | exact replica of the script's OR rule, over multiple seeds |
| measure | `bench.py` | round-robin interleaved timing, median + p90, NVML power |
| freeze | `dispatch.py` | writes the winner into a per-(arch, dtype, shape) table with its evidence |

At run time `submission.py` looks up the frozen table and runs the named plan:
no LLM call, no autotuning stall, no nondeterminism, and a fallback to the
bit-exact path if anything fails.

### The kernels

- **Fused `add + mask + LayerNorm` per block boundary** (`ops/layernorm.py`) —
  writes the new residual stream and the next sublayer's normalized input in one
  pass, reading the *next* layer's norm weights, so a block boundary costs one
  kernel instead of four.
- **FlashAttention with native causal + key-padding** (`ops/flash.py`) — fp32
  online softmax, `exp2` with a folded `log2(e)` scale so the inner loop maps to
  a single MUFU instruction. That substitution is free: against an exact
  reference a `tl.exp` variant measures the same envelope to four decimal places
  at every length from `S=128` to `S=4096`. `F.scaled_dot_product_attention`
  cannot take a causal flag and a padding mask together without materializing an
  `attn_mask` and dropping off its flash backend; we handle both predicates
  in-register.
- **Zero head-split copies** — the flash kernel consumes strided views of the
  fused QKV buffer and writes into a `[B,S,H,Dh]` buffer whose flat view is
  already the merged-head layout. The baseline spends 4% of its GPU time on
  `Memcpy DtoD` from `.contiguous()`; we spend none.
- **One fused QKV GEMM** instead of three.
- **CUDA-graph capture of the whole 6-layer stack** — removes ~105 launches per
  forward.
- **An exact S=1 specialization** — softmax over a single key is 1.0
  bit-for-bit, so attention is the identity on V. Skipping it is algebra, not
  approximation, and it turned our worst shape into a winning one.
- **`torch.compile` as a candidate, not just a yardstick** — the script lists it
  as a suggested direction, so it competes in the search under the same gate. It
  wins long-causal shapes outright.

### Unseen shapes

An unseen shape is a first-class input, not an edge case:

```bash
python -m kernelforge.cli tune --shapes-file official_shapes.txt
```

The spec is parsed, classified into a bottleneck regime, searched, benchmarked,
frozen; the whole table is then re-verified with demotion. On two shapes never in
the matrix: `B4-S777-d640-H10-F2560-L9` found `fp16[attn]+graph` at 1.99x;
`B12-S384-d768-H12-F3072-L6-causal` found nothing clearing the margin and shipped
the bit-exact plan at 1.02x.

### Shape dispatch

Keyed on
`(architecture, I/O dtype, B, S, d_model, heads, ffn, layers, causal)`, with
fallback to the nearest same-regime entry and then to a dtype-level default.
`num_layers` is part of the key because amplification grows with depth: the same
plan measures 0.55 envelope at L=6 and 0.98 at L=12.

---

## 4. Optimizations and their accuracy cost

The central tool is a **per-stage error budget** (`kernelforge/budget.py`):
narrow one stage to fp16, hold everything else wide, measure the envelope it
consumes. On the default shape:

| stage | marginal envelope cost |
|---|---|
| attention (QKV GEMM + flash) | **−0.030** |
| out_proj | +0.031 |
| ffn2 | +0.221 |
| ffn1 | +0.326 |

The attention cost is **negative** — fp16 attention lands *closer* to the
reference than our fp32 flash path, because the reference's own QK^T runs at
TF32. The FFN GEMMs dominate the budget because they carry the longest reduction
dimensions. That ordering is the opposite of intuition, and it is why precision
is chosen by measurement rather than by policy.

Single-change ablation from the baseline (envelope; **bold** = bit-identical):

| change | fp32 | fp16 | bf16 |
|---|---|---|---|
| loop rewrite, exact attention | **0.000** | **0.000** | **0.000** |
| + fused QKV GEMM | **0.000** | 2.655 FAIL | **0.000** |
| + fused Triton norm | 0.675 | 2.991 FAIL | 18.981 FAIL |
| attention → SDPA | 0.694 | 2.197 FAIL | 19.775 FAIL |
| attention → our flash kernel | 0.597 | 2.594 FAIL | 18.066 FAIL |
| residual stream → fp16 | 2.774 FAIL | — | — |

Two consequences: our structural rewrite is provably free (bit-identical on
every dtype, which is what makes a safe fast path possible), and the residual
stream must stay wide because it accumulates across 2·`num_layers` sublayers.

Full derivation, including the depth sweep and the `torch.compile` admissibility
measurements, is in [PRECISION.md](PRECISION.md). The masking-equivalence
argument is in [EQUIVALENCE.md](EQUIVALENCE.md).

---

## 5. Final test results

Full tables in [RESULTS.md](RESULTS.md). Every row cleared the accuracy gate at a
0.80 margin over three seeds **before** it was timed; the frozen table was then
re-measured with `cli verify --demote` on each machine — **no entry was
demoted**.

### The official 14 shapes (Appendix 3.7)

Both cluster nodes, every shape they can run.

| | A100-80 PCIe (sm_80) | H100 NVL (sm_90) |
|---|---|---|
| shapes measured | 13 of 14 | 13 of 14 |
| median vs reference | 5.39x | **7.35x** |
| range vs reference | 2.32x – 15.25x | 2.34x – 13.29x |
| median vs `torch.compile` | 1.53x | 1.69x |
| worst vs `torch.compile` | 1.02x | 1.03x |
| faster than both references | **13 of 13** | **13 of 13** |
| worst envelope at selection | 0.733 | 0.767 |
| worst envelope on re-verification | 0.847 | 0.837 |
| demoted on re-verification | **0** | **0** |

**26 of 26 measurements beat both the naive baseline and
`torch.compile(max-autotune)`.** There is no shape on either cluster GPU where we
lose. The narrowest margins — 2.32x over the reference on shape 8, and 1.02x over
`torch.compile` on shape 1 — are the honest edges of that claim, and shape 1's
1.02x is close enough to parity that we would not defend it as a meaningful win
on its own.

`torch.compile` clears the accuracy gate on all 14 official shapes, so each of
those comparisons is against an admissible opponent rather than a disqualified
one.

Envelope utilization is `max(abs_err / max(atol, rtol·|ref|))`; 1.0 fails. The
two envelope rows are different measurements — at selection, and re-measured
later in a fresh process. Re-measurement of the same (case, plan) pair moves by
up to **0.141** over six fresh processes, as cuBLAS selects different kernels for
the same call — wider than the 0.10 gap between the 0.80 admission margin and the
0.90 demotion threshold. That asymmetry errs toward demoting a good plan to the
bit-exact one, and no entry has in fact been demoted on any GPU. Worst seen from
a shipped entry: 0.847.

Shape 14 is missing from both columns (below); all 13 others run on both nodes.

Through the organizer's script, unmodified, on official shape 1 (A100 node
`xgph0`):

```
=== Accuracy check ===
criterion: abs_error <= 0.002 OR relative_error <= 2.00%
summary: PASS | max_abs=0.00100768 | max_rel=18461.6 | failed=0/5242880
baseline : median=1.9528 ms | throughput=4195070.86 token/s
optimized: median=0.8335 ms | throughput=9828009.49 token/s
speedup  : 2.343x based on median latency
```

This is lower than the 3.22x in the table above for the same shape: our harness
interleaves candidates round-robin over many more repeats, while the organizer's
script times each implementation in one block on an unpinned node. **The
organizer's number is the one a judge reproduces**; ours is the controlled
comparison used to *choose* between plans.

### Shape 14

`B32-S100000-d1024-H16-F1024-L2-causal`. The baseline forms `[B,H,S,S]` before
its softmax — 18.6 TB. We run it in **77.7 s on the A100-80** and **54.5 s on the
H100 NVL**, at 45.9 GB peak, with finite output of the correct shape.

No speedup is quoted: a ratio against an implementation that cannot run is not a
measurement. The claim is that the shape is reachable with a fused kernel and
unreachable without one.

Three things were required. The third is in the fp32 SDPA fallback: SDPA accepts
`is_causal` or an `attn_mask` but not both, so a causal-plus-padding shape builds
an `[S,S]` mask — 37.25 GiB here, reintroducing the quadratic term the flash
kernel removes. When no token is padded the mask is a no-op and `is_causal` alone
is exact, which holds for all 14 official shapes; that path costs 45.9 GB peak
instead of 84.6 GB.

**Where the 77 seconds go.** Profiled at batch 1 and full `S`, **97.7%** of GPU
time is a single `fmha_cutlassF_f32_aligned_64x64_rf_sm80` launch — PyTorch's
memory-efficient attention in fp32. Triton's `tl.dot` needs a narrow float type,
so an fp32 attention stage cannot use our flash kernel and falls through to SDPA.
Everything else in the stack — the GEMMs, the fused norm, the elementwise work —
is the remaining 2.3%. An fp32 flash kernel, or a bf16 attention path with an
error budget verified at that length, is the only optimization that would matter
here.

### Divergence across GPUs

**4 of the 13 official shapes chose a semantically different plan** on the A100
than on the H100 — a different set of fp16 stages, or CUDA graphs on one card and
not the other.

Counted on plan *specifications*, not names: a name lists its fp16 stages in
admission order, so two spellings of one plan look like two plans. That
distinction matters — counting names said every shape diverged.

Four in thirteen is still the argument for searching rather than hand-tuning, but
it is smaller than we expected, and the list explains why: eleven of the fourteen
official shapes are `S=128` variants of one another, far more homogeneous than a
real serving mix.

### Where we lose

**On the cluster GPUs, nowhere** — all 26 measurements beat both references. Three
things qualify that:

- **Shape 1 on the A100 is 1.02x over `torch.compile`** — parity, not a win.
  Shape 2 (1.06x) and shape 5 (1.07x) are nearly as close.
- **Long causal attention on a small GPU is our weakest regime.** Shape 13
  (`B64-S1024`) goes 4.26x and 4.11x our way on the A100 and H100, but on a
  46-SM card we have measured 0.94x of `torch.compile`. Outside our reported set,
  but a reader on a small card should expect it.
- **Shape 14 is not a win either.** It is a shape we can run and the reference
  cannot, which is a different kind of claim and quoted as one.

## 6. What the speedups are worth

Derived from the measured artifacts by `scripts/impact.py`; power is board TDP, cost assumes $0.12/kWh and PUE 1.2, and every figure scales linearly with those.

**Mean energy reduction per token across gate-passing shapes: 59% on A100, ~70% on H100.** The work is a build-time cost paid once per (architecture, shape); the saving accrues on every inference afterwards, on hardware already bought and running.

That is also why the dispatch table is per-architecture: a fleet is heterogeneous, and 4 of the 13 official shapes chose a semantically different plan on the A100 than on the H100 (section 5).

## 7. AI tools and skills used

**Claude Code (Claude Opus 5)** was the development environment for the whole
project: reading the benchmark script, writing the Triton kernels, designing the
harness, driving the cluster, and diagnosing failures.

**NUS SoC LLM-as-a-Service** (OpenAI-compatible gateway) powers the agent's LLM
proposer at build time. The client is plain `urllib` — no SDK dependency, so the
loop reproduces from a clean checkout.

**Model selection is measured.** Every model the gateway serves was run through
the kernel-generation loop — 20 kernels per sample, same targets, same gate:

| model (as served) | correct kernels (pooled) |
|---|---|
| **`qwen3.8:27b`** | **48/60 (80%)** |
| `qwen3-coder-next` | 17/40 (43%) |
| `qwen3.6:35b` | 9/40 (22%) |
| `gemma4:26b` | 7/40 (18%) |
| `ornith1.5:35b` | 3/40 (8%) |
| `qwen3.5:9b` | 0/40 |
| `llama3.1:8b` | 0/20 |

We ship **`qwen3.8:27b`**. `scripts/pick_model.py`, which scores plan-JSON
quality, was *anti-predictive* of this ranking and is not what selects the model.
Details in [CODEGEN.md](CODEGEN.md).

### Per-architecture optimization

The track asks for implementations tuned to *specific GPU hardware*, so this is
measured. The same four official shapes, the same loop, an A100 and an H100,
changing nothing but the machine; the proposer sees that machine's spec sheet
(SM count, shared memory per block, bandwidth, tensor-core support) and its
measured profile. Comparing the winning *configuration* on each card — not the
plan name, which the model writes itself:

| proposer | shapes with a different winning configuration |
|---|---|
| **LLM** (`qwen3.8:27b`) | **3 of 4** |
| heuristic | 2 of 4 |

The differences are physically sensible: on the H100 the LLM moved to
`compute=float16` on three of four shapes while staying at `float32` on the A100,
and picked larger flash tiles — the H100 has 227 KB of shared memory per block
against the A100's 163 KB, stated in the spec sheet it was given. The fourth
shape differed only in tile size.

**The LLM re-decides per architecture rather than emitting one plan and
relabelling it**, and diverges more than our heuristic, which narrows stages in a
fixed order. Every proposal still passed the same gate before it was timed.

The kernel-writing half replicates across the two cards too — same model, same
targets, independent runs:

| model | A100 (sm_80) | H100 (sm_90) |
|---|---|---|
| `qwen3.8:27b` | 16/20 | **17/20** |
| `qwen3-coder-next` | 8/20 | 9/20 |

Three independent samples of the winner (15, 16, 17 of 20) put the ranking well
outside the noise we measured on repeat arms.

**Heuristic vs LLM**, same GPU, same four shapes, same gate, 12 proposals each:

| proposer | cleared gate (A100) | cleared gate (H100) | wall clock |
|---|---|---|---|
| heuristic | 12/12 | 12/12 | 277s / 197s |
| LLM | 10/12 | 8/12 | 66s / 67s |

The heuristic proposes only what it believes legal and clears the gate every
time; the LLM proposes things the gate rejects — configurations the error budget
already ruled out (bfloat16 compute, an fp16 residual). Where it helps is
reaching combinations the cheapest-stage-first ordering cannot jump to.

### AI-generated kernel source

The model also writes complete Triton kernels against a contract, every candidate
compiled, gated and timed by the same harness. The shipped model's run:
**20 kernels, 17 correct** — `layernorm` 10/10, best 2.46x vs `torch`; `gelu`
7/10, best 3.91x.

Findings, detailed in [CODEGEN.md](CODEGEN.md):

- **Silent wrongness dominates.** One kernel computes GELU correctly in fp32
  with masked loads, then substitutes a polynomial approximation for exact `erf`
  and correctly quantifies its ~1.5e-7 error in a comment. Tolerance is measured
  against the *reference*, which calls exact `erf`: envelope 22.8. Reproducible
  on both GPUs; only the gate caught it.
- **Under-specification, not incapability.** One sentence forbidding a split of
  the reduction axis moved a target from 0/5 to 5/12.
- **Process isolation is mandatory.** An out-of-bounds generated kernel corrupts
  the CUDA context — asynchronous, uncatchable, fatal to every later CUDA call.
  Each candidate is validated in a throwaway subprocess.

We also built the obvious improvement — re-prompting with the compiler
diagnostic and a structural diagnosis of the numeric failure — and it does not
beat resampling. On the shipped model at equal attempt budget both arms reach
13/20, and the breakdown says why: a repaired kernel succeeded **1 of 4** times
against **12 of 16** for a fresh sample. Repair spends attempts at a worse rate,
so the default is `--repair 0`.

**Nothing generated is in the shipped dispatch table.** They are proposals; a
public submission should not contain code no person has read.

### Roofline

| GPU | GEMM-bound shapes | `tiny`/`decode` | long causal |
|---|---|---|---|
| sm_80 | 31–52% of tensor-core ceiling | bandwidth-bound, low intensity | **17%** |
| sm_90 | 25–48% | bandwidth-bound | **11%** |

`tiny` and `decode` have barely any arithmetic to do; their wins come from
removing launches. Long causal attention is the genuine headroom, and is where we
still fall back to a library implementation: skipping the tiles above the
diagonal halves the work but not the launch grid, so occupancy falls with it —
17% and 11% of ceiling against 45–50% for the same shape without causal masking.
The accuracy of that kernel is not the issue; a `tl.exp` variant measures the
same envelope as the shipped `exp2` one at every length tested. H100 utilization is lower because
our tiles do not saturate the larger machine — closing that means Hopper-specific
work (TMA, wgmma), which we scoped out. Per-shape table in
[RESULTS.md](RESULTS.md).

The gate is what makes an LLM proposer usable: correctness is decided
independently, *before* anything is timed, because a wrong-but-fast configuration
is what a latency-ranked search promotes.

### A non-reproducing measurement

`torch.compile` on our bit-exact rewrite at fp16 measures **0.000 envelope** with
a 2.98x speedup in about one run in five, and 2.655 / 2.502 / 3.296 / 2.853 — all
failures — in the rest. Inductor's autotuning sometimes selects a kernel set whose
rounding happens to match, and that selection is not stable across processes. The
gate runs over multiple seeds and `cli verify --demote` re-checks the frozen
table for exactly this reason.

---

## 8. Reproducing

```bash
cp .env.example .env            # optional: only the LLM proposer needs it
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install triton              # triton-windows on Windows
python -m kernelforge.cli doctor    # environment + Triton check
python -m pytest tests/ -q          # 133 tests
python -m kernelforge.cli sweep     # search + three-way benchmark
python -m kernelforge.cli verify --demote
python scripts/report.py && python scripts/impact.py
```

Bringing up a new GPU is one `sweep` invocation with no code changes: tile
legality is derived from the measured shared-memory budget and precision from a
per-stage error budget measured on that target.
