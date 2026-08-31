# KernelForge — Technical Report

**TikTok TechJam 2026, Track 3: Implement a GPU Kernel for a Transformer Layer**

This report covers what §3.2 asks for: the environment, the optimizations, the
final test results, and the AI tools and skills used to produce them.

---

## 1. Environment

Regenerate with `python -m kernelforge.cli env` on your own machine.

Every number in this report was measured on the NUS SoC Slurm cluster: five GPUs
across four architectures, each result tagged with the node that produced it.
There is no mixed reporting.

### Measurement machines

| | A100 node (`xgph0`) | H100 node (`xgpi2`) |
|---|---|---|
| GPU | NVIDIA A100 80GB PCIe — sm_80, 108 SMs, 163 KB shared/block, ~1651 GB/s | NVIDIA H100 NVL — sm_90, 132 SMs, 227 KB shared/block, ~3511 GB/s |
| CPU | Intel Xeon Gold 6326 @ 2.90 GHz, 64 cores | AMD EPYC 9334, 128 cores |
| Memory | 256 GB | 1030 GB |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, Linux 6.8 | same |
| Disk | NFS home (server-side quota ≈5.5 GB) + **3.1 TB node-local NVMe scratch** | same |

Three further nodes carry the pre-Ampere tables. They share the cluster's
software image, but build against CUDA 12.6 rather than 13.0 — PyTorch's CUDA 13
wheels contain no `sm_70` kernels at all, so on Volta the organizer's own
benchmark fails at `generate_random_case` with
`cudaErrorNoKernelImageForDevice` until torch is installed from the cu126 index.

| | TITAN RTX (`xgpe1`) | Tesla T4 (`xgpf0`) | TITAN V (`xgpd0`) |
|---|---|---|---|
| arch | sm_75, 72 SMs | sm_75, 40 SMs | sm_70, 80 SMs |
| memory | 23.5 GB | 14.6 GB | 11.8 GB |
| shared/block | 64 KB | 64 KB | 96 KB |
| measured bandwidth | ~568 GB/s | ~242 GB/s | ~600 GB/s |
| tensor cores | yes, no TF32 | yes, no TF32 | yes, no TF32 |

The two `sm_75` cards are the same ISA at 4x apart in board power and 2.3x in
bandwidth, which is why the dispatch table is keyed on architecture but the
plans within it are selected per shape *and* re-verified per card.

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
`xgph`-class nodes with `xgpj0` excluded, and `SLURMD_NODENAME` plus measured
launch overhead are recorded in every sweep artifact so two runs can be checked
for comparability. It is also why the A100 median has read between 5.16x and
5.39x across re-tunes on `xgph0` and `xgph1`: same GPU model, same code, a
different host.

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

Five GPUs, every official shape each can run.

| | H100 NVL | A100-80 PCIe | TITAN RTX | Tesla T4 | TITAN V |
|---|---|---|---|---|---|
| arch | sm_90 | sm_80 | sm_75 | sm_75 | sm_70 |
| shapes measured | 13 of 14 | 13 of 14 | 13 of 14 | 12 of 14 | 11 of 14 |
| median vs reference | **7.31x** | 5.31x | 3.31x | 3.35x | 3.31x |
| range vs reference | 2.34 – 14.32x | 2.30 – 16.24x | 1.81 – 11.12x | 1.74 – 8.68x | 1.68 – 10.47x |
| median vs `torch.compile` | 1.68x | 1.59x | 1.07x | 1.24x | 1.01x |
| faster than both references | **13 of 13** | **13 of 13** | 12 of 13 | **12 of 12** | 10 of 11 |
| worst envelope at selection | 0.767 | 0.778 | 0.796 | 0.768 | 0.684 |
| demoted on re-verification | **0** | **0** | **0** | **0** | **0** |

**60 of 62 measurements beat both the naive baseline and
`torch.compile(max-autotune)`.** The two that do not record 0.9975x and 0.9999x,
and both dispatch entries are `compile[ro]` — the search chose `torch.compile`
as the plan there, so each is one code path timed against itself. Which side of
1.0 such a case lands on changes between runs; across our re-tunes the count has
been 60 or 61 of 62, always with the marginal cases being `torch.compile` against
`torch.compile`. The defensible claim is that no shape is slower than both
references by more than measurement noise.

The gain over `torch.compile` splits cleanly by architecture: 1.59x and 1.68x
where TF32 and tensor-core bf16 exist, 1.01x–1.24x where they do not. That is the
expected consequence of the design — most of our margin is bought by spending a
*measured* precision budget (§4), and Volta and Turing have no TF32 for the
reference to be spending in the first place, so there is no headroom to claim.
Speedup over the naive reference survives the transition intact (3.31x–7.31x
median), because CUDA-graph capture, QKV fusion and FlashAttention tiling do not
depend on precision.

A caveat on what these medians can show. Eleven of the fourteen official shapes
are `S=128`, where a 128-wide query tile covers the whole sequence. The causal
loop split described in the appendix is worth 1.08x–1.21x at `S=1024`–`8192` and
nothing at `S=128`, so a real kernel improvement moves this table barely at all.
The official set under-represents long causal attention.

That split is a hypothesis about *why*, so we tested it against the obvious
alternative — that the pre-Ampere search had simply run out of budget. Re-running
`sm_70` and `sm_75` with double the per-shape budget and more timing trials, and
changing nothing else, moved the margin over the reference (2.82x → 3.32x on
Volta, 3.02x → 3.35x on the T4) and left the margin over `torch.compile` flat
(1.01x → 1.02x, 1.26x → 1.24x). Budget was not the binding constraint.
`results/budget_probe_pre_ampere.json` records both arms.

The narrowest margins on the tuned datacentre pair — 2.23x over the reference and
1.0004x over `torch.compile` on the A100 — are the honest edges of the headline
claim. That 1.0004x is parity, not a win, and its plan is `exact+compile[ro]`
too.

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
its softmax — 18.6 TB. We run it in **17.8 s on the A100-80** and **8.5 s on the
H100 NVL**, ~46 GB peak, with finite output of the correct shape.

**The opponents had to be built.** A ratio against an implementation that cannot
run is not a measurement, so for a long time we quoted none. The fix is not to
weaken the claim but to construct opponents that can run it, and to be explicit
about which is which.

The first is the organizer's own `BaselineTransformer` with exactly one edit —
the attention chunked over query rows, giving O(S) memory instead of O(S^2).
Same modules, same weights, same two-pass softmax. It is the minimum edit that
makes the reference runnable, and it is entirely our code, which makes it a weak
check on ourselves.

The second is that same model with the attention replaced by PyTorch's
`scaled_dot_product_attention`. That is a fused, O(S)-memory attention written by
the PyTorch team, it runs this shape, and swapping it in for the one line the
baseline cannot execute leaves the rest of the organizer's model untouched. It is
not our code, and it is the number we would rather be judged on.

| | A100-80 | H100 NVL |
|---|---|---|
| chunked reference, best chunk | 166.9 s (2048) | 89.7 s (1024) |
| PyTorch SDPA reference | 77.5 s | 53.3 s |
| KernelForge | **17.9 s** | **9.0 s** |
| speedup vs chunked | 9.31x | 10.00x |
| **speedup vs SDPA** | **4.32x** | **5.95x** |

Our own time reads 17.9 s and 9.0 s in the race and 17.8 s and 8.5 s in the
standalone `shape14.py` run above. The race times best-of-two inside a process
that has already run two other implementations, so its allocator and clocks are
warmer; the ratios in the table are computed within a single run and are the
figures to trust.

Both sides run the organizer's own numerics (TF32 on, `matmul_precision="high"`).
The chunked reference is timed *first*, on an unfragmented allocator, so its wide
chunks get their best chance to fit, and its chunk size is swept with the best
time taken; on the H100 every chunk from 1024 to 4096 fits and lands within 2.4%
of the others, so it is at its plateau rather than mistuned.

SDPA is the more memory-frugal opponent — 14.6 GB peak against our 30.2 GB — so
the 4.32x is bought partly with memory. On an 80 GB card that is the right trade.
The footprint is not fixed, though: the batch slice is sized against what
`mem_get_info` reports free and halved again on any slice that still does not fit,
so a smaller card trades speed for memory rather than failing, and
`set_memory_budget(bytes)` caps the working set explicitly when the card is
shared. Slicing the batch is an execution-order change, not an approximation, so
the result does not depend on the budget — `tests/test_streaming.py` pins both the
monotonicity and the override.

`python scripts/shape14.py --race` reproduces all of it;
`results/shape14_race_<arch>.json` holds the artifact.

Three things are required. The third is in the fp32 SDPA fallback: SDPA accepts
`is_causal` or an `attn_mask` but not both, so a causal-plus-padding shape builds
an `[S,S]` mask — 37.25 GiB here, reintroducing the quadratic term the flash
kernel removes. When no token is padded the mask is a no-op and `is_causal` alone
is exact, which holds for all 14 official shapes; that path costs 45.9 GB peak
instead of 84.6 GB.

**Attention dtype is where the time is.** Profiled at batch 1 and full `S`,
**97.7%** of GPU time was a single `fmha_cutlassF_f32` launch — PyTorch's
memory-efficient attention in fp32 — with the GEMMs, the fused norm and all the
elementwise work sharing the remaining 2.3%. Triton's `tl.dot` needs a narrow
float type, so an fp32 attention stage cannot use our flash kernel at all.
Narrowing that one stage to fp16 takes the shape from **77.2 s to 20.9 s** on the
A100 and from 54.5 s to 9.6 s on the H100, at unchanged peak memory. Correcting
the causal tiling default later took those to **17.8 s** and **8.5 s** — this
shape runs on the default tile, so it inherited that fix directly.

**It is gated at full length.** The shape was long treated as ungateable because
the reference materializes 18.6 TB — but the *memory* is the obstacle, not the
arithmetic. Chunking query rows and masking against the key index computes the
same thing in O(S). Against that streamed reference, run in fp32 with TF32
disabled (stricter than the organizer's own, which leaves TF32 on) and with a
two-pass softmax rather than the online rescaling our kernel uses, the whole
output at the full batch measures **envelope 0.399**, with **0 of
3,276,800,000** elements outside tolerance. `python scripts/shape14.py --gate
--batch 32` reproduces it. Every batch element is checked, so the result rests on
no independence argument at all.

Two narrower checks agree. At the full `S=100000` the attention sub-layer alone,
against an exact float64 reference on 48 sampled query rows, measures 0.034 for
the fp16 flash kernel and 0.0001 for fp32 SDPA. And at every length where the
organizer's own baseline fits, the fp16-attention plan is indistinguishable from
the fp32 one:

| S | fp32 attention | fp16 attention |
|---|---|---|
| 1024 | 0.223 | 0.205 |
| 2048 | 0.271 | 0.268 |
| 4096 | 0.214 | 0.259 |
| 8192 | 0.211 | 0.213 |
| 16384 | 0.236 | 0.200 |

Flat in `S`, and the same to within run-to-run noise. The narrowing applies only
on fp32 input and Ampere-or-newer — the conditions the fp32 default already
relies on — and `tests/test_dispatch.py` pins both exclusions.

**And nothing else in the plan helps.** `tune` cannot rank candidates here (there
is no baseline to rank against), so the rest of the plan space was searched the
same way: gate at `S=16384`, time at `S=100000`. Adding the fused norm, or
narrowing `out_proj` and both FFN GEMMs as well, all pass the gate (envelope
0.20–0.39) and all land within noise of each other at full length:

| plan | envelope @16384 | latency @100000 |
|---|---|---|
| shipped (`+fp16attn`) | 0.202 | 21.4 s |
| `+ fused norm` | 0.202 | 21.2 s |
| `+ out_proj fp16` | 0.268 | 21.4 s |
| `+ all stages fp16` | 0.372 | 21.3 s |
| `+ fused norm, all fp16` | 0.385 | 21.0 s |

That is the profile confirming itself: attention was ~98% of the runtime, so
once it is narrowed there is nothing left for the other stages to win. We ship
the plan with the smallest envelope of the group.

### Four architectures, and what the fallback path does without a table

The dispatch logic claims to generalize: tile legality is derived from the
measured shared-memory budget, precision from a measured error budget, and
pre-Ampere cards fall back to the bit-exact plan because the fp32 default's
argument (the reference itself runs TF32) does not hold without TF32.

Two separate things follow from that, and they are worth keeping apart.

**The pipeline ported without modification.** `sm_70` and `sm_75` were tuned by
running the same `tune` / `verify --demote` sequence on hardware the code had
never seen. No kernel, heuristic or threshold was changed for them; the search
re-derived tilings against a 64 KB shared-memory budget instead of 163 KB and
re-measured the error budget on cards with no TF32. It found fp16 stages that
*do* clear the gate on Volta and Turing — 4 of 11 `sm_70` entries and 5 of 13
`sm_75` entries ship one — which the architecture default alone would have
declined. Measuring beat inferring, on the architecture where we had inferred.

**The untuned fallback is separately checkable.** `verify --untuned` ignores the
dispatch table and evaluates whatever the architecture default resolves to,
which is what a card with no entries of its own would run:

```bash
python -m kernelforge.cli verify --shapes-file official_shapes.txt --untuned     --json results/portability_sm_70.json
```

On the TITAN V that path resolves to `safe(exact)+graph` for all 13 shapes and
is **bit-exact on the 12 it can fit — `max_abs = 0.0` exactly, envelope 0.000,
every one** — because without TF32 the default declines every precision trade and
computes what the reference computes, while still taking the CUDA-graph win,
which changes no arithmetic. `tests/test_dispatch.py` pins that behaviour so it
cannot regress silently. Artifact in `results/portability_sm_70.json`.

**Memory is the other half of portability.** On the largest-batch official shape,
`B10000-S128-d128-H4`, the organizer's reference peaks at **10.4 GB** and our
plan at **4.0 GB** — 2.6x less, because FlashAttention never materializes the
score matrix. Both fit the 11.8 GB card individually; what does not fit is the
*sweep*, which holds the baseline, both `torch.compile` variants and every
passing candidate resident at once in order to time them against each other.
That is why `dispatch_sm_70.json` has 11 entries rather than 13, and it is a
limit of how we measure, not of what we ship. `results/capacity_sm_70.json`
records both peaks.

One portability note that is not ours: PyTorch's CUDA 13 wheels contain no Volta
kernels at all, so the organizer's own `generate_random_case` fails before
reaching any of our code. A CUDA 12.x build is required on pre-Ampere, which is
what `requirements.txt` already recommends.

### Divergence across GPUs

If one plan were best everywhere, the whole search would be unnecessary. It is
not. Counting on plan *specifications* rather than names — a name lists its fp16
stages in admission order, so two spellings of one plan look like two plans, and
counting names inflates every figure here — of the 11 official shapes that all
five GPUs can run, **10 pick a different plan on at least one card**.

| pair | shapes with different winners |
|---|---|
| **TITAN RTX vs Tesla T4** | **6 of 12 (50%)** |
| H100 vs A100 | 7 of 13 (54%) |
| TITAN RTX vs TITAN V | 7 of 11 (64%) |
| A100 vs Tesla T4 | 8 of 12 (67%) |
| A100 vs TITAN RTX | 9 of 13 (69%) |
| A100 vs TITAN V | 9 of 11 (82%) |
| H100 vs TITAN RTX | 11 of 13 (85%) |
| H100 vs Tesla T4 | 11 of 12 (92%) |

Divergence rises with how different the two cards are, which is the sanity check
this table has to pass: the two most similar pairs sit at 50% and 54%, and the
Hopper-versus-Turing pairs at 85% and 92%.

The first row is the one that matters most. Those two cards are the *same
architecture* — both `sm_75`, both 64 KB shared memory per block, both without
TF32 — and they still disagree on half the shapes, because they differ 4x in
board power and 2.3x in memory bandwidth. A dispatch table keyed on architecture
alone would be wrong for one of them on 6 shapes; keying on architecture and
re-verifying per card is what makes that safe.

Even the lowest figure is half the shapes, and the official set understates the
effect: eleven of the fourteen shapes are `S=128` variants of one another, far
more homogeneous than a real serving mix.

### Where we lose

**Almost nowhere.** All 26 A100 and H100 measurements beat both references, as
do all 12 on the T4. Across all five GPUs, 60 of 62. Four things qualify that:

- **Both shortfalls are `torch.compile` measured against itself.** The TITAN RTX
  records 0.9975x on `B64-S128-d32-H4` and the TITAN V 0.9999x on
  `B4-S128-d128-H4`; both dispatch entries are `compile[ro]`, so the search
  shipped `torch.compile` there and each ratio is one code path timed twice.
  Across re-tunes the count has been 60 or 61 of 62, with the marginal cases
  always being of this kind — which is the honest way to read it: nothing is
  slower than both references by more than measurement noise.
- **Parity cases exist and are labelled.** The A100's narrowest margin is
  1.0046x over `torch.compile`, on a `compile[ro]` plan. That is parity, not a win,
  and we do not defend it as one.
- **Pre-Ampere is where the margin thins.** 1.01x–1.24x over `torch.compile` on
  `sm_70` and `sm_75`, against 1.59x and 1.68x on `sm_80` and `sm_90`. The cause
  is structural and tested against the budget alternative in section 5.
- **Shape 14 is measured against opponents, not the organizer's baseline.** That
  baseline cannot run it, so no number against it exists. The 4.32x and 5.95x are
  against PyTorch's SDPA substituted into the organizer's model — independently
  written, and the figure to prefer. The 9.31x and 10.00x are against a chunked
  reference that is our own code.

## 6. What the speedups are worth

Derived from the measured artifacts by `scripts/impact.py`; power is board TDP, cost assumes $0.12/kWh and PUE 1.2, and every figure scales linearly with those.

**Mean energy reduction per token across gate-passing shapes: 82% on A100, 84% on H100, and 67%-70% on the three pre-Ampere cards.** The pre-Ampere figures matter more than they look: those cards win little against `torch.compile` on latency, but the energy saving against the naive reference survives, because it comes from removing launches and memory traffic rather than from spending precision. The work is a build-time cost paid once per (architecture, shape); the saving accrues on every inference afterwards, on hardware already bought and running.

That is also why the dispatch table is per-architecture and re-verified per card: a fleet is heterogeneous, and 10 of the 11 official shapes that all five GPUs can run chose a different plan on at least one of them -- including 6 of 12 between two cards of the *same* architecture (section 5).

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

Measured over the official shapes only.

| | sm_80 (A100) | sm_90 (H100) |
|---|---|---|
| compute-bound shapes | `d1024` at **77%** of tensor-core ceiling, `S1024` at 25% | 53% and 15% |
| everything else | memory-bandwidth-bound | memory-bandwidth-bound |

**22 of the 26 rows are bandwidth-bound**, at arithmetic intensities of 37–75
FLOP/byte against ridge points an order of magnitude higher. That is a property
of the shape list rather than of the kernels: eleven of the fourteen official
shapes are `S=128` with `d=128` or smaller, so there is very little arithmetic to
do and the wins come from removing kernel launches. The roofline confirms there
is nothing further to take from better math on them.

The two shapes with real arithmetic tell the useful story. `B64-S128-d1024-H4`
reaches **77% of the A100's tensor-core ceiling**, which is a good place to be
for a mixed Triton/cuBLAS implementation. `B64-S1024-d128-H4` — the one long
causal shape in the set — reaches 25% on the A100 and 15% on the H100, and is
where the remaining headroom is.

Utilization is lower on the H100 than the A100 for 12 of the 13 shapes both cards
ran. The machine is larger and our tiles do not saturate it: the richest tiling
we offer costs 144 KB of shared memory, and while the A100 has 163 KB per block
the H100 has 227 KB, so the extra capacity is never used. We measured whether
larger tilings would help — `(256,128,8,2)` and four others, timed against the
richest current tile — and found a single 1.024x win on one arm, which does not
justify changing the default for every unmeasured shape. Closing the gap properly
means Hopper-specific work (TMA, wgmma) that we scoped out.

#### The causal path, measured at the kernel

Everything in this subsection is measured on **attention-kernel
microbenchmarks** — `(batch, heads, seq, head_dim)` tensors fed straight to
`flash_attention` — not on the official transformer shapes. They are the only way
to attribute a kernel cost to a cause, and they are kept separate from the
reported results for that reason.

Causal masking halves the FLOPs, so the causal kernel "should" cost 0.50 of the
non-causal one for the same shape. Before the work below it cost **0.62–0.72**,
measured at the A100's real 163 KB shared-memory budget.

Three things ship against it, and their sizes are very different:

- **The tiling target.** Sweeping every legal
  configuration across seven causal shapes puts the optimum at `64x64` for
  `head_dim <= 64` and `128x32` above it — not the richest legal tile the
  non-causal path wants. Targeting those brings the default to within 1.05x of
  each shape's own best configuration, from as far as **1.29x off** before.
- **The block interleave, which is worth 1.00x–1.03x.** Alternating heaviest and
  lightest program ids gives every scheduling wave a mix. It is a permutation of
  which program handles which block, so it is bit-identical (pinned in
  `tests/test_kernels.py`). An earlier round of this measurement passed
  `smem_kb=99.0`, a budget none of our cards has, and reported a larger figure;
  at the real budget the gain is small and confined to the smallest grid. It
  ships because it costs three integer instructions and cannot change a result,
  not because it is large.
- **The causal loop split, which is worth 1.08x–1.21x** — the largest of the
  three, and the one that came from measuring rather than guessing. It is
  described below.

That is the tile choice exhausted: even the *best* legal configuration sat at
0.62–0.72. The standard next step is a persistent-tile kernel that packs the
triangular work into equal units, so we built one — by source transformation on
the shipped kernel, leaving the per-block arithmetic provably unchanged, with a
fixed program count and a strided walk over m-blocks. Swept over four tiles and
three program counts (1x, 2x and 4x the 108 SMs), it was **slower on every
shape**, 1.03x–1.54x. These grids are already thousands of programs — batch times
heads times m-blocks — so there is no wave quantization left to recover, and
pinning the grid to a multiple of the SM count only removes parallelism the
scheduler was exploiting. `results/persistent_tile_probe_sm_80.json` records it.

**So we stopped guessing and decomposed the cost.** Three arms at one tile:
non-causal; the causal *work volume* with the mask suppressed (a timing probe,
not a correct result); and the shipped causal kernel.

| shape | half/full | mask cost |
|---|---|---|
| `B2-H8-S2048-d64` | 0.65 | 1.16x |
| `B8-H8-S2048-d64` | 0.55 | 1.19x |
| `B4-H16-S8192-d64` | **0.50** | 1.18x |
| `B64-H4-S1024-d32` | 0.60 | 1.34x |
| `B2-H8-S4096-d64` | 0.56 | 1.19x |

The work reduction was already nearly fully realised — exactly 0.50 at S=8192 —
so the volume was never the problem. Applying the mask was, at 1.16x–1.34x. The
kernel built the causal predicate on *every* key block, although only the
diagonal block can be affected by it: every block below the diagonal is entirely
visible, so `offs_m >= n_idx` there is arithmetic with a uniformly true answer,
computed and applied through a `tl.where` for nothing.

**The fix ships.** Splitting the key loop into an unmasked range over
`[0, start_m * BLOCK_M)` and a masked range over the diagonal block is worth
1.08x–1.21x on five of six shapes (neutral at S=512, where there is barely a
below-diagonal range to save), and takes the causal ratio to 0.53–0.72. Every
candidate tiling has `BLOCK_N` dividing `BLOCK_M`, so the boundary falls on a
block edge and the first range contains only whole, fully-visible blocks. It is
bit-identical — dropping a `where` whose predicate is uniformly true removes no
arithmetic that affects a result — verified on every shape and under key padding
with causal both on and off, and pinned in `tests/test_kernels.py`.
`results/causal_residual_sm_80.json` records both steps.

The remainder is the masking still genuinely needed on the diagonal block, plus
the lower arithmetic intensity of the smaller tile causal masking prefers. We
have no measurement isolating a further recoverable part, which is the honest
place to stop.

Accuracy is not a factor here: a `tl.exp` variant of the kernel measures the same
envelope as the shipped `exp2` one at every length tested. H100 utilization is lower because
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
python -m pytest tests/ -q          # 155 tests
python -m kernelforge.cli sweep     # search + three-way benchmark
python -m kernelforge.cli verify --demote
python scripts/report.py && python scripts/impact.py
```

Bringing up a new GPU is one `sweep` invocation with no code changes: tile
legality is derived from the measured shared-memory budget and precision from a
per-stage error budget measured on that target.
