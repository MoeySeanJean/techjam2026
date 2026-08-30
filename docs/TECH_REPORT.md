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

### The host CPU is part of the measurement, and we got caught by it

A cluster job gets an exclusive GPU but *shares* the host CPU. Because the
baseline issues ~105 kernel launches per forward and our fastest plans capture a
CUDA graph, a slower host inflates the baseline far more than it inflates us —
which inflates the speedup ratio.

This is not hypothetical. Two A100-80 runs, same GPU model, same code:

| node | µs per kernel launch | GPU time, `default` | baseline, `default` | our reported speedup |
|---|---|---|---|---|
| `xgph1` | 4.6 – 5.6 | 3.02 ms | 2.76 ms | 2.76x |
| `xgpj0` | **9.9 – 11.8** | 3.02 ms | **7.31 ms** | **10.89x** |

The GPU times are identical to three significant figures — it is the same card
doing the same work. `xgpj0` simply has a host roughly 2.2x slower at submitting
kernels, and that alone moved the headline number by 4x.

We caught it because the profiler records launch overhead per run, and the
discrepancy was too large to be real. The A100 figures in this report are from
`xgph`-class nodes; `SLURMD_NODENAME` and the measured launch overhead are now
recorded in every sweep artifact so two runs can be checked for comparability
before they are put side by side.

The general lesson is the one this project keeps relearning: **a speedup is a
property of a measurement setup, not of a kernel.** Interleaved A/B ordering
protects against drift *within* a run; nothing protects against comparing two
runs from different machines except recording which machine each came from.

### A measurement caveat we want stated plainly

Both nodes hold stable clocks under sustained load, which is the property that
makes their timings comparable across runs. We do not report any figure measured
on hardware that throttles: a power- or thermally limited GPU can settle far
below its rated clock, and we have watched the same shape measure 16.9x, 36.5x
and 45.6x across three runs on such a machine. Those numbers are not wrong so
much as meaningless, and none of them appear here.

---

## 2. The problem, as the script actually defines it

We read `torch_transformer_benchmark.py` rather than working from the prose.
Three properties of it shaped every decision that follows.

**It evaluates a 6-layer stack, not a single layer.** Any perturbation is
amplified as it propagates — measured at roughly 10³ end to end.

**The accuracy rule is an OR, not an AND:**

```python
passed_mask = finite_mask & (abs_ok | rel_ok)
```

so the per-element allowance is `max(atol, rtol·|ref|)`. We collapse that to one
number, *envelope utilization* = `max(abs_err / allowance)`, where <1.0 passes.
Script defaults are `atol=2e-3, rtol=2e-2` since the 27 August 2026 revision,
matching the problem statement's "relative error < 0.02, abs error < 0.002".
They were half that when we started, and we did not go back and re-loosen
anything, so most of this table was searched under a gate twice as tight as the
one now in force. We read both constants out of `parse_args` at import time
rather than hardcoding them, so the gate tracks the script.

**In float32 the reference already runs TF32.** `--allow-tf32` defaults true and
`matmul_precision='high'`. TF32 carries 10 mantissa bits; fp16 carries 11.
Computing in fp16 is therefore not a precision regression against *this*
reference — a fact the error budget later confirmed quantitatively.

Accuracy is a hard gate (`return 2` skips benchmarking entirely), so correctness
is not a constraint to satisfy at the end. It is the thing the whole design is
organized around.

---

## 3. What we built

Not a hand-tuned kernel — a measurement-driven optimization loop, plus the
kernels it produced.

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

**At run time nothing clever happens.** `submission.py` looks up the frozen table
and runs the named plan: no LLM call, no autotuning stall, no nondeterminism, and
a fallback to the bit-exact path if anything at all goes wrong.

### The kernels

- **Fused `add + mask + LayerNorm` per block boundary** (`ops/layernorm.py`) —
  writes the new residual stream and the next sublayer's normalized input in one
  pass, reading the *next* layer's norm weights, so a block boundary costs one
  kernel instead of four.
- **FlashAttention with native causal + key-padding** (`ops/flash.py`) — fp32
  online softmax, `exp2` with a folded `log2(e)` scale. `F.scaled_dot_product_attention`
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

### Ingesting shapes we have never seen

The brief says the official shape combinations will be told to the participants.
They had not been at build time, so the system treats an unseen shape as a
first-class input rather than an edge case:

```bash
python -m kernelforge.cli tune --shapes-file official_shapes.txt
```

A shape spec is parsed, classified into a bottleneck regime, searched,
benchmarked, frozen, and then the entire table is re-verified with demotion.
Verified on two shapes never in the matrix: `B4-S777-d640-H10-F2560-L9` found
`fp16[attn]+graph` at 1.99x, and `B12-S384-d768-H12-F3072-L6-causal` found
nothing that cleared the margin and correctly shipped the bit-exact plan at
1.02x rather than guessing.

### Shape dispatch

The problem statement invites shape specialization. Ours is keyed on
`(architecture, I/O dtype, B, S, d_model, heads, ffn, layers, causal)`, with
fallback to the nearest same-regime entry and then to a dtype-level default.
`num_layers` is part of the key because amplification grows with depth: the same
plan measures 0.55 envelope at L=6 and 0.98 at L=12.

---

## 4. Optimizations, and what each one cost in accuracy

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

Full tables in [../RESULTS.md](RESULTS.md); energy and fleet analysis in
section 6. Every row cleared the accuracy gate at a 0.80 margin over three seeds
**before** it was timed, and the frozen table was then re-measured end to end
with `cli verify --demote` on each machine — **no entry was demoted** on either
cluster GPU.

### The official 14 shapes (Appendix 3.7)

Both cluster nodes, every shape they can run. We report only hardware that
holds a stable clock, for a reason worth stating: on a throttling GPU our
measured ratios came out *higher*, not lower — a weaker card spends
proportionally more of its runtime on the kernel-launch overhead we remove, so
the speedup rises while every absolute latency gets worse. Reporting such a
machine would have flattered us for a reason that has nothing to do with the
kernels.

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

Envelope utilization is `max(abs_err / max(atol, rtol·|ref|))`, so 1.0 is the
failure line. The two envelope rows differ because they are different
measurements: the first is what a plan scored when it was selected, the second is
what the same frozen plan scores re-measured later in a fresh process.
Re-measurement moves it by up to ~0.1 as cuBLAS picks different kernels for the
same call, which is why admission is gated at 0.80 and re-verification permits up
to 0.90. The worst number we have ever seen from a shipped entry is 0.847.

The shape missing from both columns is #14, which no reference can run and which
gets its own section below. All 13 others run on both nodes, including shape 6
(`B10000`), whose baseline needs more memory than a small card has.

Through the organizer's own script, unmodified, on official shape 1 (A100 node
`xgph0`):

```
=== Accuracy check ===
criterion: abs_error <= 0.002 OR relative_error <= 2.00%
summary: PASS | max_abs=0.00100768 | max_rel=18461.6 | failed=0/5242880
baseline : median=1.9528 ms | throughput=4195070.86 token/s
optimized: median=0.8335 ms | throughput=9828009.49 token/s
speedup  : 2.343x based on median latency
```

That 2.343x is lower than the 3.22x in the table above for the same shape, and
the difference is worth explaining rather than hiding: the table comes from our
own harness, which interleaves candidates round-robin over many more repeats and
reports a median; the organizer's script times each implementation in one block
with its own defaults, on a node we do not pin. Both are honest measurements of
the same kernels. **The organizer's number is the one a judge will reproduce**,
so it is the one to hold us to; ours is the more carefully controlled comparison,
which is why we use it to *choose* between plans.

### Shape 14: the shape the reference cannot run

`B32-S100000-d1024-H16-F1024-L2-causal`. The baseline forms `[B,H,S,S]` before
its softmax — 18.6 TB. We run it in **77.7 s on the A100-80** and **54.5 s on the
H100 NVL**, at 45.9 GB peak, with finite output of the correct shape.

We quote no speedup for it. A ratio against an implementation that cannot run is
not a measurement; the claim is that the shape is reachable with a fused kernel
and unreachable without one. Getting there needed three things, and the third
was our own bug: the fp32 SDPA fallback was building an `[S,S]` causal mask —
37.25 GiB — because SDPA accepts `is_causal` or an `attn_mask` but not both. Our
fallback was reintroducing the quadratic term the flash kernel exists to remove.
Dropping a padding mask that marks nothing invalid took peak memory from 84.6 GB
to 45.9 GB. We found it by printing the failing allocation rather than reasoning
about it, after two wrong hypotheses.

### How much does the hardware actually change the answer?

**4 of the 13 official shapes chose a semantically different plan** on the A100
than on the H100 — a different set of fp16 stages, or CUDA graphs on one card and
not the other. Two generations apart, on the same vendor, with the same driver
stack.

We state this carefully because our own first count was wrong. Plan *names* list
stages in admission order, so `fp16[attn,ffn1,ffn2,out_proj]` and
`fp16[ffn1,ffn2,attn,out_proj]` read as different plans while being the same
plan. Comparing names said every shape diverged; comparing the plan
specification says 4 of 13. The smaller number is the true one.

Four in thirteen is still the argument for searching rather than hand-tuning — a
fixed kernel set is wrong for roughly a third of this list on a two-card fleet —
but it is a smaller effect than we expected, and the reason is visible in the
list itself: eleven of the fourteen official shapes are `S=128` variants of one
another, far more homogeneous than a real serving mix. On the wider matrix we
tuned against before the list was published, divergence was higher.

### Where we lose, stated plainly

**On the cluster GPUs, nowhere.** All 26 measurements are wins over both
references. That is a stronger claim than we expected to be able to make, so here
is what qualifies it.

- **Shape 1 on the A100 is 1.02x over `torch.compile`.** That is parity, not a
  win, and we would not argue otherwise. Shape 2 (1.06x) and shape 5 (1.07x) are
  nearly as close.
- **Long causal attention on a small GPU is our weakest regime.** Shape 13
  (`B64-S1024`) goes 4.26x and 4.11x our way on the A100 and H100, but on a
  46-SM card we have measured it at 0.94x of `torch.compile` — inductor wins
  there even while reporting *"Not enough SMs to use max_autotune_gemm."* That
  result is not in our reported set, because that machine's timings are not
  trustworthy; we state it because the weakness is real and a reader on a small
  card should expect it.
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

**Model selection was measured, not assumed.** `scripts/pick_model.py` scores
every model the key can reach on parse rate, field validity, shared-memory
legality of proposed tiles, constraint compliance and latency. On our gateway,
ten models were reachable and the result was mostly a tie:

| model | valid JSON | usable proposal | median latency |
|---|---|---|---|
| **`ornith1.5:35b`** | 100% | **100%** | **1.41 s** |
| `ornith1.0:35b` | 100% | 100% | 1.51 s |
| `qwen3-coder-next` | 100% | 100% | 2.02 s |
| `qwen3.6:27b` | 100% | 100% | 3.28 s |
| `qwen3.5:9b` | 100% | 100% | 4.05 s |
| `gemma4:26b` | 100% | 100% | 4.07 s |
| `qwen3.6:35b` | 88% | 88% | 2.58 s |
| `llama3.1:8b` | 100% | 75% | 2.77 s |
| `qwen3.8:27b` | 100% | 75% | 3.17 s |

Six of ten tie at 100%, so on this benchmark latency is the only thing
separating them, and it recommends `ornith1.5:35b`. Evidence in
`results/model_selection.json`.

**We do not ship that model, and the reason is the most useful thing this
section contains.** Emitting a valid plan as JSON and writing correct Triton are
different tasks, and we did not assume the first predicts the second. We ran
every reachable model through the kernel-generation loop on the A100 — 20 kernels
each, same two targets, same gate.

Two corrections had to happen first, and both are worth more than the ranking.

**The gateway aliases model ids.** Ten advertised ids resolve to **seven distinct
models**: `qwen3.6:27b` is served by `qwen3.8:27b`, `ornith1.0:35b` by
`ornith1.5:35b`, and `default` by `qwen3.6:35b`. We found it probing for
rate-limit headers, and it meant our first winner was labelled with the wrong
name and one model had been measured twice under two names. The mapping is in
`results/model_aliases.json` and every artifact now records the served id.

**A rate limit was recorded as a model's score.** Three arms returned `HTTP 429`
on every request after earlier arms drained the key's quota; our client gave up
after four retries in ~14 seconds and the taxonomy showed them as producing
nothing. That is an infrastructure limit written down as capability. The client
now backs off eight times to a two-minute ceiling and honours `Retry-After`,
unanswered requests are recorded as `api_error`, and the three arms were re-run
once the quota refilled.

| model (as served) | valid plan JSON | correct kernels |
|---|---|---|
| **`qwen3.8:27b`** | 75% | **16/20 (80%)**, 15/20 on a second sample |
| `qwen3-coder-next` | 100% | 8/20 (40%) |
| `qwen3.6:35b` | 88% | 6/20 (30%) |
| `gemma4:26b` | 100% | 5/20 (25%) |
| `ornith1.5:35b` | 100% | 0/20, 3/20 on a second sample |
| `qwen3.5:9b` | 100% | 0/20 — 17 syntax errors |
| `llama3.1:8b` | 75% | 0/20 |

The proposal benchmark is **anti-predictive** here: it ranked `ornith1.5:35b`
first (100% JSON, fastest) and that model wrote zero working kernels, while
`qwen3.8:27b` — which it ranked *last* on format at 75% — wrote the most. We ship
**`qwen3.8:27b`**.

The accidental double-measurements also bound the noise: 15 vs 16 of 20 for the
winner, 0 vs 3 for `ornith1.5:35b`. The first-to-second gap is far larger than
that; the gaps among the bottom three are not, and we do not defend their order.
Full table and failure taxonomy in [CODEGEN.md](CODEGEN.md).

### Does the LLM actually optimize *for the hardware*?

This is the claim the track is really asking about — "generate more efficient
implementations for **specific GPU hardware**" — so it needs evidence, not an
assertion that we put a spec sheet in the prompt.

We ran the same four official shapes through the same loop on an A100 and an
H100, changing nothing but the machine. The proposer sees that machine's spec
sheet (SM count, shared memory per block, bandwidth, tensor-core support) and its
measured bottleneck profile. Comparing the *configuration* that won on each card
— not the plan name, which the model writes itself and which would overstate the
difference:

| proposer | shapes with a different winning configuration |
|---|---|
| **LLM** (`qwen3.8:27b`) | **3 of 4** |
| heuristic | 2 of 4 |

The direction of the difference is consistent and physically sensible. On the
H100 the LLM moved to `compute=float16` on three of four shapes while staying at
`float32` on the A100, and it picked larger flash tiles — the H100 has 227 KB of
shared memory per block against the A100's 163 KB, which is stated in the spec
sheet it was given. On the fourth shape the two cards agreed on precision and
structure and differed only in tile size, `(128, 64, 8, 3)` against
`(128, 128, 8, 3)`.

That is the innovation claim, and it is narrow on purpose: **the LLM is
re-deciding per architecture rather than emitting one plan and relabelling it**,
and it diverges more than our hand-written heuristic does, which narrows stages
in a fixed order and cannot jump. Every one of those proposals still passed the
same accuracy gate before it was timed; the model is allowed to be wrong, and the
gate is not.

The kernel-writing half replicates across the two cards too — same model, same
targets, independent runs:

| model | A100 (sm_80) | H100 (sm_90) |
|---|---|---|
| `qwen3.8:27b` | 16/20 | **17/20** |
| `qwen3-coder-next` | 8/20 | 9/20 |

Four independent samples of the winner (15, 16, 17 of 20, counting the run served
under the `qwen3.6:27b` alias) put the ranking well outside the noise we measured
on repeat arms.

**Heuristic vs LLM, same GPU, same shapes, same gate:**

| proposer | proposals | cleared gate | API failures | wall clock |
|---|---|---|---|---|
| heuristic | 40 | 33 (82%) | 0 | 370s |
| LLM | 33 | 28 (85%) | 1 | 297s |

The LLM found the fastest gate-passing plan on 3 of 4 shapes. It is not better
at what the heuristic already encodes — it re-proposed bfloat16 compute and an
fp16 residual stream, both already ruled out by the error budget, and the gate
rejected them. Where it helped was reaching combinations the hand-written
cheapest-stage-first ordering cannot jump to.

### AI-generated kernel *source*

Configuration search is not what the track means by "AI-based code generation",
so we also had the model write complete Triton kernels against a contract, with
every candidate compiled, gated and timed by the same harness. **24 generated
kernels:**

| outcome | share | | target | success | best generated kernel |
|---|---|---|---|---|---|
| ok | 37.5% | | `gelu` | 4/12 | **5.49x** vs torch, envelope 0.061 |
| `compile_error` | 29.2% | | `layernorm` | 5/12 | **2.80x** vs torch, envelope 0.077 |
| `numeric_fail` | 25.0% | | | | |
| `syntax_error` / `triton_global_not_constexpr` | 8.4% | | | | |

Three findings, detailed in [CODEGEN.md](CODEGEN.md):

- **Silent wrongness dominates.** The worst failure did not crash: the model
  tiled a LayerNorm across the feature axis and computed mean/variance per
  64-column *tile* instead of per row. It compiles, runs at a plausible speed,
  and is wrong by an envelope of 4525. Only the gate caught it.
- **It was as much a specification gap as a model failure.** Adding one sentence
  forbidding a split of the reduction axis moved that target from **0/5 to
  5/12**. Writing a precise contract is the actual skill; the harness is what
  made the gap visible.
- **Process isolation is mandatory, not tidy.** An out-of-bounds generated
  kernel corrupts the CUDA context — asynchronous, uncatchable, and fatal to
  every later CUDA call in the process. Each candidate is validated in a
  throwaway subprocess.

We also built the obvious improvement — feeding compiler errors and a
structural diagnosis of the numeric failure back as a repair prompt — and
measured it twice, **at equal attempt budget, and got opposite answers**. On the
A100 with `qwen3-coder-next`: 8/20 correct with pure resampling against 12/20
with repair. In an earlier run on different hardware: 13/28 against 12/28, the
other way. At twenty to thirty attempts per arm neither run resolves a gap that size.

We leave `--repair 0` as the default — not because the measurement says so, but
because flipping a default on one n=20 run that contradicts the previous n=28
run is the reasoning this project exists to avoid. Full write-up in
[CODEGEN.md](CODEGEN.md); both arms are committed.

**Nothing generated is in the shipped dispatch table.** They are proposals; a
public submission should not contain code no person has read. What the loop
demonstrates is that the harness — contract, gate, isolation, taxonomy — is what
makes AI-written kernels usable at all.

### Roofline: what is left

| GPU | GEMM-bound shapes | `tiny`/`decode` | long causal |
|---|---|---|---|
| sm_86 | 46–58% of tensor-core ceiling | bandwidth-bound, low intensity | **17%** |
| sm_80 | 31–57% | bandwidth-bound | **17%** |
| sm_90 | 25–45% | bandwidth-bound | **8%** |

`tiny` and `decode` are not underusing the machine — there is barely any
arithmetic to do, which is why their wins come from removing launches. Long
causal attention is the genuine headroom and is exactly where we still fall back
to a library implementation. H100 utilization is uniformly lower because the
machine is larger than our tiles saturate; closing that means Hopper-specific
work (TMA, wgmma) we scoped out.

**The skill that mattered most was not prompting.** It was building the gate.
An LLM proposing kernel configurations is only useful if something independent
and trustworthy decides whether each one is correct — and that has to happen
*before* anything is timed, because a wrong-but-fast configuration is exactly
what a latency-ranked search will promote. A proposer is allowed to be wrong.
The gate is not.

### A near-miss that justifies the design

We measured `torch.compile` on our bit-exact rewrite at fp16 and got **0.000
envelope** with a 2.98x speedup — apparently a real finding, and confirmed as a
genuine compilation rather than a fallback. **It does not reproduce**: four
re-measurements in fresh processes gave 2.655 / 2.502 / 3.296 / 2.853, all
failures. Inductor's autotuning had happened to pick a kernel set whose rounding
matched, and that is not stable across processes.

The full sweep rejected the configuration automatically, with no special-casing.
Had we trusted the first number, we would have shipped a plan that fails roughly
three times in four. This is why the gate runs over multiple seeds, why the
sweep re-measures, and why `cli verify --demote` re-checks the frozen table
afterwards.

---

## 8. Reproducing

```bash
cp .env.example .env            # optional: only the LLM proposer needs it
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install triton              # triton-windows on Windows
python -m kernelforge.cli doctor    # environment + Triton check
python -m pytest tests/ -q          # 129 tests
python -m kernelforge.cli sweep     # search + three-way benchmark
python -m kernelforge.cli verify --demote
python scripts/report.py && python scripts/impact.py
```

Bringing up a new GPU is one `sweep` invocation with no code changes: tile
legality is derived from the measured shared-memory budget and precision from a
per-stage error budget measured on that target.
