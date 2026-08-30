# KernelForge

**TikTok TechJam 2026 — Track 3: Implement a GPU Kernel for a Transformer Layer**

We did not set out to hand-tune one fast kernel. We built the thing that
*produces* fast kernels: a loop that profiles a workload, **proposes a
configuration — from a hand-written heuristic or from an LLM**, proves it
numerically correct against the organizer's own tolerance rule, times it
honestly, and freezes the winner into a per-shape dispatch table. The kernels it
produced are the submission; the loop is the contribution.

The two proposers are the point, not a feature list. The heuristic encodes what
we already understood about this stack and searches it exhaustively; the LLM is
given the same hardware spec sheet, the same measured profile and the same error
budget, and is free to propose combinations the heuristic's fixed ordering can
never reach. Both are held to exactly the same numeric gate, so neither can win
by being wrong.

**The LLM optimizes for the specific GPU, and we measured that rather than
assuming it.** Same four shapes, same loop, changing only the machine: the LLM
chose a different winning *configuration* on the A100 than on the H100 for **3 of
4 shapes** — moving to fp16 compute and larger flash tiles on the card with 227 KB
of shared memory per block instead of 163 KB. Our hand-written heuristic diverged
on 2 of 4.

We also let the model write complete Triton source, gated every kernel it wrote,
and ran nine model ids through that loop to pick the one we ship. **`results/`
records what each proposer suggested, what cleared the gate, and what did not** —
including the kernel that replaced exact `erf` with a polynomial approximation
and left a comment explaining why that was fine. It was not.

## Project overview

The organizer's `torch_transformer_benchmark.py` runs a 6-layer Transformer
stack and compares a user implementation against a naive PyTorch baseline on
both accuracy and latency. `submission.py` supplies our
`UserOptimizedTransformer`.

What it does at run time is deliberately boring: look up the current GPU
architecture, I/O dtype and shape in a frozen table, instantiate the plan that
table names, and run it. No LLM call, no autotuning stall, no nondeterminism. All
of the search happened offline and is reproducible from the artifacts in
`results/`.

### Scope

The problem statement puts *AI-based code generation, GPU kernel fusion and
profiling-tool usage* in scope, and *production-ready deployment* out of it. This
repository is entirely the former. Everything here is a **build-time** artifact:
a profiler, a search loop, a correctness gate, generated kernels, and a frozen
lookup table. At run time `submission.py` does a dictionary lookup and calls a
kernel.

Nothing here is a serving system. [docs/USE_CASE.md](docs/USE_CASE.md) measures
what the speedups would be *worth* to a ranking workload, because relevance to a
real stakeholder is part of what is being judged — but it is an analysis with its
assumptions written down, not a deployment, and it says so.

**Start here**, depending on what you want:

| you want to... | read |
|---|---|
| **show it** to someone in 3 minutes | `python scripts/showcase.py` — five narrated acts |
| read the full technical report | [docs/TECH_REPORT.md](docs/TECH_REPORT.md) |
| see the numbers | [RESULTS.md](RESULTS.md) |
| see why it matters | [docs/USE_CASE.md](docs/USE_CASE.md) — recommendation ranking, measured |

```bash
pip install -r requirements.txt
python -m kernelforge.cli doctor     # check CUDA + Triton
python scripts/showcase.py           # guided 3-minute walkthrough
```

### The three things we would want a reviewer to look at

1. **[docs/PRECISION.md](docs/PRECISION.md)** — the measurement that reframed the
   problem. The tolerance is an OR (`abs_err <= atol` **or** `abs_err <=
   rtol*|ref|`), and the stack amplifies any perturbation by roughly 10³, so
   "how close am I to failing" is a number worth computing rather than a
   yes/no. We compute it for every candidate before timing it.

   The sharp version: **at `--dtype float16` and `--dtype bfloat16` the accuracy
   gate rejects `torch.compile(max-autotune)` applied to the organizer's own
   baseline** — envelope 2.87 and 21.00 on the A100, 2.71 and 19.62 on the
   H100, against a limit of 1.0. All 14 official
   shapes are float32, where `torch.compile` is admissible and we beat it
   anyway; but the dtype is a command-line flag, and a submission that assumed
   rather than measured would ship silent wrongness the first time anyone
   changed it.

2. **[docs/EQUIVALENCE.md](docs/EQUIVALENCE.md)** — why our structural rewrite is
   bit-identical to the baseline, including the padding-mask ordering. The
   trap: `LayerNorm` of an all-zero row returns `bias`, not zero, so the final
   mask has to be applied *after* the norm.

3. **`results/genealogy_*.json` and `results/codegen*.json`** — every
   configuration each proposer suggested, why each rejected one was rejected,
   and what each surviving one measured. Plus the head-to-head: the heuristic
   and the LLM given identical evidence on identical shapes under an identical
   gate, so "what did the AI contribute" is answered with a measurement instead
   of an assertion. The answer is more interesting than "it wins" — see
   [docs/CODEGEN.md](docs/CODEGEN.md), which also documents a repair loop we
   built, measured, and found does not help.

## Results

The organizers published the test set on 27 August 2026 (Appendix 3.7): **14
shapes**, all `float32`, all causal. We tuned against exactly those.

Results are reported on the **A100-80 and H100 cluster nodes only**. The RTX
3070 Ti laptop was the development machine and its sweeps are still in
`results/`, but it is excluded from these tables — it cannot lock clocks
(throttling to 510 MHz of 1635 under load), and its ratios are *inflated*
rather than deflated: a weaker card spends proportionally more of its time on
the kernel-launch overhead we remove, so the same optimization scores higher
there while every absolute latency is roughly four times worse. Including it
would raise our medians for a reason that has nothing to do with the kernels.
`python scripts/report.py --all` puts it back.

| | A100-80 PCIe (sm_80) | H100 NVL (sm_90) |
|---|---|---|
| shapes measured | 13 of 14 | 13 of 14 |
| median vs the reference | **5.39x** | **7.35x** |
| range vs the reference | 2.32x – 15.25x | 2.34x – 13.29x |
| median vs `torch.compile` | 1.53x | 1.69x |
| worst vs `torch.compile` | 1.02x | 1.03x |
| **faster than both references** | **13 of 13** | **13 of 13** |
| passed the accuracy gate | **all** | **all** |
| demoted on re-verification | 0 | 0 |

**Every officially specified shape that can be run at all beats both the naive
baseline and `torch.compile(max-autotune)`, on both cluster GPUs — 26 of 26
measurements.** The narrowest margins are 2.32x over the reference and 1.02x
over `torch.compile`. `torch.compile` clears the accuracy gate on all 14 official
shapes, so every one of those comparisons is against an admissible opponent.

**The fourteenth shape is the exception, and not in our disfavour.**
`B32-S100000-d1024-H16-F1024-L2` would need the reference to allocate an 18.6 TB
attention score matrix, so the reference cannot run it and neither can
`torch.compile` applied to the reference. We run it in **77.7 s on the A100** and
**54.5 s on the H100**, at 45.9 GB peak, with finite output of the correct shape.
We quote no speedup for it — a ratio against something that cannot run is not a
measurement — but it is the shape that best shows why a fused kernel is not
merely an optimization here.

Full tables in **[RESULTS.md](RESULTS.md)**, raw artifacts in
`results/sweep_*.json`. Reproduce with
`python -m kernelforge.cli tune --shapes-file official_shapes.txt` followed by
`python scripts/report.py`.

## Setup and installation

Credentials live in a gitignored `.env` (copy `.env.example`). Nothing secret
appears in any tracked file:

```bash
cp .env.example .env    # then fill in SOCLAAS_API_KEY / NUS_PASS
```

```bash
# CUDA-capable GPU required. Verified on torch 2.11 + CUDA 12.6.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install triton            # Linux
pip install triton-windows    # Windows (needs MSVC build tools)
pip install pynvml            # optional, enables energy accounting

python -m kernelforge.cli doctor      # confirms Triton compiles on this GPU
```

`doctor` prints the hardware spec sheet, the shared-memory budget, which flash
tile configurations are legal on this card, and whether clock locking is
available.

## Running this on *your* hardware

**You have neither of the two things we used, and that is fine.** Our numbers
come from an NUS SoC Slurm cluster (A100-80 and H100 NVL) and an NUS-hosted
LLM gateway. You cannot log into either. Nothing in this repository requires
them — both are substitutions, and both are one file each.

| we used | you substitute | how |
|---|---|---|
| A100-80 / H100 on a Slurm cluster | **any CUDA GPU** | nothing to configure. `cli sweep` tunes for whatever card it finds. |
| NUS SoC LLM gateway | **any OpenAI-compatible endpoint**, or none | two lines in `.env`; or skip it, and the deterministic heuristic proposer runs instead. |
| — | **no GPU at all** | the committed artifacts and the whole dashboard still work. |

Start here, on any machine:

```bash
python -m kernelforge.cli doctor
```

It ends with a capability table telling you exactly what this machine can run
and what it will fall back to, so you never discover a missing dependency
halfway through a benchmark. Nothing below fails hard; every path degrades to a
correct, slower one and says so.

### "I have a different GPU"

The submission works immediately — correctness never depends on having a tuned
entry. Dispatch resolves in this order:

1. an exact entry for your `(architecture, dtype, shape)`
2. the conservative bit-exact plan, if we have measured that architecture but
   not that shape
3. an architecture-appropriate default

We ship tuned tables for **sm_80** (A100), **sm_90** (H100) and **sm_86**
(RTX 30-series). On anything else you get case 3, which is still meaningfully
faster than the baseline because CUDA-graph capture and the structural rewrite
are architecture-independent.

One caveat on `sm_86`, since you may well be on it: its entries were selected on
a laptop that cannot lock clocks and throttles to 510 MHz of 1635 under load, so
its *timings* were not trustworthy enough to report and no sm_86 result appears
in [RESULTS.md](RESULTS.md). We still ship the table because every entry in it
passed the accuracy gate, and gating does not depend on timing — so it is
correct, and almost certainly faster than the default, but its *ranking* may not
be optimal for your card. If you are on sm_86 and want the real answer, run the
sweep below; it takes under an hour.

**On Ampere or newer** (sm_80+: A100, RTX 30/40/50-series, H100, L4) the fp32
default uses fp16 for attention and `out_proj`. That is justified because the
reference itself runs its matmuls at TF32 — see [docs/PRECISION.md](docs/PRECISION.md).

**On Turing or Volta** (sm_75 / sm_70: T4, RTX 20-series, V100) there is no
TF32, so that reasoning does not hold and dispatch automatically falls back to
the **bit-exact** plan. Slower, but it cannot fail the accuracy gate. `doctor`
tells you when this applies.

To tune your card properly (~50 min, fully automatic):

```bash
python -m kernelforge.cli sweep          # searches, gates, benchmarks, freezes
python -m kernelforge.cli verify --demote
```

That writes `results/dispatch_sm_XX.json` for your architecture. Or tune just
the shapes you care about, in a couple of minutes each:

```bash
python -m kernelforge.cli tune --shapes B8-S128-d512-H8-F2048-L6
```

### "I don't have an LLM API key"

Everything works without one. The agent has three interchangeable proposers and
picks automatically:

| what you have | what runs |
|---|---|
| nothing | **`heuristic`** — deterministic search, no network. This is the default and it produced most of our shipped plans. |
| any OpenAI-compatible endpoint | the LLM proposer |
| `ANTHROPIC_API_KEY` | the Anthropic proposer |

```bash
python -m kernelforge.cli agent --provider heuristic    # always available
```

To plug in **any** OpenAI-compatible endpoint — OpenAI, a local Ollama, vLLM,
LM Studio, or your own gateway — set two variables in `.env` (copy
`.env.example`):

```bash
# OpenAI
SOCLAAS_BASE_URL=https://api.openai.com/v1
SOCLAAS_API_KEY=sk-...
SOCLAAS_MODEL=gpt-4o-mini

# Ollama running locally — no key needed, any placeholder works
SOCLAAS_BASE_URL=http://localhost:11434/v1
SOCLAAS_API_KEY=ollama
SOCLAAS_MODEL=qwen2.5-coder:7b

# vLLM / LM Studio / any other OpenAI-compatible server
SOCLAAS_BASE_URL=http://localhost:8000/v1
```

The client is plain `urllib` against `POST /v1/chat/completions` — no SDK to
install, and no vendor lock-in. `OPENAI_BASE_URL` / `OPENAI_API_KEY` are honoured
as aliases.

Only two things need a key: `cli agent --provider llm` and `cli codegen`. Their
*results* are committed in `results/`, so you can read what the model produced —
including the kernels it got wrong — without running anything. To reproduce them
on your own endpoint:

```bash
python -m kernelforge.cli agent --provider llm --tag llm --iterations 3
python -m kernelforge.cli codegen --targets layernorm,gelu --iterations 14
python scripts/report.py        # regenerates the proposer head-to-head
```

The LLM is **never on the inference path**. It proposes configurations and
writes kernel source at build time; everything it produces passes the same
accuracy gate as everything else, and anything that fails the gate is discarded
before it can be timed. A submission that ran without ever calling a model would
produce the same outputs, just from a smaller search.

### "I don't have a GPU at all"

```bash
python -m pytest tests/ -q                  # 40 of 129 pass, 88 skip cleanly (no GPU)
python scripts/showcase.py --no-gpu         # three acts, from committed artifacts
python scripts/report.py                    # regenerate RESULTS.md
open dashboard.html                         # the full interactive explorer
```

The dashboard and every number in `RESULTS.md` come from committed JSON, so the
results are inspectable without reproducing them.

### "Triton won't install"

Triton supplies our two hand-written kernels. Without it the code paths fall
back to `F.scaled_dot_product_attention` and `torch.layer_norm` automatically —
you lose the fused kernels, keep everything else, and `doctor` says so. The
structural rewrite, fused QKV, CUDA-graph capture and `torch.compile` plans all
still work.

On Windows use `pip install triton-windows` (needs MSVC build tools); on Linux
`pip install triton`.

### "I don't have the NUS cluster"

You don't need it, and it isn't here. The Slurm job files and SSH driver that
produced our A100 and H100 numbers run against one specific university cluster,
so they would be dead weight in this repository — they are kept out of it, and
every result they produced is committed as JSON in `results/`. On any single GPU,
`cli sweep` does the same work locally, and `scripts/README.md` lists the
portable equivalent of every job we ran.

## Reproducing our results

**The headline claim**, on any CUDA GPU — every official shape beats both the
naive baseline and `torch.compile`:

```bash
python -m kernelforge.cli tune --shapes-file official_shapes.txt   # ~40 min
python -m kernelforge.cli verify --shapes-file official_shapes.txt --demote
python scripts/report.py                    # regenerates RESULTS.md
```

**The single shape that matters most**, if you have 80 GB:

```bash
python scripts/shape14.py --scan            # how far your GPU gets
```

**Through the organizer's script directly** — it imports
`torch_transformer_benchmark.py` *unmodified* and substitutes
`UserOptimizedTransformer` from `submission.py`:

```bash
python scripts/run_official.py --batch-size 64 --seq-len 128     --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --causal
```

**The AI-assisted half.** Needs an LLM endpoint; see
"[I don't have an LLM API key](#i-dont-have-an-llm-api-key)" for how to point it
at yours, or skip it — the heuristic arm runs with no key at all.

```bash
python scripts/pick_model.py                    # score the models your key can reach
python -m kernelforge.cli agent --provider heuristic --tag heuristic
python -m kernelforge.cli agent --provider llm       --tag llm
python -m kernelforge.cli codegen               # the model writes Triton; all of it gated
```

**Everything else:**

```bash
python -m pytest tests/ -q            # 129 tests, no GPU needed for most
python -m kernelforge.cli doctor      # environment: CPU, GPU, disk, software
python -m kernelforge.cli budget      # per-stage precision error budget
python -m kernelforge.cli sweep       # the full matrix, not just the official shapes
python scripts/dashboard.py           # regenerate dashboard.html
python scripts/usecase.py --cold      # the use case, untuned; drop --cold once tuned
```

`scripts/README.md` says what every script is, and which of them need our
cluster (none that you need — those are provenance for our numbers, and each
has a portable equivalent listed).

## How it works

```
profile ─► propose ─► compile ─► GATE ─► measure ─► feed back ─► freeze
                                  │
                    numerically wrong candidates stop here,
                    before they can ever post a fast time
```

- **profile** (`kernelforge/agent/profile.py`) classifies the shape into a
  bottleneck regime — launch-bound, gemm-bound, attention-bound, bandwidth-bound —
  from measured kernel time, CPU submission time and launch count.
- **propose** (`kernelforge/agent/proposers.py`) emits a `Plan`. Three
  interchangeable implementations: a deterministic proposer that encodes what the
  measurements taught us, and two LLM proposers (any OpenAI-compatible gateway,
  or the Anthropic Messages API) that receive the hardware spec sheet, the
  profile, the measured per-stage error costs, the tile configurations that fit
  this GPU, and the full attempt history. The LLM client is built on `urllib`,
  so no extra dependency is needed to reproduce the loop.
- **gate** (`kernelforge/numerics.py`) is an exact replica of the script's
  comparison, run over multiple seeds. This is the load-bearing component: a
  proposer is allowed to be wrong, the gate is not.
- **measure** (`kernelforge/bench.py`) times candidates round-robin with rotating
  order, reports median and p90, and samples power where NVML is available.
- **freeze** (`kernelforge/dispatch.py`) writes the winner into a JSON table with
  its measured envelope utilization and speedup attached as evidence.

### What the kernels actually do

- **One fused `add + mask + LayerNorm` kernel per block boundary**
  (`kernelforge/ops/layernorm.py`), which writes the new residual stream and the
  next sublayer's normalized input in a single pass, reading the *next* layer's
  norm weights so a block boundary costs one kernel instead of four.
- **FlashAttention with native causal + key-padding support**
  (`kernelforge/ops/flash.py`), fp32 online softmax, `exp2` with a folded
  `log2(e)` scale. SDPA cannot take a causal flag and a padding mask together
  without materializing an `attn_mask` and falling off its flash backend; we
  handle both predicates in-register.
- **Zero head-split copies.** The flash kernel consumes strided views of the
  fused QKV buffer and writes into a `[B,S,H,Dh]` buffer whose flat view is
  already the merged-head layout. The baseline spends 4% of its GPU time on
  `Memcpy DtoD` from `.contiguous()`; we spend none.
- **One fused QKV GEMM** instead of three.
- **CUDA graph capture of the whole 6-layer stack.** A forward issues ~105 kernel
  launches at ~13 µs of CPU each on Windows WDDM; replaying a graph collapses
  that to one submission. This is where the small-shape speedups come from.

## Absorbing the official shape list

The problem statement says the official input-shape combinations "will be told
to the participants." They were not published when we built this, so our matrix
was an informed guess — the single largest risk to the submission. We handled it
as a first-class capability rather than a hope, and then the list arrived on
27 August and we got to find out whether that worked.

It did. Absorbing all 14 official shapes was one command per machine, no code
change, and the results are in [RESULTS.md](RESULTS.md). The same command is how
you would point it at *your* shapes:

```bash
python -m kernelforge.cli tune --shapes B4-S777-d640-H10-F2560-L9
python -m kernelforge.cli tune --shapes-file official_shapes.txt
```

Any shape not in the built-in matrix is parsed from its signature — with
optional `-causal`, `-pad0.4`, `-scale64`, `-float16` suffixes in any order —
searched, benchmarked three ways, frozen into the dispatch table, and then the
*whole* table is re-verified and anything that drifted is demoted. No code
change, one command.

Demonstrated on two shapes the system had never seen:

| shape | plan found | result |
|---|---|---|
| `B4-S777-d640-H10-F2560-L9` | `fp16[attn]+graph` | **1.99x** vs baseline, 1.60x vs `torch.compile` |
| `B12-S384-d768-H12-F3072-L6-causal` | `safe(exact)` | 1.02x — correctly declined to risk a causal shape |

The second one is the more interesting result: the search found nothing that
cleared the gate with margin, so it shipped the bit-exact plan rather than a
fast guess. That is the system behaving correctly on a shape it has no
experience with.

`official_shapes.txt` is a placeholder in the repo — replace it with the
published list and run one command.

## Two safeguards worth knowing about

**Envelope utilization is not perfectly reproducible.** The same (case, plan)
pair, same seeds, re-measures within roughly ±0.1 between runs — cuBLAS kernel
selection varies with device state. A plan admitted at 0.79 can re-measure at
0.89. This is why the admission margin is 0.80 rather than 1.0, and why
`cli verify --demote` exists: it re-measures the frozen table and replaces
anything that has drifted above **0.90** with the bit-exact plan.

Those two numbers must differ by more than the noise. We briefly set the
demotion threshold equal to the admission margin and it behaved as a ratchet —
each pass re-rolled the variance, demoted another entry that was actually fine,
and a few passes reduced the whole table to bit-exact for no safety gain. The
thresholds are deliberately 0.80 and 0.90.

**Data variants share a model signature.** `default`, `default_pad` and
`default_scale` differ only in `padding_ratio` and `--input-scale`, neither of
which appears in the model config that dispatch sees. One plan must therefore
serve all of them, so collisions resolve to the *less aggressive* plan rather
than the faster one. That costs speed — the shipped `default` entry is
`fp16[attn]` rather than the faster `fp16[attn,out_proj,ffn2]+graph` the
unpadded case alone would have chosen. We consider that the right trade: the
accuracy check is a hard `return 2`.

## Limitations and what we would do next

- **We could not lock GPU clocks** (`nvidia-smi -lgc` needs elevation on this
  machine), so laptop timings carry thermal drift. We mitigate with interleaved
  A/B ordering and report p90/min spread alongside every median so the reader can
  see how much of a number is machine noise. Cluster numbers do not have this
  problem.
- **`torch.compile`'s GEMM autotuning is disabled on this GPU** — inductor
  reports *"Not enough SMs to use max_autotune_gemm"* at 46 SMs. On an A100 (108
  SMs) or H100 (132 SMs) the compile baseline will be stronger, so our margin
  over it should be expected to narrow.
- **We lose to `torch.compile` on the `decode` shape** (B32, S=1): 8.24x vs the
  naive baseline but 0.71x vs compile. At S=1 the attention kernel does no useful
  work and our fused norm consumes almost the whole error budget (structural
  envelope 0.985), forcing the bit-exact plan. A dedicated S=1 path is the
  obvious next kernel.
- **We built against a shape list we did not have.** For most of the build the
  script published no shapes — they arrive as CLI arguments — so our matrix was a
  superset partitioned by bottleneck regime, with nearest-neighbour fallback and
  a safe default beneath it. The official 14 shapes were published on 27 August
  (Appendix 3.7) and are now in `official_shapes.txt`; specializing to them took
  one `tune --shapes-file` run and no code change, which is the outcome that
  design was for. The fallback path still matters and is still tested, because a
  judge running a shape we did not tune is the normal case, not the exception.
- **`num_layers` is a first-class dispatch key.** Amplification grows with depth:
  the same plan measures 0.55 envelope at L=6 and 0.98 at L=12. Plans validated
  at one depth must not be reused at a greater one.
- **The conservative collision rule leaves speed on the table.** A better design
  would, after the sweep, re-test the *fastest* candidate for a signature against
  every data variant sharing it and keep the fastest that passes them all,
  instead of falling back to the least aggressive plan measured on any one
  variant. That needs one extra validation pass per signature; we ran out of time
  before adding it.
- **Long causal attention is still our weakest regime, and it costs us one row.**
  Official shape 13 (`B64-S1024`) on the laptop lands at 0.94x of
  `torch.compile` — the only measured shape on any GPU where we lose. The same
  shape goes 4.26x and 4.11x our way on the A100 and H100, so this is a
  small-GPU weakness rather than a kernel that is wrong, but it is a loss and it
  is in the table. We suspect the `exp2` softmax substitution and the online
  rescaling accumulate differently from the reference's full-row `torch.softmax`
  as the row length grows; a variant using `tl.exp` directly is the next
  experiment.
- **Shape 14 runs, but slowly, and we cannot prove its accuracy at full size.**
  77.7 s on an A100 is a real answer where the reference has none, but there is
  no reference output at `S=100000` to check it against — nothing can compute
  one. We verify the same code path against an exact reference at every sequence
  length that *does* fit, and we check that slicing the batch does not change the
  answer, which is the strongest statement available. It is not the same as a
  measured envelope at full size, and we do not present it as one.
- **The fp32 attention fallback is the least optimized path we ship.** When a
  shape's attention stage stays in fp32 we fall through to SDPA, because
  Triton's `tl.dot` needs a narrow float type. That path is correct and now
  memory-safe, but it is where shape 14's 77 s goes. An fp32 flash kernel using
  three-pass split-K, or a bf16 path with a verified error budget at that
  length, is the obvious next kernel.

## Testing

```bash
python -m pytest tests/ -q      # 129 tests
```

The suite targets the specific traps this project hit, not just happy paths:

- `test_numerics.py` — the gate is checked **against the organizer's own
  `compare_outputs`** on random data, not against a restatement of the rule, so
  it fails if we ever drift from it. Also asserts the rule is an OR and that it
  is genuinely stricter than `torch.isclose`.
- `test_kernels.py` — both Triton kernels against exact references across
  causal × padding × non-multiple-of-block lengths; the all-zero-row LayerNorm
  trap; the fully-masked softmax row; strided-view consumption; and that tile
  legality actually tracks shared memory across sm_86/80/90.
- `test_equivalence.py` — the structural rewrite is **bit-identical** on every
  dtype; weight sharing does not perturb the reference; a CUDA-graph result
  survives a later replay; and the submission entry point falls back rather than
  raising when dispatch fails.
- `test_dispatch.py` — the collision policy, which is where a subtle mistake
  would silently ship a plan tuned on the wrong data variant.
- `test_streaming.py` — that slicing the batch does not change the answer, and
  that the estimator which decides *when* to slice fires on official shape 14
  and stays asleep on the shapes a real run uses. Both directions are pinned,
  because a heuristic that fires when it should not is as bad as one that does
  not fire when it should.
- `test_submission.py` — the entry point a judge actually runs: that it stays a
  `BaselineTransformer` subclass so `strict=True` weight copying works, and that
  a fused path which raises falls back to the *reference answer* rather than a
  traceback — announced, not silently.
- `test_secrets.py` — including a scan asserting no tracked file contains a
  secret-shaped string.

## Team member contributions

Solo entry — all design, implementation, measurement and writing by the
submitting author, using Claude Code (Claude Opus 5) as the development
environment throughout. *(If submitting as a team, replace this section.)*

## Repository layout

```
submission.py               competition entry point (UserOptimizedTransformer)
torch_transformer_benchmark.py   organizer-provided, unmodified
official_shapes.txt         the 14 test shapes from Appendix 3.7
DEVPOST.md                  the written project description (deliverable 3.5.1)
RESULTS.md                  every official shape on every GPU, three ways
dashboard.html              interactive decision explorer (open it)

kernelforge/
  numerics.py               exact replica of the accuracy gate
  budget.py                 per-stage precision error budget
  optimized.py              the fused stack, parameterized by Plan
  ops/flash.py              FlashAttention with causal + padding (Triton)
  ops/layernorm.py          fused add + mask + LayerNorm (Triton)
  search.py                 budget-guided plan search
  bench.py                  interleaved three-way timing harness
  dispatch.py               the frozen per-(arch, dtype, shape) table
  shapes.py                 shape parsing and the regime matrix
  hw.py                     hardware probe and LLM spec sheet
  roofline.py               achieved TFLOP/s and % of hardware ceiling
  env.py                    machine description for the tech report
  secrets.py                gitignored .env loader, no dependencies
  agent/                    profile -> propose -> gate -> measure -> freeze
    proposers.py            heuristic and LLM proposers, one interface
    codegen.py              the model writes Triton; every kernel is gated
    loop.py, profile.py     the loop and the bottleneck profiler

scripts/                    see scripts/README.md for what each one does
tests/                      129 tests; see Testing above
docs/TECH_REPORT.md         the full technical report -- start here
docs/CODEGEN.md             AI-written kernels, the model bake-off, failure taxonomy
docs/USE_CASE.md            recommendation ranking: the impact case
docs/PRECISION.md           what we measured about the tolerance
docs/EQUIVALENCE.md         why the rewrite is bit-exact
results/                    every artifact behind every number above
```

## Development tools, libraries and data

- **Tools:** Claude Code (Claude Opus 5) for AI-assisted kernel development,
  VS Code, `nvidia-smi`, PyTorch profiler, Nsight Compute.
- **Libraries:** PyTorch 2.11 (+cu126), Triton 3.7.1 (`triton-windows` on
  Windows), NumPy, `pynvml` for energy sampling.
- **APIs:** NUS SoC LLM-as-a-Service (OpenAI-compatible gateway) for the LLM
  proposer; the Anthropic Messages API is supported by the same interface. Used
  at build time only — no LLM is on the inference path.
- **Datasets:** none. All inputs are generated by the organizer's
  `generate_random_case`, which we call directly so our gate sees exactly the
  distribution the benchmark uses, including padding and input scaling.
