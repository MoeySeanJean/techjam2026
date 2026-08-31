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

Measured on five GPUs of the NUS SoC Slurm cluster, spanning four architectures.
Full tables in [RESULTS.md](docs/RESULTS.md); every figure is regenerated from the
JSON artifacts in `results/`.

| GPU | arch | shapes | vs reference | vs `torch.compile` | faster than both |
|---|---|---|---|---|---|
| H100 NVL | `sm_90` | 13 of 14 | **7.31x** (2.34 – 14.32) | 1.68x | **13 of 13** |
| A100-80 PCIe | `sm_80` | 13 of 14 | 5.31x (2.30 – 16.24) | 1.59x | **13 of 13** |
| TITAN RTX | `sm_75` | 13 of 14 | 3.31x (1.81 – 11.12) | 1.07x | 12 of 13 |
| Tesla T4 | `sm_75` | 12 of 14 | 3.35x (1.74 – 8.68) | 1.24x | **12 of 12** |
| TITAN V | `sm_70` | 11 of 14 | 3.31x (1.68 – 10.47) | 1.01x | 10 of 11 |

Medians over the official shapes. Every entry passed the accuracy gate — worst
envelope 0.796 of the allowed budget — and re-verification demoted nothing.

**60 of 62 measurements beat both the naive baseline and
`torch.compile(max-autotune)`; the other two are 0.9975x and 0.9999x, on shapes
where the search *selected `torch.compile` as the plan*.** Those are one code
path timed against itself, and which side of 1.0 they land on changes between
runs. The honest statement is that nothing here is slower than both references
by more than measurement noise, and that a pipeline allowed to conclude "the
existing compiler already wins here" is more useful than one that always
substitutes.

The margin over `torch.compile` splits by architecture: 1.59x and 1.68x where
TF32 and tensor-core bf16 exist, 1.01x–1.24x where they do not. That is the
design's honest consequence — most of the win is bought by spending a *measured*
precision budget, and Volta and Turing have none to spend. The win over the naive
reference survives everywhere (3.3x – 7.3x median).

**The same search, run on different hardware, picks different kernels.** Of the
11 official shapes all five GPUs can run, 10 chose a different plan on at least
one card — including 6 of 12 between a Tesla T4 and a TITAN RTX, which are the
*same architecture* but differ 2.3x in bandwidth and 4x in board power. A table
keyed on architecture alone would be wrong for one of them on half the shapes.

**Shape 14 needed opponents built for it.** `B32-S100000-d1024-H16-F1024-L2`
would require the reference to allocate an 18.6 TB attention score matrix, so the
organizer's baseline cannot run it and neither can `torch.compile` applied to
that baseline. Rather than quote no number, we raced two opponents that can.

| | A100-80 | H100 NVL |
|---|---|---|
| the organizer's model, attention chunked to O(S) | 166.9 s | 89.7 s |
| the organizer's model, attention by **PyTorch SDPA** | 77.5 s | 53.3 s |
| KernelForge | **17.9 s** | **9.0 s** |
| speedup vs chunked | 9.31x | 10.00x |
| **speedup vs PyTorch SDPA** | **4.32x** | **5.95x** |

The second opponent is the one that matters. `scaled_dot_product_attention` is a
fused, O(S)-memory attention written by the PyTorch team — not our code — and
substituting it for the single line the organizer's baseline cannot execute
leaves the rest of their model untouched. It is the strongest available opponent
and the number we would rather be judged on. Both sides run the organizer's own
numerics, and the chunked reference's chunk size is swept so it is not a strawman.

SDPA is the more memory-frugal of the two: 14.6 GB peak against our 30.2 GB. We
buy the speed with memory, on a shape where 80 GB is available.

Correctness is established against a streamed exact reference at full length and
full batch — **0 of 3,276,800,000 elements outside tolerance, envelope 0.399**.

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
default. We ship tuned tables for **sm_70**, **sm_75**, **sm_80** and **sm_90**,
all measured on the cluster nodes named in *Steps to reproduce*.

The default for an untuned shape depends on what the architecture can do. On
**Ampere or newer** the fp32 default uses fp16 for attention and `out_proj`,
justified because the reference itself runs its matmuls at TF32 — see
[docs/PRECISION.md](docs/PRECISION.md). On **Turing or Volta** there is no TF32,
that justification does not hold, and the default falls back to the bit-exact
plan. Tuned entries can still override this per shape, and on `sm_70` and `sm_75`
several do: the search measures the actual error there rather than inferring it,
and where fp16 stages clear the gate it ships them. `doctor` reports which
defaults apply to your card.

Very large shapes are run in batch slices sized against what the device reports
free, halving again on any slice that still does not fit, so a smaller card
trades speed for memory instead of failing. To cap the working set below what the
card would allow — a serving process or a second model sharing it — set an
explicit budget:

```python
model.set_memory_budget(8 * 2**30)   # 8 GiB ceiling; None tracks free memory
```

Slicing the batch is an execution-order change, not an approximation: nothing in
this model mixes batch elements, so the result does not depend on the budget.

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
python -m pytest tests/ -q            # 46 of 155 pass, 109 skip cleanly
python scripts/showcase.py --no-gpu   # three narrated acts from committed data
python scripts/report.py              # regenerate docs/RESULTS.md
```

Every number in `docs/RESULTS.md` and `docs/dashboard.html` comes from committed
JSON, so the results are inspectable without reproducing them.

---

## Steps to reproduce your results

**The headline claim** — tune, gate and re-verify every official shape on
whatever GPU you have, then print the table:

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
input and output resident. `--gate` checks the output against a streamed exact
reference at full length:

```bash
python scripts/shape14.py --scan            # how far this GPU gets
python scripts/shape14.py                   # the full shape
python scripts/shape14.py --gate --batch 32 # correctness, full batch
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

**Which GPU produced which table.** Each tuned table is one cluster node:

| table | GPU | node |
|---|---|---|
| `results/dispatch_sm_90.json` | H100 NVL 93 GB | `xgpi*` |
| `results/dispatch_sm_80.json` | A100 80 GB PCIe | `xgph*` |
| `results/dispatch_sm_75.json` | TITAN RTX 23.5 GB, Tesla T4 14.6 GB | `xgpe*`, `xgpf*` |
| `results/dispatch_sm_70.json` | TITAN V 11.8 GB | `xgpd*` |

The matching `results/sweep_*.json` records the node, driver and measured
bandwidth for every run.

`scripts/README.md` describes each script. The Slurm job files that produced our
cluster numbers are not in this repository — they run against one specific
university cluster, so they would be dead weight here; every result they produced
is committed as JSON, and `scripts/README.md` gives the portable equivalent of
each.

---

## Limitations, and what we would improve given more time

Two things are genuinely slower than they could be. Both have been measured to
the point where we know what the next step is and why we did not take it.

- **Causal attention runs at 0.53–0.72 of the non-causal cost, against a
  theoretical floor of 0.50** (measured on attention-kernel microbenchmarks, not
  on the official shapes). We were wrong about the cause twice. First we
  blamed load imbalance and built the standard remedy, a persistent-tile kernel:
  it was slower on every shape, 1.03x–1.54x. Then we decomposed the cost and
  found the work reduction was already nearly fully realised — exactly 0.50 of
  non-causal at `S=8192` — while *applying* the mask cost 1.16x–1.34x, because
  the kernel built the causal predicate on every key block though only the
  diagonal block can be affected by it. Splitting the key loop fixed that: worth
  1.08x–1.21x, bit-identical, shipped. What remains is the masking genuinely
  needed on the diagonal block and the lower arithmetic intensity of the tile
  causal masking prefers. We have no measurement isolating a further recoverable
  part, which is where we stopped.
  (`results/causal_residual_sm_80.json`, `results/persistent_tile_probe_sm_80.json`)
- **On pre-Ampere we roughly tie `torch.compile` rather than beating it** —
  1.01x–1.24x, against 1.59x and 1.68x on Ampere and Hopper. Most of our margin
  is bought by spending a *measured* precision budget, and cards without TF32
  have none to spend. We checked this was not simply a search that ran out of
  budget: doubling the per-shape budget on both cards moved the margin over the
  naive reference (2.82x → 3.32x on Volta) and left the margin over
  `torch.compile` flat (1.01x → 1.02x). Beating the compiler there needs a
  different lever — scheduling and launch overhead rather than precision — which
  we did not build. (`results/budget_probe_pre_ampere.json`)

### Boundary conditions, not open work

These are properties of the problem rather than things left undone, and no
further effort changes them:

- **Official shape 14 has no reference to be measured against.** The organizer's
  baseline would have to allocate an 18.6 TB attention matrix, so a speedup
  against it does not exist and cannot be produced. We report against the
  strongest substitute available — PyTorch's own `scaled_dot_product_attention`,
  which we did not write — and label it as such.
- **Eleven of the fourteen official shapes are `S=128`.** That is far more
  homogeneous than a real serving mix, and it means the set under-represents long
  causal attention, which is the regime our kernel work targets. We report the
  shapes we were given.

---

## Team member contributions

Solo entry. All design, implementation, measurement and writing by the
submitting author, using Claude Code (Claude Opus 5) as the development
environment throughout.
