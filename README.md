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

Nine GPUs on the NUS SoC Slurm cluster, four architectures, official shapes only.
Every figure regenerates from the JSON in `results/`; full tables in
[RESULTS.md](docs/RESULTS.md).

The headline table is below under *Every GPU on the cluster*: **116 shape-runs,
116 passed, medians 3.12x to 7.14x** against the organizer's own baseline,
measured through their unmodified script.

**On `torch.compile`.** Each candidate is also timed against
`torch.compile(max-autotune)` during the search. Median margin over it: 1.66x and
1.67x on the Hopper parts, 1.35x–1.59x on Ampere, 1.02x–1.23x on Volta and
Turing, where there is no TF32 headroom to spend.

96 of 114 search measurements beat both references. Fifteen of the eighteen that
do not are deliberate: the search declines a `torch.compile` plan leading by
under 15% and ships our own kernel, because compile plans measured a mean 0.891x
of our kernels under the organizer's harness against our 1.017x — a narrow win in
our sweep is a loss where it counts. All nine GPUs measured faster after that
margin than before it. The other three are `torch.compile` timed against itself.

Worst envelope at selection 0.782 of the allowed budget; re-verification demoted
nothing.

### Every GPU on the cluster

Nine GPU configurations across four architectures. **Each is tuned for itself**,
then the LLM agent competes for slots in that card's table, then the result is
measured through the organizer's own `torch_transformer_benchmark.py`,
unmodified, with our layer substituted in.

| GPU | arch | SMs | memory | shapes | passed | median | range |
|---|---|---|---|---|---|---|---|
| H100 NVL | `sm_90` | 132 | 93 GB | 13 of 14 | **13/13** | **7.14x** | 2.25 – 14.58 |
| A100-PCIE-40GB | `sm_80` | 108 | 40 GB | 13 of 14 | **13/13** | **6.61x** | 2.31 – 15.14 |
| A100 80GB PCIe | `sm_80` | 108 | 79 GB | 13 of 14 | **13/13** | **5.25x** | 2.29 – 15.90 |
| H100 NVL MIG 3g.47gb | `sm_90` | 60 | 46 GB | 13 of 14 | **13/13** | **4.45x** | 2.24 – 14.90 |
| A100 80GB PCIe MIG 3g.40gb | `sm_80` | 42 | 39 GB | 13 of 14 | **13/13** | **4.38x** | 1.50 – 17.05 |
| H200 NVL | `sm_90` | 132 | 140 GB | 13 of 14 | **13/13** | **4.34x** | 2.40 – 12.45 |
| TITAN V | `sm_70` | 80 | 12 GB | 12 of 14 | **12/12** | **3.55x** | 2.15 – 10.14 |
| TITAN RTX | `sm_75` | 72 | 24 GB | 13 of 14 | **13/13** | **3.37x** | 2.15 – 9.34 |
| Tesla T4 | `sm_75` | 40 | 15 GB | 13 of 14 | **13/13** | **3.12x** | 1.02 – 5.86 |

**9 GPUs, 116 shape-runs, 116 passed — 100%**, every shape served by
its own card's table. Speedup is the organizer's figure: our median latency
against their unmodified baseline on the same input.

Two cards are MIG partitions with roughly a third of the SMs of the card they are
cut from, and one is an H200 the project had never run on before this round.
They are tuned like any other card, which is the point: the pipeline fits the
hardware in front of it rather than shipping a table and hoping.

**Shape 14 needs an opponent.** `B32-S100000-d1024-H16-F1024-L2` would require
the reference to allocate an 18.6 TB attention matrix, so the organizer's
baseline cannot run it and no speedup against it exists. We measure against two
implementations that can:

| | A100-80 | H100 NVL |
|---|---|---|
| their model, attention chunked to O(S) (our code) | 166.9 s | 89.7 s |
| their model, attention by **PyTorch SDPA** (not our code) | 77.5 s | 53.3 s |
| KernelForge | **17.9 s** | **9.0 s** |
| **speedup vs SDPA** | **4.32x** | **5.95x** |

The SDPA row is the one to read: it is an independently written fused attention
substituted for the single line their baseline cannot execute. It also uses less
memory than we do — 14.6 GB against 30.2 GB — so we buy the speed with memory.

Correctness is checked against a streamed exact reference at full length and full
batch: **0 of 3,276,800,000 elements outside tolerance**, envelope 0.399.

### Where to look

| | |
|---|---|
| every number, per shape, per GPU | [RESULTS.md](docs/RESULTS.md) |
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

Nothing here requires our hardware, and the model is **tune your card, then
run**. Dispatch resolves in this order: an exact entry for your
`(architecture, dtype, shape)` in *this card's* table; the conservative bit-exact
plan if the card has been tuned but not for that shape; then a safe
architecture-appropriate default.

Tables are per device, not per architecture, because which legal plan is
*fastest* is a property of the card rather than the instruction set. Measured on
two `sm_75` parts, six of twelve official shapes prefer different plans and
taking the wrong card's choice costs up to 1.08x — so a table tuned on an A100-80
is not the table an A100-40 wants, even though both are `sm_80`.

A GPU with no table of its own still runs **correctly** — the default path is
gated the same way as everything else — it is simply not yet fast. One command
fixes that:

```bash
python -m kernelforge.cli tune --shapes-file official_shapes.txt
```

The default for an untuned shape depends on what the architecture can do. On
**Ampere or newer** the fp32 default uses fp16 for attention and `out_proj`,
justified because the reference itself runs its matmuls at TF32 — see
[docs/PRECISION.md](docs/PRECISION.md). On **Turing or Volta** there is no TF32,
that justification does not hold, and the default falls back to the bit-exact
plan. A tuned entry can still override this per shape: the search measures the
actual error on the card in front of it rather than inferring it, and where fp16
stages clear the gate it ships them. `doctor` reports which defaults apply to
your card and whether it has a tuned table.

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

---

## Steps to reproduce your results

**The full pipeline on your GPU** — tune for the card, let the LLM compete for
slots, verify, then measure through the organizer's own script:

```bash
python -m kernelforge.cli tune   --shapes-file official_shapes.txt
python -m kernelforge.cli agent  --shapes-file official_shapes.txt --provider llm
python -m kernelforge.cli verify --shapes-file official_shapes.txt --demote
python scripts/official_all.py
python scripts/report.py                    # regenerates docs/RESULTS.md
```

`official_shapes.txt` holds the 14 shapes from Appendix 3.7. The search spends up
to `--case-budget` seconds per shape (default 300). The `agent` step needs an LLM
endpoint; without one everything else still runs. Any other shape is tuned the
same way — `cli tune --shapes B4-S777-d640-H10-F2560-L9`.

**One shape through the organizer's script**, which imports
`torch_transformer_benchmark.py` unmodified and substitutes our layer:

```bash
python scripts/run_official.py --batch-size 64 --seq-len 128     --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --causal
```

**Shape 14**, which the reference cannot run. `--scan` finds where your card
stops; the full shape peaks at 45.9 GB and needs ~80 GB resident:

```bash
python scripts/shape14.py --scan             # how far this GPU gets
python scripts/shape14.py --race             # against SDPA and a chunked reference
python scripts/shape14.py --gate --batch 32  # correctness at full length
```

**The use case** — a ranking traffic mix, measured end to end:

```bash
python scripts/usecase.py --cold    # architecture defaults
python scripts/usecase.py --tune    # tune the four segments, then measure
```

**With no GPU:** `python -m pytest tests/` runs 175 tests — 55 pass and 120
skip cleanly without one. Every number in `docs/RESULTS.md` regenerates from
committed JSON, so the results are inspectable without reproducing them.

Our Slurm job files are not in the repository — they target one university
cluster, so they are provenance rather than instructions. `scripts/README.md`
gives the portable equivalent of each.

## Limitations, and what we would improve given more time

Three things are genuinely slower than they could be. All have been measured to
the point where we know what the next step is and why we did not take it.

- **Causal attention sits 1.00x–1.28x above its arithmetic floor** (measured on
  attention-kernel microbenchmarks, not the official shapes). We were wrong about
  this three times, and the corrections are the useful part. First we blamed load
  imbalance and built the standard remedy, a persistent-tile kernel: slower on
  every shape, 1.03x–1.54x. Then we decomposed the cost and found the work
  reduction was already nearly fully realised while *applying* the mask cost
  1.16x–1.34x — the kernel built the causal predicate on every key block though
  only the diagonal block can be affected by it. Splitting the key loop fixed
  that: worth 1.08x–1.21x, bit-identical, shipped. Finally, the target itself was
  wrong: we had been comparing against 0.50, but a program reading `m+1` key
  blocks makes the floor `(M+1)/2M`, which is 0.562 at eight m-blocks and only
  approaches 0.50 as the sequence grows. Against the correct per-shape floor one
  shape sits exactly on it and the rest are 9–28% above — so the limit is
  reachable by this kernel, and what remains is headroom rather than a barrier.
  We have no measurement isolating a further recoverable part.
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

- **On the two compute-bound official shapes we reach a lower fraction of the
  H100's ceiling than the A100's** — 53% against 77%, and 15% against 25%. We
  first wrote this as "our tiles do not saturate the H100", which the data does
  not support: comparing achieved DRAM bandwidth against each card's *own* peak,
  neither card exceeds 11% on the eleven bandwidth-bound shapes, so nothing there
  is tiling-limited — those shapes are `S=128` with `d<=128` and simply have too
  little work to saturate either machine. Much of the apparent gap is arithmetic:
  the H100's tensor-core ceiling is 2.7x the A100's while its memory bandwidth is
  only 2.1x, so identical behaviour scores lower as a percentage. What remains is
  real but confined to the two shapes with enough arithmetic to be compute-bound.
  Larger tilings were measured and did not help — one 1.024x win on one shape,
  not enough to change a default — so the remaining lever is TMA and wgmma, which
  we did not build. (`results/h100_utilisation_sm_90.json`,
  `results/tile_space_probe_sm_90.json`)

- **The agent's promotion decision is noisier than its margin implies.** A
  proposal takes a slot when its measured speedup beats the frozen entry's by
  3%, but those two numbers come from different harnesses — the agent times one
  proposal, the search times many interleaved. Against what the organizer's
  benchmark then measured for the same shipped plan, the agent's figure is
  unbiased on median (1.02x) but ranges 0.15x–2.16x, with only 30 of 89 within
  10%. Some promotions are therefore luck rather than improvement. Correctness is
  unaffected — a proposal is gated before it is timed, so a noisy promotion can
  only ship something slower — and every card still measured faster after the
  agent round. The fix is to re-time the winner in the incumbent's harness before
  promoting; we did not build it. (`results/agent_measurement_noise.json`)

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
