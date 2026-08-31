# KernelForge — Devpost submission

**TikTok TechJam 2026 · Track 3: Implement a GPU Kernel for a Transformer Layer**

*This file is the written project description required by deliverable 3.5.1. It
is deliberately short; the evidence behind every number is in
[RESULTS.md](RESULTS.md) and [docs/TECH_REPORT.md](TECH_REPORT.md).*

---

## How the solution addresses the problem statement

The task asks for GPU kernels that implement a fixed Transformer layer, pass the
provided test cases within `relative error < 0.02, abs error < 0.002`, and run as
fast as possible across a published list of input shapes — with the explicit
allowance that *"participants can choose different implementations for different
shapes by adding shape checks."*

We took that last sentence literally, and it shaped the whole project. Rather
than hand-tuning one kernel, we built the loop that **produces and selects**
them:

1. **Profile** the workload and compute a per-stage error budget.
2. **Propose** candidate configurations — from a heuristic proposer, or from an
   LLM given the measured profile and the hardware's real limits.
3. **Prove** each candidate correct against the organizer's own tolerance rule,
   over multiple seeds, *before* it is allowed to be timed.
4. **Time** it honestly — interleaved round-robin, rotating order, medians.
5. **Freeze** the winner into a per-(architecture, shape) dispatch table, then
   **re-verify** the frozen table end to end and demote anything that drifts.

The kernels are the submission; the loop is the contribution. When the official
shape list was published on 27 August, specializing to it took one command and no
code change — which is the outcome the design was for.

Four claims, and where each is evidenced:

| claim | evidence |
|---|---|
| **AI and heuristics together optimize the kernel code.** Two proposers see the same spec sheet, profile and error budget, and are held to the same gate; the model also writes complete Triton source, all of it gated. | `results/genealogy_*.json`, `results/codegen*.json`, `results/generated/*.py`, [docs/CODEGEN.md](CODEGEN.md) |
| **It is worth something to a real workload.** A ranking traffic mix, measured end to end, untuned then tuned. | [docs/USE_CASE.md](USE_CASE.md), `scripts/usecase.py` |
| **The code performs.** 60 of 62 official-shape measurements across five GPUs and four architectures beat both the naive baseline and `torch.compile`; zero demoted on re-verification anywhere. | [RESULTS.md](RESULTS.md), `results/sweep_*.json` |
| **It generalizes.** Any CUDA GPU, any OpenAI-compatible LLM endpoint, or neither. Every path degrades to a correct one. | [README.md](../README.md#setup-and-installation), `cli doctor` |

### What we actually wrote

- A **Triton FlashAttention** kernel with causal **and** key-padding masking
  handled in-register. PyTorch's SDPA takes `is_causal` *or* an `attn_mask`, not
  both cheaply, so the combination the benchmark generates drops it off its fast
  path. Ours also supports `head_dim=8` (two official shapes need it) by exact
  zero-padding.
- A **fused add + mask + LayerNorm** kernel.
- A **structural rewrite** of the layer stack: fused QKV, no redundant
  `.contiguous()`, one flat `[B·S, d]` token axis, and CUDA-graph capture that
  removes ~105 kernel launches per forward.
- A **numerics gate** built from the organizer's own comparison code, collapsed
  to one number — *envelope utilization* — so "how close to failing" is
  measurable rather than a yes/no.

### Results

Measured on five cluster GPUs spanning four architectures. We report only
hardware that holds a stable clock — on a throttling GPU the measured ratios come
out *higher*, because a weaker card spends proportionally more time on the launch
overhead we remove, so such a machine would flatter us for the wrong reason.

| GPU | arch | shapes | vs reference | vs `torch.compile` | faster than both |
|---|---|---|---|---|---|
| H100 NVL | `sm_90` | 13 of 14 | **7.31x** (2.34 – 14.32) | 1.68x | **13 of 13** |
| A100-80 PCIe | `sm_80` | 13 of 14 | 5.31x (2.30 – 16.24) | 1.59x | **13 of 13** |
| TITAN RTX | `sm_75` | 13 of 14 | 3.31x (1.81 – 11.12) | 1.07x | 12 of 13 |
| Tesla T4 | `sm_75` | 12 of 14 | 3.35x (1.74 – 8.68) | 1.24x | **12 of 12** |
| TITAN V | `sm_70` | 11 of 14 | 3.31x (1.68 – 10.47) | 1.01x | 10 of 11 |

Medians over the official shapes; every entry passed the accuracy gate and
nothing was demoted on re-verification.

**60 of 62 measurements beat both the naive baseline and
`torch.compile(max-autotune)`.** The two exceptions record 0.9975x and 0.9999x,
and on both the plan the search shipped *is* `torch.compile` — one code path
timed against itself. A pipeline allowed to conclude "the existing compiler
already wins here" is more useful than one that always substitutes.

The margin over `torch.compile` is large where TF32 and tensor-core bf16 exist
(1.59x, 1.68x) and near parity on pre-Ampere, which is the design's honest
consequence: most of our win is bought by spending a *measured* precision budget,
and Volta and Turing have none to spend. We tested that against the obvious
alternative — that the pre-Ampere search was budget-starved — by doubling the
budget on both cards. The margin over the reference moved; the margin over
`torch.compile` did not.

**Official shape 14** (`B32-S100000-d1024-H16-F1024-L2`) is the one the reference
cannot run at all: it would have to allocate an 18.6 TB attention score matrix.
Rather than quote no number, we raced two opponents that can run it. The one that
counts is the organizer's model with its attention replaced by PyTorch's
`scaled_dot_product_attention` — a fused O(S) attention we did not write —
substituted for the single line their baseline cannot execute: **4.32x on an
A100-80** (77.5 s to 17.9 s) and **5.95x on an H100 NVL** (53.3 s to 9.0 s).
Against a chunked version of their own attention, which is our code and so a
weaker check, it is 9.31x and 10.00x. Correctness is checked against a streamed
exact reference at full length and full batch — 0 of 3,276,800,000 elements
outside tolerance. SDPA uses less memory than we do (14.6 GB against 30.2 GB);
we buy the speed with memory, and `set_memory_budget()` caps the working set when
the card is shared.

**Portability is measured, not asserted.** The same pipeline, unmodified, tuned
`sm_70` and `sm_75` on hardware it had never seen. And of the 11 official shapes
all five GPUs can run, 10 chose a different plan on at least one card — including
6 of 12 between a Tesla T4 and a TITAN RTX, which are the *same architecture*.
That is the argument for searching rather than hand-tuning, stated as a
measurement.

---

## Development tools used

- **Claude Code (Claude Opus 5)** — the primary development environment, used for
  kernel authoring, profiling analysis, and the AI-assisted codegen path
  described below.
- **VS Code**, Git, Windows 11 + WSL-style Git Bash.
- **NVIDIA Nsight Compute**, `nvidia-smi`, and the **PyTorch profiler** for
  bottleneck attribution.
- **Slurm** on the NUS SoC GPU cluster (`sbatch`/`srun`) for the A100 and H100
  runs, driven over SSH through a jump host. Those job files are not in the
  repository — they run on one specific cluster, so they are provenance rather
  than instructions; `scripts/README.md` lists the portable equivalent of each.

## APIs used

- **NUS SoC LLM-as-a-Service** — an OpenAI-compatible gateway, used by the LLM
  proposer and the kernel-source generator. The client speaks plain
  OpenAI-compatible HTTP, so any provider works: OpenAI, the Anthropic Messages
  API, a local Ollama, vLLM, or LM Studio. Configuration is a base URL and a key
  in `.env`; `README.md` documents the substitutions.
- **No API is on the inference path.** The LLM is a build-time proposer whose
  output must pass the same accuracy gate as everything else. With no key
  configured the system falls back to a heuristic proposer and still runs
  end to end.

## Libraries and frameworks used

- **PyTorch 2.11** (+cu126) — reference model, SDPA fallback, `torch.compile`
  as both a comparison point and a dispatch candidate.
- **Triton 3.7.1** (`triton-windows` on Windows) — the FlashAttention and fused
  LayerNorm kernels.
- **pytest** — 155 tests, of which 46 pass and 109 skip cleanly with no GPU.
- **pynvml** — energy sampling for the impact analysis.
- **paramiko** — cluster orchestration over the jump host.

## Datasets and assets used

**None.** Every input is produced by the organizer's own
`generate_random_case`, which we call directly rather than reimplementing, so the
accuracy gate sees exactly the distribution the benchmark uses — including
padding ratio and input scaling. No external dataset, no pretrained weights, no
third-party assets.

---

## Reflection: limitations, and what we would do next

Full list in [README.md](../README.md#limitations-and-what-we-would-improve-given-more-time).
The three that matter most:

1. **Causal attention: we were wrong about the cause twice, and the second time
   found a real 1.08x–1.21x.** We blamed load imbalance and built the standard
   remedy, a persistent-tile kernel — slower on every shape, 1.03x–1.54x. So we
   decomposed instead: the causal *work volume* was already nearly fully realised
   (0.50 of non-causal at S=8192), while applying the mask cost 1.16x–1.34x. The
   kernel built the causal predicate on every key block though only the diagonal
   one can be affected. Splitting the loop ships, is bit-identical including under
   key padding, and takes causal from 0.62–0.72 to 0.53–0.72 of non-causal.
   Recorded in `results/causal_residual_sm_80.json` and
   `results/persistent_tile_probe_sm_80.json`.
2. **Shape 14 spends 97.7% of its GPU time in the fp32 attention fallback.**
   Measured with the profiler. Triton's `tl.dot` needs a narrow float type, so an
   fp32 attention stage falls through to SDPA. An fp32 flash kernel, or a bf16
   path with an error budget verified at that length, would attack 98% of the
   cost.
3. **On pre-Ampere we roughly tie `torch.compile`, and we checked that this is
   structural.** Suspecting the search had simply run out of budget, we re-ran
   Volta and Turing with double the per-shape budget and more timing trials. The
   margin over the naive reference improved (2.82x → 3.32x on Volta); the margin
   over `torch.compile` did not move (1.01x → 1.02x). Most of our win is bought
   by spending a measured precision budget, and cards without TF32 have none to
   spend.

Accuracy is not the cause of the first: an `exp2`-free variant of the same
kernel measures identical envelopes to four decimal places from `S=128` to
`S=4096`, so the causal path's problem is occupancy, not arithmetic.

## Where to find the evidence

The repository is large because the measurements are kept, not because the
solution is. If you have limited time:

| to check | open |
|---|---|
| that it passes and is faster | `python scripts/run_official.py --causal` — the organizer's script, unmodified |
| the numbers, all of them | [RESULTS.md](RESULTS.md), or `dashboard.html` to click through them |
| how it was built and why | [docs/TECH_REPORT.md](TECH_REPORT.md) — environment, optimizations, results |
| the AI-assisted parts | [docs/CODEGEN.md](CODEGEN.md) + §7 of the tech report; raw transcripts in `results/` |
| whether the correctness claims hold | [docs/PRECISION.md](PRECISION.md), [docs/EQUIVALENCE.md](EQUIVALENCE.md), `python -m pytest -q` |
| that it runs on *your* hardware | [README.md](../README.md#setup-and-installation) — different GPU, no LLM key, or no GPU at all |
| what it would be worth | [docs/USE_CASE.md](USE_CASE.md) — measured, with its assumptions stated |
| it working, in three minutes | `python scripts/showcase.py` (or `--no-gpu`, which needs no hardware) |

Every claim in the documents above is backed by a JSON artifact in `results/`,
and every artifact is regenerated by a command named next to it. Where we could
not measure something — shape 14's accuracy at full size, most obviously — we say
so instead of estimating it.

## Team member contributions

Solo entry. All design, implementation, measurement and writing by the submitting
author, using Claude Code as the development environment throughout.
