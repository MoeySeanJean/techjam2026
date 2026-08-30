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
| **The code performs.** 26 of 26 official-shape measurements beat both the naive baseline and `torch.compile`; zero demoted on re-verification. | [RESULTS.md](RESULTS.md), `results/sweep_*.json` |
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

Measured on the A100-80 and H100 cluster nodes. We report only hardware that
holds a stable clock — on a throttling GPU the measured ratios come out *higher*,
because a weaker card spends proportionally more time on the launch overhead we
remove, so such a machine would flatter us for the wrong reason.

| | A100-80 PCIe (sm_80) | H100 NVL (sm_90) |
|---|---|---|
| official shapes measured | 13 of 14 | 13 of 14 |
| median vs the reference | 5.39x | **7.35x** |
| range vs the reference | 2.32x – 15.25x | 2.34x – 13.29x |
| median vs `torch.compile` | 1.53x | 1.69x |
| **faster than both references** | **13 of 13** | **13 of 13** |
| passed the accuracy gate | all | all |
| demoted on re-verification | 0 | 0 |

**Every officially specified shape that can be run beats both the naive baseline
and `torch.compile(max-autotune)`, on both cluster GPUs — 26 of 26.** The
narrowest margins are 2.32x over the reference and 1.02x over `torch.compile`;
that last one is parity rather than a win and we say so in the report.

**Official shape 14** (`B32-S100000-d1024-H16-F1024-L2`) is the one the reference
cannot run at all: it would have to allocate an 18.6 TB attention score matrix.
We run it in **77.7 s on an A100-80** and **54.5 s on an H100 NVL**. We quote no
speedup, because a ratio against something that cannot run is not a measurement.

Long causal attention on a small GPU is our weakest regime — on a 46-SM card we
have measured shape 13 at 0.94x of `torch.compile`. That is outside our reported
set, and documented rather than dropped.

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
- **pytest** — 133 tests, of which 41 pass and 92 skip cleanly with no GPU.
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

1. **We cannot prove shape 14's accuracy at full size.** Nothing can compute a
   reference output at `S=100000`, so there is no envelope to measure. We verify
   the same code path against an exact reference at every length that *does* fit,
   and that slicing the batch does not change the answer. Not closable — it is a
   property of the shape.
2. **Shape 14 spends 97.7% of its GPU time in the fp32 attention fallback.**
   Measured with the profiler. Triton's `tl.dot` needs a narrow float type, so an
   fp32 attention stage falls through to SDPA. An fp32 flash kernel, or a bf16
   path with an error budget verified at that length, would attack 98% of the
   cost.
3. **Small-batch long-causal attention is where our kernel loses to the
   library.** On the official causal shapes we win with our own kernel (shape 13:
   12.14x over the reference, 4.26x over `torch.compile`), but on `B2-S2048`
   causal the search picks `torch.compile` instead. The roofline says why: 17%
   and 11% of tensor-core ceiling against 45–50% for the same shape without
   causal masking — skipping tiles above the diagonal halves the work but not the
   launch grid.

We also closed three limitations while preparing the submission, including one
where our stated hypothesis turned out to be **wrong**: we suspected the `exp2`
softmax substitution degraded accuracy as rows grew, built the `tl.exp` variant,
and measured identical envelopes to four decimal places from `S=128` to
`S=4096`. Details in [README.md](../README.md#settled-while-preparing-the-submission).

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
