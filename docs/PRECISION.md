# Precision and the accuracy gate

This is the document that shaped every engineering decision in the project. It
records what we measured about the organizer's tolerance rule, not what we
assumed about it.

## The rule

From `torch_transformer_benchmark.py`:

```python
abs_ok = abs_error <= atol
rel_ok = abs_error <= rtol * ref.abs()
passed_mask = finite_mask & (abs_ok | rel_ok)
```

It is an **OR**. The per-element allowance is therefore `max(atol, rtol*|ref|)`,
and a single scalar captures how close a candidate is to failing:

```
envelope utilization = max over elements of  abs_err / max(atol, rtol*|ref|)
```

Utilization below 1.0 passes. We gate at **0.80**, not 1.0 — see "Margin" below.

Script defaults are `atol=2e-3`, `rtol=2e-2`, matching the problem statement's
"relative error < 0.02, abs error < 0.002". `kernelforge/numerics.py` reads both
constants out of `parse_args` with `inspect.getsource` rather than hardcoding
them, so the gate follows the script.

The script deliberately does *not* use `torch.isclose`, whose
`atol + rtol*|ref|` is more permissive, and says so in a comment.

## Finding 1 — the stack amplifies perturbations by ~10³

The benchmark evaluates a **6-layer stack** by default, not a single layer. A
change to any operation is amplified as it propagates.

Our fused LayerNorm differs from `torch.layer_norm` only by reduction order.
For a 512-wide row in fp32 that is a perturbation of about `sqrt(512) * 1.2e-7
≈ 2.7e-6`, which we measured directly in the kernel unit test. After the stack:

| depth | envelope from fused norm alone (fp32) |
|---|---|
| 1 | 0.185 |
| 2 | 0.283 |
| 4 | 0.427 |
| 6 | **0.552** |
| 12 | **0.977** |

A 2.7e-6 input perturbation becomes a ~6e-4 output difference. The amplification
is large but sub-linear in depth (it saturates), which is why the L=12 case sits
right at the edge of the gate rather than far past it.

**Consequence:** at `--layers 12` there is almost no budget left for anything
else, and precision choices that are comfortable at L=6 are not safe at L=12.
The dispatch table is keyed on `L` for exactly this reason.

## Finding 2 — in fp16 and bf16 mode the gate rejects `torch.compile` itself

We ran PyTorch's own optimizer against the organizer's own baseline, same
weights, same shape (B8 S128 d512 H8 F2048 L6). Because this is a strong claim,
the sweep now gates the library baselines on every run, on every GPU, in both
compile modes. Measured envelope utilization (limit 1.0):

| GPU | dtype | `max-autotune` | `reduce-overhead` | verdict |
|---|---|---|---|---|
| A100-80 (sm_80) | float16 | 2.808 | 2.869 | **FAIL** |
| H100 NVL (sm_90) | float16 | 2.495 | 2.708 | **FAIL** |
| Tesla T4 (sm_75) | float16 | 1.373 | 1.343 | **FAIL** |
| TITAN V (sm_70) | float16 | 1.343 | 1.282 | **FAIL** |
| A100-80 (sm_80) | bfloat16 | 20.996 | 20.020 | **FAIL** |
| H100 NVL (sm_90) | bfloat16 | 19.617 | 19.617 | **FAIL** |
| all four | float32 | passes | passes | **PASS** |

**Twelve independent failures across four architectures and both compile modes,
and no float32 failure anywhere.** The numbers come straight from
`results/sweep_*.json`, which gates the library baselines on every run.

The margin of failure shrinks on older cards — 1.28–1.37 against 2.5–2.9 on
Ampere and Hopper — but the verdict does not change: `torch.compile` is
inadmissible at fp16 on every architecture we measured. Where it wins on paper it
wins by being wrong; on the TITAN V it is 2.4x faster than the baseline at
fp16 and still fails, which is why the dispatch table ships `safe(exact)+graph`
there instead.

`bfloat16` on pre-Ampere is the one case that needs a caveat rather than a row.
Volta and Turing have no native bf16, so `torch.compile` emits arithmetic
indistinguishable from the baseline — envelope 0.000 at 1.000x and 1.002x,
identical timings to three decimal places. It passes the gate by declining to
optimize, which is not evidence either way, so it is not counted among the
failures above.

**How to read this.** Our first instinct was to call it a benchmark defect. On
reflection that is the wrong reading, and the right one is more interesting.

The organizers chose the *naive* implementation as the reference and listed
`torch.compile` as a suggested **tool**. Together those define the actual
problem: you may reach for aggressive optimizations, but correctness is measured
against the unoptimized reference, and establishing where a tool is admissible is
your job. At narrow I/O dtypes the reference's own rounding noise, amplified
through six layers, exceeds the tolerance — so a blanket `torch.compile`
submission fails, and one that never checks fails silently.

So this is not "we found a hole in your benchmark." It is **the discipline that
lets aggressive tools be used exactly where they are provably correct**, which is
why `torch.compile` is a *candidate in our dispatch table* rather than merely a
yardstick: it wins the shapes where it passes and is rejected where it does not,
with no special-casing anywhere in the code.

### Scope

All 14 official shapes are fp32, and on that list `torch.compile` clears the gate
everywhere. Every official-shape result in `RESULTS.md` is therefore a win over
an *admissible* `torch.compile`.

The admissibility finding applies to the fp16 and bf16 configurations the script
also accepts, where compiling the baseline fails. The check stays in the loop
because the shape list is fixed but the dtype is a flag.

One configuration is worth recording: compiling our bit-exact rewrite at fp16
measures 0.000 envelope in roughly one run in five and 2.5–3.3 in the rest, as
inductor's autotuning sometimes selects a kernel set whose rounding happens to
match. It is not stable across processes and the sweep rejects it automatically.

Single-change ablation:

| change (one at a time, from the baseline) | fp32 | fp16 | bf16 |
|---|---|---|---|
| loop rewrite, exact attention | **0.000** | **0.000** | **0.000** |
| + fused QKV GEMM | **0.000** | 2.655 FAIL | **0.000** |
| + fused Triton norm | 0.675 | 2.991 FAIL | 18.981 FAIL |
| + fp32 residual stream | 0.000 | 2.289 FAIL | 20.020 FAIL |
| attention → SDPA | 0.694 | 2.197 FAIL | 19.775 FAIL |
| attention → our flash kernel | 0.597 | 2.594 FAIL | 18.066 FAIL |

1. **The structural rewrite is bit-exact** on every dtype — removing redundant
   `.contiguous()` copies, hoisting the per-layer causal-mask allocation and
   moving row masking to the block boundary change nothing numerically.
2. **Fusing QKV is bit-exact in fp32 and bf16 but not fp16** (2.655): merging
   three GEMMs changes cuBLAS kernel selection, and at fp16 that changes
   accumulation order enough to matter.

**Consequence:** fp16 and bf16 inputs take a bit-exact path, with the speedup
coming from removing ~105 kernel launches per forward and from compiling the
bit-exact rewrite.

## Finding 3 — in fp32, fp16 attention is free; the FFN is not

The reference runs its matmuls at **TF32** in fp32 mode (`--allow-tf32` defaults
true, `matmul_precision='high'`). TF32 carries 10 explicit mantissa bits; fp16
carries 10 explicit plus an implicit one. Computing in fp16 is therefore not a
precision regression against this reference — it is a comparable rounding.

Measured marginal envelope cost of narrowing one stage to fp16, holding the rest
wide (default shape):

| stage | marginal envelope cost |
|---|---|
| attention (QKV GEMM + flash) | **-0.030** |
| out_proj | +0.031 |
| ffn2 | +0.221 |
| ffn1 | +0.326 |

The attention cost is **negative**: fp16 attention lands *closer* to the
reference than our fp32 flash path does, because the reference's own QK^T runs
at TF32 and fp16 is nearer to TF32 than fp32 is.

The FFN GEMMs are expensive because they carry the longest reduction dimensions
(`d→ffn` and `ffn→d`, 2048 wide here). This is not the ordering intuition
suggests — attention is the scary-looking part — and it is why the search is
driven by measurement rather than by a hand-written precision policy.

**Consequence:** the shipped fp32 plan is `fp16[attn, out_proj]` with the FFN
kept wide. On the default shape it clears the gate at 0.707 and still beats
`torch.compile` by 1.48x.

## Finding 4 — the residual stream must stay wide

Narrowing the residual stream to fp16 pushes utilization from ~0.7 to 2.8. It is
summed across `2 * num_layers` sublayers, so it is the one place where narrow
storage compounds rather than merely rounding. `residual_dtype` is a separate
knob from `compute_dtype` for this reason, and it is always fp32 in shipped
plans.

## Margin

We gate at 0.80, not 1.0. Utilization is measured over three seeds; the
organizer will run a seed we have not seen, and a hard accuracy failure skips
benchmarking entirely (`return 2`). A configuration that passes at 0.99 on our
seeds is not one we are willing to submit.

## Reproducing

```bash
python -m kernelforge.cli budget --shapes-file official_shapes.txt
```

writes the per-stage table to `results/error_budget.txt`.
