# AI-generated kernels

The model writes complete Triton source against a contract. Every candidate is
compiled, gated against an exact reference, and timed by the same harness that
governs everything else.

```bash
python -m kernelforge.cli codegen --targets layernorm,gelu --iterations 10
```

Generated modules are saved to `results/generated/` before they run; per-attempt
verdicts are in `results/codegen.json`. All runs below are on the A100-80 node
`xgph1`.

## Targets

| target | contract | reference it must match |
|---|---|---|
| `gelu` | fused bias-add + exact erf GELU | `F.gelu(x.float() + bias.float(), approximate="none")` |
| `layernorm` | fused residual-add + row mask + LayerNorm | our block-boundary kernel |

## Which model writes kernels

`scripts/pick_model.py` scores models on whether they emit a valid *plan* as
JSON. Six of ten tied at 100%, so it ranked them on latency and recommended
`ornith1.5:35b`. Measuring the actual task instead — every model, 20 kernels per
sample, same targets, same gate:

| model (as served) | valid plan JSON | correct kernels | samples |
|---|---|---|---|
| **`qwen3.8:27b`** | 75% | **48/60 (80%)** | 16/20, 15/20, 17/20 |
| `qwen3-coder-next` | 100% | 17/40 (43%) | 8/20, 9/20 |
| `qwen3.6:35b` | 88% | 9/40 (22%) | 6/20, 3/20 |
| `gemma4:26b` | 100% | 7/40 (18%) | 5/20, 2/20 |
| `ornith1.5:35b` | 100% | 3/40 (8%) | 0/20, 3/20 |
| `qwen3.5:9b` | 100% | 0/40 (0%) | 0/20, 0/20 — syntax errors dominate |
| `llama3.1:8b` | 75% | 0/20 (0%) | 0/20 |

**The JSON benchmark is anti-predictive.** It ranked `ornith1.5:35b` first; that
model wrote 3 of 40. `qwen3.8:27b`, ranked last on format at 75%, wrote 48 of 60.
We ship **`qwen3.8:27b`**.

Two properties of the gateway that the measurement has to account for:

- **It aliases model ids.** Ten advertised ids resolve to seven distinct models —
  `qwen3.6:27b` is served by `qwen3.8:27b`, `ornith1.0:35b` by `ornith1.5:35b`,
  `default` by `qwen3.6:35b`. Mapping in `results/model_aliases.json`; every
  artifact records the id that actually served the request, so a score is never
  attributed to the wrong model.
- **It rate-limits per key.** The client backs off to a two-minute ceiling and
  honours `Retry-After`; a request that never returns is logged as `api_error`
  in the taxonomy, so a throttled arm is distinguishable from a model that
  generated nothing.

Every model except `llama3.1:8b` has at least two independent samples, and the
winner has three. The spread within a model is wide — 5/20 and 2/20 for
`gemma4:26b`, 6/20 and 3/20 for `qwen3.6:35b` — so the ordering of the middle of
the table is not defended; the top two and the bottom one are separated by far
more than that spread.

## Results — the shipped model's run

`qwen3.8:27b`, 10 attempts per target:

| target | ok | how the rest failed | best generated kernel |
|---|---|---|---|
| `layernorm` | **10/10** | — | **2.46x** vs torch, envelope 0.042, 55 lines |
| `gelu` | **7/10** | 3 `numeric_fail` | **3.91x** vs torch, envelope 0.031, 67 lines |

Successful `gelu` attempts cluster at 3.59–3.91x.

## Failure modes

**1. Silent wrongness dominates.** `bias_gelu_0ea01c2425.py` tiles correctly,
masks both edges, applies its strides, accumulates in fp32 — then:

```python
# Exact erf GELU: 0.5 * v * (1 + erf(v / sqrt(2)))
# Use the Abramowitz-Stegun 7.1.26 polynomial approximation for erf
# which gives ~1.5e-7 absolute accuracy, well within float16 tolerance
abs_v = tl.abs(v)
t = 1.0 / (1.0 + 0.3275911 * abs_v)
```

The model did not forget `tl.erf`; it substituted an approximation and quantified
the error correctly. Tolerance is measured against the *reference*, which calls
exact `erf`, so the envelope is **22.8** against a limit of 1.0 — 5,724,633 of
8,388,608 elements outside tolerance. The same model produces the same
substitution with the same comment on both GPUs.

**2. Under-specification, not incapability.** The contract is worth more than
the feedback mechanism. This clause:

> CRITICAL — do not split the d axis. mean and variance are per-row statistics
> over ALL d elements, so a single program must reduce a whole row.

is worth **0/5 → 5/12** on the `layernorm` target; without it the model splits
the reduction and the kernel is silently wrong.

**3. `shared_memory_overflow`.** A tiling fine on an A100 does not launch on a
99 KB shared-memory budget. The spec sheet states the limit explicitly, and
`ops/flash.py:legal_blocks` filters tilings against the *measured* budget.

**4. `triton_global_not_constexpr`.** A Triton kernel cannot read a plain
module-level global; it must be `tl.constexpr`. The system prompt warns about it
explicitly and the model still produces it — some failure modes are properties of
the framework, not the author.

**5. Difficulty tracks structure, not size.** Elementwise-with-broadcast (`gelu`)
succeeds readily; fused reduction with masking (`layernorm`) needs a much tighter
contract. The dividing line is whether correctness depends on a *global* property
of the data layout.

## Process isolation

An illegal memory access in a generated kernel corrupts the CUDA context: the
launch returns cleanly, the error surfaces asynchronously at a later unrelated
call, and every subsequent CUDA operation in the process fails. `try/except`
cannot recover it.

Each candidate is therefore validated in a throwaway subprocess
(`agent/codegen_worker.py`). A crash costs one kernel, not the run.

## Trust model

Generated source is imported and executed. That is not sandboxed. Three
properties make it defensible:

1. It runs at **build time only**, never in the submitted inference path.
2. Every candidate is written to disk **before** it runs.
3. **Nothing generated is in the shipped dispatch table.** Promoting one is a
   deliberate human step.

## Repair loop

`--repair N` feeds a failed kernel back with its compiler diagnostic and a
structural diagnosis of numeric failures — not just "envelope 256050" but:

> the error is nearly CONSTANT within each row, which means the per-row
> statistics (mean/variance) are wrong — most likely reduced over a tile of the
> feature axis instead of the whole row

Three independent A/B replicates at equal attempt budget, same targets, same
model (`qwen3-coder-next`):

| replicate | `--repair 0` | `--repair 3` |
|---|---|---|
| n=20 per arm | 8/20 | **12/20** |
| n=28 per arm | **13/28** | 12/28 |
| n=20 per arm | **13/20** | 7/20 |
| **pooled** | **34/68 (50%)** | 31/68 (46%) |

Individual replicates disagree — the first favours repair, the other two favour
resampling — but pooled over 68 attempts per arm, **resampling is ahead**, and
`--repair 0` is the default on the measurement rather than on caution. The
replicate spread (8/20, 13/28, 13/20 for the same arm) is itself the useful
number: it is wide enough that any single n=20 comparison here is uninformative.

All arms use `qwen3-coder-next`, so each replicate is internally matched; the
result has not been reproduced on `qwen3.8:27b`.
`results/codegen_rep3_repair{0,3}_sm_80.json` and
`results/codegen_repair3_sm_80.json` hold the arms.

Repair loops stall — the model can return the same `TypeError` repeatedly,
burning the budget on one lineage. `error_signature()` fingerprints a failure
with line numbers stripped; a repair reproducing its parent's signature abandons
the lineage and samples fresh.
