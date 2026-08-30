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
`ornith1.5:35b`. Measuring the actual task instead — every model, 20 kernels
each, same targets, same gate:

| model (as served) | valid plan JSON | correct kernels |
|---|---|---|
| **`qwen3.8:27b`** | 75% | **16/20 (80%)** — also 15/20, 17/20 on further samples |
| `qwen3-coder-next` | 100% | 8/20 (40%), 9/20 on a second sample |
| `qwen3.6:35b` | 88% | 6/20 (30%) |
| `gemma4:26b` | 100% | 5/20 (25%) |
| `ornith1.5:35b` | 100% | 0/20, 3/20 on a second sample |
| `qwen3.5:9b` | 100% | 0/20 — 17 syntax errors |
| `llama3.1:8b` | 75% | 0/20 |

**The JSON benchmark is anti-predictive.** It ranked `ornith1.5:35b` first; that
model wrote zero working kernels. `qwen3.8:27b`, ranked last on format, wrote the
most. We ship **`qwen3.8:27b`**.

Two measurement defects, both fixed:

- **The gateway aliases model ids.** Ten advertised ids resolve to seven distinct
  models — `qwen3.6:27b` is served by `qwen3.8:27b`, `ornith1.0:35b` by
  `ornith1.5:35b`, `default` by `qwen3.6:35b`. Mapping in
  `results/model_aliases.json`; every artifact now records the served id.
- **`HTTP 429` was being recorded as a model's score.** Three arms returned it on
  every request after earlier arms drained the key quota. The client now backs
  off to a two-minute ceiling and honours `Retry-After`; unanswered requests are
  logged as `api_error`. Those arms were re-run.

The double-measurements bound the noise: 15 vs 16 vs 17 of 20 for the winner,
0 vs 3 for `ornith1.5:35b`. The gaps among the bottom three are within that
spread and their order is not defended.

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

**2. Under-specification, not incapability.** The first contract said "one Triton
kernel, one pass over memory" but not *do not split the reduction axis*. Adding:

> CRITICAL — do not split the d axis. mean and variance are per-row statistics
> over ALL d elements, so a single program must reduce a whole row.

moved that target from **0/5 to 5/12**.

**3. `shared_memory_overflow`.** A tiling fine on an A100 does not launch on a
99 KB shared-memory budget. The spec sheet states the limit explicitly, and
`ops/flash.py:legal_blocks` filters tilings against the *measured* budget.

**4. `triton_global_not_constexpr`.** A Triton kernel cannot read a plain
module-level global; it must be `tl.constexpr`. We hit this on our own flash
kernel, warned about it in the system prompt, and the model still produced it
once.

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

A/B at equal attempt budget, same targets, same model (`qwen3-coder-next`), run
twice:

| run | `--repair 0` | `--repair 3` |
|---|---|---|
| A100, n=20 per arm | 8/20 | **12/20** (repairs worked 4/6) |
| earlier, n=28 per arm | **13/28** | 12/28 (repairs worked 33%) |

**Opposite answers.** Neither n resolves a gap that size, so the default stays at
`--repair 0`. The ablation predates the model bake-off and used
`qwen3-coder-next`; both arms use it, so the comparison is internally matched,
but it has not been repeated on `qwen3.8:27b`.
`results/codegen_repair3_sm_80.json` holds the repair arm.

Repair loops stall — the model can return the same `TypeError` repeatedly,
burning the budget on one lineage. `error_signature()` fingerprints a failure
with line numbers stripped; a repair reproducing its parent's signature abandons
the lineage and samples fresh.
