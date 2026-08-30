# AI-generated kernels

The model writes complete Triton source against a contract. Every candidate is
compiled, gated against an exact reference, and timed by the same harness that
governs everything else.

```bash
python -m kernelforge.cli codegen --targets layernorm,gelu --iterations 10
```

Generated modules are saved to `results/generated/` before they run; per-attempt
verdicts are in `results/codegen.json`. Runs below are on the A100-80 node
`xgph1` with `qwen3.8:27b`, the model we ship.

## Targets

| target | contract | reference it must match |
|---|---|---|
| `gelu` | fused bias-add + exact erf GELU | `F.gelu(x.float() + bias.float(), approximate="none")` |
| `layernorm` | fused residual-add + row mask + LayerNorm | our block-boundary kernel |

## Results

| target | ok | how the rest failed | best generated kernel |
|---|---|---|---|
| `layernorm` | **10/10** | — | **2.46x** vs torch, envelope 0.042, 55 lines |
| `gelu` | **7/10** | 3 `numeric_fail` | **3.91x** vs torch, envelope 0.031, 67 lines |

Successful `gelu` attempts cluster at 3.59–3.91x.

**Model choice is measured.** Every model the gateway serves was run through this
loop; correct-kernel counts ranged from 48/60 down to 0/40, and `qwen3.8:27b`
won. `scripts/pick_model.py` scores plan-JSON quality instead, and on this
gateway that was *anti-predictive* of kernel-writing ability, so it is not what
selects the model. Arms are committed as `results/codegen_<model>_<arch>.json`.

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
substitution on both GPUs. Only the gate catches it.

**2. The contract does the work.** This clause is worth **0/5 → 5/12** on
`layernorm`:

> CRITICAL — do not split the d axis. mean and variance are per-row statistics
> over ALL d elements, so a single program must reduce a whole row.

**3. Hardware limits must be in the prompt.** A tiling fine on an A100 does not
launch on a 99 KB shared-memory budget. The spec sheet states the limit, and
`ops/flash.py:legal_blocks` filters tilings against the *measured* budget.

**4. Some failures are framework properties.** A Triton kernel cannot read a
plain module-level global; it must be `tl.constexpr`. The system prompt warns
about this and the model still produces it.

**5. Difficulty tracks structure, not size.** Elementwise-with-broadcast (`gelu`)
succeeds readily; fused reduction with masking (`layernorm`) needs a much tighter
contract. The dividing line is whether correctness depends on a *global* property
of the data layout.

## Process isolation

An illegal memory access in a generated kernel corrupts the CUDA context: the
launch returns cleanly, the error surfaces asynchronously at a later unrelated
call, and every subsequent CUDA operation in the process fails. `try/except`
cannot recover it. Each candidate is validated in a throwaway subprocess
(`agent/codegen_worker.py`), so a crash costs one kernel, not the run.

## Trust model

Generated source is imported and executed. That is not sandboxed. Three
properties make it defensible:

1. It runs at **build time only**, never in the submitted inference path.
2. Every candidate is written to disk **before** it runs.
3. **Nothing generated is in the shipped dispatch table.** Promoting one is a
   deliberate human step.

`--repair N` re-prompts with the failure diagnostic instead of sampling a fresh
kernel. At equal budget it does not help — both arms reach 13/20, and repaired
kernels succeed 1/4 against 12/16 for fresh samples — so the default is
`--repair 0`.
