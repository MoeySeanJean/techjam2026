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

## Where the LLM actually reaches the shipped artifact

The model contributes in two places, and they carry different risk, so they are
wired in differently.

**Plans — the model competes for slots in the frozen table.** `cli agent
--provider llm` runs *after* `cli tune`, so every shape already holds the best
configuration the heuristic search could find. The model proposes against that,
each proposal is gated and timed by the same harness, and it is promoted only if
it is **measurably faster than the plan already frozen**:

```
-> promoted into the table (7.41x vs 7.02x held)
-> not promoted: 3.18x does not beat the 5.20x already frozen
```

A proposal that merely clears the accuracy gate takes nothing. This matters
because `DispatchTable.add` supersedes a same-case row outright — correct for a
re-measurement, wrong here — so without the check, running the agent after the
search would have replaced better plans with worse ones purely by running order.
`tests/test_dispatch.py` pins the rule.

A plan is a configuration, not code: which stages run in fp16, whether the QKV
projection is fused, which flash tiling, whether a CUDA graph is captured. Every
one of those is checked by the gate, so there is no reason to keep a winning
proposal out of the table, and good reason to let it in — in the controlled
head-to-head the LLM found the fastest gate-passing plan on 3 of 4 shapes.

**Generated kernel source is different, and is not shipped.** See the trust
model below. The asymmetry is deliberate: a configuration the gate accepts is as
safe as any other configuration, while Triton source no person has read is not,
however well it measures.

## What the model actually won, across nine GPUs

Every card was tuned by the search first, then the agent proposed against the
frozen table with promotion gated on beating the incumbent by 3%. Per card the
model won roughly 8 or 9 slots out of 13 or 14 proposals, and the rejections are
logged beside them:

```
-> promoted into the table (8.66x vs 3.92x held)
-> not promoted: 1.20x does not beat the 2.31x already frozen
```

All nine GPUs measured faster after the agent round than before it. That is the
useful claim, and it comes with a caveat we would rather state than bury: this
round also introduced the `torch.compile` selection margin, so the improvement
cannot be attributed to the model alone.

**The promotion decision is noisier than the 3% margin suggests.** The agent times
one proposal in isolation; the search times many interleaved. Comparing what the
agent recorded against what the organizer's benchmark then measured for the same
shipped plan: median 1.02x, but a 0.15x-2.16x range, with only 30 of 89 within
10%. Unbiased on average, unreliable per shape -- so some promotions are real
wins and others are measurement luck. The structure is what keeps that safe: a
proposal is gated for correctness before it is timed at all, so the worst a noisy
promotion can do is ship something slower, never something wrong. Re-timing the
winner in the incumbent's harness before promoting would fix it properly, and we
did not do that. (`results/agent_measurement_noise.json`)

## Trust model

Generated source is imported and executed. That is not sandboxed. Three
properties make it defensible:

1. It runs at **build time only**, never in the submitted inference path.
2. Every candidate is written to disk **before** it runs.
3. **No generated kernel source is in the shipped dispatch table.** Promoting
   one is a deliberate human step. This applies to the *code* the model writes;
   the plans it proposes are gated configurations and do enter the table when
   they win, as described above.

`--repair N` re-prompts with the failure diagnostic instead of sampling a fresh
kernel. At equal budget it does not help — both arms reach 13/20, and repaired
kernels succeed 1/4 against 12/16 for fresh samples — so the default is
`--repair 0`.
