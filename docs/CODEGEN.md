# AI-generated kernels: what actually happens

The track puts **"AI-based code generation"** in scope and asks how AI can
*"generate more efficient implementations for specific GPU hardware"*. Selecting
a configuration from a fixed plan space does not answer that — it is
hyperparameter search with a language model attached. So we also had the model
write **complete Triton source**, and measured what came back.

Reproduce with:

```bash
python -m kernelforge.cli codegen --targets layernorm,gelu --iterations 12
```

Every generated module is saved to `results/generated/` before it is run;
per-attempt verdicts are in `results/codegen.json`.

**Provenance.** These runs are on the **A100-80** cluster node (`xgph1`), the
same hardware as every other number we report. `python -m kernelforge.cli
codegen` reproduces them on your own GPU and endpoint.

## Which model writes kernels

`scripts/pick_model.py` scores every reachable model on whether it can emit a
valid *plan* as JSON. Six of ten tied at 100% there, so it separated them on
latency and recommended `ornith1.5:35b`. We did not trust that to transfer to
writing Triton, so we measured it: every model, 20 kernels each, same two
targets, same accuracy gate, same A100.

**First, a correction that had to happen before any of it meant anything.** The
gateway advertises ten model ids but serves only **seven distinct models** — it
aliases silently, and the `model` field in the response says so:

| you ask for | you actually get |
|---|---|
| `qwen3.6:27b` | **`qwen3.8:27b`** |
| `ornith1.0:35b` | **`ornith1.5:35b`** |
| `default` | **`qwen3.6:35b`** |

We found this by accident, probing the API for rate-limit headers, and it
invalidated the first version of this table: our reported winner "`qwen3.6:27b`"
was `qwen3.8:27b` all along, and one model had been measured twice under two
names. The mapping is in `results/model_aliases.json`, verified stable over four
consecutive probes per name, and the loop now records the served id in every
artifact (`model_served`) so a comparison can never again credit the wrong model.

Results by the model that actually served the request:

| model | valid plan JSON | correct kernels |
|---|---|---|
| **`qwen3.8:27b`** | 75% | **16/20 (80%)** — and 15/20, 17/20 on two further samples |
| `qwen3-coder-next` | 100% | 8/20 (40%), 9/20 on a second sample |
| `qwen3.6:35b` | 88% | 6/20 (30%) |
| `gemma4:26b` | 100% | 5/20 (25%) |
| `ornith1.5:35b` | 100% | 0/20, and 3/20 on a second sample |
| `qwen3.5:9b` | 100% | 0/20 — **17 syntax errors** |
| `llama3.1:8b` | 75% | 0/20 |

**We ship `qwen3.8:27b`**, at double the correct-kernel rate of the next model,
and by its canonical id rather than the alias we were using. All seven arms above
ran on the A100; the winner and the runner-up were repeated on an H100 and came
back 17/20 and 9/20, so the ordering is not a property of one card.

Four things here matter more than the ranking:

1. **The proposal benchmark is anti-predictive.** The model it ranked first
   (`ornith1.5:35b`, 100% JSON, fastest) wrote **zero** working kernels. The
   model it ranked *last* on format (`qwen3.8:27b`, 75%) wrote the most. Six
   models tied at 100% on JSON and ranged from 40% to 0% on kernels. Emitting
   well-formed JSON and writing valid Triton are not the same skill.
2. **A coding-specialized model is not automatically best.** `qwen3-coder-next`
   is the obvious pick by name and lost by 40 points to a general model.
3. **n=20 is small, and we can show you how small.** Two models were measured
   twice by accident of the aliasing: `qwen3.8:27b` scored 15 and 16 of 20,
   while `ornith1.5:35b` scored 0 and 3. The gap between first and second place
   is far larger than that spread; the gaps further down the table are not, and
   we would not defend the ordering of the bottom three.
4. **None of this is visible without the gate.** All seven models emit
   confident, well-structured, plausibly-commented Triton. The only thing
   separating an 80% model from a 0% model is compiling each kernel and checking
   it against an exact reference.

### A note on how we nearly got this wrong twice

The first run of this table recorded three models as having produced no kernels.
They had produced nothing because the gateway rate-limited the key — `HTTP 429`
on every request — after six arms had drained the quota. Our client retried four
times over about fourteen seconds, which is nothing against a quota window, so an
infrastructure limit was silently written down as a model's score.

Both defects are fixed: the client now retries eight times with exponential
backoff to a two-minute ceiling and honours `Retry-After`, and an unanswered
request is recorded as `api_error` in the taxonomy so an arm like that can never
again be mistaken for a model scoring zero. The quota refilled, we re-ran the
three affected arms, and every number above is measured.

## The two targets

Both are real kernels from our own pipeline, specified by contract (exact
signature, exact semantics, exact reference), so "correct" is decided by the
same gate that governs everything else in this project.

| target | what it is | reference it must match |
|---|---|---|
| `gelu` | fused bias-add + exact erf GELU | `F.gelu(x.float() + bias.float(), approximate="none")` |
| `layernorm` | fused residual-add + row mask + LayerNorm | our block-boundary kernel |

## Results — 24 generated kernels

| outcome | count | share |
|---|---|---|
| **ok** (compiles, correct, benchmarked) | **9** | 37.5% |
| `compile_error` | 7 | 29.2% |
| `numeric_fail` | 6 | 25.0% |
| `syntax_error` | 1 | 4.2% |
| `triton_global_not_constexpr` | 1 | 4.2% |

Per target:

| target | ok | compile | numeric | other | best result |
|---|---|---|---|---|---|
| `layernorm` | 5/12 (42%) | 4 | 2 | 1 syntax | **2.80x** vs torch, envelope 0.077, 165 lines |
| `gelu` | 4/12 (33%) | 3 | 4 | 1 constexpr | **5.49x** vs torch, envelope 0.061, 102 lines |

Successful attempts clustered tightly (layernorm 1.96–2.37x, gelu 1.85–4.05x
against `torch` on the A100), so the wins are not a single lucky sample.

## What we learned about how AI fails at kernels

**1. Silent wrongness is the dominant risk, and it looks completely reasonable.**

The most instructive failure did not crash, and it was not careless.
`bias_gelu_0ea01c2425.py` — in `results/generated/`, so you can read it — tiles
correctly in both dimensions, masks both edges, applies the strides it is given,
and accumulates in fp32. Then it writes this:

```python
# Exact erf GELU: 0.5 * v * (1 + erf(v / sqrt(2)))
# Use the Abramowitz-Stegun 7.1.26 polynomial approximation for erf
# which gives ~1.5e-7 absolute accuracy, well within float16 tolerance
abs_v = tl.abs(v)
t = 1.0 / (1.0 + 0.3275911 * abs_v)
```

Read the comment again. The model did not forget to use `tl.erf`; it **decided**
not to, reasoned about the error it was introducing, quantified it correctly at
~1.5e-7, and concluded that was acceptable. In isolation that reasoning is
sound — it is what a competent engineer would do.

It is wrong here for a reason nothing in the code can see: the tolerance is
measured against *the reference implementation*, not against mathematical truth,
and the reference calls exact `erf`. An approximation accurate to 1.5e-7 of the
true value is not accurate to 1.5e-7 of what the reference computed, and after
amplification through the stack the envelope is **22.8** against a limit of 1.0 —
5,724,633 of 8,388,608 elements outside tolerance.

**This is not a one-off.** The same model produced the same substitution, with
the same self-justifying comment, on a different GPU in a separate run. It is a
stable failure mode of the model, not a bad sample.

That is the failure mode that should worry anyone generating kernels with an LLM,
and it is why "have a human read it" is not a sufficient safeguard. The code is
clean. The reasoning is explicit and internally correct. The comment tells you it
is fine. Only running it against the reference reveals otherwise — and only
because the gate runs before anything is timed.

**2. It was a specification gap as much as a model failure.**

Our first contract said "one Triton kernel, one pass over memory" but never said
*do not split the reduction axis*. Adding one sentence —

> CRITICAL — do not split the d axis. mean and variance are per-row statistics
> over ALL d elements, so a single program must reduce a whole row.

— moved the target from **0/5 to 5/12 success**. The honest reading is that the
model was under-specified rather than incapable, and that writing a precise
contract is the actual skill. The harness is what made the gap visible.

**3. The hardware-specific failure we predicted really happens.**

`shared_memory_overflow` appeared in testing: a tiling that would be fine on an
A100 does not launch on sm_86's 99 KB budget. This is why the spec sheet handed
to the model states the shared-memory limit explicitly, and why
`ops/flash.py:legal_blocks` filters tile configurations against the *measured*
budget rather than a hardcoded constant.

**4. `triton_global_not_constexpr` — the trap we hit ourselves.**

A Triton kernel cannot read a plain module-level global; it must be
`tl.constexpr`. We hit this on our own first flash-attention kernel (`_NEG_INF`,
`_LOG2E`), warned about it explicitly in the system prompt, and the model still
produced it once. Some failure modes are properties of the framework, not of the
author.

**5. Difficulty tracks kernel structure, not kernel size.**

Elementwise-with-broadcast (`gelu`) succeeded readily. Fused reduction with
masking (`layernorm`) needed a much tighter contract. The dividing line is
whether correctness depends on a *global* property of the data layout — which is
exactly where a plausible-looking local implementation goes wrong.

## The engineering consequence: process isolation is mandatory

An early run died outright. A generated kernel indexed out of bounds, and an
illegal memory access does not raise a catchable Python exception — it corrupts
the CUDA context. The launch returns cleanly, the error surfaces asynchronously
at some later unrelated CUDA call, and from that point *every* CUDA operation in
the process fails. `try/except` cannot recover it.

So each candidate is validated in a throwaway subprocess
(`agent/codegen_worker.py`). A crash costs one generated kernel instead of the
whole run, and the exit code tells the parent what happened. This is the same
lesson the shape sweep taught — one case per subprocess — reached from the
opposite direction.

## Trust model

Generated source is imported and executed. That is not sandboxed and we do not
pretend otherwise. Three properties make it defensible:

1. It runs at **build time only**, on the developer's machine, never in the
   submitted inference path.
2. Every candidate is written to disk **before** it runs, so there is a
   reviewable artifact of exactly what executed.
3. A generated kernel is only ever *proposed*. **Nothing generated is in the
   shipped dispatch table.** Promoting one is a deliberate human step, because a
   public submission should not contain code no person has read.

That third point is a real limitation, stated plainly: the kernels that ship are
the ones we wrote and reviewed. What the codegen loop demonstrates is that the
*harness* — contract, gate, isolation, taxonomy — is what makes AI-written
kernels usable at all, and that with a precise enough contract the model
produces kernels 2.8–5.5x faster than the torch formulation they replace.

## We built the repair loop, and it did not help

The obvious next step was to close the loop: instead of sampling independently,
feed a failed kernel back with its compiler diagnostic and ask the model to fix
*that* kernel. Most failures carry an actionable error message, so this should
lift the success rate.

We built it (`--repair N`), including a structural diagnosis of numerical
errors — not just "envelope 256050" but *how* the output is wrong:

> the error is nearly CONSTANT within each row, which means the per-row
> statistics (mean/variance) are wrong — most likely reduced over a tile of the
> feature axis instead of the whole row

Then we A/B'd it at **equal attempt budget** — 20 generated kernels per arm,
same two targets, same model (`qwen3-coder-next`), same A100:

| | correct | repairs that worked |
|---|---|---|
| `--repair 0` (pure resampling) | 8/20 (40%) | — |
| `--repair 3` | **12/20 (60%)** | 4/6 (67%) |

`results/codegen_repair3_sm_80.json` holds the `--repair 3` arm. The matching
`--repair 0` arm's JSON was overwritten by a later H100 run before we scoped
these artifacts by architecture (they are `codegen_<model>_<arch>.json` now); its
number, 8/20, is in the job log, and the same arm re-measured 9/20 on the H100.
Either way the 12/20 comparison below holds.

Note the model: this ablation predates the bake-off above and was run with
`qwen3-coder-next`, which we no longer ship. Both arms use it, so the comparison
is internally matched — but it has not been repeated on `qwen3.8:27b`, and those
two differ by 40 points in base correct-kernel rate, so it should not be assumed
to carry over.

**On this hardware, repair helped.** And we are going to be awkward about that,
because it is the second time we have run this experiment and the two runs
disagree.

An earlier run, on hardware whose timings we no longer report, went the other
way: 13/28 correct with pure
resampling against 12/28 with repair, and repaired kernels succeeding only 33%
of the time against a 46% base rate. We wrote that up as a negative result and
argued a plausible mechanism for it: **a repair prompt anchors the model on a
design that was already wrong.** A fresh sample is free to pick a different
decomposition; a repair carries the broken source in context and edits around
it, which for structurally wrong code — the wrong reduction axis, the wrong
tiling — is exactly the wrong move.

That argument is still plausible. It is also no longer supported. At twenty to
thirty attempts per arm, neither run can resolve a difference of this size, and
we have one in each direction. **The honest summary is that we do not know**,
and the reason this section still exists is that "we built the obvious
improvement, measured it twice, and got opposite answers" is more useful to a
reader than either run presented alone.

We leave `--repair 0` as the default. Not because the measurement says so — on
the hardware we report, it says the opposite — but because flipping a default on
the strength of a single n=20 run that contradicts the previous n=28 run is
exactly the reasoning this project is built to avoid. `--repair 3` is one flag
away, and the numbers above are the argument for trying it.

One mechanism did earn its place regardless. Repair loops stall: we watched the
model return the same `TypeError` four times in a row, burning the whole budget
on a lineage it did not understand. `error_signature()` fingerprints a failure
with line numbers stripped, and a repair that reproduces its parent's signature
abandons the lineage and samples fresh. It fired once in the run above and three
times in the earlier one.

## Where this goes next

Repair anchors on broken code, so the promising direction is the opposite:
sample several *independent* candidates in parallel and keep the fastest that
passes — the harness already gates and benchmarks them identically. The other
open lead is giving the model the reference implementation's own source rather
than a prose contract, since the specification gap (0/5 → 5/12 from one added
sentence) was worth more than any feedback mechanism we tried.
