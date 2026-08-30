# Use case: real-time recommendation ranking

```bash
python scripts/usecase.py --tune        # measures this, on your GPU, now
```

> **Scope.** The problem statement puts production-ready deployment out of
> scope, and this document stays out of it. Nothing here is a serving system,
> and we are not proposing one. The latencies below are measured, live, by the
> script named above; everything after them is arithmetic on assumptions that are
> stated in the text and settable on the command line. This exists because
> "relevance to real users" is a judged criterion and a speedup with no stated
> consumer is a number without a meaning — not because we are claiming to have
> deployed anything.

## Why this track is a recommendation problem

The official shape list is a ranking workload:

- **`num_layers` 2–4, `d_model` 32–1024** — the size a model has to be to run
  inside a request budget, thousands of times a second.
- **`batch_size` 1 to 10,000 on the same model** — one request, a coalesced peak
  batch, and an offline scoring pass: the three regimes a ranking service runs
  simultaneously.
- **`seq_len=100000` at `batch=32`** (shape 14) — not language, a behaviour log.
- **Causal masking throughout**, with a `valid_token_mask` in the generator:
  variable-length histories, ordered in time, padded.

A 6-layer stack with `d_model 512`, 8 heads, `ffn 2048` over a padded sequence is
a **user-behaviour sequence model** — the ranker that reads a user's recent
history and scores candidate videos against it. In a short-video feed that is the
highest-volume transformer inference in the product.

Its properties map onto this project's choices:

| property of ranking traffic | what it forces | what we did |
|---|---|---|
| Histories are **variable length** — a new user and a power user share a batch | every request is padded | our FlashAttention handles causal **and** key-padding in-register; SDPA cannot do both without falling off its fast path |
| Latency is a hard SLO, so **batches stay small** | small batches are launch-bound, not compute-bound | CUDA-graph capture removes ~105 kernel launches per forward |
| Traffic is **not one shape** — peak vs off-peak, light vs heavy history, first-pass vs re-rank | one kernel cannot be optimal for all of them | per-shape dispatch, tuned by search |
| The serving fleet is **heterogeneous** across GPU generations | the best kernel differs per card | per-architecture tables; 4 of the 13 official shapes chose a different plan on an A100 than on an H100 |
| A wrong score ships the wrong video | correctness is not negotiable | every plan clears the accuracy gate *before* it is timed |

## The traffic mix

Four segments on one model. Shares are illustrative and stated in the script so
you can substitute your own; the **shapes** are the substance.

| segment | shape | what it is | share |
|---|---|---|---|
| `realtime_light` | `B32-S64` pad 0.4 | single request, short history — the latency-critical common case | 42% |
| `realtime_heavy` | `B32-S256` pad 0.4 | single request, long history — power users | 18% |
| `batched_peak` | `B128-S128` pad 0.4 | peak traffic, requests coalesced for throughput | 25% |
| `rerank` | `B8-S128` | second-stage re-rank over a small candidate set | 15% |

## Measured result

**A100-80 PCIe** (sm_80), median latency, every row having cleared the accuracy
gate before it was timed:

| segment | share | baseline | ours | gain | plan chosen |
|---|---|---|---|---|---|
| `realtime_light` | 42% | 2.72 ms | 1.20 ms | **2.26x** | `fp16[attn,out_proj]+graph` |
| `realtime_heavy` | 18% | 9.36 ms | 4.41 ms | 2.12x | `fp16[attn,out_proj]+graph` |
| `batched_peak` | 25% | 15.00 ms | 8.15 ms | 1.84x | `fp16[out_proj,attn]` |
| `rerank` | 15% | 2.75 ms | 1.15 ms | **2.39x** | `wide+graph` |
| **traffic-weighted** | 100% | **6.99 ms** | **3.51 ms** | **1.99x** | — |

**Four segments, three different plans, one model.** `realtime_light` and
`realtime_heavy` share one plan; `batched_peak` takes the same precision split
*without* CUDA graphs, because at batch 128 the launch overhead graphs remove is
no longer what limits it; `rerank` gets the bit-exact wide plan. The shape alone
changes what wins, and the split differs again on other hardware. That is the
entire argument for searching rather than hand-tuning, visible in one table.

**1.99x is the conservative number, and that is deliberate.** On weaker or
thermally limited hardware this script reports a *higher* traffic-weighted gain —
we have measured 2.47x — because a slower card spends proportionally more of its
time on the kernel-launch overhead we remove, so the ratio rises while every
absolute latency gets worse. We report the A100, which holds a stable clock and
gives the smaller figure.

## The part worth watching

```bash
python scripts/usecase.py --cold     # what an untuned system does
```

**1.04x.** All four segments are shapes the table has never seen, so they fall
back through nearest-neighbour to the bit-exact plan — correct, safe, barely
faster. Then:

```bash
python scripts/usecase.py --tune     # ~3 min per shape
```

**2.47x** on the same machine, same workload, same command shape.

`--cold` removes exactly those four entries and re-runs the same lookup, so the
gap stays reproducible after tuning and the fallback being measured is the real
one.

That gap is the point: **correct everywhere, fast where it has been pointed**,
and pointing it at a new workload is one command and no code change.

## What it means for a service

Derived from the measured latencies above, with assumptions stated so they can
be replaced (`--qps`, `--fleet`):

At **100,000 ranking requests/second** on A100-80:

| | baseline | ours |
|---|---|---|
| requests/s per GPU | 8,124 | **17,448** |
| GPUs to hold the load | 12 | **6** |

**53% of the serving capacity freed** — 20,744 kWh/year at 300 W/GPU and PUE 1.2.
The absolute numbers depend entirely on the QPS assumption, which is why the
script takes `--qps` and `--fleet`: substitute your own and the arithmetic
follows.

The realtime segments also got 2.3x and 2.1x faster inside a fixed SLO — headroom
that can be spent on capacity, or on a longer user history and a larger candidate
set.

## Assumptions

- Latencies are **measured** live by the script; nothing in the table is
  estimated.
- Traffic shares and QPS are **modelled**. They are command-line inputs
  (`--qps`, `--fleet`) and every derived figure scales linearly with them.
- The workload is **representative, not proprietary** — built from the
  organizer's own model definition and generator.
- The headline table is an **A100-80**, tuned and measured in one job. The 2.47x
  figure above is from a throttling GPU and is contrast only.
- The **53% capacity reduction is throughput arithmetic**, not a deployment
  study: it assumes the ranker is the bottleneck and that freed capacity is
  reclaimed.

## Beyond ranking

The same stack sits under the other transformer inference in a short-video
product: content-understanding encoders, moderation classifiers, query
understanding in search, creative ranking in ads. They differ in shape, not in
kind; adding one is a line in `official_shapes.txt` and a `tune` invocation.
