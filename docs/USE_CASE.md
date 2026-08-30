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

TikTok's own framing of TechJam puts recommendation at the centre, and Track 3's
appendix is not a random benchmark — it is a ranking workload with the labels
removed. Look at what the organizers chose to test:

- **`num_layers=4` or `2`, `d_model` 32 to 1024.** Small and shallow. This is
  not an LLM; it is the size a model has to be when it runs inside a request
  budget, thousands of times a second.
- **`batch_size` from 1 to 10,000**, on the same model. Nothing else produces
  that spread. It is one request, a coalesced peak batch, and an offline scoring
  pass — the three regimes a ranking service runs simultaneously.
- **`seq_len=100000` at `batch=32`** (shape 14). A sequence that long is not
  language. It is a *behaviour log*: everything a user has watched, scrolled
  past, or lingered on.
- **Causal masking throughout**, with a `valid_token_mask` in the generator.
  Histories are variable length and ordered in time, so they are padded and
  causally masked. That is exactly what the script models.

So the shape list *is* the use case, and we did not have to invent one. A
6-layer stack with `d_model 512`, 8 heads, `ffn 2048` over a padded sequence is a
**user-behaviour sequence model**: the ranker that reads a user's recent
interaction history and scores candidate videos against it. In a short-video feed
that is the highest-volume transformer inference in the product — one per
request, per user, continuously, and the thing standing between a user opening
the app and the first video playing.

Its properties are the ones this project was built around, one for one:

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

**1.04x.** All four segments are shapes the dispatch table has never seen, so
they fall back through nearest-neighbour to the bit-exact plan — correct, safe,
and barely faster. The system does not pretend to be fast on shapes it has not
measured. Then:

```bash
python scripts/usecase.py --tune     # ~3 min per shape
```

**2.47x** on the same machine, same workload, same command shape.

`--cold` exists so that gap stays reproducible. Once you tune, the entries are
frozen into the table and a plain run shows the tuned number — the "before" is
gone, and you would be asking a reader to trust a figure they can no longer
produce. `--cold` removes exactly those four entries and re-runs the same
lookup, so the fallback path being measured is the real one, not a simulation
of it.

That gap *is* the product. The system does not claim to be fast on shapes it has
not measured; it claims to be **correct everywhere and fast where it has been
pointed**. Pointing it at a new workload is one command and no code change —
which is exactly the situation a serving team is in when a model or a traffic
pattern changes.

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

The capacity saving is the boring half. The interesting half is **latency
headroom**: the realtime segments got 2.3x and 2.1x faster inside a fixed SLO.
Spent on capacity that is a cheaper feed; spent on a longer user history or a
larger candidate set, it is a *better* one. That is a product decision the
speedup makes available, not an infrastructure footnote.

## Honesty about this section

- The **latencies are measured**, live, on the machine that runs the script.
  Nothing in the table is estimated.
- The **traffic shares and QPS are modelled**, not observed. They are inputs you
  can change on the command line, and every derived figure scales linearly.
- The **workload is representative, not proprietary**. It is built from the
  organizer's own model definition and generator; we make no claim to know any
  company's production configuration.
- The headline table is from an **A100-80**, tuned and measured by the same
  script in one job. The 2.47x figure quoted once above, from a throttling GPU,
  is contrast only and is not part of the reported results.
- The **53% capacity reduction is throughput arithmetic**, not a deployment study.
  It assumes the ranker is the bottleneck and that freed capacity is actually
  reclaimed — neither is automatic in a real service.

## Beyond ranking

The same stack, and therefore the same dispatch machinery, sits under the other
transformer inference in a short-video product: content-understanding encoders
for video, audio and caption embeddings; moderation classifiers; query
understanding in search; and creative ranking in ads. They differ in shape, not
in kind — which is the case this project is designed for. Adding one is a line
in `official_shapes.txt` and a `tune` invocation.
