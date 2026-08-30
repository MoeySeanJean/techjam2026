"""Official test shape #14 -- the one the reference implementation cannot run.

    B=32, S=100000, d=1024, H=16, L=2, causal, ffn=1024

`BaselineSelfAttention` materializes the score matrix `[B, H, S, S]` before
softmax. For this shape that tensor is

    32 x 16 x 100000 x 100000 x 4 bytes = 18.6 TB

which does not fit on any GPU, or any machine. The organizer's own baseline
therefore cannot execute shape 14, and neither can any implementation that
forms the score matrix -- including `torch.compile` applied to that baseline.

Our implementation never materializes it: the FlashAttention kernel streams K
and V tiles and keeps only a running softmax, so its attention memory is O(S),
not O(S^2). For this shape a fused kernel is not an optimization, it is the
precondition for running at all.

That removes the 18.6 TB but not the whole problem. Two more things are needed.

**One.** Even with O(S) attention, a single activation tensor here is

    32 x 100000 x 1024 x 4 bytes = 12.2 GB

and the forward keeps tens of them alive -- hundreds of GB of working set, on a
shape whose *per-sample* working set is under 13 GB. So the batch is streamed:
nothing in this model mixes batch elements, so the batch dimension is sliced and
results are written into one output buffer.

**Two.** The SDPA fallback -- used when the attention stage stays in fp32,
because Triton's `tl.dot` needs a narrow float type -- accepts `is_causal` OR an
`attn_mask`, never both, so a causal-plus-padding shape makes it build the mask
by hand as an `[S, S]` tensor: 37.25 GiB here, reintroducing the exact quadratic
term the flash kernel removes. When no token is padded the padding mask is a
no-op and `is_causal` alone is exact, which is the case for all 14 official
shapes; taking that path costs 45.9 GB peak instead of 84.6 GB.

Both are required. Fused attention alone will not run shape 14.

Measured:

    A100-80 PCIe   77.7 s   peak 45.9 GB   batch sliced 1 at a time
    H100 NVL 93GB  54.5 s   peak 45.9 GB   batch sliced 2 at a time

What this script does, and what it deliberately does not claim:

  * it runs OUR implementation on shape 14 and reports latency and peak memory;
  * it reports the baseline as *not runnable*, with the arithmetic above;
  * it does NOT report a speedup, because a ratio against something that cannot
    run is not a number;
  * accuracy for this code path is established at sequence lengths where the
    reference *can* be computed -- see `tests/test_kernels.py`, which checks the
    same kernel against an exact reference across causal x padding x lengths.

    python scripts/shape14.py                # full shape (needs ~60 GB)
    python scripts/shape14.py --scan         # how far this GPU gets
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import torch_transformer_benchmark as B  # noqa: E402
from kernelforge import bench, shapes  # noqa: E402
from kernelforge.dispatch import DispatchTable  # noqa: E402
from kernelforge.hw import probe  # noqa: E402
from kernelforge.optimized import build_shared  # noqa: E402

SHAPE = "B32-S100000-d1024-H16-F1024-L2-causal"


def baseline_score_bytes(case) -> int:
    """What the reference would have to allocate for its [B, H, S, S] scores."""
    return (case.batch_size * case.num_heads * case.seq_len * case.seq_len * 4)


def run_one(case, device, spec, table, verbose=True):
    cfg = case.to_config()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Weights only -- we never call the baseline's forward for this shape.
    base = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    plan, source = table.lookup(spec.arch, case.torch_dtype, cfg,
                                spec.shared_mem_per_block_kb)
    ours = build_shared(cfg, plan, base)
    x, mask = B.generate_random_case(cfg, device, case.torch_dtype, 1234,
                                     case.padding_ratio, case.input_scale)
    if verbose:
        free, total = torch.cuda.mem_get_info(device)
        print(f"{'':14}before forward: {free / 2**30:,.1f} GB free of "
              f"{total / 2**30:,.1f}; batch slice = {ours._batch_chunk(x)}"
              f"/{cfg.batch_size}", flush=True)
    with torch.inference_mode():
        out = ours(x, mask)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all())
        shape_ok = tuple(out.shape) == (cfg.batch_size, cfg.seq_len, cfg.d_model)
        # Release the first output before timing. At this shape it is 12.2 GB,
        # and holding it while each timed call allocates its own was enough on
        # its own to turn a run that fits into one that does not.
        del out
        torch.cuda.empty_cache()
        t = bench.compare({"ours": lambda: ours(x, mask)},
                          warmup=1, repeats=3, rounds=1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    ms = t["ours"].median_ms
    del base, ours, x, mask
    torch.cuda.empty_cache()
    return {"ms": ms, "peak_gb": peak, "finite": finite, "shape_ok": shape_ok,
            "plan": plan.name, "source": source}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="sweep sequence length upward to find this GPU's limit")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    spec = probe(measure=False)
    table = DispatchTable.load(spec.arch)

    case = shapes.resolve([SHAPE])[0]
    tb = baseline_score_bytes(case) / 2**40
    print(f"\n{'=' * 74}")
    print(f"Official test shape #14   {case.label()}")
    print(f"{'=' * 74}")
    print(f"  GPU                     {spec.name} [{spec.arch}], "
          f"{spec.total_mem_gb:.0f} GB")
    print(f"  reference [B,H,S,S]     {tb:,.1f} TB  -> CANNOT ALLOCATE, on any "
          f"machine")
    print("  our attention memory    O(S), not O(S^2) -- the score matrix is "
          f"never formed\n")

    targets = ([1024, 4096, 16384, 65536, 100000] if args.scan
               else [case.seq_len])
    for S in targets:
        import dataclasses
        c = dataclasses.replace(case, seq_len=S, name=f"S{S}")
        try:
            r = run_one(c, device, spec, table)
            print(f"  S={S:<7} {r['ms']:9.1f} ms   peak {r['peak_gb']:6.1f} GB   "
                  f"finite={r['finite']}  shape_ok={r['shape_ok']}   "
                  f"{r['plan']}")
        except torch.cuda.OutOfMemoryError as e:
            one = (c.batch_size * c.seq_len * c.d_model
                   * c.torch_dtype.itemsize / 2**30)
            print(f"  S={S:<7} OOM on {spec.total_mem_gb:.0f} GB "
                  f"(one activation tensor is {one:,.1f} GB in "
                  f"{str(c.torch_dtype).split('.')[-1]}; "
                  f"input+output alone need {2 * one:,.1f} GB)")
            # Which allocation failed, and how much was already held, is the
            # difference between "this shape does not fit" and "our chunking
            # did not engage". Print it rather than infer it.
            first = str(e).splitlines()[0] if str(e) else ""
            print(f"{'':14}{first[:150]}")
            import traceback
            for fr in traceback.extract_tb(e.__traceback__)[-3:]:
                print(f"{'':14}{fr.filename.split('/')[-1]}:{fr.lineno} {fr.line}")
            print(f"{'':14}held at failure: "
                  f"{torch.cuda.memory_allocated() / 2**30:,.1f} GB allocated, "
                  f"{torch.cuda.memory_reserved() / 2**30:,.1f} GB reserved")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  S={S:<7} {type(e).__name__}: {str(e)[:80]}")
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            for fr in tb[-3:]:
                print(f"{'':14}{fr.filename.split('/')[-1]}:{fr.lineno} {fr.line}")
            torch.cuda.empty_cache()

    print("\n  No speedup is quoted for this shape. A ratio against an "
          f"implementation that\n  cannot run is not a measurement. What is "
          f"true is narrower and stronger:\n  this shape is reachable with a "
          f"fused kernel and unreachable without one.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
