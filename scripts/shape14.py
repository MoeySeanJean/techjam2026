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

Measured: **20.9 s** on an A100-80 PCIe, **9.6 s** on an H100 NVL, ~46 GB peak.
The attention stage runs in fp16 so the flash kernel can take it; left in fp32 it
falls through to SDPA and costs 77.2 s and 54.5 s instead.

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
import json
import os
import sys
import time

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


CHUNK = 4096


def _ref_attention(q, k, v, chunk=None):
    """Causal attention streamed over key blocks: O(S) memory, exact result.

    The organizer's reference materializes `[B,H,S,S]` -- 18.6 TB for this shape.
    The *memory* is the obstacle, not the arithmetic: chunking the query rows and
    masking against the key index computes the same thing in O(S). This uses a
    plain two-pass `torch.softmax` rather than the online rescaling the kernel
    under test uses, so it differs in algorithm as well as in precision.
    """
    H, S, Dh = q.shape
    out = torch.empty_like(q)
    scale = Dh ** -0.5
    step = chunk or CHUNK
    for lo in range(0, S, step):
        hi = min(lo + step, S)
        s = torch.matmul(q[:, lo:hi], k[:, :hi].transpose(-2, -1)) * scale
        rows = torch.arange(lo, hi, device=q.device)[:, None]
        s = s.masked_fill(rows < torch.arange(hi, device=q.device)[None, :],
                          float("-inf"))
        out[:, lo:hi] = torch.matmul(torch.softmax(s, dim=-1), v[:, :hi])
        del s
    return out


def _ref_model(model, x, num_heads, chunk=None):
    """`BaselineTransformer.forward`, with the attention streamed."""
    h = x
    for layer in model.layers:
        a = layer.attention
        n = layer.norm1(h)
        M, d = n.shape
        Dh = d // num_heads
        q, k, v = (proj(n).view(M, num_heads, Dh).permute(1, 0, 2).contiguous()
                   for proj in (a.q_proj, a.k_proj, a.v_proj))
        ctx = _ref_attention(q, k, v, chunk).permute(1, 0, 2).reshape(M, d)
        del q, k, v
        h = h + a.out_proj(ctx)
        del ctx
        n2 = layer.norm2(h)
        h = h + layer.ffn_out(torch.nn.functional.gelu(layer.ffn_in(n2),
                                                       approximate="none"))
        del n2
    return model.final_norm(h)


def _time(fn, repeats):
    """Wall time of `fn`, best of `repeats`, with the GPU synchronized."""
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def race(device, spec, table, chunks=(1024, 2048, 4096, 8192), repeats=2):
    """Time shape 14 against a reference that can actually run it.

    The organizer's baseline cannot execute this shape -- it forms an 18.6 TB
    score matrix -- so the headline result has always been "we run it, they
    cannot", with no ratio attached. That is honest but incomplete: it says
    nothing about whether our implementation is *fast*, only that it exists.

    This closes that. The opponent is the organizer's own `BaselineTransformer`
    with exactly one change: the attention is chunked over query rows so its
    memory is O(S) instead of O(S^2). Same modules, same weights, same
    two-pass softmax, same fp32, and -- unlike the correctness gate, which
    deliberately runs stricter than the organizer does -- the organizer's own
    numerics policy, TF32 on at `matmul_precision="high"`. It is the minimum
    edit that makes the reference runnable, which makes it the fair opponent.

    Its chunk size is swept and the *best* time is taken, and it is timed
    *first*, on an unfragmented allocator, so that the large chunks get their
    best chance to fit. Racing a strawman would be easy and worthless.
    """
    import dataclasses as _dc

    case = shapes.resolve([SHAPE])[0]
    cfg = case.to_config()
    plan, source = table.lookup(spec.arch, case.torch_dtype, cfg,
                                spec.shared_mem_per_block_kb)

    # The organizer's defaults, for both sides.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"  shape            {SHAPE}")
    print(f"  our plan         {plan.name} [{source}]")
    print("  opponent         the organizer's model, attention chunked to O(S)")
    print("  numerics         fp32, TF32 on, matmul_precision=high (both sides)")
    print()

    model = B.BaselineTransformer(cfg).to(device, torch.float32).eval()
    x, mask = B.generate_random_case(cfg, device, torch.float32, 1234, 0.0, 1.0)

    # The reference goes first, on a clean allocator: its wide chunks are the
    # configurations most likely to be denied by fragmentation, and denying the
    # opponent its best setting is not a fair race.
    best_s, best_chunk, per_chunk = float("inf"), None, {}
    for chunk in chunks:
        try:
            torch.cuda.reset_peak_memory_stats()
            secs = _time(lambda: _run_ref(model, x, cfg, chunk), repeats)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            per_chunk[chunk] = round(secs, 2)
            print(f"  reference c={chunk:<5} {secs:8.2f} s   peak {peak:5.1f} GB",
                  flush=True)
            if secs < best_s:
                best_s, best_chunk = secs, chunk
        except torch.cuda.OutOfMemoryError:
            print(f"  reference c={chunk:<5}      OOM", flush=True)
        finally:
            torch.cuda.empty_cache()

    # The independent opponent: PyTorch's own fused attention, substituted for
    # the one line the organizer's baseline cannot execute.
    try:
        torch.cuda.reset_peak_memory_stats()
        sdpa_s = _time(lambda: _run_sdpa(model, x, cfg), repeats)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"  reference SDPA   {sdpa_s:8.2f} s   peak {peak:5.1f} GB",
              flush=True)
    except torch.cuda.OutOfMemoryError:
        sdpa_s = None
        print("  reference SDPA        OOM", flush=True)
    finally:
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    ours_s = _time(lambda: _run_ours(cfg, plan, model, x, mask), repeats)
    ours_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  kernelforge      {ours_s:8.2f} s   peak {ours_peak:5.1f} GB",
          flush=True)
    torch.cuda.empty_cache()

    if best_chunk is None:
        print("\n  the chunked reference did not fit either; no ratio to report")
        return False

    print()
    print(f"  speedup vs chunked reference   {best_s / ours_s:5.2f}x  "
          f"(its best chunk, {best_chunk})")
    if sdpa_s:
        print(f"  speedup vs PyTorch SDPA        {sdpa_s / ours_s:5.2f}x  "
              f"(independently written)")
    print()
    print("  Note: the chunked reference is our own code, so that ratio is a")
    print("  weak independence check. The SDPA one is not ours -- it is "
          "PyTorch's")
    print("  own fused attention substituted into the organizer's model, and it "
          "is")
    print("  the number to prefer. Neither is the organizer's unmodified "
          "baseline,")
    print("  which cannot run this shape at all.")
    print()

    out = {"shape": SHAPE, "gpu": spec.name, "arch": spec.arch,
           "plan": plan.name, "ours_s": round(ours_s, 2),
           "ours_peak_gib": round(ours_peak, 1),
           "reference_s": round(best_s, 2), "reference_chunk": best_chunk,
           "reference_by_chunk_s": per_chunk,
           "speedup": round(best_s / ours_s, 2),
           "sdpa_reference_s": round(sdpa_s, 2) if sdpa_s else None,
           "speedup_vs_sdpa": round(sdpa_s / ours_s, 2) if sdpa_s else None,
           "note": ("The organizer's BaselineTransformer cannot run this shape: "
                    "it materializes an 18.6 TB score matrix. The opponent here "
                    "is that same model with the attention chunked to O(S) "
                    "memory and nothing else changed, run under the organizer's "
                    "own numerics (TF32 on, matmul_precision=high). Its chunk "
                    "size is swept and its best time taken. `speedup_vs_sdpa` "
                    "is against a second, independently written opponent: the "
                    "same model with its attention replaced by PyTorch's "
                    "scaled_dot_product_attention, which is not our code and "
                    "does run this shape.")}
    path = os.path.join(ROOT, "results", f"shape14_race_{spec.arch}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")
    return True


def _run_ours(cfg, plan, model, x, mask):
    with torch.inference_mode():
        build_shared(cfg, plan, model)(x, mask)


def _run_ref(model, x, cfg, chunk):
    """One batch element at a time: batch elements do not interact."""
    with torch.inference_mode():
        for b in range(cfg.batch_size):
            _ref_model(model, x[b], cfg.num_heads, chunk)


def _sdpa_attention(q, k, v):
    """PyTorch's own fused attention. Not our code -- that is the point."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True).squeeze(0)


def _ref_model_sdpa(model, x, num_heads):
    """The organizer's model with its attention replaced by PyTorch's SDPA.

    The chunked reference is ours, which makes it a weak independence check: the
    same person wrote both sides of that ratio. `scaled_dot_product_attention`
    is not ours. It is a fused, O(S)-memory attention written by the PyTorch
    team, it runs this shape, and substituting it for the one line the organizer's
    baseline cannot execute leaves the rest of their model untouched.

    It is a stronger opponent than the chunked reference in every sense that
    matters, so it is the number we would rather be judged on.
    """
    h = x
    for layer in model.layers:
        a = layer.attention
        n = layer.norm1(h)
        M, d = n.shape
        Dh = d // num_heads
        q, k, v = (proj(n).view(M, num_heads, Dh).permute(1, 0, 2).contiguous()
                   for proj in (a.q_proj, a.k_proj, a.v_proj))
        ctx = _sdpa_attention(q, k, v).permute(1, 0, 2).reshape(M, d)
        del q, k, v
        h = h + a.out_proj(ctx)
        del ctx
        n2 = layer.norm2(h)
        h = h + layer.ffn_out(torch.nn.functional.gelu(layer.ffn_in(n2),
                                                       approximate="none"))
        del n2
    return model.final_norm(h)


def _run_sdpa(model, x, cfg):
    with torch.inference_mode():
        for b in range(cfg.batch_size):
            _ref_model_sdpa(model, x[b], cfg.num_heads)


def gate(device, spec, table, batch=1):
    """Gate the whole output at full sequence length.

    The reference runs in fp32 with TF32 disabled, which makes it strictly more
    precise than the organizer's own -- that one leaves TF32 on. Its score block
    is chunked, so peak memory is O(S) rather than the 18.6 TB the materialized
    `[B,H,S,S]` would need.

    `--batch N` raises how much of the batch is checked. Batch elements are
    independent in this model (`tests/test_streaming.py`), so one element is
    sufficient in principle; the full 32 is what removes the "in principle".
    """
    import dataclasses as _dc
    from kernelforge.numerics import ATOL, RTOL

    case = _dc.replace(shapes.resolve([SHAPE])[0], batch_size=batch)
    cfg = case.to_config()
    plan, source = table.lookup(spec.arch, case.torch_dtype, cfg,
                                spec.shared_mem_per_block_kb)
    print(f"  plan under test  {plan.name} [{source}]")
    print("  reference        streamed fp32, TF32 off, two-pass softmax")
    print()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = B.BaselineTransformer(cfg).to(device, torch.float32).eval()
    x, mask = B.generate_random_case(cfg, device, torch.float32, 1234, 0.0, 1.0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        ours = build_shared(cfg, plan, model)(x, mask)

    # One batch element at a time: the reference's peak is a chunk of scores for
    # a single element, so checking 32 of them costs time, not memory.
    env, failed, max_err, n = 0.0, 0, 0.0, 0
    for b in range(cfg.batch_size):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            ref_b = _ref_model(model, x[b], cfg.num_heads)
        err = (ours[b].float() - ref_b).abs()
        allow = torch.clamp(ref_b.abs() * RTOL, min=ATOL)
        env = max(env, (err / allow).max().item())
        failed += int((err > allow).sum().item())
        max_err = max(max_err, err.max().item())
        n += ref_b.numel()
        del ref_b, err, allow
        torch.cuda.empty_cache()
    print(f"  batch elements   {cfg.batch_size}")
    print(f"  elements         {n:,}")
    print(f"  max abs error    {max_err:.3e}")
    print(f"  envelope         {env:.4f}   (limit 1.0)")
    print(f"  failed           {failed}/{n:,}")
    print(f"  verdict          {'PASS' if env < 1.0 else 'FAIL'}")
    print()
    return env < 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="sweep sequence length upward to find this GPU's limit")
    ap.add_argument("--batch", type=int, default=1,
                    help="how many batch elements --gate checks (default 1; "
                         "the shape has 32)")
    ap.add_argument("--gate", action="store_true",
                    help="check the whole output at full sequence length "
                         "against a streamed exact reference")
    ap.add_argument("--race", action="store_true",
                    help="time it against the organizer's model with the "
                         "attention chunked -- the minimum edit that makes the "
                         "reference runnable, and so the fair opponent")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    spec = probe(measure=False)
    table = DispatchTable.load_for(spec)

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

    if args.gate:
        return 0 if gate(device, spec, table, args.batch) else 1
    if args.race:
        return 0 if race(device, spec, table) else 1

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
