"""Bottleneck attribution.

Produces the machine-readable profile the proposer reasons over. The point is
not to dump a profiler table but to answer one question -- which regime is this
shape in? -- because that is what selects an optimization strategy:

    launch-bound   -> fuse aggressively, capture a CUDA graph
    gemm-bound     -> precision and tensor-core efficiency
    attention-bound-> tiling, online softmax, never materialize [B,H,S,S]
    bandwidth-bound-> fuse elementwise passes into their producers

The launch-overhead measurement matters more than it looks: on Windows WDDM a
kernel launch costs ~13 us of CPU time, and a 6-layer forward issues ~105 of
them, so a shape whose GPU work is under a millisecond is being timed on its
submission cost rather than its arithmetic.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List

import torch
from torch.profiler import ProfilerActivity, profile

import torch_transformer_benchmark as B

from ..shapes import Case

# Substrings that classify a CUDA kernel name into a bucket.
BUCKETS = (
    ("gemm", ("cutlass", "gemm", "sgemm", "ampere_", "s16816", "cublas",
              "nn_", "tn_", "nt_")),
    ("softmax", ("softmax", "SoftMax")),
    ("copy", ("Memcpy", "copy_", "direct_copy")),
    ("norm", ("layer_norm", "LayerNorm", "welford", "vectorized_layer")),
    ("elementwise", ("elementwise", "vectorized_elementwise", "gelu", "fill",
                     "masked_fill", "triu", "arange", "index")),
    ("triton", ("triton", "_flash_fwd", "_add_mask_ln")),
)


def classify(name: str) -> str:
    for bucket, keys in BUCKETS:
        if any(k in name for k in keys):
            return bucket
    return "other"


@dataclasses.dataclass
class Profile:
    case: str
    total_cuda_ms: float
    total_cpu_ms: float
    launches: int
    launch_us_each: float
    buckets: Dict[str, float]
    top_kernels: List[tuple]
    regime: str
    note: str = ""

    def as_prompt_block(self) -> str:
        share = {k: 100.0 * v / self.total_cuda_ms for k, v in self.buckets.items()
                 if self.total_cuda_ms}
        lines = [
            f"Shape: {self.case}",
            f"Bottleneck regime (measured): {self.regime}",
            f"GPU time per forward: {self.total_cuda_ms:.3f} ms",
            f"CPU time per forward: {self.total_cpu_ms:.3f} ms",
            f"Kernel launches per forward: {self.launches} "
            f"(~{self.launch_us_each:.1f} us CPU each)",
            "Time by category:",
        ]
        for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<14} {v:5.1f}%")
        lines.append("Hottest kernels:")
        for name, ms, calls in self.top_kernels[:5]:
            lines.append(f"  {ms:7.3f} ms  x{calls:<5} {name[:70]}")
        if self.note:
            lines.append(f"Note: {self.note}")
        return "\n".join(lines)


def profile_model(model, x, mask, iters: int = 20) -> Profile:
    with torch.inference_mode():
        for _ in range(10):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()

    buckets: Dict[str, float] = defaultdict(float)
    kernels: List[tuple] = []
    launches = 0
    launch_cpu_us = 0.0
    total_cuda_us = 0.0
    total_cpu_us = 0.0

    for evt in p.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or \
            getattr(evt, "self_cuda_time_total", 0) or 0
        cpu_us = getattr(evt, "self_cpu_time_total", 0) or 0
        total_cpu_us += cpu_us
        if "cudaLaunchKernel" in evt.key:
            launches += evt.count
            launch_cpu_us += cpu_us
            continue
        if cuda_us <= 0:
            continue
        total_cuda_us += cuda_us
        buckets[classify(evt.key)] += cuda_us / iters / 1e3
        kernels.append((evt.key, cuda_us / iters / 1e3, evt.count // iters))

    kernels.sort(key=lambda t: -t[1])
    total_cuda_ms = total_cuda_us / iters / 1e3
    total_cpu_ms = total_cpu_us / iters / 1e3
    per_launch = (launch_cpu_us / launches) if launches else 0.0

    top = max(buckets, key=buckets.get) if buckets else "unknown"
    note = ""
    if total_cpu_ms > total_cuda_ms:
        regime = "launch-bound"
        note = ("CPU submission time exceeds GPU execution time; the shape is "
                "paying for kernel launches, not arithmetic. CUDA graph capture "
                "is the highest-value change here.")
    elif top == "gemm":
        regime = "gemm-bound"
    elif top in ("softmax",) or (buckets.get("softmax", 0) +
                                 buckets.get("elementwise", 0) >
                                 0.5 * total_cuda_ms):
        regime = "attention-bound"
    elif top in ("elementwise", "norm", "copy"):
        regime = "bandwidth-bound"
    else:
        regime = top

    return Profile("", total_cuda_ms, total_cpu_ms, launches // iters, per_launch,
                   dict(buckets), kernels[:8], regime, note)


def profile_case(case: Case, device: torch.device, iters: int = 20) -> Profile:
    cfg = case.to_config()
    model = B.BaselineTransformer(cfg).to(device, case.torch_dtype).eval()
    x, vm = B.generate_random_case(cfg, device, case.torch_dtype, 1234,
                                   case.padding_ratio, case.input_scale)
    prof = profile_model(model, x, vm, iters)
    prof.case = case.label()
    del model
    torch.cuda.empty_cache()
    return prof
