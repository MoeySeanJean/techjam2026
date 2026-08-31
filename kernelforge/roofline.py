"""Roofline analysis: how much of the machine are we actually using?

The track names the limiters explicitly -- "GPU compute throughput, memory
bandwidth, cache efficiency, kernel launch overhead, and tensor core
utilization". A speedup number says we got faster; it does not say whether
there is anything left. This module answers that by placing each measured
configuration on the roofline of its GPU:

    arithmetic intensity = FLOPs / bytes moved
    ridge point          = peak FLOP/s / peak bandwidth

Below the ridge a kernel is memory-bound and the ceiling is bandwidth; above it
the kernel is compute-bound and the ceiling is tensor-core throughput. Knowing
which side you are on is what tells you whether the next optimization should
remove memory traffic or arithmetic -- and whether a shape is already close
enough to the ceiling that further work is wasted.

Peak numbers are vendor figures for the tensor-core path with FP32
accumulation. They are stated here rather than hidden so a reader can check
them; percentages scale linearly if a figure is wrong.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

# Dense tensor-core throughput, TFLOP/s, fp16 inputs with fp32 accumulate.
#
# Only the architectures we report are listed. A ceiling is a property of the
# specific card, not of the architecture, so quoting one figure for a whole
# family would produce a confidently wrong "percent of peak". Our own `sm_75`
# pair is the demonstration: a Tesla T4 and a TITAN RTX are the same
# architecture at 2.3x apart in measured bandwidth and 4x in board power, and
# one ceiling cannot describe both. `analyse` returns None for an architecture
# that is not here, and the roofline is simply not reported for it.
PEAK_TFLOPS = {
    "sm_80": 312.0,     # A100, fp16/fp32-acc dense
    "sm_90": 835.0,     # H100 NVL, fp16/fp32-acc dense
}

# TF32 path, used when the reference runs in float32 with allow_tf32 on.
PEAK_TFLOPS_TF32 = {
    "sm_80": 156.0,
    "sm_90": 415.0,
}


@dataclasses.dataclass
class Roofline:
    flops: int
    bytes_moved: int
    seconds: float
    peak_tflops: float
    peak_bandwidth_gbs: float

    @property
    def achieved_tflops(self) -> float:
        return self.flops / self.seconds / 1e12

    @property
    def achieved_bandwidth_gbs(self) -> float:
        return self.bytes_moved / self.seconds / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes_moved, 1)

    @property
    def ridge_point(self) -> float:
        """FLOP/byte above which a kernel becomes compute-bound."""
        return self.peak_tflops * 1e12 / (self.peak_bandwidth_gbs * 1e9)

    @property
    def compute_bound(self) -> bool:
        return self.arithmetic_intensity > self.ridge_point

    @property
    def utilization(self) -> float:
        """Fraction of the binding ceiling actually reached."""
        if self.compute_bound:
            return self.achieved_tflops / self.peak_tflops
        return self.achieved_bandwidth_gbs / self.peak_bandwidth_gbs

    @property
    def limiter(self) -> str:
        return "tensor cores" if self.compute_bound else "memory bandwidth"

    def summary(self) -> str:
        return (f"{self.achieved_tflops:7.1f} TFLOP/s  "
                f"{self.achieved_bandwidth_gbs:7.0f} GB/s  "
                f"AI={self.arithmetic_intensity:6.1f} "
                f"(ridge {self.ridge_point:.1f})  "
                f"{self.limiter:<16} {self.utilization:5.1%} of ceiling")


def stack_flops(batch: int, seq: int, d_model: int, heads: int, ffn: int,
                layers: int, causal: bool = False) -> int:
    """Multiply-accumulate FLOPs for one forward pass of the whole stack.

    Counts the six GEMMs per layer plus the two attention batched matmuls, at
    2 FLOPs per MAC. Elementwise work (LayerNorm, GELU, residual, masking) is
    negligible in FLOPs though it dominates memory traffic -- which is exactly
    why the roofline, not a FLOP count alone, is the useful view.
    """
    head_dim = d_model // heads
    per_layer = 0
    per_layer += 2 * batch * seq * d_model * (3 * d_model)   # fused QKV
    per_layer += 2 * batch * seq * d_model * d_model         # out_proj
    per_layer += 2 * batch * seq * d_model * ffn             # ffn1
    per_layer += 2 * batch * seq * ffn * d_model             # ffn2
    # attention: QK^T and PV, each 2*B*H*S*S*head_dim
    attn = 4 * batch * heads * seq * seq * head_dim
    if causal:
        attn //= 2                                           # half the tiles skipped
    per_layer += attn
    return per_layer * layers


def stack_bytes(batch: int, seq: int, d_model: int, heads: int, ffn: int,
                layers: int, elem_size: int = 2) -> int:
    """Lower bound on DRAM traffic for a *fused* implementation.

    Counts the weights (read once per forward) plus the activation tensors that
    must cross DRAM between kernels. A fused implementation keeps intermediates
    in registers and shared memory, so this deliberately does not count the
    baseline's materialized [B,H,S,S] score tensor -- comparing against it would
    flatter us.
    """
    weights = layers * (4 * d_model * d_model + 2 * d_model * ffn) * elem_size
    tokens = batch * seq
    # per layer: read residual, write residual, plus the two normed activations
    # and the FFN intermediate
    acts = layers * tokens * (4 * d_model + 2 * ffn) * elem_size
    return weights + acts


def analyse(case, seconds: float, arch: str, bandwidth_gbs: float,
            dtype: str = "float32") -> Optional[Roofline]:
    peaks = PEAK_TFLOPS_TF32 if dtype == "float32" else PEAK_TFLOPS
    peak = peaks.get(arch)
    if peak is None or not bandwidth_gbs:
        return None
    elem = 4 if dtype == "float32" else 2
    return Roofline(
        flops=stack_flops(case.batch_size, case.seq_len, case.d_model,
                          case.num_heads, case.ffn_dim, case.num_layers,
                          case.causal),
        bytes_moved=stack_bytes(case.batch_size, case.seq_len, case.d_model,
                                case.num_heads, case.ffn_dim, case.num_layers,
                                elem),
        seconds=seconds / 1000.0,
        peak_tflops=peak,
        peak_bandwidth_gbs=bandwidth_gbs,
    )
