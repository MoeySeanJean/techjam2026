"""Hardware probe.

The optimizer needs a machine-readable description of the target GPU: the agent
feeds it to the LLM, and the dispatch table keys off it. Shared-memory capacity in
particular decides which Triton tile configurations are even legal, and it is the
single most common reason an LLM-generated kernel fails to compile (models trained
on A100 FlashAttention emit 164 KB configs that do not fit the 64 KB of a
Turing card).
"""
from __future__ import annotations

import dataclasses
import platform
import subprocess
from typing import Optional

import torch


@dataclasses.dataclass(frozen=True)
class GPUSpec:
    name: str
    arch: str               # "sm_80"
    major: int
    minor: int
    sm_count: int
    total_mem_gb: float
    shared_mem_per_block_kb: float
    regs_per_block: int
    warp_size: int
    l2_cache_mb: float
    clock_mhz: int
    mem_clock_mhz: int
    mem_bus_bits: int
    has_tensor_cores: bool
    has_bf16: bool
    has_tma: bool           # Hopper+
    has_fp8: bool           # Hopper+
    is_laptop: bool

    measured_bw_gbs: float = 0.0   # achieved, from a large device-to-device copy

    @property
    def peak_mem_bw_gbs(self) -> float:
        """Theoretical peak DRAM bandwidth, when the bus width is discoverable.

        Driver builds vary in whether they expose Bus Width, so this can be 0.
        `measured_bw_gbs` is the number we actually trust for roofline work: an
        achieved figure already includes clock throttling, which matters a great
        deal on a power-limited mobile part.
        """
        if not self.mem_bus_bits:
            return 0.0
        return self.mem_clock_mhz * 1e6 * 2 * (self.mem_bus_bits / 8) / 1e9

    @property
    def bw_gbs(self) -> float:
        return self.measured_bw_gbs or self.peak_mem_bw_gbs

    def summary(self) -> str:
        tags = []
        if self.has_tensor_cores:
            tags.append("tensor-cores")
        if self.has_bf16:
            tags.append("bf16")
        if self.has_tma:
            tags.append("tma")
        if self.has_fp8:
            tags.append("fp8")
        if self.is_laptop:
            tags.append("laptop")
        return (
            f"{self.name} [{self.arch}] "
            f"{self.sm_count} SMs | {self.total_mem_gb:.1f} GB | "
            f"smem/block {self.shared_mem_per_block_kb:.0f} KB | "
            f"~{self.bw_gbs:.0f} GB/s | {', '.join(tags)}"
        )

    def as_prompt_block(self) -> str:
        """Compact spec sheet handed to the LLM during kernel generation."""
        lines = [
            f"GPU: {self.name}",
            f"Compute capability: {self.arch} (major={self.major}, minor={self.minor})",
            f"SM count: {self.sm_count}",
            f"Shared memory per block (opt-in): {self.shared_mem_per_block_kb:.0f} KB"
            "   <-- HARD LIMIT for BLOCK_M*BLOCK_N*num_stages tiling",
            f"Registers per block: {self.regs_per_block}",
            f"L2 cache: {self.l2_cache_mb:.0f} MB",
            f"Device memory: {self.total_mem_gb:.1f} GB",
            (f"Achieved DRAM bandwidth (measured): ~{self.bw_gbs:.0f} GB/s"
             if self.bw_gbs else "Achieved DRAM bandwidth: not measured"),
            f"BF16 supported: {self.has_bf16}",
            f"TMA (Hopper async copy) available: {self.has_tma}",
            f"FP8 available: {self.has_fp8}",
        ]
        if self.is_laptop:
            lines.append(
                "NOTE: mobile part with a constrained power envelope - sustained clocks "
                "drop under load; prefer configurations that reduce work over ones that "
                "merely raise occupancy."
            )
        if not self.has_tma:
            lines.append(
                "NOTE: pre-Hopper. Do NOT emit TMA descriptors, wgmma, or FP8 paths. "
                "Use cp.async style pipelining via Triton num_stages instead."
            )
        return "\n".join(lines)


def _nvidia_smi(field: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def measure_bandwidth(device: int = 0, mb: int = 256, iters: int = 30) -> float:
    """Achieved device-to-device copy bandwidth in GB/s.

    A large copy reads `mb` and writes `mb`, so effective traffic is 2x the buffer.
    """
    n = mb * 1024 * 1024 // 4
    src = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_s = start.elapsed_time(end) / 1e3
    total_bytes = 2 * src.numel() * src.element_size() * iters
    del src, dst
    torch.cuda.empty_cache()
    return total_bytes / elapsed_s / 1e9


def probe(device: int = 0, measure: bool = True) -> GPUSpec:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    p = torch.cuda.get_device_properties(device)
    name = p.name

    smem = getattr(p, "shared_memory_per_block_optin", None) or p.shared_memory_per_block

    mem_clock = _nvidia_smi("clocks.max.memory")
    sm_clock = _nvidia_smi("clocks.max.sm")

    # Bus width is not exposed by torch; nvidia-smi -q carries it.
    bus_bits = 0
    try:
        q = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=15)
        for line in q.stdout.splitlines():
            if "Bus Width" in line:
                bus_bits = int("".join(c for c in line.split(":")[-1] if c.isdigit()))
                break
    except Exception:
        pass

    return GPUSpec(
        name=name,
        arch=f"sm_{p.major}{p.minor}",
        major=p.major,
        minor=p.minor,
        sm_count=p.multi_processor_count,
        total_mem_gb=p.total_memory / 2**30,
        shared_mem_per_block_kb=smem / 1024,
        regs_per_block=getattr(p, "regs_per_multiprocessor", 65536),
        warp_size=getattr(p, "warp_size", 32),
        l2_cache_mb=getattr(p, "L2_cache_size", 0) / 2**20,
        clock_mhz=int(sm_clock) if sm_clock else 0,
        mem_clock_mhz=int(mem_clock) if mem_clock else 0,
        mem_bus_bits=bus_bits,
        has_tensor_cores=p.major >= 7,
        has_bf16=p.major >= 8,
        has_tma=p.major >= 9,
        has_fp8=p.major >= 9,
        is_laptop="laptop" in name.lower() or "mobile" in name.lower(),
        measured_bw_gbs=measure_bandwidth(device) if measure else 0.0,
    )


def device_slug(spec: "GPUSpec") -> str:
    """Filename-safe identifier for a specific card, not just its architecture.

    The A100-40 and A100-80 are both sm_80: one dispatch table serves both
    correctly (same shared memory, same instruction set), but their *timings*
    are different measurements and must not overwrite one another. Dispatch
    tables stay keyed by arch; sweep and genealogy artifacts key by device.
    """
    name = spec.name.lower()
    for junk in ("nvidia", "geforce", "rtx", "gpu", "pcie", "sxm", "(r)"):
        name = name.replace(junk, " ")
    slug = "-".join(part for part in name.split() if part)
    mem = f"{spec.total_mem_gb:.0f}gb"
    if mem not in slug:
        slug = f"{slug}-{mem}"
    return f"{slug}_{spec.arch}".strip("-")


def host_summary() -> str:
    import sys
    bits = [
        f"python {sys.version.split()[0]}",
        f"torch {torch.__version__}",
        f"platform {platform.system()} {platform.release()}",
    ]
    try:
        import triton
        bits.append(f"triton {triton.__version__}")
    except Exception:
        bits.append("triton MISSING")
    if torch.cuda.is_available():
        bits.append(f"cuda {torch.version.cuda}")
    return " | ".join(bits)
