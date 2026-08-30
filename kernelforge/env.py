"""Full machine description for the tech report.

The problem statement asks for "a clear tech report illustrating what the
environment is (CPU, GPU, DISK, etc)". A latency number is meaningless without
the machine it came from, and on this project that was not a formality: the CPU
matters because small shapes are launch-bound and we measure ~13 us of CPU per
kernel launch on Windows WDDM, and the disk matters because Triton and Inductor
compile to a cache whose location changed our cluster job from "fails on an NFS
quota" to "works on node-local NVMe".

Everything here is best-effort and degrades to "unknown" rather than raising:
this runs on Windows laptops and Linux compute nodes alike.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, Optional


def _run(cmd, timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def cpu_info() -> Dict[str, str]:
    info = {"model": platform.processor() or "unknown",
            "arch": platform.machine(),
            "logical_cores": str(os.cpu_count() or "?")}
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        nproc = _run(["nproc"])
        if nproc:
            info["logical_cores"] = nproc
    elif platform.system() == "Windows":
        name = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        if name:
            info["model"] = name.splitlines()[0].strip()
    return info


def memory_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["total_gb"] = f"{kb / 1024 / 1024:.1f}"
                        break
        except Exception:
            pass
    elif platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        if out and out.strip().isdigit():
            info["total_gb"] = f"{int(out.strip()) / 2**30:.1f}"
    return info or {"total_gb": "unknown"}


def disk_info(paths=None) -> Dict[str, Dict[str, str]]:
    """Capacity of the paths that actually matter to a run.

    The compile caches are the reason this is reported: an NFS home with a
    server-side quota cannot hold a PyTorch install plus Inductor artifacts,
    which is why cluster jobs build on node-local scratch instead.
    """
    if paths is None:
        paths = [os.getcwd(), os.path.expanduser("~")]
        for var in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if os.environ.get(var):
                paths.append(os.environ[var])
        if os.path.isdir("/tmp"):
            paths.append("/tmp")
    out: Dict[str, Dict[str, str]] = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            out[path] = {"total_gb": f"{usage.total / 2**30:.1f}",
                         "free_gb": f"{usage.free / 2**30:.1f}"}
        except Exception:
            continue
    return out


def software_info() -> Dict[str, str]:
    info = {"python": sys.version.split()[0],
            "os": f"{platform.system()} {platform.release()}"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda or "n/a"
        info["cudnn"] = str(getattr(torch.backends.cudnn, "version", lambda: "n/a")())
    except Exception:
        info["torch"] = "missing"
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = "missing"
    drv = _run(["nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"])
    if drv:
        info["nvidia_driver"] = drv.splitlines()[0].strip()
    return info


def describe() -> Dict[str, object]:
    """Everything the tech report needs about this machine."""
    payload: Dict[str, object] = {
        "cpu": cpu_info(),
        "memory": memory_info(),
        "disk": disk_info(),
        "software": software_info(),
    }
    try:
        from .hw import probe
        spec = probe()
        payload["gpu"] = {
            "name": spec.name, "arch": spec.arch, "sms": spec.sm_count,
            "memory_gb": round(spec.total_mem_gb, 1),
            "shared_mem_per_block_kb": round(spec.shared_mem_per_block_kb),
            "l2_cache_mb": round(spec.l2_cache_mb),
            "measured_bandwidth_gbs": round(spec.bw_gbs),
            "max_sm_clock_mhz": spec.clock_mhz,
            "is_laptop": spec.is_laptop,
        }
    except Exception as e:
        payload["gpu"] = {"error": f"{type(e).__name__}: {e}"}
    return payload


def format_report(payload: Optional[Dict[str, object]] = None) -> str:
    p = payload or describe()
    cpu, mem, sw = p["cpu"], p["memory"], p["software"]
    gpu = p.get("gpu", {})
    lines = [
        "Environment",
        "-----------",
        f"  CPU     : {cpu.get('model')} ({cpu.get('logical_cores')} logical cores,"
        f" {cpu.get('arch')})",
        f"  Memory  : {mem.get('total_gb')} GB",
    ]
    if "error" not in gpu:
        lines.append(
            f"  GPU     : {gpu.get('name')} [{gpu.get('arch')}], "
            f"{gpu.get('sms')} SMs, {gpu.get('memory_gb')} GB, "
            f"{gpu.get('shared_mem_per_block_kb')} KB shared/block, "
            f"{gpu.get('measured_bandwidth_gbs')} GB/s measured, "
            f"{gpu.get('max_sm_clock_mhz')} MHz max SM clock")
    else:
        lines.append(f"  GPU     : {gpu['error']}")
    for path, d in p["disk"].items():
        lines.append(f"  Disk    : {path}  {d['free_gb']} GB free "
                     f"of {d['total_gb']} GB")
    lines.append(
        f"  Software: python {sw.get('python')}, torch {sw.get('torch')} "
        f"(CUDA {sw.get('cuda_runtime')}), triton {sw.get('triton')}, "
        f"driver {sw.get('nvidia_driver', 'n/a')}, {sw.get('os')}")
    return "\n".join(lines)
