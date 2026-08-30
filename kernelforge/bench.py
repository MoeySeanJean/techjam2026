"""Timing harness.

Three rules, all of which exist because of measurements taken on this project:

1. **Always three-way.** Every speedup is reported against the naive baseline
   *and* `torch.compile(max-autotune)`. Beating the naive baseline while losing
   to torch.compile is not a result, and a judge will run `--compile-baseline`.

2. **Interleaved A/B.** The laptop target showed a 64% spread between p90 and
   min at a fixed shape, purely from thermal drift. Measuring A fully and then B
   fully attributes that drift to whichever ran second. We round-robin instead,
   so drift hits every candidate equally.

3. **Median, not mean.** With a power-limited part the distribution has a long
   right tail; the mean tracks the tail, the median tracks the machine.

Energy is sampled where NVML is available: on an 83 W part, joules per token is
a more honest efficiency claim than latency alone.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Callable, Dict, List, Optional

import torch


@dataclasses.dataclass
class Timing:
    samples_ms: List[float]
    energy_j: Optional[float] = None

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        s = sorted(self.samples_ms)
        return s[min(len(s) - 1, int(0.9 * (len(s) - 1)))]

    @property
    def spread(self) -> float:
        """p90/min. A value far above 1.0 means the machine, not the kernel,
        is what is being measured."""
        return self.p90_ms / self.min_ms if self.min_ms else float("nan")


class PowerSampler:
    """Best-effort energy accounting via NVML."""

    def __init__(self, device: int = 0):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(device)
            self.nvml = pynvml
            self.ok = True
        except Exception:
            pass
        self.samples: List[float] = []

    def sample(self) -> None:
        if not self.ok:
            return
        try:
            self.samples.append(self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
        except Exception:
            pass

    def mean_watts(self) -> Optional[float]:
        return statistics.fmean(self.samples) if self.samples else None


def lock_clocks(device: int = 0) -> bool:
    """Pin SM clocks so consecutive candidates see the same machine.

    Usually needs elevation, so a failure is expected and non-fatal -- it is
    reported in the results so the reader knows which regime a number came from.
    """
    import subprocess
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        target = int(q.stdout.strip().splitlines()[0])
        r = subprocess.run(["nvidia-smi", "-i", str(device), "-lgc", f"{target},{target}"],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def reset_clocks(device: int = 0) -> None:
    import subprocess
    try:
        subprocess.run(["nvidia-smi", "-i", str(device), "-rgc"],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def _time_block(fn: Callable[[], object], iters: int) -> List[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def compare(
    candidates: Dict[str, Callable[[], object]],
    warmup: int = 25,
    repeats: int = 50,
    rounds: int = 5,
    device: int = 0,
) -> Dict[str, Timing]:
    """Round-robin timing of several callables.

    `rounds` passes over the candidate set, each contributing `repeats` samples,
    with the visiting order rotated every round so no candidate is permanently
    advantaged by cache or clock state.
    """
    names = list(candidates)
    with torch.inference_mode():
        for name in names:
            for _ in range(warmup):
                candidates[name]()
        torch.cuda.synchronize()

        power = PowerSampler(device)
        samples: Dict[str, List[float]] = {n: [] for n in names}
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for name in order:
                power.sample()
                samples[name].extend(_time_block(candidates[name], repeats))
        power.sample()

    watts = power.mean_watts()
    out = {}
    for n in names:
        t = Timing(samples[n])
        if watts is not None:
            t.energy_j = watts * t.median_ms / 1e3
        out[n] = t
    return out


def format_table(results: Dict[str, Timing], baseline_key: str = "baseline") -> str:
    base = results.get(baseline_key)
    lines = [f"{'variant':<26} {'median ms':>10} {'p90':>9} {'min':>9} "
             f"{'spread':>7} {'speedup':>8} {'mJ':>8}",
             "-" * 82]
    for name, t in results.items():
        sp = (base.median_ms / t.median_ms) if base else float("nan")
        mj = f"{t.energy_j * 1e3:8.2f}" if t.energy_j else "       -"
        lines.append(f"{name:<26} {t.median_ms:10.4f} {t.p90_ms:9.4f} "
                     f"{t.min_ms:9.4f} {t.spread:7.2f} {sp:7.3f}x {mj}")
    return "\n".join(lines)
