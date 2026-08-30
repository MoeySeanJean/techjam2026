"""The shape matrix.

The organizer's script publishes no shape list -- every dimension arrives as a CLI
argument, and the problem statement says the official combinations "will be told to
the participants". Until that list lands we partition the space by *bottleneck
regime*, because that is what actually selects an implementation:

  latency   - so small that kernel-launch overhead dominates real work
  gemm      - projection/FFN matmuls dominate (the default config lives here)
  attention - O(S^2) score computation dominates; FlashAttention territory
  bandwidth - large activations, elementwise and normalization traffic dominates

`REGIMES` is deliberately a superset of anything likely to be scored, so the
dispatch table generalizes instead of overfitting to a published list.
"""
from __future__ import annotations

import dataclasses
import re
from typing import List, Optional

import torch

from torch_transformer_benchmark import TransformerConfig


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    regime: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int = 6
    causal: bool = False
    padding_ratio: float = 0.0
    input_scale: float = 1.0
    dtype: str = "float32"

    def to_config(self) -> TransformerConfig:
        return TransformerConfig(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            causal=self.causal,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16}[self.dtype]

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    def activation_bytes(self) -> int:
        """Peak transient bytes for the *baseline*, which materializes [B,H,S,S]."""
        elt = torch.finfo(self.torch_dtype).bits // 8
        scores = self.batch_size * self.num_heads * self.seq_len * self.seq_len * 4
        acts = self.tokens * max(self.d_model, self.ffn_dim) * elt * 4
        return scores + acts

    def label(self) -> str:
        bits = [f"B{self.batch_size}", f"S{self.seq_len}", f"d{self.d_model}",
                f"H{self.num_heads}", f"F{self.ffn_dim}", f"L{self.num_layers}"]
        if self.causal:
            bits.append("causal")
        if self.padding_ratio:
            bits.append(f"pad{self.padding_ratio:g}")
        if self.input_scale != 1.0:
            bits.append(f"scale{self.input_scale:g}")
        bits.append(self.dtype)
        return "-".join(bits)

    def cli(self) -> str:
        """Equivalent invocation of the organizer's script, for the report."""
        parts = [
            f"--batch-size {self.batch_size}", f"--seq-len {self.seq_len}",
            f"--d-model {self.d_model}", f"--heads {self.num_heads}",
            f"--ffn-dim {self.ffn_dim}", f"--layers {self.num_layers}",
            f"--dtype {self.dtype}",
        ]
        if self.causal:
            parts.append("--causal")
        if self.padding_ratio:
            parts.append(f"--padding-ratio {self.padding_ratio}")
        if self.input_scale != 1.0:
            parts.append(f"--input-scale {self.input_scale}")
        return "python torch_transformer_benchmark.py " + " ".join(parts)


# --- the matrix -------------------------------------------------------------

def _base() -> List[Case]:
    return [
        # --- latency regime: work is tiny, ~105 launches/forward dominates ---
        Case("tiny",        "latency",   1,   64,  256,  4, 1024),
        Case("decode",      "latency",  32,    1,  512,  8, 2048),
        Case("short",       "latency",   4,   32,  512,  8, 2048),

        # --- gemm regime: projections and FFN dominate (script default lives here) ---
        Case("default",     "gemm",      8,  128,  512,  8, 2048),
        Case("bert_base",   "gemm",     16,  512,  768, 12, 3072),
        Case("wide",        "gemm",      8,  128, 1024, 16, 4096),
        Case("big_batch",   "gemm",     64,  128,  512,  8, 2048),

        # --- attention regime: O(S^2) dominates ---
        Case("long_seq",    "attention", 2, 2048,  512,  8, 2048),
        Case("very_long",   "attention", 1, 4096,  512,  8, 2048),
        Case("long_causal", "attention", 2, 2048,  512,  8, 2048, causal=True),

        # --- bandwidth regime: big activations, elementwise traffic ---
        Case("fat_ffn",     "bandwidth", 8,  512,  512,  8, 8192),
        Case("deep",        "bandwidth", 8,  256,  768, 12, 3072, num_layers=12),
    ]


def _variants() -> List[Case]:
    """Correctness-critical variants layered on top of a few representative shapes.

    Padding, causality and input scaling are where silently-wrong kernels hide,
    so they get first-class coverage rather than being an afterthought.
    """
    out: List[Case] = []
    seeds = [c for c in _base() if c.name in ("default", "long_seq", "tiny", "bert_base")]
    for c in seeds:
        out.append(dataclasses.replace(c, name=f"{c.name}_pad", padding_ratio=0.4))
        out.append(dataclasses.replace(c, name=f"{c.name}_causal", causal=True))
        out.append(dataclasses.replace(
            c, name=f"{c.name}_causal_pad", causal=True, padding_ratio=0.4))
        # LayerNorm is scale-invariant, so a large input scale must not move the
        # answer -- but it will overflow a naive fp16 residual stream.
        out.append(dataclasses.replace(c, name=f"{c.name}_scale", input_scale=64.0))
        for dt in ("float16", "bfloat16"):
            out.append(dataclasses.replace(c, name=f"{c.name}_{dt}", dtype=dt))
    return out


def all_cases(include_variants: bool = True) -> List[Case]:
    cases = _base()
    if include_variants:
        cases = cases + _variants()
    return cases


def select(
    names: Optional[List[str]] = None,
    regimes: Optional[List[str]] = None,
    include_variants: bool = True,
    max_activation_gb: Optional[float] = None,
) -> List[Case]:
    cases = all_cases(include_variants)
    if names:
        wanted = set(names)
        cases = [c for c in cases if c.name in wanted]
    if regimes:
        wanted = set(regimes)
        cases = [c for c in cases if c.regime in wanted]
    if max_activation_gb is not None:
        cases = [c for c in cases
                 if c.activation_bytes() / 2**30 <= max_activation_gb]
    return cases


def fits_on(spec, case: Case, headroom: float = 0.55) -> bool:
    """Whether the *baseline* can run this case without exhausting device memory.

    The baseline materializes [B,H,S,S] scores in fp32, which is what actually
    blows up -- our fused implementation never allocates it.
    """
    return case.activation_bytes() <= spec.total_mem_gb * 2**30 * headroom


REGIME_NAMES = ("latency", "gemm", "attention", "bandwidth")


# --- ingesting shapes we were not born knowing ------------------------------

_SPEC_RE = re.compile(
    r"^B(?P<batch>\d+)[-x]S(?P<seq>\d+)[-x]d(?P<d>\d+)[-x]H(?P<h>\d+)"
    r"[-x]F(?P<ffn>\d+)(?:[-x]L(?P<layers>\d+))?(?P<rest>.*)$", re.I)


def parse_spec(spec: str) -> Case:
    """Turn a shape string into a Case.

    The problem statement says the official shape combinations "will be told to
    the participants" -- but not when. Until then our matrix is an informed
    guess, so the system has to be able to ingest an arbitrary shape the moment
    one is published, without a code change. This is that entry point.

    Accepts the signature form used throughout the project, with optional
    suffixes in any order:

        B8-S128-d512-H8-F2048-L6
        B8-S128-d512-H8-F2048-L6-causal
        B8-S128-d512-H8-F2048-L6-causal-pad0.4-scale64-float16
        B2xS2048xd512xH8xF2048xL6            (x separators also fine)

    `L` defaults to 6, matching the organizer's default.
    """
    text = spec.strip()
    m = _SPEC_RE.match(text)
    if not m:
        raise ValueError(
            f"cannot parse shape {spec!r}. Expected e.g. "
            f"B8-S128-d512-H8-F2048-L6[-causal][-pad0.4][-scale64][-float16]")
    g = m.groupdict()
    rest = (g.get("rest") or "").lower()

    dtype = "float32"
    for cand in ("bfloat16", "float16", "float32"):
        if cand in rest:
            dtype = cand
            break

    pad = 0.0
    pm = re.search(r"pad([0-9.]+)", rest)
    if pm:
        pad = float(pm.group(1))

    scale = 1.0
    sm = re.search(r"scale([0-9.]+)", rest)
    if sm:
        scale = float(sm.group(1))

    batch, seq = int(g["batch"]), int(g["seq"])
    d_model, heads, ffn = int(g["d"]), int(g["h"]), int(g["ffn"])
    layers = int(g["layers"] or 6)
    if d_model % heads:
        raise ValueError(f"{spec}: d_model {d_model} is not divisible by "
                         f"{heads} heads")

    # Classify by the same bottleneck logic the hand-written matrix uses, so a
    # newly ingested shape lands in a comparable regime bucket.
    tokens = batch * seq
    if tokens <= 512:
        regime = "latency"
    elif seq >= 1024:
        regime = "attention"
    elif ffn >= 4 * d_model and tokens * ffn >= 8_000_000:
        regime = "bandwidth"
    else:
        regime = "gemm"

    return Case(name=text, regime=regime, batch_size=batch, seq_len=seq,
                d_model=d_model, num_heads=heads, ffn_dim=ffn,
                num_layers=layers, causal="causal" in rest,
                padding_ratio=pad, input_scale=scale, dtype=dtype)


def resolve(names: List[str]) -> List[Case]:
    """Look up names in the built-in matrix, parsing anything unrecognised.

    This is what lets `--cases B4-S777-d640-H10-F2560-L9` work alongside
    `--cases default,long_seq` -- and what will let the official list be tuned
    the day it lands.
    """
    known = {c.name: c for c in all_cases()}
    out: List[Case] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name in known:
            out.append(known[name])
        else:
            out.append(parse_spec(name))
    return out


def load_spec_file(path: str) -> List[Case]:
    """Read a shape list from a file: one spec per line, `#` comments allowed."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.split("#", 1)[0].strip() for ln in fh]
    return resolve([ln for ln in lines if ln])
