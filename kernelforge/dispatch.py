"""The frozen dispatch table.

The search runs offline. What ships is this: a lookup from (architecture, I/O
dtype, shape) to a `Plan`, with every entry carrying the evidence that put it
there -- measured envelope utilization, measured speedup, and the search run
that produced it.

Nothing here calls an LLM or an autotuner at run time. That is deliberate: the
serving path must be deterministic and auditable, and a warmup stall would be a
legitimate objection to the whole approach. AI is a build-time tool; the artifact
it produces is an ordinary, reviewable table.

Fallback order, most specific first:
  1. exact (arch, dtype, shape signature) entry from a search run
  2. nearest entry in the same regime for that (arch, dtype)
  3. a dtype-level default rule
  4. `SAFE` -- bit-exact, still faster than the baseline, never wrong

Rule 3 encodes the project's central numerical finding: for fp16 and bf16
inputs the tolerance is not satisfiable by any reassociating optimization (see
docs/PRECISION.md -- torch.compile itself fails it), so those dtypes get the
bit-exact plan and take their speedup from launch-overhead removal alone.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .optimized import Plan
from .search import SAFE

TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results")


def shape_signature(case_or_cfg, dtype: Optional[torch.dtype] = None) -> str:
    c = case_or_cfg
    causal = getattr(c, "causal", False)
    return (f"B{c.batch_size}-S{c.seq_len}-d{c.d_model}-H{c.num_heads}"
            f"-F{c.ffn_dim}-L{c.num_layers}{'-causal' if causal else ''}")


def _arch_major(arch: str) -> int:
    """Compute-capability major version from an `sm_NM` string; 0 if unparseable."""
    digits = "".join(c for c in (arch or "") if c.isdigit())
    if not digits:
        return 0
    return int(digits[:-1]) if len(digits) > 1 else int(digits)


def _aggressiveness(plan: Plan) -> int:
    """How much numerical risk a plan takes. Lower is safer.

    Ordered by what the error budget actually showed costs envelope: narrowing a
    stage to fp16 dominates, the fused norm is the next largest structural
    contributor, and a narrow residual stream is disproportionately expensive
    because it accumulates across 2*num_layers sublayers. CUDA graph capture
    changes no arithmetic and so carries no weight here.
    """
    score = 2 * len(plan.overrides)
    if plan.compute_dtype in ("float16", "bfloat16"):
        score += 8
    if plan.residual_dtype in ("float16", "bfloat16"):
        score += 6
    if plan.fused_norm:
        score += 1
    if plan.attention != "exact":
        score += 1
    if plan.torch_compile and plan.attention != "exact":
        # Compiling the baseline reassociates freely; compiling our bit-exact
        # rewrite measured 0.000 envelope, so only the former carries risk here.
        score += 1
    return score


@dataclasses.dataclass
class Entry:
    signature: str
    dtype: str
    arch: str
    regime: str
    plan: Plan
    utilization: float
    speedup: float
    speedup_vs_compile: float
    case: str = ""          # which shape-matrix case produced this measurement

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["plan"] = dataclasses.asdict(self.plan)
        d["plan"]["overrides"] = [list(o) for o in self.plan.overrides]
        if self.plan.flash_block:
            d["plan"]["flash_block"] = list(self.plan.flash_block)
        return d

    @staticmethod
    def from_json(d: dict) -> "Entry":
        p = dict(d["plan"])
        p["overrides"] = tuple(tuple(o) for o in p.get("overrides", []))
        fb = p.get("flash_block")
        p["flash_block"] = tuple(fb) if fb else None
        return Entry(d["signature"], d["dtype"], d["arch"], d["regime"],
                     Plan(**p), d["utilization"], d["speedup"],
                     d.get("speedup_vs_compile", float("nan")),
                     d.get("case", ""))


class DispatchTable:
    def __init__(self, entries: Optional[List[Entry]] = None,
                 owned: Optional[set] = None):
        self.entries: List[Entry] = entries or []
        # Keys this table may write back. `load` fills it from the *device*
        # layer only, and `add` registers anything new. Without it, saving a
        # table that was loaded as arch+device wrote the architecture entries
        # into the card's own file -- every sm_80 device table came out holding
        # all 27 arch rows, which made "this card's own entry" meaningless and
        # would let a stale copy shadow a re-tuned arch table.
        # A table built directly from a list owns all of it -- the caller is
        # handing us its entries. Only `load` narrows this, to the device layer,
        # so that a merged arch+device table saves back just the device rows.
        self._owned: set = (set(owned) if owned is not None else
                            {(e.arch, e.dtype, e.signature) for e in self.entries})
        self._index: Dict[Tuple[str, str, str], Entry] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._index = {(e.arch, e.dtype, e.signature): e for e in self.entries}

    def add(self, entry: Entry) -> None:
        """Insert an entry, resolving signature collisions conservatively.

        Several *data* variants share one *model* signature: `default` and
        `default_pad` differ only in `padding_ratio`, and `--input-scale` does
        not appear in the config at all. Dispatch happens from the config, so
        one plan has to serve every variant of a shape -- and padding measurably
        raises envelope utilization.

        On collision we therefore keep the **less aggressive** plan rather than
        the faster one. Picking the winner of whichever variant happened to run
        last would silently ship a plan tuned on unpadded data and let it fail on
        padded input, which is a hard `return 2` in the organizer's script.
        `cli verify` re-checks the frozen table against every variant and is the
        backstop for this.

        A re-measurement of the *same* case is a different matter: it supersedes
        the old row outright. Conservative merging is only correct across
        different data variants, and conflating the two makes a fresh sweep
        silently inherit a stale, weaker plan from a previous run.
        """
        key = (entry.arch, entry.dtype, entry.signature)
        self._owned.add(key)
        existing = self._index.get(key)

        if existing is not None and existing.case == entry.case:
            self.entries = [e for e in self.entries
                            if (e.arch, e.dtype, e.signature) != key]
            self.entries.append(entry)
            self._reindex()
            return

        old_risk, new_risk = _aggressiveness(existing.plan) if existing else 0, \
            _aggressiveness(entry.plan)
        if existing is not None and (
                old_risk < new_risk
                # Equal numerical risk: prefer whichever measured faster. This is
                # what keeps `+graph` on a tie -- CUDA graph capture changes no
                # arithmetic, so declining it would cost speed for no safety.
                or (old_risk == new_risk
                    and existing.speedup >= entry.speedup)):
            # Keep the incumbent, but remember the tighter measurement.
            existing.utilization = max(existing.utilization, entry.utilization)
            return
        self.entries = [e for e in self.entries
                        if (e.arch, e.dtype, e.signature) != key]
        if existing is not None:
            entry.utilization = max(existing.utilization, entry.utilization)
        self.entries.append(entry)
        self._reindex()

    # --- persistence -------------------------------------------------------

    @staticmethod
    def path_for(arch: str, device: Optional[str] = None) -> str:
        if device:
            return os.path.join(TABLE_DIR, f"dispatch_{arch}__{device}.json")
        return os.path.join(TABLE_DIR, f"dispatch_{arch}.json")

    def save(self, arch: str, device: Optional[str] = None) -> str:
        os.makedirs(TABLE_DIR, exist_ok=True)
        path = self.path_for(arch, device)
        entries = self.entries
        if device:
            # Only what this card measured for itself; the architecture table
            # stays the fallback and is not duplicated into the overlay.
            entries = [e for e in entries
                       if (e.arch, e.dtype, e.signature) in self._owned]
        blob = {"arch": arch, "entries": [e.to_json() for e in entries]}
        if device:
            blob["device"] = device
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        return path

    @staticmethod
    def load(arch: str, device: Optional[str] = None) -> "DispatchTable":
        """The architecture table, with this card's own entries laid over it.

        Architecture decides what is *legal* -- shared memory, TF32, tensor-core
        support -- so it is the right key for a table that must be correct on
        hardware it has never seen. Which legal plan is *fastest* is a property
        of the card: measured on two `sm_75` parts, six of twelve official
        shapes want different plans, and taking the wrong card's choice costs up
        to 1.08x (`results/same_arch_different_card_sm_75.json`).

        So a device table is an overlay, never a replacement. Entries for this
        exact card win; everything else falls through to the architecture table,
        which keeps an untuned or unknown GPU working exactly as before.
        """
        tiers = []
        # Architecture layer first, then the device layer on top of it.
        for tier in (None, device):
            path = DispatchTable.path_for(arch, tier)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            tiers.append(([Entry.from_json(e) for e in blob.get("entries", [])],
                          tier is not None))

        # Collapse to one entry per key, keeping the last -- the device layer.
        # Without this the list carries both copies, and a later `--demote`
        # would write the duplicates back out.
        merged, owned = {}, set()
        for tier_entries, is_device in tiers:
            for e in tier_entries:
                k = (e.arch, e.dtype, e.signature)
                merged[k] = e
                if is_device:
                    owned.add(k)
        return DispatchTable(list(merged.values()), owned=owned)

    @staticmethod
    def load_for(spec) -> "DispatchTable":
        """Load the table for a probed GPU, device overlay included."""
        from .hw import device_slug
        return DispatchTable.load(spec.arch, device_slug(spec))

    # --- lookup ------------------------------------------------------------

    def lookup(self, arch: str, dtype: torch.dtype, cfg, smem_kb: float = 99.0,
               ) -> Tuple[Plan, str]:
        dt = str(dtype).replace("torch.", "")
        sig = shape_signature(cfg)

        # A shape whose score matrix cannot be allocated must never be given a
        # plan that materializes one, no matter where that plan came from.
        must_stream = score_matrix_bytes(cfg) > 8 * 2**30

        hit = self._index.get((arch, dt, sig))
        if hit is not None:
            plan = dataclasses.replace(hit.plan, smem_kb=smem_kb)
            if must_stream and plan.attention in ("exact", "baseline"):
                return (default_plan(dtype, smem_kb, conservative=True,
                                     arch=arch, must_stream=True), "exact")
            return plan, "exact"

        # Same arch and dtype, nearest by total token count -- shapes close in
        # size sit in the same bottleneck regime far more often than not.
        #
        # But a neighbour's *precision* choices do not transfer, and handing them
        # to an unmeasured shape is how we produced our only correctness
        # failures: `deep` (L=12) and the causal BERT variants inherited an
        # fp16 plan tuned elsewhere and came in at envelope 1.02-1.24. The
        # structural rewrite is bit-exact and its speedup is shape-independent;
        # the fp16 stage overrides are exactly the part that is shape-sensitive.
        # So a fallback keeps the structure and the graph capture and drops the
        # precision narrowing. Slower than a tuned entry, and correct.
        same = [e for e in self.entries if e.arch == arch and e.dtype == dt]
        if same:
            return (default_plan(dtype, smem_kb, conservative=True, arch=arch,
                                 must_stream=must_stream), "nearest")

        return (default_plan(dtype, smem_kb, arch=arch,
                             must_stream=must_stream), "default")


def score_matrix_bytes(cfg) -> int:
    """Bytes the reference's [B, H, S, S] score tensor would need, in fp32.

    Official test shape #14 (B32 S100000 H16) makes this 18.6 TB. Any plan whose
    attention path materializes the score matrix is not slow there -- it is
    impossible. The fallback has to know that.
    """
    return (cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.seq_len * 4)


def default_plan(dtype: torch.dtype, smem_kb: float = 99.0,
                 conservative: bool = False, arch: Optional[str] = None,
                 must_stream: bool = False) -> Plan:
    """The rule used when the table has nothing to say.

    `conservative=True` is for a shape we have never measured (the nearest-
    neighbour fallback). It returns the bit-exact plan on every dtype, because
    precision choices are shape-sensitive and extrapolating them is precisely
    what produced our only correctness failures: `deep` (L=12) and the causal
    BERT variants inherited an fp16 plan tuned on a neighbouring shape and came
    in at envelope 1.02-1.24. The structural rewrite is bit-exact and CUDA graph
    capture changes no arithmetic, so this is still meaningfully faster than the
    baseline -- it just declines to guess at precision.

    fp16/bf16 inputs get the bit-exact plan, because the accuracy gate rejects
    every reassociating optimization at those dtypes -- including PyTorch's own
    `torch.compile(max-autotune)`. They still gain from CUDA-graph capture,
    which removes ~105 kernel launches per forward without touching arithmetic.
    """
    # Pre-Ampere cards have no TF32. Our fp32 default leans on the fact that the
    # reference's own matmuls run at TF32 (10 mantissa bits), which makes fp16 a
    # comparable rounding rather than a regression -- and that reasoning simply
    # does not hold on sm_70/sm_75, where an fp32 matmul really is fp32. We have
    # never measured those, so we do not extrapolate onto them.
    pre_ampere = bool(arch) and _arch_major(arch) < 8

    if conservative or pre_ampere or dtype in (torch.float16, torch.bfloat16):
        if must_stream:
            # The bit-exact plan uses `attention="exact"`, which forms the score
            # matrix. When that cannot fit, the conservative choice is the one
            # that still runs: flash attention, everything else left wide.
            #
            # On an fp32 input and an Ampere-or-newer card the attention stage
            # also narrows to fp16, which is the difference between the flash
            # kernel running and falling through to SDPA -- `tl.dot` needs a
            # narrow float type. On official shape 14 that is 77.2 s -> 20.9 s.
            #
            # This is the one place a *conservative* plan narrows a stage, so
            # the evidence is stated rather than assumed. At the shape-14 config
            # the full-stack envelope with fp16 attention is indistinguishable
            # from the fp32 plan at every length where the reference can be
            # computed (0.20-0.27 for both, S=1024..16384, flat in S), and at
            # the full S=100000 the attention output itself measures 0.034
            # against an exact float64 reference on sampled query rows. The
            # dtype and architecture conditions are the same ones the fp32
            # default below relies on.
            narrow_attn = (dtype not in (torch.float16, torch.bfloat16)
                           and not pre_ampere)
            return Plan(name="stream(flash)+wide" + ("+fp16attn" if narrow_attn
                                                     else ""),
                        compute_dtype="auto", residual_dtype="auto",
                        fuse_qkv=True, attention="flash", fused_norm=False,
                        smem_kb=smem_kb,
                        overrides=(("attn", "float16"),) if narrow_attn else ())
        return dataclasses.replace(SAFE, cuda_graph=True, smem_kb=smem_kb,
                                   name="safe(exact)+graph")
    # fp32: the reference runs at TF32, so fp16 attention and out_proj are
    # measured to cost essentially nothing of the error budget, while the FFN
    # GEMMs (longest reduction dimensions) are kept wide.
    return Plan(name="fp16[attn,out_proj]+graph",
                compute_dtype="auto", residual_dtype="float32",
                fuse_qkv=True, attention="flash", fused_norm=True,
                cuda_graph=True, smem_kb=smem_kb,
                overrides=(("attn", "float16"), ("out_proj", "float16")))


def summarize(table: DispatchTable) -> str:
    if not table.entries:
        return "(empty dispatch table)"
    lines = [f"{'signature':<38} {'dtype':<10} {'plan':<34} {'env':>6} "
             f"{'vs base':>8} {'vs compile':>10}", "-" * 112]
    for e in sorted(table.entries, key=lambda e: (e.dtype, e.signature)):
        lines.append(f"{e.signature:<38} {e.dtype:<10} {e.plan.name:<34} "
                     f"{e.utilization:6.3f} {e.speedup:7.3f}x {e.speedup_vs_compile:9.3f}x")
    return "\n".join(lines)
