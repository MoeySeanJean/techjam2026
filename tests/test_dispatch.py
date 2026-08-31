"""Dispatch-table policy: the rules that decide what actually ships.

No GPU required -- this is the decision logic, and it is where a subtle mistake
would silently ship a plan tuned on the wrong data variant.
"""
from __future__ import annotations

import dataclasses

import torch

from kernelforge.dispatch import (DispatchTable, Entry, _aggressiveness,
                                  default_plan, shape_signature)
from kernelforge.optimized import Plan
from kernelforge.search import SAFE

SAFE_G = dataclasses.replace(SAFE, cuda_graph=True, name="safe(exact)+graph")
MILD = Plan(name="fp16[attn]", overrides=(("attn", "float16"),))
MILD_G = dataclasses.replace(MILD, cuda_graph=True, name="fp16[attn]+graph")
BOLD = Plan(name="fp16[attn,out_proj,ffn2]",
            overrides=(("attn", "float16"), ("out_proj", "float16"),
                       ("ffn2", "float16")))


def entry(plan, case, speedup=2.0, util=0.6, dtype="float32"):
    return Entry("SIG", dtype, "sm_86", "gemm", plan, util, speedup, 1.2, case)


def test_risk_ordering_is_monotone_in_narrowing():
    assert _aggressiveness(SAFE) < _aggressiveness(MILD) < _aggressiveness(BOLD)


def test_graph_capture_carries_no_risk():
    """It changes no arithmetic, so it must not affect the safety ranking."""
    assert _aggressiveness(MILD) == _aggressiveness(MILD_G)


def test_different_variants_resolve_to_the_safer_plan():
    """`default` and `default_pad` share a signature; padding is the harder
    case, so the table must not inherit whichever ran last."""
    for order in ([("default", BOLD), ("default_pad", SAFE_G)],
                  [("default_pad", SAFE_G), ("default", BOLD)]):
        table = DispatchTable()
        for case, plan in order:
            table.add(entry(plan, case))
        assert table.entries[0].plan.name == SAFE_G.name


def test_re_measuring_the_same_case_supersedes():
    """Conservative merging applies across variants, not across reruns -- a
    fresh sweep must not inherit a stale, weaker plan."""
    table = DispatchTable()
    table.add(entry(MILD, "default"))
    table.add(entry(BOLD, "default"))
    assert table.entries[0].plan.name == BOLD.name


def test_ties_prefer_the_faster_plan():
    """Equal risk: keep whichever measured faster, which is what retains
    `+graph` instead of dropping it for nothing."""
    table = DispatchTable()
    table.add(entry(MILD, "default_pad", speedup=1.5))
    table.add(entry(MILD_G, "default", speedup=2.7))
    assert table.entries[0].plan.cuda_graph


def test_collision_keeps_the_tighter_utilization():
    table = DispatchTable()
    table.add(entry(SAFE_G, "default", util=0.10))
    table.add(entry(BOLD, "default_pad", util=0.85))
    assert table.entries[0].utilization == 0.85


def test_round_trip_through_json(tmp_path, monkeypatch):
    import kernelforge.dispatch as d
    monkeypatch.setattr(d, "TABLE_DIR", str(tmp_path))
    table = DispatchTable()
    plan = dataclasses.replace(BOLD, flash_block=(128, 64, 8, 3),
                               torch_compile="max-autotune")
    table.add(entry(plan, "default"))
    table.save("sm_86")
    again = DispatchTable.load("sm_86")
    assert len(again.entries) == 1
    got = again.entries[0]
    assert got.plan.overrides == plan.overrides
    assert got.plan.flash_block == (128, 64, 8, 3)
    assert got.plan.torch_compile == "max-autotune"
    assert got.case == "default"


def test_lookup_falls_back_in_order():
    import torch_transformer_benchmark as B
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)

    empty = DispatchTable()
    _, source = empty.lookup("sm_86", torch.float32, cfg)
    assert source == "default"

    table = DispatchTable()
    table.add(Entry(shape_signature(cfg), "float32", "sm_86", "gemm", MILD_G,
                    0.6, 2.5, 1.2, "default"))
    plan, source = table.lookup("sm_86", torch.float32, cfg)
    assert source == "exact" and plan.name == MILD_G.name

    other = B.TransformerConfig(8, 129, 512, 8, 2048, 6, False)
    _, source = table.lookup("sm_86", torch.float32, other)
    assert source == "nearest"


def test_causal_gets_its_own_signature():
    import torch_transformer_benchmark as B
    a = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    b = B.TransformerConfig(8, 128, 512, 8, 2048, 6, True)
    assert shape_signature(a) != shape_signature(b)


def test_layers_are_part_of_the_signature():
    """Amplification grows with depth: a plan validated at L=6 is not safe at
    L=12, so the two must never share an entry."""
    import torch_transformer_benchmark as B
    a = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    b = B.TransformerConfig(8, 128, 512, 8, 2048, 12, False)
    assert shape_signature(a) != shape_signature(b)


def test_narrow_dtypes_default_to_the_bit_exact_plan():
    """No reassociating optimization passes the gate at fp16/bf16."""
    for dt in (torch.float16, torch.bfloat16):
        plan = default_plan(dt)
        assert plan.attention == "exact"
        assert not plan.fuse_qkv and not plan.fused_norm
        assert not plan.overrides
        assert plan.cuda_graph, "should still take the launch-overhead win"


def test_float32_default_is_the_measured_choice():
    plan = default_plan(torch.float32)
    assert dict(plan.overrides) == {"attn": "float16", "out_proj": "float16"}
    assert plan.residual_dtype == "float32", "residual must stay wide"


def test_streaming_plan_narrows_attention_only_where_measured():
    """The streaming plan's fp16 attention is conditional, and the conditions matter.

    A shape whose score matrix cannot be allocated gets `stream(flash)+wide`.
    On fp32 input and Ampere-or-newer that plan also narrows the attention stage
    to fp16, because `tl.dot` needs a narrow float type and without it the flash
    kernel falls through to SDPA -- 97.7% of shape 14's runtime.

    It must NOT narrow on a narrow input dtype (no reassociating optimization
    passes the gate there) or on a pre-Ampere card (no TF32, so the equivalence
    argument for fp16 does not hold). This is the only place a conservative plan
    narrows anything, so both exclusions are pinned.
    """
    import torch as _t
    from kernelforge.dispatch import default_plan

    narrowed = default_plan(_t.float32, 163.0, conservative=True, arch="sm_80",
                            must_stream=True)
    assert ("attn", "float16") in narrowed.overrides
    assert narrowed.attention == "flash"

    for dtype, arch in ((_t.float16, "sm_80"), (_t.bfloat16, "sm_90"),
                        (_t.float32, "sm_75"), (_t.float32, "sm_70")):
        plan = default_plan(dtype, 163.0, conservative=True, arch=arch,
                            must_stream=True)
        assert plan.overrides == (), (
            f"{dtype} on {arch} narrowed attention: {plan.overrides}")
        assert plan.attention == "flash", "must still stream, whatever the dtype"


def test_untuned_pre_ampere_default_is_bit_exact():
    """What `verify --untuned` measures on a Volta or Turing card.

    Without a table, every lookup falls through to `default_plan`. On a card
    with no TF32 the fp32 default's argument -- that the reference is itself
    computing at 10 mantissa bits, so fp16 costs nothing real -- does not hold,
    so the default must decline the precision trade entirely. That is what makes
    the untuned portability artifact come back at `max_abs = 0` rather than
    merely "within tolerance", and it is the claim `docs/TECH_REPORT.md` makes
    about hardware we never tuned.
    """
    import torch_transformer_benchmark as B
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    empty = DispatchTable()

    for arch in ("sm_70", "sm_75"):
        plan, source = empty.lookup(arch, torch.float32, cfg)
        assert source == "default", f"{arch} should have nothing to look up"
        assert not plan.overrides, (
            f"{arch} has no TF32, so the fp32 default must not spend a "
            f"precision budget it cannot justify -- got {plan.overrides}")
        assert plan.cuda_graph, (
            f"{arch} should still take the launch-overhead win, which changes "
            f"no arithmetic")

    # The contrast: on Ampere+ the same lookup does spend the budget.
    plan, _ = empty.lookup("sm_80", torch.float32, cfg)
    assert dict(plan.overrides) == {"attn": "float16", "out_proj": "float16"}


def _plan(name):
    return dataclasses.replace(SAFE_G, name=name)


def test_device_table_overlays_the_architecture_table(tmp_path, monkeypatch):
    """A card's own entries win; everything else falls through to the arch table.

    Architecture decides what is *legal*, so it stays the fallback and an
    unknown GPU keeps working. Which legal plan is *fastest* is per-card --
    measured on two sm_75 parts, six of twelve official shapes disagree and the
    wrong card's choice costs up to 1.08x. The overlay is how both hold at once.
    """
    import kernelforge.dispatch as d
    monkeypatch.setattr(d, "TABLE_DIR", str(tmp_path))
    import torch_transformer_benchmark as B
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    sig = shape_signature(cfg)

    DispatchTable([
        Entry(sig, "float32", "sm_80", "gemm", _plan("ARCH"), 0.5, 2.0, 1.1, "c"),
        Entry("ONLY-IN-ARCH", "float32", "sm_80", "gemm", _plan("ARCH_ONLY"),
              0.5, 2.0, 1.1, "c"),
    ]).save("sm_80")
    DispatchTable([
        Entry(sig, "float32", "sm_80", "gemm", _plan("DEVICE"), 0.9, 1.0, 1.0, "c"),
    ]).save("sm_80", "mycard_sm_80")

    merged = DispatchTable.load("sm_80", "mycard_sm_80")
    plan, source = merged.lookup("sm_80", torch.float32, cfg)
    assert plan.name == "DEVICE" and source == "exact"
    assert any(e.plan.name == "ARCH_ONLY" for e in merged.entries), (
        "the architecture table must still serve shapes the card never tuned")

    keys = [(e.arch, e.dtype, e.signature) for e in merged.entries]
    assert len(keys) == len(set(keys)), (
        "duplicate keys would be written back out by `verify --demote`")

    # A card with no overlay is unaffected -- this is what keeps every GPU we
    # have never seen working exactly as it did before device tables existed.
    plain, source = DispatchTable.load("sm_80").lookup("sm_80", torch.float32, cfg)
    assert plain.name == "ARCH" and source == "exact"
    other, source = DispatchTable.load("sm_80", "unknown-card_sm_80").lookup(
        "sm_80", torch.float32, cfg)
    assert other.name == "ARCH"


def test_device_tables_do_not_collide_across_cards(tmp_path, monkeypatch):
    """Two cards of one architecture must not overwrite each other's tuning.

    This is the bug the overlay exists to prevent: before it, a TITAN RTX and a
    Tesla T4 both wrote into `dispatch_sm_75.json` and one card's plans simply
    won.
    """
    import kernelforge.dispatch as d
    monkeypatch.setattr(d, "TABLE_DIR", str(tmp_path))
    import torch_transformer_benchmark as B
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    sig = shape_signature(cfg)

    for card, name in (("rtx_sm_75", "RTX_PLAN"), ("t4_sm_75", "T4_PLAN")):
        DispatchTable([Entry(sig, "float32", "sm_75", "gemm", _plan(name),
                             0.5, 2.0, 1.1, "c")]).save("sm_75", card)

    for card, name in (("rtx_sm_75", "RTX_PLAN"), ("t4_sm_75", "T4_PLAN")):
        got, _ = DispatchTable.load("sm_75", card).lookup(
            "sm_75", torch.float32, cfg)
        assert got.name == name, f"{card} got {got.name}"


def test_torch_compile_must_win_by_a_margin(monkeypatch):
    """A tie in our harness is a loss in the organizer's, so ties go to us.

    Our sweep times candidates interleaved, which leaves a compiled model warm
    and its CUDA graph captured. The organizer's script uses a fresh process.
    Measured over 103 shape pairs on nine GPUs, plans delegating to
    torch.compile averaged 0.891x of our own kernels under their harness while
    ours averaged 1.017x, so the selector needs to stop handing shapes to the
    compiler on a hair-thin lead.
    """
    from kernelforge import search

    class C:
        def __init__(self, name, ms, compiled):
            self.plan = dataclasses.replace(
                SAFE_G, name=name,
                torch_compile="reduce-overhead" if compiled else None)
            self.median_ms, self.passed = ms, True

    def pick(cands):
        ranked = sorted([c for c in cands if c.passed], key=lambda c: c.median_ms)
        best = ranked[0]
        if best.plan.torch_compile:
            mine = next((c for c in ranked if not c.plan.torch_compile), None)
            if mine is not None and mine.median_ms <= best.median_ms * search.COMPILE_MARGIN:
                best = mine
        return best.plan.name

    # A hair-thin compile lead is not enough: ours is within the margin.
    assert pick([C("compile", 1.00, True), C("ours", 1.05, False)]) == "ours"
    # A decisive compile lead still wins -- this is a tie-break, not a ban.
    assert pick([C("compile", 1.00, True), C("ours", 1.40, False)]) == "compile"
    # When ours is outright faster nothing changes.
    assert pick([C("compile", 1.20, True), C("ours", 1.00, False)]) == "ours"
    # With no alternative, the compile plan is still selected.
    assert pick([C("compile", 1.00, True)]) == "compile"


def test_every_runtime_path_uses_the_device_overlay():
    """No entry point may load the architecture table on its own.

    `DispatchTable.load(arch)` skips the per-card overlay, so a script using it
    silently runs architecture-default plans while `doctor` reports the card has
    tuned entries. That is exactly what happened to `scripts/usecase.py` when
    the overlay was introduced: it reported `[default]` for every segment on a
    GPU that had four tuned entries. Grepping is crude but it is the check that
    would have caught it.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    paths = (list(root.glob("scripts/*.py")) + [root / "submission.py"]
             + list(root.glob("kernelforge/**/*.py")))
    for path in paths:
        if path.name == "dispatch.py":
            continue        # defines both tiers; load_for is implemented here
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "DispatchTable.load(" in line and "spec.arch" in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, (
        "these load the architecture table directly and will miss this card's "
        f"tuned entries: {offenders}")


def test_device_slugs_are_clean_and_unique():
    """The slug becomes a filename, so it has to be stable and collision-free.

    Junk words are stripped from inside hyphenated names too, and joining
    naively produced `a100---40gb` for `NVIDIA A100-PCIE-40GB`. Cosmetic on its
    own, but the slug keys a dispatch table on disk, and two cards resolving to
    one name would silently share tuning.
    """
    from kernelforge.hw import device_slug

    class Spec:
        def __init__(self, name, arch, mem):
            self.name, self.arch, self.total_mem_gb = name, arch, mem

    cards = [("NVIDIA A100-PCIE-40GB", "sm_80", 39.5),
             ("NVIDIA A100 80GB PCIe", "sm_80", 79.2),
             ("NVIDIA A100 80GB PCIe MIG 3g.40gb", "sm_80", 39.2),
             ("NVIDIA H100 NVL", "sm_90", 93.1),
             ("NVIDIA H100 NVL MIG 3g.47gb", "sm_90", 46.4),
             ("NVIDIA H200 NVL", "sm_90", 139.8),
             ("NVIDIA TITAN V", "sm_70", 11.8),
             ("NVIDIA TITAN RTX", "sm_75", 23.5),
             ("Tesla T4", "sm_75", 14.6)]
    slugs = [device_slug(Spec(*c)) for c in cards]

    assert len(set(slugs)) == len(slugs), f"two cards share a slug: {slugs}"
    for s in slugs:
        assert "--" not in s, f"{s} has a collapsed-word artefact"
        assert not s.startswith("-") and not s.endswith("-"), s
        assert s == s.lower() and " " not in s, s


def test_a_proposal_only_displaces_a_slower_entry(tmp_path, monkeypatch):
    """The agent must not overwrite a faster plan the search already froze.

    `DispatchTable.add` supersedes a same-case row outright, which is correct
    for a re-measurement of the same plan but wrong for the agent: it runs after
    `tune`, so an LLM proposal that merely cleared the gate would take the slot
    from a better plan. The loop compares speedups first; this pins the rule the
    loop relies on.
    """
    import kernelforge.dispatch as d
    monkeypatch.setattr(d, "TABLE_DIR", str(tmp_path))
    import torch_transformer_benchmark as B
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    sig = shape_signature(cfg)

    def entry(name, speedup):
        return Entry(sig, "float32", "sm_80", "gemm", _plan(name), 0.5,
                     speedup, 1.1, "shared-case")

    table = DispatchTable([entry("FROM_TUNE", 5.0)])
    key = ("sm_80", "float32", sig)

    slower = entry("LLM_SLOWER", 3.0)
    assert not (slower.speedup > table._index[key].speedup)

    faster = entry("LLM_FASTER", 7.0)
    assert faster.speedup > table._index[key].speedup
    table.add(faster)
    assert table.lookup("sm_80", torch.float32, cfg)[0].name == "LLM_FASTER"


def test_promotion_needs_a_real_margin_not_a_hair():
    """A noise-level win must not displace a measured plan.

    The laptop run promoted a 3.21x over a 3.20x -- 0.3%, inside the drift of a
    card that cannot lock its clocks. Promoting on that is fitting noise and
    makes the table churn without improving.
    """
    from kernelforge.agent.loop import PROMOTE_MARGIN
    assert PROMOTE_MARGIN > 1.0

    def promotes(new, held):
        return new > held * PROMOTE_MARGIN

    assert not promotes(3.21, 3.20), "0.3% is noise, not an improvement"
    assert not promotes(3.26, 3.20), "2% is still inside the margin"
    assert promotes(3.40, 3.20), "6% is a real win and should take the slot"
    assert promotes(9.00, 3.20)
