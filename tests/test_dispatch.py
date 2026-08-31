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
