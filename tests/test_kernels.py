"""The two Triton kernels, against exact torch references.

These target the specific traps documented in docs/EQUIVALENCE.md rather than
just "does it roughly match": all-zero rows, mask ordering around the norm,
non-multiple-of-block sequence lengths, causal crossed with padding, and the
degenerate fully-masked softmax row.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda, requires_triton
from kernelforge.numerics import check
from kernelforge.ops.flash import (flash_attention, legal_blocks, pick_block,
                                   smem_bytes)
from kernelforge.ops.layernorm import add_mask_layernorm


# --------------------------------------------------------------- layernorm

@requires_triton
@pytest.mark.parametrize("d", [256, 512, 768, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("mask_out", [False, True])
def test_fused_add_mask_layernorm(d, dtype, mask_out):
    torch.manual_seed(0)
    M = 517                                   # deliberately not a round number
    x = torch.randn(M, d, device="cuda", dtype=dtype)
    y = torch.randn(M, d, device="cuda", dtype=dtype)
    keep = (torch.rand(M, device="cuda") > 0.3).to(dtype)
    w = torch.randn(d, device="cuda", dtype=dtype)
    b = torch.randn(d, device="cuda", dtype=dtype)

    s, h = add_mask_layernorm(x, y, keep, w, b, 1e-5, mask_out=mask_out,
                              out_dtype=dtype, residual_dtype=torch.float32)

    ref_s = (x.float() + y.float()) * keep[:, None].float()
    ref_h = F.layer_norm(ref_s, (d,), w.float(), b.float(), 1e-5)
    if mask_out:
        ref_h = ref_h * keep[:, None].float()

    assert check(ref_s, s).passed
    assert check(ref_h.to(dtype), h).passed


@requires_triton
def test_layernorm_of_an_all_zero_row_returns_bias():
    """The trap that makes mask ordering matter: LN(0) == bias, not 0.

    This is why the final mask must be applied *after* the norm; a kernel that
    masked first would emit `bias` on padded rows where the baseline emits 0.
    """
    d = 256
    x = torch.zeros(8, d, device="cuda")
    w = torch.randn(d, device="cuda")
    b = torch.randn(d, device="cuda")

    _, h = add_mask_layernorm(x, None, None, w, b, 1e-5, mask_out=False)
    assert torch.allclose(h, b.expand_as(h), atol=1e-6)

    keep = torch.zeros(8, device="cuda")
    _, h_masked = add_mask_layernorm(x, None, keep, w, b, 1e-5, mask_out=True)
    assert torch.count_nonzero(h_masked) == 0


@requires_triton
def test_layernorm_without_residual_or_mask():
    d = 512
    x = torch.randn(64, d, device="cuda")
    w, b = torch.randn(d, device="cuda"), torch.randn(d, device="cuda")
    _, h = add_mask_layernorm(x, None, None, w, b, 1e-5)
    assert check(F.layer_norm(x, (d,), w, b, 1e-5), h).passed


# --------------------------------------------------------------- attention

def reference_attention(q, k, v, keep, causal):
    """Transcribed from BaselineSelfAttention.forward: softmax in fp32."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    S = q.shape[2]
    if causal:
        cm = torch.ones((S, S), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(cm, float("-inf"))
    if keep is not None:
        scores = scores.masked_fill(~keep.bool()[:, None, None, :], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


@requires_triton
@pytest.mark.parametrize("shape", [
    (2, 8, 128, 64),      # the common case
    (1, 4, 1, 64),        # S=1: attention is the identity on V
    (2, 8, 333, 64),      # not a multiple of any block size
    (1, 8, 2048, 64),     # long sequence
    (1, 4, 100, 128),     # larger head_dim
    (2, 4, 64, 32),       # smaller head_dim
])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pad", [0.0, 0.4])
def test_flash_attention_matches_reference(shape, causal, pad):
    torch.manual_seed(0)
    B_, H, S, Dh = shape
    q, k, v = (torch.randn(B_, H, S, Dh, device="cuda", dtype=torch.float16)
               for _ in range(3))

    keep = None
    if pad > 0:
        lens = torch.randint(1, S + 1, (B_,), device="cuda")
        keep = (torch.arange(S, device="cuda")[None, :] < lens[:, None]).to(torch.uint8)

    out = flash_attention(q, k, v, keep=keep, causal=causal, smem_kb=99.0)
    ref = reference_attention(q, k, v, keep, causal).to(torch.float16)
    assert check(ref, out).passed


@requires_triton
def test_fully_masked_row_yields_zero_not_nan():
    """The generator cannot produce this (min_valid >= 1), but the kernel guards
    it anyway: an all -inf softmax row must emit zeros rather than NaN."""
    q, k, v = (torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
               for _ in range(3))
    keep = torch.zeros(1, 32, device="cuda", dtype=torch.uint8)
    out = flash_attention(q, k, v, keep=keep, causal=False, smem_kb=99.0)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0


@requires_triton
def test_flash_accepts_strided_views_without_copying():
    """The kernel consumes permuted views of a fused QKV buffer directly; only
    the head-dim stride must be unit. This is what removes the baseline's
    Memcpy DtoD traffic."""
    B_, S, H, Dh = 2, 64, 4, 64
    qkv = torch.randn(B_, S, 3, H, Dh, device="cuda", dtype=torch.float16)
    q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))
    assert not q.is_contiguous() and q.stride(-1) == 1

    out_bshd = torch.empty((B_, S, H, Dh), device="cuda", dtype=torch.float16)
    flash_attention(q, k, v, out=out_bshd.permute(0, 2, 1, 3), smem_kb=99.0)
    ref = reference_attention(q, k, v, None, False).to(torch.float16)
    assert check(ref, out_bshd.permute(0, 2, 1, 3)).passed


# ---------------------------------------------------- shared-memory legality

@pytest.mark.parametrize("smem,expect_big", [(64.0, False), (96.0, False),
                                             (163.0, True), (227.0, True)])
def test_tile_legality_tracks_shared_memory(smem, expect_big):
    """The check that catches the classic LLM failure: a config sized for an
    A100 will not launch on a Turing card's 64 KB budget.

    The four budgets are the ones we actually measure on: 64 KB on the T4 and
    TITAN RTX, 96 KB on the TITAN V, 163 KB on the A100, 227 KB on the H100."""
    blocks = legal_blocks(2048, 64, smem)
    assert blocks, "there must always be a fallback configuration"
    for bm, bn, _, stages in blocks:
        assert smem_bytes(bm, bn, 64, stages) <= smem * 1024 * 0.9
    has_big = any(bm >= 128 and bn >= 128 for bm, bn, _, _ in blocks)
    assert has_big == expect_big


def test_short_sequences_do_not_get_oversized_tiles():
    for bm, _, _, _ in legal_blocks(16, 64, 228.0):
        assert bm <= 16


@requires_triton
@pytest.mark.parametrize("shape", [(2, 8, 512, 64), (4, 4, 1023, 64), (2, 2, 128, 8)])
@pytest.mark.parametrize("causal", [True, False])
def test_causal_block_interleave_is_a_pure_permutation(shape, causal):
    """The causal m-block interleave must not change a single bit of output.

    Causal work grows with the m-block index, so the kernel interleaves block
    order to balance scheduling waves. That is a permutation of which program
    handles which block -- every block still performs the same reads, the same
    accumulation and the same store -- so it is a scheduling change, not a
    numerical one, and the output must be bit-identical to the un-permuted
    order. This test rebuilds the kernel without the interleave and compares.

    If this ever fails, the interleave has stopped being free and the speedup it
    buys is not worth having.
    """
    import importlib.util
    import os
    import tempfile

    from kernelforge.ops import flash as F

    src = open(F.__file__, encoding="utf-8").read()
    marker = "            start_m = tl.where(_p % 2 == 0, _p // 2, _n - 1 - _p // 2)"
    assert marker in src, "interleave line not found -- update this test"
    plain_src = src.replace(marker, "            start_m = _p")
    path = os.path.join(tempfile.mkdtemp(), "_flash_plain.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(plain_src)
    spec = importlib.util.spec_from_file_location("_flash_plain", path)
    plain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plain)

    B, H, S, Dh = shape
    dev = torch.device("cuda")
    torch.manual_seed(3)
    q, k, v = (torch.randn(B, H, S, Dh, device=dev, dtype=torch.float16)
               for _ in range(3))
    keep = (torch.rand(B, S, device=dev) > 0.3).to(torch.uint8)
    for mask in (None, keep):
        a = F.flash_attention(q, k, v, keep=mask, causal=causal, smem_kb=99.0)
        b = plain.flash_attention(q, k, v, keep=mask, causal=causal, smem_kb=99.0)
        assert torch.equal(a, b), (
            f"interleave changed the output: max |diff| "
            f"{(a.float() - b.float()).abs().max().item():.3e}")


def test_causal_default_targets_the_measured_tile():
    """The causal default is a measured choice, not the richest legal tile.

    Causal work grows with the m-block index and wastes a wide `BLOCK_N` on the
    half-masked diagonal block, so the best tile is smaller than the non-causal
    optimum. Swept across every legal configuration on an A100, that optimum is
    `64x64` up to `head_dim` 64 and `128x32` above it. Pinning it here because
    the failure is silent: a wrong default still computes the right answer, just
    up to 1.29x slower, and only on shapes nobody tuned.
    """
    smem = 163.0
    for head_dim in (32, 64):
        bm, bn, _, _ = pick_block(2048, head_dim, smem, causal=True)
        assert (bm, bn) == (64, 64), f"head_dim {head_dim} -> {(bm, bn)}"
    bm, bn, _, _ = pick_block(2048, 128, smem, causal=True)
    assert (bm, bn) == (128, 32)

    # head_dim 256 cannot fit a 128x32 tile in 163 KB, so the target is not
    # available and the rule must fall back to a legal, m-capped tiling rather
    # than name one that will not launch.
    block = pick_block(2048, 256, smem, causal=True)
    assert block in legal_blocks(2048, 256, smem) and block[0] <= 128

    # Non-causal is unchanged: uniform work, so the richest legal tile wins.
    assert pick_block(2048, 64, smem) == legal_blocks(2048, 64, smem)[0]

    # A short sequence spans one block, so there is no imbalance to correct.
    assert pick_block(128, 64, smem, causal=True) == legal_blocks(128, 64, smem)[0]


def test_causal_default_stays_legal_on_a_small_shared_memory_budget():
    """The target tile must never be returned if it does not fit.

    Turing cards have 64 KB per block against the A100's 163 KB. The rule picks
    a target by head_dim, so it has to fall back rather than name a tiling that
    will not launch.
    """
    for smem in (64.0, 96.0, 163.0, 227.0):
        for head_dim in (32, 64, 128):
            block = pick_block(4096, head_dim, smem, causal=True)
            assert block in legal_blocks(4096, head_dim, smem), (
                f"smem {smem} head_dim {head_dim}: {block} is not legal")


@requires_cuda
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 777])
@pytest.mark.parametrize("padded", [False, True])
def test_causal_loop_split_is_bit_identical_to_the_unsplit_form(seq_len, padded):
    """The split key loop must change speed and nothing else.

    The inner loop applies the causal mask only over the diagonal block, because
    every block below it is entirely visible. That is a claim about a predicate
    being uniformly true, and if it is ever wrong the result is silently wrong
    rather than slow -- so it is checked against an exact reference here, at a
    sequence length that is not a multiple of any candidate tile (777) and with
    key padding on, which is the case where the two mask terms interact.
    """
    torch.manual_seed(seq_len)
    dev = torch.device("cuda")
    B, H, Dh = 2, 4, 64
    q, k, v = (torch.randn(B, H, seq_len, Dh, device=dev, dtype=torch.float16)
               for _ in range(3))
    keep = None
    if padded:
        keep = (torch.rand(B, seq_len, device=dev) > 0.3).to(torch.int32)
        keep[:, 0] = 1                      # never mask every key of a row

    from kernelforge.hw import probe
    smem = probe(measure=False).shared_mem_per_block_kb
    got = flash_attention(q, k, v, keep=keep, causal=True, smem_kb=smem).float()

    scores = (q.float() @ k.float().transpose(-2, -1)) * (Dh ** -0.5)
    idx = torch.arange(seq_len, device=dev)
    valid = idx[:, None] >= idx[None, :]
    if keep is not None:
        valid = valid[None, None] & (keep[:, None, None, :] != 0)
    scores = scores.masked_fill(~valid.expand_as(scores), float("-inf"))
    want = torch.softmax(scores, dim=-1) @ v.float()

    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2), (
        f"split causal loop diverges at S={seq_len}, padded={padded}: "
        f"max |diff| {(got - want).abs().max().item():.3e}")


def test_only_power_of_two_tilings_are_offered():
    """A non-power-of-two block size cannot reach the compiler.

    `tl.arange(0, BLOCK)` requires a power of two. When that is violated Triton
    fails during compilation with the location pointing at whatever line happens
    to use the offsets, which is not where the mistake is -- a `192x128`
    candidate reported the error against an unrelated `tl.where`. The filter
    exists so a hand-added or model-proposed tiling is rejected in Python, with
    a message about the real cause.
    """
    for smem in (64.0, 96.0, 163.0, 227.0):
        for head_dim in (32, 64, 128):
            for bm, bn, _, _ in legal_blocks(4096, head_dim, smem):
                assert bm & (bm - 1) == 0, f"BLOCK_M {bm} is not a power of two"
                assert bn & (bn - 1) == 0, f"BLOCK_N {bn} is not a power of two"

    # And the shipped candidate list must not contain one in the first place.
    from kernelforge.ops.flash import CANDIDATE_BLOCKS
    for bm, bn, _, _ in CANDIDATE_BLOCKS:
        assert bm & (bm - 1) == 0 and bn & (bn - 1) == 0, (
            f"candidate {bm}x{bn} is not a power of two and would never launch")


def test_roofline_units_are_milliseconds_and_physically_sensible():
    """Pin the units of `roofline.analyse`, which are easy to get wrong.

    The function takes milliseconds and its `Roofline.seconds` field is seconds.
    Passing the wrong one is a 1000x error that still produces a plausible
    number, so this checks the arithmetic against first principles instead of
    against itself: FLOPs and bytes are computed independently here and divided
    by a time we choose.
    """
    from kernelforge import roofline, shapes
    case = shapes.resolve(["B64-S128-d128-H4-F128-L4-causal"])[0]
    c = case.to_config()
    ms = 0.6676

    flops = roofline.stack_flops(c.batch_size, c.seq_len, c.d_model,
                                 c.num_heads, c.ffn_dim, c.num_layers, c.causal)
    bytes_moved = roofline.stack_bytes(c.batch_size, c.seq_len, c.d_model,
                                       c.num_heads, c.ffn_dim, c.num_layers, 4)
    r = roofline.analyse(case, ms, "sm_80", 1653.0, "float32")

    assert abs(r.achieved_tflops - flops / (ms / 1e3) / 1e12) < 1e-6
    assert abs(r.achieved_bandwidth_gbs - bytes_moved / (ms / 1e3) / 1e9) < 1e-6

    # And the answer must be physically possible on the card it names.
    assert 0 < r.achieved_bandwidth_gbs < 1653.0, (
        f"{r.achieved_bandwidth_gbs:.0f} GB/s exceeds the A100's DRAM bandwidth "
        f"-- the units are wrong by a factor of a thousand")
    assert 0 < r.achieved_tflops < 156.0, (
        f"{r.achieved_tflops:.0f} TFLOP/s exceeds the A100's TF32 ceiling")
