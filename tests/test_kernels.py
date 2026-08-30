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

from conftest import requires_triton
from kernelforge.numerics import check
from kernelforge.ops.flash import (flash_attention, legal_blocks, smem_bytes)
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

@pytest.mark.parametrize("smem,expect_big", [(99.0, False), (164.0, True),
                                             (228.0, True)])
def test_tile_legality_tracks_shared_memory(smem, expect_big):
    """The check that catches the classic LLM failure: a config sized for an
    A100 will not launch on sm_86's 99 KB budget."""
    blocks = legal_blocks(2048, 64, smem)
    assert blocks, "there must always be a fallback configuration"
    for bm, bn, _, stages in blocks:
        assert smem_bytes(bm, bn, 64, stages) <= smem * 1024 * 0.9
    has_big = any(bm >= 128 and bn >= 128 for bm, bn, _, _ in blocks)
    assert has_big == expect_big


def test_short_sequences_do_not_get_oversized_tiles():
    for bm, _, _, _ in legal_blocks(16, 64, 228.0):
        assert bm <= 16
