"""Fused residual-add + row-mask + LayerNorm.

Every block boundary in the baseline is the same three-step sequence:

    s = x + sublayer_out          # residual
    s = s * valid_token_mask      # zero padded rows
    h = LayerNorm(s)              # input to the next sublayer

The baseline spends four separate memory round-trips on that. We do it in one:
a single read of `x` and `y`, one write of the new residual stream, one write of
the normalized activation.

Equivalence to the baseline (see docs/EQUIVALENCE.md): the baseline masks the
*attention output* and then masks again at the end of the block, so its residual
stream is already zero on invalid rows before its LayerNorm sees it. Masking the
sum before the norm therefore produces bit-comparable results, while letting us
skip the baseline's separate `masked_fill` passes.

The one place the order genuinely matters is the final norm: `LayerNorm` of an
all-zero row returns `bias`, not zero, so the trailing mask must be applied
*after* normalizing. That is what `mask_out=True` does.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _add_mask_ln_kernel(
        X, Y, KEEP, W, B, S_OUT, H_OUT,
        stride_x, stride_y, stride_s, stride_h,
        n_cols, eps,
        HAS_Y: tl.constexpr,
        HAS_MASK: tl.constexpr,
        MASK_OUT: tl.constexpr,
        WRITE_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        col_mask = cols < n_cols

        # Accumulate the residual stream in fp32 regardless of storage dtype:
        # the stream is summed across 2*num_layers sublayers, and fp16
        # accumulation there is the one place error genuinely compounds.
        x = tl.load(X + row * stride_x + cols, mask=col_mask, other=0.0).to(tl.float32)
        if HAS_Y:
            y = tl.load(Y + row * stride_y + cols, mask=col_mask, other=0.0).to(tl.float32)
            x = x + y

        keep = 1.0
        if HAS_MASK:
            keep = tl.load(KEEP + row).to(tl.float32)
            x = x * keep

        if WRITE_S:
            tl.store(S_OUT + row * stride_s + cols,
                     x.to(S_OUT.dtype.element_ty), mask=col_mask)

        # nn.LayerNorm uses the biased variance and puts eps inside the sqrt.
        mean = tl.sum(x, axis=0) / n_cols
        xc = tl.where(col_mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
        h = xc * rstd * w + b

        if MASK_OUT:
            if HAS_MASK:
                h = h * keep

        tl.store(H_OUT + row * stride_h + cols,
                 h.to(H_OUT.dtype.element_ty), mask=col_mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# Triton needs the whole row in registers; beyond this we fall back to torch.
MAX_BLOCK_D = 16384


def supported(d_model: int) -> bool:
    return HAS_TRITON and _next_pow2(d_model) <= MAX_BLOCK_D


def add_mask_layernorm(
    x: torch.Tensor,                      # [M, d] residual stream in
    y: Optional[torch.Tensor],            # [M, d] sublayer output, or None
    keep: Optional[torch.Tensor],         # [M] 0/1 float, or None
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    mask_out: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    want_residual: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Returns (new_residual_stream, normalized_activation).

    `out_dtype` controls the dtype of the normalized activation, which feeds the
    next GEMM -- setting it to fp16 is how we get tensor-core inputs without ever
    materializing a separate cast.
    """
    assert x.dim() == 2, "pass [M, d]"
    M, d = x.shape
    out_dtype = out_dtype or x.dtype
    residual_dtype = residual_dtype or x.dtype

    if not supported(d):
        return _torch_fallback(x, y, keep, weight, bias, eps, mask_out,
                               out_dtype, residual_dtype, want_residual)

    x = x.contiguous()
    if y is not None:
        y = y.contiguous()

    s_out = torch.empty((M, d), dtype=residual_dtype, device=x.device) if want_residual \
        else x  # unused when WRITE_S is False
    h_out = torch.empty((M, d), dtype=out_dtype, device=x.device)

    block_d = _next_pow2(d)
    num_warps = 4
    if block_d >= 2048:
        num_warps = 8
    if block_d >= 4096:
        num_warps = 16

    _add_mask_ln_kernel[(M,)](
        x, y if y is not None else x, keep if keep is not None else x,
        weight, bias, s_out, h_out,
        x.stride(0), y.stride(0) if y is not None else 0,
        s_out.stride(0), h_out.stride(0),
        d, eps,
        HAS_Y=y is not None,
        HAS_MASK=keep is not None,
        MASK_OUT=mask_out,
        WRITE_S=want_residual,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return (s_out if want_residual else None), h_out


def _torch_fallback(x, y, keep, weight, bias, eps, mask_out,
                    out_dtype, residual_dtype, want_residual):
    s = x.float()
    if y is not None:
        s = s + y.float()
    if keep is not None:
        s = s * keep[:, None].float()
    h = torch.nn.functional.layer_norm(s, (s.shape[-1],), weight.float(),
                                       bias.float(), eps)
    if mask_out and keep is not None:
        h = h * keep[:, None].float()
    return (s.to(residual_dtype) if want_residual else None), h.to(out_dtype)
