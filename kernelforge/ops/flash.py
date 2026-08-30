"""FlashAttention with native causal + key-padding support.

Why write this rather than call `F.scaled_dot_product_attention`: SDPA takes a
causal *flag* or an explicit `attn_mask`, but not both cheaply. When the workload
has causal masking AND per-sequence padding -- which the organizer's script
produces whenever `--causal` meets `--padding-ratio > 0` -- SDPA must materialize
a boolean `attn_mask`, and that drops it off the flash backend onto the
memory-efficient or math path. Handling both predicates in-register costs us
nothing, and it is where we expect to beat the library outright.

Numerics follow the reference exactly: the reference computes its softmax in
fp32 (`torch.softmax(scores.float(), ...)`) and only then casts back to the
working dtype before the P@V matmul. Our online softmax accumulates in fp32 and
casts P to the value dtype for the second dot, which is the same sequence.

The scale is folded into a log2(e) factor so the inner loop can use `exp2`,
which maps to a single hardware MUFU instruction instead of a libdevice call.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

NEG_INF = float("-inf")
LOG2E = 1.4426950408889634


if HAS_TRITON:
    # Triton kernels may only reach globals that are declared constexpr.
    _NEG_INF = tl.constexpr(NEG_INF)
    _LOG2E = tl.constexpr(LOG2E)

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O, KEEP,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        H, N_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        ROUND_SCORES: tl.constexpr,   # 0 none, 1 fp16, 2 bf16
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh
        o_base = O + off_z * stride_oz + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < N_CTX, other=0.0,
        )

        m_i = tl.full([BLOCK_M], _NEG_INF, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        qk_scale = sm_scale * _LOG2E  # fold log2(e) so the inner loop can use exp2

        # Causal masking lets us skip every key block strictly above the diagonal.
        if CAUSAL:
            hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        else:
            hi = N_CTX

        for start_n in range(0, hi, BLOCK_N):
            n_idx = start_n + offs_n
            in_range = n_idx < N_CTX

            k = tl.load(
                k_base + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=in_range[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k))
            if ROUND_SCORES == 0:
                qk = qk * qk_scale
            else:
                # Reproduce the reference's rounding, do not improve on it. The
                # baseline evaluates `matmul(q, k^T) * scale` entirely in the
                # narrow dtype and only then calls .float() for the softmax, so
                # a kernel that keeps full fp32 precision here lands ~0.4% away
                # from the reference and fails the tolerance by being too good.
                if ROUND_SCORES == 1:
                    qk = (qk.to(tl.float16) * sm_scale).to(tl.float16).to(tl.float32)
                else:
                    qk = (qk.to(tl.bfloat16) * sm_scale).to(tl.bfloat16).to(tl.float32)
                qk = qk * _LOG2E

            valid = in_range[None, :]
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= n_idx[None, :])
            if HAS_MASK:
                km = tl.load(KEEP + off_z * N_CTX + n_idx, mask=in_range, other=0)
                valid = valid & (km[None, :] != 0)

            qk = tl.where(valid, qk, _NEG_INF)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            # A tile can be entirely masked, leaving m_ij = -inf. Subtracting that
            # would give inf - inf = NaN, so rescale against 0 in that case; the
            # corresponding probabilities are zero anyway.
            m_safe = tl.where(m_ij == _NEG_INF, 0.0, m_ij)

            p = tl.exp2(qk - m_safe[:, None])
            p = tl.where(valid, p, 0.0)

            alpha = tl.exp2(m_i - m_safe)  # exp2(-inf) = 0 on the first iteration

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(
                v_base + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=in_range[:, None], other=0.0,
            )
            acc = tl.dot(p.to(v.dtype), v, acc)

            m_i = m_ij

        # A query row with no valid keys leaves l_i == 0; emit zeros, not NaN.
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]

        tl.store(
            o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc.to(O.dtype.element_ty),
            mask=offs_m[:, None] < N_CTX,
        )


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# Triton's tl.dot needs a contraction dimension of at least 16, so a head_dim
# below that cannot be fed to the kernel directly. Two official shapes have
# head_dim 8 (d_model 32 / 4 heads, and d_model 128 / 16 heads), so rather than
# dropping them onto SDPA we pad the head dimension up to 16 with zeros. Padding
# is exact: the extra lanes contribute 0 to every dot product, and the output
# columns are sliced away again.
MIN_HEAD_DIM = 16


def flash_supported(head_dim: int) -> bool:
    return HAS_TRITON and (head_dim in (8, 16, 32, 64, 128, 256))


def _pad_head_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    """Zero-pad [B, H, S, Dh] up to `target` on the head dimension."""
    pad = target - t.shape[-1]
    if pad <= 0:
        return t
    return torch.nn.functional.pad(t, (0, pad))


# (BLOCK_M, BLOCK_N, num_warps, num_stages), richest first.
CANDIDATE_BLOCKS: Tuple[Tuple[int, int, int, int], ...] = (
    (128, 128, 8, 4),
    (128, 128, 8, 3),
    (256, 64, 8, 3),
    (128, 64, 8, 4),
    (128, 64, 8, 3),
    (128, 32, 4, 4),
    (64, 64, 4, 4),
    (64, 32, 4, 3),
    (32, 32, 4, 2),
    (16, 16, 4, 2),
)


def smem_bytes(bm: int, bn: int, head_dim: int, stages: int) -> int:
    """Shared memory a configuration needs: a resident Q tile plus multi-buffered
    K and V tiles, at 2 bytes per fp16 element."""
    return (bm * head_dim + stages * 2 * bn * head_dim) * 2


def legal_blocks(seq_len: int, head_dim: int, smem_kb: float):
    """Configurations that actually fit this GPU.

    Shared memory differs by 2.3x across our targets (99 KB on sm_86 vs 228 KB on
    sm_90), so this is derived rather than hardcoded. It is also the check that
    catches the most common LLM failure mode: models trained on A100 kernels
    happily emit 164 KB tilings that will not launch on sm_86.
    """
    budget = smem_kb * 1024 * 0.9
    out = []
    for bm, bn, warps, stages in CANDIDATE_BLOCKS:
        if bm > max(16, _next_pow2(seq_len)):
            continue
        if smem_bytes(bm, bn, head_dim, stages) <= budget:
            out.append((bm, bn, warps, stages))
    return out or [(16, 16, 4, 2)]


def pick_block(seq_len: int, head_dim: int, smem_kb: float, causal: bool = False):
    return legal_blocks(seq_len, head_dim, smem_kb)[0]


def flash_attention(
    q: torch.Tensor,                        # [B, H, S, Dh]
    k: torch.Tensor,
    v: torch.Tensor,
    keep: Optional[torch.Tensor] = None,    # [B, S] key validity, 0/1
    causal: bool = False,
    sm_scale: Optional[float] = None,
    smem_kb: float = 99.0,
    block: Optional[Tuple[int, int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
    round_scores_to: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, S, Dh = q.shape
    assert flash_supported(Dh), f"unsupported head_dim {Dh}"
    # Note the scale is computed from the ORIGINAL head_dim, before padding --
    # the reference divides by sqrt(d_k) of the real head, and padding must not
    # change the arithmetic.
    sm_scale = sm_scale if sm_scale is not None else Dh ** -0.5

    padded_from, out_original = 0, None
    if Dh < MIN_HEAD_DIM:
        padded_from, Dh = Dh, MIN_HEAD_DIM
        q, k, v = (_pad_head_dim(t, MIN_HEAD_DIM) for t in (q, k, v))
        out_original, out = out, None   # caller's buffer is the unpadded width

    # Only the head-dim stride must be unit; everything else is handled by the
    # explicit strides below. Forcing contiguity here would reintroduce the very
    # Memcpy DtoD traffic (4% of baseline GPU time) that we set out to remove.
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1,         "flash_attention needs unit stride along head_dim"
    o = out if out is not None else torch.empty_like(q)
    assert o.stride(-1) == 1, "output needs unit stride along head_dim"

    bm, bn, warps, stages = block or pick_block(S, Dh, smem_kb, causal)

    grid = (triton.cdiv(S, bm), B * H)
    _flash_fwd_kernel[grid](
        q, k, v, o, keep if keep is not None else q,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, S,
        HEAD_DIM=Dh,
        BLOCK_M=bm,
        BLOCK_N=bn,
        CAUSAL=causal,
        HAS_MASK=keep is not None,
        ROUND_SCORES={None: 0, torch.float16: 1,
                      torch.bfloat16: 2}.get(round_scores_to, 0),
        num_warps=warps,
        num_stages=stages,
    )
    if padded_from:
        narrowed = o[..., :padded_from]
        if out_original is not None:
            out_original.copy_(narrowed)
            return out_original
        return narrowed.contiguous()
    return o
