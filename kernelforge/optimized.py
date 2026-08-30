"""The optimized Transformer stack, expressed as a tunable plan.

Every optimization is a field on `Plan` rather than a hardcoded choice, because
the agent's job is to search this space per shape and per GPU. A `Plan` plus a
shape is what gets frozen into the dispatch table.

Structural rewrite relative to the baseline
-------------------------------------------
The baseline runs, per layer: 4 elementwise/reduction passes for the two
LayerNorms and two residual adds, 3 separate Q/K/V GEMMs, 4 `.contiguous()`
copies for head splitting and merging, and a materialized [B,H,S,S] score
tensor. We collapse that to:

  * one fused `add + mask + LayerNorm` kernel per block boundary, which writes
    the new residual stream and the next sublayer's normalized input in a
    single pass (and reads the *next* layer's norm weights, so a block boundary
    costs one kernel, not four);
  * one fused QKV GEMM instead of three;
  * zero head-split copies -- the flash kernel consumes strided views of the
    fused QKV buffer directly, and writes its output straight into a [B,S,H,Dh]
    buffer whose flat view is the merged-head layout;
  * FlashAttention, so [B,H,S,S] is never allocated.

Precision
---------
In `float32` mode the reference itself runs its matmuls at TF32 (the script sets
`allow_tf32=True` and `matmul_precision='high'` by default). TF32 carries 10
mantissa bits; fp16 carries 11. Computing the GEMMs in fp16 with fp32
accumulation is therefore not a precision regression against the reference --
it is a different rounding of comparable width. `residual_dtype` is kept
separately settable because the residual stream is summed across 2*num_layers
sublayers, and that is the one place where narrow accumulation actually
compounds.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

from .ops.flash import flash_attention, flash_supported
from .ops.layernorm import add_mask_layernorm
from .ops.layernorm import supported as ln_supported

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


@dataclasses.dataclass(frozen=True)
class Plan:
    """A point in the optimization space. This is what the agent searches."""

    name: str = "default"
    compute_dtype: str = "float16"     # default dtype for GEMM inputs / attention
    residual_dtype: str = "float32"    # dtype of the residual stream
    fuse_qkv: bool = True
    attention: str = "flash"           # flash | sdpa | baseline
    fused_norm: bool = True            # Triton add+mask+LN
    cuda_graph: bool = False
    # torch.compile mode ("default" | "reduce-overhead" | "max-autotune"), or
    # None for eager. The organizer's script lists torch.compile as a suggested
    # optimization direction, so it competes in the search on equal terms and is
    # held to the same accuracy gate as our own kernels -- which is why it is
    # never selected at float16/bfloat16, where it cannot pass.
    torch_compile: Optional[str] = None
    flash_block: Optional[Tuple[int, int, int, int]] = None
    smem_kb: float = 99.0
    # Per-stage precision overrides, e.g. (("ffn2", "float32"),). Stages are
    # "attn" (norm1 output, QKV GEMM, attention), "out_proj", "ffn1" (norm2
    # output, first FFN GEMM, GELU) and "ffn2". The error-budget tool measures
    # each stage independently so this can be set from evidence, not intuition.
    overrides: Tuple[Tuple[str, str], ...] = ()
    # Reproduce the reference's own precision loss rather than avoiding it: in
    # fp16/bf16 mode the baseline stores attention scores in the narrow dtype
    # before its fp32 softmax, and being *more* accurate than the reference is
    # itself a tolerance failure. See docs/PRECISION.md.
    match_score_rounding: bool = False

    def compute_torch_dtype(self, io_dtype: torch.dtype) -> torch.dtype:
        if self.compute_dtype == "auto":
            return io_dtype
        return DTYPES[self.compute_dtype]

    def residual_torch_dtype(self, io_dtype: torch.dtype) -> torch.dtype:
        if self.residual_dtype == "auto":
            return io_dtype
        return DTYPES[self.residual_dtype]

    def stage_dtype(self, stage: str, io_dtype: torch.dtype) -> torch.dtype:
        name = dict(self.overrides).get(stage, self.compute_dtype)
        return io_dtype if name == "auto" else DTYPES[name]

    def with_override(self, stage: str, dtype_name: str) -> "Plan":
        kept = tuple((s, d) for s, d in self.overrides if s != stage)
        return dataclasses.replace(
            self, overrides=kept + ((stage, dtype_name),),
            name=f"{self.name}+{stage}:{dtype_name}")

    def describe(self) -> str:
        bits = [f"compute={self.compute_dtype}", f"residual={self.residual_dtype}",
                f"attn={self.attention}"]
        if self.fuse_qkv:
            bits.append("fused-qkv")
        if self.fused_norm:
            bits.append("fused-norm")
        if self.cuda_graph:
            bits.append("cuda-graph")
        if self.torch_compile:
            bits.append(f"torch.compile[{self.torch_compile}]")
        if self.match_score_rounding:
            bits.append("score-rounding")
        if self.overrides:
            bits.append("[" + ",".join(f"{s}={d}" for s, d in self.overrides) + "]")
        if self.flash_block:
            bits.append(f"block={self.flash_block}")
        return " ".join(bits)


# Named plans forming the optimization ladder. Each step is independently
# measurable, which is what makes the ablation table in the report meaningful.
LADDER: Dict[str, Plan] = {
    "baseline":  Plan("baseline", compute_dtype="auto", residual_dtype="auto",
                      fuse_qkv=False, attention="baseline", fused_norm=False),
    "cleanup":   Plan("cleanup", compute_dtype="auto", residual_dtype="auto",
                      fuse_qkv=True, attention="sdpa", fused_norm=False),
    "fused_norm": Plan("fused_norm", compute_dtype="auto", residual_dtype="auto",
                       fuse_qkv=True, attention="sdpa", fused_norm=True),
    "half":      Plan("half", compute_dtype="float16", residual_dtype="float32",
                      fuse_qkv=True, attention="sdpa", fused_norm=True),
    "flash":     Plan("flash", compute_dtype="float16", residual_dtype="float32",
                      fuse_qkv=True, attention="flash", fused_norm=True),
    "half_res":  Plan("half_res", compute_dtype="float16", residual_dtype="float16",
                      fuse_qkv=True, attention="flash", fused_norm=True),
    "graph":     Plan("graph", compute_dtype="float16", residual_dtype="float32",
                      fuse_qkv=True, attention="flash", fused_norm=True,
                      cuda_graph=True),
    "graph_half_res": Plan("graph_half_res", compute_dtype="float16",
                           residual_dtype="float16", fuse_qkv=True,
                           attention="flash", fused_norm=True, cuda_graph=True),
}


def _mask_is_all_valid(keep_2d: torch.Tensor) -> bool:
    """True when no token is padded, so the key-padding mask can be dropped.

    Only worth asking when the mask we would otherwise materialize is large: the
    answer costs a device-to-host sync, and below a gigabyte of mask the existing
    path is cheaper than the question. Never asked during CUDA graph capture,
    where a sync is illegal -- and where the shapes are small anyway.

    Returning False is always safe; it just means we build the mask.
    """
    S = keep_2d.shape[-1]
    if S * S * 4 < 2**30:
        return False
    if keep_2d.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return bool(keep_2d.all())


class FusedTransformer(BaselineTransformer):
    """Drop-in replacement for `BaselineTransformer`.

    Subclasses the baseline so `load_state_dict(strict=True)` keeps working --
    the organizer's `copy_model_weights` requires identical parameter names.
    Fused weight buffers are derived lazily on the first forward, which lands
    inside the script's 20 warmup iterations and so never enters a timed region.
    """

    def __init__(self, config: TransformerConfig, plan: Optional[Plan] = None):
        super().__init__(config)
        self.plan = plan or LADDER["flash"]
        self._prepared_key: Optional[tuple] = None
        self._qkv_w: List[torch.Tensor] = []
        self._qkv_b: List[torch.Tensor] = []
        self._graphs: Dict[tuple, "GraphRunner"] = {}
        self._compiled_fn = None

    # --- fused weight preparation -----------------------------------------

    def _prepare(self, cdtype: torch.dtype) -> None:
        key = (cdtype, self.plan.fuse_qkv)
        if self._prepared_key == key:
            return
        self._qkv_w, self._qkv_b = [], []
        for layer in self.layers:
            a = layer.attention
            if self.plan.fuse_qkv:
                w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)
                b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0)
                self._qkv_w.append(w.to(cdtype).contiguous())
                self._qkv_b.append(b.to(cdtype).contiguous())
            else:
                self._qkv_w.append(None)
                self._qkv_b.append(None)
        self._cast_cache = {}
        self._prepared_key = key

    def _w(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Cast-and-cache a parameter. Casting every weight on every forward
        would dominate the small-shape regimes we care most about."""
        if tensor.dtype == dtype:
            return tensor
        key = (id(tensor), dtype)
        hit = self._cast_cache.get(key)
        if hit is None:
            hit = tensor.to(dtype)
            self._cast_cache[key] = hit
        return hit

    # --- forward -----------------------------------------------------------

    def _compiled(self):
        """Lazily build the `torch.compile`-wrapped forward for this plan.

        `torch.compile` is an optimization method like any other -- the
        organizer's script names it as a suggested direction -- so it competes
        in the search rather than only serving as a yardstick. It wins on shapes
        where our hand-written kernels do not, most notably long causal
        attention.

        What gets compiled is whatever the rest of the plan selects: the
        untouched baseline (`attention="baseline"`), or our structural rewrite,
        which removes the baseline's redundant copies before inductor ever sees
        it. Compilation happens on the first forward, which lands inside the
        script's 20 warmup iterations and so never enters a timed region.
        """
        if self._compiled_fn is None:
            if self.plan.attention == "baseline":
                def fn(x, m):
                    return BaselineTransformer.forward(self, x, m)
            else:
                fn = self._eager_forward
            self._compiled_fn = torch.compile(fn, mode=self.plan.torch_compile)
        return self._compiled_fn

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.plan.torch_compile:
            return self._compiled()(x, valid_token_mask)

        if self.plan.attention == "baseline":
            return super().forward(x, valid_token_mask)

        chunk = self._batch_chunk(x)
        if chunk < x.shape[0]:
            return self._chunked_forward(x, valid_token_mask, chunk)

        if self.plan.cuda_graph and not torch.cuda.is_current_stream_capturing():
            return self._graph_forward(x, valid_token_mask)

        try:
            return self._eager_forward(x, valid_token_mask)
        except torch.cuda.OutOfMemoryError:
            # The estimator said the whole batch would fit and it did not. One
            # sequence at a time is slow, but a slow answer beats no answer, and
            # this is the only path by which a shape we have never seen on a GPU
            # we have never seen can still return a correct result.
            if x.shape[0] <= 1:
                raise
            torch.cuda.empty_cache()
            return self._chunked_forward(x, valid_token_mask, 1)

    # How many d-wide activation tensors this forward keeps alive at peak.
    #
    # Counting them by hand gives about nine -- fused QKV, the attention
    # context, out_proj, the fp32 residual, the FFN hidden and output, the
    # normalized input. That number is wrong, and the way we found out is worth
    # recording: with LIVE_TENSORS=9 the estimator declined to chunk shape 14 at
    # S=16384 and the forward peaked at **64.3 GB** on an A100-80, against a
    # predicted 19.3 GB. One activation tensor there is 2.15 GB, so the real
    # peak is ~30 tensor-equivalents, not 9. The gap is the transient copies we
    # do not control: `.contiguous()` on q/k/v when an fp32 attention stage
    # falls back to SDPA, the allocator holding a freed block while the next
    # one is requested, and the benchmark harness keeping an output alive.
    #
    # So this is calibrated against that measurement rather than derived, and
    # rounded up. Over-estimating costs a smaller chunk on a shape that was
    # going to be slow anyway; under-estimating costs the whole run.
    LIVE_TENSORS = 32
    # Leave the allocator room for weights, the caller's output, and
    # fragmentation. Chunking is a fallback, so it may be pessimistic.
    MEM_HEADROOM = 0.55

    def _activation_bytes(self, B, S, dtype) -> int:
        """Peak working set of `_eager_forward` for a batch of `B`, in bytes."""
        width = max(self.config.d_model, self.config.ffn_dim)
        return B * S * width * dtype.itemsize * self.LIVE_TENSORS

    def _chunk_for_budget(self, B, S, dtype, free_bytes) -> int:
        """Largest batch slice that fits in `free_bytes`, or `B` for no chunking.

        Split out from the driver query so it can be tested on any machine: the
        answer depends only on the shape, the dtype and a byte budget, none of
        which need a GPU to reason about.
        """
        if B <= 1:
            return B
        # The output tensor is ours to allocate and lives for the whole call.
        out_bytes = B * S * self.config.d_model * dtype.itemsize
        budget = int((free_bytes - out_bytes) * self.MEM_HEADROOM)
        if budget <= 0:
            return 1
        per_sample = max(self._activation_bytes(1, S, dtype), 1)
        return max(1, min(B, budget // per_sample))

    def _batch_chunk(self, x) -> int:
        """How much of the batch to run at once, given what the device has free.

        Some shapes are too large to hold a whole batch of activations even
        though they are perfectly ordinary per sample -- official test shape 14
        (B=32, S=100000, d=1024) is 12.2 GB *per activation tensor* in fp32, so
        the forward needs hundreds of GB while one sample needs twelve.

        Nothing in this model mixes batch elements: LayerNorm normalizes over the
        feature dimension, the projections are per token, and attention is per
        (batch, head). Slicing the batch is therefore an execution-order change,
        not an approximation -- the same arithmetic in a different order, with
        only GEMM-tiling rounding to separate it from the unchunked result.

        The estimate is computed arithmetically first so that the common case
        costs no driver call; `mem_get_info` is only consulted once the shape is
        large enough for the answer to matter.
        """
        B, S, _ = x.shape
        if B <= 1 or not x.is_cuda:
            return B
        if self._activation_bytes(B, S, x.dtype) < 8 * 2**30:
            return B                    # small shapes: never chunk, never probe
        try:
            free, _total = torch.cuda.mem_get_info(x.device)
        except Exception:               # pragma: no cover - non-CUDA builds
            return B
        return self._chunk_for_budget(B, S, x.dtype, free)

    def _chunked_forward(self, x, valid_token_mask, chunk):
        """Run the batch in slices, writing into one preallocated output.

        The chunk size from `_batch_chunk` is an estimate, and estimates of peak
        allocator behaviour are not reliable enough to bet a whole run on -- ours
        was wrong by 3.3x the first time we measured it. So the estimate only has
        to get us close: if a slice runs out of memory anyway, we halve it and
        retry, down to a single sequence. A slice of one is the smallest unit
        this model has, so if that fails the shape genuinely does not fit and the
        error is the right answer.

        Slices already written stay valid -- each one is an independent function
        of its own rows -- so a retry redoes only the slice that failed.
        """
        B = x.shape[0]
        out = torch.empty_like(x)
        lo = 0
        while lo < B:
            hi = min(lo + chunk, B)
            m = None if valid_token_mask is None else valid_token_mask[lo:hi]
            try:
                out[lo:hi] = self._eager_forward(x[lo:hi], m)
            except torch.cuda.OutOfMemoryError:
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                torch.cuda.empty_cache()
                continue
            lo = hi
        return out

    def _eager_forward(self, x, valid_token_mask):
        cfg = self.config
        B, S, d = x.shape
        M = B * S
        H = cfg.num_heads
        Dh = d // H
        io_dtype = x.dtype
        rdtype = self.plan.residual_torch_dtype(io_dtype)
        d_attn = self.plan.stage_dtype("attn", io_dtype)
        d_out = self.plan.stage_dtype("out_proj", io_dtype)
        d_ffn1 = self.plan.stage_dtype("ffn1", io_dtype)
        d_ffn2 = self.plan.stage_dtype("ffn2", io_dtype)

        # tl.dot needs a narrow float type; an fp32 attention stage falls back
        # to SDPA (which is TF32 on Ampere, matching the reference's own path).
        if self.plan.attention == "exact":
            use_flash = False
        use_flash = (self.plan.attention == "flash"
                     and flash_supported(Dh)
                     and d_attn in (torch.float16, torch.bfloat16))
        use_fused_norm = self.plan.fused_norm and ln_supported(d)
        self._prepare(d_attn)

        keep_row = None      # [M] float, for the norm kernel
        keep_2d = None       # [B, S] uint8, for the attention kernel
        if valid_token_mask is not None:
            keep_2d = valid_token_mask.to(torch.uint8).contiguous()
            keep_row = valid_token_mask.reshape(M).to(rdtype)

        res = x.reshape(M, d).to(rdtype)

        # Entry LayerNorm (layer 0's norm1). Masking the residual here is safe:
        # the baseline's invalid rows are zeroed at every block boundary anyway,
        # so any difference is confined to rows that are discarded downstream.
        res, h = self._norm(res, None, keep_row, self.layers[0].norm1,
                            d_attn, rdtype, use_fused_norm, mask_out=False)

        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            attn_out = self._attention(layer, h, B, S, d, H, Dh, keep_2d,
                                       d_attn, d_out, io_dtype, use_flash, i)
            res, h = self._norm(res, attn_out, keep_row, layer.norm2,
                                d_ffn1, rdtype, use_fused_norm, mask_out=False)

            ffn = F.linear(h, self._w(layer.ffn_in.weight, d_ffn1),
                           self._w(layer.ffn_in.bias, d_ffn1))
            ffn = F.gelu(ffn, approximate="none")
            ffn = ffn.to(d_ffn2)
            ffn = F.linear(ffn, self._w(layer.ffn_out.weight, d_ffn2),
                           self._w(layer.ffn_out.bias, d_ffn2))

            last = (i == n - 1)
            next_norm = self.final_norm if last else self.layers[i + 1].norm1
            # On the final block the normalization is `final_norm`, and the mask
            # must be applied AFTER it: LayerNorm of an all-zero row returns
            # `bias`, not zero.
            res, h = self._norm(res, ffn, keep_row, next_norm, d_attn, rdtype,
                                use_fused_norm, mask_out=last,
                                out_dtype=io_dtype if last else None,
                                want_residual=not last)

        return h.reshape(B, S, d).to(io_dtype)

    def _norm(self, res, sub, keep_row, norm_mod, cdtype, rdtype, use_fused,
              mask_out=False, out_dtype=None, want_residual=True):
        out_dtype = out_dtype or cdtype
        if use_fused:
            return add_mask_layernorm(
                res, sub, keep_row, norm_mod.weight, norm_mod.bias,
                norm_mod.eps, mask_out=mask_out, out_dtype=out_dtype,
                residual_dtype=rdtype, want_residual=want_residual,
            )
        s = res if sub is None else res + sub.to(rdtype)
        if keep_row is not None:
            s = s * keep_row[:, None].to(s.dtype)
        h = F.layer_norm(s, (s.shape[-1],), norm_mod.weight.to(s.dtype),
                         norm_mod.bias.to(s.dtype), norm_mod.eps)
        if mask_out and keep_row is not None:
            h = h * keep_row[:, None].to(h.dtype)
        return s, h.to(out_dtype)

    def _attention(self, layer, h, B, S, d, H, Dh, keep_2d, cdtype, d_out,
                   io_dtype, use_flash, i):
        a = layer.attention

        if self.plan.fuse_qkv:
            qkv = F.linear(h, self._qkv_w[i], self._qkv_b[i])       # [M, 3d]
            qkv = qkv.view(B, S, 3, H, Dh)
            # Strided views, no copy: head_dim keeps unit stride, which is all
            # the flash kernel requires.
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)
        else:
            q = F.linear(h, self._w(a.q_proj.weight, cdtype),
                         self._w(a.q_proj.bias, cdtype)).view(B, S, H, Dh).permute(0, 2, 1, 3)
            k = F.linear(h, self._w(a.k_proj.weight, cdtype),
                         self._w(a.k_proj.bias, cdtype)).view(B, S, H, Dh).permute(0, 2, 1, 3)
            v = F.linear(h, self._w(a.v_proj.weight, cdtype),
                         self._w(a.v_proj.bias, cdtype)).view(B, S, H, Dh).permute(0, 2, 1, 3)

        causal = self.config.causal

        if S == 1:
            # Single-token sequences: attention is the identity on V, exactly.
            # The score matrix is [B,H,1,1], so softmax over the last dimension
            # is exp(x-x)/exp(x-x) = 1.0 bit-for-bit, and 1.0 @ V == V. Skipping
            # the whole attention path is therefore not an approximation, it is
            # algebra -- and it is the difference between losing to torch.compile
            # on this shape and beating it.
            #
            # The degenerate case (the only token masked invalid) cannot arise
            # from the organizer's generator, which guarantees min_valid >= 1;
            # the baseline would return NaN there anyway.
            ctx = v.permute(0, 2, 1, 3).reshape(B * S, d)
            ctx = ctx.to(d_out)
            return F.linear(ctx, self._w(a.out_proj.weight, d_out),
                            self._w(a.out_proj.bias, d_out))

        if use_flash:
            # Write straight into [B, S, H, Dh] so the merged-head view is free.
            ctx_bshd = torch.empty((B, S, H, Dh), dtype=cdtype, device=h.device)
            flash_attention(
                q, k, v, keep=keep_2d, causal=causal, sm_scale=Dh ** -0.5,
                smem_kb=self.plan.smem_kb, block=self.plan.flash_block,
                out=ctx_bshd.permute(0, 2, 1, 3),
                round_scores_to=io_dtype if self.plan.match_score_rounding else None,
            )
            ctx = ctx_bshd.reshape(B * S, d)
        elif self.plan.attention == "exact":
            # Transcription of BaselineSelfAttention.forward, op for op. Not fast
            # -- it exists so the ablation can separate "our loop restructuring"
            # from "our attention kernel" as error sources. If this rung does not
            # score ~0 envelope, the bug is in the surrounding rewrite.
            scale = Dh ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal:
                cm = torch.ones((S, S), device=h.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(cm, float("-inf"))
            if keep_2d is not None:
                scores = scores.masked_fill(~keep_2d.bool()[:, None, None, :],
                                            float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            ctx = torch.matmul(probs, v)
            ctx = ctx.permute(0, 2, 1, 3).reshape(B * S, d)
        else:
            # SDPA fallback, used when the attention stage stays in fp32 (Triton's
            # tl.dot needs a narrow float type). SDPA takes `is_causal` OR an
            # explicit `attn_mask`, never both, so causal-plus-padding forces us
            # to build the mask ourselves -- and that mask is [S, S].
            #
            # At S=100000 that is a 37 GiB allocation, which is how this path
            # OOM'd on an A100-80 while claiming O(S) attention memory: our own
            # fallback was reintroducing the quadratic term the flash kernel
            # exists to remove. Two guards, in order of preference:
            if keep_2d is not None and causal and _mask_is_all_valid(keep_2d):
                # Nothing is actually padded, so the padding mask is a no-op and
                # `is_causal` alone is exact. This is the common case -- and it
                # is every one of the official test shapes, none of which pad.
                keep_2d = None
            attn_mask = None
            if keep_2d is not None:
                attn_mask = keep_2d.bool()[:, None, None, :]
                if causal:
                    cm = torch.ones((S, S), device=h.device, dtype=torch.bool).tril()
                    attn_mask = attn_mask & cm[None, None, :, :]
                ctx = F.scaled_dot_product_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    attn_mask=attn_mask)
            else:
                ctx = F.scaled_dot_product_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    is_causal=causal)
            ctx = ctx.permute(0, 2, 1, 3).reshape(B * S, d)

        ctx = ctx.to(d_out)
        return F.linear(ctx, self._w(a.out_proj.weight, d_out),
                        self._w(a.out_proj.bias, d_out))

    # --- CUDA graph path ---------------------------------------------------

    def _graph_forward(self, x, valid_token_mask):
        key = (tuple(x.shape), x.dtype,
               None if valid_token_mask is None else tuple(valid_token_mask.shape))
        runner = self._graphs.get(key)
        if runner is None:
            runner = GraphRunner(self, x, valid_token_mask)
            self._graphs[key] = runner
        return runner(x, valid_token_mask)


class GraphRunner:
    """Captures the whole stack into one CUDA graph.

    A 6-layer forward issues ~105 kernel launches, and on Windows WDDM each
    costs ~13 us of CPU time -- more than the GPU work itself once the shapes get
    small. Replaying a graph collapses that to a single submission.

    The captured region is re-executed against fixed device buffers, so inputs
    are copied in and the output is cloned out.
    """

    def __init__(self, model: FusedTransformer, x, mask):
        self.model = model
        self.static_x = x.clone()
        self.static_mask = None if mask is None else mask.clone()

        # Warm up on a side stream: Triton autotuning and cuBLAS handle setup
        # must not happen during capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model._eager_forward(self.static_x, self.static_mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = model._eager_forward(self.static_x, self.static_mask)

    def __call__(self, x, mask):
        self.static_x.copy_(x)
        if self.static_mask is not None and mask is not None:
            self.static_mask.copy_(mask)
        self.graph.replay()
        # Clone: the buffer is overwritten by the next replay, and the caller
        # (the accuracy check) legitimately holds on to the result.
        return self.static_out.clone()


def build(config: TransformerConfig, plan: Plan) -> FusedTransformer:
    return FusedTransformer(config, plan)


def build_shared(config: TransformerConfig, plan: Plan,
                 reference: nn.Module) -> FusedTransformer:
    """Build a candidate that *shares* `reference`'s weight tensors.

    The search benchmarks many candidates against one another in a single
    interleaved run, so they are all resident at once. Giving each its own copy
    of the parameters is what killed an early sweep: at BERT-base size that is
    ~170 MB per candidate, and eight candidates plus the baseline and two
    compiled variants do not fit in 8 GB.

    `assign=True` rebinds the module's parameters to the reference's existing
    tensors instead of copying into freshly allocated ones, so N candidates cost
    one set of weights. Constructing under `device("meta")` avoids even the
    transient host-side allocation. This is safe here because every candidate is
    inference-only and none of them mutate a parameter.
    """
    with torch.device("meta"):
        model = FusedTransformer(config, plan)
    model.load_state_dict(reference.state_dict(), assign=True)
    return model.eval()
