# Why the fused rewrite is exactly equivalent

The baseline applies the padding mask in three places. Getting this wrong is the
single easiest way to produce a fast kernel that is silently incorrect, so the
argument is written out and then checked empirically.

## What the baseline does

```python
# BaselineSelfAttention.forward
output = self.out_proj(context)
if valid_token_mask is not None:
    output = output.masked_fill(~valid_token_mask[..., None], 0)   # (1)

# BaselineTransformerBlock.forward
x = x + self.attention(self.norm1(x), valid_token_mask, causal)
x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x))))
if valid_token_mask is not None:
    x = x.masked_fill(~valid_token_mask[..., None], 0)             # (2)

# BaselineTransformer.forward
x = self.final_norm(x)
if valid_token_mask is not None:
    x = x.masked_fill(~valid_token_mask[..., None], 0)             # (3)
```

Plus, inside attention, invalid **keys** are set to `-inf` before the softmax.

## What we do

One fused kernel per block boundary:

```
s = (x + sublayer_out) * keep
h = LayerNorm(s)                 # optionally  h = h * keep
```

## The argument

**Key masking is not optional and we keep it.** Masking invalid keys with `-inf`
changes the output of *valid* query rows, so it is real arithmetic, not
bookkeeping. Our flash kernel applies it in-register while loading each K tile.

**Output-row masking is row-local and can be deferred.** Sites (1) and (2) both
zero whole rows of invalid tokens. Row `i` of every operation in the block —
LayerNorm, the projections, the FFN — depends only on row `i`. So zeroing at (1)
and again at (2) is indistinguishable from zeroing once at (2), *provided
nothing between them reads a row other than its own*. Attention is the only
operation that mixes rows, and it sits before (1).

**Invalid rows are zero on entry to every LayerNorm, in both versions.** In the
baseline, (2) leaves invalid rows at zero, so the next block's `norm1` sees
zeros. In ours, the `* keep` inside the fused kernel does the same thing one
step earlier. Both therefore feed `LayerNorm` an all-zero row and both get
`bias` out of it.

**The final norm is the exception, and it is why `mask_out` exists.**
`LayerNorm` of an all-zero row returns `bias`, not zero. Site (3) masks *after*
normalizing. A fused kernel that masked before normalizing would emit `bias` on
padded rows where the baseline emits `0`. The last block therefore runs with
`mask_out=True`, applying `keep` on both sides of the norm.

**Entry masking is safe.** The baseline does not mask `x` on entry, relying on
the generator having zeroed padded positions. We mask anyway. If a caller passed
unmasked `x`, the two would differ on invalid rows only — and those rows are
zeroed at the next block boundary before anything reads across rows.

## Empirical check

The argument is checked, not trusted. The single-change ablation in
[PRECISION.md](PRECISION.md) measures the loop rewrite with exact attention at
**0.000 envelope, 0.000 max absolute error** on float32, float16 and bfloat16 —
bit-identical output, including under `--causal`, `--padding-ratio 0.4`, both
together, and `--input-scale 64`.

Reproduce:

```bash
python -m kernelforge.cli verify --shapes-file official_shapes.txt
```

## Degenerate case

A query row whose keys are all masked would make the softmax denominator zero.
The provided generator cannot produce it (`min_valid >= 1` guarantees a
non-empty valid prefix, and a causal row always includes its own diagonal), but
the kernel guards it anyway: `l_i == 0` yields a zero output row rather than
NaN.
