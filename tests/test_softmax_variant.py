"""The `exp2` substitution in the flash kernel is numerically free.

The inner loop folds `log2(e)` into the softmax scale so it can use `tl.exp2`,
which maps to a single hardware MUFU instruction instead of a libdevice call.
An obvious worry is that this trades accuracy for speed, and that the error grows
with row length as more terms are accumulated.

It does not. This test builds the `tl.exp` variant of the same kernel and checks
that both land the same distance from an exact fp32 reference, at row lengths
spanning a factor of 32. It exists because that worry was written down as a
suspected limitation, and measurement is how a suspicion gets settled.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest
import torch

from kernelforge.numerics import check
from kernelforge.ops import flash as F

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0
         and F.HAS_TRITON),
    reason="needs a CUDA GPU with Triton",
)


def _exp_variant():
    """The same kernel with `tl.exp` and no log2(e) folding."""
    src = open(F.__file__, encoding="utf-8").read()
    src = src.replace("qk_scale = sm_scale * _LOG2E", "qk_scale = sm_scale")
    src = src.replace("qk = qk * _LOG2E", "pass  # no log2(e) folding")
    src = src.replace("p = tl.exp2(qk - m_safe[:, None])",
                      "p = tl.exp(qk - m_safe[:, None])")
    src = src.replace("alpha = tl.exp2(m_i - m_safe)", "alpha = tl.exp(m_i - m_safe)")
    path = os.path.join(tempfile.mkdtemp(), "_flash_exp.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("_flash_exp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference(q, k, v, causal):
    """Exact attention in fp32, the same op sequence the baseline uses."""
    scale = q.shape[-1] ** -0.5
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        S = q.shape[2]
        s = s.masked_fill(
            torch.ones((S, S), device=q.device, dtype=torch.bool).triu(1),
            float("-inf"))
    return torch.matmul(torch.softmax(s.float(), -1).to(s.dtype), v)


@pytest.mark.parametrize("seq_len", [128, 512, 2048])
def test_exp2_matches_exp_at_every_row_length(seq_len):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, seq_len, 64, device=dev, dtype=torch.float16)
               for _ in range(3))
    ref = _reference(q.float(), k.float(), v.float(), causal=True)

    shipped = F.flash_attention(q, k, v, causal=True, smem_kb=99.0).float()
    variant = _exp_variant().flash_attention(q, k, v, causal=True,
                                             smem_kb=99.0).float()

    e_shipped = check(ref, shipped).envelope_utilization
    e_variant = check(ref, variant).envelope_utilization

    # Both must pass, and neither may be meaningfully better than the other.
    assert e_shipped < 1.0 and e_variant < 1.0
    assert abs(e_shipped - e_variant) < 0.05, (
        f"S={seq_len}: exp2 {e_shipped:.4f} vs exp {e_variant:.4f} -- the "
        f"substitution is no longer free")


def test_envelope_does_not_grow_with_row_length():
    """The other half of the suspicion: error accumulating as rows get longer."""
    dev = torch.device("cuda")
    envs = []
    for seq_len in (128, 512, 2048):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, seq_len, 64, device=dev, dtype=torch.float16)
                   for _ in range(3))
        ref = _reference(q.float(), k.float(), v.float(), causal=True)
        out = F.flash_attention(q, k, v, causal=True, smem_kb=99.0).float()
        envs.append(check(ref, out).envelope_utilization)
    assert max(envs) < 2 * min(envs), (
        f"envelope grows with row length: {envs} -- the online softmax is "
        f"accumulating error, which would change how we treat long sequences")
