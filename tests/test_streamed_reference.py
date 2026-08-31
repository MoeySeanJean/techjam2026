"""The streamed reference must *be* the organizer's model, not merely resemble it.

Official shape 14 was long treated as ungateable: the reference materializes
`[B,H,S,S]`, which is 18.6 TB there. But the memory is the obstacle, not the
arithmetic — chunking query rows and masking against the key index computes the
same thing in O(S), which makes the whole output checkable at full length.

That argument is only worth anything if the streamed version is genuinely the
same function. This file checks it against `BaselineTransformer.forward` itself,
including at a sequence length that is not a multiple of the chunk, where an
off-by-one in the causal mask would show up.
"""
from __future__ import annotations

import importlib.util
import os

import pytest
import torch

import torch_transformer_benchmark as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="needs a CUDA GPU",
)


def _shape14_module():
    spec = importlib.util.spec_from_file_location(
        "shape14", os.path.join(ROOT, "scripts", "shape14.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seq_len", [512, 1024, 333, 4097, 9000])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_streamed_reference_matches_the_organizers_model(seq_len, num_layers):
    """Same weights, same input, same answer.

    `333`, `4097` and `9000` are deliberate: the chunk is 4096, so these exercise
    a partial final chunk, a ragged one-row final chunk, and three chunks. A
    causal mask off by one at a chunk edge would pass at 512 and fail here.

    Usually the result is bit-identical. It need not be: splitting one large
    matmul into differently-shaped ones lets cuBLAS pick different kernels, and
    at `S=4097` the ragged final chunk shifts the last bits — 3.6e-7 absolute,
    fp32 rounding rather than a difference in what is computed. The tolerance
    below is set to catch a real divergence while allowing that.
    """
    mod = _shape14_module()
    d, heads = 128, 4
    cfg = B.TransformerConfig(batch_size=1, seq_len=seq_len, d_model=d,
                              num_heads=heads, ffn_dim=d,
                              num_layers=num_layers, causal=True)
    dev = torch.device("cuda")

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        model = B.BaselineTransformer(cfg).to(dev, torch.float32).eval()
        x, _ = B.generate_random_case(cfg, dev, torch.float32, 5, 0.0, 1.0)
        with torch.inference_mode():
            want = model(x, None)
            got = mod._ref_model(model, x.reshape(-1, d), heads).reshape(x.shape)
        diff = (want - got).abs().max().item()
        assert diff < 1e-5, (
            f"streamed reference diverges at S={seq_len}, L={num_layers}: "
            f"max |diff| {diff:.3e} -- too large to be fp32 rounding")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        torch.set_float32_matmul_precision("high")


def test_streaming_is_what_makes_it_computable():
    """The point of the exercise: the score block fits, the full matrix does not.

    Guards the constant that makes the gate possible. At shape 14's dimensions
    the streamed score block peaks around 26 GB -- large, but resident on an
    80 GB card -- against 18.6 TB for the materialized `[B,H,S,S]`. If `CHUNK`
    ever grew, that headroom would go.
    """
    mod = _shape14_module()
    S, heads, batch = 100_000, 16, 32
    streamed = heads * mod.CHUNK * S * 4        # peak score block, fp32, batch 1
    materialized = batch * heads * S * S * 4    # what the reference would need

    assert streamed < 40 * 2 ** 30, (
        f"streamed score block is {streamed / 2**30:.1f} GB -- too large to be "
        f"resident alongside the model")
    assert materialized / streamed > 500, (
        f"streaming only saves {materialized / streamed:.0f}x; the whole point "
        f"is that it turns 18.6 TB into something a GPU can hold")
