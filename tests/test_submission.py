"""The entry point a judge actually runs.

`submission.py` is the only file the organizer's script touches, so its contract
is narrow and worth pinning: it must subclass the baseline so `strict=True`
weight copying keeps working, it must produce the reference answer when anything
goes wrong, and it must never raise where the baseline would not have.

The failure paths matter more than the happy path here. A judge runs this on a
GPU we have never seen, and the difference between "slower than we hoped" and
"crashed during evaluation" is the whole submission.
"""
from __future__ import annotations

import torch

import torch_transformer_benchmark as B
from submission import UserOptimizedTransformer


def _cfg(**kw):
    base = dict(batch_size=2, seq_len=16, d_model=32, num_heads=4,
                ffn_dim=32, num_layers=2, causal=True)
    base.update(kw)
    return B.TransformerConfig(**base)


def test_is_a_baseline_subclass_so_strict_weight_copy_works():
    """The organizer copies weights with strict=True; the module tree must match."""
    cfg = _cfg()
    assert issubclass(UserOptimizedTransformer, B.BaselineTransformer)
    ours = UserOptimizedTransformer(cfg)
    ref = B.BaselineTransformer(cfg)
    assert set(ours.state_dict()) == set(ref.state_dict())
    # The organizer's own copy helper, on the real objects.
    ours.load_state_dict(ref.state_dict(), strict=True)


def test_forward_falls_back_to_the_reference_when_the_fused_path_raises(capsys):
    """A fused path that explodes must yield the reference answer, not a traceback.

    We install an implementation that always raises, which is the one thing a
    real failure on an unfamiliar GPU is guaranteed to have in common with every
    other failure mode.
    """
    cfg = _cfg()
    ours = UserOptimizedTransformer(cfg).eval()
    ref = B.BaselineTransformer(cfg).eval()
    ref.load_state_dict(ours.state_dict(), strict=True)

    class Exploding(torch.nn.Module):
        def forward(self, *a, **kw):
            raise RuntimeError("simulated Triton compile failure")

    ours._impl = Exploding()
    ours._failed = False

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)
    with torch.inference_mode():
        got = ours(x, None)
        want = ref(x, None)

    assert torch.equal(got, want), "fallback must be the reference answer exactly"
    out = capsys.readouterr().out
    assert "kernelforge" in out and "falling back" in out, \
        "the fallback must announce itself, not happen silently"


def test_the_fallback_is_sticky():
    """Having failed once, it must not retry the broken path every forward."""
    cfg = _cfg()
    ours = UserOptimizedTransformer(cfg).eval()

    calls = {"n": 0}

    class Exploding(torch.nn.Module):
        def forward(self, *a, **kw):
            calls["n"] += 1
            raise RuntimeError("boom")

    ours._impl = Exploding()
    ours._failed = False
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)
    with torch.inference_mode():
        for _ in range(4):
            ours(x, None)
    assert calls["n"] == 1, f"retried the broken path {calls['n']} times"


def test_output_shape_and_dtype_match_the_reference_on_cpu():
    """Without CUDA there is no dispatch table, so this exercises the fallback.

    The input comes from the organizer's own generator rather than a synthetic
    one. An earlier version of this test built the mask with `torch.ones(...)`,
    which is float32; the reference does `~valid_token_mask`, which only accepts
    bool or integer tensors. The test passed on a machine with a GPU and failed
    in a fresh CPU-only clone -- exactly where a judge would have met it first.
    Using the real generator means the mask dtype is whatever the benchmark
    actually produces, and stays that way if the benchmark changes.
    """
    cfg = _cfg()
    ours = UserOptimizedTransformer(cfg).eval()
    x, mask = B.generate_random_case(cfg, torch.device("cpu"), torch.float32,
                                     seed=3, padding_ratio=0.4, input_scale=1.0)
    with torch.inference_mode():
        out = ours(x, mask)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
