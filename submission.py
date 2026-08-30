"""Competition entry point.

Drop-in replacement for `UserOptimizedTransformer`. Two ways to use it:

  1. Patch the organizer's script in place (nothing else changes):

         python torch_transformer_benchmark.py --batch-size 8 --seq-len 128
         # after replacing the UserOptimizedTransformer body with:
         #     from submission import UserOptimizedTransformer

  2. Run our runner, which imports the organizer's script unmodified:

         python -m kernelforge.cli verify
         python -m kernelforge.cli sweep

The class picks its implementation from the frozen dispatch table for the GPU it
finds itself on. If the table has no entry it falls back to a dtype-level
default, and if anything at all goes wrong it falls back to the baseline. The
accuracy gate is a hard failure in the organizer's script, so "never wrong" wins
over "always fast".
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

from kernelforge.dispatch import DispatchTable
from kernelforge.hw import probe
from kernelforge.optimized import FusedTransformer, Plan

# Set KERNELFORGE_PLAN to force a named ladder plan; useful for ablations.
_FORCED = os.environ.get("KERNELFORGE_PLAN")
_VERBOSE = os.environ.get("KERNELFORGE_VERBOSE", "") not in ("", "0")


class UserOptimizedTransformer(BaselineTransformer):
    """The submitted implementation.

    Subclasses `BaselineTransformer` so the organizer's `copy_model_weights`
    keeps working with `strict=True`: parameter names and module structure are
    unchanged, only `forward` is replaced.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._impl: Optional[FusedTransformer] = None
        self._plan: Optional[Plan] = None
        self._source = "uninitialized"
        self._failed = False

    def _select(self, dtype: torch.dtype) -> None:
        if self._impl is not None or self._failed:
            return
        try:
            spec = probe(measure=False)
            smem = spec.shared_mem_per_block_kb
            if _FORCED:
                from kernelforge.optimized import LADDER
                plan, source = LADDER[_FORCED], f"env:{_FORCED}"
            else:
                table = DispatchTable.load(spec.arch)
                plan, source = table.lookup(spec.arch, dtype, self.config, smem)
            impl = FusedTransformer(self.config, plan)
            impl.load_state_dict(self.state_dict(), strict=True)
            impl = impl.to(next(self.parameters()).device, dtype).eval()
            self._impl, self._plan, self._source = impl, plan, source
            if _VERBOSE:
                print(f"[kernelforge] {spec.arch} {dtype} -> {plan.name} "
                      f"({source}): {plan.describe()}")
        except Exception as e:  # pragma: no cover - safety net
            self._failed = True
            if _VERBOSE:
                print(f"[kernelforge] falling back to baseline: "
                      f"{type(e).__name__}: {e}")

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._select(x.dtype)
        if self._impl is None:
            return super().forward(x, valid_token_mask)
        try:
            return self._impl(x, valid_token_mask)
        except Exception as e:  # pragma: no cover - safety net
            # `_select` guards *choosing* a plan; this guards *running* it. They
            # fail for different reasons: a judge runs this on a GPU we have
            # never seen, where Triton may refuse to compile a tiling that was
            # legal on ours, or an allocation may not shrink far enough. The
            # right answer there is the reference answer, not a traceback.
            #
            # Announced unconditionally, not under KERNELFORGE_VERBOSE. A silent
            # fallback would report as a ~1.0x speedup and read like a weak
            # result rather than a broken one, and we would rather a judge see
            # exactly what happened.
            self._failed, self._impl = True, None
            print(f"[kernelforge] the fused path raised {type(e).__name__}: {e}"
                  f"\n[kernelforge] falling back to the reference implementation "
                  f"for the rest of this run; results stay correct, speed does "
                  f"not.")
            return super().forward(x, valid_token_mask)
