"""Validate one generated kernel in a throwaway process.

A generated kernel that indexes out of bounds does not raise a catchable Python
exception -- it corrupts the CUDA context. The launch returns fine, the error
surfaces asynchronously at some later unrelated CUDA call, and from that point
*every* CUDA operation in the process fails. `try/except` cannot recover it;
only the process boundary can.

So each candidate is validated here, in its own process. A crash costs one
generated kernel instead of the whole run, and the exit code tells the parent
what happened. This is the same lesson the shape sweep taught (one case per
subprocess), arrived at from the opposite direction.

Not called directly; `codegen.run_codegen` spawns it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="generated module to validate")
    ap.add_argument("--target", required=True, help="KernelSpec key")
    ap.add_argument("--out", required=True, help="where to write the verdict")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    import torch

    from kernelforge.agent.codegen import TARGETS, validate_inprocess

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    spec = TARGETS[args.target]
    attempt = validate_inprocess(args.path, spec, torch.device("cuda"),
                                 trials=args.trials)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(attempt), f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
