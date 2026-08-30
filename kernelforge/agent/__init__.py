"""The optimization loop: profile, propose, gate, measure, freeze.

`__all__` marks `run_agent` as this package's deliberate public surface rather
than an unused import.
"""
from .loop import run_agent

__all__ = ["run_agent"]
