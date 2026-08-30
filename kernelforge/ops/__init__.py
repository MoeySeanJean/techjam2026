"""The two hand-written Triton kernels, re-exported as this package's surface.

`__all__` is declared so these read as deliberate re-exports rather than unused
imports -- a linter cannot otherwise tell the difference, and an earlier cleanup
pass removed two of them on exactly that misreading.
"""
from .layernorm import add_mask_layernorm, HAS_TRITON
from .flash import flash_attention, flash_supported

__all__ = ["add_mask_layernorm", "flash_attention", "flash_supported",
           "HAS_TRITON"]
