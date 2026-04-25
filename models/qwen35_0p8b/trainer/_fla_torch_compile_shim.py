"""Shim that disables torch.compile before importing fla.

torch 2.11.0+cu128 on Python 3.12 has a broken inductor: importing
torch._inductor.codegen.common raises
    TypeError: Too few arguments for <class 'torch._inductor.codegen.common.CSE'>
because of a Python 3.12 typing.Generic compatibility regression. fla
0.5.0's `fla.ops.attn.parallel` triggers this at import time via a
single `@torch.compile` decorator on a class.

The decorator was a perf hint, not required for correctness. We replace
torch.compile with a no-op pass-through BEFORE any fla import.

Usage: import this module before `import fla`.
"""
from __future__ import annotations

import torch as _torch


def _noop_compile(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def deco(fn):
        return fn
    return deco


_torch.compile = _noop_compile  # type: ignore[assignment]
