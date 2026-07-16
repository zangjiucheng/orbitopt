"""orbitopt package root.

Deliberately does not eagerly import anything heavy at package-import time:
``import orbitopt`` (and, transitively, ``import orbitopt.scene_format`` /
``import orbitopt.missions``) must stay usable in a plain
``pip install orbitopt`` environment with no numpy/pyvista/pykep/pygmo/tudatpy
installed -- see orbitopt.scene_format's module docstring for why that split
is the point. The GPU-utility re-exports below are resolved lazily via
module ``__getattr__`` (PEP 562): ``from orbitopt import GPU_AVAILABLE``
still works, but only pays numpy's import cost when actually touched, not on
every ``import orbitopt.<anything>``.
"""
from __future__ import annotations

__all__ = ["GPU_AVAILABLE", "device_info", "get_array_module"]


def __getattr__(name):
    if name in __all__:
        from orbitopt.core import gpu
        return getattr(gpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
