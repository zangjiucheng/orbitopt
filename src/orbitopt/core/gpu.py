"""GPU/CPU array-module abstraction used throughout orbitopt.

Every numerically heavy routine in this package (Lambert solves, porkchop
grids, batch fitness evaluation) is written once against an "array module"
``xp`` that is either ``numpy`` or ``cupy``. The same vectorized code then
runs on CPU or GPU depending on what's available, with no code duplication.

Set the environment variable ORBITOPT_FORCE_CPU=1 to disable CuPy even if a
GPU is present (useful for debugging / A-B benchmarking).
"""
from __future__ import annotations

import os

import numpy as np

_FORCE_CPU = os.environ.get("ORBITOPT_FORCE_CPU", "0") == "1"

cp = None
GPU_AVAILABLE = False

if not _FORCE_CPU:
    try:
        import cupy as _cp

        _cp.cuda.runtime.getDeviceCount()
        cp = _cp
        GPU_AVAILABLE = True
    except Exception:
        cp = None
        GPU_AVAILABLE = False


def get_array_module(use_gpu: bool | None = None):
    """Return the ``numpy``- or ``cupy``-compatible array module to use.

    use_gpu=None -> GPU if available, else CPU (default).
    use_gpu=True -> require GPU, raise if CuPy/CUDA is unavailable.
    use_gpu=False -> force CPU (numpy), regardless of GPU availability.
    """
    if use_gpu is False:
        return np
    if use_gpu is True:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU requested (use_gpu=True) but CuPy/CUDA is unavailable "
                "in this environment."
            )
        return cp
    return cp if GPU_AVAILABLE else np


def to_numpy(array):
    """Bring a numpy-or-cupy array back to host memory as a numpy array."""
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def device_info() -> dict:
    """Small diagnostic dict describing the active GPU, or lack thereof."""
    if not GPU_AVAILABLE:
        return {"gpu_available": False}
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "gpu_available": True,
        "name": name,
        "device_id": int(dev.id),
        "free_mem_mb": free / 1024**2,
        "total_mem_mb": total / 1024**2,
    }
