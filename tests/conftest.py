"""Shared pytest wiring for the CUDA (``gpu``-marked) tests.

Tests that need a real CuPy + CUDA device carry ``@pytest.mark.gpu``.

* Default (CPU CI, laptops without CUDA): ``gpu`` tests are *skipped* when
  ``orbitopt.core.gpu.GPU_AVAILABLE`` is false.
* ``ORBITOPT_REQUIRE_GPU=1`` (the GPU CI job, ``.github/workflows/gpu.yml``):
  a missing CUDA device is a hard error instead of a silent skip, so a
  misconfigured GPU runner can never report a green run that executed zero
  CUDA tests.
"""
from __future__ import annotations

import os

import pytest

REQUIRE_GPU = os.environ.get("ORBITOPT_REQUIRE_GPU", "0") == "1"


def _gpu_available() -> bool:
    from orbitopt.core.gpu import GPU_AVAILABLE

    return GPU_AVAILABLE


def pytest_sessionstart(session):
    if REQUIRE_GPU and not _gpu_available():
        raise pytest.UsageError(
            "ORBITOPT_REQUIRE_GPU=1 but no CuPy/CUDA device is usable "
            "(is cupy installed, the driver loaded, ORBITOPT_FORCE_CPU unset?)"
        )


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs CuPy + a CUDA device (run in the GPU CI job)")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)
