"""Correctness check for the batched GPU/CPU Lambert solver: cross-validate
against pykep's own (Izzo, reference) lambert_problem on many random
Earth-ish two-body transfer cases, on both the numpy and (if available)
CuPy backends.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

from orbitopt.core.gpu import GPU_AVAILABLE
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.lambert.gpu_batch import solve_lambert_batch

MU_SUN = pk.MU_SUN


def _random_heliocentric_transfer_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    r1_norm = rng.uniform(0.7, 1.6, n) * pk.AU
    r2_norm = rng.uniform(0.7, 1.6, n) * pk.AU
    theta1 = rng.uniform(0, 2 * np.pi, n)
    theta2 = theta1 + rng.uniform(0.2, 2.8, n)  # keep away from 0/pi degeneracies

    r1 = np.stack([r1_norm * np.cos(theta1), r1_norm * np.sin(theta1), np.zeros(n)], axis=1)
    r2 = np.stack([r2_norm * np.cos(theta2), r2_norm * np.sin(theta2), np.zeros(n)], axis=1)

    tof_days = rng.uniform(80, 400, n)
    tof = tof_days * pk.DAY2SEC
    return r1, r2, tof


@pytest.mark.parametrize("use_gpu", [False] + ([True] if GPU_AVAILABLE else []))
def test_batch_matches_pykep_reference(use_gpu):
    n = 200
    r1, r2, tof = _random_heliocentric_transfer_cases(n)

    result = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=use_gpu)
    assert result.converged.mean() > 0.95

    max_v_err = 0.0
    for i in range(n):
        if not result.converged[i]:
            continue
        ref = solve_lambert_single(r1[i], r2[i], tof[i], mu=MU_SUN, max_revs=0)
        v1_ref, v2_ref = ref[0]
        max_v_err = max(
            max_v_err,
            np.linalg.norm(result.v1[i] - v1_ref),
            np.linalg.norm(result.v2[i] - v2_ref),
        )

    # velocities are O(1e4 m/s); 1 m/s agreement is ample for optimizer/porkchop use.
    assert max_v_err < 1.0, f"max |v_batch - v_pykep| = {max_v_err} m/s"


def test_cpu_and_gpu_backends_agree():
    if not GPU_AVAILABLE:
        pytest.skip("no CUDA device available in this environment")
    n = 500
    r1, r2, tof = _random_heliocentric_transfer_cases(n, seed=1)

    cpu = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    gpu = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=True)

    both_converged = cpu.converged & gpu.converged
    assert both_converged.mean() > 0.95
    assert np.max(np.abs(cpu.v1[both_converged] - gpu.v1[both_converged])) < 1e-3
    assert np.max(np.abs(cpu.v2[both_converged] - gpu.v2[both_converged])) < 1e-3
