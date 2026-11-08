"""Correctness check for the batched GPU/CPU Lambert solver: cross-validate
against pykep's own (Izzo, reference) lambert_problem on many random
Earth-ish two-body transfer cases, on both the numpy and (if available)
CuPy backends.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

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


@pytest.mark.parametrize("use_gpu", [False, pytest.param(True, marks=pytest.mark.gpu)])
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


@pytest.mark.gpu
def test_cpu_and_gpu_backends_agree():
    n = 500
    r1, r2, tof = _random_heliocentric_transfer_cases(n, seed=1)

    cpu = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    gpu = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=True)

    both_converged = cpu.converged & gpu.converged
    assert both_converged.mean() > 0.95
    assert np.max(np.abs(cpu.v1[both_converged] - gpu.v1[both_converged])) < 1e-3
    assert np.max(np.abs(cpu.v2[both_converged] - gpu.v2[both_converged])) < 1e-3


# --------------------------------------------------------------------------
# Validity mask / degenerate-input guards / CPU fallback
# --------------------------------------------------------------------------

def _wide_random_cases(n, seed=3):
    """Much wider than the case generator above: any transfer angle
    (including near 0/180/360 deg), mildly out-of-plane, 1 day..27 yr TOFs."""
    rng = np.random.default_rng(seed)
    r1n = rng.uniform(0.3, 5.0, n) * pk.AU
    r2n = rng.uniform(0.3, 5.0, n) * pk.AU
    th1 = rng.uniform(0, 2 * np.pi, n)
    th2 = th1 + rng.uniform(0, 2 * np.pi, n)
    inc = rng.uniform(-0.3, 0.3, n)
    r1 = np.stack([r1n * np.cos(th1), r1n * np.sin(th1), np.zeros(n)], axis=1)
    r2 = np.stack([r2n * np.cos(th2), r2n * np.sin(th2), r2n * np.sin(inc)], axis=1)
    tof = 10 ** rng.uniform(0, 4, n) * pk.DAY2SEC
    return r1, r2, tof


def _assert_matches_pykep(result, r1, r2, tof, tol_m_s=1.0):
    for i in np.flatnonzero(result.valid):
        v1_ref, v2_ref = solve_lambert_single(r1[i], r2[i], tof[i], mu=MU_SUN, max_revs=0)[0]
        err = max(np.linalg.norm(result.v1[i] - v1_ref), np.linalg.norm(result.v2[i] - v2_ref))
        assert err < tol_m_s, f"case {i}: |v - v_pykep| = {err} m/s"


def test_result_exposes_validity_mask():
    r1, r2, tof = _random_heliocentric_transfer_cases(20)
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    assert res.valid.dtype == bool and res.valid.shape == (20,)
    assert res.fallback.dtype == bool and res.fallback.shape == (20,)
    np.testing.assert_array_equal(res.converged, res.valid)


@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_wide_random_cases_never_falsely_converge(cpu_fallback):
    n = 600
    r1, r2, tof = _wide_random_cases(n)
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=cpu_fallback)
    assert np.all(np.isfinite(res.v1)) and np.all(np.isfinite(res.v2))
    assert not np.any(np.isnan(res.residual))
    assert res.valid.mean() > 0.99
    _assert_matches_pykep(res, r1, r2, tof)


def test_malformed_inputs_are_invalid_and_nan_free():
    au, day = pk.AU, pk.DAY2SEC
    good_r1, good_r2 = [au, 0.0, 0.0], [0.0, 1.5 * au, 0.0]
    r1 = np.array([good_r1, [np.nan, 0, 0], good_r1, [0.0, 0.0, 0.0], good_r1, good_r1, good_r1, good_r1])
    r2 = np.array([good_r2, good_r2, [np.inf, 0, 0], good_r2, good_r2, good_r2, good_r2, good_r2])
    tof = np.array([200 * day, 200 * day, 200 * day, 200 * day, 0.0, -5 * day, np.nan, np.inf])
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    np.testing.assert_array_equal(res.valid, [True] + [False] * 7)
    assert not res.fallback[1:].any()
    for arr in (res.v1, res.v2):
        assert np.all(np.isfinite(arr))
        assert np.all(arr[1:] == 0.0)
    assert np.all(np.isinf(res.residual[1:]))
    _assert_matches_pykep(res, r1, r2, tof)


def test_invalid_mu_raises():
    r1, r2, tof = _random_heliocentric_transfer_cases(2)
    for mu in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            solve_lambert_batch(r1, r2, tof, mu, use_gpu=False)


@pytest.mark.parametrize("theta", [1e-9, 1e-4, np.pi - 1e-4, np.pi - 1e-8, np.pi, np.pi + 1e-4, 2 * np.pi - 1e-4])
def test_near_degenerate_transfer_angles(theta):
    au = pk.AU
    tofs = np.array([1.0, 30.0, 200.0, 2000.0]) * pk.DAY2SEC
    r1 = np.tile([au, 0.0, 0.0], (4, 1))
    r2 = np.tile([1.5 * au * np.cos(theta), 1.5 * au * np.sin(theta), 0.0], (4, 1))

    batch_only = solve_lambert_batch(r1, r2, tofs, MU_SUN, use_gpu=False, cpu_fallback=False)
    assert np.all(np.isfinite(batch_only.v1)) and np.all(np.isfinite(batch_only.v2))
    _assert_matches_pykep(batch_only, r1, r2, tofs)
    if abs(np.sin(theta)) < 1e-6:
        # Degenerate geometry is never claimed solved by the batched path...
        assert not batch_only.valid.any()

    res = solve_lambert_batch(r1, r2, tofs, MU_SUN, use_gpu=False)
    assert np.all(np.isfinite(res.v1)) and np.all(np.isfinite(res.v2))
    assert res.valid.all()
    # ...but the pykep CPU fallback recovers it.
    np.testing.assert_array_equal(res.fallback, ~batch_only.valid)
    _assert_matches_pykep(res, r1, r2, tofs)
