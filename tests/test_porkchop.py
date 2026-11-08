"""Porkchop ephemeris-dedup regression: the optimized grid must match the
original per-cell-ephemeris implementation, with measurably fewer
``eph`` calls."""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

from orbitopt.lambert.gpu_batch import solve_lambert_batch
from orbitopt.viz.porkchop import _eph_cubic, compute_porkchop

T0_RANGE = (9740.0, 9862.0)  # ~Sep 2026 .. Jan 2027 MJD2000 (Earth->Mars window)
TOF_RANGE = (150.0, 350.0)
N_T0, N_TOF = 31, 23


def planet(name):
    # pykep JPL low-precision ephemeris (same as orbitopt.bodies.planet, but
    # without that module's tudatpy import).
    if hasattr(pk.planet, "jpl_lp"):  # pykep 2.x (the project's target)
        return pk.planet.jpl_lp(name)
    return pk.planet(pk.udpla.jpl_lp(name))  # pykep 3.x


class CountingBody:
    def __init__(self, body):
        self._body = body
        self.calls = 0

    def eph(self, epoch):
        self.calls += 1
        return self._body.eph(epoch)


def _reference_porkchop(dep, arr, t0_range, tof_range, n_t0, n_tof, mu=pk.MU_SUN):
    """The original implementation: one dep + one arr eph call per cell."""
    t0_grid = np.linspace(t0_range[0], t0_range[1], n_t0)
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)
    tt0, ttof = np.meshgrid(t0_grid, tof_grid)
    n = tt0.size
    t0_flat, tof_flat = tt0.ravel(), ttof.ravel()
    r1, v1p, r2, v2p = (np.empty((n, 3)) for _ in range(4))
    for i in range(n):
        r1[i], v1p[i] = dep.eph(pk.epoch(float(t0_flat[i])))
        r2[i], v2p[i] = arr.eph(pk.epoch(float(t0_flat[i] + tof_flat[i])))
    res = solve_lambert_batch(r1, r2, tof_flat * pk.DAY2SEC, mu, use_gpu=False)
    c3 = np.sum(((res.v1 - v1p) / 1000.0) ** 2, axis=1)
    vinf = np.linalg.norm((res.v2 - v2p) / 1000.0, axis=1)
    return (c3.reshape(tt0.shape), vinf.reshape(tt0.shape),
            res.converged.reshape(tt0.shape))


@pytest.fixture(scope="module")
def reference():
    dep, arr = CountingBody(planet("earth")), CountingBody(planet("mars"))
    out = _reference_porkchop(dep, arr, T0_RANGE, TOF_RANGE, N_T0, N_TOF)
    return out, dep.calls, arr.calls


def test_exact_mode_matches_reference_with_fewer_departure_calls(reference):
    (c3_ref, vinf_ref, conv_ref), dep_ref, arr_ref = reference
    dep, arr = CountingBody(planet("earth")), CountingBody(planet("mars"))
    grid = compute_porkchop(dep, arr, T0_RANGE, TOF_RANGE, N_T0, N_TOF, use_gpu=False)

    assert dep_ref == arr_ref == N_T0 * N_TOF
    assert dep.calls == N_T0  # N_dep x N_TOF -> N_dep
    assert arr.calls <= N_T0 * N_TOF

    np.testing.assert_array_equal(grid.converged, conv_ref)
    m = conv_ref
    np.testing.assert_allclose(grid.c3_departure[m], c3_ref[m], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grid.v_inf_arrival[m], vinf_ref[m], rtol=1e-12, atol=1e-12)


def test_cubic_mode_within_error_budget(reference):
    (c3_ref, vinf_ref, conv_ref), _, arr_ref = reference
    dep, arr = CountingBody(planet("earth")), CountingBody(planet("mars"))
    grid = compute_porkchop(dep, arr, T0_RANGE, TOF_RANGE, N_T0, N_TOF,
                            use_gpu=False, arrival_interp_step=1.0)

    span = (T0_RANGE[1] + TOF_RANGE[1]) - (T0_RANGE[0] + TOF_RANGE[0])
    assert dep.calls == N_T0
    assert arr.calls == int(np.ceil(span)) + 1
    assert arr.calls < arr_ref

    m = conv_ref & grid.converged
    assert m.mean() > 0.95
    # Budget: 1e-4 km/s (0.1 m/s) on v_inf, 1e-3 km^2/s^2 on C3 -- far below
    # porkchop contour resolution.
    assert np.max(np.abs(grid.v_inf_arrival[m] - vinf_ref[m])) < 1e-4
    assert np.max(np.abs(grid.c3_departure[m] - c3_ref[m])) < 1e-3


def test_cubic_interpolant_error_budget():
    mars = planet("mars")
    t_query = np.linspace(9800.0, 10400.0, 997)  # off-node epochs
    r, v, n_samples = _eph_cubic(mars, t_query, 1.0)
    r_ref = np.array([mars.eph(pk.epoch(float(t)))[0] for t in t_query])
    v_ref = np.array([mars.eph(pk.epoch(float(t)))[1] for t in t_query])
    assert n_samples == 601
    pos_err = np.max(np.linalg.norm(r - r_ref, axis=1))
    vel_err = np.max(np.linalg.norm(v - v_ref, axis=1))
    assert pos_err < 200.0, pos_err  # metres (measured ~78 m)
    assert vel_err < 1e-3, vel_err  # m/s (measured ~1.4e-5)


def test_invalid_interp_step():
    with pytest.raises(ValueError):
        compute_porkchop(planet("earth"), planet("mars"), T0_RANGE, TOF_RANGE,
                         4, 4, use_gpu=False, arrival_interp_step=0.0)
