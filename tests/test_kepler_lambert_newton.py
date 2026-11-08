"""Regression tests for the analytic-derivative Newton iterations (task #135):

* ``orbitopt.dynamics.nbody_gpu`` universal Kepler propagation -- elliptic,
  hyperbolic and near-parabolic states, checked against conservation laws,
  forward/backward round trips and an independent high-accuracy numerical
  integration; plus early termination and the convergence mask.
* ``orbitopt.lambert.gpu_batch`` -- the analytic dTOF/dz against a finite
  difference (including the series branch near z = 0), and Lambert
  solutions for elliptic / hyperbolic / near-parabolic transfers recovered
  by propagating a known state with the Kepler propagator.

Neither the Kepler nor the Lambert-consistency tests need pykep.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.lambert import gpu_batch as gb

MU = nb.MU_EARTH_KM3_S2  # km^3/s^2
R0 = 7000.0  # km


def _states(kind, n=64, seed=0):
    """(r0, v0, dt) batches of a given orbit type around Earth (km, km/s, s)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(6800.0, 42000.0, n)
    th = rng.uniform(0.0, 2 * np.pi, n)
    inc = rng.uniform(-1.0, 1.0, n)
    r0 = np.stack([r * np.cos(th), r * np.sin(th) * np.cos(inc), r * np.sin(th) * np.sin(inc)], axis=1)
    # Velocity: horizontal direction tilted by a random inclination
    # (|i| <= 1 rad, half of the cases retrograde, so the angular momentum's
    # z-sign -- the Lambert prograde flag -- is unambiguous), plus a random
    # flight-path angle.
    rhat = r0 / r[:, None]
    east = np.cross(np.array([0.0, 0.0, 1.0]), rhat)
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(rhat, east)
    tilt = rng.uniform(-1.0, 1.0, n) + np.where(rng.random(n) < 0.5, 0.0, np.pi)
    t_hat = np.cos(tilt)[:, None] * east + np.sin(tilt)[:, None] * north
    gamma = rng.uniform(-0.6, 0.6, n)
    vdir = np.cos(gamma)[:, None] * t_hat + np.sin(gamma)[:, None] * rhat
    v_esc = np.sqrt(2.0 * MU / r)
    if kind == "elliptic":
        speed = v_esc * rng.uniform(0.5, 0.95, n)
        a = 1.0 / (2.0 / r - speed**2 / MU)
        period = 2 * np.pi * np.sqrt(a**3 / MU)
        dt = period * rng.uniform(-2.5, 2.5, n)
    elif kind == "hyperbolic":
        speed = v_esc * rng.uniform(1.05, 3.0, n)
        dt = rng.uniform(-2.0, 2.0, n) * 86400.0
    elif kind == "near_parabolic":
        speed = v_esc * (1.0 + rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(-12, -5, n))
        speed[:4] = v_esc[:4]  # exactly parabolic (alpha ~ 0 to rounding)
        dt = rng.uniform(-2.0, 2.0, n) * 86400.0
    else:
        raise ValueError(kind)
    return r0, speed[:, None] * vdir, dt


def _energy(r, v):
    return 0.5 * np.sum(v * v, axis=1) - MU / np.linalg.norm(r, axis=1)


KINDS = ["elliptic", "hyperbolic", "near_parabolic"]


@pytest.mark.parametrize("kind", KINDS)
def test_kepler_conserves_energy_and_angular_momentum(kind):
    r0, v0, dt = _states(kind)
    r, v, conv, _ = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False, return_info=True)
    assert conv.all()
    e0, e1 = _energy(r0, v0), _energy(r, v)
    scale = MU / np.linalg.norm(r0, axis=1)
    assert np.max(np.abs(e1 - e0) / scale) < 1e-9
    h0, h1 = np.cross(r0, v0), np.cross(r, v)
    assert np.max(np.linalg.norm(h1 - h0, axis=1) / np.linalg.norm(h0, axis=1)) < 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_kepler_round_trip(kind):
    r0, v0, dt = _states(kind, seed=1)
    r1, v1 = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False)
    r2, v2 = nb.propagate_kepler_batch(r1, v1, -dt, MU, use_gpu=False)
    rel_r = np.linalg.norm(r2 - r0, axis=1) / np.linalg.norm(r0, axis=1)
    rel_v = np.linalg.norm(v2 - v0, axis=1) / np.linalg.norm(v0, axis=1)
    assert np.max(rel_r) < 1e-8 and np.max(rel_v) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_kepler_matches_numerical_integration(kind):
    integrate = pytest.importorskip("scipy.integrate")
    r0, v0, dt = _states(kind, n=8, seed=2)
    r, v = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False)

    def rhs(_t, y):
        rr = y[:3]
        return np.concatenate([y[3:], -MU * rr / np.linalg.norm(rr) ** 3])

    for i in range(len(dt)):
        sol = integrate.solve_ivp(rhs, (0.0, dt[i]), np.concatenate([r0[i], v0[i]]),
                                  method="DOP853", rtol=1e-12, atol=1e-9)
        ref = sol.y[:, -1]
        assert np.linalg.norm(r[i] - ref[:3]) / np.linalg.norm(ref[:3]) < 1e-7
        assert np.linalg.norm(v[i] - ref[3:]) / np.linalg.norm(ref[3:]) < 1e-7


def test_kepler_full_period_returns_to_start():
    r0, v0, _ = _states("elliptic", n=32, seed=3)
    a = 1.0 / (2.0 / np.linalg.norm(r0, axis=1) - np.sum(v0 * v0, axis=1) / MU)
    period = 2 * np.pi * np.sqrt(a**3 / MU)
    r, v = nb.propagate_kepler_batch(r0, v0, 3 * period, MU, use_gpu=False)
    assert np.max(np.linalg.norm(r - r0, axis=1) / np.linalg.norm(r0, axis=1)) < 1e-8


def test_kepler_early_exit_and_zero_dt():
    r0, v0, dt = _states("elliptic", n=16, seed=4)
    dt[:3] = 0.0
    r, v, conv, iters = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False, maxiter=50, return_info=True)
    assert conv.all()
    assert 0 < iters < 50  # Newton converges quadratically; never needs the cap
    np.testing.assert_allclose(r[:3], r0[:3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(v[:3], v0[:3], rtol=0, atol=1e-12)
    # All-zero dt: converged at the initial guess, no iterations at all.
    _, _, conv0, iters0 = nb.propagate_kepler_batch(r0, v0, 0.0, MU, use_gpu=False, return_info=True)
    assert conv0.all() and iters0 == 0


def test_kepler_reports_non_convergence():
    r0, v0, dt = _states("hyperbolic", n=8, seed=5)
    _, _, conv, iters = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False, maxiter=1, return_info=True)
    assert iters == 1
    assert not conv.all()


# --------------------------------------------------------------------------
# Lambert
# --------------------------------------------------------------------------

def test_lambert_analytic_derivative_matches_finite_difference():
    r1 = np.full(5, 7000.0)
    r2 = np.full(5, 12000.0)
    dnu = np.array([0.3, 1.2, 2.5, 3.6, 5.5])
    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1.0 - np.cos(dnu)))
    for z0 in [-80.0, -4.0, -0.2, -0.051, -0.049, -1e-3, 0.0, 1e-3, 0.049, 0.051, 0.5, 5.0, 25.0, 37.0]:
        z = np.full(5, z0)
        t, dtdz, phys = gb._tof_and_dtdz(z, r1, r2, A, MU, np)
        # 4th-order (Richardson) central difference: a plain h=1e-5 central
        # difference is noisier than the analytic value near z = 0.
        h = 1e-3
        tp, _, _, pp = gb._tof_of_z(z + h, r1, r2, A, MU, np)
        tm, _, _, pm = gb._tof_of_z(z - h, r1, r2, A, MU, np)
        tp2, _, _, pp2 = gb._tof_of_z(z + 2 * h, r1, r2, A, MU, np)
        tm2, _, _, pm2 = gb._tof_of_z(z - 2 * h, r1, r2, A, MU, np)
        ok = phys & pp & pm & pp2 & pm2
        fd = (8.0 * (tp - tm) - (tp2 - tm2)) / (12.0 * h)
        np.testing.assert_allclose(dtdz[ok], fd[ok], rtol=1e-6, err_msg=f"z={z0}")


@pytest.mark.parametrize("kind", KINDS)
def test_lambert_recovers_propagated_orbit(kind):
    """Propagate a known (r0, v0) by dt, then solve Lambert(r0, r(dt), dt):
    it must give back v0 / v(dt) for elliptic, hyperbolic and near-parabolic
    transfers (z > 0, z < 0 and z ~ 0 respectively)."""
    r0, v0, dt = _states(kind, n=64, seed=6)
    # Lambert is zero-rev and forward-time: keep dt > 0 and, for ellipses,
    # under one period with a transfer angle away from 0/180 deg.
    dt = np.abs(dt)
    if kind == "elliptic":
        a = 1.0 / (2.0 / np.linalg.norm(r0, axis=1) - np.sum(v0 * v0, axis=1) / MU)
        dt = 2 * np.pi * np.sqrt(a**3 / MU) * np.random.default_rng(7).uniform(0.05, 0.9, len(dt))
    r1, v1 = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False)
    prograde = np.cross(r0, v0)[:, 2] >= 0.0
    cosang = np.sum(r0 * r1, axis=1) / np.linalg.norm(r0, axis=1) / np.linalg.norm(r1, axis=1)
    keep = np.abs(np.sin(np.arccos(np.clip(cosang, -1, 1)))) > 1e-2

    for pg in (True, False):
        sel = keep & (prograde == pg)
        if not sel.any():
            continue
        res = gb.solve_lambert_batch(r0[sel], r1[sel], dt[sel], MU, prograde=pg,
                                     use_gpu=False, cpu_fallback=False)
        assert res.valid.mean() > 0.95, f"{kind}: only {res.valid.mean():.2%} valid"
        ok = res.valid
        np.testing.assert_allclose(res.v1[ok], v0[sel][ok], rtol=0, atol=1e-6 * np.abs(v0).max())
        np.testing.assert_allclose(res.v2[ok], v1[sel][ok], rtol=0, atol=1e-6 * np.abs(v0).max())


def test_lambert_early_exit_reports_iterations():
    r0, v0, dt = _states("elliptic", n=32, seed=8)
    a = 1.0 / (2.0 / np.linalg.norm(r0, axis=1) - np.sum(v0 * v0, axis=1) / MU)
    dt = 2 * np.pi * np.sqrt(a**3 / MU) * 0.3
    r1, _ = nb.propagate_kepler_batch(r0, v0, dt, MU, use_gpu=False)
    sel = np.cross(r0, v0)[:, 2] >= 0.0
    res = gb.solve_lambert_batch(r0[sel], r1[sel], dt[sel], MU, use_gpu=False,
                                 cpu_fallback=False, maxiter=200)
    assert res.valid.all()
    assert 0 < res.iterations < 200
