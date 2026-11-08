"""Numerical boundary and regression tests (task #138).

The older Lambert cross-checks in ``test_gpu_lambert.py`` deliberately keep
away from 0/180 deg transfer angles, are mostly planar heliocentric cases and
tolerate a small fraction of unconverged samples. Every test here instead
requires *all* candidates to be solved (``valid.all()``), and covers:

* fully three-dimensional geometry (random orientation of the transfer
  plane, prograde and retrograde),
* near-degenerate transfer angles (0, 180, 360 deg) in an inclined plane,
* extreme times of flight (seconds .. millennia),
* illegal inputs (shapes, mu, non-finite / non-positive values) mixed into
  otherwise valid batches,
* a continuous sweep of the convergence tolerance (and of maxiter),
* Kepler propagation regression against textbook values and against the
  classical (elliptic / hyperbolic / parabolic) Kepler equations solved
  independently.

Everything runs on the CPU backend (``use_gpu=False``) so it is part of CPU CI.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.lambert.gpu_batch import solve_lambert_batch

MU_SUN = pk.MU_SUN
AU = pk.AU
DAY = pk.DAY2SEC
MU_EARTH = nb.MU_EARTH_KM3_S2  # km^3/s^2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _random_rotations(n, rng):
    """n uniformly distributed 3x3 rotation matrices (unit quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _pykep_rel_errors(res, r1, r2, tof, mu=MU_SUN, prograde=True):
    """Per-candidate max relative velocity error vs pykep (valid rows only)."""
    errs = np.full(len(tof), np.nan)
    for i in np.flatnonzero(res.valid):
        v1_ref, v2_ref = solve_lambert_single(r1[i], r2[i], tof[i], mu=mu, max_revs=0, prograde=prograde)[0]
        errs[i] = max(
            np.linalg.norm(res.v1[i] - v1_ref) / np.linalg.norm(v1_ref),
            np.linalg.norm(res.v2[i] - v2_ref) / np.linalg.norm(v2_ref),
        )
    return errs


def _assert_all_valid_and_match(res, r1, r2, tof, rtol, mu=MU_SUN, prograde=True):
    assert res.valid.all(), f"unsolved candidates: {np.flatnonzero(~res.valid).tolist()}"
    assert np.all(np.isfinite(res.v1)) and np.all(np.isfinite(res.v2))
    errs = _pykep_rel_errors(res, r1, r2, tof, mu=mu, prograde=prograde)
    assert np.max(errs) < rtol, f"max relative |v - v_pykep| = {np.max(errs):.3e}"


# --------------------------------------------------------------------------
# Lambert: textbook regression (3-D)
# --------------------------------------------------------------------------

def test_lambert_curtis_example_5_2_three_dimensional():
    """Curtis, *Orbital Mechanics for Engineering Students*, Example 5.2:
    a genuinely out-of-plane geocentric transfer."""
    r1 = np.array([[5000.0, 10000.0, 2100.0]])
    r2 = np.array([[-14600.0, 2500.0, 7000.0]])
    tof = np.array([3600.0])
    res = solve_lambert_batch(r1, r2, tof, MU_EARTH, use_gpu=False, cpu_fallback=False)
    assert res.valid.all() and not res.fallback.any()
    np.testing.assert_allclose(res.v1[0], [-5.9925, 1.9254, 3.2456], atol=5e-4)
    np.testing.assert_allclose(res.v2[0], [-3.3125, -4.1966, -0.38529], atol=5e-4)
    _assert_all_valid_and_match(res, r1, r2, tof, rtol=1e-8, mu=MU_EARTH)


# --------------------------------------------------------------------------
# Lambert: random 3-D geometry
# --------------------------------------------------------------------------

def _random_3d_cases(n, seed):
    """Transfer plane in a uniformly random orientation; transfer angle
    anywhere in [0.05, 2 pi - 0.05] rad; 1 day .. 27 yr TOF."""
    rng = np.random.default_rng(seed)
    R = _random_rotations(n, rng)
    r1n = rng.uniform(0.3, 5.0, n) * AU
    r2n = rng.uniform(0.3, 5.0, n) * AU
    th = rng.uniform(0.05, 2 * np.pi - 0.05, n)
    zeros = np.zeros(n)
    r1 = np.einsum("nij,nj->ni", R, np.stack([r1n, zeros, zeros], axis=1))
    r2 = np.einsum("nij,nj->ni", R, np.stack([r2n * np.cos(th), r2n * np.sin(th), zeros], axis=1))
    tof = 10 ** rng.uniform(0, 4, n) * DAY
    return r1, r2, tof


@pytest.mark.parametrize("prograde", [True, False])
def test_lambert_random_3d_geometry_all_solved_by_batch(prograde):
    r1, r2, tof = _random_3d_cases(300, seed=11)
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, prograde=prograde, use_gpu=False, cpu_fallback=False)
    # Away from 0/180 deg the batched solver alone must solve every case.
    assert not res.fallback.any()
    _assert_all_valid_and_match(res, r1, r2, tof, rtol=1e-7, prograde=prograde)


def test_lambert_3d_recovers_propagated_state():
    """Propagate inclined (incl. polar and retrograde) elliptic orbits with the
    Kepler propagator and recover the initial velocity with Lambert."""
    rng = np.random.default_rng(12)
    n = 48
    a = rng.uniform(8000.0, 40000.0, n)
    e = rng.uniform(0.0, 0.7, n)
    inc = rng.uniform(0.0, np.pi, n)
    inc[:4] = [np.pi / 2 - 0.2, np.pi / 2 + 0.2, 0.0, np.pi]  # near-polar, equatorial pro/retro
    raan = rng.uniform(0.0, 2 * np.pi, n)
    argp = rng.uniform(0.0, 2 * np.pi, n)
    nu = rng.uniform(0.0, 2 * np.pi, n)
    r0, v0 = _coe_to_rv(a, e, inc, raan, argp, nu, MU_EARTH)
    period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    dt = period * rng.uniform(0.05, 0.9, n)
    r1, v1 = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False)

    hz = np.cross(r0, v0)[:, 2]
    # The prograde flag is defined by the sign of the transfer's angular
    # momentum z-component; exactly polar planes are ambiguous, so skip any
    # with |h_z| tiny relative to |h|.
    hz_rel = hz / np.linalg.norm(np.cross(r0, v0), axis=1)
    sin_dnu = np.linalg.norm(np.cross(r0, r1), axis=1) / np.linalg.norm(r0, axis=1) / np.linalg.norm(r1, axis=1)
    keep = (np.abs(hz_rel) > 1e-3) & (sin_dnu > 1e-3)
    assert keep.sum() >= n - 8
    for pg in (True, False):
        sel = keep & ((hz > 0) == pg)
        res = solve_lambert_batch(r0[sel], r1[sel], dt[sel], MU_EARTH, prograde=pg,
                                  use_gpu=False, cpu_fallback=False)
        assert res.valid.all()
        np.testing.assert_allclose(res.v1, v0[sel], rtol=0, atol=1e-6)
        np.testing.assert_allclose(res.v2, v1[sel], rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# Lambert: near-degenerate transfer angles in an inclined plane
# --------------------------------------------------------------------------

_OFFSETS = (1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2)
_DEGENERATE_ANGLES = (
    list(_OFFSETS)
    + [np.pi + s * d for d in _OFFSETS for s in (-1, 1)]
    + [np.pi]
    + [2 * np.pi - d for d in _OFFSETS]
)


@pytest.mark.parametrize("theta", _DEGENERATE_ANGLES)
def test_lambert_near_degenerate_angles_3d(theta):
    R = _rot_x(np.deg2rad(60.0))  # transfer plane tilted 60 deg
    tofs = np.array([0.5, 30.0, 200.0, 2000.0]) * DAY
    r1 = np.tile(R @ [AU, 0.0, 0.0], (len(tofs), 1))
    r2 = np.tile(R @ [1.5 * AU * np.cos(theta), 1.5 * AU * np.sin(theta), 0.0], (len(tofs), 1))
    sin_dnu = abs(np.sin(theta))

    batch_only = solve_lambert_batch(r1, r2, tofs, MU_SUN, use_gpu=False, cpu_fallback=False)
    assert np.all(np.isfinite(batch_only.v1)) and np.all(np.isfinite(batch_only.v2))
    if sin_dnu < 1e-7:
        # Inside the angle tolerance the batched path must refuse, not guess.
        assert not batch_only.valid.any()
    elif sin_dnu > 1e-5 / 2:
        # Outside it, the batched path alone solves every case.
        assert batch_only.valid.all()

    res = solve_lambert_batch(r1, r2, tofs, MU_SUN, use_gpu=False)
    np.testing.assert_array_equal(res.fallback, ~batch_only.valid)
    # Just outside angle_tol the transfer plane is ill-conditioned: the
    # velocity is ~1/sin(dnu) times more sensitive to the 1e-6 relative TOF
    # tolerance, which bounds the achievable agreement with pykep there.
    rtol = 2e-6 if sin_dnu < 1e-4 else 1e-8
    _assert_all_valid_and_match(res, r1, r2, tofs, rtol=rtol)


# --------------------------------------------------------------------------
# Lambert: extreme times of flight
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tof_s", [1.0, 10.0, 60.0, 600.0, 3600.0])
def test_lambert_extreme_short_tof(tof_s):
    """Seconds-long transfers between AU-scale points: extremely hyperbolic
    (z << 0). The batch path may hand off to the CPU fallback but every case
    must come back solved and agree with pykep."""
    r1 = np.array([[AU, 0.0, 0.0], [AU, 0.0, 0.0]])
    r2 = np.array([[0.0, 1.5 * AU, 0.3 * AU], [0.7 * AU, 0.7 * AU, -0.1 * AU]])
    tof = np.full(2, tof_s)
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    _assert_all_valid_and_match(res, r1, r2, tof, rtol=1e-6)


@pytest.mark.parametrize("tof_days", [1e4, 3e4, 1e5, 1e6])
def test_lambert_extreme_long_tof(tof_days):
    """Decades to millennia: z approaches the (2 pi)^2 single-revolution
    limit. Solved by the batched path itself, without the fallback."""
    r1 = np.array([[AU, 0.0, 0.0], [0.4 * AU, 0.0, 0.1 * AU]])
    r2 = np.array([[0.0, 1.5 * AU, 0.3 * AU], [-3.0 * AU, 2.0 * AU, 0.0]])
    tof = np.full(2, tof_days * DAY)
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=False)
    assert not res.fallback.any()
    _assert_all_valid_and_match(res, r1, r2, tof, rtol=1e-9)


def test_lambert_tof_sweep_over_eight_decades_all_valid():
    """A continuous TOF sweep across 1 minute .. ~2700 years: no gaps."""
    tof = np.logspace(np.log10(60.0), np.log10(1e6 * DAY), 120)
    r1 = np.tile([AU, 0.0, 0.0], (len(tof), 1))
    r2 = np.tile(_rot_x(0.4) @ [1.2 * AU * np.cos(2.0), 1.2 * AU * np.sin(2.0), 0.0], (len(tof), 1))
    res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    _assert_all_valid_and_match(res, r1, r2, tof, rtol=1e-6)


# --------------------------------------------------------------------------
# Lambert: illegal inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r1,r2", [
    (np.zeros((3,)), np.zeros((3,))),          # 1-D, not (N, 3)
    (np.zeros((4, 2)), np.zeros((4, 2))),      # wrong trailing dim
    (np.zeros((4, 3)), np.zeros((5, 3))),      # mismatched N
    (np.zeros((2, 3, 1)), np.zeros((2, 3, 1))),  # too many dims
])
def test_lambert_bad_shapes_raise(r1, r2):
    with pytest.raises(ValueError):
        solve_lambert_batch(r1, r2, np.ones(len(r1)), MU_SUN, use_gpu=False)


@pytest.mark.parametrize("mu", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_lambert_bad_mu_raises(mu):
    r1 = np.array([[AU, 0.0, 0.0]])
    r2 = np.array([[0.0, AU, 0.0]])
    with pytest.raises(ValueError):
        solve_lambert_batch(r1, r2, [100 * DAY], mu, use_gpu=False)


def test_lambert_tof_wrong_length_raises():
    r1 = np.tile([AU, 0.0, 0.0], (3, 1))
    r2 = np.tile([0.0, AU, 0.0], (3, 1))
    with pytest.raises(ValueError):
        solve_lambert_batch(r1, r2, np.ones(2) * DAY, MU_SUN, use_gpu=False)


def test_lambert_illegal_rows_do_not_contaminate_valid_rows():
    """Illegal rows interleaved with a 3-D batch: the legal rows' answers are
    bit-identical to solving them alone, illegal rows are invalid, NaN-free
    and never sent to the fallback."""
    r1, r2, tof = _random_3d_cases(12, seed=13)
    clean = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False)
    assert clean.valid.all()

    bad_r1 = np.array([[np.nan, 0, 0], [0.0, 0.0, 0.0], [AU, 0, 0], [AU, 0, 0], [AU, 0, 0], [-np.inf, 0, 0]])
    bad_r2 = np.array([[0, AU, 0], [0, AU, 0], [0, 0, np.inf], [0, AU, 0], [0, AU, 0], [0, AU, 0]])
    bad_tof = np.array([DAY, DAY, DAY, -DAY, np.nan, DAY])
    m = len(tof) + len(bad_tof)
    bad_idx = np.arange(1, m, 3)[: len(bad_tof)]  # interleave: every 3rd row
    good_idx = np.setdiff1d(np.arange(m), bad_idx)
    R1, R2, T = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros(m)
    R1[good_idx], R2[good_idx], T[good_idx] = r1, r2, tof
    R1[bad_idx], R2[bad_idx], T[bad_idx] = bad_r1, bad_r2, bad_tof

    mixed = solve_lambert_batch(R1, R2, T, MU_SUN, use_gpu=False)
    assert mixed.valid[good_idx].all()
    assert not mixed.valid[bad_idx].any() and not mixed.fallback[bad_idx].any()
    assert np.all(np.isfinite(mixed.v1)) and np.all(np.isfinite(mixed.v2))
    assert np.all(mixed.v1[bad_idx] == 0.0) and np.all(np.isinf(mixed.residual[bad_idx]))
    np.testing.assert_array_equal(mixed.v1[good_idx], clean.v1)
    np.testing.assert_array_equal(mixed.v2[good_idx], clean.v2)


def test_lambert_coincident_points_are_invalid_without_fallback_crash():
    """r1 == r2 (dnu = 0, chord 0): no finite zero-rev solution is claimed
    by the batched path, and the fallback never returns NaN."""
    r = np.array([[AU, 0.0, 0.0], [0.0, 0.0, 2.0 * AU]])
    tof = np.array([100.0, 300.0]) * DAY
    res = solve_lambert_batch(r, r.copy(), tof, MU_SUN, use_gpu=False, cpu_fallback=False)
    assert not res.valid.any()
    res_fb = solve_lambert_batch(r, r.copy(), tof, MU_SUN, use_gpu=False)
    assert np.all(np.isfinite(res_fb.v1)) and np.all(np.isfinite(res_fb.v2))
    assert not np.any(np.isnan(res_fb.residual))


# --------------------------------------------------------------------------
# Lambert: continuous tolerance convergence
# --------------------------------------------------------------------------

# 1e-10 is the tightest *relative* TOF tolerance reachable for every case:
# strongly hyperbolic transfers (AU-scale chord, ~1 day TOF, z << 0) have a
# floating-point noise floor in TOF(z) of ~1e-10 relative, so a tighter tol
# is unattainable there by construction (default tol is 1e-6).
_TOLS = [10.0**-k for k in range(2, 11)]


def test_lambert_tolerance_sweep_converges_continuously():
    """For every tol in 1e-2 .. 1e-10 every candidate converges, its TOF
    residual honours that tol, and the velocity error vs pykep shrinks
    (never grows) as tol tightens -- down to the floating-point floor."""
    r1, r2, tof = _random_3d_cases(60, seed=14)
    max_errs = []
    for tol in _TOLS:
        res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=False, tol=tol)
        assert res.valid.all(), f"tol={tol:g}: {np.count_nonzero(~res.valid)} unsolved"
        assert not res.fallback.any()
        assert np.all(res.residual < tol * np.maximum(tof, 1.0)), f"tol={tol:g}"
        max_errs.append(np.max(_pykep_rel_errors(res, r1, r2, tof)))
    max_errs = np.array(max_errs)
    # Monotone non-increasing, up to round-off at the floor.
    assert np.all(np.diff(max_errs) <= 1e-12), max_errs
    # The error is bounded by (a modest multiple of) the requested tolerance...
    assert np.all(max_errs <= 10.0 * np.array(_TOLS) + 1e-9), max_errs
    # ...and the tightest setting reaches the floating-point floor.
    assert max_errs[-1] < 1e-9


def test_lambert_maxiter_cap_reaches_full_convergence_with_early_exit():
    """Once the cap is large enough every candidate converges, and raising
    it further changes nothing (the loop exits early on its own). Small caps
    are only checked for honesty: they never report more iterations than
    allowed and never return non-finite values. (Per-candidate monotonicity
    in maxiter is *not* a property of a truncated iteration: stopping on a
    bisection step can land on a worse iterate than an earlier stop.)"""
    r1, r2, tof = _random_3d_cases(100, seed=15)
    for maxiter in (1, 2, 3, 5, 8, 12):
        res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=False, maxiter=maxiter)
        assert res.iterations <= maxiter
        assert np.all(np.isfinite(res.v1)) and np.all(np.isfinite(res.v2))
    ref = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=False, maxiter=60)
    assert ref.valid.all() and ref.iterations < 60
    for maxiter in (ref.iterations, 100, 400):
        res = solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=False, cpu_fallback=False, maxiter=maxiter)
        assert res.valid.all() and res.iterations == ref.iterations
        np.testing.assert_array_equal(res.v1, ref.v1)
        np.testing.assert_array_equal(res.v2, ref.v2)


# --------------------------------------------------------------------------
# Kepler: regression against textbook values and classical Kepler equations
# --------------------------------------------------------------------------

def _coe_to_rv(a, e, inc, raan, argp, nu, mu):
    """Classical orbital elements -> inertial (r, v). Works for e < 1 (a > 0)
    and e > 1 (a < 0)."""
    p = a * (1.0 - e**2)
    r_pf = (p / (1.0 + e * np.cos(nu)))[:, None] * np.stack([np.cos(nu), np.sin(nu), np.zeros_like(nu)], 1)
    v_pf = np.sqrt(mu / p)[:, None] * np.stack([-np.sin(nu), e + np.cos(nu), np.zeros_like(nu)], 1)
    return _perifocal_to_inertial(r_pf, raan, inc, argp), _perifocal_to_inertial(v_pf, raan, inc, argp)


def _perifocal_to_inertial(x, raan, inc, argp):
    cO, sO, ci, si, cw, sw = np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc), np.cos(argp), np.sin(argp)
    Q = np.stack([
        np.stack([cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si], -1),
        np.stack([sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si], -1),
        np.stack([sw * si, cw * si, ci], -1),
    ], axis=1)
    return np.einsum("nij,nj->ni", Q, x)


def _solve_elliptic_kepler(M, e):
    """E - e sin E = M by Newton (independent scalar reference)."""
    E = np.where(e < 0.8, M, np.pi * np.sign(M) + 0 * M)
    for _ in range(100):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < 1e-15:
            break
    return E


def _solve_hyperbolic_kepler(M, e):
    """e sinh H - H = M by Newton."""
    H = np.arcsinh(M / e)
    for _ in range(200):
        dH = (e * np.sinh(H) - H - M) / (e * np.cosh(H) - 1.0)
        H = H - dH
        if np.max(np.abs(dH) / np.maximum(1.0, np.abs(H))) < 1e-15:
            break
    return H


def test_kepler_curtis_example_3_7():
    """Curtis Example 3.7: universal-variable propagation, 1 h."""
    r0 = np.array([7000.0, -12124.0, 0.0])
    v0 = np.array([2.6679, 4.6210, 0.0])
    r, v, conv, _ = nb.propagate_kepler_batch(r0, v0, 3600.0, MU_EARTH, use_gpu=False, return_info=True)
    assert conv.all()
    np.testing.assert_allclose(r[0], [-3297.8, 7413.4, 0.0], atol=0.5)
    np.testing.assert_allclose(v[0], [-8.2977, -0.96309, 0.0], atol=1e-3)


@pytest.mark.parametrize("e", [0.0, 1e-8, 0.1, 0.5, 0.9, 0.99, 0.999])
def test_kepler_elliptic_matches_kepler_equation(e):
    rng = np.random.default_rng(int(e * 1e4) + 20)
    n = 32
    a = rng.uniform(7000.0, 45000.0, n) if e < 0.9 else rng.uniform(1.0e5, 4.0e5, n)
    inc, raan, argp = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    M0 = rng.uniform(-np.pi, np.pi, n)
    ee = np.full(n, e)
    E0 = _solve_elliptic_kepler(M0, ee)
    nu0 = 2 * np.arctan2(np.sqrt(1 + ee) * np.sin(E0 / 2), np.sqrt(1 - ee) * np.cos(E0 / 2))
    mean_motion = np.sqrt(MU_EARTH / a**3)
    dt = rng.uniform(-3.0, 3.0, n) * 2 * np.pi / mean_motion  # up to +-3 revolutions
    M1 = np.mod(M0 + mean_motion * dt + np.pi, 2 * np.pi) - np.pi
    E1 = _solve_elliptic_kepler(M1, ee)
    nu1 = 2 * np.arctan2(np.sqrt(1 + ee) * np.sin(E1 / 2), np.sqrt(1 - ee) * np.cos(E1 / 2))

    r0, v0 = _coe_to_rv(a, ee, inc, raan, argp, nu0, MU_EARTH)
    r_ref, v_ref = _coe_to_rv(a, ee, inc, raan, argp, nu1, MU_EARTH)
    r, v, conv, _ = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False, return_info=True)
    assert conv.all()
    # High-e orbits magnify round-off of the elements themselves near
    # periapsis; scale the bound with the periapsis speed amplification.
    amp = (1 + e) / (1 - e)
    rtol = 1e-9 * amp
    assert np.max(np.linalg.norm(r - r_ref, axis=1) / np.linalg.norm(r_ref, axis=1)) < rtol
    assert np.max(np.linalg.norm(v - v_ref, axis=1) / np.linalg.norm(v_ref, axis=1)) < rtol


@pytest.mark.parametrize("e", [1.0 + 1e-6, 1.01, 1.5, 3.0, 10.0])
def test_kepler_hyperbolic_matches_kepler_equation(e):
    rng = np.random.default_rng(int(e * 1e3) + 30)
    n = 32
    rp = rng.uniform(6800.0, 20000.0, n)
    a = rp / (1.0 - e)  # < 0
    inc, raan, argp = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    ee = np.full(n, e)
    nu_inf = np.arccos(-1.0 / e)
    nu0 = rng.uniform(-0.8, 0.8, n) * nu_inf
    H0 = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(nu0 / 2))
    M0 = e * np.sinh(H0) - H0
    mean_motion = np.sqrt(MU_EARTH / (-a) ** 3)
    dt = rng.uniform(-2.0, 2.0, n) * 86400.0
    M1 = M0 + mean_motion * dt
    H1 = _solve_hyperbolic_kepler(M1, ee)
    nu1 = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(H1 / 2))

    r0, v0 = _coe_to_rv(a, ee, inc, raan, argp, nu0, MU_EARTH)
    r_ref, v_ref = _coe_to_rv(a, ee, inc, raan, argp, nu1, MU_EARTH)
    r, v, conv, _ = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False, return_info=True)
    assert conv.all()
    # Near e = 1 the hyperbolic elements are ill-conditioned (e - 1 -> 0).
    rtol = 1e-9 if e > 1.001 else 1e-6
    assert np.max(np.linalg.norm(r - r_ref, axis=1) / np.linalg.norm(r_ref, axis=1)) < rtol
    assert np.max(np.linalg.norm(v - v_ref, axis=1) / np.linalg.norm(v_ref, axis=1)) < rtol


def test_kepler_parabolic_matches_barker_equation():
    """Exactly parabolic (e = 1) trajectories vs Barker's equation
    D + D^3/3 = sqrt(mu/p^3) * 2 * t with D = tan(nu/2) (closed-form cubic)."""
    rng = np.random.default_rng(40)
    n = 32
    p = rng.uniform(13600.0, 40000.0, n)  # semi-latus rectum = 2 rp
    inc, raan, argp = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    nu0 = rng.uniform(-2.5, 2.5, n)
    dt = rng.uniform(-2.0, 2.0, n) * 86400.0

    def rv(nu):
        r_pf = (p / (1 + np.cos(nu)))[:, None] * np.stack([np.cos(nu), np.sin(nu), 0 * nu], 1)
        v_pf = np.sqrt(MU_EARTH / p)[:, None] * np.stack([-np.sin(nu), 1 + np.cos(nu), 0 * nu], 1)
        return _perifocal_to_inertial(r_pf, raan, inc, argp), _perifocal_to_inertial(v_pf, raan, inc, argp)

    D0 = np.tan(nu0 / 2)
    B = D0 + D0**3 / 3 + 2 * np.sqrt(MU_EARTH / p**3) * dt
    # Closed-form solution of D^3/3 + D - B = 0.
    w = np.cbrt(1.5 * B + np.sqrt(1.0 + 2.25 * B**2))
    D1 = w - 1.0 / w
    nu1 = 2 * np.arctan(D1)

    r0, v0 = rv(nu0)
    r_ref, v_ref = rv(nu1)
    r, v, conv, _ = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False, return_info=True)
    assert conv.all()
    assert np.max(np.linalg.norm(r - r_ref, axis=1) / np.linalg.norm(r_ref, axis=1)) < 1e-8
    assert np.max(np.linalg.norm(v - v_ref, axis=1) / np.linalg.norm(v_ref, axis=1)) < 1e-8


def test_kepler_tolerance_sweep_converges_continuously():
    """Kepler propagation converges for every tol in 1e-4 .. 1e-14 across
    elliptic, hyperbolic and near-parabolic states, and the error vs the
    tightest solution shrinks as tol tightens."""
    rng = np.random.default_rng(50)
    n = 48
    rp = rng.uniform(6800.0, 30000.0, n)
    e = np.concatenate([rng.uniform(0.0, 0.95, n // 3), rng.uniform(1.05, 5.0, n // 3),
                        1.0 + rng.choice([-1, 1], n - 2 * (n // 3)) * 10 ** rng.uniform(-10, -4, n - 2 * (n // 3))])
    a = rp / (1.0 - e)
    inc, raan, argp = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    nu0 = rng.uniform(-1.5, 1.5, n)
    r0, v0 = _coe_to_rv(a, e, inc, raan, argp, nu0, MU_EARTH)
    dt = rng.uniform(-1.0, 1.0, n) * 86400.0

    r_ref, _ = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False, tol=1e-15)
    errs = []
    for tol in [10.0**-k for k in range(4, 15)]:
        r, _, conv, _ = nb.propagate_kepler_batch(r0, v0, dt, MU_EARTH, use_gpu=False, tol=tol, return_info=True)
        assert conv.all(), f"tol={tol:g}: {np.count_nonzero(~conv)} unconverged"
        errs.append(np.max(np.linalg.norm(r - r_ref, axis=1) / np.linalg.norm(r_ref, axis=1)))
    errs = np.array(errs)
    assert np.all(np.diff(errs) <= 1e-13), errs
    assert errs[-1] < 1e-11


def test_lambert_near_collinear_valid_means_correct():
    """Acceptance regression: at 1.1e-6 rad (just outside angle_tol) the
    batched path used to report valid=True with v1 ~1.8 km/s away from pykep,
    because arccos(dot/(r1 r2)) quantised the transfer angle. Whenever the
    batched solver claims validity near 0/180 deg it must now agree with pykep."""
    mu = 3.986004418e14
    rng = np.random.default_rng(1)
    checked = 0
    for k in range(400):
        ang = 1.1e-6 if k == 0 else 10 ** rng.uniform(-6.5, -3)
        if k % 2:
            ang = np.pi - ang
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        w = rng.normal(size=3)
        w -= w.dot(u) * u
        w /= np.linalg.norm(w)
        r1 = rng.uniform(6.8e6, 4e7) * u
        r2 = rng.uniform(6.8e6, 4e7) * (np.cos(ang) * u + np.sin(ang) * w)
        tof = rng.uniform(500.0, 40000.0)
        prograde = bool(rng.integers(2))
        res = solve_lambert_batch(r1[None], r2[None], tof, mu, prograde=prograde,
                                  use_gpu=False, cpu_fallback=False)
        if not res.valid[0]:
            continue
        try:
            v1_ref, v2_ref = solve_lambert_single(r1, r2, tof, mu=mu, max_revs=0, prograde=prograde)[0]
        except Exception:  # noqa: BLE001 -- pykep itself rejects the geometry
            continue
        checked += 1
        scale = max(np.linalg.norm(v1_ref), 1.0)
        assert np.linalg.norm(res.v1[0] - v1_ref) < 1e-6 * scale + 1e-2, (ang, tof, prograde)
        assert np.linalg.norm(res.v2[0] - np.asarray(v2_ref)) < 1e-6 * scale + 1e-2, (ang, tof, prograde)
    assert checked > 200
