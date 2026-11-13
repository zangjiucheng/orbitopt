"""GPU-batchable, coarse Earth-Moon(-Sun) trajectory propagation for screening
translunar-injection candidates.

This is deliberately NOT navigation-grade: it exists to evaluate thousands of
candidate TLI burns (varying epoch/direction/magnitude) in one batched pass
so that only the handful of promising survivors get handed to tudatpy's
full-fidelity SPICE-based numerical propagator for refinement. Two pieces
make this possible on GPU. First, a batched universal-variable Kepler
propagator (Vallado's formulation via Curtis's f-and-g functions, Stumpff
series) advances N independent two-body states with a damped analytic-
derivative Newton iteration, no per-item branching, and an early exit once
every candidate has converged. Second, that same
propagator is reused to build a cheap analytic Moon (and optionally Sun)
ephemeris: a single real SPICE state is sampled once at J2000 to seed a
two-body osculating orbit, which is then propagated to any batch of times
without touching SPICE again. A fixed-step RK4 integrator layers Earth
point-mass gravity plus Moon (and optional Sun) third-body perturbations on
top of that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitopt.core.gpu import get_array_module, to_numpy
from orbitopt.physics_io import (
    KEPLER_BATCH,
    NBODY_SCREENING,
    UnitSystem,
    check_grav_parameter,
)

MU_EARTH_KM3_S2 = 398600.4418
MU_MOON_KM3_S2 = 4902.8
MU_SUN_KM3_S2 = 1.32712440018e11

#: This module's declared unit system: kilometres, seconds (see
#: :data:`orbitopt.physics_io.NBODY_SCREENING`).
UNITS = UnitSystem("km", "s")

_SPICE_LOADED = False
_MOON_STATE0_KM = {}  # keyed by seed_epoch_ephemeris_seconds (rounded)
_SUN_STATE0_KM = {}


def _ensure_spice_loaded():
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        from tudatpy.interface import spice

        spice.load_standard_kernels()
        _SPICE_LOADED = True


_STUMPFF_SERIES_ZMAX = 0.1
# C(z) = sum (-z)^k / (2k+2)!,  S(z) = sum (-z)^k / (2k+3)!,  k = 0..5
_C_SERIES = (1 / 2, -1 / 24, 1 / 720, -1 / 40320, 1 / 3628800, -1 / 479001600)
_S_SERIES = (1 / 6, -1 / 120, 1 / 5040, -1 / 362880, 1 / 39916800, -1 / 6227020800)


def _horner(z, coeffs):
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = c + z * acc
    return acc


def _stumpff(z, xp):
    """Stumpff functions C(z), S(z), vectorized and branchless.

    Same trick as orbitopt.lambert.gpu_batch._stumpff: xp.where evaluates
    every branch eagerly, so each branch's denominator is pre-guarded
    against division by zero (and cosh/sinh overflow) before selection.
    """
    # Below |z| = 0.1 the closed forms lose ~eps/|z| to cancellation
    # (1 - cos sqrt(z) etc.), which stalls Newton near parabolic orbits; the
    # degree-5 series there is accurate to < 1e-17.
    pos = z > _STUMPFF_SERIES_ZMAX
    neg = z < -_STUMPFF_SERIES_ZMAX

    z_pos = xp.where(pos, z, 1.0)
    sz_pos = xp.sqrt(z_pos)
    c_pos = (1.0 - xp.cos(sz_pos)) / z_pos
    s_pos = (sz_pos - xp.sin(sz_pos)) / z_pos**1.5

    z_neg = xp.where(neg, -z, 1.0)
    sz_neg = xp.sqrt(z_neg)
    c_neg = (1.0 - xp.cosh(sz_neg)) / (-z_neg)
    s_neg = (xp.sinh(sz_neg) - sz_neg) / z_neg**1.5

    c_small = _horner(z, _C_SERIES)
    s_small = _horner(z, _S_SERIES)

    C = xp.where(pos, c_pos, xp.where(neg, c_neg, c_small))
    S = xp.where(pos, s_pos, xp.where(neg, s_neg, s_small))
    return C, S


def _kepler_residual_and_derivative(chi, r0n, vr0, alpha, dt, sqrt_mu, xp):
    """Universal Kepler equation residual F(chi), its analytic derivative
    dF/dchi, and the magnitude scale of F's terms, from one Stumpff call.

        F      = (r0 vr0 / sqrt_mu) chi^2 C + (1 - alpha r0) chi^3 S + r0 chi - sqrt_mu dt
        dF/dchi = (r0 vr0 / sqrt_mu) chi (1 - z S) + (1 - alpha r0) chi^2 C + r0

    with z = alpha chi^2. dF/dchi equals the orbit radius r(chi) > 0, so the
    Newton direction is always a descent direction for |F|.
    """
    z = alpha * chi**2
    C, S = _stumpff(z, xp)
    k = r0n * vr0 / sqrt_mu
    m = 1.0 - alpha * r0n
    t1 = k * chi**2 * C
    t2 = m * chi**3 * S
    t3 = r0n * chi
    t4 = sqrt_mu * dt
    F = t1 + t2 + t3 - t4
    dF = k * chi * (1.0 - z * S) + m * chi**2 * C + r0n
    scale = xp.abs(t1) + xp.abs(t2) + xp.abs(t3) + xp.abs(t4)
    return F, dF, scale


# Maximum step halvings per Newton iteration when the full step does not
# reduce |F| (or overflows cosh/sinh on hyperbolic orbits). The halving loop
# only runs while some candidate is still rejecting.
_KEPLER_MAX_BACKTRACK = 30


def _kepler_initial_guess(r0, v0, r0n, alpha, dt, mu, sqrt_mu, xp):
    """Vallado (Alg. 8) initial universal anomaly: sqrt(mu)*alpha*dt for
    ellipses / near-parabolas, the logarithmic hyperbolic guess otherwise
    (sqrt(mu)*|alpha|*dt, the old guess, can be far too large there)."""
    chi_ell = sqrt_mu * xp.abs(alpha) * dt
    hyp = alpha < -1e-12
    a = 1.0 / xp.where(hyp, alpha, -1.0)  # < 0
    sgn = xp.where(dt >= 0.0, 1.0, -1.0)
    rdotv = xp.sum(r0 * v0, axis=-1)
    num = -2.0 * mu * alpha * dt
    den = rdotv + sgn * xp.sqrt(-mu * a) * (1.0 - r0n * alpha)
    ratio = num / xp.where(den != 0.0, den, 1.0)
    ok = hyp & (den != 0.0) & (ratio > 0.0) & xp.isfinite(ratio)
    chi_hyp = sgn * xp.sqrt(-a) * xp.log(xp.where(ok, ratio, 1.0))
    ok = ok & xp.isfinite(chi_hyp)
    return xp.where(ok, chi_hyp, chi_ell)


def _solve_universal_anomaly(r0, v0, r0n, vr0, alpha, dt, mu, xp, maxiter, tol):
    """Damped Newton on the universal Kepler equation with analytic
    derivative, per-candidate convergence and early exit.

    Returns (chi, converged, iterations). A candidate is converged once
    |F| <= tol * scale(F terms); it is then frozen. The loop stops as soon as
    no candidate is still active (one device->host sync per iteration).
    Damping: a Newton step that does not decrease |F| (or produces a
    non-finite F) is halved up to ``_KEPLER_MAX_BACKTRACK`` times; if none
    is accepted the candidate stops where it is (and is reported
    unconverged if its residual is still above ``tol``).
    """
    sqrt_mu = xp.sqrt(mu)
    chi = _kepler_initial_guess(r0, v0, r0n, alpha, dt, mu, sqrt_mu, xp)
    chi = xp.where(xp.isfinite(chi), chi, 0.0)
    F, dF, scale = _kepler_residual_and_derivative(chi, r0n, vr0, alpha, dt, sqrt_mu, xp)
    finite = xp.isfinite(F) & xp.isfinite(dF)
    converged = finite & (xp.abs(F) <= tol * scale)
    active = finite & ~converged

    iterations = 0
    for _ in range(maxiter):
        if not bool(xp.any(active)):
            break
        iterations += 1
        dF_ok = xp.isfinite(dF) & (dF > 0.0)
        step = -F / xp.where(dF_ok, dF, xp.maximum(r0n, 1e-300))
        trial = chi + step
        F_t, dF_t, scale_t = _kepler_residual_and_derivative(trial, r0n, vr0, alpha, dt, sqrt_mu, xp)
        reject = active & ~(xp.isfinite(F_t) & xp.isfinite(dF_t) & (xp.abs(F_t) < xp.abs(F)))
        for _ in range(_KEPLER_MAX_BACKTRACK):
            if not bool(xp.any(reject)):
                break
            step = xp.where(reject, 0.5 * step, step)
            trial = chi + step
            F_b, dF_b, scale_b = _kepler_residual_and_derivative(trial, r0n, vr0, alpha, dt, sqrt_mu, xp)
            F_t = xp.where(reject, F_b, F_t)
            dF_t = xp.where(reject, dF_b, dF_t)
            scale_t = xp.where(reject, scale_b, scale_t)
            reject = reject & ~(xp.isfinite(F_b) & xp.isfinite(dF_b) & (xp.abs(F_b) < xp.abs(F)))

        # Candidates for which no damped step reduces |F| have hit the
        # floating-point floor (or cannot progress): they stay put and
        # retire below, converged or not.
        take = active & ~reject
        moved = take & (trial != chi)
        chi = xp.where(take, trial, chi)
        F = xp.where(take, F_t, F)
        dF = xp.where(take, dF_t, dF)
        scale = xp.where(take, scale_t, scale)

        newly = take & (xp.abs(F) <= tol * scale)
        converged = converged | newly
        # Retire converged candidates and those that can no longer move.
        active = active & ~newly & moved

    return chi, converged, iterations


def _propagate_kepler_core(r0, v0, dt, mu, xp, maxiter, tol=1e-12, return_info=False):
    """Array-native universal-variable Kepler propagation, operating on
    whatever xp arrays it is handed and returning xp arrays (no host
    round-trip). Split out from ``propagate_kepler_batch`` so that
    ``propagate_spacecraft_batch``'s per-RK4-substage Moon/Sun lookups can
    stay resident on the GPU instead of paying a device<->host copy at
    every one of the 4*n_steps calls -- that round trip, not the Newton
    iteration itself, was the dominant GPU cost before this split existed.

    With ``return_info=True`` also returns ``(converged_mask, iterations)``.
    """
    r0 = xp.atleast_2d(xp.asarray(r0, dtype=xp.float64))
    v0 = xp.atleast_2d(xp.asarray(v0, dtype=xp.float64))
    dt = xp.atleast_1d(xp.asarray(dt, dtype=xp.float64))

    n = max(r0.shape[0], v0.shape[0], dt.shape[0])
    r0 = xp.broadcast_to(r0, (n, 3))
    v0 = xp.broadcast_to(v0, (n, 3))
    dt = xp.broadcast_to(dt, (n,))

    r0n = xp.linalg.norm(r0, axis=-1)
    v0n = xp.linalg.norm(v0, axis=-1)
    vr0 = xp.sum(r0 * v0, axis=-1) / r0n
    alpha = 2.0 / r0n - v0n**2 / mu
    sqrt_mu = xp.sqrt(mu)

    with np.errstate(all="ignore"):
        chi, converged, iterations = _solve_universal_anomaly(
            r0, v0, r0n, vr0, alpha, dt, mu, xp, maxiter, tol,
        )

    z = alpha * chi**2
    C, S = _stumpff(z, xp)

    f = 1.0 - (chi**2 / r0n) * C
    g = dt - (chi**3 / sqrt_mu) * S
    r = f[:, None] * r0 + g[:, None] * v0
    rn = xp.linalg.norm(r, axis=-1)

    fdot = (sqrt_mu / (rn * r0n)) * (alpha * chi**3 * S - chi)
    gdot = 1.0 - (chi**2 / rn) * C
    v = fdot[:, None] * r0 + gdot[:, None] * v0

    if return_info:
        return r, v, converged, iterations
    return r, v


def _on_device(a) -> bool:
    """True for a CuPy (or other non-host) array. Note that numpy arrays have
    had a ``.device`` attribute since NumPy 2, so that is not the test."""
    return type(a).__module__.split(".")[0] not in ("numpy", "builtins")


def _check_host_states(r0, v0, *, spec, where: str) -> None:
    """Run the physics I/O state checks on host arrays.

    Device arrays are skipped: :mod:`orbitopt.physics_io` is a host-side layer,
    and pulling a whole batch back to the CPU to validate it would cost more
    than the propagation does. Every public entry point here takes numpy
    arguments, so in practice the check always runs."""
    if _on_device(r0) or _on_device(v0):
        return
    spec.check_state(r0, v0, where=where)


def propagate_kepler_batch(r0, v0, dt, mu, use_gpu=None, maxiter=50, tol=1e-12, return_info=False):
    """Propagate N independent two-body states by the universal-variable
    method (Curtis Algorithms 3.3/3.4; equivalent to Vallado's universal
    Kepler-equation formulation), valid for elliptical, parabolic, and
    hyperbolic orbits alike with no branching on orbit type.

    ``dt`` broadcasts against ``r0``/``v0`` rather than requiring a strict
    one-dt-per-case shape: pass ``dt`` as (N,) for one time per initial
    state, or pass a single (3,) ``r0``/``v0`` with (N,) ``dt`` to propagate
    ONE state to N different output times (this is exactly how
    ``moon_state_batch`` reuses this function). Internally both r0/v0 and dt
    are broadcast to a common length N via ``xp.broadcast_to``, so either
    calling convention -- and anything numpy broadcasting allows between
    them -- works without special-casing.

    The universal anomaly is solved by a damped Newton iteration using the
    analytic derivative dF/dchi (= r(chi)), with backtracking whenever a
    step fails to reduce |F|. ``maxiter`` is an upper bound: converged
    candidates (|F| <= tol * magnitude of F's terms) are frozen and the loop
    exits once none remain active.

    Returns
    -------
    (r, v) : tuple of (N, 3) numpy arrays, position and velocity at t0+dt.
    With ``return_info=True``: (r, v, converged, iterations), where
    ``converged`` is an (N,) bool numpy array (the residual check) and
    ``iterations`` the number of Newton iterations actually run.
    """
    xp = get_array_module(use_gpu)
    if check_grav_parameter(mu, units=UNITS,
                           where="orbitopt.dynamics.nbody_gpu.propagate_kepler_batch: mu") is not None:
        # ``mu`` is a known body's tabulated value, so the unit system it fixes
        # is the one the states must be in too.
        _check_host_states(r0, v0, spec=KEPLER_BATCH,
                           where="orbitopt.dynamics.nbody_gpu.propagate_kepler_batch")
    out = _propagate_kepler_core(r0, v0, dt, mu, xp, maxiter, tol=tol, return_info=return_info)
    if return_info:
        r, v, converged, iterations = out
        return to_numpy(r), to_numpy(v), np.asarray(to_numpy(converged), dtype=bool), iterations
    r, v = out
    return to_numpy(r), to_numpy(v)


def get_moon_state0(seed_epoch_ephemeris_seconds=0.0):
    """Moon Cartesian state relative to Earth's center, at the given
    ephemeris time (default 0.0 = J2000 epoch, 2000-01-01 12:00 TDB), in
    the ECLIPJ2000 frame (ecliptic-and-equinox of J2000) -- queried from
    SPICE exactly once per distinct seed epoch and cached (keyed by that
    epoch), since this seeds the cheap two-body model below rather than
    being called per-sample.

    The seed epoch matters more than it looks: the two-body osculating
    approximation this module builds on it is only accurate for a handful
    of days *from the seed*, since it ignores solar perturbation, node
    regression, etc. -- real effects that only matter over long spans.
    Always seeding at J2000 (year 2000) and then asking for the Moon's
    state at a 2026 mission epoch means propagating that osculating ellipse
    forward by 26+ years of pure two-body motion, which diverges by
    hundreds of thousands of km (empirically verified) -- not a small
    correction. Callers working a real mission date MUST pass that
    mission's own reference epoch here (or, for ``propagate_spacecraft_batch``,
    rely on its default of auto-seeding at ``epoch0``), not rely on the
    J2000 default.

    ECLIPJ2000, not the equatorial "J2000" frame, to match the frame used
    by orbitopt.bodies.moon_state and orbitopt.verify.tudat_propagate's
    body settings -- SPICE's "J2000" and "ECLIPJ2000" are related by a
    ~23.4-degree rotation (Earth's obliquity), not interchangeable, and
    every other module in this package that touches Moon/Earth ephemerides
    uses ECLIPJ2000.

    Returns
    -------
    (r0, v0) : each (3,) numpy arrays, km and km/s.
    """
    NBODY_SCREENING.check_epochs(
        seed_epoch_ephemeris_seconds,
        where="orbitopt.dynamics.nbody_gpu.get_moon_state0: seed_epoch_ephemeris_seconds",
    )
    key = round(float(seed_epoch_ephemeris_seconds), 3)
    if key not in _MOON_STATE0_KM:
        _ensure_spice_loaded()
        from tudatpy.interface import spice

        state = spice.get_body_cartesian_state_at_epoch(
            "Moon", "Earth", "ECLIPJ2000", "NONE", key
        )
        _MOON_STATE0_KM[key] = (
            np.asarray(state[:3], dtype=np.float64) / 1000.0,
            np.asarray(state[3:], dtype=np.float64) / 1000.0,
        )
    return _MOON_STATE0_KM[key][0].copy(), _MOON_STATE0_KM[key][1].copy()


def get_sun_state0(seed_epoch_ephemeris_seconds=0.0):
    """Sun Cartesian state relative to Earth's center at the given ephemeris
    time, in ECLIPJ2000 -- same conventions, caching, and seed-epoch
    rationale as ``get_moon_state0``. Used only when
    ``propagate_spacecraft_batch(..., include_sun=True)``.
    """
    NBODY_SCREENING.check_epochs(
        seed_epoch_ephemeris_seconds,
        where="orbitopt.dynamics.nbody_gpu.get_sun_state0: seed_epoch_ephemeris_seconds",
    )
    key = round(float(seed_epoch_ephemeris_seconds), 3)
    if key not in _SUN_STATE0_KM:
        _ensure_spice_loaded()
        from tudatpy.interface import spice

        state = spice.get_body_cartesian_state_at_epoch(
            "Sun", "Earth", "ECLIPJ2000", "NONE", key
        )
        _SUN_STATE0_KM[key] = (
            np.asarray(state[:3], dtype=np.float64) / 1000.0,
            np.asarray(state[3:], dtype=np.float64) / 1000.0,
        )
    return _SUN_STATE0_KM[key][0].copy(), _SUN_STATE0_KM[key][1].copy()


def _moon_state_core(dt, xp, mu_earth, maxiter, seed_epoch_ephemeris_seconds=0.0, mu_moon=MU_MOON_KM3_S2):
    r0, v0 = get_moon_state0(seed_epoch_ephemeris_seconds)
    # The Moon's *relative* orbit about Earth's center obeys
    # r'' = -G(M_earth + M_moon) r / |r|^3 -- the two-body relative-motion
    # mu is the sum of both bodies' GM, not just the primary's (unlike the
    # Earth-Sun case in _sun_state_core, where mu_earth/mu_sun ~ 3e-6 is
    # genuinely negligible and dropping it is a deliberate, documented
    # approximation). mu_moon/(mu_earth+mu_moon) ~ 1.2%, which is not
    # negligible against this module's own precision budget.
    return _propagate_kepler_core(xp.asarray(r0), xp.asarray(v0), dt, mu_earth + mu_moon, xp, maxiter)


def _sun_state_core(dt, xp, mu_sun, maxiter, seed_epoch_ephemeris_seconds=0.0):
    r0, v0 = get_sun_state0(seed_epoch_ephemeris_seconds)
    return _propagate_kepler_core(xp.asarray(r0), xp.asarray(v0), dt, mu_sun, xp, maxiter)


def moon_state_batch(dt, use_gpu=None, mu_earth=MU_EARTH_KM3_S2, maxiter=20, seed_epoch_ephemeris_seconds=0.0, mu_moon=MU_MOON_KM3_S2):
    """Moon (r, v) relative to Earth at an array of times ``dt`` (seconds
    *since* ``seed_epoch_ephemeris_seconds``, default J2000), by propagating
    the single real SPICE-sampled state from ``get_moon_state0`` forward
    with the same universal-variable propagator as ``propagate_kepler_batch``
    -- a fast, fully GPU-vectorizable but two-body-approximate stand-in for
    a real lunar ephemeris (no perturbations from the Sun, Earth oblateness,
    etc.). Accurate to a few thousand km for ``dt`` up to ~10 days; for a
    real mission epoch far from J2000, pass that epoch as
    ``seed_epoch_ephemeris_seconds`` and keep ``dt`` small (elapsed mission
    time), don't pass a ``dt`` of "epoch since J2000" directly -- see
    ``get_moon_state0``'s docstring for why that silently produces
    hundred-thousand-km errors instead of an exception.

    The propagation uses ``mu_earth + mu_moon`` as the two-body gravitational
    parameter, since this models the Moon's *relative* orbit about Earth's
    center (r'' = -G(M_earth+M_moon) r/|r|^3), not a test-particle orbit
    around a fixed Earth -- see ``_moon_state_core``.

    ``maxiter`` defaults lower than ``propagate_kepler_batch``'s own default
    (empirically the near-circular, mildly-eccentric Earth-Moon two-body
    orbit converges to machine precision in under 10 Newton iterations --
    see the module's validation notes) because this function is called at
    every RK4 sub-stage of ``propagate_spacecraft_batch``, where each
    Newton iteration is a separate GPU kernel launch and launch-count, not
    per-launch array size, is what dominates wall time for that caller.

    Returns
    -------
    (r, v) : each (N, 3) numpy arrays, km and km/s.
    """
    xp = get_array_module(use_gpu)
    r, v = _moon_state_core(dt, xp, mu_earth, maxiter, seed_epoch_ephemeris_seconds, mu_moon)
    return to_numpy(r), to_numpy(v)


def sun_state_batch(dt, use_gpu=None, mu_sun=MU_SUN_KM3_S2, maxiter=20, seed_epoch_ephemeris_seconds=0.0):
    """Sun (r, v) relative to Earth at an array of times ``dt`` (seconds
    since ``seed_epoch_ephemeris_seconds``), by the same
    one-SPICE-sample-plus-two-body-propagation trick as ``moon_state_batch``
    (see its docstring for the seed-epoch caveat). Using ``mu_sun`` (rather
    than ``mu_sun + mu_earth``) for the propagation is the standard
    Earth-centered-two-body approximation of the Earth-Sun problem, exact
    to better than 1 part in 3e5 since mu_earth << mu_sun. See
    ``moon_state_batch`` for why the default ``maxiter`` is lower here than
    in ``propagate_kepler_batch`` itself.
    """
    xp = get_array_module(use_gpu)
    r, v = _sun_state_core(dt, xp, mu_sun, maxiter, seed_epoch_ephemeris_seconds)
    return to_numpy(r), to_numpy(v)


def _precompute_body_grid(
    epoch0, dt, n_steps, xp, mu_earth, mu_moon, include_sun, mu_sun, maxiter, seed_epoch
):
    """Moon (and optionally Sun) positions at every RK4 half-step epoch of
    ``propagate_spacecraft_batch``, on a *compact* grid over the distinct
    ``(epoch0, dt)`` pairs of the batch.

    Returns ``(bodies, inverse)``. ``bodies`` is a list of
    ``(mu_body, r_body, indirect)`` tuples where ``r_body`` is a
    (2*n_steps+1, U, 3) xp array of body positions at ``epoch0 + k*dt/2``
    for each of the U distinct pairs, and ``indirect`` is the matching
    ``r_body / |r_body|^3`` indirect (Earth-acceleration) term -- both
    independent of the spacecraft state, so computed once here.
    ``inverse`` is an (N,) int64 numpy array mapping each candidate to its
    column of the compact grid.

    The grid is deliberately NOT expanded to (2*n_steps+1, N, 3): with
    mixed flight times / epochs that would bring back O(N x n_steps)
    memory (~769 MB for the Moon alone at 20,000 candidates x 400 steps),
    defeating ``output="closest_approach"``. Callers gather one time slice
    at a time instead (``r_body[k][inverse]`` -- O(N) per stage).

    The half-step epochs are reproduced with the same floating-point
    accumulation the integrator historically used (``t += dt`` per step,
    ``+ 0.5*dt`` for the mid stages), so the result matches the previous
    per-stage evaluation bit-for-bit up to Kepler-solver round-off.
    """
    epoch0_h = np.asarray(to_numpy(epoch0), dtype=np.float64)
    dt_h = np.asarray(to_numpy(dt), dtype=np.float64)
    pairs = np.stack([epoch0_h, dt_h], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1).astype(np.int64)
    u_epoch0, u_dt = uniq[:, 0], uniq[:, 1]

    n_grid = 2 * n_steps + 1
    t_grid = np.empty((n_grid, uniq.shape[0]), dtype=np.float64)
    t = u_epoch0.copy()
    for step in range(n_steps):
        t_grid[2 * step] = t
        t_grid[2 * step + 1] = t + 0.5 * u_dt
        t_grid[2 * step + 2] = t + u_dt
        t = t + u_dt
    # (k4 of step s and k1 of step s+1 share index 2s+2; both were always
    # ``t + dt`` computed the same way, so one entry serves both.)

    t_since_seed = xp.asarray((t_grid - seed_epoch).reshape(-1))

    bodies = [(mu_moon, lambda ts: _moon_state_core(ts, xp, mu_earth, maxiter, seed_epoch, mu_moon))]
    if include_sun:
        bodies.append((mu_sun, lambda ts: _sun_state_core(ts, xp, mu_sun, maxiter, seed_epoch)))

    out = []
    for mu_body, state_fn in bodies:
        r_u, _ = state_fn(t_since_seed)
        r_u = r_u.reshape(n_grid, uniq.shape[0], 3)
        ind_u = r_u / xp.linalg.norm(r_u, axis=-1)[..., None] ** 3
        out.append((mu_body, r_u, ind_u))
    return out, inverse


@dataclass
class SpacecraftBatchResult:
    r: np.ndarray  # (n_steps+1, N, 3) position history, km
    v: np.ndarray  # (n_steps+1, N, 3) velocity history, km/s
    t_abs: np.ndarray  # (n_steps+1, N) absolute epoch of each stored step, s since J2000


@dataclass
class ClosestApproachBatchResult:
    """``propagate_spacecraft_batch(..., output="closest_approach")`` result:
    O(N) instead of O(N x n_steps) -- only the running Moon closest approach
    over the stored RK4 steps (0..n_steps inclusive, same sample set the
    history mode stores) plus the final state.
    """

    min_distance_km: np.ndarray  # (N,) closest spacecraft-Moon distance over the steps, km
    min_distance_epoch_s: np.ndarray  # (N,) absolute epoch of that closest step, s since J2000
    min_distance_step: np.ndarray  # (N,) int, step index (0..n_steps) of that closest step
    r: np.ndarray  # (N, 3) final position, km
    v: np.ndarray  # (N, 3) final velocity, km/s
    t_abs: np.ndarray  # (N,) final absolute epoch, s since J2000


_OUTPUT_MODES = ("history", "closest_approach")


def propagate_spacecraft_batch(
    r0,
    v0,
    t_span,
    n_steps,
    mu_earth,
    mu_moon,
    include_sun=False,
    mu_sun=None,
    epoch0=0.0,
    use_gpu=None,
    kepler_maxiter=20,
    moon_seed_epoch_ephemeris_seconds=None,
    output="history",
):
    """Fixed-step RK4 propagation of N independent spacecraft under Earth
    point-mass gravity plus Moon (and optionally Sun) third-body
    perturbation, all N cases advanced with the SAME step COUNT for
    GPU-friendliness even when their total flight times differ: each case
    gets its own internal step size ``dt_i = t_span_i / n_steps`` so it
    still reaches its own ``t_span_i`` in exactly ``n_steps`` steps.

    ``r0``/``v0`` are (N, 3) km, km/s, Earth-centered, in the same
    ECLIPJ2000 inertial frame as ``get_moon_state0``/``get_sun_state0``
    (and as ``orbitopt.bodies.moon_state`` / ``orbitopt.verify.tudat_propagate``).
    ``t_span`` is (N,) or scalar, seconds. ``epoch0`` is the absolute time
    (seconds since J2000) each case's t=0 corresponds to -- (N,) or scalar,
    defaulting to 0.0 -- needed because the Moon/Sun states depend on
    absolute time, not just elapsed flight time, and different candidate
    departures generally launch at different epochs.

    ``moon_seed_epoch_ephemeris_seconds`` controls where the internal cheap
    Moon/Sun two-body model is seeded (see ``get_moon_state0``): by default
    (``None``) it auto-seeds at ``epoch0``'s own value (its first element,
    if an array -- in practice callers almost always pass the same scalar
    epoch0 for a whole batch of candidates sharing one departure date), so
    the Moon/Sun osculating ellipse is anchored near the actual mission
    epoch instead of always J2000. This matters a lot in practice: seeding
    at J2000 and asking for a 2026 mission's Moon position propagates that
    ellipse across 26+ years of pure two-body motion, which diverges by
    hundreds of thousands of km (empirically confirmed) -- not the ~thousand-km
    error the model is actually good for over a several-day mission when
    seeded near the epoch of interest. Pass an explicit value only if you
    deliberately want a different (e.g. shared/cached-elsewhere) seed.

    The Sun term (when enabled) reuses ``sun_state_batch``, i.e. it is
    itself a two-body Earth-Sun ellipse seeded from one real SPICE sample,
    not a static/frozen position -- accurate enough over a ~10-day mission
    that its own perturbation on the trajectory is already a small
    correction.

    ``kepler_maxiter`` controls the Newton iteration count of the internal
    Moon/Sun Kepler solves (see ``moon_state_batch``) -- it is separate from
    the spacecraft RK4 step count. The Moon/Sun positions at every RK4
    half-step epoch are precomputed once before integration (one batched
    Kepler solve per distinct (epoch0, dt) pair -- see
    ``_precompute_body_grid``), so this cost is paid once per call, not
    4x per RK4 step.

    ``output`` selects what is kept:

    - ``"history"`` (default): a ``SpacecraftBatchResult`` with the full
      (n_steps+1, N, ...) position/velocity/epoch history -- O(N x n_steps)
      device memory and GPU->CPU transfer (~449 MB for 20,000 candidates x
      400 steps).
    - ``"closest_approach"``: a ``ClosestApproachBatchResult`` holding only
      the running minimum spacecraft-Moon distance (and its epoch/step) plus
      the final state -- O(N) memory, for fitness evaluation that only needs
      the closest approach. The Moon position at each step is read from the
      same precomputed half-step grid the RK4 k1 stage uses (same internal
      Moon model, same ``mu_earth``/``mu_moon``/``kepler_maxiter``), so it
      costs no extra Kepler solves. Ties keep the earliest step
      and NaN distances propagate, matching ``np.min``/``np.argmin`` on the
      history.
    """
    if output not in _OUTPUT_MODES:
        raise ValueError(f"output must be one of {_OUTPUT_MODES}, got {output!r}")
    track_closest = output == "closest_approach"

    _where = "orbitopt.dynamics.nbody_gpu.propagate_spacecraft_batch"
    check_grav_parameter(mu_earth, units=UNITS, where=f"{_where}: mu_earth")
    check_grav_parameter(mu_moon, units=UNITS, where=f"{_where}: mu_moon")
    if mu_sun is not None:
        check_grav_parameter(mu_sun, units=UNITS, where=f"{_where}: mu_sun")
    _check_host_states(r0, v0, spec=NBODY_SCREENING, where=_where)
    NBODY_SCREENING.check_epochs(epoch0, where=f"{_where}: epoch0")

    xp = get_array_module(use_gpu)

    r0 = xp.asarray(r0, dtype=xp.float64)
    v0 = xp.asarray(v0, dtype=xp.float64)
    n = r0.shape[0]

    t_span = xp.broadcast_to(xp.atleast_1d(xp.asarray(t_span, dtype=xp.float64)), (n,))
    epoch0 = xp.broadcast_to(xp.atleast_1d(xp.asarray(epoch0, dtype=xp.float64)), (n,))
    dt = t_span / n_steps

    if moon_seed_epoch_ephemeris_seconds is None:
        moon_seed_epoch_ephemeris_seconds = float(to_numpy(epoch0)[0])

    if include_sun and mu_sun is None:
        mu_sun = MU_SUN_KM3_S2

    # Every RK4 sub-stage of step ``s`` samples the Moon/Sun at one of the
    # half-step epochs ``epoch0 + k*dt/2`` (k = 2s, 2s+1, 2s+2), and those
    # epochs depend only on each candidate's (epoch0, dt) pair -- never on
    # its spacecraft state. So instead of re-running the Kepler/Newton solve
    # inside every acceleration() call (4*n_steps solves, each sized N), the
    # whole (2*n_steps+1)-point half-step grid is solved ONCE up front, for
    # each *distinct* (epoch0, dt) pair only. In the common case (a batch of
    # candidates sharing one departure epoch and coast time) that is a single
    # solve of 2*n_steps+1 points, broadcast to all N candidates.
    ephem, grid_inverse = _precompute_body_grid(
        epoch0, dt, n_steps, xp, mu_earth, mu_moon, include_sun, mu_sun,
        kepler_maxiter, moon_seed_epoch_ephemeris_seconds,
    )
    # The grid stays compact, (2*n_steps+1, U, 3); each RK4 stage gathers
    # only its own (N, 3) time slice, so ephemeris memory is
    # O(U*n_steps + N) rather than O(N*n_steps) even when the batch mixes
    # flight times / epochs. With a single distinct pair (U == 1) the
    # (1, 3) slice simply broadcasts against the (N, 3) state.
    if ephem[0][1].shape[1] == 1:
        grid_inv = None
    else:
        grid_inv = xp.asarray(grid_inverse)

    def bodies_at(k):
        """[(mu, r_body, indirect)] at half-step index ``k``, each (N|1, 3)."""
        if grid_inv is None:
            return [(mu, rb[k], ind[k]) for mu, rb, ind in ephem]
        return [(mu, rb[k][grid_inv], ind[k][grid_inv]) for mu, rb, ind in ephem]

    def acceleration(r, bodies):
        rn = xp.linalg.norm(r, axis=-1)
        a = -mu_earth * r / rn[:, None] ** 3
        for mu_body, r_b, indirect in bodies:
            d = r_b - r
            dn = xp.linalg.norm(d, axis=-1)
            a = a + mu_body * (d / dn[:, None] ** 3 - indirect)
        return a

    r = r0
    v = v0
    t_abs = epoch0

    if track_closest:
        min_dist = xp.full((n,), xp.inf, dtype=xp.float64)
        min_epoch = xp.array(epoch0, dtype=xp.float64, copy=True)
        min_step = xp.zeros((n,), dtype=xp.int64)

        def update_closest(step, r, r_moon, t_abs):
            nonlocal min_dist, min_epoch, min_step
            dist = xp.linalg.norm(r - r_moon, axis=-1)
            # Strict "<" keeps the earliest step on ties (np.argmin semantics);
            # the NaN clause makes a NaN distance stick, like np.min/np.argmin.
            better = (dist < min_dist) | (xp.isnan(dist) & ~xp.isnan(min_dist))
            min_dist = xp.where(better, dist, min_dist)
            min_epoch = xp.where(better, t_abs, min_epoch)
            min_step = xp.where(better, step, min_step)
    else:
        r_hist = xp.empty((n_steps + 1, n, 3), dtype=xp.float64)
        v_hist = xp.empty((n_steps + 1, n, 3), dtype=xp.float64)
        t_hist = xp.empty((n_steps + 1, n), dtype=xp.float64)
        r_hist[0] = r
        v_hist[0] = v
        t_hist[0] = t_abs

    dt_col = dt[:, None]
    # Grid index 2*step is exactly the epoch of stored step ``step`` (same
    # accumulation as ``t_abs``); index 2s+2 serves both k4 of step s and k1
    # of step s+1, so each slice is gathered once and carried over.
    b_start = bodies_at(0)
    for step in range(n_steps):
        b_mid = bodies_at(2 * step + 1)
        b_end = bodies_at(2 * step + 2)

        k1r = v
        if track_closest:
            # k1 is evaluated at exactly (r, t_abs) of stored step ``step``,
            # so its grid Moon position (bodies[0] is always the Moon)
            # doubles as the closest-approach sample.
            update_closest(step, r, b_start[0][1], t_abs)
        k1v = acceleration(r, b_start)

        k2r = v + 0.5 * dt_col * k1v
        k2v = acceleration(r + 0.5 * dt_col * k1r, b_mid)

        k3r = v + 0.5 * dt_col * k2v
        k3v = acceleration(r + 0.5 * dt_col * k2r, b_mid)

        k4r = v + dt_col * k3v
        k4v = acceleration(r + dt_col * k3r, b_end)
        b_start = b_end

        r = r + (dt_col / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v = v + (dt_col / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t_abs = t_abs + dt

        if not track_closest:
            r_hist[step + 1] = r
            v_hist[step + 1] = v
            t_hist[step + 1] = t_abs

    if track_closest:
        update_closest(n_steps, r, b_start[0][1], t_abs)
        return ClosestApproachBatchResult(
            min_distance_km=to_numpy(min_dist),
            min_distance_epoch_s=to_numpy(min_epoch),
            min_distance_step=to_numpy(min_step),
            r=to_numpy(r),
            v=to_numpy(v),
            t_abs=to_numpy(t_abs),
        )

    return SpacecraftBatchResult(r=to_numpy(r_hist), v=to_numpy(v_hist), t_abs=to_numpy(t_hist))
