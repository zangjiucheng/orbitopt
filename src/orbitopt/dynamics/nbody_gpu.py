"""GPU-batchable, coarse Earth-Moon(-Sun) trajectory propagation for screening
translunar-injection candidates.

This is deliberately NOT navigation-grade: it exists to evaluate thousands of
candidate TLI burns (varying epoch/direction/magnitude) in one batched pass
so that only the handful of promising survivors get handed to tudatpy's
full-fidelity SPICE-based numerical propagator for refinement. Two pieces
make this possible on GPU. First, a batched universal-variable Kepler
propagator (Vallado's formulation via Curtis's f-and-g functions, Stumpff
series) advances N independent two-body states with a fixed Newton
iteration count and no data-dependent branching. Second, that same
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

MU_EARTH_KM3_S2 = 398600.4418
MU_MOON_KM3_S2 = 4902.8
MU_SUN_KM3_S2 = 1.32712440018e11

_SPICE_LOADED = False
_MOON_STATE0_KM = {}  # keyed by seed_epoch_ephemeris_seconds (rounded)
_SUN_STATE0_KM = {}


def _ensure_spice_loaded():
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        from tudatpy.interface import spice

        spice.load_standard_kernels()
        _SPICE_LOADED = True


def _stumpff(z, xp):
    """Stumpff functions C(z), S(z), vectorized and branchless.

    Same trick as orbitopt.lambert.gpu_batch._stumpff: xp.where evaluates
    every branch eagerly, so each branch's denominator is pre-guarded
    against division by zero (and cosh/sinh overflow) before selection.
    """
    pos = z > 1e-6
    neg = z < -1e-6

    z_pos = xp.where(pos, z, 1.0)
    sz_pos = xp.sqrt(z_pos)
    c_pos = (1.0 - xp.cos(sz_pos)) / z_pos
    s_pos = (sz_pos - xp.sin(sz_pos)) / z_pos**1.5

    z_neg = xp.where(neg, -z, 1.0)
    sz_neg = xp.sqrt(z_neg)
    c_neg = (1.0 - xp.cosh(sz_neg)) / (-z_neg)
    s_neg = (xp.sinh(sz_neg) - sz_neg) / z_neg**1.5

    c_small = 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0
    s_small = 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0

    C = xp.where(pos, c_pos, xp.where(neg, c_neg, c_small))
    S = xp.where(pos, s_pos, xp.where(neg, s_neg, s_small))
    return C, S


def _kepler_time_residual(chi, r0n, vr0, alpha, dt, sqrt_mu, xp):
    z = alpha * chi**2
    C, S = _stumpff(z, xp)
    return (
        (r0n * vr0 / sqrt_mu) * chi**2 * C
        + (1.0 - alpha * r0n) * chi**3 * S
        + r0n * chi
        - sqrt_mu * dt
    )


def _propagate_kepler_core(r0, v0, dt, mu, xp, maxiter):
    """Array-native universal-variable Kepler propagation, operating on
    whatever xp arrays it is handed and returning xp arrays (no host
    round-trip). Split out from ``propagate_kepler_batch`` so that
    ``propagate_spacecraft_batch``'s per-RK4-substage Moon/Sun lookups can
    stay resident on the GPU instead of paying a device<->host copy at
    every one of the 4*n_steps calls -- that round trip, not the Newton
    iteration itself, was the dominant GPU cost before this split existed.
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

    chi = sqrt_mu * xp.abs(alpha) * dt
    for _ in range(maxiter):
        F = _kepler_time_residual(chi, r0n, vr0, alpha, dt, sqrt_mu, xp)
        h = 1e-4 * xp.maximum(xp.abs(chi), 1.0)
        F_plus = _kepler_time_residual(chi + h, r0n, vr0, alpha, dt, sqrt_mu, xp)
        F_minus = _kepler_time_residual(chi - h, r0n, vr0, alpha, dt, sqrt_mu, xp)
        dFdchi = (F_plus - F_minus) / (2.0 * h)
        dFdchi = xp.where(xp.abs(dFdchi) < 1e-12, 1e-12, dFdchi)
        chi = chi - F / dFdchi

    z = alpha * chi**2
    C, S = _stumpff(z, xp)

    f = 1.0 - (chi**2 / r0n) * C
    g = dt - (chi**3 / sqrt_mu) * S
    r = f[:, None] * r0 + g[:, None] * v0
    rn = xp.linalg.norm(r, axis=-1)

    fdot = (sqrt_mu / (rn * r0n)) * (alpha * chi**3 * S - chi)
    gdot = 1.0 - (chi**2 / rn) * C
    v = fdot[:, None] * r0 + gdot[:, None] * v0

    return r, v


def propagate_kepler_batch(r0, v0, dt, mu, use_gpu=None, maxiter=50):
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

    The universal anomaly's Newton iteration is solved with a fixed
    iteration count and a central-difference derivative (same style as
    ``orbitopt.lambert.gpu_batch.solve_lambert_batch``) rather than the
    analytic f'/F' derivative, trading a small constant-factor slowdown for
    one fewer place to get a sign wrong.

    Returns
    -------
    (r, v) : tuple of (N, 3) numpy arrays, position and velocity at t0+dt.
    """
    xp = get_array_module(use_gpu)
    r, v = _propagate_kepler_core(r0, v0, dt, mu, xp, maxiter)
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


def _moon_state_core(dt, xp, mu_earth, maxiter, seed_epoch_ephemeris_seconds=0.0):
    r0, v0 = get_moon_state0(seed_epoch_ephemeris_seconds)
    return _propagate_kepler_core(xp.asarray(r0), xp.asarray(v0), dt, mu_earth, xp, maxiter)


def _sun_state_core(dt, xp, mu_sun, maxiter, seed_epoch_ephemeris_seconds=0.0):
    r0, v0 = get_sun_state0(seed_epoch_ephemeris_seconds)
    return _propagate_kepler_core(xp.asarray(r0), xp.asarray(v0), dt, mu_sun, xp, maxiter)


def moon_state_batch(dt, use_gpu=None, mu_earth=MU_EARTH_KM3_S2, maxiter=20, seed_epoch_ephemeris_seconds=0.0):
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
    r, v = _moon_state_core(dt, xp, mu_earth, maxiter, seed_epoch_ephemeris_seconds)
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


@dataclass
class SpacecraftBatchResult:
    r: np.ndarray  # (n_steps+1, N, 3) position history, km
    v: np.ndarray  # (n_steps+1, N, 3) velocity history, km/s
    t_abs: np.ndarray  # (n_steps+1, N) absolute epoch of each stored step, s since J2000


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
    the spacecraft RK4 step count and defaults low because it is paid 4x
    per RK4 step (once per stage) and each iteration is its own GPU kernel
    launch, so its cost is launch-count- rather than array-size-bound.
    """
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

    def acceleration(r, t_abs):
        rn = xp.linalg.norm(r, axis=-1)
        a = -mu_earth * r / rn[:, None] ** 3

        t_since_seed = t_abs - moon_seed_epoch_ephemeris_seconds
        r_moon, _ = _moon_state_core(t_since_seed, xp, mu_earth, kepler_maxiter, moon_seed_epoch_ephemeris_seconds)
        d = r_moon - r
        dn = xp.linalg.norm(d, axis=-1)
        rmoon_n = xp.linalg.norm(r_moon, axis=-1)
        a = a + mu_moon * (d / dn[:, None] ** 3 - r_moon / rmoon_n[:, None] ** 3)

        if include_sun:
            r_sun, _ = _sun_state_core(t_since_seed, xp, mu_sun, kepler_maxiter, moon_seed_epoch_ephemeris_seconds)
            d_s = r_sun - r
            dsn = xp.linalg.norm(d_s, axis=-1)
            rsun_n = xp.linalg.norm(r_sun, axis=-1)
            a = a + mu_sun * (d_s / dsn[:, None] ** 3 - r_sun / rsun_n[:, None] ** 3)

        return a

    r_hist = xp.empty((n_steps + 1, n, 3), dtype=xp.float64)
    v_hist = xp.empty((n_steps + 1, n, 3), dtype=xp.float64)
    t_hist = xp.empty((n_steps + 1, n), dtype=xp.float64)

    r = r0
    v = v0
    t_abs = epoch0
    r_hist[0] = r
    v_hist[0] = v
    t_hist[0] = t_abs

    dt_col = dt[:, None]
    for step in range(n_steps):
        k1r = v
        k1v = acceleration(r, t_abs)

        k2r = v + 0.5 * dt_col * k1v
        k2v = acceleration(r + 0.5 * dt_col * k1r, t_abs + 0.5 * dt)

        k3r = v + 0.5 * dt_col * k2v
        k3v = acceleration(r + 0.5 * dt_col * k2r, t_abs + 0.5 * dt)

        k4r = v + dt_col * k3v
        k4v = acceleration(r + dt_col * k3r, t_abs + dt)

        r = r + (dt_col / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v = v + (dt_col / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t_abs = t_abs + dt

        r_hist[step + 1] = r
        v_hist[step + 1] = v
        t_hist[step + 1] = t_abs

    return SpacecraftBatchResult(r=to_numpy(r_hist), v=to_numpy(v_hist), t_abs=to_numpy(t_hist))
