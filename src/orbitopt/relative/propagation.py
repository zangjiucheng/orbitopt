"""Nonlinear two-spacecraft propagation: the acceptance model behind the CW envelope (E12-02).

:mod:`orbitopt.relative.cw` is the fast planning model; this module is what
its answers get checked against. Two spacecraft -- a chief and a deputy --
are propagated together under the nonlinear two-body equations about Earth,
optionally with the J2 oblateness term, with a fixed-step classical RK4
integrator vectorised over batches of pairs. Relative states along the
trajectory are reported in the chief's LVLH frame
(:mod:`orbitopt.relative.frames`), which is where mission questions ("how
fast is the corridor closing?") and the CW comparison are both asked.

Why fixed-step RK4 rather than an adaptive integrator or a universal-variable
Kepler solver: with J2 the pair is no longer Keplerian, and the comparison
that matters (CW vs nonlinear) wants *one* integrator applying the same
discipline to both models -- the baseline's own accuracy is then a single
step-size number, and the step-refinement test in
tests/test_e12_02_relative_dynamics.py pins the observed fourth-order
convergence so the benchmark's error budget belongs to CW, not to the
integrator. ``sub_dt`` is the knob that tightens it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.physics_io import (
    MAX_DURATION_S,
    RELATIVE_PAIR,
    UnitSystem,
    UnitMismatchError,
    check_frame,
    check_grav_parameter,
)
from orbitopt.relative.frames import eci_to_lvlh

M_S = UnitSystem("m", "s")

#: Earth gravity constants (WGS84 / EGM96 values the E12 envelopes use).
EARTH_MU = 3.9860044150e14   # m^3/s^2
EARTH_J2 = 1.08262668e-3     # dimensionless
EARTH_RE = 6378137.0         # m, equatorial


@dataclass(frozen=True)
class PairTrajectory:
    """One integrated chief/deputy pair (or batch of pairs).

    ``times`` is the requested sample grid [s since the initial state].
    ``chief_r``/``chief_v`` and ``deputy_r``/``deputy_v`` are absolute
    Earth-centred states at those times, shape ``(N, M, 3)`` (m, m/s), in
    the frame the call named. ``rel_lvlh`` is the deputy-relative-to-chief
    LVLH state at the same times, shape ``(N, M, 6)`` (rho then rho_dot,
    m, m/s).
    """

    times: np.ndarray
    chief_r: np.ndarray
    chief_v: np.ndarray
    deputy_r: np.ndarray
    deputy_v: np.ndarray
    rel_lvlh: np.ndarray
    frame: str


def _accel(r: np.ndarray, mu: float, j2: float) -> np.ndarray:
    """Two-body (plus optional J2) acceleration [m/s^2] for positions ``r``
    (...,3) in an *equatorial* geocentric frame. J2 is defined about the
    equator: callers must not hand it an oblique frame (RELATIVE_PAIR's
    vocabulary is exactly the two equatorial inertial labels for this
    reason)."""
    r2 = np.einsum("...i,...i->...", r, r)
    rn = np.sqrt(r2)
    a = (-mu / (r2 * rn))[..., None] * r
    if j2:
        # a = -grad U_J2: components -C*x*(1-5z^2/r^2) etc with C = 3*J2*mu*Re^2/(2*r^5)
        c = 1.5 * j2 * mu * EARTH_RE ** 2 / (r2 * r2 * rn)
        q = 5.0 * r[..., 2:3] ** 2 / r2[..., None]
        a = a + c[..., None] * r * np.concatenate((q - 1.0, q - 1.0, q - 3.0), axis=-1)
    return a


def propagate_pair(
    r0_chief,
    v0_chief,
    r0_deputy,
    v0_deputy,
    times,
    *,
    frame: str,
    mu: float = EARTH_MU,
    j2: float = 0.0,
    sub_dt: float | None = None,
    where: str = "",
) -> PairTrajectory:
    """Propagate a chief/deputy pair (or an ``(N,...)`` batch, broadcast)
    nonlinearly from an initial absolute state at t=0 over the sample grid
    ``times`` [s, increasing; ``times[0]`` may be > 0 -- the states given
    are always the t=0 states].

    ``frame`` is the inertial frame of every absolute state and must be one
    of :data:`orbitopt.physics_io.RELATIVE_PAIR`'s two equatorial labels:
    J2 needs the equator, and one vocabulary for the whole call removes the
    chance of mixing frames between the two spacecraft. ``mu`` is the
    central-body parameter in m^3/s^2 (unit-cross-checked), ``j2`` the zonal
    coefficient (default 0 = pure two-body). ``sub_dt`` caps the integrator
    step [s]; the default subdivides each sample interval using the
    initial chief period, targeting ~500 steps/orbit -- tighten it for
    tight error budgets.

    Returns :class:`PairTrajectory`. Position/velocity band checks are the
    RELATIVE_PAIR spec's, applied to both spacecraft states; a separation
    in kilometres reaching this metres interface lands in the band error,
    not in a silently wrong rendezvous corridor.
    """
    where = where or "propagate_pair"
    check_frame(frame, RELATIVE_PAIR.frames, where=where)
    mu_sys = check_grav_parameter(mu, where=where + ": mu", units=M_S)
    if mu_sys is not None and mu_sys != M_S:  # pragma: no cover - check_grav_parameter raises first
        raise UnitMismatchError(f"{where}: mu must be in m^3/s^2")
    j2 = float(j2)
    if not (math.isfinite(j2) and 0.0 <= j2 < 1.0e-1):
        raise ValueError(f"{where}: j2 must be finite in [0, 0.1), got {j2!r}")
    mu = float(mu)

    rc0 = RELATIVE_PAIR.check_positions(r0_chief, where=where + ": chief")
    vc0 = RELATIVE_PAIR.check_velocities(v0_chief, where=where + ": chief")
    rd0 = RELATIVE_PAIR.check_positions(r0_deputy, where=where + ": deputy")
    vd0 = RELATIVE_PAIR.check_velocities(v0_deputy, where=where + ": deputy")
    rc0, vc0, rd0, vd0 = np.broadcast_arrays(rc0, vc0, rd0, vd0)
    if rc0.ndim == 1:  # one pair, reported as N=1
        rc0, vc0, rd0, vd0 = (a[None, ...] for a in (rc0, vc0, rd0, vd0))
    pair_shape = rc0.shape[:-1]

    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"{where}: times must be a 1-D increasing array of at least 2 samples")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{where}: times must be finite")
    if t[0] < 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{where}: times must be non-negative and strictly increasing")
    if t[-1] > MAX_DURATION_S:
        raise UnitMismatchError(
            f"{where}: total propagation {t[-1]:.0f} s exceeds the {MAX_DURATION_S:.0e} s bound"
        )

    # batch over pairs: y (K,12) with stacked chief/deputy position+velocity
    y = np.concatenate((rc0.reshape(-1, 3), rd0.reshape(-1, 3),
                        vc0.reshape(-1, 3), vd0.reshape(-1, 3)), axis=1)
    k = y.shape[0]

    def rhs(y_):
        acc_c = _accel(y_[:, 0:3], mu, j2)
        acc_d = _accel(y_[:, 3:6], mu, j2)
        return np.concatenate((y_[:, 6:9], y_[:, 9:12], acc_c, acc_d), axis=1)

    if sub_dt is not None:
        sub_dt = float(sub_dt)
        if not (math.isfinite(sub_dt) and sub_dt > 0.0):
            raise ValueError(f"{where}: sub_dt must be finite and > 0, got {sub_dt!r}")
    else:
        r_mean = float(np.mean(np.linalg.norm(y[:, 0:3], axis=1)))
        period = 2.0 * math.pi * math.sqrt(max(r_mean, 1.0) ** 3 / mu)
        sub_dt = period / 500.0

    out_c = np.empty((k, t.size, 3), dtype=np.float64)
    out_cd = np.empty_like(out_c)
    out_d = np.empty_like(out_c)
    out_dd = np.empty_like(out_c)
    t_now = 0.0

    def advance(span: float):
        nonlocal y, t_now
        n_sub = max(1, int(math.ceil(span / sub_dt)))
        h = span / n_sub
        for _ in range(n_sub):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_now += span

    def store(m: int):
        out_c[:, m] = y[:, 0:3]
        out_d[:, m] = y[:, 3:6]
        out_cd[:, m] = y[:, 6:9]
        out_dd[:, m] = y[:, 9:12]

    advance(float(t[0]))
    store(0)
    for m in range(1, t.size):
        advance(float(t[m]) - t_now)
        store(m)

    shape = pair_shape + (t.size, 3)
    rc = out_c.reshape(shape)
    vc = out_cd.reshape(shape)
    rd = out_d.reshape(shape)
    vd = out_dd.reshape(shape)
    rho, rho_dot = eci_to_lvlh(rc, vc, rd - rc, vd - vc, frame, where=where + ": rel_lvlh")
    return PairTrajectory(
        times=t.copy(),
        chief_r=rc,
        chief_v=vc,
        deputy_r=rd,
        deputy_v=vd,
        rel_lvlh=np.concatenate((rho, rho_dot), axis=-1),
        frame=frame,
    )
