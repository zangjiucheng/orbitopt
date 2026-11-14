"""Two-body coast arc: propagate a Cartesian parking-orbit state forward
with classical Kepler propagation (stdlib only, no compute stack).

:func:`coast_arc` exists so a scene exporter can draw what happens AFTER
the last flown sample -- a parking orbit keeps being an orbit -- without
pulling in tudatpy. It is display-quality propagation, not verification:
the Gate-B re-fly (``orbit_refly``, tudatpy) remains the authority on
whether the insertion orbit actually holds.

Elliptical orbits only: a state whose two-body orbit is not closed (a <= 0
or e >= 1) raises -- there is no parking orbit to coast along.

:func:`j2_coast` is the short-gap companion: two-body gravity plus the J2
zonal term about a given pole, integrated with fixed-step RK4. It bridges
the second or two between a parent stage's last sample and its child's
first one, where the ascent backend (OpenTsiolkovsky) itself flies J2
gravity -- a Kepler bridge over that gap leaves ~0.03 m/s of J2 in the
staging residual. Gravity only: drag over the gap is not modelled.
"""
from __future__ import annotations

import math
from typing import Sequence

EARTH_MU = 3.986004418e14  # m^3/s^2, same constant as orbitopt.flightlab.checks
EARTH_J2 = 1.08262668e-3  # same constants as orbitopt.entry
EARTH_R_EQ_M = 6378137.0

__all__ = ["coast_arc", "j2_coast"]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def coast_arc(r0: Sequence[float], v0: Sequence[float], duration_s: float,
               samples: int = 121, mu: float = EARTH_MU) -> tuple[list, list, list]:
    """Propagate the inertial state ``(r0, v0)`` (m, m/s) along its
    two-body ellipse for ``duration_s`` seconds.

    Returns ``(times_s, positions_m, velocities_m_s)`` -- ``samples``
    uniformly spaced rows starting at 0 s (row 0 is the input state, up to
    float rounding). Raises ``ValueError`` on a non-elliptical orbit, a
    non-positive duration or fewer than two samples.
    """
    r0 = [float(x) for x in r0]
    v0 = [float(x) for x in v0]
    if len(r0) != 3 or len(v0) != 3:
        raise ValueError("coast_arc needs 3-vector position and velocity")
    if not (duration_s > 0.0) or samples < 2:
        raise ValueError(f"coast_arc needs duration_s > 0 and samples >= 2 "
                         f"(got {duration_s!r}, {samples!r})")

    r_mag = _norm(r0)
    v2 = _dot(v0, v0)
    h = _cross(r0, v0)
    h_mag = _norm(h)
    if r_mag <= 0.0 or h_mag <= 0.0:
        raise ValueError("coast_arc needs a closed (elliptical) two-body orbit")
    denom = 2.0 / r_mag - v2 / mu  # vis-viva: <= 0 means parabola or hyperbola
    if denom <= 0.0:
        raise ValueError("coast_arc needs a closed (elliptical) two-body orbit")
    a = 1.0 / denom
    rdotv = _dot(r0, v0)
    e_vec = [(v2 / mu - 1.0 / r_mag) * r0[i] - rdotv / mu * v0[i] for i in range(3)]
    e = _norm(e_vec)
    if not (0.0 <= e < 1.0):
        raise ValueError(f"coast_arc: eccentricity {e:g} is not elliptical")

    # Perifocal frame: p_hat along the eccentricity vector (orbit-frame node
    # line when e ~ 0), q_hat = h_hat x p_hat closes the in-plane pair.
    h_hat = [x / h_mag for x in h]
    if e > 1e-9:
        p_hat = [x / e for x in e_vec]
    else:
        r_perp = [r0[i] - _dot(r0, h_hat) * h_hat[i] for i in range(3)]
        n_perp = _norm(r_perp)
        if n_perp <= 0.0:  # radial velocity along h is not an orbit
            raise ValueError("coast_arc: degenerate circular state")
        p_hat = [x / n_perp for x in r_perp]
    q_hat = _cross(h_hat, p_hat)

    b = a * math.sqrt(1.0 - e * e)  # semi-minor axis
    r_p = _dot(r0, p_hat)
    r_q = _dot(r0, q_hat)
    e0 = math.atan2(r_q / b, r_p / a + e)  # eccentric anomaly at t=0
    n = math.sqrt(mu / (a * a * a))  # mean motion
    m0 = e0 - e * math.sin(e0)

    times, positions, velocities = [], [], []
    for k in range(samples):
        m = m0 + n * (duration_s * k / (samples - 1))
        ea = m + e * math.sin(m)  # Kepler, Newton iteration
        for _ in range(50):
            f = ea - e * math.sin(ea) - m
            fp = 1.0 - e * math.cos(ea)
            step = f / fp if fp else 0.0
            ea -= step
            if abs(step) < 1e-12:
                break
        cos_e, sin_e = math.cos(ea), math.sin(ea)
        rr = a * (1.0 - e * cos_e)
        rp, rq = a * (cos_e - e), b * sin_e
        vp, vq = -a * n * sin_e * a / rr, b * n * cos_e * a / rr
        times.append(duration_s * k / (samples - 1))
        positions.append([rp * p_hat[i] + rq * q_hat[i] for i in range(3)])
        velocities.append([vp * p_hat[i] + vq * q_hat[i] for i in range(3)])
    return times, positions, velocities


def j2_coast(r0: Sequence[float], v0: Sequence[float], duration_s: float, *,
             pole: Sequence[float] = (0.0, 0.0, 1.0), mu: float = EARTH_MU,
             j2: float = EARTH_J2, r_eq_m: float = EARTH_R_EQ_M,
             max_step_s: float = 0.25) -> tuple[list[float], list[float]]:
    """Propagate the inertial state ``(r0, v0)`` (m, m/s) for ``duration_s``
    seconds under two-body + J2 gravity, J2 symmetric about the unit vector
    ``pole`` (Earth's rotation axis in the state's frame; the default is the
    frame's z axis). Fixed-step RK4 with steps no longer than
    ``max_step_s``: meant for gaps of seconds, where its error is far below
    a millimetre per second. Unlike :func:`coast_arc` it needs no closed
    orbit, so a suborbital state can be coasted too.

    Returns ``(r, v)`` at ``duration_s``. Raises ``ValueError`` on a
    non-positive duration, a zero pole or a state inside the Earth's centre.
    """
    r = [float(x) for x in r0]
    v = [float(x) for x in v0]
    if len(r) != 3 or len(v) != 3 or len(pole) != 3:
        raise ValueError("j2_coast needs 3-vector position, velocity and pole")
    if not (duration_s > 0.0) or not (max_step_s > 0.0):
        raise ValueError(f"j2_coast needs duration_s > 0 (got {duration_s!r})")
    pn = _norm(pole)
    if not (pn > 0.0):
        raise ValueError("j2_coast needs a non-zero pole vector")
    k_hat = [float(x) / pn for x in pole]
    if not (_norm(r) > 0.0):
        raise ValueError("j2_coast needs a state away from the Earth's centre")

    def accel(rv: list[float]) -> list[float]:
        rn2 = _dot(rv, rv)
        rn = math.sqrt(rn2)
        z = _dot(rv, k_hat)
        a_kepler = -mu / (rn2 * rn)
        c = 1.5 * j2 * mu * r_eq_m * r_eq_m / (rn2 * rn2 * rn)
        zz = 5.0 * z * z / rn2
        return [a_kepler * rv[i] + c * ((zz - 1.0) * rv[i] - 2.0 * z * k_hat[i])
                for i in range(3)]

    n = max(1, math.ceil(duration_s / max_step_s))
    h = duration_s / n
    for _ in range(n):
        k1v = accel(r)
        k1r = v
        r2 = [r[i] + 0.5 * h * k1r[i] for i in range(3)]
        k2r = [v[i] + 0.5 * h * k1v[i] for i in range(3)]
        k2v = accel(r2)
        r3 = [r[i] + 0.5 * h * k2r[i] for i in range(3)]
        k3r = [v[i] + 0.5 * h * k2v[i] for i in range(3)]
        k3v = accel(r3)
        r4 = [r[i] + h * k3r[i] for i in range(3)]
        k4r = [v[i] + h * k3v[i] for i in range(3)]
        k4v = accel(r4)
        r = [r[i] + h / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]) for i in range(3)]
        v = [v[i] + h / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]) for i in range(3)]
    return r, v
