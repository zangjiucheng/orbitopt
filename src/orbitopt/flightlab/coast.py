"""Two-body coast arc: propagate a Cartesian parking-orbit state forward
with classical Kepler propagation (stdlib only, no compute stack).

:func:`coast_arc` exists so a scene exporter can draw what happens AFTER
the last flown sample -- a parking orbit keeps being an orbit -- without
pulling in tudatpy. It is display-quality propagation, not verification:
the Gate-B re-fly (``orbit_refly``, tudatpy) remains the authority on
whether the insertion orbit actually holds.

Elliptical orbits only: a state whose two-body orbit is not closed (a <= 0
or e >= 1) raises -- there is no parking orbit to coast along.
"""
from __future__ import annotations

import math
from typing import Sequence

EARTH_MU = 3.986004418e14  # m^3/s^2, same constant as orbitopt.flightlab.checks

__all__ = ["coast_arc"]


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
