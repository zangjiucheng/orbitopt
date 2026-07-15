"""Export real planetary orbit traces and current positions (via pykep
ephemerides/orbital elements) as plain JSON-serializable data, for
rendering in an interactive (JS/Canvas) solar-system viewer.

Orbits are traced analytically from each planet's osculating elements
(semi-major axis, eccentricity, inclination, RAAN, argument of periapsis)
at the reference epoch, sweeping true anomaly through a full revolution --
not by sampling real ephemeris time across one full period, since outer
planets' periods (Neptune: ~165 years, Pluto: ~248 years) run well outside
pykep's low-precision ephemeris valid range (1800-2050) from most reference
epochs. Current positions still come from the real eph() call, which is
always evaluated at a single, in-range epoch.
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import planet

_JPL_LP_BODIES = (
    "mercury", "venus", "earth", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
)

AU_M = pk.AU


def _eccentric_anomaly(mean_anomaly, e, maxiter=50, tol=1e-12):
    E = mean_anomaly.copy()
    for _ in range(maxiter):
        delta = (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))
        E = E - delta
        if np.max(np.abs(delta)) < tol:
            break
    return E


def _orbit_trace_from_elements(a, e, i, raan, arg_peri, n_samples):
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    E = _eccentric_anomaly(mean_anomaly, e)
    true_anomaly = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0))
    r = a * (1.0 - e * np.cos(E))

    x_pf = r * np.cos(true_anomaly)
    y_pf = r * np.sin(true_anomaly)

    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(arg_peri), np.sin(arg_peri)
    cos_i, sin_i = np.cos(i), np.sin(i)

    x = x_pf * (cos_O * cos_w - sin_O * sin_w * cos_i) - y_pf * (cos_O * sin_w + sin_O * cos_w * cos_i)
    y = x_pf * (sin_O * cos_w + cos_O * sin_w * cos_i) - y_pf * (sin_O * sin_w - cos_O * cos_w * cos_i)

    return np.stack([x, y], axis=-1) / AU_M


def export_solar_system_data(epoch_mjd2000=None, n_samples=240):
    """Return a JSON-serializable dict: reference epoch, and per-body name,
    orbit trace (list of [x, y] in AU, ecliptic J2000 projected onto the
    x-y plane), current position (AU), distance from Sun (AU), orbital
    period (days), and heliocentric speed (km/s).
    """
    if epoch_mjd2000 is None:
        epoch_mjd2000 = pk.epoch_from_string("2026-07-15 00:00:00").mjd2000
    ep = pk.epoch(epoch_mjd2000)

    bodies = {}
    for name in _JPL_LP_BODIES:
        body = planet(name)
        a, e, i, raan, arg_peri, _mean_anomaly = body.osculating_elements(ep)
        trace = _orbit_trace_from_elements(a, e, i, raan, arg_peri, n_samples)

        r_now, v_now = body.eph(ep)
        bodies[name] = {
            "orbit_au": trace.round(6).tolist(),
            "position_au": [round(r_now[0] / AU_M, 6), round(r_now[1] / AU_M, 6)],
            "distance_au": round(float(np.linalg.norm(r_now)) / AU_M, 4),
            "period_days": round(float(body.compute_period(ep)) / 86400.0, 2),
            "speed_km_s": round(float(np.linalg.norm(v_now)) / 1000.0, 3),
            "eccentricity": round(float(e), 4),
            "inclination_deg": round(float(np.degrees(i)), 3),
        }

    return {
        "epoch_mjd2000": epoch_mjd2000,
        "bodies": bodies,
    }
