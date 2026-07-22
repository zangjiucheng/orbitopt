"""Export real planetary orbits and current positions (via pykep
ephemerides/orbital elements) as a generic "scene" JSON document -- see
orbitopt.viz.scene for the shared schema consumed by the general-purpose
3D viewer in viewer/ (React + three.js), not a bespoke per-scenario format.

Orbits are traced analytically from each planet's osculating elements
(semi-major axis, eccentricity, inclination, RAAN, argument of periapsis)
at the reference epoch, sweeping true anomaly through a full revolution --
not by sampling real ephemeris time across one full period, since outer
planets' periods (Neptune: ~165 years, Pluto: ~248 years) run well outside
pykep's low-precision ephemeris valid range (1800-2050) from most reference
epochs. Current positions still come from the real eph() call, which is
always evaluated at a single, in-range epoch. Positions are exported in
true 3D (the viewer renders real inclinations, e.g. Pluto's ~17 degrees --
earlier iterations of this module flattened to the ecliptic x-y plane for a
2D canvas view; that projection is gone now that rendering is genuinely 3D).
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import body_fixed_to_eclipj2000_matrix, planet
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, ROTATION_PERIOD_HOURS, TEXTURE, body_entry, scene_document

_JPL_LP_BODIES = (
    "mercury", "venus", "earth", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
)

AU_M = pk.AU

# Real sphere radius (AU) for the textured-sphere render path, proportional
# to (not a physically exact conversion of) the existing RADIUS_DISPLAY
# scale -- true physical radii here (Sun ~0.00465 AU, Earth ~0.0000426 AU)
# would be sub-pixel at whole-solar-system zoom next to orbits spanning
# 0.4-39 AU. This factor keeps the Sun a clearly visible ~0.08 AU sphere
# while staying well inside Mercury's own 0.39 AU orbit (a past bug in an
# earlier hand-rolled renderer had the Sun's marker swallow Mercury's orbit
# entirely -- this is the same failure mode, avoided the same way: keep the
# body small relative to its neighbors' orbital scale, not to-scale with it).
_AU_RADIUS_SCALE = 0.008


def _au_radius(body_id: str) -> float:
    return RADIUS_DISPLAY.get(body_id, 4.0) * _AU_RADIUS_SCALE


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
    z = x_pf * (sin_i * sin_w) + y_pf * (sin_i * cos_w)

    return np.stack([x, y, z], axis=-1) / AU_M


def export_solar_system_data(epoch_mjd2000=None, n_samples=240):
    """Return a SceneData document (see viz.scene) for the whole solar
    system at ``epoch_mjd2000`` (defaults to 2026-07-15).
    """
    if epoch_mjd2000 is None:
        epoch_mjd2000 = pk.epoch_from_string("2026-07-15 00:00:00").mjd2000
    ep = pk.epoch(epoch_mjd2000)

    bodies = [
        body_entry(
            "sun", "Sun", COLOR["sun"], "star",
            radius_display=RADIUS_DISPLAY["sun"], position=[0.0, 0.0, 0.0],
            texture=TEXTURE["sun"], radius=_au_radius("sun"),
            rotation_period_hours=ROTATION_PERIOD_HOURS["sun"],
            orientation_basis=body_fixed_to_eclipj2000_matrix("Sun", epoch_mjd2000).tolist(),
        )
    ]

    for name in _JPL_LP_BODIES:
        body = planet(name)
        a, e, i, raan, arg_peri, _mean_anomaly = body.osculating_elements(ep)
        trace = _orbit_trace_from_elements(a, e, i, raan, arg_peri, n_samples)

        r_now, v_now = body.eph(ep)
        distance_au = float(np.linalg.norm(r_now)) / AU_M
        period_days = float(body.compute_period(ep)) / 86400.0
        speed_km_s = float(np.linalg.norm(v_now)) / 1000.0

        bodies.append(body_entry(
            name, name.capitalize(), COLOR[name],
            "dwarf-planet" if name == "pluto" else "planet",
            radius_display=RADIUS_DISPLAY.get(name, 4.0),
            orbit=trace.round(6).tolist(),
            orbit_dashed=(name == "pluto"),
            position=[round(r_now[0] / AU_M, 6), round(r_now[1] / AU_M, 6), round(r_now[2] / AU_M, 6)],
            info=[
                {"label": "Distance from Sun", "value": f"{distance_au:.3f} AU"},
                {"label": "Orbital period", "value": _format_period(period_days)},
                {"label": "Heliocentric speed", "value": f"{speed_km_s:.2f} km/s"},
                {"label": "Eccentricity", "value": f"{e:.3f}"},
                {"label": "Inclination", "value": f"{np.degrees(i):.2f}°"},
            ],
            texture=TEXTURE.get(name), radius=_au_radius(name),
            rotation_period_hours=ROTATION_PERIOD_HOURS.get(name),
            # Real axis/tilt only matters for the textured (real-sphere) bodies;
            # Pluto renders as a point marker (no texture), so skip it there.
            orientation_basis=(
                body_fixed_to_eclipj2000_matrix(name, epoch_mjd2000).tolist()
                if TEXTURE.get(name) else None
            ),
        ))

    return scene_document(
        scene_id="solar-system",
        title="Solar System",
        subtitle=f"Heliocentric positions for {pk.epoch(epoch_mjd2000)!s}",
        distance_unit="AU",
        central_body_id="sun",
        bodies=bodies,
    )


def _format_period(days):
    if days < 500:
        return f"{days:.1f} days"
    return f"{days / 365.25:.2f} years"
