"""Export a time-series dataset of the Artemis-II-like free-return mission
(see orbitopt.verify.differential_correction / examples/05_artemis2_free_return.py)
for an interactive, scrubbable timeline visualization -- as opposed to
viz.solar_system, which is a static heliocentric snapshot.

Positions are projected onto the trajectory's own orbital plane (the plane
containing the initial post-TLI-burn position and velocity), not the
ecliptic: the parking orbit is deliberately plane-aligned toward the Moon's
arrival direction (see the module-level docstring in
verify/differential_correction.py's example usage), so this plane captures
essentially all of the spacecraft's spatial extent. The Moon's true position
is projected onto the same plane for a consistent 2D picture, but its
reported *distance* from Earth in the exported data is always the true 3D
distance, not a distance measured within the projection -- the projection is
a display simplification, the physics underneath it is not.
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, moon_state
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify.differential_correction import target_lunar_flyby
from orbitopt.verify.tudat_propagate import body_position_at_absolute_epoch, propagate_multi_arc

EARTH_RADIUS_KM = 6378.0
MOON_RADIUS_KM = 1737.4
ARTEMIS_II_PERILUNE_ALTITUDE_KM = 6545.0


def _plane_aligned_parking_orbit(r_moon_arrival, altitude_km, mu_earth):
    u = r_moon_arrival / np.linalg.norm(r_moon_arrival)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    e2 = reference - np.dot(reference, u) * u
    e2 /= np.linalg.norm(e2)
    e1 = u
    theta = np.radians(150.0)
    r0_hat = np.cos(theta) * e1 - np.sin(theta) * e2
    tangent_hat = np.sin(theta) * e1 + np.cos(theta) * e2
    r0 = (EARTH_RADIUS_KM + altitude_km) * 1000.0 * r0_hat
    v0 = np.sqrt(mu_earth / np.linalg.norm(r0)) * tangent_hat
    return r0, v0


def _trajectory_plane_basis(r0, v0):
    h = np.cross(r0, v0)
    e1 = r0 / np.linalg.norm(r0)
    e2 = np.cross(h, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def _project(points_m, e1, e2):
    points_m = np.asarray(points_m)
    return np.stack([points_m @ e1, points_m @ e2], axis=-1) / 1000.0


def compute_and_export_mission(
    departure_mjd2000=None,
    coast_days_guess=4.5,
    total_days=9.6,
    step_seconds=60.0,
    parking_altitude_km=185.0,
    max_output_points=700,
):
    """Re-run the Lambert-seeded differential-correction targeter (same
    approach as examples/05_artemis2_free_return.py, without the GPU
    coarse-screening stage -- a hand-verified Lambert guess is already
    close enough that the targeter converges directly) and export the
    resulting trajectory as a scrubbable, plane-projected time series.

    ``step_seconds`` defaults to 60s, not a coarser animation-friendly
    value, for the same reason ``verify.differential_correction`` does:
    a 900s step produced a spacecraft distance range of roughly 1,100 km
    to 19,400,000 km for this exact trajectory on a first attempt -- an
    inside-the-Earth minimum and an escape-trajectory-scale maximum, both
    numerical artifacts of a fixed-step RK4 too coarse for the curvature
    near the lunar flyby, not real dynamics (see README's RK4 step-size
    gotcha). The full-resolution propagation is then decimated to
    ``max_output_points`` for export -- correctness comes from the step
    size used during integration, not from how many of those already-
    correct points get kept for the animation.
    """
    if departure_mjd2000 is None:
        departure_mjd2000 = mjd2000_from_date(2026, 8, 1)

    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    r_moon_arrival, _ = moon_state(departure_mjd2000 + coast_days_guess)
    r_moon_arrival = np.asarray(r_moon_arrival)

    r0, v0 = _plane_aligned_parking_orbit(r_moon_arrival, parking_altitude_km, mu_earth)
    (v1, _v2), = solve_lambert_single(r0, r_moon_arrival, coast_days_guess * 86400.0, mu=mu_earth, max_revs=0)[:1]
    dv_guess = v1 - v0

    target_distance_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM
    targeting = target_lunar_flyby(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=dv_guess, coast_duration_guess=coast_days_guess * 86400.0,
        target_distance_km=target_distance_km,
        reference_epoch_ephemeris_seconds=reference_et,
    )
    if not targeting.converged:
        raise RuntimeError("Differential correction did not converge; adjust departure_mjd2000 or coast_days_guess.")

    full = propagate_multi_arc(
        r0, v0, 0.0,
        arcs=[
            {"type": "impulsive_burn", "delta_v": targeting.delta_v},
            {"type": "coast", "duration": total_days * 86400.0},
        ],
        perturbing_bodies=("Earth", "Moon", "Sun"), step_size=step_seconds,
    )

    e1, e2 = _trajectory_plane_basis(r0, v0)

    perilune_day = round(targeting.coast_duration / 86400.0, 4)
    perilune_idx_fine = int(np.argmin(np.abs(full.epochs - targeting.coast_duration)))

    n_fine = len(full.epochs)
    if n_fine > max_output_points:
        keep = np.unique(np.linspace(0, n_fine - 1, max_output_points - 1).round().astype(int))
        keep = np.unique(np.concatenate([keep, [perilune_idx_fine]]))
    else:
        keep = np.arange(n_fine)

    epochs = full.epochs[keep]
    perilune_idx = int(np.searchsorted(keep, perilune_idx_fine))

    spacecraft_2d = _project(full.states[keep, :3], e1, e2)
    spacecraft_distance_km = np.linalg.norm(full.states[keep, :3], axis=1) / 1000.0

    moon_positions_m = np.array([
        body_position_at_absolute_epoch("Moon", reference_et + t) for t in epochs
    ])
    moon_2d = _project(moon_positions_m, e1, e2)
    moon_distance_km = np.linalg.norm(moon_positions_m, axis=1) / 1000.0

    days = (epochs / 86400.0).round(4).tolist()

    events = [
        {"label": "TLI burn", "day": 0.0, "note": f"delta-v {np.linalg.norm(targeting.delta_v):.0f} m/s"},
        {
            "label": "Closest approach to Moon",
            "day": perilune_day,
            "note": f"{targeting.final_distance_km:.0f} km from Moon center "
                    f"({targeting.final_distance_km - MOON_RADIUS_KM:.0f} km altitude)",
        },
    ]

    return {
        "reference_epoch_ephemeris_seconds": reference_et,
        "departure_mjd2000": departure_mjd2000,
        "target_perilune_km": target_distance_km,
        "achieved_perilune_km": targeting.final_distance_km,
        "perilune_day": perilune_day,
        "perilune_index": perilune_idx,
        "delta_v_m_s": targeting.delta_v.tolist(),
        "events": events,
        "days": days,
        "spacecraft_xy_km": spacecraft_2d.round(1).tolist(),
        "spacecraft_distance_km": spacecraft_distance_km.round(1).tolist(),
        "moon_xy_km": moon_2d.round(1).tolist(),
        "moon_distance_km": moon_distance_km.round(1).tolist(),
    }
