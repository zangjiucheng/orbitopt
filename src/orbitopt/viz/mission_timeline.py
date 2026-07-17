"""Export a time-series SceneData document (see orbitopt.viz.scene) for the
Artemis-II-like free-return mission (verify.differential_correction /
examples/05_artemis2_free_return.py), for the general-purpose 3D viewer in
viewer/ -- not a bespoke mission-specific format.

Positions are exported in true 3D, Earth-centered ECLIPJ2000 km -- earlier
iterations of this module projected onto the trajectory's own orbital plane
for a 2D canvas view; that projection is gone now that rendering is
genuinely 3D, so the real inclination of this particular free-return
trajectory relative to the ecliptic is visible directly instead of being
flattened away. See _plane_aligned_parking_orbit for how that inclination is
chosen -- close to the Moon's own orbital plane (a few degrees), the natural
minimal-plane-change choice, not an arbitrary or forced value.
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, moon_state
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify.differential_correction import target_lunar_flyby
from orbitopt.verify.tudat_propagate import (
    PropagationResult,
    body_position_at_absolute_epoch,
    find_altitude_crossing,
    propagate_multi_arc,
)
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, ROTATION_PERIOD_HOURS, TEXTURE, body_entry, scene_document

EARTH_RADIUS_KM = 6378.0
MOON_RADIUS_KM = 1737.4
ARTEMIS_II_PERILUNE_ALTITUDE_KM = 6545.0
EARTH_ENTRY_INTERFACE_ALTITUDE_KM = 121.92  # 400,000 ft, the conventional atmospheric entry interface


DEFAULT_TRANSFER_THETA_DEG = 130.0
"""Parking-orbit transfer angle behind the Moon's arrival direction (see
_plane_aligned_parking_orbit) -- also, incidentally, the free variable that
determines which side of the Moon the flyby passes on, and so whether the
post-flyby return leg bends back toward Earth or swings wide of it (the
actual "free return" part of a free-return trajectory; target_lunar_flyby's
own targeting is lunar flyby *distance* only, see its docstring). Found via
orbitopt.verify.free_return_search's B-plane-informed outer search over
this angle jointly with ``coast_days_guess`` (see that parameter on
compute_and_export_mission) -- a 2-degree/half-day grid over both
dimensions, then a targeted 2-degree scan near this point's own neighbors
-- driving the return leg's closest approach to Earth toward the
atmospheric entry interface.

Not a genuine (zero-correction-burn) free return -- the return leg still
falls ~415 km short of the entry interface. That's a real improvement
over the previous defaults (134 degrees / 4.5-day coast, 585 km short;
150 degrees / 4.5-day coast before that, 1283 km short), found by
searching coast_days_guess as a second free dimension alongside theta as
this docstring's own previous revision flagged as a real follow-up. It is
still not a smooth local optimum: theta=129 and theta=131 at this same
4.0-day coast both fail to converge at all -- this differential
corrector's feasible region remains a fragmented scatter of isolated
points, confirmed by direct search rather than assumed, just as it was
at the previous 134-degree/4.5-day point. A coarse global grid over
(theta, coast_days) at 6-degree/1-day resolution found nothing better
than the old 134/4.5 point (best: 1170 km short) -- these isolated
basins are narrow enough that only a search concentrated near an
already-known-good point reliably lands inside one. Closing the
remaining ~415 km gap for real still needs a genuine 2-DOF B-plane
targeter (jointly solving for both the lunar approach *and* the Earth
return condition, rather than two 1D dimensions layered on
target_lunar_flyby's single-condition inner solve) -- a real follow-up,
not attempted here. See free_return_search.py's own module docstring for
why this search's free variables are this angle and the coast duration,
not B-plane coordinates directly.
"""


def _plane_aligned_parking_orbit(r_moon_arrival, v_moon_arrival, altitude_km, mu_earth, theta_deg=DEFAULT_TRANSFER_THETA_DEG):
    """Choose a parking-orbit orientation whose plane contains the Moon's
    arrival direction, at a ``theta_deg`` transfer angle behind it, oriented
    so the plane also contains the Moon's own velocity at arrival -- i.e.
    (up to the ~5 degree lunar-orbit-vs-ecliptic tilt) the Moon's own
    orbital plane, the natural minimal-plane-change choice for a lunar
    transfer. See DEFAULT_TRANSFER_THETA_DEG for what the angle itself
    controls and how its default value was chosen.

    An earlier version of this used the ecliptic pole (Z) as the reference
    vector for the second in-plane basis direction instead of the Moon's
    velocity. That's a real bug, not a style choice: for any reference
    vector R and unit vector u, the plane spanned by {u, R-(R.u)u} always
    has normal u x (R-(R.u)u) = u x R, which is *exactly* perpendicular to R
    by construction (cross products are always perpendicular to both
    inputs) -- so using R=Z forced the resulting transfer plane to be
    *exactly* 90 degrees inclined to the ecliptic, always, regardless of
    where the Moon actually is. That's a near-polar departure, not the
    "~30-40 degree" inclination this module's own docstring describes, and
    it produced a visibly wrong-looking, unrealistically-oriented trajectory
    (confirmed: 86-89 degrees measured on the actual exported trajectory).
    Using the Moon's velocity as the reference instead gives a genuinely
    moderate, physically-motivated inclination (~5 degrees, matching the
    Moon's real orbit) rather than an accidental fixed 90.
    """
    u = r_moon_arrival / np.linalg.norm(r_moon_arrival)
    reference = v_moon_arrival / np.linalg.norm(v_moon_arrival)
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    e2 = reference - np.dot(reference, u) * u
    e2 /= np.linalg.norm(e2)
    e1 = u
    theta = np.radians(theta_deg)
    r0_hat = np.cos(theta) * e1 - np.sin(theta) * e2
    tangent_hat = np.sin(theta) * e1 + np.cos(theta) * e2
    r0 = (EARTH_RADIUS_KM + altitude_km) * 1000.0 * r0_hat
    v0 = np.sqrt(mu_earth / np.linalg.norm(r0)) * tangent_hat
    return r0, v0


def compute_and_export_mission(
    departure_mjd2000=None,
    coast_days_guess=4.0,
    total_days=12.0,
    step_seconds=60.0,
    parking_altitude_km=185.0,
    max_output_points=700,
):
    """Re-run the Lambert-seeded differential-correction targeter (same
    approach as examples/05_artemis2_free_return.py, without the GPU
    coarse-screening stage -- a hand-verified Lambert guess is already
    close enough that the targeter converges directly) and export the
    resulting trajectory as a scrubbable SceneData document.

    ``step_seconds`` defaults to 60s, not a coarser animation-friendly
    value, for the same reason ``verify.differential_correction`` does: a
    900s step produced a spacecraft distance range of roughly 1,100 km to
    19,400,000 km for this exact trajectory -- an inside-the-Earth minimum
    and an escape-trajectory-scale maximum, both numerical artifacts of a
    fixed-step RK4 too coarse for the curvature near the lunar flyby, not
    real dynamics (see README's RK4 step-size gotcha). The full-resolution
    propagation is then decimated to ``max_output_points`` for export --
    correctness comes from the step size used during integration, not from
    how many of those already-correct points get kept for the animation.
    """
    if departure_mjd2000 is None:
        departure_mjd2000 = mjd2000_from_date(2026, 8, 1)

    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    r_moon_arrival, v_moon_arrival = moon_state(departure_mjd2000 + coast_days_guess)
    r_moon_arrival = np.asarray(r_moon_arrival)
    v_moon_arrival = np.asarray(v_moon_arrival)

    target_distance_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM

    r0, v0 = _plane_aligned_parking_orbit(r_moon_arrival, v_moon_arrival, parking_altitude_km, mu_earth)
    (v1, _v2), = solve_lambert_single(r0, r_moon_arrival, coast_days_guess * 86400.0, mu=mu_earth, max_revs=0)[:1]
    dv_guess = v1 - v0
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

    # This targeter (see target_lunar_flyby) only aims the lunar-flyby
    # distance, not a B-plane aim point -- so whether the resulting return
    # leg happens to cross the atmospheric entry interface on its own is
    # incidental, not guaranteed by construction. Either way, this
    # propagator has no solid Earth or atmosphere model, so past the
    # trajectory's own closest return approach it doesn't stop -- it just
    # keeps going (through the massless-surface Earth if the perigee is low
    # enough, or straight back out if it isn't) and swings back out on
    # another multi-day arc, which is not what a real mission does and reads
    # as an obviously broken shape (the spacecraft appears to pass through
    # or bounce off the planet and loop back out) in the 3D view. Truncate
    # at the genuine end of the free-return leg -- the entry-interface
    # crossing if it reaches one, otherwise the return leg's own closest
    # approach to Earth -- rather than exporting that unphysical tail.
    entry = find_altitude_crossing(
        full, central_body_radius_km=EARTH_RADIUS_KM, threshold_altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
    )
    return_perigee = None
    if entry is not None:
        cutoff_epoch, cutoff_state = entry.epoch, entry.spacecraft_state
    else:
        after_perilune = full.epochs > targeting.coast_duration
        if after_perilune.any():
            radii_km = np.linalg.norm(full.states[:, :3], axis=1) / 1000.0
            idx = int(np.argmin(np.where(after_perilune, radii_km, np.inf)))
            cutoff_epoch, cutoff_state = float(full.epochs[idx]), full.states[idx]
            return_perigee = {"distance_km": radii_km[idx]}
        else:
            cutoff_epoch = None

    if cutoff_epoch is not None:
        keep_before_cutoff = full.epochs <= cutoff_epoch
        full = PropagationResult(
            epochs=np.append(full.epochs[keep_before_cutoff], cutoff_epoch),
            states=np.vstack([full.states[keep_before_cutoff], cutoff_state]),
            final_state=cutoff_state,
        )

    perilune_day = round(targeting.coast_duration / 86400.0, 4)
    perilune_idx_fine = int(np.argmin(np.abs(full.epochs - targeting.coast_duration)))

    n_fine = len(full.epochs)
    if n_fine > max_output_points:
        keep = np.unique(np.linspace(0, n_fine - 1, max_output_points - 1).round().astype(int))
        keep = np.unique(np.concatenate([keep, [perilune_idx_fine]]))
    else:
        keep = np.arange(n_fine)

    epochs = full.epochs[keep]
    days = (epochs / 86400.0).round(4).tolist()

    spacecraft_km = (full.states[keep, :3] / 1000.0).round(1)
    spacecraft_distance_km = np.linalg.norm(full.states[keep, :3], axis=1) / 1000.0

    moon_positions_km = np.array([
        body_position_at_absolute_epoch("Moon", reference_et + t) for t in epochs
    ]) / 1000.0
    moon_distance_km = np.linalg.norm(moon_positions_km, axis=1)

    events = [
        {"label": "TLI burn", "time": 0.0, "note": f"delta-v {np.linalg.norm(targeting.delta_v):.0f} m/s"},
        {
            "label": "Closest approach to Moon",
            "time": perilune_day,
            "note": f"{targeting.final_distance_km:.0f} km from Moon center "
                    f"({targeting.final_distance_km - MOON_RADIUS_KM:.0f} km altitude)",
        },
    ]
    if entry is not None:
        events.append({
            "label": "Earth entry interface",
            "time": round(entry.epoch / 86400.0, 4),
            "note": f"free return -- {EARTH_ENTRY_INTERFACE_ALTITUDE_KM:.0f} km altitude reached "
                    "with no further burns",
        })
    elif return_perigee is not None:
        events.append({
            "label": "Closest approach to Earth (return leg)",
            "time": round(cutoff_epoch / 86400.0, 4),
            "note": f"{return_perigee['distance_km']:.0f} km from Earth center "
                    f"({return_perigee['distance_km'] - EARTH_RADIUS_KM:.0f} km altitude) -- "
                    "doesn't quite reach the atmosphere on its own",
        })

    bodies = [
        body_entry(
            "earth", "Earth", COLOR["earth"], "planet", radius_display=RADIUS_DISPLAY["earth"],
            position=[0.0, 0.0, 0.0], texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
            rotation_period_hours=ROTATION_PERIOD_HOURS["earth"],
        ),
        body_entry(
            "moon", "Moon", COLOR["moon"], "moon", radius_display=RADIUS_DISPLAY["moon"],
            trail={"times": days, "positions": moon_positions_km.round(1).tolist()},
            info=[{"label": "Distance from Earth", "value": f"{moon_distance_km[0]:,.0f} km (varies over mission)"}],
            texture=TEXTURE["moon"], radius=MOON_RADIUS_KM,
            rotation_period_hours=ROTATION_PERIOD_HOURS["moon"],
        ),
        body_entry(
            "spacecraft", "Orion", COLOR["spacecraft"], "spacecraft", radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": days, "positions": spacecraft_km.tolist()},
            info=[
                {"label": "TLI delta-v", "value": f"{np.linalg.norm(targeting.delta_v):.0f} m/s"},
                {"label": "Achieved perilune", "value": f"{targeting.final_distance_km:,.0f} km from Moon center"},
            ],
        ),
    ]

    return scene_document(
        scene_id="artemis2-mission",
        title="Artemis II — Free Return",
        subtitle="Independently re-solved trajectory targeting the real Artemis II perilune altitude.\n"
                  "Not a reproduction of the flown mission -- NASA hasn't published navigation-grade state vectors.",
        distance_unit="km",
        central_body_id="earth",
        bodies=bodies,
        timeline={
            "unitLabel": "days",
            "min": 0.0,
            "max": days[-1],
            "events": events,
            "referenceEpochEt": reference_et,
        },
    )
