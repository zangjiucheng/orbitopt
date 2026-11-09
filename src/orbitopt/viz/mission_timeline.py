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

from orbitopt.bodies import (
    body_fixed_to_eclipj2000_matrix,
    mjd2000_from_date,
    mjd2000_to_ephemeris_seconds,
    moon_state,
)
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify import report as verification_report
from orbitopt.verify.differential_correction import (
    LUNAR_RETURN_ENTRY_CORRIDOR_DEG,
    LUNAR_RETURN_ENTRY_FPA_DEG,
    flight_path_angle_deg,
    target_free_return_entry_corridor,
)
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


DEFAULT_TRANSFER_THETA_DEG = 142.0
"""Parking-orbit transfer angle behind the Moon's arrival direction (see
_plane_aligned_parking_orbit) -- where along the parking orbit TLI happens.

History: this used to be the *only* lever on the return leg. The mission was
targeted with target_lunar_flyby, which aims lunar flyby distance only, so
the entry angle was incidental -- this angle was picked by scanning theta for
a return that happened to reach the entry interface, and the best such scan
point still entered at about -32.7 degrees, far steeper than survivable.
(Separately, under the current adaptive DOP853 verification integrator that
flyby-distance-only solution doesn't even return: its fixed-time "8,300 km"
miss is an inbound sample, not the periapsis, and the real flyby dips much
lower and escapes.)

Now the return is *targeted* (target_free_return_entry_corridor): the lunar
B-plane aim point and the Earth-return perigee / entry flight-path angle are
solved jointly, so theta just sets the departure geometry and no longer has
to be tuned for luck. 142 degrees is kept because it gives a clean, fast-
converging seed (~5 joint-Newton iterations).
"""

LAMBERT_AIM_OFFSET_KM = 20_000.0
"""The Lambert seed aims this far ahead of the Moon (along the Moon's own
direction of motion), not at its centre. Aiming at the centre gives a guess
that, under real dynamics, passes through (or within a few hundred km of)
the point-mass Moon -- a degenerate, near-radial relative trajectory with no
meaningful B-plane, and a propagation that can abort on the singularity. A
clear miss gives a well-defined incoming asymptote for the B-plane seed to
work from; the targeter then pulls the flyby in to the Artemis II perilune.
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


def _verify_entry_corridor(full, corridor_deg=LUNAR_RETURN_ENTRY_CORRIDOR_DEG):
    """Independently verify the exported trajectory: its first downward
    crossing of the entry interface must exist and its inertial flight-path
    angle must lie inside ``corridor_deg`` (steepest, shallowest). Returns the
    crossing; raises RuntimeError otherwise.

    This deliberately re-checks the full re-propagated trajectory instead of
    trusting ``targeting.in_corridor``: the export propagation uses its own
    step size and span, so a solution that was in-corridor inside the
    targeter can still land outside it here.
    """
    entry = find_altitude_crossing(
        full, central_body_radius_km=EARTH_RADIUS_KM, threshold_altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
    )
    steepest, shallowest = corridor_deg
    if entry is None:
        raise RuntimeError(
            "Re-propagated free-return trajectory never reaches the "
            f"{EARTH_ENTRY_INTERFACE_ALTITUDE_KM:.2f} km entry interface; refusing to export it."
        )
    fpa = flight_path_angle_deg(entry.spacecraft_state)
    if not steepest <= fpa <= shallowest:
        raise RuntimeError(
            f"Re-propagated free-return trajectory enters at {fpa:.2f} deg, outside the "
            f"[{steepest}, {shallowest}] deg lunar-return entry corridor; refusing to export it."
        )
    return entry


def compute_and_export_mission(
    departure_mjd2000=None,
    coast_days_guess=3.5,
    total_days=12.0,
    step_seconds=60.0,
    parking_altitude_km=185.0,
    max_output_points=700,
    entry_fpa_target_deg=LUNAR_RETURN_ENTRY_FPA_DEG,
    transfer_theta_deg=DEFAULT_TRANSFER_THETA_DEG,
    perilune_altitude_km=ARTEMIS_II_PERILUNE_ALTITUDE_KM,
):
    """Solve the free return with the joint lunar B-plane + Earth-return
    entry-corridor targeter (verify.differential_correction.
    target_free_return_entry_corridor), seeded by an offset Lambert arc
    (see LAMBERT_AIM_OFFSET_KM), and export the resulting trajectory as a
    scrubbable SceneData document. The return leg reaches the entry
    interface at ``entry_fpa_target_deg`` (default -6 degrees, inside the
    real lunar-return skip-entry corridor) with no further burns.

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
    scene, _targeting = _build_artemis2_mission(
        departure_mjd2000=departure_mjd2000, coast_days_guess=coast_days_guess,
        total_days=total_days, step_seconds=step_seconds,
        parking_altitude_km=parking_altitude_km, max_output_points=max_output_points,
        entry_fpa_target_deg=entry_fpa_target_deg, transfer_theta_deg=transfer_theta_deg,
        perilune_altitude_km=perilune_altitude_km,
    )
    return scene


def _build_artemis2_mission(
    departure_mjd2000=None,
    coast_days_guess=3.5,
    total_days=12.0,
    step_seconds=60.0,
    parking_altitude_km=185.0,
    max_output_points=700,
    entry_fpa_target_deg=LUNAR_RETURN_ENTRY_FPA_DEG,
    transfer_theta_deg=DEFAULT_TRANSFER_THETA_DEG,
    perilune_altitude_km=ARTEMIS_II_PERILUNE_ALTITUDE_KM,
):
    """compute_and_export_mission's body; also returns the
    FreeReturnTargetingResult so build_from_config can attach its
    machine-readable verification report to the scene."""
    if departure_mjd2000 is None:
        departure_mjd2000 = mjd2000_from_date(2026, 8, 1)

    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    r_moon_arrival, v_moon_arrival = moon_state(departure_mjd2000 + coast_days_guess)
    r_moon_arrival = np.asarray(r_moon_arrival)
    v_moon_arrival = np.asarray(v_moon_arrival)

    target_distance_km = perilune_altitude_km + MOON_RADIUS_KM

    r0, v0 = _plane_aligned_parking_orbit(
        r_moon_arrival, v_moon_arrival, parking_altitude_km, mu_earth, theta_deg=transfer_theta_deg)
    moon_h_hat = np.cross(r_moon_arrival, v_moon_arrival)
    moon_h_hat /= np.linalg.norm(moon_h_hat)
    along_track_hat = np.cross(moon_h_hat, r_moon_arrival / np.linalg.norm(r_moon_arrival))
    lambert_aim = r_moon_arrival + LAMBERT_AIM_OFFSET_KM * 1000.0 * along_track_hat
    (v1, _v2), = solve_lambert_single(r0, lambert_aim, coast_days_guess * 86400.0, mu=mu_earth, max_revs=0)[:1]
    dv_guess = v1 - v0
    # Joint lunar B-plane + Earth-return entry-corridor targeting: perilune at
    # the Artemis II altitude AND the return leg crossing the entry interface
    # at a survivable skip-entry flight-path angle, verified by TudatPy.
    targeting = target_free_return_entry_corridor(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=dv_guess, coast_duration_guess=coast_days_guess * 86400.0,
        target_periapsis_km=target_distance_km,
        reference_epoch_ephemeris_seconds=reference_et,
        target_entry_fpa_deg=entry_fpa_target_deg,
        earth_radius_km=EARTH_RADIUS_KM,
        entry_interface_altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
        entry_corridor_deg=LUNAR_RETURN_ENTRY_CORRIDOR_DEG,
    )
    # ``converged`` only means the Newton residuals (perilune, return perigee)
    # met tolerance; ``in_corridor`` is the separate check that the verified
    # entry flight-path angle is survivable. Both must hold -- a numerically
    # converged solution that skips out or burns up is not a free return.
    if not targeting.converged or not targeting.in_corridor:
        fpa = targeting.entry_flight_path_angle_deg
        raise RuntimeError(
            "Free-return entry-corridor targeting rejected "
            f"(converged={targeting.converged}, in_corridor={targeting.in_corridor}, "
            f"entry flight-path angle={'n/a' if fpa is None else f'{fpa:.2f} deg'}, "
            f"corridor={LUNAR_RETURN_ENTRY_CORRIDOR_DEG} deg); "
            "adjust departure_mjd2000, coast_days_guess or entry_fpa_target_deg."
        )

    # No need to integrate days past the (already verified) entry interface
    # only to truncate them below; half a day of margin keeps the crossing
    # comfortably inside the exported span.
    export_duration_s = total_days * 86400.0
    if targeting.entry_epoch is not None:
        export_duration_s = min(export_duration_s, targeting.entry_epoch + 0.5 * 86400.0)
    full = propagate_multi_arc(
        r0, v0, reference_et,
        arcs=[
            {"type": "impulsive_burn", "delta_v": targeting.delta_v},
            {"type": "coast", "duration": export_duration_s},
        ],
        perturbing_bodies=("Earth", "Moon", "Sun"), step_size=step_seconds,
    )

    # The targeter aims the return leg at the entry interface, but this
    # propagator has no solid Earth or atmosphere model, so past the entry
    # point it doesn't stop -- it keeps going through the massless-surface
    # Earth and swings back out on another multi-day arc, which is not what a
    # real mission does and reads as an obviously broken shape in the 3D
    # view. Truncate at the genuine end of the free-return leg -- the entry-
    # interface crossing -- after independently re-checking the corridor on
    # this re-propagated trajectory rather than trusting the targeter's own
    # internal verification run.
    entry = _verify_entry_corridor(full)
    cutoff_epoch, cutoff_state = entry.epoch, entry.spacecraft_state

    keep_before_cutoff = full.epochs <= cutoff_epoch
    full = PropagationResult(
        epochs=np.append(full.epochs[keep_before_cutoff], cutoff_epoch),
        states=np.vstack([full.states[keep_before_cutoff], cutoff_state]),
        final_state=cutoff_state,
    )

    perilune_day = round(targeting.perilune_epoch / 86400.0, 4)
    perilune_idx_fine = int(np.argmin(np.abs(full.epochs - targeting.perilune_epoch)))

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

    events = [
        {"label": "TLI burn", "time": 0.0, "note": f"delta-v {np.linalg.norm(targeting.delta_v):.0f} m/s"},
        {
            "label": "Closest approach to Moon",
            "time": perilune_day,
            "note": f"{targeting.perilune_distance_km:.0f} km from Moon center "
                    f"({targeting.perilune_distance_km - MOON_RADIUS_KM:.0f} km altitude), "
                    f"B-plane angle {targeting.b_plane_angle_deg:.1f} degrees",
        },
    ]
    entry_speed = np.linalg.norm(entry.spacecraft_state[3:])
    entry_flight_path_angle_deg = flight_path_angle_deg(entry.spacecraft_state)
    steepest, shallowest = LUNAR_RETURN_ENTRY_CORRIDOR_DEG
    events.append({
        "label": "Earth entry interface",
        "time": round(entry.epoch / 86400.0, 4),
        "note": f"free return -- {EARTH_ENTRY_INTERFACE_ALTITUDE_KM:.0f} km altitude reached "
                f"with no further burns, entry speed {entry_speed:.0f} m/s, "
                f"flight path angle {entry_flight_path_angle_deg:.1f} degrees (inside the "
                f"{abs(shallowest):.0f}-{abs(steepest):.0f} degree lunar-return skip-entry corridor)",
    })

    bodies = [
        body_entry(
            "earth", "Earth", COLOR["earth"], "planet", radius_display=RADIUS_DISPLAY["earth"],
            position=[0.0, 0.0, 0.0], texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
            rotation_period_hours=ROTATION_PERIOD_HOURS["earth"],
            orientation_basis=body_fixed_to_eclipj2000_matrix("Earth", departure_mjd2000).tolist(),
        ),
        body_entry(
            "moon", "Moon", COLOR["moon"], "moon", radius_display=RADIUS_DISPLAY["moon"],
            trail={"times": days, "positions": moon_positions_km.round(1).tolist()},
            # No static "Distance from Earth" row -- every body card already
            # shows a live "Distance" readout that tracks the scrubber (see
            # app.py's _rebuild_body_cards); a fixed mission-start snapshot
            # here duplicated it at t=0 and went stale (misleadingly) the
            # moment the timeline moved.
            texture=TEXTURE["moon"], radius=MOON_RADIUS_KM,
            rotation_period_hours=ROTATION_PERIOD_HOURS["moon"],
            orientation_basis=body_fixed_to_eclipj2000_matrix("Moon", departure_mjd2000).tolist(),
        ),
        body_entry(
            "spacecraft", "Orion", COLOR["spacecraft"], "spacecraft", radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": days, "positions": spacecraft_km.tolist()},
            info=[
                {"label": "TLI delta-v", "value": f"{np.linalg.norm(targeting.delta_v):.0f} m/s"},
                {"label": "Achieved perilune", "value": f"{targeting.perilune_distance_km:,.0f} km from Moon center"},
                {"label": "Return perigee (vacuum)", "value": f"{targeting.return_perigee_km - EARTH_RADIUS_KM:,.0f} km altitude"},
                {"label": "Entry flight-path angle",
                 "value": "n/a" if targeting.entry_flight_path_angle_deg is None
                 else f"{targeting.entry_flight_path_angle_deg:.2f} deg"},
            ],
        ),
    ]

    return scene_document(
        scene_id="artemis2-mission",
        title="Artemis II — Free Return",
        subtitle=f"Independently re-solved free return: Artemis II perilune altitude + targeted "
                 f"{entry_fpa_target_deg:.0f} deg entry flight-path angle.\n"
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
    ), targeting


def build_from_config(config: dict, recorder=None) -> dict:
    """orbitopt.mission_config adapter for mission.kind == "lunar-free-return"
    (the Artemis II free return; schema mission-lunar-free-return-1.0.json).
    Every omitted field falls back to compute_and_export_mission's defaults,
    i.e. the "artemis2" built-in mission.

    The free-return targeter is itself TudatPy-verified (perilune, return
    perigee and entry corridor are checked on a full propagation), so there
    is no separate re-fly stage to skip: ``verification.enabled: false``
    (``orbitopt run --skip-verify``) only drops the machine-readable report
    from the output -- the corridor check still gates the export.

    ``recorder`` (an orbitopt.pipeline.StageRecorder) is accepted for the
    unified pipeline's builder signature; this kind does not report
    per-stage results, so those stages stay "not run" in the pipeline
    result."""
    mission = config["mission"]
    departure = config.get("departure", {})
    parking_orbit = config.get("parking_orbit", {})
    transfer = config.get("transfer", {})
    flyby = config.get("lunar_flyby", {})
    entry = config.get("entry", {})
    propagation = config.get("propagation", {})
    verification = config.get("verification", {})

    kwargs = {}
    if "date" in departure:
        kwargs["departure_mjd2000"] = mjd2000_from_date(*departure["date"])
    elif "mjd2000" in departure:
        kwargs["departure_mjd2000"] = float(departure["mjd2000"])
    if "altitude_km" in parking_orbit:
        kwargs["parking_altitude_km"] = parking_orbit["altitude_km"]
    if "coast_days_guess" in transfer:
        kwargs["coast_days_guess"] = transfer["coast_days_guess"]
    if "theta_deg" in transfer:
        kwargs["transfer_theta_deg"] = transfer["theta_deg"]
    if "perilune_altitude_km" in flyby:
        kwargs["perilune_altitude_km"] = flyby["perilune_altitude_km"]
    if "target_fpa_deg" in entry:
        kwargs["entry_fpa_target_deg"] = entry["target_fpa_deg"]
    if "total_days" in propagation:
        kwargs["total_days"] = propagation["total_days"]
    if "step_size_s" in propagation:
        kwargs["step_seconds"] = propagation["step_size_s"]
    if "max_output_points" in propagation:
        kwargs["max_output_points"] = propagation["max_output_points"]

    scene, targeting = _build_artemis2_mission(**kwargs)
    scene["id"] = mission["id"]
    scene["title"] = mission["title"]
    if verification.get("enabled", True) and targeting.report is not None:
        # Popped and written next to the scene by `orbitopt run`.
        verification_report.attach_to_scene(scene, targeting.report)
    return scene
