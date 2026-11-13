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

import time

import numpy as np
import pykep as pk

from orbitopt.bodies import (
    body_fixed_to_eclipj2000_matrix,
    mjd2000_from_date,
    mjd2000_to_ephemeris_seconds,
    moon_state,
)
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.model_fidelity import (
    NOT_MODELLED_DRAG,
    NOT_MODELLED_IMPULSIVE_BURNS,
    NOT_MODELLED_POINT_MASS_GRAVITY,
    NOT_MODELLED_RELATIVITY,
    NOT_MODELLED_SRP,
)
from orbitopt.pipeline import (
    CONSTRAINTS,
    OPTIMIZE,
    REFINE,
    SEARCH,
    SELECT,
    VERIFY,
    StageRecorder,
)
from orbitopt.verify import report as verification_report
from orbitopt.verify.differential_correction import (
    LUNAR_RETURN_ENTRY_CORRIDOR_DEG,
    LUNAR_RETURN_ENTRY_FPA_DEG,
    flight_path_angle_deg,
    target_free_return_entry_corridor,
)
from orbitopt.verify.tudat_propagate import (
    DEFAULT_INTEGRATOR,
    PropagationResult,
    body_position_at_absolute_epoch,
    body_velocity_at_absolute_epoch,
    find_altitude_crossing,
    integrator_description,
    propagate_multi_arc,
)
from orbitopt.viz.scene import (
    COLOR,
    RADIUS_DISPLAY,
    ROTATION_PERIOD_HOURS,
    TEXTURE,
    body_entry,
    crossing_times,
    phase_spans,
    scene_document,
    thin_trail_indices,
)

EARTH_RADIUS_KM = 6378.0
MOON_RADIUS_KM = 1737.4
# Laplace sphere of influence of the Moon w.r.t. Earth, a * (m_moon/m_earth)^(2/5)
# = 384,400 km * 0.0123^0.4 ~ 66,100 km. Only used to name the flyby phase on
# the timeline; the propagation itself is full three-body, no patched conics.
MOON_SOI_RADIUS_KM = 66_100.0
ARTEMIS_II_PERILUNE_ALTITUDE_KM = 6545.0
# Trail/event times keep 1e-8 day (~1 ms): 1e-4 day (~8.6 s) put the
# spacecraft up to ~95 km off its own samples near perigee (~11 km/s).
# Positions keep 1 m, velocities 1e-3 km/day -- display data for the viewer's
# Hermite playback, not a source for metre-level error analysis.
TRAIL_TIME_DECIMALS = 8
TRAIL_POSITION_DECIMALS = 3
TRAIL_VELOCITY_DECIMALS = 3
KM_PER_DAY_PER_M_S = 86400.0 / 1000.0
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


def _record_targeting(st, targeting, target_distance_km, corridor_deg):
    """Optimize-stage contents of a FreeReturnTargetingResult."""
    steepest, shallowest = corridor_deg
    fpa = targeting.entry_flight_path_angle_deg
    seed = targeting.b_plane_seed
    st.candidates = [{
        "delta_v_m_s": np.asarray(targeting.delta_v).tolist(),
        "delta_v_magnitude_m_s": float(np.linalg.norm(targeting.delta_v)),
        "perilune_distance_km": targeting.perilune_distance_km,
        "b_plane_angle_deg": targeting.b_plane_angle_deg,
        "return_perigee_km": targeting.return_perigee_km,
        "entry_fpa_deg": fpa,
        "entry_epoch_s": targeting.entry_epoch,
        "selected": True,
    }]
    st.convergence = {
        "converged": bool(targeting.converged),
        "in_corridor": bool(targeting.in_corridor),
        "iterations": int(targeting.iterations),
        "b_plane_seed_iterations": None if seed is None else int(seed.iterations),
        "b_plane_seed_converged": None if seed is None else bool(seed.converged),
        "residual_history_km": [list(r) for r in targeting.residual_history_km],
    }
    st.constraint_violations = {
        "entry_corridor_deg": None if fpa is None else max(0.0, steepest - fpa, fpa - shallowest),
    }
    last = targeting.residual_history_km[-1] if targeting.residual_history_km else [None, None]
    st.numerical_error = {
        "perilune_residual_km": None if last[0] is None else abs(float(last[0])),
        "return_perigee_residual_km": None if last[1] is None else abs(float(last[1])),
        "perilune_vs_target_km": abs(float(targeting.perilune_distance_km) - float(target_distance_km)),
    }


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
    recorder=None,
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
    scene, targeting = _build_artemis2_mission(
        departure_mjd2000=departure_mjd2000, coast_days_guess=coast_days_guess,
        total_days=total_days, step_seconds=step_seconds,
        parking_altitude_km=parking_altitude_km, max_output_points=max_output_points,
        entry_fpa_target_deg=entry_fpa_target_deg, transfer_theta_deg=transfer_theta_deg,
        perilune_altitude_km=perilune_altitude_km, recorder=recorder,
    )
    if targeting.report is not None:
        verification_report.publish_to_scene(scene, [targeting.report])
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
    recorder=None,
    publish_report=True,
):
    """compute_and_export_mission's body; also returns the
    FreeReturnTargetingResult so build_from_config can attach its
    machine-readable verification report to the scene. ``publish_report=False``
    (``verification.enabled: false``) leaves the report out of the gate.

    ``recorder`` (orbitopt.pipeline.StageRecorder) receives every stage of
    the unified pipeline this kind has:

    - constraints: perilune target, entry-interface altitude and corridor;
    - search: the offset-Lambert TLI seed (two-body);
    - optimize: the joint B-plane + entry-corridor targeter (Newton
      iterations, final residuals, corridor violation);
    - select: the single targeted solution entering the Verification Gate
      (or its rejection when the targeter didn't converge / left the corridor);
    - refine: the fine-step export re-propagation, truncated at entry
      interface and decimated, with its drift from the targeter's solution;
    - verify: the independent entry-corridor re-check of that trajectory
      plus the targeter's TudatPy report, driving the gate verdict.
    """
    recorder = StageRecorder() if recorder is None else recorder
    gate = recorder.gate
    if departure_mjd2000 is None:
        departure_mjd2000 = mjd2000_from_date(2026, 8, 1)

    steepest, shallowest = LUNAR_RETURN_ENTRY_CORRIDOR_DEG
    target_distance_km = perilune_altitude_km + MOON_RADIUS_KM
    with recorder.stage(CONSTRAINTS) as st:
        st.physical_model = {"name": "free-return mission targets", "fidelity": "none"}
        st.candidates = [{
            "departure_mjd2000": float(departure_mjd2000),
            "parking_altitude_km": float(parking_altitude_km),
            "perilune_altitude_km": float(perilune_altitude_km),
            "target_perilune_distance_km": float(target_distance_km),
            "entry_interface_altitude_km": EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
            "entry_fpa_target_deg": float(entry_fpa_target_deg),
            "entry_corridor_deg": [steepest, shallowest],
        }]
        st.constraint_violations = {
            "entry_fpa_target_outside_corridor_deg": max(0.0, steepest - entry_fpa_target_deg,
                                                         entry_fpa_target_deg - shallowest),
        }

    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    with recorder.stage(SEARCH) as st:
        st.physical_model = {"name": "offset-aim Lambert arc to the Moon (pykep)", "fidelity": "two-body"}
        st.record_physics(
            assumptions=["one Earth-centered two-body Lambert arc to a fixed offset aim point ahead of the "
                         "Moon's predicted position"],
            limitations=["seed only: no Moon gravity on the arc, so this dv misses the free-return solution the "
                         "targeter converges"],
            not_modelled=["third_body_gravity (Moon gravity is what bends the real arc)",
                          NOT_MODELLED_SRP, NOT_MODELLED_IMPULSIVE_BURNS],
        )
        r_moon_arrival, v_moon_arrival = moon_state(departure_mjd2000 + coast_days_guess)
        r_moon_arrival = np.asarray(r_moon_arrival)
        v_moon_arrival = np.asarray(v_moon_arrival)

        r0, v0 = _plane_aligned_parking_orbit(
            r_moon_arrival, v_moon_arrival, parking_altitude_km, mu_earth, theta_deg=transfer_theta_deg)
        moon_h_hat = np.cross(r_moon_arrival, v_moon_arrival)
        moon_h_hat /= np.linalg.norm(moon_h_hat)
        along_track_hat = np.cross(moon_h_hat, r_moon_arrival / np.linalg.norm(r_moon_arrival))
        lambert_aim = r_moon_arrival + LAMBERT_AIM_OFFSET_KM * 1000.0 * along_track_hat
        (v1, _v2), = solve_lambert_single(r0, lambert_aim, coast_days_guess * 86400.0, mu=mu_earth, max_revs=0)[:1]
        dv_guess = v1 - v0
        st.candidates = [{
            "transfer_theta_deg": float(transfer_theta_deg),
            "coast_days_guess": float(coast_days_guess),
            "lambert_aim_offset_km": LAMBERT_AIM_OFFSET_KM,
            "dv_guess_m_s": np.asarray(dv_guess).tolist(),
            "dv_guess_magnitude_m_s": float(np.linalg.norm(dv_guess)),
            "selected": True,
        }]
        st.convergence = {"converged": bool(np.all(np.isfinite(dv_guess))), "lambert_revolutions": 0}
        st.notes.append("single Lambert seed aimed ahead of the Moon; the targeter does the real search")

    # Joint lunar B-plane + Earth-return entry-corridor targeting: perilune at
    # the Artemis II altitude AND the return leg crossing the entry interface
    # at a survivable skip-entry flight-path angle, verified by TudatPy.
    with recorder.stage(OPTIMIZE) as st:
        st.physical_model = {"name": "joint lunar B-plane + entry-corridor Newton (TudatPy Earth/Moon/Sun)",
                             "fidelity": "high-fidelity"}
        st.record_physics(
            assumptions=["point-mass Earth/Moon/Sun with SPICE ephemerides; a single impulsive departure burn",
                         "entry corridor checked at the interface crossing of the propagated return leg"],
            limitations=["converged means the perilune/entry-FPA residuals met tolerance; corridor survivability "
                         "is the separate in_corridor check"],
            not_modelled=[NOT_MODELLED_POINT_MASS_GRAVITY, NOT_MODELLED_SRP, NOT_MODELLED_DRAG,
                          NOT_MODELLED_RELATIVITY, NOT_MODELLED_IMPULSIVE_BURNS],
        )
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
        _record_targeting(st, targeting, target_distance_km, LUNAR_RETURN_ENTRY_CORRIDOR_DEG)

    # ``converged`` only means the Newton residuals (perilune, return perigee)
    # met tolerance; ``in_corridor`` is the separate check that the verified
    # entry flight-path angle is survivable. Both must hold -- a numerically
    # converged solution that skips out or burns up is not a free return.
    with recorder.stage(SELECT) as st:
        st.physical_model = {"name": "targeter acceptance (converged and in corridor)", "fidelity": "none"}
        fpa = targeting.entry_flight_path_angle_deg
        candidate = {"index": 0, "converged": bool(targeting.converged), "in_corridor": bool(targeting.in_corridor),
                     "entry_fpa_deg": fpa, "delta_v_m_s": float(np.linalg.norm(targeting.delta_v))}
        st.candidates = [candidate]
        gate.register(0, source="free-return entry-corridor targeter")
        if not targeting.converged or not targeting.in_corridor:
            reason = (
                "Free-return entry-corridor targeting rejected "
                f"(converged={targeting.converged}, in_corridor={targeting.in_corridor}, "
                f"entry flight-path angle={'n/a' if fpa is None else f'{fpa:.2f} deg'}, "
                f"corridor={LUNAR_RETURN_ENTRY_CORRIDOR_DEG} deg)"
            )
            gate.reject(0, reason)
            st.convergence = {"converged": False}
            raise RuntimeError(reason + "; adjust departure_mjd2000, coast_days_guess or entry_fpa_target_deg.")
        candidate["selected"] = True
        st.convergence = {"converged": True, "kept": 1, "considered": 1}
        st.notes.append("one targeted solution per run; no ranking needed")

    # No need to integrate days past the (already verified) entry interface
    # only to truncate them below; half a day of margin keeps the crossing
    # comfortably inside the exported span.
    with recorder.stage(REFINE) as st:
        # propagate_multi_arc's default integrator is adaptive RKDP 8(7);
        # step_seconds is only its maximum step, not a fixed step.
        st.physical_model = {"name": f"adaptive RKDP 8(7) TudatPy re-propagation "
                                     f"(max step {step_seconds:g} s; Earth/Moon/Sun)",
                             "fidelity": "high-fidelity",
                             "integrator": integrator_description(DEFAULT_INTEGRATOR, step_seconds)}
        st.record_physics(
            assumptions=["one impulsive TLI then a single coast; point-mass Earth/Moon/Sun on SPICE ephemerides"],
            limitations=["no solid Earth or atmosphere: past the entry interface the trajectory keeps going "
                        "through the planet, so the export is truncated at the interface crossing"],
            not_modelled=[NOT_MODELLED_POINT_MASS_GRAVITY, NOT_MODELLED_SRP, NOT_MODELLED_DRAG,
                          NOT_MODELLED_RELATIVITY, NOT_MODELLED_IMPULSIVE_BURNS],
        )
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
    with recorder.stage(VERIFY) as st:
        st.physical_model = {"name": "entry-corridor re-check on the export propagation + targeter TudatPy report",
                             "fidelity": "high-fidelity"}
        st.record_physics(
            assumptions=["same point-mass Earth/Moon/Sun dynamics as the export propagation; the corridor check "
                         "reads the vacuum flight-path angle at the interface crossing"],
            limitations=["a free-return corridor verdict only: entry dynamics themselves (lift, skip, drag) "
                         "are outside this framework's scope"],
            not_modelled=[NOT_MODELLED_POINT_MASS_GRAVITY, NOT_MODELLED_SRP, NOT_MODELLED_DRAG,
                          NOT_MODELLED_RELATIVITY, NOT_MODELLED_IMPULSIVE_BURNS],
        )
        try:
            entry = _verify_entry_corridor(full)
        except RuntimeError as exc:
            gate.reject(0, str(exc))
            st.convergence = {"converged": False, "entry_corridor": False}
            raise
        entry_fpa = flight_path_angle_deg(entry.spacecraft_state)
        reports = [targeting.report] if publish_report and targeting.report is not None else []
        if not publish_report:
            st.notes.append("verification.enabled is false: corridor re-checked, no report published")
        verdict = gate.evaluate(0, reports, entry_fpa_deg=entry_fpa)
        # Only a *failed* verification makes the export best-effort; a
        # not_checked one (report disabled) is exported with a warning.
        gate.export(0, best_effort=verdict["verification"] == "failed")
        st.candidates = [{"index": 0, "verification": verdict["verification"], "verdict": verdict["verdict"],
                          "entry_fpa_deg": entry_fpa, "entry_epoch_s": float(entry.epoch), "selected": True}]
        st.constraint_violations = {
            "entry_corridor_deg": max(0.0, steepest - entry_fpa, entry_fpa - shallowest),
        }
        st.convergence = {"converged": bool(reports) and verdict["verification"] == "passed",
                          "entry_corridor": True}
        if targeting.entry_flight_path_angle_deg is not None:
            st.numerical_error["entry_fpa_vs_targeter_deg"] = abs(entry_fpa - targeting.entry_flight_path_angle_deg)
        if targeting.entry_epoch is not None:
            st.numerical_error["entry_epoch_vs_targeter_s"] = abs(float(entry.epoch) - float(targeting.entry_epoch))

    refine_t0 = time.perf_counter()
    cutoff_epoch, cutoff_state = entry.epoch, entry.spacecraft_state

    keep_before_cutoff = full.epochs <= cutoff_epoch
    full = PropagationResult(
        epochs=np.append(full.epochs[keep_before_cutoff], cutoff_epoch),
        states=np.vstack([full.states[keep_before_cutoff], cutoff_state]),
        final_state=cutoff_state,
    )

    n_fine = len(full.epochs)
    perilune_day = round(targeting.perilune_epoch / 86400.0, TRAIL_TIME_DECIMALS)
    perilune_idx_fine = int(np.argmin(np.abs(full.epochs - targeting.perilune_epoch)))

    # The TLI burn is the first arc, so the exported trail is one burn-free
    # interpolation segment; thin_trail_indices would keep any burn pair.
    keep = thin_trail_indices(full.epochs, max_output_points - 1, keep=[perilune_idx_fine])

    epochs = full.epochs[keep]
    days = (epochs / 86400.0).round(TRAIL_TIME_DECIMALS).tolist()

    spacecraft_km = (full.states[keep, :3] / 1000.0).round(TRAIL_POSITION_DECIMALS)
    # km/day (trail distance unit per timeline unit): the viewer Hermite-
    # interpolates between samples instead of cutting chords across the
    # perigee and perilune arcs.
    spacecraft_vel = (full.states[keep, 3:] * KM_PER_DAY_PER_M_S).round(TRAIL_VELOCITY_DECIMALS)
    spacecraft_distance_km = np.linalg.norm(full.states[keep, :3], axis=1) / 1000.0

    moon_positions_km = np.array([
        body_position_at_absolute_epoch("Moon", reference_et + t) for t in epochs
    ]) / 1000.0
    moon_vel = np.array([
        body_velocity_at_absolute_epoch("Moon", reference_et + t) for t in epochs
    ]) * KM_PER_DAY_PER_M_S

    refine = recorder.add_time(REFINE, time.perf_counter() - refine_t0)
    sampled_perilune_km = float(np.min(np.linalg.norm(
        full.states[keep, :3] / 1000.0 - moon_positions_km, axis=1)))
    refine.candidates = [{
        "index": 0, "fine_samples": int(n_fine), "exported_samples": len(keep),
        "export_duration_s": float(export_duration_s), "cutoff_epoch_s": float(cutoff_epoch), "selected": True,
    }]
    refine.convergence = {"converged": True}
    refine.numerical_error = {
        "perilune_distance_vs_targeter_km": abs(sampled_perilune_km - float(targeting.perilune_distance_km)),
    }
    refine.notes.append("truncated at the entry-interface crossing, decimated to max_output_points")

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
        "time": round(entry.epoch / 86400.0, TRAIL_TIME_DECIMALS),
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
            trail={"times": days, "positions": moon_positions_km.round(TRAIL_POSITION_DECIMALS).tolist(),
                   "velocities": moon_vel.round(TRAIL_VELOCITY_DECIMALS).tolist()},
            # No static "Distance from Earth" row -- every body card already
            # shows a live "Distance" readout that tracks the scrubber (see
            # inspector.py's InspectorPanel); a fixed mission-start snapshot
            # here duplicated it at t=0 and went stale (misleadingly) the
            # moment the timeline moved.
            texture=TEXTURE["moon"], radius=MOON_RADIUS_KM,
            rotation_period_hours=ROTATION_PERIOD_HOURS["moon"],
            orientation_basis=body_fixed_to_eclipj2000_matrix("Moon", departure_mjd2000).tolist(),
        ),
        body_entry(
            "spacecraft", "Orion", COLOR["spacecraft"], "spacecraft", radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": days, "positions": spacecraft_km.tolist(), "velocities": spacecraft_vel.tolist()},
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
            "phases": _artemis_phases(days, spacecraft_km, moon_positions_km, events),
            "referenceEpochEt": reference_et,
        },
    ), targeting


def _artemis_phases(days, spacecraft_km, moon_positions_km, events):
    """Outbound coast, the lunar flyby (while inside the Moon's sphere of
    influence -- crossing times interpolated from the exported samples), the
    return coast to entry interface, and anything propagated past it."""
    entry = next(e["time"] for e in events if e["label"] == "Earth entry interface")
    moon_dist = np.linalg.norm(np.asarray(spacecraft_km) - np.asarray(moon_positions_km), axis=1)
    soi = crossing_times(days, moon_dist, MOON_SOI_RADIUS_KM)
    end = days[-1]
    if soi is None:
        return phase_spans([("Coast", 0.0, entry), ("Past entry interface", entry, end)])
    t_in, t_out = soi
    return phase_spans([
        ("Outbound coast", 0.0, t_in, "TLI to Moon SOI entry"),
        ("Lunar flyby", t_in, t_out, f"inside the Moon's sphere of influence ({MOON_SOI_RADIUS_KM:,.0f} km)"),
        ("Return coast", t_out, entry, "Moon SOI exit to the 122 km entry interface, no burns"),
        ("Past entry interface", entry, end, "propagated in vacuum -- no atmosphere model"),
    ])


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

    ``recorder`` (an orbitopt.pipeline.StageRecorder) receives the
    constraints, search, optimize, select, refine and verify stage results
    and drives the Verification Gate (see _build_artemis2_mission). With
    ``verification.enabled: false`` the corridor check still runs, but no
    report is published, so the gate records the candidate as not_checked."""
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

    if not verification.get("enabled", True):
        # Report-less gate verdict: the corridor re-check still guards the
        # export, but nothing machine-readable says the run was verified.
        kwargs["publish_report"] = False
    scene, targeting = _build_artemis2_mission(recorder=recorder, **kwargs)
    scene["id"] = mission["id"]
    scene["title"] = mission["title"]
    if verification.get("enabled", True) and targeting.report is not None:
        # Popped and written next to the scene by `orbitopt run`.
        verification_report.attach_to_scene(scene, targeting.report)
    return scene
