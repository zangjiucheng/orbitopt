"""Export a scrubbable SceneData document (see orbitopt.viz.scene) for a
Mars probe's entire journey from Earth launch through Mars orbit insertion --
the Earth-Moon (mission_timeline.py) and Earth-GEO (geo_raising.py) missions'
interplanetary analog, and the first mission in this app that leaves Earth's
sphere of influence at all.

Ends at Mars orbit insertion, not a landing: this framework has no
atmosphere model (point-mass/N-body gravity only, see verify.tudat_propagate),
so a descent-to-the-surface arc would be fabricated physics, unlike
everything else this framework claims. Same honesty-about-scope this
project already applies elsewhere (e.g. geo_raising.py not drawing the
launch ascent a two-body propagator can't represent either).

Three physics regimes, one continuous heliocentric-km trail: near-Earth
(parking orbit -> TMI burn -> hyperbolic escape), heliocentric cruise, and
near-Mars (hyperbolic approach -> MOI burn -> captured orbit). Nothing in
the scene format enforces a single reference frame for a trail (see
scene_format.py's schema -- trail positions are just numbers, continuity is
the only thing the renderer cares about), so each Earth/Mars-centered phase
is propagated in its own natural frame and then converted to heliocentric by
adding that body's own real heliocentric position at each sample's epoch --
exact vector addition (ECLIPJ2000 axes are non-rotating, so this is a plain
Galilean frame translation), not an approximation.

The idealized Lambert departure does NOT coincide with reality closely
enough to skip a correction: the real Earth-SOI-exit state (a few days after
the idealized instant, once Earth's own heliocentric velocity has rotated a
couple of degrees) misses Mars by well over a million km under real
Earth/Mars/Jupiter third-body perturbation -- measured, not assumed (see
verify.mars_insertion's module docstring). A trajectory-correction maneuver
(TCM) closes this, exactly like every real interplanetary mission flies one
for the same reason.
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, planet
from orbitopt.lambert.gpu_batch import solve_lambert_batch
from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.transfer_2body import Transfer2BodyProblem
from orbitopt.verify.mars_insertion import (
    locate_mars_periapsis,
    target_mars_approach,
    verify_mars_orbit_insertion,
)
from orbitopt.verify.tudat_propagate import (
    body_gravitational_parameter,
    body_position_at_absolute_epoch,
    body_velocity_at_absolute_epoch,
    propagate_multi_arc,
)
from orbitopt.viz.porkchop import compute_porkchop
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, ROTATION_PERIOD_HOURS, TEXTURE, body_entry, scene_document

EARTH_RADIUS_KM = 6378.0
MARS_RADIUS_KM = 3389.5
AU_KM = 1.495978707e8

PARKING_ALTITUDE_KM = 185.0  # matches mission_timeline.py's Artemis II convention
EARTH_SOI_RADIUS_KM = 924000.0  # real Earth sphere-of-influence radius
MARS_SOI_RADIUS_KM = 577000.0  # real Mars sphere-of-influence radius
MARS_TCM_TARGET_PERIAPSIS_KM = MARS_RADIUS_KM + 500.0
MARS_CAPTURE_APOAPSIS_ALT_KM = 20000.0

# Real, near-term Earth->Mars Hohmann-class transfer window (synodic period
# ~780 days; recent windows ~2020-07, ~2022-09, ~2024-10 -> next ~2026 Q4).
DEPARTURE_WINDOW_START = (2026, 9, 1)
DEPARTURE_WINDOW_END = (2027, 2, 1)
TOF_RANGE_DAYS = (130.0, 320.0)


def _search_transfer(t0_range, tof_range, pop_size, generations, seed):
    earth, mars = planet("earth"), planet("mars")
    grid = compute_porkchop(earth, mars, t0_range, tof_range, n_t0=120, n_tof=120, use_gpu=False)
    dv = np.where(grid.converged, grid.total_dv, np.inf)
    idx = np.unravel_index(int(np.argmin(dv)), dv.shape)
    seed_t0, seed_tof = float(grid.t0_grid[idx[1]]), float(grid.tof_grid[idx[0]])

    t0_bounds = (max(t0_range[0], seed_t0 - 15.0), min(t0_range[1], seed_t0 + 15.0))
    tof_bounds = (max(tof_range[0], seed_tof - 30.0), min(tof_range[1], seed_tof + 30.0))
    problem = Transfer2BodyProblem(earth, mars, t0_bounds, tof_bounds)
    x, _f, _elapsed = run_optimization(problem, pop_size=pop_size, generations=generations, seed=seed, verbose=False)
    t0, tof = float(x[0]), float(x[1])

    r1, v1p = earth.eph(pk.epoch(t0))
    r2, v2p = mars.eph(pk.epoch(t0 + tof))
    r1, v1p, r2, v2p = (np.asarray(a) for a in (r1, v1p, r2, v2p))
    res = solve_lambert_batch(r1[None, :], r2[None, :], np.array([tof * pk.DAY2SEC]), pk.MU_SUN, use_gpu=False)
    v_inf_dep = res.v1[0] - v1p
    return t0, tof, v_inf_dep


def _hyperbolic_departure_state(v_inf_dep, parking_radius_m, mu_earth):
    """Perigee (position, velocity) of the parking-orbit-radius hyperbola
    whose outgoing asymptote is v_inf_dep -- see the sign convention this
    was empirically confirmed against (rotating v_inf_hat by +nu_inf, not
    -nu_inf, around the chosen orbit normal lands the propagated asymptote
    on v_inf_hat; the other sign converges to a direction less than 51% on
    the unit-vector dot product, i.e. a different, wrong orbit)."""
    v_inf = float(np.linalg.norm(v_inf_dep))
    v_inf_hat = v_inf_dep / v_inf
    a = -mu_earth / v_inf ** 2
    e = 1.0 - parking_radius_m / a
    nu_inf = np.arccos(-1.0 / e)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v_inf_hat, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    h_hat = np.cross(v_inf_hat, ref)
    h_hat /= np.linalg.norm(h_hat)
    e_hat = v_inf_hat * np.cos(nu_inf) - np.cross(h_hat, v_inf_hat) * np.sin(nu_inf)
    e_hat /= np.linalg.norm(e_hat)
    q_hat = np.cross(h_hat, e_hat)

    r_burn = parking_radius_m * e_hat
    speed_hyp = np.sqrt(mu_earth * (2.0 / parking_radius_m - 1.0 / a))
    v_hyp = speed_hyp * q_hat
    speed_circ = np.sqrt(mu_earth / parking_radius_m)
    v_park = speed_circ * q_hat
    return r_burn, v_park, v_hyp, q_hat


def _decimate(epochs, states, points_per_arc=150):
    if len(epochs) > points_per_arc:
        keep = np.unique(np.linspace(0, len(epochs) - 1, points_per_arc).round().astype(int))
        return epochs[keep], states[keep]
    return epochs, states


def _build_mars_mission(
    seed=1,
    pop_size=160,
    generations=120,
    t0_range=None,
    tof_range=TOF_RANGE_DAYS,
    parking_altitude_km=PARKING_ALTITUDE_KM,
    mars_tcm_target_periapsis_km=MARS_TCM_TARGET_PERIAPSIS_KM,
    mars_capture_apoapsis_alt_km=MARS_CAPTURE_APOAPSIS_ALT_KM,
    max_output_points=3000,
):
    if t0_range is None:
        t0_range = (mjd2000_from_date(*DEPARTURE_WINDOW_START), mjd2000_from_date(*DEPARTURE_WINDOW_END))

    t0, tof, v_inf_dep = _search_transfer(t0_range, tof_range, pop_size, generations, seed)
    reference_et = mjd2000_to_ephemeris_seconds(t0)

    mu_earth = pk.MU_EARTH
    parking_r_m = (EARTH_RADIUS_KM + parking_altitude_km) * 1000.0
    r_burn, v_park, v_hyp, _q_hat = _hyperbolic_departure_state(v_inf_dep, parking_r_m, mu_earth)
    tmi_dv = v_hyp - v_park
    parking_period_s = 2.0 * np.pi * np.sqrt(parking_r_m ** 3 / mu_earth)

    events = [
        {"label": "Liftoff", "time": 0.0,
         "note": "Ascent to parking-orbit insertion isn't itself drawn (a two-body "
                 "propagator can't represent it) -- the trail starts already in orbit, "
                 "same as the Artemis II and GOES scenes."},
        {"label": "Parking-orbit insertion", "time": 0.0,
         "note": f"{parking_altitude_km:.0f} km circular parking orbit."},
    ]

    # --- Phase 1: Earth-centered -- parking orbit -> TMI burn -> escape ---
    escape = propagate_multi_arc(
        r_burn, v_park, reference_et,
        arcs=[
            {"type": "coast", "duration": parking_period_s},
            {"type": "impulsive_burn", "delta_v": tmi_dv},
            {"type": "coast", "duration": 12.0 * 86400.0},
        ],
        central_body="Earth", perturbing_bodies=("Earth", "Sun"), step_size=60.0,
    )
    radii_km = np.linalg.norm(escape.states[:, :3], axis=1) / 1000.0
    soi_idx = int(np.argmax(radii_km >= EARTH_SOI_RADIUS_KM))
    if radii_km[soi_idx] < EARTH_SOI_RADIUS_KM:
        raise RuntimeError("Departure hyperbola never reached Earth-SOI radius in the coast window.")
    tmi_rel_time_s = parking_period_s
    tmi_epoch_et = reference_et + tmi_rel_time_s
    tmi_r_helio = body_position_at_absolute_epoch("Earth", tmi_epoch_et, central_body="Sun") + r_burn
    events.append({
        "label": "TMI burn", "time": round(tmi_rel_time_s / 86400.0, 4),
        "note": f"apsis-preserving at the {parking_altitude_km:.0f} km parking-orbit perigee, "
                f"hyperbolic excess speed {np.linalg.norm(v_inf_dep) / 1000.0:.2f} km/s "
                f"(C3 {np.sum((v_inf_dep / 1000.0) ** 2):.1f} km^2/s^2).",
        "vector": {"position": (tmi_r_helio / 1000.0).round(1).tolist(), "deltaV": tmi_dv.round(2).tolist()},
    })

    keep_idx = np.arange(soi_idx + 1)
    phase1_epochs_rel = escape.epochs[keep_idx]  # seconds, relative to reference_et
    phase1_states = escape.states[keep_idx]
    phase1_epochs_rel, phase1_states = _decimate(phase1_epochs_rel, phase1_states)

    soi_exit_epoch_et = reference_et + float(escape.epochs[soi_idx])
    soi_exit_state_earth = escape.states[soi_idx]
    events.append({
        "label": "Earth-SOI exit", "time": round(float(escape.epochs[soi_idx]) / 86400.0, 4),
        "note": f"{EARTH_SOI_RADIUS_KM:,.0f} km from Earth center (real Earth sphere-of-influence radius)."})

    earth_r_helio = body_position_at_absolute_epoch("Earth", soi_exit_epoch_et, central_body="Sun")
    earth_v_helio = body_velocity_at_absolute_epoch("Earth", soi_exit_epoch_et, central_body="Sun")
    sc_r_helio = earth_r_helio + soi_exit_state_earth[:3]
    sc_v_helio = earth_v_helio + soi_exit_state_earth[3:]

    phase1_helio_positions_km = np.empty((len(phase1_epochs_rel), 3))
    for i, t_rel in enumerate(phase1_epochs_rel):
        et = reference_et + float(t_rel)
        earth_pos_m = body_position_at_absolute_epoch("Earth", et, central_body="Sun")
        phase1_helio_positions_km[i] = (earth_pos_m + phase1_states[i, :3]) / 1000.0

    # --- Phase 2: heliocentric cruise, with a real trajectory-correction burn ---
    remaining_tof_s = tof * 86400.0 - float(escape.epochs[soi_idx])
    tcm = target_mars_approach(
        sc_r_helio, sc_v_helio, soi_exit_epoch_et,
        coast_duration_guess=remaining_tof_s,
        target_periapsis_km=mars_tcm_target_periapsis_km,
    )
    events.append({
        "label": "Trajectory-correction maneuver", "time": round((soi_exit_epoch_et - reference_et) / 86400.0, 4),
        "note": f"closes the gap between the idealized Lambert departure and the real, "
                f"perturbed trajectory -- {'converged' if tcm.converged else 'best effort, did not fully converge'} "
                f"to {tcm.final_miss_km:,.0f} km from Mars center.",
        "vector": {"position": (sc_r_helio / 1000.0).round(1).tolist(), "deltaV": tcm.delta_v.round(2).tolist()},
    })

    # --- Phase 3 setup: locate the true periapsis under real perturbed
    # propagation. This also gives us the whole cruise-leg trail -- reusing
    # locate_mars_periapsis's own bulk+fine trajectory (see its docstring)
    # instead of separately re-propagating one keeps the exported trail a
    # single continuous numerical solution, with an exact (not merely close)
    # seam into phase 3: both start from the identical periapsis state.
    periapsis = locate_mars_periapsis(
        sc_r_helio, sc_v_helio, soi_exit_epoch_et, tcm.delta_v, tcm.coast_duration,
    )
    r_rel, v_rel, arrival_epoch_et = periapsis.r_rel, periapsis.v_rel, periapsis.arrival_epoch_et

    cruise_epochs_rel = periapsis.trajectory_epochs_et - reference_et  # absolute -> mission-relative seconds
    cruise_positions_km = periapsis.trajectory_positions_m / 1000.0  # central_body="Sun" -> already heliocentric
    cruise_epochs_rel, cruise_positions_km = _decimate(cruise_epochs_rel, cruise_positions_km, points_per_arc=500)

    # Mars-SOI-entry event: first cruise sample inside Mars's real sphere of
    # influence (~577,000 km) -- same threshold-crossing search phase1 uses
    # for Earth-SOI exit, just shrinking instead of growing.
    cruise_mars_dist_km = np.array([
        np.linalg.norm(cruise_positions_km[i] * 1000.0 - body_position_at_absolute_epoch(
            "Mars", reference_et + float(cruise_epochs_rel[i]), central_body="Sun")) / 1000.0
        for i in range(len(cruise_epochs_rel))
    ])
    if np.any(cruise_mars_dist_km <= MARS_SOI_RADIUS_KM):
        soi_entry_idx = int(np.argmax(cruise_mars_dist_km <= MARS_SOI_RADIUS_KM))
        soi_entry_time_days = round(float(cruise_epochs_rel[soi_entry_idx]) / 86400.0, 4)
    else:
        soi_entry_time_days = round((arrival_epoch_et - reference_et) / 86400.0, 4)

    periapsis_km = float(np.linalg.norm(r_rel)) / 1000.0
    moi = verify_mars_orbit_insertion(
        r_rel, v_rel, periapsis_km=periapsis_km, apoapsis_km=periapsis_km + mars_capture_apoapsis_alt_km,
        arrival_epoch_et=arrival_epoch_et,
    )
    events.append({
        "label": "Mars-SOI entry / approach", "time": soi_entry_time_days,
        "note": f"crosses Mars's real sphere-of-influence radius ({MARS_SOI_RADIUS_KM:,.0f} km), "
                "hyperbolic approach closing on periapsis."})

    a_captured_km = moi.achieved["a_km"]
    mu_mars = body_gravitational_parameter("Mars")
    capture_period_s = 2.0 * np.pi * np.sqrt((a_captured_km * 1000.0) ** 3 / mu_mars)

    moi_r_helio = body_position_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun") + r_rel
    events.append({
        "label": "Mars orbit insertion", "time": round((arrival_epoch_et - reference_et) / 86400.0, 4),
        "note": f"apsis-preserving capture burn at {periapsis_km:,.0f} km from Mars center "
                f"({periapsis_km - MARS_RADIUS_KM:,.0f} km altitude) -> "
                f"{a_captured_km:,.0f} x {moi.achieved['e']:.3f} captured orbit. "
                "Mission ends here -- this framework has no atmosphere model, so a descent "
                "to the surface isn't simulated (see the module docstring).",
        "vector": {"position": (moi_r_helio / 1000.0).round(1).tolist(), "deltaV": moi.delta_v_ms.round(2).tolist()},
    })

    # The cruise trail above already carries the trajectory through periapsis
    # itself, so phase3 only needs the MOI burn and one full lap of the
    # captured orbit -- same "show one full lap, not just the insertion
    # instant" payoff geo_raising.py's final burn uses.
    capture = propagate_multi_arc(
        r_rel, v_rel, arrival_epoch_et,
        arcs=[
            {"type": "impulsive_burn", "delta_v": moi.delta_v_ms},
            {"type": "coast", "duration": capture_period_s},
        ],
        central_body="Mars", perturbing_bodies=("Mars", "Sun"), step_size=30.0,
    )
    phase3_epochs_rel = capture.epochs + (arrival_epoch_et - reference_et)
    phase3_states = capture.states
    phase3_epochs_rel, phase3_states = _decimate(phase3_epochs_rel, phase3_states, points_per_arc=300)

    phase3_helio_positions_km = np.empty((len(phase3_epochs_rel), 3))
    for i, t_rel in enumerate(phase3_epochs_rel):
        et = reference_et + float(t_rel)
        phase3_helio_positions_km[i] = (
            body_position_at_absolute_epoch("Mars", et, central_body="Sun") + phase3_states[i, :3]
        ) / 1000.0

    # --- Assemble the one continuous heliocentric-km trail ---
    all_epochs_rel = np.concatenate([phase1_epochs_rel, cruise_epochs_rel, phase3_epochs_rel])
    all_positions_km = np.concatenate([phase1_helio_positions_km, cruise_positions_km, phase3_helio_positions_km])

    if len(all_epochs_rel) > max_output_points:
        keep = np.unique(np.linspace(0, len(all_epochs_rel) - 1, max_output_points).round().astype(int))
        all_epochs_rel = all_epochs_rel[keep]
        all_positions_km = all_positions_km[keep]

    days = (all_epochs_rel / 86400.0).round(4).tolist()
    spacecraft_positions = all_positions_km.round(1).tolist()

    events.sort(key=lambda e: e["time"])

    total_days = float(all_epochs_rel[-1] / 86400.0)
    earth_orbit_days = np.linspace(0.0, total_days, 240)
    earth_track = np.array([
        body_position_at_absolute_epoch("Earth", reference_et + d * 86400.0, central_body="Sun") for d in earth_orbit_days
    ]) / 1000.0
    mars_track = np.array([
        body_position_at_absolute_epoch("Mars", reference_et + d * 86400.0, central_body="Sun") for d in earth_orbit_days
    ]) / 1000.0

    bodies = [
        body_entry("sun", "Sun", COLOR["sun"], "sun", radius_display=RADIUS_DISPLAY["sun"],
                   position=[0.0, 0.0, 0.0], texture=TEXTURE["sun"], radius=696000.0,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["sun"]),
        body_entry("earth", "Earth", COLOR["earth"], "planet", radius_display=RADIUS_DISPLAY["earth"],
                   trail={"times": earth_orbit_days.round(4).tolist(), "positions": earth_track.round(1).tolist()},
                   texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["earth"]),
        body_entry("mars", "Mars", COLOR["mars"], "planet", radius_display=RADIUS_DISPLAY["mars"],
                   trail={"times": earth_orbit_days.round(4).tolist(), "positions": mars_track.round(1).tolist()},
                   texture=TEXTURE["mars"], radius=MARS_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["mars"]),
        body_entry(
            "spacecraft", "Mars probe", COLOR["spacecraft"], "spacecraft",
            radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": days, "positions": spacecraft_positions},
            info=[
                {"label": "TMI delta-v", "value": f"{np.linalg.norm(tmi_dv):.0f} m/s"},
                {"label": "TCM delta-v", "value": f"{np.linalg.norm(tcm.delta_v):.0f} m/s"},
                {"label": "MOI delta-v", "value": f"{np.linalg.norm(moi.delta_v_ms):.0f} m/s"},
                {"label": "Departure C3", "value": f"{np.sum((v_inf_dep / 1000.0) ** 2):.1f} km^2/s^2"},
                {"label": "Captured orbit", "value": f"{a_captured_km:,.0f} km semi-major axis, e={moi.achieved['e']:.3f}"},
            ],
        ),
    ]

    return scene_document(
        scene_id="mars-transfer",
        title="Mars — Earth to Orbit Insertion",
        subtitle="Real Earth departure window, GPU-screened + optimized Lambert transfer, a real\n"
                  "trajectory-correction maneuver, and tudatpy-verified Mars orbit insertion --\n"
                  "continuously propagated, heliocentric throughout. Ends at Mars orbit insertion,\n"
                  "not a landing: this framework has no atmosphere model.",
        distance_unit="km",
        central_body_id="sun",
        bodies=bodies,
        timeline={
            "unitLabel": "days",
            "min": 0.0,
            "max": round(total_days, 4),
            "events": events,
            "referenceEpochEt": reference_et,
        },
    )


def compute_and_export_mars_mission(**kwargs) -> dict:
    """Public entry point (mirrors compute_and_export_geo_mission): optimize
    a real Earth->Mars transfer window, fly the full launch-to-orbit-insertion
    trajectory, and export it as a SceneData document. See the module
    docstring for what this does and does not model."""
    return _build_mars_mission(**kwargs)


def build_from_config(config: dict) -> dict:
    """orbitopt.mission_config adapter for mission.kind == "mars-transfer"."""
    mission = config["mission"]
    launch_window = config.get("launch_window", {})
    parking_orbit = config.get("parking_orbit", {})
    tof_search = config.get("tof_search", {})
    mars_arrival = config.get("mars_arrival_orbit", {})
    optimizer = config.get("campaign_search", {}).get("optimizer", {})

    kwargs = {}
    if "start_date" in launch_window and "end_date" in launch_window:
        kwargs["t0_range"] = (
            mjd2000_from_date(*launch_window["start_date"]),
            mjd2000_from_date(*launch_window["end_date"]),
        )
    if "altitude_km" in parking_orbit:
        kwargs["parking_altitude_km"] = parking_orbit["altitude_km"]
    if "min_days" in tof_search and "max_days" in tof_search:
        kwargs["tof_range"] = (tof_search["min_days"], tof_search["max_days"])
    if "periapsis_km" in mars_arrival:
        kwargs["mars_tcm_target_periapsis_km"] = mars_arrival["periapsis_km"]
    if "apoapsis_altitude_km" in mars_arrival:
        kwargs["mars_capture_apoapsis_alt_km"] = mars_arrival["apoapsis_altitude_km"]
    if "population" in optimizer:
        kwargs["pop_size"] = optimizer["population"]
    if "generations" in optimizer:
        kwargs["generations"] = optimizer["generations"]
    if "seed" in optimizer:
        kwargs["seed"] = optimizer["seed"]

    scene = _build_mars_mission(**kwargs)
    scene["id"] = mission["id"]
    scene["title"] = mission["title"]
    return scene
