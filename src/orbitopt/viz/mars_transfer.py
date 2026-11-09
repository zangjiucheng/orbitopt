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

import logging
import time

import numpy as np
import pykep as pk

from orbitopt.bodies import (
    body_fixed_to_eclipj2000_matrix,
    mjd2000_from_date,
    mjd2000_to_ephemeris_seconds,
    planet,
)
from orbitopt.candidate_selection import (
    mark_scene_best_effort,
    select_candidates,
    walk_candidates,
)
from orbitopt.injection import equatorial_to_ecliptic
from orbitopt.lambert.gpu_batch import solve_lambert_batch
from orbitopt.launch_injection import LaunchSpec, launch_to_injection
from orbitopt.optimize.multistart import SINGLE_BASIN_HALF_WIDTH, find_diverse_basins, multistart_optimize
from orbitopt.pipeline import CONSTRAINTS, OPTIMIZE, REFINE, SEARCH, SELECT, VERIFY, StageRecorder
from orbitopt.problems.transfer_2body import Transfer2BodyProblem
from orbitopt.timescales import mjd2000_to_utc_datetime
from orbitopt.verify import report as verification_report
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
# A located periapsis below this altitude means the flown trajectory hits (or
# grazes the atmosphere of) Mars; such a candidate is rejected, never exported.
MARS_MIN_PERIAPSIS_ALT_KM = 100.0
MARS_CAPTURE_APOAPSIS_ALT_KM = 20000.0

# Real, near-term Earth->Mars Hohmann-class transfer window (synodic period
# ~780 days; recent windows ~2020-07, ~2022-09, ~2024-10 -> next ~2026 Q4).
DEPARTURE_WINDOW_START = (2026, 9, 1)
DEPARTURE_WINDOW_END = (2027, 2, 1)
TOF_RANGE_DAYS = (130.0, 320.0)

# Multi-basin window search (see orbitopt.optimize.multistart): how many
# distinct porkchop valleys get their own independent PSO run, how far apart
# (departure days, TOF days) two valleys must be to count as distinct, and
# how many ranked candidates tudatpy gets to try before falling back.
MULTISTART_MAX_BASINS = 4
MULTISTART_BASIN_SEPARATION = (20.0, 40.0)
MAX_TUDAT_CANDIDATES = 3
# Candidate Selection (orbitopt.candidate_selection): two optima closer than
# this (departure days, TOF days) are the same trajectory.
SELECTION_DUPLICATE_TOLERANCE = (1.0, 2.0)

logger = logging.getLogger(__name__)


def _search_transfer(
    t0_range,
    tof_range,
    pop_size,
    generations,
    seed,
    max_basins=MULTISTART_MAX_BASINS,
    basin_separation=MULTISTART_BASIN_SEPARATION,
    launch_constraints=None,
    recorder=None,
):
    """Multi-basin search for the Earth->Mars Lambert transfer.

    Screens the whole (t0, tof) window with a porkchop grid, keeps up to
    ``max_basins`` spatially distinct local minima (not just the single grid
    minimum -- adjacent windows and type-I/type-II families are separate
    valleys whose grid minima are often within discretization error of each
    other), runs one independent PSO per basin (box = the old single-basin
    +-15 d / +-30 d recipe around each) plus one PSO over the full window,
    and returns every distinct optimum, best first, together with the
    MultistartReport recording how this compares to the single-basin result.

    ``launch_constraints`` (orbitopt.launch_constraints.LaunchConstraints,
    ecliptic frame) is applied to every PSO fitness -- the porkchop screen
    stays unconstrained, it only seeds the basins.

    Returns ``(candidates, report)`` with ``candidates`` a list of
    ``(t0, tof, v_inf_dep)`` tuples ranked by Lambert total dv.
    """
    recorder = StageRecorder() if recorder is None else recorder
    earth, mars = planet("earth"), planet("mars")
    with recorder.stage(SEARCH) as st:
        grid = compute_porkchop(earth, mars, t0_range, tof_range, n_t0=120, n_tof=120, use_gpu=False)
        dv = np.where(grid.converged, grid.total_dv, np.inf)
        basins = find_diverse_basins(
            dv, grid.t0_grid, grid.tof_grid, max_basins=max_basins, min_separation=basin_separation,
        )
        st.physical_model = {"name": "Sun-centered Lambert porkchop grid (pykep ephemerides)",
                             "fidelity": "patched-conic", "grid": [120, 120]}
        st.candidates = [{"t0_mjd2000": b.x, "tof_days": b.y, "total_dv_km_s": b.value, "rank": b.rank}
                         for b in basins]
        st.convergence = {"converged": bool(basins), "grid_converged_fraction": float(np.mean(grid.converged))}

    opt = recorder[OPTIMIZE]
    t_opt = time.perf_counter()
    report = multistart_optimize(
        lambda t0_bounds, tof_bounds: Transfer2BodyProblem(
            earth, mars, t0_bounds, tof_bounds, launch_constraints=launch_constraints),
        basins,
        full_bounds=(tuple(t0_range), tuple(tof_range)),
        half_width=SINGLE_BASIN_HALF_WIDTH,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
    )
    report.notes.append(f"porkchop grid minimum {basins[0].value:.3f} km/s at t0={basins[0].x:.1f}, tof={basins[0].y:.1f}")
    logger.info("Earth->Mars window search:\n%s", report.summary(unit="m/s"))
    recorder.add_time(OPTIMIZE, time.perf_counter() - t_opt)
    opt.physical_model = {"name": "Sun-centered Lambert transfer, multistart PSO (Transfer2BodyProblem)",
                          "fidelity": "patched-conic"}
    opt.candidates = [{"t0_mjd2000": c.x, "tof_days": c.y, "total_dv_m_s": c.value, "basin_rank": c.basin_rank,
                       "merged_runs": c.merged_runs} for c in report.candidates]
    opt.convergence = {"converged": bool(report.candidates), "runs": len(report.runs),
                       "distinct_candidates": len(report.candidates),
                       "improvement_vs_single_basin_m_s": report.improvement}
    opt.notes.extend(report.notes)

    candidates = [(c.x, c.y, _lambert_v_inf_dep(earth, mars, c.x, c.y)) for c in report.candidates]
    return candidates, report


def _lambert_v_inf_dep(earth, mars, t0, tof):
    """Departure excess velocity [m/s, heliocentric ecliptic] of the
    Earth->Mars Lambert transfer leaving at t0 [MJD2000] with tof [days]."""
    r1, v1p = earth.eph(pk.epoch(t0))
    r2, v2p = mars.eph(pk.epoch(t0 + tof))
    r1, v1p, r2, v2p = (np.asarray(a) for a in (r1, v1p, r2, v2p))
    res = solve_lambert_batch(r1[None, :], r2[None, :], np.array([tof * pk.DAY2SEC]), pk.MU_SUN, use_gpu=False)
    return res.v1[0] - v1p


def _search_note(report, flown_index, rejected):
    """One human-readable sentence recording the multistart search outcome
    (it ends up in the exported scene's TMI event note)."""
    if report is None:
        return "Transfer pinned by the config's launch.departure (no window search)."
    best = report.candidates[flown_index]
    parts = [
        f"Window chosen by multi-basin search: {len(report.basins)} porkchop basins, "
        f"{len(report.runs)} independent PSO runs, {len(report.candidates)} distinct candidates; "
        f"flying candidate #{flown_index} ({best.value / 1000.0:.3f} km/s Lambert total)"
    ]
    if report.baseline is not None:
        parts.append(f"single-basin baseline {report.baseline.value / 1000.0:.3f} km/s")
    if rejected:
        parts.append("rejected by tudatpy verification: " + ", ".join(f"#{i}" for i in rejected))
    return "; ".join(parts) + "."


class _CandidateRejected(RuntimeError):
    """A Lambert candidate whose high-fidelity (tudatpy) flight did not hold up."""


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
    max_tudat_candidates=MAX_TUDAT_CANDIDATES,
    launch: LaunchSpec | None = None,
    recorder=None,
):
    """``launch`` (orbitopt.launch_injection.LaunchSpec) switches on the
    real launch: the search is launch-constrained (vehicle C3 + site DLA),
    or skipped for a pinned ``launch.departure``, and every candidate is
    flown from a launch-window lift-off into a site-reachable parking orbit
    with a numerically verified TMI burn. ``None`` (the default) is the
    original idealized-parking-orbit pipeline, unchanged."""
    recorder = StageRecorder() if recorder is None else recorder
    if t0_range is None:
        t0_range = (mjd2000_from_date(*DEPARTURE_WINDOW_START), mjd2000_from_date(*DEPARTURE_WINDOW_END))

    with recorder.stage(CONSTRAINTS) as st:
        launch_constraints = None if launch is None else launch.constraints(frame="ecliptic")
        st.physical_model = {"name": "departure window + TOF box" + ("" if launch is None else
                                                                       ", launch vehicle C3 + site DLA limits"),
                             "fidelity": "analytic",
                             "t0_range_mjd2000": list(t0_range), "tof_range_days": list(tof_range),
                             "launch_limits": None if launch_constraints is None else launch_constraints.describe()}

    if launch is not None and launch.pinned:
        t0, tof = launch.departure_mjd2000, launch.tof_days
        candidates, report = [(t0, tof, _lambert_v_inf_dep(planet("earth"), planet("mars"), t0, tof))], None
        recorder.skip(SEARCH, "transfer pinned by the config's launch.departure")
        recorder.skip(OPTIMIZE, "transfer pinned by the config's launch.departure")
    else:
        candidates, report = _search_transfer(
            t0_range, tof_range, pop_size, generations, seed,
            launch_constraints=launch_constraints, recorder=recorder,
        )
    shortlist = _select_candidates(candidates, report, max(1, max_tudat_candidates), recorder,
                                   pinned=launch is not None and launch.pinned)
    fly_kwargs = dict(
        parking_altitude_km=parking_altitude_km if launch is None else launch.parking_altitude_km,
        mars_tcm_target_periapsis_km=mars_tcm_target_periapsis_km,
        mars_capture_apoapsis_alt_km=mars_capture_apoapsis_alt_km,
        max_output_points=max_output_points,
        recorder=recorder,
    )
    refine = recorder[REFINE]
    refine.physical_model = {"name": "tudatpy N-body (Earth/Sun escape, Sun-centered cruise with TCM, "
                                     "Mars capture), fixed-step RK4", "fidelity": "high-fidelity"}
    gate = recorder.gate

    def violations(v_inf_dep):
        if launch_constraints is None:
            return {}
        ev = launch_constraints.evaluate(np.asarray(v_inf_dep)[None, :])
        return {"launch_c3_km2_s2": float(ev.c3_excess[0]), "launch_dla_deg": float(ev.dla_excess_deg[0])}

    def refine_attempt(i, cand, strict, rejected):
        t0, tof, v_inf_dep = cand
        entry = {"index": i, "t0_mjd2000": float(t0), "tof_days": float(tof), "selected": False}
        refine.candidates.append(entry)
        verify_s = recorder[VERIFY].elapsed_s
        n_verified = len(recorder[VERIFY].candidates)
        note = _search_note(report, i, rejected)
        if not strict:
            refine.notes.append("no candidate fully converged; flying the Lambert-best one best-effort")
            note += " No candidate fully converged; flying the Lambert-best one."
        try:
            with recorder.stage(REFINE):
                return _fly_transfer(t0, tof, v_inf_dep, **injection_for(t0, v_inf_dep),
                                     require_tcm_converged=strict, search_note=note, **fly_kwargs)
        except RuntimeError as exc:
            entry["rejected"] = f"refine: {exc}"
            raise _CandidateRejected(entry["rejected"]) from exc
        finally:
            # The MOI verification runs inside _fly_transfer; keep stage times disjoint.
            refine.elapsed_s -= recorder[VERIFY].elapsed_s - verify_s
            for v in recorder[VERIFY].candidates[n_verified:]:
                v["index"] = i

    def injection_for(t0, v_inf_dep):
        if launch is None:
            return {}
        try:
            return {"injection": launch_to_injection(launch, v_inf_dep, t0)}
        except ValueError as exc:  # no launch window from the site for this asymptote
            raise _CandidateRejected(str(exc)) from exc

    # Verification Gate: hand the selected candidates to tudatpy one by one.
    # A candidate whose real, perturbed flight doesn't close (escape reaches
    # the SOI, TCM converges, periapsis above the floor) is rejected; one that
    # flies but whose verification reports are failed / not_checked is
    # degraded and never counts as a success -- the next one is tried for a
    # candidate that passes. Only if none passes is the best-ranked degraded
    # one (or, every flight rejected, the Lambert-best one re-flown without
    # requiring TCM convergence) exported best-effort, with a loud warning.
    outcome = walk_candidates(
        [(i, (t0, tof, v)) for i, t0, tof, v in shortlist],
        refine_attempt,
        gate=gate,
        reports_of=lambda scene: scene.get(verification_report.SCENE_REPORTS_KEY, []),
    )
    # Trace every attempt's verdict on the refine / verify stage records.
    verdicts = {c["index"]: c for c in gate.candidates}
    attempts = refine.candidates
    for e in attempts + recorder[VERIFY].candidates:
        g = verdicts.get(e.get("index"))
        if g is not None:
            e["verdict"], e["verification"] = g["verdict"], g["verification"]
        e["selected"] = False
    for e in reversed(attempts):  # the exported flight is the last attempt of that index
        if e["index"] == outcome.index and "rejected" not in e:
            e["selected"] = True
            break
    for e in reversed(recorder[VERIFY].candidates):
        if e.get("index") == outcome.index:
            e["selected"] = True
            break
    if outcome.best_effort:
        refine.notes.append(f"no candidate passed verification; candidate #{outcome.index} exported best-effort")
    refine.constraint_violations = violations(outcome.payload[2])
    return mark_scene_best_effort(outcome.result, gate)


def _select_candidates(candidates, report, max_candidates, recorder, pinned=False):
    """Candidate Selection stage: dedupe, rank and diversity-filter the
    optimizer's ``(t0, tof, v_inf_dep)`` candidates into the short list
    tudatpy flies. Returns ``[(search_index, t0, tof, v_inf_dep), ...]``,
    best first; every candidate's fate is recorded on the select stage and
    the selected ones are registered with the Verification Gate."""
    gate = recorder.gate
    if pinned or report is None:
        recorder.skip(SELECT, "transfer pinned by the config's launch.departure")
        t0, tof, v = candidates[0]
        gate.register(0, t0_mjd2000=float(t0), tof_days=float(tof), source="pinned launch.departure")
        return [(0, t0, tof, v)]
    items = [(i, t0, tof, v, float(report.candidates[i].value)) for i, (t0, tof, v) in enumerate(candidates)]
    with recorder.stage(SELECT) as st:
        sel = select_candidates(
            items,
            cost=lambda it: it[4],
            coords=lambda it: (it[1], it[2]),
            tolerance=SELECTION_DUPLICATE_TOLERANCE,
            min_separation=MULTISTART_BASIN_SEPARATION,
            max_candidates=max_candidates,
            describe=lambda it: {"t0_mjd2000": float(it[1]), "tof_days": float(it[2]),
                                 "lambert_total_dv_m_s": it[4]},
        )
        st.physical_model = {"name": "candidate selection: dedupe (tolerance d), rank by Lambert total dv, "
                                     "diversity retention (min separation d)", "fidelity": "patched-conic",
                             "duplicate_tolerance_days": list(SELECTION_DUPLICATE_TOLERANCE),
                             "min_separation_days": list(MULTISTART_BASIN_SEPARATION),
                             "max_candidates": int(max_candidates)}
        st.candidates = sel.to_candidates()
        st.convergence = {"converged": bool(sel.selected), **sel.counts()}
        for rec in st.candidates:
            if not rec["selected"]:
                st.notes.append(f"candidate #{rec['index']} {rec['status']}: {rec['reason']}")
        if not sel.selected:
            raise RuntimeError("candidate selection kept no feasible candidate")
    for rank, (i, t0, tof, _v, dv) in enumerate(sel.selected):
        gate.register(i, t0_mjd2000=float(t0), tof_days=float(tof), lambert_total_dv_m_s=dv, selection_rank=rank)
    return [(i, t0, tof, v) for i, t0, tof, v, _dv in sel.selected]


def _fly_transfer(
    t0,
    tof,
    v_inf_dep,
    *,
    parking_altitude_km,
    mars_tcm_target_periapsis_km,
    mars_capture_apoapsis_alt_km,
    max_output_points,
    require_tcm_converged,
    search_note="",
    injection=None,
    recorder=None,
):
    """Fly one Lambert candidate through the real tudatpy pipeline and
    export it as a scene. Raises _CandidateRejected when
    ``require_tcm_converged`` and the TCM targeter did not converge.

    ``injection`` (orbitopt.launch_injection.LaunchInjection) replaces the
    idealized parking orbit by the real launch: the timeline starts at the
    launch-window lift-off, the trail at parking-orbit insertion into the
    site-reachable plane, and the TMI is the numerically verified burn."""
    mu_earth = pk.MU_EARTH
    if injection is None:
        reference_et = mjd2000_to_ephemeris_seconds(t0)
        parking_r_m = (EARTH_RADIUS_KM + parking_altitude_km) * 1000.0
        r_burn, v_park, v_hyp, _q_hat = _hyperbolic_departure_state(v_inf_dep, parking_r_m, mu_earth)
        tmi_dv = v_hyp - v_park
        parking_coast_s = 2.0 * np.pi * np.sqrt(parking_r_m ** 3 / mu_earth)
        phase1_r0, phase1_v0, phase1_start_et = r_burn, v_park, reference_et
        phase1_start_rel_s = 0.0  # parking-orbit insertion, relative to reference_et
        departure_offset_s = 0.0  # reference_et - (Lambert departure t0)
        events = [
            {"label": "Liftoff", "time": 0.0,
             "note": "Ascent to parking-orbit insertion isn't itself drawn (a two-body "
                     "propagator can't represent it) -- the trail starts already in orbit, "
                     "same as the Artemis II and GOES scenes."},
            {"label": "Parking-orbit insertion", "time": 0.0,
             "note": f"{parking_altitude_km:.0f} km circular parking orbit."},
        ]
    else:
        plan, ver, spec = injection.plan, injection.verification, injection.spec
        reference_et = mjd2000_to_ephemeris_seconds(plan.launch.liftoff_mjd2000)
        phase1_start_et = mjd2000_to_ephemeris_seconds(plan.launch.insertion_mjd2000)
        phase1_start_rel_s = phase1_start_et - reference_et
        departure_offset_s = reference_et - mjd2000_to_ephemeris_seconds(t0)
        state_ins = equatorial_to_ecliptic(plan.orbit.state_at(plan.orbit.epoch_mjd2000))
        phase1_r0, phase1_v0 = state_ins[:3], state_ins[3:]
        parking_coast_s = float(plan.burn.coast_s)
        tmi_dv = equatorial_to_ecliptic(ver.delta_v_m_s)
        r_burn = equatorial_to_ecliptic(plan.burn.pre_burn_state[:3])
        liftoff_utc = mjd2000_to_utc_datetime(plan.launch.liftoff_mjd2000)
        vehicle = spec.vehicle or "unspecified vehicle"
        events = [
            {"label": "Liftoff", "time": 0.0,
             "note": f"{vehicle} lifts off from {plan.launch.site_id} at {liftoff_utc:%Y-%m-%d %H:%M:%S} UTC "
                     f"(optimum of the {injection.window.direction} launch window, azimuth "
                     f"{plan.launch.azimuth_deg:.1f} deg), {spec.payload_kg:,.0f} kg payload. The ascent "
                     "itself isn't drawn -- the trail starts at parking-orbit insertion."},
            {"label": "Parking-orbit insertion", "time": round(phase1_start_rel_s / 86400.0, 4),
             "note": f"{plan.orbit.altitude_km:.0f} km circular parking orbit, inclination "
                     f"{plan.orbit.inclination_deg:.2f} deg (plane through the site and the departure asymptote)."},
        ]

    # --- Phase 1: Earth-centered -- parking orbit -> TMI burn -> escape ---
    escape = propagate_multi_arc(
        phase1_r0, phase1_v0, phase1_start_et,
        arcs=[
            {"type": "coast", "duration": parking_coast_s},
            {"type": "impulsive_burn", "delta_v": tmi_dv},
            {"type": "coast", "duration": 12.0 * 86400.0},
        ],
        central_body="Earth", perturbing_bodies=("Earth", "Sun"), step_size=60.0,
    )
    escape_epochs_rel = escape.epochs + phase1_start_rel_s  # seconds, relative to reference_et
    radii_km = np.linalg.norm(escape.states[:, :3], axis=1) / 1000.0
    soi_idx = int(np.argmax(radii_km >= EARTH_SOI_RADIUS_KM))
    if radii_km[soi_idx] < EARTH_SOI_RADIUS_KM:
        raise RuntimeError("Departure hyperbola never reached Earth-SOI radius in the coast window.")
    tmi_rel_time_s = phase1_start_rel_s + parking_coast_s
    tmi_epoch_et = reference_et + tmi_rel_time_s
    tmi_r_helio = body_position_at_absolute_epoch("Earth", tmi_epoch_et, central_body="Sun") + r_burn
    if injection is None:
        tmi_note = (f"apsis-preserving at the {parking_altitude_km:.0f} km parking-orbit perigee, "
                    f"hyperbolic excess speed {np.linalg.norm(v_inf_dep) / 1000.0:.2f} km/s "
                    f"(C3 {np.sum((v_inf_dep / 1000.0) ** 2):.1f} km^2/s^2). {search_note}")
    else:
        tmi_note = (f"after a {parking_coast_s / 60.0:.1f} min parking-orbit coast, hyperbolic excess speed "
                    f"{np.linalg.norm(v_inf_dep) / 1000.0:.2f} km/s (C3 {injection.c3_km2_s2:.2f} km^2/s^2, "
                    f"RLA {injection.rla_deg:.2f} deg, DLA {injection.dla_deg:.2f} deg; vehicle limit "
                    f"{spec.constraints().max_c3:.2f} km^2/s^2). Burn Newton-corrected to "
                    f"{ver.final_error:.3f} {ver.error_units} v-infinity residual "
                    f"(launch-injection verification {'passed' if ver.passed else 'NOT passed'}). {search_note}")
    events.append({
        "label": "TMI burn", "time": round(tmi_rel_time_s / 86400.0, 4),
        "note": tmi_note.rstrip(),
        "vector": {"position": (tmi_r_helio / 1000.0).round(1).tolist(), "deltaV": tmi_dv.round(2).tolist()},
    })

    keep_idx = np.arange(soi_idx + 1)
    phase1_epochs_rel = escape_epochs_rel[keep_idx]  # seconds, relative to reference_et
    phase1_states = escape.states[keep_idx]
    phase1_epochs_rel, phase1_states = _decimate(phase1_epochs_rel, phase1_states)

    soi_exit_epoch_et = reference_et + float(escape_epochs_rel[soi_idx])
    soi_exit_state_earth = escape.states[soi_idx]
    events.append({
        "label": "Earth-SOI exit", "time": round(float(escape_epochs_rel[soi_idx]) / 86400.0, 4),
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
    remaining_tof_s = tof * 86400.0 - departure_offset_s - float(escape_epochs_rel[soi_idx])
    tcm = target_mars_approach(
        sc_r_helio, sc_v_helio, soi_exit_epoch_et,
        coast_duration_guess=remaining_tof_s,
        target_periapsis_km=mars_tcm_target_periapsis_km,
    )
    recorder = StageRecorder() if recorder is None else recorder
    refine = recorder[REFINE]
    refine.convergence = {"converged": bool(tcm.converged), "tcm_iterations": int(tcm.iterations)}
    refine.numerical_error = {
        "tcm_final_miss_km": float(tcm.final_miss_km),
        "tcm_periapsis_error_km": float(abs(tcm.predicted_periapsis_km - mars_tcm_target_periapsis_km)),
    }
    if require_tcm_converged and not tcm.converged:
        raise _CandidateRejected(f"TCM did not converge (final miss {tcm.final_miss_km:,.0f} km)")
    events.append({
        "label": "Trajectory-correction maneuver", "time": round((soi_exit_epoch_et - reference_et) / 86400.0, 4),
        "note": f"closes the gap between the idealized Lambert departure and the real, "
                f"perturbed trajectory -- {'converged' if tcm.converged else 'best effort, did not fully converge'} "
                f"to a {tcm.predicted_periapsis_km:,.0f} km Mars periapsis radius "
                f"({tcm.predicted_periapsis_km - MARS_RADIUS_KM:,.0f} km altitude).",
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
    if periapsis_km - MARS_RADIUS_KM < MARS_MIN_PERIAPSIS_ALT_KM:
        raise _CandidateRejected(
            f"located Mars periapsis {periapsis_km:,.0f} km ({periapsis_km - MARS_RADIUS_KM:,.0f} km altitude) "
            f"is below the {MARS_MIN_PERIAPSIS_ALT_KM:g} km floor -- the trajectory would hit Mars")
    refine.numerical_error["located_periapsis_error_km"] = float(abs(periapsis_km - mars_tcm_target_periapsis_km))
    with recorder.stage(VERIFY) as st:
        moi = verify_mars_orbit_insertion(
            r_rel, v_rel, periapsis_km=periapsis_km, apoapsis_km=periapsis_km + mars_capture_apoapsis_alt_km,
            arrival_epoch_et=arrival_epoch_et,
        )
        st.physical_model = {"name": "tudatpy Mars point mass + Sun, MOI capture-burn targeting; "
                                     "verification reports for launch injection / TCM / MOI",
                             "fidelity": "high-fidelity"}
        st.candidates.append({"captured_orbit": moi.achieved,
                              "moi_delta_v_m_s": float(np.linalg.norm(moi.delta_v_ms)), "selected": False})
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
                   rotation_period_hours=ROTATION_PERIOD_HOURS["sun"],
                   orientation_basis=body_fixed_to_eclipj2000_matrix("Sun", t0).tolist()),
        body_entry("earth", "Earth", COLOR["earth"], "planet", radius_display=RADIUS_DISPLAY["earth"],
                   trail={"times": earth_orbit_days.round(4).tolist(), "positions": earth_track.round(1).tolist()},
                   texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["earth"],
                   orientation_basis=body_fixed_to_eclipj2000_matrix("Earth", t0).tolist()),
        body_entry("mars", "Mars", COLOR["mars"], "planet", radius_display=RADIUS_DISPLAY["mars"],
                   trail={"times": earth_orbit_days.round(4).tolist(), "positions": mars_track.round(1).tolist()},
                   texture=TEXTURE["mars"], radius=MARS_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["mars"],
                   orientation_basis=body_fixed_to_eclipj2000_matrix("Mars", t0).tolist()),
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
            ] + ([] if injection is None else [
                {"label": "Launch", "value": f"{spec.vehicle or '-'} from {plan.launch.site_id}, "
                                             f"{liftoff_utc:%Y-%m-%d %H:%M} UTC"},
            ]),
        ),
    ]

    scene = scene_document(
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
    # Verification reports (task #137) ride along for `orbitopt run`, which
    # writes them next to the scene; compute_and_export strips them.
    for rep in (None if injection is None else injection.report, tcm.report, moi.report):
        if rep is not None:
            verification_report.attach_to_scene(scene, rep)
    return scene


def compute_and_export_mars_mission(**kwargs) -> dict:
    """Public entry point (mirrors compute_and_export_geo_mission): optimize
    a real Earth->Mars transfer window, fly the full launch-to-orbit-insertion
    trajectory, and export it as a SceneData document. See the module
    docstring for what this does and does not model."""
    scene = _build_mars_mission(**kwargs)
    verification_report.pop_from_scene(scene)
    return scene


def build_from_config(config: dict, recorder=None) -> dict:
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
    # Optional real launch (site, rocket, payload, parking orbit). Absent or
    # `enabled: false` leaves kwargs -- and so the pipeline -- untouched.
    launch = LaunchSpec.from_config(
        config.get("launch"), default_altitude_km=parking_orbit.get("altitude_km", PARKING_ALTITUDE_KM))
    if launch is not None:
        kwargs["launch"] = launch

    scene = _build_mars_mission(recorder=recorder, **kwargs)
    scene["id"] = mission["id"]
    scene["title"] = mission["title"]
    return scene
