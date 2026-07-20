"""Export a scrubbable SceneData document (see orbitopt.viz.scene) for the
GOES-style launch-to-GEO campaign: the real Atlas V/Centaur launch-to-GTO
profile (docs/goes_gto_geo_mission_plan.md Section 2), followed by the
apogee-raising campaign optimized by
orbitopt.problems.geo_raising.GeoRaisingProblem -- the Earth-centered analog of
mission_timeline.py, and a new mission in the Mission Control app.

The screening optimizer produces an impulsive burn *schedule* (target
perigee/inclination after each burn), not a propagated state history, so
this module turns that schedule into an actual trajectory by real
numerical integration (tudatpy, via verify.tudat_propagate.propagate_multi_arc,
Earth point-mass gravity only -- matching the optimizer's own idealized
two-body physics, so the flown (a, e, i) at each burn matches the schedule
exactly) rather than hand-assembling idealized Keplerian arcs and hoping
consecutive ones land on the same point. Position is therefore exactly
continuous everywhere (only velocity jumps, at the instant of each
impulsive burn) -- including through the prepended launch segment, which
an earlier version of this module got wrong in three successive ways
(flying through Earth between arcs; landing on the same side but at the
wrong radius; landing at the right radius but the wrong angle) before
switching to this real-propagation approach.

Every burn here is *apsis-preserving*: it fires exactly at one apsis
(perigee or apogee) of the current orbit, changes only the OTHER apsis'
radius plus inclination, and leaves the current apsis' radius (hence its
position, on the shared x-axis by this module's own perigee/apogee
convention) untouched. This isn't just convenient -- it's load-bearing:
a single-impulse inclination change only cleanly maps to a *global*
(equator-referenced) inclination if it fires at a node (an equatorial
crossing -- see docs/goes_gto_geo_mission_plan.md Section 6, open
question #1, which flags this as the screening model's own assumption).
Every orbit here has its apse line on the x axis, so its only two nodes
ARE its two apsides -- apsis-preserving burns keep the untouched apsis
(hence the line of nodes itself) fixed, so every later burn is still
firing at a real node too. An earlier version of this module used a more
general (non-apsis-preserving) burn formula for burn 3 -- it hit the
right terminal (a, e, i) for that one burn, but the resulting orbit's
apogee (where the *next* burn fires) was no longer a node, so every LAE
burn after it silently missed its target inclination by a growing amount.
Confirmed by direct comparison against the optimizer's own schedule, not
assumed correct.

Positions are Earth-centered km. Every coast targets the next apsis
exactly (a full or half Kepler period, not each burn's own real
documented elapsed time -- e.g. burn 2 really fires ~32 min after
insertion, not one full ~91.6-minute parking-orbit lap later), which is
what apsis-preserving burns require -- narrated in each event's own note,
not silently different from the mission plan's own timeline. The LAE
campaign's timeline is still the *consecutive-apogee* lower bound (each
orbit flown back-to-back, no
skipped apogees for orbit determination) -- noted in the subtitle.
"""
from __future__ import annotations

import numpy as np

from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.geo_raising import (
    GeoRaisingProblem,
    MU_EARTH_KM3_S2,
    single_impulse_geo_insertion_ms,
)
from orbitopt.verify.tudat_propagate import propagate_multi_arc
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, ROTATION_PERIOD_HOURS, TEXTURE, body_entry, scene_document

# gto_apogee_km only feeds single_impulse_geo_insertion_ms's closed form below --
# GeoRaisingProblem always models the shared burn apogee at r_geo (see the
# "known floor" note in problems/geo_raising.py), so it isn't a constructor
# parameter of the problem itself.
GOES_GTO = dict(gto_perigee_km=8108.0, gto_apogee_km=35286.0, gto_inclination_deg=10.6)
GEO_RAISING_KWARGS = {k: v for k, v in GOES_GTO.items() if k != "gto_apogee_km"}
APOGEE_DWELL_S = 41 * 60
EARTH_RADIUS_KM = 6378.0

# Real Atlas V 541 / GOES-16 launch-to-GTO profile (docs/goes_gto_geo_mission_plan.md
# Section 2 -- the well-documented, sourced launch-vehicle elements, not the
# "[illus.]"-flagged spacecraft-phase ones). Liftoff/ascent (T0 to ~T+5 min:
# SRB burn, staging, powered flight to parking-orbit insertion) is real but
# isn't a coast/burn a two-body propagator can represent -- not drawn, same
# as mission_timeline.py's Artemis scene, which likewise starts its trail
# already in orbit, not at the pad; flagged with an event note instead of
# silently vanishing from the story. Every burn below fires exactly at an
# apsis (see the module docstring for why that's load-bearing, not just
# tidy), so each coast targets the next apsis exactly (one Kepler period,
# not the ~32 min the real Centaur burn 2 actually waits -- it fires at an
# equatorial node that isn't quite this orbit's own perigee/apogee, a gap
# already flagged as illustrative in the mission plan itself, Section 2).
# Burn 2's own perigee (167 km, not the real ~187) and burn 3's own apogee
# (see compute_and_export_geo_mission) are likewise kept at whatever this
# module's own preceding apsis already is, an apsis-preservation
# simplification -- see the module docstring for why that's
# load-bearing, not just tidy.
PARKING_ORBIT = dict(perigee_km=167.0, apogee_km=541.0, inclination_deg=28.15)
TRANSFER_ORBIT_APOGEE_KM = 32716.0
TRANSFER_ORBIT_INCLINATION_DEG = 25.68


def _apsis_state_km(r_km, incl_rad, other_apsis_km, sign, mu=MU_EARTH_KM3_S2):
    """Position (km) and velocity (km/s) at an apsis of radius r_km (this
    module's convention: perigee/apogee always on the +/-x axis, plane
    tilted incl_rad about x), given the OTHER apsis' radius (for vis-viva).
    sign=+1 for the periapsis-side apsis (+x, prograde tangential velocity
    in +y-ish), sign=-1 for the apoapsis-side one (-x, velocity reversed)."""
    a = 0.5 * (r_km + other_apsis_km)
    speed = np.sqrt(mu * (2.0 / r_km - 1.0 / a))
    pos = np.array([sign * r_km, 0.0, 0.0])
    vel = sign * speed * np.array([0.0, np.cos(incl_rad), np.sin(incl_rad)])
    return pos, vel


def _apsis_burn_km_s(r_vec_km, v_vec_km_s, other_apsis_new_km, target_incl_rad, mu=MU_EARTH_KM3_S2):
    """Post-burn velocity (km/s) for an apsis-preserving burn: the current
    position (assumed to already be an apsis -- r_vec_km . v_vec_km_s ~ 0)
    stays exactly as-is; only the OTHER apsis' radius (hence orbit shape)
    changes, and the plane tilts to reach ``target_incl_rad`` exactly (an
    ABSOLUTE inclination, not a delta from the current one -- computing the
    rotation angle from the CURRENT state's own actual inclination, fresh
    every call, rather than trusting a caller-tracked "previous" value, is
    what keeps a multi-burn chain from silently drifting: an earlier
    version took delta_incl_rad instead, and a caller loop that updated its
    own "current inclination" bookkeeping from the *schedule's* intended
    value rather than each burn's actually-achieved one baked in a small
    compounding error at every step -- confirmed by direct comparison
    against the optimizer's own per-burn schedule, not assumed).

    Tries both signs of the rotation and keeps whichever actually lands
    closer to target_incl_rad -- simpler and more robust than hand-deriving
    which sign is which, since (confirmed during development, not assumed)
    the correct sign flips between the +x and -x apsis for the exact same
    rotation formula: a rotation "around the local radial axis" reverses
    handedness when that axis itself reverses from perigee (+x) to apogee
    (-x).
    """
    r = np.linalg.norm(r_vec_km)
    r_hat = r_vec_km / r
    v_hat = v_vec_km_s / np.linalg.norm(v_vec_km_s)
    a_new = 0.5 * (r + other_apsis_new_km)
    speed_new = np.sqrt(mu * (2.0 / r - 1.0 / a_new))

    h_before = np.cross(r_vec_km, v_vec_km_s)
    incl_before_rad = np.arccos(np.clip(h_before[2] / np.linalg.norm(h_before), -1.0, 1.0))
    delta_incl_rad = incl_before_rad - target_incl_rad

    def rotated(sign):
        v_hat_new = v_hat * np.cos(sign * delta_incl_rad) + np.cross(r_hat, v_hat) * np.sin(sign * delta_incl_rad)
        return v_hat_new * speed_new

    def resulting_incl_rad(v_after):
        h = np.cross(r_vec_km, v_after)
        return np.arccos(np.clip(h[2] / np.linalg.norm(h), -1.0, 1.0))

    v_plus, v_minus = rotated(1.0), rotated(-1.0)
    err_plus = abs(resulting_incl_rad(v_plus) - target_incl_rad)
    err_minus = abs(resulting_incl_rad(v_minus) - target_incl_rad)
    return v_plus if err_plus <= err_minus else v_minus


def _kepler_period_s(a_km, mu=MU_EARTH_KM3_S2):
    return 2.0 * np.pi * np.sqrt(a_km ** 3 / mu)


def _geo_ring(r_geo_km, n=180):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([r_geo_km * np.cos(theta), r_geo_km * np.sin(theta), np.zeros_like(theta)], axis=1)


def compute_and_export_geo_mission(seed=1, step_size_s=2.0, max_output_points=900):
    """Optimize a GOES-like GTO->GEO raising campaign (minimum finite-burn-
    feasible burn count), fly the real Atlas V launch-to-GTO profile plus
    that campaign as one continuously-propagated trajectory, and export it
    as a SceneData document.

    step_size_s=2.0 is an empirically-checked tradeoff, not a round-number
    guess: at 60s (tudat_propagate's usual default), the terminal GEO
    state's own fixed-step RK4 drift alone put it ~575 km off both its
    target radius and the equatorial plane -- large enough to be visibly
    wrong for a "circular, equatorial" ending. 10s got that down to ~84 km;
    2s to ~24 km (0.03% of r_geo, i.e. visually exact) for ~15s of total
    compute, the point this stopped being worth spending more time on.
    """
    mu = MU_EARTH_KM3_S2
    cap = GeoRaisingProblem(n_burns=2, **GEO_RAISING_KWARGS).max_dv_per_pass_ms(APOGEE_DWELL_S)

    champion = None
    for n_burns in range(2, 7):
        problem = GeoRaisingProblem(n_burns=n_burns, max_dv_per_burn_ms=cap, **GEO_RAISING_KWARGS)
        x, _, _ = run_optimization(problem, pop_size=160, generations=150, seed=seed, verbose=False)
        d = problem.decode(x)
        if d["feasible"]:
            champion = (problem, d)
            break
    if champion is None:
        raise RuntimeError("No finite-burn-feasible GTO->GEO schedule within n_burns <= 6.")
    problem, sched = champion

    r_geo = problem.r_geo
    rp0 = problem.rp0
    i0 = problem.i0
    incl_after = np.radians(sched["inclination_after_deg"])
    rp_after = sched["perigee_after_km"]
    n_burns = sched["n_burns"]

    states_km = []
    epochs_s = []
    events = []
    burn_events = []
    t_cursor = 0.0

    def _propagate(r_km, v_km_s, duration_s):
        nonlocal t_cursor
        result = propagate_multi_arc(
            r_km * 1000.0, v_km_s * 1000.0, 0.0,
            arcs=[{"type": "coast", "duration": duration_s}],
            central_body="Earth", perturbing_bodies=("Earth",), step_size=step_size_s,
        )
        states_km.append(result.states / 1000.0)
        epochs_s.append(result.epochs + t_cursor)
        t_cursor += float(result.epochs[-1])
        final = result.states[-1] / 1000.0
        return final[:3], final[3:]

    events.append({
        "label": "Liftoff", "time": 0.0,
        "note": "Atlas V 541 / Centaur; ascent to parking-orbit insertion (SRB burn, "
                "staging, ~5 min) isn't itself drawn -- the trail starts already in "
                "orbit, same as the Artemis II scene.",
    })

    parking_rp = EARTH_RADIUS_KM + PARKING_ORBIT["perigee_km"]
    parking_ra = EARTH_RADIUS_KM + PARKING_ORBIT["apogee_km"]
    parking_i = np.radians(PARKING_ORBIT["inclination_deg"])
    r_km, v_km_s = _apsis_state_km(parking_rp, parking_i, parking_ra, sign=+1.0, mu=mu)
    events.append({
        "label": "Parking-orbit insertion", "time": 0.0,
        "note": f"Centaur burn 1 (~7 min 38 s) -- {PARKING_ORBIT['perigee_km']:.0f} x "
                f"{PARKING_ORBIT['apogee_km']:.0f} km, {PARKING_ORBIT['inclination_deg']:.2f} deg",
    })

    # Burn 2 (apsis-preserving, at parking perigee): raise apogee toward the
    # transfer orbit's, with the same plane change the real Centaur burn 2
    # does. Real burn 2 also nudges perigee 167->187 km; keeping THIS
    # apsis exactly fixed (167 km) is what keeps it -- and everything
    # downstream -- on a real node (see module docstring), a tradeoff for
    # a ~20 km perigee difference from the sourced figure.
    r_km, v_km_s = _propagate(r_km, v_km_s, _kepler_period_s(0.5 * (parking_rp + parking_ra), mu))
    transfer_ra = EARTH_RADIUS_KM + TRANSFER_ORBIT_APOGEE_KM
    transfer_i = np.radians(TRANSFER_ORBIT_INCLINATION_DEG)
    v_after = _apsis_burn_km_s(r_km, v_km_s, transfer_ra, transfer_i, mu=mu)
    burn_events.append((t_cursor, "Transfer-orbit injection",
                        f"Centaur burn 2 (~5 min 36 s) -- apsis-preserving: perigee stays "
                        f"{PARKING_ORBIT['perigee_km']:.0f} km (real burn 2 also nudges it to ~187 km; "
                        "kept fixed here so it -- and every later burn -- still fires at a real "
                        f"equatorial node, see module docstring), apogee -> {TRANSFER_ORBIT_APOGEE_KM:.0f} km, "
                        f"{TRANSFER_ORBIT_INCLINATION_DEG:.2f} deg",
                        r_km.tolist(), ((v_after - v_km_s) * 1000.0).tolist()))
    v_km_s = v_after

    # Coast to the transfer orbit's own apogee (a node, since perigee -- the
    # other apsis -- didn't move) -- burn 3a fires there.
    r_km, v_km_s = _propagate(r_km, v_km_s, _kepler_period_s(0.5 * (parking_rp + transfer_ra), mu) / 2.0)

    # Burn 3 (real Centaur burn 3, "apogee raise + plane cut") is split into
    # two apsis-preserving steps -- see module docstring for why a single
    # general (non-apsis) burn can't correctly hit both the target (a, e, i)
    # AND leave a node where the next burn needs one:
    #  3a, at the transfer orbit's apogee (preserved): raise perigee to the
    #      real GOES-16 injection perigee (rp0) and do the FULL plane cut.
    #  3b, at the resulting perigee (rp0, still a node since apogee didn't
    #      move at 3a): raise apogee the rest of the way to r_geo -- this
    #      is also where the GeoRaisingProblem "shared r_geo apogee"
    #      idealization (see that module's own docstring) actually starts,
    #      one step later than the real mission's own apogee (~35,286 km).
    v_after = _apsis_burn_km_s(r_km, v_km_s, rp0, i0, mu=mu)
    burn_events.append((t_cursor, "Apogee raise + plane cut (3a) / separation",
                        f"Centaur burn 3, part 1 of 2 -- apsis-preserving: apogee stays "
                        f"{TRANSFER_ORBIT_APOGEE_KM:.0f} km, perigee -> {GOES_GTO['gto_perigee_km']:.0f} km "
                        f"(matches the real GOES-16 injection exactly), {GOES_GTO['gto_inclination_deg']:.1f} deg "
                        "(the full plane cut happens here, in one step); spacecraft separates ~3.5 h "
                        "after liftoff.",
                        r_km.tolist(), ((v_after - v_km_s) * 1000.0).tolist()))
    v_km_s = v_after

    r_km, v_km_s = _propagate(r_km, v_km_s, _kepler_period_s(0.5 * (rp0 + transfer_ra), mu) / 2.0)
    v_after = _apsis_burn_km_s(r_km, v_km_s, r_geo, i0, mu=mu)
    burn_events.append((t_cursor, "Apogee raise (3b)",
                        f"Centaur-equivalent, part 2 of 2 -- apsis-preserving: perigee stays "
                        f"{GOES_GTO['gto_perigee_km']:.0f} km, apogee -> {r_geo - EARTH_RADIUS_KM:,.0f} km "
                        "(the geostationary radius -- vs. the real mission's own ~35,286 km injection "
                        "apogee, which then grows gradually across the first few LAE burns; this is "
                        "GeoRaisingProblem's own \"shared r_geo apogee\" screening-model idealization, "
                        "just made explicit as its own step here), no further plane change.",
                        r_km.tolist(), ((v_after - v_km_s) * 1000.0).tolist()))
    v_km_s = v_after

    r_km, v_km_s = _propagate(r_km, v_km_s, _kepler_period_s(0.5 * (rp0 + r_geo), mu) / 2.0)

    apogee_times = [t_cursor]  # burn k fires at the k-th apogee arrival
    for k in range(n_burns):
        v_after = _apsis_burn_km_s(r_km, v_km_s, rp_after[k], incl_after[k], mu=mu)
        burn_events.append((t_cursor, f"Apogee burn {k + 1}",
                            f"delta-v {np.linalg.norm((v_after - v_km_s) * 1000.0):.0f} m/s, "
                            f"perigee -> {rp_after[k]:,.0f} km, i -> {sched['inclination_after_deg'][k]:.1f} deg",
                            r_km.tolist(), ((v_after - v_km_s) * 1000.0).tolist()))
        v_km_s = v_after
        a_next = 0.5 * (rp_after[k] + r_geo)
        # Every pass -- including after the final burn -- coasts a full
        # period back to apogee: for the final burn this is what actually
        # reaches the circularized result (rp_after[-1] == r_geo by
        # construction) instead of leaving the last recorded trail point on
        # the previous, still-inclined orbit, and gives the animation a real
        # "arrived, going around GEO" payoff instead of stopping mid-burn.
        r_km, v_km_s = _propagate(r_km, v_km_s, _kepler_period_s(a_next, mu))
        if k < n_burns - 1:
            apogee_times.append(t_cursor)

    states_km = np.concatenate(states_km, axis=0)
    epochs_s = np.concatenate(epochs_s, axis=0)
    days = epochs_s / 86400.0

    for burn_time_s, label, note, position, delta_v in burn_events:
        events.append({
            "label": label, "time": round(burn_time_s / 86400.0, 4), "note": note,
            "vector": {"position": [round(c, 1) for c in position], "deltaV": [round(c, 2) for c in delta_v]},
        })
    events.sort(key=lambda e: e["time"])
    events.append({"label": "GEO insertion", "time": round(float(days[-1]), 4),
                   "note": f"circular {r_geo:,.0f} km, i ~ 0 deg"})

    if len(states_km) > max_output_points:
        keep = np.unique(np.linspace(0, len(states_km) - 1, max_output_points).round().astype(int))
        spacecraft_positions = states_km[keep, :3].round(1).tolist()
        spacecraft_days = days[keep].round(4).tolist()
    else:
        spacecraft_positions = states_km[:, :3].round(1).tolist()
        spacecraft_days = days.round(4).tolist()

    geo_period_h = 2.0 * np.pi * np.sqrt(r_geo ** 3 / mu) / 3600.0
    single = single_impulse_geo_insertion_ms(**GOES_GTO)

    bodies = [
        body_entry("earth", "Earth", COLOR["earth"], "planet",
                   radius_display=RADIUS_DISPLAY["earth"], position=[0.0, 0.0, 0.0],
                   texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["earth"]),
        body_entry("geo", "GEO ring", "#2fa97a", "reference", radius_display=2.4,
                   orbit=_geo_ring(r_geo).round(1).tolist(), orbit_dashed=True,
                   position=[float(r_geo), 0.0, 0.0],
                   info=[
                       {"label": "Radius", "value": f"{r_geo:,.0f} km"},
                       {"label": "Period", "value": f"{geo_period_h:.2f} h (1 sidereal day)"},
                       {"label": "Note", "value": "shared apogee — every apogee-raising burn fires here"},
                   ]),
        body_entry(
            "spacecraft", "GOES", COLOR["spacecraft"], "spacecraft",
            radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": spacecraft_days, "positions": spacecraft_positions},
            info=[
                {"label": "Apogee burns", "value": f"{n_burns} (min feasible)"},
                {"label": "Total delta-v (campaign)", "value": f"{sched['dv_total_ms']:.0f} m/s"},
                {"label": "1-impulse ideal (campaign)", "value": f"{single:.0f} m/s"},
                {"label": "Injection", "value": f"{GOES_GTO['gto_perigee_km']:.0f}x"
                                                 f"{GOES_GTO['gto_apogee_km']:.0f} km, "
                                                 f"{GOES_GTO['gto_inclination_deg']:.1f} deg"},
            ],
        ),
    ]

    return scene_document(
        scene_id="goes-gto-geo",
        title="GOES — GTO to GEO Raising",
        subtitle="Real Atlas V/Centaur launch-to-transfer-orbit profile (GOES-16), continuously\n"
                 "propagated (Earth two-body, matching the campaign's own idealized physics) into an\n"
                 "optimized minimum finite-burn-feasible apogee-burn campaign to geostationary orbit.\n"
                 "Timeline is the consecutive-apogee lower bound for the campaign (a real campaign\n"
                 "skips apogees between burns over ~2 weeks); every burn fires exactly at an apsis, so\n"
                 "the launch segment's own coast durations are apsis-exact too, not each burn's real\n"
                 "documented elapsed time (see event notes).",
        distance_unit="km",
        central_body_id="earth",
        bodies=bodies,
        timeline={
            "unitLabel": "days",
            "min": 0.0,
            "max": round(float(days[-1]), 4),
            "events": events,
        },
    )
