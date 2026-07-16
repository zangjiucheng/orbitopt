"""Export a scrubbable SceneData document (see orbitopt.viz.scene) for the
GOES-style GTO -> GEO orbit-raising campaign optimized by
orbitopt.problems.geo_raising.GeoRaisingProblem -- the Earth-centered analog of
mission_timeline.py, and a new mission in the Mission Control app.

The screening optimizer produces an impulsive burn schedule, not a propagated
state history, so this module *reconstructs* the trajectory geometry from the
schedule: a chain of Keplerian arcs that all share the apogee point (where every
burn fires, placed on +X here), starting from the inclined, eccentric injection
ellipse and rounding + de-inclining orbit by orbit to the final equatorial GEO
circle. Points are sampled uniformly in eccentric anomaly with Kepler timing so
the marker moves realistically (slow at apogee, fast at perigee) when scrubbed.

Positions are Earth-centered km. The timeline is the *consecutive-apogee* lower
bound on elapsed time (each orbit flown back-to-back); a real campaign skips
apogees between burns for orbit determination and takes ~2 weeks -- noted in the
subtitle, not modelled here.
"""
from __future__ import annotations

import numpy as np

from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.geo_raising import (
    GeoRaisingProblem,
    MU_EARTH_KM3_S2,
    single_impulse_geo_insertion_ms,
)
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, TEXTURE, body_entry, scene_document

# gto_apogee_km only feeds single_impulse_geo_insertion_ms's closed form below --
# GeoRaisingProblem always models the shared burn apogee at r_geo (see the
# "known floor" note in problems/geo_raising.py), so it isn't a constructor
# parameter of the problem itself.
GOES_GTO = dict(gto_perigee_km=8108.0, gto_apogee_km=35286.0, gto_inclination_deg=10.6)
GEO_RAISING_KWARGS = {k: v for k, v in GOES_GTO.items() if k != "gto_apogee_km"}
APOGEE_DWELL_S = 41 * 60
EARTH_RADIUS_KM = 6378.0


def _kepler_arc(rp_km, ra_km, incl_rad, e_start, e_end, n_pts, mu, t0_s):
    """Sample a Keplerian arc (apogee on +X, plane tilted ``incl`` about X) from
    eccentric anomaly ``e_start`` to ``e_end``. Returns (positions_km, times_s)."""
    a = 0.5 * (rp_km + ra_km)
    e = (ra_km - rp_km) / (ra_km + rp_km)
    mean_motion = np.sqrt(mu / a ** 3)
    ecc = np.linspace(e_start, e_end, n_pts)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(ecc / 2.0),
                          np.sqrt(1.0 - e) * np.cos(ecc / 2.0))
    r = a * (1.0 - e * np.cos(ecc))
    mean_anom = ecc - e * np.sin(ecc)
    # unwrap so a full-rev arc (e_start..e_start+2pi) gives monotonic time
    times = t0_s + (mean_anom - mean_anom[0]) / mean_motion

    e_hat = np.array([-1.0, 0.0, 0.0])                       # perigee direction
    q_hat = np.array([0.0, -np.cos(incl_rad), np.sin(incl_rad)])  # in-plane, +motion
    pos = r[:, None] * (np.cos(nu)[:, None] * e_hat + np.sin(nu)[:, None] * q_hat)
    return pos, times


def _geo_ring(r_geo_km, n=180):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([r_geo_km * np.cos(theta), r_geo_km * np.sin(theta), np.zeros_like(theta)], axis=1)


def compute_and_export_geo_mission(points_per_orbit=110, seed=1):
    """Optimize a GOES-like GTO->GEO raising campaign (minimum finite-burn-
    feasible burn count) and export it as a SceneData document."""
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

    # trajectory: injection perigee->apogee, then each post-burn orbit apogee->apogee
    positions = []
    times = []
    apogee_times = []  # time of each burn (at an apogee)
    t_cursor = 0.0

    pos, tt = _kepler_arc(rp0, r_geo, i0, 0.0, np.pi, points_per_orbit, mu, t_cursor)
    positions.append(pos)
    times.append(tt)
    t_cursor = tt[-1]
    apogee_times.append(t_cursor)  # burn 0 fires here (first apogee)

    for k in range(n_burns):
        pos, tt = _kepler_arc(rp_after[k], r_geo, incl_after[k], np.pi, 3.0 * np.pi,
                              points_per_orbit, mu, t_cursor)
        positions.append(pos[1:])   # drop duplicate apogee point shared with previous arc
        times.append(tt[1:])
        t_cursor = tt[-1]
        if k < n_burns - 1:
            apogee_times.append(t_cursor)  # next burn fires at this apogee

    positions = np.concatenate(positions, axis=0)
    times_s = np.concatenate(times, axis=0)
    days = (times_s / 86400.0)
    days = days - days[0]

    spacecraft_positions = positions.round(1).tolist()
    spacecraft_days = days.round(4).tolist()

    # 3D delta-v vector of each burn: the difference of apogee velocity vectors
    # (all burns at the shared apogee on +X, velocity direction (0, cos i, -sin i)
    # in this frame), for drawing the maneuver arrow. Units: m/s; position km.
    def _apogee_velocity(perigee_km, incl_rad):
        a = 0.5 * (perigee_km + r_geo)
        v = np.sqrt(mu * (2.0 / r_geo - 1.0 / a))
        return v * np.array([0.0, np.cos(incl_rad), -np.sin(incl_rad)])

    burn_position = [float(r_geo), 0.0, 0.0]
    v_before = _apogee_velocity(rp0, i0)
    burn_vectors = []
    for k in range(n_burns):
        v_after = _apogee_velocity(rp_after[k], incl_after[k])
        burn_vectors.append(((v_after - v_before) * 1000.0))  # km/s -> m/s
        v_before = v_after

    events = []
    for k in range(n_burns):
        burn_day = round(float((apogee_times[k]) / 86400.0), 4)
        events.append({
            "label": f"Apogee burn {k + 1}",
            "time": burn_day,
            "note": f"delta-v {sched['dv_per_burn_ms'][k]:.0f} m/s, "
                    f"perigee -> {sched['perigee_after_km'][k]:,.0f} km, "
                    f"i -> {sched['inclination_after_deg'][k]:.1f} deg",
            "vector": {"position": burn_position, "deltaV": burn_vectors[k].round(2).tolist()},
        })
    events.append({"label": "GEO insertion", "time": round(float(days[-1]), 4),
                   "note": f"circular {r_geo:,.0f} km, i ~ 0 deg"})

    geo_period_h = 2.0 * np.pi * np.sqrt(r_geo ** 3 / mu) / 3600.0
    single = single_impulse_geo_insertion_ms(**GOES_GTO)

    bodies = [
        body_entry("earth", "Earth", COLOR["earth"], "planet",
                   radius_display=RADIUS_DISPLAY["earth"], position=[0.0, 0.0, 0.0],
                   texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM),
        body_entry("geo", "GEO ring", "#2fa97a", "reference", radius_display=2.4,
                   orbit=_geo_ring(r_geo).round(1).tolist(), orbit_dashed=True,
                   position=[float(r_geo), 0.0, 0.0],
                   info=[
                       {"label": "Radius", "value": f"{r_geo:,.0f} km"},
                       {"label": "Period", "value": f"{geo_period_h:.2f} h (1 sidereal day)"},
                       {"label": "Note", "value": "shared apogee — every burn fires here"},
                   ]),
        body_entry(
            "spacecraft", "GOES", COLOR["spacecraft"], "spacecraft",
            radius_display=RADIUS_DISPLAY["spacecraft"],
            trail={"times": spacecraft_days, "positions": spacecraft_positions},
            info=[
                {"label": "Apogee burns", "value": f"{n_burns} (min feasible)"},
                {"label": "Total delta-v", "value": f"{sched['dv_total_ms']:.0f} m/s"},
                {"label": "1-impulse ideal", "value": f"{single:.0f} m/s"},
                {"label": "Injection", "value": f"{GOES_GTO['gto_perigee_km']:.0f}x"
                                                 f"{GOES_GTO['gto_apogee_km']:.0f} km, "
                                                 f"{GOES_GTO['gto_inclination_deg']:.1f} deg"},
            ],
        ),
    ]

    return scene_document(
        scene_id="goes-gto-geo",
        title="GOES — GTO to GEO Raising",
        subtitle="Optimized minimum finite-burn-feasible apogee-burn campaign from a GOES-like\n"
                 "transfer orbit to geostationary orbit. Screening model (impulsive, apogee at GEO\n"
                 "radius); timeline is the consecutive-apogee lower bound (a real campaign skips\n"
                 "apogees between burns over ~2 weeks).",
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
