"""Starship Flight 15: a pre-flight prediction of the ship's return pass over
Starbase.

SpaceX plans Flight 15 (NET 2026-11-13) as the first attempt to bring the
ship back to the launch site: about 14 orbits at ~275 km, entry at
T+22:10:46, then a tower catch at Starbase. The ascent, the deorbit burn and
the entry are not public in any detail and this framework has no powered-
ascent or atmosphere model, so this module predicts only what orbital
mechanics fixes:

* launch from Starbase into the orbit SpaceX describes (inclination taken
  from Flight 14, the only flown Starship orbit; Flight 15's is unpublished),
* propagate the ship with tudatpy (Earth J2, Moon, Sun) from engine cutoff,
* follow its ground track and find every pass near Starbase.

Two properties make the prediction checkable. The ground track relative to
the Earth does not depend on the launch time (the orbit plane is fixed by the
site's position at lift-off), so the uncertain launch time drops out. And the
pass that can bring the ship home must follow SpaceX's announced entry time
by roughly the length of an entry -- a consistency check against a number
this model never sees. After the flight, the predicted pass and its miss
distance can be compared with tracking data.

Every assumption is a named constant below, with its source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from orbitopt.bodies import body_fixed_to_eclipj2000_matrix
from orbitopt.ground_track import ecef_to_eci, ecef_to_geodetic, eci_to_ecef, geodetic_to_ecef, site_passes
from orbitopt.injection import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    TudatEarthPropagator,
    equatorial_to_ecliptic,
    mjd2000_to_ephemeris_seconds,
    parking_orbit_from_launch,
)
from orbitopt.launch_sites import get_launch_site
from orbitopt.launch_windows import as_mjd2000
from orbitopt.timescales import mjd2000_to_utc_datetime
from orbitopt.viz.scene import COLOR, RADIUS_DISPLAY, ROTATION_PERIOD_HOURS, TEXTURE, body_entry, phase_spans, scene_document

# ---------------------------------------------------------------- inputs
# Each value: what it is, and where it came from (accessed 2026-11).
LIFTOFF_UTC = "2026-11-13T23:45:00Z"
"""Window open, 5:45 pm CST (Space.com). NextSpaceflight lists 16:45 CST; the
predicted pass times relative to lift-off do not depend on which (see the
module docstring), only the calendar times shown do."""
SECO_S = 8 * 60 + 9
"""Ship engine cutoff, T+00:08:09 (Wikipedia flight table from SpaceX's
timeline). The trajectory starts here; the ascent is not modelled."""
ANNOUNCED_ENTRY_S = 22 * 3600 + 10 * 60 + 46
"""Ship entry, T+22:10:46 (SpaceX timeline via Wikipedia). Used only as a
consistency check and a timeline marker, never as an input to the orbit."""
ORBIT_ALTITUDE_KM = 275.0
"""'~275 km and approximately 14 orbits' (SpaceX via Space.com)."""
INCLINATION_DEG = 30.5
"""Flight 14's achieved inclination (Wikipedia; 262 x 275 km). Flight 15's is
not published -- an assumption, and the main one the result depends on."""
ASCENT_DOWNRANGE_DEG = 17.0
"""Arc from the pad to engine cutoff. An assumption (~1,900 km); one degree
shifts the predicted pass times by about 15 s."""
SITE_ID = "starbase"
RETURN_SEARCH_RADIUS_KM = 1500.0
PROPAGATE_HOURS = 23.5
SITE_MARKER_HEIGHT_M = 10_000.0
"""Display only: the Starbase marker sits 10 km above the WGS-84 surface so
the textured Earth sphere (equatorial radius everywhere) doesn't hide it."""

EARTH_RADIUS_KM = 6378.137
SITE_COLOR = "#2fa97a"  # the reference-object green the GOES scene uses for the GEO ring


@dataclass
class ReturnPrediction:
    liftoff_mjd2000: float
    azimuth_deg: float
    azimuth_in_corridor: bool
    inclination_deg: float
    altitude_km: float
    period_s: float
    times_s: np.ndarray            # since lift-off
    states_eq: np.ndarray          # (N, 6) m, m/s, Earth J2000 equator
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_km: np.ndarray
    passes: list                   # SitePass, time_s since lift-off
    return_pass: object | None     # the first pass after the announced entry
    model: dict = field(default_factory=dict)

    @property
    def orbits_by_announced_entry(self) -> float:
        return (ANNOUNCED_ENTRY_S - SECO_S) / self.period_s

    @property
    def entry_lead_s(self) -> float | None:
        """How long the announced entry precedes the return pass."""
        return None if self.return_pass is None else self.return_pass.time_s - ANNOUNCED_ENTRY_S


def southeast_azimuth(site, inclination_deg: float) -> float:
    """Direct-ascent azimuth into ``inclination_deg``, the south-east branch
    (cos i = cos(phi) sin(Az)): Starship heads out over the Gulf of Mexico."""
    s = math.cos(math.radians(inclination_deg)) / math.cos(math.radians(site.latitude_deg))
    if not -1.0 <= s <= 1.0:
        raise ValueError(f"inclination {inclination_deg} deg is not reachable by direct ascent from {site.id}")
    return 180.0 - math.degrees(math.asin(s))


def predict_return(
    liftoff=LIFTOFF_UTC,
    inclination_deg: float = INCLINATION_DEG,
    altitude_km: float = ORBIT_ALTITUDE_KM,
    ascent_downrange_deg: float = ASCENT_DOWNRANGE_DEG,
    hours: float = PROPAGATE_HOURS,
    max_step_s: float = 30.0,
    rtol: float | None = None,
    atol: float | None = None,
) -> ReturnPrediction:
    """Launch from Starbase into a circular ``altitude_km`` orbit at
    ``inclination_deg``, propagate from engine cutoff for ``hours`` and find
    the ground track's passes near Starbase."""
    site = get_launch_site(SITE_ID)
    t0 = as_mjd2000(liftoff)
    az = southeast_azimuth(site, inclination_deg)
    launch = parking_orbit_from_launch(site, t0, az, altitude_km=altitude_km, ascent_time_s=SECO_S,
                                       ascent_downrange_deg=ascent_downrange_deg)
    kwargs = {k: v for k, v in (("rtol", rtol), ("atol", atol)) if v is not None}
    prop = TudatEarthPropagator(perturbing_bodies=("Earth", "Moon", "Sun"), j2=True, max_step_s=max_step_s, **kwargs)
    t_ins = launch.insertion_mjd2000
    ep0 = mjd2000_to_ephemeris_seconds(t_ins)
    epochs, states = prop.coast(launch.orbit.state_at(t_ins), ep0, hours * 3600.0 - SECO_S, max_step=max_step_s)
    rel = np.asarray(epochs, dtype=float) - ep0
    utc = t_ins + rel / 86400.0
    lat, lon, h = ecef_to_geodetic(eci_to_ecef(states[:, :3], utc))
    times = rel + SECO_S
    passes = site_passes(times, lat, lon, site.latitude_deg, site.longitude_deg, RETURN_SEARCH_RADIUS_KM)
    after_entry = [p for p in passes if p.time_s >= ANNOUNCED_ENTRY_S]
    return ReturnPrediction(
        liftoff_mjd2000=t0, azimuth_deg=az, azimuth_in_corridor=launch.azimuth_in_corridor,
        inclination_deg=inclination_deg, altitude_km=altitude_km, period_s=launch.orbit.period_s,
        times_s=times, states_eq=np.asarray(states), lat_deg=lat, lon_deg=lon, alt_km=h / 1000.0,
        passes=passes, return_pass=after_entry[0] if after_entry else None,
        model=prop.model_description(),
    )


def _fmt_met(seconds: float) -> str:
    s = int(round(seconds))
    return f"T+{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _pass_text(p) -> str:
    side = "north" if p.north_of_site else "south"
    return f"{p.distance_km:,.0f} km {side} of Starbase, heading {p.direction}"


def build_flight15_scene(prediction: ReturnPrediction | None = None, convergence: dict | None = None) -> dict:
    pred = prediction if prediction is not None else predict_return()
    site = get_launch_site(SITE_ID)
    t0 = pred.liftoff_mjd2000
    days = pred.times_s / 86400.0
    km_per_day = 86400.0 / 1000.0
    sc_pos = equatorial_to_ecliptic(pred.states_eq[:, :3]) / 1000.0
    sc_vel = equatorial_to_ecliptic(pred.states_eq[:, 3:]) * km_per_day

    # Starbase rides on the rotating Earth: its inertial path, sampled every
    # two minutes over the whole timeline (from lift-off).
    site_days = np.arange(0.0, days[-1] + 1e-9, 120.0 / 86400.0)
    site_ecef = geodetic_to_ecef(site.latitude_deg, site.longitude_deg, SITE_MARKER_HEIGHT_M)
    site_eq = ecef_to_eci(np.repeat(site_ecef[None, :], len(site_days), axis=0), t0 + site_days)
    omega = np.array([0.0, 0.0, 7.2921150e-5])
    site_pos = equatorial_to_ecliptic(site_eq) / 1000.0
    site_vel = equatorial_to_ecliptic(np.cross(omega, site_eq)) * km_per_day

    ret = pred.return_pass
    events = [
        {"label": "Liftoff", "time": 0.0,
         "note": f"Starbase, azimuth {pred.azimuth_deg:.1f} deg "
                 f"({'inside' if pred.azimuth_in_corridor else 'OUTSIDE'} the site's corridor). "
                 "Ascent and booster catch are not modelled."},
        {"label": "Ship engine cutoff", "time": round(SECO_S / 86400.0, 8),
         "note": f"T+0:08:09 per SpaceX's timeline; the trajectory starts here, {ASCENT_DOWNRANGE_DEG:.0f} deg downrange (assumed)."},
    ]
    for p in pred.passes:
        if ret is not None and p is ret:
            continue
        events.append({"label": "Pass near Starbase", "time": round(p.time_s / 86400.0, 8),
                       "note": f"{_fmt_met(p.time_s)}: {_pass_text(p)}."})
    events.append({"label": "SpaceX-announced entry", "time": round(ANNOUNCED_ENTRY_S / 86400.0, 8),
                   "note": "T+22:10:46 from SpaceX's published timeline -- not computed here, and not an input "
                           "to the orbit. The deorbit burn and entry are not modelled."})
    if ret is not None:
        events.append({"label": "Return pass (predicted)", "time": round(ret.time_s / 86400.0, 8),
                       "note": f"{_fmt_met(ret.time_s)}: {_pass_text(ret)} -- the first pass after the announced "
                               f"entry, {ret.time_s - ANNOUNCED_ENTRY_S:,.0f} s after it. The ship would have to "
                               "close the miss during entry (cross-range) or with an out-of-plane deorbit burn."})
    events.sort(key=lambda e: e["time"])

    end_day = float(days[-1])
    entry_day = ANNOUNCED_ENTRY_S / 86400.0
    phases = phase_spans([
        ("Ascent (not modelled)", 0.0, SECO_S / 86400.0, "booster boostback and catch not modelled either"),
        ("Orbital coast", SECO_S / 86400.0, entry_day, f"~{pred.orbits_by_announced_entry:.1f} orbits to the announced entry"),
        ("Entry (not modelled)", entry_day, ret.time_s / 86400.0 if ret is not None else end_day,
         "the trail keeps the vacuum orbit here; the real ship is descending"),
        ("Orbit continued (hypothetical)", ret.time_s / 86400.0 if ret is not None else end_day, end_day,
         "as if there were no deorbit -- for context only"),
    ])

    info = [
        {"label": "Orbit (assumed)", "value": f"{pred.altitude_km:.0f} km circular, {pred.inclination_deg:.1f} deg "
                                              "(Flight 14's inclination)"},
        {"label": "Launch azimuth", "value": f"{pred.azimuth_deg:.1f} deg"},
        {"label": "Period", "value": f"{pred.period_s / 60.0:.1f} min"},
        {"label": "Orbits by announced entry", "value": f"{pred.orbits_by_announced_entry:.2f} (SpaceX: ~14)"},
    ]
    if ret is not None:
        info += [
            {"label": "Return pass", "value": f"{_fmt_met(ret.time_s)}, {_pass_text(ret)}"},
            {"label": "Announced entry → pass", "value": f"{(ret.time_s - ANNOUNCED_ENTRY_S) / 60.0:.1f} min"},
        ]
    if convergence:
        dt, dmiss_m = convergence["dt_s"], 1000.0 * convergence["dmiss_km"]
        info.append({"label": "Numerical check",
                     "value": "tolerances x0.01 move the pass by "
                              + ("< 0.01 s" if dt < 0.01 else f"{dt:.2f} s") + ", "
                              + ("< 1 m" if dmiss_m < 1.0 else f"{dmiss_m:,.0f} m")
                              + " (integration error only -- not the model's)"})

    liftoff_dt = mjd2000_to_utc_datetime(t0)
    bodies = [
        body_entry("earth", "Earth", COLOR["earth"], "planet", radius_display=RADIUS_DISPLAY["earth"],
                   position=[0.0, 0.0, 0.0], texture=TEXTURE["earth"], radius=EARTH_RADIUS_KM,
                   rotation_period_hours=ROTATION_PERIOD_HOURS["earth"],
                   orientation_basis=body_fixed_to_eclipj2000_matrix("Earth", t0).tolist()),
        body_entry("starbase", "Starbase", SITE_COLOR, "site", radius_display=2.6,
                   trail={"times": np.round(site_days, 8).tolist(), "positions": np.round(site_pos, 3).tolist(),
                          "velocities": np.round(site_vel, 3).tolist()},
                   info=[{"label": "Location", "value": f"{site.latitude_deg:.3f} N, {-site.longitude_deg:.3f} W"},
                         {"label": "Marker", "value": "drawn 10 km above the surface (display only)"}]),
        body_entry("spacecraft", "Ship 42", COLOR["spacecraft"], "spacecraft", radius_display=RADIUS_DISPLAY["spacecraft"],
                   trail={"times": np.round(days, 8).tolist(), "positions": np.round(sc_pos, 3).tolist(),
                          "velocities": np.round(sc_vel, 3).tolist()},
                   info=info),
    ]
    return scene_document(
        scene_id="starship-flight-15",
        title="Starship Flight 15 — Starbase return pass (prediction)",
        subtitle=(f"Pre-flight prediction, not a replay. Lift-off {liftoff_dt:%Y-%m-%d %H:%M} UTC (NET) from Starbase into "
                  f"a {pred.altitude_km:.0f} km / {pred.inclination_deg:.1f} deg orbit (Flight 15's inclination is "
                  "unpublished; Flight 14's is used),\npropagated with tudatpy from engine cutoff (Earth J2, Moon, Sun; "
                  "no drag). Ascent, deorbit burn and entry are not modelled -- this finds the pass that can bring the "
                  "ship home."),
        distance_unit="km",
        central_body_id="earth",
        bodies=bodies,
        timeline={"unitLabel": "days", "min": 0.0, "max": round(end_day, 8), "events": events,
                  "phases": phases, "referenceEpochEt": mjd2000_to_ephemeris_seconds(t0)},
    )


def compute_and_export_flight15() -> dict:
    """The built-in mission: the prediction, a tighter-tolerance re-run as a
    numerical check on the return pass, and the scene."""
    pred = predict_return()
    tight = predict_return(rtol=DEFAULT_RTOL * 0.01, atol=DEFAULT_ATOL * 0.01)
    convergence = None
    if pred.return_pass is not None and tight.return_pass is not None:
        convergence = {"dt_s": abs(pred.return_pass.time_s - tight.return_pass.time_s),
                       "dmiss_km": abs(pred.return_pass.distance_km - tight.return_pass.distance_km)}
    return build_flight15_scene(pred, convergence)
