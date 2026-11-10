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
SITE_MARKER_HEIGHT_M = 300.0
"""Display only: the Starbase marker sits just above the WGS-84 surface, so
the Earth mesh (vertices on the ellipsoid, flat faces up to ~0.5 km inside it
between them) never hides it."""

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


# ------------------------------------------------------------- return
# The deorbit burn, entry and landing burn: not published, so these are
# modelling assumptions (orbitopt.entry's 3-DOF point mass), each named.
DEORBIT_DV_M_S = 80.0
"""Retrograde deorbit burn (Raptor). Assumed: from ~275 km it lowers perigee to
~20 km, a typical shallow-entry target. Not published."""
ENTRY_MASS_KG = 150_000.0
"""Ship mass at entry (dry ship + landing propellant, payload deployed).
Assumed; SpaceX has not published V3 masses."""
ENTRY_AREA_M2 = 400.0
"""Belly-first planform, ~9 m x ~45 m. Assumed."""
ENTRY_ALPHA_DEG = 65.0
"""Belly-flop angle of attack; CD/CL from Newtonian flat-plate theory
(L/D ~ 0.47). Assumed -- Starship's aerodynamic data is not public."""
LANDING_BURN_START_M = 1000.0
"""Height where the flip and landing burn (Raptor) start. Assumed."""
CATCH_HEIGHT_M = 100.0
"""Tower catch height above the pad. Approximate."""


@dataclass
class ReturnSolution:
    burn_time_s: float              # since lift-off
    bank_deg: float                 # + = lift banked to the right of the direction of flight
    entry: object                   # orbitopt.entry.EntryTrajectory, times from the burn
    entry_interface_s: float | None  # first descent through 120 km, since lift-off
    landing_start_s: float
    landing_end_s: float
    landing_states_eq: np.ndarray   # (N, 6) of the landing-burn segment
    landing_times_s: np.ndarray
    burn_dv_eq: np.ndarray          # deorbit burn vector (m/s, inertial)
    dv_normal_m_s: float            # its out-of-plane part (0 when entry cross-range suffices)
    landing_dv_m_s: float           # landing burn delta-v incl. gravity loss
    landing_burn_vector_eq: np.ndarray
    miss_m: float                   # ground distance of the 1 km point from Starbase
    converged: bool


def _hermite_state(times, states, t):
    """Cubic Hermite state at ``t`` from position/velocity samples."""
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    t0, t1 = times[i - 1], times[i]
    h = t1 - t0
    s = (t - t0) / h
    p0, p1, v0, v1 = states[i - 1, :3], states[i, :3], states[i - 1, 3:], states[i, 3:]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    vel = (d00 * p0 + d10 * h * v0 + d01 * p1 + d11 * h * v1) / h
    return np.concatenate([pos, vel])


def _entry_vehicle():
    from orbitopt.entry import EntryVehicle, flat_plate_coefficients

    cd, cl = flat_plate_coefficients(ENTRY_ALPHA_DEG)
    return EntryVehicle(ENTRY_MASS_KG, ENTRY_AREA_M2, cd, cl)


def _fly_from(pred: ReturnPrediction, burn_time_s: float, bank_deg: float, dv_normal_m_s: float = 0.0):
    """Deorbit burn at ``burn_time_s`` -- DEORBIT_DV_M_S retrograde plus an
    optional out-of-plane component along the orbit normal -- then entry."""
    from orbitopt.entry import fly_entry

    state = _hermite_state(pred.times_s, pred.states_eq, burn_time_s)
    v = state[3:]
    h = np.cross(state[:3], v)
    dv = -DEORBIT_DV_M_S * v / np.linalg.norm(v) + dv_normal_m_s * h / np.linalg.norm(h)
    state = np.concatenate([state[:3], v + dv])
    return fly_entry(state, _entry_vehicle(), bank_deg, stop_height_m=LANDING_BURN_START_M), dv


def _miss_vector_km(pred: ReturnPrediction, burn_time_s: float, trajectory) -> np.ndarray:
    """(downrange, crossrange) of the landing-burn start point from Starbase,
    in km: the Earth-fixed offset projected on Starbase's local tangent plane
    and resolved along / to the right of the return pass's ground-track
    heading. A fixed frame on purpose -- the trajectory's own heading near the
    end is meaningless (the ship is falling almost vertically), and using it
    made the residual jump around and the solver stall."""
    site = get_launch_site(SITE_ID)
    end = trajectory.states[-1]
    utc = pred.liftoff_mjd2000 + (burn_time_s + trajectory.times_s[-1]) / 86400.0
    offset = eci_to_ecef(end[None, :3], [utc])[0] - geodetic_to_ecef(site.latitude_deg, site.longitude_deg, 0.0)
    lat, lon = math.radians(site.latitude_deg), math.radians(site.longitude_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    e, n = float(offset @ east) / 1000.0, float(offset @ north) / 1000.0
    psi = math.radians(pred.return_pass.heading_deg)
    return np.array([e * math.sin(psi) + n * math.cos(psi), e * math.cos(psi) - n * math.sin(psi)])


def solve_return(pred: ReturnPrediction, bank_guess_deg: float = 40.0) -> ReturnSolution:
    """Deorbit-burn time and constant bank angle that bring the ship to the
    landing-burn height directly above Starbase; then the landing burn to
    the tower.

    Two well-posed stages, each 2 controls for 2 targets (downrange,
    crossrange): (1) burn time + bank. If the entry's cross-range cannot
    reach (the bank saturates), (2) keep the bank stage 1 ended on -- the
    vehicle's best constant-bank reach -- and solve burn time + an
    out-of-plane component of the deorbit burn for the rest. A single
    3-control solve was tried first: its bank / out-of-plane split depended
    on the solver tolerances, so the reported delta-v was not a real answer."""
    from scipy.optimize import least_squares

    if pred.return_pass is None:
        raise ValueError("no return pass to target")
    t_guess = pred.return_pass.time_s - 2500.0
    # Burn time enters the solver as *minutes from t_guess*, not seconds
    # since lift-off (~8e4): least_squares' xtol is relative to |x|, and with
    # an 8e4 component it called a stalled stage 2 "converged" after one
    # rejected step. Every variable is O(1) this way.
    common = dict(xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=40, method="trf")

    def miss(dt_min, bank, dv_n=0.0):
        burn_t = t_guess + 60.0 * dt_min
        traj, _dv = _fly_from(pred, burn_t, bank, dv_n)
        return _miss_vector_km(pred, burn_t, traj)

    # Finite-difference steps sized to the physics (~3 s of burn time,
    # ~0.4 deg of bank, ~0.5 m/s): the default (~1e-8) differences integrator
    # / event-location noise, not the trajectory.
    stage1 = least_squares(lambda x: miss(x[0], x[1]), x0=[0.0, bank_guess_deg],
                           bounds=([-30.0, -85.0], [30.0, 85.0]), diff_step=[0.05, 0.01], **common)
    dt_min, bank, dv_n = float(stage1.x[0]), float(stage1.x[1]), 0.0
    if np.linalg.norm(stage1.fun) * 1000.0 >= 100.0:
        # Stage 2 is close to linear (crossrange moves ~0.5 km per m/s out of
        # plane, downrange ~7.5 km per s of burn time), so a plain damped
        # Newton step with fixed finite differences converges in a few
        # iterations. least_squares' trust region stalled here from some
        # stage-1 end points (exit "xtol" with no step taken).
        x = np.array([dt_min, 0.0])
        steps = np.array([2.0 / 60.0, 5.0])  # 2 s of burn time, 5 m/s
        for _ in range(20):
            f = miss(x[0], bank, x[1])
            if np.linalg.norm(f) * 1000.0 < 1.0:
                break
            jac = np.column_stack([(miss(x[0] + steps[0], bank, x[1]) - f) / steps[0],
                                   (miss(x[0], bank, x[1] + steps[1]) - f) / steps[1]])
            dx = np.linalg.solve(jac, -f)
            # Damp: at most 5 minutes / 50 m/s per iteration.
            dx *= min(1.0, 5.0 / max(abs(dx[0]), 1e-12), 50.0 / max(abs(dx[1]), 1e-12))
            x = np.clip(x + dx, [-30.0, -300.0], [30.0, 300.0])
        dt_min, dv_n = float(x[0]), float(x[1])
    burn_t = t_guess + 60.0 * dt_min
    traj, dv = _fly_from(pred, burn_t, bank, dv_n)
    miss_m = float(np.linalg.norm(_miss_vector_km(pred, burn_t, traj))) * 1000.0
    ei = traj.first_crossing_s(120_000.0)

    # Flip and landing burn: constant deceleration in the Earth-fixed frame
    # from the 1 km point straight to the tower, to rest.
    from orbitopt.entry import G0_M_S2, OMEGA_EARTH_RAD_S

    site = get_launch_site(SITE_ID)
    t_start = burn_t + float(traj.times_s[-1])
    utc0 = pred.liftoff_mjd2000 + t_start / 86400.0
    start_ecef = eci_to_ecef(traj.states[-1][None, :3], [utc0])[0]
    v_air = traj.states[-1][3:] - np.cross([0.0, 0.0, OMEGA_EARTH_RAD_S], traj.states[-1][:3])
    v0 = float(np.linalg.norm(v_air))
    tower = geodetic_to_ecef(site.latitude_deg, site.longitude_deg, CATCH_HEIGHT_M)
    path = tower - start_ecef
    d = float(np.linalg.norm(path))
    duration = 2.0 * d / v0
    ts = np.linspace(0.0, duration, 61)
    frac = 1.0 - (1.0 - ts / duration) ** 2           # s(t)/d for constant deceleration to rest
    pos_ecef = start_ecef[None, :] + frac[:, None] * path[None, :]
    utc = utc0 + ts / 86400.0
    pos_eq = ecef_to_eci(pos_ecef, utc)
    speed = v0 * (1.0 - ts / duration)
    vel_ecef = (path / d)[None, :] * speed[:, None]
    vel_eq = ecef_to_eci(vel_ecef, utc) + np.cross([0.0, 0.0, OMEGA_EARTH_RAD_S], pos_eq)
    landing_dv = v0 + G0_M_S2 * duration           # stopping + gravity loss
    up = start_ecef / np.linalg.norm(start_ecef)
    burn_dir_eq = ecef_to_eci(up[None, :], [utc0])[0]
    return ReturnSolution(
        burn_time_s=burn_t, bank_deg=bank, entry=traj,
        entry_interface_s=None if ei is None else burn_t + ei,
        landing_start_s=t_start, landing_end_s=t_start + duration,
        landing_states_eq=np.column_stack([pos_eq, vel_eq]), landing_times_s=t_start + ts,
        burn_dv_eq=dv, dv_normal_m_s=dv_n, landing_dv_m_s=landing_dv, landing_burn_vector_eq=burn_dir_eq * landing_dv,
        # Judged on the result, not the solver's exit reason: the target is a
        # landing-burn start within 100 m of the pad.
        miss_m=miss_m, converged=bool(miss_m < 100.0),
    )


def _fmt_met(seconds: float) -> str:
    s = int(round(seconds))
    return f"T+{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _pass_text(p) -> str:
    side = "north" if p.north_of_site else "south"
    return f"{p.distance_km:,.0f} km {side} of Starbase, heading {p.direction}"


def _ship_track(pred: ReturnPrediction, sol: ReturnSolution | None):
    """(times_s, states_eq) of the ship: the orbit up to the deorbit burn, the
    entry, the landing burn, then riding the tower on the rotating Earth for a
    minute (so the marker stays on the pad instead of freezing in inertial
    space while Starbase rotates away). Without a solution: the orbit only."""
    if sol is None:
        return pred.times_s, pred.states_eq
    keep = pred.times_s < sol.burn_time_s
    burn_state = _hermite_state(pred.times_s, pred.states_eq, sol.burn_time_s)
    entry_t = sol.burn_time_s + sol.entry.times_s
    entry_x = sol.entry.states.copy()
    entry_x[0] = np.concatenate([burn_state[:3], burn_state[3:]])  # the instant before the burn
    land_t, land_x = sol.landing_times_s[1:], sol.landing_states_eq[1:]
    site = get_launch_site(SITE_ID)
    after_t = sol.landing_end_s + np.arange(5.0, 61.0, 5.0)
    tower = geodetic_to_ecef(site.latitude_deg, site.longitude_deg, CATCH_HEIGHT_M)
    after_pos = ecef_to_eci(np.repeat(tower[None, :], len(after_t), axis=0), pred.liftoff_mjd2000 + after_t / 86400.0)
    after_vel = np.cross([0.0, 0.0, 7.2921150e-5], after_pos)
    times = np.concatenate([pred.times_s[keep], entry_t, land_t, after_t])
    states = np.vstack([pred.states_eq[keep], entry_x, land_x, np.column_stack([after_pos, after_vel])])
    return times, states


def _vector_event(label: str, time_s: float, position_eq, dv_eq, note: str) -> dict:
    """A timeline event carrying a burn arrow (drawn by the viewer)."""
    return {"label": label, "time": round(time_s / 86400.0, 8), "note": note,
            "vector": {"position": np.round(equatorial_to_ecliptic(np.asarray(position_eq)) / 1000.0, 3).tolist(),
                       "deltaV": np.round(equatorial_to_ecliptic(np.asarray(dv_eq)), 2).tolist()}}


def build_flight15_scene(prediction: ReturnPrediction | None = None, convergence: dict | None = None,
                         solution: ReturnSolution | None = None) -> dict:
    pred = prediction if prediction is not None else predict_return()
    sol = solution
    site = get_launch_site(SITE_ID)
    t0 = pred.liftoff_mjd2000
    times_s, states = _ship_track(pred, sol)
    days = times_s / 86400.0
    km_per_day = 86400.0 / 1000.0
    sc_pos = equatorial_to_ecliptic(states[:, :3]) / 1000.0
    sc_vel = equatorial_to_ecliptic(states[:, 3:]) * km_per_day
    end_day = float(days[-1])

    # Starbase rides on the rotating Earth: its inertial path, sampled every
    # two minutes over the whole timeline (from lift-off).
    site_days = np.unique(np.concatenate([np.arange(0.0, end_day, 120.0 / 86400.0), [end_day]]))
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
        if sol is not None and p.time_s > sol.burn_time_s:
            continue
        events.append({"label": "Pass near Starbase", "time": round(p.time_s / 86400.0, 8),
                       "note": f"{_fmt_met(p.time_s)}: {_pass_text(p)}."})
    events.append({"label": "SpaceX-announced entry", "time": round(ANNOUNCED_ENTRY_S / 86400.0, 8),
                   "note": "T+22:10:46 from SpaceX's published timeline -- not an input to anything here; "
                           "compare with the predicted entry interface."})
    if ret is not None:
        events.append({"label": "Return pass (predicted)", "time": round(ret.time_s / 86400.0, 8),
                       "note": f"{_fmt_met(ret.time_s)}: {_pass_text(ret)} -- the pass the return is flown on "
                               "(vacuum ground track, no deorbit)."})

    entry = sol.entry if sol is not None else None
    if sol is not None:
        burn_pos = _hermite_state(pred.times_s, pred.states_eq, sol.burn_time_s)[:3]
        out_of_plane = (f", {abs(sol.dv_normal_m_s):.0f} m/s of it out of plane -- the entry's cross-range "
                        "alone falls short" if abs(sol.dv_normal_m_s) > 0.5 else "")
        events.append(_vector_event(
            "Deorbit burn (Raptor)", sol.burn_time_s, burn_pos, sol.burn_dv_eq,
            f"{_fmt_met(sol.burn_time_s)}: {np.linalg.norm(sol.burn_dv_eq):.0f} m/s{out_of_plane}. "
            f"{DEORBIT_DV_M_S:.0f} m/s retrograde is an assumption; the timing is solved so the ship comes down at Starbase."))
        if sol.entry_interface_s is not None:
            events.append({"label": "Entry interface (predicted)", "time": round(sol.entry_interface_s / 86400.0, 8),
                           "note": f"{_fmt_met(sol.entry_interface_s)}, 120 km: "
                                   f"{(sol.entry_interface_s - ANNOUNCED_ENTRY_S) / 60.0:+.1f} min from SpaceX's announced "
                                   "entry -- a check this model never saw."})
        i_g = int(np.argmax(entry.aero_g))
        events.append({"label": "Peak deceleration", "time": round((sol.burn_time_s + entry.times_s[i_g]) / 86400.0, 8),
                       "note": f"{entry.aero_g[i_g]:.2f} g at {entry.heights_m[i_g] / 1000.0:.0f} km "
                               f"(peak dynamic pressure {entry.dynamic_pressure_pa.max() / 1000.0:.1f} kPa)."})
        events.append(_vector_event(
            "Flip + landing burn (Raptor)", sol.landing_start_s, sol.landing_states_eq[0, :3], sol.landing_burn_vector_eq,
            f"{_fmt_met(sol.landing_start_s)}: from {LANDING_BURN_START_M / 1000.0:.0f} km at "
            f"{np.linalg.norm(sol.entry.states[-1, 3:] - np.cross(omega, sol.entry.states[-1, :3])):.0f} m/s, "
            f"{sol.landing_end_s - sol.landing_start_s:.0f} s, ~{sol.landing_dv_m_s:.0f} m/s incl. gravity loss "
            "(constant-deceleration estimate)."))
        events.append({"label": "Tower catch (predicted)", "time": round(sol.landing_end_s / 86400.0, 8),
                       "note": f"{_fmt_met(sol.landing_end_s)}: at Starbase"
                               + ("" if sol.converged else f" -- NOT converged, {sol.miss_m / 1000.0:.1f} km off") + "."})
    events.sort(key=lambda e: e["time"])

    if sol is not None:
        ei = sol.entry_interface_s if sol.entry_interface_s is not None else sol.landing_start_s
        phases = phase_spans([
            ("Ascent (not modelled)", 0.0, SECO_S / 86400.0, "booster boostback and catch not modelled either"),
            ("Orbital coast", SECO_S / 86400.0, sol.burn_time_s / 86400.0,
             f"{(sol.burn_time_s - SECO_S) / pred.period_s:.1f} orbits"),
            ("Deorbit coast", sol.burn_time_s / 86400.0, ei / 86400.0, "after the Raptor deorbit burn"),
            ("Entry (3-DOF, assumed aero)", ei / 86400.0, sol.landing_start_s / 86400.0,
             f"belly-first, bank {sol.bank_deg:.0f} deg"),
            ("Landing burn (Raptor)", sol.landing_start_s / 86400.0, sol.landing_end_s / 86400.0, "flip and burn to the tower"),
            ("On the tower", sol.landing_end_s / 86400.0, end_day),
        ])
    else:
        entry_day = ANNOUNCED_ENTRY_S / 86400.0
        phases = phase_spans([
            ("Ascent (not modelled)", 0.0, SECO_S / 86400.0, "booster boostback and catch not modelled either"),
            ("Orbital coast", SECO_S / 86400.0, entry_day, f"~{pred.orbits_by_announced_entry:.1f} orbits to the announced entry"),
            ("Orbit continued (hypothetical)", entry_day, end_day, "no return modelled"),
        ])

    info = [
        {"label": "Orbit (assumed)", "value": f"{pred.altitude_km:.0f} km circular, {pred.inclination_deg:.1f} deg "
                                              "(Flight 14's inclination)"},
        {"label": "Launch azimuth", "value": f"{pred.azimuth_deg:.1f} deg"},
        {"label": "Period", "value": f"{pred.period_s / 60.0:.1f} min"},
        {"label": "Orbits by announced entry", "value": f"{pred.orbits_by_announced_entry:.2f} (SpaceX: ~14)"},
    ]
    if ret is not None:
        info.append({"label": "Return pass", "value": f"{_fmt_met(ret.time_s)}, {_pass_text(ret)}"})
    if sol is not None:
        veh = _entry_vehicle()
        info += [
            {"label": "Deorbit burn", "value": f"{_fmt_met(sol.burn_time_s)}, {np.linalg.norm(sol.burn_dv_eq):.0f} m/s "
                                               f"({abs(sol.dv_normal_m_s):.0f} m/s out of plane)"},
            {"label": "Entry bank angle", "value": f"{sol.bank_deg:.1f} deg (constant)"},
            {"label": "Entry interface", "value": "—" if sol.entry_interface_s is None else
             f"{_fmt_met(sol.entry_interface_s)} ({(sol.entry_interface_s - ANNOUNCED_ENTRY_S) / 60.0:+.1f} min vs SpaceX)"},
            {"label": "Peak deceleration", "value": f"{entry.aero_g.max():.2f} g, "
                                                    f"{entry.dynamic_pressure_pa.max() / 1000.0:.1f} kPa"},
            {"label": "Landing burn", "value": f"{sol.landing_end_s - sol.landing_start_s:.0f} s, ~{sol.landing_dv_m_s:.0f} m/s"},
            {"label": "Entry vehicle (assumed)",
             "value": f"{ENTRY_MASS_KG / 1000:.0f} t, {ENTRY_AREA_M2:.0f} m², L/D {veh.lift_to_drag:.2f} "
                      f"(flat plate at {ENTRY_ALPHA_DEG:.0f} deg), ballistic coeff. {veh.ballistic_coefficient:.0f} kg/m²"},
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
                         {"label": "Marker", "value": "drawn 300 m above the surface (display only)"}]),
        body_entry("spacecraft", "Ship 42", COLOR["spacecraft"], "spacecraft", radius_display=RADIUS_DISPLAY["spacecraft"],
                   trail={"times": np.round(days, 8).tolist(), "positions": np.round(sc_pos, 3).tolist(),
                          "velocities": np.round(sc_vel, 3).tolist()},
                   info=info),
    ]
    returned = ("Return modelled with assumptions: Raptor deorbit burn, 3-DOF belly-first entry (assumed mass "
                "and aero), landing burn to the tower." if sol is not None else
                "Deorbit burn and entry are not modelled.")
    scene = scene_document(
        scene_id="starship-flight-15",
        title="Starship Flight 15 — return to Starbase (prediction)",
        subtitle=(f"Pre-flight prediction, not a replay. Lift-off {liftoff_dt:%Y-%m-%d %H:%M} UTC (NET) into a "
                  f"{pred.altitude_km:.0f} km / {pred.inclination_deg:.1f} deg orbit (Flight 14's inclination; "
                  "Flight 15's is unpublished),\npropagated with tudatpy (Earth J2, Moon, Sun). Ascent not modelled. "
                  + returned),
        distance_unit="km",
        central_body_id="earth",
        bodies=bodies,
        timeline={"unitLabel": "days", "min": 0.0, "max": round(end_day, 8), "events": events,
                  "phases": phases, "referenceEpochEt": mjd2000_to_ephemeris_seconds(t0)},
    )
    # Open on the ship with Earth beneath it, as KSP's map view opens on the vessel.
    scene["initialView"] = {"focusBodyId": "spacecraft", "faceLockBodyId": "earth", "distance": 2500.0}
    return scene


def compute_and_export_flight15() -> dict:
    """The built-in mission: the prediction, a tighter-tolerance re-run as a
    numerical check on the return pass, the solved return to Starbase, and
    the scene."""
    pred = predict_return()
    tight = predict_return(rtol=DEFAULT_RTOL * 0.01, atol=DEFAULT_ATOL * 0.01)
    convergence = None
    if pred.return_pass is not None and tight.return_pass is not None:
        convergence = {"dt_s": abs(pred.return_pass.time_s - tight.return_pass.time_s),
                       "dmiss_km": abs(pred.return_pass.distance_km - tight.return_pass.distance_km)}
    return build_flight15_scene(pred, convergence, solve_return(pred))
