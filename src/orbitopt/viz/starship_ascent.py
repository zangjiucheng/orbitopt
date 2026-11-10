"""Starship Flight 15: the ascent of the stack and the return of the Super
Heavy booster to the tower.

What is published is a timeline (MECO, hot staging, boostback, booster catch)
-- not thrust, mass flow or guidance -- so the ascent is an *illustrative
profile*: smooth curves (cubic Hermite in latitude / longitude / height)
through the published event times, from the pad to the engine cutoff state
the orbit prediction starts from. It is drawn so the vehicles start on the
pad and stage where the timeline says; it is not a powered-flight
simulation.

The booster's return is physics where the physics is simple: after the
boostback burn it falls engines-first under gravity and drag (orbitopt.entry's
atmosphere), and the boostback's end velocity is *solved* so that it arrives
over the tower. The predicted catch time is then compared with SpaceX's
announced T+6:35 -- which, like the ship's announced entry, is never an input.

Every published value and every assumption is a named constant with its source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.ground_track import ecef_to_eci, ecef_to_geodetic, eci_to_ecef, geodetic_to_ecef

# --------------------------------------------------------- published plan
# Flight 15 plan as listed in Wikipedia's flight table (MECO, hot staging) and
# spacelaunchschedule.com (boostback, booster catch), accessed 2026-11. The
# sources disagree by a few seconds; treat them as provisional.
MECO_S = 2 * 60 + 19
HOT_STAGING_S = 2 * 60 + 21
BOOSTBACK_START_S = 2 * 60 + 26
BOOSTBACK_END_S = 3 * 60 + 2
BOOSTER_CATCH_ANNOUNCED_S = 6 * 60 + 35
"""Booster catch ~T+6:35 (several outlets). A check, never an input."""
SHIP_CATCH_ANNOUNCED_S = 22 * 3600 + 32 * 60 + 34
"""Ship catch T+22:32:34 (Wikipedia / spacelaunchschedule). A check only."""

# -------------------------------------------------------------- assumptions
MECO_ALTITUDE_M = 65_000.0
"""Altitude at MECO. Assumed from earlier flights' webcast telemetry (~65 km);
not published for Flight 15."""
MECO_SPEED_M_S = 1_600.0
"""Earth-relative speed at MECO (~5,800 km/h). Assumed, as above."""
MECO_FLIGHT_PATH_DEG = 30.0
"""Climb angle at MECO above the local horizontal. Assumed."""
MECO_DOWNRANGE_M = 60_000.0
"""Ground distance from the pad at MECO. Assumed."""
BOOSTER_MASS_KG = 300_000.0
"""Returning booster: dry mass plus landing propellant. Assumed (V3)."""
BOOSTER_AREA_M2 = 64.0
"""Falls engines-first: the 9 m diameter cross-section. Assumed."""
BOOSTER_CD = 1.0
"""Bluff-body drag coefficient, engines-first. Assumed."""
BOOSTER_LANDING_DECEL_G = 2.5
"""Net deceleration of the booster's landing burn (Raptor thrust minus
gravity). Assumed. The burn starts where its stopping distance equals the
height above the tower -- a fixed start height (first tried: 1.5 km) left the
booster still at ~700 m/s there, a 16 g stop."""
TOWER_CATCH_HEIGHT_M = 100.0
"""Tower catch height above the pad. Approximate."""

G0 = 9.80665
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])
_MEAN_R = 6_371_008.8


# ------------------------------------------------------------- geometry
def ground_point(site, azimuth_deg: float, downrange_m: float, height_m: float) -> np.ndarray:
    """Earth-fixed position ``downrange_m`` from the pad along the great
    circle leaving it on ``azimuth_deg``, ``height_m`` above the ellipsoid."""
    lat1, lon1 = math.radians(site.latitude_deg), math.radians(site.longitude_deg)
    az, d = math.radians(azimuth_deg), downrange_m / _MEAN_R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(az))
    lon2 = lon1 + math.atan2(math.sin(az) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return geodetic_to_ecef(math.degrees(lat2), math.degrees(lon2), height_m)


def _local_velocity(site, azimuth_deg, downrange_m, height_m, ground_speed, climb_speed) -> np.ndarray:
    """Earth-fixed velocity at a point of the great circle: ``ground_speed``
    along it (downrange positive) and ``climb_speed`` up."""
    here = ground_point(site, azimuth_deg, downrange_m, height_m)
    ahead = ground_point(site, azimuth_deg, downrange_m + 1.0, height_m)
    up = here / np.linalg.norm(here)
    along = ahead - here
    along -= (along @ up) * up
    along /= np.linalg.norm(along)
    return ground_speed * along + climb_speed * up


def eci_state_to_ecef(state_eq, utc_mjd2000) -> tuple[np.ndarray, np.ndarray]:
    r_ecef = eci_to_ecef(np.asarray(state_eq)[None, :3], [utc_mjd2000])[0]
    v_rel = np.asarray(state_eq)[3:] - np.cross(OMEGA_EARTH, np.asarray(state_eq)[:3])
    v_ecef = eci_to_ecef(v_rel[None, :], [utc_mjd2000])[0]
    return r_ecef, v_ecef


def ecef_path_to_eci_states(times_s, r_ecef, liftoff_mjd2000) -> np.ndarray:
    """(N, 6) inertial states of an Earth-fixed path sampled at ``times_s``
    (since lift-off); velocities by finite differences of the inertial path."""
    utc = liftoff_mjd2000 + np.asarray(times_s) / 86400.0
    r_eq = ecef_to_eci(r_ecef, utc)
    v_eq = np.gradient(r_eq, np.asarray(times_s, dtype=float), axis=0)
    return np.column_stack([r_eq, v_eq])


def _geo(r):
    lat, lon, h = ecef_to_geodetic(r[None, :])
    return np.array([math.radians(lat[0]), math.radians(lon[0]), h[0]])


def spherical_hermite(t0, r0, v0, t1, r1, v1, ts) -> np.ndarray:
    """Earth-fixed positions at ``ts`` on a cubic Hermite in (latitude,
    longitude, height) between two Earth-fixed states -- follows the curve of
    the Earth over hundreds of km, which a Cartesian cubic would cut through."""
    def rates(r, v, eps=0.05):
        return (_geo(r + v * eps) - _geo(r - v * eps)) / (2 * eps)
    g0, g1, d0, d1 = _geo(r0), _geo(r1), rates(r0, v0), rates(r1, v1)
    h = t1 - t0
    out = []
    for t in ts:
        s = (t - t0) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        lat, lon, ht = h00 * g0 + h10 * h * d0 + h01 * g1 + h11 * h * d1
        out.append(geodetic_to_ecef(math.degrees(lat), math.degrees(lon), ht))
    return np.array(out)


# ------------------------------------------------------- booster return
@dataclass
class BoosterReturn:
    times_s: np.ndarray            # since lift-off, from hot staging to catch
    r_ecef: np.ndarray             # (N, 3) Earth-fixed
    boostback_end_ground_speed: float  # m/s, negative = back toward the pad
    apogee_m: float
    landing_burn_start_s: float
    catch_s: float
    landing_burn_start_height_m: float
    landing_burn_speed_m_s: float  # speed when the landing burn starts
    landing_burn_dv_m_s: float


def _booster_density(h):
    from orbitopt.entry import density_kg_m3
    return density_kg_m3(h)


def _gravity(h):
    return G0 * (6_371_000.0 / (6_371_000.0 + h)) ** 2


def _drag(h, vs, vh):
    beta = BOOSTER_MASS_KG / (BOOSTER_CD * BOOSTER_AREA_M2)
    v = math.hypot(vs, vh)
    if v == 0.0:
        return 0.0, 0.0
    q = 0.5 * _booster_density(h) * v * v / beta
    return -q * vs / v, -q * vh / v


def _integrate(state, t0, accel, stop, dt=0.25, max_t=900.0):
    """RK4 in the launch plane (flat Earth over ~100 km) on state
    (s, h, vs, vh); ``accel(h, vs, vh) -> (as, ah)``; stops at the first step
    where ``stop(state)`` turns true, interpolated to its zero."""
    def f(x):
        a_s, a_h = accel(x[1], x[2], x[3])
        return np.array([x[2], x[3], a_s, a_h])
    ts, xs = [t0], [np.asarray(state, dtype=float)]
    x, t = xs[0], t0
    while t < t0 + max_t:
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        nx = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        m0, m1 = stop(x), stop(nx)
        if m1 <= 0.0:
            frac = m0 / (m0 - m1) if m0 != m1 else 1.0
            ts.append(t + frac * dt); xs.append(x + frac * (nx - x))
            break
        x, t = nx, t + dt
        ts.append(t); xs.append(x)
    return np.array(ts), np.array(xs)


def _fall_accel(h, vs, vh):
    d_s, d_h = _drag(h, vs, vh)
    return d_s, d_h - _gravity(h)


def _burn_accel(h, vs, vh):
    """Landing burn: Raptor thrust straight against the velocity, sized for
    BOOSTER_LANDING_DECEL_G of net deceleration when falling vertically,
    plus drag and gravity."""
    d_s, d_h = _drag(h, vs, vh)
    v = math.hypot(vs, vh)
    thrust = (BOOSTER_LANDING_DECEL_G + 1.0) * G0
    t_s, t_h = (-thrust * vs / v, -thrust * vh / v) if v > 1e-6 else (0.0, 0.0)
    return d_s + t_s, d_h + t_h - _gravity(h)


def _burn_from(state, t0):
    """Powered descent from ``state`` until it is (nearly) at rest. Stopping
    on "speed below 1 m/s", not "vertical speed reaches zero": with thrust
    held against the velocity, the direction is set by an ever smaller
    velocity near the end and the integration chatters around a hover whose
    vertical speed never quite turns positive (it ran to the time limit)."""
    return _integrate(state, t0, _burn_accel, stop=lambda x: math.hypot(x[2], x[3]) - 1.0, dt=0.05)


def _fall(s0, h0, vs0, vh0, t0):
    """Unpowered fall from the boostback end, then the landing burn started
    as late as possible -- the last moment from which a powered descent
    (thrust + drag + gravity) still comes to rest at the tower height; real
    boosters likewise let the dense air slow them before lighting. Returns
    the fall (times, states) up to ignition and the burn (times, states)."""
    t_fall, x_fall = _integrate((s0, h0, vs0, vh0), t0, _fall_accel,
                                stop=lambda x: x[1] - TOWER_CATCH_HEIGHT_M)

    def stop_height(i):
        _tb, xb = _burn_from(x_fall[i], t_fall[i])
        return xb[-1][1] - TOWER_CATCH_HEIGHT_M

    # Stop height falls monotonically with ignition time on the way down;
    # bisect for the last sample that still stops above the tower.
    lo = int(np.argmax(x_fall[:, 3] < 0.0))  # first descending sample
    hi = len(t_fall) - 1
    if stop_height(lo) < 0.0:
        raise ValueError("no landing burn can stop the booster with the assumed thrust")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stop_height(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    t_burn, x_burn = _burn_from(x_fall[lo], t_fall[lo])
    return t_fall[: lo + 1], x_fall[: lo + 1], t_burn, x_burn


def solve_booster_return(site, azimuth_deg: float) -> BoosterReturn:
    """Boostback-end ground speed (solved) that brings the booster's ballistic
    fall to the landing-burn height directly over the tower; then a constant-
    deceleration landing burn to the catch."""
    from scipy.optimize import brentq

    g = G0
    v_s0 = MECO_SPEED_M_S * math.cos(math.radians(MECO_FLIGHT_PATH_DEG))
    v_h0 = MECO_SPEED_M_S * math.sin(math.radians(MECO_FLIGHT_PATH_DEG))
    t_bb = BOOSTBACK_END_S - MECO_S
    vh_end = v_h0 - g * t_bb  # vertical speed keeps decaying under gravity through the burn

    def end_state(vs_end):
        s_end = MECO_DOWNRANGE_M + 0.5 * (v_s0 + vs_end) * t_bb
        h_end = MECO_ALTITUDE_M + 0.5 * (v_h0 + vh_end) * t_bb
        return s_end, h_end

    def ground_miss(vs_end):
        s_end, h_end = end_state(vs_end)
        _tf, _xf, _tb, xb = _fall(s_end, h_end, vs_end, vh_end, BOOSTBACK_END_S)
        return xb[-1][0]  # ground distance from the tower where the burn brings it to rest

    vs_end = brentq(ground_miss, -3000.0, 0.0, xtol=0.01)
    s_end, h_end = end_state(vs_end)
    tf, xf, tburn, xburn = _fall(s_end, h_end, vs_end, vh_end, BOOSTBACK_END_S)

    # Boostback (MECO -> boostback end): Hermite in the launch plane.
    tb = np.linspace(MECO_S, BOOSTBACK_END_S, 25)
    u = (tb - MECO_S) / t_bb
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    sb = h00 * MECO_DOWNRANGE_M + h10 * t_bb * v_s0 + h01 * s_end + h11 * t_bb * vs_end
    hb = h00 * MECO_ALTITUDE_M + h10 * t_bb * v_h0 + h01 * h_end + h11 * t_bb * vh_end

    # Landing burn, sampled every ~0.5 s; it ends at rest at the tower.
    step = max(1, len(tburn) // 40)
    tl, xl = tburn[1::step], xburn[1::step]
    if tl[-1] != tburn[-1]:
        tl, xl = np.append(tl, tburn[-1]), np.vstack([xl, xburn[-1]])
    sl, hl = xl[:, 0], xl[:, 1]
    sf, hf = xf[:, 0], xf[:, 1]
    v0 = float(math.hypot(xf[-1][2], xf[-1][3]))
    duration = float(tburn[-1] - tburn[0])
    times = np.concatenate([tb, tf[1:], tl])
    s_all = np.concatenate([sb, sf[1:], sl])
    h_all = np.concatenate([hb, hf[1:], hl])
    r = np.array([ground_point(site, azimuth_deg, s, h) for s, h in zip(s_all, h_all)])
    return BoosterReturn(
        times_s=times, r_ecef=r, boostback_end_ground_speed=float(vs_end), apogee_m=float(max(hb.max(), hf.max())),
        landing_burn_start_s=float(tf[-1]), catch_s=float(tl[-1]),
        landing_burn_start_height_m=float(xf[-1][1]),
        landing_burn_speed_m_s=float(v0), landing_burn_dv_m_s=float(v0 + G0 * duration),
    )


# ---------------------------------------------------------------- ascent
@dataclass
class Ascent:
    stack_times_s: np.ndarray      # 0 .. MECO
    stack_r_ecef: np.ndarray
    ship_times_s: np.ndarray       # MECO .. SECO (exclusive of MECO)
    ship_r_ecef: np.ndarray
    meco_r_ecef: np.ndarray
    meco_v_ecef: np.ndarray
    booster: BoosterReturn


def build_ascent(site, azimuth_deg: float, liftoff_mjd2000: float, seco_s: float, seco_state_eq) -> Ascent:
    """Pad -> MECO (stack), MECO -> SECO (ship, ending exactly on the orbit
    prediction's first state), and the booster's return."""
    pad = geodetic_to_ecef(site.latitude_deg, site.longitude_deg, 50.0)
    meco_r = ground_point(site, azimuth_deg, MECO_DOWNRANGE_M, MECO_ALTITUDE_M)
    meco_v = _local_velocity(site, azimuth_deg, MECO_DOWNRANGE_M, MECO_ALTITUDE_M,
                             MECO_SPEED_M_S * math.cos(math.radians(MECO_FLIGHT_PATH_DEG)),
                             MECO_SPEED_M_S * math.sin(math.radians(MECO_FLIGHT_PATH_DEG)))
    t_stack = np.linspace(0.0, MECO_S, 40)
    stack = spherical_hermite(0.0, pad, np.zeros(3), MECO_S, meco_r, meco_v, t_stack)
    seco_r, seco_v = eci_state_to_ecef(seco_state_eq, liftoff_mjd2000 + seco_s / 86400.0)
    t_ship = np.linspace(MECO_S, seco_s, 60)[1:]
    ship = spherical_hermite(MECO_S, meco_r, meco_v, seco_s, seco_r, seco_v, t_ship)
    return Ascent(stack_times_s=t_stack, stack_r_ecef=stack, ship_times_s=t_ship, ship_r_ecef=ship,
                  meco_r_ecef=meco_r, meco_v_ecef=meco_v, booster=solve_booster_return(site, azimuth_deg))
