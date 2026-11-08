"""LEO parking orbit and TLI / TMI injection modelling: launch site ->
parking orbit -> injection burn -> escape (or translunar) arc, end to end.

Chain
-----
1. **Launch -> parking orbit** (:func:`parking_orbit_from_launch`,
   :func:`parking_orbit_for_asymptote`). The parking-orbit plane is the
   inertial plane through the site at lift-off and the launch azimuth (or,
   for an interplanetary departure, the plane through the site and the
   departure asymptote -- the plane :func:`orbitopt.launch_windows.
   asymptote_launch_windows` opens windows for). Ascent is kinematic: the
   vehicle is inserted ``ascent_time_s`` after lift-off, ``ascent_downrange_deg``
   along the plane from the lift-off point, into a circular orbit at
   ``altitude_km``.
2. **Parking orbit** (:class:`ParkingOrbit`): circular, altitude +
   inclination + RAAN + argument of latitude at an epoch, with secular J2
   nodal regression and argument-of-latitude drift.
3. **Injection burn from v-infinity** (:func:`plan_injection`). For a
   hyperbolic target (TMI etc., ``C3 >= 0``) the burn point is the perigee of
   the departure hyperbola whose outgoing asymptote is the (in-plane
   projection of the) target ``v_inf``: rotate the asymptote back by the true
   anomaly of the asymptote ``theta_inf = acos(-1/e)``, ``e = 1 + r_p
   v_inf^2/mu``. For an elliptic target (TLI, ``C3 < 0``) the burn point is
   opposite the target apogee direction. The burn time is the first pass of
   the parking orbit through that argument of latitude after
   ``min_coast_s``; the burn is tangential, ``|dv| = v_p - v_circ``.
4. **Verification** (:func:`verify_injection`). The chain is propagated
   numerically (insertion -> coast -> impulsive burn -> escape to
   ``escape_radius_km`` / to apogee) with Earth point mass + J2 (scipy DOP853,
   pure numpy/scipy) or tudatpy (``backend="tudat"``, optionally with Sun /
   Moon third bodies). The analytic burn is then refined by a fixed-time
   Newton differential corrector on the 3-D burn vector -- the same square
   3x3 shooting structure as :mod:`orbitopt.verify.differential_correction`
   -- so the propagated outgoing asymptote (measured with
   :func:`orbitopt.verify.b_plane.b_plane_state`) or osculating apoapsis vector matches
   the target, and a machine-readable :mod:`orbitopt.verify.report` is
   produced (integrator step convergence, two-body conservation, parking-coast
   residual against the analytic orbit, launch-geometry checks, optional
   vehicle C3 check).
5. **Hand-off to the lunar B-plane targeter**: :meth:`InjectionPlan.
   lunar_b_plane_targeter_inputs` returns the pre-burn state / epoch / burn
   guess in the frame and time convention
   :func:`orbitopt.verify.differential_correction.target_lunar_b_plane`
   expects (Earth-centred ECLIPJ2000, tudat ephemeris seconds).

Frames and time: everything here is Earth-centred J2000 *equatorial*
(EME2000), metres / m/s, epochs as MJD2000 UTC days (the
:mod:`orbitopt.launch_windows` convention; GMST with UT1 ~= UTC). Tudat
ephemeris seconds are TDB since J2000 noon; :func:`mjd2000_to_ephemeris_seconds`
uses TT - UTC = 69.184 s (valid since 2017). V-infinity inputs may be given
in the ecliptic frame (pykep's heliocentric J2000 ecliptic) or equatorial.

Model limits (planning level): impulsive burn (no finite-burn gravity loss),
kinematic ascent, circular parking orbit, secular-only J2 in the analytic
orbit (the numerical one has full J2, so the coast residual includes J2
short-period terms of ~10 km). The corrector absorbs whatever the analytic
planning misses; its size is reported as ``correction_dv_m_s``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from orbitopt.launch_sites import LaunchSite, get_launch_site
from orbitopt.launch_windows import (
    OBLIQUITY_J2000_DEG,
    SECONDS_PER_DAY,
    LaunchWindow,
    _azimuth_mask,
    _site_vectors,
    as_mjd2000,
    asymptote_launch_windows,
)

MU_EARTH_M3_S2 = 3.986004418e14
R_EARTH_M = 6378137.0
J2_EARTH = 1.08262668e-3
# The one Earth gravity model of the plan and of both verification backends.
EARTH_GRAVITY = {"gravitational_parameter": MU_EARTH_M3_S2, "reference_radius": R_EARTH_M, "j2": J2_EARTH}
# TT - UTC since 2017-01-01 (37 leap seconds + 32.184 s).
TT_MINUS_UTC_S = 69.184
# Earth's sphere of influence w.r.t. the Sun [km] -- default end of the
# escape arc for hyperbolic departures.
EARTH_SOI_KM = 925_000.0

DEFAULT_RTOL = 1e-11
DEFAULT_ATOL = 1e-6
DEFAULT_MAX_STEP_S = 600.0


# --------------------------------------------------------------------------
# Frames / time
# --------------------------------------------------------------------------

def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# x_eq = R @ x_ecl
_ECL_TO_EQ = _rot_x(math.radians(OBLIQUITY_J2000_DEG))
_EQ_TO_ECL = _ECL_TO_EQ.T


def ecliptic_to_equatorial(vec) -> np.ndarray:
    """J2000 ecliptic -> J2000 equatorial; accepts (..., 3) or (..., 6) states."""
    return _rotate(vec, _ECL_TO_EQ)


def equatorial_to_ecliptic(vec) -> np.ndarray:
    """J2000 equatorial -> J2000 ecliptic; accepts (..., 3) or (..., 6) states."""
    return _rotate(vec, _EQ_TO_ECL)


def _rotate(vec, mat) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    if v.shape[-1] == 3:
        return v @ mat.T
    if v.shape[-1] == 6:
        return np.concatenate([v[..., :3] @ mat.T, v[..., 3:] @ mat.T], axis=-1)
    raise ValueError("expected trailing dimension 3 or 6")


def mjd2000_to_ephemeris_seconds(mjd2000: float) -> float:
    """MJD2000 UTC days -> tudat/SPICE ephemeris seconds (TDB ~ TT since
    J2000 12:00 TT)."""
    return (float(mjd2000) - 0.5) * SECONDS_PER_DAY + TT_MINUS_UTC_S


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector has no direction")
    return v / n


def _plane_normal(inclination_deg: float, raan_deg: float) -> np.ndarray:
    i, o = math.radians(inclination_deg), math.radians(raan_deg)
    return np.array([math.sin(i) * math.sin(o), -math.sin(i) * math.cos(o), math.cos(i)])


def _node_vector(raan_deg: float) -> np.ndarray:
    o = math.radians(raan_deg)
    return np.array([math.cos(o), math.sin(o), 0.0])


def _elements_of_normal(h_hat) -> tuple[float, float]:
    """(inclination, RAAN) [deg] of a plane with unit normal ``h_hat``."""
    h = _unit(h_hat)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(h[2])))))
    if math.hypot(h[0], h[1]) < 1e-12:
        return inc, 0.0  # equatorial: node undefined, use 0
    raan = math.degrees(math.atan2(h[0], -h[1])) % 360.0
    return inc, raan


def _argument_of_latitude(r_hat, inclination_deg: float, raan_deg: float) -> float:
    n = _node_vector(raan_deg)
    h = _plane_normal(inclination_deg, raan_deg)
    return math.degrees(math.atan2(float(np.dot(np.cross(n, r_hat), h)), float(np.dot(n, r_hat)))) % 360.0


def _wrap360(x: float) -> float:
    return x % 360.0


# --------------------------------------------------------------------------
# Parking orbit
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParkingOrbit:
    """Circular LEO parking orbit (Earth J2000 equatorial frame).

    ``arg_latitude_deg`` (angle from the ascending node along the orbit) holds
    at ``epoch_mjd2000``. With ``j2_secular`` the RAAN regresses and the
    argument of latitude drifts at the secular J2 rates."""
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    epoch_mjd2000: float
    arg_latitude_deg: float = 0.0
    j2_secular: bool = True
    mu_m3_s2: float = MU_EARTH_M3_S2

    def __post_init__(self):
        if not self.altitude_km > 0:
            raise ValueError("parking-orbit altitude must be > 0 km")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination must be in [0, 180] deg")

    @property
    def radius_m(self) -> float:
        return R_EARTH_M + 1000.0 * self.altitude_km

    @property
    def speed_m_s(self) -> float:
        return math.sqrt(self.mu_m3_s2 / self.radius_m)

    @property
    def mean_motion_rad_s(self) -> float:
        return math.sqrt(self.mu_m3_s2 / self.radius_m**3)

    @property
    def raan_rate_deg_s(self) -> float:
        if not self.j2_secular:
            return 0.0
        k = 1.5 * self.mean_motion_rad_s * J2_EARTH * (R_EARTH_M / self.radius_m) ** 2
        return math.degrees(-k * math.cos(math.radians(self.inclination_deg)))

    @property
    def arg_latitude_rate_deg_s(self) -> float:
        """d(omega + M)/dt for a circular orbit."""
        n = self.mean_motion_rad_s
        if not self.j2_secular:
            return math.degrees(n)
        k = 0.75 * n * J2_EARTH * (R_EARTH_M / self.radius_m) ** 2
        c2 = math.cos(math.radians(self.inclination_deg)) ** 2
        return math.degrees(n + k * (5.0 * c2 - 1.0) + k * (3.0 * c2 - 1.0))

    @property
    def period_s(self) -> float:
        return 360.0 / self.arg_latitude_rate_deg_s

    def raan_at(self, mjd2000: float) -> float:
        return _wrap360(self.raan_deg + self.raan_rate_deg_s * (mjd2000 - self.epoch_mjd2000) * SECONDS_PER_DAY)

    def arg_latitude_at(self, mjd2000: float) -> float:
        dt = (mjd2000 - self.epoch_mjd2000) * SECONDS_PER_DAY
        return _wrap360(self.arg_latitude_deg + self.arg_latitude_rate_deg_s * dt)

    def normal_at(self, mjd2000: float) -> np.ndarray:
        return _plane_normal(self.inclination_deg, self.raan_at(mjd2000))

    def state_at(self, mjd2000: float) -> np.ndarray:
        """Cartesian state (6,) [m, m/s] at ``mjd2000``."""
        raan = self.raan_at(mjd2000)
        u = math.radians(self.arg_latitude_at(mjd2000))
        n = _node_vector(raan)
        h = _plane_normal(self.inclination_deg, raan)
        m = np.cross(h, n)
        r_hat = math.cos(u) * n + math.sin(u) * m
        t_hat = np.cross(h, r_hat)
        return np.concatenate([self.radius_m * r_hat, self.speed_m_s * t_hat])

    def to_dict(self) -> dict:
        return {"altitude_km": self.altitude_km, "inclination_deg": self.inclination_deg,
                "raan_deg": self.raan_deg, "epoch_mjd2000": self.epoch_mjd2000,
                "arg_latitude_deg": self.arg_latitude_deg, "period_s": self.period_s,
                "speed_m_s": self.speed_m_s, "j2_secular": self.j2_secular}


@dataclass(frozen=True)
class LaunchToOrbit:
    """Lift-off geometry and the parking orbit it inserts into."""
    site_id: str
    liftoff_mjd2000: float
    azimuth_deg: float
    ascent_time_s: float
    ascent_downrange_deg: float
    orbit: ParkingOrbit
    site_offplane_deg: float      # angle of the lift-off site from the plane (0 by construction)
    azimuth_in_corridor: bool
    inclination_reachable: bool

    @property
    def insertion_mjd2000(self) -> float:
        return self.orbit.epoch_mjd2000

    def to_dict(self) -> dict:
        return {"site": self.site_id, "liftoff_mjd2000": self.liftoff_mjd2000,
                "azimuth_deg": self.azimuth_deg, "ascent_time_s": self.ascent_time_s,
                "ascent_downrange_deg": self.ascent_downrange_deg,
                "insertion_mjd2000": self.insertion_mjd2000, "orbit": self.orbit.to_dict(),
                "site_offplane_deg": self.site_offplane_deg,
                "azimuth_in_corridor": self.azimuth_in_corridor,
                "inclination_reachable": self.inclination_reachable}


def _resolve_site(site) -> LaunchSite:
    return get_launch_site(site) if isinstance(site, str) else site


def parking_orbit_from_launch(
    site: LaunchSite | str,
    liftoff,
    azimuth_deg: float,
    altitude_km: float = 185.0,
    ascent_time_s: float = 600.0,
    ascent_downrange_deg: float = 20.0,
    j2_secular: bool = True,
) -> LaunchToOrbit:
    """Parking orbit reached by launching from ``site`` at ``liftoff``
    (MJD2000 / datetime / ISO string) on ``azimuth_deg`` (clockwise from
    north). The plane is the inertial plane through the site and its motion
    direction at lift-off; insertion happens ``ascent_time_s`` later,
    ``ascent_downrange_deg`` downrange along that plane."""
    site = _resolve_site(site)
    t0 = as_mjd2000(liftoff)
    s, e, n = (x[0] for x in _site_vectors(site, t0))
    az = math.radians(azimuth_deg)
    d = math.sin(az) * e + math.cos(az) * n
    h = _unit(np.cross(s, d))
    inc, raan = _elements_of_normal(h)
    u_site = _argument_of_latitude(s, inc, raan)
    orbit = ParkingOrbit(
        altitude_km=float(altitude_km), inclination_deg=inc, raan_deg=raan,
        epoch_mjd2000=t0 + ascent_time_s / SECONDS_PER_DAY,
        arg_latitude_deg=_wrap360(u_site + ascent_downrange_deg), j2_secular=j2_secular,
    )
    offplane = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(h, s))))))
    return LaunchToOrbit(
        site_id=site.id, liftoff_mjd2000=t0, azimuth_deg=float(azimuth_deg) % 360.0,
        ascent_time_s=float(ascent_time_s), ascent_downrange_deg=float(ascent_downrange_deg),
        orbit=orbit, site_offplane_deg=offplane,
        azimuth_in_corridor=bool(_azimuth_mask(np.array([azimuth_deg]), site.azimuth_corridors_deg)[0]),
        # The site database's own (geodetic-latitude) model, so this check
        # agrees with orbitopt.launch_sites / launch_constraints.
        inclination_reachable=bool(site.can_reach_inclination(site.inclination_for_azimuth(azimuth_deg),
                                                              tol_deg=1e-3)),
    )


def asymptote_unit_vector(rla_deg: float, dla_deg: float) -> np.ndarray:
    ra, de = math.radians(rla_deg), math.radians(dla_deg)
    return np.array([math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)])


def parking_orbit_for_asymptote(
    site: LaunchSite | str,
    liftoff,
    rla_deg: float,
    dla_deg: float,
    altitude_km: float = 185.0,
    ascent_time_s: float = 600.0,
    ascent_downrange_deg: float = 20.0,
    j2_secular: bool = True,
) -> LaunchToOrbit:
    """Parking orbit whose plane contains both the site at lift-off and the
    departure asymptote (RLA, DLA) -- the geometry
    :func:`orbitopt.launch_windows.asymptote_launch_windows` uses, so a
    window's ``optimal_mjd2000`` is a valid ``liftoff``."""
    site = _resolve_site(site)
    t0 = as_mjd2000(liftoff)
    s, e, n = (x[0] for x in _site_vectors(site, t0))
    h = np.cross(s, asymptote_unit_vector(rla_deg, dla_deg))
    if np.linalg.norm(h) < 1e-9:
        raise ValueError("site is aligned with the asymptote; launch plane undefined")
    h = _unit(h)
    if h[2] < 0:
        h = -h  # prograde
    d = np.cross(h, s)
    az = math.degrees(math.atan2(float(np.dot(d, e)), float(np.dot(d, n)))) % 360.0
    return parking_orbit_from_launch(site, t0, az, altitude_km, ascent_time_s, ascent_downrange_deg, j2_secular)


# --------------------------------------------------------------------------
# Injection targets and burn planning
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionTarget:
    """What the injection burn must achieve (equatorial frame, SI).

    ``kind="hyperbolic"``: outgoing excess velocity ``v_inf_m_s`` (3,).
    ``kind="elliptic"``: apogee position ``apogee_m`` (3,) (radius and
    direction), e.g. the Moon's position at arrival for a TLI; verified as
    the osculating apoapsis vector (:func:`osculating_apoapsis_m`)."""
    kind: str
    v_inf_m_s: Optional[np.ndarray] = None
    apogee_m: Optional[np.ndarray] = None
    name: str = ""

    def __post_init__(self):
        if self.kind == "hyperbolic":
            if self.v_inf_m_s is None or np.linalg.norm(self.v_inf_m_s) <= 0:
                raise ValueError("hyperbolic target needs a non-zero v_inf_m_s")
        elif self.kind == "elliptic":
            if self.apogee_m is None:
                raise ValueError("elliptic target needs apogee_m")
        else:
            raise ValueError("kind must be 'hyperbolic' or 'elliptic'")

    @classmethod
    def from_v_inf(cls, v_inf_km_s, frame: str = "ecliptic", name: str = "TMI") -> "InjectionTarget":
        """Hyperbolic target from a departure V-infinity [km/s] (e.g. a
        porkchop cell's ``v_inf_departure``) in ``frame``."""
        v = np.asarray(v_inf_km_s, dtype=float).reshape(3) * 1000.0
        if frame == "ecliptic":
            v = ecliptic_to_equatorial(v)
        elif frame != "equatorial":
            raise ValueError(f"unknown frame {frame!r}")
        return cls("hyperbolic", v_inf_m_s=v, name=name)

    @classmethod
    def from_c3_rla_dla(cls, c3_km2_s2: float, rla_deg: float, dla_deg: float, name: str = "TMI") -> "InjectionTarget":
        if c3_km2_s2 <= 0:
            raise ValueError("hyperbolic target needs C3 > 0")
        v = math.sqrt(c3_km2_s2) * 1000.0 * asymptote_unit_vector(rla_deg, dla_deg)
        return cls("hyperbolic", v_inf_m_s=v, name=name)

    @classmethod
    def from_apogee(cls, apogee_radius_km: float, ra_deg: float, dec_deg: float, name: str = "TLI") -> "InjectionTarget":
        r = apogee_radius_km * 1000.0 * asymptote_unit_vector(ra_deg, dec_deg)
        return cls("elliptic", apogee_m=r, name=name)

    @property
    def direction(self) -> np.ndarray:
        """Unit direction the injection aims at (asymptote / apogee)."""
        return _unit(self.v_inf_m_s if self.kind == "hyperbolic" else self.apogee_m)

    @property
    def rla_dla_deg(self) -> tuple[float, float]:
        u = self.direction
        return math.degrees(math.atan2(u[1], u[0])) % 360.0, math.degrees(math.asin(u[2]))

    def c3_km2_s2(self, r_p_m: float, mu: float = MU_EARTH_M3_S2) -> float:
        if self.kind == "hyperbolic":
            return float(np.dot(self.v_inf_m_s, self.v_inf_m_s)) / 1e6
        r_a = float(np.linalg.norm(self.apogee_m))
        return -2.0 * mu / (r_a + r_p_m) / 1e6

    def to_dict(self) -> dict:
        rla, dla = self.rla_dla_deg
        return {"kind": self.kind, "name": self.name, "v_inf_m_s": self.v_inf_m_s,
                "apogee_m": self.apogee_m, "direction_ra_deg": rla, "direction_dec_deg": dla}


@dataclass
class InjectionBurn:
    """Impulsive TLI / TMI burn on the parking orbit (equatorial frame)."""
    epoch_mjd2000: float
    coast_s: float                 # from parking-orbit insertion to the burn
    pre_burn_state: np.ndarray     # (6,) m, m/s (analytic parking orbit)
    delta_v_m_s: np.ndarray        # (3,)
    arg_latitude_deg: float
    periapsis_speed_m_s: float
    eccentricity: float
    c3_km2_s2: float
    out_of_plane_deg: float        # target direction's angle from the parking plane
    flight_time_s: float           # burn -> escape radius (hyperbolic) or apogee (elliptic)

    @property
    def delta_v_mag_m_s(self) -> float:
        return float(np.linalg.norm(self.delta_v_m_s))

    def to_dict(self) -> dict:
        return {"epoch_mjd2000": self.epoch_mjd2000, "coast_s": self.coast_s,
                "pre_burn_state": self.pre_burn_state, "delta_v_m_s": self.delta_v_m_s,
                "delta_v_mag_m_s": self.delta_v_mag_m_s, "arg_latitude_deg": self.arg_latitude_deg,
                "periapsis_speed_m_s": self.periapsis_speed_m_s, "eccentricity": self.eccentricity,
                "c3_km2_s2": self.c3_km2_s2, "out_of_plane_deg": self.out_of_plane_deg,
                "flight_time_s": self.flight_time_s}


def hyperbolic_time_to_radius(r_p_m: float, v_inf_m_s: float, r_m: float, mu: float = MU_EARTH_M3_S2) -> float:
    """Time [s] from perigee to radius ``r_m`` on a departure hyperbola."""
    a = mu / (v_inf_m_s * v_inf_m_s)  # |a|
    e = 1.0 + r_p_m / a
    cosh_f = (r_m / a + 1.0) / e
    f = math.acosh(max(1.0, cosh_f))
    return math.sqrt(a**3 / mu) * (e * math.sinh(f) - f)


def _burn_geometry(orbit: ParkingOrbit, target: InjectionTarget, t: float):
    """(target argument of latitude [deg], out-of-plane angle [deg], v_p,
    e) for the parking plane at epoch ``t``."""
    r_p = orbit.radius_m
    mu = orbit.mu_m3_s2
    h = orbit.normal_at(t)
    u = target.direction
    oop = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(u, h))))))
    u_p = u - np.dot(u, h) * h
    if np.linalg.norm(u_p) < 1e-9:
        raise ValueError("target direction is normal to the parking plane")
    u_p = _unit(u_p)
    if target.kind == "hyperbolic":
        v_inf = float(np.linalg.norm(target.v_inf_m_s))
        e = 1.0 + r_p * v_inf * v_inf / mu
        theta = math.acos(-1.0 / e)
        p_hat = math.cos(theta) * u_p - math.sin(theta) * np.cross(h, u_p)
        v_p = math.sqrt(v_inf * v_inf + 2.0 * mu / r_p)
    else:
        r_a = float(np.linalg.norm(target.apogee_m))
        if r_a <= r_p:
            raise ValueError("apogee must be above the parking orbit")
        e = (r_a - r_p) / (r_a + r_p)
        p_hat = -u_p
        v_p = math.sqrt(2.0 * mu * r_a / (r_p * (r_a + r_p)))
    raan = orbit.raan_at(t)
    return _argument_of_latitude(p_hat, orbit.inclination_deg, raan), oop, v_p, e


def plan_injection(
    orbit: ParkingOrbit,
    target: InjectionTarget,
    min_coast_s: float = 0.0,
    escape_radius_km: float = EARTH_SOI_KM,
) -> InjectionBurn:
    """Analytic TLI / TMI burn: first pass through the injection point at
    least ``min_coast_s`` after insertion. The plane precesses under J2, so
    the burn point is re-derived in the plane of the burn epoch (fixed point,
    converges in a few iterations)."""
    t_min = orbit.epoch_mjd2000 + min_coast_s / SECONDS_PER_DAY
    rate = orbit.arg_latitude_rate_deg_s
    t_burn = t_min
    for _ in range(8):
        u_burn, oop, v_p, e = _burn_geometry(orbit, target, t_burn)
        du = _wrap360(u_burn - orbit.arg_latitude_at(t_min))
        t_new = t_min + du / rate / SECONDS_PER_DAY
        converged = abs(t_new - t_burn) * SECONDS_PER_DAY < 1e-4
        t_burn = t_new
        if converged:
            break
    u_burn, oop, v_p, e = _burn_geometry(orbit, target, t_burn)
    state = orbit.state_at(t_burn)
    t_hat = _unit(state[3:])
    dv = (v_p - orbit.speed_m_s) * t_hat
    r_p = orbit.radius_m
    if target.kind == "hyperbolic":
        flight = hyperbolic_time_to_radius(r_p, float(np.linalg.norm(target.v_inf_m_s)), escape_radius_km * 1000.0, orbit.mu_m3_s2)
    else:
        a = 0.5 * (r_p + float(np.linalg.norm(target.apogee_m)))
        flight = math.pi * math.sqrt(a**3 / orbit.mu_m3_s2)
    return InjectionBurn(
        epoch_mjd2000=t_burn, coast_s=(t_burn - orbit.epoch_mjd2000) * SECONDS_PER_DAY,
        pre_burn_state=state, delta_v_m_s=dv, arg_latitude_deg=u_burn,
        periapsis_speed_m_s=v_p, eccentricity=e, c3_km2_s2=target.c3_km2_s2(r_p, orbit.mu_m3_s2),
        out_of_plane_deg=oop, flight_time_s=flight,
    )


@dataclass
class InjectionPlan:
    """Launch -> parking orbit -> injection burn, analytically planned."""
    launch: LaunchToOrbit
    target: InjectionTarget
    burn: InjectionBurn
    window: Optional[LaunchWindow] = None

    @property
    def orbit(self) -> ParkingOrbit:
        return self.launch.orbit

    def arcs(self) -> list[dict]:
        """Arc list (insertion -> coast -> burn -> post-burn coast) in the
        :func:`orbitopt.verify.tudat_propagate.propagate_multi_arc` format."""
        return [
            {"type": "coast", "duration": float(self.burn.coast_s)},
            {"type": "impulsive_burn", "delta_v": np.asarray(self.burn.delta_v_m_s, dtype=float)},
            {"type": "coast", "duration": float(self.burn.flight_time_s)},
        ]

    def lunar_b_plane_targeter_inputs(self, coast_duration_guess_s: Optional[float] = None,
                                      pre_burn_state=None, delta_v_m_s=None) -> dict:
        """Keyword arguments for :func:`orbitopt.verify.differential_correction.
        target_lunar_b_plane` / ``target_lunar_flyby``: Earth-centred
        ECLIPJ2000 pre-burn state, tudat ephemeris epochs, burn guess. Pass
        the corrected ``pre_burn_state`` / ``delta_v_m_s`` from
        :func:`verify_injection` to start from the verified burn."""
        state = self.burn.pre_burn_state if pre_burn_state is None else np.asarray(pre_burn_state, dtype=float)
        dv = self.burn.delta_v_m_s if delta_v_m_s is None else np.asarray(delta_v_m_s, dtype=float)
        state_ecl = equatorial_to_ecliptic(state)
        et = mjd2000_to_ephemeris_seconds(self.burn.epoch_mjd2000)
        return {
            "r0": state_ecl[:3], "v0_pre_burn": state_ecl[3:], "initial_epoch": et,
            "dv_guess": equatorial_to_ecliptic(dv),
            "coast_duration_guess": float(self.burn.flight_time_s if coast_duration_guess_s is None
                                          else coast_duration_guess_s),
            "reference_epoch_ephemeris_seconds": et,
        }

    def to_dict(self) -> dict:
        return {"launch": self.launch.to_dict(), "target": self.target.to_dict(),
                "burn": self.burn.to_dict(),
                "window": None if self.window is None else self.window.to_dict()}


def plan_departure(
    site: LaunchSite | str,
    target: InjectionTarget,
    start=None,
    liftoff=None,
    days: float = 1.0,
    altitude_km: float = 185.0,
    ascent_time_s: float = 600.0,
    ascent_downrange_deg: float = 20.0,
    min_coast_s: float = 0.0,
    escape_radius_km: float = EARTH_SOI_KM,
    j2_secular: bool = True,
) -> InjectionPlan:
    """End-to-end analytic plan from a launch site to the injection burn.

    Lift-off is ``liftoff`` if given, else the optimal instant of the first
    :func:`asymptote_launch_windows` window from ``start`` within ``days``
    (raises ``ValueError`` if there is none). The parking plane contains the
    site at lift-off and the target direction."""
    site = _resolve_site(site)
    rla, dla = target.rla_dla_deg
    window = None
    if liftoff is None:
        if start is None:
            raise ValueError("pass either liftoff or start")
        windows = asymptote_launch_windows(site, rla, dla, start, days)
        if not windows:
            raise ValueError(f"no launch window from {site.id} for RLA={rla:.2f} DLA={dla:.2f} "
                             f"within {days} day(s) of {start}")
        window = windows[0]
        liftoff = window.optimal_mjd2000
    launch = parking_orbit_for_asymptote(site, liftoff, rla, dla, altitude_km, ascent_time_s,
                                         ascent_downrange_deg, j2_secular)
    burn = plan_injection(launch.orbit, target, min_coast_s, escape_radius_km)
    return InjectionPlan(launch=launch, target=target, burn=burn, window=window)


# --------------------------------------------------------------------------
# Numerical propagation backends
# --------------------------------------------------------------------------

@dataclass
class ArcPropagation:
    """Duck-typed :class:`orbitopt.verify.tudat_propagate.PropagationResult`
    (what :func:`orbitopt.verify.report.propagation_section` consumes);
    states in the Earth equatorial J2000 frame."""
    epochs: np.ndarray
    states: np.ndarray
    final_state: np.ndarray
    model: dict
    segments: list
    rerun: Optional[Callable[..., "ArcPropagation"]] = field(default=None, repr=False)
    epoch_offset: float = 0.0


def _earth_rhs(mu: float, j2: bool):
    k0 = 1.5 * J2_EARTH * mu * R_EARTH_M**2

    def f(_t, y):
        r = y[:3]
        rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        a = -mu * r / rn**3
        if j2:
            z2 = (r[2] / rn) ** 2
            k = k0 / rn**5
            a = a + k * r * (5.0 * z2 - 1.0)
            a[2] -= 2.0 * k * r[2]
        return np.concatenate([y[3:], a])
    return f


def _scipy_integrator_description(rtol, atol, max_step) -> dict:
    return {"name": "dop853", "method": "Dormand-Prince 8(5,3) (scipy DOP853), adaptive",
            "adaptive": True, "step_s": None, "rtol": float(rtol), "atol": float(atol),
            "atol_units": "m and m/s, element-wise", "initial_step_s": None,
            "min_step_s": None, "max_step_s": float(max_step)}


class NumpyEarthPropagator:
    """Earth point mass (+ J2) propagator on scipy's DOP853 -- the
    dependency-light verification backend. ``coast`` has the
    :class:`orbitopt.verify.tudat_propagate.ReusableCoastPropagator` shape."""

    name = "numpy"

    def __init__(self, j2: bool = True, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                 max_step_s: float = DEFAULT_MAX_STEP_S, mu: float = MU_EARTH_M3_S2):
        self.j2 = bool(j2)
        self.rtol, self.atol, self.max_step_s, self.mu = float(rtol), float(atol), float(max_step_s), float(mu)
        self._rhs = _earth_rhs(self.mu, self.j2)

    def coast(self, state, epoch_start_s: float, duration_s: float, max_step=None):
        from scipy.integrate import solve_ivp

        sol = solve_ivp(self._rhs, (float(epoch_start_s), float(epoch_start_s) + float(duration_s)),
                        np.asarray(state, dtype=float), method="DOP853", rtol=self.rtol, atol=self.atol,
                        max_step=self.max_step_s if max_step is None else float(max_step))
        if not sol.success:
            raise RuntimeError(f"propagation failed: {sol.message}")
        return sol.t, sol.y.T

    def final_state(self, state, duration_s: float) -> np.ndarray:
        return self.coast(state, 0.0, duration_s)[1][-1]

    def model_description(self) -> dict:
        return {"central_body": "Earth", "perturbing_bodies": ["Earth"],
                "earth_spherical_harmonic": [2, 0] if self.j2 else None,
                "central_gravity": dict(EARTH_GRAVITY, gravitational_parameter=self.mu),
                "integrator": _scipy_integrator_description(self.rtol, self.atol, self.max_step_s)}

    def propagate_arcs(self, state0, arcs, initial_epoch_s: float = 0.0) -> ArcPropagation:
        return _propagate_arcs(self, state0, arcs, initial_epoch_s)

    def with_overrides(self, rtol=None, atol=None, step_size=None, perturbing_bodies=None,
                       earth_spherical_harmonic="keep") -> "NumpyEarthPropagator":
        j2 = self.j2 if earth_spherical_harmonic == "keep" else earth_spherical_harmonic is not None
        return NumpyEarthPropagator(
            j2=j2, rtol=self.rtol if rtol is None else rtol, atol=self.atol if atol is None else atol,
            max_step_s=self.max_step_s if step_size is None else step_size, mu=self.mu)


class TudatEarthPropagator:
    """tudatpy backend: Earth-centred, Earth J2 (spherical harmonic (2,0)) and
    optional third bodies, via :class:`orbitopt.verify.tudat_propagate.
    ReusableCoastPropagator`. Takes / returns equatorial states; tudat runs in
    ECLIPJ2000 and epochs are tudat ephemeris seconds."""

    name = "tudat"

    def __init__(self, perturbing_bodies: Sequence[str] = ("Earth",), j2: bool = True,
                 rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL, max_step_s: float = DEFAULT_MAX_STEP_S):
        from orbitopt.verify.tudat_propagate import ReusableCoastPropagator

        self.perturbing_bodies = tuple(perturbing_bodies)
        if "Earth" not in self.perturbing_bodies:
            self.perturbing_bodies = ("Earth",) + self.perturbing_bodies
        self.j2 = bool(j2)
        self.rtol, self.atol, self.max_step_s = float(rtol), float(atol), float(max_step_s)
        # tudat's default Earth (SPICE GM point mass / GOCO-GGM harmonics) has
        # a different GM and reference radius from the analytic plan; give it
        # this module's constants so dynamics, B-plane / apogee targets and
        # the report's conservation check all use one Earth.
        self.mu = MU_EARTH_M3_S2
        self._prop = ReusableCoastPropagator(
            central_body="Earth", perturbing_bodies=self.perturbing_bodies, step_size=self.max_step_s,
            earth_spherical_harmonic=(2, 0) if self.j2 else None, rtol=self.rtol, atol=self.atol,
            central_gravity=EARTH_GRAVITY)

    def coast(self, state, epoch_start_s: float, duration_s: float, max_step=None):
        epochs, states = self._prop.coast(equatorial_to_ecliptic(state), epoch_start_s, duration_s, max_step)
        return np.asarray(epochs), ecliptic_to_equatorial(np.asarray(states))

    def final_state(self, state, duration_s: float, epoch_start_s: float = 0.0) -> np.ndarray:
        return self.coast(state, epoch_start_s, duration_s)[1][-1]

    def model_description(self) -> dict:
        return self._prop.model_description()

    def propagate_arcs(self, state0, arcs, initial_epoch_s: float = 0.0) -> ArcPropagation:
        return _propagate_arcs(self, state0, arcs, initial_epoch_s)

    def with_overrides(self, rtol=None, atol=None, step_size=None, perturbing_bodies=None,
                       earth_spherical_harmonic="keep") -> "TudatEarthPropagator":
        j2 = self.j2 if earth_spherical_harmonic == "keep" else earth_spherical_harmonic is not None
        return TudatEarthPropagator(
            perturbing_bodies=self.perturbing_bodies if perturbing_bodies is None else perturbing_bodies,
            j2=j2, rtol=self.rtol if rtol is None else rtol, atol=self.atol if atol is None else atol,
            max_step_s=self.max_step_s if step_size is None else step_size)


def _propagate_arcs(prop, state0, arcs, initial_epoch_s: float) -> ArcPropagation:
    """Coasts + impulsive burns, concatenated (epochs relative to
    ``initial_epoch_s``), with a ``rerun`` hook for the report."""
    state = np.asarray(state0, dtype=float).copy()
    t = float(initial_epoch_s)
    all_e, all_s, segments, n = [], [], [], 0
    for arc in arcs:
        if arc["type"] == "coast":
            e, s = prop.coast(state, t, arc["duration"])
            all_e.append(np.asarray(e))
            all_s.append(np.asarray(s))
            segments.append((n, n + len(e)))
            n += len(e)
            state, t = np.asarray(s[-1]).copy(), float(e[-1])
        elif arc["type"] == "impulsive_burn":
            state = state.copy()
            state[3:] += np.asarray(arc["delta_v"], dtype=float)
        else:
            raise ValueError(f"Unknown arc type: {arc['type']!r}")
    if not all_e:
        raise ValueError("at least one coast arc is required")
    epochs = np.concatenate(all_e) - float(initial_epoch_s)
    model = prop.model_description()
    model.update(function=f"orbitopt.injection.{type(prop).__name__}.propagate_arcs",
                 initial_state=np.asarray(state0, dtype=float), initial_epoch_ephemeris_s=float(initial_epoch_s),
                 duration_s=float(epochs[-1]),
                 arcs=[{"type": "coast", "duration": float(a["duration"])} if a["type"] == "coast"
                       else {"type": "impulsive_burn", "delta_v": np.asarray(a["delta_v"], dtype=float)}
                       for a in arcs])

    def rerun(**overrides):
        return _propagate_arcs(prop.with_overrides(**overrides), state0, arcs, initial_epoch_s)

    return ArcPropagation(epochs=epochs, states=np.concatenate(all_s, axis=0), final_state=state,
                          model=model, segments=segments, rerun=rerun, epoch_offset=float(initial_epoch_s))


def make_propagator(backend: str = "numpy", **kw):
    if backend == "numpy":
        kw.pop("perturbing_bodies", None)
        return NumpyEarthPropagator(**kw)
    if backend == "tudat":
        return TudatEarthPropagator(**kw)
    raise ValueError("backend must be 'numpy' or 'tudat'")


# --------------------------------------------------------------------------
# Verification + differential correction
# --------------------------------------------------------------------------

@dataclass
class InjectionVerification:
    plan: InjectionPlan
    passed: bool
    converged: bool
    iterations: int
    pre_burn_state: np.ndarray        # numerically propagated (equatorial)
    delta_v_m_s: np.ndarray           # corrected burn (equatorial)
    correction_dv_m_s: float          # |corrected - analytic burn|
    planned_error: float              # target error of the uncorrected burn (m/s or m)
    final_error: float                # target error after correction (m/s or m)
    error_units: str
    achieved_v_inf_m_s: Optional[np.ndarray]
    achieved_apogee_m: Optional[np.ndarray]
    coast_residual_m: float
    propagation: ArcPropagation
    report: dict
    residual_history: list = field(default_factory=list)
    out_of_plane_deg: float = 0.0     # target direction's angle from the achieved orbit plane

    @property
    def achieved_c3_km2_s2(self) -> Optional[float]:
        if self.achieved_v_inf_m_s is None:
            return None
        return float(np.dot(self.achieved_v_inf_m_s, self.achieved_v_inf_m_s)) / 1e6


def _achieved(target: InjectionTarget, final_state, mu):
    """(residual vector, achieved quantity, out-of-plane miss [deg]) for the
    end state of the post-burn arc.

    Hyperbolic: the 3-D v-infinity error [m/s] -- fully controllable by the
    3-D burn (an out-of-plane asymptote just costs a plane change).

    Elliptic: only 2 components are controllable from a fixed burn point --
    an out-of-plane burn rotates the orbit about the line through the burn
    point, i.e. (almost) about the apsidal line, so the apogee cannot be
    moved out of the parking plane. The residual is therefore
    ``[r_a - r_a*, r_a* * in-plane angle error]`` [m], and the target's
    angle from the achieved orbit plane is returned separately."""
    final_state = np.asarray(final_state, dtype=float)
    if target.kind == "hyperbolic":
        from orbitopt.verify.b_plane import b_plane_state

        _b, v_inf = b_plane_state(final_state[:3], final_state[3:], mu)
        if v_inf is None:
            return np.full(3, 1e9), None, float("nan")
        h = _unit(np.cross(final_state[:3], final_state[3:]))
        oop = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(target.direction, h))))))
        return v_inf - target.v_inf_m_s, v_inf, oop
    apo = osculating_apoapsis_m(final_state, mu)
    if apo is None:
        return np.full(2, 1e12), None, float("nan")
    h = _unit(np.cross(final_state[:3], final_state[3:]))
    t_hat = target.direction
    oop = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(t_hat, h))))))
    t_p = _unit(t_hat - np.dot(t_hat, h) * h)
    a_hat = _unit(apo)
    ang = math.atan2(float(np.dot(np.cross(t_p, a_hat), h)), float(np.dot(t_p, a_hat)))
    r_a_target = float(np.linalg.norm(target.apogee_m))
    return np.array([float(np.linalg.norm(apo)) - r_a_target, r_a_target * ang]), apo, oop


def osculating_apoapsis_m(state, mu: float = MU_EARTH_M3_S2) -> Optional[np.ndarray]:
    """Apoapsis position vector of the osculating ellipse through ``state``
    (None if not elliptic). Like the B-plane vector for a hyperbola it is an
    invariant of the two-body arc -- independent of where along the arc the
    sample is taken -- so it is a well-conditioned TLI target, unlike the
    position at a fixed time (near apogee a metre-per-second change moves
    *when* apogee happens by hours)."""
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:6], dtype=float)
    rn = float(np.linalg.norm(r))
    energy = 0.5 * float(np.dot(v, v)) - mu / rn
    if energy >= 0.0:
        return None
    a = -mu / (2.0 * energy)
    e_vec = np.cross(v, np.cross(r, v)) / mu - r / rn
    e = float(np.linalg.norm(e_vec))
    if e < 1e-9:
        return None
    return -a * (1.0 + e) * e_vec / e


def verify_injection(
    plan: InjectionPlan,
    backend: str = "numpy",
    propagator=None,
    correct: bool = True,
    tol: Optional[float] = None,
    maxiter: int = 10,
    fd_step_m_s: float = 0.05,
    coast_tolerance_km: float = 50.0,
    max_out_of_plane_deg: float = 1.0,
    launch_constraints=None,
    convergence_check: bool = True,
    report_path=None,
) -> InjectionVerification:
    """Propagate the plan numerically and (optionally) differentially
    correct the burn so the propagated arc meets the target.

    Tolerance ``tol`` on the final target error defaults to 0.5 m/s of
    v-infinity (hyperbolic) or 10 km of the in-plane apogee error
    (elliptic; see :func:`_achieved` -- the out-of-plane part of an apogee
    target is not controllable from the burn and is checked against
    ``max_out_of_plane_deg`` instead, leaving it to the lunar B-plane
    targeter / TCMs). The report's checks are: integrator step convergence,
    two-body conservation, parking-coast residual (numerical vs analytic,
    ``coast_tolerance_km``), targeting convergence, launch geometry (site in plane, azimuth in a
    range-safety corridor, inclination reachable) and, if
    ``launch_constraints`` (:class:`orbitopt.launch_constraints.
    LaunchConstraints`) is given, the vehicle C3 limit (both hyperbolic and
    elliptic targets: C3 from the energy of the corrected post-burn state,
    so a TLI with C3 ~ -2 km^2/s^2 is checked against the curve too) and,
    for a hyperbolic target, the site DLA limit."""
    from orbitopt.verify import report as _report

    prop = propagator if propagator is not None else make_propagator(backend)
    target, burn, orbit = plan.target, plan.burn, plan.orbit
    mu = getattr(prop, "mu", MU_EARTH_M3_S2)
    if tol is None:
        tol = 0.5 if target.kind == "hyperbolic" else 10_000.0
    units = "m/s" if target.kind == "hyperbolic" else "m"

    t_ins = mjd2000_to_ephemeris_seconds(orbit.epoch_mjd2000)
    t_burn = t_ins + burn.coast_s
    state_ins = orbit.state_at(orbit.epoch_mjd2000)
    _e, coast_states = prop.coast(state_ins, t_ins, burn.coast_s)
    pre = np.asarray(coast_states[-1], dtype=float)
    coast_residual = float(np.linalg.norm(pre[:3] - burn.pre_burn_state[:3]))

    def residual(dv):
        post = pre.copy()
        post[3:] += dv
        _t, s = prop.coast(post, t_burn, burn.flight_time_s)
        return _achieved(target, np.asarray(s[-1]), mu)

    dv = np.asarray(burn.delta_v_m_s, dtype=float).copy()
    res, achieved, oop = residual(dv)
    planned_error = float(np.linalg.norm(res))
    history = [planned_error]
    iterations = 0
    if correct:
        while history[-1] > tol and iterations < maxiter:
            jac = np.empty((res.size, 3))
            for k in range(3):
                step = np.zeros(3)
                step[k] = fd_step_m_s
                jac[:, k] = (residual(dv + step)[0] - res) / fd_step_m_s
            # Minimum-norm step (square 3x3 for v-infinity, 2x3 for apogee).
            delta = np.linalg.lstsq(jac, -res, rcond=None)[0]
            # Backtracking keeps a bad linearisation from overshooting.
            lam = 1.0
            while True:
                trial = dv + lam * delta
                r_trial, a_trial, o_trial = residual(trial)
                if np.linalg.norm(r_trial) < history[-1] or lam < 1e-3:
                    break
                lam *= 0.5
            dv, res, achieved, oop = trial, r_trial, a_trial, o_trial
            iterations += 1
            history.append(float(np.linalg.norm(res)))
    final_error = history[-1]
    converged = final_error <= tol

    arcs = [{"type": "coast", "duration": float(burn.coast_s)},
            {"type": "impulsive_burn", "delta_v": dv},
            {"type": "coast", "duration": float(burn.flight_time_s)}]
    full = prop.propagate_arcs(state_ins, arcs, t_ins)
    section = _report.propagation_section(full, "launch-injection", mu, convergence_check=convergence_check)
    section["dynamics"]["frame"] = "J2000 equatorial (EME2000)" if prop.name == "numpy" else "ECLIPJ2000 (reported states rotated to J2000 equatorial)"

    residuals = [_report.state_residual(
        "parking_coast_vs_analytic", "analytic circular parking orbit with secular J2",
        position=pre[:3], reference_position=burn.pre_burn_state[:3],
        tolerance={"position_m": coast_tolerance_km * 1000.0})]
    targets = {
        "converged": converged, "kind": target.kind, "iterations": iterations,
        "tolerance": tol, "error_units": units,
        "planned_error": planned_error, "final_error": final_error, "residual_history": history,
        "planned_delta_v_m_s": burn.delta_v_m_s, "delta_v_m_s": dv,
        "correction_dv_m_s": float(np.linalg.norm(dv - burn.delta_v_m_s)),
        "target_v_inf_m_s": target.v_inf_m_s, "target_apogee_m": target.apogee_m,
        "achieved_v_inf_m_s": achieved if target.kind == "hyperbolic" else None,
        "achieved_apogee_m": achieved if target.kind == "elliptic" else None,
        "target_out_of_plane_deg": oop,
    }
    lv = plan.launch
    extra_checks = {
        "launch.site_in_plane": abs(lv.site_offplane_deg) < 1e-6,
        "launch.azimuth_in_corridor": lv.azimuth_in_corridor,
        "launch.inclination_reachable": lv.inclination_reachable,
    }
    if target.kind == "elliptic":
        targets["max_out_of_plane_deg"] = float(max_out_of_plane_deg)
        extra_checks["targeting.out_of_plane"] = bool(abs(oop) <= max_out_of_plane_deg)
    if launch_constraints is not None:
        # The vehicle limit applies to every injection: C3 = 2 * specific
        # energy of the post-burn arc (negative for TLI / any elliptic
        # target), compared against the vehicle's C3 curve at the payload.
        # The site DLA limit only exists for a departure asymptote.
        # (Hyperbolic: |v_inf|^2 at the end of the escape arc, which includes
        # the J2 energy exchange; elliptic, or a failed escape: post-burn
        # point-mass energy.)
        if target.kind == "hyperbolic" and achieved is not None:
            c3_achieved = float(np.dot(achieved, achieved)) / 1e6
        else:
            post = pre.copy()
            post[3:] += dv
            c3_achieved = (float(np.dot(post[3:], post[3:]))
                           - 2.0 * mu / float(np.linalg.norm(post[:3]))) / 1e6
        targets["achieved_c3_km2_s2"] = c3_achieved
        targets["launch_constraints"] = launch_constraints.describe()
        if launch_constraints.vehicle is not None:
            targets["max_c3_km2_s2"] = launch_constraints.max_c3
            extra_checks["launch.c3_within_vehicle"] = bool(c3_achieved <= launch_constraints.max_c3)
        if target.kind == "hyperbolic":
            if achieved is None:
                extra_checks["launch.dla_within_site"] = False
            else:
                v = achieved.reshape(1, 3)
                if launch_constraints.frame == "ecliptic":
                    v = equatorial_to_ecliptic(v)
                ev = launch_constraints.evaluate(v)
                extra_checks["launch.dla_within_site"] = bool(float(ev.dla_excess_deg[0]) <= 0.0)

    report = _report.build_report(
        kind="launch-injection", label=f"{lv.site_id}-{target.name or target.kind}-{orbit.epoch_mjd2000:.5f}",
        propagations=[section], residuals=residuals, inputs=plan.to_dict(), targets=targets,
    )
    report["status"]["checks"].update(extra_checks)
    report["status"]["passed"] = all(report["status"]["checks"].values())
    report["digest"] = _report.report_digest(report)
    _report.finalize_report(report, report_path)

    return InjectionVerification(
        plan=plan, passed=bool(report["status"]["passed"]), converged=converged, iterations=iterations,
        pre_burn_state=pre, delta_v_m_s=dv, correction_dv_m_s=float(np.linalg.norm(dv - burn.delta_v_m_s)),
        planned_error=planned_error, final_error=final_error, error_units=units,
        achieved_v_inf_m_s=achieved if target.kind == "hyperbolic" else None,
        achieved_apogee_m=achieved if target.kind == "elliptic" else None,
        coast_residual_m=coast_residual, propagation=full, report=report, residual_history=history,
        out_of_plane_deg=oop,
    )
