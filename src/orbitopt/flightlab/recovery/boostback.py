"""Boostback & entry burn guidance (FL-302): parametric burns optimized so the
booster reaches the landing-burn initiation corridor.

The planner flies the E3 recovery article (FL-301's 3-DOF propagator, ECEF,
full physics) with a *scripted* two-burn mission and optimizes the script's
seven numbers against the corridor:

* boostback: ``start_delay_s`` after separation, ``duration_s`` long, at
  ``throttle``, pointing at ``pitch_deg`` above the horizon with ``yaw_deg``
  of trim right of the **live bearing to the target** in the current local
  ENU frame (bearing-guided, like a flight computer steering the landing
  site);
* entry burn: ignition when descending through ``trigger_altitude_m``,
  ``duration_s`` long at full throttle, thrust along retro-grade tilted by a
  fixed lead toward vertical-down (:data:`ENTRY_BURN_LEAD_RAD`) so the burn
  both decelerates and pitches the velocity vector vertical;
* between/after: unpowered -- drag and gravity do what drag and gravity do.

The optimization runs on scipy's Nelder-Mead (the compute stack's pygmo is
deliberately absent from the base install; see pyproject). Objective: the
corridor residual at the first descending crossing of ``corridor.h0_m`` after
the entry-burn ignition -- horizontal miss against ``radius_m``, velocity
tilt against ``max_tilt_deg``, speed against ``v_max_m_s`` -- plus propellant
at the corridor against the landing-burn need plus the vehicle reserve, and
the entry-phase dynamic-pressure / proper-g limits as penalties. Every
trajectory the optimizer ever sees is one ``propagate()`` call; nothing is
estimated with a kinematic shortcut, and the accepted plan is re-integrated
at the final (finer) step before it is reported.

Feasibility language (acceptance criteria): a plan that cannot make the
corridor comes back ``feasible=False`` with ``reason_codes[0]`` the governing
one, drawn from::

    INSUFFICIENT_PROPELLANT_BOOSTBACK    separation propellant below the
                                         analytic null-the-horizontal-velocity
                                         floor (never even tries), or a
                                         boostback burn that runs dry
    INSUFFICIENT_PROPELLANT_ENTRY_BURN   the entry burn ran dry mid-burn
    INSUFFICIENT_PROPELLANT_RESERVE      corridor reached but prop left is
                                         below landing-burn need + reserve
    ENTRY_Q_LIMIT_EXCEEDED / ENTRY_G_LIMIT_EXCEEDED   entry-phase peak q or
                                         proper g above the entry limit
    CORRIDOR_NOT_REACHED                 never crossed h0 after entry ignition
    CORRIDOR_MISSED                      crossed, outside the corridor

The corridor is the *landing-burn initiation* corridor: FL-303 picks the
plan up at ``ReturnPlan.landing_burn_initial_state`` (an ECEF ``BoosterState``
the propagator can restart from) and its ENU mirror
``landing_burn_initial_state_enu`` at the pad frame the interface specifies
(``BoosterState.frame`` is frozen to the contract frames by FL-301, so the
pad-local ENU numbers ride as a first-class field beside it).

The "entry phase" the q/g limits watch is the post-boostback descent: the
samples from the boostback burn-out to the corridor crossing (peak q and peak
proper load of that window, rebuilt from the track's own samples). Entry
limits default to :data:`DEFAULT_ENTRY_Q_MAX_PA` / :data:`DEFAULT_ENTRY_G_MAX`
-- DESIGN ASSUMPTIONS for the illustrative vehicle (an entry-rated structure
tolerates a hotter entry than the ascent max-q limit; the recovery article's
own table lives in the vehicle YAML when a real one exists), overridable per
call.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize

from orbitopt.entry import G0_M_S2
from orbitopt.flightlab.frames import WGS84_A, WGS84_F, geodetic_to_ecef
from orbitopt.flightlab.recovery.propagator import (
    AltitudeThreshold,
    BoosterModel,
    BoosterState,
    BoosterTrack,
    RecoveryError,
    ThrustCmd,
    _to_ecef_state,
    pressure_pa,
    propagate,
)

__all__ = [
    "CORRIDOR_MISSED",
    "CORRIDOR_NOT_REACHED",
    "DEFAULT_CORRIDOR",
    "DEFAULT_ENTRY_G_MAX",
    "DEFAULT_ENTRY_Q_MAX_PA",
    "ENTRY_BURN_LEAD_RAD",
    "ENTRY_G_LIMIT_EXCEEDED",
    "ENTRY_Q_LIMIT_EXCEEDED",
    "INSUFFICIENT_PROPELLANT_BOOSTBACK",
    "INSUFFICIENT_PROPELLANT_ENTRY_BURN",
    "INSUFFICIENT_PROPELLANT_RESERVE",
    "BoostbackPlan",
    "CorridorReport",
    "EntryBurnPlan",
    "LandingBurnEnu",
    "RecoveryCorridor",
    "RecoveryTarget",
    "ReturnPlan",
    "landing_burn_propellant_need_kg",
    "mission_command",
    "plan_boostback_entry",
]

# --------------------------------------------------------------------------
# reason codes (acceptance: "reports INSUFFICIENT_PROPELLANT_BOOSTBACK etc")
# --------------------------------------------------------------------------

INSUFFICIENT_PROPELLANT_BOOSTBACK = "INSUFFICIENT_PROPELLANT_BOOSTBACK"
INSUFFICIENT_PROPELLANT_ENTRY_BURN = "INSUFFICIENT_PROPELLANT_ENTRY_BURN"
INSUFFICIENT_PROPELLANT_RESERVE = "INSUFFICIENT_PROPELLANT_RESERVE"
ENTRY_Q_LIMIT_EXCEEDED = "ENTRY_Q_LIMIT_EXCEEDED"
ENTRY_G_LIMIT_EXCEEDED = "ENTRY_G_LIMIT_EXCEEDED"
CORRIDOR_NOT_REACHED = "CORRIDOR_NOT_REACHED"
CORRIDOR_MISSED = "CORRIDOR_MISSED"

#: Fixed pitch-lead of the entry-burn thrust vector from pure retro-grade
#: toward vertical-down [rad]: retro-grade brakes, lead verticalizes. An
#: internal guidance constant, not an optimizer parameter (the task scope
#: pins the entry burn to start altitude + duration).
ENTRY_BURN_LEAD_RAD = 0.22

#: Entry-phase dynamic-pressure limit [Pa] -- DESIGN ASSUMPTION (illustrative
#: entry rating; higher than the ascent max-q limit by intent, since the
#: vehicle is decelerating from orbital speed under power).
DEFAULT_ENTRY_Q_MAX_PA = 150_000.0

#: Entry-phase proper-load limit [g] -- DESIGN ASSUMPTION (illustrative).
DEFAULT_ENTRY_G_MAX = 8.0


# --------------------------------------------------------------------------
# corridor, target, plan records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryCorridor:
    """Landing-burn initiation corridor (the interface of FL-302, frozen):
    at ``h0_m`` above the ellipsoid, over the target within ``radius_m``,
    velocity tilted no more than ``max_tilt_deg`` from vertical-down and
    slower than ``v_max_m_s``."""

    h0_m: float
    radius_m: float
    v_max_m_s: float
    max_tilt_deg: float

    def __post_init__(self) -> None:
        if self.h0_m <= 0.0:
            raise ValueError(f"corridor h0_m must be positive, got {self.h0_m}")
        if self.radius_m <= 0.0:
            raise ValueError(f"corridor radius_m must be positive, got {self.radius_m}")
        if self.v_max_m_s <= 0.0:
            raise ValueError(f"corridor v_max_m_s must be positive, got {self.v_max_m_s}")
        if not 0.0 < self.max_tilt_deg <= 90.0:
            raise ValueError(f"corridor max_tilt_deg must be in (0, 90], got {self.max_tilt_deg}")


#: Illustrative landing-burn initiation corridor for the synthetic vehicle:
#: engines-on at 3 km, under 350 m/s within 8 km of the pad, within 15 deg
#: of vertical -- the FL-303 landing burn from these states closes on the pad
#: with the vehicle's landing reserve (see the module docstring).
DEFAULT_CORRIDOR = RecoveryCorridor(h0_m=3000.0, radius_m=8000.0,
                                    v_max_m_s=350.0, max_tilt_deg=15.0)


@dataclass(frozen=True)
class RecoveryTarget:
    """Where the booster is being flown back to. ``kind`` follows the Flight
    Lab target vocabulary: ``"rtls"`` (return to the launch site) or
    ``"downrange"`` (a landing site downrange of the launch site); the
    guidance math is the same either way -- the distinction is contract
    vocabulary and what ``from_scenario`` accepted as the target name."""

    kind: str
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    corridor: RecoveryCorridor = DEFAULT_CORRIDOR

    def __post_init__(self) -> None:
        if self.kind not in ("rtls", "downrange"):
            raise ValueError(
                f"recovery target kind must be 'rtls' or 'downrange', got {self.kind!r}")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude_deg}")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(f"longitude out of range: {self.longitude_deg}")

    @property
    def position_ecef_m(self) -> tuple[float, float, float]:
        return tuple(float(c) for c in geodetic_to_ecef(
            self.latitude_deg, self.longitude_deg, self.altitude_m))

    @classmethod
    def from_scenario(cls, scenario: Mapping, *, landing_sites: Mapping | None = None,
                      corridor: RecoveryCorridor | None = None) -> "RecoveryTarget":
        """Build the target from a parsed Flight Lab scenario/flight-target
        document: the ``recovery:`` section picks kind + target name, the
        ``launch_site:`` section (or a named entry of ``landing_sites``,
        ``{name: {latitude_deg, longitude_deg, altitude_m}}``) the position.
        ``kind: none`` has no recovery target to build and is an error here."""
        rec = scenario.get("recovery") or {}
        kind = str(rec.get("kind", "none"))
        if kind not in ("rtls", "downrange"):
            raise ValueError(f"scenario recovery.kind {kind!r} is not a recovery target")
        name = str(rec.get("target", "launch_site"))
        if name == "launch_site":
            site = scenario.get("launch_site") or {}
            if "latitude_deg" not in site or "longitude_deg" not in site:
                raise ValueError("recovery.target is launch_site but the scenario has no launch_site position")
            lat, lon = float(site["latitude_deg"]), float(site["longitude_deg"])
            alt = float(site.get("altitude_wgs84_m", 0.0))
        else:
            site = (landing_sites or {}).get(name)
            if site is None:
                raise ValueError(
                    f"recovery target {name!r} is not launch_site and not in landing_sites")
            lat, lon = float(site["latitude_deg"]), float(site["longitude_deg"])
            alt = float(site.get("altitude_m", site.get("altitude_wgs84_m", 0.0)))
        return cls(kind=kind, name=name, latitude_deg=lat, longitude_deg=lon,
                   altitude_m=alt, corridor=corridor or DEFAULT_CORRIDOR)


@dataclass(frozen=True)
class BoostbackPlan:
    """The optimized boostback script (FL-303/FL-304 consumers: re-fly with
    :func:`mission_command` or trust ``ReturnPlan.track``). ``yaw_deg`` is
    the heading trim [deg, +right] around the live bearing to the target."""

    start_delay_s: float
    duration_s: float
    throttle: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True)
class EntryBurnPlan:
    """The optimized entry-burn script: ignite at the descent crossing of
    ``trigger_altitude_m``, burn ``duration_s`` at full throttle."""

    trigger_altitude_m: float
    duration_s: float


@dataclass(frozen=True)
class LandingBurnEnu:
    """The corridor-insertion state in the pad-local ENU frame (position /
    velocity + mass on the mission clock) -- the FL-302 interface's
    "position/velocity in a local ENU frame at the pad + mass". East/North/
    Up components of the (ECEF) offset rotated into the pad's ENU basis; the
    pad is the target site."""

    t_s: float
    position_enu_m: tuple[float, float, float]
    velocity_enu_m: tuple[float, float, float]
    mass_kg: float


@dataclass(frozen=True)
class CorridorReport:
    """Measured corridor state at the insertion crossing (the numbers the
    feasibility check actually compared; ``reached=False`` = never crossed
    h0 after entry ignition, residual fields None where not measured)."""

    reached: bool
    t_s: float | None = None
    miss_m: float | None = None
    tilt_deg: float | None = None
    speed_m_s: float | None = None
    propellant_available_kg: float | None = None
    landing_need_kg: float | None = None
    reserve_kg: float | None = None
    entry_max_q_pa: float | None = None
    entry_max_proper_g: float | None = None


@dataclass(frozen=True)
class ReturnPlan:
    """The FL-302 output (interface frozen): ``landing_burn_initial_state``
    (ECEF ``BoosterState`` at corridor insertion, ready for FL-301's
    propagator / FL-303's landing burn), its pad-ENU mirror
    ``landing_burn_initial_state_enu``, and the ``track`` the state came
    from -- the full re-integrated rollout from separation to the corridor
    crossing or beyond (on an infeasible plan the track runs to
    impact/terminal; the corridor report says what happened).
    ``reason_codes[0]`` is the governing reason when ``feasible=False``."""

    feasible: bool
    reason_codes: tuple[str, ...]
    target: RecoveryTarget
    corridor: RecoveryCorridor
    boostback: BoostbackPlan
    entry_burn: EntryBurnPlan
    landing_burn_initial_state: BoosterState | None
    landing_burn_initial_state_enu: LandingBurnEnu | None
    track: BoosterTrack
    corridor_report: CorridorReport
    ignitions_used: int = 0
    assumptions: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# local-frame helpers (scalar, no numpy: these run inside every guidance step)
# --------------------------------------------------------------------------

_E2 = WGS84_F * (2.0 - WGS84_F)


def _geodetic_fast(r: Sequence[float]) -> tuple[float, float, float]:
    """Scalar WGS84 (lat_deg, lon_deg, h_m) of an ECEF point (Bowring's
    closed form -- sub-microsecond, geodetic accuracy; the propagator's own
    event bisection stays the authority, this only drives guidance/trigger
    logic and the corridor read-out)."""
    x, y, z = (float(c) for c in r)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
    else:
        th = math.atan2(z, p * (1.0 - WGS84_F))
        st, ct = math.sin(th), math.cos(th)
        lat = math.atan2(z + (WGS84_A * _E2 / (1.0 - WGS84_F)) * st ** 3,
                         p - WGS84_A * _E2 * ct ** 3)
    s_lat = math.sin(lat)
    n_rm = WGS84_A / math.sqrt(1.0 - _E2 * s_lat * s_lat)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n_rm


def _enu_basis(lat_deg: float, lon_deg: float) -> tuple[tuple[float, float, float], ...]:
    """(east, north, up) unit vectors in ECEF components at a geodetic point."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    sa, ca, so = math.sin(la), math.cos(la), math.sin(lo)
    co = math.cos(lo)
    return ((-so, co, 0.0), (-sa * co, -sa * so, ca), (ca * co, ca * so, sa))


# --------------------------------------------------------------------------
# landing-burn propellant need (the corridor's "landing need + reserve")
# --------------------------------------------------------------------------


def landing_burn_propellant_need_kg(mass_kg: float, speed_m_s: float,
                                    altitude_m: float, vehicle: BoosterModel) -> float:
    """Propellant [kg] to fly the vertical landing burn from a corridor state:
    cancel ``speed_m_s`` at ambient ``altitude_m`` with the engine group at
    full throttle against sea-level Isp (the pessimistic end of the Isp
    table -- the burn spends most of its time low where Isp is lowest).
    Momentum analysis of a constant-thrust vertical burn with gravity loss:
    ``v = Isp g0 ln(m0/m1) - g0 (m0 - m1) Isp g0 / T``. Returns the vehicle's
    full remaining propellant when even a burn to dry mass cannot stop the
    vehicle (the caller's margin check then fails)."""
    eng = vehicle.engine
    dry = float(vehicle.dry_mass_kg)
    m0 = float(mass_kg)
    v = float(speed_m_s)
    if m0 <= dry or v <= 0.0:
        return 0.0
    thrust_sl = eng.count * (eng.thrust_vacuum_n
                             - eng.nozzle_exit_area_m2 * pressure_pa(altitude_m))
    if thrust_sl < 0.6 * eng.count * eng.thrust_vacuum_n:
        # nozzle overexpanded past usefulness; take 60 % of vacuum thrust as
        # the pessimistic floor (a landing burn is flown where it re-attaches).
        thrust_sl = max(thrust_sl, 0.6 * eng.count * eng.thrust_vacuum_n)
    isp = eng.isp_sea_level_s
    mdot = thrust_sl / (isp * G0_M_S2)
    t_burn_per_kg = 1.0 / mdot

    def residual(m1: float) -> float:
        return (isp * G0_M_S2 * math.log(m0 / m1)
                - G0_M_S2 * (m0 - m1) * t_burn_per_kg - v)

    if residual(dry + 1e-6) <= 0.0:
        return m0 - dry  # cannot stop even burning to dry: all of it, still short
    return m0 - brentq(residual, dry + 1e-6, m0 - 1e-6)


# --------------------------------------------------------------------------
# mission script: the guidance callback flown through the FL-301 propagator
# --------------------------------------------------------------------------


class _Mission:
    """Scripted two-burn return: coast -> boostback (delay/duration/throttle/
    pitch/yaw) -> coast (drag, gravity) -> entry burn (trigger altitude /
    duration, full throttle, retro + lead) -> corridor. Sampled once per
    propagator step like a flight computer."""

    def __init__(self, t_sep: float, bb: BoostbackPlan, eb: EntryBurnPlan,
                 bearing_of: Callable[[np.ndarray], float]):
        self.t_sep = float(t_sep)
        self.bb = bb
        self.eb = eb
        self.bearing_of = bearing_of
        self.entry_t0: float | None = None

    def __call__(self, t: float, state: BoosterState) -> ThrustCmd | None:
        t0 = self.t_sep + self.bb.start_delay_s
        if t < t0:
            return None
        r = np.asarray(state.position_m, dtype=np.float64)
        v = np.asarray(state.velocity_m_s, dtype=np.float64)
        if t < t0 + self.bb.duration_s:
            lat, lon, _h = _geodetic_fast(r)
            e, n, u = _enu_basis(lat, lon)
            p = math.radians(self.bb.pitch_deg)
            y = self.bearing_of(r) + math.radians(self.bb.yaw_deg)
            d = (math.cos(p) * math.sin(y) * np.array(e)
                 + math.cos(p) * math.cos(y) * np.array(n)
                 + math.sin(p) * np.array(u))
            return ThrustCmd(self.bb.throttle, d)
        up = r / float(np.linalg.norm(r))
        if self.entry_t0 is None:
            if t < t0 + self.bb.duration_s:
                return None  # entry ignition only after boostback cutoff
            _lat, _lon, h = _geodetic_fast(r)
            if h <= self.eb.trigger_altitude_m and float(v @ up) < 0.0:
                self.entry_t0 = float(t)
            return None
        if t < self.entry_t0 + self.eb.duration_s:
            spd = float(np.linalg.norm(v))
            if spd < 1e-9:
                return None  # nothing to fly against; keep the engines shut
            vv = v / spd
            dd = -up
            pd = dd - float(dd @ vv) * vv
            nrm = float(np.linalg.norm(pd))
            if nrm < 1e-9:  # velocity already vertical: burn pure retro
                pd = vv
            else:
                pd = pd / nrm
            d = (-math.cos(ENTRY_BURN_LEAD_RAD) * vv
                 + math.sin(ENTRY_BURN_LEAD_RAD) * pd)
            return ThrustCmd(1.0, d)
        return None


def _bearing_of_target(pad_ecef: np.ndarray) -> Callable[[np.ndarray], float]:
    """Bearing-guidance function: bearing [rad, ENU convention from north] of
    the pad from any ECEF point, on the local tangent plane (accurate to
    metres at the tens-of-km distances the guidance cares about)."""
    def f(r_ecef: np.ndarray) -> float:
        lat, lon, _h = _geodetic_fast(r_ecef)
        e, n, _u = _enu_basis(lat, lon)
        d = pad_ecef - np.asarray(r_ecef, dtype=np.float64)
        return math.atan2(float(d @ np.array(e)), float(d @ np.array(n)))
    return f


# --------------------------------------------------------------------------
# rollout evaluation
# --------------------------------------------------------------------------


@dataclass
class _Rollout:
    track: BoosterTrack | None = None
    crossed: bool = False
    event: Any = None
    miss_m: float | None = None
    tilt_deg: float | None = None
    speed_m_s: float | None = None
    prop_kg: float | None = None
    landing_need_kg: float | None = None
    entry_q_pa: float = 0.0
    entry_g: float = 0.0
    exhausted: str | None = None          # "boostback" | "entry"
    impact_miss_m: float = float("inf")
    penalty: float = float("inf")


def _entry_window_loads(track: BoosterTrack, vehicle: BoosterModel,
                        t_lo: float, t_hi: float) -> tuple[float, float]:
    """(max dynamic pressure [Pa], max proper load [g]) over samples in
    ``[t_lo, t_hi]``. Proper load is rebuilt from the track's own samples
    (q, Mach, mass, flown throttle, height) as |drag| + |thrust| over m g0 --
    conservative versus the propagator's vector sum, on the sample grid, and
    windowed to the post-boostback descent (the track's global scalars would
    fold in the impact spike below the corridor)."""
    eng = vehicle.engine
    q_max = 0.0
    g_max = 0.0
    for i in range(len(track.times_s)):
        t = track.times_s[i]
        if t < t_lo or t > t_hi:
            continue
        q = float(track.dynamic_pressure_pa[i])
        q_max = max(q_max, q)
        m = float(track.mass_kg[i])
        f_aero = q * vehicle.cd(float(track.mach[i])) * vehicle.reference_area_m2
        thr = float(track.throttle[i])
        f_thrust = (eng.count * thr
                    * max(eng.thrust_vacuum_n
                          - eng.nozzle_exit_area_m2 * pressure_pa(float(track.height_m_arr[i])),
                          0.0)) if thr > 0.0 else 0.0
        g_max = max(g_max, (f_aero + f_thrust) / (G0_M_S2 * m))
    return q_max, g_max


def _evaluate(sep_state: BoosterState, mission: _Mission, vehicle: BoosterModel,
              target: RecoveryTarget, corridor: RecoveryCorridor,
              q_lim: float, g_lim: float, reserve: float, horizon_s: float,
              dt: float, bb_window: tuple[float, float]) -> _Rollout:
    """One full-physics flight through the FL-301 propagator plus the
    corridor residual. Penalty is a sum of dimensionless squared violations:
    0.0 iff every corridor/entry/propellant criterion passes in-flight."""
    out = _Rollout()
    h0_name = "fl302_corridor_h0"
    try:
        tr = propagate(sep_state, vehicle, mission, sep_state.t_s + horizon_s,
                       events=("burn", "ground", AltitudeThreshold(corridor.h0_m, h0_name, "down")),
                       dt_s=dt, sample_s=max(2.0 * dt, 2.0))
    except RecoveryError:
        out.penalty = 60.0
        return out
    out.track = tr

    burn_starts = [e for e in tr.events if e.kind == "burn_start"]
    entry_t0 = burn_starts[1].t_s if len(burn_starts) > 1 else None
    for e in tr.events:
        if e.kind == "burn_end" and e.data.get("reason") == "PROPELLANT_EXHAUSTED":
            t0, t1 = bb_window
            out.exhausted = "boostback" if t0 - 1e-6 <= e.t_s <= t1 + 1.5 * dt else "entry"

    pad = np.asarray(target.position_ecef_m, dtype=np.float64)
    crossings = [e for e in tr.events
                 if e.kind == h0_name and entry_t0 is not None and e.t_s >= entry_t0]
    if crossings:
        ev = crossings[0]
        r = np.asarray(ev.data["position_ecef_m"], dtype=np.float64)
        v = np.asarray(ev.data["velocity_ecef_m_s"], dtype=np.float64)
        lat, lon, _h = _geodetic_fast(r)
        e_, n_, _u = _enu_basis(lat, lon)
        d = pad - r
        out.miss_m = math.hypot(float(d @ np.array(e_)), float(d @ np.array(n_)))
        up = r / float(np.linalg.norm(r))
        spd = float(np.linalg.norm(v))
        out.speed_m_s = spd
        out.tilt_deg = math.degrees(math.acos(
            max(-1.0, min(1.0, -float(v @ up) / max(spd, 1e-9)))))
        out.prop_kg = float(ev.data["mass_kg"]) - float(vehicle.dry_mass_kg)
        out.landing_need_kg = landing_burn_propellant_need_kg(
            float(ev.data["mass_kg"]), spd, corridor.h0_m, vehicle)
        out.crossed = True
        out.event = ev

    t_q = burn_starts[1].t_s if len(burn_starts) > 1 else (
        burn_starts[0].t_s + mission.bb.duration_s if burn_starts else tr.times_s[0])
    t_q_end = float(ev.t_s) if out.crossed else float(tr.times_s[-1])
    out.entry_q_pa, out.entry_g = _entry_window_loads(tr, vehicle, t_q, t_q_end)

    if not out.crossed and tr.impact is not None:
        out.impact_miss_m = float(np.linalg.norm(pad - np.asarray(
            tr.impact.position_ecef_m, dtype=np.float64)))

    # ---- penalty: soft-quadratic violations, dimensionless
    pen = 0.0
    if out.exhausted == "boostback":
        pen += 50.0
    elif out.exhausted == "entry":
        pen += 10.0
    if out.crossed:
        pen += (out.miss_m / corridor.radius_m) ** 2
        pen += (max(0.0, out.tilt_deg - corridor.max_tilt_deg) / 10.0) ** 2
        pen += (max(0.0, out.speed_m_s - corridor.v_max_m_s) / 50.0) ** 2
        short = out.landing_need_kg + reserve - out.prop_kg
        pen += (max(0.0, short) / 1000.0) ** 2
        pen += (max(0.0, out.entry_q_pa - q_lim) / 50_000.0) ** 2
        pen += (max(0.0, out.entry_g - g_lim) / 2.0) ** 2
    elif tr.impact is not None:
        pen += 25.0 + 2.0 * min(out.impact_miss_m, 20.0 * corridor.radius_m) / corridor.radius_m
    else:
        pen += 35.0
    out.penalty = pen
    return out


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------


def _propellant_floor_kg(sep_mass: float, horiz_speed: float, isp_vac_s: float) -> float:
    """Analytic boostback floor [kg]: the propellant to null the horizontal
    separation velocity at vacuum Isp with zero losses. Below this the
    booster cannot even cancel its downrange motion -- no optimizer rollout
    can fix arithmetic, so the planner reports it straight away."""
    if horiz_speed <= 0.0:
        return 0.0
    return sep_mass * (1.0 - math.exp(-horiz_speed / (isp_vac_s * G0_M_S2)))


def plan_boostback_entry(sep_state: BoosterState, booster: BoosterModel,
                        recovery_target: RecoveryTarget, *,
                        entry_q_max_pa: float = DEFAULT_ENTRY_Q_MAX_PA,
                        entry_g_max: float = DEFAULT_ENTRY_G_MAX,
                        horizon_s: float = 900.0,
                        dt_opt: float = 2.0, dt_final: float = 0.5,
                        max_fevals: int = 120,
                        initial_guess: Sequence[float] | None = None) -> ReturnPlan:
    """Optimize the boostback/entry-burn script so the FL-301-propagated
    booster reaches ``recovery_target.corridor``; returns the corridor
    insertion state, its pad-ENU mirror, and the flown track.

    ``sep_state`` is the FL-301 separation state (any contract frame),
    ``booster`` the recovery vehicle model, ``recovery_target`` an rtls or
    downrange landing site. The entry-phase q/g limits default to the module
    DESIGN ASSUMPTIONs. All trajectories are flown by
    ``orbitopt.flightlab.recovery.propagator.propagate`` -- the optimizer
    only ever sees full-physics rollouts, and the accepted solution is
    re-integrated at ``dt_final`` before it is reported (no kinematic
    shortcuts). ``initial_guess`` (delay, duration, throttle, pitch, yaw
    trim, trigger altitude, entry duration) overrides the default starting
    point; ``max_fevals`` budgets the search (a restart from a jittered
    point spends at most half of it, and budgets below 40 keep the search
    where it started -- pin a script by passing both).
    Raises ``RecoveryError`` when the vehicle has fewer than two restarts
    available (boostback and entry burns each need one)."""
    corridor = recovery_target.corridor
    eng = booster.engine
    if eng.restarts_max < 2:
        raise RecoveryError(
            f"boostback + entry burns need 2 ignitions, vehicle allows {eng.restarts_max}")
    dry = float(booster.dry_mass_kg)
    sep_mass = float(sep_state.mass_kg)
    reserve = float(booster.landing_propellant_reserve_kg)

    # The propagator is frame-agnostic internally (ECEF); all guidance math
    # here is ECEF/ENU, so bring the separation state to ECEF once, here.
    if sep_state.frame != "ECEF":
        r_e, v_e = _to_ecef_state(sep_state)
        sep_state = BoosterState(t_s=sep_state.t_s, epoch_utc=sep_state.epoch_utc,
                                 frame="ECEF", position_m=tuple(float(c) for c in r_e),
                                 velocity_m_s=tuple(float(c) for c in v_e),
                                 mass_kg=sep_mass)
    sep_r = np.asarray(sep_state.position_m, dtype=np.float64)
    pad_ecef = np.asarray(recovery_target.position_ecef_m, dtype=np.float64)
    bearing_of = _bearing_of_target(pad_ecef)

    def clamp(x: np.ndarray) -> np.ndarray:
        # x[4] is the boostback yaw trim [deg, +right] around the live
        # bearing to the target (a local-frame direction parameter bounded
        # like a real trim authority); the mission law adds the bearing.
        delay, dur, thr, pitch, yaw, h_trig, dur_e = x
        delay = min(max(float(delay), 0.0), min(horizon_s * 0.4, 150.0))
        dur = min(max(float(dur), 4.0), 60.0)
        thr = min(max(float(thr), eng.throttle_min), eng.throttle_max)
        pitch = min(max(float(pitch), 0.0), 30.0)
        yaw = min(max(float(yaw), -45.0), 45.0)
        h_trig = min(max(float(h_trig), corridor.h0_m + 2000.0), 35_000.0)
        dur_e = min(max(float(dur_e), 3.0), 45.0)
        return np.array([delay, dur, thr, pitch, yaw, h_trig, dur_e])

    def to_mission(x: np.ndarray) -> _Mission:
        bb = BoostbackPlan(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        eb = EntryBurnPlan(float(x[5]), float(x[6]))
        return _Mission(sep_state.t_s, bb, eb, bearing_of)

    def rollout(x: np.ndarray, dt: float) -> _Rollout:
        x = clamp(x)
        window = (sep_state.t_s + float(x[0]), sep_state.t_s + float(x[0]) + float(x[1]))
        return _evaluate(sep_state, to_mission(x), booster, recovery_target, corridor,
                         entry_q_max_pa, entry_g_max, reserve, horizon_s, dt, window)

    # --- analytic floor check (see _propellant_floor_kg)
    sep_prop = sep_mass - dry
    sep_lat, sep_lon, _sep_h = _geodetic_fast(sep_r)
    e_, n_, _u = _enu_basis(sep_lat, sep_lon)
    sep_v = np.asarray(sep_state.velocity_m_s, dtype=np.float64)
    horiz_speed = math.hypot(float(sep_v @ np.array(e_)), float(sep_v @ np.array(n_)))
    prop_floor = _propellant_floor_kg(sep_mass, horiz_speed, eng.isp_vacuum_s)

    x0 = np.asarray(initial_guess, dtype=np.float64) if initial_guess is not None \
        else np.array([8.0, 30.0, eng.throttle_max, 5.0, 0.0, 16_000.0, 22.0])
    best_x = clamp(x0)
    best = rollout(best_x, dt_opt)

    if sep_prop >= prop_floor:
        scale = np.array([10.0, 12.0, 0.1, 8.0, 18.0, 4000.0, 10.0])

        def obj(x: np.ndarray) -> float:
            r = rollout(x, dt_opt)
            return r.penalty if math.isfinite(r.penalty) else 1e6

        simplex = np.vstack([best_x] + [best_x + v for v in np.diag(scale)])
        res = minimize(obj, best_x, method="Nelder-Mead",
                       options=dict(maxfev=max_fevals, xatol=0.5, fatol=1e-4,
                                    adaptive=True, initial_simplex=simplex))
        cand_x = clamp(np.asarray(res.x))
        cand = rollout(cand_x, dt_opt)
        if cand.penalty < best.penalty:
            best_x, best = cand_x, cand
        if best.penalty > 0.0 and max_fevals >= 40:  # restart once, half budget
            res2 = minimize(obj, clamp(best_x + 0.7 * scale * np.sign(best_x - x0 + 1e-9)),
                            method="Nelder-Mead",
                            options=dict(maxfev=max_fevals // 2, xatol=0.5,
                                         fatol=1e-4, adaptive=True))
            cand_x = clamp(np.asarray(res2.x))
            cand = rollout(cand_x, dt_opt)
            if cand.penalty < best.penalty:
                best_x, best = cand_x, cand

    # ---- verdict: re-integrate the accepted script at the finer step
    if sep_prop < prop_floor:
        # best-effort ballistic track (no burns fly); the arithmetic below
        # never lies: report the floor shortfall without pretending.
        final = _Rollout()
        final.track = propagate(sep_state, booster, None, sep_state.t_s + horizon_s,
                                events=("ground",), dt_s=dt_final,
                                sample_s=max(2.0 * dt_final, 2.0))
        final.penalty = float("inf")
        return _assemble(final, best_x, sep_state, booster, recovery_target, corridor,
                         (INSUFFICIENT_PROPELLANT_BOOSTBACK,),
                         assumption=(f"separation propellant {sep_prop:.0f} kg is below the "
                                     f"{prop_floor:.0f} kg analytic floor to null the "
                                     f"{horiz_speed:.0f} m/s horizontal separation velocity"),
                         q_lim=entry_q_max_pa, g_lim=entry_g_max, reserve=reserve)

    final = rollout(best_x, dt_final)
    return _assemble(final, best_x, sep_state, booster, recovery_target, corridor,
                     None, q_lim=entry_q_max_pa, g_lim=entry_g_max, reserve=reserve)


def _assemble(final: _Rollout, x: np.ndarray, sep_state: BoosterState,
              booster: BoosterModel, target: RecoveryTarget, corridor: RecoveryCorridor,
              forced: tuple[str, ...] | None, *, q_lim: float, g_lim: float,
              reserve: float, assumption: str | None = None) -> ReturnPlan:
    """Turn the final re-integrated rollout into the ReturnPlan: feasibility
    checks in physical order (first violation governs), corridor report,
    insertion state + ENU mirror."""
    bb = BoostbackPlan(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
    eb = EntryBurnPlan(float(x[5]), float(x[6]))
    track = final.track
    assert track is not None  # every _assemble path carries a flown track

    reasons: list[str] = []
    if forced is not None:
        reasons.extend(forced)
    else:
        if final.exhausted == "boostback":
            reasons.append(INSUFFICIENT_PROPELLANT_BOOSTBACK)
        elif final.exhausted == "entry":
            reasons.append(INSUFFICIENT_PROPELLANT_ENTRY_BURN)
        if not final.crossed:
            reasons.append(CORRIDOR_NOT_REACHED)
        else:
            if (final.miss_m > corridor.radius_m or final.tilt_deg > corridor.max_tilt_deg
                    or final.speed_m_s > corridor.v_max_m_s):
                reasons.append(CORRIDOR_MISSED)
            if final.prop_kg < final.landing_need_kg + reserve:
                reasons.append(INSUFFICIENT_PROPELLANT_RESERVE)
        if final.entry_q_pa > q_lim:
            reasons.append(ENTRY_Q_LIMIT_EXCEEDED)
        if final.entry_g > g_lim:
            reasons.append(ENTRY_G_LIMIT_EXCEEDED)

    report = CorridorReport(
        reached=final.crossed,
        t_s=float(final.event.t_s) if final.crossed else None,
        miss_m=final.miss_m, tilt_deg=final.tilt_deg, speed_m_s=final.speed_m_s,
        propellant_available_kg=final.prop_kg, landing_need_kg=final.landing_need_kg,
        reserve_kg=reserve,
        entry_max_q_pa=final.entry_q_pa if final.entry_q_pa > 0.0 or final.crossed else None,
        entry_max_proper_g=final.entry_g if final.entry_q_pa > 0.0 or final.crossed else None,
    )

    state: BoosterState | None = None
    enu: LandingBurnEnu | None = None
    if final.crossed:
        ev = final.event
        r = np.asarray(ev.data["position_ecef_m"], dtype=np.float64)
        v = np.asarray(ev.data["velocity_ecef_m_s"], dtype=np.float64)
        state = BoosterState(t_s=float(ev.t_s), epoch_utc=track.epoch_utc, frame="ECEF",
                             position_m=tuple(float(c) for c in r),
                             velocity_m_s=tuple(float(c) for c in v),
                             mass_kg=float(ev.data["mass_kg"]))
        e, n, u = _enu_basis(target.latitude_deg, target.longitude_deg)
        d = r - np.asarray(target.position_ecef_m, dtype=np.float64)
        enu = LandingBurnEnu(
            t_s=float(ev.t_s),
            position_enu_m=(float(d @ np.array(e)), float(d @ np.array(n)), float(d @ np.array(u))),
            velocity_enu_m=(float(v @ np.array(e)), float(v @ np.array(n)), float(v @ np.array(u))),
            mass_kg=float(ev.data["mass_kg"]))

    assumptions: list[str] = []
    if assumption is not None:
        assumptions.append(assumption)
    assumptions.append(f"entry q limit {q_lim:.0f} Pa, entry g limit {g_lim:.1f} g over the "
                       "post-boostback descent window (DESIGN ASSUMPTIONs unless overridden)")

    return ReturnPlan(
        feasible=not reasons, reason_codes=tuple(reasons),
        target=target, corridor=corridor, boostback=bb, entry_burn=eb,
        landing_burn_initial_state=state, landing_burn_initial_state_enu=enu,
        track=track, corridor_report=report,
        ignitions_used=int(track.ignitions_used),
        assumptions=tuple(assumptions))


def mission_command(return_plan: ReturnPlan,
                    sep_state: BoosterState) -> Callable[[float, BoosterState], ThrustCmd | None]:
    """The scripted guidance behind ``return_plan`` as an FL-301
    ``thrust_cmd`` callback: re-fly it through ``propagate`` to reproduce the
    plan's track (the acceptance "no kinematic shortcuts" check). The
    bearing-guidance law is rebuilt from ``return_plan.target``, so the
    re-fly sees the identical script."""
    pad = np.asarray(return_plan.target.position_ecef_m, dtype=np.float64)
    return _Mission(float(sep_state.t_s), return_plan.boostback, return_plan.entry_burn,
                    _bearing_of_target(pad))
