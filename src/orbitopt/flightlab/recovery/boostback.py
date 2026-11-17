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
the entry-phase dynamic-pressure / proper-g limits as penalties.

R-01 (epic #217) makes that objective *landing-aware*, co-designing the
handoff with the landing burn instead of only with the corridor: the same
corridor miss is additionally soft-constrained against the landing-divert
budget :data:`DEFAULT_LANDING_DIVERT_BUDGET_M` -- the divert the
FL-303/FL-305 landing burn can close with authority in hand, far tighter than
the 8 km corridor radius a miss is graded against (ADR flightlab-0002: the
corridor term "barely sees" a 1.76 km miss the landing burn cannot) -- and
the corridor propellant is asked to cover the landing-burn need + vehicle
reserve + a minimum fuel margin :data:`DEFAULT_LANDING_RESERVE_MARGIN_KG`
(FL-305-baseline Monte Carlo: 13 of 18 failures were reserve shortfalls the
zero-margin objective was indifferent to). Both terms are objective terms,
not feasibility verdicts: the frozen reason-code set and check order are
unchanged, and the exact FL-305 objective is reproducible with
``landing_divert_budget_m=None`` / ``landing_reserve_margin_kg=0.0``. Every
trajectory the optimizer ever sees is one ``propagate()`` call; nothing is
estimated with a kinematic shortcut, and the accepted plan is re-integrated
at the final (finer) step before it is reported.

E3-02 (epic #248) offers a third objective mode beside FL-305 and R-01: the
same landing coupling expressed as **joint chance constraints**
(``joint_chance=``; :mod:`orbitopt.flightlab.recovery.chance`, ADR
flightlab-0005). Fuel margin, divert feasibility and landing probability
enter as constraints with deterministic equivalents (a quantile margin on
the dispersion loss, a divert budget *derived* from a touchdown-scatter
requirement, a joint landing-probability bound), aggregated as a
worst-normalized-violation exact penalty instead of R-01's additive soft
terms -- so the tail a weighted sum can trade away cannot be traded.
``joint_chance`` replaces the two R-01 soft terms, so the R-01 knobs must
be pinned off alongside it; everything else (physics, checks, reason codes)
is unchanged. E7-01 (task #277) closes the semantic gap this left: the
search stays what it is -- a **penalized optimization** (exact-penalty
method, no constrained solver) -- and the constraints are enforced as a
verdict instead of implied by the penalty: a joint-mode plan whose worst
normalized violation exceeds ``JointChanceSpec.violation_tolerance`` comes
back ``feasible=False`` with ``JOINT_CHANCE_VIOLATED``.

E7-04 (task #280) adds a fourth, orthogonal planning knob:
``corridor_aim_margin_m`` (default 0.0, bit-exact). It tightens only the
*planning-time* grading of the corridor miss (against
``corridor.radius_m - margin``), so the optimizer keeps a handoff inside a
ring inset from the acceptance corridor instead of strolling up to its rim;
the ``CORRIDOR_MISSED`` verdict is still graded at the true radius. It is
a response to the E3-03 failure-driven reading (``initial_state`` ranked
first in every arm, Spearman 0.38--0.47; ``CORRIDOR_MISSED`` /
``LANDING_LATERAL_SPEED_EXCEEDED`` the dominant regression modes), but the
E7-04 paired re-fly measured it as a *dial, not the fix*: on the E3-03
grid a quarter-radius inset trades ``INSUFFICIENT_PROPELLANT_RESERVE``
down (shorter divert burns less) for lateral-speed failures up (a deeper
handoff arrives faster), and in joint-chance mode its miss-term reweight
is dominated by the joint penalty. The E3-03 report's dominant guidance-
side lever turned out to be E3-01's command-side lag compensation instead.

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
    JOINT_CHANCE_VIOLATED                joint chance mode only (E7-01):
                                         crossed within the hard limits but
                                         the accepted handoff's worst
                                         normalized joint violation exceeds
                                         the spec's violation_tolerance --
                                         the plan does not satisfy the
                                         chance constraints it was planned
                                         under

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
from orbitopt.flightlab.recovery.chance import (
    JointChanceSpec,
    evaluate_joint_chance,
    joint_penalty,
)
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
    "DEFAULT_LANDING_DIVERT_BUDGET_M",
    "DEFAULT_LANDING_RESERVE_MARGIN_KG",
    "ENTRY_BURN_LEAD_RAD",
    "ENTRY_G_LIMIT_EXCEEDED",
    "ENTRY_Q_LIMIT_EXCEEDED",
    "INSUFFICIENT_PROPELLANT_BOOSTBACK",
    "INSUFFICIENT_PROPELLANT_ENTRY_BURN",
    "INSUFFICIENT_PROPELLANT_RESERVE",
    "JOINT_CHANCE_VIOLATED",
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
#: E7-01 (task #277): joint chance mode only -- the accepted handoff's worst
#: normalized violation is above the spec's tolerance, so the plan is never
#: reported as satisfying the chance constraints.
JOINT_CHANCE_VIOLATED = "JOINT_CHANCE_VIOLATED"

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

#: R-01 landing-divert budget [m]: the horizontal offset at the corridor the
#: FL-303/FL-305 landing burn can close with throttle authority in hand --
#: the corridor miss enters the objective against this budget, not only
#: against the (far larger) corridor radius. DESIGN ASSUMPTION anchored on
#: ADR flightlab-0002 measurements: the accepted FL-305 handoff is 0.35 km
#: out (lands inside every limit), FL-304's 1.76 km overshoot missed the pad.
#: ``float("inf")`` (``None`` at the planner API) reproduces the FL-305
#: objective, which graded the miss against the corridor radius only.
DEFAULT_LANDING_DIVERT_BUDGET_M = 1000.0

#: R-01 minimum fuel reserve margin [kg] the objective keeps at the corridor
#: on top of the landing-burn need + vehicle reserve: propellant to absorb
#: the plan/reality mismatch (dispersions, divert spending) the momentum-
#: balance landing need does not see. DESIGN ASSUMPTION sized to the FL-305
#: baseline failure population (13 of 18 non-landed samples lost the margin,
#: not the corridor) and to the dispersion bands (5 % thrust / 3 % Isp on a
#: ~15 t recovery burn). ``0.0`` reproduces the FL-305 objective, which
#: penalized only the violation of need + reserve.
DEFAULT_LANDING_RESERVE_MARGIN_KG = 1000.0


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
    ``trigger_altitude_m``, burn ``duration_s`` at ``throttle`` (full
    throttle unless the planner ran with ``entry_throttle=True``, E5-05c)."""

    trigger_altitude_m: float
    duration_s: float
    throttle: float = 1.0


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
    h0 after entry ignition, residual fields None where not measured). The
    R-01 ``landing_divert_budget_m`` / ``min_landing_reserve_margin_kg``
    fields carry what the *objective* asked for (None when the R-01 term is
    disabled), so a consumer can check the accepted plan against the same
    landing-reachability and fuel-margin bar the optimizer steered to."""

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
    landing_divert_budget_m: float | None = None
    min_landing_reserve_margin_kg: float | None = None
    #: E3-02 joint chance mode: the predicted landing probability at the
    #: accepted handoff and the worst normalized constraint violation
    #: (both None when the joint mode is off).
    landing_probability: float | None = None
    joint_violation: float | None = None


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
    #: E5-05c (task #287): the second (braking) entry burn when the script was
    #: planned with ``entry_burns=2``; ``None`` for the 7-parameter one-burn
    #: script (every pre-E5-05c plan and caller).
    entry_burn_2: "EntryBurnPlan | None" = None


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


def _retro_lead_direction(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Unit thrust direction of an entry/braking burn: retrograde tilted up
    by ``ENTRY_BURN_LEAD_RAD`` toward the vertical-up component of the
    velocity's anti-parallel plane (keeps the plume off the deck and trades
    forward speed for altitude hold)."""
    spd = float(np.linalg.norm(v))
    vv = v / spd
    dd = -up
    pd = dd - float(dd @ vv) * vv
    nrm = float(np.linalg.norm(pd))
    if nrm < 1e-9:  # velocity already vertical: burn pure retro
        pd = vv
    else:
        pd = pd / nrm
    return -math.cos(ENTRY_BURN_LEAD_RAD) * vv \
        + math.sin(ENTRY_BURN_LEAD_RAD) * pd


class _Mission:
    """Scripted return: coast -> boostback (delay/duration/throttle/
    pitch/yaw) -> coast (drag, gravity) -> entry burn (trigger altitude /
    duration, full throttle, retro + lead) -> [optional second, lower
    braking burn (E5-05c, ``eb2``: same trigger/duration law at a lower
    trigger altitude) ->] corridor. Sampled once per propagator step like a
    flight computer."""

    def __init__(self, t_sep: float, bb: BoostbackPlan, eb: EntryBurnPlan,
                 bearing_of: Callable[[np.ndarray], float],
                 eb2: EntryBurnPlan | None = None):
        self.t_sep = float(t_sep)
        self.bb = bb
        self.eb = eb
        self.eb2 = eb2
        self.bearing_of = bearing_of
        self.entry_t0: float | None = None
        self.entry2_t0: float | None = None

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
            if float(np.linalg.norm(v)) < 1e-9:
                return None  # nothing to fly against; keep the engines shut
            return ThrustCmd(self.eb.throttle, _retro_lead_direction(v, up))
        if self.eb2 is None:
            return None
        if self.entry2_t0 is None:
            _lat, _lon, h = _geodetic_fast(r)
            if h <= self.eb2.trigger_altitude_m and float(v @ up) < 0.0:
                self.entry2_t0 = float(t)
            return None
        if t < self.entry2_t0 + self.eb2.duration_s:
            if float(np.linalg.norm(v)) < 1e-9:
                return None  # nothing to fly against; keep the engines shut
            return ThrustCmd(self.eb2.throttle, _retro_lead_direction(v, up))
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
    p_land: float | None = None           # E3-02 joint landing probability
    joint_violation: float | None = None  # E3-02 worst normalized violation
    #: E5-05c two-burn mode: scheduled engine-on time [s] still owed after
    #: the corridor crossing (0 = an engine-off handoff, the only valid one)
    handoff_burn_overlap_s: float = 0.0


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
              dt: float, bb_window: tuple[float, float],
              divert_budget: float = float("inf"),
              reserve_margin: float = 0.0,
              chance: JointChanceSpec | None = None,
              sep_prop_kg: float | None = None,
              aim_radius: float | None = None) -> _Rollout:
    """One full-physics flight through the FL-301 propagator plus the
    corridor residual. Penalty is a sum of dimensionless squared violations:
    the corridor/entry/propellant violations in-flight, plus the R-01 soft
    constraints -- corridor miss against ``divert_budget`` and corridor
    propellant against landing need + reserve + ``reserve_margin``. The
    defaults (infinite budget, zero margin) reproduce the FL-305 objective
    bit-for-bit. With ``chance`` (E3-02) the R-01 soft terms are replaced
    (the caller pins them off) by the joint chance constraints of
    :mod:`orbitopt.flightlab.recovery.chance`, graded on the crossed
    corridor handoff and added as the worst-violation exact penalty;
    ``sep_prop_kg`` (separation propellant) gives the burn exposure the
    loss scale is computed from. ``aim_radius`` (E7-04) replaces
    ``corridor.radius_m`` in the planning-time miss term only: the penalty
    then pulls the plan toward a handoff inside that smaller ring while the
    acceptance verdict at ``corridor.radius_m`` stays the referee
    (``None`` = the full radius, every pre-E7-04 caller)."""
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
    gate_t = entry_t0
    if getattr(mission, "eb2", None) is not None:
        # E5-05c two-burn entry: the corridor is only a valid landing-handoff
        # gate once the second (braking) burn has flown and cut off -- an
        # engine-off insertion state, like the one-burn script's coast.
        gate_t = None
        if len(burn_starts) > 2:
            ends = [e.t_s for e in tr.events
                    if e.kind == "burn_end" and e.t_s >= burn_starts[2].t_s]
            gate_t = min(ends) if ends else None
    for e in tr.events:
        if e.kind == "burn_end" and e.data.get("reason") == "PROPELLANT_EXHAUSTED":
            t0, t1 = bb_window
            out.exhausted = "boostback" if t0 - 1e-6 <= e.t_s <= t1 + 1.5 * dt else "entry"

    pad = np.asarray(target.position_ecef_m, dtype=np.float64)
    crossings = [e for e in tr.events
                 if e.kind == h0_name and gate_t is not None and e.t_s >= gate_t]
    if (not crossings and getattr(mission, "eb2", None) is not None
            and entry_t0 is not None):
        # E5-05c: the braking burn was still scheduled when the booster fell
        # through h0. Grade that (engine-on) crossing anyway and charge the
        # engine-on time still owed after it, so the objective stays
        # continuous across the "burn 2 cut off above h0" boundary instead of
        # dropping to the impact penalty; _assemble never accepts it.
        crossings = [e for e in tr.events
                     if e.kind == h0_name and e.t_s >= entry_t0]
        if crossings:
            t_x = float(crossings[0].t_s)
            if mission.entry2_t0 is None:
                owed = max(0.0, mission.entry_t0 + mission.eb.duration_s - t_x) \
                    if mission.entry_t0 is not None else 0.0
                owed += float(mission.eb2.duration_s)
            else:
                owed = max(0.0, mission.entry2_t0 + mission.eb2.duration_s - t_x)
            out.handoff_burn_overlap_s = max(owed, 1e-3)
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
    two_burn = getattr(mission, "eb2", None) is not None
    if two_burn:
        if out.exhausted == "entry":
            pen += 20.0  # two-burn mode: depletion mid-entry strands burn 2
        # Pre-crossing propellant discipline (E5-05c): the commanded entry
        # phase (burn 1 + burn 2 at ambient mdot) may not spend the landing
        # reserve plus the derived landing allowance -- the one-burn objective
        # only checked this AT the crossing, which lets a blind search burn
        # to a crossing with nothing left to land with.
        eng = vehicle.engine
        thrust_sl = eng.count * eng.thrust_vacuum_n \
            - eng.nozzle_exit_area_m2 * 101_325.0
        mdot_est = thrust_sl / (eng.isp_sea_level_s * G0_M_S2)
        m_ign = float(tr.mass_kg[0])
        if mission.entry_t0 is not None:
            t_arr = np.asarray(tr.times_s, dtype=np.float64)
            m_ign = float(np.interp(mission.entry_t0, t_arr,
                                    np.asarray(tr.mass_kg, dtype=np.float64)))
        prop_at_ign = max(0.0, m_ign - float(vehicle.dry_mass_kg))
        entry_s = (float(mission.eb.duration_s) * float(mission.eb.throttle)
                   + float(mission.eb2.duration_s) * float(mission.eb2.throttle))
        # landing allowance derived, not fixed: the landing-burn need from a
        # corridor-ceiling handoff (v_max at h0) at the lightest mass the
        # handoff can have (dry + reserve + that need; one fixed-point pass)
        dry_kg = float(vehicle.dry_mass_kg)
        allowance = landing_burn_propellant_need_kg(
            dry_kg + reserve, corridor.v_max_m_s, corridor.h0_m, vehicle)
        allowance = landing_burn_propellant_need_kg(
            dry_kg + reserve + allowance, corridor.v_max_m_s, corridor.h0_m, vehicle)
        short = max(0.0, entry_s * mdot_est + reserve + allowance - prop_at_ign)
        pen += (short / 1000.0) ** 2
    if out.crossed:
        if out.handoff_burn_overlap_s > 0.0:
            pen += 4.0 + (out.handoff_burn_overlap_s / 5.0) ** 2
        pen += (out.miss_m / (corridor.radius_m if aim_radius is None
                              else float(aim_radius))) ** 2
        # R-01 landing feasibility: the same miss against the divert the
        # landing burn can actually close, not only against the corridor
        # radius (a 1.76 km miss costs ~0.05 against 8 km and ~3.1 here).
        pen += (out.miss_m / divert_budget) ** 2
        pen += (max(0.0, out.tilt_deg - corridor.max_tilt_deg) / 10.0) ** 2
        pen += (max(0.0, out.speed_m_s - corridor.v_max_m_s) / 50.0) ** 2
        # R-01 fuel reserve: landing need + vehicle reserve + margin, so the
        # plan keeps headroom for the dispersions the momentum-balance need
        # cannot see (margin 0 = the FL-305 violation-only term).
        short = out.landing_need_kg + reserve + reserve_margin - out.prop_kg
        pen += (max(0.0, short) / 1000.0) ** 2
        pen += (max(0.0, out.entry_q_pa - q_lim) / 50_000.0) ** 2
        pen += (max(0.0, out.entry_g - g_lim) / 2.0) ** 2
        if chance is not None:
            # E3-02: fuel margin, divert feasibility and landing probability
            # as joint chance constraints (worst-violation exact penalty --
            # no cross-constraint trade). Exposure: propellant burned to the
            # corridor plus the landing need still to spend.
            burned = max(0.0, sep_prop_kg - out.prop_kg) \
                if sep_prop_kg is not None else 0.0
            verdict = evaluate_joint_chance(
                chance, miss_m=out.miss_m,
                margin_kg=out.prop_kg - out.landing_need_kg - reserve,
                exposure_kg=burned + out.landing_need_kg)
            pen += joint_penalty(verdict)
            out.p_land = verdict.p_land
            out.joint_violation = verdict.worst_violation
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
                         initial_guess: Sequence[float] | None = None,
                         landing_divert_budget_m: float | None =
                             DEFAULT_LANDING_DIVERT_BUDGET_M,
                         landing_reserve_margin_kg: float =
                             DEFAULT_LANDING_RESERVE_MARGIN_KG,
                          joint_chance: JointChanceSpec | Mapping[str, Any] | None =
                              None,
                          corridor_aim_margin_m: float = 0.0,
                          entry_burns: int = 1,
                          entry_throttle: bool = False,
                          search: Callable[[Callable[[np.ndarray], float], np.ndarray],
                                           np.ndarray] | None = None) -> ReturnPlan:
    """Optimize the boostback/entry-burn script so the FL-301-propagated
    booster reaches ``recovery_target.corridor``; returns the corridor
    insertion state, its pad-ENU mirror, and the flown track.

    ``sep_state`` is the FL-301 separation state (any contract frame),
    ``booster`` the recovery vehicle model, ``recovery_target`` an rtls or
    downrange landing site. The entry-phase q/g limits default to the module
    DESIGN ASSUMPTIONs. R-01's landing-aware objective terms (module
    docstring) default to :data:`DEFAULT_LANDING_DIVERT_BUDGET_M` and
    :data:`DEFAULT_LANDING_RESERVE_MARGIN_KG`; ``landing_divert_budget_m=
     None`` and ``landing_reserve_margin_kg=0.0`` turn them off and reproduce
    the FL-305 objective exactly (the fair-baseline switch used by the
    before/after benchmarks). ``joint_chance`` (E3-02) selects the joint
    chance-constraint objective instead: a :class:`JointChanceSpec` (or a
    mapping of its fields) whose divert / fuel-margin / landing-probability
    constraints replace the R-01 soft terms -- which must then be pinned off
    alongside it. Every rollout additionally reports the handoff's predicted
    landing probability and worst constraint violation on the corridor
    report. E7-01 feasibility semantics: in joint mode the accepted plan is
    graded against the spec -- a handoff whose worst normalized violation
    exceeds ``JointChanceSpec.violation_tolerance`` comes back
    ``feasible=False`` with reason code ``JOINT_CHANCE_VIOLATED``; the
    search itself remains a penalized optimization (exact-penalty method on
    Nelder-Mead), not a constrained solver. All trajectories are flown by
    ``orbitopt.flightlab.recovery.propagator.propagate`` -- the optimizer
    only ever sees full-physics rollouts, and the accepted solution is
    re-integrated at ``dt_final`` before it is reported (no kinematic
     shortcuts). ``initial_guess`` (delay, duration, throttle, pitch, yaw
     trim, trigger altitude, entry duration -- plus braking-burn trigger
     altitude and duration when ``entry_burns=2``) overrides the default
     starting point; ``max_fevals`` budgets the search (a restart from a jittered
    point spends at most half of it, and budgets below 40 keep the search
    where it started -- pin a script by passing both).
    ``corridor_aim_margin_m`` (E7-04, task #280) is a *planning-time* aim
    margin [m]: the objective grades the corridor miss against
    ``corridor.radius_m - corridor_aim_margin_m`` instead of the full radius,
    so the accepted script aims *inside* a ring inset from the acceptance
    corridor. The acceptance corridor itself -- and the ``CORRIDOR_MISSED``
    verdict ``_assemble`` gives against the true radius -- are untouched (the
    referee never moves); only the plan the optimizer keeps aims deeper,
    buying robustness against the plan/reality mismatch the FL-302 rollout
    cannot see (the atmospheric dispersion draws, the E3-03 top-ranked
    ``initial_state`` sensitivity): an aim that already misses 0.5 km has
    7.5 km of radius to spare, an aim that aimed *at* the rim has none. The
    tighter radius also pulls the divert handed to FL-303 down, which is the
    E7-04 attack on the terminal ``LANDING_LATERAL_SPEED_EXCEEDED`` mode.
    0.0 (default) reproduces the E3-02-era objective bit-for-bit.
    ``entry_burns`` (E5-05c, task #287) selects the entry-script length:
    1 (default) is the FL-302 7-parameter one-burn script, reproduced
    bit-for-bit; 2 extends it to a 9-parameter multi-burn entry -- a first
    entry burn (as before) plus a second, lower FULL-throttle *braking*
    burn flown with the same retro+lead law, igniting on the descent
    crossing of its own trigger altitude (clamped strictly below the first
    trigger). The corridor crossing is only counted once the braking burn
    has cut off, so the landing handoff stays an engine-off state; the
    second burn gives the script authority to decelerate *through* the
    terminal-drag asymptote a single coasting burn cannot cross from a
    computed-MECO separation energy (ADR flightlab-0008 F10). Raises
    ``RecoveryError`` when the vehicle has fewer than ``1 + entry_burns``
    restarts available (boostback and each entry burn need one; the landing
    burn needs its own on top). ``entry_throttle`` (E5-05c) appends one
    throttle parameter per entry burn (after the trigger/duration
    parameters, bounded by the engine's throttle range): the default
    full-throttle burn of the 3-engine group exceeds the 8 g entry ceiling
    on its own below ~69 t, which an RTLS booster always is by entry; off
    (default) keeps every burn at full throttle, bit-for-bit.
    ``search`` (E5-05c) replaces the Nelder-Mead search and its restart:
    called as ``search(objective, x_start)`` with the planner's own
    penalized objective (clamps its input, full-physics rollout at
    ``dt_opt``) and the clamped start, it returns the parameter vector to
    accept if it beats the start. The verdict path (re-integration at
    ``dt_final``, ``_assemble``) is unchanged -- offline global searches
    (``scripts/e505c_recovery_search.py``) plug in here; ``None`` (default)
    is the stock search, bit-for-bit."""
    corridor = recovery_target.corridor
    eng = booster.engine
    if entry_burns not in (1, 2):
        raise ValueError(f"entry_burns must be 1 or 2, got {entry_burns}")
    ignitions_needed = 1 + int(entry_burns)
    if eng.restarts_max < ignitions_needed:
        raise RecoveryError(
            f"boostback + entry burns need {ignitions_needed} ignitions, vehicle "
            f"allows {eng.restarts_max}")
    if landing_divert_budget_m is not None and not landing_divert_budget_m > 0.0:
        raise ValueError(f"landing_divert_budget_m must be positive or None, "
                         f"got {landing_divert_budget_m}")
    if not landing_reserve_margin_kg >= 0.0:
        raise ValueError(f"landing_reserve_margin_kg must be >= 0, "
                         f"got {landing_reserve_margin_kg}")
    if not corridor_aim_margin_m >= 0.0:
        raise ValueError(f"corridor_aim_margin_m must be >= 0, "
                         f"got {corridor_aim_margin_m}")
    if not corridor_aim_margin_m < corridor.radius_m:
        raise ValueError(f"corridor_aim_margin_m must stay inside the corridor "
                         f"radius {corridor.radius_m}, got {corridor_aim_margin_m}")
    aim_radius = corridor.radius_m - float(corridor_aim_margin_m)
    divert_budget = float("inf") if landing_divert_budget_m is None \
        else float(landing_divert_budget_m)
    reserve_margin = float(landing_reserve_margin_kg)
    chance = JointChanceSpec.from_option(joint_chance)
    if chance is not None:
        if landing_divert_budget_m is not None or reserve_margin != 0.0:
            raise ValueError(
                "joint_chance replaces the R-01 soft objective terms: pin them "
                "off alongside it (landing_divert_budget_m=None, "
                "landing_reserve_margin_kg=0.0)")
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

    n_base = 7 + (2 if entry_burns == 2 else 0)
    n_x = n_base + (int(entry_burns) if entry_throttle else 0)

    def clamp(x: np.ndarray) -> np.ndarray:
        # x[4] is the boostback yaw trim [deg, +right] around the live
        # bearing to the target (a local-frame direction parameter bounded
        # like a real trim authority); the mission law adds the bearing.
        # x[7:9] (entry_burns=2, E5-05c) trigger/duration of the second
        # braking burn, clamped to ignite strictly below the first trigger.
        delay, dur, thr, pitch, yaw, h_trig, dur_e = x[:7]
        delay = min(max(float(delay), 0.0), min(horizon_s * 0.4, 150.0))
        dur = min(max(float(dur), 4.0), 60.0)
        thr = min(max(float(thr), eng.throttle_min), eng.throttle_max)
        pitch = min(max(float(pitch), 0.0), 30.0)
        yaw = min(max(float(yaw), -45.0), 45.0)
        h_trig = min(max(float(h_trig), corridor.h0_m + 2000.0), 35_000.0)
        dur_e = min(max(float(dur_e), 3.0), 45.0)
        out = [delay, dur, thr, pitch, yaw, h_trig, dur_e]
        if entry_burns == 2:
            h2 = min(max(float(x[7]), corridor.h0_m + 300.0),
                     min(float(h_trig) - 1000.0, 35_000.0))
            dur2 = min(max(float(x[8]), 3.0), 45.0)
            out += [h2, dur2]
        out += [min(max(float(t), eng.throttle_min), eng.throttle_max)
                for t in x[n_base:n_x]]
        return np.array(out)

    def to_mission(x: np.ndarray) -> _Mission:
        bb, eb, eb2 = _unpack_script(x, entry_burns, entry_throttle)
        return _Mission(sep_state.t_s, bb, eb, bearing_of, eb2)

    def rollout(x: np.ndarray, dt: float) -> _Rollout:
        x = clamp(x)
        window = (sep_state.t_s + float(x[0]), sep_state.t_s + float(x[0]) + float(x[1]))
        return _evaluate(sep_state, to_mission(x), booster, recovery_target, corridor,
                         entry_q_max_pa, entry_g_max, reserve, horizon_s, dt, window,
                         divert_budget=divert_budget, reserve_margin=reserve_margin,
                         chance=chance, sep_prop_kg=sep_prop, aim_radius=aim_radius)

    # --- analytic floor check (see _propellant_floor_kg)
    sep_prop = sep_mass - dry
    sep_lat, sep_lon, _sep_h = _geodetic_fast(sep_r)
    e_, n_, _u = _enu_basis(sep_lat, sep_lon)
    sep_v = np.asarray(sep_state.velocity_m_s, dtype=np.float64)
    horiz_speed = math.hypot(float(sep_v @ np.array(e_)), float(sep_v @ np.array(n_)))
    prop_floor = _propellant_floor_kg(sep_mass, horiz_speed, eng.isp_vacuum_s)

    x0 = np.asarray(initial_guess, dtype=np.float64) if initial_guess is not None \
        else np.array([8.0, 30.0, eng.throttle_max, 5.0, 0.0, 16_000.0, 22.0,
                       *([7_000.0, 25.0] if entry_burns == 2 else []),
                       *([eng.throttle_max] * (n_x - n_base))])
    if x0.size != n_x:
        raise ValueError(
            f"initial_guess has {x0.size} entries; the entry_burns={entry_burns}, "
            f"entry_throttle={entry_throttle} script has {n_x}")
    best_x = clamp(x0)
    best = rollout(best_x, dt_opt)

    if sep_prop >= prop_floor:
        scale = np.array([10.0, 12.0, 0.1, 8.0, 18.0, 4000.0, 10.0,
                          *([3000.0, 12.0] if entry_burns == 2 else []),
                          *([-0.25] * (n_x - n_base))])

        def obj(x: np.ndarray) -> float:
            r = rollout(x, dt_opt)
            return r.penalty if math.isfinite(r.penalty) else 1e6

        if search is not None:
            cand_x = clamp(np.asarray(search(obj, best_x.copy()), dtype=np.float64))
            cand = rollout(cand_x, dt_opt)
            if cand.penalty < best.penalty:
                best_x, best = cand_x, cand
        else:
            simplex = np.vstack([best_x] + [best_x + v for v in np.diag(scale)])
            res = minimize(obj, best_x, method="Nelder-Mead",
                           options=dict(maxfev=max_fevals, xatol=0.5, fatol=1e-4,
                                        adaptive=True, initial_simplex=simplex))
            cand_x = clamp(np.asarray(res.x))
            cand = rollout(cand_x, dt_opt)
            if cand.penalty < best.penalty:
                best_x, best = cand_x, cand
        if search is None and best.penalty > 0.0 and max_fevals >= 40:  # restart once, half budget
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
                         q_lim=entry_q_max_pa, g_lim=entry_g_max, reserve=reserve,
                         divert_budget=divert_budget, reserve_margin=reserve_margin,
                         chance=chance, layout=(entry_burns, entry_throttle))

    final = rollout(best_x, dt_final)
    return _assemble(final, best_x, sep_state, booster, recovery_target, corridor,
                     None, q_lim=entry_q_max_pa, g_lim=entry_g_max, reserve=reserve,
                     divert_budget=divert_budget, reserve_margin=reserve_margin,
                     chance=chance, layout=(entry_burns, entry_throttle))


def _unpack_script(x: np.ndarray, entry_burns: int = 1, entry_throttle: bool = False
                   ) -> tuple[BoostbackPlan, EntryBurnPlan, EntryBurnPlan | None]:
    """Parameter vector -> (boostback, entry burn, braking burn | None).
    Layout: the 7 FL-302 parameters, then [trigger_2, duration_2] when
    ``entry_burns=2``, then one throttle per entry burn when
    ``entry_throttle`` (E5-05c)."""
    n_base = 7 + (2 if entry_burns == 2 else 0)
    thr = [float(t) for t in x[n_base:]] if entry_throttle else []
    thr += [1.0] * (int(entry_burns) - len(thr))
    bb = BoostbackPlan(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
    eb = EntryBurnPlan(float(x[5]), float(x[6]), thr[0])
    eb2 = EntryBurnPlan(float(x[7]), float(x[8]), thr[1]) if entry_burns == 2 else None
    return bb, eb, eb2


def _assemble(final: _Rollout, x: np.ndarray, sep_state: BoosterState,
              booster: BoosterModel, target: RecoveryTarget, corridor: RecoveryCorridor,
               forced: tuple[str, ...] | None, *, q_lim: float, g_lim: float,
               reserve: float, assumption: str | None = None,
               divert_budget: float = float("inf"),
               reserve_margin: float = 0.0,
               chance: JointChanceSpec | None = None,
               layout: tuple[int, bool] = (1, False)) -> ReturnPlan:
    """Turn the final re-integrated rollout into the ReturnPlan: feasibility
    checks in physical order (first violation governs), corridor report,
    insertion state + ENU mirror."""
    bb, eb, eb2 = _unpack_script(x, *layout)
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
            elif final.handoff_burn_overlap_s > 0.0:
                # E5-05c: an engine-on handoff (braking burn still owed) is
                # not a landing-burn initial state
                reasons.append(CORRIDOR_MISSED)
            if final.prop_kg < final.landing_need_kg + reserve:
                reasons.append(INSUFFICIENT_PROPELLANT_RESERVE)
        if final.entry_q_pa > q_lim:
            reasons.append(ENTRY_Q_LIMIT_EXCEEDED)
        if final.entry_g > g_lim:
            reasons.append(ENTRY_G_LIMIT_EXCEEDED)
        # E7-01 (task #277): the joint chance verdict is a feasibility check,
        # not just a penalty term -- a crossed handoff whose worst normalized
        # violation exceeds the spec's tolerance does not satisfy the
        # constraints and is never reported feasible.
        if (chance is not None and final.crossed
                and final.joint_violation is not None
                and not chance.satisfies(final.joint_violation)):
            reasons.append(JOINT_CHANCE_VIOLATED)

    report = CorridorReport(
        reached=final.crossed,
        t_s=float(final.event.t_s) if final.crossed else None,
        miss_m=final.miss_m, tilt_deg=final.tilt_deg, speed_m_s=final.speed_m_s,
        propellant_available_kg=final.prop_kg, landing_need_kg=final.landing_need_kg,
        reserve_kg=reserve,
        entry_max_q_pa=final.entry_q_pa if final.entry_q_pa > 0.0 or final.crossed else None,
        entry_max_proper_g=final.entry_g if final.entry_q_pa > 0.0 or final.crossed else None,
        landing_divert_budget_m=None if math.isinf(divert_budget) else divert_budget,
        min_landing_reserve_margin_kg=reserve_margin if reserve_margin > 0.0 else None,
        landing_probability=final.p_land,
        joint_violation=final.joint_violation,
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
    if not math.isinf(divert_budget) or reserve_margin > 0.0:
        assumptions.append(
            f"R-01 landing-aware objective: corridor miss soft-constrained against a "
            f"{divert_budget:.0f} m landing-divert budget, corridor propellant against "
            f"landing need + reserve + {reserve_margin:.0f} kg fuel margin "
            "(DESIGN ASSUMPTIONs unless overridden)")
    if chance is not None:
        assumptions.append(
            f"E3-02 joint chance constraints: landing probability >= "
            f"{chance.min_landing_probability:.2f} (fuel margin covering the "
            f"{chance.p_fuel_min:.2f} quantile of the dispersion loss, touchdown "
            f"{chance.scatter_quantile:.2f}-quantile radius <= "
            f"{chance.scatter_budget_m:.0f} m -> divert bar "
            f"{chance.divert_budget_m():.0f} m), worst-violation exact "
            f"penalty (penalized optimization; feasibility tolerance "
            f"{chance.violation_tolerance:g}); "
            "replaces the R-01 soft terms (DESIGN ASSUMPTIONs unless overridden)")

    if eb2 is not None:
        assumptions.append(
            "E5-05c multi-burn entry script (entry_burns=2): boostback + entry "
            "burn + lower braking burn; the corridor handoff is graded engine-"
            "off after the braking burn cutoff (DESIGN ASSUMPTION unless "
            "overridden)")
    if layout[1]:
        assumptions.append(
            "E5-05c throttled entry burns (entry_throttle=True): each entry "
            "burn flies at its optimized throttle within the engine's "
            "throttle range instead of full throttle (DESIGN ASSUMPTION "
            "unless overridden)")

    return ReturnPlan(
        feasible=not reasons, reason_codes=tuple(reasons),
        target=target, corridor=corridor, boostback=bb, entry_burn=eb,
        landing_burn_initial_state=state, landing_burn_initial_state_enu=enu,
        track=track, corridor_report=report,
        ignitions_used=int(track.ignitions_used),
        assumptions=tuple(assumptions), entry_burn_2=eb2)


def mission_command(return_plan: ReturnPlan,
                    sep_state: BoosterState) -> Callable[[float, BoosterState], ThrustCmd | None]:
    """The scripted guidance behind ``return_plan`` as an FL-301
    ``thrust_cmd`` callback: re-fly it through ``propagate`` to reproduce the
    plan's track (the acceptance "no kinematic shortcuts" check). The
    bearing-guidance law is rebuilt from ``return_plan.target``, so the
    re-fly sees the identical script."""
    pad = np.asarray(return_plan.target.position_ecef_m, dtype=np.float64)
    return _Mission(float(sep_state.t_s), return_plan.boostback, return_plan.entry_burn,
                    _bearing_of_target(pad), return_plan.entry_burn_2)
