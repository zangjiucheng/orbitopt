"""Booster 3-DOF recovery propagator (FL-301): from the separation state of
E1's FlightRun to a booster track with events, in a rotating Earth frame.

The model
---------
A point mass ``(r, v, m)`` in the Earth-fixed rotating frame (ECEF, the
ITRS/WGS84 axes of ``orbitopt.flightlab.frames``), integrated with a fixed
RK4 step and per-step event bisection:

* gravity: point mass + J2 (the axisymmetric harmonics about the ECEF z
  axis -- an oblateness term is frame-fixed in the rotating frame, which is
  one reason the propagator lives in ECEF);
* rotation: the Coriolis (``-2 Ω×v``) and centrifugal (``-Ω×(Ω×r)``)
  accelerations appear explicitly, and the atmosphere co-rotates, so the
  air-relative velocity is the ECEF velocity itself (no wind yet);
* atmosphere: the density profile of :mod:`orbitopt.entry` (Vallado Table
  8-4, US Standard Atmosphere 1976 below ~1000 km), height above the WGS-84
  ellipsoid via ``entry.height_m``. Pressure and temperature -- which
  density alone does not give, but Isp interpolation and Mach-dependent
  drag do -- come from a US76 layer table consistent with that density;
* drag: ``-q Cd(M) A r̂_air`` with the vehicle's Cd(Mach) table
  (piecewise-linear in Mach, clamped at the ends -- the same table shape
  OpenTsiolkovsky's ascent aero uses, so the same vehicle means the same
  numbers on both sides of separation);
* thrust: relightable engine group, total vacuum thrust scaled by a
  throttle clamped into ``[throttle_min, throttle_max]``, ambient-corrected
  as ``F(p) = count * throttle * (F_vac - A_e p)`` with Isp linear in the
  pressure ratio between sea level and vacuum; mass flow ``F / (Isp g0)``
  closes the propellant bookkeeping through the integration itself
  (propellant is ``m - dry_mass``, never negative: a burn cuts at dry mass
  with a ``PROPELLANT_EXHAUSTED`` burn-end event, it does not run through
  it);
* restarts: every off-to-on transition consumes one of ``restarts_max``
  ignitions; a commanded burn beyond the count raises ``RecoveryError``
  instead of quietly flying an engine-poor booster.

Events: ground impact (terminal, at the WGS-84 ellipsoid -- the geoid is
within ~±100 m, which at impact speeds is tens of milliseconds), apogee,
commanded altitude thresholds, burn start/stop, propellant exhaustion. Each
event is bisected *inside* its integration step (ground impact to
sub-millimetre in the crossing function), never interpolated between
samples: the E3 exit gate says "never a green landing from interpolation",
and that rule starts here.

The first output of this task is the *ballistic* (unpowered,
``thrust_cmd=None``) return: where the booster comes down, when, how fast
-- plus the impact-ellipse hook (``BoosterTrack.impact_ellipse()``) the E3
Monte-Carlo dispersion will fill.

Frames: the propagator integrates in ECEF whatever it is handed. A GCRS (or
LAUNCH_ECI) ``BoosterState`` is converted in with the FL-002 frame bridge,
and ``BoosterTrack.to_body_track(frame=...)`` hands back any contract frame,
so a track drops straight into a FlightResult run directory.

Time follows the mission clock: ``state0.t_s`` seconds after ``epoch_utc``
(the FlightLab t=0, usually liftoff), so a recovery track continues its
ascent run's ``t_s`` column instead of starting a new clock.

Only the state (r, v, m) is modelled: no attitude, no heating, no winds.
This answers "where/when does it land, how hard does it decelerate, does the
propellant bookkeeping close" -- it is not a loads or TPS tool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from orbitopt.entry import (
    G0_M_S2,
    J2_EARTH,
    MU_EARTH_M3_S2,
    OMEGA_EARTH_RAD_S,
    R_EARTH_M,
    density_kg_m3,
    height_m,
)
from orbitopt.flightlab import frames
from orbitopt.flightlab.contract import FLIGHT_PROVENANCE_TAGS, BodyTrack

__all__ = [
    "AltitudeThreshold",
    "BoosterEngine",
    "BoosterModel",
    "BoosterPhysics",
    "BoosterState",
    "BoosterTrack",
    "ImpactEllipse",
    "ImpactSite",
    "RecoveryError",
    "RecoveryEvent",
    "ThrustCmd",
    "ballistic_return",
    "parse_event_specs",
    "pressure_pa",
    "propagate",
    "separation_state_from_run",
    "speed_of_sound_m_s",
    "temperature_k",
]

_OMEGA_ECEF = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
_P_SEA_LEVEL_PA = 101325.0
_R_SPEC_DRY_AIR = 287.05287
_GAMMA_AIR = 1.4
_G0_BY_R = G0_M_S2 / _R_SPEC_DRY_AIR

# US Standard Atmosphere 1976 layer bases (geometric height km, base pressure
# Pa, base temperature K, lapse rate K/km) -- consistent with the density
# bases of orbitopt.entry (both are the US76 table below 50 km). Above
# 84.56 km the standard holds the base temperature, where
# p/p_sea_level < 4e-7 and nothing but a 4th-decimal Isp correction sees it.
_US76_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 101325.0, 288.150, -6.5),
    (11.0, 22632.06, 216.650, 0.0),
    (20.0, 5474.889, 216.650, 1.0),
    (32.0, 868.0187, 228.650, 2.8),
    (47.0, 110.5234, 270.650, 0.0),
    (51.0, 66.93887, 270.650, -2.8),
    (71.0, 3.956420, 214.650, 0.0),
    (84.560, 0.373388, 186.870, 0.0),
)


def temperature_k(h_m: float) -> float:
    """US76 standard temperature at geometric height ``h_m`` [m] (negative
    heights clamp to sea level)."""
    h_km = max(float(h_m), 0.0) / 1000.0
    base = _US76_LAYERS[0]
    for layer in _US76_LAYERS:
        if h_km >= layer[0]:
            base = layer
    return base[2] + base[3] * (h_km - base[0])


def pressure_pa(h_m: float) -> float:
    """US76 standard pressure at geometric height ``h_m`` [m] (hydrostatic
    within each layer, isothermal above the top layer)."""
    h_km = max(float(h_m), 0.0) / 1000.0
    base = _US76_LAYERS[0]
    for layer in _US76_LAYERS:
        if h_km >= layer[0]:
            base = layer
    p_b, t_b, lapse = base[1], base[2], base[3]
    dh = (h_km - base[0]) * 1000.0
    if lapse == 0.0:
        return p_b * math.exp(-_G0_BY_R * dh / t_b)
    t = t_b + lapse * dh / 1000.0
    return p_b * (t / t_b) ** (-_G0_BY_R / (lapse / 1000.0))


def speed_of_sound_m_s(h_m: float) -> float:
    """Speed of sound of the US76 standard atmosphere at height ``h_m``."""
    return math.sqrt(_GAMMA_AIR * _R_SPEC_DRY_AIR * temperature_k(h_m))


def density_us76_kg_m3(h_m: float) -> float:
    """US76 density at height ``h_m`` from the layer pressure/temperature
    (ideal gas) -- the textbook table itself. ``orbitopt.entry``'s density is
    the same US76 base *fitted* with piecewise exponentials (Vallado Table
    8-4), which tracks it to a few percent but reads ~15 % light near sea
    level; a cross-check against a code that evaluates US76 directly (e.g.
    OpenTsiolkovsky) should pick this model."""
    return pressure_pa(h_m) / (_R_SPEC_DRY_AIR * temperature_k(h_m))


def _naive_utc(dt: datetime) -> datetime:
    """The FlightLab epoch convention: naive datetime = UTC
    (orbitopt.timescales); an aware one is reduced to UTC first."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _iso_z(dt: datetime) -> str:
    """Contract-style ISO-8601 UTC stamp (trailing Z) for a FlightLab epoch."""
    return _naive_utc(dt).isoformat() + "Z"


# --------------------------------------------------------------------------
# errors and commands
# --------------------------------------------------------------------------


class RecoveryError(ValueError):
    """A recovery run that cannot be flown as commanded (a burn beyond the
    restart count, a start mass below the dry mass). A ValueError subclass:
    a bad input, not a bug."""


@dataclass(frozen=True)
class ThrustCmd:
    """One booster engine-command sample. ``dir_ecef`` is the thrust vector
    direction in ECEF components (need not be unit: it is normalised);
    ``throttle`` is the commanded level, interpreted through the model's
    [min, max] window (0 or less = shutdown, otherwise clamped up to the
    minimum -- an engine below minimum throttle does not exist)."""

    throttle: float
    dir_ecef: Sequence[float]

    def normalized_dir(self) -> np.ndarray:
        d = np.asarray(self.dir_ecef, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            raise RecoveryError("thrust direction is zero-length -- a directionless burn")
        return d / n


@dataclass(frozen=True)
class BoosterPhysics:
    """Physics switches, so the analytic cases of the acceptance list fly
    against exactly the physics they assume. Defaults = full model.
    ``density_model`` selects the atmosphere density table: ``"entry"`` (the
    default) reuses :func:`orbitopt.entry.density_kg_m3` (US76 fitted with
    piecewise exponentials); ``"ussa1976"`` evaluates the US76
    pressure/temperature layers directly (see ``density_us76_kg_m3``)."""

    gravity: str = "j2"          # "j2" | "point_mass" | "off"
    atmosphere: bool = True
    rotation: bool = True        # Coriolis + centrifugal + co-rotating air
    density_model: str = "entry"  # "entry" | "ussa1976"
    #: Multiplicative bias on every density read (FL-304 Monte-Carlo
    #: atmospheric-dispersion knob: 1.0 = nominal atmosphere, 1.1 = 10 %
    #: thicker). Applies to both density tables; 0 would disable aero.
    density_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.density_scale <= 0.0:
            raise ValueError(
                f"density_scale must be positive, got {self.density_scale}")


@dataclass(frozen=True)
class AltitudeThreshold:
    """An altitude-threshold crossing to record as an event (e.g. the 80 km
    entry-interface cut the E3 gates use). Direction "down" counts only
    descending crossings, "up" only ascending, "both" either."""

    height_m: float
    name: str | None = None
    direction: str = "down"

    def __post_init__(self) -> None:
        if self.direction not in ("down", "up", "both"):
            raise ValueError(f"direction must be down/up/both, got {self.direction!r}")

    @property
    def event_kind(self) -> str:
        return self.name or f"altitude_{self.height_m:g}"


# --------------------------------------------------------------------------
# state, vehicle, track
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoosterState:
    """The booster's state on the mission clock: ``t_s`` seconds after
    ``epoch_utc`` (naive datetime = UTC, the FlightLab convention), position
    and velocity in ``frame`` (any contract frame), mass in kg."""

    t_s: float
    epoch_utc: datetime
    frame: str
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    mass_kg: float

    def __post_init__(self) -> None:
        if self.frame not in frames.KNOWN_FRAMES:
            raise RecoveryError(
                f"BoosterState frame {self.frame!r} is not one of {frames.KNOWN_FRAMES}"
            )
        if self.mass_kg <= 0.0:
            raise RecoveryError(f"BoosterState mass must be positive, got {self.mass_kg}")


@dataclass(frozen=True)
class BoosterEngine:
    """The relightable engine group a booster uses for
    boostback/entry/landing (boostback and landing share the group)."""

    count: int
    usable_for: tuple[str, ...]
    thrust_vacuum_n: float          # per engine, full throttle
    isp_sea_level_s: float
    isp_vacuum_s: float
    nozzle_exit_area_m2: float
    throttle_min: float
    throttle_max: float = 1.0
    restarts_max: int = 2           # ignitions available after separation

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("an engine group needs at least one engine")
        if not (0.0 < self.throttle_min <= self.throttle_max <= 1.0):
            raise ValueError(
                "throttle window must be 0 < min <= max <= 1, got "
                f"[{self.throttle_min}, {self.throttle_max}]"
            )
        if self.restarts_max < 0:
            raise ValueError("restarts_max cannot be negative")
        if self.isp_vacuum_s < self.isp_sea_level_s:
            raise ValueError("sea-level Isp cannot exceed vacuum Isp")

    def effective_throttle(self, commanded: float) -> float:
        """0 = shutdown; anything else clamped into [min, max]."""
        if commanded <= 0.0:
            return 0.0
        return min(max(commanded, self.throttle_min), self.throttle_max)


@dataclass(frozen=True)
class BoosterModel:
    """The ``booster:`` section of a FlightLab vehicle YAML, validated and
    tagged. ``provenance_by_field`` carries the provenance tag of each
    authored number (``illustrative`` / ``public_estimate`` / ``official``),
    ``sources_by_field`` the one-line source. See ``load``."""

    name: str
    provenance: str
    dry_mass_kg: float
    landing_propellant_reserve_kg: float
    reference_area_m2: float
    cd_mach: tuple[tuple[float, float], ...]
    engine: BoosterEngine
    assumptions: tuple[str, ...] = ()
    provenance_by_field: Mapping[str, str] = field(default_factory=dict)
    sources_by_field: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dry_mass_kg <= 0.0:
            raise ValueError("dry_mass_kg must be positive")
        if self.landing_propellant_reserve_kg < 0.0:
            raise ValueError("landing_propellant_reserve_kg cannot be negative")
        if self.reference_area_m2 <= 0.0:
            raise ValueError("reference_area_m2 must be positive")
        machs = [m for m, _ in self.cd_mach]
        if len(machs) < 2 or any(b <= a for a, b in zip(machs, machs[1:])):
            raise ValueError("cd_mach must be >= 2 (mach, cd) pairs with strictly increasing Mach")
        if any(c <= 0.0 for _, c in self.cd_mach):
            raise ValueError("cd values must be positive")

    def cd(self, mach: float) -> float:
        """Cd from the vehicle's Cd(Mach) table, linear in Mach, clamped at
        the table ends (same policy as the OT aero ``ca`` curves)."""
        pts = np.asarray(self.cd_mach, dtype=np.float64)
        return float(np.interp(mach, pts[:, 0], pts[:, 1]))

    @staticmethod
    def load(path) -> "BoosterModel":
        """Read the ``booster:`` section of a FlightLab vehicle YAML. Every
        authored number may be tagged ``{value: X, provenance: tag,
        source: "..."}``; a bare number inherits the section-level
        ``provenance``. Both forms validate; the tags are kept on the model
        so a downstream gate can tell official numbers from placeholders."""
        text = Path(path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        if not isinstance(doc, dict) or not isinstance(doc.get("booster"), dict):
            raise ValueError(f"{path}: vehicle file needs a 'booster:' section mapping")
        sec = doc["booster"]
        section_prov = str(sec.get("provenance", doc.get("provenance", "illustrative")))
        if section_prov not in FLIGHT_PROVENANCE_TAGS:
            raise ValueError(f"{path}: booster provenance must be one of {FLIGHT_PROVENANCE_TAGS}")

        prov_by_field: dict[str, str] = {}
        sources: dict[str, str] = {}

        def tagged(key: str, node) -> float:
            tag = section_prov
            value = node
            if isinstance(node, dict):
                if "value" not in node:
                    raise ValueError(f"{path}: {key} mapping needs a 'value' key")
                tag = str(node.get("provenance", section_prov))
                if tag not in FLIGHT_PROVENANCE_TAGS:
                    raise ValueError(f"{path}: {key} provenance must be one of {FLIGHT_PROVENANCE_TAGS}")
                if "source" in node:
                    sources[key] = str(node["source"])
                value = node["value"]
            prov_by_field[key] = tag
            if value is None:
                raise ValueError(f"{path}: {key} is required")
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: {key} is not a number: {value!r}") from exc

        cd_node = sec.get("cd_mach")
        if isinstance(cd_node, dict):
            if "source" in cd_node:
                sources["cd_mach"] = str(cd_node["source"])
            prov_by_field["cd_mach"] = str(cd_node.get("provenance", section_prov))
            cd_node = cd_node.get("value")
        else:
            prov_by_field["cd_mach"] = section_prov
        if not isinstance(cd_node, (list, tuple)):
            raise ValueError(f"{path}: booster.cd_mach must be a list of [mach, cd] pairs")
        cd_mach = tuple((float(pair[0]), float(pair[1])) for pair in cd_node)

        eng = sec.get("engines") or {}
        if not isinstance(eng, dict):
            raise ValueError(f"{path}: booster.engines must be a mapping")
        thrust_n = tagged("engines.thrust_vacuum_n", eng.get("thrust_vacuum_n"))
        isp_sl = tagged("engines.isp_sea_level_s", eng.get("isp_sea_level_s"))
        isp_vac = tagged("engines.isp_vacuum_s", eng.get("isp_vacuum_s"))
        exit_a = tagged("engines.nozzle_exit_area_m2", eng.get("nozzle_exit_area_m2"))
        thr_min = tagged("engines.throttle_min", eng.get("throttle_min"))
        usable = eng.get("usable_for", ("boostback", "landing"))
        if isinstance(usable, str):
            usable = (usable,)
        engine = BoosterEngine(
            count=int(eng.get("count", 1)),
            usable_for=tuple(str(u) for u in usable),
            thrust_vacuum_n=thrust_n,
            isp_sea_level_s=isp_sl,
            isp_vacuum_s=isp_vac,
            nozzle_exit_area_m2=exit_a,
            throttle_min=thr_min,
            throttle_max=float(eng.get("throttle_max", 1.0)),
            restarts_max=int(eng.get("restarts_max", 2)),
        )
        return BoosterModel(
            name=str(doc.get("name", "booster")),
            provenance=section_prov,
            dry_mass_kg=tagged("dry_mass_kg", sec.get("dry_mass_kg")),
            landing_propellant_reserve_kg=tagged(
                "landing_propellant_reserve_kg", sec.get("landing_propellant_reserve_kg")),
            reference_area_m2=tagged("reference_area_m2", sec.get("reference_area_m2")),
            cd_mach=cd_mach,
            engine=engine,
            assumptions=tuple(str(a) for a in (sec.get("assumptions") or ())),
            provenance_by_field=prov_by_field,
            sources_by_field=sources,
        )


@dataclass(frozen=True)
class RecoveryEvent:
    """One recorded event on a BoosterTrack (its own vocabulary, mapped onto
    the contract event kinds by ``BoosterTrack.to_contract_events``)."""

    t_s: float
    kind: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class ImpactSite:
    """Where a track meets the ground: geodetic lat/lon on WGS84, mission
    clock, Earth-relative speed and vertical speed at the crossing."""

    t_s: float
    latitude_deg: float
    longitude_deg: float
    speed_m_s: float
    vertical_speed_m_s: float
    position_ecef_m: tuple[float, float, float]
    velocity_ecef_m_s: tuple[float, float, float]


@dataclass(frozen=True)
class ImpactEllipse:
    """The 2-sigma impact dispersion ellipse. FL-301 ships the *hook*: the
    E3 Monte-Carlo (thrust/density/separation-state dispersion) fills
    sigma/bearing; until then the state says "not assessed", never "passed"
    -- the same rule the FlightResult verification flags follow."""

    state: str = "not_assessed"
    reason_code: str = "IMPACT_ELLIPSE_NOT_ASSESSED"
    sigma_along_m: float | None = None
    sigma_cross_m: float | None = None
    bearing_deg: float | None = None


@dataclass(frozen=True)
class BoosterTrack:
    """The output of ``propagate``: the sampled ECEF track (mission clock),
    the recorded events, the impact site when there is one, and the summary
    maxima the E3 gates ask about."""

    epoch_utc: datetime
    times_s: np.ndarray
    mass_kg: np.ndarray
    position_ecef_m: np.ndarray      # (N, 3)
    velocity_ecef_m_s: np.ndarray    # (N, 3)
    height_m_arr: np.ndarray         # geodetic height above the ellipsoid
    throttle: np.ndarray             # effective throttle actually flown
    mach: np.ndarray
    dynamic_pressure_pa: np.ndarray  # on air-relative flow
    events: tuple[RecoveryEvent, ...]
    impact: ImpactSite | None
    termination: str                 # "ground_impact" | "time_limit"
    max_proper_g: float = 0.0
    max_dynamic_pressure_pa: float = 0.0
    ignitions_used: int = 0
    dry_mass_kg: float = 0.0
    landing_reserve: float = 0.0

    @property
    def propellant_remaining_kg(self) -> float:
        return float(self.mass_kg[-1]) - self.dry_mass_kg

    def reserve_margin_kg(self, reserve_kg: float | None = None) -> float:
        """Propellant left at track end above the landing reserve (negative
        = reserve spent)."""
        return self.propellant_remaining_kg - float(
            self.landing_reserve if reserve_kg is None else reserve_kg
        )

    def impact_ellipse(self) -> ImpactEllipse:
        """Impact-ellipse hook (see ImpactEllipse): a deterministic run does
        not know a dispersion ellipse; the Monte-Carlo task writes one."""
        return ImpactEllipse()

    def to_body_track(self, frame: str = "GCRS", *, body_id: str = "booster",
                      parent_body_id: str | None = None,
                      provenance: str = "illustrative") -> BodyTrack:
        """This track as a contract BodyTrack in ``frame`` ("ECEF" is a
        pass-through copy; "GCRS"/"LAUNCH_ECI" go through the FL-002 frame
        bridge at this track's epoch). The flown throttle, Mach and dynamic
        pressure ride along as extra CSV columns."""
        if frame not in frames.KNOWN_FRAMES:
            raise frames.UnknownFrameError(f"frame {frame!r} is not one of {frames.KNOWN_FRAMES}")
        r, v = self.position_ecef_m, self.velocity_ecef_m_s
        t = self.times_s
        if frame != "ECEF":
            r, v = frames.to_gcrs(r, v, "ECEF", _naive_utc(self.epoch_utc), t)
            if frame == "LAUNCH_ECI":
                r, v = frames.from_gcrs(r, v, "LAUNCH_ECI", _naive_utc(self.epoch_utc), t)
        return BodyTrack(
            body_id=body_id,
            parent_body_id=parent_body_id,
            frame=frame,
            epoch_utc=_iso_z(self.epoch_utc),
            provenance=provenance,
            t_s=tuple(float(x) for x in t),
            mass_kg=tuple(float(x) for x in self.mass_kg),
            position_m=tuple(tuple(float(c) for c in p) for p in r),
            velocity_m_s=tuple(tuple(float(c) for c in p) for p in v),
            extra_columns={
                "throttle": tuple(float(x) for x in self.throttle),
                "mach": tuple(float(x) for x in self.mach),
                "dynamic_pressure_pa": tuple(float(x) for x in self.dynamic_pressure_pa),
            },
        )

    def to_contract_events(self, body_id: str = "booster") -> tuple:
        """The events as contract ``Event``s where the recovery vocabulary
        has a FlightLab counterpart: burn_start->ignition, burn_end->meco,
        ground->impact. Recovery-only kinds (apogee, altitude thresholds)
        are left out -- events.json's kind enum would reject them -- and
        stay on ``events`` with their full data."""
        from orbitopt.flightlab.contract import Event

        mapping = {"burn_start": "ignition", "burn_end": "meco", "ground": "impact"}
        return tuple(
            Event(t_s=float(ev.t_s), kind=mapping[ev.kind], body_id=body_id, data=dict(ev.data))
            for ev in self.events if ev.kind in mapping
        )


# --------------------------------------------------------------------------
# environment / force model
# --------------------------------------------------------------------------


def _gravity_ecef(r: np.ndarray, gravity: str) -> np.ndarray:
    """Gravity (point mass, optionally + J2) at an ECEF position -- the
    Earth-fixed components of the same physical field the inertial equations
    use; J2's axisymmetry makes it frame-fixed in ECEF."""
    if gravity == "off":
        return np.zeros(3)
    x, y, z = (float(c) for c in r)
    rn2 = x * x + y * y + z * z
    rn = math.sqrt(rn2)
    f = -MU_EARTH_M3_S2 / (rn2 * rn)
    if gravity == "point_mass":
        return np.array([f * x, f * y, f * z])
    k = 1.5 * J2_EARTH * (R_EARTH_M * R_EARTH_M) / rn2
    zz = 5.0 * z * z / rn2
    return np.array([
        f * x * (1.0 + k * (1.0 - zz)),
        f * y * (1.0 + k * (1.0 - zz)),
        f * z * (1.0 + k * (3.0 - zz)),
    ])


def _aero_ecef(r: np.ndarray, v: np.ndarray, vehicle: BoosterModel,
               physics: BoosterPhysics) -> tuple[np.ndarray, float, float, float]:
    """(drag force vector [N], air-relative speed, Mach, dynamic pressure).
    The force is mass-independent; the equation of motion divides by the
    flown mass at the use site."""
    if not physics.atmosphere:
        return np.zeros(3), 0.0, 0.0, 0.0
    h = height_m(r)
    rho = (density_us76_kg_m3(h) if physics.density_model == "ussa1976"
           else density_kg_m3(h)) * float(physics.density_scale)
    if rho < 1e-14:
        return np.zeros(3), 0.0, 0.0, 0.0
    v_air = v if physics.rotation else v - np.cross(_OMEGA_ECEF, r)
    speed = float(np.linalg.norm(v_air))
    if speed < 1e-6:
        return np.zeros(3), 0.0, 0.0, 0.0
    mach = speed / speed_of_sound_m_s(h)
    q = 0.5 * rho * speed * speed
    force_mag = q * vehicle.cd(mach) * vehicle.reference_area_m2
    return -(force_mag / speed) * v_air, speed, mach, q


def _thrust_ecef(throttle: float, dir_ecef: np.ndarray | None, p_pa: float,
                 vehicle: BoosterModel) -> tuple[np.ndarray, float, float]:
    """(thrust force vector, mass flow [kg/s], effective Isp [s]) at an
    effective throttle level in ambient pressure ``p_pa``. Shutdown gives
    zeros; overexpanded to the point of ambient reversal gives zeros force
    too (no negative thrust, no suction)."""
    eng = vehicle.engine
    if throttle <= 0.0 or dir_ecef is None:
        return np.zeros(3), 0.0, eng.isp_sea_level_s
    force = eng.count * throttle * (eng.thrust_vacuum_n - eng.nozzle_exit_area_m2 * p_pa)
    if force <= 0.0:
        return np.zeros(3), 0.0, eng.isp_sea_level_s
    isp = eng.isp_sea_level_s + (eng.isp_vacuum_s - eng.isp_sea_level_s) * (
        1.0 - min(p_pa, _P_SEA_LEVEL_PA) / _P_SEA_LEVEL_PA)
    return force * dir_ecef, force / (isp * G0_M_S2), isp


# --------------------------------------------------------------------------
# event specs
# --------------------------------------------------------------------------

_BUILTIN_EVENTS = ("burn", "apogee", "ground")


def parse_event_specs(events: Iterable[str | AltitudeThreshold] | None
                      ) -> tuple[bool, bool, bool, tuple[AltitudeThreshold, ...]]:
    """(burn, apogee, ground, thresholds) from the ``events`` keyword.
    ``"burn"``/``"apogee"``/``"ground"`` opt into those events and
    ``AltitudeThreshold`` instances add threshold crossings; ground impact
    is *always* terminal either way -- a point mass flying through the
    planet is a bug, not a missing event spec. Anything else is an error:
    silently-ignored event names are how a gate stops being checked."""
    burn = apogee = ground = False
    thresholds: list[AltitudeThreshold] = []
    for spec in events or ():
        if isinstance(spec, AltitudeThreshold):
            thresholds.append(spec)
        elif spec == "burn":
            burn = True
        elif spec == "apogee":
            apogee = True
        elif spec == "ground":
            ground = True
        else:
            raise ValueError(
                f"unknown event spec {spec!r}: expected one of {_BUILTIN_EVENTS} "
                "or an AltitudeThreshold"
            )
    return burn, apogee, ground, tuple(thresholds)


# --------------------------------------------------------------------------
# integration core
# --------------------------------------------------------------------------


def _to_ecef_state(state: BoosterState) -> tuple[np.ndarray, np.ndarray]:
    """The (r, v) of ``state`` in ECEF, via the FL-002 frame bridge (GCRS /
    LAUNCH_ECI inputs pay the astropy conversion; ECEF inputs do not)."""
    r = np.asarray(state.position_m, dtype=np.float64).reshape(1, 3)
    v = np.asarray(state.velocity_m_s, dtype=np.float64).reshape(1, 3)
    t = np.array([float(state.t_s)])
    if state.frame == "ECEF":
        return r[0].copy(), v[0].copy()
    epoch = _naive_utc(state.epoch_utc)
    r_g, v_g = frames.to_gcrs(r, v, state.frame, epoch, t)
    r_e, v_e = frames.from_gcrs(r_g, v_g, "ECEF", epoch, t)
    return r_e[0], v_e[0]


def _radial_rate(s: np.ndarray) -> float:
    u = s[:3] / np.linalg.norm(s[:3])
    return float(u @ s[3:6])


def propagate(state0: BoosterState, vehicle: BoosterModel,
              thrust_cmd: Callable[[float, BoosterState], ThrustCmd | None] | None,
              t_end_s: float, *,
              events: Iterable[str | AltitudeThreshold] = (),
              dt_s: float = 0.5, sample_s: float = 2.0, ground_height_m: float = 0.0,
              physics: BoosterPhysics | None = None) -> BoosterTrack:
    """Fly the booster from ``state0`` to ``t_end_s`` (mission clock) -- or
    to the ground, whichever comes first -- with the ECEF 3-DOF model of
    this module. ``thrust_cmd(t, state)`` is sampled once per integration
    step (piecewise-constant command, like a flight computer's loop);
    ``None`` (or a callback returning ``None``) keeps the engines shut.

    ``events`` selects recorded events (``"burn"``, ``"apogee"``,
    ``"ground"``, ``AltitudeThreshold`` instances); ``dt_s`` is the fixed
    RK4 step, ``sample_s`` the output grid (event *data* is at the bisected
    event time, never interpolated between samples). Raises
    ``RecoveryError`` when a burn is commanded beyond the vehicle's restart
    count or the start mass is below the dry mass."""
    physics = physics or BoosterPhysics()
    if physics.gravity not in ("j2", "point_mass", "off"):
        raise ValueError(f"physics.gravity must be j2/point_mass/off, got {physics.gravity!r}")
    if physics.density_model not in ("entry", "ussa1976"):
        raise ValueError(
            f"physics.density_model must be entry/ussa1976, got {physics.density_model!r}"
        )
    if dt_s <= 0.0 or sample_s <= 0.0:
        raise ValueError("dt_s and sample_s must be positive")
    dry = float(vehicle.dry_mass_kg)
    if float(state0.mass_kg) < dry:
        raise RecoveryError(
            f"start mass {state0.mass_kg} kg is below the booster dry mass {dry} kg"
        )
    want_burn, want_apogee, want_ground, thresholds = parse_event_specs(events)
    t_end = float(t_end_s)
    if t_end <= state0.t_s:
        raise ValueError(f"t_end_s {t_end} must be after the start time {state0.t_s}")

    r0, v0 = _to_ecef_state(state0)
    y = np.concatenate([r0, v0, [float(state0.mass_kg)]])
    t = float(state0.t_s)
    epoch = _naive_utc(state0.epoch_utc)

    recorded: list[RecoveryEvent] = []
    times: list[float] = [t]
    pos: list[np.ndarray] = [y[:3].copy()]
    vel: list[np.ndarray] = [y[3:6].copy()]
    mass_s: list[float] = [float(y[6])]
    thr_s: list[float] = [0.0]
    mach_s: list[float] = [0.0]
    q_s: list[float] = [0.0]
    hgt_s: list[float] = [height_m(y[:3])]

    def cmd_at(tt: float, yy: np.ndarray) -> ThrustCmd | None:
        if thrust_cmd is None:
            return None
        st = BoosterState(t_s=tt, epoch_utc=epoch, frame="ECEF",
                         position_m=tuple(yy[:3]), velocity_m_s=tuple(yy[3:6]),
                         mass_kg=float(yy[6]))
        return thrust_cmd(tt, st)

    def sample(tt: float, yy: np.ndarray, throttle_eff: float) -> None:
        h = height_m(yy[:3])
        _f, speed, mach, q = _aero_ecef(yy[:3], yy[3:6], vehicle, physics)
        if tt <= times[-1] + 1e-9:  # strictly increasing (the contract says so)
            return
        times.append(tt)
        pos.append(yy[:3].copy())
        vel.append(yy[3:6].copy())
        mass_s.append(float(yy[6]))
        thr_s.append(throttle_eff)
        mach_s.append(mach)
        q_s.append(q)
        hgt_s.append(h)

    def event(tt: float, kind: str, yy: np.ndarray, extra: Mapping[str, Any] | None = None) -> None:
        h = height_m(yy[:3])
        _f, speed, mach, q = _aero_ecef(yy[:3], yy[3:6], vehicle, physics)
        payload: dict[str, Any] = {
            "height_m": h, "mach": mach, "dynamic_pressure_pa": q,
            "speed_earth_m_s": float(np.linalg.norm(yy[3:6])),
            "position_ecef_m": [float(x) for x in yy[:3]],
            "velocity_ecef_m_s": [float(x) for x in yy[3:6]],
            "mass_kg": float(yy[6]),
        }
        if extra:
            payload.update(extra)
        recorded.append(RecoveryEvent(t_s=float(tt), kind=kind, data=payload))

    def deriv(tt: float, yy: np.ndarray, on: bool, throttle: float,
              dirv: np.ndarray | None) -> np.ndarray:
        m = float(yy[6])
        a = _gravity_ecef(yy[:3], physics.gravity)
        if physics.rotation:
            a = (a - 2.0 * np.cross(_OMEGA_ECEF, yy[3:6])
                 - np.cross(_OMEGA_ECEF, np.cross(_OMEGA_ECEF, yy[:3])))
        f_aero, _s, _m, _q = _aero_ecef(yy[:3], yy[3:6], vehicle, physics)
        a = a + f_aero / m
        prop = 0.0
        if on and m > dry:
            p = pressure_pa(height_m(yy[:3])) if physics.atmosphere else 0.0
            f_thrust, mdot, _isp = _thrust_ecef(throttle, dirv, p, vehicle)
            a = a + f_thrust / m
            prop = mdot
        return np.concatenate([yy[3:6], a, [-prop]])

    def rk4(tt: float, yy: np.ndarray, h: float, on: bool, throttle: float,
            dirv: np.ndarray | None) -> np.ndarray:
        k1 = deriv(tt, yy, on, throttle, dirv)
        k2 = deriv(tt + 0.5 * h, yy + 0.5 * h * k1, on, throttle, dirv)
        k3 = deriv(tt + 0.5 * h, yy + 0.5 * h * k2, on, throttle, dirv)
        k4 = deriv(tt + h, yy + h * k3, on, throttle, dirv)
        return yy + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def bisect_fraction(tt0: float, y0: np.ndarray, span: float, on: bool,
                        throttle: float, dirv: np.ndarray | None,
                        g: Callable[[np.ndarray], float], target: float,
                        lo: float, hi: float) -> float:
        """Fraction alpha in [lo, hi] where g(rk4 partial step) crosses
        ``target``, bisection on the RK4 step itself (60 halvings -- far
        below any physical tolerance; the crossing is monotone by
        construction of the detection)."""
        g0 = g(y0) - target
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (g(rk4(tt0, y0, mid * span, on, throttle, dirv)) - target) * g0 <= 0.0:
                hi = mid
            else:
                lo = mid
        return hi

    def proper_g(throttle: float, dirv: np.ndarray | None, yy: np.ndarray) -> float:
        """Non-gravitational load [g]: aerodynamic + thrust force over g0."""
        f_aero, _s, _m, _q = _aero_ecef(yy[:3], yy[3:6], vehicle, physics)
        f = f_aero
        if throttle > 0.0:
            p = pressure_pa(height_m(yy[:3])) if physics.atmosphere else 0.0
            f_thrust, _mdot, _isp = _thrust_ecef(throttle, dirv, p, vehicle)
            f = f + f_thrust
        return float(np.linalg.norm(f)) / (G0_M_S2 * float(yy[6]))

    ignitions = 0
    burning = False
    dry_logged = False
    dir_ecef: np.ndarray | None = None
    termination = "time_limit"
    impact: ImpactSite | None = None
    max_g = 0.0
    max_q = 0.0
    grid_next = t + sample_s
    g_ground = lambda s: height_m(s[:3]) - ground_height_m

    while t < t_end - 1e-12:
        span = min(dt_s, t_end - t)

        # ---- command + mode transitions (piecewise-constant over the step)
        cmd = cmd_at(t, y)
        throttle_eff = vehicle.engine.effective_throttle(cmd.throttle) if cmd is not None else 0.0
        on = throttle_eff > 0.0
        no_prop_fired = False
        if on:
            dir_ecef = cmd.normalized_dir()
            if y[6] <= dry + 1e-9:
                # commanded burn, no propellant: the engine never lights
                # (one PROPELLANT_EXHAUSTED note, then silence -- a callback
                # that keeps asking for a dry engine is wrong once, not
                # every step)
                on = False
                throttle_eff = 0.0
                no_prop_fired = True
                if want_burn and not dry_logged:
                    event(t, "burn_end", y, {"reason": "PROPELLANT_EXHAUSTED",
                                             "attempted_throttle": cmd.throttle})
                    dry_logged = True
        if on and not burning:
            ignitions += 1
            if ignitions > vehicle.engine.restarts_max:
                raise RecoveryError(
                    f"burn commanded at t={t:g} s would be ignition {ignitions} of at most "
                    f"{vehicle.engine.restarts_max} -- restart count exhausted"
                )
            if want_burn:
                event(t, "burn_start", y, {"throttle": throttle_eff})
        elif not on and burning and not no_prop_fired:
            if want_burn:
                event(t, "burn_end", y, {"throttle": 0.0, "reason": "COMMANDED_CUTOFF"})
        burning = on

        pending: list[tuple[float, str, np.ndarray, Mapping[str, Any] | None]] = []
        grounded_at: tuple[float, np.ndarray] | None = None

        def detect(t0s: float, y0s: np.ndarray, span0: float, on0: bool,
                   thr0: float, dir0: np.ndarray | None, y1s: np.ndarray) -> bool:
            """Ground crossing (terminal, returns True) and recorded
            continuous events on the segment [y0s -> y1s]."""
            nonlocal grounded_at
            if g_ground(y0s) >= 0.0 > g_ground(y1s):
                alpha = bisect_fraction(t0s, y0s, span0, on0, thr0, dir0, g_ground, 0.0, 0.0, 1.0)
                y_imp = rk4(t0s, y0s, alpha * span0, on0, thr0, dir0)
                grounded_at = (t0s + alpha * span0, y_imp)
                return True
            if want_apogee and _radial_rate(y0s) > 0.0 >= _radial_rate(y1s):
                alpha = bisect_fraction(t0s, y0s, span0, on0, thr0, dir0, _radial_rate, 0.0, 0.0, 1.0)
                pending.append((t0s + alpha * span0, "apogee",
                                rk4(t0s, y0s, alpha * span0, on0, thr0, dir0), None))
            for thr in thresholds:
                g_t = lambda s, thr=thr: height_m(s[:3]) - thr.height_m
                crossed = (g_t(y0s) >= 0.0 > g_t(y1s)) if thr.direction in ("down", "both") \
                    else (g_t(y0s) <= 0.0 < g_t(y1s))
                if crossed:
                    alpha = bisect_fraction(t0s, y0s, span0, on0, thr0, dir0, g_t, 0.0, 0.0, 1.0)
                    pending.append((t0s + alpha * span0, thr.event_kind,
                                    rk4(t0s, y0s, alpha * span0, on0, thr0, dir0),
                                    {"direction": thr.direction}))
            return False

        # ---- integrate the step; a burn that runs dry splits it
        t_step0, y_step0, span_left, on_left, thr_left = t, y.copy(), span, burning, throttle_eff
        while True:
            y1 = rk4(t_step0, y_step0, span_left, on_left, thr_left, dir_ecef)
            if detect(t_step0, y_step0, span_left, on_left, thr_left, dir_ecef, y1):
                break
            if on_left and y_step0[6] > dry >= y1[6]:
                alpha = bisect_fraction(t_step0, y_step0, span_left, True, thr_left,
                                        dir_ecef, lambda s: float(s[6]), dry, 0.0, 1.0)
                y_cut = rk4(t_step0, y_step0, alpha * span_left, True, thr_left, dir_ecef)
                y_cut[6] = dry
                t_cut = t_step0 + alpha * span_left
                if want_burn:
                    event(t_cut, "burn_end", y_cut,
                          {"reason": "PROPELLANT_EXHAUSTED", "throttle": thr_left})
                dry_logged = True
                burning = False
                t_step0, y_step0 = t_cut, y_cut
                span_left = span - (t_cut - t)
                on_left, thr_left = False, 0.0
                continue
            break
        y_end = y1 if grounded_at is None else grounded_at[1]
        max_g = max(max_g, proper_g(thr_left if not grounded_at else 0.0, dir_ecef, y_end))
        _f, _s, _m, q_end = _aero_ecef(y_end[:3], y_end[3:6], vehicle, physics)
        max_q = max(max_q, q_end)

        for tt, kind, yy, extra in sorted(pending, key=lambda e: e[0]):
            event(tt, kind, yy, extra)

        if grounded_at is not None:
            t_imp, y_imp = grounded_at
            lat, lon, _h = frames.ecef_to_geodetic(y_imp[:3])
            up = y_imp[:3] / float(np.linalg.norm(y_imp[:3]))
            impact = ImpactSite(
                t_s=float(t_imp), latitude_deg=float(lat), longitude_deg=float(lon),
                speed_m_s=float(np.linalg.norm(y_imp[3:6])),
                vertical_speed_m_s=float(y_imp[3:6] @ up),
                position_ecef_m=tuple(float(x) for x in y_imp[:3]),
                velocity_ecef_m_s=tuple(float(x) for x in y_imp[3:6]),
            )
            if want_ground:
                event(t_imp, "ground", y_imp,
                      {"latitude_deg": float(lat), "longitude_deg": float(lon),
                       "speed_m_s": float(np.linalg.norm(y_imp[3:6]))})
            sample(t_imp, y_imp, 0.0)
            termination = "ground_impact"
            break

        t += span
        y = y_end
        if t >= grid_next - 1e-9 or t >= t_end - 1e-9:
            sample(t, y, throttle_eff)
            while grid_next <= times[-1] + 1e-9:
                grid_next += sample_s

    return BoosterTrack(
        epoch_utc=epoch,
        times_s=np.asarray(times, dtype=np.float64),
        mass_kg=np.asarray(mass_s, dtype=np.float64),
        position_ecef_m=np.asarray(pos, dtype=np.float64),
        velocity_ecef_m_s=np.asarray(vel, dtype=np.float64),
        height_m_arr=np.asarray(hgt_s, dtype=np.float64),
        throttle=np.asarray(thr_s, dtype=np.float64),
        mach=np.asarray(mach_s, dtype=np.float64),
        dynamic_pressure_pa=np.asarray(q_s, dtype=np.float64),
        events=tuple(recorded),
        impact=impact,
        termination=termination,
        max_proper_g=float(max_g),
        max_dynamic_pressure_pa=float(max_q),
        ignitions_used=ignitions,
        dry_mass_kg=dry,
        landing_reserve=float(vehicle.landing_propellant_reserve_kg),
    )


def ballistic_return(state0: BoosterState, vehicle: BoosterModel, t_end_s: float, *,
                     events: Iterable[str | AltitudeThreshold] = ("apogee", "ground"),
                     **kwargs) -> BoosterTrack:
    """The unpowered (``thrust_cmd=None``) return -- FL-301's first output:
    where, when and how fast a coasting booster comes down."""
    return propagate(state0, vehicle, None, t_end_s, events=events, **kwargs)


def separation_state_from_run(run, *, body_id: str = "stage1_debris") -> BoosterState:
    """The ``BoosterState`` to start a recovery run from, taken off a
    FlightRun: the first sample of a released-body track (the debris track
    FL-004 maps onto a separated stage) is the separation state -- frame,
    epoch and mission-clock t_s included."""
    for track in run.tracks:
        if track.body_id == body_id:
            epoch = datetime.fromisoformat(track.epoch_utc.replace("Z", "+00:00"))
            return BoosterState(
                t_s=float(track.t_s[0]),
                epoch_utc=_naive_utc(epoch),
                frame=track.frame,
                position_m=tuple(float(x) for x in track.position_m[0]),
                velocity_m_s=tuple(float(x) for x in track.velocity_m_s[0]),
                mass_kg=float(track.mass_kg[0]),
            )
    raise RecoveryError(
        f"run {run.run_id!r} has no track {body_id!r} to take a separation state from "
        f"(tracks: {', '.join(t.body_id for t in run.tracks) or 'none'})"
    )
