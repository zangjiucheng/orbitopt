"""Parametric ascent guidance: decision vector -> attitude program + vehicle (FL-201).

The E2 ascent optimizer searches a small, fixed-length decision vector; this
module turns it into the two things
:func:`orbitopt.flightlab.adapters.opentsiolkovsky.run_ascent` needs -- an
attitude program (the frozen E1 interface, a list of ``AttitudePoint``) and a
*derived vehicle document* carrying the timing/payload decisions, because
burn start/end, coast and masses live in the vehicle YAML ``stages``, not in
the attitude program (the E1 interface is frozen; the vehicle file is the
legal channel for timing and payload).

Decision vector (bounds from the vehicle file and the target -- see
:func:`bounds`):

=== ========================================= ======================================
ix  name                                      role
=== ========================================= =====================================
0   ``vertical_rise_time_s``                  fly vertical until this t, then kick
1   ``pitch_kick_deg``                        elevation drop of the initial pitch-over
2   ``pitch1_end_elev_deg``                   stage-1 linear-tangent pitch ramp ends here
3   ``pitch2_end_elev_deg``                   stage-2 linear-tangent pitch ramp ends here
4   ``launch_azimuth_deg``                    constant azimuth (nominal from inclination+site)
5   ``coast_s``                               stage-2 ``engine.burn_start_s`` after separation
6   ``stage2_burn_end_s``                     stage-2 ``engine.burn_end_s`` (cutoff, stage-local)
7   ``payload_kg``                            present only when the objective is payload
=== ========================================= =====================================

Pitch segments are linear in elevation vs time ("linear tangent"): a vertical
rise, a kick, then a straight ramp down to the segment's end elevation -- one
gravity-turn-shaped ramp per stage. The azimuth is constant; its bounds centre
on the classic flat-Earth launch azimuth for the target inclination and the
site latitude, so the optimizer tunes the achieved inclination around the
feasible nominal.

:func:`derive_vehicle` never touches the base document (deep copy) and adds
payload mass to the **last stage only** (``mass_initial_kg`` and
``mass_dry_kg`` both grow by ``payload_kg``: the vehicle file carries no
separate payload entry, so the reserve bookkeeping of
:func:`orbitopt.flightlab.checks.propellant_reserve_check` treats the payload
as non-burnable stack mass -- conservative in the vehicle's favour, matching
the note in ``checks.py``).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint

#: Standard gravity [m/s^2] for the propellant mass flow implied by thrust/Isp.
G0 = 9.80665

# --- program shape (design constants, not decision variables) --------------
KICK_TIME_S = 5.0        # duration of the initial pitch-over ramp
MECO_MARGIN_S = 5.0      # the stage-1 pitch ramp ends this far before separation
PITCH2_LEAD_S = 10.0     # the stage-2 pitch ramp ends this far before cutoff
PROGRAM_TAIL_S = 20.0    # the final waypoint holds this far past cutoff
#: Launch-azimuth bound width [deg] around the inclination/site nominal.
AZIMUTH_WINDOW_DEG = 25.0
#: Coast bound [s] after separation (upper bound; shrinks if the sim window bites).
COAST_MAX_S = 60.0
#: Payload bound: at most this fraction of the upper stage's propellant mass.
PAYLOAD_MAX_PROP_FRACTION = 0.5

OBJECTIVES = ("payload", "reserve", "booster_reserve")


class GuidanceError(ValueError):
    """A decision vector or vehicle document the parameterization cannot use."""


@dataclass(frozen=True)
class AscentParams:
    """Named view of a decision vector (see the module docstring table)."""

    vertical_rise_time_s: float
    pitch_kick_deg: float
    pitch1_end_elev_deg: float
    pitch2_end_elev_deg: float
    launch_azimuth_deg: float
    coast_s: float
    stage2_burn_end_s: float
    payload_kg: float = 0.0


def decision_names(objective: str) -> tuple[str, ...]:
    """Decision-variable names for ``objective`` (payload is a variable only
    when the objective is payload)."""
    if objective not in OBJECTIVES:
        raise GuidanceError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    names = ("vertical_rise_time_s", "pitch_kick_deg", "pitch1_end_elev_deg",
             "pitch2_end_elev_deg", "launch_azimuth_deg", "coast_s", "stage2_burn_end_s")
    return names + (("payload_kg",) if objective == "payload" else ())


def vector_size(objective: str) -> int:
    """Length of the decision vector for ``objective``."""
    return len(decision_names(objective))


def unpack(x: Sequence[float], objective: str = "payload") -> AscentParams:
    """Validate and name a decision vector."""
    expected = vector_size(objective)
    if len(x) != expected:
        raise GuidanceError(f"decision vector has {len(x)} components, expected {expected} "
                            f"for objective {objective!r} ({', '.join(decision_names(objective))})")
    vals = [float(v) for v in x]
    if not all(math.isfinite(v) for v in vals):
        raise GuidanceError(f"decision vector has non-finite entries: {vals}")
    return AscentParams(*vals[:7], payload_kg=vals[7] if len(vals) == 8 else 0.0)


def as_vector(p: AscentParams) -> list[float]:
    """Decision vector back from an :class:`AscentParams` (7 components when
    payload is zero, 8 when it is a decision)."""
    v = [p.vertical_rise_time_s, p.pitch_kick_deg, p.pitch1_end_elev_deg,
         p.pitch2_end_elev_deg, p.launch_azimuth_deg, p.coast_s, p.stage2_burn_end_s]
    return v + ([p.payload_kg] if p.payload_kg else [])


# --------------------------------------------------------------------------
# vehicle-derived quantities
# --------------------------------------------------------------------------

def _engine(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    eng = stage.get("engine")
    if not isinstance(eng, Mapping):
        raise GuidanceError("every stage needs an 'engine' mapping (burn_start_s/burn_end_s, "
                            "thrust_vac_n, isp_vac_s) for the optimizer's bounds")
    for key in ("thrust_vac_n", "isp_vac_s", "burn_end_s"):
        if key not in eng:
            raise GuidanceError(f"stage engine is missing {key!r}")
    return eng


def _thrust_range_n(thrust: Any) -> tuple[float, float]:
    """(min, max) thrust [N] from a scalar or a ``[[t_s, N], ...]`` throttle curve."""
    if isinstance(thrust, (int, float)):
        return float(thrust), float(thrust)
    if isinstance(thrust, Sequence) and thrust and all(
            isinstance(pt, Sequence) and len(pt) == 2 for pt in thrust):
        values = [float(pt[1]) for pt in thrust]
        return min(values), max(values)
    raise GuidanceError(f"cannot read thrust_vac_n {thrust!r}")


def propellant_mass_kg(stage: Mapping[str, Any]) -> float:
    """Burnable propellant mass of a stage [kg]."""
    return float(stage["mass_initial_kg"]) - float(stage["mass_dry_kg"])


def burnout_time_s(stage: Mapping[str, Any]) -> float:
    """Burn duration [s] implied by the stage's propellant, thrust and Isp,
    evaluated at the *peak* thrust -- the conservative (shortest) read for a
    throttled engine, so a ``burn_end_s`` bounded by it can never ask for more
    propellant than the stage carries."""
    eng = _engine(stage)
    prop = propellant_mass_kg(stage)
    if prop <= 0.0:
        raise GuidanceError("stage carries no propellant (mass_initial == mass_dry)")
    _, t_max = _thrust_range_n(eng["thrust_vac_n"])
    if t_max <= 0.0 or float(eng["isp_vac_s"]) <= 0.0:
        raise GuidanceError("stage engine needs positive thrust and Isp")
    return prop * float(eng["isp_vac_s"]) * G0 / t_max


def separation_time_s(vehicle_doc: Mapping[str, Any]) -> float:
    """Stage-1 separation time [s, absolute] (falling back to the stage-1
    cutoff when the file states no separation event)."""
    s1 = vehicle_doc["stages"][0]
    if "separation_time_s" in s1:
        return float(s1["separation_time_s"])
    return float(_engine(s1)["burn_end_s"])


# --------------------------------------------------------------------------
# launch azimuth from inclination + site
# --------------------------------------------------------------------------

def nominal_azimuth_deg(inclination_deg: float, latitude_deg: float) -> float:
    """Classic flat-Earth launch azimuth [deg, clockwise from north] for
    ``inclination_deg`` from ``latitude_deg``: ``sin(az) = cos(i)/cos(lat)``,
    principal (equatorward-of-pole) branch. Raises when the inclination is
    unreachable from the latitude (the site sits above the target inclination
    -- a target/site problem, not a search problem)."""
    lat = math.radians(float(latitude_deg))
    arg = math.cos(math.radians(float(inclination_deg))) / math.cos(lat)
    if not -1.0 <= arg <= 1.0:
        raise GuidanceError(
            f"inclination {inclination_deg} deg is not reachable from latitude "
            f"{latitude_deg} deg (|cos i / cos lat| > 1); the site must be at or "
            f"below the target inclination")
    return math.degrees(math.asin(arg))


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------

def bounds(vehicle_doc: Mapping[str, Any], target: Any = None,
           objective: str = "payload") -> tuple[list[float], list[float]]:
    """(lower, upper) per-component decision bounds from the vehicle document
    (timing windows, propellant-implied cutoff) and, when given, the target
    (:class:`~orbitopt.flightlab.contract.FlightTarget` -- its inclination and
    site centre the azimuth window).

    ``objective="payload"`` appends the payload bound
    ``[0, PAYLOAD_MAX_PROP_FRACTION * upper-stage propellant]``; every other
    objective leaves payload fixed (not a decision)."""
    stages = vehicle_doc.get("stages") or []
    if not stages:
        raise GuidanceError("vehicle document needs at least one stage")
    if objective not in OBJECTIVES:
        raise GuidanceError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    s1, slast = stages[0], stages[-1]
    t_burn1 = float(_engine(s1)["burn_end_s"]) - float(_engine(s1).get("burn_start_s", 0.0))
    sep = separation_time_s(vehicle_doc)
    burnout = burnout_time_s(slast)
    end_time = float((vehicle_doc.get("simulation") or {})["end_time_s"])
    az_nom = 90.0 if target is None else nominal_azimuth_deg(
        target.target_orbit.inclination_deg, target.launch_site.latitude_deg)

    tv_hi = max(5.0, min(40.0, 0.3 * t_burn1))
    coast_hi = min(COAST_MAX_S, max(0.0, end_time - sep - burnout - PROGRAM_TAIL_S))
    p1_hi = max(tv_hi + 5.0, sep - MECO_MARGIN_S - KICK_TIME_S - tv_hi)
    p2_hi = max(10.0, burnout * 0.5)
    cutoff_lo = max(30.0, 0.4 * burnout)
    cutoff_hi = burnout - 5.0
    if cutoff_hi <= cutoff_lo:
        raise GuidanceError(f"upper stage cannot hold any feasible cutoff window "
                            f"(implied burnout {burnout:.1f} s)")
    lo = [2.0, 1.0, 0.0, 0.0, az_nom - AZIMUTH_WINDOW_DEG, 0.0, cutoff_lo]
    hi = [tv_hi, 20.0, min(60.0, p1_hi), min(45.0, p2_hi),
          az_nom + AZIMUTH_WINDOW_DEG, coast_hi, cutoff_hi]
    if objective == "payload":
        lo.append(0.0)
        hi.append(PAYLOAD_MAX_PROP_FRACTION * propellant_mass_kg(slast))
    return lo, hi


# --------------------------------------------------------------------------
# decode: attitude program + derived vehicle
# --------------------------------------------------------------------------

def derive_vehicle(base_vehicle: Mapping[str, Any], x: Sequence[float], *,
                   payload_kg: float | None = None) -> dict:
    """Copy of ``base_vehicle`` with the timing/payload decisions applied: the
    **last** stage's ``engine.burn_start_s`` = coast (``x[5]``),
    ``engine.burn_end_s`` = cutoff (``x[6]``), and ``payload_kg`` (the
    vector's 8th component when present, else the keyword) added to that last
    stage's initial and dry mass. The base mapping is never mutated."""
    doc = copy.deepcopy(dict(base_vehicle))
    stages = doc.get("stages") or []
    if not stages:
        raise GuidanceError("vehicle document needs at least one stage")
    x = [float(v) for v in x]
    if len(x) not in (7, 8):
        raise GuidanceError(f"decision vector has {len(x)} components, expected 7 or 8")
    last = stages[-1]
    if "engine" not in last:
        raise GuidanceError("the last stage needs an 'engine' to carry the coast/cutoff decisions")
    last["engine"]["burn_start_s"] = x[5]
    last["engine"]["burn_end_s"] = x[6]
    mass = x[7] if (payload_kg is None and len(x) == 8) else float(payload_kg or 0.0)
    if mass > 0.0:
        last["mass_initial_kg"] = float(last["mass_initial_kg"]) + mass
        last["mass_dry_kg"] = float(last["mass_dry_kg"]) + mass
    return doc


def attitude_program(p: AscentParams, vehicle_doc: Mapping[str, Any]) -> list[AttitudePoint]:
    """The attitude program ``p`` describes: vertical rise, kick, one linear
    (gravity-turn-shaped) pitch ramp per stage, constant azimuth, a flat hold
    past cutoff. Times are strictly increasing and inside the vehicle's
    simulation window; values are clamped into a physically ordered program
    rather than rejected, so an out-of-bounds candidate degrades gracefully
    instead of aborting the optimizer."""
    stages = vehicle_doc.get("stages") or []
    if not stages:
        raise GuidanceError("vehicle document needs at least one stage")
    sep = separation_time_s(vehicle_doc)
    end_time = float((vehicle_doc.get("simulation") or {})["end_time_s"])
    az = float(p.launch_azimuth_deg)

    def clamp(v: float, lo: float, hi: float) -> float:
        return min(max(v, lo), hi)

    tv = clamp(p.vertical_rise_time_s, 0.0, max(0.0, sep - 2.0 * (KICK_TIME_S + 1.0)))
    t_kick = tv + KICK_TIME_S
    el_kick = clamp(90.0 - p.pitch_kick_deg, 0.0, 90.0)
    el1 = clamp(p.pitch1_end_elev_deg, 0.0, max(0.0, el_kick - 1.0))
    el2 = clamp(p.pitch2_end_elev_deg, 0.0, el1)
    t1_end = clamp(sep - MECO_MARGIN_S, t_kick + 1.0, max(t_kick + 1.0, sep - 1.0))
    coast = clamp(p.coast_s, 0.0, COAST_MAX_S)
    ign2 = sep + coast
    cutoff = ign2 + clamp(p.stage2_burn_end_s, 1.0,
                          max(1.0, end_time - ign2 - PROGRAM_TAIL_S))
    t2_end = clamp(cutoff - PITCH2_LEAD_S, ign2 + 1.0, cutoff - 0.5)

    pts: list[tuple[float, float, float]] = [
        (0.0, az, 90.0),
        (tv, az, 90.0),
        (t_kick, az, el_kick),
        (t1_end, az, el1),
    ]
    if coast > 1.0:  # mid-coast waypoint: the coasting stack keeps pitching down
        pts.append((ign2 - 0.5 * coast, az, 0.5 * (el1 + el2)))
    pts.append((ign2, az, el1 if coast <= 1.0 else el2))
    pts.append((t2_end, az, el2))
    pts.append((min(end_time - 1.0, cutoff + PROGRAM_TAIL_S), az, el2))

    out: list[AttitudePoint] = []
    for t, a, e in pts:
        t = max(t, (out[-1].t_s if out else -1.0) + 1e-3)  # strictly increasing
        if t > end_time - 0.5:
            break
        out.append(AttitudePoint(round(float(t), 6), round(float(a), 6), round(float(e), 6)))
    if len(out) < 3:
        raise GuidanceError(f"decision vector {p} produces a degenerate attitude program "
                            f"({len(out)} waypoints inside the {end_time} s simulation window)")
    return out


def decode(x: Sequence[float], target: Any, vehicle_doc: Mapping[str, Any],
           objective: str = "payload",
           payload_kg: float | None = None) -> tuple[list[AttitudePoint], dict]:
    """Full decode of decision vector ``x``: ``(attitude program, derived
    vehicle document)``. ``payload_kg`` overrides the vector's payload slot --
    pass the fixed payload for the ``reserve``/``booster_reserve`` objectives."""
    p = unpack(x, objective if len(x) == vector_size(objective) else "reserve")
    doc = derive_vehicle(vehicle_doc, as_vector(p), payload_kg=payload_kg)
    return attitude_program(p, vehicle_doc), doc
