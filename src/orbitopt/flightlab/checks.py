"""Flight Lab mission-constraint checks + Gate-B verdict (FL-006).

:func:`evaluate` turns a FlightResult run and its FlightTarget into the
:class:`~orbitopt.flightlab.contract.Verification` the Gate-B CLI writes:
the three independent booleans plus every check that produced them. It is a
pure function of ``(run, target)`` -- no compute stack, no subprocess, no
numpy -- so E2 can reuse it as an objective post-check and E3 as a replay
verdict, without importing an ascent backend or a propagator.

Which checks ``evaluate`` assesses
----------------------------------
* Kept from the run, but re-assessed against the target in front of us:
  the adapter's data-integrity checks (mass bookkeeping), its max-Q / max-g
  checks (OpenTsiolkovsky measures those continuously, so its summary value
  is used, but the target limit comes from the target), any liftoff T/W /
  propellant-reserve check the caller put in (those need the vehicle file,
  which ``evaluate`` deliberately does not take), and any orbit re-fly check
  (assessed once by ``orbit_refly`` in orbitopt.flightlab.simulate). A target
  limit with no stored value is a failed data-missing check, not silence.
* Recomputed here from the run's own GCRS tracks: the insertion orbit
  elements at SECO (periapsis / apoapsis / inclination / RAAN vs the target
  and its tolerances), and -- only when no backend value exists -- max-Q from
  the trajectory CSV's ``dynamic_pressure_pa`` extras.

``mission_constraints_passed`` is the conjunction of the mission-constraint
checks (see :data:`CONSTRAINT_CHECK_IDS`); data-integrity and re-fly checks
deliberately do not enter it (a re-fly failure means *physics* is not
verified, not that a *constraint* was violated). An empty constraint set
means "not assessed" (None), never "passed".

Gate G5 (declared before first run, never loosened afterwards):
the orbit re-fly tolerances below, and the 120 km atmosphere ceiling here.
The tests assert these constants before flying, and the only way to turn a
red verdict green is to fly better, not to widen a limit.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from orbitopt.flightlab.contract import (
    Check,
    FlightRun,
    FlightTarget,
    Verification,
)

# --- physical / design constants ------------------------------------------
#: WGS84 equatorial radius [m] -- same value as orbitopt.flightlab.frames.
EARTH_RADIUS_M = 6378137.0
#: Standard Earth gravitational parameter [m^3/s^2] (EGM96/WGS84). Insertion
#: elements in this module are computed with this mu; the tudatpy re-fly
#: re-computes BOTH element sets with the mu its own propagator reports.
EARTH_MU = 3.9860044150e14
#: Standard gravity [m/s^2] for the liftoff thrust-to-weight ratio.
STANDARD_GRAVITY = 9.80665
#: Gate G5: an insertion periapsis below this altitude is "inside the
#: atmosphere" -- a failed mission, whatever the target's own floor says.
ATMOSPHERE_CEILING_M = 120000.0
#: A vehicle that cannot lift off cannot fly, whatever the trajectory file
#: claims afterwards.
MIN_LIFTOFF_THRUST_TO_WEIGHT = 1.0
#: RAAN tolerance applied when the target constrains RAAN (the target schema
#: carries no RAAN-specific tolerance; 1 deg is the Flight Lab default).
RAAN_TOLERANCE_DEG = 1.0

#: Orbit re-fly drift limits (Gate G5). A point-mass, single-body, one-orbit
#: re-integration of the insertion state must reproduce the osculating
#: elements within these bounds for ``physics_verified=true``: the DOP853
#: integrator at the verify tolerances drifts a 400 km orbit by centimetres,
#: and the GCRS<->ECLIPJ2000 rotation is exact, so a visible drift means the
#: insertion state is not a self-consistent orbit -- not integrator noise.
REFLY_MAX_A_M = 100.0
REFLY_MAX_DE = 1e-6
REFLY_MAX_DI_DEG = 0.005
REFLY_MAX_DRAAN_DEG = 0.02

#: Check ids whose pass/fail decides ``mission_constraints_passed``.
CONSTRAINT_CHECK_IDS = frozenset({
    "max-q-within-limit",
    "max-accel-within-limit",
    "liftoff-thrust-to-weight",
    "propellant-reserve",
    "recovery-propellant-reserve",
    "insertion-above-atmosphere",
    "insertion-periapsis-above-minimum",
    "insertion-periapsis-within-tolerance",
    "insertion-apoapsis-within-tolerance",
    "insertion-inclination-within-tolerance",
    "insertion-raan-within-tolerance",
})

_STACK_RE = re.compile(r"stage[0-9]+")


# --------------------------------------------------------------------------
# orbital mechanics (pure stdlib math, equatorial-frame convention)
# --------------------------------------------------------------------------

def orbital_elements(r: Sequence[float], v: Sequence[float], mu: float = EARTH_MU) -> dict:
    """Classical Keplerian elements (SI, angles in deg) from a Cartesian
    state in any inertial equatorial frame -- the element convention matches
    OpenTsiolkovsky's ``summary.final_state.orbit`` and tudat's
    ``cartesian_to_keplerian`` up to the true-equinox/equinox-of-date RAAN
    convention inside the sub-metre verification noise."""
    rx, ry, rz = (float(x) for x in r)
    vx, vy, vz = (float(x) for x in v)
    rm = math.sqrt(rx * rx + ry * ry + rz * rz)
    vm2 = vx * vx + vy * vy + vz * vz
    hx, hy, hz = ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx
    hm = math.sqrt(hx * hx + hy * hy + hz * hz)
    rdotv = rx * vx + ry * vy + rz * vz
    ex = (vm2 / mu - 1.0 / rm) * rx - rdotv / mu * vx
    ey = (vm2 / mu - 1.0 / rm) * ry - rdotv / mu * vy
    ez = (vm2 / mu - 1.0 / rm) * rz - rdotv / mu * vz
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    a = 1.0 / (2.0 / rm - vm2 / mu) if vm2 / mu - 2.0 / rm < 0.0 else float("inf")
    inc_deg = math.degrees(math.acos(max(-1.0, min(1.0, hz / hm)))) if hm else 0.0
    nx, ny, nz = -hy, hx, 0.0  # node vector = k x h
    nm = math.sqrt(nx * nx + ny * ny)
    if nm < 1e-9:  # equatorial: RAAN undefined, report 0 (argument of perigee measured from x)
        raan_deg = 0.0
        argp_deg = math.degrees(math.atan2(ey, ex)) % 360.0
    else:
        raan_deg = math.degrees(math.atan2(ny, nx)) % 360.0
        cos_argp = (nx * ex + ny * ey) / (nm * e) if e > 0 else 1.0
        argp_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_argp))))
        if ez < 0:
            argp_deg = 360.0 - argp_deg
    e_cap = min(e, 0.9999999) if math.isfinite(a) else 1.0
    periapsis_alt = a * (1.0 - e_cap) - EARTH_RADIUS_M if math.isfinite(a) else -float("inf")
    apoapsis_alt = a * (1.0 + e_cap) - EARTH_RADIUS_M if math.isfinite(a) else float("inf")
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu) if a > 0 and math.isfinite(a) else float("inf")
    return {
        "semi_major_axis_m": a,
        "eccentricity": e,
        "inclination_deg": inc_deg,
        "raan_deg": raan_deg,
        "argument_of_perigee_deg": argp_deg,
        "periapsis_altitude_m": periapsis_alt,
        "apoapsis_altitude_m": apoapsis_alt,
        "period_s": period,
    }


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """|a - b| on the circle, in [0, 180]."""
    d = (a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


# --------------------------------------------------------------------------
# locating the insertion state in a run
# --------------------------------------------------------------------------

def insertion_state(run: FlightRun) -> dict:
    """The upper stage's insertion (SECO) state as written in the run: the
    sample of the ``seco`` event's body track nearest the event time, or the
    last sample of the last-ending track when the run declares no SECO (e.g.
    a hand-built test run). Returned dict: ``body_id``, ``t_s``,
    ``position_m``, ``velocity_m_s``, ``mass_kg``. The track's frame is the
    manifest's (contract: GCRS after the frame bridge)."""
    if not run.tracks:
        raise ValueError(f"run {run.run_id!r} has no body tracks -- nothing to verify")
    event = next((e for e in run.events if e.kind == "seco"), None)
    track = None
    if event is not None:
        track = next((t for t in run.tracks if t.body_id == event.body_id), None)
    if track is None:
        track = max(run.tracks, key=lambda t: t.t_end_s)
        idx = len(track.t_s) - 1
    else:
        idx = min(range(len(track.t_s)), key=lambda i: abs(track.t_s[i] - event.t_s))
    return {
        "body_id": track.body_id,
        "t_s": track.t_s[idx],
        "position_m": track.position_m[idx],
        "velocity_m_s": track.velocity_m_s[idx],
        "mass_kg": track.mass_kg[idx],
        "epoch_utc": track.epoch_utc,
        "event_t_s": event.t_s if event is not None else track.t_s[idx],
    }


# --------------------------------------------------------------------------
# checks needing the vehicle file (assessed by simulate(), kept by evaluate)
# --------------------------------------------------------------------------

def _liftoff_thrust_n(vehicle_doc: Mapping[str, Any]) -> float | None:
    """Stage-1 thrust at t=0 [N] from a flightlab vehicle document: scalar
    ``thrust_vac_n``, or the first point of the throttle curve."""
    stages = vehicle_doc.get("stages") or []
    if not stages:
        return None
    thrust = (stages[0].get("engine") or {}).get("thrust_vac_n")
    if thrust is None:
        return None
    if isinstance(thrust, (int, float)):
        return float(thrust)
    if isinstance(thrust, list) and thrust:
        return float(thrust[0][1])
    return None


def liftoff_thrust_to_weight_check(run: FlightRun, vehicle_doc: Mapping[str, Any]) -> Check | None:
    """Liftoff thrust-to-weight from the vehicle file and the run's own
    liftoff mass. Returns None (not assessed) when the vehicle states no
    stage-1 thrust."""
    thrust0 = _liftoff_thrust_n(vehicle_doc)
    if thrust0 is None:
        return None
    stack = [t for t in run.tracks if _STACK_RE.fullmatch(t.body_id)]
    if not stack:
        return None
    first = min(stack, key=lambda t: t.t_start_s)
    mass0 = first.mass_kg[0]
    tw = thrust0 / (mass0 * STANDARD_GRAVITY)
    passed = tw >= MIN_LIFTOFF_THRUST_TO_WEIGHT
    return Check(
        id="liftoff-thrust-to-weight", passed=passed, value=tw,
        limit=MIN_LIFTOFF_THRUST_TO_WEIGHT, unit="",
        reason_code="" if passed else "LIFTOFF_THRUST_TO_WEIGHT_LOW",
    )


def propellant_reserve_check(run: FlightRun, vehicle_doc: Mapping[str, Any],
                             target: FlightTarget) -> Check | None:
    """Upper-stage propellant left at insertion vs the target's upper-stage
    reserve. The mass bookkeeping treats the final stage's ``mass_dry_kg`` as
    everything that is not burnable (dry stage + payload: the vehicle file
    carries no separate payload entry), so the reserve is conservative in the
    vehicle's favour -- honest for the synthetic demonstrator, to be refined
    with real vehicle models in E2. Returns None (not assessed) when the
    target states no upper-stage reserve or the vehicle states no final-stage
    dry mass. Recovery reserve is a separate check, below."""
    limit = target.constraints.min_propellant_reserve_kg
    stages = vehicle_doc.get("stages") or []
    if limit is None or not stages or "mass_dry_kg" not in stages[-1]:
        return None
    mass_insertion = insertion_state(run)["mass_kg"]
    remaining = mass_insertion - float(stages[-1]["mass_dry_kg"])
    passed = remaining >= float(limit)
    return Check(
        id="propellant-reserve", passed=passed, value=remaining, limit=float(limit),
        unit="kg", reason_code="" if passed else "PROPELLANT_RESERVE_LOW",
    )


def _released_first_stage_track(run: FlightRun):
    for track in run.tracks:
        if track.parent_body_id == "stage1" and track.body_id.endswith("_debris"):
            return track
    for track in run.tracks:
        if track.body_id == "stage1_debris":
            return track
    return None


def recovery_propellant_reserve_check(run: FlightRun, vehicle_doc: Mapping[str, Any],
                                      target: FlightTarget) -> Check | None:
    """First-stage propellant left for recovery vs the target's recovery
    reserve. Returns None only when recovery is disabled or the target states
    no reserve; missing stage data is a failed check, not silence."""
    limit = target.recovery.min_propellant_reserve_kg
    if target.recovery.kind == "none" or limit is None:
        return None
    limit = float(limit)
    stages = vehicle_doc.get("stages") or []
    if not stages or "mass_dry_kg" not in stages[0]:
        return Check(
            id="recovery-propellant-reserve", passed=False, limit=limit, unit="kg",
            reason_code="RECOVERY_PROPELLANT_RESERVE_DATA_MISSING",
        )
    track = _released_first_stage_track(run)
    if track is None or not track.mass_kg:
        return Check(
            id="recovery-propellant-reserve", passed=False, limit=limit, unit="kg",
            reason_code="RECOVERY_PROPELLANT_RESERVE_DATA_MISSING",
        )
    remaining = float(track.mass_kg[-1]) - float(stages[0]["mass_dry_kg"])
    passed = remaining >= limit
    return Check(
        id="recovery-propellant-reserve", passed=passed, value=remaining, limit=limit,
        unit="kg", reason_code="" if passed else "PROPELLANT_RESERVE_LOW",
    )


# --------------------------------------------------------------------------
# insertion-orbit checks (recomputed by evaluate from the run's GCRS tracks)
# --------------------------------------------------------------------------

def insertion_checks(run: FlightRun, target: FlightTarget,
                     elements: Mapping[str, float] | None = None) -> list[Check]:
    """The Gate-B insertion checks: an atmosphere-floor hard constraint that
    applies to every flight, then the target's own minimum periapsis and the
    altitude / inclination / (RAAN) tolerances where the target constrains
    them. All element-dependent checks read the same element dict (computed
    with :data:`EARTH_MU` unless the caller passes one)."""
    if elements is None:
        st = insertion_state(run)
        elements = orbital_elements(st["position_m"], st["velocity_m_s"])
    constraints = target.constraints
    orbit = target.target_orbit
    checks: list[Check] = []

    rp = float(elements["periapsis_altitude_m"])
    above = rp >= ATMOSPHERE_CEILING_M
    checks.append(Check(
        id="insertion-above-atmosphere", passed=above, value=rp,
        limit=ATMOSPHERE_CEILING_M, unit="m",
        reason_code="" if above else "PERIAPSIS_IN_ATMOSPHERE",
    ))
    if constraints.minimum_insertion_periapsis_m is not None:
        limit = float(constraints.minimum_insertion_periapsis_m)
        passed = rp >= limit
        checks.append(Check(
            id="insertion-periapsis-above-minimum", passed=passed, value=rp,
            limit=limit, unit="m",
            reason_code="" if passed else "INSERTION_PERIAPSIS_BELOW_MINIMUM",
        ))
    tol = constraints.orbit_altitude_tolerance_m
    if tol is not None:
        tol = float(tol)
        for name, actual, want in (
            ("periapsis", rp, float(orbit.periapsis_altitude_m)),
            ("apoapsis", float(elements["apoapsis_altitude_m"]), float(orbit.apoapsis_altitude_m)),
        ):
            delta = actual - want
            passed = abs(delta) <= tol
            checks.append(Check(
                id=f"insertion-{name}-within-tolerance", passed=passed, value=actual,
                limit=want, unit="m",
                reason_code="" if passed else f"INSERTION_{name.upper()}_OUT_OF_TOLERANCE",
            ))
    if constraints.inclination_tolerance_deg is not None:
        tol = float(constraints.inclination_tolerance_deg)
        actual = float(elements["inclination_deg"])
        passed = abs(actual - float(orbit.inclination_deg)) <= tol
        checks.append(Check(
            id="insertion-inclination-within-tolerance", passed=passed, value=actual,
            limit=float(orbit.inclination_deg), unit="deg",
            reason_code="" if passed else "INSERTION_INCLINATION_OUT_OF_TOLERANCE",
        ))
    if orbit.raan_deg is not None:
        actual = float(elements["raan_deg"])
        passed = angle_diff_deg(actual, float(orbit.raan_deg)) <= RAAN_TOLERANCE_DEG
        checks.append(Check(
            id="insertion-raan-within-tolerance", passed=passed, value=actual,
            limit=float(orbit.raan_deg), unit="deg",
            reason_code="" if passed else "INSERTION_RAAN_OUT_OF_TOLERANCE",
        ))
    return checks


def max_q_check_from_tracks(run: FlightRun, limit_pa: float) -> Check | None:
    """Fallback max-Q: the CSV ``dynamic_pressure_pa`` extra, maxed over the
    stack tracks. Only for backends that do not report a continuous max-Q;
    returns None when no track carries the column."""
    values = [
        q for t in run.tracks if _STACK_RE.fullmatch(t.body_id)
        for q in t.extra_columns.get("dynamic_pressure_pa", ())
    ]
    if not values:
        return None
    q = max(values)
    passed = q <= float(limit_pa)
    return Check(
        id="max-q-within-limit", passed=passed, value=q, limit=float(limit_pa), unit="Pa",
        reason_code="" if passed else "max_q_exceeded",
    )


# --------------------------------------------------------------------------
# the Gate-B verdict
# --------------------------------------------------------------------------

def _find_check(checks: Sequence[Check], check_id: str) -> Check | None:
    return next((c for c in checks if c.id == check_id), None)


def _replace_or_append(checks: list[Check], replacement: Check) -> None:
    for i, c in enumerate(checks):
        if c.id == replacement.id:
            checks[i] = replacement
            return
    checks.append(replacement)


def _deduplicate_checks(checks: Sequence[Check]) -> list[Check]:
    """Collapse repeated check ids while preserving the first occurrence's
    position. A later entry with a measured value wins over an earlier
    data-missing placeholder; among equal-value states, the later entry wins."""
    order: list[str] = []
    by_id: dict[str, Check] = {}
    for check in checks:
        old = by_id.get(check.id)
        if old is None:
            order.append(check.id)
        elif old.value is not None and check.value is None:
            continue
        by_id[check.id] = check
    return [by_id[check_id] for check_id in order]


def _reassess_limit(checks: list[Check], check_id: str, limit: float, unit: str, *,
                    greater_is_ok: bool, fail_code: str) -> Check | None:
    old = _find_check(checks, check_id)
    if old is None or old.value is None:
        return None
    value = float(old.value)
    limit = float(limit)
    passed = value >= limit if greater_is_ok else value <= limit
    new = Check(
        id=check_id, passed=passed, value=value, limit=limit, unit=unit,
        reason_code="" if passed else fail_code,
    )
    _replace_or_append(checks, new)
    return new


def evaluate(run: FlightRun, target: FlightTarget) -> Verification:
    """The Gate-B verdict for ``run`` against ``target``: keeps the run's
    backend and re-fly checks, collapses duplicate ids preferring a measured
    value over an earlier data-missing placeholder, re-assesses every limit that
    ``target`` declares against its stored value, recomputes the insertion checks from the tracks,
    settles max-Q from the CSVs when no backend reported it, and conjoins the
    resulting mission-constraint checks into ``mission_constraints_passed``.
    Pure function of ``(run, target)`` -- never mutates the run; the booleans
    stay independent (None = not assessed, never "passed"). A declared reserve
    or flight-load limit with no backing value is a failed data-missing check,
    not silence. See the module docstring for the Gate G5 thresholds this
    encodes."""
    kept = [c for c in run.verification.checks if c.id not in {
        "insertion-above-atmosphere", "insertion-periapsis-above-minimum",
        "insertion-periapsis-within-tolerance", "insertion-apoapsis-within-tolerance",
        "insertion-inclination-within-tolerance", "insertion-raan-within-tolerance",
    }]
    declared_ids = set()
    if target.constraints.max_dynamic_pressure_pa is not None:
        declared_ids.add("max-q-within-limit")
    if target.constraints.max_specific_acceleration_g is not None:
        declared_ids.add("max-accel-within-limit")
    if target.constraints.min_propellant_reserve_kg is not None:
        declared_ids.add("propellant-reserve")
    if (target.recovery.kind != "none"
            and target.recovery.min_propellant_reserve_kg is not None):
        declared_ids.add("recovery-propellant-reserve")
    kept = [c for c in kept if c.id not in CONSTRAINT_CHECK_IDS or c.id in declared_ids
            or c.id == "liftoff-thrust-to-weight"]
    kept = _deduplicate_checks(kept)

    limit = target.constraints.max_dynamic_pressure_pa
    if limit is not None:
        if _reassess_limit(kept, "max-q-within-limit", limit, "Pa",
                           greater_is_ok=False, fail_code="max_q_exceeded") is None:
            fallback = max_q_check_from_tracks(run, float(limit))
            if fallback is not None:
                _replace_or_append(kept, fallback)
            else:
                _replace_or_append(kept, Check(
                    id="max-q-within-limit", passed=False, limit=float(limit), unit="Pa",
                    reason_code="MAX_Q_DATA_MISSING",
                ))

    limit = target.constraints.max_specific_acceleration_g
    if limit is not None:
        if _reassess_limit(kept, "max-accel-within-limit", limit, "g",
                           greater_is_ok=False, fail_code="max_accel_exceeded") is None:
            _replace_or_append(kept, Check(
                id="max-accel-within-limit", passed=False, limit=float(limit), unit="g",
                reason_code="MAX_ACCEL_DATA_MISSING",
            ))

    limit = target.constraints.min_propellant_reserve_kg
    if limit is not None:
        if _reassess_limit(kept, "propellant-reserve", limit, "kg",
                           greater_is_ok=True, fail_code="PROPELLANT_RESERVE_LOW") is None:
            _replace_or_append(kept, Check(
                id="propellant-reserve", passed=False, limit=float(limit), unit="kg",
                reason_code="PROPELLANT_RESERVE_DATA_MISSING",
            ))

    limit = target.recovery.min_propellant_reserve_kg
    if target.recovery.kind != "none" and limit is not None:
        if _reassess_limit(kept, "recovery-propellant-reserve", limit, "kg",
                           greater_is_ok=True,
                           fail_code="PROPELLANT_RESERVE_LOW") is None:
            _replace_or_append(kept, Check(
                id="recovery-propellant-reserve", passed=False, limit=float(limit), unit="kg",
                reason_code="RECOVERY_PROPELLANT_RESERVE_DATA_MISSING",
            ))

    checks = kept + insertion_checks(run, target)
    constraints = [c for c in checks if c.id in CONSTRAINT_CHECK_IDS]
    return Verification(
        optimizer_converged=run.verification.optimizer_converged,
        physics_verified=run.verification.physics_verified,
        mission_constraints_passed=all(c.passed for c in constraints) if constraints else None,
        checks=tuple(checks),
    )


def failed_reason_codes(verification: Verification) -> tuple[str, ...]:
    """Every non-empty reason code of a failing check -- the honest, machine
    readable "why" of a red verdict (empty when all pass)."""
    return tuple(dict.fromkeys(
        c.reason_code for c in verification.checks if not c.passed and c.reason_code
    ))
