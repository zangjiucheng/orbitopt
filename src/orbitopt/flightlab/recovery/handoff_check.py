"""R-02 (task #234): cross-phase handoff continuity, mass conservation and
joint constraints for the boostback / entry / landing chain.

The three recovery segments are joined through the M-01 handoff contract
(:mod:`orbitopt.flightlab.handoff`): FL-302's rollout hands over at the
boostback cutoff (entry ignition follows after a ballistic coast), the
corridor-insertion state is the entry -> landing handoff FL-303 actually
restarts from, and the landing track ends in the touchdown record handed to
the ground. :func:`build_handoff_report` takes a *flown* chain (a
``ReturnPlan``, the optional ``LandingResult``, the separation state and the
vehicle) -- exactly what
:class:`orbitopt.flightlab.recovery.pipeline.RecoveryOutcome` carries -- and
grades it with three families of contract checks:

*continuity* (``handoff-*``, reason ``HANDOFF_STATE_DISCONTINUITY``): the
state the consumer restarted its segment from must equal the state the
producer handed over -- separation vs the first flown sample, the
boostback-cutoff event vs the sampled track at that instant, the corridor
state vs the landing burn's first sample, the touchdown record vs the final
sample. A planner that silently re-derived a segment's IC fails here, not
in a wrong plot downstream.

*mass conservation* (``handoff-mass-*``, reason ``HANDOFF_MASS_NOT_CONSERVED``):
mass never increases along a flown track, the coast between boostback cutoff
and entry ignition changes mass by zero, and the chain ends at or above the
vehicle's dry mass (a burn that "created" propellant is a bookkeeping bug,
not a green check).

*joint cross-phase constraints* (``joint-*``): limits assessed across the
whole chain rather than inside one segment -- touchdown point and speed, the
propellant budget at the corridor (what remains must cover the landing need
plus the reserve, measured from the *handed-over* state), the reserve at
touchdown, and peak dynamic pressure per phase window. Each check id names
its phase and the report maps every check (passing or failing) to the phase
that owns it, so a violation in the report says where it happened, e.g.
``joint-max-q-entry -> ENTRY_Q_LIMIT_EXCEEDED [entry]``.

Checks the flown chain cannot answer (no burn flown, corridor never crossed,
no touchdown) are reported as *unassessed* with a reason -- never silently
dropped, never green (a Check with no value is not a pass, per the contract
rule).

    from orbitopt.flightlab.recovery.handoff_check import build_handoff_report
    report = build_handoff_report(sep_ecef, plan, landing, booster,
                                  recovery_target=target_site)
    report.passed, report.violations, report.to_dict()

``fly_recovery(..., plan_options={"handoff_check": True})`` merges the
report's checks into the combined run's verification and records the report
summary as a manifest assumption; the report object itself is always on the
returned :class:`~orbitopt.flightlab.recovery.pipeline.RecoveryOutcome`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from orbitopt.flightlab.contract import Check
from orbitopt.flightlab.handoff import (
    PHASE_BOOSTBACK,
    PHASE_ENTRY,
    PHASE_LANDING,
    PHASE_SEPARATION,
    PHASE_TOUCHDOWN,
    HandoffState,
)
from orbitopt.flightlab.recovery.boostback import (
    DEFAULT_ENTRY_G_MAX,
    DEFAULT_ENTRY_Q_MAX_PA,
    INSUFFICIENT_PROPELLANT_RESERVE,
    RecoveryTarget,
    ReturnPlan,
    _enu_basis,
    _geodetic_fast,
    landing_burn_propellant_need_kg,
)
from orbitopt.flightlab.recovery.landing import (
    DEFAULT_LANDING_LIMITS,
    LANDING_LATERAL_SPEED_EXCEEDED,
    LANDING_POSITION_OUT_OF_TOLERANCE,
    LANDING_VERTICAL_SPEED_EXCEEDED,
    LandingLimits,
    LandingResult,
)
from orbitopt.flightlab.recovery.propagator import (
    BoosterModel,
    BoosterState,
    BoosterTrack,
    _to_ecef_state,
)

__all__ = [
    "HANDOFF_MASS_NOT_CONSERVED",
    "HANDOFF_STATE_DISCONTINUITY",
    "JointConstraints",
    "RecoveryHandoffReport",
    "UnassessedCheck",
    "build_handoff_report",
    "recovery_handoff_states",
]

# ---------------------------------------------------------------------------
# reason codes (registered in orbitopt.flightlab.reasons)
# ---------------------------------------------------------------------------

#: A segment restarted from a state that is not the one its producer handed
#: over (position/velocity/mass jump beyond the boundary tolerance).
HANDOFF_STATE_DISCONTINUITY = "HANDOFF_STATE_DISCONTINUITY"
#: Mass bookkeeping broke between two artifacts of the same chain: mass
#: rose somewhere on a flown track, the coast between the burns changed it,
#: or the chain ended below the vehicle's dry mass.
HANDOFF_MASS_NOT_CONSERVED = "HANDOFF_MASS_NOT_CONSERVED"

# ---------------------------------------------------------------------------
# boundary tolerances (Gate G5, declared here, never loosened per run)
# ---------------------------------------------------------------------------

#: Producer-vs-consumer position match at a same-instant boundary [m]: the
#: states are float64 copies of the same propagate IC / event bisection, so
#: a metre-scale mismatch means a real re-derivation, not rounding.
CONTINUITY_POS_TOL_M = 1.0
#: ... and velocity [m/s], mass [kg].
CONTINUITY_VEL_TOL_M_S = 0.05
CONTINUITY_MASS_TOL_KG = 1.0
#: Event-vs-sampled-track match at the boostback cutoff [m], [m/s]: the
#: event state is bisected between 2 s-grid samples, so linear-interpolation
#: error of the accelerating arc is budgeted here (throttle-full burn, a few
#: tens of m and ~a*h^2/8 m/s of arc-chord error over one step). Mass is
#: *not* interpolated: a throttle ramp makes the mass curve kink around the
#: cutoff, and the coast-mass check compares the exact event records anyway.
CUTOFF_INTERP_POS_TOL_M = 50.0
CUTOFF_INTERP_VEL_TOL_M_S = 30.0
#: Coast mass change [kg]: no engine runs between the burns, so this is
#: numerically zero; the tolerance only absorbs float accumulation.
COAST_MASS_TOL_KG = 0.5
#: Max allowed mass *increase* between consecutive samples [kg].
MASS_MONOTONIC_TOL_KG = 1e-6


# --------------------------------------------------------------------------
# joint (cross-phase) constraint set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JointConstraints:
    """The cross-phase budget the chain is graded against: touchdown point
    and speeds, the whole-chain dynamic-pressure ceiling, and the propellant
    reserve the chain must end with. Values are maxima/minima over the
    *entire* recovery, not one segment's; segment-internal limits (the
    landing burn's tilt/g limits, the corridor geometry) keep their owners'
    verdicts -- this is the joint view the pipeline reports together."""

    touchdown_position_tol_m: float = DEFAULT_LANDING_LIMITS.position_tol_m
    max_touchdown_vertical_speed_m_s: float = DEFAULT_LANDING_LIMITS.vertical_speed_max_m_s
    max_touchdown_lateral_speed_m_s: float = DEFAULT_LANDING_LIMITS.lateral_speed_max_m_s
    max_dynamic_pressure_pa: float = DEFAULT_ENTRY_Q_MAX_PA
    min_propellant_reserve_kg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("touchdown_position_tol_m", "max_touchdown_vertical_speed_m_s",
                     "max_touchdown_lateral_speed_m_s", "max_dynamic_pressure_pa"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"JointConstraints.{name} must be positive, got {v}")
        if self.min_propellant_reserve_kg < 0.0:
            raise ValueError("min_propellant_reserve_kg cannot be negative")

    @classmethod
    def for_recovery(cls, limits: LandingLimits | None = None, *,
                     entry_q_max_pa: float = DEFAULT_ENTRY_Q_MAX_PA,
                     min_propellant_reserve_kg: float | None = None) -> "JointConstraints":
        """Derive the joint set from the limit objects the pipeline already
        carries: touchdown point/speeds from the FL-303 landing limits
        (target-declared values take effect through
        ``pipeline._landing_limits_for_target``), the chain q ceiling from
        the entry-phase rating (the recovery's hottest aerodynamic phase;
        the ascent max-q limit does not apply to the booster's descent),
        the reserve floor from the target's declared recovery reserve
        (0 = vehicle reserve only, graded through the corridor budget)."""
        lim = limits if limits is not None else DEFAULT_LANDING_LIMITS
        return cls(
            touchdown_position_tol_m=float(lim.position_tol_m),
            max_touchdown_vertical_speed_m_s=float(lim.vertical_speed_max_m_s),
            max_touchdown_lateral_speed_m_s=float(lim.lateral_speed_max_m_s),
            max_dynamic_pressure_pa=float(entry_q_max_pa),
            min_propellant_reserve_kg=float(min_propellant_reserve_kg or 0.0),
        )


@dataclass(frozen=True)
class UnassessedCheck:
    """A contract check the flown chain cannot answer -- reported, never
    dropped and never green."""

    id: str
    phase: str
    note: str

    def to_dict(self) -> dict:
        return {"id": self.id, "phase": self.phase, "note": self.note}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryHandoffReport:
    """The R-02 verdict on one flown recovery chain: the M-01 handoff states
    that exist, the continuity / mass-conservation / joint-constraint checks
    that could be graded, the ones that could not (with reasons), and the
    phase every check is attributed to. ``to_dict()`` carries the phase
    annotation per check -- the acceptance "violations name their phase"."""

    states: tuple[HandoffState, ...]
    checks: tuple[Check, ...]
    unassessed: tuple[UnassessedCheck, ...]
    constraints: JointConstraints
    phases: Mapping[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def violations(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def phase_of(self, check_id: str) -> str:
        """The recovery phase a check (graded or unassessed) is attributed
        to; ``"chain"`` for anything the report did not attribute."""
        return self.phases.get(check_id, "chain")

    def by_phase(self) -> dict[str, tuple[Check, ...]]:
        out: dict[str, list[Check]] = {}
        for c in self.checks:
            out.setdefault(self.phase_of(c.id), []).append(c)
        return {k: tuple(v) for k, v in out.items()}

    def to_dict(self) -> dict:
        return {
            "kind": "orbitopt.flightlab.recovery-handoff-report",
            "schema_version": "1.0",
            "passed": bool(self.passed),
            "constraints": {
                "touchdown_position_tol_m": self.constraints.touchdown_position_tol_m,
                "max_touchdown_vertical_speed_m_s": self.constraints.max_touchdown_vertical_speed_m_s,
                "max_touchdown_lateral_speed_m_s": self.constraints.max_touchdown_lateral_speed_m_s,
                "max_dynamic_pressure_pa": self.constraints.max_dynamic_pressure_pa,
                "min_propellant_reserve_kg": self.constraints.min_propellant_reserve_kg,
            },
            "states": [s.to_dict() for s in self.states],
            "checks": [
                {"id": c.id, "phase": self.phase_of(c.id), "passed": bool(c.passed),
                 "value": c.value, "limit": c.limit, "unit": c.unit,
                 "reason_code": c.reason_code}
                for c in self.checks
            ],
            "unassessed": [u.to_dict() for u in self.unassessed],
        }

    def assumption_note(self) -> str:
        """One line for the run manifest: verdict, per-phase violations and
        the unassessed list -- the report header the acceptance names."""
        bad = "; ".join(f"{self.phase_of(c.id)}={c.reason_code or c.id}"
                        for c in self.violations) or "none"
        ua = ", ".join(u.id for u in self.unassessed) or "none"
        return (f"R-02 handoff report: {len(self.checks)} checks, "
                f"{'PASSED' if self.passed else 'FAILED'} "
                f"(violations by phase: {bad}; unassessed: {ua})")


# --------------------------------------------------------------------------
# boundary states (M-01 payloads of a flown chain)
# --------------------------------------------------------------------------


def _ecef(state: BoosterState) -> BoosterState:
    if state.frame == "ECEF":
        return state
    r, v = _to_ecef_state(state)
    return BoosterState(t_s=state.t_s, epoch_utc=state.epoch_utc, frame="ECEF",
                        position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v),
                        mass_kg=state.mass_kg)


def _event_state(ev: Any, epoch_utc: Any) -> BoosterState:
    return BoosterState(t_s=float(ev.t_s), epoch_utc=epoch_utc, frame="ECEF",
                        position_m=tuple(float(c) for c in ev.data["position_ecef_m"]),
                        velocity_m_s=tuple(float(c) for c in ev.data["velocity_ecef_m_s"]),
                        mass_kg=float(ev.data["mass_kg"]))


def _burn_events(track: BoosterTrack) -> tuple[list, list]:
    starts = [e for e in track.events if e.kind == "burn_start"]
    ends = [e for e in track.events if e.kind == "burn_end"]
    return starts, ends


def _final_ground_state(track: BoosterTrack) -> BoosterState | None:
    """The end-of-chain state handed to the ground: the bisected impact
    record when the track terminated on the ground, else None (a track that
    ended at its horizon never reached the touchdown boundary)."""
    if track.termination != "ground_impact" or track.impact is None:
        return None
    imp = track.impact
    return BoosterState(t_s=imp.t_s, epoch_utc=track.epoch_utc, frame="ECEF",
                        position_m=tuple(imp.position_ecef_m),
                        velocity_m_s=tuple(imp.velocity_ecef_m_s),
                        mass_kg=float(track.mass_kg[-1]))


def recovery_handoff_states(sep_state: BoosterState, plan: ReturnPlan,
                            landing: LandingResult | None,
                            booster: BoosterModel) -> tuple[HandoffState, ...]:
    """The M-01 boundary states a flown chain produced, in flight order.
    Boundaries the chain never reached (no boostback burn flown, corridor
    never crossed, no touchdown) are absent -- the report says so, with a
    reason; nothing is invented."""
    dry = float(booster.dry_mass_kg)
    sep = _ecef(sep_state)
    out: list[HandoffState] = [HandoffState.of(
        sep, from_phase=PHASE_SEPARATION, to_phase=PHASE_BOOSTBACK, dry_mass_kg=dry)]

    starts, ends = _burn_events(plan.track)
    if starts and ends and ends[0].t_s >= starts[0].t_s:
        out.append(HandoffState.of(
            _event_state(ends[0], plan.track.epoch_utc),
            from_phase=PHASE_BOOSTBACK, to_phase=PHASE_ENTRY,
            dry_mass_kg=dry, ignitions_used=1))
    if plan.landing_burn_initial_state is not None:
        out.append(HandoffState.of(
            plan.landing_burn_initial_state, from_phase=PHASE_ENTRY, to_phase=PHASE_LANDING,
            dry_mass_kg=dry,
            ignitions_used=2 if (starts and len(starts) > 1) else int(plan.ignitions_used)))
    final_track = landing.track if (landing is not None and landing.track is not None) \
        else plan.track
    ground = _final_ground_state(final_track)
    if ground is not None:
        out.append(HandoffState.of(
            ground, from_phase=PHASE_LANDING, to_phase=PHASE_TOUCHDOWN,
            dry_mass_kg=dry,
            ignitions_used=int(final_track.ignitions_used)))
    return tuple(out)


# --------------------------------------------------------------------------
# small track/event helpers
# --------------------------------------------------------------------------


def _interp_at(track: BoosterTrack, t: float) -> tuple[np.ndarray, np.ndarray, float]:
    t_arr = np.asarray(track.times_s, dtype=np.float64)
    pos = np.array([float(np.interp(t, t_arr, np.asarray(track.position_ecef_m)[:, i]))
                    for i in range(3)])
    vel = np.array([float(np.interp(t, t_arr, np.asarray(track.velocity_ecef_m_s)[:, i]))
                    for i in range(3)])
    return pos, vel, float(np.interp(t, t_arr, np.asarray(track.mass_kg, dtype=np.float64)))


def _window_max_q(track: BoosterTrack, t_lo: float, t_hi: float) -> float | None:
    t = np.asarray(track.times_s, dtype=np.float64)
    keep = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    if not bool(np.any(keep)):
        return None
    return float(np.max(np.asarray(track.dynamic_pressure_pa, dtype=np.float64)[keep]))


def _horizontal_miss_m(pad_ecef: np.ndarray, pos_ecef: np.ndarray) -> float:
    """Horizontal (ENU) distance from the pad to a point -- the impact-miss
    measure, comparable with the corridor radius and the pad tolerance."""
    lat, lon, _h = _geodetic_fast(pos_ecef)
    e, n, _u = _enu_basis(lat, lon)
    d = np.asarray(pad_ecef, dtype=np.float64) - np.asarray(pos_ecef, dtype=np.float64)
    return math.hypot(float(d @ np.array(e)), float(d @ np.array(n)))


def _track_mass_increase(track: BoosterTrack) -> float:
    """Largest mass rise between consecutive samples (0.0 when monotone)."""
    d = np.diff(np.asarray(track.mass_kg, dtype=np.float64))
    return float(max(0.0, float(np.max(d)))) if d.size else 0.0


# --------------------------------------------------------------------------
# the report builder
# --------------------------------------------------------------------------


def build_handoff_report(sep_state: BoosterState, plan: ReturnPlan,
                         landing: LandingResult | None, booster: BoosterModel, *,
                         recovery_target: RecoveryTarget | None = None,
                         limits: LandingLimits | None = None,
                         entry_q_max_pa: float = DEFAULT_ENTRY_Q_MAX_PA,
                         entry_g_max: float = DEFAULT_ENTRY_G_MAX,
                         min_propellant_reserve_kg: float | None = None,
                         constraints: JointConstraints | None = None,
                         ) -> RecoveryHandoffReport:
    """Grade a flown boostback / entry / landing chain against the M-01
    contract (module docstring). ``sep_state`` in any contract frame;
    ``plan``/``landing``/``booster`` are exactly what a
    ``RecoveryOutcome`` carries (``landing`` may be ``None`` -- the chain
    was not flown that far). ``recovery_target`` (the pad) enables the
    touchdown-point check for chains that never got a ``LandingResult``;
    without it the touchdown point is graded from the landing result alone.
    The q budget defaults to the entry-phase rating; pass a stricter
    :class:`JointConstraints` (or entry limits) to grade the same flight
    against a different budget. Never raises for a bad flight -- a bad
    flight produces red checks, not an exception."""
    del entry_g_max  # the g budget stays with its owners (corridor report, touchdown checks)
    cons = constraints if constraints is not None else JointConstraints.for_recovery(
        limits, entry_q_max_pa=entry_q_max_pa,
        min_propellant_reserve_kg=min_propellant_reserve_kg)
    states = recovery_handoff_states(sep_state, plan, landing, booster)
    checks: list[Check] = []
    unassessed: list[UnassessedCheck] = []
    phases: dict[str, str] = {}

    def graded(check: Check) -> Check:
        phases[check.id] = _REPORT_PHASES[check.id]
        checks.append(check)
        return check

    def missing(check_id: str, note: str) -> None:
        unassessed.append(UnassessedCheck(id=check_id,
                                          phase=_REPORT_PHASES.get(check_id, "chain"),
                                          note=note))

    by_boundary = {s.boundary: s for s in states}
    sep = by_boundary.get((PHASE_SEPARATION, PHASE_BOOSTBACK))
    cutoff = by_boundary.get((PHASE_BOOSTBACK, PHASE_ENTRY))
    corridor = by_boundary.get((PHASE_ENTRY, PHASE_LANDING))
    touchdown = by_boundary.get((PHASE_LANDING, PHASE_TOUCHDOWN))

    # ---- continuity 1: the consumer's IC is the separation handoff -------
    if sep is not None:
        p0 = np.asarray(plan.track.position_ecef_m[0], dtype=np.float64)
        v0 = np.asarray(plan.track.velocity_ecef_m_s[0], dtype=np.float64)
        dp = float(np.linalg.norm(p0 - np.asarray(sep.position_m)))
        dv = float(np.linalg.norm(v0 - np.asarray(sep.velocity_m_s)))
        dm = abs(float(plan.track.mass_kg[0]) - sep.mass_kg)
        worst = max(dp / CONTINUITY_POS_TOL_M, dv / CONTINUITY_VEL_TOL_M_S,
                    dm / CONTINUITY_MASS_TOL_KG)
        graded(Check(id="handoff-separation-boostback", passed=worst <= 1.0,
                     value=worst, limit=1.0, unit="ratio",
                     reason_code="" if worst <= 1.0 else HANDOFF_STATE_DISCONTINUITY))

    # ---- continuity 2: cutoff event vs the sampled track at that instant -
    if cutoff is not None:
        p_i, v_i, _m_i = _interp_at(plan.track, cutoff.t_s)
        dp = float(np.linalg.norm(p_i - np.asarray(cutoff.position_m)))
        dv = float(np.linalg.norm(v_i - np.asarray(cutoff.velocity_m_s)))
        worst = max(dp / CUTOFF_INTERP_POS_TOL_M, dv / CUTOFF_INTERP_VEL_TOL_M_S)
        graded(Check(id="handoff-boostback-entry", passed=worst <= 1.0,
                     value=worst, limit=1.0, unit="ratio",
                     reason_code="" if worst <= 1.0 else HANDOFF_STATE_DISCONTINUITY))
    else:
        missing("handoff-boostback-entry", "no boostback burn was flown "
                "(no cutoff event on the plan track)")

    # ---- continuity 3: the landing burn restarted from the corridor -----
    landing_track = landing.track if (landing is not None and landing.track is not None) else None
    final_track = landing_track if landing_track is not None else plan.track
    if corridor is not None and landing_track is not None:
        p0 = np.asarray(landing_track.position_ecef_m[0], dtype=np.float64)
        v0 = np.asarray(landing_track.velocity_ecef_m_s[0], dtype=np.float64)
        dp = float(np.linalg.norm(p0 - np.asarray(corridor.position_m)))
        dv = float(np.linalg.norm(v0 - np.asarray(corridor.velocity_m_s)))
        dm = abs(float(landing_track.mass_kg[0]) - corridor.mass_kg)
        worst = max(dp / CONTINUITY_POS_TOL_M, dv / CONTINUITY_VEL_TOL_M_S,
                    dm / CONTINUITY_MASS_TOL_KG)
        graded(Check(id="handoff-entry-landing", passed=worst <= 1.0,
                     value=worst, limit=1.0, unit="ratio",
                     reason_code="" if worst <= 1.0 else HANDOFF_STATE_DISCONTINUITY))
    else:
        missing("handoff-entry-landing", "corridor never crossed" if corridor is None
                else "landing burn never flew (no landing track)")

    # ---- continuity 4: the touchdown record is the track's own end -------
    if touchdown is not None:
        p_t = np.asarray(final_track.position_ecef_m[-1], dtype=np.float64)
        v_t = np.asarray(final_track.velocity_ecef_m_s[-1], dtype=np.float64)
        dp = float(np.linalg.norm(p_t - np.asarray(touchdown.position_m)))
        dv = float(np.linalg.norm(v_t - np.asarray(touchdown.velocity_m_s)))
        graded(Check(id="handoff-landing-touchdown", passed=dp <= 0.01 and dv <= 0.01,
                     value=max(dp, dv), limit=0.01, unit="m",
                     reason_code="" if dp <= 0.01 and dv <= 0.01
                     else HANDOFF_STATE_DISCONTINUITY))
    else:
        missing("handoff-landing-touchdown", "the chain never reached the ground "
                "inside the flown horizon (no touchdown record to hand over)")

    # ---- mass conservation ----------------------------------------------
    inc_bb = _track_mass_increase(plan.track)
    graded(Check(id="handoff-mass-monotonic-boostback",
                 passed=inc_bb <= MASS_MONOTONIC_TOL_KG,
                 value=inc_bb, limit=MASS_MONOTONIC_TOL_KG, unit="kg",
                 reason_code="" if inc_bb <= MASS_MONOTONIC_TOL_KG
                 else HANDOFF_MASS_NOT_CONSERVED))
    if landing_track is not None:
        inc_l = _track_mass_increase(landing_track)
        graded(Check(id="handoff-mass-monotonic-landing", passed=inc_l <= MASS_MONOTONIC_TOL_KG,
                     value=inc_l, limit=MASS_MONOTONIC_TOL_KG, unit="kg",
                     reason_code="" if inc_l <= MASS_MONOTONIC_TOL_KG
                     else HANDOFF_MASS_NOT_CONSERVED))
    else:
        missing("handoff-mass-monotonic-landing", "landing burn never flew")

    starts, _ends = _burn_events(plan.track)
    if len(starts) > 1 and cutoff is not None:
        dm = float(starts[1].data["mass_kg"]) - cutoff.mass_kg
        graded(Check(id="handoff-mass-coast-entry", passed=abs(dm) <= COAST_MASS_TOL_KG,
                     value=abs(dm), limit=COAST_MASS_TOL_KG, unit="kg",
                     reason_code="" if abs(dm) <= COAST_MASS_TOL_KG
                     else HANDOFF_MASS_NOT_CONSERVED))
    else:
        missing("handoff-mass-coast-entry", "entry burn never ignited (no second "
                "ignition to compare the coast mass against)")

    final_track = landing_track if landing_track is not None else plan.track
    prop_final = float(final_track.mass_kg[-1]) - float(booster.dry_mass_kg)
    graded(Check(id="handoff-mass-terminal", passed=prop_final >= -MASS_MONOTONIC_TOL_KG,
                 value=prop_final, limit=0.0, unit="kg",
                 reason_code="" if prop_final >= -MASS_MONOTONIC_TOL_KG
                 else HANDOFF_MASS_NOT_CONSERVED))

    # ---- joint constraints: dynamic pressure per phase window ------------
    if starts:
        bb_end_t = cutoff.t_s if cutoff is not None else float(plan.track.times_s[-1])
        q_bb = _window_max_q(plan.track, float(starts[0].t_s), bb_end_t)
        if q_bb is not None:
            graded(Check(id="joint-max-q-boostback",
                         passed=q_bb <= cons.max_dynamic_pressure_pa,
                         value=q_bb, limit=cons.max_dynamic_pressure_pa, unit="Pa",
                         reason_code="" if q_bb <= cons.max_dynamic_pressure_pa
                         else "MAX_Q_EXCEEDED"))
        else:
            missing("joint-max-q-boostback", "boostback window empty on the sampled track")
        t_corridor = corridor.t_s if corridor is not None else float(plan.track.times_s[-1])
        q_entry = _window_max_q(plan.track, bb_end_t, t_corridor)
        if q_entry is not None:
            graded(Check(id="joint-max-q-entry", passed=q_entry <= cons.max_dynamic_pressure_pa,
                         value=q_entry, limit=cons.max_dynamic_pressure_pa, unit="Pa",
                         reason_code="" if q_entry <= cons.max_dynamic_pressure_pa
                         else "ENTRY_Q_LIMIT_EXCEEDED"))
        else:
            missing("joint-max-q-entry", "entry descent window empty on the sampled track")
    else:
        missing("joint-max-q-boostback", "no boostback burn was flown")
        missing("joint-max-q-entry", "no boostback burn was flown -- the entry "
                "descent window is undefined")
    if landing_track is not None:
        q_land = _window_max_q(landing_track, float(landing_track.times_s[0]),
                               float(landing_track.times_s[-1]))
        if q_land is not None:
            graded(Check(id="joint-max-q-landing",
                         passed=q_land <= cons.max_dynamic_pressure_pa,
                         value=q_land, limit=cons.max_dynamic_pressure_pa, unit="Pa",
                         reason_code="" if q_land <= cons.max_dynamic_pressure_pa
                         else "MAX_Q_EXCEEDED"))

    # ---- joint constraints: the propellant budget across segments --------
    if corridor is not None:
        cr = plan.corridor_report
        reserve_floor = float(booster.landing_propellant_reserve_kg)
        if cr.propellant_available_kg is not None and cr.landing_need_kg is not None:
            avail, need = float(cr.propellant_available_kg), float(cr.landing_need_kg)
        else:  # measured numbers missing -- re-derive from the handed-over state
            avail = corridor.propellant_kg
            need = landing_burn_propellant_need_kg(
                corridor.mass_kg, math.hypot(*corridor.velocity_m_s),
                plan.corridor.h0_m, booster)
        margin = avail - need - max(reserve_floor, cons.min_propellant_reserve_kg)
        graded(Check(id="joint-propellant-corridor-budget", passed=margin >= 0.0,
                     value=margin, limit=0.0, unit="kg",
                     reason_code="" if margin >= 0.0 else INSUFFICIENT_PROPELLANT_RESERVE))
    else:
        missing("joint-propellant-corridor-budget", "corridor never crossed -- "
                "no handed-over state to budget the landing burn from")

    pad = (np.asarray(recovery_target.position_ecef_m, dtype=np.float64)
           if recovery_target is not None else None)
    touch_pos_m: float | None = None
    touch_v: float | None = None
    touch_h: float | None = None
    if touchdown is not None and landing is not None:
        touch_pos_m = landing.position_error_m
        touch_v = landing.vertical_speed_m_s
        touch_h = landing.lateral_speed_m_s
    elif touchdown is not None and pad is not None:
        # a ballistic chain that hit the ground: the joint landing point is
        # the horizontal impact miss, its speed the impact speed.
        touch_pos_m = _horizontal_miss_m(pad, np.asarray(touchdown.position_m))
        speed = math.hypot(*touchdown.velocity_m_s)
        up = np.asarray(touchdown.position_m, dtype=np.float64)
        touch_v = -abs(float(np.asarray(touchdown.velocity_m_s)
                             @ (up / float(np.linalg.norm(up)))))
        touch_h = math.sqrt(max(speed * speed - touch_v * touch_v, 0.0))
    else:
        missing("joint-touchdown-position", "no touchdown -- the chain never reached the ground")
        missing("joint-touchdown-vertical-speed", "no touchdown")
        missing("joint-touchdown-lateral-speed", "no touchdown")
    if touch_pos_m is not None:
        graded(Check(id="joint-touchdown-position",
                     passed=touch_pos_m <= cons.touchdown_position_tol_m,
                     value=float(touch_pos_m),
                     limit=cons.touchdown_position_tol_m, unit="m",
                     reason_code="" if touch_pos_m <= cons.touchdown_position_tol_m
                     else LANDING_POSITION_OUT_OF_TOLERANCE))
        graded(Check(id="joint-touchdown-vertical-speed",
                     passed=touch_v is not None
                     and abs(touch_v) <= cons.max_touchdown_vertical_speed_m_s,
                     value=None if touch_v is None else abs(float(touch_v)),
                     limit=cons.max_touchdown_vertical_speed_m_s, unit="m/s",
                     reason_code="" if touch_v is not None
                     and abs(touch_v) <= cons.max_touchdown_vertical_speed_m_s
                     else LANDING_VERTICAL_SPEED_EXCEEDED))
        graded(Check(id="joint-touchdown-lateral-speed",
                     passed=touch_h is not None
                     and touch_h <= cons.max_touchdown_lateral_speed_m_s,
                     value=None if touch_h is None else float(touch_h),
                     limit=cons.max_touchdown_lateral_speed_m_s, unit="m/s",
                     reason_code="" if touch_h is not None
                     and touch_h <= cons.max_touchdown_lateral_speed_m_s
                     else LANDING_LATERAL_SPEED_EXCEEDED))

    reserve_floor = max(float(booster.landing_propellant_reserve_kg),
                        cons.min_propellant_reserve_kg)
    graded(Check(id="joint-propellant-reserve-touchdown",
                 passed=prop_final >= reserve_floor,
                 value=prop_final, limit=reserve_floor, unit="kg",
                 reason_code="" if prop_final >= reserve_floor else "PROPELLANT_RESERVE_LOW"))

    return RecoveryHandoffReport(states=states, checks=tuple(checks),
                                 unassessed=tuple(unassessed), constraints=cons,
                                 phases=phases)


#: Phase attribution of every check id the report can emit (the acceptance
#: "name the phase a violation belongs to").
_REPORT_PHASES: dict[str, str] = {
    "handoff-separation-boostback": PHASE_BOOSTBACK,
    "handoff-boostback-entry": PHASE_ENTRY,
    "handoff-entry-landing": PHASE_LANDING,
    "handoff-landing-touchdown": PHASE_LANDING,
    "handoff-mass-monotonic-boostback": PHASE_BOOSTBACK,
    "handoff-mass-monotonic-landing": PHASE_LANDING,
    "handoff-mass-coast-entry": PHASE_ENTRY,
    "handoff-mass-terminal": PHASE_LANDING,
    "joint-max-q-boostback": PHASE_BOOSTBACK,
    "joint-max-q-entry": PHASE_ENTRY,
    "joint-max-q-landing": PHASE_LANDING,
    "joint-propellant-corridor-budget": PHASE_ENTRY,
    "joint-touchdown-position": PHASE_LANDING,
    "joint-touchdown-vertical-speed": PHASE_LANDING,
    "joint-touchdown-lateral-speed": PHASE_LANDING,
    "joint-propellant-reserve-touchdown": PHASE_LANDING,
}
