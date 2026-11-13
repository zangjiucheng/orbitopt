"""FL-304: the recovery pipeline merged into a Gate-B run.

:func:`fly_recovery` takes a flown ascent run (whatever produced it -- the
OpenTsiolkovsky adapter, a synthetic test fixture, a Monte-Carlo sample),
reads its first-stage separation state (FL-301
:func:`~orbitopt.flightlab.recovery.propagator.separation_state_from_run`),
flies the full E3 recovery chain -- FL-302 boostback + entry burn, then the
FL-303 landing burn flown by FL-305's closed-loop terminal guidance (the
single-plan G-FOLD flow on request), both integrated through the FL-301
propagator -- and
returns a *combined* :class:`~orbitopt.flightlab.contract.FlightRun` whose
``stage1`` body is recovered: the ballistic ``stage1_debris`` track is replaced
by the flown ``booster`` track (separation -> boostback -> entry -> landing
burn -> touchdown), the booster events are appended, and checks covering the
landing are added to the verification so that ``mission_constraints_passed``
(once :func:`orbitopt.flightlab.checks.evaluate` has re-assessed the run)
means "insertion *and* landing inside the limits".

The verdict stays honest in both directions: a plan the physics cannot fly
returns a run whose ``recovery-landing`` gate check fails with the governing
FL-302/FL-303 reason code -- the pipeline never reports a landing it did not
fly, and it never fudges a corridor/entry/limit violation into a pass.
Deterministic by construction: every input is honoured, nothing wall-clock
enters the run, so the same inputs write a byte-identical run directory
(FL-304's determinism acceptance criterion).

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.recovery.pipeline import fly_recovery
    outcome = fly_recovery(ascent_run, load_target("scenario.yaml"),
                           "vehicle.yaml")
    print(outcome.success, outcome.reason_codes, outcome.run.verification)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab.contract import (
    Check,
    Event,
    FlightRun,
    FlightTarget,
    Verification,
)
from orbitopt.flightlab.recovery.boostback import (
    DEFAULT_ENTRY_Q_MAX_PA,
    RecoveryTarget,
    ReturnPlan,
    _enu_basis,
    _geodetic_fast,
    plan_boostback_entry,
)
from orbitopt.flightlab.recovery.handoff_check import (
    RecoveryHandoffReport,
    build_handoff_report,
)
from orbitopt.flightlab.recovery.landing import (
    DEFAULT_LANDING_LIMITS,
    GFOLD_VERSION_PIN,
    GuidanceModelMismatch,
    LandingLimits,
    LandingResult,
    LandingSolverUnavailableError,
    TerminalGuidance,
    fly_landing_burn,
    fly_landing_burn_closed_loop,
    landing_events,
)
from orbitopt.flightlab.recovery.propagator import (
    BoosterModel,
    BoosterPhysics,
    BoosterState,
    BoosterTrack,
    RecoveryError,
    _to_ecef_state,
    separation_state_from_run,
)

__all__ = ["RecoveryOutcome", "fly_recovery", "RECOVERY_GATE_CHECK_ID",
           "RECOVERY_NOT_ASSESSED", "dispersed_separation_state",
           "separation_state_ecef"]

#: Verification check id of the recovery gate: pass = the booster was flown
#: and touched down inside every landing limit; fail = reason_code carries
#: the governing FL-302/FL-303 reason (or the two codes below).
RECOVERY_GATE_CHECK_ID = "recovery-landing"
RECOVERY_NOT_ASSESSED = "RECOVERY_LANDING_NOT_ASSESSED"
RECOVERY_LANDING_NOT_ATTEMPTED = "RECOVERY_LANDING_NOT_ATTEMPTED"

_BOOSTER_BODY_ID = "booster"
_DEBRIS_BODY_ID = "stage1_debris"
_RECOVERY_ASSUMPTION = (
    "FL-304 recovery: FL-302 boostback/entry script + FL-303 landing burn "
    "(FL-305 closed-loop guidance unless plan_options select the single "
    "G-FOLD plan), integrated through the FL-301 propagator; the booster "
    "track replaces the ballistic stage1_debris track"
)

#: Warm start pinning FL-302's accepted script for the packaged synthetic
#: vehicle at its nominal RTLS separation state (delay s, duration s,
#: throttle, pitch deg, yaw deg, trigger alt m, entry duration s). Passing it
#: as ``initial_guess`` with a small ``max_fevals`` keeps a Monte-Carlo
#: re-plan a local refinement of the known-good script, not a cold search.
#: FL-305 re-pinned the boostback duration 19.1 -> 17.75 s (the gentler
#: handoff of ADR flightlab-0002): the FL-304 script overshot the pad and
#: handed the landing burn a 1.76 km divert at 41.1 t; this one hands over
#: 0.35 km out at 41.7 t, inside the landing burn's divert budget. FL-302's
#: own penalty barely sees a 1.76 km miss against its 8 km corridor radius,
#: and a short Nelder-Mead budget cannot step the 0.5 s-quantized burn.
NOMINAL_PLAN_GUESS = (18.4, 17.75, 1.0, 4.36, 0.0, 17456.0, 23.0)

#: ``plan_options["landing"]["mode"]`` values: FL-305's closed loop (the
#: default) or FL-303's single open-loop plan.
LANDING_MODES = ("closed-loop", "single")


@dataclass(frozen=True)
class RecoveryOutcome:
    """One flown recovery: the combined run, the two stage results (either
    ``None`` before it ran), the checks the pipeline contributed, the
    governing reason codes of a failure, and the honest booleans."""

    run: FlightRun
    success: bool
    reason_codes: tuple[str, ...]
    plan: ReturnPlan | None = None
    landing: LandingResult | None = None
    checks: tuple[Check, ...] = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    #: The R-02 handoff / joint-constraint report for the flown chain
    #: (:mod:`orbitopt.flightlab.recovery.handoff_check`) -- always built,
    #: merged into ``run.verification.checks`` only when
    #: ``plan_options["handoff_check"]`` is set.
    handoff: RecoveryHandoffReport | None = None


def _landing_limits_for_target(target: FlightTarget) -> LandingLimits:
    """FL-303 acceptance from the target's declared touchdown limits; the
    undeclared ones keep :data:`DEFAULT_LANDING_LIMITS`."""
    defaults = DEFAULT_LANDING_LIMITS
    vert = target.recovery.max_touchdown_speed_m_s
    lat = target.recovery.max_touchdown_lateral_speed_m_s
    return LandingLimits(
        position_tol_m=defaults.position_tol_m,
        vertical_speed_max_m_s=float(vert) if vert is not None
        else defaults.vertical_speed_max_m_s,
        lateral_speed_max_m_s=float(lat) if lat is not None
        else defaults.lateral_speed_max_m_s,
        max_tilt_deg=defaults.max_tilt_deg,
        max_proper_g=defaults.max_proper_g,
    )


def _cut_at(track: BoosterTrack, t_cut_s: float) -> BoosterTrack:
    """The track truncated at ``t_cut_s`` (inclusive): the samples before the
    cut, then the exact cut-time row rebuilt from the samples around it. Used
    on the FL-302 rollout so the corridor state is the join point, not a
    ballistic segment that kept flying past the burn handover."""
    t = np.asarray(track.times_s, dtype=np.float64)
    keep = t <= t_cut_s + 1e-9
    if bool(keep[-1]):
        return track
    n = int(np.count_nonzero(keep))
    interp = lambda a: float(np.interp(t_cut_s, t, np.asarray(a, dtype=np.float64)))  # noqa: E731
    return BoosterTrack(
        epoch_utc=track.epoch_utc,
        times_s=np.concatenate((t[:n], [t_cut_s])),
        mass_kg=np.concatenate((track.mass_kg[:n], [interp(track.mass_kg)])),
        position_ecef_m=np.vstack((track.position_ecef_m[:n],
                                   [np.interp(t_cut_s, t, track.position_ecef_m[:, i])
                                    for i in range(3)])),
        velocity_ecef_m_s=np.vstack((track.velocity_ecef_m_s[:n],
                                     [np.interp(t_cut_s, t, track.velocity_ecef_m_s[:, i])
                                      for i in range(3)])),
        height_m_arr=np.concatenate((track.height_m_arr[:n], [interp(track.height_m_arr)])),
        throttle=np.concatenate((track.throttle[:n], [interp(track.throttle)])),
        mach=np.concatenate((track.mach[:n], [interp(track.mach)])),
        dynamic_pressure_pa=np.concatenate(
            (track.dynamic_pressure_pa[:n], [interp(track.dynamic_pressure_pa)])),
        events=tuple(ev for ev in track.events if ev.t_s <= t_cut_s + 1e-9),
        impact=None,
        termination="time_limit",
        max_proper_g=float(track.max_proper_g),
        max_dynamic_pressure_pa=float(np.max(np.asarray(track.dynamic_pressure_pa[:n + 1]))),
        ignitions_used=track.ignitions_used,
        dry_mass_kg=track.dry_mass_kg,
        landing_reserve=track.landing_reserve,
    )


def _stack_tracks(first: BoosterTrack, second: BoosterTrack) -> BoosterTrack:
    """Concatenate two contiguous booster tracks (``second`` starts where
    ``first`` ends -- the corridor handover): shared boundary sample dropped
    from ``second``, events and scalars merged, ``second``'s termination and
    impact reported (the end of the flown story is the run's end)."""
    t1, t2 = first.times_s, second.times_s
    drop = 1 if len(t2) > 1 and abs(float(t2[0]) - float(t1[-1])) <= 1e-6 else 0
    cat = lambda a, b: np.concatenate((np.asarray(a), np.asarray(b)[drop:]))  # noqa: E731
    stack = lambda a, b: np.vstack((np.asarray(a), np.asarray(b)[drop:, :]))  # noqa: E731
    return BoosterTrack(
        epoch_utc=first.epoch_utc,
        times_s=cat(t1, t2),
        mass_kg=cat(first.mass_kg, second.mass_kg),
        position_ecef_m=stack(first.position_ecef_m, second.position_ecef_m),
        velocity_ecef_m_s=stack(first.velocity_ecef_m_s, second.velocity_ecef_m_s),
        height_m_arr=cat(first.height_m_arr, second.height_m_arr),
        throttle=cat(first.throttle, second.throttle),
        mach=cat(first.mach, second.mach),
        dynamic_pressure_pa=cat(first.dynamic_pressure_pa, second.dynamic_pressure_pa),
        events=tuple(first.events) + tuple(second.events),
        impact=second.impact if second.impact is not None else first.impact,
        termination=second.termination,
        max_proper_g=max(float(first.max_proper_g), float(second.max_proper_g)),
        max_dynamic_pressure_pa=max(float(first.max_dynamic_pressure_pa),
                                    float(second.max_dynamic_pressure_pa)),
        ignitions_used=int(first.ignitions_used) + int(second.ignitions_used),
        dry_mass_kg=first.dry_mass_kg or second.dry_mass_kg,
        landing_reserve=first.landing_reserve,
    )


def _booster_track(plan: ReturnPlan, landing: LandingResult | None) -> BoosterTrack:
    """The full flown booster track: FL-302 rollout (truncated at the
    corridor crossing when the landing burn was handed over) + the FL-303
    re-integration when it ran."""
    crossed = plan.corridor_report.reached and plan.corridor_report.t_s is not None
    first = _cut_at(plan.track, float(plan.corridor_report.t_s)) if crossed else plan.track
    if landing is not None and landing.track is not None:
        return _stack_tracks(first, landing.track)
    return first


def _launch_site_deg(ascent_run: FlightRun) -> tuple[float, float] | None:
    """(lat, lon) of the ascent's first sample, from the earliest track's
    geodetic columns (OpenTsiolkovsky writes them): the origin its
    ``downrange_m`` is measured from. None when no track carries them."""
    for track in sorted(ascent_run.tracks, key=lambda t: t.t_start_s):
        cols = {k.lower(): v for k, v in track.extra_columns.items()}
        lat, lon = cols.get("latitude_deg"), cols.get("longitude_deg")
        if lat and lon:
            return float(lat[0]), float(lon[0])
    return None


def _merged_run(ascent_run: FlightRun, booster_track: BoosterTrack,
                booster_events: Sequence[Event], extra_checks: Sequence[Check],
                assumptions: Sequence[str], engines: Mapping[str, str]) -> FlightRun:
    """The ascent run with the first stage recovered: ``stage1_debris`` and
    its events replaced by the flown booster body track and the recovery
    events, the recovery checks appended, the provenance/assumptions grown.
    The three verification booleans keep their ascent-run values --
    :func:`orbitopt.flightlab.checks.evaluate` re-assesses them from the
    checks, and must run after this merge for the landing to count."""
    tracks = tuple(t for t in ascent_run.tracks if t.body_id != _DEBRIS_BODY_ID)
    events = [e for e in ascent_run.events if e.body_id != _DEBRIS_BODY_ID]
    events.extend(booster_events)
    events.sort(key=lambda e: (float(e.t_s), e.body_id, e.kind))
    return replace(
        ascent_run,
        tracks=tracks + (booster_track.to_body_track(
            "GCRS", body_id=_BOOSTER_BODY_ID, parent_body_id="stage1",
            downrange_origin_deg=_launch_site_deg(ascent_run)),),
        events=tuple(events),
        engines={**ascent_run.engines, **dict(engines)},
        assumptions=tuple(ascent_run.assumptions) + tuple(assumptions),
        verification=Verification(
            optimizer_converged=ascent_run.verification.optimizer_converged,
            physics_verified=ascent_run.verification.physics_verified,
            mission_constraints_passed=ascent_run.verification.mission_constraints_passed,
            checks=tuple(ascent_run.verification.checks) + tuple(extra_checks),
        ),
    )


def fly_recovery(ascent_run: FlightRun, target: FlightTarget, vehicle_path, *,
                 sep_state: BoosterState | None = None,
                 booster: BoosterModel | None = None,
                 limits: LandingLimits | None = None,
                 physics: BoosterPhysics | None = None,
                 plan_options: Mapping[str, Any] | None = None,
                 landing_sites: Mapping[str, Mapping[str, Any]] | None = None,
                 ) -> RecoveryOutcome:
    """Fly the E3 recovery chain for ``ascent_run`` and return the combined
    run (module docstring). ``sep_state`` overrides the read-out from the
    run's ``stage1_debris`` track (FL-304 Monte Carlo injects its dispersed
    separation states through here); ``booster``/``limits``/``physics``/
    ``plan_options`` pass straight through to FL-302/FL-303 (``plan_options``
    keys: ``max_fevals``, ``initial_guess``, ``horizon_s``, ``dt_opt``,
    ``dt_final``, entry limits, the R-01 objective knobs
    ``landing_divert_budget_m`` / ``landing_reserve_margin_kg``, and
    ``handoff_check`` (merge the R-02 handoff / joint-constraint checks into
    the run's verification) plus one landing key, ``landing``: a mapping
    with ``mode`` (:data:`LANDING_MODES`, default ``"closed-loop"``), any
    :class:`~orbitopt.flightlab.recovery.landing.TerminalGuidance` fields,
    and R-03's ``mismatch``
    (:class:`~orbitopt.flightlab.recovery.landing.GuidanceModelMismatch` or
    its fields as a mapping; zero/absent = the plain FL-305 flight).
    Raises ``ValueError`` when the target declares ``recovery: none``, the
    run carries no separation state, or the landing options are unknown."""
    if target.recovery.kind == "none":
        raise ValueError("target declares recovery: none -- nothing to fly")
    booster = booster if booster is not None else BoosterModel.load(vehicle_path)
    recovery_target = RecoveryTarget.from_scenario(target.to_dict(),
                                                   landing_sites=landing_sites)
    if sep_state is None:
        sep_state = separation_state_from_run(ascent_run)

    opts = dict(plan_options) if plan_options else {}
    fly_landing = _landing_flight(opts.pop("landing", None))
    # R-02: merge the handoff / joint-constraint checks into the run only on
    # request -- Gate B reference runs compare verification byte-for-byte,
    # so the extra checks are opt-in and the report object is the default
    # surface (outcome.handoff).
    handoff_in_run = bool(opts.pop("handoff_check", False))
    plan = plan_boostback_entry(sep_state, booster, recovery_target, **opts)
    landing: LandingResult | None = None
    if plan.feasible:
        try:
            landing = fly_landing(
                plan.landing_burn_initial_state_enu,
                plan.landing_burn_initial_state,
                recovery_target, booster,
                limits=limits if limits is not None
                else _landing_limits_for_target(target),
                physics=physics,
            )
        except LandingSolverUnavailableError as exc:
            # an environment gap (gfold missing / off-pin), not a physics
            # verdict: the landing was not assessed -- never LANDING_INFEASIBLE
            landing = LandingResult(feasible=False, landed=False,
                                    reason_codes=(RECOVERY_NOT_ASSESSED,),
                                    checks=(), assumptions=(
                                        f"landing burn not assessed: {exc}",))
        except RecoveryError as exc:
            landing = LandingResult(feasible=False, landed=False,
                                    reason_codes=("LANDING_INFEASIBLE",),
                                    checks=(), assumptions=(str(exc),))

    if landing is not None and landing.landed:
        success, gate_reason = True, ""
    elif not plan.feasible:
        success, gate_reason = False, plan.reason_codes[0] if plan.reason_codes else "RECOVERY_INFEASIBLE"
    elif landing is None:
        success, gate_reason = False, RECOVERY_LANDING_NOT_ATTEMPTED
    else:
        success, gate_reason = False, landing.reason_codes[0] if landing.reason_codes else "LANDING_FAILED"

    checks: list[Check] = [Check(id=RECOVERY_GATE_CHECK_ID, passed=success,
                                 reason_code=gate_reason)]
    if landing is not None:
        checks.extend(landing.checks)
    # The target-declared recovery reserve, measured from the booster track
    # that actually flew (landing re-integration when it ran, else the
    # FL-302 rollout to its end) -- not from the ballistic debris estimate.
    track = _booster_track(plan, landing)
    reserve_limit = target.recovery.min_propellant_reserve_kg
    if reserve_limit is not None:
        remaining = track.propellant_remaining_kg
        checks.append(Check(
            id="recovery-propellant-reserve", passed=remaining >= float(reserve_limit),
            value=remaining, limit=float(reserve_limit), unit="kg",
            reason_code="" if remaining >= float(reserve_limit)
            else "PROPELLANT_RESERVE_LOW",
        ))

    booster_events = list(track.to_contract_events(_BOOSTER_BODY_ID))
    if landing is not None:
        booster_events.extend(landing_events(landing, body_id=_BOOSTER_BODY_ID))
    # R-02 (task #234): grade the flown chain against the M-01 handoff
    # contract -- continuity, mass conservation, joint cross-phase limits,
    # every check attributed to the phase that owns it.
    eff_limits = limits if limits is not None else _landing_limits_for_target(target)
    handoff = build_handoff_report(
        sep_state, plan, landing, booster,
        recovery_target=recovery_target, limits=eff_limits,
        entry_q_max_pa=float(opts.get("entry_q_max_pa", DEFAULT_ENTRY_Q_MAX_PA)),
        min_propellant_reserve_kg=target.recovery.min_propellant_reserve_kg)
    if handoff_in_run:
        checks.extend(handoff.checks)
    engines = {"propagator": "orbitopt.flightlab.recovery.propagator",
               "cvxpy": "reference",
               "gfold": (GFOLD_VERSION_PIN if landing is not None
                         and landing.plan is not None
                         and landing.plan.solver == "gfold" else "not-run")}
    assumptions = [_RECOVERY_ASSUMPTION, *plan.assumptions,
                   *(() if landing is None else landing.assumptions),
                   *([handoff.assumption_note()] if handoff_in_run else [])]
    run = _merged_run(ascent_run, track, booster_events, checks, assumptions, engines)
    reason_codes: tuple[str, ...] = () if success else (gate_reason,)
    return RecoveryOutcome(run=run, success=success, reason_codes=reason_codes,
                           plan=plan, landing=landing, checks=tuple(checks),
                           assumptions=tuple(assumptions), handoff=handoff)


def _landing_flight(landing_options: Mapping[str, Any] | None):
    """The landing-burn step for ``plan_options["landing"]``: FL-305's
    :func:`fly_landing_burn_closed_loop` with the given guidance fields, or
    FL-303's :func:`fly_landing_burn` for ``mode="single"``. Validated
    before FL-302 runs, so a typo fails fast instead of after the optimiser.
    R-03: a ``mismatch`` key (a
    :class:`~orbitopt.flightlab.recovery.landing.GuidanceModelMismatch` or a
    mapping of its fields) biases the guidance's model against the truth
    propagator; absent/zero is the plain FL-305 flight."""
    opts = dict(landing_options) if landing_options else {}
    mode = str(opts.pop("mode", "closed-loop"))
    if mode not in LANDING_MODES:
        raise ValueError(f"landing mode must be one of {LANDING_MODES}, got {mode!r}")
    if mode == "single":
        if opts:
            raise ValueError(f"landing mode 'single' takes no guidance options, got {sorted(opts)}")
        return fly_landing_burn
    mismatch = GuidanceModelMismatch.from_option(opts.pop("mismatch", None))
    guidance = TerminalGuidance(**opts)  # TypeError on an unknown field

    def fly(*args, **kwargs):
        return fly_landing_burn_closed_loop(*args, guidance=guidance,
                                            mismatch=mismatch, **kwargs)
    return fly


def separation_state_ecef(ascent_run: FlightRun) -> BoosterState:
    """The run's separation state in ECEF, whatever frame the debris track
    recorded it in (same conversion FL-302 applies internally). The Monte
    Carlo disperses in the geodetic ENU basis of the state, which is an
    ECEF-frame statement, so MC samples start here."""
    sep = separation_state_from_run(ascent_run)
    if sep.frame == "ECEF":
        return sep
    r_e, v_e = _to_ecef_state(sep)
    return replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r_e),
                   velocity_m_s=tuple(float(c) for c in v_e))


def dispersed_separation_state(sep_state: BoosterState, *,
                               altitude_m: float = 0.0, downrange_m: float = 0.0,
                               crossrange_m: float = 0.0, speed_m_s: float = 0.0,
                               heading_deg: float = 0.0) -> BoosterState:
    """A separation state shifted in the pad-free local ENU frame of its own
    position: ``altitude_m`` along up, ``downrange_m``/``crossrange_m`` along
    the horizontal velocity direction and its right-hand normal, a speed bias
    along the velocity vector, and a heading rotation of the horizontal
    velocity (deg, positive to the right). FL-304 Monte-Carlo separation
    dispersion; exact for the small sigmas dispersions use."""
    r = np.asarray(sep_state.position_m, dtype=np.float64)
    v = np.asarray(sep_state.velocity_m_s, dtype=np.float64)
    if float(np.dot(r, v)) == 0.0 and float(np.linalg.norm(v)) == 0.0:
        raise ValueError("dispersed_separation_state needs a nonzero velocity")
    lat, lon, _ = _geodetic_fast(r)
    e, n, u = (np.asarray(c, dtype=np.float64) for c in _enu_basis(lat, lon))
    horiz = np.array(v, dtype=np.float64)
    horiz -= float(np.dot(horiz, u)) * u
    hn = float(np.linalg.norm(horiz))
    if hn < 1e-9:
        raise ValueError("dispersed_separation_state needs a horizontal velocity")
    hdg = horiz / hn
    right = np.cross(hdg, u)
    if heading_deg:
        a = math.radians(heading_deg)
        hdg = math.cos(a) * hdg + math.sin(a) * right
    r_new = r + u * altitude_m + hdg * downrange_m + right * crossrange_m
    spd = float(np.linalg.norm(v)) + speed_m_s
    v_new = hdg * spd
    return BoosterState(t_s=sep_state.t_s, epoch_utc=sep_state.epoch_utc,
                        frame="ECEF",
                        position_m=tuple(float(c) for c in r_new),
                        velocity_m_s=tuple(float(c) for c in v_new),
                        mass_kg=sep_state.mass_kg)
