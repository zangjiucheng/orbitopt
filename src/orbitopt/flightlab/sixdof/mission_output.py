"""F-07 (task #232): the 6-DOF study outputs as standard mission artefacts --
a selectable-backend mission, FlightResult v1 run directories, a
verification-report-1.0 report per backend, the M-01 handoff document the
6-DOF run produces at separation, an automatically generated 3-DOF/6-DOF
comparison report, and the scene document Mission Control renders.

One mission, two backends
------------------------
:class:`TwoStageScenario` (the F-06 scenario: mass-consistent stage pair,
gravity-turn program, separation instant, delayed upper-stage ignition) is
the mission. :func:`run_mission` flies it on one of two backends::

    backend="6DOF"  F-06's closed-loop path: rigid-body stack on the F-04
                    tracker, analytic split, two independent rigid children
    backend="3DOF"  the point-mass twin: the same scenario's stack flown by
                    gravity_turn.propagate_point_mass (kinematic program
                    attitude, q-bar S CA/CN aero -- the class of model the
                    OT adapter drives), the SAME split_stack_state split on
                    the frozen trace, point-mass children (booster coasted,
                    upper ignited on the SAME IgnitionPlan schedule)

Both backends consume the identical scenario object, share the F-05
point-mass propagator and the F-06 analytic split, and run the SAME
acceptance checks (mass closure, momentum residual, handoff continuity,
scheduled ignition, trajectory divergence), so "the same task, flown twice"
means the same thing the F-05 ladder means it: the pair differs only in the
6-DOF content, and the comparison report measures that.

The output bundle
-----------------
:func:`write_mission_outputs` turns a flown backend into the standard
artefacts, beside each other under one directory::

    <out>/<backend>/                     FlightResult v1 run directory
        manifest.json  trajectories/*.csv  events.json  verification.json
        handoff.json                     M-01 HandoffState (separation ->
                                         boostback, ECEF, masses included)
        flightlab-<run_id>.verification.json   verification-report-1.0
        scene.json                       SceneData v1.0 (Mission Control)
    <out>/flightlab-comparison-*.verification.json
                                          both backends' comparison

The scene document carries the run's verification report under
``scene["verification"]`` (``orbitopt.verify.report.publish_to_scene``), so
Mission Control shows the verdicts alongside the trails. ECEF tracks convert
to GCRS through the FL-002 frame bridge at scene-export time (see
:mod:`orbitopt.flightlab.scene_export`); the run directory keeps the flown
ECEF states verbatim.

The comparison report is itself a verification-report-1.0 document (kind
``"flightlab-comparison"``): its ``inputs.checks`` are the two backends' own
acceptance checks (a comparison is only honest over two green runs) plus a
shared-span gate, and its ``inputs.bodies`` carry the F-05
:class:`AscentDiff` metrics per body (``6dof - 3dof`` on the shared time
span), the same ``d_*`` vocabulary the F-05 benchmark uses.

Numbers for the packaged scenario live in
``docs/benchmarks/fl607_mission_outputs.json``
(``PYTHONPATH=src python benchmarks/bench_fl607_mission_outputs.py``), the
regression thresholds in ``tests/test_flightlab_sixdof_mission_output.py``,
the prose in ``docs/flightlab.md`` (F-07). Like all native-plant studies
this is illustrative physics, not a flight run; the JSBSim backend (F-01)
and the OT/Gate-B pipeline (FL-006) are untouched by this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from orbitopt.flightlab import frames
from orbitopt.flightlab.contract import (
    BodyTrack,
    Check,
    Event,
    FlightRun,
    Verification,
    write_run,
)
from orbitopt.flightlab.handoff import (
    PHASE_BOOSTBACK,
    PHASE_SEPARATION,
    HandoffState,
    handoff_digest,
)
from orbitopt.flightlab.sixdof.gravity_turn import (
    RUN_COLUMNS_3DOF,
    PointMassRun,
    compare_ascent,
    propagate_point_mass,
)
from orbitopt.flightlab.sixdof.rigidbody import (
    RigidBodyState,
    quaternion_from_tec2b,
)
from orbitopt.flightlab.sixdof.two_stage import (
    TwoStageScenario,
    TwoStageStudy,
    _acceptance_checks,
    f06_two_stage_scenario,
    run_two_stage,
    split_stack_state,
    to_flight_run,
)

__all__ = [
    "BACKEND_3DOF",
    "BACKEND_6DOF",
    "BACKENDS",
    "MissionStudy",
    "f07_scenario",
    "flight_run_report",
    "run_mission",
    "to_flight_run_mission",
    "mission_comparison_report",
    "write_mission_outputs",
]

BACKEND_3DOF = "3DOF"
BACKEND_6DOF = "6DOF"
BACKENDS: tuple[str, ...] = (BACKEND_3DOF, BACKEND_6DOF)

#: The F-05 module documents WHY the native point-mass twin -- not the
#: pinned OT binary -- is the 3-DOF side of a fidelity comparison; the F-07
#: comparison inherits that rationale verbatim.
SHARED_ASSUMPTIONS: tuple[str, ...] = (
    "Illustrative native-plant mission of the F-02 reference plant, the F-04 "
    "tracker and the packaged spike_stage aero -- not flight-certified.",
    "Both backends consume the identical scenario object: same stage pair, "
    "attitude program, separation instant, ignition schedule, physics "
    "switches (US76 + J2 + rotation) and RK4 core; the pair differs only in "
    "the 6-DOF content (rigid-body attitude dynamics, the TVC actuator, "
    "body-oriented aero, the tracker loop).",
    "The 3-DOF side is the same class of model the OpenTsiolkovsky adapter "
    "drives (point mass, kinematic program attitude, q-bar S CA/CN aero) "
    "flown natively -- see sixdof/gravity_turn.py for why not the pinned OT "
    "binary; the upper stage's delayed ignition is the SAME IgnitionPlan on "
    "both sides (a coast leg to the ignition instant, then the throttle "
    "leg), so the schedule itself never differs.",
)


# --------------------------------------------------------------------------
# the 3-DOF twin of the two-stage mission
# --------------------------------------------------------------------------

def _iso_z(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _point_mass_terminal_state(run: PointMassRun, program,
                               physics) -> RigidBodyState:
    """The run's last sample as a plant state: position/velocity/mass from
    the trace, attitude kinematic from the program at the terminal site (the
    3-DOF model has no attitude states -- the program IS the attitude), body
    rates zero (a point mass carries none)."""
    s = run.samples
    r = (float(s["x_m"][-1]), float(s["y_m"][-1]), float(s["z_m"][-1]))
    lat_deg, lon_deg, _ = frames.ecef_to_geodetic(np.asarray(r, dtype=float))
    q = quaternion_from_tec2b(program.dcm_at(float(s["t_s"][-1]),
                                             float(lat_deg), float(lon_deg)))
    return RigidBodyState(
        t_s=float(s["t_s"][-1]), position_m=r,
        velocity_m_s=(float(s["vx_m_s"][-1]), float(s["vy_m_s"][-1]),
                      float(s["vz_m_s"][-1])),
        quaternion=q, omega_rad_s=(0.0, 0.0, 0.0),
        mass_kg=float(s["mass_kg"][-1]))


def _point_mass_body_track(run: PointMassRun, body_id: str, *,
                           parent_body_id: str | None = None) -> BodyTrack:
    """The point-mass run as a contract track -- the same ECEF contract shape
    ``SixDofRun.to_body_track`` emits, with the 3-DOF diagnostics kept as CSV
    extras (a point mass records no attitude block, so there are no q/omega
    columns to preserve)."""
    s = run.samples
    keep = tuple(c for c in RUN_COLUMNS_3DOF
                 if c not in ("t_s", "mass_kg", "x_m", "y_m", "z_m",
                              "vx_m_s", "vy_m_s", "vz_m_s"))
    return BodyTrack(
        body_id=body_id, parent_body_id=parent_body_id, frame="ECEF",
        epoch_utc=_iso_z(run.epoch_utc), provenance="illustrative",
        t_s=tuple(float(v) for v in s["t_s"]),
        mass_kg=tuple(float(v) for v in s["mass_kg"]),
        position_m=tuple((float(a), float(b), float(c))
                         for a, b, c in zip(s["x_m"], s["y_m"], s["z_m"])),
        velocity_m_s=tuple((float(a), float(b), float(c))
                           for a, b, c in zip(s["vx_m_s"], s["vy_m_s"],
                                              s["vz_m_s"])),
        extra_columns={c: tuple(float(v) for v in s[c]) for c in keep})


def _concat_point_mass_runs(coast: PointMassRun, burn: PointMassRun) -> PointMassRun:
    """Two point-mass legs on one body, butt-joined at the leg boundary
    (the burn leg's first sample duplicates the coast leg's last and is
    dropped). Terminal bookkeeping is the joint truth: whichever leg ends
    the run ends it."""
    samples = {c: np.concatenate([coast.samples[c], burn.samples[c][1:]])
               for c in RUN_COLUMNS_3DOF}
    terminal = burn.terminal
    ground_t_s = burn.ground_t_s
    if coast.terminal == "ground":
        samples = {c: coast.samples[c] for c in RUN_COLUMNS_3DOF}
        terminal, ground_t_s = "ground", coast.ground_t_s
    exhausted = coast.propellant_exhausted_t_s
    if exhausted is None:
        exhausted = burn.propellant_exhausted_t_s
    return PointMassRun(columns=RUN_COLUMNS_3DOF, samples=samples,
                        vehicle=burn.vehicle, physics=burn.physics,
                        epoch_utc=coast.epoch_utc, terminal=terminal,
                        ground_t_s=ground_t_s,
                        propellant_exhausted_t_s=exhausted)


def run_two_stage_point_mass(scenario: TwoStageScenario | None = None, *,
                             epoch_utc: datetime | None = None) -> TwoStageStudy:
    """The 3-DOF twin of :func:`~orbitopt.flightlab.sixdof.two_stage.run_two_stage`:
    the same scenario flown point-mass. The stack flies
    :func:`~orbitopt.flightlab.sixdof.gravity_turn.propagate_point_mass`
    (kinematic program attitude, aero on) to ``separation_t_s``; the frozen
    trace splits through the SAME
    :func:`~orbitopt.flightlab.sixdof.two_stage.split_stack_state`; the
    booster coasts point-mass with its engine shut, and the upper stage
    coasts to its scheduled ignition instant and burns from there (two legs
    butt-joined, so the ignition schedule is honoured exactly).

    The frozen split state carries the *program* attitude (the 3-DOF model's
    kinematic assumption) and zero body rates -- a point mass has no
    attitude states to inherit, and the children never see the quaternion
    beyond the impulse direction. The returned :class:`TwoStageStudy` runs
    the identical F-06 acceptance checks, so a red twin is a red mission.
    """
    s = scenario or f06_two_stage_scenario()
    epoch = epoch_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
    kw = dict(dt_s=s.dt_s, sample_every=s.sample_every, physics=s.physics,
              epoch_utc=epoch)

    stack_run = propagate_point_mass(s.state0, s.pair.stack, s.program,
                                     s.separation_t_s, use_aero=True,
                                     throttle=1.0, **kw)
    if stack_run.terminal == "ground":
        raise ValueError(f"the 3-DOF stack hit the ground at "
                         f"t={stack_run.ground_t_s} s before the separation "
                         f"instant {s.separation_t_s} s; pick a larger "
                         f"thrust-to-weight or an earlier separation_t_s")
    frozen = _point_mass_terminal_state(stack_run, s.program, s.physics)

    booster_s, upper_s, sep = split_stack_state(frozen, s.pair,
                                                stack_mass_at_t0_kg=s.state0.mass_kg)
    booster_v = replace(s.pair.booster,
                        propellant_mass_kg=sep["booster_propellant_remaining_kg"])
    upper_v = replace(s.pair.upper,
                      propellant_mass_kg=sep["upper_propellant_kg"])

    booster_run = propagate_point_mass(booster_s, booster_v, s.program,
                                       s.t_end_s, use_aero=True,
                                       throttle=0.0, **kw)
    ignition_t = s.ignition.ignition_t_s
    if ignition_t < s.t_end_s:
        coast = propagate_point_mass(upper_s, upper_v, s.program, ignition_t,
                                     use_aero=True, throttle=0.0, **kw)
        if coast.terminal == "ground":
            upper_run = coast
        else:
            burn_start = _point_mass_terminal_state(coast, s.program, s.physics)
            burn = propagate_point_mass(burn_start, upper_v, s.program,
                                        s.t_end_s, use_aero=True,
                                        throttle=s.ignition.throttle, **kw)
            upper_run = _concat_point_mass_runs(coast, burn)
    else:
        upper_run = propagate_point_mass(upper_s, upper_v, s.program,
                                         s.t_end_s, use_aero=True,
                                         throttle=s.ignition.throttle, **kw)

    handoff = HandoffState(
        from_phase=PHASE_SEPARATION, to_phase=PHASE_BOOSTBACK,
        t_s=frozen.t_s, epoch_utc=_iso_z(epoch), frame="ECEF",
        position_m=tuple(float(c) for c in booster_s.position_m),
        velocity_m_s=tuple(float(c) for c in booster_s.velocity_m_s),
        mass_kg=booster_s.mass_kg, dry_mass_kg=booster_v.dry_mass_kg,
        ignitions_used=1)

    checks = _acceptance_checks(s, sep, booster_run, upper_run)
    ascent = _PointMassAscent(stack_run)
    return TwoStageStudy(scenario=s, ascent=ascent, frozen_state=frozen,
                         separation=sep, booster_state=booster_s,
                         upper_state=upper_s, booster=booster_run,
                         upper=upper_run, booster_handoff=handoff,
                         checks=checks)


class _PointMassAscent:
    """The two_stage.to_flight_run read-shim around the point-mass stack run
    (``study.ascent.run`` / ``study.stack`` / the att_err summary keys); a
    point mass tracked nothing, so the tracker diagnostics report 0.0 -- the
    3-DOF model has, exactly, zero attitude error against its own program."""

    def __init__(self, run: PointMassRun) -> None:
        self.run = run
        self.metrics = {"att_err_max_deg": 0.0, "att_err_rms_deg": 0.0}

    def summary(self) -> dict[str, Any]:
        return {"att_err_max_deg": 0.0, "att_err_rms_deg": 0.0}


# --------------------------------------------------------------------------
# the backend-selectable mission
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionStudy:
    """One mission definition flown on one backend. ``study`` is the flown
    :class:`~orbitopt.flightlab.sixdof.two_stage.TwoStageStudy` (both
    backends produce one -- the 3-DOF twin's runs are
    :class:`PointMassRun`s); everything downstream (run directory, handoff,
    report, comparison, scene) reads this one shape."""

    backend: str
    study: TwoStageStudy

    @property
    def scenario(self) -> TwoStageScenario:
        return self.study.scenario

    @property
    def booster_handoff(self) -> HandoffState:
        return self.study.booster_handoff

    @property
    def checks(self) -> tuple[Check, ...]:
        return self.study.checks

    def summary(self) -> dict[str, Any]:
        out = self.study.summary()
        out["backend"] = self.backend
        return out


def f07_scenario(**kwargs) -> TwoStageScenario:
    """The packaged F-07 mission: the F-06 two-stage scenario (the mission
    both backends fly), named for this task. Keyword-for-key pass-through to
    :func:`~orbitopt.flightlab.sixdof.two_stage.f06_two_stage_scenario`."""
    return replace(f06_two_stage_scenario(**kwargs), name="f07-two-stage")


def run_mission(scenario: TwoStageScenario | None = None, *,
                backend: str = BACKEND_6DOF,
                epoch_utc: datetime | None = None) -> MissionStudy:
    """Fly ``scenario`` (default :func:`f07_scenario`) on ``backend``:
    ``"6DOF"`` runs F-06's closed-loop rigid-body path
    (:func:`~orbitopt.flightlab.sixdof.two_stage.run_two_stage`),
    ``"3DOF"`` the point-mass twin (:func:`run_two_stage_point_mass`).
    Case-insensitive; anything else raises ``ValueError`` naming the two
    backends. Same scenario object, same checks either way."""
    name = str(backend).strip().upper()
    if name not in BACKENDS:
        raise ValueError(f"backend {backend!r} is not supported; choose "
                         f"{', '.join(BACKENDS)}")
    s = scenario or f07_scenario()
    study = (run_two_stage(s, epoch_utc=epoch_utc) if name == BACKEND_6DOF
             else run_two_stage_point_mass(s, epoch_utc=epoch_utc))
    return MissionStudy(backend=name, study=study)


# --------------------------------------------------------------------------
# the FlightRun / VerificationReport / handoff artefacts
# --------------------------------------------------------------------------

def to_flight_run_mission(mission: MissionStudy, *,
                          run_id: str | None = None,
                          epoch_utc: str = "2026-01-01T00:00:00Z") -> FlightRun:
    """The mission as a three-body FlightResult v1 run (bodies ``stack``,
    ``booster``, ``upper``; children parented to the stack, one mission
    clock). On the 6-DOF backend this is exactly
    :func:`~orbitopt.flightlab.sixdof.two_stage.to_flight_run`; on the
    3-DOF backend it is the same manifest shape with the point-mass tracks
    and the 3-DOF twin's own check verdicts."""
    study = mission.study
    if mission.backend == BACKEND_6DOF:
        return to_flight_run(study,
                             run_id=run_id or "f07-mission-6dof",
                             epoch_utc=epoch_utc)
    rid = run_id or "f07-mission-3dof"
    s, sep = study.scenario, study.separation
    mk = lambda tag: hashlib.sha256(f"f07:{tag}".encode()).hexdigest()
    stack_track = _point_mass_body_track(study.stack, "stack")
    booster_track = _point_mass_body_track(study.booster, "booster",
                                           parent_body_id="stack")
    upper_track = _point_mass_body_track(study.upper, "upper",
                                         parent_body_id="stack")
    return FlightRun(
        run_id=rid,
        target_sha256=mk("no-flight-target-two-stage-scenario"),
        vehicle_sha256=mk("f07-stack-split-into-booster-and-upper"),
        git_sha="0" * 40,
        created_utc=epoch_utc,
        engines={"stage1": "spike_stage engine (vac 1.88 MN / Isp 320 s)",
                 "stage2": "upper engine (vac 220 kN / Isp 340 s)"},
        assumptions=SHARED_ASSUMPTIONS + tuple(s.assumptions) + (
            (f"3-DOF backend: stack flown point-mass to t={s.separation_t_s} "
             f"s, split analytically with a {sep['separation_impulse_ns']} N.s "
             f"body-+x impulse; upper ignition at t={s.ignition.ignition_t_s} s "
             f"(delay {s.ignition.ignition_delay_s} s) via a butt-joined "
             f"coast/burn leg pair."),
            ("Booster handed to recovery on the M-01 separation->boostback "
             f"boundary (handoff digest "
             f"{handoff_digest(study.booster_handoff)[:12]})."),
        ),
        tracks=(stack_track, booster_track, upper_track),
        events=(
            Event(0.0, "ignition", "stack", {"throttle": 1.0}),
            Event(study.stack.max_q.t_s, "max_q", "stack",
                  {"dynamic_pressure_pa": float(study.stack.max_q.q_pa),
                   "altitude_m": float(study.stack.max_q.altitude_m),
                   "mach": float(study.stack.max_q.mach)}),
            Event(s.separation_t_s, "separation", "stack",
                  {"impulse_ns": sep["separation_impulse_ns"],
                   "mass_kg": float(sep["stack_mass_separation_kg"])}),
            Event(s.separation_t_s, "separation", "booster",
                  {"dv_m_s": -float(sep["dv_booster_m_s"]),
                   "mass_kg": float(study.booster_state.mass_kg)}),
            Event(s.separation_t_s, "separation", "upper",
                  {"dv_m_s": float(sep["dv_upper_m_s"]),
                   "mass_kg": float(study.upper_state.mass_kg)}),
            Event(s.ignition.ignition_t_s, "ignition", "upper",
                  {"throttle": s.ignition.throttle,
                   "delay_s": s.ignition.ignition_delay_s}),
        ),
        verification=Verification(
            optimizer_converged=None,
            physics_verified=all(c.passed for c in study.checks),
            mission_constraints_passed=None,
            checks=study.checks,
        ),
    )


def flight_run_report(mission: MissionStudy,
                      target=None) -> dict:
    """The mission's verification-report-1.0 document, via the standard
    M-03 writer :func:`orbitopt.flightlab.segment.flight_run_report` -- kind
    ``"flightlab"``, label = run id, the three never-collapsed verdicts and
    every check as flown."""
    from orbitopt.flightlab.segment import flight_run_report as _report

    return _report(to_flight_run_mission(mission), target)


def handoff_document(mission: MissionStudy) -> dict:
    """The M-01 handoff contract document the mission produces at separation
    (``HandoffState.to_dict``: separation -> boostback, ECEF, wet/dry mass,
    ignitions used) -- exactly what a recovery segment restarts from."""
    return mission.booster_handoff.to_dict()


# --------------------------------------------------------------------------
# the automatically generated 3-DOF/6-DOF comparison report
# --------------------------------------------------------------------------

def _diff_summary(diff) -> dict[str, Any]:
    return {"metrics": diff.summary(),
            "samples": int(diff.t_s.size),
            "t_span_s": [float(diff.t_s[0]), float(diff.t_s[-1])]}


def mission_comparison_report(mission_3dof: MissionStudy,
                              mission_6dof: MissionStudy,
                              *, bodies: tuple[str, ...] = ("stack", "booster",
                                                            "upper")) -> dict:
    """The acceptance's auto-generated 3-DOF/6-DOF comparison report as a
    verification-report-1.0 document (kind ``"flightlab-comparison"``):
    per-body F-05 :class:`AscentDiff` metrics (``6dof - 3dof`` on the shared
    time span, the four F-05 channels), with the two backends' own
    acceptance checks as the report's checks -- the comparison is green only
    when both runs are. ``inputs.bodies[<body>]`` also carries the run
    terminal bookkeeping of both sides."""
    from orbitopt.verify.report import (CHECK_NUMERICAL, build_report, report_digest,
                                        status_from_checks)

    if mission_3dof.backend != BACKEND_3DOF or mission_6dof.backend != BACKEND_6DOF:
        raise ValueError("the comparison runs 3DOF minus 6DOF; pass the two "
                         "backends' mission studies in that order")
    s3, s6 = mission_3dof.study, mission_6dof.study
    runs3 = {"stack": s3.stack, "booster": s3.booster, "upper": s3.upper}
    runs6 = {"stack": s6.stack, "booster": s6.booster, "upper": s6.upper}
    body_docs: dict[str, Any] = {}
    span_ok = True
    for body in bodies:
        try:
            diff = compare_ascent(runs3[body], runs6[body],
                                  label=f"6dof_full minus 3dof_aero ({body})")
        except ValueError as exc:  # no shared time span -> visible, not fake
            span_ok = False
            body_docs[body] = {"error": str(exc)}
            continue
        body_docs[body] = _diff_summary(diff)
        body_docs[body]["run_3dof"] = {
            "terminal": runs3[body].terminal,
            "mass_kg": float(runs3[body].samples["mass_kg"][-1])}
        body_docs[body]["run_6dof"] = {
            "terminal": runs6[body].terminal,
            "mass_kg": float(runs6[body].samples["mass_kg"][-1])}
    checks = {f"{BACKEND_3DOF}.{c.id}": bool(c.passed) for c in s3.checks}
    checks.update({f"{BACKEND_6DOF}.{c.id}": bool(c.passed) for c in s6.checks})
    checks["comparison-shared-spans"] = span_ok
    s = s6.scenario
    inputs = {
        "scenario": {"name": s.name, "separation_t_s": s.separation_t_s,
                     "ignition_delay_s": s.ignition.ignition_delay_s,
                     "t_end_s": s.t_end_s, "dt_s": s.dt_s},
        "backends": {"3dof": "native point-mass twin (gravity_turn twin of "
                             "the OT-class model)",
                     "6dof": "native rigid-body plant (F-02/F-04/F-06 path)"},
        "headline": "6dof_minus_3dof",
        "checks": checks,
        "bodies": body_docs,
    }
    report = build_report("flightlab-comparison",
                          f"{s.name}-6dof-vs-3dof", [],
                          inputs=inputs,
                          targets={"6dof_minus_3dof": {
                              b: body_docs[b].get("metrics", {})
                              for b in bodies}})
    # V-04 (task #243): every check here is the run checking itself -- staging
    # continuity, mass/momentum closure, the scheduled ignition, the two bodies
    # parting, both backends spanning the same window -- so numerical
    # consistency; the 6dof-minus-3dof deltas are reported, not graded.
    report["status"] = status_from_checks(checks, dict.fromkeys(checks, CHECK_NUMERICAL))
    report["digest"] = report_digest(report)
    return report


# --------------------------------------------------------------------------
# the output bundle
# --------------------------------------------------------------------------

def write_mission_outputs(mission: MissionStudy, run_dir: str | Path, *,
                          scene: bool = True,
                          epoch_utc: str = "2026-01-01T00:00:00Z") -> dict:
    """Write one backend's output bundle into ``run_dir`` (created): the
    FlightResult v1 directory (``write_run``), the M-01 handoff document
    (``handoff.json``), the verification-report-1.0 document
    (``flightlab-<run_id>.verification.json``) and -- with ``scene=True`` --
    the Mission Control scene (``scene.json``, the run's verification report
    published under ``scene["verification"]``). Returns the written paths."""
    from orbitopt.verify.report import publish_to_scene, write_report

    out = Path(run_dir)
    run = to_flight_run_mission(mission, epoch_utc=epoch_utc)
    write_run(out, run)
    paths: dict[str, Path] = {"run_dir": out}
    handoff_path = out / "handoff.json"
    handoff_path.write_text(json.dumps(handoff_document(mission), indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")
    paths["handoff"] = handoff_path
    report = flight_run_report(mission)
    report_path = write_report(report, out / f"flightlab-{run.run_id}.verification.json")
    paths["report"] = report_path
    if scene:
        from orbitopt.flightlab.scene_export import run_to_scene

        scene_doc = publish_to_scene(run_to_scene(run), [report])
        scene_path = out / "scene.json"
        scene_path.write_text(json.dumps(scene_doc, indent=2) + "\n",
                              encoding="utf-8")
        paths["scene"] = scene_path
    return paths


def write_mission_pair(scenario: TwoStageScenario | None = None,
                       out_dir: str | Path = "runs/f07-mission", *,
                       scene: bool = True,
                       epoch_utc: str = "2026-01-01T00:00:00Z") -> dict:
    """Fly the same mission on both backends and write the whole F-07
    bundle: ``<out>/3DOF/``, ``<out>/6DOF/`` (each via
    :func:`write_mission_outputs`) and the automatically generated
    comparison report ``<out>/flightlab-comparison-*.verification.json``.
    Returns ``{"3DOF": paths, "6DOF": paths, "comparison": report,
    "comparison_path": Path}``."""
    from orbitopt.verify.report import default_report_path, write_report

    s = scenario or f07_scenario()
    out = Path(out_dir)
    result: dict[str, Any] = {}
    missions: dict[str, MissionStudy] = {}
    for backend in BACKENDS:
        mission = run_mission(s, backend=backend,
                             epoch_utc=_parse_utc_stamp(epoch_utc))
        missions[backend] = mission
        result[backend] = write_mission_outputs(
            mission, out / backend, scene=scene, epoch_utc=epoch_utc)
    report = mission_comparison_report(missions[BACKEND_3DOF],
                                       missions[BACKEND_6DOF])
    report_path = write_report(report, default_report_path(report, out))
    result["comparison"] = report
    result["comparison_path"] = report_path
    return result


def _parse_utc_stamp(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
