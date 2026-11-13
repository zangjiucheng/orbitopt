"""M-03 (task #221): a Flight Lab run wrapped as a mission-pipeline segment.

:class:`FlightLabSegment` follows the MissionSegment contract being defined
in M-01 (#219) -- ``run(initial_state) -> SegmentResult`` -- for the segment
that starts at the pad: a Flight Lab ascent takes *no* incoming handoff
state (``initial_state`` must be ``None``; anything else is a caller error,
not a silent ignore) and *produces* one, the SECO insertion state of the
flown upper stage, in the shape M-01's ``HandoffState`` mirrors (epoch with
timescale, frame, r, v, mass, metadata). The next segment in a chain (M-02
parking-orbit handoff, M-05 tudatpy propagation) picks up exactly that.

Everything the segment does is delegated to :func:`orbitopt.flightlab.simulate.simulate`
-- the frozen Gate-B/FL-006 pipeline (FL-302/303/304 recovery included) --
so wrapping changes nothing about the run directory written or the honest
three-boolean verdict; the segment only hands the ``FlightRun`` back with
the handoff state and a standardized ``VerificationReport``
(:func:`flight_run_report`) so the unified pipeline result (orbitopt.pipeline)
can carry Flight Lab alongside the patched-conic mission kinds.

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.segment import FlightLabSegment

    seg = FlightLabSegment(load_target("scenarios/leo_two_stage.yaml"),
                           vehicle_path=None, run_dir="runs/segment-0001")
    result = seg.run()
    result.handoff["position_m"]      # SECO state, GCRS, UTC epoch
    result.run.verification           # the same honest booleans as `orbitopt flight simulate`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from orbitopt.flightlab import checks as _checks
from orbitopt.flightlab.contract import (
    Event,
    FlightRun,
    FlightTarget,
    Verification,
)

__all__ = ["FlightLabSegment", "FlightLabSegmentResult", "flight_run_report"]

#: physical_model name/fidelity the pipeline stage records for the segment.
PHYSICAL_MODEL = {
    "name": "OpenTsiolkovsky v1.0.0 (3-DOF point-mass ascent) + tudatpy orbit re-fly",
    "fidelity": "two-body",
}


#: Gate-B check ids that are data integrity / conservation of the run itself
#: (V-04 numerical consistency); every other Gate-B check grades the flight
#: against a target band, a vehicle limit or an independent propagator
#: (model agreement).
_NUMERICAL_CHECK_PREFIXES = ("mass-", "sixdof-states-finite", "sixdof-mass-non-negative")


def _check_category(name: str) -> str:
    """V-04 (task #243) evidence category of one flight-run report check."""
    from orbitopt.verify.report import CHECK_MODEL, CHECK_NUMERICAL

    if name == "optimizer_converged":
        return CHECK_NUMERICAL          # converged targeting
    if name in ("physics_verified", "mission_constraints_passed"):
        return CHECK_MODEL              # reference propagator / vehicle limits
    if name.startswith("check."):
        cid = name[len("check."):]
        return CHECK_NUMERICAL if cid.startswith(_NUMERICAL_CHECK_PREFIXES) else CHECK_MODEL
    raise ValueError(f"flight-run report check {name!r} has no verification category")


def flight_run_report(run: FlightRun, target: FlightTarget | None = None) -> dict:
    """The run's :class:`~orbitopt.flightlab.contract.Verification` as a
    standardized verification-report-1.0 document (kind ``"flightlab"``,
    label = ``run.run_id``), so Flight Lab verdicts enter the unified run
    record exactly like the other kinds' reports (gate, warnings, written
    ``.verification.json``).

    Tri-state honesty (Gate G5) is preserved: ``optimizer_converged`` /
    ``physics_verified`` / ``mission_constraints_passed`` are recorded as
    checks *only when assessed* -- a ``None`` verdict never counts as a
    pass but also never fabricates a failure; it stays visible as ``null``
    in the report's ``targets`` block. Failed checks carry their
    ``reason_code`` next to them in ``targets.reason_codes``."""
    from orbitopt.verify.report import build_report, report_digest, status_from_checks

    v = run.verification
    checks: dict[str, bool] = {f"check.{c.id}": bool(c.passed) for c in v.checks}
    if v.optimizer_converged is not None:
        checks["optimizer_converged"] = bool(v.optimizer_converged)
    if v.physics_verified is not None:
        checks["physics_verified"] = bool(v.physics_verified)
    if v.mission_constraints_passed is not None:
        checks["mission_constraints_passed"] = bool(v.mission_constraints_passed)
    reasons = {c.id: c.reason_code for c in v.checks if not c.passed and c.reason_code}

    inputs = {
        "run_id": run.run_id,
        "target_sha256": run.target_sha256,
        "vehicle_sha256": run.vehicle_sha256,
        "git_sha": run.git_sha,
        "created_utc": run.created_utc,
        "engines": dict(run.engines),
        "assumptions": list(run.assumptions),
        "events": [{"t_s": e.t_s, "kind": e.kind, "body_id": e.body_id} for e in run.events],
        "target_file": (target.mission.id if target is not None else None),
    }
    targets = {
        "optimizer_converged": v.optimizer_converged,
        "physics_verified": v.physics_verified,
        "mission_constraints_passed": v.mission_constraints_passed,
        "reason_codes": reasons,
    }
    report = build_report("flightlab", run.run_id, [], inputs=inputs, targets=targets)
    report["status"] = status_from_checks(
        checks, {name: _check_category(name) for name in checks})
    report["digest"] = report_digest(report)
    return report


@dataclass(frozen=True)
class FlightLabSegmentResult:
    """What a Flight Lab segment hands the pipeline: the flown
    :class:`FlightRun`, the run directory it was written to, the outgoing
    handoff state (SECO insertion, GCRS/UTC -- M-01 ``HandoffState`` shape),
    the run's events and verdict, and the standardized verification report.
    ``to_dict()`` is the JSON-friendly summary recorded as the stage's
    candidate entry."""

    run: FlightRun
    run_dir: Path
    handoff: dict
    events: tuple[Event, ...] = ()
    verification: Verification = field(default=Verification)
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        v = self.verification
        return {
            "run_id": self.run.run_id,
            "run_dir": str(self.run_dir),
            "events": [e.kind for e in self.events],
            "optimizer_converged": v.optimizer_converged,
            "physics_verified": v.physics_verified,
            "mission_constraints_passed": v.mission_constraints_passed,
            "checks_total": len(v.checks),
            "checks_passed": sum(1 for c in v.checks if c.passed),
            "handoff": self.handoff,
            "selected": True,
        }


@dataclass(frozen=True)
class FlightLabSegment:
    """A Flight Lab ascent (plus optional FL-304 recovery and tudatpy
    re-fly) as one MissionSegment: constructed from a validated
    :class:`FlightTarget` + vehicle file, ``run()`` flies it through the
    frozen ``simulate`` pipeline into ``run_dir`` and returns the collected
    :class:`FlightLabSegmentResult`.

    ``vehicle_path=None`` resolves the target's declared ``vehicle.source``
    (the same 4-step order ``orbitopt flight`` uses). The remaining fields
    mirror ``simulate``'s keyword-for-key: ``attitude_program``,
    ``refly``/``refly_orbits``/``refly_max_step_s``, ``timeout_s``,
    ``run_recovery``/``recovery_options``. Raises what ``simulate`` raises
    -- a failing *check* is a red verdict in the result, never an exception."""

    target: FlightTarget
    vehicle_path: str | Path | None = None
    run_dir: str | Path = "flightrun"
    attitude_program: Sequence | None = None
    refly: bool = True
    refly_orbits: float = 1.05
    refly_max_step_s: float = 300.0
    timeout_s: float = 300.0
    run_recovery: bool = True
    recovery_options: Mapping[str, Any] | None = None
    target_path: str | Path | None = None

    def run(self, out_dir: str | Path | None = None,
            initial_state: dict | None = None) -> FlightLabSegmentResult:
        """Fly the segment. ``initial_state`` must be ``None`` (or absent):
        this segment starts at the pad, so an incoming handoff state means
        the chain was assembled wrong. ``out_dir`` overrides the configured
        ``run_dir`` (one segment object, several run directories)."""
        if initial_state is not None:
            raise ValueError(
                "FlightLabSegment is a pad-start segment: it takes no incoming "
                f"handoff state, got {type(initial_state).__name__}"
            )
        from orbitopt.flightlab.simulate import simulate
        from orbitopt.flightlab.vehicle_source import resolve_vehicle_path

        out = Path(out_dir if out_dir is not None else self.run_dir)
        vehicle = resolve_vehicle_path(
            self.vehicle_path, self.target,
            self.target_path if self.target_path is not None else out)
        run = simulate(self.target, vehicle, out,
                       attitude_program=self.attitude_program,
                       refly=self.refly, refly_orbits=self.refly_orbits,
                       refly_max_step_s=self.refly_max_step_s,
                       timeout_s=self.timeout_s,
                       run_recovery=self.run_recovery,
                       recovery_options=self.recovery_options)
        return FlightLabSegmentResult(
            run=run,
            run_dir=out,
            handoff=self.handoff_state(run),
            events=tuple(run.events),
            verification=run.verification,
            report=flight_run_report(run, self.target),
        )

    @staticmethod
    def handoff_state(run: FlightRun) -> dict:
        """The run's outgoing HandoffState (M-01 shape): the upper stage's
        SECO insertion state as flown -- frame, UTC epoch (timescale
        declared), r, v, mass, plus the metadata a downstream segment
        needs (mission epoch time, body id, source run)."""
        state = _checks.insertion_state(run)
        track = next((t for t in run.tracks if t.body_id == state["body_id"]), None)
        return {
            "frame": track.frame if track is not None else "GCRS",
            "epoch_utc": state["epoch_utc"],
            "time_scale": "UTC",
            "position_m": [float(x) for x in state["position_m"]],
            "velocity_m_s": [float(x) for x in state["velocity_m_s"]],
            "mass_kg": float(state["mass_kg"]),
            "t_s": float(state["t_s"]),
            "body_id": state["body_id"],
            "run_id": run.run_id,
        }
