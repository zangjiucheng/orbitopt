"""The stage runners: one mission leg per cross-environment subprocess.

Each runner loads the shared scenario from the resolved plan, flies its leg
with the ordinary single-environment machinery, and *seals* its outputs into
an exchange envelope (:mod:`orbitopt.crossenv.envelope`) before exiting --
the environment boundary is only ever crossed over validated, hashed,
versioned files.

``ascent`` (the tudat-environment leg): the scenario is run with its
``recovery`` block stripped -- the mission legs up to and including the
SECO-to-parking gate, exactly as E5-01/E2-01 run them -- and the flown
``FlightRun``, the M-01 SECO handoff state and the recovery separation state
are sealed for the landing leg.

``recovery`` (the gfold-environment leg): consumes and re-validates the
ascent exchange (digests re-computed before anything is read), then flies
FL-304 over the handed-over run via the mission's own
``_fly_recovery_stage`` and seals the combined run, the recovery handoff
report and its final boundary state.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from orbitopt.crossenv.envelope import (CrossEnvError, environment_record, read_envelope,
                                        read_payloads, write_envelope)
from orbitopt.crossenv.plan import CrossEnvPlan, CrossEnvStage
from orbitopt.crossenv.qualification import RECOVERY_CHECKS_KIND

STAGE_REPORT_KIND = "orbitopt.crossenv.stage-report"
STAGE_REPORT_SCHEMA_VERSION = "1.0"
REPORT_WRAPPER_KIND = "orbitopt.crossenv.report"
#: E5-05 (task #272): the E5-02 canonical separation handoff as an exchange
#: payload -- the document :class:`orbitopt.mission.coupled_branches.
#: SeparationHandoff` writes, wrapped in the versioned report envelope, so a
#: recovery leg in another environment can DRIVE from the one state both
#: branches descend from instead of re-reading one branch's track.
SEPARATION_HANDOFF_KIND = "orbitopt.mission.separation-handoff"
EXCHANGE_DIRNAME = "exchange"

#: runner name -> callable(plan, stage, out_dir)
STAGE_RUNNERS: dict = {}


def _register(name):
    def wrap(fn):
        STAGE_RUNNERS[name] = fn
        return fn
    return wrap


def _write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _absolutise(scenario):
    """Repo-relative scenario inputs to absolute paths so a stage subprocess
    flies identically from any working directory."""
    from orbitopt.crossenv.plan import resolve_path

    updates: dict[str, Any] = {}
    for field_name in ("ascent_run_dir", "ascent_target_file"):
        value = getattr(scenario, field_name)
        if value:
            updates[field_name] = str(resolve_path(value))
    if scenario.recovery is not None:
        recovery_updates: dict[str, Any] = {}
        for field_name in ("target_file", "vehicle_file"):
            value = getattr(scenario.recovery, field_name)
            if value:
                recovery_updates[field_name] = str(resolve_path(value))
        if recovery_updates:
            updates["recovery"] = replace(scenario.recovery, **recovery_updates)
    return replace(scenario, **updates) if updates else scenario


def _vehicle_path(scenario) -> Path:
    """The recovery vehicle model file: the same resolution
    ``_fly_recovery_stage`` performs (explicit file, else the packaged
    vehicle catalog under ``data_root()``)."""
    from orbitopt.flightlab.vehicle_source import data_root

    name = (scenario.recovery.vehicle_file if scenario.recovery is not None
            and scenario.recovery.vehicle_file else scenario.vehicle)
    path = Path(str(name))
    if path.is_file():
        return path
    catalog = data_root() / "vehicles" / f"{path.stem}.yaml"
    if catalog.is_file():
        return catalog
    raise FileNotFoundError(f"recovery vehicle {name!r}: not a file and not in the packaged "
                            "vehicle catalog")


@_register("ascent")
def run_ascent_stage(plan: CrossEnvPlan, stage: CrossEnvStage, out_dir: Path) -> dict:
    """Fly the ascent/mission leg and seal its exchange. Returns the stage
    report document."""
    from orbitopt.flightlab.contract import SCHEMA_VERSION as FLIGHT_SCHEMA_VERSION
    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.handoff import HANDOFF_SCHEMA_VERSION, HANDOFF_KIND, HandoffState
    from orbitopt.mission.ascent import ascent_handoff_state
    from orbitopt.mission.contract import state_to_dict
    from orbitopt.mission.two_stage_leo import load_scenario, run_two_stage_leo

    stage_dir = Path(out_dir) / stage.id
    exchange = stage_dir / EXCHANGE_DIRNAME
    scenario = _absolutise(load_scenario(plan.scenario))
    if scenario.ascent_mode != "run_dir":
        raise CrossEnvError(
            f"crossenv stage {stage.id!r}: the packaged cross-environment flow replays a "
            "flown ascent (ascent.mode run_dir); optimize mode would re-fly the ascent "
            "in the landing environment, which cannot host tudatpy")

    mission = run_two_stage_leo(replace(scenario, recovery=None), stage_dir / "mission",
                                propagate=stage.options.get("propagate"))
    run_dir = Path(mission.ascent_run_dir)
    run = read_run(run_dir)

    exchange.mkdir(parents=True, exist_ok=True)
    sealed_run = exchange / "ascent-run"
    if sealed_run.exists():
        shutil.rmtree(sealed_run)
    shutil.copytree(run_dir, sealed_run)

    seco = state_to_dict(ascent_handoff_state(run))
    _write_json(exchange / "states" / "seco-handoff.json", seco)

    from orbitopt.flightlab.recovery.propagator import BoosterModel, separation_state_from_run

    dry_mass = float(BoosterModel.load(_vehicle_path(scenario)).dry_mass_kg)
    separation = HandoffState.of(separation_state_from_run(run), from_phase="separation",
                                 to_phase="boostback", dry_mass_kg=dry_mass)
    _write_json(exchange / "states" / "separation-handoff.json", separation.to_dict())

    _write_json(exchange / "reports" / "mission-report.json",
                {"kind": REPORT_WRAPPER_KIND, "schema_version": "1.0",
                 "label": "two-stage-leo mission report (recovery leg stripped)",
                 "report": mission.to_dict()})

    payloads = [
        {"name": "ascent-run", "type": "flight-run", "schema_version": FLIGHT_SCHEMA_VERSION,
         "path": "ascent-run", "digest_kind": "run-dir"},
        {"name": "seco-handoff", "type": "mission-handoff-state",
         "schema_version": "1.0", "path": "states/seco-handoff.json", "digest_kind": "file"},
        {"name": "separation-handoff", "type": "flightlab-handoff-state",
         "schema_version": HANDOFF_SCHEMA_VERSION, "path": "states/separation-handoff.json",
         "digest_kind": "file"},
        {"name": "mission-report", "type": "report", "schema_version": "1.0",
         "report_kind": REPORT_WRAPPER_KIND, "path": "reports/mission-report.json",
         "digest_kind": "file"},
    ]
    if stage.options.get("coupled_separation"):
        # E5-05 (task #272): seal the one canonical separation state both
        # mission branches descend from, so the landing leg can be DRIVEN
        # from it across the environment boundary (E5-02 coupled branches).
        from orbitopt.mission.coupled_branches import separation_handoff

        coupled = separation_handoff(run)
        if coupled is None:
            raise CrossEnvError(f"crossenv stage {stage.id!r}: options.coupled_separation "
                                "requested but the flown run never branched at separation")
        _write_json(exchange / "states" / "coupled-separation-handoff.json",
                    {"kind": SEPARATION_HANDOFF_KIND, "schema_version": "1.0",
                     "handoff": coupled.to_dict()})
        payloads.append(
            {"name": "coupled-separation-handoff", "type": "report", "schema_version": "1.0",
             "report_kind": SEPARATION_HANDOFF_KIND,
             "path": "states/coupled-separation-handoff.json", "digest_kind": "file"})
    write_envelope(exchange, stage=stage.id, runner=stage.runner, env_label=stage.env,
                   payloads=payloads)

    failed = [c["id"] for c in mission.to_dict()["checks"] if not c["passed"]]
    return {
        "kind": STAGE_REPORT_KIND, "schema_version": STAGE_REPORT_SCHEMA_VERSION,
        "stage": stage.id, "runner": stage.runner, "env": stage.env,
        "environment": environment_record(stage.env),
        "status": "completed", "mission_ok": bool(mission.ok),
        "summary": f"status={mission.status} run_id={mission.run_id}",
        "failed_checks": failed,
        "not_assessed": [str(n.get("id")) for n in mission.to_dict()["not_assessed"]],
        "payloads": [p["name"] for p in payloads],
        "handoff_kinds": [HANDOFF_KIND],
    }


@_register("recovery")
def run_recovery_stage(plan: CrossEnvPlan, stage: CrossEnvStage, out_dir: Path) -> dict:
    """Consume the ascent exchange, fly FL-304 over the handed-over run and
    seal the recovery exchange. Returns the stage report document."""
    from orbitopt.flightlab.contract import SCHEMA_VERSION as FLIGHT_SCHEMA_VERSION
    from orbitopt.flightlab.handoff import HANDOFF_SCHEMA_VERSION, HandoffState
    from orbitopt.mission.two_stage_leo import _fly_recovery_stage, load_scenario

    if len(stage.consumes) != 1:
        raise CrossEnvError(f"crossenv stage {stage.id!r}: the recovery runner consumes "
                            "exactly one ascent exchange")
    upstream = Path(out_dir) / stage.consumes[0] / EXCHANGE_DIRNAME
    payloads = read_payloads(upstream)  # validates schema + every digest, then loads natively
    run = payloads["ascent-run"]
    payloads["seco-handoff"]     # parsed HandoffState: schema-validated by its own reader
    payloads["separation-handoff"]

    separation = None
    if stage.options.get("drive_from_separation"):
        # E5-05 (task #272): drive the booster branch from the ascent leg's
        # canonical separation handoff (E5-02), not from this leg's own
        # re-read of the debris track. The handoff document arrives inside a
        # digest-validated envelope; its identity (handoff_id) is then
        # re-graded below against the digest the recovery block records.
        from orbitopt.mission.coupled_branches import SeparationHandoff

        coupled = payloads.get("coupled-separation-handoff")
        if coupled is None:
            raise CrossEnvError(
                f"crossenv stage {stage.id!r}: options.drive_from_separation requested but "
                "the upstream exchange seals no coupled-separation-handoff payload (set "
                "options.coupled_separation on the ascent stage)")
        separation = SeparationHandoff.from_dict(coupled["handoff"])

    stage_dir = Path(out_dir) / stage.id
    exchange = stage_dir / EXCHANGE_DIRNAME
    exchange.mkdir(parents=True, exist_ok=True)
    scenario = _absolutise(load_scenario(plan.scenario))
    report_file = exchange / "states" / "recovery-handoff-report.json"

    _combined, block, recovery_checks, recovery_notes = _fly_recovery_stage(
        scenario, run, exchange, handoff_report_file=report_file, separation=separation)
    if separation is not None:
        driven = block.get("separation_handoff_id")
        if driven != separation.handoff_id:
            raise CrossEnvError(
                f"crossenv stage {stage.id!r}: the recovery leg reports separation handoff "
                f"{driven!r}, not the handed-over {separation.handoff_id!r} -- the two "
                "branches did not start at the one state")

    report_doc = json.loads(report_file.read_text(encoding="utf-8"))
    states = report_doc.get("states") or []
    if not states:
        raise CrossEnvError("the recovery leg produced no M-01 boundary states to hand over")
    final = HandoffState.from_dict(states[-1]).to_dict()
    _write_json(exchange / "states" / "recovery-final-handoff.json", final)
    _write_json(exchange / "reports" / "recovery-block.json",
                {"kind": REPORT_WRAPPER_KIND, "schema_version": "1.0",
                 "label": "recovery block of the two-stage-leo report", "report": block})
    # E5-06 (task #290): the graded recovery-leg checks cross the boundary as a
    # sealed document of their own, so the orchestrator's joint full-mission
    # gate can grade the landing evidence without re-flying anything.
    _write_json(exchange / "reports" / "recovery-checks.json",
                {"kind": RECOVERY_CHECKS_KIND, "schema_version": "1.0",
                 "label": "recovery-leg graded checks and unassessed notes (E5-06 joint gate)",
                 "success": bool(block["success"]), "landed": bool(block.get("landed")),
                 "checks": [{"id": c.id, "passed": bool(c.passed), "value": c.value,
                             "limit": c.limit, "unit": c.unit, "reason_code": c.reason_code}
                            for c in recovery_checks],
                 "not_assessed": [dict(n) for n in recovery_notes]})

    payloads_spec = [
        {"name": "recovery-run", "type": "flight-run", "schema_version": FLIGHT_SCHEMA_VERSION,
         "path": "recovery/run", "digest_kind": "run-dir"},
        {"name": "recovery-block", "type": "report", "schema_version": "1.0",
         "report_kind": REPORT_WRAPPER_KIND, "path": "reports/recovery-block.json",
         "digest_kind": "file"},
        {"name": "recovery-checks", "type": "report", "schema_version": "1.0",
         "report_kind": RECOVERY_CHECKS_KIND, "path": "reports/recovery-checks.json",
         "digest_kind": "file"},
        {"name": "recovery-handoff-report", "type": "report",
         "schema_version": report_doc["schema_version"],
         "report_kind": report_doc["kind"], "path": "states/recovery-handoff-report.json",
         "digest_kind": "file"},
        {"name": "recovery-final-handoff", "type": "flightlab-handoff-state",
         "schema_version": HANDOFF_SCHEMA_VERSION, "path": "states/recovery-final-handoff.json",
         "digest_kind": "file"},
    ]
    write_envelope(exchange, stage=stage.id, runner=stage.runner, env_label=stage.env,
                   payloads=payloads_spec)

    report = {
        "kind": STAGE_REPORT_KIND, "schema_version": STAGE_REPORT_SCHEMA_VERSION,
        "stage": stage.id, "runner": stage.runner, "env": stage.env,
        "environment": environment_record(stage.env),
        "status": "completed", "mission_ok": bool(block["success"]),
        "summary": (f"success={bool(block['success'])} landed={bool(block.get('landed'))} "
                    f"reason_codes={','.join(block.get('reason_codes') or []) or 'ok'}"),
        "failed_checks": [c.id for c in recovery_checks if not c.passed],
        "payloads": [p["name"] for p in payloads_spec],
    }
    if separation is not None:
        report["driven_from_separation"] = True
        report["separation_handoff_id"] = block["separation_handoff_id"]
    return report


@_register("rendezvous_refly")
def run_rendezvous_refly_stage(plan: CrossEnvPlan, stage: CrossEnvStage, out_dir: Path) -> dict:
    """E12-05 (task #395): re-fly a rendezvous mission's plan as a TudatPy
    two-spacecraft pulse-sequence replay in *this* environment and seal the
    evidence document. The plan's ``scenario`` is the rendezvous mission
    config (``leo_rdv_001.yaml`` and kin); the worker re-verifies the plan in
    this environment (env-parity recheck) so the sealed evidence carries its
    own verdict, not the orchestrator's say-so."""
    from orbitopt.rendezvous.config import load_rendezvous_config
    from orbitopt.verify.rendezvous_refly import refly_stage_impl

    stage_dir = Path(out_dir) / stage.id
    exchange = stage_dir / EXCHANGE_DIRNAME
    exchange.mkdir(parents=True, exist_ok=True)
    cfg = load_rendezvous_config(plan.scenario)
    payloads, doc = refly_stage_impl(cfg, plan.scenario, exchange)
    write_envelope(exchange, stage=stage.id, runner=stage.runner, env_label=stage.env,
                   payloads=payloads)
    j2 = doc["refly"]["j2"]
    return {
        "kind": STAGE_REPORT_KIND, "schema_version": STAGE_REPORT_SCHEMA_VERSION,
        "stage": stage.id, "runner": stage.runner, "env": stage.env,
        "environment": environment_record(stage.env),
        "status": "completed", "mission_ok": bool(doc["ok"]),
        "summary": (f"mission_id={doc['mission_id']} frame={doc['frame']} "
                    f"pos_err_m={j2['position_error_m']:.3f} "
                    f"rel_speed_m_s={j2['relative_speed_m_s']:.4f} "
                    f"min_app_m={j2['min_approach_distance_m']:.1f} "
                    f"dv_m_s={doc['plan']['total_delta_v_m_s']:.3f}"),
        "failed_checks": list(doc["failed_checks"]),
        "payloads": [p["name"] for p in payloads],
    }


@_register("rendezvous_refly_samples")
def run_rendezvous_sample_reflight_stage(plan: CrossEnvPlan, stage: CrossEnvStage,
                                        out_dir: Path) -> dict:
    """E12-10 (task #412): fly the per-sample re-flight spec named by
    ``ORBITOPT_RENDZVOUS_REFLIGHT_SPEC`` in *this* environment (tudatpy) and
    seal the evidence. The worker itself re-checks the spec's digests against
    its own config, so the sealed evidence carries its own provenance."""
    from orbitopt.rendezvous.config import load_rendezvous_config
    from orbitopt.verify.rendezvous_refly import sample_reflight_stage_impl

    stage_dir = Path(out_dir) / stage.id
    exchange = stage_dir / EXCHANGE_DIRNAME
    exchange.mkdir(parents=True, exist_ok=True)
    cfg = load_rendezvous_config(plan.scenario)
    payloads, doc = sample_reflight_stage_impl(cfg, plan.scenario, exchange)
    write_envelope(exchange, stage=stage.id, runner=stage.runner, env_label=stage.env,
                   payloads=payloads)
    return {
        "kind": STAGE_REPORT_KIND, "schema_version": STAGE_REPORT_SCHEMA_VERSION,
        "stage": stage.id, "runner": stage.runner, "env": stage.env,
        "environment": environment_record(stage.env),
        "status": "completed", "mission_ok": bool(doc["flights"]),
        "summary": (f"mission_id={doc['mission_id']} "
                    f"spec_digest={str(doc['spec_digest'])[:12]} "
                    f"flights={len(doc['flights'])} skipped={len(doc['skipped'])}"),
        "failed_checks": [],
        "payloads": [p["name"] for p in payloads],
    }


def execute_stage(plan: CrossEnvPlan, stage: CrossEnvStage, out_dir) -> int:
    """Run one stage and write its ``stage-report.json`` next to its exchange
    dir. Returns a process exit code: 0 = the stage completed (its *mission
    verdict* lives in ``mission_ok`` -- a red landing is a completed stage
    with an honest report, mirroring ``flightlab simulate`` semantics),
    2 = the leg blew up, reported and no sealed exchange left behind."""
    out = Path(out_dir)
    stage_dir = out / stage.id
    stage_dir.mkdir(parents=True, exist_ok=True)
    runner = STAGE_RUNNERS.get(stage.runner)
    if runner is None:
        print(f"crossenv: unknown runner {stage.runner!r} for stage {stage.id!r}",
              file=sys.stderr)
        return 2
    try:
        snapshot_path = stage.options.get("earth_orientation_snapshot")
        snapshot = None
        scope = nullcontext()
        if snapshot_path:
            from orbitopt.reference_eop import freeze_reference, load_snapshot

            snapshot = load_snapshot(Path(snapshot_path))
            scope = freeze_reference(snapshot)
        with scope:
            report = runner(plan, stage, out)
        if snapshot is not None:
            from orbitopt.crossenv.envelope import file_digest

            report["earth_orientation"] = {
                "snapshot_sha256": file_digest(snapshot_path),
                "source_version": snapshot["source_version"], "mode": snapshot["mode"],
            }
        _write_json(stage_dir / "stage-report.json", report)
        return 0
    except Exception as exc:  # noqa: BLE001 -- report, then fail the stage, never traceback-free
        traceback.print_exc()
        _write_json(stage_dir / "stage-report.json", {
            "kind": STAGE_REPORT_KIND, "schema_version": STAGE_REPORT_SCHEMA_VERSION,
            "stage": stage.id, "runner": stage.runner, "env": stage.env,
            "environment": environment_record(stage.env),
            "status": "failed", "mission_ok": False,
            "summary": f"{type(exc).__name__}: {exc}",
            "failed_checks": [], "payloads": [],
        })
        print(f"crossenv stage {stage.id!r} failed: {exc}")
        return 2
