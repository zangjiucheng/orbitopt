"""The one command that flies the mission across both environments.

:func:`run_plan` walks the plan in order, launches each stage in its declared
interpreter (:func:`orbitopt.crossenv.plan.run_stage_subprocess`), and
immediately *verifies* the exchange the stage sealed (schema + every digest
re-computed in the orchestrator's own process, not just trusted from the
stage). The joined verdict lands in ``crossenv.mission-report.json``: per
stage the runner, environment, envelope digest and mission verdict; an
``environment_matrix`` of every involved environment's provenance; and an
``overall`` status that is green only when every stage completed, every
exchange verified, every leg's mission verdict came back green AND the joint
full-mission gate (E5-06, task #290: :func:`orbitopt.crossenv.
qualification.joint_qualification` over the union of the legs' sealed
evidence -- one ``full_mission`` document with the five physical checks and
their evidence sources) came back ``passed``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from orbitopt.crossenv.envelope import (CrossEnvError, envelope_digest, file_digest,
                                        read_envelope)
from orbitopt.crossenv.plan import (RESOLVED_PLAN_FILE, load_plan, run_stage_subprocess,
                                    write_resolved_plan)
from orbitopt.crossenv.stages import STAGE_REPORT_KIND, STAGE_REPORT_SCHEMA_VERSION

MISSION_REPORT_KIND = "orbitopt.crossenv.mission-report"
MISSION_REPORT_SCHEMA_VERSION = "1.0"
MISSION_REPORT_FILE = "crossenv.mission-report.json"
STAGE_REPORT_FILE = "stage-report.json"


def _read_stage_report(stage_dir: Path) -> dict:
    path = stage_dir / STAGE_REPORT_FILE
    if not path.is_file():
        raise CrossEnvError(f"stage dir {str(stage_dir)!r} has no {STAGE_REPORT_FILE} -- the "
                            "stage subprocess died before reporting")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("kind") != STAGE_REPORT_KIND or \
            doc.get("schema_version") != STAGE_REPORT_SCHEMA_VERSION:
        raise CrossEnvError(f"{str(path)!r} is not a {STAGE_REPORT_KIND!r} v"
                            f"{STAGE_REPORT_SCHEMA_VERSION} document")
    return doc


def run_plan(plan_path, *, out=None, env_overrides: Mapping[str, str] | None = None,
             verbose: bool = True) -> dict:
    """Execute a cross-environment plan end to end. Returns the mission
    report (also written to ``<out>/crossenv.mission-report.json``); raises
    :class:`CrossEnvError` if a stage's sealed exchange is missing, foreign
    or tampered -- a broken handoff is an orchestration error, not a mission
    verdict."""
    plan = load_plan(plan_path, env_overrides=env_overrides)
    out_dir = Path(out) if out else plan.out
    if out_dir is None:
        raise CrossEnvError("the plan declares no `out` directory and none was given")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = write_resolved_plan(plan, out_dir / RESOLVED_PLAN_FILE)

    stages: list[dict] = []
    matrix: dict[str, dict] = {}
    overall_ok = True
    for stage in plan.stages:
        stage_dir = out_dir / stage.id
        entry: dict = {"id": stage.id, "runner": stage.runner, "env": stage.env,
                       "python_executable": plan.environments[stage.env],
                       "consumes": list(stage.consumes)}
        if verbose:
            print(f"crossenv: stage {stage.id!r} ({stage.runner}) -> "
                  f"{plan.environments[stage.env]}")
        proc = run_stage_subprocess(plan, stage, out_dir, resolved)
        entry["exit_code"] = proc.returncode
        if proc.returncode != 0 and verbose:
            tail = (proc.stdout or "").strip().splitlines()[-5:]
            err_tail = (proc.stderr or "").strip().splitlines()[-5:]
            for line in (*tail, *err_tail):
                print(f"crossenv:   | {line}")

        report_path = stage_dir / STAGE_REPORT_FILE
        stage_report = _read_stage_report(stage_dir) if report_path.is_file() else None
        exchange_dir = stage_dir / "exchange"
        try:
            envelope = read_envelope(exchange_dir) if (exchange_dir / "exchange.json").is_file() \
                else None
            if envelope is not None:
                entry["exchange"] = str(exchange_dir)
                entry["exchange_digest"] = envelope_digest(exchange_dir)
                entry["payloads"] = {p["name"]: p["digest"] for p in envelope["payloads"]}
                matrix.setdefault(stage.env, dict(envelope["environment"]))
        except CrossEnvError as exc:
            entry["exchange_error"] = str(exc)
            envelope = None
        if envelope is None:
            overall_ok = False
            entry["status"] = "failed"
            entry["mission_ok"] = False
        else:
            entry["status"] = stage_report["status"] if stage_report else "failed"
            entry["mission_ok"] = bool(stage_report and stage_report["mission_ok"]
                                       and proc.returncode == 0)
            if stage_report:
                entry["summary"] = stage_report["summary"]
                entry["failed_checks"] = stage_report["failed_checks"]
            if entry["status"] != "completed" or not entry["mission_ok"]:
                overall_ok = False
        stages.append(entry)

    from orbitopt.crossenv.envelope import environment_record
    from orbitopt.crossenv.qualification import joint_qualification

    joint = joint_qualification(scenario_path=plan.scenario, stages=stages,
                                plan_full_mission=plan.full_mission)
    qualified = joint["verdict"] == "passed"

    report = {
        "kind": MISSION_REPORT_KIND,
        "schema_version": MISSION_REPORT_SCHEMA_VERSION,
        "mission_id": plan.mission_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "plan": str(Path(plan_path)),
        "scenario": str(plan.scenario),
        "stages": stages,
        "environment_matrix": matrix,
        "orchestrator": environment_record("orchestrator"),
        "joint_qualification": joint,
        "overall": {
            "status": "ok" if overall_ok and qualified else "failed",
            "stages_total": len(stages),
            "exchanges_verified": all("exchange_digest" in s for s in stages),
            "mission_ok": overall_ok,
            "full_mission_verdict": joint["verdict"],
            "physical_mission_verdict": joint["physical_mission_verdict"],
            "independently_verified": joint["qualification"]["independently_verified"],
        },
        "report_file": str(out_dir / MISSION_REPORT_FILE),
    }
    (out_dir / MISSION_REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if verbose:
        for entry in stages:
            print(f"crossenv: stage {entry['id']!r}: status={entry['status']} "
                  f"mission_ok={entry['mission_ok']} "
                  f"exchange={entry.get('exchange_digest', entry.get('exchange_error', '-'))}")
        print(f"crossenv: joint full-mission gate: {joint['verdict']} "
              f"(physical {joint['physical_mission_verdict']}, "
              f"independently_verified={joint['qualification']['independently_verified']})")
        for problem in joint["problems"]:
            print(f"crossenv:   ! {problem}")
        print(f"crossenv: overall {report['overall']['status']} "
              f"-> {out_dir / MISSION_REPORT_FILE}")
    return report


def verify_exchange(exchange_dir) -> dict:
    """``orbitopt crossenv verify``: re-read and re-hash one sealed exchange
    and report what it pins. Raises :class:`CrossEnvError` when invalid."""
    envelope = read_envelope(exchange_dir)
    return {
        "exchange": str(Path(exchange_dir)),
        "envelope_digest": file_digest(Path(exchange_dir) / "exchange.json"),
        "stage": envelope["stage"],
        "runner": envelope["runner"],
        "environment": envelope["environment"],
        "payloads": {p["name"]: {"type": p["type"], "digest": p["digest"]}
                     for p in envelope["payloads"]},
        "valid": True,
    }
