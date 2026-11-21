"""E5-06 (task #290): the joint full-mission qualification of a cross-env run.

Fast half: the union of two sealed legs grades exactly one ``full_mission``
gate document (the same ``full_mission_gate`` the single-process path uses),
with the five physical checks, required-phase coverage and evidence sources
(environment + envelope digest) per check. The negative tests are the
acceptance gate of the task: deleting the propagation evidence, breaking the
shared separation state, or fabricating a landing failure must leave the
joint verdict ``failed`` or ``not_checked`` -- never a silent green. The
plan-level ``full_mission`` declaration is schema-checked here too; the real
two-environment acceptance assertions live in tests/test_e505_reference.py's
``TestCrossEnvironmentE505`` and the ``crossenv-check`` CI step.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from orbitopt.crossenv.envelope import envelope_digest, write_envelope
from orbitopt.crossenv.plan import PlanError, load_plan, write_resolved_plan
from orbitopt.crossenv.qualification import (JOINT_KIND, RECOVERY_CHECKS_KIND,
                                             joint_qualification)

REPO = Path(__file__).resolve().parents[1]
FROZEN_ASCENT = REPO / "tests" / "data" / "mission" / "reference" / "e2-04-frozen-mission" / "ascent"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_replay.yaml"
E505_PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e505_plan.yaml"
E2E_PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e2e_plan.yaml"

PHYSICAL_FIVE = ("orbit_insertion", "parking_propagation", "landing",
                 "mass_conservation", "epoch_frame_continuity")


def _c(check_id: str, passed: bool = True) -> dict:
    return {"id": check_id, "passed": bool(passed), "value": None, "limit": None,
            "unit": "", "reason_code": "" if passed else check_id}


ASCENT_CHECKS = [
    _c("station.separation_event"), _c("ascent.flight_verified"),
    _c("ascent.optimizer_converged"), _c("expected.apogee_altitude"),
    _c("expected.perigee_altitude"), _c("handoff.staging.epoch_monotonic"),
    _c("parking.epoch_monotonic"), _c("mass.bookkeeping"),
    _c("propagation.ran"), _c("propagation.epoch_forward"),
    _c("propagation.mass_conservation"), _c("propagation.report_passed"),
]
RECOVERY_CHECKS = [
    _c("recovery.landed"), _c("ascent.insertion_verified"),
    _c("recovery.propellant_reserve"), _c("recovery.touchdown_dispersion"),
    _c("recovery.handoff.epoch_monotonic"), _c("recovery.handoff.mass_conservation"),
]


def _seal_legs(tmp_path: Path, *, ascent_checks=ASCENT_CHECKS,
               recovery_checks=RECOVERY_CHECKS, coupled: bool = False):
    """Seal a synthetic two-leg run: the ascent exchange carries the frozen
    E2-04 computed ascent run plus a wrapped mission report; the recovery
    exchange carries the sealed recovery-checks document. Returns stage
    entries in mission-report shape."""
    asc = tmp_path / "ascent" / "exchange"
    shutil.copytree(FROZEN_ASCENT, asc / "ascent-run")
    payloads = [{"name": "ascent-run", "type": "flight-run", "schema_version": "1.0",
                 "path": "ascent-run", "digest_kind": "run-dir"}]
    if ascent_checks is not None:
        report = {"kind": "orbitopt.crossenv.report", "schema_version": "1.0",
                  "label": "wrapped mission report (synthetic)",
                  "report": {"status": "ok", "checks": list(ascent_checks),
                             "not_assessed": [], "propagation": {"status": "ran"}}}
        (asc / "reports").mkdir(parents=True, exist_ok=True)
        (asc / "reports" / "mission-report.json").write_text(json.dumps(report),
                                                             encoding="utf-8")
        payloads.append({"name": "mission-report", "type": "report", "schema_version": "1.0",
                         "report_kind": "orbitopt.crossenv.report",
                         "path": "reports/mission-report.json", "digest_kind": "file"})
    if coupled:
        (asc / "states").mkdir(parents=True, exist_ok=True)
        (asc / "states" / "coupled-separation-handoff.json").write_text(
            json.dumps({"kind": "orbitopt.mission.separation-handoff", "schema_version": "1.0",
                        "handoff": {"handoff_id": "sha256:" + "1" * 64}}), encoding="utf-8")
        payloads.append({"name": "coupled-separation-handoff", "type": "report",
                         "schema_version": "1.0",
                         "report_kind": "orbitopt.mission.separation-handoff",
                         "path": "states/coupled-separation-handoff.json",
                         "digest_kind": "file"})
    write_envelope(asc, stage="ascent", runner="ascent", env_label="tudat", payloads=payloads)

    rec = tmp_path / "recovery" / "exchange"
    (rec / "reports").mkdir(parents=True, exist_ok=True)
    checks_doc = {"kind": RECOVERY_CHECKS_KIND, "schema_version": "1.0",
                  "label": "recovery-leg checks (synthetic)",
                  "success": all(c["passed"] for c in recovery_checks),
                  "landed": all(c["passed"] for c in recovery_checks
                                if c["id"] == "recovery.landed"),
                  "checks": list(recovery_checks), "not_assessed": []}
    (rec / "reports" / "recovery-checks.json").write_text(json.dumps(checks_doc),
                                                          encoding="utf-8")
    (rec / "reports" / "recovery-block.json").write_text(
        json.dumps({"kind": "orbitopt.crossenv.report", "schema_version": "1.0",
                    "label": "block", "report": {"success": True}}), encoding="utf-8")
    write_envelope(rec, stage="recovery", runner="recovery", env_label="gfold", payloads=[
        {"name": "recovery-checks", "type": "report", "schema_version": "1.0",
         "report_kind": RECOVERY_CHECKS_KIND, "path": "reports/recovery-checks.json",
         "digest_kind": "file"},
        {"name": "recovery-block", "type": "report", "schema_version": "1.0",
         "report_kind": "orbitopt.crossenv.report", "path": "reports/recovery-block.json",
         "digest_kind": "file"},
    ])
    return [
        {"id": "ascent", "runner": "ascent", "env": "tudat",
         "python_executable": sys.executable, "exchange": str(asc),
         "exchange_digest": envelope_digest(asc), "status": "completed"},
        {"id": "recovery", "runner": "recovery", "env": "gfold",
         "python_executable": sys.executable, "exchange": str(rec),
         "exchange_digest": envelope_digest(rec), "status": "completed"},
    ]


def _joint(tmp_path, *, stages=None, plan_full_mission=True, **over) -> dict:
    if stages is None:
        stages = _seal_legs(tmp_path, **over)
    return joint_qualification(scenario_path=SCENARIO, stages=stages,
                               plan_full_mission=plan_full_mission)


def _physical(joint: dict) -> dict:
    return {doc["check"]: doc for doc in joint["physical"]["checks"]}


class TestJointGate:
    def test_green_union_grades_one_full_mission_document(self, tmp_path):
        joint = _joint(tmp_path)
        assert joint["kind"] == JOINT_KIND and joint["schema_version"] == "1.0"
        assert joint["problems"] == []
        assert joint["verdict"] == "passed"
        assert joint["physical_mission_verdict"] == "passed"
        assert joint["coverage"]["complete"] is True
        assert joint["full_mission_declared"] is True
        phys = _physical(joint)
        assert set(phys) == set(PHYSICAL_FIVE)
        for name in PHYSICAL_FIVE:
            assert phys[name]["verdict"] == "passed", name
            assert phys[name]["reason_codes"] == []
            srcs = [s for s in phys[name]["sources"] if s.get("exchange_digest")]
            assert srcs, f"{name}: every passed check names its evidence source"

    def test_evidence_sources_name_environment_and_envelope_digest(self, tmp_path):
        stages = _seal_legs(tmp_path)
        joint = _joint(tmp_path, stages=stages)
        phys = _physical(joint)
        landing = [s for s in phys["landing"]["sources"] if s.get("exchange_digest")]
        assert {s["env"] for s in landing} == {"gfold"}
        assert landing[0]["exchange_digest"] == stages[1]["exchange_digest"]
        assert landing[0]["payload"] == "recovery-checks"
        insertion = {s["env"] for s in phys["orbit_insertion"]["sources"]
                     if s.get("exchange_digest")}
        assert insertion == {"tudat", "gfold"}, "insertion_verified is measured by the landing leg"
        propagation = [s for s in phys["parking_propagation"]["sources"]
                       if s.get("exchange_digest")]
        assert {s["env"] for s in propagation} == {"tudat"}
        for runner in ("ascent", "recovery"):
            ev = joint["evidence"][runner]
            assert ev["exchange_digest"].startswith("sha256:")
            assert "toolchain" in ev["environment"]

    def test_qualification_reports_independent_verified_verdict(self, tmp_path):
        joint = _joint(tmp_path)
        qual = joint["qualification"]
        assert qual["independently_verified"] is False
        # the union of mission-chain checks carries no validation-grade evidence
        # (task #263 evidence semantics): the claim is false WITH a reason.
        assert "independent" in qual["reason"]

    def test_undeclared_run_grades_honestly_but_never_qualifies(self, tmp_path):
        joint = _joint(tmp_path, plan_full_mission=False)
        assert joint["verdict"] == "passed"  # the phases it flies really passed
        assert joint["full_mission_declared"] is False
        assert joint["coverage"]["summary"] == "2/3"  # propagation out of scope
        assert joint["qualification"]["independently_verified"] is False
        # ...and the physical gate still refuses the out-of-scope leg (E5-04)
        assert _physical(joint)["parking_propagation"]["verdict"] == "passed"

    def test_missing_legs_cannot_pass(self, tmp_path):
        joint = joint_qualification(scenario_path=SCENARIO, stages=[], plan_full_mission=True)
        assert joint["verdict"] == "not_checked"
        assert joint["physical_mission_verdict"] == "not_checked"
        assert any("ascent" in p for p in joint["problems"])
        assert any("recovery" in p for p in joint["problems"])


class TestNegativeEvidence:
    """Task #290 acceptance: each fabricated hole must leave the joint
    verdict failed or not_checked -- absent, unverifiable or red evidence is
    never a pass."""

    def test_deleted_propagation_evidence_is_not_checked(self, tmp_path):
        checks = [c for c in ASCENT_CHECKS if not c["id"].startswith("propagation.")]
        joint = _joint(tmp_path, ascent_checks=checks)
        assert joint["verdict"] == "not_checked"
        phys = _physical(joint)
        assert phys["parking_propagation"]["verdict"] == "not_checked"
        assert "parking_propagation_not_evidenced" in phys["parking_propagation"]["reason_codes"]
        assert joint["physical_mission_verdict"] == "not_checked"
        # the required-phase coverage says so too
        assert joint["coverage"]["phases"]["parking_propagation"]["status"] == "not_checked"

    def test_broken_shared_separation_state_is_not_checked(self, tmp_path):
        stages = _seal_legs(tmp_path, coupled=True)
        victim = Path(stages[0]["exchange"]) / "states" / "coupled-separation-handoff.json"
        doc = json.loads(victim.read_text(encoding="utf-8"))
        doc["handoff"]["handoff_id"] = "sha256:" + "0" * 64  # corrupt, no re-seal
        victim.write_text(json.dumps(doc), encoding="utf-8")
        joint = _joint(tmp_path, stages=stages)
        assert joint["verdict"] == "not_checked"
        assert any("digest mismatch" in p for p in joint["problems"])
        phys = _physical(joint)
        # the ascent leg's half of the evidence is gone: no propagation, and
        # the phase gate sees no injection evidence it can attribute
        assert phys["parking_propagation"]["verdict"] == "not_checked"
        assert joint["phases"]["parking_propagation"]["verdict"] == "not_checked"
        assert joint["physical_mission_verdict"] == "not_checked"
        assert phys["landing"]["verdict"] == "passed"  # the surviving leg still grades

    def test_fabricated_landing_failure_is_failed(self, tmp_path):
        checks = [_c("recovery.landed", False)] + RECOVERY_CHECKS[1:]
        joint = _joint(tmp_path, recovery_checks=checks)
        assert joint["verdict"] == "failed"
        assert joint["phases"]["recovery_touchdown"]["verdict"] == "failed"
        assert "recovery.landed" in joint["phases"]["recovery_touchdown"]["reason_codes"]
        assert _physical(joint)["landing"]["verdict"] == "failed"
        assert joint["qualification"]["independently_verified"] is False

    def test_report_digest_pinning_is_enforced(self, tmp_path):
        stages = _seal_legs(tmp_path)
        stages[0]["exchange_digest"] = "sha256:" + "f" * 64  # a report that lies
        joint = _joint(tmp_path, stages=stages)
        assert joint["verdict"] == "not_checked"
        assert any("digest mismatch" in p for p in joint["problems"])


class TestPlanQualificationFlag:
    def test_e505_plan_declares_full_mission(self):
        assert load_plan(E505_PLAN).full_mission is True

    def test_resolved_plan_round_trips_the_flag(self, tmp_path):
        plan = load_plan(E505_PLAN)
        resolved = write_resolved_plan(plan, tmp_path / "plan.resolved.json")
        assert load_plan(resolved).full_mission is True

    def test_e2e_plan_stays_a_partial_run(self):
        assert load_plan(E2E_PLAN).full_mission is False

    def test_full_mission_without_a_recovery_stage_is_refused(self, tmp_path):
        import yaml

        doc = {"kind": "orbitopt.crossenv.plan", "schema_version": "1.0",
               "mission": {"id": "t"}, "full_mission": True,
               "scenario": "src/orbitopt/data/mission_two_stage_leo_recovery.yaml",
               "environments": {"a": "python"},
               "stages": [{"id": "s1", "runner": "ascent", "env": "a"}]}
        path = tmp_path / "plan.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        with pytest.raises(PlanError, match="full_mission"):
            load_plan(path)


@pytest.mark.slow
class TestOrchestratorJointGate:
    """One real single-interpreter orchestration: the joined report carries
    the joint gate document and the overall verdict follows it."""

    def test_run_plan_attaches_the_joint_gate(self, tmp_path):
        from orbitopt.crossenv.orchestrator import run_plan

        report = run_plan(E2E_PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": sys.executable, "gfold": sys.executable},
                          verbose=False)
        joint = report["joint_qualification"]
        assert joint["kind"] == JOINT_KIND
        assert joint["full_mission_declared"] is False  # the e2e plan declares no full mission
        overall = report["overall"]
        assert overall["full_mission_verdict"] == joint["verdict"]
        assert overall["physical_mission_verdict"] == joint["physical_mission_verdict"]
        assert overall["independently_verified"] is joint["qualification"]["independently_verified"]
        assert (overall["status"] == "ok") == (overall["mission_ok"]
                                               and joint["verdict"] == "passed")
        on_disk = json.loads((tmp_path / "cx" / "crossenv.mission-report.json")
                             .read_text(encoding="utf-8"))
        assert on_disk["joint_qualification"]["verdict"] == joint["verdict"]


tudat_py = os.environ.get("ORBITOPT_TUDAT_PYTHON", "")
gfold_py = os.environ.get("ORBITOPT_GFOLD_PYTHON", "")


@pytest.mark.cross_env
@pytest.mark.slow
class TestCrossEnvJointGateReal:
    """The two real environments (CI's cross-env job): the E5-05 reference
    mission across the tudatpy/gfold boundary qualifies through the joint
    gate -- all five physical checks green, each pinned to the digest of the
    envelope of the environment that measured it."""

    @pytest.mark.skipif(not (tudat_py and gfold_py),
                        reason="set ORBITOPT_TUDAT_PYTHON and ORBITOPT_GFOLD_PYTHON")
    def test_reference_mission_joint_qualification(self, tmp_path):
        from orbitopt.crossenv.orchestrator import run_plan

        report = run_plan(E505_PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": tudat_py, "gfold": gfold_py},
                          verbose=False)
        joint = report["joint_qualification"]
        assert joint["verdict"] == "passed", joint["problems"]
        assert joint["physical_mission_verdict"] == "passed"
        assert joint["coverage"]["complete"] is True
        phys = _physical(joint)
        for name in PHYSICAL_FIVE:
            assert phys[name]["verdict"] == "passed", phys[name]
        envs = {name: {s["env"] for s in phys[name]["sources"] if s.get("env")}
                for name in PHYSICAL_FIVE}
        assert envs["parking_propagation"] == {"tudat"}
        assert envs["landing"] == {"gfold"}
        assert report["overall"]["full_mission_verdict"] == "passed"
        assert report["overall"]["status"] == "ok"
