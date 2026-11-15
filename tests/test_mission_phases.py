"""E2-03 (task #256): the full-mission verification gate.

Three phases -- orbit injection, parking-orbit propagation, recovery
touchdown -- each with its own tri-state verdict and failure reason codes, and
one overall verdict that is the conjunction of the applicable phases under the
unified evidence semantics (absent evidence is never a pass). The fast tests
need no compute stack: the rule and the phase attribution are graded over
synthetic check sets, the wiring over the recorded Gate-B replay (and a stubbed
FL-304 stage), the pipeline over the mission builder with propagation declared
out of scope.
"""
from __future__ import annotations

import dataclasses
import itertools
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitopt.mission.contract import HandoffCheck
from orbitopt.mission.phases import (
    INSERTION_ONLY_VERDICT_ID, MIXED_RUN_VERDICT_ID, OVERALL_RULE, PHASES,
    PHASE_ORBIT_INJECTION, PHASE_PARKING_PROPAGATION, PHASE_RECOVERY_TOUCHDOWN,
    PHASE_VERDICTS, VERDICT_FAILED, VERDICT_NOT_CHECKED, VERDICT_PASSED,
    combine_phase_verdicts, full_mission_gate, phase_of, phase_verdicts)
from orbitopt.mission.two_stage_leo import (
    RecoveryPlan, STATUS_RECOVERY_FAILED, TwoStageLeoScenario, build_from_config,
    reference_scenario, run_two_stage_leo, scenario_warnings)
from orbitopt.verify.report import (RESULT_FAILED, RESULT_NOT_CHECKED, RESULT_PASSED,
                                    validate_report)

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"
RECOVERY_TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" \
    / "leo_two_stage.yaml"


def _replay(tmp_path, run_dir=GATEB, **changes) -> TwoStageLeoScenario:
    scenario = reference_scenario()
    return dataclasses.replace(
        scenario, ascent_mode="run_dir", ascent_run_dir=str(run_dir), ascent={}, **changes)


def _stub_fly_recovery(monkeypatch, *, success, reason="INSUFFICIENT_PROPELLANT_BOOSTBACK"):
    """Stand in for FL-304 ``fly_recovery`` (same shape as the E2-01 suite):
    the recovery gate rides on the ascent run's verification, green or red."""
    from orbitopt.flightlab import checks as gate_checks
    from orbitopt.flightlab.contract import Check, Verification
    from orbitopt.flightlab.recovery.pipeline import RecoveryOutcome

    def fake(ascent_run, target, vehicle_path, **kw):
        extra = [Check(id=gate_checks.RECOVERY_GATE_CHECK_ID, passed=success,
                       reason_code="" if success else reason),
                 Check(id="recovery-propellant-reserve", passed=True, value=2500.0,
                       limit=1000.0, unit="kg")]
        run = dataclasses.replace(
            ascent_run, verification=Verification(
                optimizer_converged=ascent_run.verification.optimizer_converged,
                physics_verified=ascent_run.verification.physics_verified,
                mission_constraints_passed=(
                    success and ascent_run.verification.mission_constraints_passed),
                checks=tuple(ascent_run.verification.checks) + tuple(extra)))
        return RecoveryOutcome(run=run, success=success,
                               reason_codes=() if success else (reason,),
                               plan=None, landing=None, checks=tuple(extra),
                               assumptions=(), handoff=None)

    monkeypatch.setattr("orbitopt.flightlab.recovery.pipeline.fly_recovery", fake)


def _ck(check_id, passed, reason_code=""):
    return HandoffCheck(id=check_id, passed=passed, value=None, limit=None, unit="",
                        reason_code=reason_code)


# --- vocabulary ---------------------------------------------------------------

def test_phase_verdict_vocabulary_is_the_unified_evidence_tristate():
    # E4-01 (task #243) unified what a verdict means; the gate must not invent
    # a fourth state or rename the three.
    assert (VERDICT_PASSED, VERDICT_FAILED, VERDICT_NOT_CHECKED) == (
        RESULT_PASSED, RESULT_FAILED, RESULT_NOT_CHECKED)
    assert PHASE_VERDICTS == (RESULT_PASSED, RESULT_FAILED, RESULT_NOT_CHECKED)


def test_gate_reports_all_three_phases_in_chain_order():
    assert PHASES == ("orbit_injection", "parking_propagation", "recovery_touchdown")


@pytest.mark.parametrize("check_id,phase", [
    ("station.liftoff", PHASE_ORBIT_INJECTION),
    ("station.event_chain_order", PHASE_ORBIT_INJECTION),
    ("ascent.flight_verified", PHASE_ORBIT_INJECTION),
    (INSERTION_ONLY_VERDICT_ID, PHASE_ORBIT_INJECTION),
    ("handoff.stage1.position_continuity", PHASE_ORBIT_INJECTION),
    ("parking.mass_conservation", PHASE_ORBIT_INJECTION),
    ("mass.bookkeeping", PHASE_ORBIT_INJECTION),
    ("mass.separation_fork", PHASE_ORBIT_INJECTION),
    ("expected.apoapsis", PHASE_ORBIT_INJECTION),
    ("propagation.ran", PHASE_PARKING_PROPAGATION),
    ("propagation.report_passed", PHASE_PARKING_PROPAGATION),
    ("recovery.landed", PHASE_RECOVERY_TOUCHDOWN),
    ("recovery.handoff.position_continuity", PHASE_RECOVERY_TOUCHDOWN),
    ("recovery.landing-tilt", PHASE_RECOVERY_TOUCHDOWN),
])
def test_every_chain_namespace_lands_in_exactly_one_phase(check_id, phase):
    assert phase_of(check_id) == phase


@pytest.mark.parametrize("check_id", ["digest", "status", "", "ascent", "handoffs"])
def test_ids_outside_the_phase_vocabulary_are_recorded_not_attributed(check_id):
    assert phase_of(check_id) is None


# --- per-phase verdicts ---------------------------------------------------------

def test_each_phase_is_graded_only_by_its_own_checks():
    docs = phase_verdicts(
        [_ck("station.liftoff", True), _ck("expected.apoapsis", False, "out_of_tolerance"),
         _ck("propagation.ran", True), _ck("recovery.landed", False, "TOUCHDOWN_MISSED")],
        applicable={p: True for p in PHASES})
    inj, prop, rec = (docs[p] for p in PHASES)
    assert (inj["verdict"], prop["verdict"], rec["verdict"]) == (
        VERDICT_FAILED, VERDICT_PASSED, VERDICT_FAILED)
    assert inj["reason_codes"] == ["out_of_tolerance"]
    assert rec["reason_codes"] == ["TOUCHDOWN_MISSED"]
    assert prop["checks"] == {"propagation.ran": True}
    assert rec["checks_total"] == 1 and rec["checks_passed"] == 0


def test_a_phase_with_no_checks_is_not_checked_never_defaulted_to_passed():
    docs = phase_verdicts([_ck("station.liftoff", True)],
                          applicable={p: True for p in PHASES},
                          reasons={PHASE_RECOVERY_TOUCHDOWN: "the booster never flew"})
    assert docs[PHASE_RECOVERY_TOUCHDOWN]["verdict"] == VERDICT_NOT_CHECKED
    assert docs[PHASE_RECOVERY_TOUCHDOWN]["reason"] == "the booster never flew"
    assert docs[PHASE_PARKING_PROPAGATION]["verdict"] == VERDICT_NOT_CHECKED
    assert docs[PHASE_PARKING_PROPAGATION]["reason"] != ""      # always explained


@pytest.mark.parametrize("check_id", ["expected.apoapsis", "recovery.landed"])
def test_failed_check_without_its_own_reason_code_falls_back_to_its_id(check_id):
    docs = phase_verdicts([_ck(check_id, False, "")], applicable={p: True for p in PHASES})
    phase = phase_of(check_id)
    assert docs[phase]["reason_codes"] == [check_id]


def test_mixed_run_verdict_measures_injection_alone_until_recovery_runs():
    without = phase_verdicts([_ck(MIXED_RUN_VERDICT_ID, True)],
                             applicable={p: True for p in PHASES})
    assert without[PHASE_ORBIT_INJECTION]["checks"] == {MIXED_RUN_VERDICT_ID: True}
    # once recovery runs, the run-level conjunction spans both branches of the
    # fork; its halves are graded by insertion_verified and recovery.*, and the
    # mixed verdict is recorded but attributed to neither phase.
    with_recovery = phase_verdicts(
        [_ck(MIXED_RUN_VERDICT_ID, False), _ck(INSERTION_ONLY_VERDICT_ID, True),
         _ck("recovery.landed", False, "TOUCHDOWN_MISSED")],
        applicable={p: True for p in PHASES})
    assert MIXED_RUN_VERDICT_ID not in with_recovery[PHASE_ORBIT_INJECTION]["checks"]
    assert with_recovery[PHASE_ORBIT_INJECTION]["checks"] == {INSERTION_ONLY_VERDICT_ID: True}
    # task semantics 入轨成功 + 回收失败: local injection success, overall failure
    assert with_recovery[PHASE_ORBIT_INJECTION]["verdict"] == VERDICT_PASSED
    assert with_recovery[PHASE_RECOVERY_TOUCHDOWN]["verdict"] == VERDICT_FAILED
    assert combine_phase_verdicts(with_recovery.values()) == VERDICT_FAILED


# --- the overall rule, every combination ---------------------------------------

@pytest.mark.parametrize("inj,prop,rec",
                         list(itertools.product(PHASE_VERDICTS, repeat=3)))
def test_overall_verdict_rule_covers_every_phase_combination(inj, prop, rec):
    docs = [{"phase": p, "applicable": True, "verdict": v, "reason_codes": []}
            for p, v in zip(PHASES, (inj, prop, rec))]
    got = combine_phase_verdicts(docs)
    verdicts = (inj, prop, rec)
    if VERDICT_FAILED in verdicts:
        assert got == VERDICT_FAILED          # a failed phase dominates, always
    elif VERDICT_NOT_CHECKED in verdicts:
        assert got == VERDICT_NOT_CHECKED     # absent evidence is never a pass
    else:
        assert got == VERDICT_PASSED          # all three graded green


def test_a_phase_the_mission_does_not_fly_never_blocks_the_overall_verdict():
    docs = phase_verdicts([_ck(INSERTION_ONLY_VERDICT_ID, True), _ck("propagation.ran", True)],
                          applicable={PHASE_ORBIT_INJECTION: True,
                                      PHASE_PARKING_PROPAGATION: True,
                                      PHASE_RECOVERY_TOUCHDOWN: "expendable mission"})
    assert docs[PHASE_RECOVERY_TOUCHDOWN]["verdict"] == VERDICT_NOT_CHECKED
    assert docs[PHASE_RECOVERY_TOUCHDOWN]["applicable"] is False
    assert "expendable mission" in docs[PHASE_RECOVERY_TOUCHDOWN]["reason"]
    assert combine_phase_verdicts(docs.values()) == VERDICT_PASSED
    # ... but a phase the mission DOES fly and that never ran caps at not_checked
    docs = phase_verdicts([_ck(MIXED_RUN_VERDICT_ID, True)],
                          applicable={p: True for p in PHASES})
    assert combine_phase_verdicts(docs.values()) == VERDICT_NOT_CHECKED


# --- the gate document over a mission -------------------------------------------

def _mission(scenario, checks, *, not_assessed=(), propagation=None):
    return SimpleNamespace(scenario=scenario, checks=tuple(checks),
                           not_assessed=list(not_assessed),
                           propagation=propagation or {"status": "not_run"})


@pytest.mark.parametrize('physics', [True, False, None])
@pytest.mark.parametrize('with_recovery', [False, True])
def test_mission_preserves_the_independent_ascent_physics_verdict(tmp_path, monkeypatch, physics, with_recovery):
    from orbitopt.flightlab.contract import read_run
    from orbitopt.mission_config import load_mission_config
    from orbitopt.mission.two_stage_leo import scenario_from_config
    import orbitopt.mission.two_stage_leo as chain

    reference = REPO / 'tests/data/mission/reference/e2-04-frozen-mission/ascent'
    run = read_run(reference)
    run = dataclasses.replace(run, verification=dataclasses.replace(
        run.verification, physics_verified=physics))
    monkeypatch.setattr(chain, '_acquire_ascent', lambda *args: (run, None, None))
    scenario = scenario_from_config(load_mission_config(
        REPO / 'src/orbitopt/data/mission_two_stage_leo_recovery.yaml'))
    if with_recovery:
        _stub_fly_recovery(monkeypatch, success=True)
    else:
        scenario = dataclasses.replace(scenario, recovery=None)
    reference_script = runpy.run_path(str(REPO / 'scripts/e204_reference.py'))
    with reference_script['frozen_earth_orientation']():
        mission = run_two_stage_leo(scenario, tmp_path / 'mission', propagate=False)
    status = mission.report['status']
    injection = next(p for p in mission.full_mission['phases']
                     if p['phase'] == PHASE_ORBIT_INJECTION)
    if physics is None:
        assert 'ascent.physics_verified' not in status['checks']
        assert status['independent_validation']['result'] == RESULT_NOT_CHECKED
        assert status['independent_validation']['checks_total'] == 0
        assert any(n['id'] == 'ascent.physics_verified' for n in mission.not_assessed)
    else:
        expected = RESULT_PASSED if physics else RESULT_FAILED
        assert status['checks']['ascent.physics_verified'] is physics
        assert status['check_evidence']['ascent.physics_verified'] == 'independent_model'
        assert status['independent_validation']['result'] == expected
        assert status['independent_validation']['checks_total'] == 1
        assert status['result'] == injection['verdict'] == mission.full_mission['verdict'] == expected
        if not physics:
            assert 'physics_verification_failed' in injection['reason_codes']
    validate_report(mission.report)


def test_gate_reads_applicability_from_the_scenario_not_from_the_run():
    expendable = SimpleNamespace(recovery=None,
                                 expected=SimpleNamespace(propagation_result="not_checked"))
    gate = full_mission_gate(_mission(expendable, [_ck("expected.apoapsis", True)]))
    phases = {p["phase"]: p for p in gate["phases"]}
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["applicable"] is False
    assert "no recovery stage declared" in phases[PHASE_RECOVERY_TOUCHDOWN]["reason"]
    assert phases[PHASE_PARKING_PROPAGATION]["applicable"] is False
    assert "not_checked" in phases[PHASE_PARKING_PROPAGATION]["reason"]
    assert gate["verdict"] == VERDICT_PASSED       # every applicable phase green
    assert gate["rule"] == OVERALL_RULE


def test_declared_but_unflown_phases_are_not_checked_with_their_reason():
    scenario = SimpleNamespace(recovery=SimpleNamespace(),
                               expected=SimpleNamespace(propagation_result="passed"))
    gate = full_mission_gate(_mission(
        scenario, [_ck(MIXED_RUN_VERDICT_ID, True)],
        not_assessed=[{"id": "recovery",
                       "reason": "the first stage never separated"}],
        propagation={"status": "not_run", "reason": "skipped by caller"}))
    phases = {p["phase"]: p for p in gate["phases"]}
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["verdict"] == VERDICT_NOT_CHECKED
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["reason"] == "the first stage never separated"
    assert phases[PHASE_PARKING_PROPAGATION]["reason"] == "skipped by caller"
    assert gate["verdict"] == VERDICT_NOT_CHECKED


def test_overall_reason_codes_are_the_union_of_the_failed_phases():
    scenario = SimpleNamespace(recovery=SimpleNamespace(),
                               expected=SimpleNamespace(propagation_result="passed"))
    gate = full_mission_gate(_mission(
        scenario, [_ck("expected.apoapsis", False, "out_of_tolerance"),
                   _ck("recovery.landed", False, "TOUCHDOWN_MISSED")]))
    assert gate["verdict"] == VERDICT_FAILED
    assert gate["reason_codes"] == ["TOUCHDOWN_MISSED", "out_of_tolerance"]


# --- the chain, wired ------------------------------------------------------------

def test_replayed_reference_mission_carries_the_gate(tmp_path):
    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07", propagate=False)

    gate = mission.report["inputs"]["full_mission"]
    assert gate["verdict"] == mission.full_mission["verdict"] == VERDICT_FAILED
    phases = {p["phase"]: p for p in gate["phases"]}
    # injection: red because the recorded flight is off-target (the chain
    # status says which link, the phase says which checks)
    assert phases[PHASE_ORBIT_INJECTION]["verdict"] == VERDICT_FAILED
    assert "out_of_tolerance" in phases[PHASE_ORBIT_INJECTION]["reason_codes"]
    # propagation: declared, skipped by the caller -> not_checked, capped and explained
    assert phases[PHASE_PARKING_PROPAGATION]["verdict"] == VERDICT_NOT_CHECKED
    assert phases[PHASE_PARKING_PROPAGATION]["reason"] != ""
    # recovery: this mission flies none -> not applicable, never failed
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["applicable"] is False
    assert "expendable mission" in phases[PHASE_RECOVERY_TOUCHDOWN]["reason"]
    assert "full-mission gate" in " | ".join(scenario_warnings(mission.scenario))
    assert mission.to_dict()["full_mission"]["verdict"] == VERDICT_FAILED
    validate_report(mission.report)


def test_recovery_failure_keeps_the_injection_phase_verdict_its_own(tmp_path, monkeypatch):
    _stub_fly_recovery(monkeypatch, success=False)
    scenario = _replay(tmp_path, recovery=RecoveryPlan(target_file=str(RECOVERY_TARGET)))
    mission = run_two_stage_leo(scenario, tmp_path / "m07", propagate=False)

    assert mission.status == STATUS_RECOVERY_FAILED
    phases = {p["phase"]: p for p in mission.full_mission["phases"]}
    rec = phases[PHASE_RECOVERY_TOUCHDOWN]
    assert rec["verdict"] == VERDICT_FAILED
    assert rec["reason_codes"] == ["INSUFFICIENT_PROPELLANT_BOOSTBACK"]
    # the merged run grades touchdown inside ascent.flight_verified; the gate
    # must not let that verdict red-grade injection through the back door
    assert phases[PHASE_ORBIT_INJECTION]["checks"].get(MIXED_RUN_VERDICT_ID) is None
    assert INSERTION_ONLY_VERDICT_ID in phases[PHASE_ORBIT_INJECTION]["checks"]
    assert mission.full_mission["verdict"] == VERDICT_FAILED
    assert {"INSUFFICIENT_PROPELLANT_BOOSTBACK", "out_of_tolerance"} <= set(
        mission.full_mission["reason_codes"])


def test_green_recovery_makes_the_recovery_phase_pass(tmp_path, monkeypatch):
    _stub_fly_recovery(monkeypatch, success=True)
    scenario = _replay(tmp_path, recovery=RecoveryPlan(target_file=str(RECOVERY_TARGET)))
    mission = run_two_stage_leo(scenario, tmp_path / "m07", propagate=False)

    phases = {p["phase"]: p for p in mission.full_mission["phases"]}
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["applicable"] is True
    assert phases[PHASE_RECOVERY_TOUCHDOWN]["verdict"] == VERDICT_PASSED
    # the recorded insertion is still off-target: local recovery success, overall failed
    assert phases[PHASE_ORBIT_INJECTION]["verdict"] == VERDICT_FAILED
    assert mission.full_mission["verdict"] == VERDICT_FAILED


def test_gate_survives_report_validation_and_digest_round_trip(tmp_path):
    from orbitopt.verify.report import read_report

    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07", propagate=False)
    report = read_report(mission.report_path)
    validate_report(report)
    assert report["inputs"]["full_mission"] == mission.report["inputs"]["full_mission"]


def test_pipeline_verify_stage_records_the_full_mission_verdict(tmp_path):
    pytest.importorskip("astropy")  # the scene export at the end of the builder
    from orbitopt.pipeline import VERIFY, StageRecorder

    scenario = _replay(tmp_path, expected=dataclasses.replace(
        reference_scenario().expected, propagation_result="not_checked"))
    config = {"schemaVersion": "1.0",
              "mission": {"id": "m07-two-stage-leo", "title": scenario.title,
                          "kind": "two-stage-leo"},
              "scenario": scenario.to_dict(), "out_dir": str(tmp_path / "m07")}
    scene = build_from_config(dict(config))
    assert scene["id"] == "m07-two-stage-leo"

    recorder = StageRecorder()
    build_from_config(dict(config), recorder=recorder)
    verify = recorder[VERIFY]
    assert verify.convergence["full_mission_verdict"] == VERDICT_FAILED   # off-target replay
    assert any("full-mission gate" in n for n in verify.notes)
