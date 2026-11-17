"""E5-04 (task #271): the full physical mission gate.

E2-03's phase gate excuses declared out-of-scope phases -- E2-04 declared the
parking propagation out of scope and still passed (``docs/verification.md``).
The physical gate closes that hole for a COMPUTED mission: five checks --
orbit insertion, parking propagation, landing, mass conservation,
epoch/frame continuity -- are ALL required, each graded independently with its
own tri-state verdict and its own reason codes, and any one of them ``failed``
OR merely ``not_checked`` keeps the whole gate from passing. No new system:
the gate re-grades exactly the checks the chain already produced
(:mod:`orbitopt.mission.phases`), and rides in the same report document.

The fast tests grade the rule over synthetic check sets (no compute stack),
the wiring over the recorded Gate-B replay through the real chain.
"""
from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitopt.mission.contract import HandoffCheck
from orbitopt.mission.phases import (
    INSERTION_ONLY_VERDICT_ID, MIXED_RUN_VERDICT_ID, PHYSICAL_CHECKS, PHYSICAL_LABELS,
    PHYSICAL_NOT_EVIDENCED_CODE, PHYSICAL_RULE, VERDICT_FAILED, VERDICT_NOT_CHECKED,
    VERDICT_PASSED, full_mission_gate, physical_check_of, physical_mission_gate)
from orbitopt.mission.two_stage_leo import (
    reference_scenario, run_two_stage_leo, scenario_warnings)
from orbitopt.verify.report import (RESULT_FAILED, RESULT_NOT_CHECKED, RESULT_PASSED,
                                    validate_report)

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"


def _ck(check_id, passed, reason_code=""):
    return HandoffCheck(id=check_id, passed=passed, value=None, limit=None, unit="",
                        reason_code=reason_code)


def _scenario(recovery=None, propagation_result="passed"):
    return SimpleNamespace(recovery=recovery,
                           expected=SimpleNamespace(propagation_result=propagation_result))


def _mission(checks, *, scenario=None, not_assessed=(), propagation=None):
    return SimpleNamespace(scenario=scenario or _scenario(recovery=object()),
                           checks=tuple(checks), not_assessed=list(not_assessed),
                           propagation=propagation or {"status": "not_run"})


#: One passing check inside every one of the five physical checks -- the
#: green baseline the failure/absence parametrizations mutate.
GREEN = ("station.liftoff", "ascent.insertion_verified", "expected.apoapsis",
         "propagation.ran", "propagation.report_passed", "recovery.landed",
         "mass.stage1.non_increasing", "handoff.stage2.mass_conservation",
         "handoff.stage2.epoch_monotonic", "handoff.coupled.upper-stage.shared_source")

#: One red probe per physical check: (probe check id, its reason code).
RED = {
    "orbit_insertion": ("expected.apoapsis", "out_of_tolerance"),
    "parking_propagation": ("propagation.report_passed", "report_failed"),
    "landing": ("recovery.landed", "TOUCHDOWN_MISSED"),
    "mass_conservation": ("mass.stage1.non_increasing", "mass_gain"),
    "epoch_frame_continuity": ("handoff.stage2.epoch_monotonic", "epoch_backwards"),
}

#: A check id belonging to each physical check (for the absence tests).
MEMBER = {
    "orbit_insertion": "station.liftoff",
    "parking_propagation": "propagation.ran",
    "landing": "recovery.landed",
    "mass_conservation": "mass.stage1.non_increasing",
    "epoch_frame_continuity": "handoff.stage2.epoch_monotonic",
}


def _docs(gate):
    return {d["check"]: d for d in gate["checks"]}


# --- vocabulary ---------------------------------------------------------------

def test_the_five_required_checks_are_named_ordered_and_labelled():
    assert PHYSICAL_CHECKS == ("orbit_insertion", "parking_propagation", "landing",
                               "mass_conservation", "epoch_frame_continuity")
    assert set(PHYSICAL_LABELS) == set(PHYSICAL_CHECKS)
    # the tri-state is the unified evidence vocabulary, not a fourth invention
    assert (RESULT_PASSED, RESULT_FAILED, RESULT_NOT_CHECKED) == (
        VERDICT_PASSED, VERDICT_FAILED, VERDICT_NOT_CHECKED)


@pytest.mark.parametrize("check_id,name", [
    ("station.liftoff", "orbit_insertion"),
    ("ascent.physics_verified", "orbit_insertion"),
    ("expected.raan", "orbit_insertion"),
    ("propagation.ran", "parking_propagation"),
    ("propagation.epoch_forward", "parking_propagation"),
    ("recovery.landed", "landing"),
    ("recovery.handoff.position_continuity", "landing"),
    ("mass.stage2.non_increasing", "mass_conservation"),
    ("mass.separation_fork", "mass_conservation"),
    ("mass.coupled.branch_mass_split", "mass_conservation"),
    ("handoff.stage1.mass_conservation", "mass_conservation"),
    ("parking.mass_conservation", "mass_conservation"),
    ("propagation.mass_conservation", "mass_conservation"),
    ("recovery.handoff.mass_conservation", "mass_conservation"),
    ("handoff.stage2.epoch_monotonic", "epoch_frame_continuity"),
    ("parking.position_continuity", "epoch_frame_continuity"),
    ("handoff.coupled.booster.shared_source", "epoch_frame_continuity"),
])
def test_every_chain_namespace_lands_in_exactly_one_physical_check(check_id, name):
    assert physical_check_of(check_id) == name


@pytest.mark.parametrize("check_id", ["digest", "status", "", "ascent", "handoffs"])
def test_ids_outside_the_physical_vocabulary_are_recorded_not_attributed(check_id):
    assert physical_check_of(check_id) is None


def test_every_check_is_required_and_graded_independently():
    docs = _docs(physical_mission_gate(_mission([_ck(i, True) for i in GREEN])))
    assert list(docs) == list(PHYSICAL_CHECKS)
    assert all(d["required"] is True for d in docs.values())
    assert all(d["verdict"] == VERDICT_PASSED for d in docs.values())
    assert all(d["reason_codes"] == [] for d in docs.values())
    # the conjunction itself is the same tri-state rule as the phase gate
    assert physical_mission_gate(_mission([_ck(i, True) for i in GREEN]))["rule"] == PHYSICAL_RULE


# --- one failing check fails the whole gate (and only names its own codes) ----

@pytest.mark.parametrize("name", PHYSICAL_CHECKS)
def test_each_required_check_failing_fails_the_whole_gate(name):
    probe, code = RED[name]
    checks = [_ck(i, True) for i in GREEN if i != probe] + [_ck(probe, False, code)]
    gate = physical_mission_gate(_mission(checks))
    docs = _docs(gate)
    assert docs[name]["verdict"] == VERDICT_FAILED
    assert code in docs[name]["reason_codes"]
    assert gate["verdict"] == VERDICT_FAILED
    assert gate["reason_codes"] == [code]
    # the other four keep their own independent verdict
    for other, doc in docs.items():
        if other != name:
            assert doc["verdict"] == VERDICT_PASSED, other


@pytest.mark.parametrize("name", PHYSICAL_CHECKS)
def test_a_failing_check_without_its_own_code_falls_back_to_its_id(name):
    probe, _ = RED[name]
    checks = [_ck(i, True) for i in GREEN if i != probe] + [_ck(probe, False, "")]
    docs = _docs(physical_mission_gate(_mission(checks)))
    assert docs[name]["reason_codes"] == [probe]


# --- one un-evidenced check caps the whole gate at not_checked -----------------

@pytest.mark.parametrize("name", PHYSICAL_CHECKS)
def test_each_required_check_un_evidenced_caps_the_gate_at_not_checked(name):
    assert physical_check_of(MEMBER[name]) == name
    checks = [_ck(i, True) for i in GREEN if i != MEMBER[name]
              and physical_check_of(i) != name]      # remove the whole group
    gate = physical_mission_gate(_mission(checks))
    docs = _docs(gate)
    assert docs[name]["verdict"] == VERDICT_NOT_CHECKED
    assert docs[name]["reason"] != ""            # always explained
    assert docs[name]["reason_codes"] == [PHYSICAL_NOT_EVIDENCED_CODE[name]]
    assert gate["verdict"] == VERDICT_NOT_CHECKED  # absent evidence is never a pass
    assert PHYSICAL_NOT_EVIDENCED_CODE[name] in gate["reason_codes"]
    for other, doc in docs.items():
        if other != name:
            assert doc["verdict"] == VERDICT_PASSED, other


@pytest.mark.parametrize("inj,prop,land,mass,cont",
                         list(itertools.product((VERDICT_PASSED, VERDICT_FAILED), repeat=5)))
def test_the_gate_passes_only_when_all_five_checks_pass(inj, prop, land, mass, cont):
    verdicts = dict(zip(PHYSICAL_CHECKS, (inj, prop, land, mass, cont)))
    probes = {"orbit_insertion": ("station.liftoff", "station_missing"),
              "parking_propagation": ("propagation.ran", "propagation_unavailable"),
              "landing": ("recovery.landed", "TOUCHDOWN_MISSED"),
              "mass_conservation": ("mass.stage1.non_increasing", "mass_gain"),
              "epoch_frame_continuity": ("handoff.stage2.epoch_monotonic", "epoch_backwards")}
    checks = [_ck(i, True) for i in GREEN]
    for name, verdict in verdicts.items():
        if verdict == VERDICT_FAILED:
            probe, code = probes[name]
            checks = [_ck(probe, False, code) if c.id == probe else c for c in checks]
    gate = physical_mission_gate(_mission(checks))
    expected = VERDICT_FAILED if VERDICT_FAILED in verdicts.values() else VERDICT_PASSED
    assert gate["verdict"] == expected


def test_a_failed_check_beats_a_missing_one():
    # propagation failed, landing never flew: the mission FAILED, it is not
    # merely unassessed (failed > not_checked beats, exactly like the phases)
    gate = physical_mission_gate(_mission(
        [_ck(i, True) for i in GREEN if not i.startswith("recovery.")
         and i != "propagation.report_passed"]
        + [_ck("propagation.report_passed", False, "report_failed")]))
    docs = _docs(gate)
    assert docs["parking_propagation"]["verdict"] == VERDICT_FAILED
    assert docs["landing"]["verdict"] == VERDICT_NOT_CHECKED
    assert gate["verdict"] == VERDICT_FAILED


# --- the E2-04 hole: a declaration is not evidence ------------------------------

def test_declared_out_of_scope_propagation_never_passes_the_physical_gate():
    """The regression E5-04 exists for: E2-04 declared ``propagation_result:
    not_checked``, its phase gate passed -- the physical gate must not."""
    checks = [_ck(i, True) for i in GREEN if not i.startswith("propagation.")]
    gate = physical_mission_gate(_mission(
        checks, scenario=_scenario(recovery=object(), propagation_result="not_checked"),
        propagation={"status": "not_run",
                     "reason": "propagation not requested (expected result: 'not_checked')"}))
    docs = _docs(gate)
    assert docs["parking_propagation"]["verdict"] == VERDICT_NOT_CHECKED
    assert "out of scope" in docs["parking_propagation"]["reason"]
    assert "never waives" in docs["parking_propagation"]["reason"]
    assert gate["verdict"] == VERDICT_NOT_CHECKED


def test_no_recovery_declared_is_not_checked_not_waived():
    checks = [_ck(i, True) for i in GREEN if not i.startswith("recovery.")]
    gate = physical_mission_gate(_mission(checks, scenario=_scenario(recovery=None)))
    docs = _docs(gate)
    assert docs["landing"]["verdict"] == VERDICT_NOT_CHECKED
    assert "no recovery stage declared" in docs["landing"]["reason"]
    assert gate["verdict"] == VERDICT_NOT_CHECKED


def test_declared_but_unflown_recovery_keeps_its_recorded_reason():
    checks = [_ck(i, True) for i in GREEN if not i.startswith("recovery.")]
    gate = physical_mission_gate(_mission(
        checks, not_assessed=[{"id": "recovery",
                               "reason": "the first stage never separated"}]))
    docs = _docs(gate)
    assert docs["landing"]["verdict"] == VERDICT_NOT_CHECKED
    assert docs["landing"]["reason"] == "the first stage never separated"


# --- mixed-run conjunction follows the E2-03 attribution rule ------------------

def test_mixed_run_verdict_cannot_red_grade_insertion_through_the_back_door():
    # touchdown red drags ascent.flight_verified with it; the insertion half is
    # graded by ascent.insertion_verified, the landing by recovery.*
    gate = physical_mission_gate(_mission(
        [_ck(MIXED_RUN_VERDICT_ID, False, "mission_constraints_failed"),
         _ck(INSERTION_ONLY_VERDICT_ID, True)]
        + [_ck(i, True) for i in GREEN if i not in ("ascent.insertion_verified",
                                                    "recovery.landed")]
        + [_ck("recovery.landed", False, "TOUCHDOWN_MISSED")]))
    docs = _docs(gate)
    assert docs["orbit_insertion"]["verdict"] == VERDICT_PASSED
    assert MIXED_RUN_VERDICT_ID not in docs["orbit_insertion"]["checks"]
    assert docs["landing"]["verdict"] == VERDICT_FAILED
    assert gate["verdict"] == VERDICT_FAILED


def test_expendable_flight_verified_still_measures_insertion():
    gate = physical_mission_gate(_mission(
        [_ck(MIXED_RUN_VERDICT_ID, True)]
        + [_ck(i, True) for i in GREEN
           if physical_check_of(i) not in ("orbit_insertion", "landing")],
        scenario=_scenario(recovery=None)))
    docs = _docs(gate)
    assert docs["orbit_insertion"]["checks"][MIXED_RUN_VERDICT_ID] is True
    assert docs["orbit_insertion"]["verdict"] == VERDICT_PASSED
    assert docs["landing"]["verdict"] == VERDICT_NOT_CHECKED
    assert gate["verdict"] == VERDICT_NOT_CHECKED


# --- the chain, wired ----------------------------------------------------------

def _replay(tmp_path):
    scenario = reference_scenario()
    return dataclasses.replace(scenario, ascent_mode="run_dir",
                               ascent_run_dir=str(GATEB), ascent={})


def test_physical_gate_rides_in_the_full_mission_gate_document(tmp_path):
    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07", propagate=False)
    gate = mission.report["inputs"]["full_mission"]
    physical = gate["physical"]
    assert physical["rule"] == PHYSICAL_RULE
    docs = _docs(physical)
    assert list(docs) == list(PHYSICAL_CHECKS)
    # the recorded Gate-B replay is off-target: insertion FAILED with its code
    assert docs["orbit_insertion"]["verdict"] == VERDICT_FAILED
    assert "out_of_tolerance" in docs["orbit_insertion"]["reason_codes"]
    assert physical["verdict"] == VERDICT_FAILED          # failed dominates
    # propagation was declared and skipped -> not_checked with its own reason,
    # landing was never flown -> not_checked: the physical verdict is stricter
    # than a "phase not applicable" excuse even where the phase gate excuses
    assert docs["parking_propagation"]["verdict"] == VERDICT_NOT_CHECKED
    assert docs["parking_propagation"]["reason"] != ""
    assert docs["landing"]["verdict"] == VERDICT_NOT_CHECKED
    assert docs["landing"]["reason_codes"] == ["landing_not_evidenced"]
    assert "landing_not_evidenced" in physical["reason_codes"]
    assert "out_of_tolerance" in physical["reason_codes"]
    validate_report(mission.report)


def test_replayed_declared_out_of_scope_propagation_stays_required_physically(tmp_path):
    """E2-04's shape, replayed: the phase gate passes what it deems
    applicable, the physical gate keeps the out-of-scope propagation required
    and refuses to pass."""
    scenario = dataclasses.replace(
        _replay(tmp_path),
        expected=dataclasses.replace(reference_scenario().expected,
                                     propagation_result="not_checked"))
    mission = run_two_stage_leo(scenario, tmp_path / "m07", propagate=False)
    phase_verdict = mission.full_mission["verdict"]
    physical = mission.full_mission["physical"]
    docs = _docs(physical)
    phases = {p["phase"]: p for p in mission.full_mission["phases"]}
    # the phase gate excuses the declared out-of-scope phase (not applicable)
    assert phases["parking_propagation"]["applicable"] is False
    # the physical gate keeps it required: no evidence, its own reason code
    assert docs["parking_propagation"]["verdict"] == VERDICT_NOT_CHECKED
    assert "never waives" in docs["parking_propagation"]["reason"]
    assert "parking_propagation_not_evidenced" in physical["reason_codes"]
    # this replay is off-target anyway, so the physical verdict is outright red
    assert phase_verdict == VERDICT_FAILED == physical["verdict"]
    # and the warning names the split so a report reader sees both verdicts
    notes = " | ".join(scenario_warnings(mission.scenario))
    assert "E5-04 physical gate" in notes


def test_physical_gate_survives_the_report_digest_round_trip(tmp_path):
    from orbitopt.verify.report import read_report

    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07", propagate=False)
    report = read_report(mission.report_path)
    validate_report(report)
    assert report["inputs"]["full_mission"]["physical"] == \
        mission.report["inputs"]["full_mission"]["physical"]


def test_pipeline_verify_stage_records_the_physical_verdict(tmp_path):
    pytest.importorskip("astropy")
    from orbitopt.pipeline import VERIFY, StageRecorder
    from orbitopt.mission.two_stage_leo import build_from_config

    scenario = _replay(tmp_path)
    config = {"schemaVersion": "1.0",
              "mission": {"id": "m07-two-stage-leo", "title": scenario.title,
                          "kind": "two-stage-leo"},
              "scenario": scenario.to_dict(), "out_dir": str(tmp_path / "m07")}
    build_from_config(dict(config))                     # scene export, once
    recorder = StageRecorder()
    build_from_config(dict(config), recorder=recorder)
    verify = recorder[VERIFY]
    assert verify.convergence["physical_mission_verdict"] == RESULT_FAILED
    assert any("full physical mission gate" in n for n in verify.notes)
    assert any("orbit_insertion=failed" in n for n in verify.notes)


def test_full_mission_gate_still_grades_its_three_phases():
    # E5-04 rides alongside, it does not replace: the phase documents and the
    # phase rule are untouched, and an out-of-scope phase still passes there
    expendable = _scenario(recovery=None, propagation_result="not_checked")
    gate = full_mission_gate(_mission([_ck("expected.apoapsis", True)], scenario=expendable))
    phases = {p["phase"]: p for p in gate["phases"]}
    assert gate["verdict"] == "passed"
    assert phases["parking_propagation"]["applicable"] is False
    assert phases["recovery_touchdown"]["applicable"] is False
    # ... while the physical gate over the same run refuses to pass
    assert gate["physical"]["verdict"] == VERDICT_NOT_CHECKED
