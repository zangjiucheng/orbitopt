"""E5-02 (task #269): coupled branch execution.

The fast tests need no compute stack: they lift the canonical separation
``HandoffState`` off the recorded Gate-B ascent (the same fork fixture
E2-02's tests use), grade the coupled document against it, and drive the
booster branch through the mission chain with a stubbed ``fly_recovery``
that captures the state it was launched from. The acceptance the task names
-- both branches' initial states descend from one digest-traceable
``HandoffState``, mass conserved and epoch/frame continuous at the split,
both branch results in one run -- is asserted against the produced report.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.mission.contract import state_digest, state_to_dict
from orbitopt.mission.coupled_branches import (
    booster_state_from_run,
    coupled_branches,
    separation_handoff,
)
from orbitopt.mission.two_stage_leo import (
    STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE,
    RecoveryPlan,
    TwoStageLeoScenario,
    reference_scenario,
    run_two_stage_leo,
    staging_handoffs,
)
from orbitopt.verify.report import read_report, validate_report

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"
RECOVERY_TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" \
    / "leo_two_stage.yaml"


@pytest.fixture
def gateb():
    return read_run(GATEB)


def _replay(tmp_path, **changes) -> TwoStageLeoScenario:
    scenario = reference_scenario()
    return dataclasses.replace(
        scenario, ascent_mode="run_dir", ascent_run_dir=str(GATEB), ascent={}, **changes)


def _track(run, match_id, **changes):
    return dataclasses.replace(
        run, tracks=[dataclasses.replace(t, **changes) if t.body_id == match_id else t
                     for t in run.tracks])


def _stub_fly_recovery(monkeypatch, *, success=True):
    """Stand in for FL-304 ``fly_recovery`` (same shape as the E2-01 wiring
    test), capturing the separation state it was driven from."""
    from orbitopt.flightlab import checks as gate_checks
    from orbitopt.flightlab.contract import Check, Verification
    from orbitopt.flightlab.recovery.pipeline import RecoveryOutcome

    calls = {}

    def fake(ascent_run, target, vehicle_path, **kw):
        calls.update(target=target, vehicle_path=vehicle_path, kw=kw)
        extra = [Check(id=gate_checks.RECOVERY_GATE_CHECK_ID, passed=success,
                       reason_code="" if success else "INSUFFICIENT_PROPELLANT_BOOSTBACK"),
                 Check(id="recovery-propellant-reserve", passed=True, value=2500.0,
                       limit=1000.0, unit="kg")]
        run = dataclasses.replace(
            ascent_run, verification=Verification(
                optimizer_converged=ascent_run.verification.optimizer_converged,
                physics_verified=ascent_run.verification.physics_verified,
                mission_constraints_passed=(
                    success and ascent_run.verification.mission_constraints_passed),
                checks=tuple(ascent_run.verification.checks) + tuple(extra)))
        return RecoveryOutcome(run=run, success=success, reason_codes=(), plan=None,
                               landing=None, checks=tuple(extra), assumptions=(),
                               handoff=None)

    monkeypatch.setattr("orbitopt.flightlab.recovery.pipeline.fly_recovery", fake)
    return calls


# --- the canonical separation state ------------------------------------------


def test_the_separation_fork_lifts_to_one_digest_identified_handoff(gateb):
    h = separation_handoff(gateb)
    assert h.parent_body == "stage1"
    assert dict(h.branch_bodies) == {"upper-stage": "stage2", "booster": "stage1_debris"}
    stage1 = next(t for t in gateb.tracks if t.body_id == "stage1")
    assert h.t_s == float(stage1.t_s[-1]) == pytest.approx(206.0)
    assert h.separation_event_t_s == pytest.approx(207.0)  # the event trails the sample
    assert h.state.frame == "GCRS"  # frame-continuous across the fork
    assert h.handoff_id == state_digest(h.state) == separation_handoff(gateb).handoff_id
    indep = hashlib.sha256(json.dumps(state_to_dict(h.state), sort_keys=True,
                                      separators=(",", ":"),
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    assert h.handoff_id == indep  # pinned to the canonical bytes, not to this code
    assert h.epoch_utc.tzinfo is not None  # the handoff names an absolute UTC epoch


def test_the_digest_pins_the_split_not_the_naming(gateb):
    want = separation_handoff(gateb).handoff_id
    # FL-304 renames the debris arc to the flown booster: the separation
    # physics did not change, so the handoff id must not either.
    assert separation_handoff(_track(gateb, "stage1_debris", body_id="booster")).handoff_id \
        == want
    # five grams at the final stack sample is a different separation state.
    stage1 = next(t for t in gateb.tracks if t.body_id == "stage1")
    bumped = _track(gateb, "stage1",
                    mass_kg=stage1.mass_kg[:-1] + (stage1.mass_kg[-1] + 5.0,))
    assert separation_handoff(bumped).handoff_id != want


def test_coupled_document_gates_every_branch_on_the_shared_source(gateb):
    scenario = reference_scenario()
    staging = staging_handoffs(gateb, scenario.handoff, scenario.staging_velocity_m_s)
    h = separation_handoff(gateb)
    doc = coupled_branches(gateb, scenario.handoff, scenario.staging_velocity_m_s,
                           staging, handoff=h)
    assert doc["passed"] is True
    assert doc["handoff_id"] == h.handoff_id
    assert doc["driven_from_separation"] is False
    assert doc["mass_residual_kg"] == pytest.approx(0.0, abs=1e-6)
    by_kind = {b["kind"]: b for b in doc["branches"]}
    assert set(by_kind) == {"upper-stage", "booster"}
    for branch in by_kind.values():
        assert branch["handoff_id"] == h.handoff_id  # both initial states trace to one id
        ids = {c["id"] for c in branch["checks"]}
        assert {f"handoff.coupled.{branch['kind']}.{gate}" for gate in
                ("position_continuity", "velocity_continuity", "epoch_monotonic",
                 "mass_conservation", "shared_source")} <= ids
        assert all(c["passed"] for c in branch["checks"])
    assert any(c["id"] == "mass.coupled.branch_mass_split" and c["passed"]
               for c in doc["checks"])


def test_a_fork_that_moves_or_masses_its_way_goes_red(gateb):
    scenario = reference_scenario()

    def grade(run):
        staging = staging_handoffs(run, scenario.handoff, scenario.staging_velocity_m_s)
        return coupled_branches(run, scenario.handoff, scenario.staging_velocity_m_s,
                                staging)

    moved = _track(gateb, "stage2",
                   position_m=((next(t for t in gateb.tracks if t.body_id == "stage2")
                                .position_m[0][0] + 1000.0, 0.0, 0.0),)
                   + next(t for t in gateb.tracks if t.body_id == "stage2").position_m[1:])
    red = {c["id"]: c for c in grade(moved)["checks"]}
    assert red["handoff.coupled.upper-stage.position_continuity"]["passed"] is False
    assert grade(moved)["passed"] is False

    heavier = next(t for t in gateb.tracks if t.body_id == "stage1_debris")
    faked = _track(gateb, "stage1_debris",
                   mass_kg=(heavier.mass_kg[0] + 5000.0,) + heavier.mass_kg[1:])
    red = {c["id"]: c for c in grade(faked)["checks"]}
    assert red["mass.coupled.branch_mass_split"]["passed"] is False


def test_the_booster_branch_is_flown_from_the_shared_state(gateb):
    h = separation_handoff(gateb)
    booster = booster_state_from_run(gateb, h)
    debris = next(t for t in gateb.tracks if t.body_id == "stage1_debris")
    assert booster.frame == "GCRS" and booster.t_s == h.t_s
    # the propagator's mission clock: t_s seconds after the run's reference
    # epoch, landing on the handoff's absolute instant (E5-05 convention fix)
    assert (booster.epoch_utc + timedelta(seconds=booster.t_s)
            == h.epoch_utc.replace(tzinfo=None))
    assert booster.mass_kg == pytest.approx(debris.mass_kg[0])  # its own separation mass
    assert booster.position_m == tuple(h.state.r_m)
    assert booster.velocity_m_s == tuple(h.state.v_m_s)


# --- the mission chain drives (or honestly does not) from the handoff ---------


def test_mission_drives_both_branches_from_one_handoff(tmp_path, monkeypatch):
    calls = _stub_fly_recovery(monkeypatch, success=True)
    scenario = _replay(tmp_path, recovery=RecoveryPlan(target_file=str(RECOVERY_TARGET),
                                                       drive_from_separation=True))
    mission = run_two_stage_leo(scenario, tmp_path / "m", propagate=False)

    # the recovery stage started AT the canonical handoff, at the branch mass
    sep = calls["kw"]["sep_state"]
    want = separation_handoff(read_run(GATEB))
    assert sep.frame == "GCRS" and sep.t_s == want.t_s
    assert sep.mass_kg == pytest.approx(30000.0, rel=1e-9)
    assert sep.position_m == tuple(want.state.r_m)

    ids = {c.id: c for c in mission.checks}
    for want_id in ("handoff.coupled.upper-stage.position_continuity",
                    "handoff.coupled.booster.shared_source",
                    "mass.coupled.branch_mass_split"):
        assert want_id in ids and ids[want_id].passed
    assert mission.recovery["driven_from_separation"] is True
    assert mission.recovery["separation_handoff_id"] == want.handoff_id
    assert mission.coupled_branches["driven_from_separation"] is True
    assert mission.coupled_branches["handoff_id"] == want.handoff_id

    # both branch results land in the same run's records
    by_kind = {b["kind"]: b for b in mission.coupled_branches["branches"]}
    upper = by_kind["upper-stage"]["result"]
    assert isinstance(upper["insertion_passed"], bool)  # the graded verdict, whatever it is
    assert upper["parking_elements"]["periapsis_altitude_m"]
    assert by_kind["upper-stage"]["handoff_id"] == want.handoff_id
    booster = by_kind["booster"]["result"]
    assert booster["success"] is True and booster["driven_from_separation"] is True

    report = read_report(mission.report_path)
    validate_report(report)
    handoffs = report["inputs"]["handoffs"]
    assert handoffs["coupled_branches"]["handoff_id"] == want.handoff_id
    assert handoffs["coupled_branches"]["driven_from_separation"] is True
    # the GATEB ascent is off-target by fixture design: the chain still stops
    # at the parking orbit, the coupled checks never masked it
    assert mission.status == STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE


def test_an_undriven_run_records_the_document_and_grades_no_coupled_checks(
        tmp_path, monkeypatch):
    calls = _stub_fly_recovery(monkeypatch, success=True)
    scenario = _replay(tmp_path, recovery=RecoveryPlan(target_file=str(RECOVERY_TARGET)))
    mission = run_two_stage_leo(scenario, tmp_path / "m", propagate=False)

    assert "sep_state" not in calls["kw"]  # the E2-01 read-out path, untouched
    assert not any(".coupled." in c.id for c in mission.checks)
    assert "driven_from_separation" not in mission.recovery
    doc = mission.coupled_branches
    assert doc is not None and doc["driven_from_separation"] is False and doc["passed"]
    booster = {b["kind"]: b for b in doc["branches"]}["booster"]["result"]
    assert "driven_from_separation" not in booster


def test_drive_without_a_fork_is_a_red_check_not_a_silence(tmp_path, monkeypatch):
    import orbitopt.mission.coupled_branches as cb

    _stub_fly_recovery(monkeypatch, success=True)
    monkeypatch.setattr(cb, "separation_handoff", lambda run: None)
    scenario = _replay(tmp_path, recovery=RecoveryPlan(target_file=str(RECOVERY_TARGET),
                                                       drive_from_separation=True))
    mission = run_two_stage_leo(scenario, tmp_path / "m", propagate=False)
    failed = {c.id for c in mission.checks if not c.passed}
    assert "handoff.coupled.separation_found" in failed


# --- the scenario knob itself --------------------------------------------------


def test_the_recovery_knob_round_trips_and_keeps_the_default_doc_clean(tmp_path):
    assert "drive_from_separation" not in RecoveryPlan().to_dict()
    plan = RecoveryPlan.from_dict({"target_file": "t.yaml", "drive_from_separation": True})
    assert plan.drive_from_separation is True and plan.to_dict()["drive_from_separation"] \
        is True
    scenario = _replay(tmp_path, recovery=plan)
    back = TwoStageLeoScenario.from_dict(scenario.to_dict())
    assert back.recovery.drive_from_separation is True
