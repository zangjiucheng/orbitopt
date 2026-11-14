"""E2-02 (task #255): the branched mission timeline.

From separation on the mission is two trajectories, not one, and both must
start from the same handoff: the stack's state at separation. These tests
grade the **recorded** Gate-B two-stage ascent (the same compute-free fixture
M-07 uses) through :mod:`orbitopt.mission.branches`: the fork must be found,
both branch boundaries must be graded and continuous under M-01, and the
split must conserve mass. A tampered fixture (a branch whose first sample
jumps, or whose mass was fabricated) must come back red, never quietly.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.mission.branches import (
    BRANCH_BOOSTER, BRANCH_LABELS, BRANCH_UPPER_STAGE,
    branch_handoffs, fork_lanes, separation_fork)
from orbitopt.mission.two_stage_leo import reference_scenario, staging_handoffs

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"


@pytest.fixture
def gateb():
    return read_run(GATEB)


@pytest.fixture
def tolerances():
    return reference_scenario().handoff


def _docs(run, tolerances):
    return staging_handoffs(run, tolerances, reference_scenario().staging_velocity_m_s)


# --- finding the fork ---------------------------------------------------------


def test_fork_is_the_stack_shedding_two_children(gateb):
    fork = separation_fork(gateb)
    assert fork is not None
    assert fork["parent_body"] == "stage1"          # the ascent stack, at separation
    assert fork["t_s"] == pytest.approx(207.0)      # the recorded separation event
    by_kind = {b["kind"]: b for b in fork["branches"]}
    assert set(by_kind) == {BRANCH_UPPER_STAGE, BRANCH_BOOSTER}
    assert by_kind[BRANCH_UPPER_STAGE]["body_id"] == "stage2"
    assert by_kind[BRANCH_BOOSTER]["body_id"] == "stage1_debris"


def test_fork_lanes_are_upper_stage_above_the_first_stage(gateb):
    lanes = fork_lanes(separation_fork(gateb))
    assert [lane["id"] for lane in lanes] == [BRANCH_UPPER_STAGE, BRANCH_BOOSTER]
    assert [lane["label"] for lane in lanes] == [
        BRANCH_LABELS[BRANCH_UPPER_STAGE], BRANCH_LABELS[BRANCH_BOOSTER]]
    assert [lane["bodyId"] for lane in lanes] == ["stage2", "stage1_debris"]
    assert fork_lanes(None) == []


def test_a_fairing_off_a_branch_is_not_a_fork(gateb):
    # stage2_fairing is a child of the *branch* (stage2): the mission fork is
    # still the stack's split, and the fairing is not one of its branches.
    fork = separation_fork(gateb)
    assert "stage2_fairing" not in {b["body_id"] for b in fork["branches"]}


def test_a_run_that_shed_only_one_child_never_forked(gateb):
    single = replace(gateb, tracks=tuple(
        t for t in gateb.tracks if t.body_id != "stage1_debris"))
    assert separation_fork(single) is None
    assert branch_handoffs(single, reference_scenario().handoff,
                           reference_scenario().staging_velocity_m_s, []) is None


# --- grading the fork ---------------------------------------------------------


def test_both_branch_boundaries_are_graded_and_continuous(gateb, tolerances):
    docs = {d["child_body"]: d for d in _docs(gateb, tolerances)}
    for body in ("stage2", "stage1_debris"):     # E2-02: both branches of the fork
        assert docs[body]["graded"] is True
        assert docs[body]["passed"] is True
    assert docs["stage2_fairing"]["graded"] is False   # shed off a branch: recorded


def test_fork_doc_carries_both_handoffs_and_passes_on_the_recorded_flight(gateb, tolerances):
    docs = _docs(gateb, tolerances)
    fork = branch_handoffs(gateb, tolerances,
                           reference_scenario().staging_velocity_m_s, docs)
    assert fork["passed"] is True
    assert fork["t_s"] == pytest.approx(207.0)
    assert [b["kind"] for b in fork["branches"]] == [BRANCH_UPPER_STAGE, BRANCH_BOOSTER]
    assert all(b["handoff"] is not None and b["graded"] for b in fork["branches"])
    assert [c["id"] for c in fork["checks"]] == ["mass.separation_fork"]
    assert fork["checks"][0]["passed"] is True


def test_fork_conserves_mass_on_the_recorded_flight(gateb, tolerances):
    fork = branch_handoffs(gateb, tolerances,
                           reference_scenario().staging_velocity_m_s,
                           _docs(gateb, tolerances))
    # upper stage + first stage at separation = the stack before the split
    assert fork["mass_sum_kg"] == pytest.approx(fork["parent_mass_kg"], abs=1e-3)
    assert fork["mass_residual_kg"] == pytest.approx(0.0, abs=1e-3)


def test_a_branch_whose_first_state_jumps_fails_the_fork(gateb, tolerances):
    # 10 km of drift between the stack's last sample and the branch's first
    # is a wiring error, not physics: the branch boundary must go red.
    tampered = tuple(
        replace(t, position_m=tuple(
            tuple(row) for row in (np.asarray(t.position_m, dtype=float)
                                   + np.array([10e3, 0.0, 0.0]))))
        if t.body_id == "stage1_debris" else t
        for t in gateb.tracks)
    run = replace(gateb, tracks=tampered)
    fork = branch_handoffs(run, tolerances,
                           reference_scenario().staging_velocity_m_s, _docs(run, tolerances))
    booster = next(b for b in fork["branches"] if b["kind"] == BRANCH_BOOSTER)
    assert booster["handoff"]["passed"] is False
    assert fork["passed"] is False


def test_a_fabricated_branch_mass_fails_the_conservation_gate(gateb, tolerances):
    tampered = tuple(
        replace(t, mass_kg=tuple(
            np.asarray(t.mass_kg, dtype=float) - 1000.0))
        if t.body_id == "stage1_debris" else t
        for t in gateb.tracks)
    run = replace(gateb, tracks=tampered)
    fork = branch_handoffs(run, tolerances,
                           reference_scenario().staging_velocity_m_s, _docs(run, tolerances))
    assert fork["mass_residual_kg"] == pytest.approx(-1000.0, abs=1e-3)
    assert fork["checks"][0]["passed"] is False
    assert fork["checks"][0]["reason_code"] == "mass_not_conserved_across_fork"
    # the boundary itself is still continuous -- mass bookkeeping is the fork's
    # own gate, invisible to the per-boundary checks
    assert all(b["handoff"]["passed"] for b in fork["branches"])


def test_a_branch_without_a_handoff_doc_is_reported_never_skipped(gateb, tolerances):
    docs = [d for d in _docs(gateb, tolerances) if d["child_body"] != "stage1_debris"]
    fork = branch_handoffs(gateb, tolerances,
                           reference_scenario().staging_velocity_m_s, docs)
    assert fork["passed"] is False
    assert "stage1_debris" in fork["reason"]
    booster = next(b for b in fork["branches"] if b["kind"] == BRANCH_BOOSTER)
    assert booster["handoff"] is None and booster["passed"] is None
