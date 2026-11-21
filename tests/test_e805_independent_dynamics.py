"""E8-05 independent dynamics validation of the E5-07 Starship-class reference.

Gate for :mod:`orbitopt.verify.independent_dynamics`: the ascent twin and the
two landing re-flights must agree with the sealed reference inside the
declared bands, the error budget must add up with a *reported* residual, the
independent-model evidence must really be independent models (never the
OpenTsiolkovsky raw-output self-comparison), and no check may carry
historical-telemetry evidence. Building the report flies the twin four times
(nominal plus three probes) in parallel, so the report-dependent tests are
marked ``slow``; the pure-table tests run in the fast suite.
"""
from __future__ import annotations

import math

import pytest

from orbitopt.verify.independent_dynamics import (
    REFERENCE_ROOT, declared_tolerances, verify_independent_dynamics)
from orbitopt.verify.report import EVIDENCE_TYPES

NOT_RECORDED = pytest.mark.skipif(
    not (REFERENCE_ROOT / "PROVENANCE.json").is_file(),
    reason="E5-07 reference not recorded (scripts/e5-07-reference.py update)")

_TWIN_LINES = ("engine", "aero", "numerical", "frame_time", "controller",
               "residual")


@pytest.fixture(scope="module")
def report() -> dict:
    # the landing re-flight needs the pinned convex solver; jobs without it
    # (Python 3.10 layer) skip, the gfold job runs it unfiltered
    pytest.importorskip("gfold")
    return verify_independent_dynamics()


def test_declared_tolerances_are_auditable() -> None:
    tol = declared_tolerances()
    assert tol, "no bands declared"
    for cid, band in tol.items():
        assert isinstance(band["basis"], str) and len(band["basis"]) >= 40, cid
        assert band["limit"] > 0.0, cid
        assert band["unit"], cid
    # the acceptance surface: MECO / separation / SECO + debris, landing pair
    for cid in ("ascent.meco.state_position", "ascent.meco.state_velocity",
                "ascent.meco.mass", "ascent.separation.state_position",
                "ascent.separation.debris_mass", "ascent.seco.state_position",
                "ascent.seco.state_velocity", "ascent.debris_impact.position",
                "landing.cutoff_position", "landing.touchdown_position",
                "landing.touchdown_vertical_speed", "landing.propellant"):
        assert cid in tol, f"declared band missing for {cid}"


def test_error_budget_frame_and_controller_lines_are_declared_zero() -> None:
    # The twin flies the same authored program at the same epoch with the same
    # Earth rate: the frame/time and controller lines are *declared* zero (and
    # the residual absorbs the schedule-tracking implementation difference),
    # never fitted. The three measured lines must exist for every budgeted
    # quantity in the source module.
    import inspect
    import orbitopt.verify.independent_dynamics as m
    src = inspect.getsource(m.verify_independent_dynamics)
    assert 'lines["frame_time"] = 0.0' in src
    assert 'lines["controller"] = 0.0' in src


@NOT_RECORDED
@pytest.mark.slow
def test_report_verdict_and_evidence_taxonomy(report: dict) -> None:
    assert report["schemaVersion"] == "1.0"
    assert report["kind"] == "independent-dynamics"
    status = report["status"]
    assert status["result"] == "passed" and status["passed"] is True
    assert len(status["checks"]) >= 20
    assert all(status["checks"].values())
    assert set(status["check_evidence"].values()) <= set(EVIDENCE_TYPES)
    # a public-estimate vehicle has no telemetry: no check may claim it
    assert "historical_telemetry" not in status["check_evidence"].values()
    assert status["evidence"]["historical_telemetry"]["result"] == "not_checked"
    # the acceptance events carry independent-model evidence ...
    ev = status["check_evidence"]
    for key in ("ascent.meco.state_position", "ascent.separation.state_position",
                "ascent.seco.state_position", "ascent.debris_impact.position",
                "landing.touchdown_position", "landing.propellant"):
        assert ev[key] == "independent_model"
    # ... the FL-301 design grading is labelled honestly, and the independent
    # rollup passes on checks that are independent models, not OT self-talk
    assert ev["landing.fl301_gates"] == "design_input"
    assert status["independent_validation"]["result"] == "passed"
    assert status["independent_validation"]["checks_passed"] >= 15


@NOT_RECORDED
@pytest.mark.slow
def test_error_budget_adds_up_with_reported_residual(report: dict) -> None:
    budget = report["inputs"]["error_budget"]
    tol = report["inputs"]["tolerances"]
    for q, entry in budget.items():
        assert set(entry["lines"]) == set(_TWIN_LINES)
        total = sum(entry["lines"].values())
        assert math.isclose(total, entry["measured_delta"], rel_tol=1e-9)
        assert entry["lines"]["frame_time"] == 0.0
        assert entry["lines"]["controller"] == 0.0
    # the gated residuals sit inside their declared bands
    for q, cid in (("meco_pos", "ascent.meco.residual_position"),
                   ("meco_vel", "ascent.meco.residual_velocity"),
                   ("separation_pos", "ascent.separation.residual_position"),
                   ("seco_pos", "ascent.seco.residual_position")):
        res = budget[q]["lines"]["residual"]
        assert abs(res) <= tol[cid]["limit"], (q, res, tol[cid]["limit"])


@NOT_RECORDED
@pytest.mark.slow
def test_landing_corridor_and_pair_are_recorded(report: dict) -> None:
    lc = report["inputs"]["landing_corridor"]
    assert lc["engines"] == 3 and lc["mass_kg"] == 456000.0
    assert lc["fl301_position_error_m"] <= 25.0
    assert abs(lc["fl301_vertical_speed_m_s"]) <= 3.0
    # the independent re-flight touched down near the pad with the reserve
    assert abs(lc["independent_touchdown"]["pos_enu_m"][0]) < 50.0
    assert lc["independent_touchdown"]["mass_kg"] >= 425000.0
    notes = " ".join(report["inputs"]["honesty_notes"]
                     + report["inputs"]["assumption_differences"])
    for topic in ("design input", "historical_telemetry", "residual"):
        assert topic in notes


@NOT_RECORDED
@pytest.mark.slow
def test_reference_digests_verified(report: dict) -> None:
    ref = report["inputs"]["reference"]
    assert ref["provenance_ok"] is True
    assert ref["provenance_files_checked"] >= 10
