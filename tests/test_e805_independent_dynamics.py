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
from pathlib import Path

import numpy as np
import pytest

import orbitopt.verify.independent_dynamics as idyn
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


# ---------------------------------------------------------------------------
# E10-04 (task #301): event-aware interpolation of the raw 2-s history
# ---------------------------------------------------------------------------

E508_DC7_REF = Path(__file__).resolve().parents[1] / \
    "tests/data/mission/reference/e5-08-new-vehicle-dc7"


def _mk_row(t, x, v, a, thrust, isp, mass):
    return {"time_s": t, "pos_eci_x_m": x, "pos_eci_y_m": 0.0, "pos_eci_z_m": 0.0,
            "vel_eci_x_mps": v, "vel_eci_y_mps": 0.0, "vel_eci_z_mps": 0.0,
            "accel_eci_x_mps2": a, "accel_eci_y_mps2": 0.0, "accel_eci_z_mps2": 0.0,
            "thrust_n": thrust, "isp_s": isp, "mass_kg": mass}


def test_chord_across_a_cutoff_kink_is_replaced_by_the_left_segment():
    """A 2-s chord spanning a cutoff averages the powered and unpowered
    slopes; the state AT the cutoff instant must come from the powered side."""
    g, a_pow, mdot, v0, m0 = -9.8, 20.0, 1000.0, 100.0, 10_000.0
    v1 = v0 + a_pow * 1.0                       # cutoff at t = 1.0 mid-segment
    rows = [_mk_row(0.0, v0 * 0.5, v0, g + a_pow, 1e6, 100.0, m0),
            _mk_row(2.0, 0.0, v1 + g * 1.0, g, 0.0, 0.0, m0 - mdot),
            _mk_row(4.0, 0.0, v1 + g * 3.0, g, 0.0, 0.0, m0 - mdot)]
    rec = {"raw": rows, "trace_split": len(rows)}
    s = idyn._raw_interp(rec, 1.0, None)
    assert s["vel_eci_x_mps"] == pytest.approx(v0 + (g + a_pow), abs=1e-9)
    assert s["mass_kg"] == pytest.approx(m0 - mdot, abs=1e-6)
    # the plain chord on a smooth segment is unchanged
    s2 = idyn._raw_interp(rec, 3.0, None)
    assert s2["vel_eci_x_mps"] == pytest.approx(v1 + g * 2.0, abs=1e-9)


def test_ignition_inside_a_segment_holds_mass_at_the_left_value():
    rows = [_mk_row(0.0, 0.0, 50.0, -5.0, 0.0, 0.0, 5000.0),
            _mk_row(2.0, 0.0, 40.0, 15.0, 1e6, 100.0, 3000.0)]
    s = idyn._raw_interp({"raw": rows, "trace_split": 2}, 0.5, None)
    assert s["mass_kg"] == pytest.approx(5000.0)             # not yet ignited


@pytest.mark.skipif(not E508_DC7_REF.is_dir(), reason="E5-08 DC7 reference absent")
def test_recorded_dc7_event_states_reconstruct_their_own_segments() -> None:
    """DC7's sealed history: MECO 123.0 and SECO 514.663 sit mid-segment. The
    reconstruction must return the state the trace itself implies: the MECO
    mass equals the coast rows, and the SECO state lies on the orbit the
    coast rows fly (a straight chord was ~13 m/s slow, ~22 km low in SMA)."""
    rec = idyn._load_recorded(E508_DC7_REF)
    _, v_me, m_me = idyn._rec_state(rec, rec["event_times"]["meco"], 1)
    rows = {float(r["time_s"]): r for r in rec["raw"]}
    coast = rows[124.0]
    assert m_me == pytest.approx(float(coast["mass_kg"]), abs=50.0)
    assert float(np.linalg.norm(v_me)) > 1765.0            # powered-side speed
    r_se, v_se, m_se = idyn._rec_state(rec, rec["event_times"]["seco"], 2)
    row = rows[516.0]
    r_c = np.array([float(row[c]) for c in idyn._RAW_POS])
    v_c = np.array([float(row[c]) for c in idyn._RAW_VEL])
    a_se = idyn._orbit_elements(r_se, v_se)["semi_major_axis_m"]
    a_c = idyn._orbit_elements(r_c, v_c)["semi_major_axis_m"]
    assert abs(a_se - a_c) < 3000.0                        # same post-burn orbit
    assert m_se == pytest.approx(float(row["mass_kg"]), abs=50.0)
