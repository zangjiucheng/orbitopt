"""E11-02 (task #309): recovery feasibility envelope gates.

Fast gates on the committed benchmark JSON
(docs/benchmarks/e11_02_recovery_feasibility_envelope.json) plus no-flight
unit gates on the envelope grading itself. The acceptance criteria appear
as: zero false rejects on the E10-03 hold-out with truth counts matching
the sealed E10-03 report (part C), a boundary that crosses in the DC7
speed/downrange map and a wall of tilt-OUT for the fins-less DC8 grid
(part B), the E10-07 twin agreeing with the envelope on the boundary it
seals (part D), and a screened call spending zero search on an OUT sample
while an IN sample still gets its search (part E). Regenerate with

    PYTHONPATH=src python benchmarks/bench_e11_02_recovery_feasibility_envelope.py \
        --parts A,B,C,D,E --workers 8 --cache /tmp/e1102-cache.json

(~1 h of physics on 16 workers -- truth flights replicate the E10-03
pipeline, which needs the same astropy/iers frame environment it ran in).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitopt.flightlab.recovery import (
    CORRIDOR_MISSED,
    CORRIDOR_NOT_REACHED,
    ENTRY_Q_LIMIT_EXCEEDED,
    INSUFFICIENT_PROPELLANT_BOOSTBACK,
    INSUFFICIENT_PROPELLANT_ENTRY_BURN,
    INSUFFICIENT_PROPELLANT_RESERVE,
    RecoveryCorridor,
)
from orbitopt.flightlab.recovery.feasibility import (
    ENVELOPE_CONDITIONS,
    CONDITION_REASON,
    EnvelopeSpec,
    grade_probe,
)

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs" / "benchmarks" / "e11_02_recovery_feasibility_envelope.json"
E1003 = REPO / "docs" / "benchmarks" / "e10_03_robust_rtls_closure.json"

CORRIDOR = RecoveryCorridor(h0_m=3000.0, radius_m=8000.0, v_max_m_s=350.0,
                            max_tilt_deg=15.0)


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


def _spec() -> EnvelopeSpec:
    return EnvelopeSpec(reference_script=(0.0,) * 8, landing_need_model="momentum")


def _probe_ok(**over):
    p = {"reached": True, "miss_m": 500.0, "tilt_deg": 4.0, "speed_m_s": 300.0,
         "prop_kg": 120000.0, "landing_need_kg": 40000.0, "entry_q_pa": 800.0,
         "entry_g": 3.0, "exhausted": None, "landing_forecast_ok": True,
         "handoff_overlap_s": 0.0}
    p.update(over)
    return p


def _grade(probe, spec=None):
    return grade_probe(probe, spec or _spec(), corridor=CORRIDOR,
                       landing_reserve_kg=10000.0, sep_propellant_kg=150000.0,
                       prop_floor_kg=40000.0)


class TestEnvelopeGrading:
    def test_conditions_predict_flight_reasons(self):
        assert set(CONDITION_REASON) == set(ENVELOPE_CONDITIONS)
        assert set(CONDITION_REASON.values()) <= {
            CORRIDOR_MISSED, CORRIDOR_NOT_REACHED, ENTRY_Q_LIMIT_EXCEEDED,
            "ENTRY_G_LIMIT_EXCEEDED", INSUFFICIENT_PROPELLANT_BOOSTBACK,
            INSUFFICIENT_PROPELLANT_ENTRY_BURN, INSUFFICIENT_PROPELLANT_RESERVE}

    def test_clean_probe_inside(self):
        v = _grade(_probe_ok())
        assert v.in_envelope and v.governing_condition is None
        assert v.worst_severity < 1.0

    def test_analytic_rejection_without_probe(self):
        v = grade_probe(None, _spec(), corridor=CORRIDOR,
                        landing_reserve_kg=10000.0, sep_propellant_kg=30000.0,
                        prop_floor_kg=40000.0)
        assert not v.in_envelope
        assert v.governing_condition == "fuel_floor"
        assert v.governing_reason == INSUFFICIENT_PROPELLANT_BOOSTBACK

    def test_out_codes_follow_the_planner_violation_order(self):
        v = _grade(_probe_ok(reached=False, tilt_deg=60.0, entry_q_pa=200000.0))
        assert not v.in_envelope
        assert v.governing_condition == "corridor_reached"
        assert v.governing_reason == CORRIDOR_NOT_REACHED
        assert ENTRY_Q_LIMIT_EXCEEDED in v.violated_codes

    def test_handoff_overlap_only_when_the_rest_is_clean(self):
        v = _grade(_probe_ok(handoff_overlap_s=1.2))
        assert not v.in_envelope
        assert v.governing_condition == "handoff_overlap"
        assert v.governing_reason == CORRIDOR_MISSED
        # a crossing outside the speed window outranks it: with no forecast
        # modelled the raw v_max is the gate, and the overlap elif is not
        # even evaluated (mirrors the planner's elif chain)
        v2 = _grade(_probe_ok(handoff_overlap_s=1.2, speed_m_s=420.0,
                              landing_forecast_ok=None))
        assert v2.governing_condition == "arrival_speed"


class TestSpecRoundTrip:
    def test_mapping_roundtrip_and_dataclass_rebuild(self):
        spec = EnvelopeSpec(
            reference_script=(1.0, 2.0), dt_opt=0.25, dt_final=0.1,
            landing_limits={"position_tol_m": 25.0},
            landing_guidance={"max_thrust_accel_g": 4.0},
            landing_divert_budget_m=None)
        back = EnvelopeSpec.from_mapping(spec.to_dict())
        assert back.dt_final == 0.1
        kw = back.planner_kwargs()
        assert kw["landing_limits"].position_tol_m == 25.0
        assert kw["landing_guidance"].max_thrust_accel_g == 4.0
        assert kw["landing_divert_budget_m"] is None


@pytest.fixture(scope="module")
def e1003() -> dict:
    return json.loads(E1003.read_text(encoding="utf-8"))


class TestHoldoutAgreement:
    """Part C acceptance: zero false rejects, truth = sealed E10-03."""

    def test_coverage(self, rep):
        recs = rep["C_records"]
        assert len(recs) == 4 * 168
        for arm, c in rep["C_confusion"].items():
            assert c["n"] == 168, arm

    def test_zero_false_rejects(self, rep):
        for arm, c in rep["C_confusion"].items():
            assert c["false_rejects"] == 0, (arm, c["false_reject_detail"])

    def test_truth_matches_sealed_e10_03(self, rep, e1003):
        for arm, c in rep["C_confusion"].items():
            landed = sum(1 for r in rep["C_records"]
                         if r["arm"] == arm and r["truth_landed"])
            committed = e1003["arms"][arm]["landed"]
            assert landed == committed, (arm, landed, committed)

    def test_rejection_codes_match_truth(self, rep):
        # predicted reason = the flight's first reason code; disagreement is
        # tolerated only as a small residual on near-tied violations
        for arm, c in rep["C_confusion"].items():
            rej = c["reject_code_match"] + c["reject_code_mismatch"]
            assert rej == c["out_not_landed"] + c["out_but_landed"]
            assert c["reject_code_mismatch"] <= 0.06 * rej, (arm, c)

    def test_predicted_codes_are_flight_codes(self, rep):
        flight = {CORRIDOR_MISSED, CORRIDOR_NOT_REACHED, ENTRY_Q_LIMIT_EXCEEDED,
                  INSUFFICIENT_PROPELLANT_BOOSTBACK,
                  INSUFFICIENT_PROPELLANT_ENTRY_BURN,
                  INSUFFICIENT_PROPELLANT_RESERVE, "LANDED"}
        for r in rep["C_records"]:
            if not r["pred_in_envelope"]:
                assert r["pred_code"] in flight - {"LANDED"}


class TestBoundaryMaps:
    def test_dc7_boundary_crosses(self, rep):
        cells = [v for k, v in rep["B_cells"].items() if k.startswith("dc7|")]
        assert len(cells) == 49
        ins = sum(1 for c in cells if c["in_envelope"])
        assert 0 < ins < 49

    def test_dc8_fins_grid_is_all_tilt_out(self, rep):
        cells = [v for k, v in rep["B_cells"].items() if k.startswith("dc8_fins|")]
        assert len(cells) == 49
        assert all(not c["in_envelope"] for c in cells)
        assert all(c["governing_condition"] == "corridor_tilt" for c in cells)


class TestTwinAgreement:
    def test_twin_confirms_the_envelope_boundary(self, rep):
        pair = [v for v in rep["D"].values()
                if v.get("pair") in ("IN", "OUT") and "error" not in v]
        assert len(pair) == 2
        assert rep["D_pair_agreement"]
        assert all(v["boundary_agreement"] for v in pair)


class TestSearchBudget:
    def test_out_sample_spends_no_search(self, rep):
        seen = 0
        for r in rep["E"]:
            if "out" in str(r.get("key")):
                seen += 1
                assert r["screen_rejected"], r["key"]
                assert r["screen_fevals"] == 0, r["key"]
                # below the propellant floor the planner's own guard also
                # skips the search, so the budget saving is in the wall: the
                # screen pays one cheap reference rollout where the unscreened
                # planner pays its full-dt grading integrations
                assert r["screen_wall_s"] < 0.25 * r["full_wall_s"], r["key"]
        assert seen >= 2

    def test_in_sample_still_gets_its_search(self, rep):
        r = next(r for r in rep["E"] if r.get("key") == "E|dc7|in-nominal")
        assert not r["screen_rejected"]
        assert r["screen_fevals"] > 0
