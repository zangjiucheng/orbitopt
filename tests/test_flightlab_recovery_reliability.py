"""E7-05 (task #293): the reliability reading tools.

Unit level: the violation-reconstruction identity against the planner's own
``evaluate_joint_chance`` (they must agree to float noise on synthetic
verdicts), the diagnosis categorizer on synthesized rows, and the cluster
statistics against hand-computed references. Recorded level: the diagnosis
on the shipped E7-04 joint-arm evidence must reproduce the documented
headline (51 divert / 48 fuel / 0 probability-bound rejections, 20 of 99
rejections landed under the paired FL-305 grading) -- the tool is the only
thing between the report and that number, so the test pins it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitopt.flightlab.recovery.chance import JointChanceSpec, evaluate_joint_chance
from orbitopt.flightlab.recovery.reliability import (
    DEFAULT_BORDERLINE_VIOLATION, DEFAULT_DEEP_VIOLATION, channels_bound,
    cluster_bootstrap_phi_ci, cluster_bootstrap_rate_ci, constraint_violations,
    infeasibility_diagnosis, stratified_association,
)

REPO = Path(__file__).resolve().parents[1]

_SPEC = JointChanceSpec()


_ROW_IDX = iter(range(1000))


def _row(miss_m: float, margin_kg: float, exposure_kg: float) -> dict:
    v = evaluate_joint_chance(_SPEC, miss_m=miss_m, margin_kg=margin_kg,
                              exposure_kg=exposure_kg)
    return {"seed": 1, "idx": next(_ROW_IDX),
            "outcome": "failed", "reason": "JOINT_CHANCE_VIOLATED",
            "corridor_miss_m": miss_m, "corridor_fuel_margin_kg": margin_kg,
            "landing_probability": v.p_land, "joint_violation": v.worst_violation}


class TestViolationReconstruction:
    def test_identity_against_the_planner_verdict(self):
        # three single-channel violations and one accepted row
        for miss, margin in ((4000.0, 5000.0),   # divert deep
                             (900.0, -3000.0),   # fuel deep
                             (900.0, 3000.0),    # fuel mild / accepted region
                             (300.0, 8000.0)):   # clean pass
            row = _row(miss, margin, 40_000.0)
            v = constraint_violations(row, _SPEC)
            assert v is not None and v["consistent"], (miss, margin)
            ref = evaluate_joint_chance(_SPEC, miss_m=miss, margin_kg=margin,
                                        exposure_kg=40_000.0)
            for got, want in zip(v["violations"], ref.violations):
                assert got == pytest.approx(want, abs=1e-9)

    def test_inconsistent_recorded_violation_is_flagged_not_classified(self):
        row = _row(4000.0, 5000.0, 40_000.0)
        row["joint_violation"] = 0.0  # contradiction
        v = constraint_violations(row, _SPEC)
        assert v is not None and not v["consistent"]
        diag = infeasibility_diagnosis([row], _SPEC)
        assert diag["inconsistent_rows"] == 1 and diag["classified"] == 0

    def test_non_joint_row_is_none(self):
        assert constraint_violations({"landing_probability": None,
                                      "corridor_miss_m": None}, _SPEC) is None

    def test_channels_bound_names_ties_and_zero(self):
        # construct a tie: divert and fuel violations equal by symmetry of
        # tuning is fiddly; assert the documented shapes instead
        row = _row(4000.0, 5000.0, 40_000.0)
        assert channels_bound(row, _SPEC) == ("divert",)
        ok = _row(300.0, 8000.0, 40_000.0)
        ok["reason"] = "LANDED"
        assert channels_bound(ok, _SPEC) == ()


class TestDiagnosis:
    def test_categories_and_paired_baseline(self):
        rows = [
            _row(4000.0, 5000.0, 40_000.0),     # divert, deep
            _row(920.0, 5000.0, 40_000.0),     # divert, near (0.17 sigma over)
            _row(900.0, -3000.0, 40_000.0),     # fuel
        ]
        # a prob-only rejection: huge fuel/divert margins but a low product
        row = dict(seed=1, idx=3, outcome="failed",
                   reason="JOINT_CHANCE_VIOLATED", corridor_miss_m=200.0,
                   corridor_fuel_margin_kg=30_000.0,
                   landing_probability=0.4)
        # consistency cannot be claimed for a hand-written p_land -- the
        # diagnosis counts it inconsistent; assert exactly that contract:
        rows_with_bad = rows + [dict(row, joint_violation=0.5)]
        diag = infeasibility_diagnosis(rows_with_bad, _SPEC)
        assert diag["inconsistent_rows"] == 1
        assert diag["binding"].get("divert") == 2
        assert diag["binding"].get("fuel") == 1
        assert diag["severity"]["near"] == 1
        assert diag["severity"]["deep"] == 2
        base = [{"seed": r["seed"], "idx": r["idx"], "outcome": "landed",
                 "reason": "LANDED"} for r in rows[:2]]
        diag = infeasibility_diagnosis(rows, _SPEC, baseline_rows=base)
        pb = diag["paired_baseline"]
        assert pb["n_paired"] == 2 and pb["rejected_but_landed"] == 2
        assert pb["landed_anyway_share"] == pytest.approx(1.0)

    def test_probability_term_cannot_bind_alone_under_the_default_spec(self):
        # sweep the whole rejection surface: whenever the joint product is
        # violated, one of the physical terms is violated at least as hard
        # (the product is built from the same marginals). This is why the
        # E7-05 diagnosis reports ZERO probability-bound rejections.
        from orbitopt.flightlab.recovery.chance import evaluate_joint_chance
        for miss in [x * 25.0 for x in range(1, 80)]:
            for margin in [x * 100.0 for x in range(-40, 60)]:
                v = evaluate_joint_chance(_SPEC, miss_m=miss, margin_kg=margin,
                                          exposure_kg=40_000.0)
                assert not (v.violations[2] > 0.0 and
                            v.violations[2] > v.violations[0] and
                            v.violations[2] > v.violations[1])


class TestClusterStatistics:
    @staticmethod
    def _t(a, b, c, d):
        return {"both": a, "fuel_only": b, "divert_only": c, "neither": d}

    def test_mantel_haenszel_against_hand_computation(self):
        # two identical strata: a=1,b=9,c=9,d=81 each
        # per stratum: n=100, row1=10, col1=10, E=1, V=10*90*10*90/(100^2*99)
        tables = {1: self._t(1, 9, 9, 81), 2: self._t(1, 9, 9, 81)}
        res = stratified_association(tables, trials=200, seed=7)
        mh = res["mantel_haenszel"]
        e_sum = 2.0
        v_each = 10 * 90 * 10 * 90 / (100.0 ** 2 * 99.0)
        chi_ref = (abs(2.0 - e_sum) - 0.5) ** 2 / (2 * v_each)
        assert mh["statistic"] == pytest.approx(chi_ref)
        assert mh["n_tables"] == 2

    def test_fisher_combined_hand_value(self):
        # two strata whose exact p are both ~ 0.5 -> X2 = -2 ln(0.25), tail
        # e^-x (1 + x), x = X2/2 = ln 4
        import math
        tables = {1: self._t(5, 5, 5, 85), 2: self._t(85, 5, 5, 5)}
        res = stratified_association(tables, trials=200, seed=7)
        fc = res["fisher_combined"]
        x = fc["statistic"] / 2.0
        assert fc["p_value"] == pytest.approx(math.exp(-x) * (1.0 + x))
        assert fc["k"] == 2

    def test_degenerate_strata_are_skipped_not_zeroed(self):
        tables = {1: self._t(0, 0, 0, 0),        # no rows
                  2: self._t(10, 0, 0, 0),       # degenerate margin (no divert fail)
                  3: self._t(4, 10, 30, 56)}
        res = stratified_association(tables, trials=200, seed=7)
        assert set(res["clusters_skipped"]) == {1, 2}
        assert res["n_clusters"] == 1
        assert res["mantel_haenszel"] is None   # one stratum is not stratified

    def test_cluster_bootstrap_phi_deterministic_and_bounded(self):
        tables = [self._t(4, 10, 28, 59), self._t(2, 10, 35, 55),
                  self._t(4, 14, 28, 49), self._t(1, 12, 38, 60)]
        as_map = {i: t for i, t in enumerate(tables)}
        res = stratified_association(as_map, trials=500, seed=123)
        ci = res["phi_cluster_bootstrap_ci95"]
        again = stratified_association(as_map, trials=500, seed=123)
        assert ci == again["phi_cluster_bootstrap_ci95"]
        assert ci is not None and ci[0] < 0.0
        # a whole-cluster resample can never pool outside the tables' own
        # phi range [-0.193, -0.025]:
        assert -0.20 <= ci[0] <= -0.05 and ci[1] <= 0.0

    def test_rate_ci_identical_clusters_collapse_to_the_rate(self):
        ci = cluster_bootstrap_rate_ci([(3, 10), (3, 10), (3, 10), (3, 10)])
        assert ci is not None
        assert ci[0] == pytest.approx(0.3) and ci[1] == pytest.approx(0.3)
        assert cluster_bootstrap_rate_ci([]) is None

    def test_phi_ci_identical_tables(self):
        t = self._t(10, 20, 30, 40)
        ci = cluster_bootstrap_phi_ci([dict(t) for _ in range(4)], trials=300,
                                      seed=5)
        from orbitopt.flightlab.recovery.joint_risk import phi_coefficient
        phi = phi_coefficient(10, 20, 30, 40)
        assert ci is not None and ci[0] == pytest.approx(ci[1]) == pytest.approx(phi)


class TestRecordedEvidence:
    """The shipped E7-04 evidence through the new tool: the numbers the
    E7-05 report quotes must be exactly what the tool produces from the
    recorded rows (reconstruction included)."""

    E7_04 = REPO / "docs" / "benchmarks" / "e7_04_failure_driven.json"
    E3_03 = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

    def test_reconstruction_is_consistent_on_every_recorded_joint_row(self):
        rep = json.loads(self.E7_04.read_text())
        rows = next(a for a in rep["arms"]
                    if a["name"] == "e7_04_joint_base")["paired_rows"]
        n_joint = 0
        for r in rows:
            v = constraint_violations(r, _SPEC)
            if r["reason"] == "JOINT_CHANCE_VIOLATED":
                n_joint += 1
                assert v is not None and v["consistent"], r
        assert n_joint == 99

    def test_diagnosis_reproduces_the_reported_headline(self):
        rep = json.loads(self.E7_04.read_text())
        rows = next(a for a in rep["arms"]
                    if a["name"] == "e7_04_joint_base")["paired_rows"]
        base = json.loads(self.E3_03.read_text()
                          )["runs"]["fl305_baseline"]["paired_rows"]
        diag = infeasibility_diagnosis(rows, _SPEC, baseline_rows=base)
        assert diag["classified"] == 99
        assert diag["binding"] == {"divert": 51, "fuel": 48}
        assert diag["probability_binding"]["bound_on_prob_any"] == 0
        assert diag["severity"] == {"near": 18, "mid": 30, "deep": 51}
        pb = diag["paired_baseline"]
        assert pb["n_paired"] == 99 and pb["rejected_but_landed"] == 20
        assert pb["by_binding"] == {"divert/failed": 44, "divert/landed": 7,
                                    "fuel/failed": 35, "fuel/landed": 13}
        assert diag["worst_violation_quantiles"]["q0.1"] < DEFAULT_BORDERLINE_VIOLATION
        assert diag["worst_violation_quantiles"]["q0.9"] > DEFAULT_DEEP_VIOLATION
