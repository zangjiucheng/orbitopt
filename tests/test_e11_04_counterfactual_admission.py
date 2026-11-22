"""E11-04 (task #311): counterfactual admission re-flight gates.

Fast gates on the committed benchmark JSON
(docs/benchmarks/e11_04_counterfactual_admission.json) -- no physics:
every measured landing count is recomputed from the recorded flight records
and must match the analysis; the flipped rows must be exactly the E10-06
counts; the pairing verdict must rest on the bit-valid cells reproducing
their recorded rows; and the honest readings (measured vs the FL-305
inference, rule-effect pairing, no undoing of old-rule landings) must be
what the report claims. Regenerate with

    PYTHONPATH=src python benchmarks/bench_e11_04_counterfactual_admission.py \
        --workers 8 -o docs/benchmarks/e11_04_counterfactual_admission.json \
        --md docs/benchmarks/e11_04_counterfactual_admission.md

(~1 min of physics on 16 workers; era drift in the pre-#287 E7-04 arm is
expected and gated as a diagnostic, see the report's era-validity section).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs" / "benchmarks" / "e11_04_counterfactual_admission.json"

#: E10-06 recorded newly-admitted counts this re-flight must cover
RECORDED_FLIPS = {"train": 15, "a_seeds": 16, "b_bands": 1, "c_rtls": 0}
BIT_VALID = ("a_seeds", "b_bands")


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def landed_by_flip(rep) -> dict:
    out: dict[tuple[str, int, int], bool] = {}
    for f in rep["flights"]:
        if f["kind"] == "admit":
            out[(f["cell"], f["seed"], f["idx"])] = (
                f["record"].get("outcome") == "landed")
    return out


class TestReportIntegrity:
    def test_kind_and_coverage(self, rep):
        assert rep["meta"]["kind"] == (
            "orbitopt.e11-04.counterfactual-admission-validation")
        assert rep["meta"]["max_flips_per_cell"] is None
        for c in rep["analysis"]["cells"]:
            assert c["newly_admitted"] == RECORDED_FLIPS[c["name"]]
            assert c["recorded_newly_admitted"] == RECORDED_FLIPS[c["name"]]

    def test_measured_counts_recomputed_from_flights(self, rep, landed_by_flip):
        for c in rep["analysis"]["cells"]:
            keys = [(c["name"], f["seed"], f["idx"]) for f in c["flips"]]
            assert all(k in landed_by_flip for k in keys)
            mz = c["newly_admitted_measured"]
            assert mz["n"] == len(keys)
            assert mz["landed"] == sum(landed_by_flip[k] for k in keys)
        assert not any("SAMPLE_ERROR" in str(m["diff"])
                       for m in rep["analysis"]["hard_violations"])

    def test_pairing_verdict_rests_on_bit_valid_cells(self, rep):
        v = rep["verdict"]["A1_pairing_verified"]
        assert v["pass"] and v["no_sample_errors"]
        for c in rep["analysis"]["cells"]:
            ev = c["era_validity"]
            if c["name"] in BIT_VALID:
                assert ev["flips_drifted"] == 0
                assert ev["controls_drifted"] == 0
                assert ev["flips_reproducing_recorded"] == c["newly_admitted"]
            elif c["name"] == "train":
                # pre-#287 arm: drift is the documented diagnostic, and the
                # drift rows must all be RECOVERY-era, never missing flights
                assert ev["bit_valid_on_head"] is False
                assert (ev["flips_reproducing_recorded"]
                        + ev["flips_drifted"]) == c["newly_admitted"]
                assert rep["analysis"]["era_validity"]["drifted_rows"]

    def test_policy_never_undoes_an_old_rule_landing(self, rep):
        assert all(n == 0 for n in rep["analysis"]["structural_premise"]
                   ["satisfied_rows_newly_rejected"].values())
        m = rep["analysis"]["pooled"]["rule_effect_paired"]
        assert m["successes_b_only"] == 0
        assert m["n_pairs"] == sum(RECORDED_FLIPS.values())

    def test_joint_violated_admits_counted_as_planning_infeasible(self, rep):
        # a JOINT_CHANCE_VIOLATED admit means the LIVE gate rejected on
        # today's numbers where the offline regrade admitted (only possible
        # on the stale E7-04 arm) -- allowed, but it must be classified as
        # planning_infeasible, never left unexplained
        for c in rep["analysis"]["cells"]:
            fc = c["failure_categories"]
            if not fc:
                continue
            for cls, per in fc["codes_by_class"].items():
                if "JOINT_CHANCE_VIOLATED" in per:
                    assert cls == "planning_infeasible"


class TestHonestReadings:
    def test_measured_not_inferred_and_passes(self, rep):
        v = rep["verdict"]
        assert v["A2_measured_not_inferred"]["pass"]
        assert v["A3_failure_categories_reported"]["pass"]
        assert v["overall_pass"]

    def test_pooled_rate_clears_the_floor(self, rep):
        mz = rep["analysis"]["pooled"]["newly_admitted"]["measured"]
        floor = rep["analysis"]["policy"]["min_admission_probability"]
        assert mz["n"] == 32
        assert mz["rate"] >= floor
        assert rep["verdict"]["rule_update_required"] is False

    def test_inference_comparison_is_reported(self, rep):
        ti = rep["analysis"]["training_vs_inference"]
        assert ti["inferred"]["rate"] == pytest.approx(10 / 15)
        assert ti["measured"]["n"] == 15
        assert 0 <= ti["exact_binomial_vs_inferred_p"] <= 1
        if ti["measured"]["rate"] < ti["inferred"]["rate"]:
            ex = ti["below_inference_explanation"]
            assert ex is not None
            assert (ex["fl305_landed_but_measured_failed"]
                    or ex["fl305_failed_but_measured_landed"])
            assert rep["verdict"]["measured_below_inference"] is True

    def test_rule_effect_claim_only_when_supported(self, rep):
        t = rep["verdict"]["claim_calibration_translates_to_landings"]
        if t["supported"]:
            assert t["paired_p_value"] is not None
            assert t["paired_p_value"] < 0.05
            assert t["b"] > t["c"]
            assert t["measured_admitted_rate"] >= t["floor"]
