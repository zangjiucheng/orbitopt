"""E11-03 (task #310): robust recovery co-design gates.

Fast consistency gates on the committed campaign report
(docs/benchmarks/e11_03_robust_recovery_co_design.json): the seed manifest
is disjoint and sized from the CI-width target, the paired arms reproduce
the sealed E10-03 baseline tallies exactly, the failure categories
partition the failures, the Holm family is the planned two, the constraint
audit reports no relaxation, and the recorded verdict -- ascent-constrained
co-design cannot move the DC8 separation state toward the envelope, so the
co-designed arm stays at the E10-03 floor and is significantly WORSE than
both pinned baselines -- is asserted so a future re-run that does close
trips the test loudly. Regenerate with
``python benchmarks/bench_e11_03_robust_recovery_co_design.py --workers 30
--cache ...``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs/benchmarks/e11_03_robust_recovery_co_design.json"
SCRIPT = REPO / "benchmarks/bench_e11_03_robust_recovery_co_design.py"


def _bench():
    spec = importlib.util.spec_from_file_location("bench_e11_03", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def benchmod():
    return _bench()


class TestSeedManifest:
    def test_full_scale_and_ci_sized(self, rep, benchmod):
        sm = rep["seed_manifest"]
        assert rep["smoke_scale"] is False
        n = sm["required_sample_size_for_ci_width"]
        assert n["actual"] == n["required"] + 1 >= n["required"]
        assert n["required"] > 100
        assert sm["holdout_seeds"] == benchmod.HOLDOUT_SEEDS
        assert sm["n_per_seed"] * len(sm["holdout_seeds"]) == n["actual"] == 168

    def test_streams_disjoint(self, rep, benchmod):
        sm = rep["seed_manifest"]
        ho = set(sm["holdout_seeds"])
        tr = set(sm["train_seeds"])
        assert ho.isdisjoint(tr)
        for other in (benchmod.E508_MC_SEEDS, benchmod.E1005_TRAIN_SEEDS,
                      benchmod.E1003_TRAIN_SEEDS):
            assert ho.isdisjoint(other)
            assert tr.isdisjoint(other)
        assert benchmod.RECORDED_SEARCH_SEED not in ho | tr


class TestPairedArms:
    def test_arms_complete_and_partition(self, rep):
        for a in rep["arms"]:
            assert a["n"] == 168
            fc = a["failure_categories"]
            assert sum(fc.values()) == a["n"] - a["successes"]
            assert set(fc) == {"planning_infeasible", "execution_failure",
                               "accuracy", "fuel"}

    def test_pinned_baselines_reproduce_sealed_e1003(self, rep):
        got = {a["arm"]: a["successes"] for a in rep["arms"]}
        assert got["dc7_pinned"] == 44
        assert got["starbase_pinned"] == 11
        assert rep["baseline_dc8_e10_03"]["holdout_successes"] == 0

    def test_tables_consistent_with_tallies(self, rep):
        succ = {a["arm"]: a["successes"] for a in rep["arms"]}
        for c in rep["comparisons"]:
            a, b = c["arms"]
            t = c["table"]
            assert t["a_only"] + t["both"] == succ[a]
            assert t["b_only"] + t["both"] == succ[b]

    def test_holm_family_planned_and_monotone(self, rep):
        cs = rep["comparisons"]
        assert [tuple(c["arms"]) for c in cs] == [
            ("codesign", "dc7_pinned"), ("codesign", "starbase_pinned")]
        ps = sorted(c["p_raw"] for c in cs)
        assert abs(cs[0]["p_holm"] - min(1.0, 2 * ps[0])) < 1e-12
        for c in cs:
            assert c["p_holm"] >= c["p_raw"]
        assert max(c["p_holm"] for c in cs) <= max(
            min(1.0, 2 * p) for p in ps) + 1e-12


class TestRecordedVerdict:
    def test_co_design_closed_negative(self, rep):
        """The honest E11-03 verdict: the ascent constraint set pins the
        separation state, co-design does not close, and the recorded
        significance is a significant REGRESSION on both comparisons."""
        cs = rep["comparisons"]
        assert all(c["table"]["a_only"] == 0 and c["significant"] for c in cs)
        assert all(c["paired_ci"]["diff"] < 0 for c in cs)
        codesign = next(a for a in rep["arms"] if a["arm"] == "codesign")
        assert codesign["successes"] == 0
        assert codesign["failure_categories"]["planning_infeasible"] >= 160

    def test_nominal_corridor_missed(self, rep):
        nom = rep["e5_06_joint_gate"]["nominal"]
        assert nom["outcome"] == "failed"
        assert nom["reason"] == "CORRIDOR_MISSED"
        st = {k: v["status"] for k, v in rep["e5_06_joint_gate"]["checks"].items()}
        assert st["physical_check_landing"] == "failed"
        assert st["physical_check_mass_conservation"] == "passed"
        assert st["physical_check_epoch_frame_continuity"] == "passed"
        assert st["physical_check_orbit_insertion"] == "passed"
        assert st["physical_check_parking_propagation"] == \
            "not_evidenced_single_environment"

    def test_ascent_grid_is_the_constraint_map(self, rep):
        grid = rep["ascent_grid"]
        rows = grid["rows"]
        assert len(rows) == grid["n_cells"]
        assert sum(1 for r in rows if r["ascent_pass"]) == grid["n_pass"]
        assert grid["n_pass"] == 1
        assert [r["vkey"] for r in rows if r["ascent_pass"]] == ["m+0_p+0_b+0_n+0"]
        for r in rows:
            if not r["ascent_pass"]:
                assert any(k.startswith("insertion") or k.startswith("max-")
                           for k in r["failed_checks"])

    def test_chosen_design_is_the_sealed_ascent(self, rep):
        ch = rep["codesign"]
        assert ch["vkey"] == "m+0_p+0_b+0_n+0"
        assert ch["ascent"]["ascent_pass"] is True
        assert ch["ascent"]["checks"]["max-accel-within-limit"] is True
        assert ch["ascent"]["checks"]["max-q-within-limit"] is True
        assert ch["es_selected"]["in_frac"] <= 0.5

    def test_selected_script_inside_bounds(self, rep):
        x = rep["codesign"]["script"]
        assert len(x) == 8
        for v, (lo, hi) in zip(x, rep["codesign"]["script_bounds"]):
            assert lo - 1e-9 <= v <= hi + 1e-9

    def test_constraints_unrelaxed(self, rep):
        ca = rep["constraint_audit"]
        assert ca["constraints_unrelaxed"] is True
        assert ca["ascent_all_pass"] is True
        assert ca["graded_reserve_identical"] is True
        assert ca["g_max_ascent"] == 3.0
        assert ca["insertion_band_m"] == [230000.0, 250000.0]

    def test_insertion_margin_reported(self, rep):
        im = rep["insertion_margin"]
        assert abs(im["baseline_dc8_sealed"]["ship_margin_kg"] - 103548.46) < 0.1
        assert abs(im["codesign"]["ship_margin_kg"]
                   - im["baseline_dc8_sealed"]["ship_margin_kg"] - 0.0) < 0.5
        assert im["delta_ship_margin_kg"] == 0.0

    def test_train_agreement_honest(self, rep):
        ag = rep["codesign"]["train_agreement"]
        assert ag["screen_on"] <= ag["n_train"] == 10
        assert ag["planned"] <= ag["screen_on"]
        assert ag["landed"] <= ag["planned"]


class TestCategoryMapping:
    def test_known_reasons_mapped(self, benchmod):
        assert benchmod.failure_category("CORRIDOR_MISSED") == "planning_infeasible"
        assert benchmod.failure_category("PROPELLANT_RESERVE_LOW") == "fuel"
        assert benchmod.failure_category("LANDING_POSITION_OUT_OF_TOLERANCE") \
            == "accuracy"
        assert benchmod.failure_category("LANDING_FAILED") == "execution_failure"
        assert benchmod.failure_category("WHATEVER_NEW") == "execution_failure"

    def test_variant_yaml_zero_knobs_preserves_the_ascent(self, benchmod, tmp_path):
        """The co-design file format must be behaviour-preserving at the
        identity knobs: OT's insertion checks pass exactly as the sealed
        E10-02 DC8 record (periapsis-above-minimum is the sensitive one)."""
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.flightlab.contract import load_target
        knobs = {"meco_shift": 0.0, "pitch_trim": 0.0, "dburn_s": 0.0,
                 "node_trim": 0.0}
        vp = benchmod.build_variant_yaml(knobs, tmp_path / "veh.yaml")
        import yaml
        doc = yaml.safe_load(vp.read_text())
        base = yaml.safe_load(benchmod.DC8_FILE.read_text())
        assert doc["staging"] == base["staging"]
        assert doc["attitude_program"] == base["attitude_program"]
        assert doc["stages"][0]["engine"] == base["stages"][0]["engine"]
