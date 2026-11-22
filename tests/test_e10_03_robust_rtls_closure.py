"""E10-03 (task #300): robust RTLS design-closure gates.

Fast consistency gates on the committed campaign report
(docs/benchmarks/e10_03_robust_rtls_closure.json): the seed manifest is
disjoint and sized from the CI-width target, the failure-category tallies
add up, the Holm family is the planned three with monotone adjusted p, the
constraint audit reports no relaxation, the DC8 start-state digest
recomputes, and the recorded verdicts match the recorded data. The campaign
verdict itself (DC8 + E10-05 planning + E10-06 graded admission does not
close -- the honest NOT-CLOSED sibling of E10-01/E10-07) is asserted so a
future re-run that DOES close trips the test loudly. Regenerate with
``python benchmarks/bench_e10_03_robust_rtls_closure.py --fly-cache ...``.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs/benchmarks/e10_03_robust_rtls_closure.json"
SCRIPT = REPO / "benchmarks/bench_e10_03_robust_rtls_closure.py"


def _bench():
    spec = importlib.util.spec_from_file_location("bench_e10_03", SCRIPT)
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
    def test_streams_disjoint_and_sizes(self, rep, benchmod):
        sm = rep["seed_manifest"]
        assert rep["smoke_scale"] is False
        assert sm["n_per_arm"] == sm["n_per_seed"] * len(sm["holdout_seeds"])
        assert sm["n_per_arm"] >= sm["required_sample_size"]
        assert sm["required_sample_size"] > 36
        assert sm["n_per_arm"] > 4 * 36
        assert sm["holdout_seeds"] == benchmod.HOLDOUT_SEEDS
        assert set(sm["train_seeds"]).isdisjoint(sm["holdout_seeds"])
        for stream in sm["excluded_streams"].values():
            stream = {stream} if isinstance(stream, int) else set(stream)
            assert stream.isdisjoint(sm["holdout_seeds"])
            assert stream.isdisjoint(sm["train_seeds"])


class TestFailureCategories:
    def test_categories_partition_the_failures(self, rep, benchmod):
        for arm, a in rep["arms"].items():
            assert a["n"] == 168, arm
            cats = a["failure_categories"]
            assert sum(cats.values()) == a["n"] - a["landed"], arm
            assert set(cats) == set(benchmod.FAILURE_CATEGORIES), arm
            assert sum(a["reason_mix"].values()) == a["n"] - a["landed"], arm

    def test_dc8_failure_is_planning_class(self, rep):
        # the closure gap is a planning/feasibility gap (corridor + joint
        # chance), not a guidance-execution gap
        for arm in ("dc8_robust", "dc8_binary"):
            a = rep["arms"][arm]
            assert a["failure_categories"]["planning_infeasible"] == a["n"], arm
            assert set(a["reason_mix"]) == {"CORRIDOR_MISSED"}, arm

    def test_known_reasons_are_mapped(self, benchmod):
        for r in ("CORRIDOR_MISSED", "INSUFFICIENT_PROPELLANT_RESERVE",
                  "JOINT_CHANCE_VIOLATED", "LANDING_FAILED",
                  "LANDING_POSITION_OUT_OF_TOLERANCE", "PROPELLANT_RESERVE_LOW"):
            assert benchmod.failure_category(r) in benchmod.FAILURE_CATEGORIES


class TestStatistics:
    def test_holm_family_planned_and_monotone(self, rep):
        paired = rep["paired"]
        assert len(paired) == 3
        assert {tuple(p["arms"]) for p in paired} == {
            ("dc8_robust", "dc7_pinned"), ("dc8_robust", "starbase_pinned"),
            ("dc8_robust", "dc8_binary")}
        rows = sorted(paired, key=lambda p: p["p_raw"])
        assert [p["p_holm"] for p in rows] == sorted(p["p_holm"] for p in rows) \
            or rows[0]["p_holm"] <= rows[1]["p_holm"] <= rows[2]["p_holm"]
        for p in paired:
            t = p["table"]
            assert t["both"] + t["a_only"] + t["b_only"] + t["neither"] == 168
            assert p["significant"] is (p["p_holm"] < 0.05 and
                                        p["paired_ci"]["diff"] > 0.0)

    def test_verdict_no_improvement_recorded(self, rep):
        # honest NOT-CLOSED: the new scheme must be recorded as NOT
        # significantly better; a future run that closes trips this gate
        assert all(p["significant"] is False for p in rep["paired"])
        assert rep["arms"]["dc8_robust"]["landed"] <= rep["arms"]["dc7_pinned"]["landed"]
        assert rep["e5_06_joint_gate"]["recovery_leg_checks_passed"] is False


class TestDesignAndAudit:
    def test_dc8_start_digest_recomputes(self, rep):
        from orbitopt.flightlab.handoff import HandoffState, handoff_digest
        from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState
        doc = json.loads((REPO / "docs/benchmarks/e10_07_full_recovery_reflight.json")
                         .read_text(encoding="utf-8"))
        hs_doc = doc["handoff"]
        s = hs_doc["state"]
        sep = BoosterState(t_s=float(s["t_s"]),
                           epoch_utc=dt.datetime.fromisoformat(s["epoch_utc"]),
                           frame=s["frame"],
                           position_m=tuple(float(c) for c in s["position_m"]),
                           velocity_m_s=tuple(float(c) for c in s["velocity_m_s"]),
                           mass_kg=float(s["mass_kg"]))
        hs = HandoffState.of(sep, from_phase="separation", to_phase="boostback",
                             dry_mass_kg=BoosterModel.load(
                                 REPO / "src/orbitopt/data/flightlab/vehicles"
                                 / "starship_class_rtls_dc8.yaml").dry_mass_kg)
        assert handoff_digest(hs) == hs_doc["digest"]

    def test_constraints_unrelaxed(self, rep):
        ca = rep["constraint_audit"]
        assert ca["constraints_unrelaxed"] is True
        assert ca["dc8_shares_dc7_scenario"] is True
        assert ca["graded_reserve_identical"] is True
        assert ca["recipe_entry_window"] == {"entry_q_max_pa": 250000.0,
                                             "entry_g_max": 12.0}

    def test_nominal_gate_rows_match(self, rep):
        nom = rep["nominals"]
        assert nom["dc7"]["outcome"] == "landed"
        assert nom["starbase"]["outcome"] == "landed"
        gate = rep["e5_06_joint_gate"]
        assert gate["nominal"]["outcome"] == nom["dc8"]["outcome"]
        assert gate["recovery_leg_checks_passed"] == (
            gate["nominal"]["outcome"] == "landed"
            and bool(gate["checks"]["physical_check_mass_conservation"]["checks"]))

    def test_sealed_baselines_untouched(self, rep):
        sb = rep["sealed_baselines"]
        assert sb["dc7"]["successes"] == 11 and sb["dc7"]["n"] == 36
        assert sb["starbase"]["successes"] == 4 and sb["starbase"]["n"] == 36

    def test_selected_script_sane(self, rep):
        x = rep["robust_arm"]["selected_script"]
        assert len(x) == 8
        assert 0.0 <= x[0] <= 150.0          # boostback delay
        assert -20.0 <= x[1] <= 80.0         # aim past pad [km]
        assert 3200.0 <= x[5] <= 35000.0     # entry trigger altitude
        assert 60.0 <= x[6] <= 1500.0        # entry cutoff speed
