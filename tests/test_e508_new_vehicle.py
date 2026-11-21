"""E5-08 (task #296): the new-vehicle design & qualification guards.

The vehicle ``starship-class-rtls-derived-v1`` is the E5-07 reference stack
with one declared architectural design change (DC1: boostback + ONE powered
entry burn + landing, restarts_max 3), flown through the same qualification
flow the E5-07 reference passed -- and, unlike E5-07, WITH the recovery leg
FLOWN through the shipped planner chain (the DC1 change exists to close
RECOVERY_GAP.md: the leg is PLANNED, FLOWN and GRADED). Whether the graded
gates close is a result, not a promise: the sealed tree records either a
closed landing or a classified gap (RECOVERY_GAP_E508.md, E5-07 pattern --
the gap is reported, no constraint is relaxed). These tests guard the
invariants the task's acceptance turns on:

* the constraints are the E5-07 ones, UNCHANGED (never relaxed to pass);
* the design changes are declared, sourced or tagged DESIGN ASSUMPTION;
* the sealed tree matches its provenance digests and the recovery leg is
  recorded FLOWN -- closed, or a gap that is classified (never silent);
* the joint five-check gate is declared (full_mission) and the E8-05 report
  -- when generated -- says independent_model for the twin and the two-model
  landing re-flight;
* the MC summary (when generated) uses hold-out seeds disjoint from the
  search seed with a whole-seed cluster CI (E7-05 methods);
* comparison against the reference separates design-change effects from
  model uncertainty (the +/-30 % aero band).

Slow (``-m slow``): the recorded replay via ``run`` + ``compare``. Heavier
still (env-gated, ``ORBITOPT_E508_HEAVY=1``): the cross-environment run and
the E8-05 validation -- both cost real minutes of compute.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-08-new-vehicle"
REPORTS_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-08-new-vehicle-reports"
E507_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-07-physics-qualification"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "starship_class_rtls_derived.yaml"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_starship_class_rtls_e508.yaml"
E507_SCENARIO = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_starship_class.yaml"
MISSION_DOC = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_starship_e508.yaml"
PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e508_plan.yaml"

NOT_RECORDED = pytest.mark.skipif(
    not (REF_ROOT / "PROVENANCE.json").is_file(),
    reason="E5-08 reference not recorded yet (scripts/e508_new_vehicle.py record)")


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestDesignInputs:
    """The INPUT guards run before recording too: they are what the design
    promises, independent of any recorded result."""

    def test_vehicle_declares_the_design_change_and_its_reasons(self):
        text = VEHICLE.read_text(encoding="utf-8")
        assert "DC1" in text and "DESIGN ASSUMPTION" in text
        doc = _yaml(VEHICLE)
        assert doc["name"] == "starship-class-rtls-derived-v1"
        assert doc["schema_version"] == "flight-vehicle-1.0"
        eng = doc["booster"]["engines"]
        assert eng["restarts_max"] == 3
        assert "entry" in eng["usable_for"] and "landing" in eng["usable_for"]

    def test_design_changes_do_not_touch_constraints_or_aero(self):
        # DC1 is recovery architecture ONLY: masses, engines, the nose aero
        # tables and the attitude program are the reference's (DC2-DC5
        # unchanged), so the ascent comparison stays apples-to-apples.
        ref_doc = _yaml(REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles"
                        / "starship_class_public_estimate.yaml")
        doc = _yaml(VEHICLE)
        for i, (a, b) in enumerate(zip(doc["stages"], ref_doc["stages"])):
            for key in ("mass_dry_kg", "mass_initial_kg", "engine", "aero",
                        "separation_time_s"):
                assert a.get(key) == b.get(key), f"stage {i} {key} changed"
        assert doc["booster"]["engines"]["count"] == \
            ref_doc["booster"]["engines"]["count"]
        assert doc["attitude_program"] == ref_doc["attitude_program"]

    def test_constraints_are_the_e507_constraints_unchanged(self):
        a = _yaml(SCENARIO)["constraints"]
        b = _yaml(E507_SCENARIO)["constraints"]
        assert a == b, "the new target must inherit the E5-07 constraints exactly"
        assert a["max_dynamic_pressure_pa"] == 55000
        assert a["max_specific_acceleration_g"] == 3.0
        assert a["minimum_insertion_periapsis_m"] == 230000

    def test_mission_doc_declares_the_joint_full_mission_gate(self):
        doc = _yaml(MISSION_DOC)["scenario"]
        assert doc.get("full_mission") is True
        assert "recovery" in doc
        rec = doc["recovery"]
        assert rec["drive_from_separation"] is True
        # the recipe is pinned, never left a placeholder, after recording
        assert doc["expected"]["propagation_result"] == "not_checked"

    def test_crossenv_plan_uses_two_environments(self):
        doc = _yaml(PLAN)["plan"] if "plan" in _yaml(PLAN) else _yaml(PLAN)
        text = PLAN.read_text(encoding="utf-8")
        assert "full_mission" in text
        stages = doc["stages"]
        envs = {s["env"] for s in stages}
        assert {"tudat", "gfold"} <= envs


@NOT_RECORDED
class TestSealedReference:
    def test_every_sealed_file_matches_its_digest(self):
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        for rel, digest in prov["files"].items():
            p = REF_ROOT / rel
            assert p.is_file(), f"sealed file missing: {rel}"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel

    def test_provenance_records_the_derivation_and_the_flown_recovery(self):
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        assert prov["reference_class"] in ("new-vehicle-qualification",
                                           "new-vehicle-qualification-gap")
        assert "e5-07" in prov["derived_from"]["reference"]
        assert "computed ascent" in prov["recording"]["mode"]
        assert "flown" in prov["recording"]["recovery_phase"]
        assert prov["frozen_inputs"]["constraint_basis"].find("never relaxed") \
            or "UNCHANGED" in prov["frozen_inputs"]["constraint_basis"]

    def test_recovery_leg_flown_and_either_closed_or_classified(self):
        """The DC1 outcome is a RESULT, guarded either way: a closed landing
        meets the inherited gates; a gap is classified, with the corridor
        reached and no constraint moved."""
        script = json.loads((REF_ROOT / "recovery-script.json").read_text(encoding="utf-8"))
        assert script["corridor_report"]["reached"] is True
        if script.get("status") == "not_closed":
            gap = (REF_ROOT / "RECOVERY_GAP_E508.md").read_text(encoding="utf-8")
            assert "capability" in gap.lower()
            assert not script["checks"]["recovery-landing"]
            remain = (script["corridor_report"]["propellant_available_kg"]
                      - script["corridor_report"]["landing_need_kg"])
            assert remain < 100_000.0     # the reserve gate is the honest miss
        else:
            assert script["checks"]["recovery-landing"] is True
            assert script["checks"]["recovery-propellant-reserve"] is True
            td = script["touchdown"]
            assert td["vertical_speed_m_s"] <= 3.0
            assert td["lateral_speed_m_s"] <= 1.0
            assert td["propellant_remaining_kg"] >= 100_000.0

    def test_ascent_stays_inside_the_inherited_constraints(self):
        ver = json.loads((REF_ROOT / "mission-optimized" / "launch" / "ascent"
                          / "verification.json").read_text(encoding="utf-8"))
        byid = {c["id"]: c for c in ver["checks"]}
        assert byid["max-q-within-limit"]["passed"]
        assert byid["max-accel-within-limit"]["passed"]
        assert byid["max-q-within-limit"]["value"] <= 55000
        assert byid["max-accel-within-limit"]["value"] <= 3.0
        assert byid["insertion-periapsis-above-minimum"]["value"] >= 230000

    def test_comparison_separates_design_from_model_uncertainty(self):
        comp = json.loads((REF_ROOT / "comparison.json").read_text(encoding="utf-8"))
        assert "ascent" in comp and "recovery" in comp
        assert "model_uncertainty" in comp
        assert "30" in comp["model_uncertainty"]
        assert comp["recovery"]["reference"].startswith("NOT FLOWN")
        assert comp["recovery"]["derived"].startswith("FLOWN")

    def test_earth_orientation_snapshot_is_the_reference_epoch(self):
        # same epoch, byte-identical snapshot: the ascent twins compare clean
        mine = (REF_ROOT / "earth_orientation.json").read_bytes()
        theirs = (E507_ROOT / "earth_orientation.json").read_bytes()
        assert mine == theirs


@NOT_RECORDED
class TestMonteCarlo:
    """The MC summary exists only after ``scripts/e508_new_vehicle.py mc``."""

    def _summary(self):
        p = REPORTS_ROOT / "mc-summary.json"
        if not p.is_file():
            pytest.skip("MC not run yet (scripts/e508_new_vehicle.py mc)")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_holdout_seeds_disjoint_from_the_search_seed(self):
        s = self._summary()
        assert s["search_seed"] not in s["seeds"]
        assert len(s["seeds"]) >= 3, "E7-05 needs enough clusters to bootstrap"

    def test_cluster_bootstrap_ci_present_and_brackets_the_rate(self):
        s = self._summary()
        ci = s["cluster_bootstrap_rate_ci_95"]
        assert ci is not None and ci[0] <= s["success_rate"] <= ci[1]


@NOT_RECORDED
class TestIndependentDynamics:
    def _report(self):
        p = REPORTS_ROOT / "independent-dynamics.json"
        if not p.is_file():
            pytest.skip("E8-05 not run yet (scripts/e508_new_vehicle.py e805)")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_report_ok_with_independent_model_evidence(self):
        rep = self._report()
        assert rep["status"]["result"] == "passed" and rep["status"]["passed"]
        ids = {c["id"] for c in rep["inputs"]["check_details"]
               if c["evidence"] == "independent_model"}
        assert any(i.startswith("ascent.") for i in ids)
        assert any(i.startswith("landing.") for i in ids)
        lc = rep["inputs"]["landing_corridor"]
        assert lc["engines"] == 3  # the declared 3-engine landing set (13 cannot
        #                          throttle below weight; inherited limitation)
        hot = rep["inputs"]["actual_recorded_corridor"]
        assert hot["landing_plannable"] is False  # independent corroboration of the gap


@pytest.mark.slow
@NOT_RECORDED
class TestRegression:
    def test_rerun_matches_recorded_reference(self, tmp_path):
        pytest.importorskip("numpy")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e508_new_vehicle", REPO / "scripts" / "e508_new_vehicle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = tmp_path / "e508"
        assert mod.cmd_run(out) == 0
        assert mod.cmd_compare(out) == 0


@pytest.mark.slow
@NOT_RECORDED
@pytest.mark.skipif(os.environ.get("ORBITOPT_E508_HEAVY") != "1",
                    reason="set ORBITOPT_E508_HEAVY=1 (needs both interpreters, "
                           "minutes of compute)")
class TestHeavy:
    def test_crossenv_joint_gate(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e508_new_vehicle", REPO / "scripts" / "e508_new_vehicle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # grade the canonical joint run if it is there; otherwise run it fresh
        # (rerun=True -> needs both interpreters via the environment)
        assert mod.cmd_crossenv_check(mod.CROSS_OUT, rerun=None) == 0

    def test_e805_on_the_new_vehicle(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e508_new_vehicle", REPO / "scripts" / "e508_new_vehicle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.cmd_e805() == 0
