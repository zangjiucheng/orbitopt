"""E5-07: the committed Physics Qualification Reference mission (task #291).

The reference is a Starship-class (public-estimate) LEO insertion computed
end to end with OpenTsiolkovsky v1.0.0, graded under the E5-06 joint gate
-- and it is deliberately NOT the E5-05 mission: that one is the Software
Integration Reference (synthetic vehicle, relaxed-then-assumed limits,
proves the chain works); this one carries public-sourced constraints and
reports the one leg it cannot close (see RECOVERY_GAP.md) instead of
relaxing anything to close it.

Fast guards (every CI job, no compute stack needed):

* the files under ``tests/data/mission/reference/e5-07-physics-qualification/``
  match their recorded digests (``scripts/e507_reference.py verify``);
* the recorded replay (``mission/``) matches the mission of record within
  the declared tolerances (``compare``, files only);
* the provenance names a COMPUTED OpenTsiolkovsky ascent, records the
  toolchain/dependency matrix and the frozen Earth-orientation snapshot;
* the mission doc flies no recovery, declares no ``full_mission``, and the
  reference set ships RECOVERY_GAP.md with the verbatim planner rejections
  -- the unclosed leg is reported, not relaxed;
* the flight target keeps max-Q on a public source and every ASSUMPTION-tagged
  limit visible, and the vehicle keeps the aero +/-30% sensitivity note;
* the two references stay separate: E5-05 (Software Integration Reference)
  is still recorded and neither doc string conflates the roles.

Slow (``flightlab-landing`` job, OpenTsiolkovsky CLI available): ``run``
re-derives the whole chain from the recorded ascent and ``compare`` grades
it against the reference. ``cross_env`` (two real interpreters): the
packaged plan ``crossenv_e507_plan.yaml`` flies the mission leg in the
tudatpy environment -- the parking-orbit propagation PHYSICALLY EVIDENCED
there -- and ``crossenv-check`` grades it against the reference, including
the Earth-orientation snapshot identity; the joint full-mission gate is
asserted NOT-DECLARED (the recovery leg is unrepresentable, RECOVERY_GAP.md).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = Path(__file__).parent / "data" / "mission" / "reference" / "e5-07-physics-qualification"
MISSION_DOC = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_starship_e507.yaml"
PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e507_plan.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_starship_class.yaml"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "starship_class_public_estimate.yaml"
E505_REF_ROOT = Path(__file__).parent / "data" / "mission" / "reference" / "e5-05-reference-mission"

NOT_RECORDED = pytest.mark.skipif(
    not REF_ROOT.is_dir(),
    reason="E5-07 reference not recorded yet (scripts/e507_reference.py update)")


def _script():
    spec = importlib.util.spec_from_file_location(
        "e507_reference", REPO / "scripts" / "e507_reference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))["scenario"]


@NOT_RECORDED
class TestProvenance:
    def test_committed_reference_matches_its_provenance(self):
        assert _script().cmd_verify() == 0, (
            "reference files changed without scripts/e507_reference.py update")

    def test_provenance_names_a_computed_opentsiolkovsky_ascent(self):
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        assert prov["reference_class"] == "physics-qualification"
        assert "computed ascent" in prov["recording"]["mode"]
        assert prov["recording"]["ascent_engines"]["opentsiolkovsky-cli"] == "1.0.0"
        assert prov["toolchain"]["opentsiolkovsky"] == "1.0.0"
        # the report must never let this read as the software-integration run
        assert "software-integration" in prov["distinct_from"].lower()

    def test_frozen_earth_orientation_snapshot_matches_its_digest(self):
        import hashlib

        blob = (REF_ROOT / "earth_orientation.json").read_bytes()
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        assert prov["files"]["earth_orientation.json"] == hashlib.sha256(blob).hexdigest()


@NOT_RECORDED
class TestReferenceFiles:
    def test_recorded_replay_matches_the_mission_of_record(self):
        assert _script().cmd_compare(REF_ROOT / "mission") == 0

    def test_recorded_ascent_is_a_dense_computed_flight(self):
        asc = REF_ROOT / "mission-optimized" / "launch" / "ascent" / "opentsiolkovsky"
        assert (asc / "summary.json").is_file()
        assert (asc / "bodies" / "stage1.csv").is_file()
        trajs = REF_ROOT / "mission-optimized" / "launch" / "ascent" / "trajectories"
        n = 0
        for f in [asc / "trajectory.csv", asc / "bodies" / "stage1.csv",
                  trajs / "stage1.csv", trajs / "stage1_debris.csv", trajs / "stage2.csv"]:
            with f.open(encoding="utf-8") as fh:
                n += sum(1 for _ in fh)
        # 10 Hz sampled-data RK4 plant outputs downsampled at the recorder's
        # cadence: a fixed-program 516 s flight yields hundreds of samples
        # per body -- a two-sample preset handoff would be under a dozen.
        assert n > 800, f"recorded ascent has only {n} samples -- not a computed flight"
        events = {e["kind"] for e in
                  json.loads((asc / "summary.json").read_text(encoding="utf-8"))["events"]}
        assert {"liftoff", "max_q", "separation", "burnout"} <= events

    def test_recovery_gap_report_documents_the_planner_rejections(self):
        text = (REF_ROOT / "RECOVERY_GAP.md").read_text(encoding="utf-8")
        assert "entry_burns=0 -> ValueError" in text
        assert "3 ignitions, vehicle allows 2" in text
        assert "relaxation" in text.lower()

    def test_declared_tolerances_are_finite_and_positive(self):
        tol = json.loads((REF_ROOT / "tolerances.json").read_text(encoding="utf-8"))
        for key in ("achieved_element_rel", "achieved_element_abs_m",
                    "achieved_inc_abs_deg", "epoch_abs_s"):
            assert float(tol[key]) > 0.0, key
            assert float(tol[key]) < 1.0, key


class TestScenarioDeclaration:
    """These guard the reference INPUTS, so they run even before recording."""

    def test_mission_doc_flies_no_recovery_and_declares_no_full_mission(self):
        text = MISSION_DOC.read_text(encoding="utf-8")
        doc = _yaml(MISSION_DOC)
        assert doc.get("full_mission") is not True
        assert "recovery" not in doc          # no recovery stage declared
        assert doc["expected"]["propagation_result"] == "not_checked"
        # the recording itself keeps propagation not_checked; the E5-04
        # phase is evidenced on the cross-environment leg instead
        assert "physics qualification" in text.lower()
        assert "recovery reported not flown" in text.lower()

    def test_flight_target_keeps_public_source_and_assumption_tags(self):
        import yaml

        text = TARGET.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        assert "ASSUMPTION" in text
        cons = doc["constraints"]
        assert cons["max_dynamic_pressure_pa"] == 55000
        assert cons["max_specific_acceleration_g"] == 3.0
        assert cons["minimum_insertion_periapsis_m"] == 230000
        # the recorded flight must sit INSIDE these, not the other way round
        assert "never relaxed" in text.lower() or "never relaxed" in text.replace(
            "\n", " ").lower()

    def test_vehicle_keeps_the_aero_sensitivity_note(self):
        text = VEHICLE.read_text(encoding="utf-8")
        assert "30" in text  # +/-30 % aerodynamic sensitivity, recorded range

    def test_software_integration_reference_is_still_recorded(self):
        # E5-07 must not replace E5-05: both references coexist, and the
        # task's separation of roles is only meaningful if both are here.
        assert (E505_REF_ROOT / "PROVENANCE.json").is_file()
        assert (REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_reference.yaml").is_file()


@pytest.mark.slow
@NOT_RECORDED
class TestRegression:
    """``run`` re-derives the mission from the recorded ascent (no new
    ascent computation); needs the flightlab runtime (numpy, the repo's
    mission runner) in the test interpreter."""

    def test_rerun_matches_recorded_reference(self, tmp_path):
        try:
            import numpy  # noqa: F401
        except Exception:
            pytest.skip("needs numpy (flightlab runtime)")
        script = _script()
        assert script.cmd_run(tmp_path / "e507") == 0
        assert script.cmd_compare(tmp_path / "e507") == 0


@pytest.mark.slow
@NOT_RECORDED
class TestCrossEnvSingleInterpreter:
    """The packaged E5-07 plan end to end on one interpreter: the mission
    leg, the sealed exchange, the propagation evidenced in the stage
    environment, and the joined gate reported NOT-DECLARED for full_mission
    (the recovery evidence gap must be visible in ``problems``)."""

    def test_packaged_e507_plan_end_to_end(self, tmp_path):
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.crossenv.orchestrator import run_plan

        report = run_plan(PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": sys.executable}, verbose=False)
        assert report["overall"]["exchanges_verified"] is True
        assert [s["id"] for s in report["stages"]] == ["mission"]
        stage = report["stages"][0]
        assert stage["status"] == "completed" and stage["exit_code"] == 0
        assert stage["mission_ok"] is True
        joint = report["joint_qualification"]
        assert joint["full_mission_declared"] is False
        assert any("recovery" in p for p in joint["problems"])
        try:
            import tudatpy  # noqa: F401

            have_tudat = True
        except Exception:
            have_tudat = False
        if not have_tudat:
            # crossenv-check requires the parking propagation to be PHYSICALLY
            # EVIDENCED in the stage environment; the mission runner honestly
            # declines it (not_assessed: compute stack missing) without
            # tudatpy here -- that declination is itself the E5-04 rule, not
            # a failure. The full check is TestCrossEnvironmentE507.
            pytest.skip("stage-environment propagation evidence needs tudatpy")
        assert _script().cmd_crossenv_check(tmp_path / "cx") == 0


@pytest.mark.cross_env
@pytest.mark.slow
@NOT_RECORDED
class TestCrossEnvironmentE507:
    """The acceptance test across a real environment boundary: the E5-07
    mission leg (run-dir ascent, parking-orbit propagation) in the tudatpy
    interpreter, then ``crossenv-check`` against the committed reference --
    which requires the propagation to be PHYSICALLY EVIDENCED there and the
    Earth-orientation snapshot to be the reference's own."""

    tudat_py = os.environ.get("ORBITOPT_TUDAT_PYTHON", "")

    @pytest.mark.skipif(not tudat_py, reason="set ORBITOPT_TUDAT_PYTHON")
    def test_reference_mission_cross_environment(self, tmp_path):
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.crossenv.orchestrator import run_plan

        out = tmp_path / "cx-e507"
        report = run_plan(PLAN, out=out,
                          env_overrides={"tudat": self.tudat_py}, verbose=False)
        asc = report["environment_matrix"]["tudat"]
        assert asc["python_executable"] == self.tudat_py
        assert report["overall"]["exchanges_verified"] is True
        assert report["overall"]["mission_ok"] is True, f"verdict: {report['stages']}"
        joint = report["joint_qualification"]
        assert joint["full_mission_declared"] is False
        assert joint["physical_mission_verdict"] != "passed"  # recovery not flown
        assert _script().cmd_crossenv_check(out) == 0
        # a foreign Earth-orientation snapshot cannot be blessed
        stage_report = out / "mission" / "stage-report.json"
        original = stage_report.read_bytes()
        changed = json.loads(original)
        changed["earth_orientation"]["snapshot_sha256"] = "sha256:" + "0" * 64
        try:
            stage_report.write_text(json.dumps(changed))
            assert _script().cmd_crossenv_check(out) == 1
        finally:
            stage_report.write_bytes(original)
