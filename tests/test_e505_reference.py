"""E5-05: the committed reproducible full-physical reference mission (task #272).

Fast guards (every CI job, no compute stack needed):

* the files under ``tests/data/mission/reference/e5-05-reference-mission/``
  match their recorded digests (``scripts/e505_reference.py verify``);
* the provenance records code version, dependency matrix, backend pins and
  IERS snapshot, and names the reference as recorded on a COMPUTED ascent
  (OpenTsiolkovsky v1.0.0 engines + the recorded handoff identity);
* the recorded ascent really is a dense computed flight (multi-thousand
  sample tracks, raw backend outputs on disk) -- not a preset two-sample
  handoff; the E5-01 rule holds for the mission of record, not just its
  M-07 sibling;
* the scenario pair (mission of record / replay) declares the full physical
  mission: computed ascent, RTLS from the launch site, coupled branches, the
  pinned E5-05c fixed-script recovery recipe, and the honest ``not_checked``
  propagation;
* the tolerance file exists and the script documents where every band comes
  from;
* E2-04 is still here: its scenario, script, test and frozen reference files
  are present -- the acceptance criterion keeps it as the fast integration
  test, it was not replaced.

Slow (``flightlab-landing`` job, gfold 0.3.2): ``run`` replays the recorded
ascent, re-flies the recovery from the canonical separation handoff and
``compare`` grades every number against the reference within the declared
tolerances. ``cross_env`` (the cross-env job, two real interpreters): the
packaged plan ``crossenv_e505_plan.yaml`` flies the mission across the
tudatpy/gfold boundary and ``crossenv-check`` grades the sealed recovery
against the reference, including the cross-boundary handoff identity.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = Path(__file__).parent / "data" / "mission" / "reference" / "e5-05-reference-mission"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_reference.yaml"
REPLAY = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_replay.yaml"
PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e505_plan.yaml"

# The recording is made by `python scripts/e505_reference.py update` on the
# E5-05a..d re-base onto the RTLS-capable `synthetic_two_stage_rtls` vehicle
# (docs/adr/flightlab-0008); until it has run, every guard below consumes the
# recorded reference, so it arms itself on the directory's existence -- a
# fresh recording turns them green without touching this file.
NOT_RECORDED = pytest.mark.skipif(
    not REF_ROOT.is_dir(),
    reason="E5-05 reference not recorded yet (scripts/e505_reference.py update)")


def _script():
    spec = importlib.util.spec_from_file_location(
        "e505_reference", REPO / "scripts" / "e505_reference.py")
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
            "reference files changed without scripts/e505_reference.py update "
            "-- see tests/test_e505_reference.py docstring")

    def test_provenance_records_code_dependencies_and_backend_pins(self):
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        script = _script()
        assert prov["requires_gfold"] == script.GOLDEN_GFOLD
        assert prov["requires_opentsiolkovsky"] == script.GOLDEN_OT
        assert prov["requires_commit_tag"] == script.COMMIT_TAG
        assert len(prov["git_sha_recording"]) == 40, "code version of the recording"
        tool = prov["toolchain"]
        for pkg in ("python", "platform", "gfold", "pygmo", "numpy", "scipy",
                    "astropy", "astropy_iers_data"):
            assert tool.get(pkg), f"toolchain record missing {pkg!r}"
        assert tool["opentsiolkovsky"] == script.GOLDEN_OT
        assert "IERS-A" in prov["earth_orientation"]

    def test_provenance_names_the_computed_recording(self):
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        rec = prov["recording"]
        assert rec["ascent_engines"] == {"opentsiolkovsky-cli": "1.0.0"}, (
            "the reference ascent must be computed by the pinned binary")
        assert rec["separation_handoff_id"] and len(rec["separation_handoff_id"]) == 64
        assert rec["physical_gate_verdict"] in ("passed", "not_checked")
        assert prov["frozen_inputs"]["plan_options_source"], "tolerance/recipe provenance"


@NOT_RECORDED
class TestRecordedAscentIsComputed:
    def test_recorded_ascent_is_a_dense_computed_flight(self):
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.flightlab.contract import read_run

        run = read_run(REF_ROOT / "mission-optimized" / "launch" / "ascent")
        assert dict(run.engines) == {"opentsiolkovsky-cli": "1.0.0"}
        assert run.verification.optimizer_converged is None, \
            "fixed-program recording: no optimizer in the loop (task #272 re-base)"
        assert run.backend is not None, "a computed flight records its backend"
        assert any(len(t.t_s) > 100 for t in run.tracks), (
            "dense backend tracks: the E5-01 rule (no preset two-sample "
            "handoffs on mission paths) holds for the reference too")
        body_ids = {t.body_id for t in run.tracks}
        assert {"stage1", "stage1_debris", "stage2"} <= body_ids
        assert (REF_ROOT / "mission-optimized" / "launch" / "ascent"
                / "opentsiolkovsky").is_dir(), "raw backend outputs stay with the run"


class TestScenarioPair:
    def test_mission_of_record_computes_and_couples(self):
        s = _yaml(SCENARIO)
        r = _yaml(REPLAY)
        assert s["ascent"]["mode"] == "run_dir", \
            "the recording flies a fixed-program computed ascent, not a pygmo search " \
            "(task #272 re-base: the FL-201 cutoff window degenerates on the restart " \
            "vehicle's peak-thrust burnout; docs/adr/flightlab-0008)"
        assert s["ascent"]["run_dir"] == r["ascent"]["run_dir"], \
            "the replay consumes exactly what the recording flew"
        assert s["recovery"]["drive_from_separation"] is True
        assert str(s["recovery"].get("target_file") or ""), \
            "the recovery gates are graded against a recorded flight target"
        assert s["expected"]["propagation_result"] == "not_checked"
        assert s["launch"]["vehicle"] == r["launch"]["vehicle"] == "synthetic_two_stage_rtls", \
            "the pair flies the RTLS-capable redesign (task #272 re-base)"
        assert s["recovery"]["plan_options"] == r["recovery"]["plan_options"], \
            "the replay pins the same recovery recipe the recording flew"
        assert s["expected"]["parking_orbit"] == r["expected"]["parking_orbit"]
        assert s["expected"]["tolerances"] == r["expected"]["tolerances"]

    @NOT_RECORDED
    def test_replay_replays_the_recorded_computed_ascent(self):
        s = _yaml(REPLAY)
        assert s["ascent"]["mode"] == "run_dir"
        run_dir = REPO / s["ascent"]["run_dir"]
        target = REPO / s["ascent"]["target_file"]
        assert str(REF_ROOT) in str(run_dir.resolve()), "replay consumes the reference"
        assert (run_dir / "manifest.json").is_file() and target.is_file()
        assert s["recovery"]["drive_from_separation"] is True
        for key in ("initial_guess", "search"):
            assert key in s["recovery"]["plan_options"]
            assert key in _yaml(SCENARIO)["recovery"]["plan_options"], (
                "the replay pins the same recovery recipe the recording flew")

    @NOT_RECORDED
    def test_recorded_target_declares_the_rtls_recovery(self):
        import yaml

        doc = yaml.safe_load((REF_ROOT / "mission-optimized" / "launch"
                              / "target.yaml").read_text(encoding="utf-8"))
        assert doc["recovery"]["kind"] == "rtls"


class TestTolerances:
    @NOT_RECORDED
    def test_declared_bands_exist_and_are_documented(self):
        tol = json.loads((REF_ROOT / "tolerances.json").read_text(encoding="utf-8"))
        script = _script()
        for key in ("position_m_abs", "velocity_m_s_abs", "check_value_abs",
                    "report_value_abs", "landing_plan_diagnostic_abs"):
            assert key in tol, f"declared tolerance {key!r} missing"
        doc = script.__doc__ or ""
        assert "Tolerance provenance" in doc, "every band must carry a source"

    def test_e204_fast_integration_test_is_preserved(self):
        # Acceptance: E2-04 stays the fast integration test; it was kept,
        # not replaced. Guard the four artifacts that make it runnable.
        for path in (REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_recovery.yaml",
                     REPO / "scripts" / "e204_reference.py",
                     Path(__file__).parent / "test_e204_reference.py",
                     Path(__file__).parent / "data" / "mission" / "reference" / "e2-04-" "frozen-mission"):
            assert Path(path).exists(), f"E2-04 artifact {path} must not be removed"


@NOT_RECORDED
class TestSlow:
    """The landing stack re-flies the recovery; everything else is replay."""

    def _gfold_ready(self) -> bool:
        try:
            from importlib.metadata import version

            return version("gfold") == "0.3.2"
        except Exception:
            return False

    def test_replay_matches_recorded_reference(self, tmp_path):
        if not self._gfold_ready():
            pytest.skip("needs the pinned landing stack (gfold 0.3.2)")
        script = _script()
        assert script.cmd_run(tmp_path / "e505") == 0
        assert script.cmd_compare(tmp_path / "e505") == 0


@pytest.mark.slow
@NOT_RECORDED
class TestCrossEnvSingleInterpreter:
    """The packaged E5-05 plan end to end on one interpreter: the subprocess
    protocol, the coupled-separation payload, the drive-from-separation
    recovery and the joined verdict. The real two-environment acceptance run
    is the cross_env-marked test in tests/test_crossenv.py's spirit, here as
    TestCrossEnvironmentE505 below."""

    def test_packaged_e505_plan_end_to_end(self, tmp_path):
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.crossenv.orchestrator import run_plan

        report = run_plan(PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": sys.executable, "gfold": sys.executable},
                          verbose=False)
        assert report["overall"]["exchanges_verified"] is True
        assert [s["id"] for s in report["stages"]] == ["ascent", "recovery"]
        for stage in report["stages"]:
            assert stage["status"] == "completed" and stage["exit_code"] == 0
        assert "coupled-separation-handoff" in report["stages"][0]["payloads"]
        stage_report = json.loads((tmp_path / "cx" / "recovery" / "stage-report.json")
                                  .read_text(encoding="utf-8"))
        assert stage_report.get("driven_from_separation") is True
        prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        assert stage_report.get("separation_handoff_id") == \
            prov["recording"]["separation_handoff_id"], (
            "the recovery leg drove from the one recorded separation state")
        try:
            from importlib.metadata import version

            landed = version("gfold") == "0.3.2"
        except Exception:
            landed = False
        if landed:
            assert report["overall"]["mission_ok"] is True


@pytest.mark.cross_env
@pytest.mark.slow
@NOT_RECORDED
class TestCrossEnvironmentE505:
    """The acceptance test, two real environments: the E5-05 plan across the
    tudatpy 3.10 / gfold 3.12 boundary, then ``crossenv-check`` against the
    committed reference (envelopes, handoff identity, recovery numerics)."""

    tudat_py = os.environ.get("ORBITOPT_TUDAT_PYTHON", "")
    gfold_py = os.environ.get("ORBITOPT_GFOLD_PYTHON", "")

    @pytest.mark.skipif(not (tudat_py and gfold_py),
                        reason="set ORBITOPT_TUDAT_PYTHON and ORBITOPT_GFOLD_PYTHON")
    def test_reference_mission_two_environments(self, tmp_path):
        sys.path.insert(0, str(REPO / "src"))
        from orbitopt.crossenv.orchestrator import run_plan

        out = tmp_path / "cx-e505"
        report = run_plan(PLAN, out=out,
                          env_overrides={"tudat": self.tudat_py, "gfold": self.gfold_py},
                          verbose=False)
        asc = report["environment_matrix"]["tudat"]
        rec = report["environment_matrix"]["gfold"]
        assert asc["python_executable"] != rec["python_executable"]
        assert asc["toolchain"]["gfold"] == "absent" and rec["toolchain"]["gfold"] == "0.3.2"
        assert report["overall"]["exchanges_verified"] is True
        assert report["overall"]["mission_ok"] is True, f"verdict: {report['stages']}"
        assert _script().cmd_crossenv_check(out) == 0
