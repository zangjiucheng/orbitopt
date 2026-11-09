"""Checks for the unified mission pipeline (orbitopt.pipeline, task #170).

The fast tests are compute-free: they swap each mission kind's builder for a
stand-in that records stage results the way the real geo-raising /
mars-transfer builders do, and check the lifecycle bookkeeping and the
pipeline-result-1.0 document. The @pytest.mark.slow tests run the real
geo-raising and mars-transfer missions through ``run_pipeline`` and check
that both produce the same standardized structure.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from orbitopt import mission_config
from orbitopt.pipeline import (
    STAGES,
    StageRecorder,
    run_pipeline,
    validate_pipeline_result,
)
from orbitopt.verify import report as verification_report
from orbitopt.viz.scene import body_entry, scene_document

FIXTURES = Path(__file__).parent / "fixtures"

GEO_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "geo-test", "title": "GEO test", "kind": "geo-raising"},
    "output": {"scene_file": "geo-test.json"},
}
MARS_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "mars-test", "title": "Mars test", "kind": "mars-transfer"},
    "output": {"scene_file": "mars-test.json"},
}


def _scene(config):
    trail = {"times": [0.0, 1.0], "positions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    return scene_document(config["mission"]["id"], config["mission"]["title"],
                          [body_entry("spacecraft", "Probe", "#ffffff", "spacecraft", trail=trail)],
                          distance_unit="km")


def fake_builder(config, recorder=None):
    """Mimics a migrated builder: constraints/search/optimize/refine are
    recorded through the recorder; geo skips verification when disabled,
    the verification report rides along on the scene."""
    recorder = StageRecorder() if recorder is None else recorder
    with recorder.stage("constraints") as st:
        st.physical_model = {"name": "cap", "fidelity": "analytic"}
    with recorder.stage("search") as st:
        st.candidates = [{"n": 1, "selected": True}, {"n": 2}]
        st.convergence = {"converged": True}
    recorder.add_time("optimize", 0.25).constraint_violations = {"cap_ms": 0.0}
    with recorder.stage("select") as st:
        st.candidates = [{"n": 1, "selected": True}]
    for attempt in range(2):  # first refine attempt rejected, second accepted
        try:
            with recorder.stage("refine") as st:
                st.candidates.append({"index": attempt})
                if attempt == 0:
                    raise RuntimeError("rejected")
                st.numerical_error = {"miss_km": 1.5}
        except RuntimeError:
            pass
    scene = _scene(config)
    if config.get("verification", {}).get("enabled", True):
        verification_report.attach_to_scene(scene, {
            "kind": "fake", "label": "check",
            "status": verification_report.status_from_checks({"a": True, "b": True}),
            "residuals": [{"name": "final", "position_residual_m": 3.0}],
        })
    else:
        recorder.skip("verify", "verification.enabled is false")
    return scene


@pytest.fixture
def fake_builders(monkeypatch):
    for kind in ("geo-raising", "mars-transfer"):
        monkeypatch.setitem(mission_config._KIND_BUILDERS, kind, (__name__, "fake_builder"))


def _shape(doc):
    """Kind-independent structure of a pipeline-result document."""
    return {
        "keys": sorted(doc),
        "stages": [(s["stage"], sorted(s)) for s in doc["stages"]],
        "verification": sorted(doc["verification"]),
    }


def test_stage_order_is_the_documented_lifecycle():
    assert STAGES == ("mission_config", "constraints", "search", "optimize", "select", "refine", "verify", "export")


def test_recorder_times_stages_and_recovers_from_a_retried_failure():
    rec = StageRecorder()
    with pytest.raises(ValueError), rec.stage("refine"):
        raise ValueError("boom")
    assert rec["refine"].status == "failed" and "boom" in rec["refine"].error
    with rec.stage("refine"):
        pass
    assert rec["refine"].status == "completed" and rec["refine"].error is None
    assert rec.add_time("search", 2.0).elapsed_s == 2.0
    assert rec.skip("verify", "off").status == "skipped"


def test_geo_and_mars_produce_the_same_structure(fake_builders):
    geo = run_pipeline(copy.deepcopy(GEO_CONFIG))
    mars = run_pipeline(copy.deepcopy(MARS_CONFIG))
    for result in (geo, mars):
        doc = result.to_dict()
        validate_pipeline_result(doc)
        json.dumps(doc, allow_nan=False)
        assert doc["status"] == "completed"
        assert [s["stage"] for s in doc["stages"]] == list(STAGES)
        assert all(s["status"] == "completed" for s in doc["stages"])
        assert result.scene["id"] == doc["mission"]["id"]
        assert verification_report.SCENE_REPORTS_KEY not in result.scene
    assert _shape(geo.to_dict()) == _shape(mars.to_dict())


def test_verify_stage_is_summarized_from_reports(fake_builders):
    result = run_pipeline(copy.deepcopy(GEO_CONFIG))
    verify = result.stage("verify")
    assert verify.converged is True
    assert verify.convergence["checks_passed"] == verify.convergence["checks_total"] == 2
    assert verify.numerical_error == {"check.final.position_residual_m": 3.0}
    assert result.to_dict()["verification"] == {"result": "passed", "reports": ["fake:check"]}
    # The viewer-facing copy stays in the scene (the private key is popped).
    published = result.scene[verification_report.SCENE_VERIFICATION_KEY]
    assert published["result"] == "passed" and [r["kind"] for r in published["reports"]] == ["fake"]
    refine = result.stage("refine")
    assert refine.status == "completed" and len(refine.candidates) == 2


def test_disabled_verification_is_skipped_not_passed(fake_builders):
    config = copy.deepcopy(GEO_CONFIG)
    config["verification"] = {"enabled": False}
    doc = run_pipeline(config).to_dict()
    assert doc["stages"][STAGES.index("verify")]["status"] == "skipped"
    assert doc["verification"]["result"] == "not_checked"
    validate_pipeline_result(doc)
    # Nothing published: the viewer then shows "not checked", not a pass.
    assert verification_report.SCENE_VERIFICATION_KEY not in run_pipeline(copy.deepcopy(config)).scene


def minimal_builder(config, recorder=None):
    """A not-yet-migrated builder: ignores the recorder entirely."""
    return _scene(config)


def test_stages_a_kind_lacks_are_marked_skipped(monkeypatch):
    monkeypatch.setitem(mission_config._KIND_BUILDERS, "geo-raising", (__name__, "minimal_builder"))
    doc = run_pipeline(copy.deepcopy(GEO_CONFIG)).to_dict()
    statuses = {s["stage"]: s["status"] for s in doc["stages"]}
    assert statuses["mission_config"] == statuses["export"] == "completed"
    assert {statuses[s] for s in ("constraints", "search", "optimize", "select", "refine", "verify")} == {"skipped"}
    validate_pipeline_result(doc)


def test_invalid_config_fails_in_the_mission_config_stage():
    config = copy.deepcopy(GEO_CONFIG)
    config["bogus"] = 1
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        run_pipeline(config)
    doc = excinfo.value.pipeline_result.to_dict()
    assert doc["status"] == "failed"
    assert doc["stages"][0]["status"] == "failed"
    assert all(s["status"] == "not_run" for s in doc["stages"][1:])
    validate_pipeline_result(doc)

    result = run_pipeline(config, raise_on_error=False)
    assert result.status == "failed" and result.scene is None


def test_write_outputs_writes_scene_report_and_pipeline_result(fake_builders, tmp_path):
    config = copy.deepcopy(MARS_CONFIG)
    result = run_pipeline(config, write_outputs=True, scene_path=tmp_path / "out.json")
    assert result.outputs == {
        "scene_id": "mars-test",
        "scene_file": str(tmp_path / "out.json"),
        "verification_report": str(tmp_path / "out.verification.json"),
        "pipeline_result": str(tmp_path / "out.pipeline.json"),
    }
    doc = json.loads((tmp_path / "out.pipeline.json").read_text())
    validate_pipeline_result(doc)
    assert doc["outputs"]["scene_file"] == str(tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())["id"] == "mars-test"


def test_build_scene_from_config_keeps_reports_attached(fake_builders):
    scene = mission_config.build_scene_from_config(copy.deepcopy(GEO_CONFIG))
    assert [r["kind"] for r in verification_report.pop_from_scene(scene)] == ["fake"]


def test_cli_run_goes_through_the_pipeline(fake_builders, tmp_path, capsys):
    from orbitopt.cli import main

    cfg = tmp_path / "geo.yaml"
    cfg.write_text(json.dumps(GEO_CONFIG))
    main(["run", str(cfg), "-o", str(tmp_path / "geo.json")])
    out = capsys.readouterr().out
    assert "wrote pipeline result" in out and "Pipeline geo-test (geo-raising): COMPLETED" in out
    validate_pipeline_result(json.loads((tmp_path / "geo.pipeline.json").read_text()))


def test_builtin_gui_missions_have_valid_pipeline_configs():
    from orbitopt.missions import builtin_mission_config

    for mission_id in ("goes", "mars"):
        mission_config.validate_mission_config(builtin_mission_config(mission_id))


@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["geo_raising_config.yaml", "mars_transfer_config.yaml"])
def test_real_missions_run_through_the_unified_pipeline(fixture):
    pytest.importorskip("pykep")
    pytest.importorskip("tudatpy")
    config = mission_config.load_mission_config(FIXTURES / fixture)
    result = run_pipeline(config)
    doc = result.to_dict()
    validate_pipeline_result(doc)
    statuses = {s["stage"]: s["status"] for s in doc["stages"]}
    assert doc["status"] == "completed"
    # geo-raising optimizes one burn sequence -- no candidate list, so its
    # select stage is legitimately skipped; mars-transfer selects for real.
    skipped = {"select"} if config["mission"]["kind"] == "geo-raising" else set()
    for stage in ("mission_config", "constraints", "search", "optimize", "select", "refine", "verify", "export"):
        assert statuses[stage] == ("skipped" if stage in skipped else "completed"), (stage, statuses)
    assert result.stage("optimize").candidates
    assert result.stage("refine").numerical_error
    assert result.verification_reports
    gate = doc["gate"]
    assert gate["exported_candidate"] is not None
    assert doc["success"] == (gate["result"] == "passed" and not gate["best_effort"])
