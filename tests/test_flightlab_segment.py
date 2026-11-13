"""M-03 (task #221): Flight Lab as a mission-pipeline segment.

Fast tests reuse the FL-004 CLI stub (recorded v1.0.0 outputs) so the
segment -> simulate -> verdict pipeline runs without the real binary, and
run the ``flight-lab`` mission kind end-to-end through
``orbitopt.pipeline.run_pipeline``: the run lands in the standardized
pipeline result (stages + gate), the verification report enters the unified
run record, and an honest red verdict flows through as a best-effort export
-- never a silent pass. The reverse case (verdict green when the target
grades the flight it actually flew) proves ``result.success`` is reachable.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from orbitopt.flightlab.contract import (
    Check,
    FlightRun,
    Verification,
    load_target,
    read_run,
)
from orbitopt.flightlab.segment import FlightLabSegment, flight_run_report
from orbitopt.flightlab.target_export import save_target
from orbitopt.mission_config import is_valid_mission_config, known_kinds, validate_mission_config
from orbitopt.pipeline import run_pipeline, validate_pipeline_result
from orbitopt.verify.report import RESULT_FAILED, RESULT_PASSED, status_result, validate_report

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
EXAMPLE_CONFIG = REPO / "src" / "orbitopt" / "data" / "flightlab" / "mission_leo_two_stage.yaml"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def target():
    return load_target(TARGET)


def _config(tmp_path, target_file=TARGET, **fl):
    fl.setdefault("vehicle_file", str(VEHICLE))
    fl.setdefault("refly", False)
    return {
        "schemaVersion": "1.0",
        "mission": {"id": "fl-seg", "title": "Flight Lab segment run", "kind": "flight-lab"},
        "flightlab": {"target_file": str(target_file), "run_dir": str(tmp_path / "flightrun"), **fl},
        "output": {"scene_file": str(tmp_path / "scene.json")},
    }


# ----------------------------------------------------------------- segment


def test_segment_run_collects_flightrun_and_handoff(stub, target, tmp_path):
    seg = FlightLabSegment(target, vehicle_path=VEHICLE, run_dir=tmp_path / "seg-run",
                           refly=False, target_path=TARGET)
    result = seg.run()

    assert (tmp_path / "seg-run" / "manifest.json").is_file()
    assert read_run(tmp_path / "seg-run").run_id == result.run.run_id
    assert result.events and any(e.kind == "seco" for e in result.events)
    # The outgoing handoff state is the SECO insertion state, frame/epoch/
    # r/v/mass straight off the manifest (GCRS after the frame bridge).
    h = result.handoff
    assert h["frame"] == "GCRS" and h["time_scale"] == "UTC"
    assert len(h["position_m"]) == 3 and len(h["velocity_m_s"]) == 3
    assert h["mass_kg"] > 0 and h["run_id"] == result.run.run_id
    from orbitopt.flightlab.checks import insertion_state

    ref = insertion_state(result.run)
    assert h["position_m"] == [float(x) for x in ref["position_m"]]
    assert h["mass_kg"] == pytest.approx(ref["mass_kg"])
    d = result.to_dict()
    assert d["selected"] is True and d["run_id"] == result.run.run_id
    assert d["physics_verified"] is None  # refly skipped: not assessed, never passed


def test_segment_takes_no_incoming_handoff_state(target, tmp_path):
    seg = FlightLabSegment(target, vehicle_path=VEHICLE, run_dir=tmp_path / "x")
    with pytest.raises(ValueError, match="pad-start"):
        seg.run(initial_state={"frame": "GCRS"})


def test_handoff_state_rejects_unknown_seco_body(target):
    run = FlightRun(run_id="r", target_sha256="0" * 64, vehicle_sha256="0" * 64,
                    git_sha="x", created_utc="2026-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="no body tracks"):
        FlightLabSegment.handoff_state(run)


# ------------------------------------------------------------ report bridge


def test_flight_run_report_is_a_valid_verification_report(target):
    v = Verification(optimizer_converged=True, physics_verified=None,
                     mission_constraints_passed=False,
                     checks=(Check("insertion-apoapsis-within-tolerance", False,
                                   value=533000.0, limit=20000.0, unit="m",
                                   reason_code="INSERTION_APOAPSIS_OUT_OF_TOLERANCE"),))
    run = FlightRun(run_id="hand-0001", target_sha256="0" * 64, vehicle_sha256="0" * 64,
                    git_sha="deadbeef", created_utc="2026-11-10T12:00:00Z", verification=v)
    report = flight_run_report(run, target)

    validate_report(report)  # schema + digest
    assert report["kind"] == "flightlab" and report["label"] == "hand-0001"
    assert status_result(report["status"]) == RESULT_FAILED
    assert report["status"]["checks"]["check.insertion-apoapsis-within-tolerance"] is False
    # Tri-state honesty: physics_verified was not assessed -- it must appear
    # as null in the verdict block and must NOT be one of the (all-must-pass)
    # status checks.
    assert report["targets"]["physics_verified"] is None
    assert "physics_verified" not in report["status"]["checks"]
    assert report["targets"]["reason_codes"] == {
        "insertion-apoapsis-within-tolerance": "INSERTION_APOAPSIS_OUT_OF_TOLERANCE"}

    green = replace(run, verification=Verification(
        optimizer_converged=True, physics_verified=True, mission_constraints_passed=True,
        checks=(Check("max-q-within-limit", True),)))
    green_report = flight_run_report(green)
    assert status_result(green_report["status"]) == RESULT_PASSED
    assert green_report["status"]["checks"]["physics_verified"] is True


# ------------------------------------------------------------ config layer


def test_flight_lab_kind_is_registered():
    assert "flight-lab" in known_kinds()
    assert is_valid_mission_config(yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8")))


def test_flight_lab_config_schema_is_strict():
    base = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(base))
    bad["flightlab"]["target_filez"] = bad["flightlab"]["target_file"]
    with pytest.raises(Exception, match="target_filez|Additional properties"):
        validate_mission_config(bad)


# ------------------------------------------------------------- the pipeline


def test_pipeline_runs_flight_lab_segment_with_honest_red_verdict(stub, tmp_path):
    result = run_pipeline(_config(tmp_path))

    assert result.status == "completed"
    assert result.mission["kind"] == "flight-lab"
    validate_pipeline_result(result.to_dict())

    search = result.stage("search")
    assert search.status == "completed"
    assert len(search.candidates) == 1 and search.candidates[0]["selected"] is True
    assert search.candidates[0]["run_dir"] == str(tmp_path / "flightrun")
    assert search.physical_model["fidelity"] == "two-body"
    assert "check.insertion-apoapsis-within-tolerance" in search.constraint_violations

    verify = result.stage("verify")
    assert verify.status == "completed"
    assert verify.converged is False  # the tuned program overshoots apogee: honest red
    assert verify.convergence["physics_verified"] is None
    assert verify.convergence["checks_total"] >= 3

    # The unified run record: one flightlab report drives the gate.
    assert len(result.verification_reports) == 1
    report = result.verification_reports[0]
    assert report["kind"] == "flightlab"
    validate_report(report)
    assert result.verification_result == RESULT_FAILED
    assert not result.success and result.gate.result == RESULT_FAILED
    assert result.warnings  # best-effort export is never silent
    assert "warnings" in result.scene

    # The run directory is a full FlightResult the contract can re-read.
    assert read_run(tmp_path / "flightrun").run_id == report["label"]


def test_pipeline_flight_lab_success_when_target_grades_the_flight(stub, target, tmp_path):
    """Grade the stub's actual insertion (a 400 x ~533 km orbit, no declared
    recovery): every check then passes and the pipeline's success becomes
    True -- success is reachable, not hard-wired red."""
    probe = FlightLabSegment(target, vehicle_path=VEHICLE, run_dir=tmp_path / "probe",
                             refly=False, run_recovery=False, target_path=TARGET)
    flown = probe.run()
    by_id = {c.id: c for c in flown.run.verification.checks}
    apo = by_id["insertion-apoapsis-within-tolerance"].value
    peri = by_id["insertion-periapsis-within-tolerance"].value
    inc = by_id["insertion-inclination-within-tolerance"].value
    assert None not in (apo, peri, inc)

    graded = replace(target, target_orbit=replace(
        target.target_orbit, periapsis_altitude_m=peri, apoapsis_altitude_m=apo,
        inclination_deg=inc), recovery=replace(target.recovery, kind="none",
                                               min_propellant_reserve_kg=None))
    graded_path = save_target(graded, tmp_path / "graded_target.yaml")

    result = run_pipeline(_config(tmp_path, target_file=graded_path))
    assert result.success is True
    assert result.verification_result == RESULT_PASSED
    assert result.stage("verify").converged is True
    assert not result.warnings
    assert result.gate.candidates[0]["verdict"] == "accepted"


def test_pipeline_flight_lab_reports_write_outputs(stub, tmp_path):
    config = _config(tmp_path)
    result = run_pipeline(config, write_outputs=True, raise_on_error=False)

    scene_file = Path(result.outputs["scene_file"])
    assert scene_file.is_file()
    report_file = Path(result.outputs["verification_report"])
    written = json.loads(report_file.read_text(encoding="utf-8"))
    assert written["kind"] == "flightlab"
    validate_report(written)
    pipeline_file = Path(result.outputs["pipeline_result"])
    doc = json.loads(pipeline_file.read_text(encoding="utf-8"))
    validate_pipeline_result(doc)
    assert doc["mission"]["kind"] == "flight-lab"
    assert doc["gate"]["exported_candidate"] == 0


def test_skip_verify_disables_refly_through_schema(stub, tmp_path):
    config = _config(tmp_path)
    config.setdefault("verification", {})["enabled"] = False
    validate_mission_config(config)  # --skip-verify injects this section
    result = run_pipeline(config)
    assert result.stage("verify").convergence["physics_verified"] is None
