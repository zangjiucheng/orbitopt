"""Artemis II (lunar-free-return) through the unified pipeline (task #181).

The fast tests stub the targeter, the propagation and the ephemerides, then
run the real lunar-free-return builder through ``run_pipeline``. They check
that every stage has a record and that the Verification Gate shows the
verdict. They still need pykep/tudatpy to import orbitopt.viz.mission_timeline.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("pykep")
pytest.importorskip("tudatpy")

from orbitopt import mission_config
from orbitopt.pipeline import STAGES, run_pipeline, validate_pipeline_result
from orbitopt.verify import report as verification_report
from orbitopt.verify.differential_correction import (
    EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
    EARTH_RADIUS_KM,
    FreeReturnTargetingResult,
)

CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "artemis2-test", "title": "Artemis II test", "kind": "lunar-free-return"},
    "departure": {"mjd2000": 9709.0},
    "output": {"scene_file": "artemis2-test.json"},
}
ENTRY_EPOCH_S = 9.0 * 86400.0


def _report(passed=True):
    return {
        "kind": "lunar-free-return-entry-corridor", "label": "artemis",
        "status": verification_report.status_from_checks({"targeting.converged": True,
                                                          "residual.entry_flight_path_angle_in_corridor": passed}),
        "residuals": [{"name": "entry", "position_residual_m": 12.0}],
    }


def _targeting(*, converged=True, in_corridor=True, report=None):
    return FreeReturnTargetingResult(
        delta_v=np.array([3100.0, 0.0, 0.0]), converged=converged, in_corridor=in_corridor,
        iterations=4, coast_duration=3.5 * 86400.0, perilune_epoch=4.0 * 86400.0,
        perilune_distance_km=8282.4, b_plane_angle_deg=12.0,
        return_perigee_km=EARTH_RADIUS_KM + 30.0, target_return_perigee_km=EARTH_RADIUS_KM + 30.0,
        entry_epoch=ENTRY_EPOCH_S, entry_state=None, entry_speed_m_s=11_000.0,
        entry_flight_path_angle_deg=-6.0, residual_history_km=[[900.0, 40.0], [3.0, 0.5]],
        report=_report() if report is None else report,
    )


def _crossing(fpa_deg):
    from orbitopt.verify.tudat_propagate import AltitudeCrossingResult

    r = (EARTH_RADIUS_KM + EARTH_ENTRY_INTERFACE_ALTITUDE_KM) * 1000.0
    v = 11_000.0
    state = np.array([r, 0.0, 0.0, v * np.sin(np.radians(fpa_deg)), v * np.cos(np.radians(fpa_deg)), 0.0])
    return AltitudeCrossingResult(epoch=ENTRY_EPOCH_S, altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
                                  spacecraft_state=state)


@pytest.fixture
def stubbed(monkeypatch):
    """Compute-free stand-ins for SPICE, Lambert, the targeter and TudatPy."""
    from orbitopt.verify.tudat_propagate import PropagationResult
    from orbitopt.viz import mission_timeline as mt

    state = {"targeting": _targeting(), "entry_fpa": -6.0}
    monkeypatch.setattr(mt, "mjd2000_to_ephemeris_seconds", lambda mjd: 0.0)
    monkeypatch.setattr(mt, "moon_state", lambda mjd: (np.array([3.84e8, 0.0, 0.0]), np.array([0.0, 1022.0, 0.0])))
    monkeypatch.setattr(mt, "solve_lambert_single",
                        lambda *a, **kw: [(np.array([0.0, 10_900.0, 0.0]), np.zeros(3))])
    monkeypatch.setattr(mt, "target_free_return_entry_corridor", lambda **kw: state["targeting"])
    n = 40
    epochs = np.linspace(0.0, 9.5 * 86400.0, n)
    phase = np.linspace(0.0, 2.0 * np.pi, n)
    states = np.column_stack([3.0e8 * np.sin(phase / 2) + 7.0e6, 1.0e7 * np.sin(phase), np.zeros(n),
                              np.zeros(n), np.full(n, 1.0e3), np.zeros(n)])
    monkeypatch.setattr(mt, "propagate_multi_arc",
                        lambda *a, **kw: PropagationResult(epochs=epochs, states=states, final_state=states[-1]))
    monkeypatch.setattr(mt, "find_altitude_crossing", lambda *a, **kw: _crossing(state["entry_fpa"]))
    monkeypatch.setattr(mt, "body_position_at_absolute_epoch", lambda body, t: np.array([3.84e8, 0.0, 0.0]))
    monkeypatch.setattr(mt, "body_fixed_to_eclipj2000_matrix", lambda body, mjd: np.eye(3))
    return state


def _statuses(doc):
    return {s["stage"]: s["status"] for s in doc["stages"]}


def test_artemis_records_every_stage_and_the_gate(stubbed):
    result = run_pipeline(copy.deepcopy(CONFIG))
    doc = result.to_dict()
    validate_pipeline_result(doc)
    assert doc["status"] == "completed" and doc["success"] is True
    assert _statuses(doc) == {name: "completed" for name in STAGES}
    for name in ("constraints", "search", "optimize", "select", "refine", "verify"):
        st = result.stage(name)
        assert st.candidates, name
        assert st.physical_model.get("fidelity"), name
        assert st.elapsed_s >= 0.0
    opt = result.stage("optimize")
    assert opt.converged is True and opt.convergence["iterations"] == 4
    assert opt.numerical_error["perilune_residual_km"] == pytest.approx(3.0)
    assert opt.constraint_violations["entry_corridor_deg"] == 0.0
    assert "perilune_distance_vs_targeter_km" in result.stage("refine").numerical_error
    verify = result.stage("verify")
    assert verify.converged is True
    assert verify.numerical_error["entry_fpa_vs_targeter_deg"] == pytest.approx(0.0, abs=1e-9)
    gate = doc["gate"]
    assert gate["exported_candidate"] == 0 and gate["result"] == "passed"
    assert gate["counts"]["accepted"] == 1
    assert doc["verification"]["result"] == "passed"
    assert result.scene["id"] == "artemis2-test"


def test_failed_report_is_a_best_effort_export(stubbed):
    stubbed["targeting"] = _targeting(report=_report(passed=False))
    doc = run_pipeline(copy.deepcopy(CONFIG)).to_dict()
    validate_pipeline_result(doc)
    assert doc["success"] is False
    assert doc["gate"]["best_effort"] is True and doc["gate"]["result"] == "failed"
    assert doc["stages"][STAGES.index("verify")]["convergence"]["converged"] is False


def test_disabled_verification_leaves_the_gate_not_checked(stubbed):
    config = copy.deepcopy(CONFIG)
    config["verification"] = {"enabled": False}
    result = run_pipeline(config)
    doc = result.to_dict()
    assert not result.verification_reports
    assert doc["gate"]["result"] == "not_checked" and doc["success"] is False
    assert any("not verified" in w for w in doc["gate"]["warnings"])


def test_rejected_targeting_fails_in_select_and_is_rejected_by_the_gate(stubbed):
    stubbed["targeting"] = _targeting(converged=False)
    result = run_pipeline(copy.deepcopy(CONFIG), raise_on_error=False)
    statuses = _statuses(result.to_dict())
    assert statuses["optimize"] == "completed" and statuses["select"] == "failed"
    assert statuses["refine"] == "not_run" and statuses["export"] == "not_run"
    assert result.gate.candidates[0]["verdict"] == "rejected"


def test_out_of_corridor_export_fails_in_verify(stubbed):
    stubbed["entry_fpa"] = -9.0
    result = run_pipeline(copy.deepcopy(CONFIG), raise_on_error=False)
    statuses = _statuses(result.to_dict())
    assert statuses["refine"] == "completed" and statuses["verify"] == "failed"
    assert result.gate.candidates[0]["verdict"] == "rejected"


def test_builtin_artemis_runs_through_the_pipeline(monkeypatch):
    from orbitopt import missions

    config = missions.builtin_mission_config("artemis2")
    mission_config.validate_mission_config(config)
    assert config["mission"]["id"] == "artemis2-mission"
    seen = []
    monkeypatch.setattr("orbitopt.pipeline.run_pipeline",
                        lambda cfg: seen.append(cfg) or type("R", (), {"scene": {"id": "x"}})())
    assert missions._run_builtin_pipeline("artemis2") == {"id": "x"}
    assert seen[0]["mission"]["kind"] == "lunar-free-return"
