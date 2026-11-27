"""E14-09 Optimize workspace widgets, offscreen: every state renders through the real job threads, on the
offline mock provider. The states are the ones ``scripts/capture_optimization_studio.py`` photographs."""
from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("scipy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_optimization_studio.py"
CAPTURE = runpy.run_path(str(SCRIPT))


def _has_ink(widget) -> bool:
    image = widget.grab().toImage()
    seen = {image.pixel(x, y) for x in range(0, image.width(), 7) for y in range(0, image.height(), 7)}
    return len(seen) > 8


def _badge(badge) -> str:
    return badge._label.text()


@pytest.fixture
def workspace(tmp_path):
    app, ws = CAPTURE["make_workspace"](tmp_path)
    yield app, ws
    CAPTURE["finish"](app, ws)


@pytest.mark.parametrize("name", list(CAPTURE["STATES"]))
def test_every_state_renders(workspace, name):
    app, ws = workspace
    driver, phase = CAPTURE["STATES"][name]
    driver(app, ws)
    app.processEvents()
    assert ws.session.phase.value == phase
    assert _has_ink(ws)
    assert "OFFLINE" in ws.banner.text() and ws.banner.property("offline") == "true"
    statuses = {name: _badge(badge) for name, (badge, _reason) in ws.status_rows.items()}
    assert len(statuses) == 4                                    # never one combined check mark
    if name == "idle":
        assert not ws.run_button.isEnabled() and not ws.confirm_button.isEnabled()
        assert ws.prompt_edit.toPlainText().startswith("Minimize x squared")
    elif name == "clarification":
        assert ws.questions_box.isVisibleTo(ws) and len(ws.answer_edits) == 1
        assert "unit cost" in ws.questions_form.itemAt(0).widget().text()
        assert "not valid IR" in ws.digest_label.text()
    elif name == "editing-error":
        assert "/variables/0/upper" in ws.issues_label.text() and not ws.confirm_button.isEnabled()
        cell = ws.tables["variables"].item(0, 3)
        assert cell.text() == "ten" and cell.toolTip()
    elif name == "running":
        assert ws.cancel_button.isEnabled() and not ws.run_button.isEnabled()
        assert _badge(ws.run_badge) == "RUNNING" and ws.iteration_table.rowCount() == 2
        assert not ws.provider_combo.isEnabled() and not ws.compile_button.isEnabled()
    elif name == "finished-verified":
        assert _badge(ws.run_badge) == "VERIFIED"
        assert statuses == {"solver_converged": "PASSED", "constraints_satisfied": "PASSED",
                            "independently_verified": "PASSED", "optimality_certified": "NOT ASSESSED"}
        assert "CONFIRMED" in ws.digest_label.text() and "unchanged" in ws.violations_label.text()
    elif name == "finished-unverified":
        assert _badge(ws.run_badge) == "UNVERIFIED BEST" and ws.chart.view.best_label == "iteration-0001"
        assert not ws.chart.view.verified and statuses["independently_verified"] == "NOT ASSESSED"
    elif name == "policy-violation":
        assert "problem_edit" in ws.violations_label.text() and "never applied" in ws.violations_label.text()
        assert ws.iteration_table.item(0, 1).text() == "rejected"
    elif name == "cancelled":
        assert _badge(ws.run_badge) == "CANCELLED" and ws.replay_button.isEnabled()
    elif name == "replay":
        assert _badge(ws.replay_badge) == "EQUAL" and ws.step_slider.isEnabled()
        assert ws.iteration_table.rowCount() == 1 and "iteration 1 of 2" in ws.step_label.text()


def test_a_cancelled_run_replays_equal_from_the_ui(workspace):
    app, ws = workspace
    CAPTURE["cancelled"](app, ws)
    ws.replay_button.click()
    CAPTURE["pump"](app, lambda: not ws.busy)
    assert ws.session.replay_result.equal and _badge(ws.replay_badge) == "EQUAL"


def test_the_table_editor_round_trips_through_the_validator(workspace):
    app, ws = workspace
    CAPTURE["_compile"](app, ws, "halfplane")
    table = ws.tables["constraints"]
    table.item(0, 3).setText("2")                                # rhs, typed in the cell
    CAPTURE["pump"](app, lambda: ws.session.draft.doc["constraints"][0]["rhs"] == 2.0)
    assert "changed_locked_constraint" in ws.protected_label.text()
    combo = ws.tables["objectives"].cellWidget(0, 2)
    combo.setCurrentText("maximize")
    CAPTURE["pump"](app, lambda: ws.session.draft.doc["objectives"][0]["sense"] == "maximize")
    assert ws.tables["objectives"].cellWidget(0, 2) is combo     # reused, never rebuilt under the user
    ws.confirm_button.click()
    assert ws.session.phase.value == "confirmed" and not ws.tables["constraints"].editTriggers()


def test_export_and_reopen_with_the_digest_kept_out_of_band(workspace, tmp_path):
    app, ws = workspace
    CAPTURE["finished_verified"](app, ws)
    doc = ws.export_report(tmp_path / "report.json", tmp_path / "report.html")
    assert doc["run_status"] == "verified" and json.loads((tmp_path / "report.json").read_text()) == doc
    assert "Independently verified" in (tmp_path / "report.html").read_text()
    exp = ws.session.result.experiment_dir
    assert ws._digests()[str(exp.resolve())] == ws.session.result.manifest_digest
    ws.reset_button.click()
    CAPTURE["pump"](app, lambda: ws.session.phase.value == "idle")
    ws.open_run(exp)
    CAPTURE["pump"](app, lambda: not ws.busy)
    assert ws.session.replay_result.digest_checked and ws.session.replay_result.equal


def test_without_the_agent_extras_the_workspace_shows_an_install_hint(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.optimize import backend
    from orbitopt.viz.optimize.workspace import OptimizationWorkspace

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(backend, "missing_extras", lambda: ("scipy",))
    ws = OptimizationWorkspace()
    assert ws.session is None and "orbitopt[agent]" in ws.hint_label.text() and "scipy" in ws.hint_label.text()
    assert _has_ink(ws)
