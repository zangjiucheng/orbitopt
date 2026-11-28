"""Render every state of Mission Control's Optimize workspace offscreen and save one PNG per state.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/capture_optimization_studio.py [--out docs/assets]

Each state starts from a fresh workspace on the offline mock provider and is
driven through the real UI: the job threads, the presenter and the widgets
are the ones Mission Control uses. No network and no model are involved:
the bundled examples carry scripted compiler replies, and the running and
cancelled states use a scripted planner that holds its second reply until
the Cancel button has been pressed, so the screenshot shows a run in flight
on any machine.

``tests/test_e14_09_studio_widgets.py`` drives the same :data:`STATES`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
SIZE = (1600, 940)


def pump(app, cond, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError(f"timed out waiting for {cond}")
        app.processEvents()
        time.sleep(0.005)


def make_workspace(root: Path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.optimize.session import StudioSession
    from orbitopt.viz.optimize.workspace import OptimizationWorkspace
    from orbitopt.viz.theme import load_bundled_fonts, stylesheet

    app = QApplication.instance() or QApplication([])
    load_bundled_fonts()
    root.mkdir(parents=True, exist_ok=True)
    session = StudioSession(root=root / "runs", environ={}, scratch=root / "scratch")
    settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
    ws = OptimizationWorkspace(session, settings=settings)
    ws.setStyleSheet(stylesheet() + ws.styleSheet())
    ws.resize(*SIZE)
    ws.show()
    app.processEvents()
    return app, ws


def _choose(app, ws, key: str) -> None:
    ws.example_combo.setCurrentIndex(ws.example_combo.findData(key))
    pump(app, lambda: ws.session.example is not None and ws.session.example.key == key)


def _compile(app, ws, key: str) -> None:
    _choose(app, ws, key)
    ws.compile_button.click()
    pump(app, lambda: not ws.busy)


def _run(app, ws) -> None:
    ws.confirm_button.click()
    ws.run_button.click()
    pump(app, lambda: not ws.busy)


def idle(app, ws) -> None:
    _choose(app, ws, "halfplane")


def settings(app, ws) -> None:
    ws.pages.setCurrentIndex(1)
    ws.resize(960, 430)
    ws.settings_page.provider_combo.setCurrentIndex(ws.settings_page.provider_combo.findData("local"))


def clarification(app, ws) -> None:
    _compile(app, ws, "clarify")
    ws.answer_edits[0].setText("c = 3")


def editing_error(app, ws) -> None:
    _compile(app, ws, "halfplane")
    ws.editor_tabs.setCurrentIndex(0)
    ws.session.set_cell("variables", 0, "upper", "ten")


def finished_verified(app, ws) -> None:
    _compile(app, ws, "halfplane")
    _run(app, ws)


def finished_unverified(app, ws) -> None:
    _compile(app, ws, "fc001")
    _run(app, ws)


def policy_violation(app, ws) -> None:
    _compile(app, ws, "halfplane")
    ws.settings_page.planner_combo.setCurrentIndex(1)
    ws.settings_page.apply_button.click()
    pump(app, lambda: ws.session.planner == "llm")
    ws.session.set_planner_replies([
        {"action": "solve", "solver_id": "scipy.slsqp", "constraints": [], "rationale": "drop the constraint"},
        {"action": "solve", "solver_id": "scipy.slsqp", "options": {"constraint_tol": 0.5}},
        {"action": "solve", "solver_id": "scipy.slsqp"}, {"action": "stop"}])
    _run(app, ws)


class HeldPlanner:
    """A scripted LLM planner whose second reply waits until ``release`` is set (a slow model, on cue)."""

    def __init__(self):
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, request) -> str:
        self.calls += 1
        if self.calls > 1:
            self.release.wait(30)
        state = json.loads(request.messages[-1].content)
        return json.dumps({"action": "solve", "solver_id": state["solvers"][0]["solver_id"]})


def _start_held_run(app, ws) -> HeldPlanner:
    _compile(app, ws, "halfplane")
    ws.settings_page.planner_combo.setCurrentIndex(1)
    ws.settings_page.apply_button.click()
    pump(app, lambda: ws.session.planner == "llm")
    held = HeldPlanner()
    ws.session.set_planner_replies(held)
    ws.confirm_button.click()
    ws.run_button.click()
    pump(app, lambda: len(ws.session.live) >= 1 and held.calls >= 2)
    return held


def running(app, ws) -> None:
    held = _start_held_run(app, ws)
    ws._release = held.release       # the caller cancels and releases after the screenshot


def cancelled(app, ws) -> None:
    held = _start_held_run(app, ws)
    ws.cancel_button.click()
    held.release.set()
    pump(app, lambda: not ws.busy)


def replay(app, ws) -> None:
    _compile(app, ws, "halfplane")
    _run(app, ws)
    ws.replay_button.click()
    pump(app, lambda: not ws.busy)
    ws.step_slider.setValue(1)
    pump(app, lambda: ws.session.step == 1)


STATES = {
    "idle": (idle, "idle"),
    "settings": (settings, "idle"),
    "clarification": (clarification, "needs_clarification"),
    "editing-error": (editing_error, "editing"),
    "running": (running, "running"),
    "finished-verified": (finished_verified, "finished"),
    "finished-unverified": (finished_unverified, "finished"),
    "policy-violation": (policy_violation, "finished"),
    "cancelled": (cancelled, "cancelled"),
    "replay": (replay, "replay"),
}
"""name -> (driver, the session phase it ends in)"""


def finish(app, ws) -> None:
    """Let a held run go (the running state) and wait for every thread, before the widget is dropped."""
    release = getattr(ws, "_release", None)
    if release is not None:
        ws.session.cancel()
        release.set()
    pump(app, lambda: not ws.busy)
    ws.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    parser.add_argument("--states", nargs="*", default=list(STATES))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orbitopt-studio-shots-") as tmp:
        for name in args.states:
            driver, _phase = STATES[name]
            app, ws = make_workspace(Path(tmp) / name)
            driver(app, ws)
            app.processEvents()
            path = args.out / f"optimization-studio-{name}.png"
            ws.grab().save(str(path))
            print(f"{name:20s} phase={ws.session.phase.value:20s} -> {path}")
            finish(app, ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
