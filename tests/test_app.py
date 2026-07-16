"""Smoke tests for the Mission Control Qt app. These need a real windowing
system -- VTK's native Win32 OpenGL context fails to get a valid pixel
format under Qt's "offscreen" platform plugin (confirmed empirically while
building this), so unlike the rest of this package's off_screen=True tests,
these actually open a (real, if briefly-shown) window. Fine for local runs
on a machine with a display; skip in a headless CI environment.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")


def _pump_until(app, predicate, timeout=30.0):
    """Drive the real Qt event loop with processEvents() until `predicate()`
    is true or `timeout` elapses. Mission loading now runs on a background
    QThread (see app.py's SceneLoader), so tests must wait for it the same
    way the real event loop does -- a handful of fixed processEvents() calls
    raced the load and either passed by luck or silently asserted on stale
    state. A bounded wall-clock loop is safe here because the fix under test
    (see SceneLoader in app.py) closed the actual deadlock this app used to
    hit; if that regresses, this loop times out and fails loudly rather than
    hanging forever.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {predicate}")
        app.processEvents()


@pytest.mark.slow
def test_mission_control_loads_and_switches_missions():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow()
    window.show()
    _pump_until(app, lambda: not window._loading)

    assert window.mission_list.count() == 3  # Solar System, Artemis II, GOES GTO->GEO
    assert window._current_scene is not None
    assert window._current_scene["id"] == "solar-system"
    assert not window.timeline_bar.isVisible()

    window.mission_list.setCurrentRow(1)
    # The switch must not block: setCurrentRow() returns immediately and the
    # mission list is disabled (not destroyed/frozen) while the background
    # thread computes the new scene -- this is exactly the "UI gets stuck
    # when switching missions" behavior that was reported and fixed.
    assert window._loading
    assert not window.mission_list.isEnabled()

    _pump_until(app, lambda: not window._loading)

    assert window._current_scene["id"] == "artemis2-mission"
    assert window.timeline_bar.isVisible()
    assert window.mission_list.isEnabled()

    timeline = window._current_scene["timeline"]
    perilune_day = next(e["time"] for e in timeline["events"] if "approach" in e["label"].lower())
    frac = (perilune_day - timeline["min"]) / (timeline["max"] - timeline["min"])
    window.time_slider.setValue(int(frac * window.time_slider.maximum()))
    for _ in range(5):
        app.processEvents()

    assert abs(window._current_time - perilune_day) < 0.05
    orion_distance_text = window._body_cards["spacecraft"]["dist_val"].text()
    assert "km" in orion_distance_text

    window.plotter.close()  # release the VTK/OpenGL context before Qt tears down the widget
    window.close()
