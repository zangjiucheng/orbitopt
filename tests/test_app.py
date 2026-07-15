"""Smoke tests for the Mission Control Qt app. These need a real windowing
system -- VTK's native Win32 OpenGL context fails to get a valid pixel
format under Qt's "offscreen" platform plugin (confirmed empirically while
building this), so unlike the rest of this package's off_screen=True tests,
these actually open a (real, if briefly-shown) window. Fine for local runs
on a machine with a display; skip in a headless CI environment.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")


@pytest.mark.slow
def test_mission_control_loads_and_switches_missions():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow()
    window.show()
    for _ in range(5):
        app.processEvents()

    assert window.mission_list.count() == 2
    assert window._current_scene is not None
    assert window._current_scene["id"] == "solar-system"
    assert not window.timeline_bar.isVisible()

    window.mission_list.setCurrentRow(1)
    for _ in range(10):
        app.processEvents()

    assert window._current_scene["id"] == "artemis2-mission"
    assert window.timeline_bar.isVisible()

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
