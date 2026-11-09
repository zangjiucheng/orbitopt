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


# Upper bound for one mission compute, not a performance target: it only
# turns a deadlock into a failure instead of a hang. A cold Artemis II
# compute takes ~19 s on an M-series Mac and longer on a 2-vCPU CI runner
# (it timed out at 30 s on GitHub Actions once the verification integrator
# became adaptive RKDP8(7)).
LOAD_TIMEOUT_S = 180.0


def _pump_until(app, predicate, timeout=LOAD_TIMEOUT_S):
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

    sidebar = window.sidebar
    state = window.state
    assert sidebar.count() == 4  # Solar System, Artemis II, GOES GTO->GEO, Mars
    assert state.scene is not None
    assert state.scene["id"] == "solar-system"
    assert not window.timeline_bar.isVisible()

    sidebar.select_row(1)
    # The switch must not block: select_row() returns immediately and the
    # mission list is disabled (not destroyed/frozen) while the background
    # thread computes the new scene -- this is exactly the "UI gets stuck
    # when switching missions" behavior that was reported and fixed.
    assert window._loading
    assert not sidebar.mission_list.isEnabled()

    _pump_until(app, lambda: not window._loading)

    assert state.scene["id"] == "artemis2-mission"
    assert window.timeline_bar.isVisible()
    assert sidebar.mission_list.isEnabled()

    # Scrubbing the slider moves the shared time, and the inspector follows.
    timeline = state.timeline
    perilune_day = next(e["time"] for e in timeline["events"] if "approach" in e["label"].lower())
    frac = (perilune_day - timeline["min"]) / (timeline["max"] - timeline["min"])
    slider = window.timeline_bar.time_slider
    slider.setValue(int(frac * slider.maximum()))
    for _ in range(5):
        app.processEvents()

    assert abs(state.time - perilune_day) < 0.05
    assert "km" in window.inspector.distance_text("spacecraft")
    assert window.timeline_bar.met_label.text().startswith(f"T+{state.time:.2f}")

    # Moving time through the state (not the slider) moves the slider too.
    state.set_time(timeline["min"])
    app.processEvents()
    assert slider.value() == 0

    # Clicking a maneuver row seeks to it. Artemis II's events carry no burn
    # vectors, so this is the seek-only path (no auto-frame/selection).
    event = timeline["events"][-1]
    assert "vector" not in event
    window.inspector.event_clicked.emit(event)
    app.processEvents()
    assert abs(state.time - event["time"]) < 1e-9
    assert window.timeline_bar.event_label.text() == f"▸ {event['label']}"

    window.inspector.body_clicked.emit("spacecraft", False)
    app.processEvents()
    assert state.selected_body_id == "spacecraft"

    # Measuring: ⌘/Ctrl-click a second body shows the measure card.
    other = next(b["id"] for b in state.scene["bodies"] if b["id"] != "spacecraft")
    window.inspector.body_clicked.emit(other, True)
    app.processEvents()
    assert state.measure_body_id == other
    assert not window.inspector.measure_card.isHidden()

    # Re-centering on the Moon: the button drives state, the renderer follows,
    # and the Moon is drawn at the origin. Back to Earth restores the scene frame.
    strip = window.view_strip
    assert strip.frame_button(None).text() == "Earth"
    strip.frame_button("moon").click()
    app.processEvents()
    assert state.frame_body_id == "moon" == window.renderer.frame_body_id
    assert sum(abs(c) for c in window.renderer.body_display_position("moon", state.time)) < 1e-6
    strip.frame_button(None).click()
    app.processEvents()
    assert state.frame_body_id is None and window.renderer.frame_body_id is None

    # Playback ticks advance time through the same state.
    state.set_time(timeline["min"])
    before = state.time
    window._set_playing(True)
    _pump_until(app, lambda: state.time > before, timeout=5.0)
    window._set_playing(False)

    # Switching back to a cached scene resets selection and hides the bar.
    sidebar.select_row(0)
    _pump_until(app, lambda: not window._loading)
    assert state.scene["id"] == "solar-system"
    assert state.selected_body_id is None and state.measure_body_id is None
    assert window.inspector.measure_card.isHidden()
    # The solar-system scene draws planets far larger than life and says so.
    assert "not to scale" in window.view_strip.scale_label.text()
    assert window.view_strip.frame_button("earth") is None  # static scene: no re-centering

    window.plotter.close()  # release the VTK/OpenGL context before Qt tears down the widget
    window.close()
