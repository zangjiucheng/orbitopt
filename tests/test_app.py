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
def test_mission_control_loads_and_switches_missions(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow()
    window.show()
    _pump_until(app, lambda: not window._loading)

    # try/finally: a failed assertion must still close the window and its
    # compute worker, or every later Qt / worker test fails on the leftovers.
    try:
        sidebar = window.sidebar
        state = window.state
        assert sidebar.count() == 5  # Solar System, Artemis II, GOES GTO->GEO, Mars, Starship Flight 15 (Flight Lab is a mode, not a mission)
        assert state.scene is not None
        assert state.scene["id"] == "solar-system"
        assert not window.timeline_bar.isVisible()
        assert window.verification_badge.isHidden()  # an ephemeris snapshot: nothing to verify
        assert window.chart_panel.isHidden()  # ...and nothing to chart

        # Renders are coalesced: five requests in one event-loop turn are one
        # draw (each draw waits for vsync; a tick used to draw 3-4 times).
        real_render, drawn = window.plotter._render, []
        window.plotter._render = lambda *a, **k: (drawn.append(1), real_render(*a, **k))
        try:
            for _ in range(5):
                window.plotter.render()
            assert drawn == []  # queued, not drawn yet
            app.processEvents()
            assert 1 <= len(drawn) <= 2  # 1; a timer firing in the same turn may add one
        finally:
            window.plotter._render = real_render

        # Hot reload: an edited source file shows the reload button; ⌘R
        # restarts the compute worker and recomputes the current mission.
        from orbitopt.viz.source_watch import SourceWatch

        (tmp_path / "mission.py").write_text("x = 1\n")
        window._source_watch = SourceWatch(tmp_path)
        window._check_source()
        assert not window.reload_button.isVisible()
        (tmp_path / "mission.py").write_text("x = 2  # edited\n")
        window._check_source()
        assert window.reload_button.isVisible() and "mission.py" in window.reload_button.toolTip()
        assert window.reload_action.shortcut() == QKeySequence("Ctrl+R")
        old_workers = window._compute_pool.worker_pids()
        window.reload_action.trigger()
        assert window._loading and not window.reload_button.isVisible()
        _pump_until(app, lambda: not window._loading)
        assert state.scene["id"] == "solar-system"
        assert not set(old_workers) & set(window._compute_pool.worker_pids())

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

        # The targeting report rides along: the badge states its tri-state result
        # and the Verification tab shows what was checked.
        result = state.scene["verification"]["result"]
        assert window.verification_badge._label.text() == {
            "passed": "VERIFIED", "failed": "VERIFY FAILED", "not_checked": "NOT CHECKED"}[result]
        window._show_verification_tab()
        assert window.right_tabs.currentWidget() is window.verification_panel
        assert "Step convergence" in window.verification_panel.text()
        assert "Verification report" not in window.scene_subtitle_label.text()
        window.right_tabs.setCurrentWidget(window.inspector)

        # Phases: the lunar flyby is the Moon-SOI crossing and contains perilune.
        phases = window.timeline_bar.track.phases
        flyby = next(p for p in phases if p["label"] == "Lunar flyby")
        closest = next(e["time"] for e in state.events if "approach" in e["label"].lower())
        assert flyby["start"] < closest < flyby["end"]

        # KSP map-view focus: Tab goes to the spacecraft first and tracks it,
        # framed with Earth behind but without turning face lock on (it is
        # only on when picked); Shift+Tab wraps around the other way.
        # Checked through the menu actions, not synthetic key presses: Qt only
        # delivers shortcuts to the active window, which this one need not be
        # in a full test run (a key press here passed alone and failed there).
        from PySide6.QtGui import QAction, QKeySequence

        actions = {a.text(): a for a in window.findChildren(QAction)}
        focus_next, focus_prev = actions["Focus next (Tab)"], actions["Focus previous (⇧Tab)"]
        assert focus_next.shortcut() == QKeySequence(Qt.Key_Tab)
        focus_next.trigger()
        app.processEvents()
        assert state.selected_body_id == "spacecraft" and window._tracking
        assert window._lock_reference_body_id is None and window.lock_combo.currentData() is None
        sc = window.renderer.body_display_position("spacecraft", state.time)
        earth = window.renderer.body_display_position("earth", state.time)
        cam = window.plotter.camera
        import numpy as _np
        look = _np.asarray(cam.focal_point) - _np.asarray(cam.position)
        assert _np.dot(look, _np.asarray(earth) - _np.asarray(sc)) > 0  # Earth is in front, behind the ship
        focus_prev.trigger()
        app.processEvents()
        assert state.selected_body_id == "moon" and window._lock_reference_body_id is None
        window._set_tracking(False)
        window.state.select_body(None)

        # Flight-data label next to the spacecraft (L toggles it). From a camera
        # beyond the spacecraft, looking back at Earth -- the label hides when
        # Earth is in the way (checked below), so the view has to be known.
        import numpy as np

        sc = np.asarray(window.renderer.body_display_position("spacecraft", state.time))
        earth = np.asarray(window.renderer.body_display_position("earth", state.time))
        window.plotter.camera.position = tuple(sc + (sc - earth) * 2.0)
        window.plotter.camera.focal_point = tuple(earth)
        window._refresh_view()
        label = window.renderer.flight_label_text("spacecraft")
        assert label is not None and "\nH " in label and "\nV " in label
        assert "\nΔv " not in label  # Artemis II's events carry no burn vectors: no Δv line
        window.labels_action.setChecked(False)
        assert window.renderer.flight_label_text("spacecraft") is None
        window.labels_action.setChecked(True)
        # Help links open the guide, website and README online.
        guide = window.help_url("guide")
        assert guide.toString() == "https://zangjiucheng.github.io/orbitopt/guide.html"  # the docs site's page
        assert window.help_url("website").toString() == "https://zangjiucheng.github.io/orbitopt/"
        assert window.help_url("readme").toString().endswith("/blob/main/README.md")
        # Seen from the far side of Earth the spacecraft is hidden, and so is its label.
        saved = window.plotter.camera_position
        window.plotter.camera.position = tuple(earth - (sc - earth) * 3.0)
        window.plotter.camera.focal_point = tuple(earth)
        window._refresh_view()
        assert window.renderer.flight_label_text("spacecraft") is None
        window.plotter.camera_position = saved
        window._refresh_view()
        assert window.renderer.flight_label_text("spacecraft") is not None

        # Linked charts: Artemis II has no launch-window screen, only distances.
        charts = window.chart_panel
        assert not charts.isHidden()
        assert [charts.tabText(i) for i in range(charts.count())] == ["Distance"]

        # Scrubbing the slider moves the shared time, and the inspector follows.
        timeline = state.timeline
        perilune_day = next(e["time"] for e in timeline["events"] if "approach" in e["label"].lower())
        frac = (perilune_day - timeline["min"]) / (timeline["max"] - timeline["min"])
        track = window.timeline_bar.track
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        x = track.track_left + frac * track.track_width
        y = 0.5 * sum(track._rows()["ruler"])
        QTest.mouseClick(track, Qt.LeftButton, Qt.NoModifier, QPoint(round(x), round(y)))
        for _ in range(5):
            app.processEvents()

        assert abs(state.time - perilune_day) < 0.05
        assert "km" in window.inspector.distance_text("spacecraft")
        assert window.timeline_bar.met_edit.text() == window.timeline_bar.met_text()

        # Moving time through the state (not the track) moves the playhead too.
        state.set_time(timeline["min"])
        app.processEvents()
        assert window.timeline_bar.met_edit.text() == "T+0:00:00"

        # Clicking a maneuver row seeks to it. Artemis II's events carry no burn
        # vectors, so this is the seek-only path (no auto-frame/selection).
        event = timeline["events"][-1]
        assert "vector" not in event
        window.inspector.event_clicked.emit(event)
        app.processEvents()
        assert abs(state.time - event["time"]) < 1e-9
        assert window.timeline_bar.event_label.text() == f"▸ {event['label']}"

        # Playback runs backwards too (Shift+Space), and the same key pauses.
        state.set_time(event["time"])
        window._toggle_play(-1)
        assert window._playing and window._play_direction == -1
        window._last_tick_ms = time.monotonic() * 1000.0 - 50.0
        window._on_tick()
        assert state.time < event["time"]
        window._toggle_play(+1)  # the other direction turns around
        assert window._playing and window._play_direction == 1
        window._toggle_play(+1)
        assert not window._playing
        window.play_action.trigger()  # the menu passes checked=False: still forward
        assert window._playing and window._play_direction == 1

        # Dragging on the track pauses (playback carried the playhead away
        # from the pointer), and it stays paused after the drag.
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        track = window.timeline_bar.track
        top, bottom = track._rows()["ruler"]
        at = QPoint(int(track.x_of(state.time) + 40), int((top + bottom) / 2))
        QTest.mousePress(track, Qt.LeftButton, Qt.NoModifier, at)
        assert not window._playing
        QTest.mouseMove(track, at + QPoint(30, 0))
        QTest.mouseRelease(track, Qt.LeftButton, Qt.NoModifier, at + QPoint(30, 0))
        app.processEvents()
        assert not window._playing

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
    finally:
        window.plotter.close()  # release the VTK/OpenGL context before Qt tears down the widget
        window.close()
