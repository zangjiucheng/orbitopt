"""TimelineBar readouts follow the loaded scene."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _scene(ref_et=None):
    timeline = {"min": 0.0, "max": 4.0, "unitLabel": "days", "events": []}
    if ref_et is not None:
        timeline["referenceEpochEt"] = ref_et
    return {"id": "t", "title": "T", "distanceUnit": "km", "bodies": [], "timeline": timeline}


def test_date_readout_is_cleared_for_a_timeline_without_an_epoch():
    """GOES has no calendar epoch: switching to it from a mission that has one
    used to leave the previous mission's date on screen."""
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    QApplication.instance() or QApplication([])
    state = MissionState()
    bar = TimelineBar(state)
    state.set_scene(_scene(ref_et=838814469.0))
    assert bar.date_label.text().endswith("UTC")
    state.set_scene(_scene())
    assert bar.date_label.text() == ""
    state.set_time(2.0)
    assert bar.date_label.text() == "" and bar.met_edit.text() == "T+2d 00:00:00"


def test_warp_is_ksp_style_real_time_multiplier():
    from orbitopt.viz.timeline_bar import (
        WARP_LEVELS,
        default_warp,
        format_rate,
        format_warp,
        slider_from_warp,
        step_warp,
        timeline_unit_seconds,
        warp_from_slider,
    )

    # 1x is real time; the range reaches from slow motion to interplanetary cruise.
    assert WARP_LEVELS[0] == 0.1 and 1.0 in WARP_LEVELS and WARP_LEVELS[-1] == 1.0e7
    assert format_warp(1.0) == "1×" and format_warp(1000.0) == "1,000×"
    assert format_warp(100_000.0) == "100k×" and format_warp(1.0e7) == "10M×"
    assert format_rate(1.0) == "1 s/s" and format_rate(1000.0) == "16.7 min/s" and format_rate(86400.0) == "1 d/s"
    # KSP keys step between levels, also from an in-between value.
    assert step_warp(1.0, +1) == 5.0 and step_warp(1.0, -1) == 0.1
    assert step_warp(1500.0, +1) == 10_000.0 and step_warp(1500.0, -1) == 1_000.0
    assert step_warp(1.0e7, +1) == 1.0e7  # clamps at the top
    for w in (0.1, 3.7, 1234.0, 1.0e7):
        assert warp_from_slider(slider_from_warp(w)) == pytest.approx(w, rel=0.01)
    # Default: the level that plays the whole timeline in about a minute.
    assert default_warp(23.5 * 3600) == 1_000.0      # Starship Flight 15
    assert default_warp(7.5 * 86400) == 10_000.0     # Artemis II
    assert default_warp(303 * 86400) == 1_000_000.0  # Mars
    assert timeline_unit_seconds("days") == 86400.0 and timeline_unit_seconds("hours") == 3600.0
    assert timeline_unit_seconds(None) == 86400.0


def test_timeline_bar_picks_the_scene_default_warp_and_steps():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    QApplication.instance() or QApplication([])
    state = MissionState()
    bar = TimelineBar(state)
    state.set_scene(_scene())  # 4 days
    assert bar.warp == 10_000.0
    bar.step_warp(+1)
    assert bar.warp == 100_000.0 and bar.speed_value_label.text().startswith("100k×")
    bar.set_warp(1.0)
    assert bar.speed_value_label.text() == "1× · 1 s/s"


def test_typing_a_time_jumps_there_and_the_track_zooms_and_follows():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    app = QApplication.instance() or QApplication([])
    state = MissionState()
    bar = TimelineBar(state)
    bar.resize(1000, 140)
    bar.show()
    scene = _scene(ref_et=838814469.0)
    scene["timeline"]["events"] = [{"time": 1.0, "label": "Burn", "vector": {"deltaV": [0.0, 0.01, 0.0]}},
                                   {"time": 1.0 + 30.0 / 86400.0, "label": "Note"}]
    state.set_scene(scene)
    app.processEvents()
    track = bar.track
    assert track.window.is_fit and bar.zoom_label.text() == "4 d"

    bar.met_edit.setText("T+1d 06:00:00")
    bar.met_edit.returnPressed.emit()
    assert state.time == pytest.approx(1.25)
    bar.met_edit.setText("-6h")  # relative to now
    bar.met_edit.returnPressed.emit()
    assert state.time == pytest.approx(1.0)
    bar.met_edit.setText("not a time")
    bar.met_edit.returnPressed.emit()
    assert state.time == pytest.approx(1.0) and bar.met_edit.property("invalid") is True

    # Zoom in on the playhead: the ruler, step and readout get finer.
    step_fit = bar.step_size()
    track.zoom(1000)
    assert track.window.span == pytest.approx(4.0 / 1000)
    assert track.window.start <= state.time <= track.window.end
    assert bar.step_size() == pytest.approx(step_fit / 1000)
    assert bar.zoom_label.text() == "5.76 min"
    # Scrubbing at this zoom is sub-second.
    x = track.x_of(state.time)
    assert track.time_at(x + 1) - track.time_at(x) < 1.0 / 86400.0
    # The event lane snaps to the burn; double-clicking frames it.
    assert track.events_near(track.x_of(1.0))[0]["label"] == "Burn"
    track.zoom_to_event(scene["timeline"]["events"][0])
    assert track.window.start < 1.0 < track.window.end and track.window.span < 1.0 / 24

    # Playback running off the right edge pages the view along.
    state.set_time(track.window.end + 0.1 * track.window.span)
    assert track.window.start <= state.time <= track.window.end
    # A far jump centers on it.
    state.set_time(3.5)
    assert track.window.start < 3.5 < track.window.end
    track.fit()
    assert track.window.is_fit and not bar.fit_btn.isEnabled()
    assert not bar.grab().isNull()
    bar.close()
