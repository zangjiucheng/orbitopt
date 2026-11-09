"""Timeline phases: defined by each exporter from its own event times / the
trajectory, validated, and drawn under the scrubber."""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.scene_format import validate_scene
from orbitopt.viz.scene import crossing_times, phase_spans


def test_phase_spans_drop_zero_length_and_keep_notes():
    phases = phase_spans([("A", 0.0, 0.0), ("B", 0.0, 1.5, "note"), ("C", 1.5, 3.0)])
    assert phases == [{"label": "B", "start": 0.0, "end": 1.5, "note": "note"},
                      {"label": "C", "start": 1.5, "end": 3.0}]


def test_crossing_times_interpolate_between_samples():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    d = np.array([100.0, 60.0, 20.0, 60.0, 100.0])
    t_in, t_out = crossing_times(t, d, 50.0)
    assert t_in == pytest.approx(1.25) and t_out == pytest.approx(2.75)
    assert crossing_times(t, d, 10.0) is None
    assert crossing_times(t, np.full(5, 1.0), 50.0) == (0.0, 4.0)  # inside the whole time


def test_mars_phases_follow_the_event_times():
    pytest.importorskip("pykep")
    from orbitopt.viz.mars_transfer import _mars_phases

    events = [{"label": "Liftoff", "time": 0.0}, {"label": "Parking-orbit insertion", "time": 0.0},
              {"label": "TMI burn", "time": 0.06}, {"label": "Earth-SOI exit", "time": 3.14},
              {"label": "Trajectory-correction maneuver", "time": 3.14},
              {"label": "Mars-SOI entry / approach", "time": 300.4}, {"label": "Mars orbit insertion", "time": 302.4}]
    phases = _mars_phases(events, 302.97)
    assert [p["label"] for p in phases] == [
        "Parking orbit", "Earth departure", "Interplanetary cruise", "Mars approach", "Mars orbit"]  # no ascent
    assert phases[2]["start"] == 3.14 and phases[2]["end"] == 300.4
    # Contiguous: each phase starts where the previous one ended.
    assert all(a["end"] == b["start"] for a, b in zip(phases, phases[1:]))


def test_geo_phases_bracket_the_burns():
    pytest.importorskip("pykep")
    from orbitopt.viz.geo_raising import _geo_phases

    events = [{"label": "Liftoff", "time": 0.0}, {"label": "Apogee burn 1", "time": 0.4, "vector": {}},
              {"label": "Apogee burn 2", "time": 2.1, "vector": {}}, {"label": "GEO insertion", "time": 3.0}]
    assert [(p["label"], p["start"], p["end"]) for p in _geo_phases(events, 3.0)] == [
        ("Coast to first apogee", 0.0, 0.4), ("Apogee-raising burns", 0.4, 2.1), ("Final coast", 2.1, 3.0)]


def test_artemis_flyby_phase_is_the_moon_soi_crossing():
    pytest.importorskip("pykep")
    from orbitopt.viz.mission_timeline import MOON_SOI_RADIUS_KM, _artemis_phases

    days = np.linspace(0.0, 8.0, 81)
    moon = np.column_stack([np.full(81, 3.84e5), np.zeros(81), np.zeros(81)])
    # Straight past the Moon at 10,000 km closest approach on day 4.
    sc = moon + np.column_stack([(days - 4.0) * 1.0e5, np.full(81, 1.0e4), np.zeros(81)])
    events = [{"label": "TLI burn", "time": 0.0}, {"label": "Earth entry interface", "time": 7.5}]
    phases = _artemis_phases(days, sc, moon, events)
    flyby = next(p for p in phases if p["label"] == "Lunar flyby")
    half = np.sqrt(MOON_SOI_RADIUS_KM ** 2 - 1.0e4 ** 2) / 1.0e5
    assert flyby["start"] == pytest.approx(4.0 - half, abs=2e-3)
    assert flyby["end"] == pytest.approx(4.0 + half, abs=2e-3)
    assert [p["label"] for p in phases] == ["Outbound coast", "Lunar flyby", "Return coast", "Past entry interface"]


def _scene(phases):
    return {"schemaVersion": "1.0", "id": "p", "title": "P", "distanceUnit": "km",
            "bodies": [{"id": "sc", "name": "SC", "color": "#e8a23e", "kind": "spacecraft", "radiusDisplay": 3.0,
                        "trail": {"times": [0.0, 10.0], "positions": [[0, 0, 0], [1, 0, 0]]}}],
            "timeline": {"min": 0.0, "max": 10.0, "unitLabel": "days", "events": [], "phases": phases}}


def test_validator_rejects_a_phase_that_ends_before_it_starts():
    validate_scene(_scene(phase_spans([("A", 0.0, 4.0), ("B", 4.0, 10.0)])))
    with pytest.raises(ValueError, match="ends"):
        validate_scene(_scene([{"label": "A", "start": 5.0, "end": 1.0}]))


def test_phase_strip_highlights_current_phase_and_seeks_on_click():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    app = QApplication.instance() or QApplication([])
    state = MissionState()
    bar = TimelineBar(state)
    bar.resize(900, 100)
    bar.show()
    state.set_scene(_scene(phase_spans([("Outbound", 0.0, 4.0), ("Flyby", 4.0, 6.0), ("Return", 6.0, 10.0)])))
    app.processEvents()
    strip = bar.phase_strip
    assert strip.isVisible() and bar.height() == TimelineBar._BASE_HEIGHT + strip.HEIGHT + 2

    # Bands sit on the scrubber's own x scale: phase boundaries line up with
    # the slider groove's on-screen ends.
    groove = bar.time_slider._groove()
    left = strip.mapFromGlobal(bar.time_slider.mapToGlobal(groove.topLeft())).x()
    assert strip._x(0.0) == pytest.approx(left, abs=1.0)
    assert strip._x(10.0) == pytest.approx(left + groove.width(), abs=1.0)
    assert 0 < strip._x(0.0) < strip.width() and strip._x(10.0) <= strip.width()

    state.set_time(5.0)
    assert strip.current_phase()["label"] == "Flyby"
    x_return = 0.5 * (strip._x(6.0) + strip._x(10.0))
    assert strip.phase_at(x_return)["label"] == "Return"
    strip._state.set_time(strip.phase_at(x_return)["start"])
    assert state.time == 6.0
    assert not strip.grab().isNull()

    state.set_scene(_scene([]))  # no phases: strip hidden, bar back to its base height
    assert strip.isHidden() and bar.height() == TimelineBar._BASE_HEIGHT
    bar.close()
