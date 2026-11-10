"""MissionState and its timeline helpers -- no window, no display needed."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from orbitopt.viz.state import (  # noqa: E402
    MissionState,
    active_event,
    adjacent_event,
    eased_speed,
    fraction_to_time,
    nearest_maneuver,
    time_to_fraction,
)

EVENTS = [
    {"label": "TLI", "time": 1.0, "vector": {"position": [1, 0, 0], "deltaV": [0, 3, 0]}},
    {"label": "Closest approach", "time": 5.0},
    {"label": "TCM", "time": 7.0, "vector": {"position": [0, 1, 0], "deltaV": [0, 0, 1]}},
]
TIMELINE = {"min": 0.0, "max": 10.0, "unitLabel": "days", "events": EVENTS}
SCENE = {
    "id": "test",
    "title": "Test",
    "distanceUnit": "km",
    "bodies": [
        {"id": "earth", "name": "Earth", "kind": "planet"},
        {"id": "sc", "name": "Probe", "kind": "spacecraft"},
        {"id": "moon", "name": "Moon", "kind": "moon"},
    ],
    "timeline": TIMELINE,
}


def test_fraction_round_trip_and_clamp():
    assert fraction_to_time(TIMELINE, time_to_fraction(TIMELINE, 2.5)) == pytest.approx(2.5)
    assert time_to_fraction(TIMELINE, -5.0) == 0.0
    assert time_to_fraction(TIMELINE, 50.0) == 1.0
    # A zero-length timeline must not divide by zero.
    assert time_to_fraction({"min": 3.0, "max": 3.0}, 3.0) == 0.0


def test_active_event_is_last_at_or_before():
    assert active_event(EVENTS, 0.5) is None
    assert active_event(EVENTS, 1.0)["label"] == "TLI"
    assert active_event(EVENTS, 6.0)["label"] == "Closest approach"
    assert active_event(EVENTS, 99.0)["label"] == "TCM"


def test_adjacent_event_is_strict():
    assert adjacent_event(EVENTS, 1.0, +1)["label"] == "Closest approach"
    assert adjacent_event(EVENTS, 1.0, -1) is None
    assert adjacent_event(EVENTS, 5.0, -1)["label"] == "TLI"
    assert adjacent_event(EVENTS, 7.0, +1) is None


def test_nearest_maneuver_ignores_events_without_vectors():
    event, dt = nearest_maneuver(EVENTS, 5.1)
    assert event["label"] == "TCM"  # 1.9 away; the closer flyby has no burn
    assert dt == pytest.approx(1.9)
    assert nearest_maneuver([{"time": 1.0}], 1.0) == (None, None)


def test_eased_speed_only_slows_down_and_ramps():
    kwargs = dict(span=10.0, window_fraction=0.1, target_speed=0.02)  # 1-day window
    assert eased_speed(100.0, EVENTS, 3.0, **kwargs) == 100.0  # far from events
    assert eased_speed(100.0, EVENTS, 5.0, **kwargs) == pytest.approx(0.02)  # on an event
    mid = eased_speed(100.0, EVENTS, 5.5, **kwargs)
    assert 0.02 < mid < 100.0
    # Never speeds up a deliberately slow rate.
    assert eased_speed(0.001, EVENTS, 5.0, **kwargs) == pytest.approx(0.001)
    assert eased_speed(5.0, [], 5.0, **kwargs) == 5.0


def _recorder(state):
    log = []
    state.scene_changed.connect(lambda s: log.append(("scene", s["id"] if s else None)))
    state.time_changed.connect(lambda t: log.append(("time", t)))
    state.selection_changed.connect(lambda: log.append(("selection",)))
    return log


def test_set_scene_resets_time_and_selection():
    state = MissionState()
    state.set_scene(SCENE)
    state.set_time(4.0)
    state.select_body("sc")
    state.toggle_measure_partner("earth")
    log = _recorder(state)

    state.set_scene(SCENE)

    assert log == [("scene", "test")]
    assert state.time == 0.0
    assert state.selected_body_id is None and state.measure_body_id is None
    assert state.primary_spacecraft_id() == "sc"


def test_set_time_clamps_and_ignores_static_scenes():
    state = MissionState()
    state.set_scene(SCENE)
    log = _recorder(state)
    state.set_time(42.0)
    state.set_fraction(0.25)
    assert log == [("time", 10.0), ("time", 2.5)]
    assert state.active_event()["label"] == "TLI"

    static = dict(SCENE)
    static.pop("timeline")
    state.set_scene(static)
    log.clear()
    state.set_time(3.0)
    assert log == [] and state.time == 0.0


def test_selection_and_measure_partner_rules():
    state = MissionState()
    state.set_scene(SCENE)
    log = _recorder(state)

    state.select_body("nope")  # unknown id: ignored, no signal
    assert state.selected_body_id is None and log == []

    state.select_body("sc")
    assert state.toggle_measure_partner("sc") is False  # not against itself
    assert state.toggle_measure_partner("earth") is True
    assert state.measure_body_id == "earth"
    assert state.toggle_measure_partner("earth") is True  # second click unpicks
    assert state.measure_body_id is None

    state.toggle_measure_partner("moon")
    state.select_body("earth")  # a plain select drops the partner
    assert state.selected_body_id == "earth" and state.measure_body_id is None
    assert log.count(("selection",)) == 5


def test_maneuver_windows_follow_event_spacing():
    from orbitopt.viz.state import maneuver_windows

    events = [
        {"label": "Deorbit", "time": 10.0, "vector": {}},
        {"label": "Peak g", "time": 10.30},
        {"label": "Landing burn", "time": 10.34, "vector": {}},
        {"label": "Catch", "time": 10.345},
        {"label": "SOI exit", "time": 3.0},
        {"label": "TCM", "time": 3.0, "vector": {}},  # same time as SOI exit: that one doesn't count
    ]
    w = maneuver_windows(events, span=20.0)
    assert w[id(events[2])] == pytest.approx(0.0025)    # half the 0.005 gap to the catch
    assert w[id(events[0])] == pytest.approx(0.15)      # half the gap to peak g
    assert w[id(events[5])] == pytest.approx(1.0)       # capped at 5% of the span (gap to deorbit is 7)
    assert id(events[1]) not in w                       # not a burn
