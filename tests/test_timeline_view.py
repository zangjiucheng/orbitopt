"""The zoomable timeline's view math (no Qt)."""
from __future__ import annotations

import pytest

from orbitopt.viz.timeline_view import (
    TimeWindow,
    format_met,
    format_ruler_label,
    parse_time_input,
    tick_step_s,
    ticks,
)

DAY = 86400.0


def test_window_zooms_about_an_anchor_and_clamps():
    w = TimeWindow.fit(0.0, 1.0, DAY)  # one day
    assert w.is_fit and w.min_span == pytest.approx(2.0 / DAY)
    w.zoom(4.0, anchor=0.5)
    assert (w.start, w.end) == pytest.approx((0.375, 0.625))
    w.zoom(4.0, anchor=0.4)  # the anchor keeps its place on screen
    assert w.to_fraction(0.4) == pytest.approx((0.4 - 0.375) / 0.25)
    w.zoom(1e9)  # never deeper than two seconds
    assert w.span == pytest.approx(2.0 / DAY)
    w.zoom(1e-9)  # never wider than the mission
    assert w.is_fit
    w.set(0.9, 1.3)  # past the end: slides back inside, keeps the span
    assert (w.start, w.end) == pytest.approx((0.6, 1.0))
    w.pan(-5.0)
    assert (w.start, w.end) == pytest.approx((0.0, 0.4))


def test_follow_pages_for_playback_and_centers_on_a_jump():
    w = TimeWindow.fit(0.0, 10.0, DAY)
    w.set(0.0, 1.0)
    assert not w.follow(0.5)
    assert w.follow(1.02)  # just past the edge: page, playhead 8% in
    assert w.start == pytest.approx(1.02 - 0.08)
    assert w.follow(7.0)  # far away: center
    assert 0.5 * (w.start + w.end) == pytest.approx(7.0)


def test_ruler_steps_follow_the_zoom():
    assert tick_step_s(22.5 * 3600, 1000) == 3 * 3600  # Starship at fit
    assert tick_step_s(360, 1000) == 60  # six minutes: 30 s would be 83 px apart
    assert tick_step_s(2, 1000) == 0.2
    assert ticks(0.0, 1.0 / 24, DAY, 900) == pytest.approx([k * 900 / DAY for k in range(5)])


def test_met_formatting():
    assert format_met(395) == "T+0:06:35"
    assert format_met(81154) == "T+22:32:34"
    assert format_met(-5) == "T-0:00:05"
    assert format_met(12 * DAY + 4 * 3600 + 12 * 60) == "T+12d 04:12:00"
    assert format_met(139.44, resolution_s=0.05) == "T+0:02:19.4"
    assert format_ruler_label(81000, 1800) == "22:30:00"
    assert format_ruler_label(79200, 3 * 3600) == "22h"
    assert format_ruler_label(DAY + 6 * 3600, 6 * 3600) == "1d 06h"
    assert format_ruler_label(60, 60) == "0:01:00"
    assert format_ruler_label(2 * DAY, 2 * DAY) == "2d"
    assert format_ruler_label(139.4, 0.2) == "2:19.4"
    assert format_ruler_label(30, 15) == "0:00:30"


@pytest.mark.parametrize("text, expected", [
    ("T+22:32:34", 81154.0),
    ("t+6m35s", 395.0),
    ("T-10s", -10.0),
    ("22:32:34", 81154.0),
    ("6:35", 395.0),
    ("1d 02:00:00", DAY + 7200.0),
    ("1d 2h", DAY + 7200.0),
    ("395 s", 395.0),
    ("1.5h", 5400.0),
    ("+30s", 1030.0),   # relative to now (1000 s)
    ("-5m", 700.0),
    ("2", 2 * DAY),     # a bare number is in the timeline's unit
    ("nonsense", None),
    ("5 parsecs", None),
    ("", None),
])
def test_parse_time_input(text, expected):
    got = parse_time_input(text, now_s=1000.0, unit_s=DAY)
    assert got == (None if expected is None else pytest.approx(expected))


def test_parse_utc_date_needs_an_epoch():
    from datetime import datetime, timezone

    from orbitopt.timescales import utc_datetime_to_tdb_seconds

    ref = utc_datetime_to_tdb_seconds(datetime(2026, 11, 13, 23, 0, tzinfo=timezone.utc))
    assert parse_time_input("2026-11-14 01:30", 0.0, DAY, reference_et=ref) == pytest.approx(9000.0)
    assert parse_time_input("2026-11-14 01:30", 0.0, DAY) is None
