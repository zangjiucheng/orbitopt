"""Flight Lab telemetry charts: hover readout, linked crosshair, zoom / pan,
keyboard stepping -- and the timeline drawing one bar per entity lane."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _channel(channel_id="altitude"):
    from orbitopt.flightlab.ui.telemetry import TelemetryChannel, TelemetrySeries

    stage1 = TelemetrySeries(channel_id, "Stage1", "km", "#e8a23e", "stage1",
                             tuple((float(t), float(t)) for t in range(0, 101, 10)))
    booster = TelemetrySeries(channel_id, "Booster", "km", "#3ecf8e", "booster",
                              tuple((float(t), 200.0 - t) for t in range(60, 301, 20)))
    return TelemetryChannel(channel_id, "Altitude", "km", "", (stage1, booster))


def _chart(qapp, width=420, height=220):
    from orbitopt.flightlab.ui.charts import TelemetryChart

    chart = TelemetryChart("altitude", "Altitude", "km")
    chart.resize(width, height)
    chart.set_channel(_channel())
    chart.set_time_range(0.0, 300.0)
    return chart


# ------------------------------------------------------------------ readout


def test_readout_is_interpolated_and_only_inside_a_series_own_span():
    from orbitopt.flightlab.ui.charts import readout_rows

    channel = _channel()
    rows = {label: value for label, _c, _t, value in readout_rows(channel, 35.0)}
    assert rows == {"Stage1": pytest.approx(35.0)}               # the booster is not flying yet
    both = {label: value for label, _c, _t, value in readout_rows(channel, 70.0)}
    assert both == {"Stage1": pytest.approx(70.0), "Booster": pytest.approx(130.0)}
    after = {label for label, *_ in readout_rows(channel, 250.0)}
    assert after == {"Booster"}                                  # stage 1's track has ended: no clamped value
    assert readout_rows(channel, 400.0) == []


def test_mission_elapsed_time_format():
    from orbitopt.flightlab.ui.charts import fmt_met

    assert fmt_met(0.0) == "T+0:00:00.0"
    assert fmt_met(135.25) == "T+0:02:15.2" or fmt_met(135.25) == "T+0:02:15.3"
    assert fmt_met(3725.0) == "T+1:02:05.0"
    assert fmt_met(-12.5) == "T-0:00:12.5"


# -------------------------------------------------------------------- hover


def test_hover_reports_its_time_and_clears_on_leave(qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    chart = _chart(qapp)
    seen = []
    chart.hover_time_changed.connect(seen.append)
    rect = chart.plot_rect()
    mid = QPointF(rect.center().x(), rect.center().y())
    chart.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, mid, mid, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert seen and seen[-1] == pytest.approx(150.0, abs=2.0) and chart.hover_time == seen[-1]
    outside = QPointF(2.0, 2.0)                                  # over the title, not the plot
    chart.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, outside, outside, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert seen[-1] is None and chart.hover_time is None


def test_a_hover_on_one_chart_is_mirrored_on_the_others_without_feedback(qapp):
    from orbitopt.flightlab.ui.charts import link_charts

    a, b, c = _chart(qapp), _chart(qapp), _chart(qapp)
    link_charts([a, b, c])
    echoes = []
    b.hover_time_changed.connect(echoes.append)                  # b must only RECEIVE the mirror
    a._set_hover(120.0)
    assert b.hover_time == c.hover_time == 120.0 and echoes == []
    a._set_hover(None)
    assert b.hover_time is None and c.hover_time is None


def test_the_hover_box_and_markers_paint(qapp):
    from PySide6.QtGui import QImage

    chart = _chart(qapp)
    chart.set_time(70.0)
    chart.set_hover_time(70.0)
    def shot(hover):
        chart.set_hover_time(hover)
        image = QImage(chart.size(), QImage.Format_ARGB32)
        image.fill(0)
        chart.render(image)
        return image

    plain, hovered = shot(None), shot(70.0)
    rect = chart.plot_rect()
    # the readout box (and the crosshair) change the pixels in the plot's top rows
    changed = sum(1 for x in range(int(rect.left()), int(rect.right()))
                  for y in range(int(rect.top()), int(rect.top()) + 40)
                  if plain.pixel(x, y) != hovered.pixel(x, y))
    assert changed > 400


# --------------------------------------------------------------- zoom / pan


def test_wheel_zoom_keeps_the_point_under_the_pointer_and_refits_y(qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    chart = _chart(qapp)
    full_ymax = chart._ymax
    rect = chart.plot_rect()
    pos = QPointF(rect.left() + rect.width() * 0.25, rect.center().y())   # t = 75 s
    anchor = chart._time_at_x(pos.x())
    seen = []
    chart.view_range_changed.connect(lambda lo, hi: seen.append((lo, hi)))
    event = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, 240), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    chart.wheelEvent(event)
    lo, hi = chart.view_range
    assert chart.is_zoomed and (hi - lo) < 300.0 and seen
    assert chart._time_at_x(pos.x()) == pytest.approx(anchor, abs=1e-6)   # the pointer's time did not move
    assert chart._ymax < full_ymax                                         # y refits to what is visible


def test_zoom_is_clamped_to_the_run_and_resets(qapp):
    chart = _chart(qapp)
    chart.set_view_range(-50.0, 40.0)                           # asks for before the run: slides in
    assert chart.view_range == (0.0, 90.0)
    chart.set_view_range(250.0, 400.0)
    assert chart.view_range[1] == 300.0 and chart.view_range[0] == 150.0
    chart.set_view_range(100.0, 100.0)                          # never a zero-width window
    lo, hi = chart.view_range
    assert hi > lo
    chart.reset_view()                                          # (double-click opens the detail plot)
    assert chart.view_range == (0.0, 300.0) and not chart.is_zoomed


def test_zoom_and_pan_are_shared_by_linked_charts(qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from orbitopt.flightlab.ui.charts import link_charts

    a, b = _chart(qapp), _chart(qapp)
    link_charts([a, b])
    a.set_view_range(60.0, 180.0, emit=True)
    assert b.view_range == (60.0, 180.0)
    # shift-drag pans: dragging left moves the window to later times
    rect = a.plot_rect()
    press = QPointF(rect.center())
    a.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, press, press, Qt.LeftButton, Qt.LeftButton,
                                  Qt.ShiftModifier))
    drag = QPointF(press.x() - rect.width() * 0.25, press.y())
    a.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, drag, drag, Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier))
    a.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, drag, drag, Qt.LeftButton, Qt.NoButton,
                                    Qt.ShiftModifier))
    assert a.view_range == pytest.approx((90.0, 210.0), abs=1.0) and b.view_range == a.view_range


def test_a_plain_click_still_seeks_and_does_not_zoom(qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    chart = _chart(qapp)
    seeks = []
    chart.time_requested.connect(seeks.append)
    pos = QPointF(chart.plot_rect().center())
    chart.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, pos, Qt.LeftButton, Qt.LeftButton,
                                      Qt.NoModifier))
    assert seeks == [pytest.approx(150.0, abs=2.0)] and not chart.is_zoomed


# ----------------------------------------------------------------- keyboard


def test_arrow_keys_step_the_cursor_one_sample(qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    chart = _chart(qapp)
    seeks = []
    chart.time_requested.connect(seeks.append)
    chart.set_time(40.0)
    chart.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    assert seeks[-1] == 50.0
    chart.set_time(60.0)
    chart.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
    assert seeks[-1] == 50.0
    chart.set_time(300.0)
    chart.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    assert seeks[-1] == 300.0                                    # clamped at the end of the run


# ----------------------------------------------------------------- timeline


def _fork_scene():
    return {
        "bodies": [],
        "timeline": {
            "min": 0.0, "max": 600.0, "unitLabel": "s",
            "phases": [
                {"label": "Boost", "start": 0.0, "end": 140.0},
                {"label": "Staging", "start": 140.0, "end": 150.0},
                {"label": "Upper-stage burn", "start": 150.0, "end": 400.0, "lane": "upper-stage"},
                {"label": "Parking orbit", "start": 400.0, "end": 600.0, "lane": "upper-stage"},
                {"label": "First stage", "start": 150.0, "end": 580.0, "lane": "booster"},
            ],
            "branches": {"lanes": [{"id": "upper-stage", "label": "Upper stage"},
                                   {"id": "booster", "label": "First stage"}]},
            "events": [],
        },
    }


def test_a_lane_is_one_continuous_bar_per_entity(qapp):
    from PySide6.QtGui import QImage

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    state = MissionState()
    bar = TimelineBar(state)
    state.set_scene(_fork_scene())
    assert bar.track._lane_span("") == (0.0, 150.0)                    # Boost + Staging: the stack, one span
    assert bar.track._lane_span("upper-stage") == (150.0, 600.0)       # burn + parking orbit: ONE upper stage
    assert bar.track._lane_span("booster") == (150.0, 580.0)
    assert bar.track._lane_span("nobody") is None
    bar.track.resize(900, bar.track.height())
    image = QImage(bar.track.size(), QImage.Format_ARGB32)
    image.fill(0)
    bar.track.render(image)                                            # paints without error with merged lanes
    assert not image.isNull()
