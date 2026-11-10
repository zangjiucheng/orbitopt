"""The mission timeline under the 3D view: play / event-step buttons, a time
box you can type into, the time-warp control, and a zoomable track.

The track (TimelineTrack) shows a window of the mission that zooms from the
whole span down to two seconds (timeline_view.TimeWindow): an overview strip
with the visible window bracketed, a ruler with MET ticks that adapt to the
zoom, the mission phases, and the events with their names. Time is
continuous -- scrubbing at the deepest zoom moves it by milliseconds, where
the old 10,000-step slider moved it by eight seconds on Starship's timeline
and by forty minutes on a Mars cruise.

Mouse: drag on the track to scrub (Shift-drag: ten times finer; grabbing
the playhead nudges it from where it is instead of jumping), scroll to zoom
about the cursor (scroll
sideways, or right-drag, to pan), drag the bracket in the overview strip to
pan, double-click a phase or an event to zoom to it. Keys (window menu):
= / - zoom, 0 fit, G focus the time box.

It reads and writes the current time through MissionState only, so anything
else that moves time (playback, a maneuver click, a keyboard step) moves the
track too without the window having to update it by hand.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz import icons
from orbitopt.viz.state import MissionState, maneuver_windows
from orbitopt.viz.theme import (
    ACCENT,
    ACCENT_DIM,
    ACCENT_MID,
    HAIRLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PANEL,
    PANEL_RAISED,
    SURFACE_HOVER,
)
from orbitopt.viz.timeline_view import (
    TimeWindow,
    format_met,
    format_ruler_label,
    minor_step_s,
    parse_time_input,
    tick_step_s,
    ticks,
)

# KSP-style time warp: the multiplier is simulated seconds per real second,
# so 1x is real time. The range has to span both ends of what a scene can
# hold -- a Starship landing burn lasts tens of seconds and is worth watching
# at 1x, a Mars cruise lasts 300 days and needs ~10^6x to cross in seconds.
# The slider is continuous on a log scale; `,` / `.` step through the KSP
# warp levels (WARP_LEVELS) and `/` drops back to real time, as in KSP.
# (Before this, "1x" meant one simulated *day* per second -- 86,400x real
# time -- and the slowest setting was still 86x, too fast for a burn.)
WARP_MIN = 0.1
WARP_MAX = 1.0e7
WARP_LEVELS = (0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)
WARP_SLIDER_STEPS = 1000

# Seconds in one timeline unit (timeline.unitLabel); unknown labels fall back
# to days, the unit every orbitopt exporter writes.
_UNIT_SECONDS = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "min": 60.0, "minutes": 60.0,
                 "h": 3600.0, "hr": 3600.0, "hours": 3600.0, "d": 86400.0, "day": 86400.0, "days": 86400.0}


def timeline_unit_seconds(unit_label: str | None) -> float:
    return _UNIT_SECONDS.get((unit_label or "days").strip().lower(), 86400.0)


def _sig(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def format_warp(warp: float) -> str:
    """'0.1×', '1×', '50×', '1,000×', '100k×', '10M×'."""
    if warp >= 1.0e6:
        return f"{_sig(warp / 1.0e6)}M×"
    if warp >= 1.0e5:
        return f"{_sig(warp / 1.0e3)}k×"
    return f"{_sig(warp)}×"


def format_rate(warp: float) -> str:
    """Simulated time per real second at ``warp``: '1 s/s', '17 min/s', '1.2 d/s'."""
    seconds = warp
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if seconds >= size:
            return f"{_sig(seconds / size)} {unit}/s"
    return f"{_sig(seconds)} s/s"


def warp_from_slider(value: int) -> float:
    return WARP_MIN * (WARP_MAX / WARP_MIN) ** (value / WARP_SLIDER_STEPS)


def slider_from_warp(warp: float) -> int:
    warp = min(WARP_MAX, max(WARP_MIN, warp))
    return int(round(math.log(warp / WARP_MIN) / math.log(WARP_MAX / WARP_MIN) * WARP_SLIDER_STEPS))


def step_warp(warp: float, direction: int) -> float:
    """The next KSP warp level above (``direction > 0``) or below the current
    warp -- from an in-between value, the nearest level in that direction."""
    if direction > 0:
        return next((w for w in WARP_LEVELS if w > warp * (1 + 1e-9)), WARP_LEVELS[-1])
    return next((w for w in reversed(WARP_LEVELS) if w < warp * (1 - 1e-9)), WARP_LEVELS[0])



def epoch_to_date_string(reference_et: float, day_offset: float) -> str:
    from orbitopt.timescales import tdb_seconds_to_utc_datetime

    dt = tdb_seconds_to_utc_datetime(reference_et + day_offset * 86400.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_view_span(seconds: float) -> str:
    """The zoom readout: '2 s', '45 s', '6.5 min', '3.2 h', '12 d'."""
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if seconds >= 1.5 * size:
            return f"{_sig(seconds / size)} {unit}"
    return f"{_sig(seconds)} s"


def _mono_font(point_size: float) -> QFont:
    font = QFont()
    font.setFamilies(["IBM Plex Mono", "SF Mono", "Menlo", "Consolas", "monospace"])
    font.setPointSizeF(point_size)
    return font


class TimelineTrack(QWidget):
    """The zoomable, painted timeline (see the module docstring)."""

    view_changed = Signal()

    PAD_X = 10
    OVERVIEW_H = 10
    RULER_H = 22
    PHASE_H = 16
    EVENT_H = 22
    GAP = 3
    _HIT_TOLERANCE_PX = 7
    _PLAYHEAD_GRAB_PX = 6
    FINE_SCRUB = 0.1  # Shift-drag moves time at a tenth of the pointer's rate
    _WHEEL_ZOOM_PER_NOTCH = 1.25  # one 120-unit wheel notch

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self.window: TimeWindow | None = None
        self._unit_s = 86400.0
        self._phases: list[dict] = []
        self._drag: str | None = None  # "scrub" | "pan" | "overview"
        self._drag_anchor = (0.0, 0.0)  # (x, window start) when a pan began
        self._scrub_anchor = (0.0, 0.0, False)  # (x, time, fine) the scrub is measured from
        self._hover_x: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(self._height(False))
        state.scene_changed.connect(self._on_scene_changed)
        state.time_changed.connect(self._on_time_changed)

    # ------------------------------------------------------------ layout
    def _height(self, phases: bool) -> int:
        return (self.OVERVIEW_H + self.GAP + self.RULER_H + self.EVENT_H
                + ((self.PHASE_H + self.GAP) if phases else 0) + 2)

    def _rows(self) -> dict[str, tuple[float, float]]:
        """(top, bottom) of each row in widget coordinates."""
        y = 0.0
        rows = {"overview": (y, y + self.OVERVIEW_H)}
        y += self.OVERVIEW_H + self.GAP
        rows["ruler"] = (y, y + self.RULER_H)
        y += self.RULER_H
        if self._phases:
            rows["phases"] = (y + 1, y + 1 + self.PHASE_H)
            y += self.PHASE_H + self.GAP
        rows["events"] = (y, y + self.EVENT_H)
        return rows

    def _row_at(self, y: float) -> str | None:
        for name, (top, bottom) in self._rows().items():
            if top - 1 <= y <= bottom + 1:
                return name
        return None

    @property
    def track_left(self) -> float:
        return float(self.PAD_X)

    @property
    def track_width(self) -> float:
        return max(1.0, self.width() - 2.0 * self.PAD_X)

    def x_of(self, t: float) -> float:
        """Widget x of timeline time ``t`` on the zoomed rows."""
        return self.track_left + self.window.to_fraction(t) * self.track_width

    def time_at(self, x: float) -> float:
        f = (x - self.track_left) / self.track_width
        return min(self.window.hi, max(self.window.lo, self.window.from_fraction(f)))

    def _overview_x(self, t: float) -> float:
        w = self.window
        return self.track_left + (t - w.lo) / w.full_span * self.track_width

    def _overview_time(self, x: float) -> float:
        w = self.window
        return w.lo + (x - self.track_left) / self.track_width * w.full_span

    # ------------------------------------------------------------ state
    @property
    def phases(self) -> list[dict]:
        return self._phases

    def _on_scene_changed(self, _scene):
        timeline = self._state.timeline
        self._phases = list((timeline or {}).get("phases", []))
        if timeline is None:
            self.window = None
        else:
            self._unit_s = timeline_unit_seconds(timeline.get("unitLabel"))
            self.window = TimeWindow.fit(timeline["min"], timeline["max"], self._unit_s)
        self.setFixedHeight(self._height(bool(self._phases)))
        self.view_changed.emit()
        self.update()

    def _on_time_changed(self, t: float):
        if self.window is not None and self._drag != "scrub" and self.window.follow(t):
            self.view_changed.emit()
        self.update()

    def current_phase(self) -> dict | None:
        t = self._state.time
        return next((p for p in self._phases if p["start"] <= t <= p["end"]), None)

    def phase_at(self, x: float) -> dict | None:
        if self.window is None:
            return None
        t = self.time_at(x)
        return next((p for p in self._phases if p["start"] <= t <= p["end"]), None)

    def events_near(self, x: float, tol: float | None = None) -> list[dict]:
        """Events within ``tol`` px of ``x`` on the zoomed rows, nearest first."""
        if self.window is None:
            return []
        tol = self._HIT_TOLERANCE_PX if tol is None else tol
        near = sorted(((abs(self.x_of(e.get("time", 0.0)) - x), k, e)
                       for k, e in enumerate(self._state.events)), key=lambda p: (p[0], p[1]))
        return [e for d, _k, e in near if d <= tol]

    # ------------------------------------------------------------ view
    def set_view(self, start: float, end: float):
        if self.window is not None:
            self.window.set(start, end)
            self.view_changed.emit()
            self.update()

    def zoom(self, factor: float, anchor: float | None = None):
        """Zoom by ``factor`` about ``anchor`` (default: the playhead when it
        is in view, else the window's center)."""
        if self.window is None:
            return
        if anchor is None:
            t = self._state.time
            anchor = t if self.window.start <= t <= self.window.end else None
        self.window.zoom(factor, anchor)
        self.view_changed.emit()
        self.update()

    def fit(self):
        if self.window is not None:
            self.window.reset()
            self.view_changed.emit()
            self.update()

    def zoom_to_phase(self, phase: dict):
        pad = 0.04 * (phase["end"] - phase["start"])
        self.set_view(phase["start"] - pad, phase["end"] + pad)

    def zoom_to_event(self, event: dict):
        """Frame an event with its neighbourhood: its burn-arrow window (half
        the gap to the nearest other event, capped), at least a minute."""
        if self.window is None:
            return
        windows = maneuver_windows(self._state.events, self.window.full_span)
        half = max(windows.get(id(event), 0.0) * 1.5, 60.0 / self._unit_s)
        t = event.get("time", 0.0)
        self.set_view(t - half, t + half)

    # ------------------------------------------------------------ paint
    def paintEvent(self, _event):  # noqa: N802
        if self.window is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rows = self._rows()
        self._paint_overview(p, *rows["overview"])
        self._paint_ruler(p, *rows["ruler"])
        if "phases" in rows:
            self._paint_phases(p, *rows["phases"])
        self._paint_events(p, *rows["events"])
        self._paint_playhead(p, rows["ruler"][0], rows["events"][1])
        p.end()

    def _paint_overview(self, p: QPainter, top: float, bottom: float):
        w, left, width = self.window, self.track_left, self.track_width
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(PANEL_RAISED))
        p.drawRoundedRect(QRectF(left, top, width, bottom - top), 2, 2)
        for k, phase in enumerate(self._phases):
            x0, x1 = self._overview_x(phase["start"]), self._overview_x(phase["end"])
            p.setBrush(QColor(SURFACE_HOVER if k % 2 else HAIRLINE))
            p.drawRect(QRectF(x0, top + 3, max(1.0, x1 - x0), bottom - top - 6))
        p.setPen(QPen(QColor(ACCENT_MID), 1.0))
        for e in self._state.events:
            x = self._overview_x(e.get("time", 0.0))
            p.drawLine(QPointF(x, top + 1), QPointF(x, bottom - 1))
        # The visible window, bracketed (the whole strip at fit).
        x0, x1 = self._overview_x(w.start), self._overview_x(w.end)
        x1 = max(x1, x0 + 3.0)  # stays grabbable however deep the zoom
        p.setPen(QPen(QColor(INK_SECONDARY if not w.is_fit else HAIRLINE), 1.2))
        p.setBrush(QColor(255, 255, 255, 18 if not w.is_fit else 0))
        p.drawRoundedRect(QRectF(x0, top + 0.5, x1 - x0, bottom - top - 1), 2, 2)
        xt = self._overview_x(self._state.time)
        p.setPen(QPen(QColor(ACCENT), 1.6))
        p.drawLine(QPointF(xt, top - 1), QPointF(xt, bottom + 1))

    def _paint_ruler(self, p: QPainter, top: float, bottom: float):
        w = self.window
        span_s = w.span * self._unit_s
        major = tick_step_s(span_s, self.track_width)
        minor = minor_step_s(major)
        p.setPen(QPen(QColor(HAIRLINE), 1.0))
        p.drawLine(QPointF(self.track_left, bottom), QPointF(self.track_left + self.track_width, bottom))
        for t in ticks(w.start, w.end, self._unit_s, minor):
            x = self.x_of(t)
            p.drawLine(QPointF(x, bottom - 4), QPointF(x, bottom))
        p.setFont(_mono_font(8.5))
        fm = p.fontMetrics()
        for t in ticks(w.start, w.end, self._unit_s, major):
            x = self.x_of(t)
            p.setPen(QPen(QColor(INK_MUTED), 1.0))
            p.drawLine(QPointF(x, bottom - 9), QPointF(x, bottom))
            text = format_ruler_label(t * self._unit_s, major)
            tw = fm.horizontalAdvance(text)
            tx = min(max(x + 3, self.track_left), self.track_left + self.track_width - tw)
            p.setPen(QColor(INK_SECONDARY))
            p.drawText(QPointF(tx, top + fm.ascent() + 1), text)
        if self._hover_x is not None and self.track_left <= self._hover_x <= self.track_left + self.track_width:
            p.setPen(QPen(QColor(INK_MUTED), 1.0, Qt.DashLine))
            p.drawLine(QPointF(self._hover_x, top), QPointF(self._hover_x, self._rows()["events"][1]))

    def _paint_phases(self, p: QPainter, top: float, bottom: float):
        current = self.current_phase()
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        p.setFont(font)
        left, right = self.track_left, self.track_left + self.track_width
        for k, phase in enumerate(self._phases):
            x0, x1 = max(self.x_of(phase["start"]), left), min(self.x_of(phase["end"]), right)
            if x1 <= x0:
                continue
            rect = QRectF(x0, top, max(1.0, x1 - x0), bottom - top)
            active = phase is current
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT_DIM if active else (PANEL_RAISED if k % 2 == 0 else SURFACE_HOVER)))
            p.drawRect(rect)
            p.setPen(QPen(QColor(HAIRLINE), 1.0))
            p.drawLine(rect.topLeft(), rect.bottomLeft())
            text = p.fontMetrics().elidedText(phase["label"], Qt.ElideRight, int(rect.width() - 8))
            if text and text != "…":
                p.setPen(QColor(INK_PRIMARY if active else INK_SECONDARY))
                p.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def _paint_events(self, p: QPainter, top: float, bottom: float):
        left, right = self.track_left, self.track_left + self.track_width
        visible = sorted(((self.x_of(e.get("time", 0.0)), e) for e in self._state.events
                          if left - 1 <= self.x_of(e.get("time", 0.0)) <= right + 1), key=lambda p_: p_[0])
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        p.setFont(font)
        fm = p.fontMetrics()
        active = self._state.active_event()
        cy = top + 7.0
        label_end = -1e9  # right edge of the last label drawn, so labels never overlap
        for k, (x, e) in enumerate(visible):
            burn = "vector" in e
            p.setPen(QPen(QColor(ACCENT), 1.3))
            p.setBrush(QColor(ACCENT) if burn else QColor(PANEL))
            if burn:
                tri = QPainterPath()
                tri.moveTo(x - 4.0, cy - 5.0)
                tri.lineTo(x + 4.0, cy - 5.0)
                tri.lineTo(x, cy + 2.0)
                tri.closeSubpath()
                p.drawPath(tri)
            else:
                p.drawEllipse(QRectF(x - 3.0, cy - 4.5, 6.0, 6.0))
            next_x = visible[k + 1][0] if k + 1 < len(visible) else right + 1e9
            room = min(next_x - x - 10.0, right - x - 6.0)
            if x - 2 < label_end or room < 24:
                continue
            text = fm.elidedText(e.get("label", ""), Qt.ElideRight, int(room))
            if not text or text == "…":
                continue
            p.setPen(QColor(INK_PRIMARY if e is active else INK_SECONDARY))
            p.drawText(QPointF(x + 6.0, top + 7.0 + fm.ascent() - 1), text)
            label_end = x + 6.0 + fm.horizontalAdvance(text)

    def _paint_playhead(self, p: QPainter, top: float, bottom: float):
        t = self._state.time
        if not (self.window.start <= t <= self.window.end):
            return
        x = self.x_of(t)
        p.setPen(QPen(QColor(ACCENT), 1.6))
        p.drawLine(QPointF(x, top + 4), QPointF(x, bottom))
        head = QPainterPath()
        head.moveTo(x - 5.0, top)
        head.lineTo(x + 5.0, top)
        head.lineTo(x, top + 6.0)
        head.closeSubpath()
        p.setBrush(QColor(ACCENT))
        p.setPen(Qt.NoPen)
        p.drawPath(head)

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, event):  # noqa: N802
        if self.window is None:
            return
        x, y = event.position().x(), event.position().y()
        row = self._row_at(y)
        if event.button() in (Qt.RightButton, Qt.MiddleButton) or (
                event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._drag, self._drag_anchor = "pan", (x, self.window.start)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        if row == "overview":
            w = self.window
            x0, x1 = self._overview_x(w.start), self._overview_x(w.end)
            if not (x0 - 3 <= x <= max(x1, x0 + 3) + 3):
                w.center_on(self._overview_time(x))
                self.view_changed.emit()
            self._drag, self._drag_anchor = "overview", (x, self.window.start)
            self.update()
            return
        if row == "phases":
            phase = self.phase_at(x)
            if phase is not None:
                self._state.set_time(phase["start"])
            return
        near = self.events_near(x)
        on_playhead = abs(x - self.x_of(self._state.time)) <= self._PLAYHEAD_GRAB_PX
        if row == "events" and near and not on_playhead:
            self._state.set_time(near[0].get("time", 0.0))  # snap to the event
            return
        self._drag = "scrub"
        if not on_playhead:  # grabbing the playhead nudges it from where it is: no jump
            self._state.set_time(self.time_at(x))
        self._scrub_anchor = (x, self._state.time, bool(event.modifiers() & Qt.ShiftModifier))

    def mouseMoveEvent(self, event):  # noqa: N802
        if self.window is None:
            return
        x, y = event.position().x(), event.position().y()
        if self._drag == "scrub":
            fine = bool(event.modifiers() & Qt.ShiftModifier)
            x0, t0, was_fine = self._scrub_anchor
            if fine != was_fine:  # Shift pressed / released mid-drag: carry on from here
                x0, t0 = x, self._state.time
                self._scrub_anchor = (x0, t0, fine)
            # Relative to where the drag started, at 1/10 the rate with Shift.
            gain = self.FINE_SCRUB if fine else 1.0
            t = t0 + (x - x0) / self.track_width * self.window.span * gain
            self._state.set_time(min(self.window.hi, max(self.window.lo, t)))
        elif self._drag == "pan":
            x_anchor, start = self._drag_anchor
            dt = -(x - x_anchor) / self.track_width * self.window.span
            self.window.pan(start + dt - self.window.start)
            self.view_changed.emit()
        elif self._drag == "overview":
            x_anchor, start = self._drag_anchor
            dt = (x - x_anchor) / self.track_width * self.window.full_span
            self.window.pan(start + dt - self.window.start)
            self.view_changed.emit()
        self._hover_x = x
        self._update_tooltip(x, y, event)
        self.update()

    def mouseReleaseEvent(self, _event):  # noqa: N802
        self._drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if self.window is None:
            return
        x, row = event.position().x(), self._row_at(event.position().y())
        near = self.events_near(x)
        if row == "events" and near:
            self.zoom_to_event(near[0])
        elif row == "phases" and self.phase_at(x) is not None:
            self.zoom_to_phase(self.phase_at(x))
        elif row == "overview":
            self.fit()

    def leaveEvent(self, _event):  # noqa: N802
        self._hover_x = None
        self.update()

    def wheelEvent(self, event):  # noqa: N802
        if self.window is None:
            return
        pixel, angle = event.pixelDelta(), event.angleDelta()
        dx = pixel.x() if not pixel.isNull() else angle.x() / 4.0
        dy = pixel.y() if not pixel.isNull() else angle.y() / 4.0
        if event.modifiers() & Qt.ShiftModifier and not dx:
            dx, dy = dy, 0.0
        if abs(dx) > abs(dy):  # sideways: pan
            self.window.pan(-dx / self.track_width * self.window.span)
        else:  # vertical: zoom about the cursor (one notch = 30 px here)
            factor = self._WHEEL_ZOOM_PER_NOTCH ** (dy / 30.0)
            self.window.zoom(factor, self.time_at(event.position().x()))
        self.view_changed.emit()
        self.update()
        event.accept()

    def _update_tooltip(self, x: float, y: float, event):
        row = self._row_at(y)
        tips = []
        if row == "phases":
            phase = self.phase_at(x)
            if phase is not None:
                tips.append(f"{phase['label']}\n{format_met(phase['start'] * self._unit_s)} – "
                            f"{format_met(phase['end'] * self._unit_s)}")
                if phase.get("note"):
                    tips.append(phase["note"])
                tips.append("Click: jump to its start · double-click: zoom to it")
        elif row == "events":
            for e in self.events_near(x):
                tips.append(f"{e.get('label', '')}  {format_met(e.get('time', 0.0) * self._unit_s)}"
                            + (f"\n{e['note']}" if e.get("note") else ""))
            if tips:
                tips.append("Click: jump · double-click: zoom to it")
        elif row == "overview":
            tips.append("Whole mission · drag the bracket to pan · double-click: fit")
        if row in ("ruler", "events", "phases") and not tips:
            res = self.window.span * self._unit_s / self.track_width
            tips.append(format_met(self.time_at(x) * self._unit_s, res))
        text = "\n\n".join(tips) if len(tips) > 1 and row == "events" else "\n".join(tips)
        if text != self.toolTip():
            self.setToolTip(text)
            if text:
                QToolTip.showText(event.globalPosition().toPoint(), text, self)


class WarpLadder(QWidget):
    """KSP-style warp selector: one bar per warp level (WARP_LEVELS), evenly
    spaced and rising with the warp, filled up to the current one. Click or
    drag along it to pick a level, scroll to fine-tune between levels. The
    1x bar carries a tick underneath so real time is easy to find.

    Even spacing is the point: on the old log slider 0.1x..10Mx was squeezed
    into 170 px, so landing on 1x (or any round level) took several tries."""

    level_picked = Signal(float)
    nudged = Signal(int)

    BAR_W = 9
    GAP = 3
    H = 26

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._warp = 1.0
        n = len(WARP_LEVELS)
        self.setFixedSize(n * self.BAR_W + (n - 1) * self.GAP + 2, self.H)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_warp(self, warp: float):
        self._warp = warp
        self.update()

    def index_at(self, x: float) -> int:
        step = self.BAR_W + self.GAP
        return min(len(WARP_LEVELS) - 1, max(0, int((x - 1) // step)))

    def bar_rect(self, i: int) -> QRectF:
        n = len(WARP_LEVELS)
        h = 7.0 + (self.H - 13.0) * i / (n - 1)
        x = 1 + i * (self.BAR_W + self.GAP)
        return QRectF(x, self.H - 6.0 - h, self.BAR_W, h)

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        log_w = math.log(self._warp)
        for i, level in enumerate(WARP_LEVELS):
            rect = self.bar_rect(i)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(SURFACE_HOVER))
            p.drawRoundedRect(rect, 1.5, 1.5)
            # Filled up to the warp; the bar above a between-levels warp
            # (after a fine nudge) fills in proportion, on the log scale.
            if level <= self._warp * (1 + 1e-9):
                fill = 1.0
            elif i > 0 and WARP_LEVELS[i - 1] < self._warp:
                lo = math.log(WARP_LEVELS[i - 1])
                fill = (log_w - lo) / (math.log(level) - lo)
            else:
                fill = 0.0
            if fill > 0:
                part = QRectF(rect.left(), rect.bottom() - rect.height() * fill, rect.width(), rect.height() * fill)
                p.setBrush(QColor(ACCENT))
                p.drawRoundedRect(part, 1.5, 1.5)
            if level == 1.0:
                p.setBrush(QColor(INK_PRIMARY if abs(self._warp - 1.0) < 1e-9 else INK_MUTED))
                cx = rect.center().x()
                tri = QPainterPath()
                tri.moveTo(cx - 3.0, self.H - 0.5)
                tri.lineTo(cx + 3.0, self.H - 0.5)
                tri.lineTo(cx, self.H - 4.0)
                tri.closeSubpath()
                p.drawPath(tri)
        p.end()

    def _pick(self, x: float):
        self.level_picked.emit(WARP_LEVELS[self.index_at(x)])

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._pick(event.position().x())

    def mouseMoveEvent(self, event):  # noqa: N802
        i = self.index_at(event.position().x())
        self.setToolTip(f"{format_warp(WARP_LEVELS[i])} · {format_rate(WARP_LEVELS[i])}\n"
                        "Click a bar for that warp · scroll to fine-tune")
        if event.buttons() & Qt.LeftButton:
            self._pick(event.position().x())

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.nudged.emit(1 if delta > 0 else -1)
        event.accept()


class TimelineBar(QWidget):
    """Controls row (play, previous / next event, the time box, the zoom
    controls, time warp) over the zoomable track.

    Time flows one way: the track and the time box write to ``state`` when
    the *user* acts, and follow ``state.time_changed`` otherwise. The bar
    owns the time-warp setting (``warp``) and the auto-slow toggle; the
    playback timer that consumes them lives in the window.
    """

    play_clicked = Signal()
    reverse_clicked = Signal()
    auto_slow_toggled = Signal(bool)
    # The previous / next event buttons: the window seeks (and frames a burn)
    # the same way as for Shift+arrow.
    event_step_requested = Signal(int)

    _CONTROLS_H = 44
    _MARGINS = (16, 6, 16, 6)

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._warp = 1.0  # every scene opens at real time (_on_scene_changed)
        self.setObjectName("timelineBar")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._build()
        state.scene_changed.connect(self._on_scene_changed)
        state.time_changed.connect(self._on_time_changed)
        self._fit_height()

    # ------------------------------------------------------------- build
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(*self._MARGINS)
        outer.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        outer.addLayout(row)

        self.reverse_btn = QPushButton()
        self.reverse_btn.setObjectName("playButton")
        self.reverse_btn.setIcon(icons.play_icon(INK_SECONDARY, reverse=True))
        self.reverse_btn.setIconSize(QSize(15, 15))
        self.reverse_btn.setToolTip("Play backwards / pause (Shift+Space)")
        self.reverse_btn.clicked.connect(self.reverse_clicked)
        row.addWidget(self.reverse_btn)
        self.play_btn = QPushButton()
        self.play_btn.setObjectName("playButton")
        self.play_btn.setIcon(icons.play_icon())
        self.play_btn.setIconSize(QSize(15, 15))
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.clicked.connect(self.play_clicked)
        row.addWidget(self.play_btn)

        self.prev_event_btn = self._tool_button(icons.chevron_left_icon(), "Previous event (Shift+←)")
        self.prev_event_btn.clicked.connect(lambda: self.event_step_requested.emit(-1))
        self.next_event_btn = self._tool_button(icons.chevron_right_icon(), "Next event (Shift+→)")
        self.next_event_btn.clicked.connect(lambda: self.event_step_requested.emit(+1))
        row.addWidget(self.prev_event_btn)
        row.addWidget(self.next_event_btn)

        self.met_edit = QLineEdit("T+0:00:00")
        self.met_edit.setObjectName("metReadout")
        self.met_edit.setToolTip(
            "Mission elapsed time -- type to jump (G): T+22:32:34, T+6m35s, "
            "+30s / -5m (relative), 395 s, or a UTC date 2026-11-14 03:12"
        )
        self.met_edit.setFixedWidth(150)
        self.met_edit.returnPressed.connect(self._on_met_entered)
        self.met_edit.editingFinished.connect(self._refresh_readouts)
        row.addWidget(self.met_edit)

        self.date_label = QLabel("")
        self.date_label.setObjectName("dateReadout")
        row.addWidget(self.date_label)
        row.addStretch(1)
        self.event_label = QLabel("")
        self.event_label.setObjectName("eventReadout")
        row.addWidget(self.event_label)
        row.addSpacing(10)

        self.zoom_out_btn = self._tool_button(None, "Zoom out (-)", "−")
        self.zoom_out_btn.clicked.connect(lambda: self.track.zoom(1 / 2))
        self.zoom_label = QLabel("")
        self.zoom_label.setObjectName("scaleReadout")
        self.zoom_label.setMinimumWidth(56)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setToolTip("Visible span of the timeline -- scroll on the track to zoom")
        self.zoom_in_btn = self._tool_button(None, "Zoom in (=)", "+")
        self.zoom_in_btn.clicked.connect(lambda: self.track.zoom(2))
        self.fit_btn = self._tool_button(None, "Fit the whole mission (0)", "Fit")
        self.fit_btn.setObjectName("timelineTextButton")
        self.fit_btn.clicked.connect(lambda: self.track.fit())
        for w in (self.zoom_out_btn, self.zoom_label, self.zoom_in_btn, self.fit_btn):
            row.addWidget(w)
        row.addSpacing(10)

        speed_col = QVBoxLayout()
        speed_col.setSpacing(0)
        speed_header = QHBoxLayout()
        speed_header.addWidget(QLabel("TIME WARP"))
        self.auto_slow_btn = QPushButton()
        self.auto_slow_btn.setObjectName("autoSlowButton")
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT))
        self.auto_slow_btn.setIconSize(QSize(12, 12))
        self.auto_slow_btn.setCheckable(True)
        self.auto_slow_btn.setChecked(True)
        self.auto_slow_btn.setToolTip(
            "Auto-slow near timeline events (on by default) -- click to play "
            "through events at full speed instead"
        )
        self.auto_slow_btn.toggled.connect(self._on_auto_slow_toggled)
        speed_header.addWidget(self.auto_slow_btn)
        speed_header.addStretch(1)
        self.speed_value_label = QLabel(format_warp(self._warp))
        self.speed_value_label.setObjectName("speedValue")
        speed_header.addWidget(self.speed_value_label)
        speed_col.addLayout(speed_header)

        warp_row = QHBoxLayout()
        warp_row.setSpacing(4)
        self.warp_down_btn = self._tool_button(None, "Slower: previous warp level (,)", "«")
        self.warp_down_btn.clicked.connect(lambda: self.step_warp(-1))
        self.warp_ladder = WarpLadder()
        self.warp_ladder.level_picked.connect(self.set_warp)
        self.warp_ladder.nudged.connect(self.nudge_speed)
        self.warp_up_btn = self._tool_button(None, "Faster: next warp level (.)", "»")
        self.warp_up_btn.clicked.connect(lambda: self.step_warp(+1))
        self.real_time_btn = self._tool_button(None, "Real time, 1× (/)", "1×")
        self.real_time_btn.setObjectName("realTimeButton")
        self.real_time_btn.setCheckable(True)
        self.real_time_btn.clicked.connect(lambda: self.set_warp(1.0))
        for w in (self.warp_down_btn, self.warp_ladder, self.warp_up_btn, self.real_time_btn):
            warp_row.addWidget(w, 0, Qt.AlignVCenter)
        speed_col.addLayout(warp_row)
        row.addLayout(speed_col)

        self.track = TimelineTrack(self._state)
        self.track.view_changed.connect(self._on_view_changed)
        outer.addWidget(self.track)

    def _tool_button(self, icon, tip: str, text: str = "") -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("timelineToolButton")
        if icon is not None:
            btn.setIcon(icon)
            btn.setIconSize(QSize(14, 14))
        btn.setToolTip(tip)
        btn.setFocusPolicy(Qt.NoFocus)
        return btn

    def _fit_height(self):
        top, bottom = self._MARGINS[1], self._MARGINS[3]
        self.setFixedHeight(top + self._CONTROLS_H + 4 + self.track.minimumHeight() + bottom)

    # ------------------------------------------------------- state sync
    @property
    def unit_seconds(self) -> float:
        return timeline_unit_seconds((self._state.timeline or {}).get("unitLabel"))

    def _on_scene_changed(self, _scene):
        timeline = self._state.timeline
        self.setVisible(timeline is not None)
        if timeline is not None:
            # Real time (1×), as in KSP: warp up with "." or the ladder.
            self.set_warp(1.0)
        self._fit_height()
        self.event_label.setText("")
        self._refresh_readouts()

    def _on_time_changed(self, _t: float):
        self._refresh_readouts()

    def _on_view_changed(self):
        w = self.track.window
        self.zoom_label.setText("" if w is None else format_view_span(w.span * self.unit_seconds))
        if w is not None:
            self.zoom_in_btn.setEnabled(w.span > w.min_span * (1 + 1e-9))
            self.zoom_out_btn.setEnabled(not w.is_fit)
            self.fit_btn.setEnabled(not w.is_fit)

    def met_text(self) -> str:
        timeline = self._state.timeline
        if not timeline:
            return ""
        w = self.track.window
        res = 1.0 if w is None else w.span * self.unit_seconds / self.track.track_width
        return format_met(self._state.time * self.unit_seconds, res)

    def _refresh_readouts(self):
        timeline = self._state.timeline
        if not timeline:
            return
        t = self._state.time
        # Don't overwrite what's being typed. isModified() is set only by
        # typing (setText clears it): focus alone isn't enough, a widget can
        # hold focus without anyone editing (it did on the macOS CI runners).
        if not (self.met_edit.hasFocus() and self.met_edit.isModified()):
            self.met_edit.setText(self.met_text())
        ref_et = timeline.get("referenceEpochEt")
        # Cleared, not left alone, for a timeline with no calendar epoch:
        # otherwise the previous mission's date stayed on screen.
        self.date_label.setText("" if ref_et is None else epoch_to_date_string(ref_et, t / 86400.0 * self.unit_seconds))
        current = self._state.active_event()
        self.event_label.setText(f"▸ {current['label']}" if current else "")

    def _on_met_entered(self):
        timeline = self._state.timeline
        if not timeline:
            return
        unit_s = self.unit_seconds
        seconds = parse_time_input(self.met_edit.text(), self._state.time * unit_s, unit_s,
                                   timeline.get("referenceEpochEt"))
        if seconds is None:
            self.met_edit.setProperty("invalid", True)
        else:
            self.met_edit.setProperty("invalid", False)
            self._state.set_time(seconds / unit_s)
            self.met_edit.clearFocus()
        self.met_edit.style().unpolish(self.met_edit)
        self.met_edit.style().polish(self.met_edit)
        self._refresh_readouts()

    def focus_time_box(self):
        self.met_edit.setFocus(Qt.ShortcutFocusReason)
        self.met_edit.selectAll()

    def step_size(self) -> float:
        """The ←/→ step: 1% of the visible span, so it gets finer as you zoom."""
        w = self.track.window
        return 0.0 if w is None else 0.01 * w.span

    # -------------------------------------------------------- playback
    def set_playing(self, playing: bool, direction: int = 1):
        """Pause icon on the button of the direction that is playing."""
        forward, backward = playing and direction > 0, playing and direction < 0
        self.play_btn.setIcon(icons.pause_icon() if forward else icons.play_icon())
        self.reverse_btn.setIcon(icons.pause_icon() if backward
                                 else icons.play_icon(INK_SECONDARY, reverse=True))

    @property
    def warp(self) -> float:
        """The user-chosen time warp: simulated seconds per real second (1 = real time)."""
        return self._warp

    @property
    def auto_slow_enabled(self) -> bool:
        return self.auto_slow_btn.isChecked()

    def set_warp(self, warp: float):
        self._warp = min(WARP_MAX, max(WARP_MIN, float(warp)))
        self.warp_ladder.set_warp(self._warp)
        self.real_time_btn.setChecked(abs(self._warp - 1.0) < 1e-9)
        self.warp_down_btn.setEnabled(self._warp > WARP_LEVELS[0] * (1 + 1e-9))
        self.warp_up_btn.setEnabled(self._warp < WARP_LEVELS[-1] * (1 - 1e-9))
        self.show_effective_speed(self._warp)

    def nudge_speed(self, direction: int):
        """Fine multiplicative nudge of the warp (the [ / ] shortcuts)."""
        self.set_warp(self._warp * (1.25 ** direction))

    def step_warp(self, direction: int):
        """Jump to the next KSP warp level up or down (the . / , shortcuts)."""
        self.set_warp(step_warp(self._warp, direction))

    def show_effective_speed(self, effective_speed: float):
        """Keep the TIME WARP label showing what's *actually* playing back,
        not just the slider-set speed -- without this the label kept reading
        (e.g.) "10x" while auto-slow had actually eased playback down to 0.02x
        near an event, which looked like the readout and the real playback
        rate had drifted out of sync with each other."""
        eased = effective_speed < self._warp * (1 - 1e-9)
        self.speed_value_label.setText(f"{format_warp(effective_speed)} · {format_rate(effective_speed)}")
        self.speed_value_label.setProperty("eased", eased)
        timeline = self._state.timeline
        tip = "Time warp: simulated time per real second (1× = real time)"
        if timeline:
            span_s = (timeline["max"] - timeline["min"]) * self.unit_seconds
            tip += f"\nWhole mission at this warp: {format_view_span(span_s / effective_speed)}"
        if eased:
            tip += f"\nEased down near an event (auto-slow); set to {format_warp(self._warp)}"
        self.speed_value_label.setToolTip(tip)
        self.speed_value_label.style().unpolish(self.speed_value_label)
        self.speed_value_label.style().polish(self.speed_value_label)

    def _on_auto_slow_toggled(self, on: bool):
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT if on else INK_MUTED))
        self.auto_slow_toggled.emit(bool(on))
