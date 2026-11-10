"""The mission timeline strip under the 3D view: play/pause, the time-warp
rate, and the event-marked scrubber with its MET / date / active-event
readouts. It reads and writes the current time through MissionState only, so
anything else that moves time (playback, a maneuver click, a keyboard step)
moves this bar too without the window having to update it by hand."""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz import icons
from orbitopt.viz.state import MissionState, time_to_fraction
from orbitopt.viz.theme import (
    ACCENT,
    ACCENT_DIM,
    HAIRLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PANEL_RAISED,
    SURFACE_HOVER,
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
DEFAULT_PLAYBACK_REAL_S = 60.0  # a scene's default warp plays its whole timeline in about this long

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


def default_warp(span_seconds: float, playback_real_s: float = DEFAULT_PLAYBACK_REAL_S) -> float:
    """The warp level (nearest on a log scale) that plays ``span_seconds`` of
    timeline in about ``playback_real_s`` real seconds."""
    target = max(span_seconds, 1e-9) / playback_real_s
    return min(WARP_LEVELS[1:], key=lambda w: abs(math.log(w) - math.log(target)))


# Resolution of the time scrubber (slider steps across the whole timeline).
TIME_SLIDER_STEPS = 10000


def epoch_to_date_string(reference_et: float, day_offset: float) -> str:
    from orbitopt.timescales import tdb_seconds_to_utc_datetime

    dt = tdb_seconds_to_utc_datetime(reference_et + day_offset * 86400.0)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


class EventSlider(QSlider):
    """The timeline scrubber, with a marker painted over the groove at each
    mission event (burns, flybys, insertions). Hovering near a marker shows
    every event within reach (not just one -- events that land within a few
    pixels of each other, e.g. a burn immediately followed by an SOI-exit
    note, used to be visually indistinguishable and only the first-added one
    was ever reachable by click); clicking snaps to whichever is nearest.
    Falls back to a plain slider when the scene has no timeline events."""

    _HIT_TOLERANCE_PX = 7
    # A normal marker's own triangle is ~6px wide (see paintEvent's `half`),
    # so anything closer than that already visually overlaps -- clustering
    # only below 3px (an earlier value here) left genuinely-touching markers
    # rendered as two separate, ambiguous shapes instead of one clear
    # cluster mark. Confirmed empirically against a real rendered frame, not
    # assumed: the Mars mission's Mars-SOI-entry/MOI events, ~4.8px apart in
    # practice, looked like touching-but-unmerged marks at the old value.
    _CLUSTER_TOLERANCE_PX = 6

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._events = []  # list of (slider_value:int, label:str, note:str)
        self.setMouseTracking(True)

    def set_events(self, events, t_min, t_max):
        self._events = []
        span = (t_max - t_min) or 1.0
        lo, hi = self.minimum(), self.maximum()
        for e in events:
            frac = min(1.0, max(0.0, (e.get("time", t_min) - t_min) / span))
            self._events.append((int(round(lo + frac * (hi - lo))),
                                 e.get("label", ""), e.get("note", "")))
        self.update()

    def _groove(self):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

    def _value_to_x(self, value):
        groove = self._groove()
        span = (self.maximum() - self.minimum()) or 1
        return groove.left() + int((value - self.minimum()) / span * groove.width())

    def _events_near(self, x, tol):
        """Every event within `tol` px of `x`, nearest first -- not just the
        first one added, so a click/hover near a cluster of close-together
        events reaches all of them instead of being stuck on whichever
        happened to be first in the scene's event list."""
        near = sorted(
            ((abs(self._value_to_x(ev[0]) - x), ev) for ev in self._events),
            key=lambda pair: pair[0],
        )
        return [ev for dist, ev in near if dist <= tol]

    def _event_near(self, x, tol=None):
        near = self._events_near(x, self._HIT_TOLERANCE_PX if tol is None else tol)
        return near[0] if near else None

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if not self._events:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() // 2

        # Bucket by rounded pixel position so events within
        # _CLUSTER_TOLERANCE_PX draw as one bolder cluster mark instead of
        # silently overlapping into what looks like (and, before this fix,
        # behaved as) a single ordinary event.
        buckets: dict[int, list] = {}
        for value, label, note in self._events:
            x = self._value_to_x(value)
            buckets.setdefault(round(x / self._CLUSTER_TOLERANCE_PX), []).append(x)
        centers = {key: round(sum(xs) / len(xs)) for key, xs in buckets.items()}
        counts = {key: len(xs) for key, xs in buckets.items()}

        for key, x in centers.items():
            clustered = counts[key] > 1
            painter.setPen(QPen(QColor(ACCENT), 1.8 if clustered else 1.4))
            painter.setBrush(QColor(ACCENT))
            painter.drawLine(x, 7, x, cy + 2)
            half = 4.2 if clustered else 3.0
            tri = QPainterPath()
            tri.moveTo(x - half, 1.0)
            tri.lineTo(x + half, 1.0)
            tri.lineTo(x, 7.0)
            tri.closeSubpath()
            painter.drawPath(tri)
            if clustered:
                # A small dot below the marker's guide line signals "more
                # than one event here" at a glance, without needing to hover
                # first. Below, not above: the widget's actual rendered
                # height is ~16px (QSlider::handle's 15px + a hair), so a
                # mark above y=0 is silently clipped by Qt's paint-event
                # clip rect -- confirmed empirically by grabbing a real
                # rendered frame and finding zero accent-colored pixels in
                # that band, not assumed from the geometry alone.
                painter.drawEllipse(QRectF(x - 1.6, cy + 4.0, 3.2, 3.2))
        painter.end()

    def mouseMoveEvent(self, event):  # noqa: N802
        near = self._events_near(int(event.position().x()), self._HIT_TOLERANCE_PX)
        text = "\n\n".join(f"{label}\n{note}".strip() for _value, label, note in near)
        self.setToolTip(text)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        ev = self._event_near(int(event.position().x()))
        if ev is not None:
            self.setValue(ev[0])  # snap to the nearest event
            return
        super().mousePressEvent(event)



class PhaseStrip(QWidget):
    """The scene's mission phases as bands under the scrubber, on the same x
    scale as its groove: the current phase in the accent tint, the others in
    two alternating neutrals (phases differ per mission, so they get no
    categorical colors). Hover names the span; click jumps to its start."""

    HEIGHT = 18

    def __init__(self, slider: QSlider, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._slider = slider
        self._state = state
        self._phases: list[dict] = []
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        state.scene_changed.connect(self._on_scene_changed)
        state.time_changed.connect(lambda _t: self.update())

    @property
    def phases(self) -> list[dict]:
        return self._phases

    def _on_scene_changed(self, _scene):
        timeline = self._state.timeline
        self._phases = list((timeline or {}).get("phases", []))
        self.setVisible(bool(self._phases))
        self.update()

    def _x(self, t: float) -> float:
        groove = self._slider._groove()
        # The slider is a sibling, not an ancestor: go through global
        # coordinates (QWidget.mapTo needs a parent).
        origin = self.mapFromGlobal(self._slider.mapToGlobal(groove.topLeft())).x()
        return origin + time_to_fraction(self._state.timeline, t) * groove.width()

    def current_phase(self) -> dict | None:
        t = self._state.time
        return next((p for p in self._phases if p["start"] <= t <= p["end"]), None)

    def phase_at(self, x: float) -> dict | None:
        return next((p for p in self._phases if self._x(p["start"]) <= x <= self._x(p["end"])), None)

    def paintEvent(self, _event):  # noqa: N802
        if not self._phases or self._state.timeline is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        current = self.current_phase()
        for k, phase in enumerate(self._phases):
            x0, x1 = self._x(phase["start"]), self._x(phase["end"])
            rect = QRectF(x0, 2.0, max(1.0, x1 - x0), self.height() - 4.0)
            active = phase is current
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(ACCENT_DIM if active else (PANEL_RAISED if k % 2 == 0 else SURFACE_HOVER)))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor(HAIRLINE), 1.0))
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
            text = painter.fontMetrics().elidedText(phase["label"], Qt.ElideRight, int(rect.width() - 8))
            if text and text != "…":
                painter.setPen(QColor(INK_PRIMARY if active else INK_SECONDARY))
                painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.end()

    def mouseMoveEvent(self, event):  # noqa: N802
        phase = self.phase_at(event.position().x())
        if phase is None:
            self.setToolTip("")
        else:
            unit = self._state.timeline.get("unitLabel", "")
            tip = f"{phase['label']}\nT+{phase['start']:.2f} – T+{phase['end']:.2f} {unit}"
            if phase.get("note"):
                tip += f"\n{phase['note']}"
            self.setToolTip(tip + "\nClick to jump to its start")
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        phase = self.phase_at(event.position().x())
        if phase is not None:
            self._state.set_time(phase["start"])


class TimelineBar(QWidget):
    """Play button, time-warp control and the event-marked scrubber.

    Time flows one way: the scrubber writes to ``state`` when the *user*
    drags it, and follows ``state.time_changed`` otherwise (with its own
    signals blocked, so following never writes back). The bar owns the
    time-warp setting (``speed``) and the auto-slow toggle; the playback
    timer that consumes them lives in the window.
    """

    play_clicked = Signal()
    auto_slow_toggled = Signal(bool)

    _BASE_HEIGHT = 72

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._warp = 1_000.0  # replaced per scene by default_warp() in _on_scene_changed
        self.setObjectName("timelineBar")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(self._BASE_HEIGHT)
        self._build()
        state.scene_changed.connect(self._on_scene_changed)
        state.time_changed.connect(self._on_time_changed)

    # ------------------------------------------------------------- build
    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        self.play_btn = QPushButton()
        self.play_btn.setObjectName("playButton")
        self.play_btn.setIcon(icons.play_icon())
        self.play_btn.setIconSize(QSize(15, 15))
        self.play_btn.clicked.connect(self.play_clicked)
        layout.addWidget(self.play_btn)

        speed_col = QVBoxLayout()
        speed_col.setSpacing(2)
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

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setObjectName("speedSlider")
        self.speed_slider.setRange(0, WARP_SLIDER_STEPS)
        self.speed_slider.setValue(slider_from_warp(self._warp))
        self.speed_slider.setToolTip(
            "Time warp, 0.1x to 10,000,000x (1x = real time). "
            ", / . step through the KSP warp levels, / returns to 1x, [ / ] fine-tune"
        )
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)
        speed_col.addWidget(self.speed_slider)
        layout.addLayout(speed_col)

        slider_col = QVBoxLayout()
        slider_col.setSpacing(2)
        readout_row = QHBoxLayout()
        self.met_label = QLabel("T+0.00")
        self.met_label.setObjectName("metReadout")
        self.date_label = QLabel("")
        self.date_label.setObjectName("dateReadout")
        self.event_label = QLabel("")
        self.event_label.setObjectName("eventReadout")
        readout_row.addWidget(self.met_label)
        readout_row.addWidget(self.date_label)
        readout_row.addStretch(1)
        readout_row.addWidget(self.event_label)
        slider_col.addLayout(readout_row)

        self.time_slider = EventSlider()
        self.time_slider.setRange(0, TIME_SLIDER_STEPS)
        self.time_slider.valueChanged.connect(self._on_time_slider)
        slider_col.addWidget(self.time_slider)

        self.phase_strip = PhaseStrip(self.time_slider, self._state)
        slider_col.addWidget(self.phase_strip)

        # Fixed start/end scale reference under the bar: the MET readout above
        # only ever shows the *current* scrubbed position, so there was no way
        # to tell at a glance how long the whole mission span is -- easy to
        # misjudge for a mission like Mars (300+ days) where nearly every event
        # piles up within the first/last ~1% of the bar.
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        self.scale_start_label = QLabel("")
        self.scale_start_label.setObjectName("scaleReadout")
        self.scale_end_label = QLabel("")
        self.scale_end_label.setObjectName("scaleReadout")
        self.scale_end_label.setAlignment(Qt.AlignRight)
        scale_row.addWidget(self.scale_start_label)
        scale_row.addStretch(1)
        scale_row.addWidget(self.scale_end_label)
        slider_col.addLayout(scale_row)

        layout.addLayout(slider_col, stretch=1)

    # ------------------------------------------------------- state sync
    def _on_scene_changed(self, _scene):
        timeline = self._state.timeline
        self.setVisible(timeline is not None)
        if timeline is not None:
            span_s = (timeline["max"] - timeline["min"]) * timeline_unit_seconds(timeline.get("unitLabel"))
            self.set_warp(default_warp(span_s))
        has_phases = bool((timeline or {}).get("phases"))
        self.setFixedHeight(self._BASE_HEIGHT + (PhaseStrip.HEIGHT + 2 if has_phases else 0))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(0)
        if timeline:
            self.time_slider.set_events(self._state.events, timeline["min"], timeline["max"])
            unit = timeline.get("unitLabel", "")
            self.scale_start_label.setText(f"{timeline['min']:.1f} {unit}".strip())
            self.scale_end_label.setText(f"{timeline['max']:.1f} {unit}".strip())
        else:
            self.time_slider.set_events([], 0.0, 1.0)
            self.scale_start_label.setText("")
            self.scale_end_label.setText("")
        self.time_slider.blockSignals(False)
        self.event_label.setText("")
        self._refresh_readouts()

    def _on_time_changed(self, t: float):
        timeline = self._state.timeline
        if timeline is None:
            return
        value = int(round(time_to_fraction(timeline, t) * self.time_slider.maximum()))
        if value != self.time_slider.value():
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(value)
            self.time_slider.blockSignals(False)
        self._refresh_readouts()

    def _on_time_slider(self, value: int):
        # The user dragged (or clicked an event marker on) the scrubber.
        self._state.set_fraction(value / self.time_slider.maximum())

    def _refresh_readouts(self):
        timeline = self._state.timeline
        if not timeline:
            return
        t = self._state.time
        self.met_label.setText(f"T+{t:.2f} {timeline['unitLabel']}")
        ref_et = timeline.get("referenceEpochEt")
        # Cleared, not left alone, for a timeline with no calendar epoch:
        # otherwise the previous mission's date stayed on screen.
        self.date_label.setText("" if ref_et is None else epoch_to_date_string(ref_et, t))
        current = self._state.active_event()
        self.event_label.setText(f"▸ {current['label']}" if current else "")

    # -------------------------------------------------------- playback
    def set_playing(self, playing: bool):
        self.play_btn.setIcon(icons.pause_icon() if playing else icons.play_icon())

    @property
    def warp(self) -> float:
        """The user-chosen time warp: simulated seconds per real second (1 = real time)."""
        return self._warp

    @property
    def auto_slow_enabled(self) -> bool:
        return self.auto_slow_btn.isChecked()

    def set_warp(self, warp: float):
        self._warp = min(WARP_MAX, max(WARP_MIN, float(warp)))
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(slider_from_warp(self._warp))
        self.speed_slider.blockSignals(False)
        self.show_effective_speed(self._warp)

    def nudge_speed(self, direction: int):
        """Fine multiplicative nudge of the warp (the [ / ] shortcuts)."""
        self.set_warp(self._warp * (1.25 ** direction))

    def step_warp(self, direction: int):
        """Jump to the next KSP warp level up or down (the . / , shortcuts)."""
        self.set_warp(step_warp(self._warp, direction))

    def _on_speed_slider(self, value: int):
        self._warp = warp_from_slider(value)
        self.show_effective_speed(self._warp)

    def show_effective_speed(self, effective_speed: float):
        """Keep the TIME WARP label showing what's *actually* playing back,
        not just the slider-set speed -- without this the label kept reading
        (e.g.) "10x" while auto-slow had actually eased playback down to 0.02x
        near an event, which looked like the readout and the real playback
        rate had drifted out of sync with each other."""
        eased = effective_speed < self._warp * (1 - 1e-9)
        self.speed_value_label.setText(f"{format_warp(effective_speed)} · {format_rate(effective_speed)}")
        self.speed_value_label.setProperty("eased", eased)
        self.speed_value_label.style().unpolish(self.speed_value_label)
        self.speed_value_label.style().polish(self.speed_value_label)

    def _on_auto_slow_toggled(self, on: bool):
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT if on else INK_MUTED))
        self.auto_slow_toggled.emit(bool(on))
