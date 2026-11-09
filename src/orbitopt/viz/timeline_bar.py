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
from orbitopt.viz.theme import ACCENT, INK_MUTED

# Time-warp speed is a continuous log-scale control (0.001x .. 1000x), not
# fixed presets, so it can be fine-tuned to any rate. Speed is in simulated
# *days* per real second (see MissionControlWindow._on_tick), so even the old
# floor of 0.1x still meant 1 simulated hour flew by in under half a real
# second -- too fast to watch a close lunar flyby unfold. At the current
# floor, 0.001x, 1 simulated hour takes ~41.7 real seconds and 1 simulated
# minute takes <1 real second, actually slow enough to watch closely.
SPEED_MIN = 0.001
SPEED_MAX = 1000.0
SPEED_SLIDER_STEPS = 1000

# Resolution of the time scrubber (slider steps across the whole timeline).
TIME_SLIDER_STEPS = 10000


def format_speed(speed: float) -> str:
    if speed >= 10:
        return f"{speed:.0f}×"
    if speed >= 1:
        return f"{speed:.1f}×"
    # Below 1x: a fixed 2 decimals reads fine down to 0.1, but prints as
    # "0.00×" for anything under 0.005 -- indistinguishable from stopped now
    # that SPEED_MIN goes as low as 0.001. Scale the decimal count to keep ~2
    # significant figures visible at any speed below 1x instead.
    decimals = max(2, -int(math.floor(math.log10(speed))) + 1)
    return f"{speed:.{decimals}f}×"


def speed_from_slider(value: int) -> float:
    frac = value / SPEED_SLIDER_STEPS
    return SPEED_MIN * (SPEED_MAX / SPEED_MIN) ** frac


def slider_from_speed(speed: float) -> int:
    speed = min(SPEED_MAX, max(SPEED_MIN, speed))
    frac = math.log(speed / SPEED_MIN) / math.log(SPEED_MAX / SPEED_MIN)
    return int(round(frac * SPEED_SLIDER_STEPS))


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

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._speed = 1.0
        self.setObjectName("timelineBar")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(72)
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
        self.speed_value_label = QLabel(format_speed(self._speed))
        self.speed_value_label.setObjectName("speedValue")
        speed_header.addWidget(self.speed_value_label)
        speed_col.addLayout(speed_header)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setObjectName("speedSlider")
        self.speed_slider.setRange(0, SPEED_SLIDER_STEPS)
        self.speed_slider.setValue(slider_from_speed(self._speed))
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
        if ref_et is not None:
            self.date_label.setText(epoch_to_date_string(ref_et, t))
        current = self._state.active_event()
        self.event_label.setText(f"▸ {current['label']}" if current else "")

    # -------------------------------------------------------- playback
    def set_playing(self, playing: bool):
        self.play_btn.setIcon(icons.pause_icon() if playing else icons.play_icon())

    @property
    def speed(self) -> float:
        """The user-chosen time-warp rate, in simulated days per real second."""
        return self._speed

    @property
    def auto_slow_enabled(self) -> bool:
        return self.auto_slow_btn.isChecked()

    def nudge_speed(self, direction: int):
        """Fine multiplicative nudge of the time-warp rate (the [ / ] shortcuts);
        a continuous log step, not a jump between fixed presets."""
        self._speed = min(SPEED_MAX, max(SPEED_MIN, self._speed * (1.25 ** direction)))
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(slider_from_speed(self._speed))
        self.speed_slider.blockSignals(False)
        self.show_effective_speed(self._speed)

    def _on_speed_slider(self, value: int):
        self._speed = speed_from_slider(value)
        self.show_effective_speed(self._speed)

    def show_effective_speed(self, effective_speed: float):
        """Keep the TIME WARP label showing what's *actually* playing back,
        not just the slider-set speed -- without this the label kept reading
        (e.g.) "10x" while auto-slow had actually eased playback down to 0.02x
        near an event, which looked like the readout and the real playback
        rate had drifted out of sync with each other."""
        eased = effective_speed < self._speed - 1e-9
        self.speed_value_label.setText(format_speed(effective_speed))
        self.speed_value_label.setProperty("eased", eased)
        self.speed_value_label.style().unpolish(self.speed_value_label)
        self.speed_value_label.style().polish(self.speed_value_label)

    def _on_auto_slow_toggled(self, on: bool):
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT if on else INK_MUTED))
        self.auto_slow_toggled.emit(bool(on))
