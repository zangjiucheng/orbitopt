"""Small reusable chrome widgets shared by the pixel-mission-control UI.

Kept as one module (not one file per widget) because there are only two of
them and they're each a handful of lines -- this project's existing
convention is a few cohesive modules (icons.py bundles a dozen icon
functions the same way), not one file per tiny class.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from orbitopt.viz.theme import DANGER, INFO, INK_MUTED, SUCCESS, WARNING

# Every StatusBadge state maps to exactly one semantic color -- never an
# arbitrary per-badge choice (see design brief: "颜色必须具有语义").
_STATE_COLORS = {
    "idle": INK_MUTED,
    "loading": INFO,
    "ready": SUCCESS,
    "verified": SUCCESS,
    "warning": WARNING,
    "failed": DANGER,
}
_PULSE_STATES = {"loading"}
_PULSE_INTERVAL_MS = 600  # a slow, single on/off blink -- not a continuous animation


class StatusBadge(QWidget):
    """A small state dot + uppercase label, e.g. "● READY". Text is always
    shown alongside the color (never color alone) so state reads correctly
    for anyone who can't distinguish the color (design brief section 13).
    The dot pulses gently only while in a "loading"-like state, and the
    timer driving that is stopped the rest of the time -- no idle repaint
    cost for a badge that's just sitting at READY/FAILED/etc."""

    _DOT = 7

    def __init__(self, state: str = "idle", text: str | None = None, parent=None):
        super().__init__(parent)
        self._state = state
        self._pulse_on = True
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self._dot = QWidget()
        self._dot.setFixedSize(self._DOT, self._DOT)
        self._label = QLabel()
        self._label.setObjectName("statusBadgeText")
        layout.addWidget(self._dot, alignment=Qt.AlignVCenter)
        layout.addWidget(self._label, alignment=Qt.AlignVCenter)

        self._timer = QTimer(self)
        self._timer.setInterval(_PULSE_INTERVAL_MS)
        self._timer.timeout.connect(self._on_pulse)

        self._dot.paintEvent = self._paint_dot
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        self._state = state
        self._label.setText((text or state).upper())
        self._label.setStyleSheet(f"color: {_STATE_COLORS.get(state, INK_MUTED)};")
        if state in _PULSE_STATES:
            if not self._timer.isActive():
                self._pulse_on = True
                self._timer.start()
        else:
            self._timer.stop()
            self._pulse_on = True
        self._dot.update()

    def _on_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self._dot.update()

    def _paint_dot(self, _event) -> None:
        painter = QPainter(self._dot)
        color = QColor(_STATE_COLORS.get(self._state, INK_MUTED))
        if self._state in _PULSE_STATES and not self._pulse_on:
            color.setAlpha(90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRect(0, 0, self._DOT, self._DOT)
        painter.end()


class MetricRow(QWidget):
    """One "LABEL ... VALUE UNIT" line -- the same shape as the body/maneuver
    stat rows already hand-built in app.py's info panel, formalized here for
    the new compute-status cluster so that one isn't duplicating ad hoc
    QHBoxLayout code a third time."""

    def __init__(self, label: str, value: str = "—", unit: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._label = QLabel(label.upper())
        self._label.setObjectName("metricLabel")
        self._value = QLabel(value)
        self._value.setObjectName("metricValue")
        self._unit = QLabel(unit)
        self._unit.setObjectName("metricUnit")
        layout.addWidget(self._label)
        layout.addStretch(1)
        layout.addWidget(self._value)
        layout.addWidget(self._unit)

    def set_value(self, value: str, unit: str | None = None) -> None:
        self._value.setText(value)
        if unit is not None:
            self._unit.setText(unit)
