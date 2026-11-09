"""Small self-contained Qt widgets used by Mission Control's chrome."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QWidget

from orbitopt.viz.theme import ACCENT


class SpinnerWidget(QWidget):
    """A rotating-arc spinner painted with QPainter and driven by a QTimer.

    It animates while the Qt event loop is free -- i.e. throughout the ~10s
    background scene *computation* (which runs on a QThread, see SceneLoader)
    -- and simply holds its last painted frame during the short synchronous
    VTK scene rebuild that follows (SceneRenderer.load has to run on the main
    thread). That rebuild is the brief stutter the user reported; showing this
    over the view for the whole switch is the "loading effect" that makes the
    stutter read as deliberate progress rather than a freeze.
    """

    def __init__(self, parent=None, diameter: int = 64, color: str = ACCENT):
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self.setFixedSize(diameter, diameter)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def _advance(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802 -- Qt override signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        margin = 5
        rect = QRectF(margin, margin, self.width() - 2 * margin, self.height() - 2 * margin)

        track = QPen(QColor(self._color.red(), self._color.green(), self._color.blue(), 40), 4)
        track.setCapStyle(Qt.RoundCap)
        painter.setPen(track)
        painter.drawArc(rect, 0, 360 * 16)

        arc = QPen(self._color, 4)
        arc.setCapStyle(Qt.RoundCap)
        painter.setPen(arc)
        painter.drawArc(rect, -self._angle * 16, 100 * 16)


class ViewControlsPopover(QFrame):
    """Compact popover for the less-frequently-used view settings (see
    MissionControlWindow._build_view_controls_popover). A plain Qt.Popup
    window rather than a QMenu with QWidgetAction rows: a QComboBox (the
    face-lock control) embedded in a QMenu has flaky click-through behavior
    on some platforms, where a normal popup window doesn't. Qt.Popup already
    closes on any click outside it; Escape is wired explicitly below since
    Qt.Popup doesn't bind that key on its own (design brief: "支持 Escape 关闭")."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setObjectName("viewControlsPopover")

    def keyPressEvent(self, event):  # noqa: N802 -- Qt override signature
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)
