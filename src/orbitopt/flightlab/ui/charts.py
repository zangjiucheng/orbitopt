"""Shared QPainter chart widgets for the Flight Lab pages (FL-501/FL-503).

The synchronized telemetry chart started life in the replay page; the
compare view overlays two runs on the same widget, so it lives here, free of
the pyvista/VTK imports the replay page's 3D view needs.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from orbitopt.flightlab.ui.telemetry import TelemetryChannel
from orbitopt.viz.theme import HAIRLINE, INFO, INK_MUTED, INK_SECONDARY, PANEL

__all__ = ["TelemetryChart", "fmt_clock", "fmt_number", "nice_ticks", "qfont"]


def _font(size: float, bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


def _nice_ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, target)
    if raw <= 0 or not math.isfinite(raw):
        return [lo]
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    return [start + k * step for k in range(int((hi - start) / step + 1e-9) + 1)]


def _fmt_number(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 10_000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:,.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.001:
        return f"{value:.3f}"
    return "0" if value == 0 else f"{value:.2e}"


def _fmt_clock(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "—"
    seconds = float(seconds)
    if abs(seconds) < 90:
        return f"{seconds:.1f} s"
    if abs(seconds) < 7200:
        return f"{seconds / 60:.1f} min"
    if abs(seconds) < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def _slider_resolution(seconds: float) -> float:
    if seconds <= 600:
        return 1000.0
    if seconds <= 6000:
        return 100.0
    if seconds <= 60000:
        return 10.0
    return 1.0


class _TelemetryChart(QWidget):
    """One synchronized channel chart.

    Static pixels are cached the same way the app's other QPainter charts do:
    the data lines do not change on every playback tick, only the cursor does.
    Click/drag requests a seek so scrubbing a chart keeps the 3D view, event
    highlight and all other channels at the same time.
    """

    time_requested = Signal(float)

    MARGINS = (66, 26, 18, 28)
    _TITLE_X = 8.0
    legend_placement: str | None = None  # "header" | "inside" | None, set when painted

    def __init__(self, channel_id: str, label: str, unit: str, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self._channel = TelemetryChannel(channel_id, label, unit, "waiting for a Flight Lab run")
        self._xmin = 0.0
        self._xmax = 1.0
        self._ymin = 0.0
        self._ymax = 1.0
        self._time = 0.0
        self._dragging = False
        self._cache: QPixmap | None = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(120, 110)

    def set_channel(self, channel: TelemetryChannel) -> None:
        self._channel = channel
        self._recompute_y_range()
        self.invalidate()

    def set_time_range(self, lo: float, hi: float) -> None:
        self._xmin = float(lo)
        self._xmax = float(hi)
        self.invalidate()

    def clear(self) -> None:
        self.set_channel(TelemetryChannel(self.channel_id, self._label(), self._unit(), "waiting for a Flight Lab run"))

    def set_time(self, t: float) -> None:
        self._time = float(t)
        self.update()

    def invalidate(self) -> None:
        self._cache = None
        self.update()

    def _label(self) -> str:
        return self._channel.label

    def _unit(self) -> str:
        return self._channel.unit

    def resizeEvent(self, event):  # noqa: N802
        self._cache = None
        super().resizeEvent(event)

    def plot_rect(self) -> QRectF:
        left, top, right, bottom = self.MARGINS
        return QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))

    def _recompute_y_range(self) -> None:
        values = [value for series in self._channel.series for _, value in series.points if math.isfinite(value)]
        if not values:
            self._ymin, self._ymax = 0.0, 1.0
            return
        lo, hi = min(values), max(values)
        if hi <= lo:
            pad = max(1.0, abs(lo) * 0.05)
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.06
            lo, hi = lo - pad, hi + pad
        self._ymin, self._ymax = lo, hi

    def _x(self, t: float) -> float:
        rect = self.plot_rect()
        if self._xmax <= self._xmin:
            return rect.left() + rect.width() / 2.0
        fraction = min(1.0, max(0.0, (float(t) - self._xmin) / (self._xmax - self._xmin)))
        return rect.left() + rect.width() * fraction

    def _y(self, value: float) -> float:
        rect = self.plot_rect()
        if self._ymax <= self._ymin:
            return rect.bottom()
        fraction = min(1.0, max(0.0, (float(value) - self._ymin) / (self._ymax - self._ymin)))
        return rect.bottom() - rect.height() * fraction

    def _time_at_x(self, x: float) -> float:
        rect = self.plot_rect()
        if rect.width() <= 0 or self._xmax <= self._xmin:
            return self._xmin
        fraction = min(1.0, max(0.0, (float(x) - rect.left()) / rect.width()))
        return self._xmin + fraction * (self._xmax - self._xmin)

    def paintEvent(self, _event):  # noqa: N802
        ratio = self.devicePixelRatioF()
        pixmap_width = max(1, int(round(self.width() * ratio)))
        pixmap_height = max(1, int(round(self.height() * ratio)))
        if self._cache is None or self._cache.width() != pixmap_width or self._cache.height() != pixmap_height:
            pixmap = QPixmap(QSize(pixmap_width, pixmap_height))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(QColor(PANEL))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            self._paint_static(painter)
            painter.end()
            self._cache = pixmap

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_dynamic(painter)
        painter.end()

    def _paint_static(self, p: QPainter) -> None:
        rect = self.plot_rect()
        # The title owns the header row from the widget's left edge (the
        # y-axis labels are below it): a fixed box ``rect.width() - 90`` wide
        # cut narrow charts' titles to "Altitud" / "Dynam" / "Mass (".
        p.setFont(_font(10, True))
        p.setPen(QColor(INK_SECONDARY))
        title = f"{self._channel.label} ({self._channel.unit})"
        title_box = QRectF(self._TITLE_X, 3, max(1.0, self.width() - 2 * self._TITLE_X), 18)
        shown = p.fontMetrics().elidedText(title, Qt.ElideRight, int(title_box.width()))
        p.drawText(title_box, Qt.AlignLeft | Qt.AlignVCenter, shown)
        title_right = self._TITLE_X + p.fontMetrics().horizontalAdvance(shown) + 10
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)

        if not self._channel.has_data:
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(10))
            p.drawText(rect, Qt.AlignCenter, self._channel.missing_reason or "no telemetry")
            return

        for yv in _nice_ticks(self._ymin, self._ymax):
            py = self._y(yv)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(8.5))
            p.drawText(QRectF(0, py - 8, self.MARGINS[0] - 6, 16), Qt.AlignRight | Qt.AlignVCenter, _fmt_number(yv))

        for tv in _nice_ticks(self._xmin, self._xmax):
            px = self._x(tv)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(8.5))
            p.drawText(QRectF(px - 35, rect.bottom() + 4, 70, 16), Qt.AlignCenter, _fmt_clock(tv))

        for series in self._channel.series:
            points = series.points
            if not points:
                continue
            color = QColor(series.color)
            if len(points) == 1:
                px, py = self._x(points[0][0]), self._y(points[0][1])
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(color))
                p.drawRect(QRectF(px - 2.0, py - 2.0, 4.0, 4.0))
                continue
            max_samples = max(2, int(rect.width() * 1.5))
            stride = max(1, len(points) // max_samples)
            samples = list(points[::stride])
            if samples[-1] != points[-1]:
                samples.append(points[-1])
            line = Qt.DashLine if getattr(series, "style", "solid") == "dash" else Qt.SolidLine
            p.setPen(QPen(color, 1.8, line, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(QPolygonF([QPointF(self._x(t), self._y(v)) for t, v in samples]))
        self._paint_legend(p, rect, title_right)  # over the lines, so an inside box stays legible

    def _paint_legend(self, p: QPainter, rect: QRectF, min_x: float = 0.0) -> None:
        """Every labelled series, or none: beside the title when the whole
        legend fits there, else in a box inside the plot's top-right corner.
        It used to paint from the right only what fitted beside the title,
        so a four-body chart could be labelled "Booster" alone."""
        series = [item for item in self._channel.series if item.label]
        self.legend_placement = None
        if not series:
            return
        p.setFont(_font(8.5))
        fm = p.fontMetrics()
        widths = [fm.horizontalAdvance(item.label) + 2 for item in series]
        if rect.right() - 2.0 - sum(w + 22 for w in widths) >= min_x:
            self.legend_placement = "header"
            x = rect.right() - 2.0
            for entry, text_width in zip(reversed(series), reversed(widths)):
                x -= text_width
                p.setPen(QColor(entry.color))
                p.drawText(QRectF(x, 3, text_width, 18), Qt.AlignLeft | Qt.AlignVCenter, entry.label)
                x -= 12
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(entry.color)))
                p.drawRect(QRectF(x, 9, 7, 7))
                x -= 10
            return
        self.legend_placement = "inside"
        row_h = fm.height() + 1
        box_w = min(rect.width() - 8.0, max(widths) + 22.0)
        box = QRectF(rect.right() - box_w - 4.0, rect.top() + 4.0, box_w, row_h * len(series) + 6.0)
        backdrop = QColor(PANEL)
        backdrop.setAlpha(215)
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.setBrush(QBrush(backdrop))
        p.drawRect(box)
        for row, entry in enumerate(series):
            y = box.top() + 3.0 + row * row_h
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(entry.color)))
            p.drawRect(QRectF(box.left() + 6.0, y + (row_h - 7.0) / 2.0, 7, 7))
            p.setPen(QColor(entry.color))
            text = fm.elidedText(entry.label, Qt.ElideRight, int(box.width() - 22.0))
            p.drawText(QRectF(box.left() + 18.0, y, box.width() - 20.0, row_h), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.setBrush(Qt.NoBrush)

    def _paint_dynamic(self, p: QPainter) -> None:
        if self._xmax <= self._xmin:
            return
        x = self._x(self._time)
        rect = self.plot_rect()
        p.setPen(QPen(QColor(INFO), 1, Qt.DashLine))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.time_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._dragging:
            self.time_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# Public aliases (the leading-underscore names stay for the replay page).
TelemetryChart = _TelemetryChart
qfont = _font
nice_ticks = _nice_ticks
fmt_number = _fmt_number
fmt_clock = _fmt_clock
