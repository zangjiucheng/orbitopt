"""Shared QPainter chart widgets for the Flight Lab pages (FL-501/FL-503).

The synchronized telemetry chart started life in the replay page; the
compare view overlays two runs on the same widget, so it lives here, free of
the pyvista/VTK imports the replay page's 3D view needs.
"""
from __future__ import annotations

import bisect
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from orbitopt.flightlab.ui.telemetry import TelemetryChannel, interpolated_value
from orbitopt.viz.theme import HAIRLINE, INFO, INK_MUTED, INK_SECONDARY, PANEL

__all__ = ["TelemetryChart", "fmt_clock", "fmt_met", "fmt_number", "link_charts", "nice_ticks", "qfont",
           "readout_rows"]


def _font(size: float, bold: bool = False) -> QFont:
    from orbitopt.viz.theme import resolve_family

    font = QFont()
    family = resolve_family()
    if family:
        font.setFamily(family)
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


def _integer_ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    """Whole-number ticks for a count (engines lit): steps of 1, 2, 5, 10 ... never 2.5."""
    if hi <= lo:
        return [float(round(lo))]
    raw = (hi - lo) / max(1, target)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    step = max(1.0, next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw))
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


def _fmt_met(seconds: float) -> str:
    """Mission elapsed time as ``T+h:mm:ss.s`` (the readout's exact clock)."""
    if not math.isfinite(seconds):
        return "T+—"
    negative = seconds < 0
    hours, rest = divmod(abs(float(seconds)), 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"T{'-' if negative else '+'}{int(hours)}:{int(minutes):02d}:{secs:04.1f}"


def readout_rows(channel: TelemetryChannel, t: float) -> list[tuple[str, str, str, float]]:
    """``(label, colour, value text, value)`` of every series with data at time ``t`` -- linearly interpolated, and
    only inside the series' own span (a stage that has not flown yet, or has landed, reads nothing rather than its
    clamped end value). The hover box and the tests share this."""
    rows: list[tuple[str, str, str, float]] = []
    for series in channel.series:
        pts = series.points
        if not pts or not (pts[0][0] - 1e-9 <= t <= pts[-1][0] + 1e-9):
            continue
        value = interpolated_value(pts, t)
        if value is None or not math.isfinite(value):
            continue
        rows.append((series.label or series.body_id, series.color, _fmt_number(value), float(value)))
    return rows


class _TelemetryChart(QWidget):
    """One synchronized channel chart.

    Static pixels are cached the same way the app's other QPainter charts do:
    the data lines do not change on every playback tick, only the cursor does.
    Click/drag requests a seek so scrubbing a chart keeps the 3D view, event
    highlight and all other channels at the same time.
    """

    time_requested = Signal(float)
    hovered = Signal(float)
    #: the mouse's time over the plot (``None`` when it leaves): :func:`link_charts` mirrors it on the other charts
    #: so one crosshair reads every channel
    hover_time_changed = Signal(object)
    #: the visible time window changed through this chart's own wheel / pan / reset
    view_range_changed = Signal(float, float)

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
        self._hover: float | None = None
        self._pan_x = None
        self._full_range = (0.0, 1.0)
        self._detail_chart = None
        self._detail_dialog = None
        self.setToolTip("Hover: read every channel at that time · left-click or drag: seek · scroll: zoom · "
                        "right- or shift-drag: pan · double-click: expand · Left/Right: step one sample")
        self._cache: QPixmap | None = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # arrow keys step the cursor
        self.setMinimumSize(120, 110)
        from orbitopt.flightlab.ui.plot_export import ExportButton, channel_to_csv

        # download: the picture as shown, and the data behind it (the right margin is kept clear of the legend)
        self.export_button = ExportButton(self, self._label, csv=lambda: channel_to_csv(self._channel))

    def set_channel(self, channel: TelemetryChannel) -> None:
        self._channel = channel
        self._recompute_y_range()
        self.invalidate()
        if self._detail_chart is not None:
            self._detail_chart.set_channel(channel)

    def set_time_range(self, lo: float, hi: float) -> None:
        self._xmin = float(lo)
        self._xmax = float(hi)
        self._full_range = (float(lo), float(hi))
        self._recompute_y_range()
        self.invalidate()
        if self._detail_chart is not None:
            self._detail_chart.set_time_range(lo, hi)

    def clear(self) -> None:
        self.set_channel(TelemetryChannel(self.channel_id, self._label(), self._unit(), "waiting for a Flight Lab run"))

    def set_time(self, t: float) -> None:
        self._time = float(t)
        self.update()
        if self._detail_chart is not None:
            self._detail_chart.set_time(t)

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
        self.export_button.reposition()

    def plot_rect(self) -> QRectF:
        left, top, right, bottom = self.MARGINS
        return QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))

    def _recompute_y_range(self) -> None:
        from orbitopt.flightlab.ui.telemetry import interpolated_value
        values = [value for series in self._channel.series for t, value in series.points
                  if self._xmin <= t <= self._xmax and math.isfinite(value)]
        values += [v for series in self._channel.series for t in (self._xmin, self._xmax)
                   if series.points and series.points[0][0] <= t <= series.points[-1][0]
                   and (v := interpolated_value(series.points, t)) is not None]
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

        counts = self._channel.unit == "engines"          # a whole-number quantity: whole-number axis
        for yv in (_integer_ticks(self._ymin, self._ymax) if counts else _nice_ticks(self._ymin, self._ymax)):
            py = self._y(yv)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(8.5))
            p.drawText(QRectF(0, py - 8, self.MARGINS[0] - 6, 16), Qt.AlignRight | Qt.AlignVCenter,
                       f"{yv:.0f}" if counts else _fmt_number(yv))

        for tv in _nice_ticks(self._xmin, self._xmax):
            px = self._x(tv)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(8.5))
            p.drawText(QRectF(px - 35, rect.bottom() + 4, 70, 16), Qt.AlignCenter, _fmt_clock(tv))

        for series in self._channel.series:
            points = series.points
            from orbitopt.flightlab.ui.telemetry import interpolated_value
            if points:
                lo, hi = max(self._xmin, points[0][0]), min(self._xmax, points[-1][0])
                if hi < lo:
                    continue
                points = ((lo, interpolated_value(points, lo)),
                          *[(t, v) for t, v in points if lo < t < hi],
                          (hi, interpolated_value(points, hi)))
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
            # Keep narrow spikes when decimating: each pixel bucket retains
            # both extrema, ordered in time, rather than every nth sample.
            samples = [point for start in range(0, len(points), stride)
                       for point in sorted({min(points[start:start+stride], key=lambda p: p[1]),
                                            max(points[start:start+stride], key=lambda p: p[1])})]
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
        if self._hover is not None and self._xmin <= self._hover <= self._xmax and self._channel.has_data:
            hx = self._x(self._hover)
            p.setPen(QPen(QColor(INK_SECONDARY), 1, Qt.SolidLine))
            p.drawLine(QPointF(hx, rect.top()), QPointF(hx, rect.bottom()))
            rows = readout_rows(self._channel, self._hover)
            for _label, color, _text, value in rows:
                p.setPen(QPen(QColor(PANEL), 1.2))
                p.setBrush(QBrush(QColor(color)))
                p.drawEllipse(QPointF(hx, self._y(value)), 3.6, 3.6)
            p.setBrush(Qt.NoBrush)
            self._paint_readout(p, rect, hx, rows)

    def _paint_readout(self, p: QPainter, rect: QRectF, hx: float, rows: list[tuple[str, str, str, float]]) -> None:
        """The hover box: the exact time, then one coloured row per series with data there, on the side of the
        crosshair that has room."""
        p.setFont(_font(8.5))
        fm = p.fontMetrics()
        unit = self._channel.unit
        lines = [(None, _fmt_met(self._hover), "")]
        lines += [(color, label, f"{text} {unit}".strip()) for label, color, text, _v in rows]
        if not rows:
            lines.append((None, "no data here", ""))
        label_w = max(fm.horizontalAdvance(a) for _c, a, _b in lines)
        value_w = max(fm.horizontalAdvance(b) for _c, _a, b in lines)
        row_h = fm.height() + 2
        box_w = min(rect.width() - 8.0, 22.0 + label_w + (12.0 + value_w if value_w else 0.0))
        box_h = row_h * len(lines) + 6.0
        x = hx + 10.0
        if x + box_w > rect.right() - 2.0:
            x = hx - 10.0 - box_w
        x = max(rect.left() + 2.0, x)
        y = rect.top() + 4.0
        if self.legend_placement == "inside":      # the legend box owns the top-right: read from below it
            y = max(rect.top() + 4.0, rect.bottom() - box_h - 4.0)
        box = QRectF(x, y, box_w, box_h)
        backdrop = QColor(PANEL)
        backdrop.setAlpha(238)
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.setBrush(QBrush(backdrop))
        p.drawRect(box)
        for n, (color, label, value) in enumerate(lines):
            ry = box.top() + 3.0 + n * row_h
            if color:
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(color)))
                p.drawRect(QRectF(box.left() + 6.0, ry + (row_h - 7.0) / 2.0, 7, 7))
                p.setPen(QColor(INK_SECONDARY))
            else:
                p.setPen(QColor(INFO) if n == 0 else QColor(INK_MUTED))
            p.drawText(QRectF(box.left() + 18.0 if color else box.left() + 6.0, ry,
                              box.width() - 20.0, row_h), Qt.AlignLeft | Qt.AlignVCenter, label)
            if value:
                p.setPen(QColor(color or INK_SECONDARY))
                p.drawText(QRectF(box.left(), ry, box.width() - 6.0, row_h), Qt.AlignRight | Qt.AlignVCenter, value)
        p.setBrush(Qt.NoBrush)

    def set_hover_time(self, t: float | None) -> None:
        """Mirror another chart's crosshair (no signal, so no feedback loop)."""
        t = None if t is None else float(t)
        if t != self._hover:
            self._hover = t
            self.update()

    @property
    def hover_time(self) -> float | None:
        return self._hover

    def _set_hover(self, t: float | None) -> None:
        if t != self._hover:
            self._hover = t
            self.update()
            self.hover_time_changed.emit(t)

    def leaveEvent(self, event):  # noqa: N802
        if not self._dragging:
            self._set_hover(None)
        super().leaveEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        """Left/Right move the cursor to the previous/next sample time of the channel: step through the run one
        record at a time."""
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self._channel.has_data:
            times = sorted({pt[0] for s in self._channel.series for pt in s.points})
            if times:
                if key == Qt.Key.Key_Right:
                    t = times[min(bisect.bisect_right(times, self._time + 1e-9), len(times) - 1)]
                else:
                    t = times[max(bisect.bisect_left(times, self._time - 1e-9) - 1, 0)]
                self.time_requested.emit(t)
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        shift_pan = (event.button() == Qt.MouseButton.LeftButton
                     and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        if shift_pan or event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_x = event.position().x()
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setFocus()
            self.time_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._pan_x is not None:
            delta = (self._pan_x - event.position().x()) / self.plot_rect().width() * (self._xmax - self._xmin)
            self._pan_x = event.position().x()
            self._set_visible_range(self._xmin + delta, self._xmax + delta, emit=True)
            event.accept()
        elif self._dragging:
            self.time_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
        else:
            t = self._time_at_x(event.position().x())
            self.hovered.emit(t)
            self._set_hover(t if self.plot_rect().adjusted(-1, -1, 1, 1).contains(event.position()) else None)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._pan_x is not None and event.button() in (
                Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._pan_x = None
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def set_view_range(self, lo: float, hi: float, *, emit: bool = False) -> None:
        """Show ``[lo, hi]``; silent unless ``emit`` (a linked chart's move must not echo back)."""
        self._set_visible_range(lo, hi, emit=emit)

    @property
    def view_range(self) -> tuple[float, float]:
        return self._xmin, self._xmax

    @property
    def is_zoomed(self) -> bool:
        return (self._xmin, self._xmax) != self._full_range

    def _set_visible_range(self, lo, hi, *, emit: bool = False):
        full_lo, full_hi = self._full_range
        width = min(hi - lo, full_hi - full_lo)
        if width <= 0:
            return
        lo = max(full_lo, min(lo, full_hi - width))
        if (lo, lo + width) == (self._xmin, self._xmax):
            return
        self._xmin, self._xmax = lo, lo + width
        self._recompute_y_range()
        self.invalidate()
        if emit:
            self.view_range_changed.emit(self._xmin, self._xmax)

    def reset_view(self):
        self._set_visible_range(*self._full_range, emit=True)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            event.ignore()
            return
        steps = event.angleDelta().y() / 120
        if not steps:
            return
        anchor = self._time_at_x(event.position().x())
        factor = 1.3 ** max(-8, min(8, -steps))
        if (self._xmax - self._xmin) * factor >= max(1e-6, (self._full_range[1] - self._full_range[0]) * 1e-6):
            self._set_visible_range(anchor + (self._xmin - anchor) * factor,
                                    anchor + (self._xmax - anchor) * factor, emit=True)
        event.accept()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_detail()
            event.accept()

    def open_detail(self):
        """Open a modeless full-size plot; seeks stay linked to playback."""
        if self._detail_dialog is not None:
            self._detail_dialog.show()
            self._detail_dialog.raise_()
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._channel.label + " · Flight telemetry")
        dialog.resize(1100, 680)
        layout = QVBoxLayout(dialog)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Scroll: zoom · right-drag: pan · left-click: seek"))
        toolbar.addStretch(1)
        reset = QPushButton("Reset view")
        reset.setFixedWidth(110)
        toolbar.addWidget(reset)
        layout.addLayout(toolbar)
        chart = _TelemetryChart(self.channel_id, self._label(), self._unit(), dialog)
        chart.set_channel(self._channel)
        chart.set_time_range(*self._full_range)
        chart.set_time(self._time)
        chart.time_requested.connect(self.time_requested)
        reset.clicked.connect(chart.reset_view)
        layout.addWidget(chart, 1)
        readout = QLabel()
        readout.setWordWrap(True)
        readout.setTextFormat(Qt.PlainText)
        layout.addWidget(readout)

        def show_values(t):
            from orbitopt.flightlab.ui.telemetry import interpolated_value
            values = []
            for series in chart._channel.series:
                if series.points and series.points[0][0] <= t <= series.points[-1][0]:
                    value = interpolated_value(series.points, t)
                    values.append(f"{series.label}: {value:,.6g} {series.unit}")
            readout.setText(f"T+{t:.3f} s (interpolated)  |  " + (" · ".join(values) or "No samples at this time"))

        chart.hovered.connect(show_values)
        show_values(self._time)
        self._detail_chart, self._detail_dialog = chart, dialog
        dialog.show()


# Public aliases (the leading-underscore names stay for the replay page).
TelemetryChart = _TelemetryChart
qfont = _font
nice_ticks = _nice_ticks
fmt_number = _fmt_number
fmt_clock = _fmt_clock


def link_charts(charts) -> None:
    """Make ``charts`` one instrument: a crosshair on any of them is mirrored on the others (one hover reads every
    channel at that time) and a zoom or pan of any is applied to all."""
    charts = list(charts)
    for source in charts:
        source.hover_time_changed.connect(lambda t, src=source: [c.set_hover_time(t) for c in charts if c is not src])
        source.view_range_changed.connect(
            lambda lo, hi, src=source: [c.set_view_range(lo, hi) for c in charts if c is not src])


fmt_met = _fmt_met
