"""Linked 2D charts under the 3D view, drawn with plain QPainter (no chart
library: the data is small -- a ~120x120 grid and a few thousand samples --
and a new dependency would mean regenerating the conda lockfiles).

* PorkchopChart -- the launch-window screen the optimizer searched (Lambert
  total delta-v over departure date x time of flight) with every candidate on
  it. Clicking a candidate selects it everywhere (3D arc, list, inspector).
* DistanceChart -- the spacecraft's distance to each natural body over the
  mission timeline, on one log axis, with the current time as a cursor.
  Clicking or dragging moves the mission time.

Colors follow the dataviz rules: the porkchop is magnitude, so one hue (the
reference sequential blue), with the low-delta-v valleys lightest on the dark
surface and unsolved cells left empty; the distance lines are entities, so each
takes its body's own scene color, the same as in 3D. Text stays in ink colors.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QTabWidget, QToolTip, QWidget

from orbitopt.timescales import mjd2000_to_utc_datetime
from orbitopt.viz.state import MissionState
from orbitopt.viz.theme import ACCENT, BG_VOID, HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, PANEL

# Reference sequential blue ramp, steps 100..700 (dataviz palette.md).
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                   "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CANDIDATE_SELECTED = "#5ec8ff"  # same as the selected arc in 3D (scene_renderer)
_HIT_PX = 12
_NATURAL_KINDS = {"sun", "star", "planet", "dwarf-planet", "moon"}


def _ramp_rgb(t: np.ndarray) -> np.ndarray:
    """(…, 3) uint8 colors along SEQUENTIAL_BLUE for t in [0, 1] (0 = lightest)."""
    stops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEQUENTIAL_BLUE], dtype=float)
    x = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1.0 - f) + stops[i + 1] * f).round().astype(np.uint8)


def color_scale(values: np.ndarray) -> tuple[float, float]:
    """(vmin, vmax) for the porkchop colors: from the best value to the
    median, so the low valleys get the whole ramp and everything above the
    median shares the darkest step (the legend says so)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(finite.min()), float(np.median(finite))
    return vmin, (vmax if vmax > vmin else vmin + 1.0)


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    return [start + k * step for k in range(int((hi - start) / step + 1e-9) + 1)]


def _three_digits(v: float) -> str:
    # Never scientific notation ("1e+03"): whole numbers from 100 up.
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def format_km(km: float) -> str:
    if km >= 1e6:
        return f"{_three_digits(km / 1e6)} M km"
    if km >= 1e3:
        return f"{_three_digits(km / 1e3)} k km"
    return f"{km:,.0f} km"


def _font(size: float, bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


class _CachedChart(QWidget):
    """Static content rendered once into a pixmap (on data change / resize);
    paintEvent blits it and draws only the cheap dynamic overlay."""

    MARGINS = (58, 14, 18, 34)  # left, top, right, bottom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(150)
        self._cache: QPixmap | None = None

    def plot_rect(self) -> QRectF:
        left, top, right, bottom = self.MARGINS
        return QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))

    def invalidate(self):
        self._cache = None
        self.update()

    def resizeEvent(self, event):  # noqa: N802
        self._cache = None
        super().resizeEvent(event)

    def paintEvent(self, _event):  # noqa: N802
        ratio = self.devicePixelRatioF()
        if self._cache is None or self._cache.size() != self.size() * ratio:
            pixmap = QPixmap(self.size() * ratio)
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(QColor(PANEL))
            p = QPainter(pixmap)
            p.setRenderHint(QPainter.Antialiasing)
            self.paint_static(p)
            p.end()
            self._cache = pixmap
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        p.setRenderHint(QPainter.Antialiasing)
        self.paint_dynamic(p)
        p.end()

    def paint_static(self, p: QPainter):
        raise NotImplementedError

    def paint_dynamic(self, p: QPainter):
        pass

    def _empty(self, p: QPainter, text: str):
        p.setPen(QColor(INK_MUTED))
        p.setFont(_font(10))
        p.drawText(self.rect(), Qt.AlignCenter, text)


# ---------------------------------------------------------------- porkchop
class PorkchopChart(_CachedChart):
    candidate_clicked = Signal(str)

    MARGINS = (64, 22, 92, 38)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = np.empty(0)
        self._tof = np.empty(0)
        self._dv = np.empty((0, 0))
        self._image: QImage | None = None
        self._vmin, self._vmax = 0.0, 1.0
        self._candidates: list[dict] = []
        self._selected: str | None = None
        self._model = ""
        self._hover: QPointF | None = None

    def set_data(self, porkchop: dict | None, candidates: list[dict], selected: str | None):
        self._candidates = [c for c in candidates if c.get("departureMjd2000") is not None]
        self._selected = selected
        if porkchop:
            self._t0 = np.asarray(porkchop["departureMjd2000"], dtype=float)
            self._tof = np.asarray(porkchop["tofDays"], dtype=float)
            self._dv = np.array([[np.nan if v is None else v for v in row] for row in porkchop["totalDv"]],
                                dtype=float)
            self._model = porkchop.get("model", "")
            self._vmin, self._vmax = color_scale(self._dv)
            self._image = self._build_image()
        else:
            self._t0, self._tof, self._dv, self._image = np.empty(0), np.empty(0), np.empty((0, 0)), None
        self.invalidate()

    def set_selected(self, candidate_id: str | None):
        self._selected = candidate_id
        self.invalidate()

    def _build_image(self) -> QImage:
        n_tof, n_t0 = self._dv.shape
        t = (self._dv - self._vmin) / (self._vmax - self._vmin)
        rgba = np.zeros((n_tof, n_t0, 4), dtype=np.uint8)
        rgba[..., :3] = _ramp_rgb(np.nan_to_num(t, nan=1.0))
        rgba[..., 3] = np.where(np.isfinite(self._dv), 255, 0)  # unsolved: empty, not a color
        rgba = np.ascontiguousarray(rgba[::-1])  # row 0 = longest TOF, drawn at the top
        image = QImage(rgba.data, n_t0, n_tof, 4 * n_t0, QImage.Format_RGBA8888)
        return image.copy()  # own the pixels (rgba is a temporary)

    # coordinate maps
    def _x(self, t0: float) -> float:
        r = self.plot_rect()
        return r.left() + (t0 - self._t0[0]) / ((self._t0[-1] - self._t0[0]) or 1.0) * r.width()

    def _y(self, tof: float) -> float:
        r = self.plot_rect()
        return r.bottom() - (tof - self._tof[0]) / ((self._tof[-1] - self._tof[0]) or 1.0) * r.height()

    def _inverse(self, pos: QPointF) -> tuple[float, float]:
        r = self.plot_rect()
        fx = (pos.x() - r.left()) / r.width()
        fy = (r.bottom() - pos.y()) / r.height()
        return (self._t0[0] + fx * (self._t0[-1] - self._t0[0]), self._tof[0] + fy * (self._tof[-1] - self._tof[0]))

    def candidate_at(self, pos: QPointF) -> str | None:
        best, best_d = None, _HIT_PX
        for c in self._candidates:
            d = math.hypot(self._x(c["departureMjd2000"]) - pos.x(), self._y(c["tofDays"]) - pos.y())
            if d <= best_d:
                best, best_d = c["id"], d
        return best

    def paint_static(self, p: QPainter):
        if self._image is None:
            self._empty(p, "No launch-window screen in this scene")
            return
        r = self.plot_rect()
        p.drawImage(r, self._image)

        p.setFont(_font(8.5))
        p.setPen(QColor(INK_MUTED))
        for t in nice_ticks(self._t0[0], self._t0[-1], 4):
            x = self._x(t)
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 4))
            p.drawText(QRectF(x - 40, r.bottom() + 5, 80, 14), Qt.AlignHCenter,
                       mjd2000_to_utc_datetime(t).strftime("%Y-%m-%d"))
        for tof in nice_ticks(self._tof[0], self._tof[-1], 4):
            y = self._y(tof)
            p.drawLine(QPointF(r.left() - 4, y), QPointF(r.left(), y))
            p.drawText(QRectF(r.left() - 44, y - 7, 38, 14), Qt.AlignRight | Qt.AlignVCenter, f"{tof:.0f}")
        p.setPen(QColor(INK_SECONDARY))
        p.drawText(QRectF(r.left(), r.bottom() + 20, r.width(), 14), Qt.AlignHCenter, "Departure (UTC)")
        p.save()
        p.translate(12, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-80, -7, 160, 14), Qt.AlignCenter, "Time of flight (days)")
        p.restore()
        p.setPen(QColor(INK_MUTED))
        p.drawText(QRectF(r.left(), 3, r.width(), 16), Qt.AlignLeft, "Screening grid: " + self._model)

        # Legend: the color bar, low (best) at the top.
        bar = QRectF(r.right() + 16, r.top(), 12, r.height())
        ramp = QImage(1, 64, QImage.Format_RGB888)
        for k, rgb in enumerate(_ramp_rgb(np.linspace(0.0, 1.0, 64))):
            ramp.setPixelColor(0, k, QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        p.drawImage(bar, ramp)
        p.setPen(QColor(INK_SECONDARY))
        p.drawText(QRectF(bar.right() + 4, bar.top() - 2, 60, 14), Qt.AlignLeft, f"{self._vmin:.2f}")
        p.drawText(QRectF(bar.right() + 4, bar.bottom() - 12, 60, 14), Qt.AlignLeft, f"≥{self._vmax:.2f}")
        p.setPen(QColor(INK_MUTED))
        p.drawText(QRectF(bar.right() + 4, bar.center().y() - 14, 60, 28), Qt.AlignLeft | Qt.AlignVCenter,
                   "Δv\nkm/s")

        # Candidates: >= 8 px markers with a 2 px surface ring. All markers
        # first, then all direct labels with a dark halo, so close candidates
        # never hide each other's labels.
        drawn = []
        for c in sorted(self._candidates, key=lambda c: c["id"] == self._selected):
            x, y = self._x(c["departureMjd2000"]), self._y(c["tofDays"])
            if not r.adjusted(-6, -6, 6, 6).contains(QPointF(x, y)):
                continue
            selected = c["id"] == self._selected
            radius = 6.0 if selected else 4.5
            p.setPen(QPen(QColor(BG_VOID), 2.0))
            p.setBrush(QColor(CANDIDATE_SELECTED if selected else INK_PRIMARY))
            p.drawEllipse(QPointF(x, y), radius, radius)
            if c.get("exported"):
                p.setPen(QPen(QColor(ACCENT), 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x, y), radius + 3.0, radius + 3.0)
            drawn.append((c, x, y, radius, selected))
        placed: list[QRectF] = []
        for c, x, y, radius, selected in drawn:
            font = _font(8.5, bold=selected)
            p.setFont(font)
            text = f"#{c.get('searchIndex', '?')}"
            w = p.fontMetrics().horizontalAdvance(text)
            # Right of the marker, else left, else below: first spot that does
            # not collide with a label already placed.
            spots = [QRectF(x + radius + 5, y - 7, w, 14), QRectF(x - radius - 5 - w, y - 7, w, 14),
                     QRectF(x - w / 2, y + radius + 3, w, 14)]
            box = next((b for b in spots if not any(b.intersects(o) for o in placed)), spots[0])
            placed.append(box)
            path = QPainterPath()
            path.addText(QPointF(box.left(), box.top() + 11), font, text)
            p.setPen(QPen(QColor(BG_VOID), 3.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.fillPath(path, QColor(INK_PRIMARY))

    def paint_dynamic(self, p: QPainter):
        if self._image is None or self._hover is None or not self.plot_rect().contains(self._hover):
            return
        r = self.plot_rect()
        p.setPen(QPen(QColor(255, 255, 255, 70), 1.0))
        p.drawLine(QPointF(self._hover.x(), r.top()), QPointF(self._hover.x(), r.bottom()))
        p.drawLine(QPointF(r.left(), self._hover.y()), QPointF(r.right(), self._hover.y()))

    def value_at(self, t0: float, tof: float) -> float | None:
        if self._dv.size == 0:
            return None
        i = int(np.clip(np.searchsorted(self._tof, tof), 0, len(self._tof) - 1))
        j = int(np.clip(np.searchsorted(self._t0, t0), 0, len(self._t0) - 1))
        v = self._dv[i, j]
        return None if not np.isfinite(v) else float(v)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._hover = event.position()
        if self._image is not None and self.plot_rect().contains(self._hover):
            cid = self.candidate_at(self._hover)
            cand = next((c for c in self._candidates if c["id"] == cid), None)
            if cand is not None:
                dv = cand.get("lambertTotalDvMs")
                text = (f"{cand['label']}\nDepart {mjd2000_to_utc_datetime(cand['departureMjd2000']):%Y-%m-%d}"
                        f" · {cand['tofDays']:.0f} d"
                        + ("" if dv is None else f"\nLambert Δv {dv / 1000.0:.3f} km/s") + "\nClick to select")
            else:
                t0, tof = self._inverse(self._hover)
                v = self.value_at(t0, tof)
                text = (f"Depart {mjd2000_to_utc_datetime(t0):%Y-%m-%d} · {tof:.0f} d\n"
                        + ("no Lambert solution" if v is None else f"Δv {v:.3f} km/s"))
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()
        self.update()

    def leaveEvent(self, _event):  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        cid = self.candidate_at(event.position())
        if cid is not None:
            self.candidate_clicked.emit(cid)


# ---------------------------------------------------------------- distance
def distance_series(scene: dict) -> tuple[np.ndarray, list[dict]]:
    """``(times, series)`` for the scene's primary spacecraft: one series per
    natural body, ``{"id", "name", "color", "km": (n,)}``, the spacecraft's
    distance to that body at its own trail times (other trails Hermite- or
    chord-interpolated there, as the 3D view draws them)."""
    from orbitopt.viz.scene_renderer import _sample_trail_many

    bodies = scene.get("bodies", [])
    sc = next((b for b in bodies if b.get("kind") == "spacecraft" and "trail" in b), None)
    if sc is None:
        return np.empty(0), []
    times = np.asarray(sc["trail"]["times"], dtype=float)
    sc_pos = np.asarray(sc["trail"]["positions"], dtype=float)
    series = []
    for body in bodies:
        if body.get("kind") not in _NATURAL_KINDS:
            continue
        if "trail" in body:
            vel = body["trail"].get("velocities")
            pos = _sample_trail_many(
                np.asarray(body["trail"]["times"], dtype=float), np.asarray(body["trail"]["positions"], dtype=float),
                times, None if vel is None else np.asarray(vel, dtype=float))
        else:
            pos = np.broadcast_to(np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float), sc_pos.shape)
        series.append({"id": body["id"], "name": body["name"], "color": body["color"],
                       "km": np.linalg.norm(sc_pos - pos, axis=1)})
    return times, series


class DistanceChart(_CachedChart):
    time_requested = Signal(float)

    MARGINS = (64, 26, 18, 30)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._times = np.empty(0)
        self._series: list[dict] = []
        self._events: list[dict] = []
        self._unit = ""
        self._time = 0.0
        self._lo, self._hi = 1.0, 10.0
        self._hover: QPointF | None = None
        self._dragging = False

    def set_data(self, scene: dict | None, time: float):
        self._times, self._series = (np.empty(0), []) if not scene else distance_series(scene)
        timeline = (scene or {}).get("timeline") or {}
        self._events = timeline.get("events", [])
        self._unit = timeline.get("unitLabel", "")
        self._time = time
        if self._series:
            all_km = np.concatenate([s["km"] for s in self._series])
            positive = all_km[all_km > 0]
            self._lo = 10 ** math.floor(math.log10(max(positive.min(), 1.0)))
            self._hi = 10 ** math.ceil(math.log10(positive.max()))
        self.invalidate()

    def set_time(self, t: float):
        self._time = t
        self.update()  # only the cursor moves: the cached lines stay

    def has_data(self) -> bool:
        return bool(self._series)

    def _x(self, t: float) -> float:
        r = self.plot_rect()
        span = (self._times[-1] - self._times[0]) or 1.0
        return r.left() + (t - self._times[0]) / span * r.width()

    def _y(self, km: float) -> float:
        r = self.plot_rect()
        f = (math.log10(max(km, self._lo)) - math.log10(self._lo)) / (math.log10(self._hi) - math.log10(self._lo))
        return r.bottom() - f * r.height()

    def time_at(self, x: float) -> float:
        r = self.plot_rect()
        f = min(1.0, max(0.0, (x - r.left()) / r.width()))
        return float(self._times[0] + f * (self._times[-1] - self._times[0]))

    def paint_static(self, p: QPainter):
        if not self._series:
            self._empty(p, "No spacecraft trajectory in this scene")
            return
        r = self.plot_rect()
        p.setFont(_font(8.5))
        decade = self._lo
        while decade <= self._hi * 1.0001:  # recessive decade grid
            y = self._y(decade)
            p.setPen(QPen(QColor(HAIRLINE), 1.0))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(2, y - 7, r.left() - 8, 14), Qt.AlignRight | Qt.AlignVCenter, format_km(decade))
            decade *= 10
        for t in nice_ticks(self._times[0], self._times[-1], 6):
            x = self._x(t)
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 4))
            p.drawText(QRectF(x - 30, r.bottom() + 5, 60, 14), Qt.AlignHCenter, f"{t:g}")
        # Axis title in the free corner under the y labels, so it can't
        # collide with the last x tick.
        p.setPen(QColor(INK_SECONDARY))
        p.drawText(QRectF(2, r.bottom() + 5, r.left() - 8, 14), Qt.AlignRight, f"T+ {self._unit}")

        # Events: small ticks on the time axis, same accent as the scrubber.
        p.setPen(QPen(QColor(ACCENT), 1.2))
        for e in self._events:
            x = self._x(e.get("time", 0.0))
            if r.left() <= x <= r.right():
                p.drawLine(QPointF(x, r.bottom() - 5), QPointF(x, r.bottom()))

        # One 2 px line per body in its scene color; legend in ink text.
        step = max(1, len(self._times) // int(max(1.0, r.width() * 2)))
        xs = [self._x(t) for t in self._times[::step]]
        legend_x = r.left()
        for s in self._series:
            poly = QPolygonF([QPointF(x, self._y(km)) for x, km in zip(xs, s["km"][::step])])
            p.setPen(QPen(QColor(s["color"]), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(poly)
            p.setBrush(QColor(s["color"]))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(legend_x, 7, 10, 10), 2, 2)
            p.setPen(QColor(INK_SECONDARY))
            label = f"to {s['name']}"
            p.drawText(QPointF(legend_x + 14, 16), label)
            legend_x += 22 + p.fontMetrics().horizontalAdvance(label)

    def paint_dynamic(self, p: QPainter):
        if not self._series:
            return
        r = self.plot_rect()
        x = self._x(self._time)
        if r.left() - 1 <= x <= r.right() + 1:
            p.setPen(QPen(QColor(INK_PRIMARY), 1.2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            path = QPainterPath()
            path.moveTo(x - 4, r.top() - 6)
            path.lineTo(x + 4, r.top() - 6)
            path.lineTo(x, r.top())
            path.closeSubpath()
            p.fillPath(path, QColor(INK_PRIMARY))
        if self._hover is not None and r.contains(self._hover):
            p.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
            p.drawLine(QPointF(self._hover.x(), r.top()), QPointF(self._hover.x(), r.bottom()))

    def _readout(self, t: float) -> str:
        i = int(np.clip(np.searchsorted(self._times, t), 0, len(self._times) - 1))
        lines = [f"T+{self._times[i]:.2f} {self._unit}"]
        lines += [f"to {s['name']}: {format_km(float(s['km'][i]))}" for s in self._series]
        return "\n".join(lines + ["Click or drag to move the mission time"])

    def mouseMoveEvent(self, event):  # noqa: N802
        self._hover = event.position()
        if self._series and self.plot_rect().contains(self._hover):
            t = self.time_at(self._hover.x())
            QToolTip.showText(event.globalPosition().toPoint(), self._readout(t), self)
            if self._dragging:
                self.time_requested.emit(t)
        else:
            QToolTip.hideText()
        self.update()

    def leaveEvent(self, _event):  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if self._series and self.plot_rect().contains(event.position()):
            self._dragging = True
            self.time_requested.emit(self.time_at(event.position().x()))

    def mouseReleaseEvent(self, _event):  # noqa: N802
        self._dragging = False


# ------------------------------------------------------------------ panel
class ChartPanel(QTabWidget):
    """The chart tabs, wired to MissionState both ways. Hidden for a scene
    with nothing to chart."""

    def __init__(self, state: MissionState, parent=None):
        super().__init__(parent)
        self._state = state
        self.setObjectName("chartTabs")
        self.setDocumentMode(True)
        self.tabBar().setDrawBase(False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.porkchop = PorkchopChart()
        self.distance = DistanceChart()
        self.porkchop.candidate_clicked.connect(state.select_candidate)
        self.distance.time_requested.connect(state.set_time)
        state.scene_changed.connect(self._on_scene_changed)
        state.candidate_changed.connect(self.porkchop.set_selected)
        state.time_changed.connect(self.distance.set_time)

    def _on_scene_changed(self, scene):
        self.clear()
        self.porkchop.set_data((scene or {}).get("porkchop"), self._state.candidates, self._state.candidate_id)
        self.distance.set_data(scene, self._state.time)
        if (scene or {}).get("porkchop"):
            self.addTab(self.porkchop, "Launch window")
        if self.distance.has_data():
            self.addTab(self.distance, "Distance")
        self.setVisible(self.count() > 0)
