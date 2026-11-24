"""Grid-fin demonstration view: what the booster's fins are doing, from the flown fin telemetry.

Three parts, in the style of the engine HUD:
  * a top-down view of the fins around the body (the real layout: Super Heavy V3 has three, in a 90/90/180
    degree arrangement with the catch side bare; an X of four otherwise). Each fin is drawn as a lattice, folded
    against the body while stowed and swung out when deployed, its visible width narrowing as it is deflected;
  * a side view: the airflow against the body, the angle of attack between them, and a fin turned by its
    deflection;
  * angle-of-attack, deflection and bank gauges with pointers, and the state (STOWED / DEPLOYED), Mach and q.

The geometry and the lookups are plain functions so they can be tested; :class:`GridFinHud` only paints. The
numbers are the model's flown telemetry (``fin_alpha_deg``, ``fin_delta_deg``, ``fin_bank_deg``): the track records
the angles, not a deploy flag, so "deployed" follows the model's own rule (below 70 km, above 1.5 km, with
dynamic pressure) or any non-zero angle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

DEPLOY_ALTITUDE_M = 70_000.0
STOW_ALTITUDE_M = 1_500.0
DEPLOY_Q_MIN_PA = 100.0
#: gauge full scales: the model's authority (alpha up to 60 deg, hinge 40 deg, bank 60 deg)
ALPHA_MAX_DEG, DELTA_MAX_DEG, BANK_MAX_DEG = 60.0, 40.0, 60.0


#: half-extents (in ring radii) of the three parts of the view: top view | side view | the three gauges
HALF_TOP, HALF_SIDE, HALF_GAUGES = 1.15, 1.05, 1.7
EXTENT_SUM = 2.0 * (HALF_TOP + HALF_SIDE + HALF_GAUGES)


def fin_azimuths_deg(count: int) -> tuple[float, ...]:
    """Screen angle of each fin around the body, degrees clockwise from the top. Three fins sit in the Super Heavy
    V3 90/90/180 arrangement (the gap of 180 is the catch side, at the bottom); four make an X; otherwise even."""
    count = int(count)
    if count == 3:
        return (-90.0, 0.0, 90.0)
    if count == 4:
        return (45.0, 135.0, 225.0, 315.0)
    if count <= 0:
        return ()
    return tuple(360.0 * k / count for k in range(count))


@dataclass(frozen=True)
class GridFinReadout:
    t_s: float
    has_data: bool
    deployed: bool
    alpha_deg: float
    delta_deg: float
    bank_deg: float
    mach: float | None
    q_kpa: float | None
    altitude_km: float | None


NO_FIN_DATA = GridFinReadout(0.0, False, False, 0.0, 0.0, 0.0, None, None, None)


def _step(t: Sequence[float], values: Sequence[float], at: float) -> float | None:
    best = None
    for tt, vv in zip(t, values):
        if tt <= at + 1e-9:
            best = float(vv)
        else:
            break
    return best


def gridfin_readout_at(t_s: Sequence[float], extra: Mapping[str, Sequence[float]], at: float) -> GridFinReadout:
    """The fins' state at ``at`` from a track's fin columns (``NO_FIN_DATA`` for a track without them, or outside
    the span of its fin telemetry)."""
    cols = {k.lower(): v for k, v in extra.items()}
    if "fin_alpha_deg" not in cols and "fin_delta_deg" not in cols:
        return NO_FIN_DATA
    t = [float(x) for x in t_s]
    if not t or at < t[0] - 1e-9 or at > t[-1] + 1e-9:
        return NO_FIN_DATA
    valid = _step(t, cols["fin_telemetry_valid"], at) if "fin_telemetry_valid" in cols else 1.0
    if valid is not None and valid < 0.5:
        return NO_FIN_DATA                                  # a joined arc that carries no fin data at this time
    alpha = _step(t, cols["fin_alpha_deg"], at) if "fin_alpha_deg" in cols else 0.0
    delta = _step(t, cols["fin_delta_deg"], at) if "fin_delta_deg" in cols else 0.0
    bank = _step(t, cols["fin_bank_deg"], at) if "fin_bank_deg" in cols else 0.0
    alt = _step(t, cols["altitude_m"], at) if "altitude_m" in cols else None
    q = _step(t, cols["dynamic_pressure_pa"], at) if "dynamic_pressure_pa" in cols else None
    mach = _step(t, cols["mach"], at) if "mach" in cols else None
    angles = any(abs(v or 0.0) > 1e-3 for v in (alpha, delta, bank))
    in_window = alt is not None and STOW_ALTITUDE_M <= alt <= DEPLOY_ALTITUDE_M and (q or 0.0) >= DEPLOY_Q_MIN_PA
    return GridFinReadout(float(at), True, bool(angles or in_window), alpha or 0.0, delta or 0.0, bank or 0.0,
                          mach, None if q is None else q / 1000.0, None if alt is None else alt / 1000.0)


def visible_chord_fraction(delta_deg: float) -> float:
    """How much of a fin's chord the top-down view shows when it is turned by ``delta_deg`` about its hinge."""
    return max(0.12, math.cos(math.radians(max(-89.0, min(89.0, delta_deg)))))


def row_layout_ok(width: float, height: float) -> bool:
    """True when the three parts of the view (top view | side view | gauges) leave clear space between them at
    this size -- the same geometry the painter uses."""
    label_h, margin, gap = 30.0, 10.0, 16.0
    r = max(18.0, min((height - label_h) / 2.0 - 12.0, (width - 2 * margin - 2 * gap) / EXTENT_SUM))
    spare = max(gap, (width - 2 * margin - EXTENT_SUM * r) / 2.0)
    x1 = margin + HALF_TOP * r
    x2 = x1 + HALF_TOP * r + spare + HALF_SIDE * r
    x3 = x2 + HALF_SIDE * r + spare + HALF_GAUGES * r
    inside = x3 + HALF_GAUGES * r <= width - margin + 1e-6                  # nothing runs off the right edge
    return (inside and (x2 - HALF_SIDE * r) - (x1 + HALF_TOP * r) >= gap - 1e-6
            and (x3 - HALF_GAUGES * r) - (x2 + HALF_SIDE * r) >= gap - 1e-6)


# ------------------------------------------------------------------ the widget

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
    from PySide6.QtWidgets import QWidget
except Exception:                                          # pragma: no cover
    QWidget = object  # type: ignore[assignment,misc]


def _font(widget, size: float, bold: bool = False):
    f = QFont(widget.font())
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


class GridFinHud(QWidget):                                 # type: ignore[misc,valid-type]
    """The painted grid-fin demonstration. ``set_readout`` takes a :class:`GridFinReadout`."""

    def __init__(self, parent=None, fin_count: int = 3):
        super().__init__(parent)
        self.setObjectName("flightLabGridFinHud")
        self.setMinimumSize(280, 84)
        self._r: GridFinReadout = NO_FIN_DATA
        self._count = int(fin_count)
        self._fixed = True          # Super Heavy's fins are fixed: always out. A vehicle that folds them says so.
        from orbitopt.flightlab.ui.plot_export import ExportButton

        self.export_button = ExportButton(self, lambda: "grid-fins")        # the card as a picture

    def resizeEvent(self, event) -> None:                  # noqa: N802
        super().resizeEvent(event)
        self.export_button.reposition()

    @property
    def readout(self) -> GridFinReadout:
        return self._r

    @property
    def fin_count(self) -> int:
        return self._count

    @property
    def fixed(self) -> bool:
        return self._fixed

    def set_fixed(self, fixed: bool) -> None:
        """Fixed fins (Super Heavy) are drawn out at every moment, their state being ACTIVE (turned) or NEUTRAL;
        fold-away fins (Falcon 9, or any vehicle declaring ``stowable``) also show STOWED."""
        self._fixed = bool(fixed)
        self.update()

    def _out(self, rd) -> bool:
        """Whether the fins are drawn extended: always for fixed fins, else only while deployed."""
        return self._fixed or (rd.has_data and rd.deployed)

    def set_fin_count(self, count: int) -> None:
        self._count = max(0, int(count))
        self.update()

    def set_readout(self, readout: GridFinReadout | None) -> None:
        self._r = readout or NO_FIN_DATA
        self.update()

    # ---- layout: top view | side view | gauges, never touching
    def _layout(self):
        w, h = float(self.width()), float(self.height())
        label_h, margin, gap = 30.0, 10.0, 16.0
        r = max(18.0, min((h - label_h) / 2.0 - 12.0, (w - 2 * margin - 2 * gap) / EXTENT_SUM))
        spare = max(gap, (w - 2 * margin - EXTENT_SUM * r) / 2.0)
        cy = (h - label_h) / 2.0 + 4.0
        x1 = margin + HALF_TOP * r
        x2 = x1 + HALF_TOP * r + spare + HALF_SIDE * r
        x3 = x2 + HALF_SIDE * r + spare + HALF_GAUGES * r
        return r, cy, (x1, x2, x3), label_h

    def paintEvent(self, _event) -> None:                  # noqa: N802
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, PANEL

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.drawLine(QPointF(8, self.height() - 1.0), QPointF(self.width() - 8, self.height() - 1.0))
        r, cy, (x1, x2, x3), label_h = self._layout()
        rd = self._r
        top = 8.0
        # one small heading above each part, so each picture says what it shows
        p.setPen(QColor(INK_MUTED)); p.setFont(_font(self, 7.5, True))
        for x, half, text in ((x1, HALF_TOP, f"TOP VIEW · {self._count} FINS"), (x2, HALF_SIDE, "AIRFLOW"),
                              (x3, HALF_GAUGES, "CONTROL")):
            p.drawText(QRectF(x - half * r, top, 2 * half * r, 12), Qt.AlignCenter, text)
        cy = cy + 6.0
        self._paint_top(p, QPointF(x1, cy), r, rd)
        self._paint_side(p, QPointF(x2, cy), r, rd)
        self._paint_gauges(p, QPointF(x3, cy), r, rd)
        p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 11, True))
        p.drawText(QRectF(0, self.height() - label_h, self.width(), label_h), Qt.AlignCenter, "GRID FINS")
        p.end()

    # ---- top-down: the fins around the body, drawn as outlined lattices (lines, not solid blocks)
    def _paint_top(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_PRIMARY

        live = rd.has_data
        out = QColor(INK_PRIMARY if live else "#4a4a4a")
        body = r * 0.34
        p.setPen(QPen(QColor("#cfcfcf" if live else "#3a3a3a"), 1.4)); p.setBrush(QBrush(QColor("#121212")))
        p.drawEllipse(c, body, body)
        p.setPen(QPen(QColor("#2e2e2e"), 1)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, body * 0.5, body * 0.5)                      # the engine cluster, for scale
        for az in fin_azimuths_deg(self._count):
            p.save(); p.translate(c); p.rotate(az)                      # a fin pointing up, turned to its place
            if self._out(rd):
                length = r * 0.64
                chord = r * 0.52 * visible_chord_fraction(rd.delta_deg if live else 0.0)  # narrower the more it is turned
                p.setPen(QPen(out if live else QColor("#6a6a6a"), 1.6)); p.setBrush(QBrush(QColor(255, 255, 255, 38 if live else 18)))
                cells_l, cells_c = 4, 3
            else:
                length = r * 0.20                                       # folded flat against the body
                chord = r * 0.52 * 0.34
                p.setPen(QPen(QColor("#6a6a6a" if live else "#3a3a3a"), 1.2)); p.setBrush(Qt.NoBrush)
                cells_l, cells_c = 1, 1
            y0 = -body - 0.03 * r - length
            p.drawRect(QRectF(-chord / 2, y0, chord, length))
            p.setPen(QPen(QColor(INK_PRIMARY if live and self._out(rd) else "#4a4a4a"), 0.8))
            for i in range(1, cells_l):                                 # the lattice cells
                y = y0 + length * i / cells_l
                p.drawLine(QPointF(-chord / 2, y), QPointF(chord / 2, y))
            for j in range(1, cells_c):
                x = -chord / 2 + chord * j / cells_c
                p.drawLine(QPointF(x, y0), QPointF(x, y0 + length))
            p.setPen(QPen(QColor("#7a7a7a"), 1.0, Qt.DotLine))          # the hinge it turns about
            p.drawLine(QPointF(0, -body), QPointF(0, y0 + length))
            p.restore()

    # ---- side view: wind line, body axis, the angle between them, and one fin turned by its deflection
    def _paint_side(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import INK_MUTED, INK_PRIMARY, INK_SECONDARY

        live = rd.has_data
        alpha = rd.alpha_deg if live else 0.0
        delta = rd.delta_deg if live and self._out(rd) else 0.0
        top, bot = c.y() - r * 1.0, c.y() + r * 1.0
        # the relative wind: a dashed vertical through the centre, arrows pointing down (the booster falls engines first)
        p.setPen(QPen(QColor("#5a5a5a"), 1.0, Qt.DashLine))
        p.drawLine(QPointF(c.x(), top), QPointF(c.x(), bot))
        p.setPen(QPen(QColor("#8a8a8a"), 1.4))
        for k in range(3):
            y = top + r * (0.10 + 0.20 * k)
            p.drawLine(QPointF(c.x() - r * 0.55, y), QPointF(c.x() - r * 0.55, y + r * 0.14))
            p.drawLine(QPointF(c.x() - r * 0.55, y + r * 0.14), QPointF(c.x() - r * 0.55 - 3, y + r * 0.14 - 4))
            p.drawLine(QPointF(c.x() - r * 0.55, y + r * 0.14), QPointF(c.x() - r * 0.55 + 3, y + r * 0.14 - 4))
        p.setPen(QColor(INK_MUTED)); p.setFont(_font(self, 7))
        p.drawText(QRectF(c.x() - r * 1.05, bot - r * 0.12, r * 0.95, 11), Qt.AlignLeft, "wind")
        # the body, nose up, pitched by alpha about the centre; its axis dash-dotted
        p.save(); p.translate(c); p.rotate(alpha)
        bw, bh = r * 0.19, r * 1.5
        p.setPen(QPen(QColor("#4a4a4a"), 1.0, Qt.DashDotLine))
        p.drawLine(QPointF(0, -bh * 0.72), QPointF(0, bh * 0.58))
        p.setPen(QPen(QColor("#e6e6e6" if live else "#4a4a4a"), 1.6)); p.setBrush(QBrush(QColor("#121212")))
        p.drawRoundedRect(QRectF(-bw / 2, -bh * 0.5, bw, bh), bw * 0.2, bw * 0.2)
        p.drawPolygon(QPolygonF([QPointF(-bw / 2, -bh * 0.5), QPointF(0, -bh * 0.64), QPointF(bw / 2, -bh * 0.5)]))
        # the fin: hinged near the nose end; out from the body when deployed, flat against it when stowed
        hy = -bh * 0.26
        p.save(); p.translate(bw / 2, hy)
        if self._out(rd):
            p.rotate(-delta)                                               # out from the body, turned by delta
            fl, ft = r * 0.50, r * 0.11
            p.setPen(QPen(QColor(INK_PRIMARY), 1.5)); p.setBrush(QBrush(QColor(255, 255, 255, 38)))
            p.drawRect(QRectF(0, -ft / 2, fl, ft))
            p.setPen(QPen(QColor(INK_PRIMARY), 0.8))
            for i in range(1, 4):
                x = fl * i / 4
                p.drawLine(QPointF(x, -ft / 2), QPointF(x, ft / 2))
            p.drawLine(QPointF(0, 0), QPointF(fl, 0))
        else:
            p.rotate(0.0)
            p.setPen(QPen(QColor("#6a6a6a" if live else "#3a3a3a"), 1.2)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(0, -r * 0.03, r * 0.34, r * 0.06))
        p.restore()
        p.restore()
        # alpha: the arc between the wind line (up) and the body axis, labelled to the side, clear of the body
        if live and abs(alpha) > 0.5:
            rad = r * 0.38
            p.setPen(QPen(QColor(INK_SECONDARY), 1.3)); p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), 90 * 16, int(-alpha * 16))
            p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 8.5, True))
            p.drawText(QRectF(c.x() - r * 0.50, c.y() - r * 0.50, r * 0.46, 13), Qt.AlignRight, f"α {alpha:.0f}°")

    # ---- the three gauges and the state
    def _paint_gauges(self, p, c, r, rd) -> None:
        from orbitopt.flightlab.ui.engine_hud import _gauge
        from orbitopt.viz.theme import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SUCCESS, WARNING

        rows = (("AoA", rd.alpha_deg, ALPHA_MAX_DEG), ("FIN δ", rd.delta_deg, DELTA_MAX_DEG),
                ("BANK", rd.bank_deg, BANK_MAX_DEG))
        gr = r * 0.52
        cy = c.y() - r * 0.22
        for i, (name, value, top) in enumerate(rows):
            gx = c.x() + (i - 1) * r * 1.12
            frac = 0.0 if not rd.has_data else max(0.0, min(1.0, abs(value) / top))
            _gauge(p, QPointF(gx, cy), gr, frac, width=3.5, pointer=rd.has_data)
            p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 7.5, True))
            p.drawText(QRectF(gx - gr, cy - gr * 0.5, 2 * gr, 12), Qt.AlignCenter, name)
            p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 10.5, True))
            txt = "—" if not rd.has_data else (f"{value:+.0f}°" if name == "BANK" else f"{value:.0f}°")
            p.drawText(QRectF(gx - gr, cy - gr * 0.08, 2 * gr, 18), Qt.AlignCenter, txt)
        active = rd.has_data and any(abs(v) > 0.01 for v in (rd.alpha_deg, rd.delta_deg, rd.bank_deg))
        if self._fixed:                                   # always out: the question is only whether it is turning
            if not rd.has_data:
                state, color = "FIXED · NO FIN DATA", INK_MUTED
            elif active:
                state, color = "FIXED · ACTIVE", SUCCESS
            else:
                state, color = "FIXED · NEUTRAL", INK_SECONDARY
        elif not rd.has_data:
            state, color = "NO FIN DATA", INK_MUTED
        elif rd.deployed:
            state, color = "DEPLOYED", SUCCESS
        else:
            state, color = "STOWED", WARNING
        p.setFont(_font(self, 9.5, True)); p.setPen(QColor(color))
        p.drawText(QRectF(c.x() - r * 1.7, c.y() + r * 0.62, r * 3.4, 16), Qt.AlignCenter, state)
        bits = []
        if rd.has_data and rd.mach is not None:
            bits.append(f"Mach {rd.mach:.1f}")
        if rd.has_data and rd.q_kpa is not None:
            bits.append(f"q {rd.q_kpa:.1f} kPa")
        if rd.has_data and rd.altitude_km is not None:
            bits.append(f"{rd.altitude_km:.0f} km")
        p.setPen(QColor(INK_MUTED)); p.setFont(_font(self, 8))
        p.drawText(QRectF(c.x() - r * 1.7, c.y() + r * 0.62 + 16, r * 3.4, 14), Qt.AlignCenter, " · ".join(bits))
