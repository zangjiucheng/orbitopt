"""Flight Lab's Vehicle page: a read-only picture of a vehicle definition.

Opened from the *Vehicles* list of Flight Lab's sidebar page. It draws what a
vehicle file declares -- the stack in profile, the staging and thrust schedule
against time, the masses and ideal delta-v, where each number comes from -- and,
when a flown run is selected in *Runs*, sets the declared timings and masses
beside what that run flew. Nothing on the page edits a file: vehicles are
changed in code (the YAML), like every Flight Lab input.

The model is :mod:`orbitopt.flightlab.vehicle_view` (Qt-free); this module
only paints it.
"""
from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.ui.charts import nice_ticks, qfont
from orbitopt.flightlab.ui.recovery_panel import FLIGHTLAB_PANEL_QSS
from orbitopt.flightlab.vehicle_view import OverlayRow, VehicleView
from orbitopt.viz.theme import (
    ACCENT,
    BG_VOID,
    HAIRLINE,
    INFO,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PANEL,
    SURFACE_HOVER,
    SUCCESS,
    WARNING,
)

__all__ = ["VehiclePage", "VehicleProfile", "VehicleTimeline"]

#: tag colour of a provenance level (the file's own vocabulary)
PROVENANCE_COLORS = {"official": SUCCESS, "public_estimate": WARNING, "illustrative": INK_MUTED}
#: an overlay row is flagged beyond this (seconds: the output step of a run; kg: 0.5 %)
OVERLAY_TIME_TOL_S = 2.0
OVERLAY_MASS_TOL = 0.005

_STAGE_FILLS = (QColor("#262626"), QColor("#363636"), QColor("#454545"))
_BURNED = QColor("#555555")               # propellant the declared schedule burns
_LEFT = QColor("#d6d6d6")                 # propellant left at cutoff


def _t(seconds: float) -> str:
    return f"{seconds:.1f} s"


def _kg(value: float) -> str:
    return f"{value / 1000.0:,.1f} t"


# --------------------------------------------------------------------------
# profile drawing
# --------------------------------------------------------------------------

class VehicleProfile(QWidget):
    """The stack in profile, bottom-up, widths and lengths in one scale. A
    vehicle file with a ``geometry:`` block is drawn to scale with its
    lengths marked in metres; without one the stage heights are a schematic
    (mass over diameter squared) and no length is printed."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.view: VehicleView | None = None
        self.setMinimumSize(QSize(260, 420))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def set_vehicle(self, view: VehicleView | None) -> None:
        self.view = view
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        view = self.view
        if view is None:
            return
        geo = view.geometry
        pad_top, pad_bottom, pad_x = 64, 34, 14
        heights = list(geo.stage_length)
        fairing_len = geo.fairing_length or 0.0
        total = sum(heights) + fairing_len
        widest = max([*geo.stage_diameter, geo.fairing_diameter or 0.0]) or 1.0
        avail_h = self.height() - pad_top - pad_bottom
        scale = min(avail_h / total, 90.0 / widest) if total > 0 else 1.0
        cx = pad_x + widest * scale / 2 + 4
        label_x = cx + widest * scale / 2 + (44 if geo.declared else 14)
        y = self.height() - pad_bottom

        p.setFont(qfont(8.5, True))
        p.setPen(QColor(INK_MUTED))
        p.drawText(QRectF(pad_x, 8, self.width() - 2 * pad_x, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "PROFILE · to scale" if geo.declared else "PROFILE · schematic, not to scale")
        self._legend(p, pad_x, 30)

        for i, stage in enumerate(view.stages):
            h, d = heights[i] * scale, geo.stage_diameter[i] * scale
            nose_h = (geo.nose_length or 0.0) * scale if i == len(view.stages) - 1 else 0.0
            rect = QRectF(cx - d / 2, y - h + nose_h, d, h - nose_h)
            # first-stage nozzle exit, to scale in width, under the stage
            if i == 0 and stage.nozzle_exit_area_m2:
                exit_d = math.sqrt(4.0 * stage.nozzle_exit_area_m2 / math.pi) * scale
                drop = min(exit_d * 0.7, pad_bottom - 8)
                p.setPen(QPen(QColor(INK_MUTED), 1))
                p.setBrush(QBrush(QColor(SURFACE_HOVER)))
                p.drawPolygon(QPolygonF([QPointF(cx - exit_d * 0.28, y), QPointF(cx + exit_d * 0.28, y),
                                         QPointF(cx + exit_d / 2, y + drop),
                                         QPointF(cx - exit_d / 2, y + drop)]))
            p.setPen(QPen(QColor(INK_SECONDARY), 1.4))
            p.setBrush(QBrush(_STAGE_FILLS[i % len(_STAGE_FILLS)]))
            p.drawRect(rect)
            # the tank: brown is what the declared schedule burns, blue what stays
            if stage.propellant_kg > 0 and h > 8:
                burned = min(max(stage.burned_kg / stage.propellant_kg, 0.0), 1.0)
                tank = QRectF(rect.left() + 3, rect.top() + 3, rect.width() - 6, rect.height() - 6)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(_BURNED))
                p.drawRect(QRectF(tank.left(), tank.bottom() - tank.height() * burned,
                                  tank.width(), tank.height() * burned))
                if burned < 1.0:
                    p.setBrush(QBrush(_LEFT))
                    p.drawRect(QRectF(tank.left(), tank.top(), tank.width(),
                                      tank.height() * (1.0 - burned)))
            if nose_h > 0.0:
                self._nose(p, view, cx, rect.top(), scale)
            lines = [f"Stage {stage.index}", f"{_kg(stage.mass_initial_kg)} wet"]
            if geo.declared:
                self._bracket(p, rect, cx + widest * scale / 2 + 8)
                lines.append(f"{heights[i]:.1f} m")
            label_rect = QRectF(rect.left(), rect.top() - nose_h, rect.width(), rect.height() + nose_h)
            self._label(p, label_x, label_rect, lines)
            y -= h

        if geo.fairing_length:
            fh = geo.fairing_length * scale
            fd = (geo.fairing_diameter or geo.stage_diameter[-1]) * scale
            body = fh * 0.62
            p.setPen(QPen(QColor(INK_SECONDARY), 1.4, Qt.DashLine))
            p.setBrush(QBrush(QColor(SURFACE_HOVER)))
            p.drawPolygon(QPolygonF([QPointF(cx - fd / 2, y), QPointF(cx + fd / 2, y),
                                     QPointF(cx + fd / 2, y - body),
                                     QPointF(cx + fd * 0.12, y - fh), QPointF(cx - fd * 0.12, y - fh),
                                     QPointF(cx - fd / 2, y - body)]))
            rect = QRectF(cx - fd / 2, y - fh, fd, fh)
            self._bracket(p, rect, cx + widest * scale / 2 + 8)
            self._label(p, label_x, rect, ["Fairing", f"{geo.fairing_length:.1f} m"])
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.drawLine(int(pad_x), int(self.height() - pad_bottom), int(self.width() - pad_x),
                   int(self.height() - pad_bottom))

    @staticmethod
    def _nose(p: QPainter, view: VehicleView, cx: float, base_y: float, scale: float) -> None:
        """The declared nose, from its base (``base_y``) up, to scale."""
        from orbitopt.flightlab.nose_aero import NoseSpec

        geo = view.geometry
        diameter = geo.stage_diameter[-1]
        nose = NoseSpec(str(geo.nose_shape), float(geo.nose_length), diameter)
        x, r = nose.profile(60)
        left = [QPointF(cx - ri * scale, base_y - (nose.length_m - xi) * scale) for xi, ri in zip(x, r)]
        right = [QPointF(cx + ri * scale, base_y - (nose.length_m - xi) * scale) for xi, ri in zip(x, r)]
        p.setPen(QPen(QColor(INK_SECONDARY), 1.4))
        p.setBrush(QBrush(_STAGE_FILLS[(len(view.stages) - 1) % len(_STAGE_FILLS)]))
        p.drawPolygon(QPolygonF(left + right[::-1]))

    @staticmethod
    def _legend(p: QPainter, x: float, y: float) -> None:
        p.setFont(qfont(8))
        for k, (colour, text) in enumerate(((_BURNED, "burned by the declared schedule"),
                                            (_LEFT, "left at cutoff"))):
            top = y + k * 14
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(colour))
            p.drawRect(QRectF(x, top + 3, 8, 8))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(x + 13, top, 230, 14), Qt.AlignLeft | Qt.AlignVCenter, text)

    @staticmethod
    def _label(p: QPainter, x: float, rect: QRectF, lines: list[str]) -> None:
        p.setPen(QColor(INK_PRIMARY))
        p.setFont(qfont(9, True))
        height = 14.0 * len(lines)
        top = rect.center().y() - height / 2
        for k, line in enumerate(lines):
            if k == 1:
                p.setPen(QColor(INK_SECONDARY))
                p.setFont(qfont(8.5))
            p.drawText(QRectF(x, top + 14.0 * k, 150, 14), Qt.AlignLeft | Qt.AlignVCenter, line)

    @staticmethod
    def _bracket(p: QPainter, rect: QRectF, x: float) -> None:
        p.setPen(QPen(QColor(INK_MUTED), 1))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        p.drawLine(QPointF(x - 3, rect.top()), QPointF(x + 3, rect.top()))
        p.drawLine(QPointF(x - 3, rect.bottom()), QPointF(x + 3, rect.bottom()))


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------

class VehicleTimeline(QWidget):
    """Thrust of each stage against time (declared), the staging events as
    dashed verticals, and -- with a flown run -- a lane of where each event
    actually happened. Hover a marker for its name and time."""

    ROW_H = 58
    LEFT, RIGHT, TOP, AXIS_H, LANE_H = 70, 16, 22, 26, 26

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.view: VehicleView | None = None
        self.flown: list[tuple[float, str]] = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._hits: list[tuple[float, float, str]] = []   # (x, y, text)
        self._relayout()

    def set_vehicle(self, view: VehicleView | None, overlay: Sequence[OverlayRow] = ()) -> None:
        self.view = view
        self.flown = [(r.flown, r.label) for r in overlay if r.unit == "s" and r.flown is not None]
        self._relayout()
        self.update()

    def _relayout(self) -> None:
        rows = len(self.view.stages) if self.view else 1
        lane = self.LANE_H if self.flown else 0
        self.setFixedHeight(self.TOP + rows * self.ROW_H + lane + self.AXIS_H)

    # the time axis spans liftoff to the last declared (or flown) event
    def _t_max(self) -> float:
        assert self.view is not None
        # burnouts and flown events set the axis: a stage separation long after
        # the last cutoff (payload deploy) would only empty the plot
        latest = [self.view.last_burnout_s, *[t for t, _ in self.flown]]
        return max(latest) * 1.06 or 1.0

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        self._hits = []
        view = self.view
        if view is None:
            return
        t_max = self._t_max()
        x0, x1 = self.LEFT, self.width() - self.RIGHT

        def px(t: float) -> float:
            return x0 + (x1 - x0) * t / t_max

        rows_bottom = self.TOP + len(view.stages) * self.ROW_H
        p.setFont(qfont(8.5, True))
        p.setPen(QColor(INK_MUTED))
        p.drawText(QRectF(8, 2, 300, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "SCHEDULE · declared thrust and staging")
        # grid + axis
        axis_y = rows_bottom + (self.LANE_H if self.flown else 0)
        p.setFont(qfont(8))
        for tick in nice_ticks(0.0, t_max, 8):
            p.setPen(QPen(QColor(HAIRLINE), 1))
            p.drawLine(QPointF(px(tick), self.TOP), QPointF(px(tick), axis_y))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(px(tick) - 30, axis_y + 3, 60, 16), Qt.AlignHCenter, f"{tick:g} s")

        for i, stage in enumerate(view.stages):
            top = self.TOP + i * self.ROW_H
            base = top + self.ROW_H - 8
            height = self.ROW_H - 20
            peak = max(stage.thrust_vac_n, 1.0)
            p.setPen(QColor(INK_SECONDARY))
            p.setFont(qfont(8.5, True))
            p.drawText(QRectF(8, top + 4, self.LEFT - 12, 16), Qt.AlignLeft, f"Stage {stage.index}")
            p.setFont(qfont(8))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(8, top + 20, self.LEFT - 12, 16), Qt.AlignLeft,
                       f"{peak / 1e6:.1f} MN")
            colour = _STAGE_FILLS[i % len(_STAGE_FILLS)].lighter(180)
            poly = QPolygonF([QPointF(px(stage.thrust[0][0]), base)]
                             + [QPointF(px(t), base - height * f / peak) for t, f in stage.thrust]
                             + [QPointF(px(stage.thrust[-1][0]), base)]) if stage.thrust else QPolygonF()
            p.setPen(QPen(colour.lighter(130), 1.4))
            fill = QColor(colour)
            fill.setAlpha(120)
            p.setBrush(QBrush(fill))
            p.drawPolygon(poly)
            for seg in stage.segments:
                if seg.kind == "coast":
                    p.setPen(QPen(QColor(INK_MUTED), 1, Qt.DotLine))
                    p.drawLine(QPointF(px(seg.t0_s), base - 2), QPointF(px(seg.t1_s), base - 2))
                    p.setFont(qfont(8))
                    p.drawText(QRectF(px(seg.t0_s), base - height, px(seg.t1_s) - px(seg.t0_s), 16),
                               Qt.AlignCenter, f"coast {seg.t1_s - seg.t0_s:.0f} s")
            ign_x = px(stage.ignition_s)
            self._hits.append((ign_x, base, f"Stage {stage.index} ignition · {_t(stage.ignition_s)}"))
            out_x = px(stage.burnout_s)
            self._hits.append((out_x, base, f"Stage {stage.index} cutoff · {_t(stage.burnout_s)}"))

        for t, label, kind in view.markers:
            if t > t_max:
                continue
            p.setPen(QPen(QColor(ACCENT if kind == "separation" else INFO), 1, Qt.DashLine))
            p.drawLine(QPointF(px(t), self.TOP), QPointF(px(t), rows_bottom))
            p.setFont(qfont(7.5))
            p.drawText(QRectF(px(t) + 2, self.TOP - 1, 90, 12), Qt.AlignLeft,
                       "sep" if kind == "separation" else "jettison")
            self._hits.append((px(t), self.TOP + 6, f"{label} · {_t(t)}"))

        if self.flown:
            lane_y = rows_bottom + self.LANE_H / 2
            p.setPen(QColor(INK_MUTED))
            p.setFont(qfont(8.5, True))
            p.drawText(QRectF(8, rows_bottom + 4, self.LEFT - 12, 18), Qt.AlignLeft, "Flown")
            for t, label in self.flown:
                p.setPen(QPen(QColor(SUCCESS), 1.4))
                p.setBrush(QBrush(QColor(SUCCESS)))
                p.drawPolygon(QPolygonF([QPointF(px(t), lane_y + 6), QPointF(px(t) - 5, lane_y - 5),
                                         QPointF(px(t) + 5, lane_y - 5)]))
                self._hits.append((px(t), lane_y, f"{label} · flown {_t(t)}"))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt)
        pos = event.position()
        best = min(self._hits, key=lambda h: abs(h[0] - pos.x()) + 0.5 * abs(h[1] - pos.y()),
                   default=None)
        if best is not None and abs(best[0] - pos.x()) <= 8 and abs(best[1] - pos.y()) <= 30:
            QToolTip.showText(event.globalPosition().toPoint(), best[2], self)
        else:
            QToolTip.hideText()


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def _table(headers: Sequence[str], stretch_col: int | None = None) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)   # read-only, like the whole page
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.setFocusPolicy(Qt.NoFocus)
    table.setWordWrap(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch_col is not None:
        table.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    return table


def _fit_height(table: QTableWidget) -> None:
    rows = table.rowCount()
    height = table.horizontalHeader().height() + 2 + sum(table.rowHeight(r) for r in range(rows))
    table.setFixedHeight(height + 2)


def _display_path(path: str) -> str:
    """A catalog file as ``orbitopt/data/flightlab/vehicles/x.yaml`` (the full
    path, with the user's home folder in it, stays in the tooltip)."""
    parts = path.replace("\\", "/").split("/")
    if "data" not in parts:
        return path
    pkg = [i for i, part in enumerate(parts[:-1]) if part == "orbitopt"]
    return "/".join(parts[pkg[-1]:]) if pkg else path


class VehiclePage(QWidget):
    """The page: header, profile + numbers, schedule, declared-vs-flown, notes."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("vehiclePage")
        self.setStyleSheet(
            FLIGHTLAB_PANEL_QSS
            + f"#vehiclePage, #vehicleBody {{ background: {BG_VOID}; }}"
            + " QScrollArea { border: none; background: transparent; }"
            + " #vehicleBody QLabel { background: transparent; }")
        self.view: VehicleView | None = None
        self.overlay: list[OverlayRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("vehicleBody")
        scroll.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(18, 14, 18, 18)
        col.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setFont(qfont(15, True))
        self.title_label.setStyleSheet(f"color: {INK_PRIMARY};")
        self.tag_label = QLabel()
        self.tag_label.setTextFormat(Qt.RichText)
        self.disclaimer_label = QLabel()
        self.disclaimer_label.setWordWrap(True)
        self.disclaimer_label.setStyleSheet(f"color: {WARNING};")
        self.source_label = QLabel()
        self.source_label.setStyleSheet(f"color: {INK_MUTED};")
        self.readonly_label = QLabel(
            "Read-only · edit the vehicle YAML in code; Flight Lab only draws it.")
        self.readonly_label.setStyleSheet(f"color: {INK_MUTED};")
        for w in (self.title_label, self.tag_label, self.disclaimer_label, self.source_label,
                  self.readonly_label):
            col.addWidget(w)

        mid = QHBoxLayout()
        mid.setSpacing(14)
        self.profile = VehicleProfile()
        self.profile.setFixedWidth(300)
        mid.addWidget(self.profile)
        right = QVBoxLayout()
        self.numbers = _table(["Property", "Value", "Source"], stretch_col=0)
        right.addWidget(self.numbers)
        mid.addLayout(right, 1)
        col.addLayout(mid)

        self.timeline = VehicleTimeline()
        col.addWidget(self.timeline)

        self.overlay_header = QLabel()
        self.overlay_header.setWordWrap(True)
        self.overlay_header.setStyleSheet(f"color: {INK_SECONDARY};")
        self.overlay_table = _table(["Declared vs flown", "Declared", "Flown", "Difference"],
                                    stretch_col=0)
        col.addWidget(self.overlay_header)
        col.addWidget(self.overlay_table)

        self.notes_label = QLabel()
        self.notes_label.setWordWrap(True)
        self.notes_label.setTextFormat(Qt.RichText)
        self.notes_label.setStyleSheet(f"color: {INK_SECONDARY};")
        col.addWidget(self.notes_label)
        col.addStretch(1)
        self.clear()

    # ------------------------------------------------------------- content
    def clear(self) -> None:
        self.view = None
        self.title_label.setText("No vehicle selected")
        for w in (self.tag_label, self.disclaimer_label, self.source_label, self.notes_label,
                  self.overlay_header):
            w.setText("")
        self.numbers.setRowCount(0)
        self.overlay_table.setRowCount(0)
        self.profile.set_vehicle(None)
        self.timeline.set_vehicle(None)

    def set_vehicle(self, view: VehicleView, overlay: Sequence[OverlayRow] = (),
                    overlay_label: str | None = None, digest_match: bool | None = None) -> None:
        """Show ``view``; with ``overlay`` rows (see
        :func:`~orbitopt.flightlab.vehicle_view.flown_overlay`) the declared
        values are set beside the run ``overlay_label`` flew.
        ``digest_match`` says whether that run flew this very file (its
        manifest digest equals the file's), ``None`` when unknown."""
        self.view = view
        self.overlay = list(overlay)
        colour = PROVENANCE_COLORS.get(view.provenance, INK_MUTED)
        self.title_label.setText(view.name)
        self.tag_label.setText(
            f'<span style="color:{colour}">● {view.provenance or "untagged"}</span>'
            f'&nbsp;&nbsp;<span style="color:{INK_MUTED}">as of {view.as_of or "—"} · '
            f'{len(view.stages)} stage{"s" if len(view.stages) != 1 else ""}'
            f'{" · recovery stage" if view.booster else ""}'
            f'{" · restart" if view.restart else ""}</span>')
        self.disclaimer_label.setText(view.disclaimer)
        self.disclaimer_label.setVisible(bool(view.disclaimer))
        self.source_label.setToolTip(view.source_path)
        self.source_label.setText(_display_path(view.source_path))
        self.profile.set_vehicle(view)
        self.timeline.set_vehicle(view, self.overlay)
        self._fill_numbers(view)
        self._fill_overlay(overlay_label, digest_match)
        notes = list(view.notes)
        if not view.geometry.declared:
            notes.append("No `geometry:` block in this vehicle file: the profile is a schematic "
                         "(stage height from mass over diameter², no lengths shown).")
        notes.append("Delta-v is ideal: vacuum Isp over the declared thrust schedule, no gravity, "
                     "drag or steering losses.")
        notes.extend(view.assumptions)
        self.notes_label.setText("<b>Notes</b><ul style='margin-top:2px'>"
                                 + "".join(f"<li>{_esc(n)}</li>" for n in notes) + "</ul>")

    def _row(self, table: QTableWidget, cells: Sequence[str], *, colours: Sequence[str | None] = (),
             tips: Sequence[str] = (), bold: bool = False) -> None:
        r = table.rowCount()
        table.insertRow(r)
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled)
            if c < len(colours) and colours[c]:
                item.setForeground(QBrush(QColor(colours[c])))
            if c < len(tips) and tips[c]:
                item.setToolTip(tips[c])
            if bold:
                item.setFont(qfont(9, True))
            if c > 0:
                left = table is self.numbers and c == len(cells) - 1   # the Source column
                item.setTextAlignment((Qt.AlignLeft if left else Qt.AlignRight) | Qt.AlignVCenter)
            table.setItem(r, c, item)

    def _fill_numbers(self, view: VehicleView) -> None:
        t = self.numbers
        t.setRowCount(0)

        def tag(path: str) -> tuple[str, str | None, str]:
            prov = view.provenance_for(path)
            if prov is None:
                return "—", INK_MUTED, "No provenance tag for this number."
            text = prov.provenance + (f" ±{prov.uncertainty_pct:g}%" if prov.uncertainty_pct else "")
            tip = f"{prov.source}\nas of {prov.as_of}" + (f"\n{prov.accuracy}" if prov.accuracy else "")
            return text, PROVENANCE_COLORS.get(prov.provenance, INK_MUTED), tip

        def add(label: str, value: str, path: str | None = None, *, bold: bool = False) -> None:
            if path is None:
                self._row(t, [label, value, ""], bold=bold)
                return
            text, colour, tip = tag(path)
            self._row(t, [label, value, text], colours=[None, None, colour],
                      tips=["", "", tip], bold=bold)

        add("Liftoff mass", _kg(view.liftoff_mass_kg), bold=True)
        if view.payload_kg:
            add("Payload", _kg(view.payload_kg), "payload_kg")
        add("Liftoff thrust (vacuum)", f"{view.liftoff_thrust_vac_n / 1e6:.2f} MN")
        add("Liftoff T/W (vacuum)", f"{view.liftoff_twr_vac:.2f}")
        if view.liftoff_twr_sl is not None:
            add("Liftoff T/W (sea level est.)", f"{view.liftoff_twr_sl:.2f}")
        add("Ideal Δv, all stages", f"{view.total_delta_v_m_s:,.0f} m/s", bold=True)
        for i, s in enumerate(view.stages):
            base = f"stages[{i}]"
            add(f"Stage {s.index} wet / dry", f"{_kg(s.mass_initial_kg)} / {_kg(s.mass_dry_kg)}",
                f"{base}.mass_initial_kg", bold=True)
            add("  burns", f"{_kg(s.burned_kg)} of {_kg(s.propellant_kg)}")
            add("  Isp (vac) · thrust (peak)", f"{s.isp_vac_s:g} s · {s.thrust_vac_n / 1e6:.2f} MN",
                f"{base}.engine.isp_vac_s")
            add("  burn", f"{_t(s.ignition_s)} → {_t(s.burnout_s)}",
                f"{base}.engine.burn_end_s")
            if s.diameter_m:
                add("  diameter", f"{s.diameter_m:g} m", f"{base}.aero.diameter_m")
            add("  ideal Δv", f"{s.delta_v_m_s:,.0f} m/s")
        b = view.booster
        if b is not None:
            add("Recovery stage", "", bold=True)
            add("  dry mass", _kg(b.dry_mass_kg), "booster.dry_mass_kg")
            add("  landing reserve (planning)", _kg(b.landing_reserve_kg),
                "booster.landing_propellant_reserve_kg")
            eng = (f"{b.engine_count} × " if b.engine_count else "") + \
                  f"{b.thrust_vacuum_n / 1e6:.2f} MN vac"
            add("  landing engines", eng, "booster.engines.thrust_vacuum_n")
            add("  throttle", f"{b.throttle_min:.0%} – {(b.throttle_max or 1.0):.0%}"
                + (f" · {b.restarts_max} relights" if b.restarts_max is not None else ""),
                "booster.engines.throttle_min")
        geo = view.geometry
        if geo.nose_shape and geo.nose_length:
            diameter = geo.stage_diameter[-1]
            add("Nose", f"{geo.nose_shape.replace('_', ' ')} · {geo.nose_length:g} m · "
                f"L/D {geo.nose_length / diameter:.2f}", "geometry.nose")
            ca = _ca_at(view, (0.8, 1.5, 4.0))
            if ca:
                add("  axial Ca, power on (M 0.8 / 1.5 / 4)", " / ".join(f"{c:.2f}" for c in ca),
                    "stages[0].aero")
        if view.geometry.declared:
            add("Geometry", " + ".join(f"{x:g} m" for x in view.geometry.stage_length)
                + (f" + fairing {view.geometry.fairing_length:g} m" if view.geometry.fairing_length else ""),
                "geometry")
        t.resizeRowsToContents()
        _fit_height(t)

    def _fill_overlay(self, label: str | None, digest_match: bool | None) -> None:
        t = self.overlay_table
        t.setRowCount(0)
        show = bool(self.overlay)
        t.setVisible(show)
        if not show:
            self.overlay_header.setText(
                "Select a flown run in <b>Runs</b> to set this vehicle's declared timings "
                "and masses beside what the run flew.")
            self.overlay_header.setTextFormat(Qt.RichText)
            return
        same = {True: "this run flew this exact vehicle file",
                False: "the run's vehicle digest differs from this file (an optimised or edited "
                       "copy): differences may come from that",
                None: "vehicle digest unknown"}[digest_match]
        self.overlay_header.setTextFormat(Qt.PlainText)
        self.overlay_header.setText(f"Flown: {label or 'selected run'} — {same}.")
        for row in self.overlay:
            flag = self._flag(row)
            colour = {"warn": WARNING, "ok": SUCCESS}.get(flag or "")
            self._row(t, [row.label, self._fmt(row, row.declared), self._fmt(row, row.flown),
                          self._fmt_delta(row)], colours=[None, None, None, colour])
        t.resizeRowsToContents()
        _fit_height(t)

    @staticmethod
    def _fmt(row: OverlayRow, value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.1f} s" if row.unit == "s" else f"{value / 1000.0:,.1f} t"

    @staticmethod
    def _fmt_delta(row: OverlayRow) -> str:
        d = row.delta
        if d is None:
            return ""
        return f"{d:+.1f} s" if row.unit == "s" else f"{d / 1000.0:+,.2f} t"

    @staticmethod
    def _flag(row: OverlayRow) -> str | None:
        d = row.delta
        if d is None:
            return None
        if row.unit == "s":
            return "warn" if abs(d) > OVERLAY_TIME_TOL_S else "ok"
        base = abs(row.declared or 0.0) or 1.0
        return "warn" if abs(d) / base > OVERLAY_MASS_TOL else "ok"

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt)
        return QSize(900, 800)


def _esc(text: str) -> str:
    import html

    return html.escape(text)


def _ca_at(view: VehicleView, machs) -> list[float]:
    """Stage 1's declared power-on axial-force coefficient at ``machs`` (the
    stack's drag while it flies the atmosphere); empty when it is a scalar."""
    import numpy as np

    table = ((view.document.get("stages") or [{}])[0].get("aero") or {}).get("ca")
    if not isinstance(table, list):
        return []
    arr = np.asarray(table, dtype=float)
    return [float(np.interp(m, arr[:, 0], arr[:, 1])) for m in machs]
