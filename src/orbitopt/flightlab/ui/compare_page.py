"""Mission Control *Compare Runs* page for Flight Lab (FL-503).

Pick two precomputed runs (A and B) and see them side by side:

* overlaid trajectory profiles (altitude vs downrange; A solid, B dashed)
  and the six telemetry channels with both runs on the same axes;
* a diff table of inputs (vehicle, guidance, mission) and outputs (payload,
  insertion errors, margins, booster propellant at touchdown, landing
  errors), with the headline trade-offs in plain sentences (e.g. "Recovery
  reserve cost X kg payload");
* each run's failure reasons in plain language (with a suggested fix), and
  each run's recovery panel (touchdown checks, Monte Carlo summary).

All numbers come from :mod:`orbitopt.flightlab.compare`; this module only
draws them. Like the replay page it never flies anything.
"""
from __future__ import annotations

import json

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.compare import (
    CompareResult,
    DiffRow,
    RunBundle,
    compare_runs,
    discover_runs,
    failure_explanations,
    load_bundle,
    overlay_channels,
    recovery_summary,
    trajectory_profiles,
)
from orbitopt.flightlab.scene_export import TRAIL_PALETTE
from orbitopt.flightlab.ui.charts import TelemetryChart, fmt_number, nice_ticks, qfont
from orbitopt.flightlab.ui.recovery_panel import FLIGHTLAB_PANEL_QSS, ReasonPanel, RecoveryPanel
from orbitopt.flightlab.ui.telemetry import build_telemetry_channels
from orbitopt.viz.theme import (
    ACCENT,
    DANGER,
    HAIRLINE,
    INFO,
    INK_MUTED,
    INK_SECONDARY,
    PANEL,
    SUCCESS,
)

__all__ = ["FlightLabComparePage", "ProfileChart"]

_TELEMETRY_SPECS = (
    ("altitude", "Altitude", "km"),
    ("speed", "Speed", "km/s"),
    ("dynamic_pressure", "Dynamic pressure", "kPa"),
    ("g_load", "G-load", "g"),
    ("mass", "Mass", "kg"),
    ("throttle", "Throttle", "%"),
)


def _body_colors(bundles) -> dict[str, str]:
    ids: list[str] = []
    for bundle in bundles:
        for track in bundle.run.tracks:
            if track.body_id not in ids:
                ids.append(track.body_id)
    return {body_id: TRAIL_PALETTE[i % len(TRAIL_PALETTE)] for i, body_id in enumerate(ids)}


class ProfileChart(QWidget):
    """Altitude vs downrange for every body of up to two runs (A solid,
    B dashed; one colour per body id, shared between runs)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabProfileChart")
        self.setMinimumSize(240, 180)
        self._series: list[tuple[str, str, str, tuple[tuple[float, float], ...]]] = []
        self._note = ""

    @property
    def series(self):
        return tuple(self._series)

    def set_runs(self, a: RunBundle | None, b: RunBundle | None) -> None:
        bundles = [x for x in (a, b) if x is not None]
        colors = _body_colors(bundles)
        self._series = []
        derived = False
        for tag, bundle, style in (("A", a, "solid"), ("B", b, "dash")):
            if bundle is None:
                continue
            for body_id, pts in trajectory_profiles(bundle.run):
                if pts:
                    self._series.append((f"{tag} {body_id}", colors.get(body_id, INFO), style, pts))
            if not any("downrange_m" in {k.lower() for k in t.extra_columns} for t in bundle.run.tracks):
                derived = True
        self._note = "inertial downrange (great-circle from liftoff)" if derived else "downrange"
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        rect = QRectF(60, 24, max(1.0, self.width() - 74), max(1.0, self.height() - 52))
        p.setFont(qfont(10, True))
        p.setPen(QColor(INK_SECONDARY))
        p.drawText(QRectF(rect.left(), 3, rect.width(), 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "Trajectory: altitude (km) vs downrange (km) — A solid, B dashed")
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.drawRect(rect)
        pts = [pt for *_x, s in self._series for pt in s]
        if not pts:
            p.setPen(QColor(INK_MUTED))
            p.drawText(rect, Qt.AlignCenter, "Pick two runs to overlay")
            p.end()
            return
        x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
        y0, y1 = min(0.0, min(y for _, y in pts)), max(y for _, y in pts)
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        y1 += (y1 - y0) * 0.05

        def px(x: float, y: float) -> QPointF:
            return QPointF(rect.left() + (x - x0) / (x1 - x0) * rect.width(),
                           rect.bottom() - (y - y0) / (y1 - y0) * rect.height())

        p.setFont(qfont(8.5))
        for xv in nice_ticks(x0, x1):
            q = px(xv, y0)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(q.x(), rect.top()), QPointF(q.x(), rect.bottom()))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(q.x() - 35, rect.bottom() + 3, 70, 14), Qt.AlignCenter, fmt_number(xv))
        for yv in nice_ticks(y0, y1):
            q = px(x0, yv)
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(rect.left(), q.y()), QPointF(rect.right(), q.y()))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(0, q.y() - 8, 54, 16), Qt.AlignRight | Qt.AlignVCenter, fmt_number(yv))
        p.drawText(QRectF(rect.left(), rect.bottom() + 15, rect.width(), 14), Qt.AlignCenter, self._note)
        for _label, color, style, series in self._series:
            line = Qt.DashLine if style == "dash" else Qt.SolidLine
            p.setPen(QPen(QColor(color), 1.8, line, Qt.RoundCap, Qt.RoundJoin))
            stride = max(1, len(series) // max(2, int(rect.width() * 1.5)))
            sample = list(series[::stride])
            if sample[-1] != series[-1]:
                sample.append(series[-1])
            p.drawPolyline(QPolygonF([px(x, y) for x, y in sample]))
        # legend
        p.setFont(qfont(8.5))
        y = rect.top() + 4
        for label, color, style, _s in self._series:
            p.setPen(QPen(QColor(color), 2, Qt.DashLine if style == "dash" else Qt.SolidLine))
            p.drawLine(QPointF(rect.right() - 150, y + 6), QPointF(rect.right() - 128, y + 6))
            p.setPen(QColor(INK_SECONDARY))
            p.drawText(QRectF(rect.right() - 124, y, 122, 13), Qt.AlignLeft | Qt.AlignVCenter, label)
            y += 14
        p.end()


def _fmt_cell(value, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = fmt_number(value)
        return f"{text} {unit}" if unit else text
    return str(value)


def _fmt_delta(row: DiffRow) -> str:
    if row.delta is None:
        # A value one run does not record (an ascent-only run has no booster
        # figures; a run without vehicle.yaml no vehicle inputs) is not a
        # change of value: say which side has it.
        if row.changed and (row.a is None) != (row.b is None):
            return "only in B" if row.a is None else "only in A"
        return "changed" if row.changed else ""
    if row.delta == 0:
        return "0"
    sign = "+" if row.delta > 0 else "−"
    text = fmt_number(abs(row.delta))
    return f"{sign}{text} {row.unit}".rstrip()


class _RunPicker(QWidget):
    """Label + combo of known runs + Open… button for one side."""

    picked = Signal(str)

    def __init__(self, tag: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"Run {tag}")
        label.setObjectName("sectionHeader")
        self.combo = QComboBox()
        # Sized to ~16 characters and growing with the row: sized to its
        # longest run label, the two pickers held the page (and the window)
        # near 1,000 px wide. The drop-down still shows full labels.
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(16)
        self.combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self.combo.activated.connect(self._activated)
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self._browse)
        layout.addWidget(label)
        layout.addWidget(self.combo, stretch=1)
        layout.addWidget(self.open_button)

    def set_runs(self, runs: list[tuple[str, Path]], current: Path | None) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("— pick a run —", "")
        for label, path in runs:
            self.combo.addItem(label, str(path))
            self.combo.setItemData(self.combo.count() - 1, f"{label}\n{path}", Qt.ItemDataRole.ToolTipRole)
        if current is not None:
            idx = self.combo.findData(str(current))
            if idx < 0:
                self.combo.addItem(current.name, str(current))
                idx = self.combo.count() - 1
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def _activated(self, index: int) -> None:
        path = self.combo.itemData(index)
        if path:
            self.picked.emit(str(path))

    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Open Flight Lab run directory", str(Path.cwd()))
        if path:
            self.picked.emit(path)


def _run_label(run_dir: Path) -> str:
    """A run opened from disk, by its run id (its folder may be just "run")."""
    try:
        doc = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Path(run_dir).name
    return str(doc.get("run_id") or Path(run_dir).name)


class FlightLabComparePage(QWidget):
    """Compare two Flight Lab runs (FL-503)."""

    compared = Signal()
    runs_changed = Signal()  # known_runs() changed (the sidebar's run list follows)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabCompare")
        self.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        self.bundle_a: RunBundle | None = None
        self.bundle_b: RunBundle | None = None
        self.result: CompareResult | None = None
        self._extra_runs: list[Path] = []
        self._build_ui()
        self.refresh_run_list()

    # ------------------------------------------------------------- public API
    def add_run(self, run_dir: Path | str) -> None:
        """Make a run directory pickable (e.g. one just replayed/simulated)."""
        path = Path(run_dir)
        if path not in self._extra_runs:
            self._extra_runs.append(path)
        self.refresh_run_list()

    def known_runs(self) -> list[tuple[str, Path]]:
        """Designer-cached runs (newest first), then every run opened or
        replayed this session: the run pickers' list, and the sidebar's."""
        runs = [(label, path) for label, path in discover_runs()]
        known = {p for _l, p in runs}
        runs += [(_run_label(p), p) for p in self._extra_runs if p not in known]
        return runs

    def refresh_run_list(self) -> None:
        runs = self.known_runs()
        self.picker_a.set_runs(runs, self.bundle_a.run_dir if self.bundle_a else None)
        self.picker_b.set_runs(runs, self.bundle_b.run_dir if self.bundle_b else None)
        self.runs_changed.emit()

    def set_run(self, side: str, run_dir: Path | str) -> bool:
        """Load ``run_dir`` as run ``"A"`` or ``"B"``; False (and a message)
        when it is not a readable FlightResult directory."""
        try:
            bundle = load_bundle(run_dir)
        except Exception as exc:  # noqa: BLE001 -- explain a bad dir, don't stack-trace the UI
            self.status_label.setText(f"Could not open {run_dir}: {type(exc).__name__}: {exc}")
            self.status_label.setStyleSheet(f"color: {DANGER};")
            return False
        if side.upper() == "A":
            self.bundle_a = bundle
        else:
            self.bundle_b = bundle
        if bundle.run_dir not in self._extra_runs:
            self._extra_runs.append(bundle.run_dir)
        self.refresh_run_list()
        self._update()
        return True

    def set_runs(self, a: Path | str, b: Path | str) -> bool:
        return self.set_run("A", a) and self.set_run("B", b)

    def swap(self) -> None:
        self.bundle_a, self.bundle_b = self.bundle_b, self.bundle_a
        self.refresh_run_list()
        self._update()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        title = QLabel("Compare runs")
        title.setObjectName("appTitle")
        root.addWidget(title)
        pick_row = QHBoxLayout()
        self.picker_a = _RunPicker("A")
        self.picker_b = _RunPicker("B")
        self.picker_a.picked.connect(lambda p: self.set_run("A", p))
        self.picker_b.picked.connect(lambda p: self.set_run("B", p))
        self.swap_button = QPushButton("Swap A↔B")
        self.swap_button.clicked.connect(self.swap)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_run_list)
        # A over B (one row each held the page near 850 px wide).
        pick_row.addWidget(self.picker_a, stretch=1)
        pick_row.addWidget(self.swap_button)
        root.addLayout(pick_row)
        pick_row_b = QHBoxLayout()
        pick_row_b.addWidget(self.picker_b, stretch=1)
        pick_row_b.addWidget(self.refresh_button)
        root.addLayout(pick_row_b)
        width = max(self.swap_button.sizeHint().width(), self.refresh_button.sizeHint().width())
        self.swap_button.setMinimumWidth(width)  # the two buttons line up
        self.refresh_button.setMinimumWidth(width)
        self.status_label = QLabel("Pick run A and run B (designer cache runs are listed; Open… for any run dir).")
        self.status_label.setObjectName("metricLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self.headline_label = QLabel("")
        self.headline_label.setObjectName("flightLabCompareHeadline")
        self.headline_label.setWordWrap(True)
        self.headline_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.headline_label)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_chart = ProfileChart()
        chart_host = QWidget()
        grid = QGridLayout(chart_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.charts: dict[str, TelemetryChart] = {}
        for i, (cid, label, unit) in enumerate(_TELEMETRY_SPECS):
            chart = TelemetryChart(cid, label, unit, chart_host)
            chart.time_requested.connect(self._cursor_to)
            self.charts[cid] = chart
            grid.addWidget(chart, i // 3, i % 3)
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(self.profile_chart)
        vsplit.addWidget(chart_host)
        vsplit.setSizes([300, 320])
        left_layout.addWidget(vsplit)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        diff_host = QWidget()
        diff_layout = QVBoxLayout(diff_host)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        self.changed_only = QCheckBox("Changed rows only")
        self.changed_only.setChecked(True)
        self.changed_only.toggled.connect(lambda _c: self._populate_diff())
        self.diff_table = QTableWidget(0, 5)
        self.diff_table.setObjectName("flightLabDiffTable")
        self.diff_table.setHorizontalHeaderLabels(("Section", "Quantity", "A", "B", "Δ (B−A)"))
        self.diff_table.verticalHeader().setVisible(False)
        self.diff_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.diff_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.diff_table.horizontalHeader()
        # Quantity takes the slack; the others are sized to their contents up
        # to a cap after every fill (_fit_columns). Sized to contents alone, a
        # long value (the vehicle name in A/B) pushed A, B and Δ off the panel.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.diff_table.setTextElideMode(Qt.TextElideMode.ElideRight)
        diff_layout.addWidget(self.changed_only)
        diff_layout.addWidget(self.diff_table)
        self.tabs.addTab(diff_host, "Diff")

        why_host = QWidget()
        why_layout = QHBoxLayout(why_host)
        why_layout.setContentsMargins(0, 0, 0, 0)
        self.reasons_a = ReasonPanel()
        self.reasons_b = ReasonPanel()
        self.reasons_a.clear_entries("Run A not loaded.")
        self.reasons_b.clear_entries("Run B not loaded.")
        why_layout.addWidget(self.reasons_a)
        why_layout.addWidget(self.reasons_b)
        self.tabs.addTab(why_host, "Why")

        rec_host = QWidget()
        rec_layout = QHBoxLayout(rec_host)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        self.recovery_a = RecoveryPanel()
        self.recovery_b = RecoveryPanel()
        for tag, panel in (("A", self.recovery_a), ("B", self.recovery_b)):
            col = QVBoxLayout()
            head = QLabel(f"RUN {tag}")
            head.setObjectName("sectionHeader")
            col.addWidget(head)
            col.addWidget(panel)
            rec_layout.addLayout(col)
        self.tabs.addTab(rec_host, "Recovery")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 560])
        root.addWidget(splitter, stretch=1)

    def _cursor_to(self, t: float) -> None:
        for chart in self.charts.values():
            chart.set_time(t)

    # --------------------------------------------------------------- update
    def _update(self) -> None:
        a, b = self.bundle_a, self.bundle_b
        self.profile_chart.set_runs(a, b)
        self._populate_telemetry(a, b)
        for bundle, reasons, recovery, tag in ((a, self.reasons_a, self.recovery_a, "A"),
                                               (b, self.reasons_b, self.recovery_b, "B")):
            if bundle is None:
                reasons.clear_entries(f"Run {tag} not loaded.")
                recovery.clear(f"Run {tag} not loaded.")
        if a is None or b is None:
            self.result = None
            self.headline_label.setText("")
            self.diff_table.setRowCount(0)
            loaded = a or b
            if loaded is not None:
                self.status_label.setText(f"Run {'A' if a else 'B'}: {loaded.label}. Pick the other run.")
                self.status_label.setStyleSheet("")
                self._fill_side(loaded, "A" if a else "B")
            return
        self.result = compare_runs(a, b)
        self.status_label.setText(f"A = {a.label}  ({a.run_dir})\nB = {b.label}  ({b.run_dir})")
        self.status_label.setStyleSheet("")
        heads = self.result.headlines or ("No payload / recovery trade-off to report between these runs.",)
        self.headline_label.setText("<br>".join(f"<span style='color:{ACCENT}'>▸</span> {h}" for h in heads))
        self._fill_side(a, "A")
        self._fill_side(b, "B")
        self._populate_diff()
        self.compared.emit()

    def _fill_side(self, bundle: RunBundle, tag: str) -> None:
        reasons = self.reasons_a if tag == "A" else self.reasons_b
        recovery = self.recovery_a if tag == "A" else self.recovery_b
        entries = failure_explanations(bundle.run)
        reasons.set_entries(entries, title=f"Run {tag} · {bundle.label}",
                            empty="All checks passed — nothing to explain.")
        recovery.set_summary(recovery_summary(bundle))

    def _populate_telemetry(self, a: RunBundle | None, b: RunBundle | None) -> None:
        colors = _body_colors([x for x in (a, b) if x is not None])
        ch_a = build_telemetry_channels(a.run, colors) if a else ()
        ch_b = build_telemetry_channels(b.run, colors) if b else ()
        merged = overlay_channels(ch_a, ch_b)
        bundles = [x for x in (a, b) if x is not None]
        tmin = min((t.t_start_s for x in bundles for t in x.run.tracks), default=0.0)
        tmax = max((t.t_end_s for x in bundles for t in x.run.tracks), default=1.0)
        by_id = {c.id: c for c in merged}
        for cid, chart in self.charts.items():
            if cid in by_id:
                chart.set_channel(by_id[cid])
                chart.set_time_range(tmin, tmax)
            else:
                chart.clear()

    def _populate_diff(self) -> None:
        if self.result is None:
            self.diff_table.setRowCount(0)
            return
        only = self.changed_only.isChecked()
        rows = [("Input", r) for r in self.result.inputs if r.changed or not only]
        rows += [("Output", r) for r in self.result.outputs if r.changed or not only]
        self.diff_table.setRowCount(len(rows))
        for i, (kind, row) in enumerate(rows):
            texts = (f"{kind} · {row.section}", row.label, _fmt_cell(row.a, row.unit),
                     _fmt_cell(row.b, row.unit), _fmt_delta(row))
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, row.key)
                item.setToolTip(str(text))  # the full text when a capped column elides it
                if col == 4 and row.trend:
                    item.setForeground(QBrush(QColor(SUCCESS if row.trend == "better" else DANGER)))
                elif col in (2, 3) and str(text) in ("FAIL", "FAILED"):
                    item.setForeground(QBrush(QColor(DANGER)))
                elif col in (2, 3) and str(text) in ("PASS", "LANDED"):
                    item.setForeground(QBrush(QColor(SUCCESS)))
                self.diff_table.setItem(i, col, item)
        self._fit_columns()
        self.diff_table.resizeRowsToContents()

    _MAX_COLUMN_PX = {0: 130, 2: 120, 3: 120, 4: 110}

    def _fit_columns(self) -> None:
        """Every column but Quantity to its contents, capped (elided past
        the cap, full text in the tooltip): A, B and Δ stay on the panel."""
        for col, cap in self._MAX_COLUMN_PX.items():
            hint = max(self.diff_table.sizeHintForColumn(col),
                       self.diff_table.horizontalHeader().sectionSizeHint(col))
            self.diff_table.setColumnWidth(col, min(cap, hint + 12))

    def diff_keys(self) -> list[str]:
        """Row keys currently shown in the diff table (tests)."""
        return [self.diff_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
                for r in range(self.diff_table.rowCount())]
