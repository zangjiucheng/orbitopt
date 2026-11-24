"""Flight Lab's panels in Mission Control's main window.

A Flight Lab run is rendered exactly like a mission -- same 3D view, timeline,
inspector and verification tab -- and these two widgets add what only a
launch-vehicle flight has:

* :class:`FlightRunPanel` -- the right column's *Flight* tab: the launch
  summary (site, liftoff epoch, vehicle, target and achieved orbit, payload,
  the three verdicts), the key events with altitude/speed at each (click one to seek),
  and the Checks / Why / Recovery detail;
* :class:`TelemetryPanel` -- the chart panel's *Telemetry* tab: altitude,
  speed, dynamic pressure, g-load, mass, throttle, estimated stage propellant,
  radial vertical speed and recorded downrange against mission time,
  linked to the timeline both ways.

Both are read-only views of a precomputed run: nothing here simulates.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.compare import RunBundle, recovery_summary
from orbitopt.flightlab.contract import Verification
from orbitopt.flightlab.reasons import explain_all
from orbitopt.flightlab.run_index import KeyEvent, SummaryRow, key_events, launch_summary
from orbitopt.flightlab.ui.charts import _fmt_clock, _TelemetryChart
from orbitopt.flightlab.ui.engine_hud import EngineHud
from orbitopt.flightlab.ui.recovery_panel import (
    FLIGHTLAB_PANEL_QSS,
    ReasonPanel,
    RecoveryPanel,
    format_check_value,
    reason_label,
    reason_tooltip,
)
from orbitopt.flightlab.ui.telemetry import TelemetryChannel, build_telemetry_channels
from orbitopt.viz.theme import ACCENT, DANGER, INK_MUTED, INK_SECONDARY, PANEL, SUCCESS, WARNING

#: (channel id, label, unit) of the telemetry grid, in display order.
TELEMETRY_CHARTS = (
    ("altitude", "Altitude", "km"),
    ("speed", "Speed", "km/s"),
    ("dynamic_pressure", "Dynamic pressure", "kPa"),
    ("g_load", "G-load", "g"),
    ("mass", "Mass", "kg"),
    ("throttle", "Throttle", "%"),
    ("engines_lit", "Engines lit", "engines"),
    ("propellant", "Propellant remaining", "t"),
    ("vertical_speed", "Vertical speed (radial)", "m/s"),
    ("downrange", "Downrange", "km"),
    ("grid_fins", "Grid Fins", "deg"),
)
_HIDDEN_PLOTS_SETTING = "flightlab/hiddenTelemetryPlots"


#: Colour of a verdict value in the summary (the verdict rows read PASS / FAIL / N/A).
VERDICT_COLORS = {"PASS": SUCCESS, "FAIL": DANGER, "N/A": WARNING}


def summary_html(rows: Sequence[SummaryRow]) -> str:
    """The launch summary as compact sectioned HTML (two columns per row);
    verdict values are coloured."""
    parts: list[str] = []
    section = None
    for row in rows:
        if row.section != section:
            if section is not None:
                parts.append("</table>")
            section = row.section
            parts.append(f'<p style="margin:8px 0 2px 0; color:{ACCENT}; font-weight:600;'
                         f' letter-spacing:1px">{html.escape(section.upper())}</p>'
                         '<table cellspacing="0" cellpadding="1" width="100%">')
        parts.append(f'<tr><td width="42%" style="color:{INK_MUTED}; padding-right:8px">'
                     f'{html.escape(row.label)}</td>'
                     f'<td style="color:{VERDICT_COLORS.get(row.value, INK_SECONDARY)}">'
                     f'{html.escape(row.value)}</td></tr>')
    if section is not None:
        parts.append("</table>")
    return "".join(parts)


def _fmt_opt(value: float | None, fmt: str) -> str:
    if value is None:
        return "—"
    return format(0.0 if abs(value) < 0.05 else value, fmt)  # ground contact reads 0.0, not -0.0


class FlightRunPanel(QWidget):
    """The *Flight* tab. Signal ``time_requested(t)``: an event row was clicked."""

    time_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightRunPanel")
        self.bundle: RunBundle | None = None
        self.events: tuple[KeyEvent, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self.summary = QTextBrowser()
        self.summary.setObjectName("flightSummary")
        self.summary.setOpenLinks(False)
        self.summary.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        splitter.addWidget(self.summary)

        events_host = QWidget()
        ev_layout = QVBoxLayout(events_host)
        ev_layout.setContentsMargins(0, 0, 0, 0)
        ev_layout.setSpacing(2)
        header = QLabel("KEY EVENTS")
        header.setObjectName("sectionHeader")
        ev_layout.addWidget(header)
        self.event_table = QTableWidget(0, 4)
        self.event_table.setObjectName("flightEventTable")
        self.event_table.setStyleSheet(FLIGHTLAB_PANEL_QSS)   # dark cells; the system style is white
        self.event_table.setHorizontalHeaderLabels(("T+", "Event", "Alt km", "km/s"))
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.event_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.event_table.cellClicked.connect(self._event_clicked)
        ev_layout.addWidget(self.event_table)
        splitter.addWidget(events_host)

        self.check_table = QTableWidget(0, 5)
        self.check_table.setObjectName("flightLabChecksTable")
        self.check_table.setHorizontalHeaderLabels(("Check", "Result", "Value", "Limit", "Reason"))
        self.check_table.verticalHeader().setVisible(False)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.check_table.setWordWrap(True)
        self.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.check_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.reason_panel = ReasonPanel()
        self.reason_panel.clear_entries()
        self.recovery_panel = RecoveryPanel()
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("flightLabDetailTabs")
        self.detail_tabs.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        self.detail_tabs.addTab(self.check_table, "Checks")
        self.detail_tabs.addTab(self.reason_panel, "Why")
        self.detail_tabs.addTab(self.recovery_panel, "Recovery")
        self.engine_hud = EngineHud()                      # SpaceX-webcast-style: both vehicles, engines + propellant
        self.detail_tabs.addTab(self.engine_hud, "Engines")
        self._time = 0.0
        splitter.addWidget(self.detail_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        layout.addWidget(splitter, stretch=1)
        self.clear()

    # ------------------------------------------------------------- content
    def clear(self) -> None:
        self.bundle = None
        self.events = ()
        self.summary.setHtml(f'<p style="color:{INK_MUTED}">No Flight Lab run loaded.</p>')
        self.event_table.setRowCount(0)
        self.check_table.setRowCount(0)
        self.reason_panel.clear_entries()
        self.recovery_panel.clear()
        self.engine_hud.clear()

    def set_run(self, bundle: RunBundle, target=None,
                channels: Sequence[TelemetryChannel] | None = None) -> None:
        """Show ``bundle`` (a run plus its side files); ``target`` adds the
        site name and target orbit when the run's target file was found."""
        self.bundle = bundle
        run = bundle.run
        self.summary.setHtml(summary_html(launch_summary(bundle, target)))
        self.events = tuple(key_events(run, channels))
        self._populate_events()
        self._populate_checks(run.verification)
        summary = recovery_summary(bundle)
        self.recovery_panel.set_summary(summary)
        # the engine HUD: Super Heavy (stage 1, then the recovering booster) and Starship, each with its engines,
        # speed and propellant -- the telemetry channels carry the speed and the propellant series
        fins = ((bundle.vehicle or {}).get("booster") or {}).get("grid_fins") or {}
        # fixed (never folded) unless the vehicle's grid_fins block says it can stow: Super Heavy's fins are always out
        fixed = (not fins) or not bool(fins.get("stowable", True))
        self.engine_hud.set_run(run.tracks, channels if channels is not None else build_telemetry_channels(run),
                                summary.engine_burns, fin_count=fins.get("count"), fins_fixed=fixed)
        self.engine_hud.set_time(self._time)

    def set_time(self, t: float) -> None:
        """Highlight the latest event at or before ``t``; the engine HUD follows the same clock."""
        self._time = float(t)
        self.engine_hud.set_time(self._time)
        current = -1
        for row, event in enumerate(self.events):
            if event.t_s <= t + 1e-6:
                current = row
        if current != self.event_table.currentRow():
            self.event_table.blockSignals(True)
            if current < 0:
                self.event_table.clearSelection()
            else:
                self.event_table.selectRow(current)
            self.event_table.blockSignals(False)

    # ------------------------------------------------------------- helpers
    def _populate_events(self) -> None:
        self.event_table.setRowCount(len(self.events))
        for row, event in enumerate(self.events):
            cells = (f"{_fmt_clock(event.t_s)}", event.title,
                     _fmt_opt(event.altitude_km, ",.1f"), _fmt_opt(event.speed_km_s, ".2f"))
            tip = f"{event.title} · {event.body_id} · T+{_fmt_clock(event.t_s)}"
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                item.setData(Qt.ItemDataRole.UserRole, event.t_s)
                if col != 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.event_table.setItem(row, col, item)
        self.event_table.resizeRowsToContents()

    def _populate_checks(self, verification: Verification) -> None:
        checks = verification.checks
        self.check_table.setRowCount(len(checks))
        for row, check in enumerate(checks):
            labels = (check.id, "PASS" if check.passed else "FAIL",
                      format_check_value(check.value, check.unit),
                      format_check_value(check.limit, check.unit), reason_label(check))
            tooltip = reason_tooltip(check)
            for column, text in enumerate(labels):
                item = QTableWidgetItem(text)
                if tooltip:
                    item.setToolTip(tooltip)
                if not check.passed:
                    item.setForeground(QBrush(QColor(DANGER)))
                    item.setBackground(QBrush(QColor("#2a1010")))
                self.check_table.setItem(row, column, item)
        self.check_table.resizeRowsToContents()
        entries = explain_all(c.reason_code for c in checks if not c.passed and c.reason_code)
        self.reason_panel.set_entries(entries, title="Why this run is red" if entries else "")

    def _event_clicked(self, row: int, _col: int) -> None:
        item = self.event_table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
            self.time_requested.emit(float(item.data(Qt.ItemDataRole.UserRole)))


class TelemetryPanel(QScrollArea):
    """The *Telemetry* chart tab. Signal ``time_requested(t)``: a chart was
    clicked or dragged."""

    time_requested = Signal(float)

    def __init__(self, parent=None, *, settings: QSettings | None = None):
        super().__init__(parent)
        self.setObjectName("flightTelemetry")
        self._settings = settings if settings is not None else QSettings("orbitopt", "Mission Control")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(
            f"QScrollArea#flightTelemetry, QWidget#telemetryGrid {{ background: {PANEL}; }}"
        )
        chart_host = QWidget()
        chart_host.setObjectName("telemetryGrid")
        self._grid = QGridLayout(chart_host)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(6)
        self._empty = QLabel("No plots selected. Use Plots to show a chart.", chart_host)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plots_button = QPushButton("Plots")
        self.plots_button.setObjectName("telemetryPlots")
        self.plots_button.setToolTip("Choose which telemetry plots to display")
        menu = QMenu(self.plots_button)
        self.plots_button.setMenu(menu)
        hidden = self._settings.value(_HIDDEN_PLOTS_SETTING, "", type=str).split(",")
        self._plot_actions = {}
        self.charts: dict[str, _TelemetryChart] = {}
        for channel_id, label, unit in TELEMETRY_CHARTS:
            chart = _TelemetryChart(channel_id, label, unit, chart_host)
            chart.time_requested.connect(self.time_requested)
            self.charts[channel_id] = chart
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(channel_id not in hidden)
            action.triggered.connect(self._save_plot_selection)
            self._plot_actions[channel_id] = action
        self._set_fin_plot_available(False)
        menu.addSeparator()
        menu.addAction("Show all plots", self._show_all_plots)
        for column in range(3):
            self._grid.setColumnStretch(column, 1)
        self._layout_plots()
        self.setWidget(chart_host)

    def _layout_plots(self) -> None:
        self._grid.removeWidget(self._empty)
        index = 0
        for channel_id, chart in self.charts.items():
            action = self._plot_actions[channel_id]
            visible = action.isEnabled() and action.isChecked()
            self._grid.removeWidget(chart)
            chart.setVisible(visible)
            if visible:
                row, column = divmod(index, 3)
                self._grid.addWidget(chart, row, column)
                index += 1
        if index == 0:
            self._grid.addWidget(self._empty, 0, 0, 1, 3)
        self._empty.setVisible(index == 0)
        for row in range(self._grid.rowCount()):
            self._grid.setRowStretch(row, int(row == 0 or row * 3 < index))

    def _save_plot_selection(self) -> None:
        hidden = [channel_id for channel_id, action in self._plot_actions.items()
                  if not action.isChecked()]
        self._settings.setValue(_HIDDEN_PLOTS_SETTING, ",".join(hidden))
        self._layout_plots()

    def _show_all_plots(self) -> None:
        for action in self._plot_actions.values():
            action.setChecked(True)
        self._save_plot_selection()

    def _set_fin_plot_available(self, available: bool) -> None:
        action = self._plot_actions["grid_fins"]
        action.setEnabled(available)
        action.setText("Grid Fins" if available else "Grid Fins (no data)")

    def set_channels(self, channels: Sequence[TelemetryChannel], t_min: float, t_max: float) -> None:
        for chart in self.charts.values():
            chart.clear()
        for channel in channels:
            chart = self.charts.get(channel.id)
            if chart is not None:
                chart.set_channel(channel)
                chart.set_time_range(t_min, t_max)
        self._set_fin_plot_available(any(channel.id == "grid_fins" and channel.has_data
                                         for channel in channels))
        self._layout_plots()

    def set_time(self, t: float) -> None:
        for chart in self.charts.values():
            chart.set_time(t)


def scene_track_colors(scene: dict | None) -> dict[str, str]:
    """Body id -> trail colour of the scene's spacecraft, so a telemetry
    series wears its body's colour in the 3D view."""
    return {str(b.get("id", "")): str(b.get("color", "")) for b in (scene or {}).get("bodies", [])
            if b.get("kind") == "spacecraft"}


def load_flight_bundle(run_dir: Path | str, run=None, scene: dict | None = None):
    """``(bundle, target, channels)`` for a run directory (``run`` skips
    re-reading it; ``scene`` colours the telemetry like the 3D trails). Pure
    data, safe off the UI thread."""
    from orbitopt.flightlab.compare import bundle_from_run, load_side_files
    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.run_index import find_target

    run_dir = Path(run_dir).expanduser()
    run = run if run is not None else read_run(run_dir)
    bundle = bundle_from_run(run, run_dir, **load_side_files(run_dir))
    from orbitopt.flightlab.replay_context import replay_inputs
    vehicle, target = replay_inputs(run, run_dir)
    return bundle, target, build_telemetry_channels(run, scene_track_colors(scene), vehicle=vehicle)
