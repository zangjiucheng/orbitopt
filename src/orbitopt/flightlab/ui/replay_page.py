"""Mission Control replay page for precomputed Flight Lab runs (FL-501).

Qt belongs here and nowhere else under ``flightlab``: the run contract, scene
export and telemetry derivation are deliberately Qt-free so a headless install
can still inspect a run. This widget only plays those precomputed artifacts --
it never invokes an optimizer or propagation backend.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.compare import bundle_from_run, load_side_files, recovery_summary
from orbitopt.flightlab.contract import FlightRun, Verification
from orbitopt.flightlab.reasons import explain_all
from orbitopt.flightlab.ui.charts import (  # noqa: F401 -- re-exported for callers/tests
    _TelemetryChart,
    _fmt_clock,
    _fmt_number,
    _font,
    _nice_ticks,
    _slider_resolution,
)
from orbitopt.flightlab.ui.recovery_panel import (
    FLIGHTLAB_PANEL_QSS,
    ReasonPanel,
    RecoveryPanel,
    format_check_value,
    reason_label,
    reason_tooltip,
)
from orbitopt.flightlab.ui.telemetry import (
    TelemetryChannel,
    build_telemetry_channels,
    channel_values_at,
    load_run_scene,
)
from orbitopt.viz.scene_renderer import SceneRenderer
from orbitopt.viz.theme import BG_VOID, DANGER, INK_MUTED
from orbitopt.viz.ui_kit import StatusBadge
from orbitopt.viz.viewport import DebouncedInteractor

FLIGHTLAB_FIDELITY = "fast-3dof"
_WARP_LEVELS = (1.0, 10.0, 60.0, 300.0, 1000.0)


class FlightLabReplayPage(QWidget):
    """Open a FlightResult v1 directory and replay it without recomputing it."""

    run_loaded = Signal(str)
    time_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabReplay")
        self._run: FlightRun | None = None
        #: Directory of the loaded run (None when nothing is loaded).
        self.run_dir: Path | None = None
        self._scene: dict | None = None
        self._events: tuple[dict, ...] = ()
        self._channels: tuple[TelemetryChannel, ...] = ()
        self._body_names: dict[str, str] = {}
        self._charts: dict[str, _TelemetryChart] = {}
        self._time = 0.0
        self._tmin = 0.0
        self._tmax = 0.0
        self._playing = False
        self._warp = 60.0
        self._last_tick_ms: float | None = None
        self._slider_resolution = 1000.0
        self._plotter = None
        self._renderer: SceneRenderer | None = None
        self._plotter_index: int | None = None

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._update_badges(Verification())

    def load_run(self, run_dir: Path) -> None:
        """Open one Flight Lab run directory in the current page."""
        self._clear_loaded()
        try:
            run, scene = load_run_scene(run_dir, earth_texture=True)
        except Exception as exc:  # noqa: BLE001 -- UI should explain a bad/closed dir, not stack-trace
            self._show_error(f"Could not open {run_dir}\n{type(exc).__name__}: {exc}")
            return

        self._run = run
        self._scene = scene
        self.run_dir = Path(run_dir).expanduser()
        timeline = scene.get("timeline") or {}
        self._tmin = float(timeline.get("min", 0.0))
        self._tmax = float(timeline.get("max", 0.0))
        self._events = tuple(sorted((timeline.get("events") or ()), key=lambda item: float(item.get("time", 0.0))))
        self._body_names = {str(body.get("id", "")): str(body.get("name", body.get("id", ""))) for body in scene.get("bodies", [])}
        self._channels = build_telemetry_channels(run, self._scene_colors())

        self._update_header()
        self._update_badges(run.verification)
        self._populate_events()
        self._populate_checks(run.verification)
        self._populate_recovery(run, self.run_dir)
        self._populate_charts()
        self._configure_timeline()

        try:
            self._ensure_plotter()
            if self._renderer is None:
                raise RuntimeError("3D viewport could not be created")
            self._renderer.load(scene)
        except Exception as exc:  # noqa: BLE001 -- no GL / missing optional viewer dependency still leaves telemetry useful
            self._show_error(f"Run loaded, but 3D playback is unavailable.\n{type(exc).__name__}: {exc}")
            return

        self.view_stack.setCurrentIndex(self._plotter_index)
        self._set_time(self._tmin)
        QTimer.singleShot(0, self._refit_view)
        self.run_loaded.emit(run.run_id)

    def stop_playback(self) -> None:
        self._set_playing(False)

    def toggle_playback(self) -> None:
        if self._playing:
            self._set_playing(False)
        else:
            self._start_playback()

    def restart(self) -> None:
        self._set_playing(False)
        self._restart()

    def step_time(self, direction: float) -> None:
        """Nudge by 1% of the run (scaled by ``direction``), like the main timeline."""
        if self._run is None or self._tmax <= self._tmin:
            return
        self._seek_to_time(self._time + direction * 0.01 * (self._tmax - self._tmin))

    def seek_relative_event(self, direction: int) -> None:
        times = [float(event.get("time", 0.0)) for event in self._events]
        if direction > 0:
            later = [t for t in times if t > self._time + 1e-6]
            target = min(later) if later else None
        else:
            earlier = [t for t in times if t < self._time - 1e-6]
            target = max(earlier) if earlier else None
        if target is not None:
            self._seek_to_time(target)

    def hideEvent(self, event):  # noqa: N802
        self._set_playing(False)
        super().hideEvent(event)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QWidget()
        header.setObjectName("flightLabHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self.run_title = QLabel("Flight Lab replay")
        self.run_title.setObjectName("appTitle")
        self.run_subtitle = QLabel("Open a FlightResult v1 run directory; playback is computed ahead of time.")
        self.run_subtitle.setObjectName("metricLabel")
        self.run_subtitle.setWordWrap(True)  # assumptions can be long; never widen the window
        title_col.addWidget(self.run_title)
        title_col.addWidget(self.run_subtitle)
        title_row.addLayout(title_col, stretch=1)
        self.open_button = QPushButton("Open run directory…")
        self.open_button.clicked.connect(self._choose_run_directory)
        title_row.addWidget(self.open_button, alignment=Qt.AlignTop)
        header_layout.addLayout(title_row)

        verdict_row = QHBoxLayout()
        verdict_row.addStretch(1)
        self.badge_optimizer = StatusBadge("idle", "Optimizer —")
        self.badge_physics = StatusBadge("idle", "Physics —")
        self.badge_constraints = StatusBadge("idle", "Mission —")
        self.fidelity_label = QLabel(FLIGHTLAB_FIDELITY)
        self.fidelity_label.setObjectName("metricLabel")
        for badge in (self.badge_optimizer, self.badge_physics, self.badge_constraints):
            verdict_row.addWidget(badge)
        verdict_row.addWidget(self.fidelity_label)
        header_layout.addLayout(verdict_row)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("flightLabSplitter")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.view_stack = QStackedWidget()
        self._empty_label = QLabel("No Flight Lab run loaded")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {INK_MUTED}; padding: 24px;")
        self._error_label = QLabel("")
        self._error_label.setAlignment(Qt.AlignCenter)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {DANGER}; padding: 24px;")
        self._empty_index = self.view_stack.addWidget(self._empty_label)
        self._error_index = self.view_stack.addWidget(self._error_label)

        chart_host = QWidget()
        chart_grid = QGridLayout(chart_host)
        chart_grid.setContentsMargins(0, 0, 0, 0)
        chart_grid.setSpacing(6)
        specs = (
            ("altitude", "Altitude", "km"),
            ("speed", "Speed", "km/s"),
            ("dynamic_pressure", "Dynamic pressure", "kPa"),
            ("g_load", "G-load", "g"),
            ("mass", "Mass", "kg"),
            ("throttle", "Throttle", "%"),
        )
        for index, (channel_id, label, unit) in enumerate(specs):
            chart = _TelemetryChart(channel_id, label, unit, chart_host)
            chart.time_requested.connect(self._seek_to_time)
            self._charts[channel_id] = chart
            chart_grid.addWidget(chart, index // 3, index % 3)
        for column in range(3):
            chart_grid.setColumnStretch(column, 1)
        for row in range(2):
            chart_grid.setRowStretch(row, 1)

        self.view_splitter = QSplitter(Qt.Vertical)
        self.view_splitter.setObjectName("flightLabViewSplitter")
        self.view_splitter.addWidget(self.view_stack)
        self.view_splitter.addWidget(chart_host)
        self.view_splitter.setCollapsible(0, False)
        self.view_splitter.setCollapsible(1, False)
        self.view_splitter.setStretchFactor(0, 5)
        self.view_splitter.setStretchFactor(1, 2)
        self.view_splitter.setSizes([430, 220])
        left_layout.addWidget(self.view_splitter, stretch=1)

        timeline_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.toggled.connect(self._play_toggled)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setEnabled(False)
        self.restart_button.clicked.connect(self._restart)
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.valueChanged.connect(self._slider_changed)
        self.time_label = QLabel("T+0.0 s")
        self.time_label.setObjectName("metReadout")
        self.readout_label = QLabel("no telemetry loaded")
        self.readout_label.setObjectName("metricLabel")
        self.readout_label.setMinimumWidth(240)
        self.warp_combo = QComboBox()
        for warp in _WARP_LEVELS:
            self.warp_combo.addItem(f"{warp:g}×", warp)
        self.warp_combo.setCurrentIndex(2)
        self.warp_combo.currentIndexChanged.connect(self._warp_changed)
        timeline_row.addWidget(self.play_button)
        timeline_row.addWidget(self.restart_button)
        timeline_row.addWidget(self.timeline_slider, stretch=1)
        timeline_row.addWidget(self.time_label)
        timeline_row.addWidget(self.readout_label, stretch=2)
        timeline_row.addWidget(self.warp_combo)
        left_layout.addLayout(timeline_row)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("flightLabRightPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        event_title = QLabel("EVENTS")
        event_title.setObjectName("sectionHeader")
        checks_title = QLabel("VERIFICATION CHECKS")
        checks_title.setObjectName("sectionHeader")
        self.event_list = QListWidget()
        self.event_list.itemClicked.connect(self._event_clicked)
        self.check_table = QTableWidget(0, 5)
        self.check_table.setObjectName("flightLabChecksTable")
        self.check_table.setHorizontalHeaderLabels(("Check", "Result", "Value", "Limit", "Reason"))
        self.check_table.verticalHeader().setVisible(False)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.check_table.setWordWrap(True)
        self.check_table.setMinimumWidth(270)
        self.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.check_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.reason_panel = ReasonPanel()
        self.reason_panel.clear_entries()
        self.recovery_panel = RecoveryPanel()
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("flightLabDetailTabs")
        self.detail_tabs.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        self.detail_tabs.addTab(self.check_table, "Checks")
        self.detail_tabs.addTab(self.reason_panel, "Why · 原因")
        self.detail_tabs.addTab(self.recovery_panel, "Recovery · 回收")
        right_layout.addWidget(event_title)
        right_layout.addWidget(self.event_list, stretch=2)
        right_layout.addWidget(checks_title)
        right_layout.addWidget(self.detail_tabs, stretch=3)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([980, 320])
        root.addWidget(splitter, stretch=1)

    def _choose_run_directory(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Open Flight Lab run directory", str(Path.cwd()))
        if path:
            self.load_run(Path(path))

    def _scene_colors(self) -> dict[str, str]:
        if self._scene is None:
            return {}
        return {str(body.get("id", "")): str(body.get("color", "")) for body in self._scene.get("bodies", [])
                if body.get("kind") == "spacecraft"}

    def _update_header(self) -> None:
        if self._run is None or self._scene is None:
            return
        self.run_title.setText(str(self._scene.get("title", f"Flight Lab — {self._run.run_id}")))
        subtitle = str(self._scene.get("subtitle", ""))
        assumptions = ", ".join(self._run.assumptions) if self._run.assumptions else "none"
        self.run_subtitle.setText(f"{subtitle} · fidelity {FLIGHTLAB_FIDELITY} · assumptions: {assumptions}")
        self.fidelity_label.setText(FLIGHTLAB_FIDELITY)

    def _update_badges(self, verification: Verification) -> None:
        self._set_verdict_badge(self.badge_optimizer, "Optimizer", verification.optimizer_converged)
        self._set_verdict_badge(self.badge_physics, "Physics", verification.physics_verified)
        self._set_verdict_badge(self.badge_constraints, "Mission", verification.mission_constraints_passed)

    @staticmethod
    def _set_verdict_badge(badge: StatusBadge, label: str, passed: bool | None) -> None:
        if passed is True:
            badge.set_state("verified", f"{label} PASS")
            badge.setToolTip("Confirmed true.")
        elif passed is False:
            badge.set_state("failed", f"{label} FAIL")
            badge.setToolTip("Confirmed false.")
        else:
            badge.set_state("warning", f"{label} N/A")
            badge.setToolTip("Not assessed -- treated as not passed by gates.")

    def _populate_events(self) -> None:
        self.event_list.clear()
        for event in self._events:
            t = float(event.get("time", 0.0))
            body = event.get("bodyId") or ""
            body_label = self._body_names.get(str(body), str(body))
            label = str(event.get("label", "event"))
            item = QListWidgetItem(f"{label} · {_fmt_clock(t)} · {body_label}")
            note = str(event.get("note", ""))
            item.setToolTip("\n".join(part for part in (label, f"T={_fmt_clock(t)}", body_label, note) if part))
            item.setData(Qt.ItemDataRole.UserRole, t)
            self.event_list.addItem(item)

    def _populate_checks(self, verification: Verification) -> None:
        checks = verification.checks
        self.check_table.clearContents()
        self.check_table.setRowCount(len(checks))
        for row, check in enumerate(checks):
            value = self._format_check_value(check.value, check.unit)
            limit = self._format_check_value(check.limit, check.unit)
            labels = (check.id, "PASS" if check.passed else "FAIL", value, limit, reason_label(check))
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
        self.reason_panel.set_entries(entries, title="Why this run is red · 失败原因" if entries else "")

    def _populate_recovery(self, run: FlightRun, run_dir: Path) -> None:
        bundle = bundle_from_run(run, run_dir, **load_side_files(run_dir))
        summary = recovery_summary(bundle)
        self.recovery_panel.set_summary(summary)

    @staticmethod
    def _format_check_value(value: float | None, unit: str) -> str:
        return format_check_value(value, unit)

    def _populate_charts(self) -> None:
        for channel in self._channels:
            chart = self._charts[channel.id]
            chart.set_channel(channel)
            chart.set_time_range(self._tmin, self._tmax)

    def _configure_timeline(self) -> None:
        span = max(0.0, self._tmax - self._tmin)
        self._slider_resolution = _slider_resolution(span)
        maximum = max(0, min(1_000_000, int(round(span * self._slider_resolution))))
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(maximum)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        enabled = bool(self._run is not None and self._tmax > self._tmin)
        self.play_button.setEnabled(enabled)
        self.restart_button.setEnabled(enabled)
        self.timeline_slider.setEnabled(enabled)

    def _ensure_plotter(self) -> None:
        if self._plotter is not None:
            return
        self._plotter = DebouncedInteractor(self.view_stack)
        self._plotter.set_background(BG_VOID)
        try:
            self._plotter.enable_anti_aliasing()
        except Exception:
            pass
        self._renderer = SceneRenderer(self._plotter)
        self._plotter_index = self.view_stack.addWidget(self._plotter)

    def _show_error(self, text: str) -> None:
        self._clear_loaded()
        self._error_label.setText(text)
        self.view_stack.setCurrentIndex(self._error_index)

    def _clear_loaded(self) -> None:
        self._set_playing(False)
        self._run = None
        self.run_dir = None
        self._scene = None
        self._events = ()
        self._channels = ()
        self._body_names = {}
        self._tmin = self._tmax = 0.0
        self._time = 0.0
        self.event_list.clear()
        self.check_table.clearContents()
        self.check_table.setRowCount(0)
        self.reason_panel.clear_entries()
        self.recovery_panel.clear()
        self._update_badges(Verification())
        for chart in self._charts.values():
            chart.clear()
        self.play_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.time_label.setText("T+0.0 s")
        self.readout_label.setText("no telemetry loaded")

    def _play_toggled(self, checked: bool) -> None:
        if not checked and self._playing:
            self._set_playing(False)
        elif checked and not self._playing:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._run is None or self._tmax <= self._tmin:
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.blockSignals(False)
            return
        if self._time >= self._tmax - 1e-9:
            self._set_time(self._tmin)
        self._playing = True
        self._last_tick_ms = time.monotonic() * 1000.0
        self._timer.start()
        self._update_play_button()

    def _set_playing(self, playing: bool) -> None:
        if self._playing == playing:
            return
        self._playing = playing
        if not playing:
            self._timer.stop()
            self._last_tick_ms = None
        self._update_play_button()

    def _update_play_button(self) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(self._playing)
        self.play_button.setText("Pause" if self._playing else "Play")
        self.play_button.blockSignals(False)

    def _on_tick(self) -> None:
        if not self._playing or self._run is None:
            return
        now_ms = time.monotonic() * 1000.0
        if self._last_tick_ms is None:
            self._last_tick_ms = now_ms
            return
        dt_ms = max(0.0, min(250.0, now_ms - self._last_tick_ms))
        self._last_tick_ms = now_ms
        self._set_time(self._time + (dt_ms / 1000.0) * self._warp)
        if self._time >= self._tmax - 1e-9:
            self._set_playing(False)

    def _slider_changed(self, value: int) -> None:
        if self._run is None:
            return
        self._set_time(self._tmin + float(value) / self._slider_resolution)

    def _seek_to_time(self, t: float) -> None:
        self._set_playing(False)
        self._set_time(t)

    def _restart(self) -> None:
        self._set_time(self._tmin)

    def _warp_changed(self, _index: int) -> None:
        data = self.warp_combo.currentData()
        if data is not None:
            self._warp = float(data)

    def _set_time(self, t: float) -> None:
        if self._run is None:
            return
        t = float(t)
        if math.isfinite(t):
            self._time = max(self._tmin, min(self._tmax, t))
        else:
            self._time = self._tmin
        self.time_changed.emit(self._time)
        self._sync_time_controls()
        self._highlight_event_at_time()
        self._update_readout()
        if self._renderer is not None:
            self._renderer.set_time(self._time, render=False)
            if self._plotter is not None:
                self._plotter.render()

    def _sync_time_controls(self) -> None:
        value = int(round((self._time - self._tmin) * self._slider_resolution))
        maximum = max(0, self.timeline_slider.maximum())
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(max(0, min(maximum, value)))
        self.timeline_slider.blockSignals(False)
        self.time_label.setText(f"T+{_fmt_clock(self._time)}")
        for chart in self._charts.values():
            chart.set_time(self._time)

    def _update_readout(self) -> None:
        parts = []
        for channel in self._channels:
            values = channel_values_at(channel, self._time)
            if not values:
                continue
            rendered = " / ".join(f"{_fmt_number(value)} {channel.unit}" for _, value in values)
            parts.append(f"{channel.label} {rendered}")
        text = " · ".join(parts)
        if len(text) > 240:
            text = text[:237] + "…"
        self.readout_label.setText(text or "no telemetry at this time")

    def _highlight_event_at_time(self) -> None:
        current = -1
        for index, event in enumerate(self._events):
            if float(event.get("time", 0.0)) <= self._time + 1e-6:
                current = index
        if self.event_list.currentRow() != current:
            self.event_list.blockSignals(True)
            self.event_list.setCurrentRow(current)
            self.event_list.blockSignals(False)

    def _event_clicked(self, item: QListWidgetItem) -> None:
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self._seek_to_time(float(t))

    def _refit_view(self) -> None:
        if self._plotter is None or self._scene is None:
            return
        try:
            self._plotter.reset_camera()
        except Exception:
            pass
        try:
            self._plotter.render()
        except Exception:
            pass
