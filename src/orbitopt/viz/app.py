"""Mission Control: a persistent, general-purpose desktop app for browsing
every scene this framework can produce (solar system, the Artemis II
mission, and any future one -- or any scene JSON someone hands it via
File > Open) -- not a script that renders one scene and exits. Built with
PySide6 + pyvistaqt: the 3D content is the same PyVista/VTK renderer as
pv_viewer.py (see scene_renderer.py, shared by both), the surrounding
chrome (mission list, orbit-info cards, time-warp strip) is real Qt
widgets styled after a dark "tracking station" look (theme.py) -- KSP's
map view was the reference point for what that chrome should *do*
(a vessel/mission list you switch between, a time-warp control strip, a
per-body info readout), not a literal skin to imitate.

Layout of the package:

* state.py        -- MissionState: the loaded scene, current time and body
                     selection, shared by every panel through Qt signals.
* sidebar.py      -- the mission list (left).
* candidate_panel.py -- the optimizer's candidate trajectories (left).
* inspector.py    -- measure readout, maneuver list and body cards (right).
* timeline_bar.py -- play / time-warp / event scrubber (bottom).
* view_strip.py   -- which body the view is centered on, and body scale.
* verification_panel.py -- the scene's verification reports (right, 2nd tab).
* charts.py       -- linked 2D charts under the 3D view (porkchop, distance).
* viewport.py     -- the embedded VTK view; compute.py -- background loads.

This module only assembles those pieces and owns what is specific to the
3D view: camera presets, tracking/face-lock/focus, viewport picking, the
maneuver arrow, and the playback timer.

Run: orbitopt  (or orbitopt-app / python -m orbitopt); main() is the entry point.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QSize, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from orbitopt.missions import high_fidelity_loader, list_missions, verification_off_note
from orbitopt.viz import icons
from orbitopt.viz.candidate_panel import CandidatePanel
from orbitopt.viz.charts import ChartPanel
from orbitopt.viz.flight_labels import label_text, labelled_bodies
from orbitopt.viz.compute import BackendProbeWorker, ComputePool, SceneLoader
from orbitopt.viz.source_watch import SourceWatch
from orbitopt.viz.icon import app_icon
from orbitopt.viz.inspector import InspectorPanel
from orbitopt.viz.pv_viewer import load_scene
from orbitopt.viz.scene_renderer import SceneRenderer
from orbitopt.viz.sidebar import MissionSidebar
from orbitopt.viz.state import (
    MissionState,
    adjacent_event,
    altitude_reference,
    eased_speed,
    maneuver_windows,
    nearest_maneuver,
    timeline_span,
)
from orbitopt.viz.theme import BG_VOID, DANGER, STYLESHEET, SUCCESS
from orbitopt.viz.timeline_bar import TimelineBar, timeline_unit_seconds
from orbitopt.viz.ui_kit import MetricRow, StatusBadge
from orbitopt.viz.verification_panel import (
    VerificationPanel,
    display_subtitle,
    selection_badge_state,
    verification_badge_state,
)
from orbitopt.viz.view_strip import ViewStrip
from orbitopt.viz.viewport import DebouncedInteractor
from orbitopt.viz.widgets import SpinnerWidget, ViewControlsPopover

# Minimum spacing between two timeline-scrub scene refreshes (~60 Hz). A
# slider drag emits valueChanged once per mouse-move event -- often several
# hundred Hz on a high-rate mouse -- and every extra refresh beyond what the
# display can show is pure CPU/GPU cost that made fast scrubbing stutter.
SCRUB_MIN_INTERVAL_MS = 16.0

WEBSITE_URL = "https://zangjiucheng.github.io/orbitopt/"
REPOSITORY_URL = "https://github.com/zangjiucheng/orbitopt"
FLIGHTLAB_SIDEBAR_KEY = "__orbitopt_flightlab__"
FLIGHTLAB_SIDEBAR_LABEL = "Flight Lab"

# Auto-slow near a timeline event (TLI burn, closest approach, entry interface,
# ...) so fast-forwarding through a mission doesn't blow past the moments that
# actually matter -- the window is a *fraction* of the mission's own timeline
# span (the same 5% that caps each burn's arrow window, see state.maneuver_windows),
# not a fixed day count, since a short and a long mission should each get a
# proportionally-sized "slow down here" zone. Playback eases toward
# EVENT_SLOWDOWN_TARGET_WARP, an absolute warp (simulated seconds per real
# second), not a ratio of the chosen warp -- see state.eased_speed for why.
# 1,000x is the KSP level closest to the old 0.02 day/s (1,728x).
EVENT_SLOWDOWN_WINDOW_FRACTION = 0.05
EVENT_SLOWDOWN_TARGET_WARP = 1_000.0

# Axial rotation is ambient motion for a scene with no timeline of its own
# (the Solar System view) -- driven by this continuous wall-clock timer (see
# SceneRenderer.advance_rotation), since there's no mission "current time"
# for it to track otherwise. A scene *with* a timeline (a mission) instead
# gets its rotation driven exactly by the mission's own current time (see
# SceneRenderer._apply_rotation_for_time, called from set_time()), so it
# stays in lockstep with scrubbing/pausing/time-warp speed instead of this
# constant rate; this timer's ticks are then a no-op for such a scene (see
# advance_rotation). 1 simulated hour per real second is a chosen, clearly-
# labelled acceleration (real-time rotation would be imperceptibly slow
# over a normal viewing session): Earth completes a visible spin in ~24s,
# Jupiter (fastest real rotator, ~10h) in ~10s, Venus/the Moon (both
# 500-6000h) stay visually still, same as they really would.
ROTATION_SIM_HOURS_PER_REAL_SECOND = 1.0

# Clicking a maneuver row auto-frames its burn: the camera zooms in on the
# spacecraft at the burn instant, to an ABSOLUTE distance of this multiple of
# the scene's maneuver-arrow length (_maneuver_base_len, itself a fraction of
# the orbit radius -- so the burn scale, shared by every event). Absolute, not
# a relative dolly-in, so repeated clicks land on the same framing instead of
# compounding. At 3x, the (camera-distance-clamped) arrow fills roughly half
# the view with the spacecraft centered -- close enough to read the maneuver,
# wide enough to keep some orbit context. Falls back to zoom_to_body's default
# relative zoom for a scene whose events carry no burn vectors.
EVENT_ZOOM_ARROW_MULTIPLE = 3.0


def _norm3(v) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


class MissionControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mission Control — orbitopt")
        self.setWindowIcon(app_icon())
        self._size_to_screen()
        self.setStyleSheet(STYLESHEET)

        # The scene, current time and body selection live here; every panel
        # subscribes to its signals (see state.py).
        self.state = MissionState(self)

        self._scene_cache: dict[str, dict] = {}
        # Sidebar key -> registry mission id (built-ins/plugins only; File >
        # Open scenes have none), the active key, and the keys whose loader
        # was switched to its high-fidelity-verification variant.
        self._mission_ids: dict[str, str] = {}
        self._active_key: str | None = None
        self._active_label: str | None = None
        self._flightlab_active = False
        self._verified_keys: set[str] = set()
        self._loaders: dict[str, callable] = {}
        self._playing = False
        self._play_direction = 1  # +1 forward, -1 backwards

        self._loading = False
        self._compute_pool = ComputePool()
        # closeEvent terminates the worker, but an app.quit() that never closes
        # this window (or a window that was never shown) would not; then
        # concurrent.futures' exit hook joins the executor and waits out a
        # possibly minutes-long compute. aboutToQuit covers that path.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._compute_pool.terminate)
        self._load_thread: QThread | None = None
        self._load_worker: SceneLoader | None = None
        self._load_start_ts: float = 0.0
        self._backend_probe_thread: QThread | None = None
        self._backend_probe_worker: BackendProbeWorker | None = None
        self._closing = False

        # 3D-view-only state (camera behavior, not mission data).
        self._viewport_press_pos: tuple[int, int] | None = None
        self._tracking = False
        self._lock_reference_body_id: str | None = None
        self._view_axis: str | None = None
        self._view_negative = False
        self._focus_zoomed = False
        self._pre_focus_camera: tuple | None = None
        self._maneuver_base_len = 0.0
        self._maneuver_max_dv = 1.0
        self._maneuver_windows: dict[int, float] = {}

        self._timer = QTimer(self)
        # ~60 fps playback. It was 33 ms (30 fps) while each tick drew the
        # view 3-4 times; with renders coalesced (viewport.render) a tick
        # costs well under a frame. A PreciseTimer: the default coarse timer
        # may fire up to 5% late, which shows as uneven frame pacing.
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)
        self._last_tick_ms = None
        # Set only while _on_tick is moving time, so _on_time_changed knows to
        # refresh immediately (with this dt for camera-tracking smoothing)
        # instead of going through the scrub throttle.
        self._tick_dt: float | None = None

        # Scrub render throttle (see _on_time_changed): a drag can move time
        # far faster than the screen refreshes, so at most one scene
        # refresh+render runs per SCRUB_MIN_INTERVAL_MS; anything in between
        # is coalesced into one trailing refresh at the latest time.
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._apply_scrub)
        self._last_scrub_render_ms = float("-inf")

        # Always-on (not gated by _playing/timeline presence, see
        # ROTATION_SIM_HOURS_PER_REAL_SECOND) -- started once here, stopped
        # only in closeEvent.
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setInterval(33)
        self._rotation_timer.timeout.connect(self._on_rotation_tick)
        self._last_rotation_tick_ms = None
        self._rotation_timer.start()

        self._build_ui()
        self.state.time_changed.connect(self._on_time_changed)

        # Streamed Earth imagery: re-pick tiles for the camera and show the
        # ones that finished downloading. Cheap when nothing changed (see
        # tile_layer.EarthTileLayer.update), so it simply polls.
        self._tile_timer = QTimer(self)
        self._tile_timer.setInterval(200)
        self._tile_timer.timeout.connect(self._on_tile_tick)
        self._tile_timer.start()

        # Hot reload: missions are computed by a worker process that imported
        # the code once, so an edited source file changes nothing until that
        # worker is replaced. Poll for edits and offer a reload (title bar
        # button, Cmd/Ctrl+R) -- never recompute on our own, a Starship
        # compute takes minutes.
        self._source_watch = SourceWatch()
        self._source_changed = False
        self._source_change_text = ""
        # Sidebar keys whose scene is a shipped precomputed one made by older
        # code (scene_cache.SHIPPED_STALE_KEY): the same button offers to
        # recompute them.
        self._stale_keys: set[str] = set()
        self._source_timer = QTimer(self)
        self._source_timer.setInterval(1500)
        self._source_timer.timeout.connect(self._check_source)
        self._source_timer.start()
        self.state.selection_changed.connect(self._update_measure)
        self.state.frame_changed.connect(self._on_frame_changed)
        self.state.candidate_changed.connect(self._on_candidate_changed)
        self._populate_builtin_missions()
        self._start_backend_probe()

    def _size_to_screen(self):
        """Open at a comfortable size that always fits the current display.

        The old hard-coded 1500x950 was taller than a laptop's usable height
        (menu bar + Dock eat into it), so the window opened partly off-screen.
        availableGeometry() already excludes the macOS menu bar and Dock, so
        clamping to a fraction of it -- and centering within it -- keeps the
        whole window on screen on any monitor while still preferring a roomy
        default on large ones.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1360, 820)
            return
        avail = screen.availableGeometry()
        width = min(1500, int(avail.width() * 0.9))
        height = min(900, int(avail.height() * 0.9))
        self.resize(width, height)
        self.move(
            avail.x() + (avail.width() - width) // 2,
            avail.y() + (avail.height() - height) // 2,
        )

    # ------------------------------------------------------------------ UI
    _RAIL_W = 30          # width of a collapsed panel's rail strip
    _LEFT_MIN = 170       # min drag width of the missions panel
    _RIGHT_MIN = 200      # min drag width of the body-info panel

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # A splitter (not a fixed HBox) makes both side panels drag-resizable.
        # Each side is a QStackedWidget that flips between the full panel and a
        # thin rail carrying an expand button -- so the collapse/expand control
        # lives on the panel itself, and a collapsed panel still leaves a
        # handle to bring it back.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(4)

        self.sidebar = MissionSidebar()
        self.sidebar.mission_activated.connect(self._on_mission_activated)
        self.sidebar.open_file_clicked.connect(self._open_file)
        self.sidebar.collapse_clicked.connect(self._toggle_sidebar)
        self.candidate_panel = CandidatePanel(self.state)
        self.candidate_panel.show_all_box.toggled.connect(lambda _on: self._apply_candidate_highlight())
        self.sidebar.add_section(self.candidate_panel)
        self.left_rail = self._build_rail(icons.chevron_right_icon, self._toggle_sidebar, "Show missions")
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.sidebar)
        self.left_stack.addWidget(self.left_rail)
        self.left_stack.setMinimumWidth(self._LEFT_MIN)
        self.splitter.addWidget(self.left_stack)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self._build_title_bar())
        self.view_strip = ViewStrip(self.state)
        center_layout.addWidget(self.view_strip)

        self.plotter = DebouncedInteractor(center)
        self.plotter.set_background(BG_VOID)
        self.plotter.enable_anti_aliasing()
        self.renderer = SceneRenderer(self.plotter)
        self._install_viewport_picking()

        # The 3D view and a loading page share one slot. We swap to the loading
        # page (rather than overlaying a Qt widget on top of the view) because
        # the embedded VTK surface is a native GL context Qt's compositor
        # doesn't reliably draw sibling widgets over -- the same reason
        # QWidget.grab() comes back black for it (see README). A QStackedWidget
        # swap composites cleanly.
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.plotter)
        self.center_stack.addWidget(self._build_loading_page())
        self.flightlab_page = self._build_flightlab_page()
        self._flightlab_index = self.center_stack.addWidget(self.flightlab_page)

        # The 3D view over the linked 2D charts; the splitter lets the charts
        # be dragged smaller, never the 3D view away.
        self.chart_panel = ChartPanel(self.state)
        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.setObjectName("centerSplitter")
        self.center_splitter.setHandleWidth(4)
        self.center_splitter.addWidget(self.center_stack)
        self.center_splitter.addWidget(self.chart_panel)
        self.center_splitter.setCollapsible(0, False)
        self.center_splitter.setStretchFactor(0, 3)
        self.center_splitter.setStretchFactor(1, 1)
        self.center_splitter.setSizes([560, 210])
        center_layout.addWidget(self.center_splitter, stretch=1)

        self.timeline_bar = TimelineBar(self.state)
        self.timeline_bar.play_clicked.connect(lambda: self._toggle_play(+1))
        self.timeline_bar.reverse_clicked.connect(lambda: self._toggle_play(-1))
        self.timeline_bar.auto_slow_toggled.connect(self._on_auto_slow_toggled)
        self.timeline_bar.event_step_requested.connect(self._seek_relative_event)
        # Dragging the playhead pauses: playback kept moving it out from under
        # the pointer. It stays paused when the drag ends.
        self.timeline_bar.scrub_started.connect(lambda: self._playing and self._set_playing(False))
        center_layout.addWidget(self.timeline_bar)
        self.splitter.addWidget(center)

        self.inspector = InspectorPanel(self.state)
        self.inspector.body_clicked.connect(self._on_card_clicked)
        self.inspector.event_clicked.connect(self._seek_to_event)
        self.verification_panel = VerificationPanel(self.state)
        self.verification_panel.verify_requested.connect(self._run_high_fidelity_verification)
        # Inspector and Verification share the right column as tabs; the
        # collapse chevron moves to the tab bar's corner so it serves both.
        self.inspector.collapse_btn.hide()
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.setDocumentMode(True)
        # Without these the native tab-bar base paints a light strip past the
        # last tab, ignoring the dark stylesheet.
        self.right_tabs.tabBar().setDrawBase(False)
        self.right_tabs.setAttribute(Qt.WA_StyledBackground, True)
        self.right_tabs.addTab(self.inspector, "Inspector")
        self.right_tabs.addTab(self.verification_panel, "Verification")
        collapse = QPushButton()
        collapse.setObjectName("panelCollapse")
        collapse.setIcon(icons.chevron_right_icon())
        collapse.setIconSize(QSize(12, 12))
        collapse.setToolTip("Collapse the right panel")
        collapse.clicked.connect(self._toggle_info_panel)
        self.right_tabs.setCornerWidget(collapse, Qt.TopRightCorner)
        self.right_rail = self._build_rail(icons.chevron_left_icon, self._toggle_info_panel, "Show body info")
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self.right_tabs)
        self.right_stack.addWidget(self.right_rail)
        self.right_stack.setMinimumWidth(self._RIGHT_MIN)
        self.splitter.addWidget(self.right_stack)

        # Only the center grows when the window resizes; the panels keep width.
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([220, 900, 260])
        self._left_w = 220
        self._right_w = 260

        root.addWidget(self.splitter)
        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_rail(self, icon_fn, handler, tip: str) -> QWidget:
        """A thin vertical strip shown when a panel is collapsed: just an expand
        button, so the panel can be summoned back from its own edge. ``icon_fn``
        is one of icons.py's chevron_*_icon functions, not a text glyph -- see
        icons.py's module docstring for why."""
        rail = QWidget()
        rail.setObjectName("railBar")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(3, 9, 3, 9)
        layout.setSpacing(0)
        btn = QPushButton()
        btn.setObjectName("railButton")
        btn.setIcon(icon_fn())
        btn.setIconSize(QSize(13, 13))
        btn.setToolTip(tip)
        btn.clicked.connect(handler)
        layout.addWidget(btn, alignment=Qt.AlignHCenter | Qt.AlignTop)
        layout.addStretch(1)
        return rail

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 12, 8)
        layout.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self.scene_title_label = QLabel("Select a mission")
        self.scene_title_label.setObjectName("sceneTitle")
        self.scene_subtitle_label = QLabel("")
        self.scene_subtitle_label.setObjectName("sceneSubtitle")
        self.scene_subtitle_label.setWordWrap(True)
        title_col.addWidget(self.scene_title_label)
        title_col.addWidget(self.scene_subtitle_label)
        layout.addLayout(title_col, stretch=1)

        layout.addWidget(self._build_compute_status(), alignment=Qt.AlignVCenter)
        layout.addWidget(self._build_view_controls(), alignment=Qt.AlignVCenter)

        return bar

    def _build_compute_status(self) -> QWidget:
        """Real-state badges only, no fabricated telemetry: this app runs one
        blocking compute-and-export call per mission on a background thread
        (see _start_loading), not a live generation-by-generation optimizer
        loop, so there's no genuine eval-rate/generation counter to show.
        What *is* real: whether a scene is idle/loading/ready/failed, how
        long the last load took, and a one-time CPU/GPU capability probe
        (orbitopt.core.gpu.GPU_AVAILABLE, see BackendProbeWorker)."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        self.load_status_badge = StatusBadge("idle", "IDLE")
        self.load_status_badge.setToolTip("Current mission scene load state")
        row.addWidget(self.load_status_badge)

        # Shown once orbitopt's source changed on disk (see _check_source).
        self.reload_button = QPushButton("Code changed · Reload")
        self.reload_button.setObjectName("reloadBanner")
        self.reload_button.clicked.connect(self._reload_mission)
        self.reload_button.setVisible(False)
        row.addWidget(self.reload_button)

        # Always on screen for a mission, so "not checked" can't be mistaken
        # for verified; click it for the reports.
        self.verification_badge = StatusBadge("idle", "")
        self.verification_badge.setToolTip("High-fidelity verification of this trajectory -- click for the reports")
        self.verification_badge.mousePressEvent = lambda _e: self._show_verification_tab()
        self.verification_badge.setVisible(False)
        row.addWidget(self.verification_badge)

        # Shown only while a candidate other than the exported trajectory is
        # selected: its arc is screening fidelity and was never verified, so
        # the verification badge beside it must not read as its status.
        self.selection_badge = StatusBadge("warning", "")
        self.selection_badge.mousePressEvent = lambda _e: self._show_verification_tab()
        self.selection_badge.setVisible(False)
        row.addWidget(self.selection_badge)

        self.backend_badge = StatusBadge("idle", "…")
        self.backend_badge.setToolTip(
            "Whether this environment has a usable CUDA/CuPy backend "
            "(orbitopt.core.gpu) -- probed once at startup"
        )
        row.addWidget(self.backend_badge)

        self.load_time_metric = MetricRow("Load", "—", "s")
        self.load_time_metric.setToolTip("Wall-clock time for the most recent mission load")
        row.addWidget(self.load_time_metric)

        return box

    def _build_view_controls(self) -> QWidget:
        """KSP-style camera view presets -- snap the camera to a fixed
        orthographic angle without hunting for it by dragging. Pressing the
        same preset a second time flips it to the opposite side (see
        _apply_axis_view) -- Top a second time looks up from below, etc."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        label = QLabel("VIEW")
        label.setObjectName("viewLabel")
        row.addWidget(label)

        for text, handler, tip in (
            ("Top", self._view_top, "Top-down (look along −Z); press again to flip to bottom-up"),
            ("Front", self._view_front, "Front (look along −Y); press again to flip to back"),
            ("Side", self._view_side, "Side (look along −X); press again to flip to the other side"),
            ("Iso", self._view_iso, "Isometric; press again to flip to the opposite corner"),
        ):
            btn = QPushButton(text)
            btn.setObjectName("viewButton")
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            row.addWidget(btn)

        sep = QFrame()
        sep.setObjectName("viewSep")
        sep.setFrameShape(QFrame.VLine)
        row.addWidget(sep)

        self.track_btn = QPushButton("Track")
        self.track_btn.setObjectName("viewButton")
        self.track_btn.setCheckable(True)
        self.track_btn.setToolTip("Keep the selected body centered as time advances (T)")
        self.track_btn.toggled.connect(self._set_tracking)
        row.addWidget(self.track_btn)

        help_sep = QFrame()
        help_sep.setObjectName("viewSep")
        help_sep.setFrameShape(QFrame.VLine)
        row.addWidget(help_sep)

        # Less-frequently-used view settings (face-lock reference, reset
        # camera, panel visibility) live in a compact popover behind this
        # gear icon instead of sitting inline -- keeps the header lean per
        # the design brief ("控件尽量不切割主视图" / low chrome), while Top/
        # Front/Side/Iso/Track above stay inline since camera navigation is
        # frequent enough to earn permanent screen space.
        self.view_controls_popover = self._build_view_controls_popover()
        self.view_controls_btn = QPushButton()
        self.view_controls_btn.setObjectName("pixelIconButton")
        self.view_controls_btn.setIcon(icons.gear_icon())
        self.view_controls_btn.setIconSize(QSize(13, 13))
        self.view_controls_btn.setToolTip("View controls — face-lock, reset camera, panels")
        self.view_controls_btn.clicked.connect(self._toggle_view_controls_popover)
        row.addWidget(self.view_controls_btn)

        self.shortcuts_btn = QPushButton()
        self.shortcuts_btn.setObjectName("helpButton")
        self.shortcuts_btn.setIcon(icons.keyboard_icon())
        self.shortcuts_btn.setIconSize(QSize(13, 13))
        self.shortcuts_btn.setToolTip("Keyboard shortcuts (F1)")
        self.shortcuts_btn.clicked.connect(self._show_shortcuts_dialog)
        row.addWidget(self.shortcuts_btn)
        return box

    def _build_view_controls_popover(self) -> ViewControlsPopover:
        popover = ViewControlsPopover(self)
        # Qt.Popup makes this its own top-level window, which (like
        # _show_shortcuts_dialog's QDialog) does not inherit self's
        # stylesheet automatically -- set it explicitly.
        popover.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout(popover)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        cam_label = QLabel("CAMERA")
        cam_label.setObjectName("menuSectionLabel")
        layout.addWidget(cam_label)

        face_row = QHBoxLayout()
        face_label = QLabel("FACE LOCK")
        face_label.setObjectName("viewLabel")
        face_row.addWidget(face_label)
        self.lock_combo = QComboBox()
        self.lock_combo.setObjectName("lockCombo")
        self.lock_combo.setToolTip(
            "While tracking, keep this body lined up behind the tracked body "
            "instead of an arbitrary offset direction -- e.g. track a "
            "spacecraft, lock to Earth, and Earth stays framed as the "
            "spacecraft moves"
        )
        self.lock_combo.addItem("None", None)
        self.lock_combo.currentIndexChanged.connect(self._on_lock_reference_changed)
        face_row.addWidget(self.lock_combo, stretch=1)
        layout.addLayout(face_row)

        reset_btn = QPushButton("Reset camera (R)")
        reset_btn.setObjectName("viewButton")
        reset_btn.clicked.connect(self._reset_camera_from_popover)
        layout.addWidget(reset_btn)

        sep = QFrame()
        sep.setObjectName("viewSep")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        panels_label = QLabel("PANELS")
        panels_label.setObjectName("menuSectionLabel")
        layout.addWidget(panels_label)

        rail_btn = QPushButton("Toggle mission rail")
        rail_btn.setObjectName("viewButton")
        rail_btn.clicked.connect(lambda: (self._toggle_sidebar(), popover.close()))
        layout.addWidget(rail_btn)

        inspector_btn = QPushButton("Toggle inspector (I)")
        inspector_btn.setObjectName("viewButton")
        inspector_btn.clicked.connect(lambda: (self._toggle_info_panel(), popover.close()))
        layout.addWidget(inspector_btn)

        popover.setFixedWidth(210)
        return popover

    def _reset_camera_from_popover(self):
        self.plotter.reset_camera()
        self.plotter.render()
        self.view_controls_popover.close()

    def _toggle_view_controls_popover(self):
        popover = self.view_controls_popover
        if popover.isVisible():
            popover.close()
            return
        btn = self.view_controls_btn
        anchor = btn.mapToGlobal(btn.rect().bottomRight())
        popover.adjustSize()
        anchor.setX(anchor.x() - popover.width())
        popover.move(anchor)
        popover.show()

    def _build_loading_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("loadingPage")
        layout = QVBoxLayout(page)
        layout.addStretch(1)

        spinner_row = QHBoxLayout()
        spinner_row.addStretch(1)
        self._spinner = SpinnerWidget(diameter=68)
        spinner_row.addWidget(self._spinner)
        spinner_row.addStretch(1)
        layout.addLayout(spinner_row)

        self._loading_label = QLabel("Loading…")
        self._loading_label.setObjectName("loadingLabel")
        self._loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._loading_label)

        layout.addStretch(1)
        return page

    def _build_flightlab_page(self) -> QWidget:
        from orbitopt.flightlab.ui import FlightLabReplayPage

        return FlightLabReplayPage()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("Open scene file…")
        open_action.triggered.connect(self._open_file)
        self.reload_action = file_menu.addAction("Reload mission with edited code")
        self.reload_action.setShortcut(QKeySequence("Ctrl+R"))  # Cmd+R on macOS
        self.reload_action.triggered.connect(self._reload_mission)
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)  # Cmd+Q / Ctrl+Q
        quit_action.triggered.connect(self.close)

        play_menu = self.menuBar().addMenu("&Playback")
        self.play_action = play_menu.addAction("Play / Pause", lambda: self._toggle_play(+1))
        self.play_action.setShortcut(QKeySequence(Qt.Key_Space))
        play_menu.addAction("Play backwards / Pause", lambda: self._toggle_play(-1)).setShortcut(
            QKeySequence("Shift+Space"))
        play_menu.addAction("Step forward", lambda: self._step_time(+1)).setShortcut(QKeySequence(Qt.Key_Right))
        play_menu.addAction("Step back", lambda: self._step_time(-1)).setShortcut(QKeySequence(Qt.Key_Left))
        play_menu.addAction("Step forward (fine)", lambda: self._step_time(+0.1)).setShortcut(QKeySequence("Alt+Right"))
        play_menu.addAction("Step back (fine)", lambda: self._step_time(-0.1)).setShortcut(QKeySequence("Alt+Left"))
        play_menu.addAction("Restart", self._restart_timeline).setShortcut(QKeySequence(Qt.Key_Home))
        play_menu.addSeparator()
        # KSP's warp keys: . / , step through the warp levels, / is real time.
        play_menu.addAction("Warp up", lambda: self.timeline_bar.step_warp(+1)).setShortcut(QKeySequence("."))
        play_menu.addAction("Warp down", lambda: self.timeline_bar.step_warp(-1)).setShortcut(QKeySequence(","))
        play_menu.addAction("Real time (1×)", lambda: self.timeline_bar.set_warp(1.0)).setShortcut(QKeySequence("/"))
        play_menu.addAction("Faster (fine)", lambda: self.timeline_bar.nudge_speed(+1)).setShortcut(QKeySequence("]"))
        play_menu.addAction("Slower (fine)", lambda: self.timeline_bar.nudge_speed(-1)).setShortcut(QKeySequence("["))
        play_menu.addSeparator()
        next_event_action = play_menu.addAction("Next event", lambda: self._seek_relative_event(+1))
        next_event_action.setShortcut(QKeySequence("Shift+Right"))
        prev_event_action = play_menu.addAction("Previous event", lambda: self._seek_relative_event(-1))
        prev_event_action.setShortcut(QKeySequence("Shift+Left"))
        play_menu.addSeparator()
        # The zoomable timeline track (timeline_bar.TimelineTrack).
        play_menu.addAction("Zoom timeline in", lambda: self.timeline_bar.track.zoom(2)).setShortcut(QKeySequence("="))
        play_menu.addAction("Zoom timeline out", lambda: self.timeline_bar.track.zoom(0.5)).setShortcut(QKeySequence("-"))
        play_menu.addAction("Fit whole mission", lambda: self.timeline_bar.track.fit()).setShortcut(QKeySequence("0"))
        play_menu.addAction("Go to time…", self.timeline_bar.focus_time_box).setShortcut(QKeySequence("G"))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("Top view", self._view_top)
        view_menu.addAction("Front view", self._view_front)
        view_menu.addAction("Side view", self._view_side)
        view_menu.addAction("Isometric view", self._view_iso)
        reset_action = view_menu.addAction("Reset camera")
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        view_menu.addSeparator()
        focus_action = view_menu.addAction("Focus selected body", self._focus_selected)
        focus_action.setShortcut(QKeySequence("M"))
        self.track_action = view_menu.addAction("Track selected body")
        self.track_action.setCheckable(True)
        self.track_action.setShortcut(QKeySequence("T"))
        self.track_action.toggled.connect(self._set_tracking)
        # KSP-style flight data (height, speed, delta-v) next to each vehicle.
        self.labels_action = view_menu.addAction("Flight data labels")
        self.labels_action.setCheckable(True)
        self.labels_action.setChecked(True)
        self.labels_action.setShortcut(QKeySequence("L"))
        self.labels_action.toggled.connect(lambda _on: self._refresh_view())
        self.imagery_action = view_menu.addAction("High-res Earth imagery (EOX Sentinel-2)")
        self.imagery_action.setCheckable(True)
        self.imagery_action.setChecked(self.renderer.tiles.enabled)
        self.imagery_action.setToolTip("Stream 10 m satellite imagery for the part of Earth in view "
                                       "(downloads tiles, cached on disk)")
        self.imagery_action.toggled.connect(self._on_imagery_toggled)
        # KSP map view: Tab / Shift+Tab cycle the focus target.
        focus_next = view_menu.addAction("Focus next (Tab)", lambda: self._cycle_focus(+1))
        focus_next.setShortcut(QKeySequence(Qt.Key_Tab))
        focus_next.setShortcutContext(Qt.ApplicationShortcut)
        focus_prev = view_menu.addAction("Focus previous (⇧Tab)", lambda: self._cycle_focus(-1))
        focus_prev.setShortcut(QKeySequence(Qt.SHIFT | Qt.Key_Backtab))
        focus_prev.setShortcutContext(Qt.ApplicationShortcut)
        view_menu.addSeparator()
        view_menu.addAction("Toggle missions panel", self._toggle_sidebar)
        info_toggle_action = view_menu.addAction("Toggle body-info panel", self._toggle_info_panel)
        info_toggle_action.setShortcut(QKeySequence("I"))

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts_action = help_menu.addAction("Keyboard Shortcuts…", self._show_shortcuts_dialog)
        shortcuts_action.setShortcut(QKeySequence(Qt.Key_F1))
        help_menu.addSeparator()
        self.guide_action = help_menu.addAction("User Guide", lambda: self._open_help_link("guide"))
        help_menu.addAction("orbitopt Website", lambda: self._open_help_link("website"))
        help_menu.addAction("README on GitHub", lambda: self._open_help_link("readme"))

    # ----------------------------------------------------------------- Help
    # Every entry here matches an actual wired shortcut (see _build_menu
    # above) or a documented mouse gesture (see _on_card_clicked) -- kept as
    # one static list so this panel can't silently drift out of sync with
    # what the app actually does.
    _SHORTCUTS = [
        ("Space / ⇧Space", "Play / play backwards (the same key again pauses)"),
        ("← / →", "Step back / forward (1% of the visible timeline)"),
        ("Alt+← / →", "Fine step (0.1% of the visible timeline)"),
        ("Shift-drag on timeline", "Scrub 10× finer; drag the playhead to nudge it"),
        ("Shift+← / →", "Previous / next mission event"),
        ("Home", "Restart timeline"),
        ("= / - / 0", "Zoom the timeline in / out / fit the whole mission"),
        ("G", "Go to time: type T+22:32:34, +30s, 6m35s or a UTC date"),
        ("Scroll on timeline", "Zoom about the cursor (sideways / right-drag: pan)"),
        (", / .", "Time warp down / up a KSP level (1×, 5×, 10×, 50×, … 10M×)"),
        ("/", "Back to real time (1×)"),
        ("[ / ]", "Fine-tune time warp"),
        ("Tab / ⇧Tab", "Cycle focus: spacecraft, then bodies (KSP map view)"),
        ("T", "Track selected body"),
        ("L", "Flight data labels (height, speed, Δv) on / off"),
        ("M", "Focus (zoom to) selected body"),
        ("R", "Reset camera"),
        ("I", "Toggle inspector (body-info) panel"),
        ("F1", "Show this panel"),
        ("Ctrl+R / ⌘R", "Reload the mission with edited code (new compute worker)"),
        ("Ctrl+Q / ⌘Q", "Quit"),
        ("Click", "Select a body"),
        ("Ctrl/⌘-click", "Measure distance to another body"),
    ]

    def help_url(self, which: str) -> QUrl:
        """Where Help > User Guide / Website / README points: the guide is
        read from the source checkout when there is one (it works offline
        and matches this version), else from the website."""
        if which == "guide":
            local = Path(__file__).resolve().parents[3] / "docs" / "guide.html"
            if local.is_file():
                return QUrl.fromLocalFile(str(local))
            return QUrl(f"{WEBSITE_URL}guide.html")
        if which == "website":
            return QUrl(WEBSITE_URL)
        return QUrl(f"{REPOSITORY_URL}/blob/main/README.md")

    def _open_help_link(self, which: str) -> None:
        url = self.help_url(which)
        if not QDesktopServices.openUrl(url):
            self.statusBar().showMessage(f"Could not open {url.toString()}")

    def _show_shortcuts_dialog(self):
        """A QDialog is its own top-level window, so it does not inherit
        self's stylesheet automatically -- set it explicitly so this panel
        matches the rest of the app instead of falling back to the OS
        default look. Built once and reused (cached on
        self._shortcuts_dialog) since its content is static."""
        if getattr(self, "_shortcuts_dialog", None) is None:
            dialog = QDialog(self)
            dialog.setObjectName("shortcutsDialog")
            dialog.setWindowTitle("Keyboard Shortcuts")
            dialog.setStyleSheet(STYLESHEET)

            outer = QVBoxLayout(dialog)
            outer.setContentsMargins(22, 20, 22, 20)
            outer.setSpacing(14)

            title = QLabel("KEYBOARD SHORTCUTS")
            title.setObjectName("sectionHeader")
            outer.addWidget(title)

            grid = QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(8)
            for row, (key, desc) in enumerate(self._SHORTCUTS):
                cap = QLabel(key)
                cap.setObjectName("keyCap")
                cap.setAlignment(Qt.AlignCenter)
                grid.addWidget(cap, row, 0)
                label = QLabel(desc)
                label.setObjectName("shortcutDesc")
                grid.addWidget(label, row, 1)
            outer.addLayout(grid)

            close_btn = QPushButton("Close")
            close_btn.setObjectName("viewButton")
            close_btn.clicked.connect(dialog.close)
            close_row = QHBoxLayout()
            close_row.addStretch(1)
            close_row.addWidget(close_btn)
            outer.addLayout(close_row)

            self._shortcuts_dialog = dialog
        self._shortcuts_dialog.show()
        self._shortcuts_dialog.raise_()
        self._shortcuts_dialog.activateWindow()

    # -------------------------------------------------------- Panels / camera
    def _toggle_sidebar(self):
        self._toggle_side(self.left_stack, "_left_w", 0, self._LEFT_MIN)

    def _toggle_info_panel(self):
        self._toggle_side(self.right_stack, "_right_w", 2, self._RIGHT_MIN)

    def _toggle_side(self, stack: QStackedWidget, width_attr: str, index: int, min_w: int):
        """Collapse a splitter side to its rail, or expand it back. Collapsing
        remembers the current drag width; expanding restores it and takes the
        space back from the center view."""
        if stack.currentIndex() == 0:
            sizes = self.splitter.sizes()
            setattr(self, width_attr, max(sizes[index], min_w))
            stack.setCurrentIndex(1)
            stack.setFixedWidth(self._RAIL_W)
            # QSplitter won't reflow just because a child's max width changed --
            # hand the freed space to the center view explicitly.
            sizes[1] += sizes[index] - self._RAIL_W
            sizes[index] = self._RAIL_W
            self.splitter.setSizes(sizes)
        else:
            stack.setMaximumWidth(16777215)
            stack.setMinimumWidth(min_w)
            stack.setCurrentIndex(0)
            sizes = self.splitter.sizes()
            target = getattr(self, width_attr)
            sizes[1] = max(200, sizes[1] - (target - sizes[index]))
            sizes[index] = target
            self.splitter.setSizes(sizes)

    def _view_top(self):
        self._apply_axis_view("xy")

    def _view_front(self):
        self._apply_axis_view("xz")

    def _view_side(self):
        self._apply_axis_view("yz")

    def _view_iso(self):
        self._apply_axis_view("isometric")

    def _apply_axis_view(self, axis: str):
        """Snap to a canonical camera preset along ``axis`` (one of the
        pyvista Plotter.view_{xy,xz,yz,isometric} names) -- each of those
        only ever looks from one fixed direction (e.g. Top is always
        look-down-from-above), so on its own there's no way to see the
        opposite side (e.g. a look-up-from-below "Bottom" view) short of
        manually dragging there. Pressing the *same* preset again instead
        flips to its negative direction (pyvista's own `negative=` param on
        each view_* method) -- Top/Top flips between above and below,
        Front/Front between front and back, Side/Side between the two
        sides, Iso/Iso between the two isometric corners -- so every preset
        button doubles as its own up-down/front-back/left-right flip.
        Picking a *different* preset always starts from its canonical
        (non-flipped) side, regardless of whatever was flipped before.
        """
        if self._view_axis == axis:
            self._view_negative = not self._view_negative
        else:
            self._view_axis = axis
            self._view_negative = False
        getattr(self.plotter, f"view_{axis}")(negative=self._view_negative)
        self.plotter.render()

    def _on_card_clicked(self, body_id: str, measure: bool):
        """Plain click selects (focus/track anchor); ⌘/Ctrl-click picks the
        second body to measure a distance to."""
        if measure:
            self.state.toggle_measure_partner(body_id)
        else:
            self._select_body(body_id)

    def _select_body(self, body_id: str):
        """Select a body: remember it as the focus/track target (the inspector
        highlights its card from state.selection_changed) and center the
        camera on it once. A plain select also clears any pending distance
        measurement."""
        # A plain select recenters but doesn't zoom (that's what M is for) --
        # if a previous M-zoom is still active for a different body, drop it
        # rather than leave a stale "press M to restore" state pointing at
        # a camera position framed for a body that's no longer selected.
        self._focus_zoomed = False
        self._pre_focus_camera = None
        self.state.select_body(body_id)
        self.renderer.track_body(body_id, self.state.time, reference_body_id=self._lock_reference_body_id)
        self.statusBar().showMessage(
            f"Focused {self.state.body(body_id)['name']} — ⌘/Ctrl-click another to measure"
        )

    def _update_measure(self, render: bool = True):
        """Redraw the measurement line + readout for the current (A, B) pair, or
        clear it if a full pair isn't selected. Called on selection changes and
        every timeline step so the distance tracks the bodies as they move.
        ``render=False`` for the hot playback-tick path -- see _refresh_view,
        which batches its scene mutations into one render instead of one each."""
        a, b = self.state.selected_body_id, self.state.measure_body_id
        if a and b:
            dist = self.renderer.measure_line(a, b, self.state.time, render=render)
            self.inspector.show_measure(dist)
        else:
            self.renderer.clear_measure_line(render=render)
            self.inspector.hide_measure()

    def _focus_selected(self):
        """KSP-style focus (the M shortcut): the first press zooms the camera
        in on the selected body, remembering exactly where the camera was so
        a second press zooms back out to that same view -- not a fixed
        reset, an actual toggle. Selecting a different body (see
        _select_body) or loading a new scene (see _apply_scene) drops the
        saved state, so M always starts a fresh zoom-in on whatever's
        currently selected rather than "restoring" an unrelated old view."""
        selected = self.state.selected_body_id
        if not selected:
            self.statusBar().showMessage("Select a body first — click its card")
            return
        if self._focus_zoomed:
            if self._pre_focus_camera is not None:
                position, focal_point, up = self._pre_focus_camera
                cam = self.plotter.camera
                cam.position = position
                cam.focal_point = focal_point
                cam.up = up
                self.plotter.render()
            self._focus_zoomed = False
            self._pre_focus_camera = None
        else:
            cam = self.plotter.camera
            self._pre_focus_camera = (tuple(cam.position), tuple(cam.focal_point), tuple(cam.up))
            self.renderer.zoom_to_body(selected, self.state.time)
            self._focus_zoomed = True

    # ------------------------------------------------------- Viewport click
    # Clicking a body directly in the 3D view (not just its card) selects and
    # focuses it in one gesture -- the brentmcnatt.com-style "click a planet"
    # interaction. This has to distinguish a genuine click from the start of
    # a click-and-drag camera orbit, since VTK's own left-button-press fires
    # immediately on press, before any drag happens. So picking is done by
    # hand, at release: record the press position, and only actually pick
    # (at the release position) if the pointer barely moved in between --
    # otherwise this was a drag, not a click, and does nothing.
    _VIEWPORT_CLICK_TOLERANCE_PX = 4

    def _install_viewport_picking(self):
        # "cell" (vtkCellPicker) does a geometric ray/mesh intersection test
        # against the actual actor geometry -- unlike "prop" (vtkPropPicker,
        # a hardware/selection-buffer pick), it doesn't depend on reading
        # back a rendered frame, so it isn't sensitive to render/composite
        # timing. Also gets pyvista's own default 2.5%-of-window tolerance
        # (see RenderWindowInteractor.picker's setter), which forgives a
        # near-miss on a small body the same way a real click naturally would.
        self.plotter.iren.picker = "cell"
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_viewport_left_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_viewport_left_release)
        # The burn arrow's length follows the camera distance; re-size it when
        # an interaction (orbit, zoom) ends, or a zoom while paused leaves the
        # previous, now-huge arrow filling the view.
        self.plotter.iren.add_observer("EndInteractionEvent", lambda *_a: self._resize_maneuver_arrow())
        self.plotter.iren.add_observer("MouseWheelForwardEvent", lambda *_a: self._resize_maneuver_arrow())
        self.plotter.iren.add_observer("MouseWheelBackwardEvent", lambda *_a: self._resize_maneuver_arrow())

    def _on_viewport_left_press(self, *_args):
        self._viewport_press_pos = self.plotter.iren.interactor.GetEventPosition()

    def _on_viewport_left_release(self, *_args):
        press_pos = self._viewport_press_pos
        self._viewport_press_pos = None
        if press_pos is None or self.state.scene is None:
            return
        x, y = self.plotter.iren.interactor.GetEventPosition()
        if (x - press_pos[0]) ** 2 + (y - press_pos[1]) ** 2 > self._VIEWPORT_CLICK_TOLERANCE_PX ** 2:
            return  # the pointer moved -- this was a camera-orbit drag, not a click
        picker = self.plotter.iren.picker
        picker.Pick(x, y, 0, self.plotter.renderer)
        body_id = self.renderer.body_id_for_actor(picker.GetActor())
        if self.state.body(body_id) is None:
            return
        if QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self.state.toggle_measure_partner(body_id)
            return
        # Same lightweight-select-then-toggle-focus composition as pressing M
        # after clicking a card, just in one gesture: selecting a *different*
        # body always focuses in (see _select_body resetting _focus_zoomed);
        # clicking the *same*, already-focused body again toggles back out,
        # matching the M shortcut's own zoom in/out toggle instead of
        # zooming in further on every repeat click.
        if body_id != self.state.selected_body_id:
            self._select_body(body_id)
        self._focus_selected()

    def _set_tracking(self, on: bool):
        """Turn continuous tracking on/off, keeping the button and menu item in
        sync (either can drive this)."""
        self._tracking = bool(on)
        for w in (self.track_btn, self.track_action):
            w.blockSignals(True)
            w.setChecked(self._tracking)
            w.blockSignals(False)
        if self._tracking:
            selected = self.state.selected_body_id
            if selected:
                self.renderer.track_body(selected, self.state.time, reference_body_id=self._lock_reference_body_id)
                self.statusBar().showMessage(f"Tracking {self.state.body(selected)['name']}")
            else:
                self.statusBar().showMessage("Select a body to track — click its card")
        else:
            self.statusBar().showMessage("Tracking off")

    def _on_lock_reference_changed(self, _index: int):
        """The FACE dropdown: lock the camera's offset direction so this body
        stays lined up behind whatever's selected (see SceneRenderer.track_body).
        Applies immediately -- once, to the current selection -- whether or not
        continuous Track is on, so e.g. picking "Earth" here re-orients an
        already M-zoomed view right away."""
        self._lock_reference_body_id = self.lock_combo.currentData()
        selected = self.state.selected_body_id
        if selected:
            self.renderer.track_body(selected, self.state.time, reference_body_id=self._lock_reference_body_id)

    def _update_tracking(self, dt: float | None = None, render: bool = True):
        """Follow the tracked body after a timeline change, if tracking is on.
        ``dt`` is passed through for smoothing on continuous playback ticks;
        omitted (snaps instantly) for a manual slider drag. ``render=False``
        for the hot playback-tick path -- see _refresh_view."""
        selected = self.state.selected_body_id
        if self._tracking and selected:
            self.renderer.track_body(
                selected, self.state.time, dt=dt,
                reference_body_id=self._lock_reference_body_id, render=render,
            )

    # ------------------------------------------------------------- Missions
    def _populate_builtin_missions(self):
        # Sourced from orbitopt.missions -- the registry shared with the CLI,
        # so a mission (built-in or a third-party plugin) added there shows up
        # here with no separate list to keep in sync.
        for mission in list_missions():
            self._loaders[mission.title] = mission.load
            self._mission_ids[mission.title] = mission.id
            self.sidebar.add_mission(mission.title, mission.title, mission.title)
        self.sidebar.add_mission(
            FLIGHTLAB_SIDEBAR_KEY,
            FLIGHTLAB_SIDEBAR_LABEL,
            "Open a precomputed Flight Lab run directory",
        )
        if self.sidebar.count():
            self.sidebar.select_row(0)

    def _open_file(self):
        if self._loading:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open scene JSON", "", "Scene files (*.json)")
        if not path:
            return
        # Keyed by the full absolute path, not just the basename: two different
        # scene files that happen to share a filename (e.g. from different
        # folders) would otherwise collide in _loaders/_scene_cache, with the
        # second load silently overwriting the first's cache entry. The list
        # item's *displayed* text stays the short filename -- only the lookup
        # key is the full path.
        key = os.path.abspath(path)
        self._loaders[key] = (lambda p=path: load_scene(p))
        self.sidebar.add_mission(key, os.path.basename(path), path)
        self.sidebar.select_row(self.sidebar.count() - 1)

    def _on_mission_activated(self, key: str, label: str):
        if self._loading:
            return
        self._set_playing(False)
        if key == FLIGHTLAB_SIDEBAR_KEY:
            self._active_key, self._active_label = key, label
            self._update_reload_button()
            self._show_flightlab(label)
            return
        self._hide_flightlab()
        self._active_key, self._active_label = key, label
        self._update_reload_button()

        if key in self._scene_cache:
            # A cached scene skips the ~10s compute but still stutters: applying
            # it runs SceneRenderer.load, which tears down and rebuilds every
            # VTK actor synchronously on the main thread. Show the loading page
            # first, then apply on the next event-loop turn so the page actually
            # paints before that blocking rebuild -- otherwise the swap and the
            # freeze happen in the same turn and the user sees only the freeze.
            self._begin_switch(f"Loading {label}…")
            scene = self._scene_cache[key]
            QTimer.singleShot(0, lambda: self._finish_cached(scene, key))
            return

        self._start_loading(key, label)

    def _show_flightlab(self, label: str | None = None):
        self._set_playing(False)
        self._flightlab_active = True
        self._active_key = FLIGHTLAB_SIDEBAR_KEY
        self._active_label = label or FLIGHTLAB_SIDEBAR_LABEL
        self._update_reload_button()
        self.chart_panel.hide()
        self.timeline_bar.hide()
        self.center_stack.setCurrentIndex(self._flightlab_index)
        self.statusBar().showMessage("Flight Lab replay")

    def _hide_flightlab(self):
        self._flightlab_active = False
        self.flightlab_page.stop_playback()
        if self.center_stack.currentWidget() is self.flightlab_page:
            self.center_stack.setCurrentIndex(0)
        self.chart_panel.setVisible(self.chart_panel.count() > 0)
        self.timeline_bar.setVisible((self.state.scene or {}).get("timeline") is not None)

    def _begin_switch(self, message: str):
        self._loading = True
        self._load_start_ts = time.perf_counter()
        self.load_status_badge.set_state("loading", "LOADING")
        self.sidebar.set_busy(True)
        # No wait cursor: on macOS 27 (PySide6 6.11 / VTK 9.6) setting any
        # cursor on a window that contains the pyvistaqt QtInteractor
        # segfaults -- QApplication.setOverrideCursor and widget.setCursor
        # alike -- and the first mission load runs from __init__, so the app
        # crashed on launch. The spinner overlay + LOADING badge + disabled
        # mission list already signal the busy state.
        self._show_loading(message)
        self.statusBar().showMessage(message)

    def _end_switch(self):
        self._hide_loading()
        self.sidebar.set_busy(False)
        self._loading = False
        self.statusBar().showMessage("Ready")

    def _show_loading(self, message: str):
        self._loading_label.setText(message)
        self._spinner.start()
        self.center_stack.setCurrentIndex(1)

    def _hide_loading(self):
        self._spinner.stop()
        if self._flightlab_active:
            self.center_stack.setCurrentIndex(self._flightlab_index)
            return
        self.center_stack.setCurrentIndex(0)
        # The scene was built while the view was hidden, so SceneRenderer.load's
        # reset_camera() fit the camera to a stale/zero viewport (most visibly
        # on the very first load, when the view had never been shown). Re-fit on
        # the next event-loop turn, once the show/resize events have given the
        # view its real size -- the "iso" orientation load() set is preserved.
        QTimer.singleShot(0, self._refit_camera)

    def _refit_camera(self):
        if self.state.scene is None:
            return
        self.plotter.reset_camera()
        view = self.state.scene.get("initialView") or {}
        focus = view.get("focusBodyId")
        if focus and self.state.body(focus) is not None:
            # The scene asks to open focused on a body (KSP's map view opens
            # on the vessel), optionally with another body locked behind it.
            lock = view.get("faceLockBodyId")
            self._focus_on(focus, lock_body_id=lock if self.state.body(lock) is not None else None,
                           distance=view.get("distance"))
        else:
            self.plotter.render()

    # ---------------------------------------------------------- KSP focus
    def _focus_targets(self) -> list[str]:
        """Tab order: spacecraft first, then natural bodies, then ground sites
        -- the way KSP's map view cycles vessels before planets."""
        order = {"spacecraft": 0, "site": 2}
        bodies = (self.state.scene or {}).get("bodies", [])
        return [b["id"] for b in sorted(bodies, key=lambda b: order.get(b.get("kind"), 1))]

    def _cycle_focus(self, direction: int):
        targets = self._focus_targets()
        if not targets:
            return
        current = self.state.selected_body_id
        i = targets.index(current) if current in targets else -1
        target = targets[(i + direction) % len(targets)]
        ref = altitude_reference(self.state.scene)
        body = self.state.body(target)
        # A spacecraft around a planet: lock the planet behind it, so the
        # camera looks down at the ship with the planet as the backdrop.
        lock = ref.body_id if ref is not None and body.get("kind") == "spacecraft" else None
        self._focus_on(target, lock_body_id=lock)

    def _focus_distance(self, body_id: str) -> float:
        """A camera distance that frames ``body_id``: a few radii for a body
        drawn as a sphere, a fixed close-up for a spacecraft over a planet,
        else a small fraction of its distance from the scene's center."""
        body = self.state.body(body_id)
        if body.get("radius") and body.get("kind") not in ("spacecraft", "site"):
            return 4.0 * float(body["radius"])
        r = float(np.linalg.norm(self.renderer.body_display_position(body_id, self.state.time)))
        if altitude_reference(self.state.scene) is not None:
            return max(1500.0, 0.25 * r)
        return max(0.02 * r, 1e-6)

    def _focus_on(self, body_id: str, lock_body_id: str | None = None, distance: float | None = None):
        """Select, track and frame ``body_id``."""
        self._lock_reference_body_id = lock_body_id
        self._select_lock_combo(lock_body_id)
        self._select_body(body_id)
        if not self._tracking:
            self._set_tracking(True)
        self.renderer.track_body(body_id, self.state.time, reference_body_id=lock_body_id, render=False)
        self.renderer.zoom_to_body(body_id, self.state.time,
                                   distance_km=float(distance) if distance else self._focus_distance(body_id))
        self._focus_zoomed = False
        self._pre_focus_camera = None
        self.statusBar().showMessage(f"Focus: {self.state.body(body_id)['name']} — Tab / ⇧Tab to cycle")

    def _select_lock_combo(self, body_id: str | None):
        index = self.lock_combo.findData(body_id)
        self.lock_combo.blockSignals(True)
        self.lock_combo.setCurrentIndex(max(index, 0))
        self.lock_combo.blockSignals(False)

    def _finish_cached(self, scene: dict, key: str):
        # try/finally: if applying the scene raises (e.g. a malformed cached
        # scene dict), _end_switch must still run -- otherwise self._loading
        # stays True forever and the mission list / open-file button stay
        # disabled, requiring an app restart to recover.
        try:
            self._apply_scene(scene)
            self.sidebar.set_status(key, SUCCESS)
        finally:
            self._end_switch()

    def _start_backend_probe(self):
        """Runs BackendProbeWorker once, at startup, on its own QThread --
        same reasoning as _start_loading below, just for a much smaller job.
        The badge shows "…" (see _build_compute_status) until this reports
        back."""
        thread = QThread(self)
        worker = BackendProbeWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_backend_probed, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Same "keep a Python-side owner alive" reasoning as _load_thread/
        # _load_worker -- cleared in _on_backend_probed once the result lands.
        self._backend_probe_thread = thread
        self._backend_probe_worker = worker
        thread.start()

    def _on_backend_probed(self, gpu_available: bool):
        self._backend_probe_thread = None
        self._backend_probe_worker = None
        if self._closing:
            return
        if gpu_available:
            self.backend_badge.set_state("verified", "GPU")
        else:
            self.backend_badge.set_state("idle", "CPU")

    def _start_loading(self, key: str, label: str | None = None):
        """Runs self._loaders[name]() on a background QThread instead of
        the UI thread -- compute_and_export_mission() alone is ~10s of
        Lambert-solve + differential-correction + tudatpy propagation, and
        running that inline (as an earlier version of this app did) froze
        the whole window -- no repaints, no input, Windows marks it "Not
        Responding" -- for the entire computation. The sidebar stays
        disabled for the same reason _on_mission_activated is guarded:
        re-entering this while a load is already in flight would leak a
        second thread/worker pair.

        The loading page (with its spinner) is shown for the whole switch --
        it animates during this background compute, then holds while the
        finished scene is applied on the main thread.
        """
        label = label if label is not None else key
        self._begin_switch(f"Computing {label}…")

        thread = QThread(self)
        worker = SceneLoader(key, self._loaders[key], self._compute_pool)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # QueuedConnection explicitly, not AutoConnection -- see the comment
        # on SceneLoader.finished/failed for why this matters here.
        worker.finished.connect(self._on_scene_loaded, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_scene_load_failed, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_load_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Keep references alive for the thread's lifetime -- PySide doesn't
        # keep a Python-side owner of these alive on its own, and a
        # garbage-collected QThread/QObject mid-run is a crash, not a
        # graceful cancellation.
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    def _check_source(self):
        """Offer a reload once the code on disk is not the code the compute
        worker runs."""
        if self._source_changed:
            return
        changed = self._source_watch.changed_files()
        if changed:
            self._source_changed = True
            shown = ", ".join(changed[:4]) + (f" and {len(changed) - 4} more" if len(changed) > 4 else "")
            self._source_change_text = f"Edited: {shown}."
            self._update_reload_button()

    def _update_reload_button(self):
        """One button, two reasons: edited code (reload) or a shipped
        precomputed scene made by older code (recompute)."""
        stale = self._active_key in self._stale_keys
        if self._source_changed:
            self.reload_button.setText("Code changed · Reload")
            self.reload_button.setToolTip(f"{self._source_change_text}\nReload (⌘R / Ctrl+R) restarts the compute "
                                          "worker with the new code and recomputes this mission.")
        elif stale:
            self.reload_button.setText("Precomputed by older code · Recompute")
            self.reload_button.setToolTip("This is the scene shipped with orbitopt, computed by a different version "
                                          "of the code. Recompute (⌘R / Ctrl+R) runs your code -- for some missions "
                                          "that takes minutes.")
        self.reload_button.setVisible(self._source_changed or stale)

    def _reload_mission(self):
        """Pick up edited mission code: a fresh compute worker (the new code),
        every in-memory scene dropped (all computed by the old code), and the
        current mission recomputed. The disk cache is keyed on the source, so
        it misses by itself. GUI code itself is not reloaded."""
        if self._loading:
            return
        self._source_watch.accept()
        self._source_changed = False
        mission_id = self._mission_ids.get(self._active_key) if self._active_key is not None else None
        if self._active_key in self._stale_keys and mission_id is not None:
            # Skip its shipped scene from now on (the worker reads this at
            # load time; it is spawned after this point, so it inherits it).
            from orbitopt.scene_cache import ENV_RECOMPUTE

            ids = {x for x in os.environ.get(ENV_RECOMPUTE, "").split(",") if x}
            os.environ[ENV_RECOMPUTE] = ",".join(sorted(ids | {mission_id}))
            self._stale_keys.discard(self._active_key)
        self.reload_button.setVisible(False)
        self._compute_pool.restart()
        self._scene_cache.clear()
        if self._active_key is not None and self._active_key in self._loaders:
            self._start_loading(self._active_key, self._active_label)

    def _on_load_thread_finished(self):
        self._end_switch()
        self._load_thread = None
        self._load_worker = None

    def _on_scene_loaded(self, name: str, scene: dict):
        if self._closing:
            # A finished signal that was already queued (posted to this
            # thread's event queue) by the time closeEvent disconnected it
            # still gets delivered once the event loop resumes -- Qt doesn't
            # retroactively cancel an already-posted call. This flag is the
            # real guard against touching self.plotter/self.renderer after
            # closeEvent has closed them; the disconnect there is just hygiene.
            return
        from orbitopt.scene_cache import pop_shipped_marker

        if pop_shipped_marker(scene) is not None:
            self._stale_keys.add(name)
        else:
            self._stale_keys.discard(name)
        self._update_reload_button()
        self._scene_cache[name] = scene
        # Applied while the loading page is still up (plotter hidden); the swap
        # back to the view happens in _on_load_thread_finished -> _end_switch.
        self._apply_scene(scene)
        self.sidebar.set_status(name, SUCCESS)

    def _on_scene_load_failed(self, name: str, error_msg: str):
        if self._closing:
            return
        self.load_status_badge.set_state("failed", "FAILED")
        self.sidebar.set_status(name, DANGER)
        QMessageBox.critical(self, f"Failed to load {name}", error_msg)
        self.statusBar().showMessage("Ready")

    def _verify_offer_note(self) -> str | None:
        """The Verification tab's "not verified by default -- run it" note for
        the active mission, or None (no offer: not such a built-in, or it was
        already re-run with verification on)."""
        key = self._active_key
        mission_id = self._mission_ids.get(key) if key is not None else None
        if mission_id is None or key in self._verified_keys:
            return None
        return verification_off_note(mission_id)

    def _run_high_fidelity_verification(self):
        """One-click entry from the Verification tab: re-run the active
        built-in mission with high-fidelity verification enabled, replacing
        its sidebar entry's scene with the verified one."""
        key = self._active_key
        mission_id = self._mission_ids.get(key) if key is not None else None
        loader = high_fidelity_loader(mission_id) if mission_id is not None else None
        if self._loading or loader is None:
            return
        self._set_playing(False)
        self._loaders[key] = loader
        self._verified_keys.add(key)
        self._scene_cache.pop(key, None)
        self._start_loading(key, f"{self._active_label or key} with high-fidelity verification")

    def _apply_scene(self, scene: dict):
        verify_offer = self._verify_offer_note()
        self.verification_panel.set_verify_offer(verify_offer)
        self.renderer.load(scene)

        # Camera state belongs to the old scene's bodies: stop tracking so we
        # never chase a body id that no longer exists, drop the face-lock, and
        # forget the M-key "pre-focus" camera (it framed bodies that are gone).
        # (renderer.load already cleared the measure line via plotter.clear.)
        if self._tracking:
            self._set_tracking(False)
        self._lock_reference_body_id = None
        self._focus_zoomed = False
        self._pre_focus_camera = None
        self._rebuild_lock_combo(scene)

        # Precompute the maneuver-arrow display scale: physical delta-v (m/s) is
        # invisible next to orbit radii (km), so arrows are drawn at a fraction of
        # the burn-site radius (_maneuver_base_len), lengthened in proportion to
        # |delta-v|. That fraction is only the *wide-view cap*, though -- when the
        # camera is dollied in on the spacecraft during a burn, set_maneuver_vector
        # clamps it down to a fraction of the camera distance so the arrow doesn't
        # swamp the very body it's drawn from. The arrow only appears while the
        # scrubber is within that burn's window (state.maneuver_windows), so it flashes past each
        # maneuver instead of hanging on screen the whole time.
        timeline = scene.get("timeline")
        events = timeline.get("events", []) if timeline else []
        vectors = [e["vector"] for e in events if "vector" in e]
        if vectors:
            self._maneuver_max_dv = max(_norm3(v["deltaV"]) for v in vectors) or 1.0
            self._maneuver_base_len = 0.22 * max(_norm3(v["position"]) for v in vectors)
        else:
            self._maneuver_max_dv, self._maneuver_base_len = 1.0, 0.0
        self._maneuver_windows = maneuver_windows(events, timeline_span(timeline)) if timeline else {}
        self.renderer.clear_maneuver_vector()

        self.scene_title_label.setText(scene["title"])
        self.scene_subtitle_label.setText(display_subtitle(scene))
        badge = verification_badge_state(scene)
        self.verification_badge.setVisible(badge is not None)
        if badge is not None:
            state, text, tooltip = badge
            self.verification_badge.set_state(state, text)
            if verify_offer and not scene.get("verification"):
                tooltip = ("NOT VERIFIED: high-fidelity verification is off by default for this mission. "
                           "Click to open the Verification tab and run it.")
            self.verification_badge.setToolTip(tooltip)

        # Resets time + selection and lets the timeline bar and the inspector
        # rebuild themselves for the new scene.
        self.state.set_scene(scene)
        self._refresh_selection_badge()
        self._apply_candidate_highlight(render=False)
        self._refresh_view()

        # Set last, after everything above has succeeded -- these two are the
        # "this load fully worked" signal for the title bar (this badge) and
        # the mission-list row (set by the caller right after this returns,
        # see _on_scene_loaded/_finish_cached). Setting READY any earlier let
        # the badge announce success before the scene was actually usable: a
        # KeyError/etc. anywhere above would leave it stuck on READY while the
        # mission's list marker never left NEUTRAL.
        self.load_status_badge.set_state("ready", "READY")
        self.load_time_metric.set_value(f"{time.perf_counter() - self._load_start_ts:.1f}")

    def _rebuild_lock_combo(self, scene: dict):
        self.lock_combo.blockSignals(True)
        self.lock_combo.clear()
        self.lock_combo.addItem("None", None)
        for body in scene["bodies"]:
            self.lock_combo.addItem(body["name"], body["id"])
        self.lock_combo.setCurrentIndex(0)
        self.lock_combo.blockSignals(False)

    # ------------------------------------------------------- Scene refresh
    def _refresh_view(self, dt: float | None = None):
        """Bring the 3D view and the inspector's distances up to state.time.
        Every mutation below skips its own render and the view renders once
        at the end: separate renders per call made per-frame cost noisy
        enough to read as playback jitter/stutter."""
        distances = self.renderer.set_time(self.state.time, render=False)
        self._update_flight_labels()
        t = self.state.time
        self.inspector.set_distances(distances, {bid: self.renderer.body_world_position(bid, t)
                                                 for bid in self.inspector.altitude_body_ids()})
        self._update_maneuver_arrow()
        self._update_tracking(dt=dt, render=False)
        self._update_measure(render=False)
        self.plotter.render()

    def _resize_maneuver_arrow(self):
        if self.state.scene is not None and self.state.events:
            self._update_maneuver_arrow()
            self.plotter.render()

    def _update_flight_labels(self):
        scene = self.state.scene
        if scene is None or not self.labels_action.isChecked():
            self.renderer.set_flight_labels({}, self.state.time)
            return
        t = self.state.time
        texts = {
            body_id: label_text(scene, body_id, t,
                                lambda tt, bid=body_id: self.renderer.body_world_position(bid, tt),
                                self._maneuver_windows)
            for body_id in labelled_bodies(scene)
        }
        self.renderer.set_flight_labels(texts, t)

    def _update_maneuver_arrow(self):
        """Draw a maneuver's delta-v vector only while the scrubber is close (in
        time) to that burn -- the nearest one, within its own window (see
        state.maneuver_windows) -- so the
        arrow flashes past each maneuver rather than hanging on the whole time."""
        maneuver, nearest_dt = nearest_maneuver(self.state.events, self.state.time)
        if (maneuver is not None and self._maneuver_base_len > 0.0
                and nearest_dt <= self._maneuver_windows.get(id(maneuver), 0.0)):
            vec = maneuver["vector"]
            magnitude_fraction = _norm3(vec["deltaV"]) / self._maneuver_max_dv
            self.renderer.set_maneuver_vector(
                vec["position"], vec["deltaV"], self._maneuver_base_len, magnitude_fraction,
                t=maneuver.get("time"))
        else:
            self.renderer.clear_maneuver_vector()

    def _on_time_changed(self, _t: float):
        if self._tick_dt is not None:
            # A playback tick: already paced by the 33 ms timer, so refresh
            # now and pass dt through for camera-tracking smoothing.
            self._refresh_view(dt=self._tick_dt)
            return
        # Throttled: state.time is always up to date (anything reading it
        # right after a scrub sees the new time), but the 3D refresh + render
        # runs at most once per SCRUB_MIN_INTERVAL_MS. The first change of a
        # burst applies immediately (leading edge, so a single click or
        # keyboard step has no added latency); the rest coalesce into one
        # trailing refresh at the latest time, so the final frame always
        # matches where the drag stopped.
        if self._scrub_timer.isActive():
            return
        elapsed_ms = time.monotonic() * 1000.0 - self._last_scrub_render_ms
        if elapsed_ms >= SCRUB_MIN_INTERVAL_MS:
            self._apply_scrub()
        else:
            self._scrub_timer.start(max(1, int(SCRUB_MIN_INTERVAL_MS - elapsed_ms)))

    def _apply_scrub(self):
        if self.state.timeline is None:
            return
        self._last_scrub_render_ms = time.monotonic() * 1000.0
        self._refresh_view()

    def _on_frame_changed(self, body_id: str | None):
        """Re-center the 3D view on ``body_id`` and frame its neighborhood
        (or, back in the scene's own frame, fit the whole scene). The M-key
        "zoom back out" camera was saved in the old frame's coordinates, so it
        is dropped."""
        self._focus_zoomed = False
        self._pre_focus_camera = None
        self.renderer.set_frame(body_id, render=False)
        if body_id is None:
            self.plotter.reset_camera()
        else:
            self.renderer.frame_camera_on_origin(self.renderer.local_extent(self.state.time))
        self._refresh_view()
        name = self.state.body(body_id)["name"] if body_id else "the scene's own frame"
        self.statusBar().showMessage(f"View centered on {name}")

    def _show_verification_tab(self):
        if self.right_stack.currentIndex() != 0:
            self._toggle_info_panel()  # expand the collapsed right column first
        self.right_tabs.setCurrentWidget(self.verification_panel)

    def _apply_candidate_highlight(self, render: bool = True):
        self.renderer.highlight_candidate(
            self.state.candidate_id, show_all=self.candidate_panel.show_all_box.isChecked(), render=render,
        )

    def _refresh_selection_badge(self):
        badge = selection_badge_state(self.state.scene, self.state.candidate())
        self.selection_badge.setVisible(badge is not None)
        if badge is not None:
            state, text, tooltip = badge
            self.selection_badge.set_state(state, text)
            self.selection_badge.setToolTip(tooltip)

    def _on_candidate_changed(self, _candidate_id):
        self._apply_candidate_highlight()
        self._refresh_selection_badge()
        cand = self.state.candidate()
        if cand is not None:
            self.statusBar().showMessage(
                f"{cand['label']} highlighted"
                + ("" if cand.get("exported") else " — screening arc only; verification shows the exported trajectory")
                + ("" if self.state.frame_body_id is None or self.renderer.candidate_in_frame(cand["id"])
                   else " — no ephemeris for this frame at its dates; its arc shows in the scene's own frame")
            )

    # ------------------------------------------------------- Event seeking
    def _seek_to_event(self, event: dict):
        """Jump to a maneuver's time AND auto-frame its burn -- zoom the camera
        in on the spacecraft at that instant, so clicking a maneuver row takes
        you right up to the burn (and its delta-v arrow) instead of only
        moving the clock. An event with no burn vector (e.g. the terminal
        "insertion" marker) just seeks. The zoom is to an absolute distance
        (see EVENT_ZOOM_ARROW_MULTIPLE) so clicking rows in succession lands
        on the same framing each time rather than diving in further on every
        click."""
        sc_id = self.state.primary_spacecraft_id()
        zoom = sc_id is not None and "vector" in event and self._maneuver_base_len > 0.0
        # Remember the pre-zoom framing so the M-key focus toggle can zoom back
        # out to it -- the same contract _focus_selected sets up for its zoom.
        if zoom:
            cam = self.plotter.camera
            pre_camera = (tuple(cam.position), tuple(cam.focal_point), tuple(cam.up))
        self.state.set_time(event.get("time", 0.0))
        if not zoom:
            return
        self.state.select_body(sc_id)
        self.renderer.zoom_to_body(
            sc_id, self.state.time,
            distance_km=self._maneuver_base_len * EVENT_ZOOM_ARROW_MULTIPLE,
        )
        self._pre_focus_camera = pre_camera
        self._focus_zoomed = True
        self.statusBar().showMessage(f"▸ {event.get('label', 'maneuver')} — framed the burn")

    def _seek_relative_event(self, direction: int):
        """Shift+Right / Shift+Left: jump to the next/previous mission event
        relative to the current time -- distinct from the "most recent event
        at or before now" lookup the readouts use, which only ever looks
        backward."""
        if self._flightlab_active:
            self.flightlab_page.seek_relative_event(direction)
            return
        target = adjacent_event(self.state.events, self.state.time, direction)
        if target is not None:
            self._seek_to_event(target)

    # -------------------------------------------------------------- Timing
    def _step_time(self, direction: float):
        """Nudge the current time one step forward/back: 1% of the visible
        timeline, so the step gets finer as the timeline zooms in."""
        if self._flightlab_active:
            self.flightlab_page.step_time(direction)
            return
        timeline = self.state.timeline
        if not timeline:
            return
        self._set_playing(False)
        self.state.set_time(self.state.time + direction * (self.timeline_bar.step_size() or 0.01 * timeline_span(timeline)))

    def _restart_timeline(self):
        if self._flightlab_active:
            self.flightlab_page.restart()
            return
        timeline = self.state.timeline
        if not timeline:
            return
        self._set_playing(False)
        self.state.set_time(timeline["min"])

    def _toggle_play(self, direction: int = 1):
        """Space plays forward, Shift+Space backwards; the key of the
        direction already playing pauses, the other one turns around."""
        if self._flightlab_active:
            # The mission scene is hidden behind the Flight Lab page; Space
            # drives the replay instead of animating an invisible view.
            self.flightlab_page.toggle_playback()
            return
        timeline = self.state.timeline
        if not timeline:
            return
        if self._playing and direction == self._play_direction:
            self._set_playing(False)
            return
        # From the end it can only go back, from the start only forward: start
        # over from the other end rather than doing nothing.
        if direction > 0 and self.state.time >= timeline["max"]:
            self.state.set_time(timeline["min"])
        elif direction < 0 and self.state.time <= timeline["min"]:
            self.state.set_time(timeline["max"])
        self._play_direction = 1 if direction > 0 else -1
        self._set_playing(True)

    def _set_playing(self, playing: bool):
        self._playing = playing
        self.timeline_bar.set_playing(playing, self._play_direction)
        if playing:
            self._last_tick_ms = None
            self._timer.start()
        else:
            self._timer.stop()
            # Drop any leftover "eased" auto-slow display.
            self.timeline_bar.show_effective_speed(self.timeline_bar.warp)

    def _on_auto_slow_toggled(self, on: bool):
        if not self._playing:
            self.timeline_bar.show_effective_speed(self.timeline_bar.warp)
        self.statusBar().showMessage("Auto-slow near events: on" if on else "Auto-slow near events: off")

    def _effective_speed(self, timeline: dict) -> float:
        """The chosen time warp (simulated seconds per real second), eased
        down near events when auto-slow is on (see state.eased_speed)."""
        warp = self.timeline_bar.warp
        if not self.timeline_bar.auto_slow_enabled:
            return warp
        return eased_speed(
            warp, self.state.events, self.state.time, timeline_span(timeline),
            EVENT_SLOWDOWN_WINDOW_FRACTION, EVENT_SLOWDOWN_TARGET_WARP,
        )

    def _on_tick(self):
        now_ms = time.monotonic() * 1000.0
        if self._last_tick_ms is None:
            self._last_tick_ms = now_ms
            return
        dt = (now_ms - self._last_tick_ms) / 1000.0
        self._last_tick_ms = now_ms
        # Cap dt so a single stalled frame (GC pause, texture/mesh rebuild,
        # window drag) can't advance sim time by one big visible jump -- the
        # timer's nominal interval is 33ms, so anything past ~3x that is
        # treated as a hitch to smooth over rather than catch up on exactly.
        dt = min(dt, 0.1)

        timeline = self.state.timeline
        if not timeline:
            self._set_playing(False)
            return

        effective_speed = self._effective_speed(timeline)
        self.timeline_bar.show_effective_speed(effective_speed)
        # Warp is in simulated seconds per real second; the timeline counts
        # in its own unit (days for every orbitopt exporter).
        unit_s = timeline_unit_seconds(timeline.get("unitLabel"))
        step = self._play_direction * dt * effective_speed / unit_s
        new_time = min(timeline["max"], max(timeline["min"], self.state.time + step))
        self._tick_dt = dt
        try:
            self.state.set_time(new_time)
        finally:
            self._tick_dt = None
        if new_time >= timeline["max"] or new_time <= timeline["min"]:
            self._set_playing(False)

    def _on_rotation_tick(self):
        now_ms = time.monotonic() * 1000.0
        if self._last_rotation_tick_ms is None:
            self._last_rotation_tick_ms = now_ms
            return
        dt = (now_ms - self._last_rotation_tick_ms) / 1000.0
        self._last_rotation_tick_ms = now_ms

        if self.state.scene is not None:
            self.renderer.advance_rotation(dt * ROTATION_SIM_HOURS_PER_REAL_SECOND)

    def _on_tile_tick(self):
        if self._loading or self.renderer.scene is None:
            return
        if self.renderer.update_tiles():
            self.plotter.render()
        # Arrived tiles are built a few per tick (tile_layer._BUILD_BUDGET_S):
        # poll fast until they are all up, then back to the idle rate.
        self._tile_timer.setInterval(40 if self.renderer.tiles.pending else 200)

    def _on_imagery_toggled(self, on: bool):
        self.renderer.tiles.set_enabled(on)
        self._on_tile_tick()

    def closeEvent(self, event):  # noqa: N802 -- Qt override signature
        """Tear down cleanly on quit (incl. Cmd+Q). The embedded VTK render
        window has to release its native OpenGL context *before* Qt destroys the
        widget under it -- otherwise VTK finalizes a context Qt has already torn
        down and prints errors on the way out (the same reason tests call
        plotter.close() before window.close()). Also stop the animation timer and
        any in-flight scene-loading thread so nothing fires mid-teardown."""
        self._set_playing(False)
        self._timer.stop()
        self._scrub_timer.stop()
        self._rotation_timer.stop()
        self._tile_timer.stop()
        self.renderer.tiles.shutdown()
        self._spinner.stop()  # the spinner has its own QTimer, independent of self._timer
        self.plotter._resize_settle_timer.stop()  # so it can't fire post-teardown
        # Set *before* touching the load thread, unconditionally -- not just on
        # a wait() timeout. disconnect() below is best-effort hygiene, not the
        # real guard: SceneLoader.run() emits finished/failed as its very last
        # statement, so a fast-finishing load can have that call already queued
        # to this thread's event loop before we ever get to disconnect it, and
        # Qt does not retroactively cancel an already-posted queued call. This
        # flag is what _on_scene_loaded/_on_scene_load_failed actually check to
        # refuse to touch self.plotter/self.renderer once we're closing.
        self._closing = True
        thread = self._load_thread
        worker = self._load_worker
        if thread is not None:
            if worker is not None:
                for signal, slot in (
                    (worker.finished, self._on_scene_loaded),
                    (worker.failed, self._on_scene_load_failed),
                ):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            # Kill the compute process first: the loader thread is blocked in
            # future.result() and unblocks as soon as the worker is gone.
            self._compute_pool.terminate()
            thread.quit()
            # Scene loading (e.g. compute_and_export_mission) can take ~10s --
            # give it generous room to finish before tearing down the plotter
            # below. Bounded, not indefinite: self._closing above already
            # makes proceeding safe even if the thread is still running, so a
            # genuinely hung loader can't block quit forever.
            if not thread.wait(2000) and not thread.wait(15000):
                print(
                    "orbitopt: scene-loading thread still running ~17s after "
                    "quit was requested; not waiting further.",
                    file=sys.stderr,
                )
        probe_thread = self._backend_probe_thread
        probe_worker = self._backend_probe_worker
        if probe_thread is not None:
            if probe_worker is not None:
                try:
                    probe_worker.finished.disconnect(self._on_backend_probed)
                except (RuntimeError, TypeError):
                    pass
            probe_thread.quit()
            probe_thread.wait(1000)  # a device-count query, never ~10s of compute like scene loads
        self._compute_pool.terminate()  # idle worker process, if any
        try:
            self.plotter.close()
        except Exception:  # noqa: BLE001 -- best-effort teardown, never block quit
            pass
        super().closeEvent(event)


def main():
    # No-op except in a frozen Windows build (PyInstaller etc.), where the
    # spawned compute worker re-runs this executable and must not open a
    # second window.
    import multiprocessing

    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    # App-level icon drives the macOS Dock tile / taskbar entry; the window
    # inherits it for its title bar too.
    app.setWindowIcon(app_icon())
    window = MissionControlWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
