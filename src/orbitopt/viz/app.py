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

Run: orbitopt  (or orbitopt-app / python -m orbitopt); main() is the entry point.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkRenderWindow

from orbitopt.viz.icon import app_icon
from orbitopt.viz.pv_viewer import load_scene
from orbitopt.viz.scene_renderer import SceneRenderer
from orbitopt.viz.theme import ACCENT, BG_VOID, INK_SECONDARY, STYLESHEET

SPEED_PRESETS = [0.15, 0.5, 1.0, 5.0, 20.0, 100.0]


class DebouncedInteractor(QtInteractor):
    """QtInteractor renders synchronously on *every* resizeEvent (see
    pyvistaqt.rwi.QVTKRenderWindowInteractor.resizeEvent -> self.update() ->
    paintEvent -> Iren.Render()). On Windows, a live edge-drag delivers
    resizeEvent for every intermediate size through the OS's own native
    modal message loop (WM_ENTERSIZEMOVE..WM_EXITSIZEMOVE), which won't hand
    control back to the rest of the app until each repaint finishes -- so a
    scene with anti-aliasing, orbit trails and point labels compounds into
    the whole window reading as frozen for the length of the drag. Measured
    ~45ms/intermediate-size even with anti-aliasing off, which isn't the
    render cost itself (that dropped from ~20ms to ~2ms) but Qt/VTK resize
    bookkeeping run on every single one of dozens of events during a drag.
    Fix: keep the VTK render window's size/DPI in sync on every event
    (cheap), but defer the actual repaint until resizing has been idle for a
    short settle period -- the 3D view visually lags slightly behind the
    window edge during the drag itself, but the app never stops responding.
    """

    _RESIZE_SETTLE_MS = 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resize_settle_timer = QTimer(self)
        self._resize_settle_timer.setSingleShot(True)
        self._resize_settle_timer.setInterval(self._RESIZE_SETTLE_MS)
        self._resize_settle_timer.timeout.connect(self._render_after_resize_settles)

    def resizeEvent(self, event):  # noqa: N802 -- Qt override signature
        if self._RenderWindow is None:
            return
        scale = self._getPixelRatio()
        w = int(round(scale * self.width()))
        h = int(round(scale * self.height()))
        self._RenderWindow.SetDPI(int(round(72 * scale)))
        vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self._resize_settle_timer.start()

    def _render_after_resize_settles(self):
        # Guard against firing after the widget is closed: pyvistaqt swaps _Iren
        # for a stub with no Render() during teardown, so a still-pending settle
        # timer would otherwise raise on quit (Cmd+Q).
        iren = self._Iren
        if iren is not None and hasattr(iren, "Render"):
            iren.Render()


class SceneLoader(QObject):
    """Runs a scene loader (e.g. compute_and_export_mission, ~10s of pure
    Python/numpy/tudatpy work) on a background QThread so it can't block
    the UI event loop -- moveToThread + signals is the standard PySide
    pattern for this, not a raw threading.Thread, because Qt needs the
    result handed back via a queued signal/slot connection to land safely
    on the main thread rather than touching any Qt/VTK objects directly
    from the worker thread (which isn't thread-safe). The loader itself
    must stay pure-Python -- it must never touch self.plotter/self.renderer,
    both of which live on the main thread.
    """

    # (name, scene) rather than just (scene): the name has to travel with the
    # signal itself, not via a Python closure. A `lambda scene: cb(name, scene)`
    # connected to a cross-thread signal is invoked *directly on the emitting
    # (worker) thread* -- PySide can only infer "this needs to be queued to
    # the receiver's thread" when the receiver is a bound method of a QObject;
    # a plain lambda has no thread affinity of its own, so AutoConnection
    # silently falls back to a direct call. That let _on_scene_loaded (which
    # touches self.renderer/self.plotter, i.e. VTK's OpenGL context) run on
    # the background thread, fighting the main thread for the same Win32 GL
    # context -- observed as `wglMakeCurrent failed ... resource in use` and
    # the whole app hanging. Emitting (name, scene) and connecting straight to
    # the bound method (no lambda) lets Qt detect the correct thread and queue
    # the call properly; the connections below also pass QueuedConnection
    # explicitly so this doesn't regress silently if a lambda creeps back in.
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, name, loader):
        super().__init__()
        self._name = name
        self._loader = loader

    def run(self):
        try:
            scene = self._loader()
        except Exception as exc:  # noqa: BLE001 -- reported to the user via the failed signal
            self.failed.emit(self._name, str(exc))
            return
        self.finished.emit(self._name, scene)


def _builtin_missions():
    """(display name, zero-arg loader) pairs -- computed lazily, only when
    first selected, and cached after that (see MissionControlWindow._scene_cache)."""

    def load_solar_system():
        from orbitopt.viz.solar_system import export_solar_system_data
        return export_solar_system_data()

    def load_artemis2():
        from orbitopt.viz.mission_timeline import compute_and_export_mission
        return compute_and_export_mission()

    def load_geo_raising():
        from orbitopt.viz.geo_raising import compute_and_export_geo_mission
        return compute_and_export_geo_mission()

    return [
        ("Solar System", load_solar_system),
        ("Artemis II — Free Return", load_artemis2),
        ("GOES — GTO to GEO", load_geo_raising),
    ]


def _format_distance(value: float, unit: str) -> str:
    if unit == "AU":
        return f"{value:,.3f} AU"
    return f"{value:,.0f} km"


def _norm3(v) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


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


class EventSlider(QSlider):
    """The timeline scrubber, with a marker painted over the groove at each
    mission event (burns, flybys, insertions). Hovering a marker shows the
    event's label + note; clicking one seeks straight to it. Falls back to a
    plain slider when the scene has no timeline events."""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._events = []  # list of (slider_value:int, label:str, note:str)
        self.setMouseTracking(True)

    def set_events(self, events, t_min, t_max):
        self._events = []
        span = (t_max - t_min) or 1.0
        lo, hi = self.minimum(), self.maximum()
        for e in events:
            frac = min(1.0, max(0.0, (e.get("time", t_min) - t_min) / span))
            self._events.append((int(round(lo + frac * (hi - lo))),
                                 e.get("label", ""), e.get("note", "")))
        self.update()

    def _groove(self):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

    def _value_to_x(self, value):
        groove = self._groove()
        span = (self.maximum() - self.minimum()) or 1
        return groove.left() + int((value - self.minimum()) / span * groove.width())

    def _event_near(self, x, tol=6):
        for ev in self._events:
            if abs(self._value_to_x(ev[0]) - x) <= tol:
                return ev
        return None

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if not self._events:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(ACCENT), 1.4))
        painter.setBrush(QColor(ACCENT))
        cy = self.height() // 2
        for value, _label, _note in self._events:
            x = self._value_to_x(value)
            painter.drawLine(x, 7, x, cy + 2)
            tri = QPainterPath()
            tri.moveTo(x - 3.0, 1.0)
            tri.lineTo(x + 3.0, 1.0)
            tri.lineTo(x, 7.0)
            tri.closeSubpath()
            painter.drawPath(tri)
        painter.end()

    def mouseMoveEvent(self, event):  # noqa: N802
        ev = self._event_near(int(event.position().x()))
        self.setToolTip(f"{ev[1]}\n{ev[2]}".strip() if ev else "")
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        ev = self._event_near(int(event.position().x()))
        if ev is not None:
            self.setValue(ev[0])  # snap to the event
            return
        super().mousePressEvent(event)


class MissionControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mission Control — orbitopt")
        self.setWindowIcon(app_icon())
        self._size_to_screen()
        self.setStyleSheet(STYLESHEET)

        self._scene_cache: dict[str, dict] = {}
        self._loaders: dict[str, callable] = {}
        self._current_scene: dict | None = None
        self._current_time = 0.0
        self._speed = SPEED_PRESETS[2]
        self._playing = False

        self._loading = False
        self._load_thread: QThread | None = None
        self._load_worker: SceneLoader | None = None

        self._selected_body_id: str | None = None
        self._measure_body_id: str | None = None
        self._tracking = False
        self._events: list[dict] = []
        self._maneuver_base_len = 0.0
        self._maneuver_max_dv = 1.0
        self._maneuver_window = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._last_tick_ms = None

        self._build_ui()
        self._populate_builtin_missions()

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

        self.sidebar = self._build_sidebar()
        self.left_rail = self._build_rail("»", self._toggle_sidebar, "Show missions")
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

        self.plotter = DebouncedInteractor(center)
        self.plotter.set_background(BG_VOID)
        self.plotter.enable_anti_aliasing()
        self.renderer = SceneRenderer(self.plotter)

        # The 3D view and a loading page share one slot. We swap to the loading
        # page (rather than overlaying a Qt widget on top of the view) because
        # the embedded VTK surface is a native GL context Qt's compositor
        # doesn't reliably draw sibling widgets over -- the same reason
        # QWidget.grab() comes back black for it (see README). A QStackedWidget
        # swap composites cleanly.
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.plotter)
        self.center_stack.addWidget(self._build_loading_page())
        center_layout.addWidget(self.center_stack, stretch=1)

        self.timeline_bar = self._build_timeline_bar()
        center_layout.addWidget(self.timeline_bar)
        self.splitter.addWidget(center)

        self.info_panel = self._build_info_panel()
        self.right_rail = self._build_rail("«", self._toggle_info_panel, "Show body info")
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self.info_panel)
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

    def _build_rail(self, glyph: str, handler, tip: str) -> QWidget:
        """A thin vertical strip shown when a panel is collapsed: just an expand
        button, so the panel can be summoned back from its own edge."""
        rail = QWidget()
        rail.setObjectName("railBar")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(3, 9, 3, 9)
        layout.setSpacing(0)
        btn = QPushButton(glyph)
        btn.setObjectName("railButton")
        btn.setToolTip(tip)
        btn.clicked.connect(handler)
        layout.addWidget(btn, alignment=Qt.AlignHCenter | Qt.AlignTop)
        layout.addStretch(1)
        return rail

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(0)

        title = QLabel("ORBITOPT MISSION CONTROL")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        header_row = QHBoxLayout()
        header = QLabel("MISSIONS")
        header.setObjectName("sectionHeader")
        self.sidebar_collapse = QPushButton("❮")
        self.sidebar_collapse.setObjectName("panelCollapse")
        self.sidebar_collapse.setToolTip("Collapse missions panel")
        self.sidebar_collapse.clicked.connect(self._toggle_sidebar)
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(self.sidebar_collapse)
        layout.addLayout(header_row)

        self.mission_list = QListWidget()
        self.mission_list.currentRowChanged.connect(self._on_mission_selected)
        layout.addWidget(self.mission_list, stretch=1)

        self._open_file_btn = QPushButton("Open scene file…")
        self._open_file_btn.clicked.connect(self._open_file)
        layout.addWidget(self._open_file_btn)

        return sidebar

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

        layout.addWidget(self._build_view_controls(), alignment=Qt.AlignVCenter)

        return bar

    def _build_view_controls(self) -> QWidget:
        """KSP-style camera view presets -- snap the camera to a fixed
        orthographic angle without hunting for it by dragging."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        label = QLabel("VIEW")
        label.setObjectName("viewLabel")
        row.addWidget(label)

        for text, handler, tip in (
            ("Top", self._view_top, "Top-down (look along −Z)"),
            ("Front", self._view_front, "Front (look along −Y)"),
            ("Side", self._view_side, "Side (look along −X)"),
            ("Iso", self._view_iso, "Isometric"),
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
        return box

    def _build_timeline_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("timelineBar")
        bar.setFixedHeight(72)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_btn)

        speed_row = QHBoxLayout()
        speed_row.setSpacing(4)
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self._speed_buttons = {}
        for i, s in enumerate(SPEED_PRESETS):
            btn = QPushButton(f"{s:g}x")
            btn.setCheckable(True)
            btn.setChecked(s == self._speed)
            btn.clicked.connect(lambda _checked, s=s: self._set_speed(s))
            self.speed_group.addButton(btn)
            self._speed_buttons[s] = btn
            speed_row.addWidget(btn)
        speed_col = QVBoxLayout()
        speed_col.addWidget(QLabel("TIME WARP"))
        speed_col.addLayout(speed_row)
        layout.addLayout(speed_col)

        slider_col = QVBoxLayout()
        slider_col.setSpacing(2)
        readout_row = QHBoxLayout()
        self.met_label = QLabel("T+0.00")
        self.met_label.setObjectName("metReadout")
        self.date_label = QLabel("")
        self.date_label.setObjectName("dateReadout")
        self.event_label = QLabel("")
        self.event_label.setObjectName("eventReadout")
        readout_row.addWidget(self.met_label)
        readout_row.addWidget(self.date_label)
        readout_row.addStretch(1)
        readout_row.addWidget(self.event_label)
        slider_col.addLayout(readout_row)

        self.time_slider = EventSlider()
        self.time_slider.setRange(0, 10000)
        self.time_slider.valueChanged.connect(self._on_slider_moved)
        slider_col.addWidget(self.time_slider)
        layout.addLayout(slider_col, stretch=1)

        return bar

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("infoPanel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header_row = QHBoxLayout()
        self.info_collapse = QPushButton("❯")
        self.info_collapse.setObjectName("panelCollapse")
        self.info_collapse.setToolTip("Collapse body-info panel")
        self.info_collapse.clicked.connect(self._toggle_info_panel)
        header = QLabel("BODIES")
        header.setObjectName("sectionHeader")
        header_row.addWidget(self.info_collapse)
        header_row.addWidget(header)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        # Distance-between-two-bodies readout, shown only while measuring.
        self.measure_card = QFrame()
        self.measure_card.setObjectName("measureCard")
        mlay = QVBoxLayout(self.measure_card)
        mlay.setContentsMargins(10, 7, 10, 7)
        mlay.setSpacing(2)
        self.measure_pair_label = QLabel("")
        self.measure_pair_label.setObjectName("measurePair")
        self.measure_dist_label = QLabel("")
        self.measure_dist_label.setObjectName("measureDist")
        mlay.addWidget(self.measure_pair_label)
        mlay.addWidget(self.measure_dist_label)
        self.measure_card.setVisible(False)
        outer.addWidget(self.measure_card)

        # The body cards go in a scroll area, not straight into the panel: a
        # scene with ~10 bodies stacks that many fixed-height cards, and without
        # this the panel's minimum height is the sum of all of them -- which
        # forced the whole window's minimum height taller than the screen (so it
        # opened oversized and couldn't be shrunk). Scrolling keeps the panel's
        # minimum height bounded so the window stays freely resizable.
        scroll = QScrollArea()
        scroll.setObjectName("infoScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        cards_host = QWidget()
        cards_host.setObjectName("infoCards")
        self.info_layout = QVBoxLayout(cards_host)
        self.info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_layout.setSpacing(8)
        self.info_layout.addStretch(1)
        scroll.setWidget(cards_host)
        outer.addWidget(scroll, stretch=1)

        self._body_cards: dict[str, dict] = {}
        return panel

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

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("Open scene file…")
        open_action.triggered.connect(self._open_file)
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)  # Cmd+Q / Ctrl+Q
        quit_action.triggered.connect(self.close)

        play_menu = self.menuBar().addMenu("&Playback")
        self.play_action = play_menu.addAction("Play / Pause", self._toggle_play)
        self.play_action.setShortcut(QKeySequence(Qt.Key_Space))
        play_menu.addAction("Step forward", lambda: self._step_time(+1)).setShortcut(QKeySequence(Qt.Key_Right))
        play_menu.addAction("Step back", lambda: self._step_time(-1)).setShortcut(QKeySequence(Qt.Key_Left))
        play_menu.addAction("Restart", self._restart_timeline).setShortcut(QKeySequence(Qt.Key_Home))
        play_menu.addSeparator()
        play_menu.addAction("Faster", lambda: self._cycle_speed(+1)).setShortcut(QKeySequence("]"))
        play_menu.addAction("Slower", lambda: self._cycle_speed(-1)).setShortcut(QKeySequence("["))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("Top view", self._view_top)
        view_menu.addAction("Front view", self._view_front)
        view_menu.addAction("Side view", self._view_side)
        view_menu.addAction("Isometric view", self._view_iso)
        reset_action = view_menu.addAction("Reset camera")
        reset_action.triggered.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        view_menu.addSeparator()
        focus_action = view_menu.addAction("Focus selected body", self._focus_selected)
        focus_action.setShortcut(QKeySequence("M"))
        self.track_action = view_menu.addAction("Track selected body")
        self.track_action.setCheckable(True)
        self.track_action.setShortcut(QKeySequence("T"))
        self.track_action.toggled.connect(self._set_tracking)
        view_menu.addSeparator()
        view_menu.addAction("Toggle missions panel", self._toggle_sidebar)
        view_menu.addAction("Toggle body-info panel", self._toggle_info_panel)

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
        self.plotter.view_xy()
        self.plotter.render()

    def _view_front(self):
        self.plotter.view_xz()
        self.plotter.render()

    def _view_side(self):
        self.plotter.view_yz()
        self.plotter.render()

    def _view_iso(self):
        self.plotter.view_isometric()
        self.plotter.render()

    def _on_card_clicked(self, event, body_id: str):
        """Plain click selects (focus/track anchor); ⌘/Ctrl-click picks the
        second body to measure a distance to."""
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self._set_measure_partner(body_id)
        else:
            self._select_body(body_id)

    def _select_body(self, body_id: str):
        """Select a body (clicking its card): highlight the card, remember it as
        the focus/track target, and center the camera on it once. A plain select
        also clears any pending distance measurement."""
        self._selected_body_id = body_id
        self._measure_body_id = None
        self._refresh_card_highlights()
        self.renderer.track_body(body_id, self._current_time)
        self._update_measure()
        self.statusBar().showMessage(
            f"Focused {self._body_cards[body_id]['name']} — ⌘/Ctrl-click another to measure"
        )

    def _set_measure_partner(self, body_id: str):
        """Pick (or unpick) the second body for a distance measurement."""
        if body_id == self._selected_body_id:
            return  # can't measure a body against itself
        self._measure_body_id = None if body_id == self._measure_body_id else body_id
        self._refresh_card_highlights()
        self._update_measure()

    def _refresh_card_highlights(self):
        for bid, card in self._body_cards.items():
            frame = card["frame"]
            frame.setProperty("selected", bid == self._selected_body_id)
            frame.setProperty("measure", bid == self._measure_body_id)
            frame.style().unpolish(frame)
            frame.style().polish(frame)

    def _update_measure(self):
        """Redraw the measurement line + readout for the current (A, B) pair, or
        clear it if a full pair isn't selected. Called on selection changes and
        every timeline step so the distance tracks the bodies as they move."""
        a, b = self._selected_body_id, self._measure_body_id
        cards = self._body_cards
        if a and b and a in cards and b in cards:
            dist = self.renderer.measure_line(a, b, self._current_time)
            self.measure_pair_label.setText(f"{cards[a]['name']}  ↔  {cards[b]['name']}")
            self.measure_dist_label.setText(_format_distance(dist, cards[a]["unit"]))
            self.measure_card.setVisible(True)
        else:
            self.renderer.clear_measure_line()
            self.measure_card.setVisible(False)

    def _focus_selected(self):
        """Re-center the camera on the selected body (the M shortcut)."""
        if self._selected_body_id and self._selected_body_id in self._body_cards:
            self.renderer.track_body(self._selected_body_id, self._current_time)
        else:
            self.statusBar().showMessage("Select a body first — click its card")

    def _set_tracking(self, on: bool):
        """Turn continuous tracking on/off, keeping the button and menu item in
        sync (either can drive this)."""
        self._tracking = bool(on)
        for w in (self.track_btn, self.track_action):
            w.blockSignals(True)
            w.setChecked(self._tracking)
            w.blockSignals(False)
        if self._tracking:
            if self._selected_body_id and self._selected_body_id in self._body_cards:
                self.renderer.track_body(self._selected_body_id, self._current_time)
                self.statusBar().showMessage(f"Tracking {self._body_cards[self._selected_body_id]['name']}")
            else:
                self.statusBar().showMessage("Select a body to track — click its card")
        else:
            self.statusBar().showMessage("Tracking off")

    def _update_tracking(self):
        """Follow the tracked body after a timeline change, if tracking is on."""
        if self._tracking and self._selected_body_id and self._selected_body_id in self._body_cards:
            self.renderer.track_body(self._selected_body_id, self._current_time)

    # ------------------------------------------------------------- Missions
    def _populate_builtin_missions(self):
        for name, loader in _builtin_missions():
            self._loaders[name] = loader
            self.mission_list.addItem(QListWidgetItem(name))
        if self.mission_list.count():
            self.mission_list.setCurrentRow(0)

    def _open_file(self):
        if self._loading:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open scene JSON", "", "Scene files (*.json)")
        if not path:
            return
        name = os.path.basename(path)
        self._loaders[name] = (lambda p=path: load_scene(p))
        self.mission_list.addItem(QListWidgetItem(name))
        self.mission_list.setCurrentRow(self.mission_list.count() - 1)

    def _on_mission_selected(self, row: int):
        if row < 0 or self._loading:
            return
        name = self.mission_list.item(row).text()
        self._set_playing(False)

        if name in self._scene_cache:
            # A cached scene skips the ~10s compute but still stutters: applying
            # it runs SceneRenderer.load, which tears down and rebuilds every
            # VTK actor synchronously on the main thread. Show the loading page
            # first, then apply on the next event-loop turn so the page actually
            # paints before that blocking rebuild -- otherwise the swap and the
            # freeze happen in the same turn and the user sees only the freeze.
            self._begin_switch(f"Loading {name}…")
            scene = self._scene_cache[name]
            QTimer.singleShot(0, lambda: self._finish_cached(scene))
            return

        self._start_loading(name)

    def _begin_switch(self, message: str):
        self._loading = True
        self.mission_list.setEnabled(False)
        self._open_file_btn.setEnabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self._show_loading(message)
        self.statusBar().showMessage(message)

    def _end_switch(self):
        self._hide_loading()
        QApplication.restoreOverrideCursor()
        self.mission_list.setEnabled(True)
        self._open_file_btn.setEnabled(True)
        self._loading = False
        self.statusBar().showMessage("Ready")

    def _show_loading(self, message: str):
        self._loading_label.setText(message)
        self._spinner.start()
        self.center_stack.setCurrentIndex(1)

    def _hide_loading(self):
        self._spinner.stop()
        self.center_stack.setCurrentIndex(0)
        # The scene was built while the view was hidden, so SceneRenderer.load's
        # reset_camera() fit the camera to a stale/zero viewport (most visibly
        # on the very first load, when the view had never been shown). Re-fit on
        # the next event-loop turn, once the show/resize events have given the
        # view its real size -- the "iso" orientation load() set is preserved.
        QTimer.singleShot(0, self._refit_camera)

    def _refit_camera(self):
        if self._current_scene is not None:
            self.plotter.reset_camera()
            self.plotter.render()

    def _finish_cached(self, scene: dict):
        self._apply_scene(scene)
        self._end_switch()

    def _start_loading(self, name: str):
        """Runs self._loaders[name]() on a background QThread instead of
        the UI thread -- compute_and_export_mission() alone is ~10s of
        Lambert-solve + differential-correction + tudatpy propagation, and
        running that inline (as an earlier version of this app did) froze
        the whole window -- no repaints, no input, Windows marks it "Not
        Responding" -- for the entire computation. The sidebar/open-file
        button stay disabled and the cursor shows busy for the same reason
        the mission list itself is guarded above: re-entering this while a
        load is already in flight would leak a second thread/worker pair.

        The loading page (with its spinner) is shown for the whole switch --
        it animates during this background compute, then holds while the
        finished scene is applied on the main thread.
        """
        self._begin_switch(f"Computing {name}…")

        thread = QThread(self)
        worker = SceneLoader(name, self._loaders[name])
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

    def _on_load_thread_finished(self):
        self._end_switch()
        self._load_thread = None
        self._load_worker = None

    def _on_scene_loaded(self, name: str, scene: dict):
        self._scene_cache[name] = scene
        # Applied while the loading page is still up (plotter hidden); the swap
        # back to the view happens in _on_load_thread_finished -> _end_switch.
        self._apply_scene(scene)

    def _on_scene_load_failed(self, name: str, error_msg: str):
        QMessageBox.critical(self, f"Failed to load {name}", error_msg)
        self.statusBar().showMessage("Ready")

    def _apply_scene(self, scene: dict):
        self._current_scene = scene
        self.renderer.load(scene)

        # A new scene has different bodies -- drop the old selection/measurement
        # and stop tracking so we never chase a body id that no longer exists.
        # (renderer.load already cleared the measure line via plotter.clear.)
        self._selected_body_id = None
        self._measure_body_id = None
        self.measure_card.setVisible(False)
        if self._tracking:
            self._set_tracking(False)

        self.scene_title_label.setText(scene["title"])
        self.scene_subtitle_label.setText(scene.get("subtitle", "").replace("\n", "  "))

        timeline = scene.get("timeline")
        self.timeline_bar.setVisible(timeline is not None)
        self._events = timeline.get("events", []) if timeline else []
        if timeline:
            self._current_time = timeline["min"]
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(0)
            self.time_slider.set_events(self._events, timeline["min"], timeline["max"])
            self.time_slider.blockSignals(False)
        else:
            self._current_time = 0.0
            self.time_slider.set_events([], 0.0, 1.0)
        self.event_label.setText("")

        # Precompute the maneuver-arrow display scale: physical delta-v (m/s) is
        # invisible next to orbit radii (km), so arrows are drawn at a fraction of
        # the burn-site radius, lengthened in proportion to |delta-v|. The arrow
        # only appears while the scrubber is within _maneuver_window of a burn, so
        # it flashes past each maneuver instead of hanging on screen the whole time.
        vectors = [e["vector"] for e in self._events if "vector" in e]
        if vectors:
            self._maneuver_max_dv = max(_norm3(v["deltaV"]) for v in vectors) or 1.0
            self._maneuver_base_len = 0.22 * max(_norm3(v["position"]) for v in vectors)
        else:
            self._maneuver_max_dv, self._maneuver_base_len = 1.0, 0.0
        if timeline:
            self._maneuver_window = 0.05 * ((timeline["max"] - timeline["min"]) or 1.0)
        else:
            self._maneuver_window = 0.0
        self.renderer.clear_maneuver_vector()

        self._rebuild_body_cards(scene)
        self._refresh_readouts(self.renderer.set_time(self._current_time))
        self.plotter.render()

    # --------------------------------------------------------------- Cards
    def _rebuild_body_cards(self, scene: dict):
        for card in self._body_cards.values():
            card["frame"].deleteLater()
        self._body_cards = {}
        for widget in getattr(self, "_event_widgets", []):
            widget.deleteLater()
        self._event_widgets = []
        self._event_rows = []

        stretch_item = self.info_layout.takeAt(self.info_layout.count() - 1)
        del stretch_item

        timeline = scene.get("timeline")
        events = timeline.get("events", []) if timeline else []
        if events:
            header = QLabel("MANEUVERS")
            header.setObjectName("sectionHeader")
            self.info_layout.addWidget(header)
            self._event_widgets.append(header)
            unit = timeline.get("unitLabel", "")
            for e in events:
                row = self._make_maneuver_row(e, unit)
                self.info_layout.addWidget(row)
                self._event_widgets.append(row)
                self._event_rows.append((e, row))

            bodies_header = QLabel("BODIES")
            bodies_header.setObjectName("sectionHeader")
            self.info_layout.addWidget(bodies_header)
            self._event_widgets.append(bodies_header)

        for body in scene["bodies"]:
            frame = QFrame()
            frame.setObjectName("bodyCard")
            v = QVBoxLayout(frame)
            v.setContentsMargins(10, 8, 10, 8)
            v.setSpacing(3)

            name_row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {body['color']}; font-size: 13px;")
            name_label = QLabel(body["name"])
            name_label.setObjectName("bodyName")
            name_row.addWidget(dot)
            name_row.addWidget(name_label)
            name_row.addStretch(1)
            v.addLayout(name_row)

            dist_row = QHBoxLayout()
            dist_key = QLabel("Distance")
            dist_key.setObjectName("bodyStat")
            dist_val = QLabel("—")
            dist_val.setObjectName("bodyStatValue")
            dist_row.addWidget(dist_key)
            dist_row.addStretch(1)
            dist_row.addWidget(dist_val)
            v.addLayout(dist_row)

            for row in body.get("info", []):
                r = QHBoxLayout()
                k = QLabel(row["label"])
                k.setObjectName("bodyStat")
                val = QLabel(row["value"])
                val.setObjectName("bodyStatValue")
                r.addWidget(k)
                r.addStretch(1)
                r.addWidget(val)
                v.addLayout(r)

            frame.setToolTip("Click to focus · ⌘/Ctrl-click to measure distance")
            frame.mousePressEvent = lambda e, bid=body["id"]: self._on_card_clicked(e, bid)

            self.info_layout.addWidget(frame)
            self._body_cards[body["id"]] = {
                "frame": frame,
                "dist_val": dist_val,
                "unit": scene["distanceUnit"],
                "name": body["name"],
            }

        self.info_layout.addStretch(1)

    def _make_maneuver_row(self, event: dict, unit: str) -> QFrame:
        row = QFrame()
        row.setObjectName("maneuverRow")
        row.setProperty("active", False)
        v = QVBoxLayout(row)
        v.setContentsMargins(9, 6, 9, 6)
        v.setSpacing(2)

        top = QHBoxLayout()
        name = QLabel(event.get("label", ""))
        name.setObjectName("maneuverName")
        time_label = QLabel(f"T+{event.get('time', 0.0):.2f} {unit}")
        time_label.setObjectName("maneuverTime")
        top.addWidget(name)
        top.addStretch(1)
        top.addWidget(time_label)
        v.addLayout(top)

        note = event.get("note")
        if note:
            note_label = QLabel(note)
            note_label.setObjectName("maneuverNote")
            note_label.setWordWrap(True)
            v.addWidget(note_label)

        row.setToolTip("Jump to this event")
        row.mousePressEvent = lambda _e, t=event.get("time", 0.0): self._seek_to_time(t)
        return row

    def _refresh_readouts(self, distances: dict[str, float]):
        for body_id, card in self._body_cards.items():
            if body_id in distances:
                card["dist_val"].setText(_format_distance(distances[body_id], card["unit"]))

        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if timeline:
            self.met_label.setText(f"T+{self._current_time:.2f} {timeline['unitLabel']}")
            ref_et = timeline.get("referenceEpochEt")
            if ref_et is not None:
                self.date_label.setText(_epoch_to_date_string(ref_et, self._current_time))
            self._refresh_active_event()

    def _refresh_active_event(self):
        """Name the most recent event at or before the current time, so the
        readout says which maneuver the spacecraft is on (e.g. just after a
        burn). Highlights that event's row in the maneuver list too."""
        current = None
        for e in self._events:
            if e.get("time", 0.0) <= self._current_time + 1e-9:
                current = e
        self.event_label.setText(f"▸ {current['label']}" if current else "")
        active_id = id(current) if current else None
        for e, row in getattr(self, "_event_rows", []):
            row.setProperty("active", id(e) == active_id)
            row.style().unpolish(row)
            row.style().polish(row)

        # Draw a maneuver's delta-v vector only while the scrubber is close (in
        # time) to that burn -- the nearest one within _maneuver_window -- so the
        # arrow flashes past each maneuver rather than hanging on the whole time.
        maneuver, nearest_dt = None, None
        for e in self._events:
            if "vector" not in e:
                continue
            dt = abs(e.get("time", 0.0) - self._current_time)
            if nearest_dt is None or dt < nearest_dt:
                maneuver, nearest_dt = e, dt
        if (maneuver is not None and self._maneuver_base_len > 0.0
                and nearest_dt <= self._maneuver_window):
            vec = maneuver["vector"]
            length = self._maneuver_base_len * (_norm3(vec["deltaV"]) / self._maneuver_max_dv)
            self.renderer.set_maneuver_vector(vec["position"], vec["deltaV"], length)
        else:
            self.renderer.clear_maneuver_vector()

    def _seek_to_time(self, day: float):
        """Move the scrubber to a specific timeline time (used by the maneuver
        list rows)."""
        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if not timeline:
            return
        span = (timeline["max"] - timeline["min"]) or 1.0
        frac = min(1.0, max(0.0, (day - timeline["min"]) / span))
        self.time_slider.setValue(int(round(frac * self.time_slider.maximum())))

    # -------------------------------------------------------------- Timing
    def _on_slider_moved(self, value: int):
        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if not timeline:
            return
        frac = value / self.time_slider.maximum()
        self._current_time = timeline["min"] + frac * (timeline["max"] - timeline["min"])
        self._refresh_readouts(self.renderer.set_time(self._current_time))
        self._update_tracking()
        self._update_measure()

    def _set_speed(self, speed: float):
        self._speed = speed
        btn = self._speed_buttons.get(speed)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def _cycle_speed(self, direction: int):
        """Step to the next/previous time-warp preset (the ] / [ shortcuts)."""
        try:
            idx = SPEED_PRESETS.index(self._speed)
        except ValueError:
            idx = SPEED_PRESETS.index(1.0)
        idx = max(0, min(len(SPEED_PRESETS) - 1, idx + direction))
        self._set_speed(SPEED_PRESETS[idx])

    def _step_time(self, direction: int):
        """Nudge the scrubber one step (2% of the span) forward/back."""
        if not (self._current_scene and self._current_scene.get("timeline")):
            return
        self._set_playing(False)
        step = max(1, int(round(0.02 * self.time_slider.maximum())))
        self.time_slider.setValue(
            max(0, min(self.time_slider.maximum(), self.time_slider.value() + direction * step))
        )

    def _restart_timeline(self):
        if not (self._current_scene and self._current_scene.get("timeline")):
            return
        self._set_playing(False)
        self.time_slider.setValue(0)

    def _toggle_play(self):
        if not (self._current_scene and self._current_scene.get("timeline")):
            return
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool):
        self._playing = playing
        self.play_btn.setText("⏸" if playing else "▶")
        if playing:
            self._last_tick_ms = None
            self._timer.start()
        else:
            self._timer.stop()

    def _on_tick(self):
        import time as _time

        now_ms = _time.monotonic() * 1000.0
        if self._last_tick_ms is None:
            self._last_tick_ms = now_ms
            return
        dt = (now_ms - self._last_tick_ms) / 1000.0
        self._last_tick_ms = now_ms

        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if not timeline:
            self._set_playing(False)
            return

        self._current_time += dt * self._speed
        if self._current_time >= timeline["max"]:
            self._current_time = timeline["max"]
            self._set_playing(False)

        frac = (self._current_time - timeline["min"]) / (timeline["max"] - timeline["min"])
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(frac * self.time_slider.maximum()))
        self.time_slider.blockSignals(False)
        self._refresh_readouts(self.renderer.set_time(self._current_time))
        self._update_tracking()
        self._update_measure()

    def closeEvent(self, event):  # noqa: N802 -- Qt override signature
        """Tear down cleanly on quit (incl. Cmd+Q). The embedded VTK render
        window has to release its native OpenGL context *before* Qt destroys the
        widget under it -- otherwise VTK finalizes a context Qt has already torn
        down and prints errors on the way out (the same reason tests call
        plotter.close() before window.close()). Also stop the animation timer and
        any in-flight scene-loading thread so nothing fires mid-teardown."""
        self._set_playing(False)
        self._timer.stop()
        self.plotter._resize_settle_timer.stop()  # so it can't fire post-teardown
        thread = self._load_thread
        if thread is not None:
            thread.quit()
            thread.wait(2000)
        try:
            self.plotter.close()
        except Exception:  # noqa: BLE001 -- best-effort teardown, never block quit
            pass
        super().closeEvent(event)


def _epoch_to_date_string(reference_et: float, day_offset: float) -> str:
    import datetime

    et2000 = datetime.datetime(2000, 1, 1, 11, 58, 55, 816000, tzinfo=datetime.timezone.utc)
    dt = et2000 + datetime.timedelta(seconds=reference_et + day_offset * 86400.0)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def main():
    app = QApplication(sys.argv)
    # App-level icon drives the macOS Dock tile / taskbar entry; the window
    # inherits it for its title bar too.
    app.setWindowIcon(app_icon())
    window = MissionControlWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
