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

import math
import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
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

from orbitopt.missions import list_missions
from orbitopt.viz import icons
from orbitopt.viz.icon import app_icon
from orbitopt.viz.pv_viewer import load_scene
from orbitopt.viz.scene_renderer import SceneRenderer
from orbitopt.viz.theme import ACCENT, BG_VOID, INK_MUTED, INK_SECONDARY, STYLESHEET

# Time-warp speed is a continuous log-scale control (0.001x .. 1000x), not
# fixed presets, so it can be fine-tuned to any rate. self._speed is in
# simulated *days* per real second (see _on_tick), so even the old floor of
# 0.1x still meant 1 simulated hour flew by in under half a real second --
# too fast to watch a close lunar flyby unfold. At the current floor,
# 0.001x, 1 simulated hour takes ~41.7 real seconds and 1 simulated minute
# takes <1 real second, actually slow enough to watch closely.
SPEED_MIN = 0.001
SPEED_MAX = 1000.0
SPEED_SLIDER_STEPS = 1000

# Auto-slow near a timeline event (TLI burn, closest approach, entry interface,
# ...) so fast-forwarding through a mission doesn't blow past the moments that
# actually matter -- the window is a *fraction* of the mission's own timeline
# span (matching _maneuver_window's existing precedent, see _apply_scene),
# not a fixed day count, since a short and a long mission should each get a
# proportionally-sized "slow down here" zone.
#
# This has to clamp to an ABSOLUTE speed near the event, not just scale the
# current speed down by a fixed ratio -- a first version did the latter
# (effective = self._speed * 0.15) and it was imperceptible at any real
# fast-forward rate: 15% of 1000x is still 150x, which crosses a sub-day
# window in a few milliseconds of real time, i.e. no visible slowdown at
# exactly the speeds where this feature matters. EVENT_SLOWDOWN_TARGET_SPEED
# is a comfortably-watchable absolute pace (simulated days per real second,
# same units as self._speed) that playback eases toward as the scrubber
# nears an event, regardless of how high self._speed itself is -- but never
# *speeds up* a deliberately slower self._speed (see the min() in
# _event_proximity_effective_speed), so a user who already set 0.01x doesn't
# get sped up just for approaching an event. The smoothstep ramp (not linear)
# avoids an audible/visible speed "snap" at the window boundary.
EVENT_SLOWDOWN_WINDOW_FRACTION = 0.05
EVENT_SLOWDOWN_TARGET_SPEED = 0.02

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
        self._install_dynamic_clipping()

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

    # KSP-style dynamic depth range. VTK's default fit-to-scene clipping can't
    # pull the near plane closer than a fixed fraction of the WHOLE scene's far
    # plane, so in a scene tens of thousands of km across (a GEO ring) or many
    # AU (the solar system), zooming in to inspect a local stretch of orbit
    # shoves the geometry inside the near plane and it just vanishes -- capping
    # how far you can usefully zoom. Instead we set the near plane proportional
    # to the live camera-to-focal distance, so you can keep zooming in
    # arbitrarily close (tracking-station style) with the focus staying visible,
    # while the far plane still reaches the scene's far corner so distant bodies
    # keep drawing. This has to catch BOTH ways the range gets reset: pyvista's
    # wheel-zoom calls reset_camera_clipping_range() (overridden below), while
    # track_body / zoom_to_body / reset_camera just move the camera and render,
    # so we also recompute on the camera's own ModifiedEvent -- setting the
    # range there, before the render, means VTK's fit never runs.
    _CLIP_NEAR_FRACTION = 1.0e-3

    def _install_dynamic_clipping(self) -> None:
        self._clipping_busy = False
        try:
            self.camera.AddObserver("ModifiedEvent", lambda *_a: self._apply_dynamic_clipping())
        except Exception:  # noqa: BLE001 -- no camera yet is fine; the override still covers zoom
            pass

    def _apply_dynamic_clipping(self) -> None:
        if getattr(self, "_clipping_busy", False):
            return  # our own SetClippingRange re-fires ModifiedEvent -- don't recurse
        self._clipping_busy = True
        try:
            camera = self.camera
            fx, fy, fz = camera.focal_point
            px, py, pz = camera.position
            distance = math.sqrt((px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2)
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            if distance <= 0.0 or not all(
                math.isfinite(b) for b in (xmin, xmax, ymin, ymax, zmin, zmax)
            ):
                super().reset_camera_clipping_range()
                return
            far_corner = max(
                math.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)
            )
            near = max(distance * self._CLIP_NEAR_FRACTION, 1.0e-9)
            far = far_corner * 1.05 + near
            if far <= near:
                super().reset_camera_clipping_range()
                return
            camera.SetClippingRange(near, far)
        except Exception:  # noqa: BLE001 -- clipping is cosmetic; never break a render over it
            super().reset_camera_clipping_range()
        finally:
            self._clipping_busy = False

    def reset_camera_clipping_range(self):  # noqa: N802 -- overrides pyvista's snake_case method
        self._apply_dynamic_clipping()


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
    mission event (burns, flybys, insertions). Hovering near a marker shows
    every event within reach (not just one -- events that land within a few
    pixels of each other, e.g. a burn immediately followed by an SOI-exit
    note, used to be visually indistinguishable and only the first-added one
    was ever reachable by click); clicking snaps to whichever is nearest.
    Falls back to a plain slider when the scene has no timeline events."""

    _HIT_TOLERANCE_PX = 7
    # A normal marker's own triangle is ~6px wide (see paintEvent's `half`),
    # so anything closer than that already visually overlaps -- clustering
    # only below 3px (an earlier value here) left genuinely-touching markers
    # rendered as two separate, ambiguous shapes instead of one clear
    # cluster mark. Confirmed empirically against a real rendered frame, not
    # assumed: the Mars mission's Mars-SOI-entry/MOI events, ~4.8px apart in
    # practice, looked like touching-but-unmerged marks at the old value.
    _CLUSTER_TOLERANCE_PX = 6

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

    def _events_near(self, x, tol):
        """Every event within `tol` px of `x`, nearest first -- not just the
        first one added, so a click/hover near a cluster of close-together
        events reaches all of them instead of being stuck on whichever
        happened to be first in the scene's event list."""
        near = sorted(
            ((abs(self._value_to_x(ev[0]) - x), ev) for ev in self._events),
            key=lambda pair: pair[0],
        )
        return [ev for dist, ev in near if dist <= tol]

    def _event_near(self, x, tol=None):
        near = self._events_near(x, self._HIT_TOLERANCE_PX if tol is None else tol)
        return near[0] if near else None

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if not self._events:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() // 2

        # Bucket by rounded pixel position so events within
        # _CLUSTER_TOLERANCE_PX draw as one bolder cluster mark instead of
        # silently overlapping into what looks like (and, before this fix,
        # behaved as) a single ordinary event.
        buckets: dict[int, list] = {}
        for value, label, note in self._events:
            x = self._value_to_x(value)
            buckets.setdefault(round(x / self._CLUSTER_TOLERANCE_PX), []).append(x)
        centers = {key: round(sum(xs) / len(xs)) for key, xs in buckets.items()}
        counts = {key: len(xs) for key, xs in buckets.items()}

        for key, x in centers.items():
            clustered = counts[key] > 1
            painter.setPen(QPen(QColor(ACCENT), 1.8 if clustered else 1.4))
            painter.setBrush(QColor(ACCENT))
            painter.drawLine(x, 7, x, cy + 2)
            half = 4.2 if clustered else 3.0
            tri = QPainterPath()
            tri.moveTo(x - half, 1.0)
            tri.lineTo(x + half, 1.0)
            tri.lineTo(x, 7.0)
            tri.closeSubpath()
            painter.drawPath(tri)
            if clustered:
                # A small dot below the marker's guide line signals "more
                # than one event here" at a glance, without needing to hover
                # first. Below, not above: the widget's actual rendered
                # height is ~16px (QSlider::handle's 15px + a hair), so a
                # mark above y=0 is silently clipped by Qt's paint-event
                # clip rect -- confirmed empirically by grabbing a real
                # rendered frame and finding zero accent-colored pixels in
                # that band, not assumed from the geometry alone.
                painter.drawEllipse(QRectF(x - 1.6, cy + 4.0, 3.2, 3.2))
        painter.end()

    def mouseMoveEvent(self, event):  # noqa: N802
        near = self._events_near(int(event.position().x()), self._HIT_TOLERANCE_PX)
        text = "\n\n".join(f"{label}\n{note}".strip() for _value, label, note in near)
        self.setToolTip(text)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        ev = self._event_near(int(event.position().x()))
        if ev is not None:
            self.setValue(ev[0])  # snap to the nearest event
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
        self._speed = 1.0
        self._playing = False

        self._loading = False
        self._load_thread: QThread | None = None
        self._load_worker: SceneLoader | None = None
        self._closing = False

        self._selected_body_id: str | None = None
        self._measure_body_id: str | None = None
        self._tracking = False
        self._lock_reference_body_id: str | None = None
        self._view_axis: str | None = None
        self._view_negative = False
        self._focus_zoomed = False
        self._pre_focus_camera: tuple | None = None
        self._events: list[dict] = []
        self._event_slowdown_enabled = True
        self._maneuver_base_len = 0.0
        self._maneuver_max_dv = 1.0
        self._maneuver_window = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._last_tick_ms = None

        # Always-on (not gated by _playing/timeline presence, see
        # ROTATION_SIM_HOURS_PER_REAL_SECOND) -- started once here, stopped
        # only in closeEvent.
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setInterval(33)
        self._rotation_timer.timeout.connect(self._on_rotation_tick)
        self._last_rotation_tick_ms = None
        self._rotation_timer.start()

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
        self.right_rail = self._build_rail(icons.chevron_left_icon, self._toggle_info_panel, "Show body info")
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
        self.sidebar_collapse = QPushButton()
        self.sidebar_collapse.setObjectName("panelCollapse")
        self.sidebar_collapse.setIcon(icons.chevron_left_icon())
        self.sidebar_collapse.setIconSize(QSize(12, 12))
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

        face_label = QLabel("FACE")
        face_label.setObjectName("viewLabel")
        row.addWidget(face_label)

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
        row.addWidget(self.lock_combo)

        help_sep = QFrame()
        help_sep.setObjectName("viewSep")
        help_sep.setFrameShape(QFrame.VLine)
        row.addWidget(help_sep)

        self.shortcuts_btn = QPushButton()
        self.shortcuts_btn.setObjectName("helpButton")
        self.shortcuts_btn.setIcon(icons.keyboard_icon())
        self.shortcuts_btn.setIconSize(QSize(13, 13))
        self.shortcuts_btn.setToolTip("Keyboard shortcuts (F1)")
        self.shortcuts_btn.clicked.connect(self._show_shortcuts_dialog)
        row.addWidget(self.shortcuts_btn)
        return box

    def _build_timeline_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("timelineBar")
        bar.setFixedHeight(72)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        self.play_btn = QPushButton()
        self.play_btn.setObjectName("playButton")
        self.play_btn.setIcon(icons.play_icon())
        self.play_btn.setIconSize(QSize(15, 15))
        self.play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_btn)

        speed_col = QVBoxLayout()
        speed_col.setSpacing(2)
        speed_header = QHBoxLayout()
        speed_header.addWidget(QLabel("TIME WARP"))
        self.auto_slow_btn = QPushButton()
        self.auto_slow_btn.setObjectName("autoSlowButton")
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT))
        self.auto_slow_btn.setIconSize(QSize(12, 12))
        self.auto_slow_btn.setCheckable(True)
        self.auto_slow_btn.setChecked(True)
        self.auto_slow_btn.setToolTip(
            "Auto-slow near timeline events (on by default) -- click to play "
            "through events at full speed instead"
        )
        self.auto_slow_btn.toggled.connect(self._set_event_slowdown_enabled)
        speed_header.addWidget(self.auto_slow_btn)
        speed_header.addStretch(1)
        self.speed_value_label = QLabel(self._format_speed(self._speed))
        self.speed_value_label.setObjectName("speedValue")
        speed_header.addWidget(self.speed_value_label)
        speed_col.addLayout(speed_header)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setObjectName("speedSlider")
        self.speed_slider.setRange(0, SPEED_SLIDER_STEPS)
        self.speed_slider.setValue(self._slider_from_speed(self._speed))
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)
        speed_col.addWidget(self.speed_slider)
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

        # Fixed start/end scale reference under the bar: the MET readout above
        # only ever shows the *current* scrubbed position, so there was no way
        # to tell at a glance how long the whole mission span is -- easy to
        # misjudge for a mission like Mars (300+ days) where nearly every event
        # piles up within the first/last ~1% of the bar.
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        self.scale_start_label = QLabel("")
        self.scale_start_label.setObjectName("scaleReadout")
        self.scale_end_label = QLabel("")
        self.scale_end_label.setObjectName("scaleReadout")
        self.scale_end_label.setAlignment(Qt.AlignRight)
        scale_row.addWidget(self.scale_start_label)
        scale_row.addStretch(1)
        scale_row.addWidget(self.scale_end_label)
        slider_col.addLayout(scale_row)

        layout.addLayout(slider_col, stretch=1)

        return bar

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("infoPanel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header_row = QHBoxLayout()
        self.info_collapse = QPushButton()
        self.info_collapse.setObjectName("panelCollapse")
        self.info_collapse.setIcon(icons.chevron_right_icon())
        self.info_collapse.setIconSize(QSize(12, 12))
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

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts_action = help_menu.addAction("Keyboard Shortcuts…", self._show_shortcuts_dialog)
        shortcuts_action.setShortcut(QKeySequence(Qt.Key_F1))

    # ----------------------------------------------------------------- Help
    # Every entry here matches an actual wired shortcut (see _build_menu
    # above) or a documented mouse gesture (see _on_card_clicked) -- kept as
    # one static list so this panel can't silently drift out of sync with
    # what the app actually does.
    _SHORTCUTS = [
        ("Space", "Play / pause"),
        ("← / →", "Step back / forward"),
        ("Home", "Restart timeline"),
        ("[ / ]", "Slower / faster time-warp"),
        ("T", "Track selected body"),
        ("M", "Focus (zoom to) selected body"),
        ("F1", "Show this panel"),
        ("Ctrl+Q / ⌘Q", "Quit"),
        ("Click", "Select a body"),
        ("Ctrl/⌘-click", "Measure distance to another body"),
    ]

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
        # A plain select recenters but doesn't zoom (that's what M is for) --
        # if a previous M-zoom is still active for a different body, drop it
        # rather than leave a stale "press M to restore" state pointing at
        # a camera position framed for a body that's no longer selected.
        self._focus_zoomed = False
        self._pre_focus_camera = None
        self._refresh_card_highlights()
        self.renderer.track_body(body_id, self._current_time, reference_body_id=self._lock_reference_body_id)
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

    def _update_measure(self, render: bool = True):
        """Redraw the measurement line + readout for the current (A, B) pair, or
        clear it if a full pair isn't selected. Called on selection changes and
        every timeline step so the distance tracks the bodies as they move.
        ``render=False`` for the hot playback-tick path -- see _on_tick, which
        batches its scene mutations into one render instead of one each."""
        a, b = self._selected_body_id, self._measure_body_id
        cards = self._body_cards
        if a and b and a in cards and b in cards:
            dist = self.renderer.measure_line(a, b, self._current_time, render=render)
            self.measure_pair_label.setText(f"{cards[a]['name']}  ↔  {cards[b]['name']}")
            self.measure_dist_label.setText(_format_distance(dist, cards[a]["unit"]))
            self.measure_card.setVisible(True)
        else:
            self.renderer.clear_measure_line(render=render)
            self.measure_card.setVisible(False)

    def _focus_selected(self):
        """KSP-style focus (the M shortcut): the first press zooms the camera
        in on the selected body, remembering exactly where the camera was so
        a second press zooms back out to that same view -- not a fixed
        reset, an actual toggle. Selecting a different body (see
        _select_body) or loading a new scene (see _apply_scene) drops the
        saved state, so M always starts a fresh zoom-in on whatever's
        currently selected rather than "restoring" an unrelated old view."""
        if not (self._selected_body_id and self._selected_body_id in self._body_cards):
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
            self.renderer.zoom_to_body(self._selected_body_id, self._current_time)
            self._focus_zoomed = True

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
                self.renderer.track_body(
                    self._selected_body_id, self._current_time, reference_body_id=self._lock_reference_body_id
                )
                self.statusBar().showMessage(f"Tracking {self._body_cards[self._selected_body_id]['name']}")
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
        if self._selected_body_id and self._selected_body_id in self._body_cards:
            self.renderer.track_body(
                self._selected_body_id, self._current_time, reference_body_id=self._lock_reference_body_id
            )

    def _update_tracking(self, dt: float | None = None, render: bool = True):
        """Follow the tracked body after a timeline change, if tracking is on.
        ``dt`` is passed through for smoothing on continuous playback ticks;
        omitted (snaps instantly) for a manual slider drag. ``render=False``
        for the hot playback-tick path -- see _on_tick."""
        if self._tracking and self._selected_body_id and self._selected_body_id in self._body_cards:
            self.renderer.track_body(
                self._selected_body_id, self._current_time, dt=dt,
                reference_body_id=self._lock_reference_body_id, render=render,
            )

    # ------------------------------------------------------------- Missions
    def _populate_builtin_missions(self):
        # Sourced from orbitopt.missions -- the registry shared with the CLI,
        # so a mission (built-in or a third-party plugin) added there shows up
        # here with no separate list to keep in sync.
        for mission in list_missions():
            self._loaders[mission.title] = mission.load
            item = QListWidgetItem(mission.title)
            item.setData(Qt.UserRole, mission.title)
            self.mission_list.addItem(item)
        if self.mission_list.count():
            self.mission_list.setCurrentRow(0)

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
        # key (stashed on the item via Qt.UserRole) is the full path.
        key = os.path.abspath(path)
        label = os.path.basename(path)
        self._loaders[key] = (lambda p=path: load_scene(p))
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, key)
        self.mission_list.addItem(item)
        self.mission_list.setCurrentRow(self.mission_list.count() - 1)

    def _on_mission_selected(self, row: int):
        if row < 0 or self._loading:
            return
        item = self.mission_list.item(row)
        key = item.data(Qt.UserRole)
        label = item.text()
        self._set_playing(False)

        if key in self._scene_cache:
            # A cached scene skips the ~10s compute but still stutters: applying
            # it runs SceneRenderer.load, which tears down and rebuilds every
            # VTK actor synchronously on the main thread. Show the loading page
            # first, then apply on the next event-loop turn so the page actually
            # paints before that blocking rebuild -- otherwise the swap and the
            # freeze happen in the same turn and the user sees only the freeze.
            self._begin_switch(f"Loading {label}…")
            scene = self._scene_cache[key]
            QTimer.singleShot(0, lambda: self._finish_cached(scene))
            return

        self._start_loading(key, label)

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
        # try/finally: if applying the scene raises (e.g. a malformed cached
        # scene dict), _end_switch must still run -- otherwise self._loading
        # stays True forever and the mission list / open-file button stay
        # disabled, requiring an app restart to recover.
        try:
            self._apply_scene(scene)
        finally:
            self._end_switch()

    def _start_loading(self, key: str, label: str | None = None):
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
        label = label if label is not None else key
        self._begin_switch(f"Computing {label}…")

        thread = QThread(self)
        worker = SceneLoader(key, self._loaders[key])
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
        if self._closing:
            # A finished signal that was already queued (posted to this
            # thread's event queue) by the time closeEvent disconnected it
            # still gets delivered once the event loop resumes -- Qt doesn't
            # retroactively cancel an already-posted call. This flag is the
            # real guard against touching self.plotter/self.renderer after
            # closeEvent has closed them; the disconnect there is just hygiene.
            return
        self._scene_cache[name] = scene
        # Applied while the loading page is still up (plotter hidden); the swap
        # back to the view happens in _on_load_thread_finished -> _end_switch.
        self._apply_scene(scene)

    def _on_scene_load_failed(self, name: str, error_msg: str):
        if self._closing:
            return
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
        self._lock_reference_body_id = None
        # Same reasoning for the M-key focus-zoom toggle: a saved "pre-focus"
        # camera state from the old scene means nothing once the bodies (and
        # their positions) it was framing are gone.
        self._focus_zoomed = False
        self._pre_focus_camera = None

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
            unit = timeline.get("unitLabel", "")
            self.scale_start_label.setText(f"{timeline['min']:.1f} {unit}".strip())
            self.scale_end_label.setText(f"{timeline['max']:.1f} {unit}".strip())
        else:
            self._current_time = 0.0
            self.time_slider.set_events([], 0.0, 1.0)
            self.scale_start_label.setText("")
            self.scale_end_label.setText("")
        self.event_label.setText("")

        # Precompute the maneuver-arrow display scale: physical delta-v (m/s) is
        # invisible next to orbit radii (km), so arrows are drawn at a fraction of
        # the burn-site radius (_maneuver_base_len), lengthened in proportion to
        # |delta-v|. That fraction is only the *wide-view cap*, though -- when the
        # camera is dollied in on the spacecraft during a burn, set_maneuver_vector
        # clamps it down to a fraction of the camera distance so the arrow doesn't
        # swamp the very body it's drawn from. The arrow only appears while the
        # scrubber is within _maneuver_window of a burn, so it flashes past each
        # maneuver instead of hanging on screen the whole time.
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

        self.lock_combo.blockSignals(True)
        self.lock_combo.clear()
        self.lock_combo.addItem("None", None)
        for body in scene["bodies"]:
            self.lock_combo.addItem(body["name"], body["id"])
        self.lock_combo.setCurrentIndex(0)
        self.lock_combo.blockSignals(False)
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

        row.setToolTip("Jump to this event and zoom in on the burn")
        row.mousePressEvent = lambda _e, ev=event: self._seek_to_event(ev)
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
            magnitude_fraction = _norm3(vec["deltaV"]) / self._maneuver_max_dv
            self.renderer.set_maneuver_vector(
                vec["position"], vec["deltaV"], self._maneuver_base_len, magnitude_fraction)
        else:
            self.renderer.clear_maneuver_vector()

    def _primary_spacecraft_id(self) -> str | None:
        """The scene's main spacecraft body (kind == "spacecraft") -- GOES,
        Orion, the Mars probe -- provided it's a real selectable body (has a
        card). The auto-frame on a maneuver click (see _seek_to_event) points
        the camera at this body, since that's what every burn is applied to."""
        scene = self._current_scene
        if not scene:
            return None
        for body in scene.get("bodies", []):
            if body.get("kind") == "spacecraft" and body.get("id") in self._body_cards:
                return body["id"]
        return None

    def _seek_to_event(self, event: dict):
        """Jump the scrubber to a maneuver's time AND auto-frame its burn --
        zoom the camera in on the spacecraft at that instant, so clicking a
        maneuver row takes you right up to the burn (and its delta-v arrow)
        instead of only moving the clock. An event with no burn vector (e.g.
        the terminal "insertion" marker) just seeks. The zoom is to an
        absolute distance (see EVENT_ZOOM_ARROW_MULTIPLE) so clicking rows in
        succession lands on the same framing each time rather than diving in
        further on every click."""
        sc_id = self._primary_spacecraft_id()
        zoom = sc_id is not None and "vector" in event and self._maneuver_base_len > 0.0
        # Remember the pre-zoom framing so the M-key focus toggle can zoom back
        # out to it -- the same contract _focus_selected sets up for its zoom.
        if zoom:
            cam = self.plotter.camera
            pre_camera = (tuple(cam.position), tuple(cam.focal_point), tuple(cam.up))
        self._seek_to_time(event.get("time", 0.0))
        if not zoom:
            return
        self._selected_body_id = sc_id
        self._measure_body_id = None
        self._refresh_card_highlights()
        self._update_measure()
        self.renderer.zoom_to_body(
            sc_id, self._current_time,
            distance_km=self._maneuver_base_len * EVENT_ZOOM_ARROW_MULTIPLE,
        )
        self._pre_focus_camera = pre_camera
        self._focus_zoomed = True
        self.statusBar().showMessage(f"▸ {event.get('label', 'maneuver')} — framed the burn")

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
        # Batched into a single render, same fix _on_tick already applies to
        # playback: a manual scrub drag fires this handler at high frequency,
        # and set_time/_update_tracking/_update_measure each doing their own
        # render made dragging visibly jankier than playback at the same
        # effective rate.
        self._refresh_readouts(self.renderer.set_time(self._current_time, render=False))
        self._update_tracking(render=False)
        self._update_measure(render=False)
        self.plotter.render()

    @staticmethod
    def _format_speed(speed: float) -> str:
        if speed >= 10:
            return f"{speed:.0f}×"
        if speed >= 1:
            return f"{speed:.1f}×"
        # Below 1x: a fixed 2 decimals (the old behavior) reads fine down to
        # SPEED_MIN's old floor of 0.1, but prints as "0.00×" for anything
        # under 0.005 -- indistinguishable from stopped now that SPEED_MIN
        # goes as low as 0.001. Scale the decimal count to keep ~2
        # significant figures visible at any speed below 1x instead.
        decimals = max(2, -int(math.floor(math.log10(speed))) + 1)
        return f"{speed:.{decimals}f}×"

    def _speed_from_slider(self, value: int) -> float:
        frac = value / SPEED_SLIDER_STEPS
        return SPEED_MIN * (SPEED_MAX / SPEED_MIN) ** frac

    def _slider_from_speed(self, speed: float) -> int:
        speed = min(SPEED_MAX, max(SPEED_MIN, speed))
        frac = math.log(speed / SPEED_MIN) / math.log(SPEED_MAX / SPEED_MIN)
        return int(round(frac * SPEED_SLIDER_STEPS))

    def _on_speed_slider(self, value: int):
        self._speed = self._speed_from_slider(value)
        self._sync_speed_readout(self._speed)

    def _set_speed(self, speed: float):
        self._speed = min(SPEED_MAX, max(SPEED_MIN, float(speed)))
        if hasattr(self, "speed_slider"):
            self.speed_slider.blockSignals(True)
            self.speed_slider.setValue(self._slider_from_speed(self._speed))
            self.speed_slider.blockSignals(False)
            self._sync_speed_readout(self._speed)

    def _cycle_speed(self, direction: int):
        """Fine multiplicative nudge of the time-warp rate (the [ / ] shortcuts);
        a continuous log step, not a jump between fixed presets."""
        self._set_speed(self._speed * (1.25 ** direction))

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
        self.play_btn.setIcon(icons.pause_icon() if playing else icons.play_icon())
        if playing:
            self._last_tick_ms = None
            self._timer.start()
        else:
            self._timer.stop()
            self._sync_speed_readout(self._speed)  # drop any leftover "eased" display

    def _set_event_slowdown_enabled(self, on: bool):
        self._event_slowdown_enabled = bool(on)
        self.auto_slow_btn.setIcon(icons.clock_icon(ACCENT if self._event_slowdown_enabled else INK_MUTED))
        if not self._playing:
            self._sync_speed_readout(self._speed)
        self.statusBar().showMessage(
            "Auto-slow near events: on" if self._event_slowdown_enabled else "Auto-slow near events: off"
        )

    def _event_proximity_effective_speed(self, timeline) -> float:
        """self._speed, eased toward EVENT_SLOWDOWN_TARGET_SPEED (an absolute
        days-per-real-second pace, not a ratio of self._speed -- see that
        constant's comment for why a ratio doesn't work) as the scrubber
        nears the closest timeline event, over EVENT_SLOWDOWN_WINDOW_FRACTION
        of the mission's own span. min() means this only ever slows playback
        down, never speeds it up past whatever the user actually chose.
        Returns self._speed unchanged (no easing) if there are no events or
        the user has switched auto-slow off via the timeline bar's toggle.
        """
        if not self._events or not self._event_slowdown_enabled:
            return self._speed
        span = (timeline["max"] - timeline["min"]) or 1.0
        window = EVENT_SLOWDOWN_WINDOW_FRACTION * span
        if window <= 0.0:
            return self._speed
        nearest_dt = min(abs(e.get("time", 0.0) - self._current_time) for e in self._events)
        if nearest_dt >= window:
            return self._speed
        near_speed = min(self._speed, EVENT_SLOWDOWN_TARGET_SPEED)
        t = nearest_dt / window
        smoothstep = t * t * (3.0 - 2.0 * t)
        return near_speed + (self._speed - near_speed) * smoothstep

    def _on_tick(self):
        import time as _time

        now_ms = _time.monotonic() * 1000.0
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

        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if not timeline:
            self._set_playing(False)
            return

        effective_speed = self._event_proximity_effective_speed(timeline)
        self._sync_speed_readout(effective_speed)
        self._current_time += dt * effective_speed
        if self._current_time >= timeline["max"]:
            self._current_time = timeline["max"]
            self._set_playing(False)

        frac = (self._current_time - timeline["min"]) / (timeline["max"] - timeline["min"])
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(frac * self.time_slider.maximum()))
        self.time_slider.blockSignals(False)
        # Batched into a single render at the end instead of one render per
        # call (set_time/track_body/measure_line each used to render on their
        # own) -- three separate GPU submits every 33ms tick made per-tick
        # cost noisy enough to read as playback jitter/stutter.
        self._refresh_readouts(self.renderer.set_time(self._current_time, render=False))
        self._update_tracking(dt=dt, render=False)
        self._update_measure(render=False)
        self.plotter.render()

    def _sync_speed_readout(self, effective_speed: float):
        """Keep the TIME WARP label showing what's *actually* playing back,
        not just the slider-set self._speed -- without this the label kept
        reading (e.g.) "10x" while auto-slow had actually eased playback down
        to 0.02x near an event, which looked like the readout and the real
        playback rate had drifted out of sync with each other."""
        eased = effective_speed < self._speed - 1e-9
        self.speed_value_label.setText(self._format_speed(effective_speed))
        self.speed_value_label.setProperty("eased", eased)
        self.speed_value_label.style().unpolish(self.speed_value_label)
        self.speed_value_label.style().polish(self.speed_value_label)

    def _on_rotation_tick(self):
        import time as _time

        now_ms = _time.monotonic() * 1000.0
        if self._last_rotation_tick_ms is None:
            self._last_rotation_tick_ms = now_ms
            return
        dt = (now_ms - self._last_rotation_tick_ms) / 1000.0
        self._last_rotation_tick_ms = now_ms

        if self._current_scene is not None:
            self.renderer.advance_rotation(dt * ROTATION_SIM_HOURS_PER_REAL_SECOND)

    def closeEvent(self, event):  # noqa: N802 -- Qt override signature
        """Tear down cleanly on quit (incl. Cmd+Q). The embedded VTK render
        window has to release its native OpenGL context *before* Qt destroys the
        widget under it -- otherwise VTK finalizes a context Qt has already torn
        down and prints errors on the way out (the same reason tests call
        plotter.close() before window.close()). Also stop the animation timer and
        any in-flight scene-loading thread so nothing fires mid-teardown."""
        self._set_playing(False)
        self._timer.stop()
        self._rotation_timer.stop()
        self._spinner._timer.stop()  # the spinner has its own QTimer, independent of self._timer
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
