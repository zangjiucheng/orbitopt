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

Run: python examples/09_mission_control_app.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCursor
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
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from orbitopt.viz.pv_viewer import load_scene
from orbitopt.viz.scene_renderer import SceneRenderer
from orbitopt.viz.theme import BG_VOID, STYLESHEET

SPEED_PRESETS = [0.15, 0.5, 1.0, 5.0, 20.0, 100.0]


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

    return [
        ("Solar System", load_solar_system),
        ("Artemis II — Free Return", load_artemis2),
    ]


def _format_distance(value: float, unit: str) -> str:
    if unit == "AU":
        return f"{value:,.3f} AU"
    return f"{value:,.0f} km"


class MissionControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mission Control — orbitopt")
        self.resize(1500, 950)
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

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._last_tick_ms = None

        self._build_ui()
        self._populate_builtin_missions()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self._build_title_bar())

        self.plotter = QtInteractor(center)
        self.plotter.set_background(BG_VOID)
        self.plotter.enable_anti_aliasing()
        self.renderer = SceneRenderer(self.plotter)
        center_layout.addWidget(self.plotter, stretch=1)

        self.timeline_bar = self._build_timeline_bar()
        center_layout.addWidget(self.timeline_bar)
        root.addWidget(center, stretch=1)

        root.addWidget(self._build_info_panel())

        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(0)

        title = QLabel("ORBITOPT MISSION CONTROL")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        header = QLabel("MISSIONS")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        self.mission_list = QListWidget()
        self.mission_list.currentRowChanged.connect(self._on_mission_selected)
        layout.addWidget(self.mission_list, stretch=1)

        self._open_file_btn = QPushButton("Open scene file…")
        self._open_file_btn.clicked.connect(self._open_file)
        layout.addWidget(self._open_file_btn)
        layout.setContentsMargins(8, 10, 8, 10)

        return sidebar

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)
        self.scene_title_label = QLabel("Select a mission")
        self.scene_title_label.setObjectName("sceneTitle")
        self.scene_subtitle_label = QLabel("")
        self.scene_subtitle_label.setObjectName("sceneSubtitle")
        self.scene_subtitle_label.setWordWrap(True)
        layout.addWidget(self.scene_title_label)
        layout.addWidget(self.scene_subtitle_label)
        return bar

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
        for i, s in enumerate(SPEED_PRESETS):
            btn = QPushButton(f"{s:g}x")
            btn.setCheckable(True)
            btn.setChecked(s == self._speed)
            btn.clicked.connect(lambda _checked, s=s: self._set_speed(s))
            self.speed_group.addButton(btn)
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
        readout_row.addWidget(self.met_label)
        readout_row.addWidget(self.date_label)
        readout_row.addStretch(1)
        slider_col.addLayout(readout_row)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 10000)
        self.time_slider.valueChanged.connect(self._on_slider_moved)
        slider_col.addWidget(self.time_slider)
        layout.addLayout(slider_col, stretch=1)

        return bar

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("infoPanel")
        panel.setFixedWidth(260)
        self.info_layout = QVBoxLayout(panel)
        self.info_layout.setContentsMargins(10, 10, 10, 10)
        self.info_layout.setSpacing(8)

        header = QLabel("BODIES")
        header.setObjectName("sectionHeader")
        self.info_layout.addWidget(header)
        self.info_layout.addStretch(1)
        self._body_cards: dict[str, dict] = {}
        return panel

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("Open scene file…")
        open_action.triggered.connect(self._open_file)
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("&View")
        reset_action = view_menu.addAction("Reset camera")
        reset_action.triggered.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))

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
            self._apply_scene(self._scene_cache[name])
            return

        self._start_loading(name)

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
        """
        self._loading = True
        self.mission_list.setEnabled(False)
        self._open_file_btn.setEnabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.statusBar().showMessage(f"Computing {name}…")

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
        self._loading = False
        self.mission_list.setEnabled(True)
        self._open_file_btn.setEnabled(True)
        QApplication.restoreOverrideCursor()
        self._load_thread = None
        self._load_worker = None

    def _on_scene_loaded(self, name: str, scene: dict):
        self._scene_cache[name] = scene
        self._apply_scene(scene)
        self.statusBar().showMessage("Ready")

    def _on_scene_load_failed(self, name: str, error_msg: str):
        QMessageBox.critical(self, f"Failed to load {name}", error_msg)
        self.statusBar().showMessage("Ready")

    def _apply_scene(self, scene: dict):
        self._current_scene = scene
        self.renderer.load(scene)

        self.scene_title_label.setText(scene["title"])
        self.scene_subtitle_label.setText(scene.get("subtitle", "").replace("\n", "  "))

        timeline = scene.get("timeline")
        self.timeline_bar.setVisible(timeline is not None)
        if timeline:
            self._current_time = timeline["min"]
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
        else:
            self._current_time = 0.0

        self._rebuild_body_cards(scene)
        self._refresh_readouts(self.renderer.set_time(self._current_time))
        self.plotter.render()

    # --------------------------------------------------------------- Cards
    def _rebuild_body_cards(self, scene: dict):
        for card in self._body_cards.values():
            card["frame"].deleteLater()
        self._body_cards = {}

        stretch_item = self.info_layout.takeAt(self.info_layout.count() - 1)
        del stretch_item

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

            frame.mousePressEvent = lambda _e, bid=body["id"]: self._focus_body(bid)

            self.info_layout.addWidget(frame)
            self._body_cards[body["id"]] = {"frame": frame, "dist_val": dist_val, "unit": scene["distanceUnit"]}

        self.info_layout.addStretch(1)

    def _focus_body(self, body_id: str):
        self.renderer.focus_on(body_id, self._current_time)

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

    # -------------------------------------------------------------- Timing
    def _on_slider_moved(self, value: int):
        timeline = self._current_scene.get("timeline") if self._current_scene else None
        if not timeline:
            return
        frac = value / self.time_slider.maximum()
        self._current_time = timeline["min"] + frac * (timeline["max"] - timeline["min"])
        self._refresh_readouts(self.renderer.set_time(self._current_time))

    def _set_speed(self, speed: float):
        self._speed = speed

    def _toggle_play(self):
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


def _epoch_to_date_string(reference_et: float, day_offset: float) -> str:
    import datetime

    et2000 = datetime.datetime(2000, 1, 1, 11, 58, 55, 816000, tzinfo=datetime.timezone.utc)
    dt = et2000 + datetime.timedelta(seconds=reference_et + day_offset * 86400.0)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def main():
    app = QApplication(sys.argv)
    window = MissionControlWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
