"""Designer + replay + compare container: Mission Control's Flight Lab mode
(FL-502, FL-503).

Flight Lab has its own mode in Mission Control (the sidebar's Missions |
Flight Lab switch), not a row among the missions. The three pages live in a
tab widget whose tab bar is hidden: :class:`FlightLabNav` (``self.nav``),
shown as the sidebar's Flight Lab page, picks the page and lists the runs.
Finishing a simulation (or hitting the input-hash cache) switches straight to
the replay page with the run loaded, and the existing Mission Control
transport shortcuts keep driving playback via the delegating methods below.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from orbitopt.flightlab.ui.compare_page import FlightLabComparePage
from orbitopt.flightlab.ui.designer_page import FlightLabDesignerPage
from orbitopt.flightlab.ui.nav import FlightLabNav
from orbitopt.flightlab.ui.replay_page import FlightLabReplayPage


class FlightLabWorkbench(QWidget):
    """Mission Control's Flight Lab tab: *Designer* and *Replay* pages with
    the designer → replay handoff wired, plus the playback delegation
    Mission Control's shortcuts/visibility handling already expect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("flightLabTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().hide()  # the sidebar's FlightLabNav picks the page
        layout.addWidget(self.tabs)
        self.designer_page = FlightLabDesignerPage(self)
        self.replay_page = FlightLabReplayPage(self)
        self.tabs.addTab(self.designer_page, "Designer")
        self.compare_page = FlightLabComparePage(self)
        self.tabs.addTab(self.replay_page, "Replay")
        self.tabs.addTab(self.compare_page, "Compare")
        self.designer_page.run_ready.connect(self._open_run)
        self.replay_page.run_loaded.connect(self._replay_loaded)

        self.nav = FlightLabNav()
        self.nav.page_selected.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.nav.select_page)
        self.nav.run_activated.connect(self._open_run)
        self.nav.open_clicked.connect(self.replay_page._choose_run_directory)
        self.compare_page.runs_changed.connect(self._refresh_runs)
        self._refresh_runs()

    def _refresh_runs(self):
        self.nav.set_runs(self.compare_page.known_runs(), self.replay_page.run_dir)

    def _replay_loaded(self, _run_id: str):
        if self.replay_page.run_dir is not None:
            self.compare_page.add_run(self.replay_page.run_dir)
        self.tabs.setCurrentWidget(self.replay_page)

    def _open_run(self, run_dir: str):
        self.replay_page.load_run(Path(run_dir))
        self.compare_page.add_run(Path(run_dir))
        self.tabs.setCurrentWidget(self.replay_page)
        self._refresh_runs()  # the loaded run is marked current

    # --------------------------------------------------- playback delegation
    def stop_playback(self):
        self.replay_page.stop_playback()

    def toggle_playback(self):
        self.replay_page.toggle_playback()

    def restart(self):
        self.replay_page.restart()

    def step_time(self, dt: float):
        self.replay_page.step_time(dt)

    def seek_relative_event(self, direction: int):
        self.replay_page.seek_relative_event(direction)

    # ------------------------------------------------------------- teardown
    def terminate_compute(self):
        """Cancel/stop any in-flight simulate; called from app shutdown."""
        self.designer_page.terminate_compute()
