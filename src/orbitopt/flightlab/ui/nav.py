"""Flight Lab's page of Mission Control's sidebar: the Designer / Replay /
Compare navigation and the runs Flight Lab knows about.

Flight Lab is not a mission, so it is not a row in the mission list: the
sidebar's Missions | Flight Lab switch swaps the mission list for this panel
(orbitopt.viz.sidebar.MissionSidebar.set_flightlab_panel).
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz import icons
from orbitopt.viz.theme import DANGER, INK_MUTED, SUCCESS

#: (page title, tooltip), in the workbench's page order.
PAGES = (
    ("Designer", "Mission Designer + Vehicle Builder: fill the form, Simulate"),
    ("Replay", "A flown run in 3D with its telemetry, checks and recovery"),
    ("Compare", "Two runs side by side: inputs, outcomes and why they differ"),
)


def run_verdict(run_dir: Path) -> bool | None:
    """The run's ``mission_constraints_passed`` (None: not assessed or the
    file is unreadable) -- the marker beside it in the run list."""
    try:
        doc = json.loads((Path(run_dir) / "verification.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = doc.get("mission_constraints_passed")
    return value if isinstance(value, bool) else None


class FlightLabNav(QWidget):
    """Signals: ``page_selected(index)``, ``run_activated(run_dir)``,
    ``open_clicked()``."""

    page_selected = Signal(int)
    run_activated = Signal(str)
    open_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("flightLabNav")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("FLIGHT LAB")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)
        self.page_list = QListWidget()
        self.page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for title, tip in PAGES:
            item = QListWidgetItem(title)
            item.setToolTip(tip)
            self.page_list.addItem(item)
        # Exactly as tall as its rows: sized here, before the stylesheet's
        # item padding applied, it cut "Compare" off (see showEvent).
        self.page_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.page_list.currentRowChanged.connect(self._page_changed)
        layout.addWidget(self.page_list)

        runs_header = QLabel("RECENT RUNS")
        runs_header.setObjectName("sectionHeader")
        layout.addSpacing(10)
        layout.addWidget(runs_header)
        self.run_list = QListWidget()
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.run_list.itemActivated.connect(self._run_clicked)
        self.run_list.itemClicked.connect(self._run_clicked)
        layout.addWidget(self.run_list, stretch=1)
        self.empty_label = QLabel("No runs yet: Simulate one in the Designer, or open a run directory.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(f"color: {INK_MUTED}; padding: 6px 4px;")
        layout.addWidget(self.empty_label)

        self.open_button = QPushButton("Open run…")
        self.open_button.setToolTip("Open a FlightResult run directory in Replay")
        self.open_button.clicked.connect(self.open_clicked)
        layout.addWidget(self.open_button)
        self.page_list.setCurrentRow(0)

    def showEvent(self, event):  # noqa: N802 -- Qt override
        super().showEvent(event)
        self._fit_page_list()

    def _fit_page_list(self) -> None:
        self.page_list.ensurePolished()
        rows = sum(self.page_list.sizeHintForRow(i) for i in range(self.page_list.count()))
        self.page_list.setFixedHeight(rows + 2 * self.page_list.frameWidth() + 2)

    # ------------------------------------------------------------ pages
    def select_page(self, index: int) -> None:
        """Follow the workbench (e.g. a finished simulate opens Replay)
        without re-emitting page_selected."""
        if self.page_list.currentRow() != index:
            self.page_list.blockSignals(True)
            self.page_list.setCurrentRow(index)
            self.page_list.blockSignals(False)

    def _page_changed(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)

    # ------------------------------------------------------------- runs
    def set_runs(self, runs: list[tuple[str, Path]], current: Path | None = None) -> None:
        self.run_list.clear()
        for label, path in runs:
            path = Path(path)
            verdict = run_verdict(path)
            color = INK_MUTED if verdict is None else (SUCCESS if verdict else DANGER)
            word = "not assessed" if verdict is None else ("mission passed" if verdict else "mission failed")
            item = QListWidgetItem(icons.mission_marker_icon(color), label.split(" · ")[0])
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(f"{label}\n{path}\n{word}")
            self.run_list.addItem(item)
            if current is not None and path == Path(current):
                self.run_list.setCurrentItem(item)
        self.empty_label.setVisible(not runs)

    def run_paths(self) -> list[str]:
        return [self.run_list.item(i).data(Qt.UserRole) for i in range(self.run_list.count())]

    def _run_clicked(self, item: QListWidgetItem) -> None:
        self.run_activated.emit(item.data(Qt.UserRole))
