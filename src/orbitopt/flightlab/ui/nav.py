"""Flight Lab's page of Mission Control's sidebar: the flight runs on disk.

Flight Lab renders runs, it does not make them: runs come from code and the
CLI (``orbitopt flight simulate -o DIR``, ``orbitopt launch -o DIR``).
This page lists every run found under the watched folders
(:func:`orbitopt.flightlab.run_index.default_run_roots` plus any folder added
with *Add folder…*, remembered across sessions), the way the Missions page
lists missions: click one and it loads into the same 3D view, timeline and
panels. *Compare runs…* opens the side-by-side comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.run_index import RunEntry, default_run_roots, discover_flight_runs
from orbitopt.viz import icons
from orbitopt.viz.theme import DANGER, INK_MUTED, SUCCESS

#: QSettings key of the folders added with "Add folder…".
SETTINGS_ROOTS = "flightlab/run_roots"


def run_verdict(run_dir: Path) -> bool | None:
    """The run's ``mission_constraints_passed`` (None: not assessed or the
    file is unreadable) -- the marker beside it in the run list."""
    try:
        doc = json.loads((Path(run_dir) / "verification.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = doc.get("mission_constraints_passed")
    return value if isinstance(value, bool) else None


def _verdict_color(verdict: bool | None) -> str:
    return INK_MUTED if verdict is None else (SUCCESS if verdict else DANGER)


def _verdict_word(verdict: bool | None) -> str:
    return "not assessed" if verdict is None else ("mission passed" if verdict else "mission failed")


class FlightLabNav(QWidget):
    """Signals: ``run_activated(run_dir, label)`` (a run row was picked) and
    ``compare_clicked()``."""

    run_activated = Signal(str, str)
    compare_clicked = Signal()

    def __init__(self, parent: QWidget | None = None, *, settings: QSettings | None = None,
                 default_roots: list[Path] | None = None):
        super().__init__(parent)
        self.setObjectName("flightLabNav")
        self._settings = settings if settings is not None else QSettings("orbitopt", "Mission Control")
        self._default_roots = list(default_roots) if default_roots is not None else default_run_roots()
        self.entries: list[RunEntry] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("FLIGHT RUNS")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)
        self.run_list = QListWidget()
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.run_list.currentRowChanged.connect(self._row_changed)
        layout.addWidget(self.run_list, stretch=1)
        self.empty_label = QLabel(
            "No runs found. Make one from code or the CLI (orbitopt flight simulate -o DIR, "
            "orbitopt launch -o DIR), then Add folder… or Refresh.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(f"color: {INK_MUTED}; padding: 6px 4px;")
        layout.addWidget(self.empty_label)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.add_button = QPushButton("Add folder…")
        self.add_button.setToolTip("Watch a folder for run directories (a run directory itself works too)")
        self.add_button.clicked.connect(self._choose_folder)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Look for new runs in the watched folders")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.add_button, 1)
        row.addWidget(self.refresh_button, 1)
        layout.addLayout(row)
        self.compare_button = QPushButton("Compare runs…")
        self.compare_button.setToolTip("Two runs side by side: inputs, outcomes and why they differ")
        self.compare_button.clicked.connect(self.compare_clicked)
        layout.addWidget(self.compare_button)
        self.refresh()

    # ------------------------------------------------------------ folders
    def roots(self) -> list[Path]:
        return self._default_roots + self.added_roots()

    def added_roots(self) -> list[Path]:
        raw = self._settings.value(SETTINGS_ROOTS, [])
        if isinstance(raw, str):
            raw = [raw]
        return [Path(p) for p in (raw or []) if p]

    def add_root(self, folder: Path | str, *, select: bool = True) -> None:
        """Watch ``folder`` from now on (persisted) and list its runs; with
        ``select`` the newest run under it becomes current."""
        folder = Path(folder).expanduser().resolve()
        added = self.added_roots()
        if folder not in added and folder not in self._default_roots:
            self._settings.setValue(SETTINGS_ROOTS, [str(p) for p in added + [folder]])
        self.refresh()
        if select:
            inside = [e for e in self.entries if folder == e.path.resolve() or folder in e.path.resolve().parents]
            if inside:
                self.select_path(inside[0].path)

    def _choose_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Add a folder of Flight Lab runs", str(Path.cwd()))
        if path:
            self.add_root(path)

    # --------------------------------------------------------------- runs
    def refresh(self) -> None:
        """Re-scan the watched folders, keeping the current run selected."""
        current = self.current_path()
        self.entries = discover_flight_runs(self.roots())
        self.run_list.blockSignals(True)
        self.run_list.clear()
        labels = [e.label for e in self.entries]
        for entry in self.entries:
            text = entry.label
            if labels.count(text) > 1:  # same folder name in two places: say where
                text += f" · {entry.path.parent.parent.name if entry.path.name == 'ascent' else entry.path.parent.name}"
            item = QListWidgetItem(icons.mission_marker_icon(_verdict_color(entry.verdict)), text)
            item.setData(Qt.UserRole, str(entry.path))
            item.setToolTip(f"{entry.detail}\n{entry.path}\n{_verdict_word(entry.verdict)}")
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)
        if current is not None:
            self.select_path(current, emit=False)
        self.empty_label.setVisible(not self.entries)

    def run_paths(self) -> list[str]:
        return [self.run_list.item(i).data(Qt.UserRole) for i in range(self.run_list.count())]

    def current_path(self) -> Path | None:
        item = self.run_list.currentItem()
        return Path(item.data(Qt.UserRole)) if item is not None else None

    def select_path(self, path: Path | str, *, emit: bool = True) -> bool:
        target = str(Path(path))
        for i in range(self.run_list.count()):
            if self.run_list.item(i).data(Qt.UserRole) == target:
                if not emit:
                    self.run_list.blockSignals(True)
                self.run_list.setCurrentRow(i)
                self.run_list.blockSignals(False)
                return True
        return False

    def select_first(self) -> bool:
        if self.run_list.count():
            self.run_list.setCurrentRow(0)
            return True
        return False

    def set_busy(self, busy: bool) -> None:
        for widget in (self.run_list, self.add_button, self.refresh_button, self.compare_button):
            widget.setEnabled(not busy)

    def _row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.run_list.item(row)
        self.run_activated.emit(item.data(Qt.UserRole), item.text())
