"""Mission Control's top navigation: the workspace bar and the command palette.

The window used to stack every destination into one left column (a Missions |
Flight Lab switch, then Runs | Vehicles inside it). It is now three layers:

1. **Workspace bar** (top, always visible): *Missions · Runs · Vehicles ·
   Compare* -- what kind of thing you are looking at;
2. **Context list** (left, collapsible): only the list of the current
   workspace -- the missions, the runs, or the vehicles;
3. **Inspector** (right): the tabs for the thing selected.

A breadcrumb in the title bar says where you are (``Runs › e505 · recovery``)
and the **command palette** (Ctrl/Cmd+K) jumps to any mission, run, vehicle or
workspace by typing -- the shortcut layer over the visible navigation, never
the only way to get anywhere.

:func:`filter_items` is the palette's matching, Qt-free so it is testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["CommandPalette", "PaletteItem", "WORKSPACES", "WorkspaceBar", "filter_items"]

MISSIONS, RUNS, VEHICLES, COMPARE = "missions", "runs", "vehicles", "compare"
#: (id, label, tooltip), in display order
WORKSPACES = (
    (MISSIONS, "Missions", "Mission scenes in the 3D view"),
    (RUNS, "Runs", "Flight runs: replay, telemetry, checks"),
    (VEHICLES, "Vehicles", "The vehicle catalog, read-only"),
    (COMPARE, "Compare", "Two runs side by side"),
)


@dataclass(frozen=True)
class PaletteItem:
    """One destination of the palette: ``kind`` (``Mission``, ``Run``,
    ``Vehicle``, ``Workspace``), what to show, and what to do."""

    kind: str
    label: str
    detail: str = ""
    action: Callable[[], None] = field(default=lambda: None, compare=False, repr=False)


def filter_items(items: Iterable[PaletteItem], query: str, limit: int = 40) -> list[PaletteItem]:
    """The items matching ``query``, best first. Every whitespace-separated
    word must occur (case-insensitively) in the kind, label or detail; a word
    that starts the label, or starts a word of it, ranks higher. An empty query
    keeps the given order."""
    words = query.lower().split()
    scored: list[tuple[float, int, PaletteItem]] = []
    for order, item in enumerate(items):
        label, kind, detail = item.label.lower(), item.kind.lower(), item.detail.lower()
        haystack = f"{kind} {label} {detail}"
        if not all(w in haystack for w in words):
            continue
        score = 0.0
        for w in words:
            if label.startswith(w):
                score += 3.0
            elif any(part.startswith(w) for part in label.replace("·", " ").replace("-", " ").split()):
                score += 2.0
            elif w in label:
                score += 1.0
        scored.append((-score, order, item))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored[:limit]]


class WorkspaceBar(QWidget):
    """The top bar: the product mark, the four workspaces, the palette hint."""

    workspace_changed = Signal(str)
    palette_requested = Signal()
    sidebar_toggle_requested = Signal()
    inspector_toggle_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("workspaceBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)
        self.panel_toggles: dict[str, QPushButton] = {}
        self.panel_toggles["left"] = self._panel_toggle(
            "left", "Show / hide the list (Ctrl+B)", self.sidebar_toggle_requested)
        layout.addWidget(self.panel_toggles["left"])
        layout.addSpacing(8)
        from orbitopt.viz.icon import render_logo_mark
        from orbitopt.viz.theme import INK_PRIMARY

        logo = QLabel()
        logo.setObjectName("workspaceLogo")
        logo.setPixmap(render_logo_mark(40, INK_PRIMARY).scaled(
            22, 22, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setFixedSize(26, 26)
        layout.addWidget(logo)
        layout.addSpacing(8)
        mark = QLabel("ORBITOPT")
        mark.setObjectName("workspaceMark")
        layout.addWidget(mark)
        layout.addSpacing(22)
        self.buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for key, text, tip in WORKSPACES:
            btn = QPushButton(text.upper())
            btn.setObjectName("workspaceTab")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setCursor(Qt.PointingHandCursor)
            self._group.addButton(btn)
            self.buttons[key] = btn
            layout.addWidget(btn)
            btn.toggled.connect(lambda on, k=key: on and self.workspace_changed.emit(k))
        layout.addStretch(1)
        self.search_button = QPushButton("Search   Ctrl K")
        self.search_button.setObjectName("workspaceSearch")
        self.search_button.setToolTip("Jump to any mission, run or vehicle (Ctrl/Cmd+K)")
        self.search_button.clicked.connect(self.palette_requested)
        layout.addWidget(self.search_button)
        layout.addSpacing(6)
        self.panel_toggles["right"] = self._panel_toggle(
            "right", "Show / hide the inspector (Ctrl+Shift+B)", self.inspector_toggle_requested)
        layout.addWidget(self.panel_toggles["right"])
        self.setFixedHeight(40)

    def _panel_toggle(self, side: str, tip: str, signal) -> QPushButton:
        from orbitopt.viz import icons

        btn = QPushButton()
        btn.setObjectName("panelToggle")
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setIconSize(QSize(18, 18))
        btn.setIcon(icons.panel_toggle_icon(side, True))
        btn.setProperty("side", side)
        btn.clicked.connect(signal)
        return btn

    def set_panel_open(self, side: str, open_: bool) -> None:
        """Draw the toggle of ``side`` as open (pane filled) or collapsed."""
        from orbitopt.viz import icons

        self.panel_toggles[side].setIcon(icons.panel_toggle_icon(side, open_))
        self.panel_toggles[side].setProperty("open", open_)

    def set_panel_enabled(self, side: str, enabled: bool) -> None:
        self.panel_toggles[side].setEnabled(enabled)

    @property
    def workspace(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), MISSIONS)

    def set_workspace(self, key: str, *, emit: bool = False) -> None:
        """Show ``key`` as current (no signal unless ``emit``: the window
        mirrors its own state here)."""
        btn = self.buttons[key]
        if btn.isChecked():
            return
        self._group.blockSignals(True)
        btn.blockSignals(not emit)
        btn.setChecked(True)
        btn.blockSignals(False)
        self._group.blockSignals(False)


class CommandPalette(QDialog):
    """A search box over a list of destinations. ``open_with(items)`` shows
    it; Enter (or a double click) runs the chosen item's action and closes."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("commandPalette")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.resize(560, 380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.query = QLineEdit()
        self.query.setObjectName("paletteQuery")
        self.query.setPlaceholderText("Jump to a mission, run, vehicle or workspace…")
        self.query.textChanged.connect(self._refilter)
        self.query.installEventFilter(self)
        layout.addWidget(self.query)
        self.results = QListWidget()
        self.results.setObjectName("paletteResults")
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results.itemActivated.connect(self._activate)
        layout.addWidget(self.results, 1)
        self._items: list[PaletteItem] = []
        self._shown: list[PaletteItem] = []

    def open_with(self, items: Sequence[PaletteItem]) -> None:
        self._items = list(items)
        self.query.clear()
        self._refilter("")
        if self.parentWidget() is not None:
            top = self.parentWidget().window()
            self.move(top.x() + (top.width() - self.width()) // 2, top.y() + 90)
        self.show()
        self.query.setFocus()

    def _refilter(self, text: str) -> None:
        self._shown = filter_items(self._items, text)
        self.results.clear()
        for item in self._shown:
            row = QListWidgetItem(f"{item.kind.upper():<10} {item.label}"
                                  + (f"   {item.detail}" if item.detail else ""))
            self.results.addItem(row)
        if self._shown:
            self.results.setCurrentRow(0)

    def _activate(self, row_item: QListWidgetItem | None = None) -> None:
        row = self.results.row(row_item) if row_item is not None else self.results.currentRow()
        if 0 <= row < len(self._shown):
            action = self._shown[row].action
            self.close()
            action()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        if obj is self.query and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                step = 1 if key == Qt.Key_Down else -1
                row = min(max(self.results.currentRow() + step, 0), max(self.results.count() - 1, 0))
                self.results.setCurrentRow(row)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._activate()
                return True
        return super().eventFilter(obj, event)
