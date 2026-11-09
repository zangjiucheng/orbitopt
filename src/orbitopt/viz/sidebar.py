"""The left-hand mission list: built-in missions from orbitopt.missions plus
any scene file opened via File > Open, each with a load-status marker."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from orbitopt.viz import icons
from orbitopt.viz.theme import INK_MUTED


class MissionSidebar(QWidget):
    """Signals:
      * ``mission_activated(key, label)`` -- a row became current. ``key`` is
        the loader key (mission title, or a scene file's absolute path).
      * ``open_file_clicked()`` / ``collapse_clicked()``.
    """

    mission_activated = Signal(str, str)
    open_file_clicked = Signal()
    collapse_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._items: dict[str, QListWidgetItem] = {}
        self.setObjectName("sidebar")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        self._layout = layout
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(0)

        title = QLabel("ORBITOPT MISSION CONTROL")
        title.setObjectName("appTitle")
        # The sidebar (default ~220px, narrower once the splitter is dragged)
        # is well short of this string's unwrapped width at 14px/700 weight --
        # it was silently clipped mid-word with no ellipsis. Wrap instead of
        # widening the sidebar just to fit a static branding string.
        title.setWordWrap(True)
        layout.addWidget(title)

        header_row = QHBoxLayout()
        header = QLabel("MISSIONS")
        header.setObjectName("sectionHeader")
        self.collapse_btn = QPushButton()
        self.collapse_btn.setObjectName("panelCollapse")
        self.collapse_btn.setIcon(icons.chevron_left_icon())
        self.collapse_btn.setIconSize(QSize(12, 12))
        self.collapse_btn.setToolTip("Collapse missions panel")
        self.collapse_btn.clicked.connect(self.collapse_clicked)
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(self.collapse_btn)
        layout.addLayout(header_row)

        self.mission_list = QListWidget()
        # Unlike the info panel's QScrollArea, this never had its horizontal
        # scrollbar explicitly turned off -- Qt's native, unthemed scrollbar
        # showed up as a stray gray line under the mission rows whenever an
        # item's content requested slightly more width than available.
        self.mission_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mission_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.mission_list, stretch=1)

        self.open_file_btn = QPushButton("Open scene file…")
        self.open_file_btn.clicked.connect(self.open_file_clicked)
        layout.addWidget(self.open_file_btn)

    def add_section(self, widget: QWidget) -> None:
        """Add a panel below the mission list (above the open-file button)."""
        self._layout.insertWidget(self._layout.indexOf(self.open_file_btn), widget)

    def add_mission(self, key: str, label: str, tooltip: str) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, key)
        # A long title (e.g. "Mars — Earth to Orbit Insertion") elides with
        # "…" now that the list's horizontal scrollbar is off -- the tooltip
        # is the only way to read it in full at a narrow sidebar width.
        item.setToolTip(tooltip)
        # Neutral marker until this mission has actually been loaded once (or
        # fails to) -- see set_status, driven by the real load outcome.
        item.setIcon(icons.mission_marker_icon(INK_MUTED))
        self.mission_list.addItem(item)
        self._items[key] = item

    def select_row(self, row: int) -> None:
        self.mission_list.setCurrentRow(row)

    def count(self) -> int:
        return self.mission_list.count()

    def set_status(self, key: str, color: str) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setIcon(icons.mission_marker_icon(color))

    def set_busy(self, busy: bool) -> None:
        """Disable mission switching while a load is in flight -- re-entering a
        load would leak a second thread/worker pair."""
        self.mission_list.setEnabled(not busy)
        self.open_file_btn.setEnabled(not busy)

    def _on_row_changed(self, row: int):
        if row < 0:
            return
        item = self.mission_list.item(row)
        self.mission_activated.emit(item.data(Qt.UserRole), item.text())
