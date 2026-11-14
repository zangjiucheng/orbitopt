"""The left-hand sidebar: a Missions | Flight Lab mode switch over two pages.
Missions lists the built-in missions from orbitopt.missions plus any scene
file opened via File > Open, each with a load-status marker; the Flight Lab
page lists the flight runs on disk (set_flightlab_panel)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz import icons
from orbitopt.viz.theme import INK_MUTED


class MissionSidebar(QWidget):
    """Signals:
      * ``mission_activated(key, label)`` -- a row became current. ``key`` is
        the loader key (mission title, or a scene file's absolute path).
      * ``open_file_clicked()`` / ``collapse_clicked()``.
      * ``mode_changed(mode)`` -- the switch at the top went to ``"missions"``
        or ``"flightlab"``. Flight Lab has its own page of the sidebar (the
        flight runs on disk); a run then renders in the same view as a
        mission.
    """

    MISSIONS, FLIGHTLAB = "missions", "flightlab"

    mission_activated = Signal(str, str)
    mode_changed = Signal(str)
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

        # Missions | Flight Lab, with the collapse chevron serving both pages.
        switch_row = QHBoxLayout()
        switch_row.setContentsMargins(0, 8, 0, 6)
        switch_row.setSpacing(0)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, text, tip in ((self.MISSIONS, "Missions", "Mission scenes in the 3D view"),
                                (self.FLIGHTLAB, "Flight Lab", "Replay and compare launch-vehicle flight runs")):
            btn = QPushButton(text)
            btn.setObjectName("modeSwitch")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setProperty("side", "left" if mode == self.MISSIONS else "right")
            self._mode_group.addButton(btn)
            self.mode_buttons[mode] = btn
            switch_row.addWidget(btn, 1)
            btn.toggled.connect(lambda on, m=mode: on and self._on_mode_toggled(m))
        initial = self.mode_buttons[self.MISSIONS]
        initial.blockSignals(True)  # the pages below do not exist yet
        initial.setChecked(True)
        initial.blockSignals(False)
        self.collapse_btn = QPushButton()
        self.collapse_btn.setObjectName("panelCollapse")
        self.collapse_btn.setIcon(icons.chevron_left_icon())
        self.collapse_btn.setIconSize(QSize(12, 12))
        self.collapse_btn.setToolTip("Collapse the sidebar")
        self.collapse_btn.clicked.connect(self.collapse_clicked)
        switch_row.addSpacing(6)
        switch_row.addWidget(self.collapse_btn)
        layout.addLayout(switch_row)

        self.pages = QStackedWidget()
        layout.addWidget(self.pages, stretch=1)

        missions = QWidget()
        mlayout = QVBoxLayout(missions)
        mlayout.setContentsMargins(0, 0, 0, 0)
        mlayout.setSpacing(0)
        self._missions_layout = mlayout
        header = QLabel("MISSIONS")
        header.setObjectName("sectionHeader")
        mlayout.addWidget(header)

        self.mission_list = QListWidget()
        # Unlike the info panel's QScrollArea, this never had its horizontal
        # scrollbar explicitly turned off -- Qt's native, unthemed scrollbar
        # showed up as a stray gray line under the mission rows whenever an
        # item's content requested slightly more width than available.
        self.mission_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mission_list.currentRowChanged.connect(self._on_row_changed)
        mlayout.addWidget(self.mission_list, stretch=1)

        self.open_file_btn = QPushButton("Open scene file…")
        self.open_file_btn.clicked.connect(self.open_file_clicked)
        mlayout.addWidget(self.open_file_btn)
        self.pages.addWidget(missions)
        self._flightlab_panel: QWidget | None = None

    # ------------------------------------------------------------ modes
    @property
    def mode(self) -> str:
        return self.FLIGHTLAB if self.mode_buttons[self.FLIGHTLAB].isChecked() else self.MISSIONS

    def set_flightlab_panel(self, widget: QWidget) -> None:
        """The Flight Lab page of the sidebar (the flight-run list)."""
        self._flightlab_panel = widget
        self.pages.addWidget(widget)

    def set_mode(self, mode: str) -> None:
        """Switch pages as if the user clicked the switch (emits mode_changed)."""
        self.mode_buttons[mode].setChecked(True)

    def _on_mode_toggled(self, mode: str) -> None:
        if mode == self.FLIGHTLAB and self._flightlab_panel is not None:
            self.pages.setCurrentWidget(self._flightlab_panel)
        else:
            self.pages.setCurrentIndex(0)
        self.mode_changed.emit(mode)

    def add_section(self, widget: QWidget) -> None:
        """Add a panel below the mission list (above the open-file button)."""
        self._missions_layout.insertWidget(self._missions_layout.indexOf(self.open_file_btn), widget)

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
        self.mode_buttons[self.FLIGHTLAB].setEnabled(not busy)

    def _on_row_changed(self, row: int):
        if row < 0:
            return
        item = self.mission_list.item(row)
        self.mission_activated.emit(item.data(Qt.UserRole), item.text())
