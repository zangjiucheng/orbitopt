"""The right-hand inspector panel: the distance-measure readout, the mission's
maneuver list, and one card per body with its live distance and info rows.

It rebuilds itself from MissionState.scene_changed and re-highlights from
selection_changed / time_changed; clicks are reported as signals and never
change state directly, so the window decides what a click means (select,
measure, seek-and-frame).
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from orbitopt.viz import icons
from orbitopt.viz.state import MissionState


def format_distance(value: float, unit: str) -> str:
    if unit == "AU":
        return f"{value:,.3f} AU"
    return f"{value:,.0f} km"


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class InspectorPanel(QWidget):
    """Signals:
      * ``body_clicked(body_id, measure)`` -- a body card was clicked;
        ``measure`` is True for a ⌘/Ctrl-click (pick a measure partner).
      * ``event_clicked(event)`` -- a maneuver row was clicked.
      * ``collapse_clicked()`` -- the panel's own collapse chevron.
    """

    body_clicked = Signal(str, bool)
    event_clicked = Signal(object)
    collapse_clicked = Signal()

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._body_cards: dict[str, dict] = {}
        self._event_widgets: list[QWidget] = []
        self._event_rows: list[tuple[dict, QFrame]] = []
        self._active_event_id: int | None = None
        self.setObjectName("infoPanel")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._build()
        state.scene_changed.connect(self._rebuild)
        state.selection_changed.connect(self._refresh_card_highlights)
        state.time_changed.connect(lambda _t: self._refresh_active_event())

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header_row = QHBoxLayout()
        self.collapse_btn = QPushButton()
        self.collapse_btn.setObjectName("panelCollapse")
        self.collapse_btn.setIcon(icons.chevron_right_icon())
        self.collapse_btn.setIconSize(QSize(12, 12))
        self.collapse_btn.setToolTip("Collapse body-info panel")
        self.collapse_btn.clicked.connect(self.collapse_clicked)
        header = QLabel("BODIES")
        header.setObjectName("sectionHeader")
        header_row.addWidget(self.collapse_btn)
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

    # ------------------------------------------------------------ public
    def set_distances(self, distances: dict[str, float]) -> None:
        unit = self._state.scene["distanceUnit"] if self._state.scene else ""
        for body_id, card in self._body_cards.items():
            if body_id in distances:
                card["dist_val"].setText(format_distance(distances[body_id], unit))

    def distance_text(self, body_id: str) -> str:
        return self._body_cards[body_id]["dist_val"].text()

    def show_measure(self, distance: float) -> None:
        state = self._state
        a, b = state.body(state.selected_body_id), state.body(state.measure_body_id)
        self.measure_pair_label.setText(f"{a['name']}  ↔  {b['name']}")
        self.measure_dist_label.setText(format_distance(distance, state.scene["distanceUnit"]))
        self.measure_card.setVisible(True)

    def hide_measure(self) -> None:
        self.measure_card.setVisible(False)

    # --------------------------------------------------------- rebuild
    def _rebuild(self, scene: dict | None):
        for card in self._body_cards.values():
            card["frame"].deleteLater()
        self._body_cards = {}
        for widget in self._event_widgets:
            widget.deleteLater()
        self._event_widgets = []
        self._event_rows = []
        self._active_event_id = None
        self.measure_card.setVisible(False)

        stretch_item = self.info_layout.takeAt(self.info_layout.count() - 1)
        del stretch_item

        if scene is None:
            self.info_layout.addStretch(1)
            return

        timeline = scene.get("timeline")
        events = self._state.events
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
            self._add_body_card(body)

        self.info_layout.addStretch(1)
        self._refresh_active_event()

    def _add_body_card(self, body: dict):
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
        dist_val.setWordWrap(True)
        dist_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
            # Without word-wrap, a long value (e.g. "8,312 km from Moon
            # center") reports its full unwrapped width as this row's
            # (and so the whole scrollable card column's) minimum width --
            # wider than the actual panel, with horizontal scrolling off,
            # so the overflow silently became invisible instead of
            # visibly clipped. Confirmed against a real rendered frame:
            # the scroll content measured 318px against a 260px-wide
            # panel, and even *short* values on unrelated rows vanished
            # because addStretch pushed them into the same off-panel
            # region. Word-wrap keeps every row's width bounded by the
            # panel instead of by its longest value.
            val.setWordWrap(True)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r.addWidget(k)
            r.addStretch(1)
            r.addWidget(val)
            v.addLayout(r)

        frame.setToolTip("Click to focus · ⌘/Ctrl-click to measure distance")
        frame.mousePressEvent = lambda e, bid=body["id"]: self.body_clicked.emit(
            bid, bool(e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        )

        self.info_layout.addWidget(frame)
        self._body_cards[body["id"]] = {"frame": frame, "dist_val": dist_val}

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
        row.mousePressEvent = lambda _e, ev=event: self.event_clicked.emit(ev)
        return row

    # ------------------------------------------------------- highlights
    def _refresh_card_highlights(self):
        selected, measure = self._state.selected_body_id, self._state.measure_body_id
        for bid, card in self._body_cards.items():
            frame = card["frame"]
            frame.setProperty("selected", bid == selected)
            frame.setProperty("measure", bid == measure)
            _repolish(frame)

    def _refresh_active_event(self):
        """Highlight the row of the most recent event at or before the current
        time. Re-polishes only when the active event actually changes --
        time moves every playback tick, and restyling every row each time is
        wasted work."""
        current = self._state.active_event()
        active_id = id(current) if current is not None else None
        if active_id == self._active_event_id:
            return
        self._active_event_id = active_id
        for e, row in self._event_rows:
            row.setProperty("active", id(e) == active_id)
            _repolish(row)
