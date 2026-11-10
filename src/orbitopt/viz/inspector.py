"""The right-hand inspector panel: the distance-measure readout, then up to
three framed sections -- the selected candidate, the mission's maneuver
list, and one card per body with its live distance and info rows -- each
with its own header and its own scroll, in a vertical splitter so their
share of the height can be dragged. (They used to share one scroll area,
so scrolling to a body card carried the maneuver list away with it.)

It rebuilds itself from MissionState.scene_changed and re-highlights from
selection_changed / time_changed; clicks are reported as signals and never
change state directly, so the window decides what a click means (select,
measure, seek-and-frame).
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz import icons
from orbitopt.viz.candidate_panel import (
    FLOWN_FIDELITY_LABEL,
    SCREENING_FIDELITY_LABEL,
    candidate_status,
    exported_candidate,
    format_date,
    format_dv,
)
from orbitopt.viz.state import MissionState, altitude_reference
from orbitopt.viz.timeline_bar import timeline_unit_seconds
from orbitopt.viz.timeline_view import format_met


def format_distance(value: float, unit: str) -> str:
    if unit == "AU":
        return f"{value:,.3f} AU"
    return f"{value:,.0f} km"


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class InspectorSection(QFrame):
    """A framed section: header (title + item count) over its own scroll area.

    The cards go in a scroll area, not straight into the panel: a scene with
    ~10 bodies stacks that many fixed-height cards, and without it the
    panel's minimum height is the sum of all of them -- which forced the
    whole window taller than the screen. Each section scrolling on its own
    keeps that bound and lets the maneuver list stay put while the body
    cards scroll (and the reverse)."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("inspectorSection")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        header = QWidget()
        header.setObjectName("inspectorSectionHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(10, 6, 10, 6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionHeader")
        self.count_label = QLabel("")
        self.count_label.setObjectName("sectionCount")
        h.addWidget(self.title_label)
        h.addStretch(1)
        h.addWidget(self.count_label)
        v.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("infoScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        host.setObjectName("infoCards")
        self.items = QVBoxLayout(host)
        self.items.setContentsMargins(4, 2, 4, 6)  # narrow: the panel can be dragged down to 200 px
        self.items.setSpacing(6)
        self.items.addStretch(1)
        self.scroll.setWidget(host)
        v.addWidget(self.scroll, stretch=1)

    def add(self, widget: QWidget) -> None:
        self.items.insertWidget(self.items.count() - 1, widget)  # above the trailing stretch

    def clear(self) -> None:
        while self.items.count() > 1:
            item = self.items.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                _clear_layout(item.layout())

    def set_count(self, n: int | None) -> None:
        self.count_label.setText("" if n is None else str(n))

    def item_count(self) -> int:
        return self.items.count() - 1


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
        self._event_rows: list[tuple[dict, QFrame]] = []
        self._active_event_id: int | None = None
        self._candidate_card: QFrame | None = None
        self.setObjectName("infoPanel")
        # A QWidget *subclass* only paints its stylesheet background/border
        # with this attribute set (a plain QWidget gets it implicitly).
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._build()
        state.scene_changed.connect(self._rebuild)
        state.selection_changed.connect(self._refresh_card_highlights)
        state.time_changed.connect(lambda _t: self._refresh_active_event())
        state.candidate_changed.connect(lambda _cid: self._fill_candidate_card())

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
        header = QLabel("INSPECTOR")
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

        self.candidate_section = InspectorSection("CANDIDATE")
        self.maneuver_section = InspectorSection("MANEUVERS")
        self.body_section = InspectorSection("BODIES")
        self.sections = QSplitter(Qt.Vertical)
        self.sections.setObjectName("inspectorSplitter")
        self.sections.setChildrenCollapsible(False)
        self.sections.setHandleWidth(8)
        for k, section in enumerate((self.candidate_section, self.maneuver_section, self.body_section)):
            section.setMinimumHeight(90)
            self.sections.addWidget(section)
            self.sections.setStretchFactor(k, 1)
        outer.addWidget(self.sections, stretch=1)

    # ------------------------------------------------------------ public
    def altitude_body_ids(self) -> list[str]:
        """Bodies whose card shows an altitude (it needs their position, not
        just their distance -- the ellipsoid's radius depends on latitude)."""
        return [bid for bid, card in self._body_cards.items() if card["minus"]]

    def set_distances(self, distances: dict[str, float], positions: dict | None = None) -> None:
        """Distance cards from ``distances``; altitude cards from the scene-
        frame ``positions`` of :meth:`altitude_body_ids` (height above the
        central body's drawn ellipsoid)."""
        unit = self._state.scene["distanceUnit"] if self._state.scene else ""
        ref = altitude_reference(self._state.scene)
        for body_id, card in self._body_cards.items():
            if card["dist_val"] is None:
                continue
            if card["minus"]:
                if positions is not None and body_id in positions and ref is not None:
                    card["dist_val"].setText(format_distance(ref.altitude(positions[body_id]), unit))
            elif body_id in distances:
                card["dist_val"].setText(format_distance(distances[body_id], unit))

    def distance_text(self, body_id: str) -> str:
        label = self._body_cards[body_id]["dist_val"]
        return "" if label is None else label.text()

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
        # setParent(None) takes each old widget out of the panel right away;
        # deleteLater alone waits for the event loop, and until then the old
        # cards still show above the new ones.
        for section in (self.candidate_section, self.maneuver_section, self.body_section):
            section.clear()
        self._body_cards = {}
        self._event_rows = []
        self._active_event_id = None
        self._candidate_card = None
        self.measure_card.setVisible(False)

        scene_ok = scene is not None
        has_candidates = scene_ok and bool(self._state.candidates)
        events = self._state.events if scene_ok else []
        self.candidate_section.setVisible(has_candidates)
        self.maneuver_section.setVisible(bool(events))
        self.body_section.setVisible(scene_ok)
        if not scene_ok:
            return

        if has_candidates:
            self._candidate_card = QFrame()
            self._candidate_card.setObjectName("bodyCard")
            QVBoxLayout(self._candidate_card).setContentsMargins(10, 8, 10, 8)
            self.candidate_section.add(self._candidate_card)
            self.candidate_section.set_count(None)
            self._fill_candidate_card()

        if events:
            unit = scene["timeline"].get("unitLabel", "")
            for e in events:
                row = self._make_maneuver_row(e, unit)
                self.maneuver_section.add(row)
                self._event_rows.append((e, row))
            self.maneuver_section.set_count(len(events))

        for body in scene["bodies"]:
            self._add_body_card(body)
        self.body_section.set_count(len(scene["bodies"]))
        self._balance_sections()
        # Again once Qt has laid the new cards out (their size hints and the
        # splitter's height are only final then).
        QTimer.singleShot(0, self._balance_sections)
        self._refresh_active_event()

    def _balance_sections(self) -> None:
        """Share the height by content: a short list gets just what it needs,
        the rest goes to the longer ones (the user can drag from there)."""
        visible = [s for s in (self.candidate_section, self.maneuver_section, self.body_section) if s.isVisibleTo(self)]
        if not visible:
            return
        total = max(self.sections.height(), 300)
        wants = [s.items.sizeHint().height() + 34 for s in visible]
        share = total / len(visible)
        sizes = []
        spare = sum(max(0.0, share - w) for w in wants)
        greedy = sum(1 for w in wants if w > share) or 1
        for w in wants:
            sizes.append(int(w if w <= share else share + spare / greedy))
        it = iter(sizes)
        self.sections.setSizes([next(it) if s in visible else 0
                                for s in (self.candidate_section, self.maneuver_section, self.body_section)])

    def candidate_card_text(self) -> str:
        """Every label's text in the candidate card (for tests)."""
        if self._candidate_card is None:
            return ""
        return "\n".join(w.text() for w in self._candidate_card.findChildren(QLabel))

    def _fill_candidate_card(self):
        card, cand = self._candidate_card, self._state.candidate()
        if card is None or cand is None:
            return
        layout = card.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # item.widget() is None after this
                widget.deleteLater()
            elif item.layout() is not None:
                _clear_layout(item.layout())
        layout.setSpacing(3)

        name_row = QHBoxLayout()
        name = QLabel(cand["label"])
        name.setObjectName("bodyName")
        status_text, color = candidate_status(cand)
        status = QLabel(status_text)
        status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
        name_row.addWidget(name)
        name_row.addStretch(1)
        name_row.addWidget(status)
        layout.addLayout(name_row)

        best = min((c["lambertTotalDvMs"] for c in self._state.candidates
                    if c.get("lambertTotalDvMs") is not None), default=None)
        dv = cand.get("lambertTotalDvMs")
        dv_text = format_dv(dv)
        if dv is not None and best is not None and dv > best:
            dv_text += f"  (+{(dv - best) / 1000.0:.3f} vs best)"
        elif dv is not None and best is not None:
            dv_text += "  (best)"
        departure = cand.get("departureMjd2000")
        tof = cand.get("tofDays")
        rank = cand.get("rank")
        exported = bool(cand.get("exported"))
        rows = [
            ("Role", "Exported trajectory" if exported else "Selected candidate (screening)"),
            ("Screening fidelity", SCREENING_FIDELITY_LABEL),
            ("Flown fidelity", FLOWN_FIDELITY_LABEL if exported else "not exported — no high-fidelity flight"),
            ("Departure", format_date(departure)),
            ("Arrival", format_date(None if departure is None or tof is None else departure + tof)),
            ("Time of flight", "—" if tof is None else f"{tof:.1f} days"),
            ("Lambert Δv", dv_text),
            ("Selection", f"{cand.get('selection', '—')}" + ("" if rank is None else f", rank {rank}")),
            ("Verification", (cand.get("verification") or "not flown").replace("_", " ")),
        ]
        for key, value in rows:
            r = QHBoxLayout()
            k = QLabel(key)
            k.setObjectName("bodyStat")
            val = QLabel(value)
            val.setObjectName("bodyStatValue")
            val.setWordWrap(True)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r.addWidget(k)
            r.addStretch(1)
            r.addWidget(val)
            layout.addLayout(r)

        notes = list(cand.get("verdictReasons") or [])
        if exported:
            notes.append("This is the mission trajectory (amber): flown in high fidelity; "
                         "the cyan arc is its two-body screening estimate.")
        else:
            flown = exported_candidate(self._state.candidates)
            notes.append("Screening arc only (two-body Lambert) -- not flown in high fidelity."
                         + (f" The timeline, maneuvers and verification are {flown['label']}'s." if flown else ""))
        for text in notes:
            note = QLabel(text)
            note.setObjectName("maneuverNote")
            note.setWordWrap(True)
            layout.addWidget(note)

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

        # Spacecraft around a planet read as an altitude (above the drawn
        # sphere); a ground site gets no distance row at all (its marker is
        # lifted for display, see its info rows); everything else reads as
        # distance from the scene's center.
        ref = altitude_reference(self._state.scene)
        kind = body.get("kind")
        dist_val, minus = None, 0.0
        if kind != "site":
            as_altitude = ref is not None and kind == "spacecraft"
            minus = 1.0 if as_altitude else 0.0  # marks an altitude card (see set_distances)
            dist_row = QHBoxLayout()
            dist_key = QLabel("Altitude" if as_altitude else "Distance")
            dist_key.setObjectName("bodyStat")
            if as_altitude:
                shape = "ellipsoid" if ref.flattening else "sphere"
                dist_key.setToolTip(f"Height above the {self._state.body(ref.body_id)['name']} {shape} as drawn "
                                    f"({ref.radius:,.0f} km equatorial radius)")
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

        self.body_section.add(frame)
        self._body_cards[body["id"]] = {"frame": frame, "dist_val": dist_val, "minus": minus}

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
        # Long names ("Trajectory-correction maneuver") next to the time would
        # otherwise set a minimum width wider than the panel, and with
        # horizontal scrolling off every card got clipped on the right.
        name.setWordWrap(True)
        # MET, not "T+0.00 days": a day-unit timeline put every ascent event
        # of a Starship flight at the same "T+0.00".
        time_label = QLabel(format_met(event.get("time", 0.0) * timeline_unit_seconds(unit)))
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
            if id(e) == active_id:
                # Its own scroll now: keep the current event in view.
                self.maneuver_section.scroll.ensureWidgetVisible(row, 0, 24)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)  # item.widget() is None after this
            widget.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())
