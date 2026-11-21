"""Persistent body focus controls above the flight plots."""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget

from orbitopt.viz.theme import BG_VOID, BORDER_SOFT, HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE_HOVER


class BodyStrip(QWidget):
    focus_requested = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setObjectName("bodyStrip")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QWidget#bodyStrip {{ background: {BG_VOID}; border-top: 1px solid {BORDER_SOFT}; }}
            QScrollArea#bodyStripScroll, QWidget#bodyStripViewport,
            QWidget#bodyStripContents {{ background: {BG_VOID}; border: none; }}
            QLabel#bodyStripLabel {{ color: {INK_MUTED}; font-size: 11px; border: none; }}
            QPushButton#bodyFocus {{ background: transparent; color: {INK_SECONDARY};
                border: none; border-bottom: 2px solid transparent; border-radius: 0;
                padding: 0 10px; font-size: 12px; font-weight: 400; }}
            QPushButton#bodyFocus:hover {{ background: {SURFACE_HOVER}; color: {INK_PRIMARY}; }}
            QPushButton#bodyFocus:checked {{ color: {INK_PRIMARY}; border-bottom-color: {INK_PRIMARY};
                font-weight: 600; }}
            QPushButton#bodyFocus:disabled {{ color: {INK_MUTED}; background: transparent; }}
            QPushButton#bodyFocus:focus {{ border-bottom-color: {INK_PRIMARY}; }}
            QScrollBar:horizontal {{ background: {BG_VOID}; height: 4px; border: none; margin: 0; }}
            QScrollBar::handle:horizontal {{ background: {HAIRLINE}; min-width: 24px; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: {BG_VOID}; }}
        """)
        self.state, self.buttons = state, {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 8, 3)
        layout.setSpacing(10)
        label = QLabel("Bodies")
        label.setObjectName("bodyStripLabel")
        layout.addWidget(label)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("bodyStripScroll")
        self.scroll.viewport().setObjectName("bodyStripViewport")
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.content = QWidget()
        self.content.setObjectName("bodyStripContents")
        self.content.setAttribute(Qt.WA_StyledBackground, True)
        self.row = QHBoxLayout(self.content)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(2)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)
        self.setFixedHeight(40)
        state.scene_changed.connect(self.set_scene)
        state.time_changed.connect(self.refresh)
        state.selection_changed.connect(self.refresh)

    def set_scene(self, scene):
        while self.row.count():
            item = self.row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.buttons = {}
        for body in (scene or {}).get("bodies", []):
            name = body.get("name", body["id"])
            if body["id"].startswith("stage") and body["id"][5:].isdigit():
                name = "Stage " + body["id"][5:]
            elif body.get("kind") == "site":
                name = name.split(" · ", 1)[0]
            button = QPushButton(name)
            button.setObjectName("bodyFocus")
            button.setFixedHeight(28)
            dot = QPixmap(12, 12)
            dot.fill(Qt.transparent)
            painter = QPainter(dot)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(body.get("color", INK_SECONDARY)))
            painter.drawEllipse(3, 3, 6, 6)
            painter.end()
            button.setIcon(QIcon(dot))
            button.setIconSize(QSize(12, 12))
            button.setCheckable(True)
            button.clicked.connect(lambda _on=False, key=body["id"]: self.focus_requested.emit(key))
            button.setToolTip("Focus and follow " + body.get("name", body["id"]))
            self.row.addWidget(button)
            self.buttons[body["id"]] = button
        self.row.addStretch(1)
        self.refresh()

    def refresh(self, *_):
        for key, button in self.buttons.items():
            body = self.state.body(key)
            visible = (body.get("tStart", float("-inf")) <= self.state.time
                       <= body.get("tEnd", float("inf")))
            button.setEnabled(visible)
            button.setChecked(key == self.state.selected_body_id)
