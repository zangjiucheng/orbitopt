"""The thin strip between the title bar and the 3D view: which body the view
is centered on (with buttons to re-center it), and whether body sizes are
drawn to scale.

It sits *above* the viewport rather than floating over it because the
embedded VTK surface is a native GL context that Qt does not reliably
composite sibling widgets on top of (see the loading-page note in app.py).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QWidget

from orbitopt.viz.state import MissionState, frame_options, size_exaggeration_range


def _format_factor(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def scale_note(scene: dict | None) -> tuple[str, bool]:
    """``(text, exaggerated)`` describing body scale for ``scene``."""
    rng = size_exaggeration_range(scene)
    if rng is not None:
        lo, hi = rng
        span = f"×{_format_factor(lo)}" if lo == hi else f"×{_format_factor(lo)}–{_format_factor(hi)}"
        return f"Body sizes enlarged {span} — not to scale", True
    has_spheres = any(b.get("texture") and b.get("radius") for b in (scene or {}).get("bodies", []))
    if has_spheres:
        return "Body sizes to scale · spacecraft marker is symbolic", False
    return "", False


class ViewStrip(QWidget):
    """Re-centers the view through ``MissionState.set_frame``; rebuilds its
    buttons from ``scene_changed`` and follows ``frame_changed``."""

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._buttons: dict[str | None, QPushButton] = {}
        self.setObjectName("viewStrip")
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 5, 12, 5)
        layout.setSpacing(4)

        self.frame_label = QLabel("CENTER")
        self.frame_label.setObjectName("viewLabel")
        self.frame_label.setToolTip(
            "Which body the view is centered on. Re-centering on a planet shows the "
            "spacecraft's path as seen from that planet, at its own scale -- e.g. the "
            "Earth parking orbit and escape, or the Mars capture. Display only: data "
            "and distance readouts do not change."
        )
        layout.addWidget(self.frame_label)

        self._button_row = QHBoxLayout()
        self._button_row.setSpacing(4)
        layout.addLayout(self._button_row)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout.addStretch(1)
        self.scale_label = QLabel("")
        self.scale_label.setObjectName("scaleNote")
        layout.addWidget(self.scale_label)

        state.scene_changed.connect(self._on_scene_changed)
        state.frame_changed.connect(self._on_frame_changed)

    def frame_button(self, body_id: str | None) -> QPushButton | None:
        return self._buttons.get(body_id)

    def _on_scene_changed(self, scene):
        for button in self._buttons.values():
            self._group.removeButton(button)
            button.deleteLater()
        self._buttons = {}
        options = frame_options(scene)
        for body_id, name in options:
            button = QPushButton(name)
            button.setObjectName("viewButton")
            button.setCheckable(True)
            button.setToolTip(
                f"Center the view on {name} (the scene's own frame)" if body_id is None
                else f"Center the view on {name} and frame its neighborhood"
            )
            button.clicked.connect(lambda _checked=False, bid=body_id: self._state.set_frame(bid))
            self._group.addButton(button)
            self._button_row.addWidget(button)
            self._buttons[body_id] = button
        # One option is not a choice: show it as a plain label instead.
        multiple = len(options) > 1
        for button in self._buttons.values():
            button.setVisible(multiple)
        self.frame_label.setText("CENTER" if multiple else (f"CENTER  {options[0][1]}" if options else ""))
        self._on_frame_changed(self._state.frame_body_id)

        text, exaggerated = scale_note(scene)
        self.scale_label.setText(text)
        self.scale_label.setProperty("warn", exaggerated)
        self.scale_label.style().unpolish(self.scale_label)
        self.scale_label.style().polish(self.scale_label)
        self.setVisible(bool(options))

    def _on_frame_changed(self, body_id):
        button = self._buttons.get(body_id)
        if button is not None:
            button.setChecked(True)
