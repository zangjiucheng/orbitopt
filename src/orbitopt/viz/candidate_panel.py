"""The TRAJECTORIES list: the optimizer's distinct candidates for the loaded
mission, each with its numbers and what happened to it, one click away from
being highlighted in the 3D view and detailed in the inspector.

Viewer-only (no pykep/tudatpy): it reads the scene's ``candidates`` entries
that orbitopt.viz.candidates wrote.
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from orbitopt.timescales import mjd2000_to_utc_datetime
from orbitopt.viz.state import MissionState
from orbitopt.viz.theme import DANGER, INK_MUTED, SUCCESS, WARNING

# Verdict -> (short status, color). Wording follows the gate's meaning: only
# "accepted" is a verified trajectory.
_VERDICT_STATUS = {
    "accepted": ("VERIFIED", SUCCESS),
    "degraded": ("NOT VERIFIED", WARNING),
    "rejected": ("REJECTED", DANGER),
    "not_evaluated": ("NOT FLOWN", INK_MUTED),
}
_OFF_LIST = ("NOT SHORTLISTED", INK_MUTED)

# What each kind of trajectory in a candidates scene is, at what fidelity.
# Only the exported trajectory was flown (and verified) in high fidelity; every
# candidate's arc, the exported one's included, is a screening estimate.
SCREENING_FIDELITY_LABEL = "Two-body Lambert"
FLOWN_FIDELITY_LABEL = "TudatPy N-body"


def exported_candidate(candidates) -> dict | None:
    """The candidate whose high-fidelity flight is the scene's trajectory."""
    return next((c for c in candidates if c.get("exported")), None)


def candidate_status(candidate: dict) -> tuple[str, str]:
    """``(label, color)`` for a candidate's outcome at the verification gate."""
    verdict = candidate.get("verdict")
    if verdict is None:
        return _OFF_LIST
    text, color = _VERDICT_STATUS.get(verdict, (verdict.upper(), INK_MUTED))
    if verdict == "degraded" and candidate.get("verification"):
        text = f"VERIFICATION {candidate['verification'].replace('_', ' ').upper()}"
    return text, color


def format_date(mjd2000: float | None) -> str:
    if mjd2000 is None:
        return "—"
    return mjd2000_to_utc_datetime(mjd2000).strftime("%Y-%m-%d")


def format_dv(dv_m_s: float | None) -> str:
    return "—" if dv_m_s is None else f"{dv_m_s / 1000.0:.3f} km/s"


def candidate_summary(candidate: dict) -> str:
    """One line: departure date, time of flight, Lambert delta-v."""
    return (f"Depart {format_date(candidate.get('departureMjd2000'))} · "
            f"{candidate.get('tofDays', 0.0):.0f} d · Δv {format_dv(candidate.get('lambertTotalDvMs'))}")


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class CandidatePanel(QWidget):
    """Selects through ``MissionState.select_candidate``; hides itself for a
    scene without candidates. ``show_all`` (the checkbox) tells the 3D view
    whether to draw every screening arc or only the selected one."""

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self._rows: dict[str, QFrame] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header = QLabel("TRAJECTORIES")
        header.setObjectName("sectionHeader")
        header.setToolTip(
            "Distinct transfers the optimizer found, best Lambert delta-v first. Arcs are "
            "screening fidelity (two-body Lambert); only the exported one was flown in "
            "high fidelity -- that is the mission trajectory and its timeline."
        )
        self.show_all_box = QCheckBox("All")
        self.show_all_box.setChecked(True)
        self.show_all_box.setToolTip("Draw every candidate's screening arc, not just the selected one")
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(self.show_all_box)
        layout.addLayout(header_row)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(6)
        layout.addLayout(self._rows_layout)

        self.frame_hint = QLabel("Arcs are drawn in the scene's own frame")
        self.frame_hint.setObjectName("maneuverNote")
        self.frame_hint.setWordWrap(True)
        self.frame_hint.setVisible(False)
        layout.addWidget(self.frame_hint)

        state.scene_changed.connect(self._rebuild)
        state.candidate_changed.connect(lambda _cid: self._refresh_highlight())
        state.frame_changed.connect(lambda fid: self.frame_hint.setVisible(fid is not None))

    def row(self, candidate_id: str) -> QFrame | None:
        return self._rows.get(candidate_id)

    def _rebuild(self, _scene):
        for row in self._rows.values():
            row.setParent(None)  # off the panel now, not at the next event-loop turn
            row.deleteLater()
        self._rows = {}
        self.frame_hint.setVisible(False)
        candidates = self._state.candidates
        for cand in candidates:
            row = self._make_row(cand)
            self._rows_layout.addWidget(row)
            self._rows[cand["id"]] = row
        self.setVisible(bool(candidates))
        self._refresh_highlight()

    def _make_row(self, cand: dict) -> QFrame:
        # Reuses the maneuver-row look: same card, same "active" highlight.
        row = QFrame()
        row.setObjectName("maneuverRow")
        row.setProperty("active", False)
        v = QVBoxLayout(row)
        v.setContentsMargins(9, 6, 9, 6)
        v.setSpacing(2)

        # Three short lines rather than a name/status row: the sidebar is
        # ~200 px wide, too narrow for "Candidate #0 · FLOWN" and a status.
        name = QLabel(cand["label"])
        name.setObjectName("maneuverName")
        v.addWidget(name)
        status_text, color = candidate_status(cand)
        status = QLabel(status_text + ("  ·  flown" if cand.get("exported") else ""))
        status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
        v.addWidget(status)

        summary = QLabel(candidate_summary(cand))
        summary.setObjectName("maneuverNote")
        summary.setWordWrap(True)
        v.addWidget(summary)

        row.setToolTip(cand.get("selectionReason") or cand["label"])
        # No pointing-hand cursor: setting any cursor in a window holding the
        # pyvistaqt view segfaults on macOS 27 (see app.py's _begin_switch).
        row.mousePressEvent = lambda _e, cid=cand["id"]: self._state.select_candidate(cid)
        return row

    def _refresh_highlight(self):
        selected = self._state.candidate_id
        for cid, row in self._rows.items():
            row.setProperty("active", cid == selected)
            _repolish(row)
