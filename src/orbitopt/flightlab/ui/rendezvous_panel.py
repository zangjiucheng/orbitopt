"""Rendezvous verdict tab for Flight Lab (E12-06 mission studio runs).

Fed by the ``rendezvous-studio-1.0`` sidecar a studio run carries beside its
``manifest.json``: the two-impulse plan, the verification verdict graded by
``verify_plan``, and the keep-out / hold-point safety results. Pure display --
the sidecar is producer-computed metadata, and the numbers shown are the same
ones the rel_* telemetry curves end on.
"""
from __future__ import annotations

from typing import Mapping

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.ui.recovery_panel import FLIGHTLAB_PANEL_QSS
from orbitopt.viz.theme import DANGER, INK_MUTED, INK_SECONDARY, SUCCESS

VERDICT_COLORS = {"PASS": SUCCESS, "FAIL": DANGER, "N/A": INK_MUTED}


def _f(value, spec: str) -> str:
    return format(float(value), spec)


def rendezvous_rows(doc: Mapping) -> tuple[tuple[str, str, str], ...]:
    """(label, value, verdict) rows for the sidecar document's tables."""
    plan = doc.get("plan") or {}
    ver = doc.get("verification") or {}
    keepout = doc.get("keepout") or {}
    hold = doc.get("hold_point") or {}
    maneuvers = {str(m.get("kind")): m for m in plan.get("maneuvers", ())}
    dv1 = maneuvers.get("transfer_injection", {})
    dv2 = maneuvers.get("hold_point_insertion", {})
    rel = ver.get("arrival_rel_lvlh_m", (0.0, 0.0, 0.0))
    checks = ver.get("checks") or {}
    rows: list[tuple[str, str, str]] = [
        ("Verdict", "PASS" if ver.get("passed") else "FAIL",
         "PASS" if ver.get("passed") else "FAIL"),
        ("Mission", str(doc.get("mission_id", "?")) + " · " + str(plan.get("frame", "GCRS")), "N/A"),
        ("Drift gate T+", _f(plan.get("drift_end_s", 0.0), ",.1f") + " s",
         "PASS" if checks.get("gate_within_drift_window") else "FAIL"),
        ("Transfer TOF", _f(plan.get("transfer_tof_s", 0.0), ",.1f") + " s",
         "PASS" if checks.get("tof_within_transfer_window") else "FAIL"),
        ("Arrival T+", _f(plan.get("arrival_epoch_s", 0.0), ",.1f") + " s", "N/A"),
        ("Δv transfer injection", _f(dv1.get("magnitude_m_s", 0.0), ".3f") + " m/s", "N/A"),
        ("Δv hold-point insertion", _f(dv2.get("magnitude_m_s", 0.0), ".3f") + " m/s", "N/A"),
        ("Δv total / budget",
         _f(plan.get("total_delta_v_m_s", 0.0), ".3f") + " / "
         + _f(plan.get("delta_v_budget_m_s", 0.0), ".1f") + " m/s",
         "PASS" if checks.get("delta_v_within_budget") else "FAIL"),
        ("Δv margin", _f(plan.get("budget_margin_m_s", 0.0), ".3f") + " m/s", "N/A"),
        ("Position error at arrival", _f(ver.get("position_error_m", 0.0), ".3e") + " m",
         "PASS" if checks.get("position_error_within_tolerance") else "FAIL"),
        ("Relative speed at arrival", _f(ver.get("relative_speed_m_s", 0.0), ".3e") + " m/s",
         "PASS" if checks.get("relative_speed_within_tolerance") else "FAIL"),
        ("Rel. position at arrival (LVLH)",
         "(" + ", ".join(_f(v, "+.1f") for v in rel) + ") m", "N/A"),
        ("Keep-out sphere",
         "radius " + _f(keepout.get("radius_m", 0.0), ".0f") + " m · closest approach "
         + _f(keepout.get("min_range_m", 0.0), ",.0f") + " m",
         "PASS" if keepout.get("respected") else "FAIL"),
        ("Hold point",
         "tolerance " + _f(hold.get("tolerance_m", 0.0), ".0f") + " m", "N/A"),
    ]
    return tuple(rows)


class RendezvousPanel(QWidget):
    """Rendezvous plan + graded verdict + safety checks for a studio run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabRendezvousPanel")
        self.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.verdict_label = QLabel("No rendezvous run loaded.")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setObjectName("flightLabRendezvousVerdict")
        layout.addWidget(self.verdict_label)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("flightLabRendezvousTable")
        self.table.setHorizontalHeaderLabels(("Rendezvous check", "Result", "Verdict"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setObjectName("flightLabRendezvousNote")
        layout.addWidget(self.note_label)
        layout.addStretch(1)
        self.clear()

    def set_doc(self, doc: Mapping | None) -> None:
        if not doc:
            self.clear()
            return
        rows = rendezvous_rows(doc)
        passed = bool((doc.get("verification") or {}).get("passed"))
        verdict = "PASS" if passed else "FAIL"
        self.verdict_label.setText(
            f'<span style="color:{VERDICT_COLORS[verdict]}; font-weight:600">{verdict}</span>'
            f' · <span style="color:{INK_SECONDARY}">{doc.get("mission_id", "rendezvous")}</span>'
            ' · replay of the two-impulse plan with the verified arrival state')
        self.table.setRowCount(len(rows))
        for row, (label, value, state) in enumerate(rows):
            for column, text in enumerate((label, value, state)):
                item = QTableWidgetItem(text)
                color = VERDICT_COLORS.get(state if column == 2 else "N/A")
                if column == 2:
                    item.setForeground(_brush(color))
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()
        self.note_label.setText(
            f'<span style="color:{INK_MUTED}">Maneuvers and hold/keep-out results come from the '
            'run\u0027s rendezvous sidecar; the rel_* telemetry curves end on exactly these '
            'arrival numbers.</span>')

    def clear(self) -> None:
        self.verdict_label.setText("No rendezvous run loaded.")
        self.table.setRowCount(0)
        self.note_label.setText("")


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(color))
