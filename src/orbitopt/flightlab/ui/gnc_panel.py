"""GNC replay tab for Flight Lab (E13-08 mission studio runs).

Fed by the ``gnc-studio-1.0`` sidecar a GNC studio run carries beside its
``manifest.json``: the sealed-run replay evidence, the E13-07 independent
verification verdict, and the navigation / control / measurement summaries.
The two evidence lines are deliberately kept apart -- the replay curve is the
record grading itself, the verdict line only counts what the independent
verifier actually ran (a not-checked line is shown as such, never as a pass).
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

RESULT_COLORS = {"passed": SUCCESS, "failed": DANGER, "not_checked": INK_MUTED}
RESULT_WORDS = {"passed": "PASS", "failed": "FAIL", "not_checked": "NOT CHECKED"}

ERROR_CLASS_LABELS = {
    "numerical_error": "Numerical error",
    "model_error": "Model error (independent re-flight)",
    "navigation_inconsistency": "Navigation inconsistency",
    "optimizer_control_nonconvergence": "Optimizer control non-convergence",
}


def _f(value, spec: str) -> str:
    return format(float(value), spec)


def gnc_rows(doc: Mapping) -> tuple[tuple[str, str, str], ...]:
    """(label, value, result) rows for a gnc-studio-1.0 sidecar document."""
    local = doc.get("local_replay") or {}
    indep = doc.get("independent_verification") or {}
    nav = doc.get("nav_summary") or {}
    ctl = doc.get("control_summary") or {}
    meas = doc.get("measurement") or {}
    supervision = doc.get("supervision") or {}
    tracking = doc.get("tracking") or {}
    rows: list[tuple[str, str, str]] = [
        ("Run log (local replay)", str(local.get("run_id", "?"))
         + " · digest " + str(local.get("run_digest", ""))[:16] + "…", "N/A"),
    ]
    classes = indep.get("error_classes") or {}
    if indep:
        rows.append(("Verification report (independent)",
                     str(indep.get("report_kind", "?")) + " · digest "
                     + str(indep.get("report_digest", ""))[:16] + "…", "N/A"))
    else:
        rows.append(("Verification report (independent)", "none consumed", "not_checked"))
    for cls, label in ERROR_CLASS_LABELS.items():
        line = classes.get(cls)
        if line is None:
            continue
        result = str(line.get("result", "not_checked"))
        value = RESULT_WORDS.get(result, result)
        if result == "not_checked" and line.get("reason"):
            value += f" ({line['reason']})"
        rows.append((label, value, result))
    est = nav.get("estimates") or {}
    if est:
        rows += [
            ("Estimates scored / gate violations",
             f"{est.get('scored', 0)} / {est.get('gate_violations', 0)}",
             "failed" if est.get("gate_violations") else "passed"),
            ("NEES max / mean per DoF",
             _f(est.get("max_nis", 0.0), ".1f") + " / "
             + _f(est.get("mean_nis_per_dof", 0.0), ".2f"), "N/A"),
        ]
    obs = nav.get("observations") or {}
    if obs:
        rows.append(("Observations scored / gate violations",
                     f"{obs.get('scored', 0)} / {obs.get('gate_violations', 0)}",
                     "failed" if obs.get("gate_violations") else "passed"))
    if ctl:
        rows += [
            ("Burn events executed", _f(ctl.get("events", 0), ".0f"), "N/A"),
            ("Execution error max",
             _f(ctl.get("execution_error_max_m_s", 0.0), ".3e") + " m/s", "N/A"),
        ]
    if tracking:
        rows.append(("Estimate-vs-truth RMSE (pos)",
                     _f(tracking.get("rmse_pos_m", 0.0), ".3e") + " m", "N/A"))
    drops = sum(int(s.get("dropped", 0)) for s in (meas.get("sensors") or {}).values())
    rows.append(("Measurement frames dropped", _f(drops, ".0f"),
                 "N/A" if not drops else "failed"))
    if supervision:
        rows.append(("Supervisor final state", str(supervision.get("final_state", "?")),
                     "N/A"))
        rows.append(("Supervisor transitions", _f(len(supervision.get("events", ())), ".0f"),
                     "N/A"))
    return tuple(rows)


class GncPanel(QWidget):
    """Evidence-line + summary table for a GNC studio replay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabGncPanel")
        self.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.verdict_label = QLabel("No GNC run loaded.")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setObjectName("flightLabGncVerdict")
        layout.addWidget(self.verdict_label)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("flightLabGncTable")
        self.table.setHorizontalHeaderLabels(("GNC replay check", "Result", "Verdict"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setObjectName("flightLabGncNote")
        layout.addWidget(self.note_label)
        layout.addStretch(1)
        self.clear()

    def set_doc(self, doc: Mapping | None) -> None:
        if not doc:
            self.clear()
            return
        rows = gnc_rows(doc)
        indep = doc.get("independent_verification") or {}
        classes = indep.get("error_classes") or {}
        failed = [cls for cls, line in classes.items()
                  if str(line.get("result")) == "failed"]
        unchecked = [cls for cls, line in classes.items()
                     if str(line.get("result")) not in ("passed", "failed")]
        if failed:
            head, color = "FAIL", DANGER
        elif indep:
            head, color = "PASS (checked lines)", SUCCESS
        else:
            head, color = "REPLAY ONLY — no independent report", INK_SECONDARY
        self.verdict_label.setText(
            f'<span style="color:{color}; font-weight:600">{head}</span>'
            f' · <span style="color:{INK_SECONDARY}">{doc.get("title", "GNC replay")}</span>')
        self.table.setRowCount(len(rows))
        for row, (label, value, result) in enumerate(rows):
            for column, text in enumerate((label, value, result.upper()
                                           if result in RESULT_WORDS else result)):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setForeground(_brush(RESULT_COLORS.get(result, INK_MUTED)))
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()
        extra = (f" · {len(unchecked)} error class(es) did not run: shown with their "
                 "reason, not as a pass" if unchecked else "")
        self.note_label.setText(
            f'<span style="color:{INK_MUTED}">Curves and ellipsoids on the scene replay the '
            'sealed run log (local, self-consistent — not independent proof). The verdict line '
            f'is the consumed E13-07 report{extra}.</span>')

    def clear(self) -> None:
        self.verdict_label.setText("No GNC run loaded.")
        self.table.setRowCount(0)
        self.note_label.setText("")


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(color))
