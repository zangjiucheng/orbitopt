"""Reason explanations + recovery panel widgets (FL-503).

Thin Qt views over :mod:`orbitopt.flightlab.reasons` and
:func:`orbitopt.flightlab.compare.recovery_summary`, shared by the replay page
and the Compare Runs view:

* :class:`ReasonPanel` -- the plain-language "why it failed" list (EN + 中文,
  with the suggested fix) for a set of reason codes;
* :class:`RecoveryPanel` -- recovery verdict, touchdown checks, and the Monte
  Carlo summary (success rate, failure histogram, 1-sigma dispersion ellipse
  over the touchdown scatter) when ``montecarlo.json`` exists.
"""
from __future__ import annotations

import html
from typing import Iterable, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab.compare import RecoverySummary, ellipse_polygon
from orbitopt.flightlab.contract import Check
from orbitopt.flightlab.reasons import ReasonEntry, explain
from orbitopt.flightlab.ui.charts import fmt_number, nice_ticks, qfont
from orbitopt.viz.theme import (
    ACCENT,
    DANGER,
    HAIRLINE,
    INFO,
    INK_MUTED,
    INK_SECONDARY,
    PANEL,
    SUCCESS,
    WARNING,
)

__all__ = ["FLIGHTLAB_PANEL_QSS", "DispersionPlot", "ReasonPanel", "RecoveryPanel",
           "format_check_value", "reason_label"]

#: Dark styling for the generic item views/inputs these Flight Lab panels
#: use (the Mission Control stylesheet only themes its own named widgets).
FLIGHTLAB_PANEL_QSS = f"""
QTextBrowser, QTableWidget, QComboBox, QComboBox QAbstractItemView {{
    background: {PANEL}; color: {INK_SECONDARY}; border: 1px solid {HAIRLINE};
    selection-background-color: #1b1729; selection-color: {ACCENT};
}}
QHeaderView::section {{
    background: #13101e; color: {INK_MUTED}; border: none;
    border-bottom: 1px solid {HAIRLINE}; padding: 3px 6px;
}}
QTableWidget {{ gridline-color: {HAIRLINE}; }}
QTabWidget::pane {{ border: 1px solid {HAIRLINE}; background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL}; color: {INK_MUTED}; padding: 5px 12px; border: none; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QCheckBox {{ color: {INK_SECONDARY}; }}
"""


def format_check_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    text = fmt_number(float(value))
    return f"{text} {unit}" if unit else text


def reason_label(check: Check) -> str:
    """Table text for a check's reason: plain title + raw code, or pass."""
    if not check.reason_code:
        return "pass" if check.passed else "no reason code"
    entry = explain(check.reason_code)
    return f"{entry.title_en} ({check.reason_code})" if entry.known else check.reason_code


def reason_tooltip(check: Check) -> str:
    if not check.reason_code:
        return ""
    entry = explain(check.reason_code)
    return f"{check.reason_code}\n\n{entry.bilingual()}"


def _severity_color(entry: ReasonEntry) -> str:
    return {"fail": DANGER, "missing": WARNING, "info": SUCCESS}.get(entry.severity, INK_SECONDARY)


def reasons_html(entries: Sequence[ReasonEntry], *, empty: str = "All checks passed — nothing to explain.",
                 counts: Sequence[int] | None = None) -> str:
    if not entries:
        return f'<p style="color:{SUCCESS}">{html.escape(empty)}</p>'
    parts = []
    for i, e in enumerate(entries):
        color = _severity_color(e)
        count = f" ×{counts[i]}" if counts is not None else ""
        parts.append(
            f'<div style="margin-bottom:8px">'
            f'<b style="color:{color}">{html.escape(e.title_en)}{count}</b>'
            f' <span style="color:{INK_MUTED}">· {html.escape(e.title_zh)} · <code>{html.escape(e.code)}</code></span><br>'
            f'<span style="color:{INK_SECONDARY}">{html.escape(e.explanation_en)}</span><br>'
            f'<span style="color:{INK_MUTED}">{html.escape(e.explanation_zh)}</span><br>'
            f'<span style="color:{ACCENT}">Fix:</span> {html.escape(e.fix_en)}<br>'
            f'<span style="color:{ACCENT}">建议：</span>{html.escape(e.fix_zh)}'
            f'</div>')
    return "".join(parts)


class ReasonPanel(QTextBrowser):
    """Read-only list of plain-language failure explanations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabReasonPanel")
        self.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        self.setOpenLinks(False)
        self._entries: tuple[ReasonEntry, ...] = ()
        self.set_entries(())

    @property
    def entries(self) -> tuple[ReasonEntry, ...]:
        return self._entries

    def set_entries(self, entries: Iterable[ReasonEntry], *, title: str = "",
                    empty: str = "All checks passed — nothing to explain.") -> None:
        self._entries = tuple(entries)
        head = f'<p style="color:{INK_SECONDARY}"><b>{html.escape(title)}</b></p>' if title else ""
        self.setHtml(head + reasons_html(self._entries, empty=empty))

    def clear_entries(self, message: str = "No run loaded.") -> None:
        self._entries = ()
        self.setHtml(f'<p style="color:{INK_MUTED}">{html.escape(message)}</p>')


class DispersionPlot(QWidget):
    """Touchdown scatter (site ENU, metres) with the 1- and 3-sigma ellipse
    FL-304 reports. Landed samples are green, failed ones red; the pad is
    the origin cross."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 160)
        self._points: tuple[tuple[float, float, bool], ...] = ()
        self._ellipse: dict = {}
        self._message = "no Monte Carlo"

    def set_data(self, points: Sequence[tuple[float, float, bool]], ellipse: dict | None,
                 message: str = "") -> None:
        self._points = tuple(points)
        self._ellipse = dict(ellipse or {})
        self._message = message
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        rect = QRectF(46, 22, max(1.0, self.width() - 56), max(1.0, self.height() - 44))
        p.setPen(QColor(INK_SECONDARY))
        p.setFont(qfont(9.5, True))
        p.drawText(QRectF(rect.left(), 2, rect.width(), 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "Touchdown dispersion (east/north, m)")
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.drawRect(rect)
        ell3 = ellipse_polygon(self._ellipse, k=3.0)
        xs = [pt[0] for pt in self._points] + [pt[0] for pt in ell3] + [0.0]
        ys = [pt[1] for pt in self._points] + [pt[1] for pt in ell3] + [0.0]
        if not self._points and not ell3:
            p.setPen(QColor(INK_MUTED))
            p.drawText(rect, Qt.AlignCenter, self._message or "no touchdowns")
            p.end()
            return
        span = max(max(xs) - min(xs), max(ys) - min(ys), 10.0) * 1.15
        cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        scale = min(rect.width(), rect.height()) / span

        def to_px(e: float, n: float) -> QPointF:
            return QPointF(rect.center().x() + (e - cx) * scale, rect.center().y() - (n - cy) * scale)

        p.setFont(qfont(8))
        for tick in nice_ticks(cx - span / 2, cx + span / 2, 3):
            x = to_px(tick, cy).x()
            if rect.left() <= x <= rect.right():
                p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.setPen(QColor(INK_MUTED))
                p.drawText(QRectF(x - 30, rect.bottom() + 2, 60, 14), Qt.AlignCenter, fmt_number(tick))
        for tick in nice_ticks(cy - span / 2, cy + span / 2, 3):
            y = to_px(cx, tick).y()
            if rect.top() <= y <= rect.bottom():
                p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                p.setPen(QColor(INK_MUTED))
                p.drawText(QRectF(0, y - 7, 42, 14), Qt.AlignRight | Qt.AlignVCenter, fmt_number(tick))
        pad = to_px(0.0, 0.0)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawLine(pad + QPointF(-6, 0), pad + QPointF(6, 0))
        p.drawLine(pad + QPointF(0, -6), pad + QPointF(0, 6))
        for k, style in ((1.0, Qt.SolidLine), (3.0, Qt.DashLine)):
            poly = ellipse_polygon(self._ellipse, k=k)
            if poly:
                p.setPen(QPen(QColor(INFO), 1.4, style))
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(QPolygonF([to_px(e, n) for e, n in poly]))
        for e, n, ok in self._points:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(SUCCESS if ok else DANGER)))
            p.drawEllipse(to_px(e, n), 3.0, 3.0)
        p.end()


class RecoveryPanel(QWidget):
    """Recovery verdict + touchdown checks + Monte Carlo summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabRecoveryPanel")
        self.setStyleSheet(FLIGHTLAB_PANEL_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.verdict_label = QLabel("No recovery data.")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setObjectName("flightLabRecoveryVerdict")
        self.check_table = QTableWidget(0, 4)
        self.check_table.setObjectName("flightLabTouchdownTable")
        self.check_table.setHorizontalHeaderLabels(("Touchdown check", "Result", "Value", "Limit"))
        self.check_table.verticalHeader().setVisible(False)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.check_table.horizontalHeader().setStretchLastSection(True)
        self.mc_label = QLabel("")
        self.mc_label.setWordWrap(True)
        self.mc_label.setObjectName("flightLabMonteCarloSummary")
        self.dispersion = DispersionPlot()
        self.reasons = ReasonPanel()
        layout.addWidget(self.verdict_label)
        layout.addWidget(self.check_table, stretch=2)
        layout.addWidget(self.mc_label)
        layout.addWidget(self.dispersion, stretch=3)
        layout.addWidget(self.reasons, stretch=2)
        self.summary: RecoverySummary | None = None
        self.clear()

    def clear(self, message: str = "No recovery data.") -> None:
        self.summary = None
        self.verdict_label.setText(message)
        self.verdict_label.setStyleSheet(f"color: {INK_MUTED};")
        self.check_table.setRowCount(0)
        self.check_table.setVisible(False)
        self.mc_label.setText("")
        self.mc_label.setVisible(False)
        self.dispersion.set_data((), None, "no Monte Carlo")
        self.dispersion.setVisible(False)
        self.reasons.clear_entries("")
        self.reasons.setVisible(False)

    def set_summary(self, summary: RecoverySummary) -> None:
        self.clear()
        self.summary = summary
        if summary.state == "none":
            self.verdict_label.setText("Ascent-only run: the booster is expended (no recovery planned).")
            return
        gate = summary.gate
        if gate is None and not summary.checks:
            self.verdict_label.setText(
                f"Recovery requested ({summary.kind}) but not assessed: the run has no touchdown checks.")
            self.verdict_label.setStyleSheet(f"color: {WARNING};")
        elif gate is not None and gate.passed:
            self.verdict_label.setText("Booster LANDED inside every touchdown limit · 助推级成功着陆")
            self.verdict_label.setStyleSheet(f"color: {SUCCESS}; font-weight: 600;")
        else:
            why = explain(gate.reason_code).title_en if gate is not None and gate.reason_code else "see checks"
            self.verdict_label.setText(f"Booster recovery FAILED — {why} · 助推级回收失败")
            self.verdict_label.setStyleSheet(f"color: {DANGER}; font-weight: 600;")
        if summary.touchdown:
            td = summary.touchdown
            extra = ", ".join(f"{k}={fmt_number(float(v))}" for k, v in td.items()
                              if k not in ("t_s", "body_id") and isinstance(v, (int, float)))
            self.verdict_label.setText(self.verdict_label.text()
                                       + f"\nTouchdown T+{fmt_number(float(td['t_s']))} s"
                                       + (f" ({extra})" if extra else ""))
        self.check_table.setRowCount(len(summary.checks))
        self.check_table.setVisible(bool(summary.checks))
        for row, check in enumerate(summary.checks):
            texts = (check.id, "PASS" if check.passed else "FAIL",
                     format_check_value(check.value, check.unit),
                     format_check_value(check.limit, check.unit))
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if not check.passed:
                    item.setForeground(QBrush(QColor(DANGER)))
                    item.setToolTip(reason_tooltip(check))
                self.check_table.setItem(row, col, item)
        self.check_table.resizeRowsToContents()
        if summary.explanations:
            self.reasons.set_entries(summary.explanations, title="Why the recovery failed · 回收失败原因")
            self.reasons.setVisible(True)
        mc = summary.montecarlo
        if mc:
            rate = summary.success_rate
            n = int(mc.get("n", 0) or 0)
            ell = summary.ellipse
            if ell.get("state") == "assessed":
                ell_text = (f"1σ ellipse {fmt_number(float(ell['sigma_along_m']))} × "
                            f"{fmt_number(float(ell['sigma_cross_m']))} m, bearing "
                            f"{fmt_number(float(ell.get('bearing_deg', 0.0)))}°, centre "
                            f"({fmt_number(float(ell.get('mean_east_m', 0.0)))} E, "
                            f"{fmt_number(float(ell.get('mean_north_m', 0.0)))} N) m")
            else:
                ell_text = f"dispersion ellipse not assessed ({ell.get('state', 'no data')})"
            rate_text = "—" if rate is None else f"{rate * 100:.0f}%"
            hist = ", ".join(f"{entry.title_en if entry.known else code} ×{count}"
                             for code, count, entry in summary.mc_reasons)
            self.mc_label.setText(
                f"<b>Monte Carlo</b> (n={n}, seed={mc.get('seed', '—')}): success rate "
                f"<b>{rate_text}</b> · 成功率<br>{html.escape(ell_text)}<br>"
                f"<span style='color:{INK_MUTED}'>{html.escape(hist)}</span>")
            color = SUCCESS if rate is not None and rate >= 0.9 else WARNING if rate and rate > 0.5 else DANGER
            self.mc_label.setStyleSheet(f"color: {color if rate is not None else INK_SECONDARY};")
            self.mc_label.setVisible(True)
            self.dispersion.set_data(summary.touchdown_points, dict(ell),
                                     "no touchdowns" if not summary.touchdown_points else "")
            self.dispersion.setVisible(True)
            failures = [(entry, count) for code, count, entry in summary.mc_reasons
                        if entry.severity != "info"]
            if failures and not summary.explanations:
                self.reasons.setHtml(
                    f'<p style="color:{INK_SECONDARY}"><b>Monte Carlo failure modes · 失败模式</b></p>'
                    + _counts_html(failures))
                self.reasons.setVisible(True)


def _counts_html(items: Sequence[tuple[ReasonEntry, int]]) -> str:
    return reasons_html([e for e, _ in items], counts=[c for _, c in items])
