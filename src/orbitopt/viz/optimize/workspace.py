"""The Optimize workspace page: renders a :class:`StudioSession` and runs its jobs off the UI thread.

Left, the problem in words (an example or your own text, the compile
outcome, the compiler's questions). Centre, the compiled problem as math and
the structured IR editor with the validator's errors and the digest.
Right, the run: budget meters, the convergence curve, the iteration table,
the four verification statuses one by one, policy violations, and replay.

Jobs run on a QThread with the same worker pattern as ``viz.compute``
(``moveToThread`` and signals connected to bound methods, so results are
queued back to the UI thread). Cancel sets the runtime's cooperative cancel
flag; the run stops at its next iteration boundary and is saved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from orbitopt.viz.charts import _CachedChart, nice_ticks
from orbitopt.viz.optimize import backend
from orbitopt.viz.optimize.editing import FIELDS, SECTIONS
from orbitopt.viz.optimize.session import Action, Phase, StudioSession
from orbitopt.viz.optimize.views import (OUTCOME_BADGE, RUN_STATUS_BADGE, RUN_STATUS_TEXT, STATUS_LABELS, RunView,
                                         problem_html, report_document, report_html)
from orbitopt.viz.theme import (ACCENT, BG_VOID, DANGER, HAIRLINE, INFO, INK_MUTED, INK_PRIMARY, INK_SECONDARY, PANEL,
                                PANEL_RAISED, SUCCESS, WARNING)
from orbitopt.viz.ui_kit import MetricRow, StatusBadge

__all__ = ["ConvergenceChart", "OptimizationWorkspace"]

PHASE_BADGE = {Phase.IDLE: "idle", Phase.COMPILING: "loading", Phase.CLARIFYING: "warning", Phase.DRAFT: "warning",
               Phase.EDITING: "idle", Phase.CONFIRMED: "ready", Phase.RUNNING: "loading", Phase.FINISHED: "ready",
               Phase.CANCELLED: "warning", Phase.FAILED: "failed", Phase.REPLAYING: "loading", Phase.REPLAY: "ready"}
KINDS = ("continuous", "integer", "binary", "discrete")
SENSES = {"objectives": ("minimize", "maximize"), "constraints": ("<=", ">=", "==")}
_QSS = f"""
#optimizeWorkspace, #optimizeColumn {{ background: {BG_VOID}; }}
#optimizeWorkspace QLabel {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QTextBrowser, QTableWidget, QComboBox, QComboBox QAbstractItemView, QPlainTextEdit, QLineEdit {{
    background: {PANEL}; color: {INK_SECONDARY}; border: 1px solid {HAIRLINE};
    selection-background-color: {PANEL_RAISED}; selection-color: {ACCENT};
}}
QPlainTextEdit, QLineEdit {{ color: {INK_PRIMARY}; padding: 4px; }}
QHeaderView::section {{ background: {PANEL_RAISED}; color: {INK_MUTED}; border: none;
    border-bottom: 1px solid {HAIRLINE}; padding: 3px 6px; }}
QTableWidget {{ gridline-color: {HAIRLINE}; }}
QTabWidget::pane {{ border: 1px solid {HAIRLINE}; background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL}; color: {INK_MUTED}; padding: 5px 12px; border: none; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton:disabled {{ color: #4a4a4a; border-color: #1f1f1f; }}
QPushButton#primaryAction {{ color: {INK_PRIMARY}; border-color: {INK_SECONDARY}; }}
QPushButton#primaryAction:disabled {{ color: #4a4a4a; border-color: #1f1f1f; }}
QFrame#mathCard {{ background: {PANEL}; border: 1px solid {HAIRLINE}; }}
QLabel#optimizeBanner {{ padding: 6px 12px; border-bottom: 1px solid {HAIRLINE}; background: {PANEL}; }}
QLabel#optimizeBanner[offline="true"] {{ color: {WARNING}; }}
QLabel#optimizeBanner[offline="false"] {{ color: {INK_SECONDARY}; }}
"""


def _header(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionHeader")
    return label


def _muted(text: str = "", *, wrap: bool = True, color: str = INK_SECONDARY) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setTextFormat(Qt.RichText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setStyleSheet(f"color: {color};")
    return label


def _button(text: str, tip: str, *, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tip)
    if primary:
        button.setObjectName("primaryAction")
    return button


def _esc(value: Any) -> str:
    import html

    return html.escape("—" if value is None else str(value))


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


# ------------------------------------------------------------------------------------------- the background
class _JobWorker(QObject):
    """Runs one session job on a QThread; results travel back through queued signals."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self) -> None:
        try:
            result = self.job(self.progress.emit)
        except Exception as exc:  # noqa: BLE001 -- shown to the user as a failed job
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


# ---------------------------------------------------------------------------------------------- the chart
class ConvergenceChart(_CachedChart):
    """Best-so-far objective against evaluations, every candidate marked by how the guard graded it.

    Feasible points are filled, infeasible ones a red cross, points the guard
    could not grade hollow grey, alternate-solver points hollow diamonds. The
    best point is ringed green only when the run is verified; otherwise its
    ring is amber and labelled ``unverified``.
    """

    MARGINS = (62, 30, 18, 34)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.view: RunView | None = None

    def set_view(self, view: RunView | None) -> None:
        self.view = view
        self.invalidate()

    def _ranges(self):
        v = self.view
        xs = [p.evaluations for p in v.points] + [x for x, _ in v.best_path]
        ys = [p.objective for p in v.points] + [y for _, y in v.best_path]
        x_hi = max(xs) if xs else 1.0
        lo, hi = (min(ys), max(ys)) if ys else (0.0, 1.0)
        if hi <= lo:
            pad = max(abs(lo) * 0.1, 1e-3)
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.08
            lo, hi = lo - pad, hi + pad
        return 0.0, max(float(x_hi), 1.0) * 1.04, lo, hi

    def paint_static(self, p: QPainter):
        from orbitopt.viz.charts import _font

        rect = self.plot_rect()
        p.setFont(_font(9.5, True))
        p.setPen(QColor(INK_SECONDARY))
        name = self.view.objective_name if self.view else ""
        title = " ".join(f"Objective {name} vs evaluations".split()) + (f" ({self.view.sense})" if self.view else "")
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignLeft | Qt.AlignVCenter, title)
        self._legend(p)
        p.setPen(QPen(QColor(HAIRLINE), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        if self.view is None or not self.view.points:
            p.setPen(QColor(INK_MUTED))
            p.setFont(_font(10))
            p.drawText(rect, Qt.AlignCenter, "No candidate yet" if self.view else "Run a confirmed problem")
            return
        x0, x1, y0, y1 = self._ranges()

        def px(x):
            return rect.left() + rect.width() * (x - x0) / (x1 - x0)

        def py(y):
            return rect.bottom() - rect.height() * (y - y0) / (y1 - y0)

        p.setFont(_font(8.5))
        ticks = nice_ticks(y0, y1, 4)
        digits = _tick_digits(ticks)
        for yv in ticks:
            p.setPen(QPen(QColor(HAIRLINE), 1, Qt.DotLine))
            p.drawLine(QPointF(rect.left(), py(yv)), QPointF(rect.right(), py(yv)))
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(0, py(yv) - 8, self.MARGINS[0] - 6, 16), Qt.AlignRight | Qt.AlignVCenter,
                       f"{yv:.{digits}g}")
        for xv in [t for t in nice_ticks(x0, x1, 5) if float(t).is_integer()]:     # evaluations are counts
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(px(xv) - 30, rect.bottom() + 4, 60, 14), Qt.AlignCenter, f"{xv:,.0f}")
        p.drawText(QRectF(rect.left(), rect.bottom() + 17, rect.width(), 14), Qt.AlignCenter, "evaluations")
        if self.view.best_path:
            path = QPainterPath()
            (fx, fy), *rest = self.view.best_path
            path.moveTo(px(fx), py(fy))
            last_y = fy
            for x, y in rest:
                path.lineTo(px(x), py(last_y))
                path.lineTo(px(x), py(y))
                last_y = y
            p.setPen(QPen(QColor(INK_SECONDARY), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        for point in self.view.points:
            self._mark(p, QPointF(px(point.evaluations), py(point.objective)), point.kind)
        best = next((pt for pt in self.view.points if pt.label == self.view.best_label), None)
        if best is not None:
            verified = self.view.verified
            color = QColor(SUCCESS if verified else WARNING)
            centre = QPointF(px(best.evaluations), py(best.objective))
            p.setPen(QPen(color, 2.0, Qt.SolidLine if verified else Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(centre, 9, 9)
            p.setPen(color)
            p.setFont(_font(8.5, True))
            text = "verified best" if verified else "best, unverified"
            width = p.fontMetrics().horizontalAdvance(text) + 4
            x = centre.x() + 13 if centre.x() + 13 + width < rect.right() else centre.x() - 13 - width
            y = centre.y() - 24 if centre.y() - 24 > rect.top() + 2 else centre.y() + 10      # off the best-so-far line
            p.drawText(QRectF(x, y, width, 14), Qt.AlignLeft | Qt.AlignVCenter, text)

    @staticmethod
    def _mark(p: QPainter, c: QPointF, kind: str) -> None:
        r = 4.0
        if kind == "feasible":
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(INFO)))
            p.drawEllipse(c, r, r)
        elif kind == "infeasible":
            p.setPen(QPen(QColor(DANGER), 2))
            p.drawLine(QPointF(c.x() - r, c.y() - r), QPointF(c.x() + r, c.y() + r))
            p.drawLine(QPointF(c.x() - r, c.y() + r), QPointF(c.x() + r, c.y() - r))
        elif kind == "alternate":
            p.setPen(QPen(QColor(INK_PRIMARY), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon([QPointF(c.x(), c.y() - r - 1), QPointF(c.x() + r + 1, c.y()),
                           QPointF(c.x(), c.y() + r + 1), QPointF(c.x() - r - 1, c.y())])
        else:
            p.setPen(QPen(QColor(INK_MUTED), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, r, r)

    def _legend(self, p: QPainter) -> None:
        from orbitopt.viz.charts import _font

        p.setFont(_font(8.5))
        x = float(self.width() - 18)
        for kind, text in reversed((("feasible", "feasible"), ("infeasible", "infeasible"),
                                    ("not_assessed", "not graded"), ("alternate", "alternate solver"))):
            width = p.fontMetrics().horizontalAdvance(text)
            x -= width
            p.setPen(QColor(INK_MUTED))
            p.drawText(QRectF(x, 4, width + 2, 18), Qt.AlignLeft | Qt.AlignVCenter, text)
            self._mark(p, QPointF(x - 9, 13), kind)
            x -= 24


# ------------------------------------------------------------------------------------------- the workspace
class OptimizationWorkspace(QWidget):
    """The page. ``session`` defaults to a fresh one; ``settings`` keeps the manifest digests out of band."""

    def __init__(self, session: StudioSession | None = None, *, settings=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("optimizeWorkspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(_QSS)
        self.missing = backend.missing_extras()
        self.session: StudioSession | None = None
        self._thread: QThread | None = None
        self._worker: _JobWorker | None = None
        self._job = None
        self._shown_doc = None
        self._settings = settings
        self._filling = True        # widget set-up and rendering never feed back into the session
        self._shown_questions = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if self.missing:
            self.hint_label = _muted(f"<b>Optimization needs the agent extras</b><br><br>"
                                     f"Missing: {', '.join(self.missing)}.<br><br>"
                                     + _esc(backend.INSTALL_HINT).replace("\n", "<br>"), color=INK_SECONDARY)
            self.hint_label.setAlignment(Qt.AlignCenter)
            outer.addWidget(self.hint_label, 1)
            return
        self.session = session or StudioSession()
        self.banner = QLabel()
        self.banner.setObjectName("optimizeBanner")
        self.banner.setWordWrap(True)
        outer.addWidget(self.banner)
        outer.addWidget(self._build_header())
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._scroll(self._build_problem_column()))
        splitter.addWidget(self._scroll(self._build_ir_column()))
        splitter.addWidget(self._scroll(self._build_run_column()))
        splitter.setSizes([330, 560, 640])
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        outer.addWidget(splitter, 1)
        self._filling = False
        self.session.subscribe(self.refresh)
        self.refresh()

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setMinimumWidth(260)
        return area

    # ---------------------------------------------------------------------------------------------- layout
    def _build_header(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(10)
        title = QLabel("OPTIMIZE")
        title.setObjectName("sectionHeader")
        row.addWidget(title)
        self.phase_badge = StatusBadge("idle", "IDLE")
        self.phase_badge.setMinimumWidth(190)
        self.phase_badge.setToolTip("Where the workspace is: idle, compiling, editing, confirmed, running, ...")
        row.addWidget(self.phase_badge)
        self.message_label = _muted("", wrap=False, color=DANGER)
        row.addWidget(self.message_label, 1)
        row.addWidget(QLabel("Provider"))
        self.provider_combo = QComboBox()
        self.provider_combo.setToolTip("The model that compiles the prompt (and plans, with the LLM planner). "
                                       "Credentials come from environment variables only.")
        for choice in self.session.choices:
            self.provider_combo.addItem(choice.label, choice.key)
        self.provider_combo.currentIndexChanged.connect(
            lambda i: self._input(self.session.set_provider, self.provider_combo.itemData(i)))
        row.addWidget(self.provider_combo)
        row.addWidget(QLabel("Planner"))
        self.planner_combo = QComboBox()
        self.planner_combo.addItem("Rule-based (no LLM)", "rule")
        self.planner_combo.addItem("LLM planner (selected provider)", "llm")
        self.planner_combo.currentIndexChanged.connect(
            lambda i: self._input(self.session.set_planner, self.planner_combo.itemData(i)))
        row.addWidget(self.planner_combo)
        self.open_button = _button("Open run…", "Replay a saved experiment or run directory, with no model call")
        self.open_button.clicked.connect(self._open_run)
        row.addWidget(self.open_button)
        self.reset_button = _button("New", "Start over: clear the problem and the run")
        self.reset_button.clicked.connect(lambda: self._safely(self.session.reset))
        row.addWidget(self.reset_button)
        return bar

    def _column(self) -> tuple[QWidget, QVBoxLayout]:
        column = QWidget()
        column.setObjectName("optimizeColumn")
        layout = QVBoxLayout(column)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)
        return column, layout

    def _build_problem_column(self) -> QWidget:
        column, col = self._column()
        col.addWidget(_header("Problem in words"))
        self.example_combo = QComboBox()
        self.example_combo.addItem("Your own text", None)
        for ex in backend.EXAMPLES:
            self.example_combo.addItem(f"Example: {ex.title}", ex.key)
        self.example_combo.setToolTip("Bundled examples carry scripted replies, so the offline mock can compile them")
        self.example_combo.currentIndexChanged.connect(
            lambda i: self._input(self.session.select_example, self.example_combo.itemData(i)))
        col.addWidget(self.example_combo)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the problem: what to minimise or maximise, the variables and "
                                            "their ranges, and every hard constraint.")
        self.prompt_edit.setMinimumHeight(110)
        self.prompt_edit.textChanged.connect(self._on_prompt)
        col.addWidget(self.prompt_edit)
        buttons = QHBoxLayout()
        self.compile_button = _button("Compile", "Turn the text into a mathematical problem (IR)", primary=True)
        self.compile_button.clicked.connect(lambda: self._start(self.session.compile))
        self.blank_button = _button("Write by hand", "Skip the compiler: start from a blank problem")
        self.blank_button.clicked.connect(lambda: self._safely(self.session.start_blank))
        self.replies_button = _button("Load replies…", "Script the offline mock with a JSON list of replies")
        self.replies_button.clicked.connect(self._load_replies)
        for b in (self.compile_button, self.blank_button, self.replies_button):
            buttons.addWidget(b)
        col.addLayout(buttons)
        col.addWidget(_header("Compiler"))
        self.compile_label = _muted("Not compiled yet.")
        col.addWidget(self.compile_label)
        self.questions_box = QWidget()
        self.questions_form = QFormLayout(self.questions_box)
        self.questions_form.setContentsMargins(0, 0, 0, 0)
        self.answer_edits: list[QLineEdit] = []
        col.addWidget(self.questions_box)
        self.answer_button = _button("Answer and compile", "Add your answers to the prompt and compile again",
                                     primary=True)
        self.answer_button.clicked.connect(
            lambda: self._start(lambda: self.session.answer([e.text() for e in self.answer_edits])))
        col.addWidget(self.answer_button)
        col.addStretch(1)
        return column

    def _build_ir_column(self) -> QWidget:
        column, col = self._column()
        col.addWidget(_header("Compiled problem"))
        card = QFrame()
        card.setObjectName("mathCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        self.math_label = _muted("", color=INK_PRIMARY)
        self.math_label.setMinimumHeight(90)
        card_layout.addWidget(self.math_label)
        col.addWidget(card)
        self.digest_label = _muted("", color=INK_MUTED)
        col.addWidget(self.digest_label)
        self.editor_tabs = QTabWidget()
        self.tables: dict[str, QTableWidget] = {}
        for section in SECTIONS:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(4, 4, 4, 4)
            table = QTableWidget(0, len(FIELDS[section]) + 1)
            table.setHorizontalHeaderLabels([*FIELDS[section], "origin"])
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setMinimumHeight(130)
            table.itemChanged.connect(lambda item, s=section: self._on_cell(s, item))
            layout.addWidget(table)
            row = QHBoxLayout()
            add = _button("+ Add", f"Add one item to {section}")
            add.clicked.connect(lambda _c=False, s=section: self._safely(self.session.add_item, s))
            remove = _button("− Remove selected", f"Remove the selected item from {section}")
            remove.clicked.connect(lambda _c=False, s=section: self._remove(s))
            row.addWidget(add)
            row.addWidget(remove)
            row.addStretch(1)
            layout.addLayout(row)
            self.tables[section] = table
            self.editor_tabs.addTab(page, section.capitalize())
        col.addWidget(self.editor_tabs)
        self.issues_label = _muted("", color=DANGER)
        col.addWidget(self.issues_label)
        self.protected_label = _muted("", color=WARNING)
        col.addWidget(self.protected_label)
        buttons = QHBoxLayout()
        self.edit_button = _button("Edit", "Open the problem in the editor (unfreezes a confirmed problem)")
        self.edit_button.clicked.connect(lambda: self._safely(self.session.edit_draft))
        self.confirm_button = _button("Confirm", "Freeze the problem; its digest is what the run solves",
                                      primary=True)
        self.confirm_button.clicked.connect(lambda: self._safely(self.session.confirm))
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.confirm_button)
        buttons.addStretch(1)
        col.addLayout(buttons)
        col.addStretch(1)
        return column

    def _build_run_column(self) -> QWidget:
        column, col = self._column()
        col.addWidget(_header("Run"))
        buttons = QHBoxLayout()
        self.run_button = _button("Run", "Run the bounded agent on the confirmed problem", primary=True)
        self.run_button.clicked.connect(lambda: self._start(self.session.run))
        self.cancel_button = _button("Cancel", "Stop at the next iteration; the run is saved as cancelled")
        self.cancel_button.clicked.connect(lambda: self._safely(self.session.cancel))
        self.replay_button = _button("Replay", "Re-execute this run from its record, with no model call")
        self.replay_button.clicked.connect(lambda: self._start(self.session.replay))
        for b in (self.run_button, self.cancel_button, self.replay_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        col.addLayout(buttons)
        status = QHBoxLayout()
        self.run_badge = StatusBadge("idle", "NO RUN")
        self.run_badge.setMinimumWidth(150)
        status.addWidget(self.run_badge)
        self.run_text = _muted("", color=INK_SECONDARY)
        status.addWidget(self.run_text, 1)
        col.addLayout(status)
        meters = QHBoxLayout()
        self.meters = {name: MetricRow(name, "—") for name in ("Iterations", "Evaluations", "Tokens")}
        for m in self.meters.values():
            meters.addWidget(m)
        col.addLayout(meters)
        self.chart = ConvergenceChart()
        col.addWidget(self.chart)
        self.iteration_table = QTableWidget(0, 9)
        self.iteration_table.setHorizontalHeaderLabels(["#", "kind", "solver", "evals", "Σ evals", "objective",
                                                        "graded", "diagnosis", "note"])
        self.iteration_table.verticalHeader().setVisible(False)
        self.iteration_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.iteration_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.iteration_table.horizontalHeader().setStretchLastSection(True)
        self.iteration_table.setMinimumHeight(140)
        col.addWidget(self.iteration_table)
        col.addWidget(_header("Verification report"))
        self.source_label = _muted("", color=INK_MUTED)
        col.addWidget(self.source_label)
        self.status_rows: dict[str, tuple[StatusBadge, QLabel]] = {}
        grid = QFormLayout()
        grid.setHorizontalSpacing(10)
        for name, label in STATUS_LABELS.items():
            badge = StatusBadge("idle", "NOT RUN")
            badge.setMinimumWidth(118)
            reason = _muted("", color=INK_SECONDARY)
            box = QHBoxLayout()
            box.addWidget(badge)
            box.addWidget(reason, 1)
            grid.addRow(_muted(label, wrap=False, color=INK_PRIMARY), box)
            self.status_rows[name] = (badge, reason)
        col.addLayout(grid)
        self.codes_label = _muted("", color=INK_MUTED)
        col.addWidget(self.codes_label)
        self.violations_label = _muted("", color=INK_SECONDARY)
        col.addWidget(self.violations_label)
        col.addWidget(_header("Replay"))
        replay_row = QHBoxLayout()
        self.replay_badge = StatusBadge("idle", "NOT REPLAYED")
        self.replay_badge.setMinimumWidth(120)
        replay_row.addWidget(self.replay_badge)
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setToolTip("Step through the replayed iterations")
        self.step_slider.valueChanged.connect(self._on_step)
        replay_row.addWidget(self.step_slider, 1)
        self.step_label = _muted("", wrap=False, color=INK_MUTED)
        replay_row.addWidget(self.step_label)
        col.addLayout(replay_row)
        self.replay_label = _muted("", color=INK_SECONDARY)
        col.addWidget(self.replay_label)
        export = QHBoxLayout()
        self.export_button = _button("Export report…", "Write the verification report as JSON and HTML")
        self.export_button.clicked.connect(self._export_dialog)
        export.addWidget(self.export_button)
        export.addStretch(1)
        col.addLayout(export)
        col.addStretch(1)
        return column

    # -------------------------------------------------------------------------------------------- actions
    def _safely(self, fn, *args) -> None:
        """Run a session action; an illegal one becomes the header message, never an exception in a slot."""
        from orbitopt.viz.optimize.session import IllegalTransition

        try:
            fn(*args)
        except (IllegalTransition, ValueError) as exc:
            self.message_label.setText(_esc(exc))

    def _input(self, fn, *args) -> None:
        """A user input (not the rendering echoing state back), applied after the current signal returns, so a
        re-render can replace the widget that sent it."""
        if not self._filling:
            QTimer.singleShot(0, lambda: self._safely(fn, *args))

    def _start(self, make_job) -> None:
        from orbitopt.viz.optimize.session import IllegalTransition

        if self._thread is not None:
            return
        try:
            job = make_job()
        except (IllegalTransition, ValueError) as exc:
            self.message_label.setText(_esc(exc))
            return
        self._job = job
        self._thread = QThread(self)
        self._worker = _JobWorker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.failed.connect(self._on_failed, Qt.QueuedConnection)
        self._thread.start()

    def _end_thread(self) -> None:
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.quit()
            thread.wait(5000)

    def _on_progress(self, update) -> None:
        if self._job is not None:
            self.session.job_progress(self._job, update)

    def _on_finished(self, result) -> None:
        job, self._job = self._job, None
        self._end_thread()
        if job is not None:
            if job.kind == "run":
                self._remember_digest(result.experiment_dir, result.manifest_digest)
            self.session.job_finished(job, result)

    def _on_failed(self, message: str) -> None:
        job, self._job = self._job, None
        self._end_thread()
        if job is not None:
            self.session.job_failed(job, message)

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def shutdown(self) -> None:
        """Cancel a running job and wait for its thread (window close)."""
        if self._job is not None:
            self._job.cancel.set()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10000)

    def _on_prompt(self) -> None:
        text = self.prompt_edit.toPlainText()
        if not self._filling and self.session.prompt != text:
            self._safely(self.session.set_prompt, text)

    def _on_cell(self, section: str, item: QTableWidgetItem) -> None:
        fields = FIELDS[section]
        if item.column() < len(fields):
            self._input(self.session.set_cell, section, item.row(), fields[item.column()], item.text())

    def _remove(self, section: str) -> None:
        row = self.tables[section].currentRow()
        if row >= 0:
            self._safely(self.session.remove_item, section, row)

    def _on_step(self, value: int) -> None:
        if not self._filling and self.session.phase is Phase.REPLAY:
            self._safely(self.session.set_step, value)

    def _load_replies(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Mock replies (a JSON list)", "", "JSON (*.json)")
        if path:
            try:
                self.session.load_mock_replies(json.loads(Path(path).read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as exc:
                self.message_label.setText(_esc(f"{path}: {exc}"))

    def _open_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open an experiment or run directory", str(self.session.root))
        if path:
            self.open_run(Path(path))

    def open_run(self, path: Path) -> None:
        """Replay ``path`` with the manifest digest this machine kept for it, when there is one."""
        digest = self._digests().get(str(Path(path).resolve()))
        self._start(lambda: self.session.replay(path, expected_manifest_digest=digest))

    def _digests(self) -> dict:
        if self._settings is None:
            return {}
        try:
            return json.loads(self._settings.value("optimize/manifest_digests", "{}") or "{}")
        except ValueError:
            return {}

    def _remember_digest(self, experiment_dir: Path, digest: str) -> None:
        """Keep the run's manifest digest outside the run directory (whoever can edit the run can edit its
        manifest)."""
        if self._settings is None:
            return
        digests = self._digests()
        digests[str(Path(experiment_dir).resolve())] = digest
        self._settings.setValue("optimize/manifest_digests", json.dumps(digests, sort_keys=True))

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export the verification report", "verification-report.json",
                                              "JSON (*.json)")
        if path:
            self.export_report(Path(path), Path(path).with_suffix(".html"))

    def export_report(self, json_path: Path, html_path: Path) -> dict | None:
        """Write the shown run's verification report as JSON and HTML; ``None`` when no run is shown."""
        s = self.session
        view = s.run_view()
        if view is None:
            return None
        replay = s.replay_result
        problem = (replay.problem if s.phase is Phase.REPLAY and replay else None) or \
            (s.result.problem if s.result else {})
        meta = {"problem_digest": s.digest, "planner": s.planner, "provider": s.provider.label, "seed": s.seed,
                "manifest_digest": s.result.manifest_digest if s.result else (replay.meta.get("manifest_digest")
                                                                               if replay else None),
                "run_dir": str(s.result.experiment_dir if s.result else (replay.path if replay else "")),
                "replay": replay.status if replay else None}
        doc = report_document(view, problem=problem or {}, meta=meta)
        json_path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        html_path.write_text(report_html(doc), encoding="utf-8")
        return doc

    # ------------------------------------------------------------------------------------------- rendering
    def refresh(self) -> None:
        """Render the whole session (cheap: the problem and the trace are small)."""
        if self.session is None:
            return
        self._filling = True
        try:
            self._render_header()
            self._render_problem()
            self._render_ir()
            self._render_run()
        finally:
            self._filling = False

    def _render_header(self) -> None:
        s = self.session
        self.banner.setText(s.banner)
        self.banner.setProperty("offline", "false" if s.real_provider else "true")
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)
        text = s.phase.value.replace("_", " ")
        if s.phase is Phase.RUNNING and s.cancel_requested:
            text = "cancelling"
        self.phase_badge.set_state(PHASE_BADGE[s.phase], text)
        self.message_label.setText(_esc(s.error) if s.error else "")
        idle = s.phase not in (Phase.COMPILING, Phase.RUNNING, Phase.REPLAYING)
        for widget in (self.provider_combo, self.planner_combo, self.open_button, self.reset_button):
            widget.setEnabled(idle)
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(s.provider.key)))
        self.planner_combo.setCurrentIndex(max(0, self.planner_combo.findData(s.planner)))

    def _render_problem(self) -> None:
        s = self.session
        if self.prompt_edit.toPlainText() != s.prompt:
            self.prompt_edit.setPlainText(s.prompt)
        idle = s.phase not in (Phase.COMPILING, Phase.RUNNING, Phase.REPLAYING)
        self.prompt_edit.setReadOnly(not idle)
        self.example_combo.setEnabled(idle)
        self.example_combo.setCurrentIndex(max(0, self.example_combo.findData(s.example.key if s.example else None)))
        self.compile_button.setEnabled(s.can(Action.COMPILE))
        self.compile_button.setText("Recompile (revision)" if s.phase is Phase.EDITING else "Compile")
        self.blank_button.setEnabled(Action.EDIT in _legal(s))
        self.replies_button.setEnabled(idle)
        self.compile_label.setText(self._compile_html())
        clarifying = s.phase is Phase.CLARIFYING and s.compile_view is not None
        questions = s.compile_view.questions if clarifying else ()
        if questions is not self._shown_questions:     # rebuilt only for new questions: typed answers survive
            self._shown_questions = questions
            while self.questions_form.rowCount():
                self.questions_form.removeRow(0)
            self.answer_edits = []
            for q in questions:
                edit = QLineEdit()
                edit.setPlaceholderText("your answer, e.g. c = 3")
                self.answer_edits.append(edit)
                self.questions_form.addRow(_muted(f"<b>{_esc(q.get('kind', '').replace('_', ' '))}</b>: "
                                                  f"{_esc(q.get('text'))}", color=INK_PRIMARY), edit)
        self.questions_box.setVisible(clarifying)
        self.answer_button.setVisible(clarifying)

    def _compile_html(self) -> str:
        s = self.session
        if s.phase is Phase.COMPILING:
            return "<span style='color:%s'>Compiling…</span>" % INFO
        cv = s.compile_view
        if s.phase is Phase.REPLAY and s.replay_result is not None:
            return "Showing the problem the replayed run solved. Edit it to start a new experiment from it."
        if cv is None:
            return "Not compiled yet." if s.draft is None else "Written by hand: no compiler involved."
        title = {"compile": ("Accepted", SUCCESS), "needs_clarification": ("Needs clarification", WARNING),
                 "unsupported": ("Unsupported", DANGER), "rejected_unauthorized_change":
                 ("Rejected: unauthorised change to a locked item", DANGER), "compile_failed":
                 ("Compile failed", DANGER), "provider_error": ("Provider error", DANGER)}.get(cv.outcome,
                                                                                            (cv.outcome, INK_PRIMARY))
        parts = [f"<b style='color:{title[1]}'>{_esc(title[0])}</b> &nbsp;<span style='color:{INK_MUTED}'>"
                 f"{cv.attempts} round(s), {cv.tokens} tokens, {cv.wall_s * 1000:.0f} ms</span>"]
        if cv.questions and s.phase is not Phase.CLARIFYING:     # while clarifying, the form below asks them
            parts.append("<br>".join(f"• <b>{_esc(q.get('kind', '').replace('_', ' '))}</b>: {_esc(q.get('text'))}"
                                     for q in cv.questions))
        if cv.assumptions:
            parts.append("<b>Assumptions</b><br>" + "<br>".join(f"• {_esc(a.get('about'))}: {_esc(a.get('text'))}"
                                                                 for a in cv.assumptions))
        if cv.unsupported:
            parts.append(f"<b>{_esc(cv.unsupported.get('category'))}</b>: {_esc(cv.unsupported.get('explanation'))}")
        if cv.violations:
            parts.append("<b>Authorisation findings</b><br>" + "<br>".join(
                f"• {_esc(v.get('kind'))} {_esc(v.get('item'))}: {_esc(v.get('message'))}" for v in cv.violations)
                + "<br>Your problem was kept as it was.")
        if cv.errors:
            parts.append("<br>".join(f"• {_esc(e)}" for e in cv.errors[:8]))
        if s.phase is Phase.DRAFT:
            parts.append("Edit the problem by hand, or change the text and compile again.")
        return "<br><br>".join(parts)

    def _render_ir(self) -> None:
        s = self.session
        draft = s.draft
        doc = draft.doc if draft else {}
        problem_class = reference = None
        if draft is not None and draft.valid:
            try:
                from orbitopt.optimization import classify

                problem_class = classify(draft.problem).problem_class.value
            except Exception:  # noqa: BLE001 -- the class is a label, not a gate
                problem_class = None
        if s.example is not None and s.example.reference and s.rounds:
            reference = s.example.reference
        self.math_label.setText(problem_html(doc, problem_class=problem_class, reference=reference) if draft
                                else f"<span style='color:{INK_MUTED}'>Compile a problem, or write one by hand.</span>")
        if s.confirmed is not None:
            self.digest_label.setText(f"<b style='color:{SUCCESS}'>CONFIRMED</b> &nbsp;digest "
                                      f"<tt>{_esc(s.confirmed.digest())}</tt>")
        elif draft is not None:
            self.digest_label.setText(f"digest <tt>{_esc(draft.digest)}</tt>" if draft.valid else
                                      f"<span style='color:{DANGER}'>not valid IR: no digest</span>")
        else:
            self.digest_label.setText("")
        editable = s.phase is Phase.EDITING
        if draft is not None and draft.doc is not self._shown_doc:
            self._shown_doc = draft.doc
            for section, table in self.tables.items():
                self._fill_table(section, table, draft)
        elif draft is None:
            self._shown_doc = None
            for table in self.tables.values():
                table.setRowCount(0)
        for section, table in self.tables.items():
            table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                  | QAbstractItemView.AnyKeyPressed if editable else QAbstractItemView.NoEditTriggers)
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    widget = table.cellWidget(r, c)
                    if widget is not None:
                        widget.setEnabled(editable)
        for button in self.editor_tabs.findChildren(QPushButton):
            button.setEnabled(editable)
        issues = draft.issues if draft else ()
        self.issues_label.setText("<b>Validation errors</b><br>" + "<br>".join(f"• {_esc(i)}" for i in issues[:12])
                                  if issues else "")
        changes = draft.protected_changes() if draft and s.phase in (Phase.EDITING, Phase.CONFIRMED) else ()
        self.protected_label.setText(
            "<b>Your changes to protected items</b> (a later compile may not undo them)<br>"
            + "<br>".join(f"• {_esc(c.kind)} {_esc(c.item)}: {_esc(c.message)}" for c in changes) if changes else "")
        self.edit_button.setEnabled(Action.EDIT in _legal(s) and s.phase is not Phase.EDITING and draft is not None)
        self.confirm_button.setEnabled(s.can(Action.CONFIRM))

    def _fill_table(self, section: str, table: QTableWidget, draft) -> None:
        items = draft.doc.get(section) or []
        fields = FIELDS[section]
        table.setRowCount(len(items))
        for r, item in enumerate(items):
            bad: dict[str, str] = {}
            for issue in draft.issues_for(section, r):      # /<section>/<row>/<field>; a whole-item issue: name
                parts = issue.path.split("/")
                name = parts[3] if len(parts) > 3 and parts[3] in fields else "name"
                bad[name] = "; ".join(filter(None, (bad.get(name), issue.message)))
            for c, name in enumerate(fields):
                value = item.get(name)
                options = {"kind": KINDS, "sense": SENSES.get(section)}.get(name)
                if options:
                    combo = table.cellWidget(r, c)
                    if not isinstance(combo, QComboBox):
                        combo = self._combo(section, r, name, options)
                        table.setCellWidget(r, c, combo)
                    if combo.findText(str(value)) < 0:
                        combo.addItem(str(value))
                    combo.setCurrentText(str(value))
                    continue
                cell = QTableWidgetItem(_cell_text(value))
                if name in bad:
                    cell.setForeground(QBrush(QColor(DANGER)))
                    cell.setBackground(QBrush(QColor("#2a1210")))
                    cell.setToolTip(bad[name])
                table.setItem(r, c, cell)
            origin = item.get("origin", "")
            tag = (origin + (" · locked" if section == "constraints" and item.get("locked", origin == "user") else "")
                   if section in ("objectives", "constraints") else "—")
            cell = QTableWidgetItem(tag)
            cell.setFlags(Qt.ItemIsEnabled)
            cell.setForeground(QBrush(QColor(INK_MUTED if origin != "compiler" else WARNING)))
            table.setItem(r, len(fields), cell)

    def _combo(self, section: str, row: int, name: str, options) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(options))
        combo.currentTextChanged.connect(
            lambda text, s=section, r=row, n=name: self._input(self.session.set_cell, s, r, n, text))
        return combo

    def _render_run(self) -> None:
        s = self.session
        self.run_button.setEnabled(s.can(Action.RUN))
        self.cancel_button.setEnabled(s.can(Action.CANCEL))
        self.replay_button.setEnabled(Action.REPLAY in _legal(s) and s.result is not None)
        view = s.run_view()
        self.chart.set_view(view)
        self._render_status(view)
        self._render_iterations(view)
        self._render_report(view)
        self._render_replay()
        self.export_button.setEnabled(view is not None and not view.live)

    def _render_status(self, view: RunView | None) -> None:
        s = self.session
        if s.phase is Phase.RUNNING:
            self.run_badge.set_state("loading", "cancelling" if s.cancel_requested else "running")
            self.run_text.setText(f"Iteration {len(s.live) + 1}: {_esc(s.running_label)}"
                                  + (" · stops at the next iteration boundary" if s.cancel_requested else ""))
        elif s.phase is Phase.REPLAY and s.replay_result is not None and s.replay_result.trace is not None:
            outcome = s.replay_result.trace["outcome"]
            self.run_badge.set_state(RUN_STATUS_BADGE.get(outcome["status"], "idle"),
                                     outcome["status"].replace("_", " "))
            total = len(s.replay_result.trace["iterations"])
            self.run_text.setText(f"Recorded run: {_esc(RUN_STATUS_TEXT.get(outcome['status'], outcome['status']))}. "
                                  f"Showing iteration {s.step} of {total}.")
        elif view is not None and view.status:
            self.run_badge.set_state(RUN_STATUS_BADGE.get(view.status, "idle"), view.status.replace("_", " "))
            where = f" · saved as <tt>{_esc(s.result.experiment_dir.name)}</tt>" if s.result else ""
            self.run_text.setToolTip(str(s.result.experiment_dir) if s.result else "")
            self.run_text.setText(_esc(RUN_STATUS_TEXT.get(view.status, view.status)) + f" (stop: "
                                  f"{_esc(view.stop_reason)})" + where)
        elif s.phase is Phase.FAILED:
            self.run_badge.set_state("failed", "failed")
            self.run_text.setText(_esc(s.error))
        else:
            self.run_badge.set_state("idle", "no run")
            self.run_text.setText("Confirm the problem, then run." if s.confirmed is None else "Ready to run.")
        for meter in (view.meters if view else ()):
            self.meters[meter.label].set_value(f"{meter.used:,.0f} / {meter.limit:,.0f}")
        if view is None:
            for m in self.meters.values():
                m.set_value("—")

    def _render_iterations(self, view: RunView | None) -> None:
        rows = view.rows if view else ()
        table = self.iteration_table
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            cells = [str(row.index), row.kind, row.solver, f"{row.evaluations:,}", f"{row.cumulative:,}",
                     _fmt(row.objective), row.feasibility.replace("_", " "), row.diagnosis.replace("_", " "), row.note]
            color = {"rejected": DANGER, "running": INFO}.get(row.kind) or \
                (DANGER if row.feasibility == "infeasible" else None)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if color:
                    item.setForeground(QBrush(QColor(color)))
                table.setItem(r, c, item)
        if rows:
            table.scrollToBottom()

    def _render_report(self, view: RunView | None) -> None:
        s = self.session
        self.source_label.setText(_esc(view.report_source) if view else "No run yet: every status reads NOT RUN.")
        for line in (view.statuses if view else ()):
            badge, reason = self.status_rows[line.name]
            badge.set_state(OUTCOME_BADGE.get(line.outcome, "idle"), line.outcome.replace("_", " "))
            reason.setText(_esc(line.reason))
        if view is None:
            for badge, reason in self.status_rows.values():
                badge.set_state("idle", "not run")
                reason.setText("")
        codes = ", ".join(view.codes) if view else ""
        self.codes_label.setText(f"Verdict {_esc(view.verdict)} · finding codes: {_esc(codes or 'none')}"
                                 if view and view.verdict else "")
        lines = []
        if view and view.violations:
            lines.append(f"<b style='color:{DANGER}'>Policy violations</b>: {len(view.violations)} proposal(s) "
                         "refused by the runtime and never applied")
            lines += [f"<span style='color:{DANGER}'>• iteration {v.iteration}: <b>{_esc(v.code)}</b> "
                      f"{_esc(v.path)}</span> {_esc(v.message)}" for v in view.violations[:10]]
        problem = s.confirmed
        if view and view.fingerprint and problem is not None and s.phase is not Phase.REPLAY:
            same = view.fingerprint == problem.hard_constraint_fingerprint()
            lines.append(f"<span style='color:{SUCCESS if same else DANGER}'>Hard constraints "
                         f"{'unchanged' if same else 'CHANGED'}</span>: fingerprint <tt>{_esc(view.fingerprint[:16])}"
                         "</tt> " + ("matches the confirmed problem" if same else "differs from the confirmed problem"))
        self.violations_label.setText("<br>".join(lines))

    def _render_replay(self) -> None:
        s = self.session
        result = s.replay_result
        replaying = s.phase is Phase.REPLAY and result is not None
        total = len(result.trace["iterations"]) if replaying and result.trace else 0
        self.step_slider.setEnabled(replaying and total > 0)
        self.step_slider.setRange(0, total)
        self.step_slider.setValue(s.step if replaying else 0)
        self.step_label.setText(f"after iteration {s.step} of {total}" if replaying else "")
        if s.phase is Phase.REPLAYING:
            self.replay_badge.set_state("loading", "replaying")
            self.replay_label.setText("Re-executing from the record; no model is called.")
        elif replaying:
            self.replay_badge.set_state("verified" if result.equal else "failed", result.status)
            parts = "<br>".join(f"<b>{_esc(p.name)}</b> {_esc(p.status.replace('_', ' '))}: "
                                + "; ".join(_esc(_short(d)) for d in p.details[:4]) for p in result.parts)
            self.replay_label.setToolTip("\n".join(d for p in result.parts for d in p.details))
            digest = ("checked against the digest kept on this machine" if result.digest_checked else
                      "no out-of-band digest known: checked against the directory's own manifest")
            self.replay_label.setText(f"<tt>{_esc(result.path.name)}</tt><br>{parts}<br>"
                                      f"<span style='color:{INK_MUTED}'>Manifest {digest}. "
                                      f"{result.wall_s * 1000:.0f} ms, no model call.</span>")
        else:
            self.replay_badge.set_state("idle", "not replayed")
            self.replay_label.setText("Replay re-runs the compile from its recorded cassette and the run from its "
                                      "recorded proposals; no model is called.")


def _tick_digits(ticks) -> int:
    """Significant digits that keep neighbouring tick labels apart (0.499999 and 0.5, not 0.5 and 0.5)."""
    import math

    if len(ticks) < 2:
        return 4
    step = abs(ticks[1] - ticks[0])
    scale = max(abs(t) for t in ticks) or 1.0
    return max(3, min(12, int(math.ceil(math.log10(scale / step))) + 2)) if step > 0 else 4


_HEX = re.compile(r"\b([0-9a-f]{12})[0-9a-f]{40,}\b")


def _short(text: str) -> str:
    """Long digests shortened to their first 12 hex digits (the full text stays in the tooltip)."""
    return _HEX.sub(r"\1…", text)


def _cell_text(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value)
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _legal(session: StudioSession):
    from orbitopt.viz.optimize.session import TRANSITIONS

    return TRANSITIONS[session.phase]
