"""Mission Designer + Vehicle Builder forms for Mission Control (FL-502).

The thin Qt view over :mod:`orbitopt.flightlab.designer`: every widget maps to
a plain dict field, every rejection from the Qt-free engine (``build_target``
schema errors, OT-schema violations) lands back on the *field* that caused it,
and the Simulate button pushes the picklable job through the compute-pool
worker with Cancel and an input-hash result cache. The Explorer/Engineer mode
switch only changes which fields are visible -- both modes collect the same
dict and hand it to the same physics.

Nothing on this page edits YAML: presets load into the form, edits produce a
vehicle document the form wrote field-by-field, and a finished run opens in
the sibling replay tab (:class:`orbitopt.flightlab.ui.workbench.FlightLabWorkbench`).
"""
from __future__ import annotations

import functools
import json
import os
import signal
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QThread, QDate, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from orbitopt.flightlab import designer as D
from orbitopt.flightlab.contract import FLIGHT_PROVENANCE_TAGS
from orbitopt.viz.compute_pool import ComputePool
from orbitopt.viz.theme import DANGER, INK_MUTED, INK_SECONDARY, SUCCESS, WARNING
from orbitopt.viz.ui_kit import StatusBadge

_RECOVERY_KINDS = (("no recovery", "none"), ("RTLS at launch site", "rtls"), ("downrange landing", "downrange"))
_FREE_FLIGHT = (("ballistic", "ballistic"), ("weathercock (aero-stable)", "aero_stable"),
                ("follow program", "attitude_program"))

_FIELD_ERROR_QSS = f"""
QDoubleSpinBox[flErr="1"], QLineEdit[flErr="1"], QComboBox[flErr="1"],
QDateEdit[flErr="1"], QPlainTextEdit[flErr="1"] {{ border: 1px solid {DANGER}; }}
"""


def _num(lo: float, hi: float, value: float, step: float = 1.0, decimals: int = 1,
         suffix: str = "", special: str | None = None) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(suffix)
    if special is not None:
        box.setMinimum(lo)
        box.setSpecialValueText(special)
    box.setValue(value)
    return box


class _CurveDialog(QDialog):
    """Engineer-mode editor for a tabulated curve (Cd/Ca over Mach, thrust
    throttling over time): JSON rows ``[[x, y], ...]`` straight from the
    vehicle document. Escape/Cancel changes nothing."""

    def __init__(self, title: str, value, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(420, 260)
        self.result_value = value
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{title} — JSON rows [x, y]. Blank / 'null' restores the default."))
        self._text = QPlainTextEdit()
        self._text.setPlainText(json.dumps(value) if value is not None else "")
        layout.addWidget(self._text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self):
        raw = self._text.toPlainText().strip()
        if raw in ("", "null"):
            self.result_value = None
            self.accept()
            return
        try:
            parsed = json.loads(raw)
            assert isinstance(parsed, list) and parsed and all(
                isinstance(r, list) and len(r) == 2 for r in parsed)
            self.result_value = [[float(x), float(y)] for x, y in parsed]
        except Exception:  # noqa: BLE001 -- keep the dialog open, explain, never half-apply
            QMessageBox.warning(self, "Not a curve",
                                "Expected JSON rows like [[0, 0.35], [0.8, 0.38], ...]")
            return
        self.accept()


class _StageCard(QGroupBox):
    """One stage of the Vehicle Builder. Spinners carry the scalar case;
    tabulated values loaded from a preset (thrust throttling, Cd/Ca over Mach,
    fairing jettisons) survive round-trips untouched unless the user edits
    them -- the engineer-mode table dialogs edit the curve itself."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, page: "FlightLabDesignerPage", index: int):
        super().__init__(f"Stage {index + 1}")
        self.page = page
        self.thrust_curve = None
        self.ca_curve = None
        self.jettisons: list[dict] = []
        self._loading = False

        grid = QGridLayout(self)
        grid.setContentsMargins(10, 6, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        def add(row: int, col: int, label: str, widget: QWidget, engineer: bool = False):
            cap = QLabel(label)
            cap.setStyleSheet(f"color: {INK_SECONDARY};")
            grid.addWidget(cap, row, 2 * col)
            grid.addWidget(widget, row, 2 * col + 1)
            if engineer:
                page._register_engineer(cap, widget)
            return widget

        self.mass_initial = add(row := 0, col := 0, "Initial mass", _num(0.1, 1e6, 50.0, 5.0, 1, " t"))
        self.mass_dry = add(row, 1, "Dry mass", _num(0.0, 1e6, 5.0, 1.0, 1, " t"))
        self.thrust = add(row := 1, col := 0, "Thrust (vac)", _num(0.0, 1e6, 1000.0, 50.0, 1, " kN"))
        self.isp = add(row, 1, "Isp (vac)", _num(100.0, 600.0, 320.0, 5.0, 1, " s"))
        self.burn_end = add(row := 2, col := 0, "Burn ends", _num(1.0, 20000.0, 160.0, 5.0, 0, " s"))
        self.separation = add(row, 1, "Separation", _num(0.0, 20000.0, 0.0, 1.0, 0, " s", "— last stage"))
        self.burn_start = add(row := 3, col := 0, "Burn starts", _num(0.0, 20000.0, 0.0, 1.0, 0, " s"), True)
        self.throttle = add(row, 1, "Min throttle", _num(0.0, 100.0, 100.0, 5.0, 0, " %", "full"), True)
        self.nozzle = add(row := 4, col := 0, "Nozzle exit area", _num(0.0, 500.0, 0.0, 0.5, 2, " m²", "—"), True)
        self.diameter = add(row, 1, "Diameter", _num(0.1, 40.0, 3.7, 0.1, 2, " m"))
        self.cn_alpha = add(row := 5, col := 0, "CN-alpha", _num(0.0, 20.0, 0.0, 0.1, 2, " /rad", "off"), True)
        self.ballistic = add(row, 1, "Ballistic coeff β", _num(1.0, 1e5, 100.0, 10.0, 1, " kg/m²"), True)
        self.cd_button = add(row := 6, col := 0, "Cd/Ca table", QPushButton("edit…"), True)
        self.free_flight = add(row, 1, "Coast behaviour", QComboBox(), True)
        for text, data in _FREE_FLIGHT:
            self.free_flight.addItem(text, data)
        rm = QPushButton("remove stage")
        rm.setStyleSheet(f"color: {DANGER};")
        rm.clicked.connect(lambda: self.removed.emit(self))
        grid.addWidget(rm, 6, 3)

        self.cd_button.clicked.connect(self._edit_cd)
        for w in (self.mass_initial, self.mass_dry, self.thrust, self.isp, self.burn_end,
                  self.separation, self.burn_start, self.throttle, self.nozzle, self.diameter,
                  self.cn_alpha, self.ballistic):
            w.valueChanged.connect(lambda _v, key=w: self._touched(key))
        self.free_flight.currentIndexChanged.connect(lambda _i: self._touched(self.free_flight))
        self._fields = [self.mass_initial, self.mass_dry, self.thrust, self.isp, self.burn_end,
                        self.separation, self.burn_start, self.throttle, self.nozzle,
                        self.diameter, self.cn_alpha, self.ballistic, self.free_flight]

    # -------------------------------------------------------------- round-trip
    @staticmethod
    def _scalar(curve):
        if isinstance(curve, list) and curve:
            return float(curve[0][1])
        return float(curve) if isinstance(curve, (int, float)) else None

    def load(self, stage: dict):
        self._loading = True
        try:
            self.mass_initial.setValue(stage["initial_mass_t"])
            self.mass_dry.setValue(stage["dry_mass_t"])
            self.thrust_curve = stage["thrust_kn"] if isinstance(stage["thrust_kn"], list) else None
            self.thrust.setValue(max(self._scalar(stage["thrust_kn"]) or 0.0, 0.1))
            self.isp.setValue(stage["isp_vac_s"])
            self.burn_start.setValue(stage.get("burn_start_s", 0.0))
            self.burn_end.setValue(max(stage.get("burn_end_s", 1.0), 1.0))
            self.throttle.setValue(float(stage.get("min_throttle_pct")
                                         if stage.get("min_throttle_pct") is not None else 100.0))
            self.nozzle.setValue(stage.get("nozzle_area_m2", 0.0))
            self.diameter.setValue(stage.get("diameter_m", 3.7))
            self.cn_alpha.setValue(stage.get("cn_alpha", 0.0))
            self.ballistic.setValue(stage.get("ballistic_coeff_kg_m2", 100.0))
            self.ca_curve = stage.get("ca") if isinstance(stage.get("ca"), list) else None
            self.jettisons = list(stage.get("jettisons") or [])
            self.separation.setValue(float(stage["separation_time_s"] or 0.0))
            idx = self.free_flight.findData(stage.get("free_flight", "attitude_program"))
            self.free_flight.setCurrentIndex(max(idx, 0))
        finally:
            self._loading = False

    def to_form(self) -> dict:
        return {
            "initial_mass_t": self.mass_initial.value(),
            "dry_mass_t": self.mass_dry.value(),
            "thrust_kn": self.thrust_curve if self.thrust_curve is not None else self.thrust.value(),
            "isp_vac_s": self.isp.value(),
            "burn_start_s": self.burn_start.value(),
            "burn_end_s": self.burn_end.value(),
            "min_throttle_pct": None if self.throttle.value() >= 100.0 else self.throttle.value(),
            "nozzle_area_m2": self.nozzle.value() or None,
            "diameter_m": self.diameter.value(),
            "ca": self.ca_curve,
            "cn_alpha": self.cn_alpha.value() or None,
            "ballistic_coeff_kg_m2": self.ballistic.value(),
            "free_flight": self.free_flight.currentData(),
            "separation_time_s": self.separation.value() if self.separation.value() > 0 else None,
            "jettisons": self.jettisons,
        }

    def _touched(self, widget):
        if self._loading:
            return
        if widget is self.thrust:
            self.thrust_curve = None  # an edited thrust scalar replaces a preset curve
        self.page._demote_provenance()
        self.changed.emit()

    def _edit_cd(self):
        dlg = _CurveDialog("Cd/Ca over Mach",
                           self.ca_curve if self.ca_curve is not None else D.DEFAULT_CA_TABLE,
                           self)
        if dlg.exec() == QDialog.Accepted:
            self.ca_curve = dlg.result_value
            self._touched(self.cd_button)

    def set_last(self, last: bool):
        self.separation.setDisabled(last)
        if last:
            self.separation.setValue(0.0)


class FlightLabDesignerPage(QWidget):
    """The FL-502 designer: forms in, cache-checked ``simulate`` job out,
    finished runs announced through :attr:`run_ready` (the workbench opens
    them in the replay tab)."""

    run_ready = Signal(str)          # run directory to open in the replay tab
    simulate_finished = Signal(dict)  # worker result summary
    simulate_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flightLabDesigner")
        self._engineer = False
        self._engineer_widgets: list[QWidget] = []
        self._fields: dict[str, QWidget] = {}
        self._cards: list[_StageCard] = []
        self._loading = False
        self._pool: ComputePool | None = None
        self._thread: _SimWorker | None = None
        self._job: dict | None = None
        self._t0 = 0.0
        self._timer = QTimer(self, interval=500)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self.setStyleSheet(_FIELD_ERROR_QSS)
        self._apply_mode()
        self._sync_recovery(0)
        presets = list(D.vehicle_presets())
        if presets:  # start from a template, not from empty — "no YAML" is the point
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(self.preset_combo.findData(presets[0]))
            self.preset_combo.blockSignals(False)
            self._load_preset(presets[0])
        self._validate()

    # ------------------------------------------------------------------- UI
    def _register_engineer(self, *widgets: QWidget):
        for w in widgets:
            self._engineer_widgets.append(w)

    def _field(self, name: str, widget: QWidget) -> QWidget:
        self._fields[name] = widget
        return widget

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Mission Designer")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        header.addWidget(title)
        self.mode_button = QPushButton("Engineer mode")
        self.mode_button.setCheckable(True)
        self.mode_button.setToolTip("Explorer ↔ Engineer: only which fields are visible. The physics is the same.")
        self.mode_button.toggled.connect(self._toggle_mode)
        header.addWidget(self.mode_button)
        header.addStretch(1)
        self.badge = StatusBadge("ready", "ready")
        header.addWidget(self.badge)
        self.status = QLabel("Fill the form — Simulate runs when the documents are valid.")
        self.status.setStyleSheet(f"color: {INK_MUTED};")
        header.addWidget(self.status)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulate)
        header.addWidget(self.cancel_button)
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        header.addWidget(self.simulate_button)
        root.addLayout(header)

        self.optimize_note = QLabel()
        self.optimize_note.setStyleSheet(f"color: {INK_MUTED}; font-style: italic;")
        root.addWidget(self.optimize_note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, 1)
        body = QWidget()
        scroll.setWidget(body)
        columns = QGridLayout(body)
        columns.setContentsMargins(2, 2, 2, 2)
        columns.setColumnStretch(0, 2)
        columns.setColumnStretch(1, 3)
        columns.addWidget(self._mission_group(), 0, 0, Qt.AlignTop)
        columns.addWidget(self._vehicle_group(), 0, 1)

        self.derived = QLabel()
        self.derived.setObjectName("flDesignerDerived")
        self.derived.setStyleSheet(f"color: {INK_SECONDARY};")
        root.addWidget(self.derived)

    def _mission_group(self) -> QWidget:
        box = QGroupBox("Mission")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        def row(label: str, widget: QWidget, engineer: bool = False):
            cap = QLabel(label)
            cap.setStyleSheet(f"color: {INK_SECONDARY};")
            if engineer:
                self._register_engineer(cap, widget)
            self._fields[label] = widget
            form.addRow(cap, widget)
            return widget

        self.site_combo = QComboBox()
        for site in D.mission_sites():
            self.site_combo.addItem(site.name, site.id)
        row("Launch site", self.site_combo)
        self.periapsis = row("Periapsis", _num(120.0, 1e6, 400.0, 10.0, 1, " km"))
        self.apoapsis = row("Apoapsis", _num(120.0, 1e6, 400.0, 10.0, 1, " km"))
        self.inclination = row("Inclination", _num(0.0, 180.0, 51.6, 1.0, 1, "°"))
        self.raan = _num(0.0, 360.0, 0.0, 5.0, 1, "°")
        self.raan_auto = QCheckBox("auto (unconstrained)")
        self.raan_auto.setChecked(True)
        raan_row = QHBoxLayout()
        raan_row.addWidget(self.raan)
        raan_row.addWidget(self.raan_auto)
        raan_w = QWidget()
        raan_w.setLayout(raan_row)
        self._fields["raan"] = raan_w
        self._register_engineer(QLabel("RAAN"), raan_w)
        form.addRow(QLabel("RAAN"), raan_w)
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        row("Launch date", self.date_edit)
        self.window_days = row("Window", _num(0.01, 365.0, 2.0, 0.5, 2, " days"), True)
        self.recovery = QComboBox()
        for text, data in _RECOVERY_KINDS:
            self.recovery.addItem(text, data)
        row("Recovery", self.recovery)
        self.reserve = row("Recovery reserve", _num(0.0, 1e6, 1000.0, 100.0, 0, " kg"), True)
        self.max_q = row("Max dynamic pressure", _num(0.0, 500.0, 60.0, 5.0, 1, " kPa", "—"), True)
        self.max_g = row("Max acceleration", _num(0.0, 20.0, 5.0, 0.5, 2, " g", "—"), True)
        self.min_periapsis = row("Min insertion periapsis", _num(0.0, 1e6, 380.0, 10.0, 1, " km", "—"), True)

        self.azimuth = row("Launch azimuth", _num(0.0, 359.5, 90.0, 0.5, 1, "°"), True)
        self.attitude_auto = QCheckBox("generate attitude program from site + inclination")
        self.attitude_auto.setChecked(True)
        row("Attitude", self.attitude_auto, True)
        self.attitude_text = QPlainTextEdit()
        self.attitude_text.setFixedHeight(110)
        self.attitude_text.setPlaceholderText('[{"t_s": 0, "azimuth_deg": 90, "elevation_deg": 45}, ...]')
        self.attitude_text.textChanged.connect(lambda: self._changed())
        row("Attitude program (t, az, elev)", self.attitude_text, True)

        for w in (self.periapsis, self.apoapsis, self.inclination, self.raan,
                  self.window_days, self.reserve, self.max_q, self.max_g,
                  self.min_periapsis, self.azimuth):
            w.valueChanged.connect(lambda _v: self._changed())
        for w in (self.site_combo, self.recovery):
            w.currentIndexChanged.connect(lambda _i: self._changed())
        self.raan_auto.toggled.connect(lambda _v: self._changed())
        self.date_edit.dateChanged.connect(lambda _d: self._changed())
        self.attitude_auto.toggled.connect(self._regenerate_attitude)
        self.attitude_text.textChanged.connect(self._attitude_edited)
        self.recovery.currentIndexChanged.connect(self._sync_recovery)
        self._fields.update({"site": self.site_combo, "periapsis_km": self.periapsis,
                             "apoapsis_km": self.apoapsis, "inclination_deg": self.inclination,
                             "raan_deg": self.raan, "date": self.date_edit,
                             "window_days": self.window_days, "recovery": self.recovery,
                             "recovery_reserve_kg": self.reserve, "max_q_kpa": self.max_q,
                             "max_g": self.max_g, "min_periapsis_km": self.min_periapsis})
        return box

    def _vehicle_group(self) -> QWidget:
        box = QGroupBox("Vehicle")
        layout = QVBoxLayout(box)

        bar = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— custom —", "")
        for name in D.vehicle_presets():
            self.preset_combo.addItem(name, name)
        self.preset_combo.currentIndexChanged.connect(
            lambda _i: self._load_preset(self.preset_combo.currentData()))
        bar.addWidget(QLabel("preset"))
        bar.addWidget(self.preset_combo)
        self.vehicle_name = QLineEdit("designer-vehicle")
        self.vehicle_name.setFixedWidth(160)
        self.vehicle_name.textChanged.connect(lambda _t: self._changed())
        bar.addWidget(QLabel("name"))
        bar.addWidget(self.vehicle_name)
        self.provenance = QComboBox()
        for tag in FLIGHT_PROVENANCE_TAGS:
            self.provenance.addItem(tag, tag)
        bar.addWidget(QLabel("provenance"))
        bar.addWidget(self.provenance)
        self.payload = _num(0.0, 1e5, 0.0, 0.5, 2, " t")
        bar.addWidget(QLabel("payload"))
        bar.addWidget(self.payload)
        self.end_time = _num(1.0, 1e5, 950.0, 30.0, 0, " s")
        eng_label = QLabel("sim ends")
        self._register_engineer(eng_label, self.end_time)
        bar.addWidget(eng_label)
        bar.addWidget(self.end_time)
        bar.addStretch(1)
        add_btn = QPushButton("＋ stage")
        add_btn.clicked.connect(lambda: self._add_stage() or self._changed())
        bar.addWidget(add_btn)
        layout.addLayout(bar)

        self.stages_host = QWidget()
        self.stages_layout = QVBoxLayout(self.stages_host)
        self.stages_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stages_host)
        layout.addStretch(1)

        self.payload.valueChanged.connect(lambda _v: (self._demote_provenance(), self._changed()))
        self.end_time.valueChanged.connect(lambda _v: self._changed())
        self._fields["name"] = self.vehicle_name
        return box

    # -------------------------------------------------------------- behaviour
    def _toggle_mode(self, engineer: bool):
        self._engineer = engineer
        self.mode_button.setText("Explorer mode" if engineer else "Engineer mode")
        self._apply_mode()

    def _apply_mode(self):
        for w in self._engineer_widgets:
            w.setVisible(self._engineer)
        note = ("E2 ascent optimizer available — Simulate will pre-optimize."
                if D.optimize_available() else
                "E2 not merged yet — Simulate runs plain OpenTsiolkovsky ascent (feature-checked).")
        self.optimize_note.setText(note)

    def _add_stage(self):
        card = _StageCard(self, len(self._cards))
        card.changed.connect(self._changed)
        card.removed.connect(self._remove_stage)
        self._cards.append(card)
        self.stages_layout.addWidget(card)
        self._sync_last_stage()

    def _remove_stage(self, card):
        if len(self._cards) <= 1:
            return
        self._cards.remove(card)
        card.deleteLater()
        self._sync_last_stage()
        self._changed()

    def _sync_last_stage(self):
        for i, card in enumerate(self._cards):
            card.set_last(i == len(self._cards) - 1)
            card.setTitle(f"Stage {i + 1}")

    def _demote_provenance(self):
        """Editing a value a preset supplied invalidates its provenance tag."""
        if not self._loading and self.provenance.currentData() != "illustrative":
            idx = self.provenance.findData("illustrative")
            self.provenance.blockSignals(True)
            self.provenance.setCurrentIndex(idx)
            self.provenance.blockSignals(False)

    def _changed(self):
        if not self._loading:
            self._validate()

    def _load_preset(self, name: str):
        if not name:
            return
        doc = D.load_vehicle_preset(name)
        form = D.vehicle_form_from_document(doc)
        self._loading = True
        try:
            self.vehicle_name.setText(form["name"])
            idx = self.provenance.findData(form["provenance"])
            self.provenance.setCurrentIndex(max(idx, 0))
            self.payload.setValue(form["payload_t"])
            self.end_time.setValue(form["end_time_s"])
            for card in self._cards:
                card.deleteLater()
            self._cards.clear()
            self.stages_layout.setContentsMargins(0, 0, 0, 0)
            for stage in form["stages"]:
                self._add_stage()
                self._cards[-1].load(stage)
            self._sync_last_stage()
            self.attitude_text.setPlainText(json.dumps(form["attitude_program"], indent=1))
        finally:
            self._loading = False
        self._validate()

    def _regenerate_attitude(self, _auto=True):
        if self._loading or not self.attitude_auto.isChecked():
            return
        self._loading = True
        try:
            self.attitude_text.setPlainText(
                json.dumps(D.default_attitude_program(self.azimuth.value()), indent=1))
        finally:
            self._loading = False

    def _attitude_edited(self):
        if not self._loading and self.attitude_auto.isChecked():
            self.attitude_auto.setChecked(False)  # manual edit stops auto-generation

    def _sync_recovery(self, _i):
        self.reserve.setEnabled(self.recovery.currentData() != "none")

    # ------------------------------------------------------------ form -> doc
    def mission_form(self) -> dict:
        return {
            "site": self.site_combo.currentData(),
            "periapsis_km": self.periapsis.value(),
            "apoapsis_km": self.apoapsis.value(),
            "inclination_deg": self.inclination.value(),
            "raan_deg": None if self.raan_auto.isChecked() else self.raan.value(),
            "date": self.date_edit.date().toString("yyyy-MM-dd"),
            "window_days": self.window_days.value(),
            "recovery": self.recovery.currentData(),
            "recovery_reserve_kg": self.reserve.value() if self.recovery.currentData() != "none" else None,
            "max_q_kpa": self.max_q.value(),
            "max_g": self.max_g.value(),
            "min_periapsis_km": self.min_periapsis.value(),
        }

    def vehicle_form(self) -> dict:
        return {
            "name": self.vehicle_name.text().strip() or "designer-vehicle",
            "provenance": self.provenance.currentData(),
            "payload_t": self.payload.value(),
            "end_time_s": self.end_time.value(),
            "output_step_s": 2.0,
            "attitude_program": json.loads(self.attitude_text.toPlainText() or "[]"),
            "stages": [card.to_form() for card in self._cards],
            "preset": self.preset_combo.currentData() or "",
        }

    # ------------------------------------------------------------- validation
    def _mark(self, widget: QWidget, message: str | None):
        widget.setProperty("flErr", "1" if message else None)
        widget.setToolTip(message or "")
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _clear_error_marks(self):
        for w in list(self._fields.values()) + [self.attitude_text]:
            self._mark(w, None)
        for card in self._cards:
            for f in card._fields:
                self._mark(f, None)

    def _validate(self) -> bool:
        self._clear_error_marks()

        errors: dict[str, str] = {}
        target, mission_errors = D.target_from_form(self.mission_form())
        errors.update(mission_errors)
        try:
            vform = self.vehicle_form()
        except (ValueError, KeyError, TypeError) as exc:
            vform = None
            errors["attitude_program"] = f"attitude program is not valid JSON: {exc}"
        doc = None
        if vform is not None:
            try:
                doc = D.vehicle_document(vform)
                errors.update(D.validate_vehicle_document(doc))
            except (ValueError, KeyError, TypeError) as exc:
                doc = None
                errors.setdefault("attitude_program", f"vehicle form rejected: {exc}")

        if errors:
            self._show_errors(errors)
            ok = target is not None and doc is not None and not errors
        else:
            ok = True
        if ok:
            self._update_derived(doc)
            digest, _ = D.build_job(target, doc, {"timeout_s": 300.0, "refly": True})
            cached = D.cached_run_dir(digest)
            self.badge.set_state("ready", "ready to simulate")
            self.simulate_button.setText("Simulate (cached — opens replay)" if cached else "Simulate")
        else:
            self.badge.set_state("warning", "fix fields")
            self.simulate_button.setText("Simulate")
        self.simulate_button.setEnabled(bool(ok) and self._thread is None)
        return ok

    def _show_errors(self, errors: dict[str, str]):
        for key, message in sorted(errors.items()):
            widget = self._resolve_field(key)
            if widget is not None:
                self._mark(widget, message)

    def _resolve_field(self, key: str):
        if "[" in key and key.startswith("stages["):
            try:
                head, rest = key.split("]", 1)
                card = self._cards[int(head[len("stages["):])]
            except (ValueError, IndexError):
                return None
            attr = {"mass_initial_kg": "mass_initial", "mass_dry_kg": "mass_dry",
                    "engine.thrust_vac_n": "thrust", "engine.isp_vac_s": "isp",
                    "engine.burn_end_s": "burn_end", "engine.burn_start_s": "burn_start",
                    "separation_time_s": "separation", "aero.diameter_m": "diameter",
                    "engine.nozzle_exit_area_m2": "nozzle"}.get(rest.lstrip("."))
            return getattr(card, attr, card) if attr else card
        return {"attitude_program": self.attitude_text, "stages": self.stages_host,
                "constraints": self.max_q, "name": self.vehicle_name}.get(key, self._fields.get(key))

    def _update_derived(self, doc: dict):
        m = D.derived_metrics(doc)
        tw = m.get("liftoff_thrust_to_weight")
        stages = m["stages"]
        tw_txt = ("—" if tw is None else
                  f"<b style='color:{SUCCESS if m['liftoff_tw_ok'] else DANGER}'>{tw:.2f}</b> "
                  f"(min {D.MIN_LIFTOFF_THRUST_TO_WEIGHT:.2f})")
        dv = " + ".join(f"{(s['ideal_delta_v_m_s'] or 0.0) / 1000.0:.1f}" for s in stages)
        prop = sum(s["propellant_kg"] for s in stages) + float(doc.get("payload_kg", 0.0))
        self.derived.setText(
            f"liftoff T/W {tw_txt}   ·   ideal Δv {dv} km/s (vac, no losses)   ·   "
            f"mass {float(stages and doc['stages'][0].get('mass_initial_kg') or 0) / 1000.0:,.1f} t liftoff   ·   "
            f"propellant {prop / 1000.0:,.1f} t   ·   "
            f"Cd area {stages[0]['reference_area_m2']:.1f} m²"
            + ("" if stages[0]["sea_level_thrust_n"] is None else
               f"   ·   SL thrust {stages[0]['sea_level_thrust_n'] / 1e6:,.1f} MN"))
        for card, s in zip(self._cards, stages):
            dv_s = s["ideal_delta_v_m_s"]
            card.setTitle(f"Stage {s['index'] + 1} — "
                          + (f"Δv {dv_s / 1000.0:.2f} km/s · prop {s['propellant_kg'] / 1000.0:,.1f} t"
                             if dv_s is not None else "no engine"))

    # ------------------------------------------------------------ run/cancel
    def _pool_instance(self) -> ComputePool | None:
        if not ComputePool.enabled():
            return None
        if self._pool is None or self._pool.closed:
            self._pool = ComputePool()
        return self._pool

    def simulate(self):
        if self._thread is not None or not self._validate():
            return
        target, _ = D.target_from_form(self.mission_form())
        vform = self.vehicle_form()
        doc = D.vehicle_document(vform)
        digest, job = D.build_job(target, doc, {"timeout_s": 300.0, "refly": True},
                                  mission_form=self.mission_form(), vehicle_form=vform)
        cached = D.cached_run_dir(digest)
        if cached is not None:
            self.status.setText("cache hit — reopening the run for these exact inputs")
            self.run_ready.emit(str(cached))
            return
        self._job = job
        self._t0 = time.monotonic()
        self._timer.start()
        self.badge.set_state("loading", "simulating")
        self.status.setText("queuing on the compute worker…")
        self.simulate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._thread = _SimWorker(self._pool_instance(), job, self)
        self._thread.done.connect(self._on_done)
        self._thread.bad.connect(self._on_bad)
        self._thread.gone.connect(self._on_gone)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def cancel_simulate(self):
        if self._thread is not None:
            self.cancel_button.setEnabled(False)
            self.status.setText("cancelling…")
            self._thread.request_cancel()

    def _tick(self):
        if self._thread is not None:
            self.status.setText(f"simulating… {time.monotonic() - self._t0:,.0f} s"
                                " (OpenTsiolkovsky ascent in the compute worker)")

    def _on_done(self, result: dict):
        self.badge.set_state("verified", "verified" if result.get("physics_verified")
                             else "run complete")
        failed = result.get("failed_checks") or []
        self.status.setText("opening replay…" + (f" ({len(failed)} check(s) flagged)" if failed else ""))
        self.simulate_finished.emit(result)
        self.run_ready.emit(str(result["run_dir"]))

    def _on_bad(self, message: str):
        self.badge.set_state("failed", "simulate failed")
        self.status.setText(message.splitlines()[0][:140])
        self.simulate_failed.emit(message)

    def _on_gone(self):
        self.badge.set_state("idle", "cancelled")
        self.status.setText("cancelled — the worker and its simulator process group were killed")

    def _on_thread_finished(self):
        if self._job is not None:  # a worker killed before its own cleanup left a marker
            (Path(self._job["run_dir"]) / D.IN_PROGRESS).unlink(missing_ok=True)
        self._job = None
        self._timer.stop()
        self._thread = None
        self.cancel_button.setEnabled(False)
        self._validate()

    def terminate_compute(self):
        """App-quit teardown: stop any in-flight simulate and close the pool."""
        thread = self._thread
        if thread is not None:
            thread.request_cancel()
            thread.wait(3000)
        if self._pool is not None:
            self._pool.terminate()


class _SimWorker(QThread):
    """Runs the picklable job on the compute-pool worker (its own GIL and —
    thanks to :func:`orbitopt.flightlab.designer.run_form_job` — its own
    process group), or inline on a base install with pool disabled."""

    done = Signal(dict)
    bad = Signal(str)
    gone = Signal()

    def __init__(self, pool: ComputePool | None, job: dict, parent=None):
        super().__init__(parent)
        self._pool = pool
        self._job = job
        self._cancel = False

    def run(self):
        try:
            fn = functools.partial(D.run_form_job, self._job)
            self.done.emit(self._pool.run(fn) if self._pool is not None else fn())
        except Exception as exc:  # noqa: BLE001 -- reported verbatim on the status line
            if self._cancel:
                self.gone.emit()
            else:
                self.bad.emit(f"{type(exc).__name__}: {exc}")

    def request_cancel(self):
        if self._cancel:
            return
        self._cancel = True
        # Kill the worker's *process group* from the marker file first, so the
        # OpenTsiolkovsky grandchild dies too even when the pool escalates
        # SIGTERM to SIGKILL. Guarded hard: never signal a group we belong to.
        try:
            text = (Path(self._job["run_dir"]) / D.IN_PROGRESS).read_text(encoding="ascii")
            pgid = int(text)
            if os.name == "posix" and pgid > 1 and pgid not in (os.getpid(), os.getpgrp()):
                os.killpg(pgid, signal.SIGTERM)
        except (OSError, ValueError):
            pass
        if self._pool is not None and not self._pool.closed:
            self._pool.restart()  # kills the worker -> run() above raises
