"""The VERIFICATION tab: what the scene's high-fidelity verification actually
checked, read straight from its verification reports (the scene's
``verification`` field, orbitopt's verification-report-1.0 documents).

For each report: its tri-state result and every check, the residuals against
their references and tolerances, and per propagation the force model (and
what it leaves out), the integrator and its step-convergence test, and the
energy / angular-momentum conservation diagnostic. Nothing is summarized
into a friendlier claim than the report makes -- "not checked" stays "not
checked", and a conservation test that does not apply to perturbed dynamics
says so instead of passing or failing.

A scene with a ``handoffs`` block (M-06: a mission that crossed segment
boundaries, e.g. the Flight Lab ascent feeding a parking orbit) also gets a
MISSION HANDOFFS section: one card per boundary with its pass/fail/not-checked
badge, the state handed over, and the checks behind the verdict -- rendered
straight from the scene, inferred by nothing.

Viewer-only: reads JSON, imports no compute code.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from orbitopt.viz.candidate_panel import FLOWN_FIDELITY_LABEL, SCREENING_FIDELITY_LABEL, exported_candidate
from orbitopt.viz.state import MissionState
from orbitopt.viz.theme import DANGER, INK_MUTED, SUCCESS, WARNING

RESULT_TEXT = {"passed": "PASSED", "failed": "FAILED", "not_checked": "NOT CHECKED"}
RESULT_COLOR = {"passed": SUCCESS, "failed": DANGER, "not_checked": WARNING}
# StatusBadge state + text for the title bar.
RESULT_BADGE = {"passed": ("verified", "VERIFIED"), "failed": ("failed", "VERIFY FAILED"),
                "not_checked": ("warning", "NOT CHECKED")}


def scene_verification_result(scene: dict | None) -> str | None:
    """Tri-state verification result for a scene, or None when there is no
    trajectory to verify (no timeline, e.g. an ephemeris snapshot). A mission
    scene without a ``verification`` field is "not_checked", never passed."""
    if not scene or not scene.get("timeline"):
        return None
    verification = scene.get("verification")
    if not verification:
        return "not_checked"
    return verification.get("result", "not_checked")


def verification_badge_state(scene: dict | None) -> tuple[str, str, str] | None:
    """``(state, text, tooltip)`` for the title-bar verification badge, or
    None when the scene has nothing to verify. The reports belong to the
    exported trajectory only, so in a scene with candidates the badge says
    so ("EXPORTED VERIFIED") whichever candidate is selected."""
    result = scene_verification_result(scene)
    if result is None:
        return None
    state, text = RESULT_BADGE[result]
    candidates = (scene or {}).get("candidates") or []
    if not candidates:
        return state, text, "High-fidelity verification of this trajectory -- click for the reports"
    exported = exported_candidate(candidates)
    name = exported["label"] if exported else "the mission trajectory"
    return state, f"EXPORTED {text}", (
        f"High-fidelity ({FLOWN_FIDELITY_LABEL}) verification of the exported trajectory ({name}) only. "
        f"Other candidates are screening arcs ({SCREENING_FIDELITY_LABEL}) and were not verified "
        "-- click for the reports")


def selection_badge_state(scene: dict | None, selected: dict | None) -> tuple[str, str, str] | None:
    """``(state, text, tooltip)`` for the title-bar badge naming a selected
    candidate that is *not* the exported trajectory, or None (no candidates,
    or the exported one is selected and the verification badge covers it)."""
    if not (scene or {}).get("candidates") or selected is None or selected.get("exported"):
        return None
    return "warning", f"{selected['label']} · SCREENING ONLY", (
        f"The selected candidate is a {SCREENING_FIDELITY_LABEL} screening arc, not flown or verified "
        "in high fidelity. The verification badge and reports are the exported trajectory's.")


def verification_scope(scene: dict | None, selected: dict | None) -> list[tuple[str, str, str | None]]:
    """``(key, value, color)`` rows saying which trajectory the verification
    reports describe and what the selected candidate is. Empty for a scene
    without candidates (its one trajectory is the one verified)."""
    candidates = (scene or {}).get("candidates") or []
    if not candidates:
        return []
    exported = exported_candidate(candidates)
    rows = [
        ("Exported Trajectory",
         f"{exported['label']} — the mission trajectory, propagated in high fidelity; "
         "the verification below is for it alone" if exported
         else "none of the candidates — the verification below is for the mission trajectory alone", None),
        ("Flown Fidelity", FLOWN_FIDELITY_LABEL, None),
    ]
    if selected is not None and not selected.get("exported"):
        rows.append(("Selected Candidate",
                     f"{selected['label']} — screening trajectory, NOT verified in high fidelity", WARNING))
    elif selected is not None:
        rows.append(("Selected Candidate", f"{selected['label']} — the exported trajectory", None))
    rows.append(("Screening Fidelity", SCREENING_FIDELITY_LABEL, None))
    return rows


def handoffs_of(scene: dict | None) -> list:
    """The scene's ``handoffs`` block (M-06 segment-boundary verdicts, e.g.
    from orbitopt.flightlab.scene_export.run_to_scene) -- possibly empty."""
    return (scene or {}).get("handoffs") or []


def display_subtitle(scene: dict) -> str:
    """The scene subtitle for the title bar. When the scene carries its
    reports, the one-line "Verification report ...: PASSED (...)" summaries
    appended for the CLI are dropped here: the badge and this tab show the
    same thing in full, and they had grown the Mars title bar to 9 lines."""
    lines = (scene.get("subtitle") or "").split("\n")
    if scene.get("verification"):
        lines = [ln for ln in lines if not ln.startswith("Verification report ")]
    return "  ".join(ln for ln in lines if ln)


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def format_length_m(meters) -> str:
    if meters is None:
        return "—"
    m = abs(float(meters))
    if m >= 1000.0:
        return f"{meters / 1000.0:,.3f} km"
    if m >= 1.0:
        return f"{meters:,.2f} m"
    if m == 0.0:
        return "0 m"
    return f"{meters * 1000.0:.3g} mm"


def format_ratio(x) -> str:
    return "—" if x is None else f"{float(x):.2e}"


def _format_tolerance(tol) -> str:
    if not tol:
        return ""
    parts = []
    for key, value in tol.items():
        if isinstance(value, (list, tuple)):
            value = " to ".join(f"{v:g}" for v in value)
        elif isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{_humanize(key)} {value}")
    return "; ".join(parts)


def force_model_text(dynamics: dict) -> str:
    """e.g. "Earth point mass + Moon, Sun (ECLIPJ2000)"."""
    summary = dynamics.get("summary", {})
    central = dynamics.get("central_body", "?")
    harmonics = [t for t, on in (("J2", summary.get("central_J2") or summary.get("earth_J2")),
                                 ("J22", summary.get("central_J22") or summary.get("earth_J22"))) if on]
    text = f"{central} " + (" + ".join(["point mass"] + harmonics) if harmonics else "point mass")
    third = summary.get("third_bodies") or []
    if third:
        text += " + " + ", ".join(third)
    frame = dynamics.get("frame")
    return f"{text} ({frame})" if frame else text


def step_convergence_text(conv: dict) -> tuple[str, str]:
    """(text, color) for an integrator's step-convergence test."""
    if not conv.get("checked"):
        return f"not checked — {conv.get('reason', 'no reason given')}", WARNING
    delta, tol = conv.get("delta_position_m"), conv.get("position_tolerance_m")
    ok = conv.get("converged")
    text = f"Δr {format_length_m(delta)}" + (f" (tol {format_length_m(tol)})" if tol is not None else "")
    if conv.get("method"):
        text += f" · {conv['method']}"
    return text, SUCCESS if ok else DANGER


def conservation_text(conservation: dict) -> tuple[str, str]:
    """(text, color) for the propagated model's conservation diagnostic."""
    block = conservation.get("propagated_model") or {}
    tol = conservation.get("tolerance_rel")
    drift = (f"energy {format_ratio(block.get('energy_rel_drift_max'))}, "
             f"|h| {format_ratio(block.get('angular_momentum_rel_drift_max'))}")
    if not block.get("applicable", False):
        # The report's reason already explains why (and that the numbers are
        # only a diagnostic); just append the numbers.
        return f"not applicable — {block.get('reason', '')} (max relative drift: {drift})", INK_MUTED
    ok = block.get("within_tolerance")
    return f"max relative drift {drift} (tol {format_ratio(tol)})", SUCCESS if ok else DANGER


def _label(text: str, name: str = "bodyStatValue", color: str | None = None, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    if color is not None:
        label.setStyleSheet(f"color: {color};")
    return label


def _row(layout: QVBoxLayout, key: str, value: str, color: str | None = None) -> None:
    """A key over its value: values here are long (method names, reasons), so
    stacking reads better than squeezing both onto one line."""
    layout.addWidget(_label(key, "bodyStat", wrap=False))
    layout.addWidget(_label(value, "maneuverNote", color))


VERIFY_BUTTON_TEXT = "Run high-fidelity verification (TudatPy)"


class VerificationPanel(QWidget):
    """Rebuilt from MissionState.scene_changed.

    For a mission whose default load skips verification (e.g. GOES) the host
    calls :meth:`set_verify_offer` with the reason before applying the scene;
    the panel then explains it and shows a button that emits
    ``verify_requested`` so the host can re-run the mission with high-fidelity
    verification on."""

    verify_requested = Signal()

    def __init__(self, state: MissionState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self.setObjectName("infoPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setObjectName("infoScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        host.setObjectName("infoCards")
        self._layout = QVBoxLayout(host)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, stretch=1)

        self._scene: dict | None = None
        self._verify_offer: str | None = None
        self.verify_button: QPushButton | None = None
        state.scene_changed.connect(self._rebuild)
        state.candidate_changed.connect(lambda _cid: self._rebuild(self._scene))

    def text(self) -> str:
        """All label text in the panel (for tests)."""
        return "\n".join(w.text() for w in self.findChildren(QLabel) if w.parent() is not None)

    def set_verify_offer(self, note: str | None) -> None:
        """Why the next scene(s) carry no verification report and that it can
        be switched on (None: no offer). Shown only while the scene has no
        reports; takes effect at the next rebuild."""
        self._verify_offer = note

    # ------------------------------------------------------------ build
    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # off the panel now, not at the next event-loop turn
                widget.deleteLater()

    def _rebuild(self, scene):
        self._scene = scene
        self.verify_button = None
        self._clear()
        result = scene_verification_result(scene)
        scope = verification_scope(scene, self._state.candidate())
        header = QLabel("VERIFICATION · EXPORTED TRAJECTORY" if scope else "VERIFICATION")
        header.setObjectName("sectionHeader")
        self._layout.addWidget(header)
        if scope:
            self._layout.addWidget(self._scope_card(scope))

        if result is None:
            self._layout.addWidget(self._note_card(
                "Nothing to verify: this scene is an ephemeris snapshot, not a computed trajectory."))
        elif not (scene or {}).get("verification"):
            self._layout.addWidget(self._note_card(
                "NOT CHECKED — no high-fidelity verification report came with this trajectory "
                "(verification was off or produced nothing). Do not treat it as verified.",
                WARNING))
            if self._verify_offer:
                self._layout.addWidget(self._verify_offer_card(self._verify_offer))
        else:
            reports = scene["verification"].get("reports", [])
            n_passed = sum(1 for r in reports if (r.get("status") or {}).get("result") == "passed")
            self._layout.addWidget(self._note_card(
                f"{RESULT_TEXT.get(result, result)} — {n_passed} of {len(reports)} report(s) passed. "
                "Passed means every check in a report ran and held; not checked is never success.",
                RESULT_COLOR.get(result)))
            for report in reports:
                self._layout.addWidget(self._report_card(report))

        handoffs = handoffs_of(scene)
        if handoffs:
            self._layout.addWidget(self._handoffs_section(handoffs))
        self._layout.addStretch(1)

    # ----------------------------------------------------- handoffs (M-06)
    def _handoffs_section(self, handoffs: list) -> None:
        """Header + one card per segment boundary the mission crossed."""
        header = QLabel("MISSION HANDOFFS")
        header.setObjectName("sectionHeader")
        self._layout.addWidget(header)
        self._layout.addWidget(self._note_card(
            "State continuity and mission checks at each segment boundary. "
            "PASSED means every check at that boundary ran and held; NOT CHECKED is never success."))
        for handoff in handoffs:
            self._layout.addWidget(self._handoff_card(handoff))

    def _handoff_card(self, handoff: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("bodyCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(3)

        status = handoff.get("status", "not_checked")
        title_row = QHBoxLayout()
        title_row.addWidget(_label(handoff.get("label", "Handoff"), "bodyName"), stretch=1)
        badge = QLabel(RESULT_TEXT.get(status, str(status).upper()))
        badge.setStyleSheet(f"color: {RESULT_COLOR.get(status, INK_MUTED)}; font-size: 10px; font-weight: 600;")
        title_row.addWidget(badge, alignment=Qt.AlignTop)
        v.addLayout(title_row)
        met = f"T+{float(handoff.get('time', 0.0)):,.0f} s"
        if handoff.get("detail"):
            met += f" \u00b7 {handoff['detail']}"
        v.addWidget(_label(met, "maneuverNote"))

        for key, row_key in (("from", "Handed over"), ("to", "Received"), ("state", "Handoff state")):
            state = handoff.get(key)
            if not isinstance(state, dict):
                continue
            r_m = state.get("position_m") or [0.0, 0.0, 0.0]
            v_m_s = state.get("velocity_m_s") or [0.0, 0.0, 0.0]
            summary = (f"|r| {math.dist([0.0] * 3, r_m) / 1000.0:,.1f} km"
                       f" \u00b7 |v| {math.dist([0.0] * 3, v_m_s):,.1f} m/s"
                       f" \u00b7 mass {float(state.get('mass_kg', 0.0)):,.0f} kg"
                       f" \u00b7 {state.get('frame', '?')} @ {state.get('epoch_utc', '?')}")
            _row(v, f"{row_key} \u00b7 {state.get('body', '?')}", summary)

        checks = handoff.get("checks") or []
        if checks:
            v.addWidget(_label("CHECKS", "sectionHeader", wrap=False))
            for check in checks:
                ok = check.get("passed")
                mark, color = ("✓", SUCCESS) if ok else ("✗", DANGER)
                v.addWidget(_label(f"{mark}  {_humanize(check.get('id', ''))}", "maneuverNote", color))
                bits = []
                if check.get("value") is not None:
                    bits.append(f"{float(check['value']):,.3g} {check.get('unit', '')}".rstrip())
                if check.get("limit") is not None:
                    bits.append(f"limit {float(check['limit']):,.3g}")
                if check.get("reason_code"):
                    bits.append(str(check["reason_code"]))
                if bits:
                    v.addWidget(_label("    " + " \u00b7 ".join(bits), "maneuverNote"))
        else:
            v.addWidget(_label("No checks ran at this boundary.", "maneuverNote", WARNING))
        return card

    def _scope_card(self, rows) -> QFrame:
        card = QFrame()
        card.setObjectName("bodyCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(3)
        for key, value, color in rows:
            _row(v, key, value, color)
        return card

    def _verify_offer_card(self, note: str) -> QFrame:
        card = QFrame()
        card.setObjectName("bodyCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        v.addWidget(_label("NOT VERIFIED BY DEFAULT", "sectionHeader", wrap=False))
        v.addWidget(_label(note, "maneuverNote", WARNING))
        button = QPushButton(VERIFY_BUTTON_TEXT)
        button.setToolTip("Re-run this mission with the TudatPy high-fidelity re-fly enabled "
                          "(about a minute of extra compute the first time, then cached) "
                          "and show its verification report here.")
        button.clicked.connect(self.verify_requested.emit)
        v.addWidget(button)
        self.verify_button = button
        return card

    def _note_card(self, text: str, color: str | None = None) -> QFrame:
        card = QFrame()
        card.setObjectName("bodyCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.addWidget(_label(text, "maneuverNote", color))
        return card

    def _report_card(self, report: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("bodyCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(3)

        status = report.get("status") or {}
        result = status.get("result", "not_checked")
        title_row = QHBoxLayout()
        title = _label(_humanize(report.get("kind", "report")).capitalize(), "bodyName")
        title_row.addWidget(title, stretch=1)
        badge = QLabel(RESULT_TEXT.get(result, result.upper()))
        badge.setStyleSheet(f"color: {RESULT_COLOR.get(result, INK_MUTED)}; font-size: 10px; font-weight: 600;")
        title_row.addWidget(badge, alignment=Qt.AlignTop)
        v.addLayout(title_row)
        v.addWidget(_label(report.get("label", ""), "maneuverNote"))

        checks = status.get("checks") or {}
        metrics = (report.get("targets") or {}).get("metrics") if isinstance(report.get("targets"), dict) else None
        if metrics:
            # Reports that carry their numbers (Starship's): each check with
            # what was measured and the limit it was held to.
            v.addWidget(_label("CHECKS", "sectionHeader", wrap=False))
            for m in metrics:
                ok = m.get("ok")
                mark, color = ("✓", SUCCESS) if ok else (("✗", DANGER) if ok is False else ("—", INK_MUTED))
                v.addWidget(_label(f"{mark}  {m.get('name', '')}", "maneuverNote", color))
                detail = " · ".join(x for x in (m.get("value", ""), f"limit {m['limit']}" if m.get("limit") else "") if x)
                if detail:
                    v.addWidget(_label(f"    {detail}", "maneuverNote"))
        elif checks:
            v.addWidget(_label("CHECKS", "sectionHeader", wrap=False))
            for name, ok in checks.items():
                v.addWidget(_label(f"{'✓' if ok else '✗'}  {_humanize(name)}", "maneuverNote",
                                   SUCCESS if ok else DANGER))
        else:
            v.addWidget(_label("No checks ran.", "maneuverNote", WARNING))

        for res in report.get("residuals", []):
            v.addWidget(_label("RESIDUAL", "sectionHeader", wrap=False))
            within = res.get("within_tolerance")
            mark, color = ("✓", SUCCESS) if within else (("✗", DANGER) if within is False else ("—", INK_MUTED))
            v.addWidget(_label(f"{mark}  {_humanize(res.get('name', ''))}", "maneuverNote", color))
            v.addWidget(_label(f"vs {res.get('reference', '')}", "maneuverNote"))
            values = []
            if res.get("position_residual_m") is not None:
                values.append(f"|Δr| {format_length_m(res['position_residual_m'])}")
            if res.get("velocity_residual_m_s") is not None:
                values.append(f"|Δv| {float(res['velocity_residual_m_s']):.3g} m/s")
            tol = _format_tolerance(res.get("tolerance"))
            if values or tol:
                v.addWidget(_label(" · ".join(values + ([f"tolerance: {tol}"] if tol else [])), "maneuverNote"))

        for prop in report.get("propagations", []):
            v.addWidget(_label(f"PROPAGATION · {prop.get('label', '')}", "sectionHeader"))
            dynamics = prop.get("dynamics") or {}
            _row(v, "Force model", force_model_text(dynamics))
            not_modelled = dynamics.get("not_modelled") or []
            if not_modelled:
                _row(v, "Not modelled", ", ".join(not_modelled))
            integ = prop.get("integrator") or {}
            steps = (integ.get("steps") or {}).get("n_steps")
            tol_text = (f"rtol {integ['rtol']:g}, atol {integ['atol']:g}"
                        if integ.get("rtol") is not None and integ.get("atol") is not None else "")
            _row(v, "Integrator", ", ".join(x for x in (integ.get("method", ""), tol_text,
                                                       f"{steps} steps" if steps is not None else "") if x))
            text, color = step_convergence_text(integ.get("convergence") or {})
            _row(v, "Step convergence", text, color)
            text, color = conservation_text(prop.get("conservation") or {})
            _row(v, "Conservation", text, color)

        provenance = report.get("provenance") or {}
        versions = ", ".join(f"{k} {provenance[k]}" for k in ("tudatpy", "orbitopt") if provenance.get(k))
        digest = report.get("digest", "")
        footer = " · ".join(x for x in (versions, f"digest {digest[:12]}…" if digest else "") if x)
        if footer:
            v.addWidget(_label(footer, "maneuverNote", INK_MUTED))
        return card
