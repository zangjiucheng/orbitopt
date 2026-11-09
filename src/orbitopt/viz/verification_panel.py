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

Viewer-only: reads JSON, imports no compute code.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

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


class VerificationPanel(QWidget):
    """Rebuilt from MissionState.scene_changed."""

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

        state.scene_changed.connect(self._rebuild)

    def text(self) -> str:
        """All label text in the panel (for tests)."""
        return "\n".join(w.text() for w in self.findChildren(QLabel) if w.parent() is not None)

    # ------------------------------------------------------------ build
    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # off the panel now, not at the next event-loop turn
                widget.deleteLater()

    def _rebuild(self, scene):
        self._clear()
        result = scene_verification_result(scene)
        header = QLabel("VERIFICATION")
        header.setObjectName("sectionHeader")
        self._layout.addWidget(header)

        if result is None:
            self._layout.addWidget(self._note_card(
                "Nothing to verify: this scene is an ephemeris snapshot, not a computed trajectory."))
        elif not (scene or {}).get("verification"):
            self._layout.addWidget(self._note_card(
                "NOT CHECKED — no high-fidelity verification report came with this trajectory "
                "(verification was off or produced nothing). Do not treat it as verified.",
                WARNING))
        else:
            reports = scene["verification"].get("reports", [])
            n_passed = sum(1 for r in reports if (r.get("status") or {}).get("result") == "passed")
            self._layout.addWidget(self._note_card(
                f"{RESULT_TEXT.get(result, result)} — {n_passed} of {len(reports)} report(s) passed. "
                "Passed means every check in a report ran and held; not checked is never success.",
                RESULT_COLOR.get(result)))
            for report in reports:
                self._layout.addWidget(self._report_card(report))
        self._layout.addStretch(1)

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
        if checks:
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
