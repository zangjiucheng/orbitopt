"""The VERIFICATION tab and badge: reports published into the scene and read
back faithfully -- "not checked" never shown as success."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("PySide6")

from orbitopt.verify import report as vr  # noqa: E402
from orbitopt.viz.state import MissionState  # noqa: E402
from orbitopt.viz.theme import DANGER, INK_MUTED, SUCCESS, WARNING  # noqa: E402
from orbitopt.viz.verification_panel import (  # noqa: E402
    VerificationPanel,
    conservation_text,
    display_subtitle,
    force_model_text,
    format_length_m,
    scene_verification_result,
    step_convergence_text,
)

# Shaped like the real Artemis II targeting report (trimmed).
REPORT = {
    "schemaVersion": "1.0",
    "kind": "lunar-free-return-entry-corridor",
    "label": "ref-et-838814469.183",
    "status": {"result": "passed", "passed": True, "checks": {
        "tli-flyby-entry.step_convergence": True,
        "tli-flyby-entry.two_body_conservation": True,
        "residual.entry_flight_path_angle_in_corridor": True,
        "targeting.converged": True,
    }},
    "inputs": {},
    "residuals": [{
        "name": "entry_flight_path_angle_in_corridor",
        "reference": "TudatPy-propagated inertial flight-path angle vs the entry corridor",
        "position_residual_m": None, "velocity_residual_m_s": None,
        "tolerance": {"corridor_deg": [-7.0, -5.0]}, "within_tolerance": True,
    }],
    "propagations": [{
        "label": "tli-flyby-entry",
        "final_state": [0, 0, 0, 0, 0, 0],
        "dynamics": {
            "central_body": "Earth", "frame": "ECLIPJ2000", "forces": [],
            "summary": {"central_point_mass_only": False, "central_J2": False, "central_J22": False,
                        "earth_J2": False, "earth_J22": False, "sun_gravity": True, "moon_gravity": True,
                        "third_bodies": ["Moon", "Sun"]},
            "not_modelled": ["solar_radiation_pressure", "atmospheric_drag"],
        },
        "integrator": {
            "name": "dop853", "method": "Runge-Kutta-Dormand-Prince 8(7)", "adaptive": True,
            "rtol": 1e-11, "atol": 1e-6, "max_step_s": None,
            "steps": {"n_steps": 1177, "min_s": 1.0, "max_s": 900.0, "mean_s": 300.0, "median_s": 280.0,
                      "n_segments": 1},
            "convergence": {"checked": True, "converged": True, "method": "re-run with rtol/atol x0.01",
                            "delta_position_m": 0.016, "position_tolerance_m": 10.0},
        },
        "conservation": {"tolerance_rel": 1e-8, "propagated_model": {
            "mu_m3_s2": 3.986e14, "energy_rel_drift_max": 0.045, "angular_momentum_rel_drift_max": 3.29,
            "segments": [], "applicable": False,
            "reason": "perturbed dynamics do not conserve two-body energy"}},
    }],
    "targets": None,
    "provenance": {"python": "3.10", "numpy": "1.26", "tudatpy": "1.0.0"},
}
REPORT["digest"] = vr.report_digest(REPORT)


def _mission_scene(**extra):
    scene = {"id": "m", "title": "M", "distanceUnit": "km", "bodies": [],
             "timeline": {"min": 0.0, "max": 1.0, "unitLabel": "days", "events": []},
             "subtitle": "Line one\nVerification report x:y: PASSED (4/4 checks passed)"}
    scene.update(extra)
    return scene


def test_publish_combines_reports_tri_state():
    scene = vr.publish_to_scene(_mission_scene(), [REPORT])
    assert scene["verification"]["result"] == "passed"
    failed = copy.deepcopy(REPORT)
    failed["status"] = vr.status_from_checks({"a": True, "b": False})
    not_checked = copy.deepcopy(REPORT)
    not_checked["status"] = vr.status_from_checks({})
    assert vr.publish_to_scene(_mission_scene(), [REPORT, failed])["verification"]["result"] == "failed"
    assert vr.publish_to_scene(_mission_scene(), [REPORT, not_checked])["verification"]["result"] == "not_checked"
    assert "verification" not in vr.publish_to_scene(_mission_scene(), [])


def test_result_for_scene_never_upgrades_missing_reports():
    assert scene_verification_result({"bodies": []}) is None  # snapshot: nothing to verify
    assert scene_verification_result(_mission_scene()) == "not_checked"
    assert scene_verification_result(vr.publish_to_scene(_mission_scene(), [REPORT])) == "passed"


def test_subtitle_drops_report_lines_only_when_reports_are_shown():
    assert display_subtitle(_mission_scene()) == "Line one  Verification report x:y: PASSED (4/4 checks passed)"
    assert display_subtitle(vr.publish_to_scene(_mission_scene(), [REPORT])) == "Line one"


def test_formatting_states_what_the_report_says():
    prop = REPORT["propagations"][0]
    assert force_model_text(prop["dynamics"]) == "Earth point mass + Moon, Sun (ECLIPJ2000)"
    text, color = step_convergence_text(prop["integrator"]["convergence"])
    assert text.startswith("Δr 16 mm (tol 10.00 m)") and color == SUCCESS
    text, color = conservation_text(prop["conservation"])
    assert text.startswith("not applicable") and "max relative drift: energy 4.50e-02" in text
    assert color == INK_MUTED
    assert step_convergence_text({"checked": False, "reason": "fixed step"}) == ("not checked — fixed step", WARNING)
    bad = {"tolerance_rel": 1e-8, "propagated_model": {"applicable": True, "within_tolerance": False,
                                                       "energy_rel_drift_max": 1e-6,
                                                       "angular_momentum_rel_drift_max": 1e-7}}
    assert conservation_text(bad)[1] == DANGER
    assert format_length_m(1234.5) == "1.234 km" and format_length_m(None) == "—"


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_panel_shows_checks_residuals_and_model(app):
    state = MissionState()
    panel = VerificationPanel(state)
    state.set_scene(vr.publish_to_scene(_mission_scene(), [REPORT]))
    text = panel.text()
    assert "PASSED — 1 of 1 report(s) passed" in text
    assert "✓  targeting.converged" in text
    assert "corridor deg -7 to -5" in text
    assert "Earth point mass + Moon, Sun (ECLIPJ2000)" in text
    assert "solar_radiation_pressure, atmospheric_drag" in text
    assert "tudatpy 1.0.0" in text


def test_panel_says_not_checked_without_reports(app):
    state = MissionState()
    panel = VerificationPanel(state)
    state.set_scene(_mission_scene())
    assert "NOT CHECKED — no high-fidelity verification report" in panel.text()
    state.set_scene({"id": "s", "title": "S", "distanceUnit": "AU", "bodies": []})
    text = panel.text()
    assert "ephemeris snapshot" in text and "NOT CHECKED" not in text  # old content is gone at once
