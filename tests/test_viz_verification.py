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
    selection_badge_state,
    step_convergence_text,
    verification_badge_state,
    verification_scope,
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
    }, "check_categories": {
        "tli-flyby-entry.step_convergence": "numerical_consistency",
        "tli-flyby-entry.two_body_conservation": "numerical_consistency",
        "residual.entry_flight_path_angle_in_corridor": "model_agreement",
        "targeting.converged": "numerical_consistency",
    }},
    "inputs": {},
    "residuals": [{
        "name": "entry_flight_path_angle_in_corridor",
        "reference": "TudatPy-propagated inertial flight-path angle vs the entry corridor",
        "category": "model_agreement",
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
    failed["status"] = vr.status_from_checks({"a": True, "b": False},
                                                  {"a": "model_agreement", "b": "model_agreement"})
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


def test_panel_separates_the_three_check_categories(app):
    """Task #243: checks are grouped under their category header, and the
    residual says which kind of evidence it is."""
    state = MissionState()
    panel = VerificationPanel(state)
    state.set_scene(vr.publish_to_scene(_mission_scene(), [REPORT]))
    text = panel.text()
    assert "NUMERICAL CONSISTENCY (3/3)" in text
    assert "MODEL AGREEMENT (1/1)" in text
    assert "HISTORICAL VALIDATION" not in text  # a category with no checks is not shown
    assert "RESIDUAL · MODEL AGREEMENT" in text
    # Checks without categories (documents written before categorisation) still
    # render under the plain heading.
    unclassified = copy.deepcopy(REPORT)
    unclassified["status"] = {"result": "passed", "passed": True, "checks": dict(REPORT["status"]["checks"])}
    unclassified["digest"] = vr.report_digest(unclassified)
    state.set_scene(vr.publish_to_scene(_mission_scene(), [unclassified]))
    text = panel.text()
    assert "NUMERICAL CONSISTENCY" not in text and "✓  targeting.converged" in text
    lines = [ln.strip() for ln in text.splitlines()]
    assert "CHECKS" in lines  # unclassified checks fall back to the plain list


def test_panel_says_not_checked_without_reports(app):
    state = MissionState()
    panel = VerificationPanel(state)
    state.set_scene(_mission_scene())
    assert "NOT CHECKED — no high-fidelity verification report" in panel.text()
    state.set_scene({"id": "s", "title": "S", "distanceUnit": "AU", "bodies": []})
    text = panel.text()
    assert "ephemeris snapshot" in text and "NOT CHECKED" not in text  # old content is gone at once


# ------------------------------------------- candidates vs the exported trajectory
def _candidates_scene():
    def cand(index, exported, verdict, verification):
        return {"id": f"candidate-{index}", "label": f"Candidate #{index}", "searchIndex": index,
                "exported": exported, "verdict": verdict, "verification": verification}
    return vr.publish_to_scene(_mission_scene(candidates=[
        cand(1, False, "rejected", None), cand(0, True, "accepted", "passed")]), [REPORT])


def test_badges_scope_verification_to_the_exported_trajectory():
    scene = _candidates_scene()
    exported, other = scene["candidates"][1], scene["candidates"][0]
    state, text, tooltip = verification_badge_state(scene)
    assert (state, text) == ("verified", "EXPORTED VERIFIED")
    assert "Candidate #0" in tooltip and "TudatPy N-body" in tooltip and "Two-body Lambert" in tooltip
    assert selection_badge_state(scene, exported) is None
    state, text, _ = selection_badge_state(scene, other)
    assert state == "warning" and text == "Candidate #1 · SCREENING ONLY" and "VERIFIED" not in text
    # No candidates: the one trajectory is the one verified, badge unchanged.
    plain = vr.publish_to_scene(_mission_scene(), [REPORT])
    assert verification_badge_state(plain)[:2] == ("verified", "VERIFIED")
    assert selection_badge_state(plain, None) is None and verification_scope(plain, None) == []
    assert verification_badge_state({"bodies": []}) is None


def test_scope_rows_name_both_trajectories_and_fidelities():
    scene = _candidates_scene()
    rows = {k: (v, c) for k, v, c in verification_scope(scene, scene["candidates"][0])}
    assert rows["Exported Trajectory"][0].startswith("Candidate #0")
    assert rows["Selected Candidate"] == ("Candidate #1 — screening trajectory, NOT verified in high fidelity",
                                          WARNING)
    assert rows["Screening Fidelity"][0] == "Two-body Lambert"
    assert rows["Flown Fidelity"][0] == "TudatPy N-body"
    rows = {k: v for k, v, _c in verification_scope(scene, scene["candidates"][1])}
    assert rows["Selected Candidate"] == "Candidate #0 — the exported trajectory"


def test_panel_follows_the_candidate_selection(app):
    state = MissionState()
    panel = VerificationPanel(state)
    state.set_scene(_candidates_scene())
    text = panel.text()
    assert "VERIFICATION · EXPORTED TRAJECTORY" in text
    assert "Candidate #0 — the exported trajectory" in text and "NOT verified" not in text
    state.select_candidate("candidate-1")
    text = panel.text()
    assert "Candidate #1 — screening trajectory, NOT verified in high fidelity" in text
    assert "Exported Trajectory\nCandidate #0" in text  # the reports stay the exported one's
    assert "PASSED — 1 of 1 report(s) passed" in text


def test_goes_default_load_is_flagged_unverified_with_a_verified_variant():
    import pickle

    from orbitopt import missions

    assert missions.builtin_mission_config("goes")["verification"] == {"enabled": False}
    note = missions.verification_off_note("goes")
    assert note and "does NOT mean" in note and "TudatPy" in note
    loader = missions.high_fidelity_loader("goes")
    assert loader is not None and pickle.loads(pickle.dumps(loader)) is loader  # runs in the worker process
    # Missions whose default load already verifies get no offer.
    for mission_id in ("artemis2", "mars", "solar-system", "nope"):
        assert missions.verification_off_note(mission_id) is None
        assert missions.high_fidelity_loader(mission_id) is None


def test_panel_offers_one_click_verification_only_while_unverified(app):
    state = MissionState()
    panel = VerificationPanel(state)
    requested = []
    panel.verify_requested.connect(lambda: requested.append(True))

    state.set_scene(_mission_scene())
    assert panel.verify_button is None  # no offer set: the plain NOT CHECKED note only

    panel.set_verify_offer("Verification is OFF by default for GOES.")
    state.set_scene(_mission_scene())
    text = panel.text()
    assert "NOT CHECKED" in text and "NOT VERIFIED BY DEFAULT" in text
    assert "Verification is OFF by default for GOES." in text
    assert panel.verify_button is not None
    panel.verify_button.click()
    assert requested == [True]

    # Once the verified scene arrives its reports are shown, not the offer.
    state.set_scene(vr.publish_to_scene(_mission_scene(), [REPORT]))
    assert panel.verify_button is None
    assert "PASSED — 1 of 1 report(s) passed" in panel.text()
