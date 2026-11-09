"""Optimizer candidates in the scene and in the viewer: the exporter helper
(screening arcs + gate bookkeeping), state, renderer highlighting, and the
candidate panel / inspector card in a real window."""
from __future__ import annotations

import numpy as np
import pytest

pk = pytest.importorskip("pykep")
pv = pytest.importorskip("pyvista")
pytest.importorskip("PySide6")

from orbitopt.bodies import mjd2000_to_ephemeris_seconds, planet  # noqa: E402
from orbitopt.scene_format import validate_scene  # noqa: E402
from orbitopt.viz.candidate_panel import candidate_status, candidate_summary  # noqa: E402
from orbitopt.viz.candidates import SCREENING_FIDELITY, build_candidate_entries, lambert_arc_km  # noqa: E402
from orbitopt.viz.scene_renderer import CANDIDATE_SELECTED_COLOR, SceneRenderer  # noqa: E402
from orbitopt.viz.state import MissionState  # noqa: E402

T0 = 9800.0  # MJD2000, late 2026
TOF = 250.0


def test_lambert_arc_connects_earth_at_departure_to_mars_at_arrival():
    arc = lambert_arc_km(T0, TOF, n_points=50)
    earth_r = np.asarray(planet("earth").eph(pk.epoch(T0))[0]) / 1000.0
    mars_r = np.asarray(planet("mars").eph(pk.epoch(T0 + TOF))[0]) / 1000.0
    assert arc.shape == (50, 3)
    assert np.linalg.norm(arc[0] - earth_r) < 1e-3
    assert np.linalg.norm(arc[-1] - mars_r) < 50.0  # km, after a 250-day Kepler propagation
    radii = np.linalg.norm(arc, axis=1)
    assert radii.min() > 1.3e8 and radii.max() < 2.6e8  # between ~Earth and ~Mars orbits


def _records():
    # The select stage's records: three distinct candidates, one duplicate.
    return [
        {"index": 0, "status": "selected", "rank": 1, "reason": "rank 1 by cost",
         "t0_mjd2000": T0 + 20, "tof_days": 230.0, "lambert_total_dv_m_s": 5800.0},
        {"index": 1, "status": "selected", "rank": 0, "reason": "rank 0 by cost",
         "t0_mjd2000": T0, "tof_days": TOF, "lambert_total_dv_m_s": 5600.0},
        {"index": 2, "status": "duplicate", "rank": None, "reason": "duplicate of candidate #1",
         "t0_mjd2000": T0 + 0.5, "tof_days": TOF, "lambert_total_dv_m_s": 5601.0},
        {"index": 3, "status": "truncated", "rank": 2, "reason": "beyond max_candidates",
         "t0_mjd2000": T0 - 30, "tof_days": 280.0, "lambert_total_dv_m_s": 6100.0},
    ]


def _gate():
    return [
        {"index": 1, "verdict": "rejected", "verification": None, "reasons": ["refine: TCM did not converge"]},
        {"index": 0, "verdict": "accepted", "verification": "passed", "reasons": []},
    ]


def _fake_arc(t0, tof):
    return np.column_stack([np.linspace(1.5e8, 2.2e8, 20), np.full(20, t0), np.full(20, tof)])


def test_entries_rank_order_fates_and_timeline_times():
    ref_et = mjd2000_to_ephemeris_seconds(T0 + 20)  # the exported one departs at timeline 0
    entries = build_candidate_entries(_records(), _gate(), exported_index=0, reference_et=ref_et,
                                      flown_body_id="spacecraft", arc_fn=_fake_arc)
    assert [e["searchIndex"] for e in entries] == [1, 0, 3]  # by rank; duplicate dropped
    best, flown, extra = entries
    assert best["verdict"] == "rejected" and best["verdictReasons"] == ["refine: TCM did not converge"]
    assert flown["exported"] and flown["flownBodyId"] == "spacecraft" and flown["verification"] == "passed"
    assert extra["verdict"] is None and extra["selection"] == "truncated"  # never reached the gate
    assert "flownBodyId" not in best
    assert flown["screeningPath"]["fidelity"] == SCREENING_FIDELITY
    assert flown["screeningPath"]["times"][0] == pytest.approx(0.0, abs=1e-6)
    assert flown["screeningPath"]["times"][-1] == pytest.approx(230.0, abs=1e-6)
    assert best["screeningPath"]["times"][0] == pytest.approx(-20.0, abs=1e-6)  # departs 20 days earlier


def test_pinned_transfer_falls_back_to_gate_candidates():
    gate = [{"index": 0, "t0_mjd2000": T0, "tof_days": TOF, "verdict": "accepted", "verification": "passed",
             "reasons": [], "source": "pinned launch.departure"}]
    entries = build_candidate_entries([], gate, 0, mjd2000_to_ephemeris_seconds(T0), "spacecraft",
                                      arc_fn=_fake_arc)
    assert len(entries) == 1 and entries[0]["exported"] and entries[0]["selectionReason"] == "pinned launch.departure"


def test_status_labels_never_call_an_unverified_candidate_verified():
    assert candidate_status({"verdict": "accepted"})[0] == "VERIFIED"
    assert candidate_status({"verdict": "degraded", "verification": "not_checked"})[0] == "VERIFICATION NOT CHECKED"
    assert candidate_status({"verdict": "rejected"})[0] == "REJECTED"
    assert candidate_status({"verdict": None})[0] == "NOT SHORTLISTED"
    summary = candidate_summary({"departureMjd2000": T0, "tofDays": TOF, "lambertTotalDvMs": 5600.0})
    assert "250 d" in summary and "5.600 km/s" in summary


# ------------------------------------------------------------- viewer side
def _scene():
    ref_et = mjd2000_to_ephemeris_seconds(T0 + 20)
    days = np.linspace(0.0, 230.0, 24)
    earth = np.column_stack([1.5e8 * np.cos(days / 58), 1.5e8 * np.sin(days / 58), np.zeros_like(days)])
    scene = {
        "schemaVersion": "1.0", "id": "cand-test", "title": "Candidates", "distanceUnit": "km",
        "centralBodyId": "sun",
        "bodies": [
            {"id": "sun", "name": "Sun", "color": "#fff4d6", "kind": "sun", "radiusDisplay": 10.0,
             "position": [0.0, 0.0, 0.0]},
            {"id": "earth", "name": "Earth", "color": "#4c8eff", "kind": "planet", "radiusDisplay": 4.6,
             "radius": 6371.0, "trail": {"times": days.tolist(), "positions": earth.tolist()}},
            {"id": "spacecraft", "name": "Probe", "color": "#e8a23e", "kind": "spacecraft", "radiusDisplay": 3.0,
             "trail": {"times": days.tolist(), "positions": (earth * 1.1).tolist()}},
        ],
        "timeline": {"min": 0.0, "max": 230.0, "unitLabel": "days", "events": [],
                     "referenceEpochEt": ref_et},
    }
    scene["candidates"] = build_candidate_entries(_records(), _gate(), 0, ref_et, "spacecraft", arc_fn=_fake_arc)
    return scene


def test_scene_with_candidates_validates_and_rejects_ragged_paths():
    scene = _scene()
    validate_scene(scene)
    scene["candidates"][0]["screeningPath"]["times"].pop()
    with pytest.raises(ValueError, match="screening path"):
        validate_scene(scene)


def test_state_starts_on_the_exported_candidate():
    state = MissionState()
    seen = []
    state.candidate_changed.connect(seen.append)
    state.set_scene(_scene())
    assert state.candidate_id == "candidate-0" and state.candidate()["exported"]
    state.select_candidate("candidate-3")
    state.select_candidate("nope")
    assert seen == ["candidate-3"] and state.candidate()["searchIndex"] == 3


def test_renderer_highlights_and_hides_arcs_outside_the_scene_frame():
    plotter = pv.Plotter(off_screen=True)
    try:
        r = SceneRenderer(plotter)
        r.load(_scene())
        r.highlight_candidate("candidate-1", show_all=True)
        actor = plotter.renderer.actors["candidate-candidate-1"]
        assert np.allclose(actor.prop.color.float_rgb, pv.Color(CANDIDATE_SELECTED_COLOR).float_rgb)
        assert all(r.candidate_visible(c) for c in ("candidate-0", "candidate-1", "candidate-3"))

        r.highlight_candidate("candidate-1", show_all=False)
        assert r.candidate_visible("candidate-1") and not r.candidate_visible("candidate-0")

        r.set_frame("earth")
        assert not r.candidate_visible("candidate-1")
        r.set_frame(None)
        assert r.candidate_visible("candidate-1")
    finally:
        plotter.close()


@pytest.mark.slow
def test_candidate_panel_and_inspector_follow_the_selection():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow()
    window.show()
    try:
        deadline = __import__("time").monotonic() + 180
        while window._loading and __import__("time").monotonic() < deadline:
            app.processEvents()
        assert not window.candidate_panel.isVisible()  # solar system: no candidates

        window._apply_scene(_scene())
        app.processEvents()
        # The solar system's cards are gone at once, not at some later
        # event-loop turn: one card per body plus the candidate card.
        from PySide6.QtWidgets import QFrame
        cards = window.inspector.findChildren(QFrame, "bodyCard")
        assert len(cards) == len(_scene()["bodies"]) + 1
        panel = window.candidate_panel
        assert panel.isVisible()
        assert panel.row("candidate-0").property("active")  # exported one preselected
        assert "mission trajectory" in window.inspector.candidate_card_text()

        panel.row("candidate-1").mousePressEvent(None)
        app.processEvents()
        assert window.state.candidate_id == "candidate-1"
        assert panel.row("candidate-1").property("active") and not panel.row("candidate-0").property("active")
        card = window.inspector.candidate_card_text()
        assert "REJECTED" in card and "(best)" in card and "not flown in high fidelity" in card
        assert window.renderer.candidate_visible("candidate-1")

        panel.show_all_box.setChecked(False)
        app.processEvents()
        assert not window.renderer.candidate_visible("candidate-0")
    finally:
        window.plotter.close()
        window.close()


def test_inspector_fits_its_minimum_width_with_long_maneuver_names():
    """A long maneuver name used to set a minimum width wider than the panel;
    with horizontal scrolling off, every card was clipped on the right."""
    from PySide6.QtWidgets import QApplication, QWidget

    from orbitopt.viz.inspector import InspectorPanel
    from orbitopt.viz.theme import STYLESHEET

    app = QApplication.instance() or QApplication([])
    scene = _scene()
    scene["timeline"]["events"] = [
        {"label": "Trajectory-correction maneuver", "time": 3.14, "note": "closes the gap"},
        {"label": "Mars-SOI entry / approach", "time": 200.0},
    ]
    state = MissionState()
    inspector = InspectorPanel(state)
    inspector.setStyleSheet(STYLESHEET)
    inspector.resize(200, 800)  # the panel's minimum drag width (app._RIGHT_MIN)
    inspector.show()
    state.set_scene(scene)
    app.processEvents()
    host = inspector.findChild(QWidget, "infoCards")
    assert host.minimumSizeHint().width() <= host.parentWidget().width()
    inspector.close()
