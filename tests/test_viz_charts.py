"""Linked 2D charts: porkchop data in the scene, the color scale, the
distance series, and both charts driving / following MissionState."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF  # noqa: E402

from orbitopt.scene_format import validate_scene  # noqa: E402
from orbitopt.viz.charts import (  # noqa: E402
    ChartPanel,
    _ramp_rgb,
    color_scale,
    distance_series,
    format_km,
    nice_ticks,
)
from orbitopt.viz.state import MissionState  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _porkchop_doc():
    from orbitopt.viz.porkchop import PorkchopGrid, porkchop_document

    t0 = np.linspace(9800.0, 9860.0, 7)
    tof = np.linspace(200.0, 320.0, 5)
    dv = 5.0 + np.add.outer((tof - 260.0) ** 2 / 2000.0, (t0 - 9830.0) ** 2 / 500.0)
    converged = np.ones_like(dv, dtype=bool)
    converged[0, 0] = False
    grid = PorkchopGrid(t0, tof, dv, dv, dv, converged)
    return porkchop_document(grid)


def _scene():
    days = np.linspace(0.0, 10.0, 101)
    earth = np.zeros((len(days), 3))
    sc = np.column_stack([7000.0 + 4.0e4 * days, np.zeros_like(days), np.zeros_like(days)])
    moon_t = np.linspace(0.0, 10.0, 11)
    moon = np.column_stack([np.full(11, 3.84e5), np.zeros(11), np.zeros(11)])
    return {
        "schemaVersion": "1.0", "id": "c", "title": "C", "distanceUnit": "km", "centralBodyId": "earth",
        "bodies": [
            {"id": "earth", "name": "Earth", "color": "#4c8eff", "kind": "planet", "radiusDisplay": 4.6,
             "position": [0.0, 0.0, 0.0]},
            {"id": "moon", "name": "Moon", "color": "#928da3", "kind": "moon", "radiusDisplay": 3.4,
             "trail": {"times": moon_t.tolist(), "positions": moon.tolist()}},
            {"id": "spacecraft", "name": "Probe", "color": "#e8a23e", "kind": "spacecraft", "radiusDisplay": 3.0,
             "trail": {"times": days.tolist(), "positions": (earth + sc).tolist()}},
        ],
        "timeline": {"min": 0.0, "max": 10.0, "unitLabel": "days",
                     "events": [{"label": "Burn", "time": 2.0}]},
        "porkchop": _porkchop_doc(),
        "candidates": [
            {"id": "candidate-0", "label": "Candidate #0", "searchIndex": 0, "exported": True,
             "departureMjd2000": 9830.0, "tofDays": 260.0, "lambertTotalDvMs": 5000.0,
             "screeningPath": {"fidelity": "x", "times": [0.0, 1.0], "positions": [[0, 0, 0], [1, 1, 1]]}},
            {"id": "candidate-1", "label": "Candidate #1", "searchIndex": 1, "exported": False,
             "departureMjd2000": 9850.0, "tofDays": 300.0, "lambertTotalDvMs": 5900.0,
             "screeningPath": {"fidelity": "x", "times": [0.0, 1.0], "positions": [[0, 0, 0], [1, 1, 1]]}},
        ],
    }


def test_porkchop_document_shape_and_gaps():
    doc = _porkchop_doc()
    assert len(doc["totalDv"]) == len(doc["tofDays"]) == 5
    assert all(len(row) == len(doc["departureMjd2000"]) == 7 for row in doc["totalDv"])
    assert doc["totalDv"][0][0] is None  # not converged: a gap, not a value
    scene = _scene()
    validate_scene(scene)
    scene["porkchop"]["totalDv"][1].pop()
    with pytest.raises(ValueError, match="porkchop"):
        validate_scene(scene)


def test_color_scale_and_ramp():
    values = np.array([[np.nan, 5.0, 6.0], [7.0, 9.0, 20.0]])
    vmin, vmax = color_scale(values)
    assert vmin == 5.0 and vmax == 7.0  # best .. median
    lightest, darkest = _ramp_rgb(np.array([0.0, 1.0]))
    assert lightest.sum() > darkest.sum()  # low delta-v reads lightest on the dark surface
    assert nice_ticks(0.0, 10.0, 5) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert format_km(384_400.0) == "384 k km" and format_km(2.3e8) == "230 M km"
    assert format_km(1e9) == "1,000 M km" and format_km(6563.0) == "6.56 k km"


def test_distance_series_per_natural_body():
    times, series = distance_series(_scene())
    assert [s["id"] for s in series] == ["earth", "moon"]  # not the spacecraft itself
    earth = series[0]["km"]
    assert earth[0] == pytest.approx(7000.0) and earth[-1] == pytest.approx(407000.0)
    moon = series[1]["km"]
    assert moon[0] == pytest.approx(3.84e5 - 7000.0)
    assert len(times) == len(earth) == 101


def test_charts_drive_and_follow_state(app):
    state = MissionState()
    panel = ChartPanel(state)
    panel.resize(800, 260)
    panel.show()
    state.set_scene(_scene())
    app.processEvents()
    assert [panel.tabText(i) for i in range(panel.count())] == ["Launch window", "Distance"]

    # Clicking a candidate marker on the porkchop selects it.
    chart = panel.porkchop
    chart.resize(800, 240)
    marker = QPointF(chart._x(9850.0), chart._y(300.0))
    assert chart.candidate_at(marker) == "candidate-1"
    assert chart.candidate_at(QPointF(marker.x() + 40, marker.y())) is None
    chart.candidate_clicked.emit(chart.candidate_at(marker))
    assert state.candidate_id == "candidate-1"
    assert chart._selected == "candidate-1"

    # Clicking the distance chart moves time; moving time moves its cursor.
    dist = panel.distance
    dist.resize(800, 240)
    r = dist.plot_rect()
    dist.time_requested.emit(dist.time_at(r.left() + 0.5 * r.width()))
    assert state.time == pytest.approx(5.0, abs=0.05)
    state.set_time(7.5)
    assert dist._time == 7.5

    # Grab both so the paint paths actually run.
    assert not chart.grab().isNull() and not dist.grab().isNull()

    # A scene with nothing to chart hides the panel.
    state.set_scene({"id": "s", "title": "S", "distanceUnit": "AU", "bodies": []})
    assert panel.count() == 0 and panel.isHidden()
    panel.close()


def test_planet_centered_scene_reads_as_altitude(app):
    from orbitopt.viz.inspector import InspectorPanel
    from orbitopt.viz.state import altitude_reference

    scene = _scene()
    scene["bodies"][0]["radius"] = 6378.137  # Earth, the central body, now has a drawn sphere
    ref = altitude_reference(scene)
    assert (ref.body_id, ref.radius, ref.flattening) == ("earth", 6378.137, 0.0)
    _times, series = distance_series(scene)
    earth = next(s for s in series if s["id"] == "earth")
    assert earth["label"] == "altitude above Earth"
    assert earth["km"][0] == pytest.approx(7000.0 - 6378.137)
    assert next(s for s in series if s["id"] == "moon")["label"] == "to Moon"  # other bodies: distance

    state = MissionState()
    inspector = InspectorPanel(state)
    state.set_scene(scene)
    assert inspector.altitude_body_ids() == ["spacecraft"]
    inspector.set_distances({"spacecraft": 6653.137, "moon": 384_000.0},
                            {"spacecraft": np.array([6653.137, 0.0, 0.0])})
    assert inspector.distance_text("spacecraft") == "275 km"   # altitude, not 6,653 km
    assert inspector.distance_text("moon") == "384,000 km"     # a natural body keeps its distance

    sun_centered = _scene()
    sun_centered["centralBodyId"] = None
    assert altitude_reference(sun_centered) is None


def test_altitude_is_measured_from_the_ellipsoid():
    from orbitopt.viz.state import altitude_reference

    f = 1.0 / 298.257223563
    scene = _scene()
    scene["bodies"][0].update(radius=6378.137, flattening=f)
    ref = altitude_reference(scene)
    polar = 6378.137 * (1.0 - f)
    # On the ground at the pole and on the equator: both ~0, not -21 km at the pole.
    assert ref.altitude([0.0, 0.0, polar]) == pytest.approx(0.0, abs=0.1)
    assert ref.altitude([6378.137, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # At Starbase's latitude a sphere of the equatorial radius put the ground ~4 km up.
    lat = np.radians(25.997)
    r_ground = 6378.137 * (1.0 - f * np.sin(lat) ** 2)
    assert ref.altitude(r_ground * np.array([np.cos(lat), 0.0, np.sin(lat)])) == pytest.approx(0.0, abs=0.05)
    # The pole follows orientationBasis (third column), not the scene's +Z.
    scene["bodies"][0]["orientationBasis"] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    assert altitude_reference(scene).altitude([0.0, polar, 0.0]) == pytest.approx(0.0, abs=0.1)


def test_flattened_body_renders_as_an_ellipsoid():
    import pyvista as pv

    from orbitopt.viz.scene_renderer import SceneRenderer

    f = 1.0 / 298.257223563
    scene = {"schemaVersion": "1.0", "id": "e", "title": "E", "distanceUnit": "km", "centralBodyId": "earth",
             "bodies": [{"id": "earth", "name": "Earth", "color": "#4c8eff", "kind": "planet", "radiusDisplay": 4.6,
                         "position": [0.0, 0.0, 0.0], "texture": "earth_daymap.jpg", "radius": 6378.137,
                         "flattening": f}]}
    plotter = pv.Plotter(off_screen=True)
    try:
        SceneRenderer(plotter).load(scene)
        mesh = plotter.renderer.actors["marker-earth"].mapper.dataset
        pts = np.asarray(mesh.points)
        assert np.abs(pts[:, 2]).max() == pytest.approx(6378.137 * (1.0 - f), rel=1e-6)
        assert np.hypot(pts[:, 0], pts[:, 1]).max() == pytest.approx(6378.137, rel=1e-6)
        # Texture rows map to geodetic latitude: a vertex's v is 0.5 + phi / pi
        # where phi is the latitude of the ellipsoid normal there.
        v = np.asarray(mesh.active_texture_coordinates)[:, 1]
        normals = pts.copy()
        normals[:, 2] /= (1.0 - f) ** 2
        phi = np.arctan2(normals[:, 2], np.hypot(normals[:, 0], normals[:, 1]))
        np.testing.assert_allclose(v, 0.5 + phi / np.pi, atol=1e-6)
    finally:
        plotter.close()
