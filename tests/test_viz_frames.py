"""Multi-scale view: Hermite trail sampling and re-centering the display on a
body (SceneRenderer.set_frame), plus the frame/scale metadata helpers."""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("PySide6")

from orbitopt.scene_format import validate_scene  # noqa: E402
from orbitopt.viz.scene_renderer import (  # noqa: E402
    SceneRenderer,
    _sample_trail,
    _sample_trail_many,
)
from orbitopt.viz.state import MissionState, frame_options, size_exaggeration_range  # noqa: E402
from orbitopt.viz.view_strip import scale_note  # noqa: E402

# A planet on a 1.5e8 km circular orbit with a 365-day period, sampled once a
# day -- the Mars scene's Earth trail in miniature.
R_PLANET = 1.496e8
OMEGA = 2.0 * np.pi / 365.25  # rad/day


def _planet(t):
    t = np.asarray(t, dtype=float)
    return np.stack([R_PLANET * np.cos(OMEGA * t), R_PLANET * np.sin(OMEGA * t), np.zeros_like(t)], axis=-1)


def _planet_velocity(t):
    t = np.asarray(t, dtype=float)
    return np.stack([-R_PLANET * OMEGA * np.sin(OMEGA * t), R_PLANET * OMEGA * np.cos(OMEGA * t),
                     np.zeros_like(t)], axis=-1)


# A spacecraft in a 6,563 km (185 km altitude) circular orbit around that
# planet, densely sampled (90 s), for the first 0.2 days.
R_LEO = 6563.0
LEO_OMEGA = 2.0 * np.pi / (88.5 / 1440.0)  # rad/day, ~88.5 min period


def _spacecraft(t):
    t = np.asarray(t, dtype=float)
    local = np.stack([R_LEO * np.cos(LEO_OMEGA * t), R_LEO * np.sin(LEO_OMEGA * t), np.zeros_like(t)], axis=-1)
    return _planet(t) + local


def _scene():
    days = np.linspace(0.0, 10.0, 11)
    sc_t = np.arange(0.0, 0.2, 90.0 / 86400.0)
    return {
        "schemaVersion": "1.0",
        "id": "frames-test",
        "title": "Frames",
        "distanceUnit": "km",
        "centralBodyId": "sun",
        "bodies": [
            {"id": "sun", "name": "Sun", "color": "#fff4d6", "kind": "sun", "radiusDisplay": 10.0,
             "position": [0.0, 0.0, 0.0]},
            {"id": "earth", "name": "Earth", "color": "#4c8eff", "kind": "planet", "radiusDisplay": 4.6,
             "radius": 6371.0,
             "trail": {"times": days.tolist(), "positions": _planet(days).tolist(),
                       "velocities": _planet_velocity(days).tolist()}},
            {"id": "sc", "name": "Probe", "color": "#e8a23e", "kind": "spacecraft", "radiusDisplay": 3.0,
             "trail": {"times": sc_t.tolist(), "positions": _spacecraft(sc_t).tolist()}},
        ],
        "timeline": {"min": 0.0, "max": 0.2, "unitLabel": "days", "events": []},
    }


# ---------------------------------------------------------------- sampling
def test_hermite_beats_chord_on_a_sparse_orbit():
    days = np.linspace(0.0, 10.0, 11)
    pos, vel = _planet(days), _planet_velocity(days)
    t = 3.5  # mid-interval: worst case for a chord
    truth = _planet(t)
    chord_err = np.linalg.norm(_sample_trail(days, pos, t)[1] - truth)
    hermite_err = np.linalg.norm(_sample_trail(days, pos, t, vel)[1] - truth)
    # The chord misses the arc by ~5,500 km -- more than Earth's radius.
    assert chord_err > 5000.0
    assert hermite_err < 0.1  # tens of meters


def test_sample_many_matches_scalar_sampling():
    days = np.linspace(0.0, 10.0, 11)
    pos, vel = _planet(days), _planet_velocity(days)
    ts = np.array([-1.0, 0.0, 0.3, 1.0, 4.75, 9.99, 10.0, 12.0])
    for velocities in (None, vel):
        many = _sample_trail_many(days, pos, ts, velocities)
        one = np.array([_sample_trail(days, pos, t, velocities)[1] for t in ts])
        np.testing.assert_allclose(many, one, rtol=0, atol=1e-6)


# ------------------------------------------------------------------ frames
@pytest.fixture
def renderer():
    plotter = pv.Plotter(off_screen=True)
    r = SceneRenderer(plotter)
    r.load(_scene())
    yield r
    plotter.close()


def _drawn(renderer, name):
    return np.asarray(renderer.plotter.renderer.actors[name].mapper.dataset.points)


def test_planet_frame_shows_the_parking_orbit_at_its_true_radius(renderer):
    sc_times = renderer._moving["sc"]["times"]
    t = float(sc_times[91])  # on a probe sample
    renderer.set_time(t)
    renderer.set_frame("earth")

    # The frame body sits at the origin; the probe is on its 6,563 km orbit.
    assert np.linalg.norm(renderer.body_display_position("earth", t)) < 1e-6
    assert np.linalg.norm(renderer.body_display_position("sc", t)) == pytest.approx(R_LEO, abs=1.0)

    # Between probe samples the probe itself is still a straight chord (it has
    # no velocities): inside the circle by at most the chord sagitta R*θ²/8,
    # ~9 km for a 90 s step -- the frame transform adds nothing to that.
    mid = 0.5 * (sc_times[91] + sc_times[92])
    theta = LEO_OMEGA * (sc_times[92] - sc_times[91])
    sagitta = R_LEO * theta ** 2 / 8.0
    r_mid = np.linalg.norm(renderer.body_display_position("sc", mid))
    assert R_LEO - sagitta - 1.0 <= r_mid <= R_LEO

    # Every drawn trail vertex is at the parking-orbit radius: Earth is
    # Hermite-interpolated at the probe's own (non-daily) sample times. With
    # straight chords these would scatter by thousands of km.
    trail = renderer._moving["sc"]["full_positions"]
    radii = np.linalg.norm(trail, axis=1)
    assert np.max(np.abs(radii - R_LEO)) < 1.0


def test_frame_change_is_display_only(renderer):
    t = 0.05
    world_before = renderer.body_world_position("sc", t)
    distances_before = renderer.set_time(t)
    measured_before = renderer.measure_line("sc", "sun", t)

    renderer.set_frame("earth")
    distances_after = renderer.set_time(t)

    np.testing.assert_allclose(renderer.body_world_position("sc", t), world_before)
    assert distances_after == pytest.approx(distances_before)  # readouts stay world-frame
    assert renderer.measure_line("sc", "sun", t) == pytest.approx(measured_before)  # translation-invariant

    # The static Sun moved with the frame: it is drawn at -r_earth(t).
    sun = renderer.body_display_position("sun", t)
    np.testing.assert_allclose(sun, -_planet(t), rtol=1e-9)


def test_returning_to_the_scene_frame_restores_world_geometry(renderer):
    world_trail = renderer._moving["sc"]["world_positions"].copy()
    renderer.set_frame("earth")
    renderer.set_frame(None)
    np.testing.assert_array_equal(renderer._moving["sc"]["full_positions"], world_trail)
    np.testing.assert_array_equal(renderer.frame_origin(0.1), np.zeros(3))


def test_unknown_frame_falls_back_to_scene_frame(renderer):
    renderer.set_frame("nope")
    assert renderer.frame_body_id is None


def test_local_extent_frames_the_parking_orbit(renderer):
    renderer.set_frame("earth")
    extent = renderer.local_extent(0.0)
    assert R_LEO <= extent <= 20 * 6371.0


# ------------------------------------------------------- metadata helpers
def test_frame_options_list_moving_natural_bodies_only():
    scene = _scene()
    assert frame_options(scene) == [(None, "Sun"), ("earth", "Earth")]  # not the spacecraft
    static = dict(scene)
    static.pop("timeline")
    assert frame_options(static) == [(None, "Sun")]
    assert frame_options(None) == []


def test_state_frame_rules():
    state = MissionState()
    state.set_scene(_scene())
    seen = []
    state.frame_changed.connect(seen.append)
    state.set_frame("sc")  # a spacecraft is not a frame option
    state.set_frame("earth")
    state.set_frame("earth")  # no-op
    assert seen == ["earth"] and state.frame_body_id == "earth"
    state.set_scene(_scene())  # a new scene starts in its own frame
    assert state.frame_body_id is None


def test_scale_note_reports_exaggeration():
    scene = _scene()
    assert scale_note(scene) == ("", False)  # no real spheres drawn
    for body in scene["bodies"][:2]:
        body["texture"] = "x.jpg"
        body["radius"] = 1.0
    assert scale_note(scene)[0].startswith("Body sizes to scale")
    scene["bodies"][0]["radiusExaggeration"] = 17.0
    scene["bodies"][1]["radiusExaggeration"] = 860.0
    assert size_exaggeration_range(scene) == (17.0, 860.0)
    text, warn = scale_note(scene)
    assert warn and "×17–860" in text


def test_schema_accepts_velocities_and_rejects_ragged_ones():
    scene = _scene()
    validate_scene(scene)
    scene["bodies"][1]["trail"]["velocities"].pop()
    with pytest.raises(ValueError, match="velocities"):
        validate_scene(scene)
