"""SceneRenderer honours the optional per-body lifespan (tStart/tEnd).

FL-005 scene export writes ``tStart``/``tEnd`` (timeline units) per body so a
staged booster disappears at its ``t_end_s`` and a child body first appears
at its ``t_start_s``; the renderer gates every actor of a trailed body's
marker/full-trail/past-trail set by them. Bodies without the bounds -- every
scene written before FL-005 -- must never have an actor touched.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from orbitopt.viz.scene_renderer import SceneRenderer  # noqa: E402


def _scene() -> dict:
    t = np.linspace(100.0, 200.0, 21)
    pos = np.c_[np.full_like(t, 7000.0), 60.0 * (t - 100.0), np.zeros_like(t)]
    return {
        "schemaVersion": "1.0", "id": "life", "title": "life", "distanceUnit": "km",
        "bodies": [
            {"id": "earth", "name": "Earth", "color": "#3399ff", "position": [0.0, 0.0, 0.0]},
            {"id": "bstr", "name": "Booster", "color": "#ffaa00", "kind": "spacecraft",
             "radiusDisplay": 3.0, "tStart": 100.0, "tEnd": 200.0,
             "trail": {"times": t.tolist(), "positions": pos.tolist()}},
        ],
        "timeline": {"unitLabel": "seconds", "min": 0.0, "max": 300.0},
    }


def _visible(renderer) -> bool:
    entry = renderer._moving["bstr"]
    flags = {bool(a.GetVisibility()) for a in entry["actors"]}
    return flags == {True}


def test_lifespan_hides_body_outside_its_window():
    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = SceneRenderer(plotter)
        renderer.load(_scene())  # load leaves the timeline at min = 0 s
        assert not _visible(renderer)  # before tStart: nothing drawn
        renderer.set_time(99.9, render=False)
        assert not _visible(renderer)
        renderer.set_time(100.0, render=False)
        assert _visible(renderer)      # appears exactly at tStart
        renderer.set_time(150.0, render=False)
        assert _visible(renderer)
        renderer.set_time(200.0, render=False)
        assert _visible(renderer)      # lives through its last sample
        renderer.set_time(200.001, render=False)
        assert not _visible(renderer)  # disappears at tEnd
    finally:
        plotter.close()


def test_bodies_without_lifespan_are_never_hidden():
    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = SceneRenderer(plotter)
        scene = _scene()
        scene["bodies"][1].pop("tStart")
        scene["bodies"][1].pop("tEnd")
        renderer.load(scene)
        renderer.set_time(0.0, render=False)
        assert _visible(renderer)
    finally:
        plotter.close()


def test_earth_rotation_tracks_non_day_timelines():
    # A seconds-based timeline must spin a textured rotator with real-time
    # physics: timeline t drives the spin angle via timeline.unitLabel, so
    # exactly one sidereal day later the orientation must return to the
    # identity -- not to it 86400x too often (the old hardcoded days factor).
    scene = {
        "schemaVersion": "1.0", "id": "spin", "title": "spin", "distanceUnit": "km",
        "bodies": [
            {"id": "earth", "name": "Earth", "color": "#3399ff", "kind": "planet",
             "radiusDisplay": 4.6, "radius": 6378.137, "texture": "earth_daymap.jpg",
             "rotationPeriodHours": 23.9344696,
             "orientationBasis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        ],
        "timeline": {"unitLabel": "seconds", "min": 0.0, "max": 540.0},
    }
    period_s = 23.9344696 * 3600.0
    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = SceneRenderer(plotter)
        renderer.load(scene)
        actor = renderer._rotating[0]["actor"]
        renderer.set_time(0.0, render=False)
        o0 = np.asarray(actor.orientation)
        renderer.set_time(period_s, render=False)
        o1 = np.asarray(actor.orientation)  # one full sidereal day of t
        renderer.set_time(period_s / 2.0, render=False)
        o_half = np.asarray(actor.orientation)
        np.testing.assert_allclose(o1, o0, atol=1e-6)  # whole spin => same orientation
        assert not np.allclose(o_half, o0, atol=1e-3)  # half a day => 180 deg off
    finally:
        plotter.close()
