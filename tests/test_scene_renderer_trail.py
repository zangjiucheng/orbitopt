"""SceneRenderer.set_time updates the "traveled so far" trail in place.

Pins the timeline hot-path fix: set_time used to build a new
pv.MultipleLines and add_mesh() it (replacing the actor) on every tick/scrub
event. Now one persistent PolyData/actor per trailed body is mutated in
place, and the drawn polyline must still be exactly the old vertex sequence
``full_positions[:max(idx, 1)] + [pos]`` for any (forward, backward, or
jumping) sequence of times.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from orbitopt.viz.scene_renderer import (  # noqa: E402
    SceneRenderer,
    position_at_time,
    trail_index_at_time,
)


def _scene(n: int = 200) -> dict:
    t = np.linspace(0.0, 10.0, n)
    pos = np.c_[np.cos(t), np.sin(t), 0.05 * t] * 7000.0
    return {
        "bodies": [
            {"id": "earth", "name": "Earth", "color": "#3399ff", "position": [0.0, 0.0, 0.0]},
            {"id": "sc", "name": "SC", "color": "#ffaa00",
             "trail": {"times": t.tolist(), "positions": pos.tolist()}},
        ],
        "timeline": {"min": 0.0, "max": 10.0},
    }


def _expected_polyline(body: dict, t: float) -> np.ndarray:
    full = np.asarray(body["trail"]["positions"], dtype=float)
    idx = trail_index_at_time(body, t)
    return np.vstack([full[: max(idx, 1)], position_at_time(body, t).reshape(1, 3)])


def _drawn_polyline(poly) -> np.ndarray:
    lines = np.asarray(poly.lines)
    assert poly.n_lines == 1
    count = int(lines[0])
    return np.asarray(poly.points)[lines[1: 1 + count]]


def test_set_time_updates_past_trail_in_place():
    scene = _scene()
    body = scene["bodies"][1]
    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = SceneRenderer(plotter)
        renderer.load(scene)
        actor = plotter.renderer.actors["past-sc"]
        poly = actor.mapper.dataset

        rng = np.random.default_rng(1)
        times = np.concatenate([
            [0.0, -1.0, 10.0, 11.0, 5.0, 5.0],
            np.linspace(0.0, 10.0, 40),
            np.linspace(10.0, 0.0, 40),
            rng.uniform(-0.5, 10.5, 60),
            body["trail"]["times"][:5],  # exactly on samples
        ])
        for t in times:
            distances = renderer.set_time(float(t), render=False)
            # Same actor and dataset every time: no add_mesh() replacement.
            assert plotter.renderer.actors["past-sc"] is actor
            assert actor.mapper.dataset is poly
            np.testing.assert_allclose(_drawn_polyline(poly), _expected_polyline(body, float(t)))
            assert distances["sc"] == pytest.approx(np.linalg.norm(position_at_time(body, float(t))))
            np.testing.assert_allclose(
                renderer.body_world_position("sc", float(t)), position_at_time(body, float(t)))
    finally:
        plotter.close()

