"""Populates a PyVista plotter (a plain ``pv.Plotter`` for the scripting
API in pv_viewer.py, or a ``pyvistaqt.QtInteractor`` embedded in the
Mission Control app in app.py -- both share the same drawing API, so one
renderer serves both) from a SceneData dict. This is the one place that
knows how to turn {bodies, orbits, trails} into VTK meshes; neither caller
duplicates that logic.

Deliberately has NO on-screen text/slider/legend widgets of its own (unlike
this package's simpler pv_viewer.show_scene, which draws those with VTK's
built-in 2D widgets for a quick, dependency-light script viewer) -- the app
wants real, richly-stylable Qt widgets for that chrome instead, so this
module's job stops at the 3D content and a small data API
(``set_time``/``distances_at``) for whatever HUD the caller builds around it.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

_MARKER_BASE_SIZE = 8.0
_MARKER_WEIGHT_SIZE = 1.1


def marker_size(body: dict) -> float:
    return _MARKER_BASE_SIZE + _MARKER_WEIGHT_SIZE * body.get("radiusDisplay", 4.0)


def position_at_time(body: dict, t: float) -> np.ndarray:
    if "trail" not in body:
        return np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
    times = body["trail"]["times"]
    positions = body["trail"]["positions"]
    if t <= times[0]:
        return np.asarray(positions[0], dtype=float)
    if t >= times[-1]:
        return np.asarray(positions[-1], dtype=float)
    idx = int(np.searchsorted(times, t))
    idx = max(1, min(idx, len(times) - 1))
    t0, t1 = times[idx - 1], times[idx]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0 = np.asarray(positions[idx - 1], dtype=float)
    p1 = np.asarray(positions[idx], dtype=float)
    return p0 + frac * (p1 - p0)


def trail_index_at_time(body: dict, t: float) -> int:
    return int(np.searchsorted(body["trail"]["times"], t))


class SceneRenderer:
    """Owns the 3D content of one scene inside a plotter. ``load()`` tears
    down and rebuilds everything (switching missions is rare -- a handful
    of times per session -- so simplicity beats trying to diff old/new
    scenes); ``set_time()`` is the hot path, called every timeline tick or
    time-warp step, and only mutates existing mesh point arrays plus swaps
    the "traveled so far" trail actor, never rebuilds from scratch.
    """

    def __init__(self, plotter):
        self.plotter = plotter
        self.scene: dict | None = None
        self._moving: dict[str, dict] = {}
        self._static_positions: dict[str, np.ndarray] = {}

    def load(self, scene: dict) -> None:
        self.plotter.clear()
        self.scene = scene
        self._moving = {}
        self._static_positions = {}

        for body in scene["bodies"]:
            color = body["color"]

            if "orbit" in body:
                pts = np.asarray(body["orbit"], dtype=float)
                pts = np.vstack([pts, pts[0]])
                orbit_mesh = pv.MultipleLines(pts)
                opacity = 0.32 if body.get("orbitDashed") else 0.5
                self.plotter.add_mesh(orbit_mesh, color=color, opacity=opacity, line_width=1, pickable=False)

            if "trail" in body:
                t0 = scene.get("timeline", {}).get("min", body["trail"]["times"][0])
                pos0 = position_at_time(body, t0)

                point_poly = pv.PolyData(pos0.reshape(1, 3))
                self.plotter.add_mesh(
                    point_poly, color=color, point_size=marker_size(body),
                    render_points_as_spheres=True, name=f"marker-{body['id']}",
                )

                full_positions = np.asarray(body["trail"]["positions"], dtype=float)
                full_poly = pv.MultipleLines(full_positions)
                self.plotter.add_mesh(full_poly, color=color, line_width=1.1, opacity=0.3, pickable=False)

                past_poly = pv.MultipleLines(np.vstack([full_positions[0], pos0]))
                self.plotter.add_mesh(
                    past_poly, color=color, line_width=2.6, opacity=0.95,
                    name=f"past-{body['id']}", pickable=False,
                )

                self._moving[body["id"]] = {
                    "body": body,
                    "point_poly": point_poly,
                    "past_actor_name": f"past-{body['id']}",
                    "full_positions": full_positions,
                    "color": color,
                }
            else:
                pos = np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
                self._static_positions[body["id"]] = pos
                marker = pv.PolyData(pos.reshape(1, 3))
                self.plotter.add_mesh(
                    marker, color=color, point_size=marker_size(body),
                    render_points_as_spheres=True, name=f"marker-{body['id']}",
                )
                self.plotter.add_point_labels(
                    pos.reshape(1, 3), [body["name"]], font_size=12, text_color=color,
                    shape=None, always_visible=True, show_points=False,
                )

        timeline = scene.get("timeline")
        self.set_time(timeline["min"] if timeline else 0.0)
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()

    def set_time(self, t: float) -> dict[str, float]:
        """Move every trailed body to its position at time ``t``, regrow
        its "traveled so far" trail, and return {body_id: distance from
        origin} for every body (moving or static) -- the HUD's job to
        display, not this class's.
        """
        distances = {}
        for body_id, entry in self._moving.items():
            body = entry["body"]
            pos = position_at_time(body, t)
            entry["point_poly"].points = pos.reshape(1, 3)

            idx = trail_index_at_time(body, t)
            past_pts = np.vstack([entry["full_positions"][: max(idx, 1)], pos.reshape(1, 3)])
            new_past_poly = pv.MultipleLines(past_pts) if len(past_pts) > 1 else pv.MultipleLines(np.vstack([pos, pos]))
            self.plotter.add_mesh(
                new_past_poly, color=entry["color"], line_width=2.6, opacity=0.95,
                name=entry["past_actor_name"], pickable=False,
            )
            distances[body_id] = float(np.linalg.norm(pos))

        for body_id, pos in self._static_positions.items():
            distances[body_id] = float(np.linalg.norm(pos))

        self.plotter.render()
        return distances

    def body_world_position(self, body_id: str, t: float) -> np.ndarray:
        if body_id in self._moving:
            return position_at_time(self._moving[body_id]["body"], t)
        return self._static_positions.get(body_id, np.zeros(3))

    def focus_on(self, body_id: str, t: float) -> None:
        pos = self.body_world_position(body_id, t)
        self.plotter.camera.focal_point = tuple(pos)
        self.plotter.render()
