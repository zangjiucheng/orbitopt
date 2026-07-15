"""General-purpose interactive 3D scene viewer, built on PyVista/VTK --
a real desktop window, not a browser page, and no HTML/CSS/JS anywhere in
this package. Consumes the same SceneData dict produced by every
orbitopt.viz.* exporter (solar_system.py, mission_timeline.py, and any
future one) -- this module has no per-scenario code path, only a generic
renderer for that one schema (see orbitopt.viz.scene for the schema
helpers, and load_scene/show_scene below for how you actually open one).

Usage:
    from orbitopt.viz.pv_viewer import load_scene, show_scene
    show_scene(load_scene("path/to/some_scene.json"))

or, to view something you just computed without writing it to disk first:
    from orbitopt.viz.solar_system import export_solar_system_data
    show_scene(export_solar_system_data())

Marker sizes are rendered as VTK point sprites (`render_points_as_spheres`),
which VTK draws at a constant *screen-pixel* size regardless of camera
distance -- this is what a hand-rolled "constant apparent size" scheme in
a from-scratch 3D renderer has to reimplement (and, in an earlier
Three.js-based iteration of this viewer, got wrong: the Sun's marker
stayed large enough in world-space to swallow Mercury's entire orbit even
fully zoomed in). Using VTK's own point rendering sidesteps that whole
class of bug for free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

BACKGROUND = "#06050c"
INK_PRIMARY = "#f4f2ea"
INK_SECONDARY = "#a9a6bb"
ACCENT = "#e8a23e"

_MARKER_BASE_SIZE = 8.0
_MARKER_WEIGHT_SIZE = 1.1


def load_scene(path) -> dict:
    """Read a SceneData JSON document (as produced by any orbitopt.viz.*
    exporter) from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _position_at_time(body: dict, t: float) -> np.ndarray:
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


def _trail_index_at_time(body: dict, t: float) -> int:
    times = body["trail"]["times"]
    return int(np.searchsorted(times, t))


def _marker_size(body: dict) -> float:
    return _MARKER_BASE_SIZE + _MARKER_WEIGHT_SIZE * body.get("radiusDisplay", 4.0)


def show_scene(scene: dict, window_size=(1400, 900), off_screen: bool = False):
    """Open an interactive PyVista window rendering ``scene`` (a SceneData
    dict). Blocks until the window is closed, same as any desktop app.

    ``off_screen=True`` renders without opening a window and returns the
    Plotter instead of blocking (``plotter.screenshot(path)``) -- used by
    this package's own tests/tooling, not part of the normal interactive
    workflow.
    """
    plotter = pv.Plotter(window_size=list(window_size), off_screen=off_screen)
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing()

    plotter.add_text(scene["title"], font_size=16, color=INK_PRIMARY, position=(20, window_size[1] - 36))
    if scene.get("subtitle"):
        n_subtitle_lines = scene["subtitle"].count("\n") + 1
        plotter.add_text(
            scene["subtitle"], font_size=8, color=INK_SECONDARY,
            position=(20, window_size[1] - 62 - 14 * (n_subtitle_lines - 1)),
        )

    legend_entries = []
    moving_bodies = {}  # id -> dict of live handles updated by the timeline callback

    for body in scene["bodies"]:
        color = body["color"]
        legend_entries.append([body["name"], color])

        if "orbit" in body:
            pts = np.asarray(body["orbit"], dtype=float)
            pts = np.vstack([pts, pts[0]])
            orbit_mesh = pv.MultipleLines(pts)
            opacity = 0.32 if body.get("orbitDashed") else 0.5
            plotter.add_mesh(orbit_mesh, color=color, opacity=opacity, line_width=1, pickable=False)

        if "trail" in body:
            t0 = scene.get("timeline", {}).get("min", body["trail"]["times"][0])
            pos0 = _position_at_time(body, t0)

            point_poly = pv.PolyData(pos0.reshape(1, 3))
            point_poly.point_data["name"] = [body["id"]]
            plotter.add_mesh(
                point_poly, color=color, point_size=_marker_size(body),
                render_points_as_spheres=True, name=f"marker-{body['id']}",
            )

            full_positions = np.asarray(body["trail"]["positions"], dtype=float)

            # Full path, always visible at low opacity, so the shape of the
            # whole trajectory reads at a glance -- without this, a body at
            # its start-of-timeline position (little or no "traveled so
            # far" path yet) looks like an isolated dot with no indication
            # of where it's headed.
            full_poly = pv.MultipleLines(full_positions)
            plotter.add_mesh(full_poly, color=color, line_width=1.1, opacity=0.3, pickable=False)

            past_poly = pv.MultipleLines(np.vstack([full_positions[0], pos0]))
            plotter.add_mesh(past_poly, color=color, line_width=2.6, opacity=0.95, name=f"past-{body['id']}", pickable=False)

            moving_bodies[body["id"]] = {
                "body": body,
                "point_poly": point_poly,
                "past_actor_name": f"past-{body['id']}",
                "full_positions": full_positions,
                "color": color,
            }
        else:
            pos = np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
            marker = pv.PolyData(pos.reshape(1, 3))
            plotter.add_mesh(marker, color=color, point_size=_marker_size(body), render_points_as_spheres=True, name=f"marker-{body['id']}")
            plotter.add_point_labels(
                pos.reshape(1, 3), [body["name"]], font_size=12, text_color=color,
                shape=None, always_visible=True, show_points=False,
            )

    if legend_entries:
        plotter.add_legend(
            legend_entries, bcolor=(0.07, 0.06, 0.12), face="circle",
            size=(0.16, 0.03 * len(legend_entries) + 0.02), loc="lower left",
        )

    info_text_actor = plotter.add_text("", font_size=9, color=INK_PRIMARY, position="upper_right")

    def _update_info(t):
        lines = []
        timeline = scene.get("timeline")
        if timeline:
            lines.append(f"T+{t:.2f} {timeline['unitLabel']}")
        for entry in moving_bodies.values():
            body = entry["body"]
            pos = _position_at_time(body, t)
            dist = float(np.linalg.norm(pos))
            unit = scene["distanceUnit"]
            lines.append(f"{body['name']}: {dist:,.1f} {unit}")
        info_text_actor.set_text("upper_right", "\n".join(lines))

    def _on_time_change(t):
        for entry in moving_bodies.values():
            body = entry["body"]
            pos = _position_at_time(body, t)
            entry["point_poly"].points = pos.reshape(1, 3)

            idx = _trail_index_at_time(body, t)
            past_pts = np.vstack([entry["full_positions"][: max(idx, 1)], pos.reshape(1, 3)])
            new_past_poly = pv.MultipleLines(past_pts) if len(past_pts) > 1 else pv.MultipleLines(np.vstack([pos, pos]))
            plotter.add_mesh(
                new_past_poly, color=entry["color"], line_width=2.4, opacity=0.9,
                name=entry["past_actor_name"], pickable=False,
            )
        _update_info(t)
        plotter.render()

    timeline = scene.get("timeline")
    if timeline:
        plotter.add_slider_widget(
            _on_time_change,
            [timeline["min"], timeline["max"]],
            value=timeline["min"],
            title=f"Mission time ({timeline['unitLabel']})",
            color=INK_PRIMARY,
            style="modern",
            pointa=(0.28, 0.08),
            pointb=(0.85, 0.08),
            interaction_event="always",  # live-update while dragging, not just on release
        )
        for event in timeline.get("events", []):
            frac = (event["time"] - timeline["min"]) / (timeline["max"] - timeline["min"])
            x = 0.28 + frac * (0.85 - 0.28)
            plotter.add_text(
                "|", position=(x * window_size[0] - 4, 40), font_size=10, color=ACCENT, viewport=False,
            )
        _on_time_change(timeline["min"])

    plotter.camera_position = "iso"
    plotter.reset_camera()

    if off_screen:
        plotter.render()
        return plotter

    plotter.show(title=scene["title"])
    return None
