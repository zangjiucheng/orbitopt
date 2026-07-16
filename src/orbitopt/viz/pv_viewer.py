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

A body with `texture`+`radius` set (see orbitopt.viz.scene.TEXTURE and
scene_renderer._sphere_actor) renders as a real, textured 3D sphere instead
of a marker. Everything else renders as a VTK point sprite
(`render_points_as_spheres`), which VTK draws at a constant *screen-pixel*
size regardless of camera distance -- this is what a hand-rolled "constant
apparent size" scheme in a from-scratch 3D renderer has to reimplement (and,
in an earlier Three.js-based iteration of this viewer, got wrong: the Sun's
marker stayed large enough in world-space to swallow Mercury's entire orbit
even fully zoomed in). Using VTK's own point rendering sidesteps that whole
class of bug for free.

The 3D content itself (orbits, trails, markers/spheres, set_time) is
SceneRenderer's job, not reimplemented here -- see scene_renderer.py's own
docstring. This module's job is the VTK 2D-overlay chrome around that
content (title/subtitle text, legend, timeline slider, per-body distance
readout) that a script-based viewer wants and the Qt-based Mission Control
app (app.py) builds with real Qt widgets instead.
"""
from __future__ import annotations

import pyvista as pv

from orbitopt.scene_format import read_scene
from orbitopt.viz.scene_renderer import SceneRenderer

BACKGROUND = "#06050c"
INK_PRIMARY = "#f4f2ea"
INK_SECONDARY = "#a9a6bb"
ACCENT = "#e8a23e"


def load_scene(path) -> dict:
    """Read + validate a SceneData JSON document (as produced by any
    orbitopt.viz.* exporter) from disk. A thin, viewer-side name for
    orbitopt.scene_format.read_scene -- kept so existing callers importing
    load_scene from here don't need to change."""
    return read_scene(path)


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

    # The 3D content (orbits, trails, textured-sphere or point-marker
    # bodies) is SceneRenderer's job -- same renderer the Qt app uses, see
    # this module's docstring -- this function only adds the VTK 2D-overlay
    # chrome (title, legend, slider, info text) around it.
    renderer = SceneRenderer(plotter)
    renderer.load(scene)

    legend_entries = [[body["name"], body["color"]] for body in scene["bodies"]]
    moving_body_ids = [body["id"] for body in scene["bodies"] if "trail" in body]

    if legend_entries:
        plotter.add_legend(
            legend_entries, bcolor=(0.07, 0.06, 0.12), face="circle",
            size=(0.16, 0.03 * len(legend_entries) + 0.02), loc="lower left",
        )

    info_text_actor = plotter.add_text("", font_size=9, color=INK_PRIMARY, position="upper_right")
    body_names = {body["id"]: body["name"] for body in scene["bodies"]}

    def _update_info(t, distances):
        lines = []
        timeline = scene.get("timeline")
        if timeline:
            lines.append(f"T+{t:.2f} {timeline['unitLabel']}")
        unit = scene["distanceUnit"]
        for body_id in moving_body_ids:
            lines.append(f"{body_names[body_id]}: {distances[body_id]:,.1f} {unit}")
        info_text_actor.set_text("upper_right", "\n".join(lines))

    def _on_time_change(t):
        distances = renderer.set_time(t)  # moves every trailed body + regrows its trail
        _update_info(t, distances)
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
