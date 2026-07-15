"""Smoke tests for the PyVista scene viewer: renders both real scenes
off-screen and checks nothing throws and the slider-driven position update
actually moves the trajectory (a real bug once: the slider's default
interaction_event='end' meant a naive test of the 'InteractionEvent' path
silently never called back -- this pins the fix, interaction_event='always').
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.viz.pv_viewer import show_scene
from orbitopt.viz.solar_system import export_solar_system_data


def test_show_scene_offscreen_solar_system():
    scene = export_solar_system_data(n_samples=60)
    plotter = show_scene(scene, off_screen=True)
    img = plotter.screenshot(return_img=True)
    assert img.shape[0] > 0 and img.shape[1] > 0
    plotter.close()


@pytest.mark.slow
def test_mission_timeline_slider_updates_spacecraft_position():
    from orbitopt.viz.mission_timeline import compute_and_export_mission

    scene = compute_and_export_mission()
    plotter = show_scene(scene, off_screen=True)

    slider = plotter.slider_widgets[0]
    rep = slider.GetRepresentation()

    marker_name = "marker-spacecraft"
    initial_points = plotter.renderer.actors[marker_name].mapper.dataset.points.copy()

    perilune_day = next(e["time"] for e in scene["timeline"]["events"] if "approach" in e["label"].lower())
    rep.SetValue(perilune_day)
    slider.InvokeEvent("InteractionEvent")
    plotter.render()

    updated_points = plotter.renderer.actors[marker_name].mapper.dataset.points
    assert not np.allclose(initial_points, updated_points), (
        "spacecraft marker did not move when the timeline slider changed -- "
        "check that add_slider_widget uses interaction_event='always' (or "
        "that whatever test/consumer triggers the matching VTK event)."
    )

    plotter.close()
