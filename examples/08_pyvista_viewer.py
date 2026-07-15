"""Open the general-purpose PyVista 3D viewer on any scene -- a real desktop
window (rotate/pan/zoom with the mouse, scrub the timeline slider if the
scene has one), no browser involved.

Run with a built-in scenario:
    python examples/08_pyvista_viewer.py solar-system
    python examples/08_pyvista_viewer.py artemis2

Or load any previously-exported SceneData JSON file (see
orbitopt.viz.scene / orbitopt.viz.solar_system / orbitopt.viz.mission_timeline
for how one gets built -- any future scene exporter that produces the same
document shape works here with no changes to this script or to
orbitopt.viz.pv_viewer):
    python examples/08_pyvista_viewer.py path/to/some_scene.json
"""
from __future__ import annotations

import sys

from orbitopt.viz.pv_viewer import load_scene, show_scene


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)

    arg = sys.argv[1]

    if arg == "solar-system":
        from orbitopt.viz.solar_system import export_solar_system_data
        scene = export_solar_system_data()
    elif arg == "artemis2":
        from orbitopt.viz.mission_timeline import compute_and_export_mission
        print("Solving the free-return trajectory and propagating the mission -- a few seconds...")
        scene = compute_and_export_mission()
    else:
        scene = load_scene(arg)

    show_scene(scene)


if __name__ == "__main__":
    main()
