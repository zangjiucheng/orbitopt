"""Open the general-purpose PyVista 3D viewer on any scene -- a real desktop
window (rotate/pan/zoom with the mouse, scrub the timeline slider if the
scene has one), no browser involved.

Prefer the installed console command, which does exactly this:

    orbitopt view solar-system
    orbitopt view artemis2
    orbitopt view goes
    orbitopt view path/to/some_scene.json

This script is kept as a thin reference around orbitopt.cli.resolve_scene +
orbitopt.viz.pv_viewer.show_scene; any scene exporter that produces the same
SceneData document (see orbitopt.viz.scene) works here with no changes.
"""
from __future__ import annotations

import sys

from orbitopt.cli import resolve_scene
from orbitopt.viz.pv_viewer import show_scene


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    show_scene(resolve_scene(sys.argv[1]))


if __name__ == "__main__":
    main()
