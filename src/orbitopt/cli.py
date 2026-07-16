"""Command-line entry point for orbitopt -- launch the Mission Control app or
the single-scene 3D viewer directly, without going through examples/.

    orbitopt                    # Mission Control (the default)
    orbitopt app                # Mission Control
    orbitopt view               # single-scene viewer, solar system
    orbitopt view artemis2      # a built-in scene (solar-system | artemis2 | goes)
    orbitopt view scene.json    # any exported SceneData JSON file
    orbitopt export goes -o goes.json   # cache a built-in scene's JSON

The same commands work as ``python -m orbitopt ...``. Installed as the
``orbitopt`` and ``orbitopt-app`` console scripts (see pyproject.toml), so after
``pip install -e .`` they are on PATH. The examples/ scripts remain as annotated
references; they are no longer the way to start the app.
"""
from __future__ import annotations

import argparse
import importlib
import json

# built-in scene name -> (module, factory function). The Mission Control app has
# the same set in viz.app._builtin_missions; kept small and explicit rather than
# shared to avoid importing Qt just to name a scene.
_BUILTIN_SCENES = {
    "solar-system": ("orbitopt.viz.solar_system", "export_solar_system_data"),
    "artemis2": ("orbitopt.viz.mission_timeline", "compute_and_export_mission"),
    "goes": ("orbitopt.viz.geo_raising", "compute_and_export_geo_mission"),
}


def resolve_scene(name: str) -> dict:
    """Return a SceneData dict for a built-in scene name or a JSON file path."""
    if name in _BUILTIN_SCENES:
        module_name, func_name = _BUILTIN_SCENES[name]
        if name != "solar-system":
            print(f"Computing the '{name}' scene -- this can take a few seconds...")
        return getattr(importlib.import_module(module_name), func_name)()
    from orbitopt.viz.pv_viewer import load_scene
    return load_scene(name)


def _cmd_app(_args):
    from orbitopt.viz.app import main as app_main
    app_main()


def _cmd_view(args):
    from orbitopt.viz.pv_viewer import show_scene
    show_scene(resolve_scene(args.scene))


def _cmd_export(args):
    scene = resolve_scene(args.scene)
    out = args.output or f"{args.scene}.json"
    with open(out, "w") as handle:
        json.dump(scene, handle, indent=2)
    print(f"wrote {out} ({len(scene.get('bodies', []))} bodies)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbitopt",
        description="orbitopt -- orbital trajectory optimization + Mission Control 3D viewer",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("app", help="launch the Mission Control desktop app (default)")

    view = sub.add_parser("view", help="open the single-scene 3D viewer")
    view.add_argument("scene", nargs="?", default="solar-system",
                      help="solar-system | artemis2 | goes | path/to/scene.json")

    export = sub.add_parser("export", help="write a built-in scene's SceneData JSON to a file")
    export.add_argument("scene", help="solar-system | artemis2 | goes")
    export.add_argument("-o", "--output", help="output path (default: <scene>.json)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in (None, "app"):
        _cmd_app(args)
    elif args.command == "view":
        _cmd_view(args)
    elif args.command == "export":
        _cmd_export(args)


if __name__ == "__main__":
    main()
