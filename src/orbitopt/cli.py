"""Command-line entry point for orbitopt -- launch the Mission Control app or
the single-scene 3D viewer directly, without going through examples/.

    orbitopt                    # Mission Control (the default)
    orbitopt app                # Mission Control
    orbitopt view               # single-scene viewer, solar system
    orbitopt view artemis2      # a built-in mission (see `orbitopt view --help`)
    orbitopt view scene.json    # any exported SceneData JSON file
    orbitopt export goes -o goes.json   # cache a built-in mission's scene JSON

The same commands work as ``python -m orbitopt ...``. Installed as the
``orbitopt`` and ``orbitopt-app`` console scripts (see pyproject.toml), so
after ``pip install -e .`` they are on PATH. The examples/ scripts remain as
annotated references; they are no longer the way to start the app.

Built-in and plugin missions both come from ``orbitopt.missions`` -- see that
module for how a third-party package adds its own without touching this file.
"""
from __future__ import annotations

import argparse
import json

from orbitopt.missions import list_missions
from orbitopt.scene_format import read_scene


def resolve_scene(name: str) -> dict:
    """Return a SceneData dict for a registered mission id (built-in or a
    third-party plugin -- see orbitopt.missions) or a JSON file path."""
    from orbitopt.missions import get

    try:
        mission = get(name)
    except KeyError:
        return read_scene(name)
    if mission.id != "solar-system":
        print(f"Computing the {mission.id!r} scene -- this can take a few seconds...")
    return mission.load()


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


def _cmd_missions(_args):
    for mission in list_missions():
        print(f"{mission.id:<16} {mission.title}")


def build_parser() -> argparse.ArgumentParser:
    mission_ids = ", ".join(m.id for m in list_missions())

    parser = argparse.ArgumentParser(
        prog="orbitopt",
        description="orbitopt -- orbital trajectory optimization + Mission Control 3D viewer",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("app", help="launch the Mission Control desktop app (default)")

    view = sub.add_parser("view", help="open the single-scene 3D viewer")
    view.add_argument("scene", nargs="?", default="solar-system",
                      help=f"a mission id ({mission_ids}) or path/to/scene.json")

    export = sub.add_parser("export", help="write a mission's SceneData JSON to a file")
    export.add_argument("scene", help=f"a mission id ({mission_ids})")
    export.add_argument("-o", "--output", help="output path (default: <scene>.json)")

    sub.add_parser("missions", help="list every registered mission id (built-in + plugins)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in (None, "app"):
        _cmd_app(args)
    elif args.command == "view":
        _cmd_view(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "missions":
        _cmd_missions(args)


if __name__ == "__main__":
    main()
