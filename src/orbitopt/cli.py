"""Command-line entry point for orbitopt -- launch the Mission Control app or
the single-scene 3D viewer directly, without going through examples/.

    orbitopt                    # Mission Control (the default)
    orbitopt app                # Mission Control
    orbitopt view               # single-scene viewer, solar system
    orbitopt view artemis2      # a built-in mission (see `orbitopt view --help`)
    orbitopt view scene.json    # any exported SceneData JSON file
    orbitopt view my_mission.yaml       # run (or load cached) a mission config, then view it
    orbitopt export goes -o goes.json   # cache a built-in mission's scene JSON
    orbitopt validate my_mission.yaml   # schema-check a mission config (no compute)
    orbitopt run my_mission.yaml        # run a mission config, write + verify its scene
    orbitopt launch-window kourou --inclination 51.64 --raan 100 --epoch 2026-11-08
                                        # daily launch windows (no compute stack)

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
from pathlib import Path

from orbitopt.missions import list_missions
from orbitopt.scene_format import read_scene


_CONFIG_SUFFIXES = (".yaml", ".yml")


def resolve_scene(name: str) -> dict:
    """Return a SceneData dict for a registered mission id (built-in or a
    third-party plugin -- see orbitopt.missions), a mission config YAML path
    (see orbitopt.mission_config), or a scene JSON file path."""
    from orbitopt.missions import get

    try:
        mission = get(name)
    except KeyError:
        if Path(name).suffix.lower() in _CONFIG_SUFFIXES:
            from orbitopt.mission_config import cached_scene_from_config

            config = _load_and_validate_config(name)
            print(f"Computing the {config['mission']['id']!r} scene from {name} "
                  "(cached after the first run)...")
            return cached_scene_from_config(config)
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
    missions = list_missions()
    width = max([16] + [len(m.id) for m in missions])  # a long plugin id must not break the column
    for mission in missions:
        print(f"{mission.id:<{width}} {mission.title}")


def _load_and_validate_config(path):
    """Shared by `validate`/`run`: parse+validate a mission config, printing
    a clean one-line error (not a traceback) on a schema/format mismatch --
    unlike the rest of this file, which lets exceptions surface raw, a
    readable pass/fail is the entire point of these two commands."""
    import jsonschema

    from orbitopt.mission_config import load_mission_config

    try:
        return load_mission_config(path)
    except (jsonschema.ValidationError, ValueError) as exc:
        print(f"INVALID: {path}\n  {exc}")
        raise SystemExit(1) from None


def _cmd_validate(args):
    config = _load_and_validate_config(args.config)
    mission = config["mission"]
    print(f"OK: {mission['id']} (kind: {mission['kind']})")


def _cmd_run(args):
    from orbitopt.pipeline import run_pipeline
    from orbitopt.verify.report import RESULT_PASSED, format_status, status_result

    config = _load_and_validate_config(args.config)
    if args.skip_verify:
        config.setdefault("verification", {})["enabled"] = False

    mission = config["mission"]
    print(f"Running {mission['id']!r} (kind: {mission['kind']})...")
    # The same unified pipeline (orbitopt.pipeline) the GUI and Python API
    # use: MissionConfig -> Constraints -> Search -> Optimize -> Refine ->
    # Verify -> Export, writing the scene, its verification report (task
    # #137) and the standardized pipeline result next to each other.
    result = run_pipeline(config, write_outputs=True, scene_path=args.out)
    scene = result.scene
    print(f"wrote {result.outputs['scene_file']} ({len(scene.get('bodies', []))} bodies)")
    if "verification_report" in result.outputs:
        doc = result.verification_document()
        print(("" if status_result(doc["status"]) == RESULT_PASSED else "WARNING: ")
              + f"wrote verification report {result.outputs['verification_report']} "
              f"({format_status(doc['status'])})")
    print(f"wrote pipeline result {result.outputs['pipeline_result']}")
    print(result.summary())
    # build_from_config appends its verification summary as trailing
    # subtitle lines -- surface it here too, so a non-convergence WARNING
    # is visible in the terminal, not just in the scene the UI shows later.
    for line in scene.get("subtitle", "").splitlines():
        if line.startswith("High-fidelity verification") or line.startswith("Verification report"):
            bad = "DID NOT CONVERGE" in line or (
                line.startswith("Verification report") and ": PASSED (" not in line)
            print(("WARNING: " if bad else "") + line)
    # Verification Gate (task #171): a failed / not_checked / best-effort
    # export is never reported as a success -- make that impossible to miss.
    if result.warnings:
        bar = "!" * 78
        print(bar, *(f"WARNING: {w}" for w in result.warnings), bar, sep="\n")
    print("verification gate: " + ("SUCCESS -- exported candidate passed verification" if result.success
                                   else f"NOT SUCCESSFUL (verification: {result.gate.result})"))


def _cmd_launch_window(args):
    from orbitopt import launch_windows as lw

    start = args.start or args.epoch
    if args.rla is not None or args.dla is not None:
        if args.rla is None or args.dla is None:
            raise SystemExit("--rla and --dla must be given together")
        windows = lw.asymptote_launch_windows(
            args.site, args.rla, args.dla, start, args.days, args.azimuth, args.step)
    else:
        if args.inclination is None:
            raise SystemExit("give --inclination (with --raan or --ltan), or --rla/--dla")
        if args.ltan is not None:
            plane = lw.sun_synchronous_plane(args.ltan, args.epoch, args.inclination)
        elif args.altitude_km is not None:
            plane = lw.OrbitPlane.from_altitude(args.inclination, args.raan, args.epoch, args.altitude_km)
        else:
            plane = lw.OrbitPlane(args.inclination, args.raan, lw.as_mjd2000(args.epoch))
        windows = lw.plane_launch_windows(
            args.site, plane, start, args.days, args.dogleg,
            respect_corridors=not args.ignore_corridors, azimuth_range_deg=args.azimuth,
            step_s=args.step)
    if args.json:
        print(json.dumps([w.to_dict() for w in windows], indent=2, ensure_ascii=False))
        return
    if windows:
        print(lw.format_windows(windows))
    else:
        print("no launch window in the requested interval")
    from orbitopt.data_provenance import format_data_sources
    from orbitopt.launch_sites import get_launch_site

    # Windows depend on the site's location and (unless overridden on the
    # command line) its range-safety azimuth corridors.
    fields = ["location"]
    if args.azimuth is None and not (args.rla is None and args.ignore_corridors):
        fields.append("azimuth_corridors")
    print()
    print(format_data_sources(get_launch_site(args.site).data_sources(fields)))


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
                      help=f"a mission id ({mission_ids}), path/to/mission.yaml or path/to/scene.json")

    export = sub.add_parser("export", help="write a mission's SceneData JSON to a file")
    export.add_argument("scene", help=f"a mission id ({mission_ids})")
    export.add_argument("-o", "--output", help="output path (default: <scene>.json)")

    sub.add_parser("missions", help="list every registered mission id (built-in + plugins)")

    validate = sub.add_parser("validate", help="schema-check a mission config file (no compute)")
    validate.add_argument("config", help="path to a mission config YAML/JSON file")

    run = sub.add_parser("run", help="validate, run, and export a mission config to a scene file")
    run.add_argument("config", help="path to a mission config YAML/JSON file")
    run.add_argument("-o", "--out", help="output scene path (default: the config's output.scene_file)")
    run.add_argument("--skip-verify", action="store_true",
                     help="skip the high-fidelity result verification (faster, less trustworthy)")

    lw = sub.add_parser("launch-window",
                        help="daily launch windows from a launch site into an orbit plane / asymptote")
    lw.add_argument("site", help="launch site id, name or alias (see orbitopt/data/launch_sites.yaml)")
    lw.add_argument("--inclination", type=float, help="target orbit inclination [deg]")
    lw.add_argument("--raan", type=float, default=0.0, help="target RAAN at --epoch [deg] (default 0)")
    lw.add_argument("--epoch", default="2000-01-01T00:00:00Z",
                    help="UTC epoch of --raan / --ltan, ISO-8601 or MJD2000")
    lw.add_argument("--altitude-km", type=float,
                    help="circular altitude; makes RAAN regress at the J2 rate")
    lw.add_argument("--ltan", type=float,
                    help="sun-synchronous plane: local time of ascending node [h] (overrides --raan)")
    lw.add_argument("--rla", type=float, help="departure asymptote right ascension [deg]")
    lw.add_argument("--dla", type=float, help="departure asymptote declination [deg]")
    lw.add_argument("--start", help="start of the search, UTC (default: --epoch)")
    lw.add_argument("--days", type=float, default=1.0, help="search span [days] (default 1)")
    lw.add_argument("--dogleg", type=float, default=0.0,
                    help="plane-change budget [deg]; 0 = instantaneous in-plane windows")
    lw.add_argument("--azimuth", type=float, nargs=2, action="append", metavar=("START", "END"),
                    help="allowed azimuth corridor, clockwise [deg]; repeatable "
                         "(default: the site's range-safety corridors)")
    lw.add_argument("--ignore-corridors", action="store_true",
                    help="plane windows: do not filter by the site's azimuth corridors")
    lw.add_argument("--step", type=float, default=60.0, help="scan step [s] (default 60)")
    lw.add_argument("--json", action="store_true", help="print JSON instead of a table")
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
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "launch-window":
        _cmd_launch_window(args)


if __name__ == "__main__":
    main()
