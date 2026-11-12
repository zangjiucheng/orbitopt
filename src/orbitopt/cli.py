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
    orbitopt flight validate target.yaml # schema-check a Flight Lab flight target ...
    orbitopt flight validate runs/run-1  # ... or a FlightResult run directory
    orbitopt flight export-target --site kourou --orbit 400x400@51.6 \
        --date 2026-11-10 -o target.yaml  # build a FlightTarget from site/orbit/window
    orbitopt flight to-scene runs/run-1 -o scene.json  # run dir -> viewer scene

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
        print(f"Loading the {mission.id!r} scene (precomputed or cached, else computed -- "
              "that can take minutes)...")
    from orbitopt.scene_cache import ENV_RECOMPUTE, pop_shipped_marker

    scene = mission.load()
    if pop_shipped_marker(scene) is not None:
        print(f"Note: this is the shipped precomputed {mission.id!r} scene, made by different orbitopt code. "
              f"Set {ENV_RECOMPUTE}={mission.id} to recompute it with yours.")
    return scene


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


def _cmd_flight(args):
    """`orbitopt flight ...`: the Flight Lab tools (orbitopt.flightlab) --
    readable pass/fail or a written scene JSON, no compute stack, no GUI
    imports (orbitopt.flightlab.scene_export builds the scene itself).
    `orbitopt flight validate <target.yaml|run_dir>`: the contract check
    (orbitopt.flightlab.contract). `orbitopt flight export-target`: build a
    FlightTarget YAML (FL-003). `orbitopt flight simulate`: Gate B (FL-006)
    -- ascend, check, re-fly, honest verdict. `orbitopt flight optimize`:
    FL-202 -- pygmo ascent optimization, best flight written as a run +
    optimization.json (conda-forge pygmo only)."""
    if getattr(args, "flight_command", None) == "simulate":
        _cmd_flight_simulate(args)
        return
    if getattr(args, "flight_command", None) == "montecarlo":
        _cmd_flight_montecarlo(args)
        return
    if getattr(args, "flight_command", None) == "optimize":
        _cmd_flight_optimize(args)
        return
    if args.flight_command == "to-scene":
        return _cmd_flight_to_scene(args)
    if getattr(args, "flight_command", None) == "export-target":
        _cmd_flight_export_target(args)
        return

    import jsonschema

    from orbitopt.flightlab.contract import validate_path

    try:
        summary = validate_path(args.path)
    except (jsonschema.ValidationError, ValueError) as exc:
        print(f"INVALID: {args.path}\n  {exc}")
        raise SystemExit(1) from None
    print(f"OK: {summary}")


def _cmd_flight_export_target(args):
    """`orbitopt flight export-target --site <id> --orbit <peri_km>x<apo_km>@<inc_deg>
    [--raan <deg>] (--date <UTC> | --window <UTC>[/days]) -o target.yaml`:
    the CLI front of orbitopt.flightlab.target_export.build_target -- a
    readable error + exit 1 on an unknown site, an infeasible inclination or
    an empty launch-window search, never a traceback."""
    import jsonschema

    from orbitopt.flightlab.contract import target_digest
    from orbitopt.flightlab.target_export import build_target, parse_orbit_spec, parse_window_spec, save_target

    if args.date is not None:
        anchor, window_days = args.date, None
    else:
        anchor, window_days = parse_window_spec(args.window)
    try:
        peri_m, apo_m, inc = parse_orbit_spec(args.orbit)
        kw = {} if window_days is None else {"window_days": window_days}
        target = build_target(args.site, peri_m, apo_m, inc, raan_deg=args.raan,
                              date_utc=anchor, **kw)
        save_target(target, args.output)
    except KeyError as exc:
        print(f"ERROR: {exc.args[0] if exc.args else exc}")  # KeyError repr would wrap the message in quotes
        raise SystemExit(1) from None
    except (ValueError, jsonschema.ValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
    raan_note = (f", RAAN {target.target_orbit.raan_deg:g} deg"
                 if target.target_orbit.raan_deg is not None else ", RAAN unconstrained")
    print(f"wrote {args.output}: {target.mission.id} @ {target.mission.epoch_utc} "
          f"(site {target.launch_site.name}{raan_note})")
    print("validated against flight-target-1.0.json before writing "
          f"(digest {target_digest(target.to_dict())[:12]}...)")


def _cmd_flight_to_scene(args):
    """`orbitopt flight to-scene <run_dir> -o scene.json`: the FL-005 scene
    importer -- a FlightResult run dir becomes a scene-1.0 JSON that
    `orbitopt view` (or Mission Control) opens as a multi-body Earth scene.
    A run that fails the contract (or carries a frame scene export cannot
    convert to GCRS) is rejected and no file is written: never render a
    suspect run."""
    import jsonschema

    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.scene_export import run_to_scene

    try:
        scene = run_to_scene(read_run(args.run_dir), earth_texture=not args.no_earth_texture)
    except (jsonschema.ValidationError, ValueError) as exc:
        print(f"REJECTED: {args.run_dir}\n  {exc}")
        raise SystemExit(1) from None
    except ImportError as exc:
        # A non-GCRS track (e.g. LAUNCH_ECI) needs the FL-002 frame bridge,
        # whose numpy/astropy dependencies a base install does not carry.
        print(f"ERROR: {args.run_dir} has a track that needs the flightlab frame bridge "
              f"to convert to GCRS, but its optional dependency is missing ({exc}). "
              "Install numpy and astropy (or use the conda env), or export an all-GCRS run.")
        raise SystemExit(1) from None
    out = args.output or f"{scene['flightRun']['runId']}.json"
    with open(out, "w") as handle:
        json.dump(scene, handle, indent=2)
    n_bodies = len(scene["bodies"]) - 1  # the Earth is not a run body
    n_events = len(scene["timeline"].get("events", []))
    print(f"wrote {out}: {n_bodies} bodies, {n_events} events "
          f"(open with: orbitopt view {out})")


def _cmd_flight_simulate(args):
    """`orbitopt flight simulate <target.yaml> --vehicle <v.yaml> -o runs/<id>
    [--scene]`: the Gate-B pipeline (FL-006) -- OpenTsiolkovsky ascent,
    mission-constraint checks, tudatpy insertion re-fly -- printing the three
    independent verdicts and every failing reason code. The exit code reports
    pipeline health (0 = a run was produced and verified, whatever the
    verdict; 1 = it could not be produced), never the verdict itself: an
    honest red verdict exits 0. `--no-refly` skips the physics re-fly
    (physics_verified then stays "not assessed")."""
    from pathlib import Path

    import jsonschema

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.simulate import simulate

    try:
        target = load_target(args.target)
        run = simulate(target, Path(args.vehicle), Path(args.output),
                       refly=not args.no_refly, timeout_s=args.timeout)
    except (OSError, ValueError, KeyError, RuntimeError, jsonschema.ValidationError) as exc:
        # RuntimeError covers OTBinaryMissingError and every adapter subprocess failure
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    def word(v):
        return "not assessed" if v is None else ("PASSED" if v else "FAILED")

    v = run.verification
    print(f"run {run.run_id} -> {args.output}")
    print(f"  optimizer_converged:        {word(v.optimizer_converged)}"
          "  (E1 flies a hand-tuned program; no optimizer ran)")
    print(f"  physics_verified:           {word(v.physics_verified)}")
    print(f"  mission_constraints_passed: {word(v.mission_constraints_passed)}")
    gate = next((c for c in v.checks if c.id == "recovery-landing"), None)
    if gate is not None:
        print(f"  recovery (FL-304):              {'LANDED' if gate.passed else 'NOT LANDED'}"
              + (f" [{gate.reason_code}]" if gate.reason_code else ""))
    for c in v.checks:
        if not c.passed:
            value = "-" if c.value is None else f"{c.value:.6g} {c.unit}".strip()
            limit = "-" if c.limit is None else f"{c.limit:.6g} {c.unit}".strip()
            tag = "SKIP" if c.id.endswith("-not-assessed") else "FAIL"
            print(f"  {tag} {c.id}: {value} vs limit {limit}"
                  + (f" [{c.reason_code}]" if c.reason_code else ""))
    if args.scene:
        from orbitopt.flightlab.scene_export import run_to_scene

        scene = run_to_scene(run)
        scene_path = Path(args.output) / "scene.json"
        scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
        print(f"  scene: {scene_path} ({len(scene['bodies'])} bodies "
              "-- open with: orbitopt view that file)")


def _cmd_flight_montecarlo(args):
    """`orbitopt flight montecarlo <target.yaml> --vehicle <v.yaml> -n N
    --seed S [-o montecarlo.json] [--jobs J]`: FL-304 -- fly the ascent once,
    then re-plan and re-fly ``n`` dispersed recovery samples (thrust, Isp,
    atmosphere, drag, separation state) in a process pool, printing the
    success rate, the reason-code histogram and the landing dispersion.
    Deterministic per seed: the same seed and setup write a byte-identical
    JSON regardless of ``--jobs``. Exit code reports pipeline health, like
    `flight simulate`."""
    from pathlib import Path

    import jsonschema

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.recovery.montecarlo import run_monte_carlo
    from orbitopt.flightlab.simulate import simulate

    out = Path(args.out)
    ascent_dir = Path(args.ascent_run_dir) if args.ascent_run_dir else out.with_suffix(".ascent")
    try:
        target = load_target(args.target)
        ascent = simulate(target, Path(args.vehicle), ascent_dir,
                          refly=not args.no_refly, timeout_s=args.timeout,
                          run_recovery=False)
        mc = run_monte_carlo(target, Path(args.vehicle), ascent,
                             seed=args.seed, n=args.n, workers=args.jobs,
                             out_path=out)
    except (OSError, ValueError, KeyError, RuntimeError, jsonschema.ValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    print(f"montecarlo: n={mc.n} seed={mc.seed} -> {out}")
    print(f"  success_rate: {mc.success_rate:.3f} "
          f"({sum(1 for s in mc.samples if s.get('outcome') == 'landed')}/{mc.n} landed in limits)")
    for code, count in mc.reason_histogram.items():
        print(f"  {count:>4}  {code}")
    ell = mc.ellipse
    if ell.get("state") == "assessed":
        print(f"  ellipse (1-sigma, site ENU): along {ell['sigma_along_m']:.1f} m x "
              f"cross {ell['sigma_cross_m']:.1f} m @ {ell['bearing_deg']:.1f} deg "
              f"(mean {ell['mean_east_m']:+.1f} E {ell['mean_north_m']:+.1f} N, n={ell['n']})")
    else:
        print(f"  ellipse: {ell['state']} (n={ell.get('n', 0)})")
    for key, st in mc.landing_stats.items():
        if st.get("n"):
            print(f"  {key}: p50 {st['p50']:.3g}  p95 {st['p95']:.3g} (n={st['n']})")


def _cmd_flight_optimize(args):
    """`orbitopt flight optimize <target.yaml> [--vehicle <v.yaml>] -o runs/<id>`:
    the FL-202 ascent optimization loop (pygmo, conda-forge only) -- global
    search over the FL-201 parametric ascent, SLSQP local refinement, and the
    winner re-flown through the Gate-B pipeline. Prints the same three
    independent verdicts as `flight simulate` plus the convergence report
    written to `<RUN_DIR>/optimization.json`. Exit code: 0 when a run was
    produced (whatever the verdict), 1 when nothing flyable came out or the
    compute stack is missing -- always with an actionable message, never a
    traceback."""
    from pathlib import Path

    import jsonschema

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.optimize.run import OptimizeAbort, optimize_ascent
    from orbitopt.flightlab.vehicle_source import resolve_vehicle_path

    try:
        target = load_target(args.target)
        vehicle = resolve_vehicle_path(args.vehicle, target, args.target)
    except (OSError, ValueError, KeyError, jsonschema.ValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
    try:
        result = optimize_ascent(
            target, vehicle, Path(args.output), target_path=args.target,
            objective=args.objective, payload_kg=args.payload_kg,
            seed=args.seed, algorithm=args.algorithm,
            generations=args.generations, population=args.population,
            workers=args.workers or None, local=not args.no_local,
            refly=not args.no_refly, timeout_s=args.timeout,
            derive_window=not args.fixed_epoch, window_days=args.window_days,
            sensitivity_offsets_s=tuple(args.sensitivity_offsets))
    except OptimizeAbort as exc:
        print(f"ERROR: {exc}")
        for code in exc.reason_codes:
            print(f"  {code}")
        raise SystemExit(1) from None
    except (RuntimeError, OSError, ValueError, KeyError,
            jsonschema.ValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    run = result["run"]
    best = result["best_fitness"]

    def word(v):
        return "not assessed" if v is None else ("PASSED" if v else "FAILED")

    v = run.verification
    print(f"run {run.run_id} -> {args.output}")
    print(f"  optimizer_converged:        {word(v.optimizer_converged)}"
          f"  ({result['status']})")
    print(f"  physics_verified:           {word(v.physics_verified)}")
    print(f"  mission_constraints_passed: {word(v.mission_constraints_passed)}")
    print(f"  objective {result['objective']}: {-best['objective']:.6g} kg"
          f"  [{result['algorithm']['global']}, {result['evaluations']} "
          f"evaluations, local: {result['algorithm']['local']}]")
    raan = result.get("raan")
    if raan:
        print(f"  raan (GCRS):                {raan['achieved_raan_deg']:.3f} deg vs "
              f"{raan['target_raan_deg']:.3f} deg, error {raan['error_deg']:+.3f} deg "
              f"(tolerance {raan['tolerance_deg']:g} deg): {word(raan['passed'])}")
        for row in raan.get("window_sensitivity", ()):
            print(f"    liftoff {row['liftoff_offset_s']:+g} s "
                  f"({row['liftoff_epoch_utc']}): RAAN {row['achieved_raan_deg']:.3f} deg, "
                  f"error {row['error_deg']:+.3f} deg "
                  f"{'PASS' if row['passed_within_tolerance'] else 'FAIL'}")
    for code in result["reason_codes"]:
        print(f"  {code}")
    for c in v.checks:
        if not c.passed:
            value = "-" if c.value is None else f"{c.value:.6g} {c.unit}".strip()
            limit = "-" if c.limit is None else f"{c.limit:.6g} {c.unit}".strip()
            tag = "SKIP" if c.id.endswith("-not-assessed") else "FAIL"
            print(f"  {tag} {c.id}: {value} vs limit {limit}"
                  + (f" [{c.reason_code}]" if c.reason_code else ""))
    print(f"  report: {Path(args.output) / 'optimization.json'}")


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

    flight = sub.add_parser("flight", help="Flight Lab contract tools (FlightTarget / FlightResult)")
    flight_sub = flight.add_subparsers(dest="flight_command", required=True)
    flight_validate = flight_sub.add_parser(
        "validate", help="schema-check a flight target YAML or a FlightResult run directory (no compute)")
    flight_validate.add_argument("path",
                                 help="a FlightTarget YAML/JSON file, or a run dir (manifest.json + trajectories/)")
    flight_export = flight_sub.add_parser(
        "export-target",
        help="build a schema-valid FlightTarget YAML from a launch site, an orbit spec and a window (FL-003)")
    flight_export.add_argument("--site", required=True, metavar="ID",
                               help="launch-site id from orbitopt's site database "
                                    "(cape-canaveral, vandenberg, starbase, kourou, jiuquan, wenchang, baikonur)")
    flight_export.add_argument("--orbit", required=True, metavar="PERIxAPO@INC",
                               help="target orbit: <peri_km>x<apo_km>@<inclination_deg>, e.g. 400x400@51.6")
    flight_export.add_argument("--raan", type=float, default=None, metavar="DEG",
                               help="constrain the target RAAN [deg]; the epoch becomes the first in-plane "
                                    "node pass (J2 regression included) inside the search window")
    flight_when = flight_export.add_mutually_exclusive_group(required=True)
    flight_when.add_argument("--date", metavar="UTC",
                             help="earliest-liftoff anchor, e.g. 2026-11-10 or 2026-11-10T12:00:00Z")
    flight_when.add_argument("--window", metavar="UTC[/DAYS]",
                             help="same anchor with an explicit node-search window in days, e.g. 2026-11-10/5")
    flight_export.add_argument("-o", "--output", required=True, metavar="PATH",
                               help="output FlightTarget YAML path (written only after schema validation)")
    flight_scene = flight_sub.add_parser(
        "to-scene", help="convert a FlightResult run dir to a scene JSON (opens with `orbitopt view`)")
    flight_scene.add_argument("run_dir", help="a run dir (manifest.json + trajectories/)")
    flight_scene.add_argument("-o", "--output", help="output scene JSON path (default: <run_id>.json)")
    flight_scene.add_argument("--no-earth-texture", action="store_true",
                              help="draw Earth as a point marker instead of the textured globe")
    flight_sim = flight_sub.add_parser(
        "simulate",
        help="Gate B (FL-006): fly a FlightTarget with a vehicle file, verify constraints "
             "and re-fly the insertion orbit with tudatpy (needs the OpenTsiolkovsky binary)")
    flight_sim.add_argument("target", help="a FlightTarget YAML (see `flight export-target`)")
    flight_sim.add_argument("--vehicle", required=True, metavar="PATH",
                            help="flightlab vehicle YAML (OT stage configs + attitude program)")
    flight_sim.add_argument("-o", "--output", required=True, metavar="RUN_DIR",
                            help="output run directory (FlightResult v1)")
    flight_sim.add_argument("--scene", action="store_true",
                            help="also write <RUN_DIR>/scene.json (FL-005; opens with `orbitopt view`)")
    flight_sim.add_argument("--no-refly", action="store_true",
                            help="skip the tudatpy insertion re-fly (physics_verified stays not-assessed)")
    flight_sim.add_argument("--timeout", type=float, default=300.0, metavar="S",
                            help="OpenTsiolkovsky subprocess timeout [s] (default 300)")
    flight_mc = flight_sub.add_parser(
        "montecarlo",
        help="FL-304: Monte-carlo the booster recovery of a flight target "
             "(ascent once, then n dispersed closed-loop recovery samples; "
             "needs the OpenTsiolkovsky binary)")
    flight_mc.add_argument("target", help="a FlightTarget YAML (see `flight export-target`)")
    flight_mc.add_argument("--vehicle", required=True, metavar="PATH",
                           help="flightlab vehicle YAML (OT stage configs + booster section)")
    flight_mc.add_argument("-n", type=int, default=8, metavar="N",
                           help="number of samples (default 8)")
    flight_mc.add_argument("--seed", type=int, default=1, metavar="S",
                           help="master seed; the statistics are byte-reproducible per seed (default 1)")
    flight_mc.add_argument("--jobs", type=int, default=1, metavar="J",
                           help="process-pool width (default 1 = in-process serial)")
    flight_mc.add_argument("-o", "--out", default="montecarlo.json", metavar="PATH",
                           help="output JSON path (default: montecarlo.json)")
    flight_mc.add_argument("--ascent-run-dir", default=None, metavar="RUN_DIR",
                           help="where the (single) ascent run is written "
                                "(default: <out>.ascent/)")
    flight_mc.add_argument("--no-refly", action="store_true",
                           help="skip the tudatpy insertion re-fly of the ascent run")
    flight_mc.add_argument("--timeout", type=float, default=300.0, metavar="S",
                            help="OpenTsiolkovsky subprocess timeout [s] (default 300)")
    flight_opt = flight_sub.add_parser(
        "optimize",
        help="FL-202: optimize the parametric ascent toward a FlightTarget (pygmo; "
             "conda-forge only) and write the best flight as a run + optimization.json")
    flight_opt.add_argument("target", help="a FlightTarget YAML (see `flight export-target`)")
    flight_opt.add_argument("--vehicle", metavar="PATH", default=None,
                            help="vehicle YAML (default: resolve target.vehicle.source "
                                 "against the target dir, then package data)")
    flight_opt.add_argument("--objective", choices=("payload", "reserve", "booster_reserve"),
                            default="payload", help="objective to maximize (default payload)")
    flight_opt.add_argument("--payload-kg", type=float, default=0.0, metavar="KG",
                            help="fixed payload for the reserve objectives (default 0)")
    flight_opt.add_argument("--seed", type=int, default=42,
                            help="RNG seed -- fixed seed gives byte-identical results (default 42)")
    flight_opt.add_argument("-o", "--output", required=True, metavar="RUN_DIR",
                            help="output run directory (FlightResult v1 + optimization.json)")
    flight_opt.add_argument("--algorithm", choices=("cmaes", "pso_gen", "sade"), default="cmaes",
                            help="global search algorithm (default cmaes; sade is serial)")
    flight_opt.add_argument("--generations", type=int, default=60, metavar="G",
                            help="global search generations (default 60)")
    flight_opt.add_argument("--population", type=int, default=48, metavar="N",
                            help="population / swarm size (default 48)")
    flight_opt.add_argument("--workers", type=int, default=0, metavar="N",
                            help="parallel fitness processes (default: up to 8 cores)")
    flight_opt.add_argument("--no-local", action="store_true",
                            help="skip the SLSQP local refinement")
    flight_opt.add_argument("--no-refly", action="store_true",
                            help="skip the tudatpy insertion re-fly of the winner")
    flight_opt.add_argument("--timeout", type=float, default=300.0, metavar="S",
                            help="OpenTsiolkovsky subprocess timeout [s] (default 300)")
    flight_opt.add_argument("--window-days", type=float, default=14.0, metavar="D",
                            help="RAAN-constrained only: in-plane epoch search horizon "
                                 "[days] from the target epoch (default 14)")
    flight_opt.add_argument("--fixed-epoch", action="store_true",
                            help="RAAN-constrained only: fly the epoch the target file "
                                 "carries, do not derive the in-plane liftoff epoch (FL-203)")
    flight_opt.add_argument("--sensitivity-offsets", type=float, nargs="*",
                            default=[-60.0, 60.0], metavar="S",
                            help="RAAN-constrained only: liftoff offsets [s] the winner is "
                                 "re-flown at for the RAAN window-sensitivity report "
                                 "(default: -60 60; pass no values to skip)")
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
    elif args.command == "flight":
        _cmd_flight(args)


if __name__ == "__main__":
    main()
