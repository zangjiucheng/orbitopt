"""OpenTsiolkovsky v1.0.0 ascent adapter (FL-004): FlightTarget in, FlightRun out.

OpenTsiolkovsky (https://github.com/istellartech/OpenTsiolkovsky, Rust, MIT;
see the repository NOTICE) is an EXTERNAL pinned binary -- never vendored
into this GPL tree, never built here. This adapter generates an OT config
JSON from a :class:`~orbitopt.flightlab.contract.FlightTarget` plus a vehicle
YAML (and an optional attitude program), validates that config against OT's
own JSON Schema (the output of ``openTsiolkovsky-cli schema``, cached in this
package since FL-004), runs ``openTsiolkovsky-cli run <cfg> -o <dir> --json``
as a subprocess under a timeout, and converts the outputs (``trajectory.csv``,
``summary.json``, ``bodies/*.csv``) into a FlightResult v1 run directory via
``contract.write_run``.

The frame of OT's ``pos_eci_*`` / ``vel_eci_*`` CSV columns, verified in the
v1.0.0 source (``src/physics/coordinates.rs``): these are **not** J2000/GCRS.
OT integrates in a launch-aligned pseudo-inertial frame whose axes equal the
ECEF frame at the launch epoch (its ECI state at ``t=0`` is the ECEF state
unrotated, and it propagates Earth rotation only as ``Rz(Omega_E * t)``) --
exactly the contract's ``LAUNCH_ECI``. This adapter therefore reads those
columns as ``LAUNCH_ECI`` and passes every track through
:func:`orbitopt.flightlab.frames.to_gcrs` before writing the manifest as
``GCRS``; a cross-check that survives this conversion is that |r| is a
rotation invariant, asserted against OT's own summary in the slow tests.

Event mapping (OT ``summary.json`` events -> contract kinds; unmapped OT
kinds -- ``mach1``, ``rail_exit``, ``apogee`` -- survive in the raw copy, see
below): ``ignition``/``liftoff``/``max_q`` pass through; stage-1 ``burnout``
becomes ``meco`` and any later-stage ``burnout`` ``seco``; ``separation``
passes through; a fairing-class ``jettison`` becomes ``separation`` with
``data.ot_event == "jettison"``; each released body's ground ``impact``
(summary ``bodies[].impact``) becomes an ``impact`` event.

Body mapping ("split at separation/jettison"): ``trajectory.csv`` covers the
active stack for the whole flight, so it is sliced per active stage -- stack
track ``stage{k}`` runs from the previous separation to this stage's own
separation (parent ``stage{k-1}``), and OT's released-body CSVs become extra
tracks: spent stage ``stage{k}`` as ``stage{k}_debris`` (parent
``stage{k}``), jettisons under OT's own id (parent the carrying stack).
Every track's CSV mass is non-increasing within the track, which keeps
``flightlab.scene_export`` happy across staging.

Determinism and concurrency (E2 will call this from parallel optimizer
processes): ``run_ascent`` is a pure function of ``(target, vehicle file,
attitude program)`` -- the run id is a digest of the target and the generated
config, ``created_utc`` records the mission epoch rather than wall-clock
``now``, and the OT scratch directory is created inside ``out_dir`` with a
unique ``tempfile.mkdtemp`` prefix, so simultaneous calls never collide. The
raw OT outputs are copied into ``<out_dir>/opentsiolkovsky/`` (a directory
the contract reader ignores) for audit and for the unmapped event kinds; the
binary version, the pinned release digest and the generated config's SHA-256
are recorded in the manifest (``engines`` / ``assumptions`` -- the manifest
schema's field set is frozen, so the config hash rides in assumptions).

    from pathlib import Path
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent
    run = run_ascent(load_target("scenarios/leo_two_stage.yaml"),
                     Path("vehicles/synthetic_two_stage.yaml"), None,
                     Path("runs/fl004-0001"))

Install the pinned binary first: ``python scripts/fetch_opentsiolkovsky.py``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import yaml

from orbitopt.flightlab import frames
from orbitopt.flightlab.contract import (
    Event,
    FlightRun,
    FlightTarget,
    FLIGHT_PROVENANCE_TAGS,
    BodyTrack,
    Check,
    Verification,
    target_digest,
    write_run,
)

#: The single OT version this adapter is written against -- the binary is
#: pinned (scripts/fetch_opentsiolkovsky.py) and a version probe mismatch is
#: a hard error, because outputs and event semantics are version-specific.
OT_VERSION = "1.0.0"

#: Pinned release asset + SHA-256 of the asset archive (kept in sync with
#: scripts/fetch_opentsiolkovsky.py; recorded in the run manifest so a run
#: names exactly the release it was flown with, even when the binary came in
#: through ORBITOPT_OPENTSIOLKOVSKY from a manual install).
PINNED_ASSET = f"openTsiolkovsky-v{OT_VERSION}-<platform>.tar.gz"
PINNED_DIGEST_NOTE = (
    "pinned per-platform SHA-256 digests live in scripts/fetch_opentsiolkovsky.py"
)

_ENV_BINARY = "ORBITOPT_OPENTSIOLKOVSKY"
_RAW_DIR = "opentsiolkovsky"

#: Trajectory CSV columns the adapter carries across as contract extras --
#: all numerically parseable in every OT row (the IIP columns can be empty).
EXTRA_COLUMNS = ("altitude_m", "latitude_deg", "longitude_deg", "downrange_m",
                 "dynamic_pressure_pa", "mach")


class OTBinaryMissingError(RuntimeError):
    """No OpenTsiolkovsky binary was found, or the one found is not the
    pinned version -- with the fix (fetch script / env override) spelled out."""


@dataclass(frozen=True)
class AttitudePoint:
    """One waypoint of an azimuth/elevation attitude program, ``t_s`` seconds
    after liftoff -- the input form of OT's attitude curves (and of a vehicle
    file's ``attitude_program`` section)."""

    t_s: float
    azimuth_deg: float
    elevation_deg: float


# --------------------------------------------------------------------------
# binary location / version probe / cached schema
# --------------------------------------------------------------------------

def find_binary() -> Path:
    """Locate the OT binary: ``ORBITOPT_OPENTSIOLKOVSKY`` (an explicit path,
    the adapter's and fetch script's shared override) wins, else the fetch
    script's install location ``~/.cache/orbitopt/bin/openTsiolkovsky-cli``,
    else ``PATH``. Raises ``OTBinaryMissingError`` with the install command."""
    override = os.environ.get(_ENV_BINARY, "").strip()
    if override:
        cand = Path(override)
        if cand.is_file():
            return cand
        raise OTBinaryMissingError(
            f"{_ENV_BINARY}={override} is not an existing file. Fix the path, or unset it and "
            f"install the pinned v{OT_VERSION} release with  python scripts/fetch_opentsiolkovsky.py")
    name = _binary_name()
    candidates = [Path.home() / ".cache" / "orbitopt" / "bin" / name]
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise OTBinaryMissingError(
        f"no OpenTsiolkovsky binary found (looked at: {', '.join(str(c) for c in candidates)}). "
        f"Install the pinned v{OT_VERSION} release:  python scripts/fetch_opentsiolkovsky.py"
        f"   -- or set {_ENV_BINARY}=/path/to/{name}")


def _binary_name() -> str:
    return "openTsiolkovsky-cli.exe" if os.name == "nt" else "openTsiolkovsky-cli"


def probe_version(binary: Path) -> str:
    """Ask the binary its version and require the pinned one -- a stale or
    substituted binary must fail here, not midway through a run. Returns the
    bare version string (``'1.0.0'``)."""
    try:
        proc = subprocess.run([str(binary), "--version"], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OTBinaryMissingError(f"{binary}: version probe failed ({exc})") from exc
    out = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0 or OT_VERSION not in out:
        raise OTBinaryMissingError(
            f"{binary}: reports {out!r}, but this adapter is pinned to OpenTsiolkovsky "
            f"v{OT_VERSION} -- re-run  python scripts/fetch_opentsiolkovsky.py --force")
    return OT_VERSION


@lru_cache(maxsize=None)
def pinned_config_schema() -> dict:
    """OT's own config JSON Schema (``openTsiolkovsky-cli schema``) as cached
    in ``orbitopt/data/flightlab/schemas/``, so config validation needs no
    binary (CI). Regenerate with ``fetch_opentsiolkovsky.py --write-schema``
    when re-pinning."""
    text = (resources.files("orbitopt")
            .joinpath("data", "flightlab", "schemas", f"opentsiolkovsky-config-v{OT_VERSION}.json")
            .read_text(encoding="utf-8"))
    return json.loads(text)


# --------------------------------------------------------------------------
# vehicle file + config generation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Vehicle:
    """Parsed flightlab vehicle file: everything the adapter needs beyond
    the target (mission/site/epoch) to write an OT config -- the OT
    ``stages`` array passed through, the simulation window, and a default
    attitude program the caller may override."""

    name: str
    provenance: str
    end_time_s: float
    output_step_s: float
    integrator: dict
    stages: tuple[dict, ...]
    attitude_program: tuple[AttitudePoint, ...] | None
    assumptions: tuple[str, ...]

    @staticmethod
    def load(path) -> "Vehicle":
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: flightlab vehicle file must be a YAML mapping")
        missing = [k for k in ("name", "provenance", "stages") if k not in doc]
        if missing:
            raise ValueError(f"{path}: vehicle file is missing required key(s): {', '.join(missing)}")
        if str(doc["provenance"]) not in FLIGHT_PROVENANCE_TAGS:
            raise ValueError(f"{path}: provenance must be one of {FLIGHT_PROVENANCE_TAGS}")
        stages = doc["stages"]
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{path}: 'stages' must be a non-empty list (OT stage configs)")
        sim = doc.get("simulation") or {}
        if "end_time_s" not in sim:
            raise ValueError(f"{path}: 'simulation.end_time_s' is required (where to stop the ascent)")
        program = doc.get("attitude_program")
        points = None
        if program is not None:
            if not isinstance(program, list) or not program:
                raise ValueError(f"{path}: 'attitude_program' must be a non-empty list of waypoints")
            points = tuple(_attitude_point(p, f"{path}: attitude_program[{i}]") for i, p in enumerate(program))
        return Vehicle(
            name=str(doc["name"]),
            provenance=str(doc["provenance"]),
            end_time_s=float(sim["end_time_s"]),
            output_step_s=float(sim.get("output_step_s", 1.0)),
            integrator=dict(sim.get("integrator") or {"method": "rk45"}),
            stages=tuple(stages),
            attitude_program=points,
            assumptions=tuple(str(a) for a in (doc.get("assumptions") or ())),
        )


def _attitude_point(p: Any, where: str) -> AttitudePoint:
    if isinstance(p, AttitudePoint):
        return p
    if not isinstance(p, Mapping) or not {"t_s", "azimuth_deg", "elevation_deg"} <= set(p):
        raise ValueError(f"{where}: an attitude waypoint needs t_s, azimuth_deg, elevation_deg")
    return AttitudePoint(float(p["t_s"]), float(p["azimuth_deg"]), float(p["elevation_deg"]))


def _iso_utc(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if dt.microsecond:
        base += f".{dt.microsecond:06d}".rstrip("0")
    return base + "Z"


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def build_config(target: FlightTarget, vehicle: Vehicle,
                 attitude_program: Sequence[AttitudePoint] | None = None) -> dict:
    """The OT config JSON document this run would fly: ``launch`` from the
    target's site/epoch, ``attitude`` from the caller's program (or the
    vehicle file's), ``simulation`` and ``stages`` from the vehicle file.
    Validated against OT's own cached schema before returning, so a malformed
    vehicle/program fails here with a schema error instead of as an opaque
    subprocess exit code."""
    program = list(attitude_program) if attitude_program else \
        (list(vehicle.attitude_program) if vehicle.attitude_program else None)
    if not program:
        raise ValueError(
            "no attitude program: pass attitude_program to run_ascent, or give the "
            f"vehicle file {vehicle.name!r} an attitude_program section"
        )
    az = [[float(p.t_s), float(p.azimuth_deg)] for p in program]
    el = [[float(p.t_s), float(p.elevation_deg)] for p in program]
    config = {
        "name": f"{target.mission.id} (orbitopt FL-004 / {vehicle.name})",
        "simulation": {
            "end_time_s": vehicle.end_time_s,
            "output_step_s": vehicle.output_step_s,
            "integrator": dict(vehicle.integrator),
        },
        "launch": {
            "latitude_deg": float(target.launch_site.latitude_deg),
            "longitude_deg": float(target.launch_site.longitude_deg),
            "altitude_m": float(target.launch_site.altitude_wgs84_m),
            "time_utc": target.mission.epoch_utc,
        },
        "attitude": {"azimuth_deg": az, "elevation_deg": el},
        "stages": [json.loads(json.dumps(s)) for s in vehicle.stages],
    }
    jsonschema.validate(instance=config, schema=pinned_config_schema())
    return config


# --------------------------------------------------------------------------
# subprocess run of the binary
# --------------------------------------------------------------------------

def _run_binary(binary: Path, config: Mapping[str, Any], ot_out_dir: Path, timeout_s: float) -> dict:
    """One ``openTsiolkovsky-cli run <cfg> -o <dir> --json`` in ``ot_out_dir``
    (a caller-owned fresh scratch dir -> parallel-safe). Returns the parsed
    single JSON document the CLI prints. Every failure mode (spawn, timeout,
    non-zero exit, unparsable JSON, error document) raises RuntimeError naming
    the command and OT's own error kind/message where available."""
    config_path = ot_out_dir.parent / "config.json"
    text = json.dumps(dict(config), indent=1, sort_keys=False, ensure_ascii=False)
    config_path.write_text(text, encoding="utf-8")
    cmd = [str(binary), "run", str(config_path), "-o", str(ot_out_dir), "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{' '.join(cmd)}: timed out after {timeout_s} s") from exc
    except OSError as exc:
        raise RuntimeError(f"{cmd[0]}: cannot execute ({exc})") from exc
    if proc.returncode != 0:
        detail = _error_detail(proc.stdout) or proc.stderr.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"OpenTsiolkovsky failed ({detail}); stderr: {proc.stderr.strip()[:400]}")
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenTsiolkovsky --json output is not JSON: {exc}") from exc
    if doc.get("status") != "ok":
        raise RuntimeError(f"OpenTsiolkovsky reported status {doc.get('status')!r}: "
                           f"{json.dumps(doc.get('error') or doc.get('warnings') or '')[:400]}")
    return doc


def _error_detail(stdout: str) -> str:
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return ""
    err = doc.get("error") if isinstance(doc, dict) else None
    if isinstance(err, dict):
        return f"{err.get('kind', 'error')}: {err.get('message', '')}"
    return ""


# --------------------------------------------------------------------------
# OT outputs -> FlightRun
# --------------------------------------------------------------------------

def _num(row: Mapping[str, str], col: str) -> float:
    val = (row.get(col) or "").strip()
    return float(val) if val else 0.0


def _read_ot_csv(path: Path) -> list[dict]:
    """One OT trajectory CSV -> list of row dicts with the float columns the
    adapter needs (plus 'stage'), ordered as written."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        out.append({
            "t": _num(row, "time_s"),
            "mass": _num(row, "mass_kg"),
            "r": [_num(row, "pos_eci_x_m"), _num(row, "pos_eci_y_m"), _num(row, "pos_eci_z_m")],
            "v": [_num(row, "vel_eci_x_mps"), _num(row, "vel_eci_y_mps"), _num(row, "vel_eci_z_mps")],
            "extras": [_num(row, c) for c in EXTRA_COLUMNS],
            "stage": int(float(row.get("stage") or "1")),
        })
    return out


def _dedupe(rows: list[dict]) -> list[dict]:
    """OT writes a sample at every event time, so a grid time can appear
    twice (e.g. ignition and liftoff both at t=0). The contract requires
    strictly increasing t_s -- keep the LAST row for a repeated time (the
    event state) and drop the earlier duplicate."""
    seen: dict[float, dict] = {}
    order: list[float] = []
    for row in rows:
        if row["t"] not in seen:
            order.append(row["t"])
        seen[row["t"]] = row
    return [seen[t] for t in order]


def _track(body_id: str, parent: str | None, epoch_utc: str, provenance: str,
           rows: list[dict], epoch: datetime) -> BodyTrack:
    """One BodyTrack from OT rows, with the LAUNCH_ECI pseudo-inertial states
    converted to GCRS by the frame bridge (the only legal frame move)."""
    if not rows:
        raise RuntimeError("adapter bug: empty OT row slice for body " + body_id)
    r_gcrs, v_gcrs = frames.to_gcrs([row["r"] for row in rows], [row["v"] for row in rows],
                                    "LAUNCH_ECI", epoch, [row["t"] for row in rows])
    return BodyTrack(
        body_id=body_id,
        parent_body_id=parent,
        frame="GCRS",
        epoch_utc=epoch_utc,
        provenance=provenance,
        t_s=tuple(float(row["t"]) for row in rows),
        mass_kg=tuple(float(row["mass"]) for row in rows),
        position_m=tuple((float(x), float(y), float(z)) for x, y, z in r_gcrs),
        velocity_m_s=tuple((float(x), float(y), float(z)) for x, y, z in v_gcrs),
        extra_columns={name: tuple(float(row["extras"][i]) for row in rows)
                       for i, name in enumerate(EXTRA_COLUMNS)},
    )


_EVENT_MAP = {"ignition": "ignition", "liftoff": "liftoff", "max_q": "max_q",
              "separation": "separation"}
_STACK_RE = re.compile(r"stage[0-9]+")


def _mapped_events(summary: Mapping[str, Any], n_stages: int) -> list[Event]:
    """OT summary events -> contract events (see the module docstring for the
    table). Unmapped OT kinds are dropped here but survive in the raw copy."""
    events: list[Event] = []
    for e in summary.get("events") or []:
        kind, stage, t = e.get("kind"), int(e.get("stage", 1)), float(e.get("time_s", 0.0))
        data: dict[str, Any] = {"ot_event": kind, "stage": stage}
        if kind in ("mach1", "rail_exit", "apogee"):
            continue
        if kind == "burnout":
            mapped, body = ("meco" if stage == 1 else "seco"), f"stage{stage}"
        elif kind == "separation":
            mapped, body = "separation", (f"stage{stage + 1}" if stage < n_stages else f"stage{stage}")
            data["separated_from"] = f"stage{stage}_debris"
        elif kind == "jettison":
            mapped, body = "separation", f"stage{stage}_{e.get('body', 'jettison')}"
        elif kind in _EVENT_MAP:
            mapped, body = _EVENT_MAP[kind], f"stage{stage}"
        else:
            continue
        events.append(Event(t_s=t, kind=mapped, body_id=body, data=data))
    main_impact = summary.get("impact")
    if main_impact:  # suborbital flights end with the (final) stack on the ground
        events.append(Event(t_s=float(main_impact["time_s"]), kind="impact",
                            body_id=f"stage{n_stages}",
                            data={"ot_event": "impact", "latitude_deg": main_impact["latitude_deg"],
                                  "longitude_deg": main_impact["longitude_deg"],
                                  "speed_mps": main_impact["speed_mps"]}))
    for b in summary.get("bodies") or []:
        impact = b.get("impact")
        if impact:
            body = _released_body_id(b)
            events.append(Event(t_s=float(impact["time_s"]), kind="impact", body_id=body,
                                data={"ot_event": "impact", "latitude_deg": impact["latitude_deg"],
                                      "longitude_deg": impact["longitude_deg"], "speed_mps": impact["speed_mps"]}))
    events.sort(key=lambda ev: (ev.t_s, ev.kind, ev.body_id))
    return events


def _released_body_id(b: Mapping[str, Any]) -> str:
    """Manifest body id for an OT released body: spent stages get the
    ``_debris`` suffix (OT's id ``stage1`` names the falling booster while our
    stack track ``stage1`` names the flying stack up to separation);
    jettisons keep OT's already-qualified id (``stage2_fairing``)."""
    return f"{b['id']}_debris" if b.get("kind") == "stage" else str(b["id"])


def _verification(summary: Mapping[str, Any], tracks: Sequence[BodyTrack],
                  target: FlightTarget) -> Verification:
    """Checks the adapter can settle from the adapter's own outputs: masses
    non-negative, and stack-mass continuity across every separation (stack
    mass on the parent track at separation = continuing-stack mass +
    released-body mass, as a fraction of the stack mass). Mission constraints
    are evaluated against the target's own limits when it states them.
    Convergence and physics verification are NOT assessed here: OT is a
    simulator (no optimizer) and a higher-fidelity check (e.g. tudatpy) is
    what sets physics_verified -- both stay None."""
    checks: list[Check] = []
    checks.append(Check(id="mass-non-negative",
                        passed=all(min(tr.mass_kg) >= 0.0 for tr in tracks),
                        value=min(min(tr.mass_kg) for tr in tracks), limit=0.0, unit="kg",
                        reason_code="" if all(min(tr.mass_kg) >= 0.0 for tr in tracks) else "negative_mass"))
    by_id = {tr.body_id: tr for tr in tracks}
    frac_limit = 0.02
    for tr in tracks:
        if not tr.parent_body_id or not _STACK_RE.fullmatch(tr.body_id):
            continue
        parent = by_id.get(tr.parent_body_id)
        if parent is None:
            continue
        released = by_id.get(f"{tr.parent_body_id}_debris")
        if released is None:
            continue
        stack_mass = parent.mass_kg[-1]
        accounted = tr.mass_kg[0] + released.mass_kg[0]
        rel = abs(stack_mass - accounted) / stack_mass if stack_mass else 0.0
        checks.append(Check(id=f"mass-continuity-{tr.body_id}", passed=rel <= frac_limit,
                            value=rel, limit=frac_limit, unit="fraction",
                            reason_code="" if rel <= frac_limit else "mass_discontinuity"))
    maxima = (summary.get("maxima") or {})
    constraints = target.constraints
    verdicts: list[bool] = []
    if constraints.max_dynamic_pressure_pa is not None and "dynamic_pressure_pa" in maxima:
        value = float(maxima["dynamic_pressure_pa"]["value"])
        passed = value <= constraints.max_dynamic_pressure_pa
        verdicts.append(passed)
        checks.append(Check(id="max-q-within-limit", passed=passed, value=value,
                            limit=float(constraints.max_dynamic_pressure_pa), unit="Pa",
                            reason_code="" if passed else "max_q_exceeded"))
    if constraints.max_specific_acceleration_g is not None and "acceleration_g" in maxima:
        value = float(maxima["acceleration_g"]["value"])
        passed = value <= constraints.max_specific_acceleration_g
        verdicts.append(passed)
        checks.append(Check(id="max-accel-within-limit", passed=passed, value=value,
                            limit=float(constraints.max_specific_acceleration_g), unit="g",
                            reason_code="" if passed else "max_accel_exceeded"))
    return Verification(
        optimizer_converged=None,
        physics_verified=None,
        mission_constraints_passed=(all(verdicts) if verdicts else None),
        checks=tuple(checks),
    )


def parse_ot_outputs(ot_out_dir: Path, target: FlightTarget, vehicle: Vehicle,
                     run_id: str, git_sha: str, config_sha256: str,
                     vehicle_sha256: str) -> FlightRun:
    """The OT output directory (trajectory.csv + summary.json + bodies/) ->
    a FlightRun object (no disk write). Exposed for tests and re-analysis of
    recorded OT runs; run_ascent is the full pipeline. Frame conversion,
    body splitting and event mapping are documented at the module docstring."""
    summary = json.loads((ot_out_dir / "summary.json").read_text(encoding="utf-8"))
    main_rows = _read_ot_csv(ot_out_dir / "trajectory.csv")
    n_stages = len(vehicle.stages)
    epoch = _parse_utc(target.mission.epoch_utc)
    epoch_utc = _iso_utc(epoch)
    provenance = vehicle.provenance

    # stack tracks: trajectory.csv sliced by active stage, deduped per event
    tracks: list[BodyTrack] = []
    stack: dict[int, list[dict]] = {}
    for row in main_rows:
        stack.setdefault(row["stage"], []).append(row)
    for stage in sorted(stack):
        rows = _dedupe(stack[stage])
        tracks.append(_track(f"stage{stage}", f"stage{stage - 1}" if stage > 1 else None,
                             epoch_utc, provenance, rows, epoch))

    # released bodies: OT's bodies/<id>.csv, from release to impact
    for b in summary.get("bodies") or []:
        body_path = ot_out_dir / "bodies" / f"{b['id']}.csv"
        if not body_path.is_file():
            raise RuntimeError(f"{body_path}: missing from the OT outputs for released body {b['id']!r}")
        rows = _dedupe(_read_ot_csv(body_path))
        tracks.append(_track(_released_body_id(b), f"stage{b['stage']}", epoch_utc,
                             provenance, rows, epoch))

    events = _mapped_events(summary, n_stages)
    known = {tr.body_id for tr in tracks}
    events = [e for e in events if e.body_id in known]  # defensive; OT ids always map
    assumptions = (
        f"OpenTsiolkovsky {OT_VERSION} (MIT, external binary -- not vendored; see NOTICE; "
        f"{PINNED_ASSET}, {PINNED_DIGEST_NOTE})",
        f"opentsiolkovsky run-config sha256 {config_sha256}",
        "OT pos_eci_*/vel_eci_* read as LAUNCH_ECI (axes = ECEF at the launch epoch, held "
        "inertial) and converted to GCRS by orbitopt.flightlab.frames.to_gcrs at the epoch",
        "created_utc is the mission epoch: this adapter is a pure function of "
        "(target, vehicle, attitude program), so repeated calls write byte-identical runs",
    ) + tuple(f"vehicle: {a}" for a in vehicle.assumptions) + tuple(target.assumptions)
    return FlightRun(
        run_id=run_id,
        target_sha256=target_digest(target.to_dict()),
        vehicle_sha256=vehicle_sha256,
        git_sha=git_sha,
        created_utc=epoch_utc,
        engines={"opentsiolkovsky-cli": OT_VERSION},
        assumptions=assumptions,
        tracks=tuple(tracks),
        events=tuple(events),
        verification=_verification(summary, tracks, target),
    )


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def run_ascent(target: FlightTarget, vehicle_path: Path,
               attitude_program: list[AttitudePoint] | None, out_dir: Path,
               *, timeout_s: float = 120) -> FlightRun:
    """Fly ``target``'s mission in OpenTsiolkovsky and write the result to
    ``out_dir`` as a FlightResult v1 run directory (see the module docstring
    for mapping rules). Returns the FlightRun that was written.

    Pure: the same ``(target, vehicle_path, attitude_program)`` produce
    byte-identical files; safe for parallel processes (fresh mkdtemp scratch
    per call). Raises ``OTBinaryMissingError`` (install/override message),
    ``jsonschema.ValidationError`` (config vs OT's schema) or ``RuntimeError``
    (subprocess/timeout/output problems)."""
    binary = find_binary()
    probe_version(binary)
    vehicle = Vehicle.load(vehicle_path)
    vehicle_sha = hashlib.sha256(Path(vehicle_path).read_bytes()).hexdigest()
    config = build_config(target, vehicle, attitude_program)
    config_bytes = json.dumps(config, indent=1, sort_keys=False, ensure_ascii=False).encode("utf-8")
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    target_sha = target_digest(target.to_dict())
    run_id = f"ot-{target_sha[:8]}-{config_sha[:8]}"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".ot-scratch-", dir=out))
    try:
        _run_binary(binary, config, scratch / "ot", timeout_s)
        run = parse_ot_outputs(scratch / "ot", target, vehicle, run_id, _git_sha(),
                               config_sha, vehicle_sha)
        raw = out / _RAW_DIR
        if raw.exists():
            shutil.rmtree(raw)
        shutil.copytree(scratch / "ot", raw)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    write_run(out, run)
    return run


def _git_sha() -> str:
    """HEAD sha of this checkout, or 'unknown' -- same policy (and same
    command) as orbitopt.flightlab.target_export._git_sha."""
    root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"
