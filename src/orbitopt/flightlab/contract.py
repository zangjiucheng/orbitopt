"""orbitopt Flight Lab v1 data contract: ``FlightTarget`` (what a flight run
is asked to do) and ``FlightRun`` / ``FlightResult`` (what it produced), as
frozen dataclasses over independently-versioned JSON Schemas
(``orbitopt/schemas/flight-target-1.0.json``,
``orbitopt/schemas/flight-result-1.0.json``).

Like orbitopt.scene_format / orbitopt.mission_config, this module needs
nothing but the standard library + jsonschema + pyyaml -- no numpy, no Qt,
no pyvista, no compute stack -- so a producer (an ascent optimizer, an
external CLI adapter) and a consumer (the scene importer, Mission Control)
can stay on either side of this file without importing each other. Physics
backends must NOT be trusted with frame bookkeeping: a body's ``frame`` /
``epoch_utc`` travel in the run manifest, and an external backend's
pseudo-ECI (e.g. OpenTsiolkovsky's launch-aligned ECI) must be converted to
GCRS by the frame bridge before it is written here as ``"GCRS"``.

Run directory layout (the FlightResult v1 document spans four files):

    runs/<run_id>/
      manifest.json           # root document (flight-result-1.0.json)
      trajectories/<body_id>.csv   # t_s,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s [+ extras]
      events.json             # [ {"t_s", "kind", "body_id", "data"}, ... ]
      verification.json       # {optimizer_converged, physics_verified,
                              #  mission_constraints_passed, checks}

The three verification booleans answer three different questions and are
never collapsed into one "success": optimizer convergence is not physics
verification is not mission-constraint satisfaction; ``null`` means
"not assessed", never "passed".

    from orbitopt.flightlab.contract import load_target, read_run, write_run
    target = load_target("scenarios/leo_two_stage.yaml")   # raises on mismatch
    run = read_run("runs/demo-0001")                        # schema + CSV + cross-file checks
    write_run("runs/copy", run)                             # round-trips

Compatibility policy: same as orbitopt.scene_format -- a document's own
``schema_version`` picks the schema; readers ignore unknown OPTIONAL fields,
so adding one is not breaking; removing/renaming/narrowing is, and gets a
new schema file (flight-target-2.0.json etc.).
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml

SCHEMA_VERSION = "1.0"

#: The only frames a Flight Lab v1 body track may declare. ``LAUNCH_ECI`` is
#: the launch-aligned, non-rotating frame aligned with ECEF at the track's
#: epoch (what OpenTsiolkovsky reports); converting it to ``GCRS`` is the
#: frame bridge's (FL-002) job, not each adapter's.
FRAMES = ("GCRS", "ECEF", "LAUNCH_ECI")

#: The Flight Lab vocabulary of orbitopt.data_provenance tags -- every
#: authored number in a target/run is one of these (``derived`` is not used:
#: flight artifacts carry their own provenance, not the launch DBs').
FLIGHT_PROVENANCE_TAGS = ("official", "public_estimate", "illustrative")

EVENT_KINDS = (
    "ignition", "liftoff", "max_q", "meco", "separation", "seco",
    "boostback_start", "boostback_end", "entry_start", "entry_end",
    "landing_start", "touchdown", "impact",
)

#: Every trajectories/<body_id>.csv starts with exactly these columns, in
#: this order; any further numeric columns are extras the reader preserves
#: (e.g. dynamic pressure, throttle level).
REQUIRED_CSV_COLUMNS = ("t_s", "mass_kg", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")

_RUN_MANIFEST = "manifest.json"
_RUN_TRAJECTORIES = "trajectories"
_RUN_EVENTS = "events.json"
_RUN_VERIFICATION = "verification.json"


@lru_cache(maxsize=None)
def _schema(name: str) -> dict:
    """Load and cache a packaged JSON Schema by filename. Uses
    importlib.resources, not a hand-built filesystem path, so this works from
    an installed wheel as well as a source checkout -- same approach as
    orbitopt.scene_format._schema."""
    traversable = resources.files("orbitopt").joinpath("schemas", name)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"No packaged schema file orbitopt/schemas/{name}") from exc
    return json.loads(text)


@lru_cache(maxsize=None)
def _flight_result_def(name: str) -> dict:
    """The flight-result schema's root document re-rooted at one of its
    ``$defs`` (events / verification), so validating a run sidecar document
    keeps the ``$defs`` its ``$ref``\\ s resolve against -- validating the
    bare subschema alone would strand every ``#/$defs/...`` pointer."""
    schema = _schema("flight-result-1.0.json")
    return {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"}


def _parse_utc(text: str, where: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp (``Z`` or explicit offset required)
    into an aware datetime. ``datetime.fromisoformat`` is used rather than
    strptime because the schema pattern accepts fractional seconds and both
    zone spellings."""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: {text!r} is not an ISO-8601 timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{where}: timestamp {text!r} needs an explicit UTC zone (e.g. trailing 'Z')")
    return dt


# --------------------------------------------------------------------------
# FlightTarget
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mission:
    id: str
    title: str
    epoch_utc: str


@dataclass(frozen=True)
class LaunchSite:
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_wgs84_m: float


@dataclass(frozen=True)
class TargetOrbit:
    central_body: str
    periapsis_altitude_m: float
    apoapsis_altitude_m: float
    inclination_deg: float
    raan_deg: float | None = None
    ltan_hours: float | None = None
    epoch_utc: str | None = None


@dataclass(frozen=True)
class TerminalState:
    frame: str
    epoch_utc: str
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    origin: str = "earth_center"
    time_scale: str = "UTC"


@dataclass(frozen=True)
class Constraints:
    max_dynamic_pressure_pa: float | None = None
    max_specific_acceleration_g: float | None = None
    minimum_insertion_periapsis_m: float | None = None
    orbit_altitude_tolerance_m: float | None = None
    inclination_tolerance_deg: float | None = None
    min_propellant_reserve_kg: float | None = None


@dataclass(frozen=True)
class Recovery:
    kind: str = "none"
    target: str | None = None
    max_touchdown_speed_m_s: float | None = None
    max_touchdown_lateral_speed_m_s: float | None = None
    min_propellant_reserve_kg: float | None = None


@dataclass(frozen=True)
class VehicleRef:
    source: str
    provenance: str


@dataclass(frozen=True)
class Simulation:
    ascent_backend: str | None = None
    atmosphere_6dof_backend: str | None = None
    orbit_verify_backend: str | None = None
    enabled_phases: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlightTarget:
    """Parsed + schema-validated view of a FlightTarget v1 document. All
    fields below the top level are themselves frozen dataclasses, so a target
    handed to another component cannot be mutated underneath it."""

    mission: Mission
    launch_site: LaunchSite
    target_orbit: TargetOrbit
    vehicle: VehicleRef
    terminal_state: TerminalState | None = None
    constraints: Constraints = field(default_factory=Constraints)
    recovery: Recovery = field(default_factory=Recovery)
    simulation: Simulation = field(default_factory=Simulation)
    assumptions: tuple[str, ...] = ()

    @staticmethod
    def from_dict(doc: Mapping[str, Any]) -> "FlightTarget":
        """Build from an already-validated mapping (see validate_target);
        raises KeyError/TypeError on a structurally impossible input, so call
        it only after validation, exactly as the dataclass mirrors the
        schema)."""
        orbit = doc["target_orbit"]
        terminal = doc.get("terminal_state")
        sim = doc.get("simulation") or {}
        return FlightTarget(
            mission=Mission(**doc["mission"]),
            launch_site=LaunchSite(**doc["launch_site"]),
            target_orbit=TargetOrbit(
                central_body=orbit["central_body"],
                periapsis_altitude_m=orbit["periapsis_altitude_m"],
                apoapsis_altitude_m=orbit["apoapsis_altitude_m"],
                inclination_deg=orbit["inclination_deg"],
                raan_deg=orbit.get("raan_deg"),
                ltan_hours=orbit.get("ltan_hours"),
                epoch_utc=orbit.get("epoch_utc"),
            ),
            vehicle=VehicleRef(**doc["vehicle"]),
            terminal_state=TerminalState(
                frame=terminal["frame"],
                epoch_utc=terminal["epoch_utc"],
                position_m=tuple(terminal["position_m"]),
                velocity_m_s=tuple(terminal["velocity_m_s"]),
                origin=terminal.get("origin", "earth_center"),
                time_scale=terminal.get("time_scale", "UTC"),
            ) if terminal is not None else None,
            constraints=Constraints(**(doc.get("constraints") or {})),
            recovery=Recovery(**(doc.get("recovery") or {})),
            simulation=Simulation(
                ascent_backend=sim.get("ascent_backend"),
                atmosphere_6dof_backend=sim.get("atmosphere_6dof_backend"),
                orbit_verify_backend=sim.get("orbit_verify_backend"),
                enabled_phases=tuple(sim.get("enabled_phases") or ()),
            ),
            assumptions=tuple(doc.get("assumptions") or ()),
        )

    def to_dict(self) -> dict:
        """The FlightTarget v1 document this dataclass represents, in the
        field order of the schema (canonical key order is the caller's
        business -- see target_digest for the hashing form)."""
        doc: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mission": dataclasses.asdict(self.mission),
            "launch_site": dataclasses.asdict(self.launch_site),
            "target_orbit": {
                "central_body": self.target_orbit.central_body,
                "periapsis_altitude_m": self.target_orbit.periapsis_altitude_m,
                "apoapsis_altitude_m": self.target_orbit.apoapsis_altitude_m,
                "inclination_deg": self.target_orbit.inclination_deg,
                "raan_deg": self.target_orbit.raan_deg,
            },
            "vehicle": dataclasses.asdict(self.vehicle),
        }
        if self.target_orbit.ltan_hours is not None:
            doc["target_orbit"]["ltan_hours"] = self.target_orbit.ltan_hours
        if self.target_orbit.epoch_utc is not None:
            doc["target_orbit"]["epoch_utc"] = self.target_orbit.epoch_utc
        if self.terminal_state is not None:
            doc["terminal_state"] = {
                "frame": self.terminal_state.frame,
                "origin": self.terminal_state.origin,
                "time_scale": self.terminal_state.time_scale,
                "epoch_utc": self.terminal_state.epoch_utc,
                "position_m": list(self.terminal_state.position_m),
                "velocity_m_s": list(self.terminal_state.velocity_m_s),
            }
        constraints = {k: v for k, v in dataclasses.asdict(self.constraints).items() if v is not None}
        if constraints:
            doc["constraints"] = constraints
        if self.recovery.kind != "none" or self.recovery.target is not None:
            recovery = {"kind": self.recovery.kind}
            recovery.update({k: v for k, v in dataclasses.asdict(self.recovery).items()
                             if k != "kind" and v is not None})
            doc["recovery"] = recovery
        simulation = {k: v for k, v in {
            "ascent_backend": self.simulation.ascent_backend,
            "atmosphere_6dof_backend": self.simulation.atmosphere_6dof_backend,
            "orbit_verify_backend": self.simulation.orbit_verify_backend,
            "enabled_phases": list(self.simulation.enabled_phases),
        }.items() if v not in (None, [])}
        if simulation:
            doc["simulation"] = simulation
        if self.assumptions:
            doc["assumptions"] = list(self.assumptions)
        return doc


def validate_target(target: Mapping[str, Any]) -> None:
    """Validate an already-parsed FlightTarget document. Raises
    ``jsonschema.ValidationError`` on a schema mismatch, or ``ValueError``
    for missing/unknown ``schema_version``, an unparseable timestamp, or an
    orbit with apoapsis below periapsis -- the cross-field checks JSON Schema
    cannot express, same split as orbitopt.scene_format.validate_scene."""
    version = target.get("schema_version") if isinstance(target, Mapping) else None
    if version is None:
        raise ValueError(
            "flight target document has no 'schema_version' field -- not a "
            f"valid FlightTarget. Set schema_version={SCHEMA_VERSION!r} (see "
            "orbitopt/schemas/flight-target-1.0.json)."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"no FlightTarget schema for version {version!r} (this orbitopt "
            f"implements {SCHEMA_VERSION!r}; see orbitopt/schemas/)"
        )
    doc = dict(target)
    jsonschema.validate(instance=doc, schema=_schema("flight-target-1.0.json"))

    _parse_utc(doc["mission"]["epoch_utc"], "mission.epoch_utc")
    orbit = doc["target_orbit"]
    if orbit["apoapsis_altitude_m"] < orbit["periapsis_altitude_m"]:
        raise ValueError(
            f"flight target {doc['mission']['id']!r} is invalid: target orbit apoapsis "
            f"({orbit['apoapsis_altitude_m']} m) is below periapsis "
            f"({orbit['periapsis_altitude_m']} m) -- swap them or widen the orbit."
        )
    if orbit.get("epoch_utc") is not None:
        _parse_utc(orbit["epoch_utc"], "target_orbit.epoch_utc")
    if doc.get("terminal_state") is not None:
        _parse_utc(doc["terminal_state"]["epoch_utc"], "terminal_state.epoch_utc")


def is_valid_target(target: Mapping[str, Any]) -> bool:
    """Non-raising form of validate_target() -- True/False instead of an
    exception, for callers that just want a filter/check."""
    try:
        validate_target(target)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


def load_target(path, *, validate: bool = True) -> FlightTarget:
    """Read a FlightTarget YAML (or JSON -- YAML is a superset) document from
    disk and return it as a validated, frozen FlightTarget. Pass
    ``validate=False`` to load a document as-is."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(doc).__name__}")
    if validate:
        validate_target(doc)
    return FlightTarget.from_dict(doc)


def target_digest(target: Mapping[str, Any]) -> str:
    """Full SHA-256 hex of a target document's canonical JSON -- what a run
    manifest records as ``target_sha256`` so a run is auditable against the
    exact target it flew (two targets with the same content regardless of
    key order / YAML formatting share a digest)."""
    text = json.dumps(dict(target), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# FlightRun (the FlightResult v1 run directory)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyTrack:
    """One body's sampled trajectory: CSV columns held as parallel tuples,
    with the frame/epoch/provenance the manifest declares for this body.
    ``extra_columns`` preserves any additional CSV columns (name -> values)
    so a write_run/read_run round-trip is lossless."""

    body_id: str
    parent_body_id: str | None
    frame: str
    epoch_utc: str
    provenance: str
    t_s: tuple[float, ...]
    mass_kg: tuple[float, ...]
    position_m: tuple[tuple[float, float, float], ...]
    velocity_m_s: tuple[tuple[float, float, float], ...]
    extra_columns: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def t_start_s(self) -> float:
        return self.t_s[0]

    @property
    def t_end_s(self) -> float:
        return self.t_s[-1]

    def to_manifest_entry(self) -> dict:
        """This track's ``bodies[]`` entry in the run manifest (frame/epoch/
        time bounds live there, not in the CSV)."""
        return {
            "body_id": self.body_id,
            "parent_body_id": self.parent_body_id,
            "frame": self.frame,
            "epoch_utc": self.epoch_utc,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Event:
    """One mission event (ignition, max_q, separation, touchdown, ...), at
    ``t_s`` seconds after the mission epoch, attributed to a body id."""

    t_s: float
    kind: str
    body_id: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Check:
    """One verification check: what was tested, the verdict, and the numbers
    behind it (value/limit/unit may be null for non-numeric checks)."""

    id: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    reason_code: str = ""


@dataclass(frozen=True)
class Verification:
    """The three independent success verdicts plus their per-check evidence.
    A ``None`` verdict means "not assessed" -- callers that treat verification
    as a gate must map None to "not passed", never to "passed"."""

    optimizer_converged: bool | None = None
    physics_verified: bool | None = None
    mission_constraints_passed: bool | None = None
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class FlightRun:
    """A FlightResult v1 run: manifest fields + per-body tracks + events +
    verification. Construct one, then ``write_run(dir, run)`` materialises
    the directory; ``read_run(dir)`` is the inverse."""

    run_id: str
    target_sha256: str
    vehicle_sha256: str
    git_sha: str
    created_utc: str
    engines: Mapping[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    tracks: tuple[BodyTrack, ...] = ()
    events: tuple[Event, ...] = ()
    verification: Verification = field(default_factory=Verification)


def _validate_run_manifest(manifest: Mapping[str, Any], where: str) -> None:
    version = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    if version is None:
        raise ValueError(
            f"{where}: manifest has no 'schema_version' field -- not a valid "
            f"FlightResult. Set schema_version={SCHEMA_VERSION!r}."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(f"{where}: no FlightResult schema for version {version!r}")
    jsonschema.validate(instance=dict(manifest), schema=_schema("flight-result-1.0.json"))
    _parse_utc(manifest["created_utc"], f"{where}: created_utc")
    for body in manifest["bodies"]:
        _parse_utc(body["epoch_utc"], f"{where}: body {body['body_id']!r} epoch_utc")
        if body["t_end_s"] < body["t_start_s"]:
            raise ValueError(
                f"{where}: body {body['body_id']!r} ends (t_end_s={body['t_end_s']}) "
                f"before it starts (t_start_s={body['t_start_s']})."
            )
    ids = [b["body_id"] for b in manifest["bodies"]]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{where}: duplicate body_id(s) in manifest: {', '.join(dupes)}")
    parents = {b["parent_body_id"] for b in manifest["bodies"] if b["parent_body_id"] is not None}
    unknown_parents = sorted(parents - set(ids))
    if unknown_parents:
        raise ValueError(
            f"{where}: manifest parent_body_id(s) not among the run's bodies: {', '.join(unknown_parents)}"
        )


def _parse_track_csv(text: str, body: Mapping[str, Any], where: str) -> BodyTrack:
    """Parse + sanity-check one trajectory CSV (schema-expressible rules
    live in flight-result-1.0.json; column order, monotonic t_s and
    non-negative mass -- things JSON Schema cannot see inside a CSV -- are
    checked here, so a bad trajectory is a catchable load error, never a
    silently wrong render downstream)."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or not rows[0]:
        raise ValueError(f"{where}: trajectory CSV is empty (no header row)")
    header = rows[0]
    required = list(REQUIRED_CSV_COLUMNS)
    if header[:len(required)] != required:
        raise ValueError(
            f"{where}: trajectory CSV header must start with the required columns "
            f"{','.join(required)} in order, got {','.join(header)}"
        )
    extras = header[len(required):]
    if len(set(extras)) != len(extras) or set(extras) & set(required):
        raise ValueError(f"{where}: trajectory CSV has duplicate/overlapping column names: {','.join(header)}")

    t_s: list[float] = []
    mass_kg: list[float] = []
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    extra_values: list[list[float]] = [[] for _ in extras]
    for lineno, row in enumerate(rows[1:], start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue  # tolerate a trailing newline's empty row
        if len(row) != len(header):
            raise ValueError(f"{where}:{lineno}: row has {len(row)} fields, header has {len(header)}")
        try:
            nums = [float(x) for x in row]
        except ValueError as exc:
            raise ValueError(f"{where}:{lineno}: non-numeric trajectory value: {exc}") from None
        if any(not math.isfinite(v) for v in nums):
            raise ValueError(f"{where}:{lineno}: non-finite trajectory value -- NaN/Inf is not a valid sample")
        t, m, x, y, z, vx, vy, vz = nums[:8]
        t_s.append(t)
        mass_kg.append(m)
        positions.append((x, y, z))
        velocities.append((vx, vy, vz))
        for i, extra in enumerate(nums[8:]):
            extra_values[i].append(extra)
        if m < 0:
            raise ValueError(f"{where}:{lineno}: mass_kg is negative ({m}) -- mass is never negative; "
                             "a propellant-depletion bookkeeping bug is the usual cause")
    if not t_s:
        raise ValueError(f"{where}: trajectory CSV has a header but no sample rows")
    for i in range(1, len(t_s)):
        if t_s[i] <= t_s[i - 1]:
            raise ValueError(
                f"{where}: t_s is not strictly increasing ({t_s[i - 1]} s followed by {t_s[i]} s) "
                "-- a trajectory must be sampled in strictly increasing time"
            )
    if t_s[0] < body["t_start_s"] or t_s[-1] > body["t_end_s"]:
        raise ValueError(
            f"{where}: trajectory samples span t={t_s[0]}..{t_s[-1]} s, outside the manifest-declared "
            f"window {body['t_start_s']}..{body['t_end_s']} s for body {body['body_id']!r}"
        )
    return BodyTrack(
        body_id=body["body_id"],
        parent_body_id=body["parent_body_id"],
        frame=body["frame"],
        epoch_utc=body["epoch_utc"],
        provenance=body["provenance"],
        t_s=tuple(t_s),
        mass_kg=tuple(mass_kg),
        position_m=tuple(positions),
        velocity_m_s=tuple(velocities),
        extra_columns={name: tuple(values) for name, values in zip(extras, extra_values)},
    )


def _manifest_document(run: FlightRun) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "target_sha256": run.target_sha256,
        "vehicle_sha256": run.vehicle_sha256,
        "engines": dict(run.engines),
        "git_sha": run.git_sha,
        "created_utc": run.created_utc,
        "assumptions": list(run.assumptions),
        "bodies": [track.to_manifest_entry() for track in run.tracks],
    }


def read_run(dir, *, validate: bool = True) -> FlightRun:
    """Read a FlightResult v1 run directory (manifest.json +
    trajectories/*.csv + events.json + verification.json) into a FlightRun.
    Validated by default -- JSON Schema for each JSON document, plus the
    cross-file/cross-column checks (CSV column contract, monotonic t_s,
    non-negative mass, manifest window bounds, events referencing known
    bodies, every manifest body having exactly one trajectory file and vice
    versa) that keep a corrupt run a load error instead of a wrong plot."""
    root = Path(dir)
    manifest_path = root / _RUN_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{root}: no {_RUN_MANIFEST} -- not a FlightResult v1 run directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if validate:
        _validate_run_manifest(manifest, str(manifest_path))

    events = json.loads((root / _RUN_EVENTS).read_text(encoding="utf-8"))
    verification = json.loads((root / _RUN_VERIFICATION).read_text(encoding="utf-8"))
    if validate:
        jsonschema.validate(instance=events, schema=_flight_result_def("events"))
        jsonschema.validate(instance=verification, schema=_flight_result_def("verification"))
        known = {b["body_id"] for b in manifest["bodies"]}
        unknown_events = sorted({e["body_id"] for e in events} - known)
        if unknown_events:
            raise ValueError(
                f"{root / _RUN_EVENTS}: events reference body_id(s) not in the manifest: "
                f"{', '.join(unknown_events)} -- every event must be attributed to a body this run tracked"
            )

    tracks: list[BodyTrack] = []
    for body in manifest["bodies"]:
        csv_path = root / _RUN_TRAJECTORIES / f"{body['body_id']}.csv"
        if not csv_path.is_file():
            raise ValueError(f"{csv_path}: missing -- the manifest declares body {body['body_id']!r}")
        tracks.append(_parse_track_csv(csv_path.read_text(encoding="utf-8"), body, str(csv_path)))
    if validate:
        trajectory_dir = root / _RUN_TRAJECTORIES
        declared = {b["body_id"] for b in manifest["bodies"]}
        orphans = sorted(p.stem for p in trajectory_dir.glob("*.csv") if p.stem not in declared)
        if orphans:
            raise ValueError(
                f"{trajectory_dir}: trajectory file(s) for body_id(s) not in the manifest: "
                f"{', '.join(orphans)} -- every tracked body belongs in manifest.bodies"
            )

    checks = tuple(
        Check(id=c["id"], passed=c["passed"], value=c.get("value"), limit=c.get("limit"),
              unit=c.get("unit", ""), reason_code=c.get("reason_code", ""))
        for c in verification["checks"]
    )
    return FlightRun(
        run_id=manifest["run_id"],
        target_sha256=manifest["target_sha256"],
        vehicle_sha256=manifest["vehicle_sha256"],
        git_sha=manifest["git_sha"],
        created_utc=manifest["created_utc"],
        engines=dict(manifest["engines"]),
        assumptions=tuple(manifest["assumptions"]),
        tracks=tuple(tracks),
        events=tuple(
            Event(t_s=e["t_s"], kind=e["kind"], body_id=e["body_id"], data=dict(e["data"]))
            for e in events
        ),
        verification=Verification(
            optimizer_converged=verification["optimizer_converged"],
            physics_verified=verification["physics_verified"],
            mission_constraints_passed=verification["mission_constraints_passed"],
            checks=checks,
        ),
    )


def _write_track_csv(track: BodyTrack) -> str:
    buf = io.StringIO()
    extras = list(track.extra_columns)
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(REQUIRED_CSV_COLUMNS) + extras)
    n = len(track.t_s)
    for i in range(n):
        row = [track.t_s[i], track.mass_kg[i],
               *track.position_m[i], *track.velocity_m_s[i],
               *(track.extra_columns[name][i] for name in extras)]
        writer.writerow(row)
    return buf.getvalue()


def write_run(dir, run: FlightRun, *, validate: bool = True) -> None:
    """Write ``run`` as a FlightResult v1 directory (created if absent):
    manifest.json, trajectories/<body_id>.csv, events.json,
    verification.json. A run written by write_run always round-trips through
    read_run -- validated again on read. Pass ``validate=False`` to skip the
    manifest/events/verification schema checks on write (the read side still
    checks)."""
    root = Path(dir)
    (root / _RUN_TRAJECTORIES).mkdir(parents=True, exist_ok=True)
    manifest = _manifest_document(run)
    events = [dataclasses.asdict(e) | {"data": dict(e.data)} for e in run.events]
    verification = {
        "optimizer_converged": run.verification.optimizer_converged,
        "physics_verified": run.verification.physics_verified,
        "mission_constraints_passed": run.verification.mission_constraints_passed,
        "checks": [dataclasses.asdict(c) for c in run.verification.checks],
    }
    if validate:
        _validate_run_manifest(manifest, f"{root / _RUN_MANIFEST} (in memory)")
        jsonschema.validate(instance=events, schema=_flight_result_def("events"))
        jsonschema.validate(instance=verification, schema=_flight_result_def("verification"))
    (root / _RUN_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for track in run.tracks:
        (root / _RUN_TRAJECTORIES / f"{track.body_id}.csv").write_text(
            _write_track_csv(track), encoding="utf-8")
    (root / _RUN_EVENTS).write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    (root / _RUN_VERIFICATION).write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")


def validate_run(dir) -> None:
    """Schema- and cross-check a run directory without keeping it around --
    what ``orbitopt flight validate <run_dir>`` runs. Raises the same
    errors as read_run; returns None on success."""
    read_run(dir)


def is_valid_run(dir) -> bool:
    """Non-raising form of validate_run() -- True/False instead of an
    exception, for callers that just want a filter/check."""
    try:
        validate_run(dir)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


def validate_path(path) -> str:
    """Validate whatever ``path`` is -- a FlightTarget document (YAML/JSON
    file) or a FlightResult v1 run directory -- and return a one-line
    summary of what was found. Shared by the CLI and any caller that is
    handed a user path without knowing its kind; raises the same errors as
    validate_target / validate_run."""
    p = Path(path)
    if p.is_dir():
        validate_run(p)
        manifest = json.loads((p / _RUN_MANIFEST).read_text(encoding="utf-8"))
        events = json.loads((p / _RUN_EVENTS).read_text(encoding="utf-8"))
        return f"run {manifest['run_id']} ({len(manifest['bodies'])} bodies, {len(events)} events)"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: expected a YAML/JSON mapping at the top level, got {type(doc).__name__}")
    validate_target(doc)
    return f"flight target {doc['mission']['id']} ({doc['mission']['title']})"
