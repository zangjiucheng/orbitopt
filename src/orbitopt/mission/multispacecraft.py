"""orbitopt multi-spacecraft mission contract (E12-01, task #391): the
shared reference frame/epoch two or more spacecraft are coordinated on, the
Target/Chaser roster, the multi-body event vocabulary, and the gate that
binds per-spacecraft trajectories (reused ``BodyTrack``s, FlightResult v1
format) to one mission and replays them on ONE unified time axis.

``orbitopt.mission.contract`` is a single-spacecraft seam: one
:class:`~orbitopt.mission.contract.HandoffState` crosses one segment
boundary. A rendezvous mission has *two* living trajectories at once, so
this module adds the mission level *around* that contract -- nothing here
changes ``HandoffState``, ``SegmentResult`` or any existing interface:

* :class:`RelativeState` -- chaser state minus target state at one shared
  epoch in one shared frame, built by :meth:`RelativeState.from_states` from
  two ``HandoffState``s (frame mismatch or epoch skew beyond tolerance is a
  construction failure, not a silent mix). Serialized as
  ``relative-state-1.0.json`` (:func:`relative_state_to_dict` /
  :func:`relative_state_from_dict`).
* :class:`MultiSpacecraftMission` -- mission config document
  (``multi-spacecraft-mission-1.0.json``): a shared frame + shared epoch
  (the one Epoch/timescale and FRAMES vocabulary), a spacecraft roster with
  mandatory TARGET/CHASER roles, optional named :class:`Zone` spheres and
  :class:`MissionEvent` entries from the multi-body vocabulary
  ``maneuver / hold_point_arrival / zone_entry / zone_exit``.
* :func:`validate_mission` -- the gate: roster/reference checks, plus the
  per-track binding checks against reused ``BodyTrack`` inputs -- frame
  mixing, epoch misalignment, unit incoherence (position and velocity
  sampled at inconsistent scales, caught by chord-vs-velocity agreement),
  event coverage. Reports (like ``check_handoff``), never silently fixes.
* :func:`build_unified_replay` -- takes a mission + its tracks, requires a
  passing validation, and emits the ``multispacecraft-replay-1.0.json``
  document: every spacecraft resampled onto one strictly increasing
  TDB-seconds axis (absolute, timescale-aware via
  :meth:`~orbitopt.mission.contract.Epoch.to_tdb_seconds`), so any replay
  reader advances all spacecraft with one clock.

Units are SI everywhere and carried in the field names (``*_m``,
``*_m_s``, ``*_tdb_s``) -- there is no unit field to get wrong, and a
producer that wrote kilometres into one field is caught by the
unit-coherence gate (its ``v_m_s`` and the time-difference of its
``position_m`` then disagree by the scale factor), one that swapped both
fields at once by the radius-plausibility bound inside the same gate, and
non-finite values at construction.

    from orbitopt.mission.multispacecraft import (
        MultiSpacecraftMission, SpacecraftSpec, RelativeState,
        validate_mission, build_unified_replay)

    report = validate_mission(mission, tracks)      # gate the binding
    assert report.passed, [c.reason_code for c in report.failures]
    replay = build_unified_replay(mission, tracks)  # one clock, two tracks

Same contract-first dependency discipline as ``orbitopt.mission.contract``:
standard library + jsonschema (+ orbitopt.timescales / the stdlib-only
flightlab contract types) -- no numpy, no Qt, no compute stack.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping, Sequence

import jsonschema

from orbitopt.flightlab.contract import BodyTrack
from orbitopt.mission.contract import (
    EPOCH_TIMESCALES,
    MISSION_FRAMES,
    Epoch,
    HandoffState,
)
from orbitopt.timescales import (
    SECONDS_PER_DAY,
    tdb_seconds_to_utc_mjd2000,
    utc_datetime_to_mjd2000,
    utc_mjd2000_to_tdb_seconds,
)

#: Schema version of all three documents this module writes
#: (``relative-state-1.0.json``, ``multi-spacecraft-mission-1.0.json``,
#: ``multispacecraft-replay-1.0.json``). Same compatibility policy as
#: orbitopt.mission.contract: optional fields are additive; breaking changes
#: bump to 2.0 with a new schema file.
SCHEMA_VERSION = "1.0"

#: The two roles of a rendezvous mission. Mandatory on every roster entry:
#: exactly one TARGET, at least one CHASER (checked by
#: :func:`validate_mission`).
ROLES = ("TARGET", "CHASER")

#: The multi-body event vocabulary. Single-vehicle ascent kinds (meco,
#: separation, ...) stay in ``orbitopt.flightlab.contract.EVENT_KINDS`` --
#: these are the kinds that only exist because two spacecraft share a
#: scenario.
MISSION_EVENT_KINDS = (
    "maneuver",
    "hold_point_arrival",
    "zone_entry",
    "zone_exit",
)


# --------------------------------------------------------------------------
# schema loading and value coercion (same discipline as mission.contract)
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _schema(name: str) -> dict:
    """Load and cache a packaged JSON Schema by filename -- same
    importlib.resources approach as orbitopt.mission.contract._schema."""
    traversable = resources.files("orbitopt").joinpath("schemas", name)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"No packaged schema file orbitopt/schemas/{name}") from exc
    return json.loads(text)


def _finite(value: Any, where: str) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {value!r} is not a number") from exc
    if not math.isfinite(t):
        raise ValueError(f"{where}: {value!r} is not finite")
    return t


def _vec3(value: Any, where: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{where}: expected a 3-vector, got {value!r}")
    parts = tuple(value)
    if len(parts) != 3:
        raise ValueError(f"{where}: expected 3 components, got {len(parts)}")
    return tuple(_finite(c, f"{where}[{i}]") for i, c in enumerate(parts))  # type: ignore[return-value]


def _mapping_of_str(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{where}: keys must be strings, got {key!r}")
        out[key] = item
    return out


def _nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _epoch_doc(epoch: Epoch) -> dict:
    return {"mjd2000": epoch.mjd2000, "timescale": epoch.timescale}


def _epoch_from_doc(doc: Mapping[str, Any], where: str) -> Epoch:
    if not isinstance(doc, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {doc!r}")
    return Epoch(mjd2000=_finite(doc["mjd2000"], f"{where}.mjd2000"),
                 timescale=doc["timescale"])


def _version_gate(doc: Any, schema_file: str, label: str) -> None:
    """schema_version picks the schema, like state_from_dict /
    validate_target: missing/unknown version is a ValueError, a document of
    THIS version is then jsonschema-validated."""
    version = doc.get("schema_version") if isinstance(doc, Mapping) else None
    if version is None:
        raise ValueError(
            f"{label} document has no 'schema_version' field -- set "
            f"schema_version={SCHEMA_VERSION!r} (see orbitopt/schemas/{schema_file})."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"no {label} schema for version {version!r} (this orbitopt "
            f"implements {SCHEMA_VERSION!r}; see orbitopt/schemas/)"
        )
    jsonschema.validate(instance=dict(doc), schema=_schema(schema_file))


# --------------------------------------------------------------------------
# RelativeState
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelativeState:
    """One spacecraft's state RELATIVE to another: ``r_chaser - r_target``
    and ``v_chaser - v_target``, valid only at one shared epoch in one
    shared frame (a difference of vectors is meaningless otherwise). Build
    it from two handoff states with :meth:`from_states` -- that is where
    the shared-frame / aligned-epoch rule is enforced -- or interpolate it
    off a replay timeline with
    :func:`relative_state_from_tracks`. Derived: :attr:`range_m`,
    :attr:`range_rate_m_s`."""

    target_id: str
    chaser_id: str
    epoch: Epoch
    frame: str
    r_rel_m: tuple[float, float, float]
    v_rel_m_s: tuple[float, float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_id", _nonempty_str(self.target_id, "RelativeState.target_id"))
        object.__setattr__(self, "chaser_id", _nonempty_str(self.chaser_id, "RelativeState.chaser_id"))
        if self.target_id == self.chaser_id:
            raise ValueError(
                f"RelativeState: target_id == chaser_id ({self.target_id!r}) -- a spacecraft "
                "is not relative to itself; use two distinct roster members"
            )
        if not isinstance(self.epoch, Epoch):
            raise ValueError(f"RelativeState.epoch must be an Epoch, got {self.epoch!r}")
        if self.frame not in MISSION_FRAMES:
            raise ValueError(f"RelativeState.frame {self.frame!r} not in {MISSION_FRAMES}")
        object.__setattr__(self, "r_rel_m", _vec3(self.r_rel_m, "RelativeState.r_rel_m"))
        object.__setattr__(self, "v_rel_m_s", _vec3(self.v_rel_m_s, "RelativeState.v_rel_m_s"))
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "RelativeState.metadata"))

    @property
    def range_m(self) -> float:
        """|r_rel| [m]."""
        return math.sqrt(sum(c * c for c in self.r_rel_m))

    @property
    def range_rate_m_s(self) -> float:
        """d/dt |r_rel| = (r . v) / |r| [m/s]; 0.0 at zero range (closing
        rate of an overlapping state is undefined -- report 0, not NaN)."""
        r = self.range_m
        if r == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(self.r_rel_m, self.v_rel_m_s, strict=True)) / r

    @classmethod
    def from_states(
        cls,
        target: HandoffState,
        chaser: HandoffState,
        epoch_alignment_s: float = 1.0,
    ) -> "RelativeState":
        """Form the chaser-minus-target relative state from two handoff
        states. Raises ``ValueError`` -- this is an input error, not a
        reportable handoff -- when the states are not comparable:

        * ``frame_mismatch``: the two states declare different frames. A
          GCRS r minus an ECEF r is not a relative position; convert first
          (orbitopt.flightlab.frames), inside whichever producer mixed them.
        * ``epoch_mismatch``: the states' epochs (each converted from its
          own declared timescale via ``Epoch.to_tdb_seconds``) differ by
          more than ``epoch_alignment_s``. At 7.5 km/s a 1 s skew is a 7.5 km
          fake relative position, so the skew gate is the whole point of
          sharing an epoch.
        """
        if not isinstance(target, HandoffState) or not isinstance(chaser, HandoffState):
            raise ValueError(
                f"RelativeState.from_states needs two HandoffStates, got "
                f"{type(target).__name__} and {type(chaser).__name__}"
            )
        if target.frame != chaser.frame:
            raise ValueError(
                f"frame_mismatch: target {target.frame} != chaser {chaser.frame} -- "
                "convert to one frame (orbitopt.flightlab.frames) before differencing"
            )
        alignment = _finite(epoch_alignment_s, "epoch_alignment_s")
        if alignment < 0.0:
            raise ValueError(f"epoch_alignment_s must be >= 0, got {alignment}")
        skew = abs(target.epoch.delta_seconds(chaser.epoch))
        if skew > alignment:
            raise ValueError(
                f"epoch_mismatch: target and chaser epochs differ by {skew:g} s "
                f"(> {alignment:g} s tolerance) -- align the timeline or the relative "
                "state measures the two spacecraft at two different times"
            )
        target_id = str(target.metadata.get("spacecraft_id", target.stage_config.get("spacecraft_id", "")))
        chaser_id = str(chaser.metadata.get("spacecraft_id", chaser.stage_config.get("spacecraft_id", "")))
        if not target_id or not chaser_id:
            raise ValueError(
                "RelativeState.from_states: state.metadata['spacecraft_id'] is required to "
                "name the pair (HandoffState carries no id field of its own)"
            )
        return cls(
            target_id=target_id,
            chaser_id=chaser_id,
            epoch=target.epoch,
            frame=target.frame,
            r_rel_m=tuple(a - b for a, b in zip(chaser.r_m, target.r_m, strict=True)),
            v_rel_m_s=tuple(a - b for a, b in zip(chaser.v_m_s, target.v_m_s, strict=True)),
        )

    def to_dict(self) -> dict:
        """The relative-state-1.0 JSON document for this state."""
        return {
            "schema_version": SCHEMA_VERSION,
            "target_id": self.target_id,
            "chaser_id": self.chaser_id,
            "epoch": _epoch_doc(self.epoch),
            "frame": self.frame,
            "position_m": list(self.r_rel_m),
            "velocity_m_s": list(self.v_rel_m_s),
            "metadata": dict(self.metadata),
        }


def relative_state_to_dict(state: RelativeState) -> dict:
    """Serialize a validated RelativeState (construction already checked
    everything; this mirrors orbitopt.mission.contract.state_to_dict)."""
    if not isinstance(state, RelativeState):
        raise ValueError(f"not a RelativeState: {state!r}")
    return state.to_dict()


def relative_state_from_dict(doc: Mapping[str, Any]) -> RelativeState:
    """Parse a relative-state-1.0 document. Raises
    ``jsonschema.ValidationError`` on a schema mismatch (unknown extra
    fields and wrong field names -- e.g. a ``position_km`` -- are schema
    rejections; there is no unit field to mis-set, the unit IS the field
    name) or ``ValueError`` on a missing/unknown schema_version, a
    non-finite number or a bad epoch."""
    _version_gate(doc, "relative-state-1.0.json", "relative state")
    epoch = _epoch_from_doc(doc["epoch"], "epoch")
    if epoch.timescale == "UTC":
        utc_mjd2000_to_tdb_seconds(epoch.mjd2000)  # reject pre-1972 UTC labels early
    else:
        epoch.to_tdb_seconds()
    return RelativeState(
        target_id=doc["target_id"],
        chaser_id=doc["chaser_id"],
        epoch=epoch,
        frame=doc["frame"],
        r_rel_m=_vec3(doc["position_m"], "position_m"),
        v_rel_m_s=_vec3(doc["velocity_m_s"], "velocity_m_s"),
        metadata=doc.get("metadata", {}),
    )


def is_valid_relative_state_document(doc: Mapping[str, Any]) -> bool:
    """Non-raising form of :func:`relative_state_from_dict`."""
    try:
        relative_state_from_dict(doc)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------
# Roster members, zones, events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpacecraftSpec:
    """One roster entry: a unique ``spacecraft_id`` (matching its
    ``BodyTrack.body_id``) and its mandatory role. The role is a required
    constructor argument -- a missing role is a TypeError at the call site
    and a ``required``-violation from the schema, never a silent default."""

    spacecraft_id: str
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "spacecraft_id", _nonempty_str(self.spacecraft_id, "SpacecraftSpec.spacecraft_id"))
        if self.role not in ROLES:
            raise ValueError(f"SpacecraftSpec.role {self.role!r} not in {ROLES} -- every spacecraft is a TARGET or a CHASER")
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "SpacecraftSpec.metadata"))

    def to_dict(self) -> dict:
        return {"spacecraft_id": self.spacecraft_id, "role": self.role, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class Zone:
    """A named spherical region the event vocabulary refers to (keep-out
    sphere, approach corridor volume, hold-point box circumsphere). Centre
    in the mission frame, metres -- the same origin every track uses."""

    name: str
    center_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, "name", _nonempty_str(self.name, "Zone.name"))
        object.__setattr__(self, "center_m", _vec3(self.center_m, "Zone.center_m"))
        radius = _finite(self.radius_m, "Zone.radius_m")
        if radius <= 0.0:
            raise ValueError(f"Zone.radius_m must be > 0, got {radius}")
        object.__setattr__(self, "radius_m", radius)

    def to_dict(self) -> dict:
        return {"name": self.name, "center_m": list(self.center_m), "radius_m": self.radius_m}


@dataclass(frozen=True)
class MissionEvent:
    """One multi-body event: who (``spacecraft_id``, plus optional
    ``counterpart_id``), what (``kind`` from :data:`MISSION_EVENT_KINDS`),
    when (an ``Epoch`` on the shared timeline). Contract per kind:
    ``zone_entry``/``zone_exit`` must name a declared ``zone``;
    ``hold_point_arrival`` must name a ``hold_point_id``."""

    spacecraft_id: str
    kind: str
    epoch: Epoch
    counterpart_id: str | None = None
    zone: str | None = None
    hold_point_id: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "spacecraft_id", _nonempty_str(self.spacecraft_id, "MissionEvent.spacecraft_id"))
        if self.kind not in MISSION_EVENT_KINDS:
            raise ValueError(f"MissionEvent.kind {self.kind!r} not in {MISSION_EVENT_KINDS}")
        if not isinstance(self.epoch, Epoch):
            raise ValueError(f"MissionEvent.epoch must be an Epoch, got {self.epoch!r}")
        if self.kind in ("zone_entry", "zone_exit"):
            object.__setattr__(self, "zone", _nonempty_str(self.zone, f"MissionEvent(zone={self.kind}).zone"))
        elif self.zone is not None:
            raise ValueError(f"MissionEvent.zone is only meaningful for zone_entry/zone_exit, not {self.kind!r}")
        if self.kind == "hold_point_arrival":
            object.__setattr__(
                self, "hold_point_id", _nonempty_str(self.hold_point_id, "MissionEvent(hold_point_arrival).hold_point_id")
            )
        elif self.hold_point_id is not None:
            raise ValueError(f"MissionEvent.hold_point_id is only meaningful for hold_point_arrival, not {self.kind!r}")
        if self.counterpart_id is not None:
            _nonempty_str(self.counterpart_id, "MissionEvent.counterpart_id")
            if self.counterpart_id == self.spacecraft_id:
                raise ValueError(f"MissionEvent: counterpart_id == spacecraft_id ({self.spacecraft_id!r})")
        object.__setattr__(self, "data", _mapping_of_str(self.data, "MissionEvent.data"))

    def to_dict(self) -> dict:
        doc: dict[str, Any] = {
            "spacecraft_id": self.spacecraft_id,
            "kind": self.kind,
            "epoch": _epoch_doc(self.epoch),
            "data": dict(self.data),
        }
        if self.counterpart_id is not None:
            doc["counterpart_id"] = self.counterpart_id
        if self.zone is not None:
            doc["zone"] = self.zone
        if self.hold_point_id is not None:
            doc["hold_point_id"] = self.hold_point_id
        return doc


# --------------------------------------------------------------------------
# MultiSpacecraftMission
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiSpacecraftMission:
    """The mission-level contract: an id/title, the ONE shared frame and the
    ONE shared epoch (its ``timescale`` is part of the Epoch and never
    guessed), the roster, optional zones and events. The reference is
    declared here once so no per-spacecraft document can quietly disagree --
    :func:`validate_mission` fails the tracks that do.

    Construct with keyword arguments; round-trips through
    :func:`mission_from_dict` / :meth:`to_dict` against the versioned
    ``multi-spacecraft-mission-1.0.json`` schema. Counting rules (unique
    ids, exactly one TARGET, event/zone references) need the whole document
    and live in :func:`validate_mission`."""

    mission_id: str
    title: str
    frame: str
    epoch: Epoch
    spacecraft: tuple[SpacecraftSpec, ...]
    zones: tuple[Zone, ...] = ()
    events: tuple[MissionEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "mission_id", _nonempty_str(self.mission_id, "MultiSpacecraftMission.mission_id"))
        object.__setattr__(self, "title", _nonempty_str(self.title, "MultiSpacecraftMission.title"))
        if self.frame not in MISSION_FRAMES:
            raise ValueError(f"MultiSpacecraftMission.frame {self.frame!r} not in {MISSION_FRAMES}")
        if not isinstance(self.epoch, Epoch):
            raise ValueError(f"MultiSpacecraftMission.epoch must be an Epoch, got {self.epoch!r}")
        specs = _typed_tuple(self.spacecraft, SpacecraftSpec, "MultiSpacecraftMission.spacecraft")
        if len(specs) < 2:
            raise ValueError(f"MultiSpacecraftMission needs at least 2 spacecraft, got {len(specs)}")
        object.__setattr__(self, "spacecraft", specs)
        object.__setattr__(self, "zones", _typed_tuple(self.zones, Zone, "MultiSpacecraftMission.zones"))
        object.__setattr__(self, "events", _typed_tuple(self.events, MissionEvent, "MultiSpacecraftMission.events"))
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "MultiSpacecraftMission.metadata"))

    @property
    def target(self) -> SpacecraftSpec:
        """The single TARGET (raises unless the roster has exactly one --
        counting is the caller's intent here, validation reports it)."""
        targets = [s for s in self.spacecraft if s.role == "TARGET"]
        if len(targets) != 1:
            raise ValueError(f"mission has {len(targets)} TARGET spacecraft, need exactly 1")
        return targets[0]

    def to_dict(self) -> dict:
        """The multi-spacecraft-mission-1.0 document."""
        return {
            "schema_version": SCHEMA_VERSION,
            "mission_id": self.mission_id,
            "title": self.title,
            "frame": self.frame,
            "epoch": _epoch_doc(self.epoch),
            "spacecraft": [s.to_dict() for s in self.spacecraft],
            "zones": [z.to_dict() for z in self.zones],
            "events": [e.to_dict() for e in self.events],
            "metadata": dict(self.metadata),
        }


def _typed_tuple(value: Any, cls: type, where: str) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{where}: expected an iterable of {cls.__name__}, got {value!r}")
    items = tuple(value)
    for i, item in enumerate(items):
        if not isinstance(item, cls):
            raise ValueError(f"{where}[{i}]: expected {cls.__name__}, got {item!r}")
    return items


def mission_to_dict(mission: MultiSpacecraftMission) -> dict:
    """Serialize a constructed mission."""
    if not isinstance(mission, MultiSpacecraftMission):
        raise ValueError(f"not a MultiSpacecraftMission: {mission!r}")
    return mission.to_dict()


def _event_from_doc(doc: Mapping[str, Any], where: str) -> MissionEvent:
    return MissionEvent(
        spacecraft_id=doc["spacecraft_id"],
        kind=doc["kind"],
        epoch=_epoch_from_doc(doc["epoch"], f"{where}.epoch"),
        counterpart_id=doc.get("counterpart_id"),
        zone=doc.get("zone"),
        hold_point_id=doc.get("hold_point_id"),
        data=doc.get("data", {}),
    )


def mission_from_dict(doc: Mapping[str, Any]) -> MultiSpacecraftMission:
    """Parse a multi-spacecraft-mission-1.0 document. Raises
    ``jsonschema.ValidationError`` on a schema mismatch (a missing
    ``role``, an unknown role or a ``PROBE`` role name is a schema
    rejection) or ``ValueError`` on a missing/unknown schema_version, a
    non-finite / bad value, or an event the dataclass rejects (a
    ``zone_entry`` without a zone). Counting rules (duplicate ids, TARGET
    multiplicity, zone/event references) come out of
    :func:`validate_mission`, not here."""
    _version_gate(doc, "multi-spacecraft-mission-1.0.json", "multi-spacecraft mission")
    epoch = _epoch_from_doc(doc["epoch"], "epoch")
    if epoch.timescale == "UTC":
        utc_mjd2000_to_tdb_seconds(epoch.mjd2000)  # reject pre-1972 UTC labels early
    else:
        epoch.to_tdb_seconds()
    return MultiSpacecraftMission(
        mission_id=doc["mission_id"],
        title=doc["title"],
        frame=doc["frame"],
        epoch=epoch,
        spacecraft=tuple(
            SpacecraftSpec(spacecraft_id=s["spacecraft_id"], role=s["role"], metadata=s.get("metadata", {}))
            for s in doc["spacecraft"]
        ),
        zones=tuple(
            Zone(name=z["name"], center_m=_vec3(z["center_m"], f"zones[{i}].center_m"), radius_m=_finite(z["radius_m"], f"zones[{i}].radius_m"))
            for i, z in enumerate(doc.get("zones", []))
        ),
        events=tuple(_event_from_doc(e, f"events[{i}]") for i, e in enumerate(doc.get("events", []))),
        metadata=doc.get("metadata", {}),
    )


def is_valid_mission_document(doc: Mapping[str, Any]) -> bool:
    """Non-raising form of :func:`mission_from_dict`."""
    try:
        mission_from_dict(doc)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------
# Mission validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionCheck:
    """One validation gate: id, verdict, the measured number and the limit
    it was judged against (may be None when the check failed before
    measuring), and a machine-readable ``reason_code`` -- same shape as
    ``orbitopt.mission.contract.HandoffCheck``, so CI gates both alike."""

    id: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    reason_code: str = ""


@dataclass(frozen=True)
class MissionValidationReport:
    """The verdict of :func:`validate_mission`: every check that ran.
    ``passed`` is the AND of the checks; ``failures`` names the ones that
    said no. Never raises for a bad mission -- it reports it; invalid
    *documents* raise at parse time."""

    mission: MultiSpacecraftMission
    checks: tuple[MissionCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[MissionCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True)
class MissionTolerances:
    """The numeric gates of :func:`validate_mission`.
    ``epoch_alignment_s``: how far a track's own epoch (BodyTrack carries
    ``epoch_utc``) may sit from the mission epoch -- it bounds the timing
    error the shared reference tolerates, not the physical drift INSIDE a
    track (that is the producer's problem, and the track's own samples
    carry it). ``unit_coherence_m_s``: the largest agreement error between
    a track's velocity samples and the time-difference of its position
    samples (midpoint velocity vs chord velocity); a track that wrote m or
    km/s into an m/s field disagrees by orders of magnitude, a genuinely
    curved trajectory sampled finely disagrees by much less than the
    default."""

    epoch_alignment_s: float = 1.0
    unit_coherence_m_s: float = 10.0

    def __post_init__(self):
        for name in ("epoch_alignment_s", "unit_coherence_m_s"):
            value = _finite(getattr(self, name), f"MissionTolerances.{name}")
            if value < 0.0:
                raise ValueError(f"MissionTolerances.{name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)


def _check_any(checks: list[MissionCheck]) -> MissionCheck:
    passed = all(c.passed for c in checks)
    first_bad = next((c for c in checks if not c.passed), None)
    codes = ";".join(c.reason_code for c in checks if not c.passed)
    if passed:
        worst = max(checks, key=lambda c: c.value or 0.0)
        return MissionCheck(id=checks[0].id, passed=True, value=worst.value, limit=worst.limit,
                            unit=checks[0].unit)
    return MissionCheck(id=first_bad.id, passed=False, value=first_bad.value, limit=first_bad.limit,
                        unit=first_bad.unit, reason_code=codes)


def _roles_checks(mission: MultiSpacecraftMission) -> list[MissionCheck]:
    ids = [s.spacecraft_id for s in mission.spacecraft]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        return [MissionCheck(
            id="msm.spacecraft_roles", passed=False, unit="",
            reason_code="duplicate_spacecraft_id:" + ",".join(dupes),
        )]
    n_target = sum(1 for s in mission.spacecraft if s.role == "TARGET")
    n_chaser = sum(1 for s in mission.spacecraft if s.role == "CHASER")
    checks = []
    if n_target == 0:
        checks.append(MissionCheck(id="msm.spacecraft_roles", passed=False, reason_code="no_target"))
    elif n_target > 1:
        checks.append(MissionCheck(id="msm.spacecraft_roles", passed=False, reason_code=f"multiple_targets:{n_target}"))
    if n_chaser == 0:
        checks.append(MissionCheck(id="msm.spacecraft_roles", passed=False, reason_code="no_chaser"))
    if not checks:
        checks.append(MissionCheck(id="msm.spacecraft_roles", passed=True))
    return checks


def _reference_checks(mission: MultiSpacecraftMission) -> list[MissionCheck]:
    """Zone-name and event/counterpart references -- everything that needs
    more than the single value the dataclass already checked."""
    failures: list[str] = []
    names = [z.name for z in mission.zones]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        failures.append("duplicate_zone_name:" + ",".join(dupes))
    return [MissionCheck(id="msm.zones", passed=not failures, reason_code=";".join(failures))]


def _event_reference_checks(mission: MultiSpacecraftMission) -> MissionCheck:
    known = {s.spacecraft_id for s in mission.spacecraft}
    zones = {z.name for z in mission.zones}
    failures: list[str] = []
    for i, event in enumerate(mission.events):
        if event.spacecraft_id not in known:
            failures.append(f"unknown_spacecraft:events[{i}]={event.spacecraft_id!r}")
        if event.counterpart_id is not None and event.counterpart_id not in known:
            failures.append(f"unknown_counterpart:events[{i}]={event.counterpart_id!r}")
        if event.zone is not None and event.zone not in zones:
            failures.append(f"unknown_zone:events[{i}]={event.zone!r}")
    return MissionCheck(id="msm.event_references", passed=not failures, reason_code=";".join(failures))


def _track_epoch_tdb(track: BodyTrack) -> float:
    """Absolute TDB seconds of a BodyTrack's own epoch (BodyTrack epoch_utc
    is UTC by format definition)."""
    try:
        dt = datetime.fromisoformat(track.epoch_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"track {track.body_id!r}: {track.epoch_utc!r} is not an ISO-8601 timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError(f"track {track.body_id!r}: epoch_utc {track.epoch_utc!r} needs an explicit UTC zone")
    return Epoch(mjd2000=utc_datetime_to_mjd2000(dt.astimezone(timezone.utc)), timescale="UTC").to_tdb_seconds()


def _track_checks(mission: MultiSpacecraftMission, tracks: Sequence[BodyTrack],
                  tolerances: MissionTolerances) -> list[MissionCheck]:
    known = [s.spacecraft_id for s in mission.spacecraft]
    by_id: dict[str, BodyTrack] = {}
    coverage: list[MissionCheck] = []
    aligned: dict[str, tuple[float, BodyTrack]] = {}
    for track in tracks:
        if not isinstance(track, BodyTrack):
            raise ValueError(f"validate_mission tracks must be orbitopt.flightlab.contract.BodyTrack, got {track!r}")
        if track.body_id not in known:
            coverage.append(MissionCheck(id="msm.track_coverage", passed=False, reason_code=f"unknown_track:{track.body_id}"))
        elif track.body_id in by_id:
            coverage.append(MissionCheck(id="msm.track_coverage", passed=False, reason_code=f"duplicate_track:{track.body_id}"))
        else:
            by_id[track.body_id] = track
    missing = sorted(set(known) - set(by_id))
    for mid in missing:
        coverage.append(MissionCheck(id="msm.track_coverage", passed=False, reason_code=f"missing_track:{mid}"))

    frame_failures = [f"frame_mismatch:{t.body_id}:{t.frame}" for t in by_id.values() if t.frame != mission.frame]
    epoch_failures: list[str] = []
    try:
        mission_tdb = mission.epoch.to_tdb_seconds()
    except ValueError:
        mission_tdb = None
        epoch_failures.append("mission_epoch_unparseable")
    if mission_tdb is not None:
        for t in by_id.values():
            try:
                tdb = _track_epoch_tdb(t)
            except ValueError:
                epoch_failures.append(f"epoch_unparseable:{t.body_id}")
                continue
            skew = abs(tdb - mission_tdb)
            if skew > tolerances.epoch_alignment_s:
                epoch_failures.append(f"epoch_mismatch:{t.body_id}:{skew:g}s")
            else:
                aligned[t.body_id] = (tdb, t)

    time_failures: list[str] = []
    coherence_failures: list[MissionCheck] = []
    for t in by_id.values():
        if len(t.t_s) < 2:
            time_failures.append(f"track_too_short:{t.body_id}")
            continue
        ts = [float(v) for v in t.t_s]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            time_failures.append(f"time_not_monotonic:{t.body_id}")
            continue
        err = _unit_coherence_error(t)
        scale = _radius_plausibility(t)
        if err > tolerances.unit_coherence_m_s:
            coherence_failures.append(MissionCheck(
                id="msm.unit_coherence", passed=False, value=err, limit=tolerances.unit_coherence_m_s,
                unit="m/s", reason_code=f"unit_incoherence:{t.body_id}",
            ))
        elif scale is not None:
            coherence_failures.append(MissionCheck(
                id="msm.unit_coherence", passed=False, value=scale,
                unit="m", reason_code=f"implausible_radius:{t.body_id}:{scale:g}",
            ))
        else:
            coherence_failures.append(MissionCheck(
                id="msm.unit_coherence", passed=True, value=err, limit=tolerances.unit_coherence_m_s, unit="m/s"))

    window: MissionCheck
    if len(aligned) == len(by_id) and by_id:
        lo = max(tdb + t.t_start_s for tdb, t in aligned.values())
        hi = min(tdb + t.t_end_s for tdb, t in aligned.values())
        window = MissionCheck(id="msm.common_window", passed=lo < hi, value=hi - lo, unit="s",
                              reason_code="" if lo < hi else "empty_common_window")
    else:
        window = MissionCheck(id="msm.common_window", passed=False, reason_code="empty_common_window")

    event_failures: list[str] = []
    for i, event in enumerate(mission.events):
        t = aligned.get(event.spacecraft_id)
        if t is None:
            continue  # already reported by track_coverage / epoch / roles
        tdb, track = t
        rel = event.epoch.to_tdb_seconds() - tdb
        if rel < track.t_start_s or rel > track.t_end_s:
            event_failures.append(f"event_outside_track:events[{i}]={event.kind}")
    checks = [
        _check_any(coverage) if coverage else MissionCheck(id="msm.track_coverage", passed=True),
        MissionCheck(id="msm.frame_consistency", passed=not frame_failures,
                     reason_code=";".join(frame_failures)),
        MissionCheck(id="msm.epoch_consistency", passed=not epoch_failures,
                     limit=tolerances.epoch_alignment_s, unit="s",
                     reason_code=";".join(epoch_failures)),
        MissionCheck(id="msm.track_time", passed=not time_failures, reason_code=";".join(time_failures)),
        _check_any(coherence_failures) if coherence_failures else MissionCheck(id="msm.unit_coherence", passed=True),
        window,
        MissionCheck(id="msm.event_timeline_coverage", passed=not event_failures, reason_code=";".join(event_failures)),
    ]
    return checks


_PLAUSIBLE_RADIUS_M = (1.0e5, 1.0e13)


def _radius_plausibility(track: BodyTrack) -> float | None:
    """Median |position| [m] when the track sits outside the range that any
    body-track around a natural body can occupy, else None. A track that
    swapped BOTH fields from m to km stays chord-coherent -- no scale test
    sees a mixed ratio -- but it leaves the plausible range on the small
    side (a mislabelled LEO reads 7 000 m, under any real radius), so the
    absolute bound catches what the coherence gate cannot."""
    medians = sorted(math.dist(p, (0.0, 0.0, 0.0)) for p in track.position_m)
    median = medians[len(medians) // 2]
    if _PLAUSIBLE_RADIUS_M[0] <= median <= _PLAUSIBLE_RADIUS_M[1]:
        return None
    return median


def _unit_coherence_error(track: BodyTrack) -> float:
    """Max over consecutive sample pairs of |(r[i+1]-r[i])/dt - (v[i]+v[i+1])/2|
    [m/s]. A track whose position is in m and velocity in km/s (or vice
    versa) trips this by the scale factor: the declared velocity says 7500,
    the positions say 7.5. A straight-line constant-velocity track gives
    exactly 0; a finely sampled curved one gives its O(omega*dt/2) chord
    error, well inside the default tolerance. This gate cannot see a track
    that swapped BOTH fields at once -- :func:`_radius_plausibility`
    covers that side."""
    worst = 0.0
    ts = track.t_s
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0:
            continue
        chord = tuple((b - a) / dt for a, b in zip(track.position_m[i], track.position_m[i + 1], strict=True))
        vmid = tuple((a + b) * 0.5 for a, b in zip(track.velocity_m_s[i + 1], track.velocity_m_s[i], strict=True))
        err = math.sqrt(sum((c - v) ** 2 for c, v in zip(chord, vmid, strict=True)))
        worst = max(worst, err)
    return worst


def validate_mission(
    mission: MultiSpacecraftMission,
    tracks: Sequence[BodyTrack] = (),
    tolerances: MissionTolerances | None = None,
) -> MissionValidationReport:
    """Gate a mission, and (when tracks are given) the binding of its
    reused ``BodyTrack`` trajectories to the shared reference. Checks:

    ==========================================  =====================================
    id                                          rule / reason codes
    ==========================================  =====================================
    ``msm.spacecraft_roles``                    unique ids, exactly one TARGET, >= 1
                                                CHASER (``duplicate_spacecraft_id``,
                                                ``no_target``, ``multiple_targets``,
                                                ``no_chaser``)
    ``msm.zones``                               unique zone names
    ``msm.event_references``                    events name roster spacecraft, declared
                                                zones, roster counterparts
    ``msm.track_coverage``                      one track per spacecraft, no orphans
                                                (``missing_track:*``, ``unknown_track:*``,
                                                ``duplicate_track:*``)
    ``msm.frame_consistency``                   every track flies in the mission frame
                                                (``frame_mismatch:*`` -- frame mixing
                                                across a mission is a failure, convert
                                                upstream)
    ``msm.epoch_consistency``                   every track's epoch within
                                                ``epoch_alignment_s`` of the mission
                                                epoch, compared on TDB seconds
                                                (``epoch_mismatch:*``)
    ``msm.track_time``                          >= 2 samples, strictly increasing time
                                                (``track_too_short:*``, ``time_not_monotonic:*``)
    ``msm.unit_coherence``                      velocity samples agree with the
                                                time-difference of the position samples
                                                (``unit_incoherence:*`` -- catches
                                                km-written-as-m / km/s-written-as-m/s)
    ``msm.common_window``                       the tracks overlap on a non-empty time
                                                window (``empty_common_window``)
    ``msm.event_timeline_coverage``             every event falls inside its own
                                                spacecraft's track coverage
                                                (``event_outside_track:*``)
    ==========================================  =====================================

    ``passed`` is the AND of all of them; the report is data, so callers
    gate-or-collect as they see fit (same policy as ``check_handoff``)."""
    tol = tolerances if tolerances is not None else MissionTolerances()
    checks = _roles_checks(mission) + _reference_checks(mission) + [_event_reference_checks(mission)]
    if tracks:
        checks += _track_checks(mission, tracks, tol)
    return MissionValidationReport(mission=mission, checks=tuple(checks))


# --------------------------------------------------------------------------
# Unified replay
# --------------------------------------------------------------------------


def _epoch_from_tdb(tdb_seconds: float) -> Epoch:
    """A RelativeState epoch for an absolute TDB-seconds instant, expressed
    on the TDB day label (round-trips exactly through to_tdb_seconds)."""
    return Epoch(mjd2000=tdb_seconds / SECONDS_PER_DAY + 0.5, timescale="TDB")


def _interp3(ts: Sequence[float], values: Sequence[tuple[float, float, float]],
             t: float) -> tuple[float, float, float]:
    """Component-wise linear interpolation of a vec3 column at time ``t``
    (``ts`` strictly increasing; clamped at both ends -- the caller has
    already restricted ``t`` to the track's coverage)."""
    if t <= ts[0]:
        return tuple(values[0])  # type: ignore[return-value]
    if t >= ts[-1]:
        return tuple(values[-1])  # type: ignore[return-value]
    i = bisect.bisect_right(ts, t) - 1
    frac = (t - ts[i]) / (ts[i + 1] - ts[i])
    a, b = values[i], values[i + 1]
    return tuple(av + (bv - av) * frac for av, bv in zip(a, b, strict=True))  # type: ignore[return-value]


def build_unified_replay(
    mission: MultiSpacecraftMission,
    tracks: Sequence[BodyTrack],
    tolerances: MissionTolerances | None = None,
) -> dict:
    """Resample every spacecraft's ``BodyTrack`` onto ONE shared time axis
    and return the ``multispacecraft-replay-1.0`` document: the
    acceptance-carrying artefact that two trajectories can be replayed on a
    unified timeline. Refuses (``ValueError``) a mission whose
    :func:`validate_mission` gate fails -- an epoch-inconsistent,
    frame-mixed or unit-mixed mission must not reach a replay reader as a
    smooth-looking animation.

    The axis is absolute **TDB seconds since J2000** (the one comparable
    scale, leap-second and TDB-TT aware): every track's own ``epoch_utc``
    plus its relative ``t_s`` is converted through
    :meth:`Epoch.to_tdb_seconds`, so two producers that picked different
    declared timescales still land on the same number. Axis times are the
    union of all track sample times inside the tracks' common overlap
    window, so a reader advances all spacecraft with one clock and one
    ``seek``. Each spacecraft's entry carries its role and its
    interpolated position/velocity at each axis time; ``relative_states``
    gives the TARGET-referenced relative state of every CHASER at every
    axis time; events are placed on the same axis as ``t_tdb_s``."""
    report = validate_mission(mission, tracks, tolerances)
    if not report.passed:
        detail = "; ".join(f"{c.id}({c.reason_code})" for c in report.failures)
        raise ValueError(f"build_unified_replay: mission {mission.mission_id!r} failed validation: {detail}")
    if len(tracks) < 2:
        raise ValueError(
            f"build_unified_replay: a unified replay needs a track per spacecraft "
            f"(the roster has {len(mission.spacecraft)}), got {len(tracks)}")
    by_id = {t.body_id: t for t in tracks}
    track_tdb = {t.body_id: _track_epoch_tdb(t) for t in tracks}

    lo = max(track_tdb[b] + by_id[b].t_start_s for b in track_tdb)
    hi = min(track_tdb[b] + by_id[b].t_end_s for b in track_tdb)
    axis = _overlap_axis(by_id, track_tdb, lo, hi)
    if len(axis) < 2:
        raise ValueError(
            f"build_unified_replay: common overlap window [{lo!r}, {hi!r}] yields fewer "
            "than 2 axis samples -- the tracks do not share a replayable interval"
        )

    target = mission.target
    roles = {s.spacecraft_id: s.role for s in mission.spacecraft}
    spacecraft_docs = []
    for spec in mission.spacecraft:
        track = by_id[spec.spacecraft_id]
        tdb0 = track_tdb[spec.spacecraft_id]
        samples = []
        for t_tdb in axis:
            rel = t_tdb - tdb0
            samples.append({
                "t_tdb_s": t_tdb,
                "position_m": list(_interp3(track.t_s, track.position_m, rel)),
                "velocity_m_s": list(_interp3(track.t_s, track.velocity_m_s, rel)),
            })
        spacecraft_docs.append({
            "spacecraft_id": spec.spacecraft_id,
            "role": spec.role,
            "frame": mission.frame,
            "samples": samples,
        })

    relative_docs = []
    for spec in mission.spacecraft:
        if spec.role != "CHASER":
            continue
        ch, tg = by_id[spec.spacecraft_id], by_id[target.spacecraft_id]
        rc, rt = track_tdb[spec.spacecraft_id], track_tdb[target.spacecraft_id]
        for t_tdb in axis:
            r_rel = tuple(a - b for a, b in zip(
                _interp3(ch.t_s, ch.position_m, t_tdb - rc),
                _interp3(tg.t_s, tg.position_m, t_tdb - rt), strict=True))
            v_rel = tuple(a - b for a, b in zip(
                _interp3(ch.t_s, ch.velocity_m_s, t_tdb - rc),
                _interp3(tg.t_s, tg.velocity_m_s, t_tdb - rt), strict=True))
            relative_docs.append({
                "t_tdb_s": t_tdb,
                "target_id": target.spacecraft_id,
                "chaser_id": spec.spacecraft_id,
                "frame": mission.frame,
                "position_m": list(r_rel),
                "velocity_m_s": list(v_rel),
            })

    events_docs = []
    for event in sorted(mission.events, key=lambda e: e.epoch.to_tdb_seconds()):
        doc = event.to_dict()
        doc.pop("epoch")
        events_docs.append({"t_tdb_s": event.epoch.to_tdb_seconds(), **doc})

    document = {
        "schema_version": SCHEMA_VERSION,
        "mission_id": mission.mission_id,
        "frame": mission.frame,
        "time_axis_tdb_s": list(axis),
        "spacecraft": spacecraft_docs,
        "relative_states": relative_docs,
        "events": events_docs,
    }
    jsonschema.validate(instance=document, schema=_schema("multispacecraft-replay-1.0.json"))
    return document


def _overlap_axis(by_id: dict, track_tdb: dict, lo: float, hi: float) -> list[float]:
    """The shared axis: every track sample instant (absolute TDB seconds)
    that lies inside the common overlap window, de-duplicated and sorted."""
    times = set()
    for body_id, track in by_id.items():
        tdb0 = track_tdb[body_id]
        for t in track.t_s:
            absolute = tdb0 + float(t)
            if lo <= absolute <= hi:
                times.add(absolute)
    return sorted(times)


def is_valid_replay_document(doc: Mapping[str, Any]) -> bool:
    """Non-raising schema check of a multispacecraft-replay-1.0 document."""
    try:
        _version_gate(doc, "multispacecraft-replay-1.0.json", "multi-spacecraft replay")
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


def replay_from_dict(doc: Mapping[str, Any]) -> dict:
    """Validate a replay document (schema, version gate, strictly
    increasing axis, per-spacecraft sample times on the axis) and return
    it. Raises on anything invalid -- this is the reader a replay UI calls
    before it animates."""
    _version_gate(doc, "multispacecraft-replay-1.0.json", "multi-spacecraft replay")
    axis = [float(t) for t in doc["time_axis_tdb_s"]]
    for a, b in zip(axis, axis[1:]):
        if not b > a:
            raise ValueError(f"replay time_axis_tdb_s is not strictly increasing at {a!r} -> {b!r}")
    axis_set = set(axis)
    for craft in doc["spacecraft"]:
        prev = None
        for sample in craft["samples"]:
            t = float(sample["t_tdb_s"])
            if t not in axis_set:
                raise ValueError(f"replay: {craft['spacecraft_id']!r} sample at {t!r} s is off the shared axis")
            if prev is not None and t <= prev:
                raise ValueError(f"replay: {craft['spacecraft_id']!r} samples are not in ascending time")
            prev = t
        if craft["samples"][0]["t_tdb_s"] < axis[0] or craft["samples"][-1]["t_tdb_s"] > axis[-1]:
            raise ValueError(f"replay: {craft['spacecraft_id']!r} samples leave the axis window")
    return dict(doc)


def relative_state_from_tracks(
    mission: MultiSpacecraftMission,
    target_track: BodyTrack,
    chaser_track: BodyTrack,
    t_tdb_s: float,
) -> RelativeState:
    """The TARGET-to-CHASER relative state at an absolute TDB-seconds
    instant, interpolated linearly off the two (mission-validated) tracks.
    This is the live-number counterpart of
    :meth:`RelativeState.from_states` for the replay reader: ask for the
    range at any axis time without holding the whole document. On a roster
    of more than two spacecraft this validates the PAIR against the
    mission (a two-craft view of the roster), not the full fleet -- the
    other craft's tracks belong to other queries."""
    pair = {target_track.body_id: target_track, chaser_track.body_id: chaser_track}
    if len(pair) != 2:
        raise ValueError("relative_state_from_tracks: target and chaser must be different spacecraft")
    wanted = tuple(s for s in mission.spacecraft if s.spacecraft_id in pair)
    if {s.spacecraft_id for s in wanted} != set(pair):
        raise ValueError("relative_state_from_tracks: tracks name spacecraft outside the mission roster")
    roles = {s.spacecraft_id: s.role for s in wanted}
    if roles[target_track.body_id] != "TARGET" or roles[chaser_track.body_id] != "CHASER":
        raise ValueError("relative_state_from_tracks: the TARGET track must come first, the CHASER track second")
    pair_mission = _dc_replace(mission, spacecraft=(
        wanted[0], wanted[1]) if wanted[1].role == "CHASER" else (wanted[1], wanted[0]))
    report = validate_mission(pair_mission, (target_track, chaser_track))
    if not report.passed:
        detail = "; ".join(f"{c.id}({c.reason_code})" for c in report.failures)
        raise ValueError(f"relative_state_from_tracks: tracks failed validation: {detail}")
    t_tdb = _finite(t_tdb_s, "t_tdb_s")
    rt = _track_epoch_tdb(target_track)
    rc = _track_epoch_tdb(chaser_track)
    r_rel = tuple(a - b for a, b in zip(
        _interp3(chaser_track.t_s, chaser_track.position_m, t_tdb - rc),
        _interp3(target_track.t_s, target_track.position_m, t_tdb - rt), strict=True))
    v_rel = tuple(a - b for a, b in zip(
        _interp3(chaser_track.t_s, chaser_track.velocity_m_s, t_tdb - rc),
        _interp3(target_track.t_s, target_track.velocity_m_s, t_tdb - rt), strict=True))
    return RelativeState(
        target_id=target_track.body_id,
        chaser_id=chaser_track.body_id,
        epoch=_epoch_from_tdb(t_tdb),
        frame=mission.frame,
        r_rel_m=r_rel,
        v_rel_m_s=v_rel,
    )


__all__ = [
    "EPOCH_TIMESCALES",
    "MISSION_EVENT_KINDS",
    "MISSION_FRAMES",
    "ROLES",
    "SCHEMA_VERSION",
    "Epoch",
    "HandoffState",
    "MissionCheck",
    "MissionEvent",
    "MissionTolerances",
    "MissionValidationReport",
    "MultiSpacecraftMission",
    "RelativeState",
    "SpacecraftSpec",
    "Zone",
    "build_unified_replay",
    "is_valid_mission_document",
    "is_valid_relative_state_document",
    "is_valid_replay_document",
    "mission_from_dict",
    "mission_to_dict",
    "relative_state_from_dict",
    "relative_state_from_tracks",
    "relative_state_to_dict",
    "replay_from_dict",
    "validate_mission",
]
