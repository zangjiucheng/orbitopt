"""The five closed-loop GNC data contracts (E13-01, task #418).

An open-loop E12 campaign planned a trajectory and asked "would it work?"
against injected dispersions. A closed-loop GNC simulation has to answer the
sharper question -- "what did the spacecraft *know*, and what did it *do*
about it?" -- and that only means something if the records separate five
things that an open-loop log blurs into one state history:

* **Truth** (``TruthSample``) -- the actual simulated absolute state of one
  spacecraft, produced by the dynamics integrator. Guidance must never read
  it; :mod:`orbitopt.gnc.feed` is the interface-layer wall, and
  ``tests/test_e13_01_truth_isolation.py`` is the proof.
* **Observation** -- one sensor measurement: the measured relative state at
  ``t_sample_s``, delivered to the filter at ``t_available_s`` (= sample +
  declared latency), with its measurement covariance. A dropped frame is
  the *absence* of a record, never a shifted timestamp.
* **Estimate** -- a navigation-filter update: relative-state estimate, full
  6x6 covariance, and the ids of the observations it used. An estimate with
  no provenance is indistinguishable from a truth leak.
* **Command** -- a guidance/control decision at a control tick: the asked-for
  delta-v, the execution instant, and the estimate the law read.
* **ControlEvent** -- what the actuator actually delivered (E12-10 measured
  execution error as the dominant failure source: the commanded-minus-realized
  gap is data, not folklore).

The payload vocabulary is E12-01's, not a new dialect: ``Observation`` and
``Estimate`` carry an embedded ``relative-state-1.0`` document parsed by
:func:`orbitopt.mission.multispacecraft.relative_state_from_dict`, epochs are
MJD2000 labels on a declared timescale, loop timing is elapsed seconds from
the clock's t0 (the E12-03 convention). All five kinds plus the clock
(:mod:`orbitopt.gnc.clock`) and the replay log
(:mod:`orbitopt.gnc.recording`) share ONE versioned schema bundle,
``orbitopt/schemas/gnc-contracts-1.0.json``: the kinds reference each other
(an estimate names its observations, a command names its estimate), so they
version together or the references dangle. Documents are produced by
``to_document()`` and parsed by the module-level ``*_from_document``
functions -- construction validates, exactly like
:mod:`orbitopt.mission.multispacecraft`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping

import jsonschema

from orbitopt.mission.contract import Epoch, MISSION_FRAMES
from orbitopt.mission.multispacecraft import RelativeState, relative_state_from_dict

#: Schema file implementing these contracts (one bundle, one version).
CONTRACT_SCHEMA_FILE = "gnc-contracts-1.0.json"
#: ``schema_version`` every document in the bundle carries.
CONTRACT_SCHEMA_VERSION = "1.0"

#: The five record kinds, in causal order (truth feeds sensors, sensors feed
#: estimates, estimates feed commands, commands feed control events).
RECORD_KINDS = ("truth", "observation", "estimate", "command", "control_event")

#: Timescale of the simulation clock: uniform scales only. A UTC label inside
#: a control loop silently jumps by a leap second mid-run; UTC belongs at the
#: handoff boundary, not between sensor tick and actuator fire.
CLOCK_TIMESCALES = ("TT", "TDB")


class GNCContractError(ValueError):
    """A GNC record/clock/log document is malformed, of the wrong kind or
    version, or violates a cross-field rule the JSON Schema cannot express."""


# ---------------------------------------------------------------------------
# schema loading and value coercion (same discipline as mission.multispacecraft)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _bundle() -> dict:
    traversable = resources.files("orbitopt").joinpath("schemas", CONTRACT_SCHEMA_FILE)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover -- packaging bug
        raise GNCContractError(f"No packaged schema orbitopt/schemas/{CONTRACT_SCHEMA_FILE}") from exc
    return json.loads(text)


def _def_schema(name: str) -> dict:
    """Draft-2020-12 schema for one bundle member: the bundle's $defs with
    ``$defs/<name>`` as the root reference."""
    bundle = _bundle()
    return {"$schema": bundle["$schema"], "$id": bundle["$id"],
            "$defs": bundle["$defs"], "$ref": f"#/$defs/{name}"}


def _version_gate(doc: Any, def_name: str, label: str) -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"{label} document must be a mapping, got {type(doc).__name__}")
    version = doc.get("schema_version")
    if version != CONTRACT_SCHEMA_VERSION:
        raise GNCContractError(
            f"no {label} schema for version {version!r} (this orbitopt implements "
            f"{CONTRACT_SCHEMA_VERSION!r}; see orbitopt/schemas/{CONTRACT_SCHEMA_FILE})")
    if doc.get("record_kind") != def_name:
        raise GNCContractError(
            f"{label} document has record_kind {doc.get('record_kind')!r}, expected {def_name!r}")
    try:
        jsonschema.validate(instance=dict(doc), schema=_def_schema(def_name))
    except jsonschema.ValidationError as exc:
        raise GNCContractError(f"{label} document violates {CONTRACT_SCHEMA_FILE}: "
                               f"{exc.message}") from exc
    return doc


def _finite(value: Any, where: str) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise GNCContractError(f"{where}: {value!r} is not a number") from exc
    if not math.isfinite(t):
        raise GNCContractError(f"{where}: {value!r} is not finite")
    return t


def _nonneg(value: Any, where: str) -> float:
    t = _finite(value, where)
    if t < 0.0:
        raise GNCContractError(f"{where}: must be >= 0, got {t}")
    return t


def _pos(value: Any, where: str) -> float:
    t = _finite(value, where)
    if t <= 0.0:
        raise GNCContractError(f"{where}: must be > 0, got {t}")
    return t


def _vec(value: Any, n: int, where: str) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise GNCContractError(f"{where}: expected {n} numbers, got {value!r}")
    parts = tuple(value)
    if len(parts) != n:
        raise GNCContractError(f"{where}: expected {n} components, got {len(parts)}")
    return tuple(_finite(c, f"{where}[{i}]") for i, c in enumerate(parts))  # type: ignore[return-value]


def _nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise GNCContractError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _mapping_of_str(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise GNCContractError(f"{where}: expected a mapping, got {type(value).__name__}")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise GNCContractError(f"{where}: keys must be strings, got {key!r}")
        out[key] = item
    return out


def _epoch_doc(epoch: Epoch) -> dict:
    return {"mjd2000": epoch.mjd2000, "timescale": epoch.timescale}


def _uniform_epoch(doc: Any, where: str) -> Epoch:
    """An Epoch on a uniform (simulation) timescale."""
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"{where}: expected a mapping, got {doc!r}")
    epoch = Epoch(mjd2000=_finite(doc["mjd2000"], f"{where}.mjd2000"),
                  timescale=doc["timescale"])
    if epoch.timescale not in CLOCK_TIMESCALES:
        raise GNCContractError(
            f"{where}: timescale {epoch.timescale!r} -- a simulation clock runs on a "
            f"uniform scale {CLOCK_TIMESCALES}, never UTC (leap seconds)")
    return epoch


def _symmetric_cov(value: Any, dim: int, where: str) -> tuple:
    """Row-major flattened square covariance: finite, symmetric, non-negative
    diagonals (variances). Correlation structure is the producer's business."""
    flat = _vec(value, dim * dim, where)
    scale = max(1.0, max(abs(x) for x in flat))
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(flat[i * dim + j] - flat[j * dim + i]) > 1.0e-9 * scale:
                raise GNCContractError(
                    f"{where}: not symmetric at ({i},{j}): {flat[i * dim + j]!r} != "
                    f"{flat[j * dim + i]!r} -- a covariance must equal its transpose")
    for i in range(dim):
        if flat[i * dim + i] < 0.0:
            raise GNCContractError(f"{where}: variance [{i}] is negative "
                                   f"({flat[i * dim + i]}) -- variances cannot be")
    return flat


def _relative_state_doc(rs: RelativeState) -> dict:
    return rs.to_dict()


def _relative_state_from(doc: Any, where: str) -> RelativeState:
    try:
        rs = relative_state_from_dict(doc)
    except (jsonschema.ValidationError, ValueError) as exc:
        raise GNCContractError(f"{where}: not a valid relative-state-1.0 document: {exc}") from exc
    if rs.epoch.timescale not in CLOCK_TIMESCALES:
        raise GNCContractError(
            f"{where}: epoch timescale {rs.epoch.timescale!r} -- loop records are timestamped "
            f"on the simulation clock's uniform scale {CLOCK_TIMESCALES}")
    return rs


def _record_doc(kind: str, record_id: str, fields: dict, metadata: Mapping[str, Any]) -> dict:
    return {"schema_version": CONTRACT_SCHEMA_VERSION, "record_kind": kind,
            "record_id": record_id, **fields, "metadata": dict(metadata)}


def _id_tuple(value: Any, where: str) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise GNCContractError(f"{where}: expected a list of record ids, got {value!r}")
    return tuple(_nonempty_str(v, f"{where}[{i}]") for i, v in enumerate(value))


# ---------------------------------------------------------------------------
# the five record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthSample:
    """The ACTUAL simulated absolute state of one spacecraft at a dynamics
    epoch (what the integrator says, not what anyone measured). Visible to
    the simulator, the logger and offline evaluators; guidance reads it
    nowhere -- see :mod:`orbitopt.gnc.feed`."""

    record_id: str
    spacecraft_id: str
    t_s: float
    epoch: Epoch
    frame: str
    r_m: tuple
    v_m_s: tuple
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "record_id", _nonempty_str(self.record_id, "TruthSample.record_id"))
        object.__setattr__(self, "spacecraft_id",
                           _nonempty_str(self.spacecraft_id, "TruthSample.spacecraft_id"))
        object.__setattr__(self, "t_s", _nonneg(self.t_s, "TruthSample.t_s"))
        if not isinstance(self.epoch, Epoch):
            raise GNCContractError(f"TruthSample.epoch must be an Epoch, got {self.epoch!r}")
        if self.epoch.timescale not in CLOCK_TIMESCALES:
            raise GNCContractError(f"TruthSample.epoch timescale {self.epoch.timescale!r} "
                                   f"not in {CLOCK_TIMESCALES}")
        if self.frame not in MISSION_FRAMES:
            raise GNCContractError(f"TruthSample.frame {self.frame!r} not in {MISSION_FRAMES}")
        object.__setattr__(self, "r_m", _vec(self.r_m, 3, "TruthSample.r_m"))
        object.__setattr__(self, "v_m_s", _vec(self.v_m_s, 3, "TruthSample.v_m_s"))
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "TruthSample.metadata"))

    def to_document(self) -> dict:
        return _record_doc("truth", self.record_id, {
            "t_s": self.t_s, "epoch": _epoch_doc(self.epoch),
            "spacecraft_id": self.spacecraft_id, "frame": self.frame,
            "position_m": list(self.r_m), "velocity_m_s": list(self.v_m_s)}, self.metadata)


@dataclass(frozen=True)
class Observation:
    """One sensor measurement of the chaser-minus-target relative state:
    sampled at ``t_sample_s``, usable by the filter at ``t_available_s``
    (== sample + ``latency_s``, enforced on construction), with the
    measurement position covariance (3x3 row-major)."""

    record_id: str
    sensor_id: str
    t_sample_s: float
    t_available_s: float
    latency_s: float
    relative_state: RelativeState
    covariance_position_m2: tuple
    covariance_velocity_m2_s2: tuple = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "record_id", _nonempty_str(self.record_id, "Observation.record_id"))
        object.__setattr__(self, "sensor_id", _nonempty_str(self.sensor_id, "Observation.sensor_id"))
        object.__setattr__(self, "t_sample_s", _nonneg(self.t_sample_s, "Observation.t_sample_s"))
        object.__setattr__(self, "t_available_s", _nonneg(self.t_available_s, "Observation.t_available_s"))
        object.__setattr__(self, "latency_s", _nonneg(self.latency_s, "Observation.latency_s"))
        if abs(self.t_available_s - (self.t_sample_s + self.latency_s)) > 1.0e-6:
            raise GNCContractError(
                f"Observation {self.record_id}: t_available_s {self.t_available_s} != t_sample_s "
                f"{self.t_sample_s} + latency_s {self.latency_s} -- a measurement cannot arrive "
                "before its shutter opened or out of order with its declared latency")
        object.__setattr__(self, "relative_state",
                           _relative_state_from(self.relative_state.to_dict()
                                                if isinstance(self.relative_state, RelativeState)
                                                else self.relative_state,
                                                "Observation.relative_state"))
        object.__setattr__(self, "covariance_position_m2",
                           _symmetric_cov(self.covariance_position_m2, 3,
                                          "Observation.covariance_position_m2"))
        vel = tuple(self.covariance_velocity_m2_s2)
        if vel:
            object.__setattr__(self, "covariance_velocity_m2_s2",
                               _symmetric_cov(vel, 3, "Observation.covariance_velocity_m2_s2"))
        else:
            object.__setattr__(self, "covariance_velocity_m2_s2", ())
        object.__setattr__(self, "metadata",
                           _mapping_of_str(self.metadata, "Observation.metadata"))

    def to_document(self) -> dict:
        fields = {"sensor_id": self.sensor_id, "t_sample_s": self.t_sample_s,
                  "t_available_s": self.t_available_s, "latency_s": self.latency_s,
                  "relative_state": _relative_state_doc(self.relative_state),
                  "covariance_position_m2": list(self.covariance_position_m2)}
        if self.covariance_velocity_m2_s2:
            fields["covariance_velocity_m2_s2"] = list(self.covariance_velocity_m2_s2)
        return _record_doc("observation", self.record_id, fields, self.metadata)


@dataclass(frozen=True)
class Estimate:
    """One navigation-filter update: relative-state estimate + 6x6 covariance
    (rho then rho_dot, row-major), consumed by guidance through
    :class:`orbitopt.gnc.feed.GuidanceInput` and by nothing else.
    ``source_observation_ids`` is mandatory provenance -- which measurements
    produced this update is a checkable fact, not an assumption."""

    record_id: str
    estimator_id: str
    t_update_s: float
    relative_state: RelativeState
    covariance_rel_m2: tuple
    source_observation_ids: tuple
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "record_id", _nonempty_str(self.record_id, "Estimate.record_id"))
        object.__setattr__(self, "estimator_id",
                           _nonempty_str(self.estimator_id, "Estimate.estimator_id"))
        object.__setattr__(self, "t_update_s", _nonneg(self.t_update_s, "Estimate.t_update_s"))
        object.__setattr__(self, "relative_state",
                           _relative_state_from(self.relative_state.to_dict()
                                                if isinstance(self.relative_state, RelativeState)
                                                else self.relative_state,
                                                "Estimate.relative_state"))
        object.__setattr__(self, "covariance_rel_m2",
                           _symmetric_cov(self.covariance_rel_m2, 6, "Estimate.covariance_rel_m2"))
        object.__setattr__(self, "source_observation_ids",
                           _id_tuple(self.source_observation_ids, "Estimate.source_observation_ids"))
        if not self.source_observation_ids:
            raise GNCContractError(
                f"Estimate {self.record_id}: source_observation_ids is empty -- an update with "
                "no measurements behind it is a truth leak in disguise; state a time-update "
                "propagation via the previous estimate in metadata instead")
        object.__setattr__(self, "metadata",
                           _mapping_of_str(self.metadata, "Estimate.metadata"))

    def to_document(self) -> dict:
        return _record_doc("estimate", self.record_id, {
            "estimator_id": self.estimator_id, "t_update_s": self.t_update_s,
            "relative_state": _relative_state_doc(self.relative_state),
            "covariance_rel_m2": list(self.covariance_rel_m2),
            "source_observation_ids": list(self.source_observation_ids)}, self.metadata)


@dataclass(frozen=True)
class Command:
    """A guidance/control decision taken at a control tick from one named
    estimate: commanded inertial delta-v and the instant it asks the actuator
    to fire. What actually happened is the linked ControlEvent."""

    record_id: str
    actuator_id: str
    t_tick_s: float
    t_execute_s: float
    delta_v_eci_m_s: tuple
    source_estimate_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "record_id", _nonempty_str(self.record_id, "Command.record_id"))
        object.__setattr__(self, "actuator_id", _nonempty_str(self.actuator_id, "Command.actuator_id"))
        object.__setattr__(self, "t_tick_s", _nonneg(self.t_tick_s, "Command.t_tick_s"))
        object.__setattr__(self, "t_execute_s", _nonneg(self.t_execute_s, "Command.t_execute_s"))
        if self.t_execute_s + 1.0e-9 < self.t_tick_s:
            raise GNCContractError(
                f"Command {self.record_id}: t_execute_s {self.t_execute_s} < t_tick_s "
                f"{self.t_tick_s} -- a command cannot fire before the tick that made it")
        object.__setattr__(self, "delta_v_eci_m_s",
                           _vec(self.delta_v_eci_m_s, 3, "Command.delta_v_eci_m_s"))
        object.__setattr__(self, "source_estimate_id",
                           _nonempty_str(self.source_estimate_id, "Command.source_estimate_id"))
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "Command.metadata"))

    def to_document(self) -> dict:
        return _record_doc("command", self.record_id, {
            "actuator_id": self.actuator_id, "t_tick_s": self.t_tick_s,
            "t_execute_s": self.t_execute_s,
            "delta_v_eci_m_s": list(self.delta_v_eci_m_s),
            "source_estimate_id": self.source_estimate_id}, self.metadata)


@dataclass(frozen=True)
class ControlEvent:
    """A maneuver as ACTUALLY executed: the realized delta-v (zero vector
    when ``executed`` is false) against what the command asked for. The
    commanded-minus-realized gap is execution error -- the quantity E12-10
    measured as the dominant failure source, recorded per burn."""

    record_id: str
    actuator_id: str
    t_exec_s: float
    executed: bool
    commanded_delta_v_m_s: tuple
    realized_delta_v_m_s: tuple
    source_command_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "record_id", _nonempty_str(self.record_id, "ControlEvent.record_id"))
        object.__setattr__(self, "actuator_id",
                           _nonempty_str(self.actuator_id, "ControlEvent.actuator_id"))
        object.__setattr__(self, "t_exec_s", _nonneg(self.t_exec_s, "ControlEvent.t_exec_s"))
        if not isinstance(self.executed, bool):
            raise GNCContractError(f"ControlEvent {self.record_id}: executed must be a bool")
        if not self.executed and any(x != 0.0 for x in self.realized_delta_v_m_s):
            raise GNCContractError(f"ControlEvent {self.record_id}: executed=False with non-zero "
                                   "realized delta-v -- a burn that did not happen delivered "
                                   "nothing; record the misfire as executed with the shortfall")
        object.__setattr__(self, "commanded_delta_v_m_s",
                           _vec(self.commanded_delta_v_m_s, 3, "ControlEvent.commanded_delta_v_m_s"))
        object.__setattr__(self, "realized_delta_v_m_s",
                           _vec(self.realized_delta_v_m_s, 3, "ControlEvent.realized_delta_v_m_s"))
        object.__setattr__(self, "source_command_id",
                           _nonempty_str(self.source_command_id, "ControlEvent.source_command_id"))
        object.__setattr__(self, "metadata",
                           _mapping_of_str(self.metadata, "ControlEvent.metadata"))

    def to_document(self) -> dict:
        return _record_doc("control_event", self.record_id, {
            "actuator_id": self.actuator_id, "t_exec_s": self.t_exec_s,
            "executed": self.executed,
            "commanded_delta_v_m_s": list(self.commanded_delta_v_m_s),
            "realized_delta_v_m_s": list(self.realized_delta_v_m_s),
            "source_command_id": self.source_command_id}, self.metadata)


RECORD_TYPES = {"truth": TruthSample, "observation": Observation, "estimate": Estimate,
                "command": Command, "control_event": ControlEvent}


# ---------------------------------------------------------------------------
# document parsing (dispatch + per-kind)
# ---------------------------------------------------------------------------


def truth_from_document(doc: Mapping[str, Any]) -> TruthSample:
    """Parse a gnc-contracts-1.0 ``truth`` record document."""
    d = _version_gate(doc, "truth", "truth")
    return TruthSample(record_id=d["record_id"], spacecraft_id=d["spacecraft_id"],
                       t_s=d["t_s"], epoch=_uniform_epoch(d["epoch"], "epoch"), frame=d["frame"],
                       r_m=d["position_m"], v_m_s=d["velocity_m_s"], metadata=d.get("metadata", {}))


def observation_from_document(doc: Mapping[str, Any]) -> Observation:
    """Parse a gnc-contracts-1.0 ``observation`` record document."""
    d = _version_gate(doc, "observation", "observation")
    return Observation(record_id=d["record_id"], sensor_id=d["sensor_id"],
                       t_sample_s=d["t_sample_s"], t_available_s=d["t_available_s"],
                       latency_s=d["latency_s"],
                       relative_state=_relative_state_from(d["relative_state"], "relative_state"),
                       covariance_position_m2=d["covariance_position_m2"],
                       covariance_velocity_m2_s2=d.get("covariance_velocity_m2_s2", ()),
                       metadata=d.get("metadata", {}))


def estimate_from_document(doc: Mapping[str, Any]) -> Estimate:
    """Parse a gnc-contracts-1.0 ``estimate`` record document."""
    d = _version_gate(doc, "estimate", "estimate")
    return Estimate(record_id=d["record_id"], estimator_id=d["estimator_id"],
                    t_update_s=d["t_update_s"],
                    relative_state=_relative_state_from(d["relative_state"], "relative_state"),
                    covariance_rel_m2=d["covariance_rel_m2"],
                    source_observation_ids=d["source_observation_ids"],
                    metadata=d.get("metadata", {}))


def command_from_document(doc: Mapping[str, Any]) -> Command:
    """Parse a gnc-contracts-1.0 ``command`` record document."""
    d = _version_gate(doc, "command", "command")
    return Command(record_id=d["record_id"], actuator_id=d["actuator_id"],
                   t_tick_s=d["t_tick_s"], t_execute_s=d["t_execute_s"],
                   delta_v_eci_m_s=d["delta_v_eci_m_s"],
                   source_estimate_id=d["source_estimate_id"], metadata=d.get("metadata", {}))


def control_event_from_document(doc: Mapping[str, Any]) -> ControlEvent:
    """Parse a gnc-contracts-1.0 ``control_event`` record document."""
    d = _version_gate(doc, "control_event", "control_event")
    return ControlEvent(record_id=d["record_id"], actuator_id=d["actuator_id"],
                        t_exec_s=d["t_exec_s"], executed=d["executed"],
                        commanded_delta_v_m_s=d["commanded_delta_v_m_s"],
                        realized_delta_v_m_s=d["realized_delta_v_m_s"],
                        source_command_id=d["source_command_id"], metadata=d.get("metadata", {}))


_DOCUMENT_PARSERS = {
    "truth": truth_from_document, "observation": observation_from_document,
    "estimate": estimate_from_document, "command": command_from_document,
    "control_event": control_event_from_document}


def record_from_document(doc: Mapping[str, Any]):
    """Parse any of the five record kinds by dispatching on ``record_kind``."""
    if not isinstance(doc, Mapping) or "record_kind" not in doc:
        raise GNCContractError("record document has no 'record_kind'")
    kind = doc["record_kind"]
    if kind not in _DOCUMENT_PARSERS:
        raise GNCContractError(f"unknown record kind {kind!r}; expected one of {RECORD_KINDS}")
    return _DOCUMENT_PARSERS[kind](doc)


def is_valid_record_document(doc: Mapping[str, Any]) -> bool:
    """Non-raising form of :func:`record_from_document`."""
    try:
        record_from_document(doc)
    except (GNCContractError, jsonschema.ValidationError):
        return False
    return True
