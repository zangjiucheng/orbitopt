"""The replayable GNC run log: measure/estimate/control/config as one sealed record (E13-01).

A closed-loop run is only evidence when the *complete* exchange can be
re-read and found self-consistent: every truth sample, observation,
estimate, command and control event, plus the clock declaration and the run
configuration that produced them. :class:`GncRunLog` is that document, and
:func:`read_run_log` is where replay is checked -- not "the file parses" but
"the log agrees with itself":

* ids are unique, and every reference (estimate→observations,
  command→estimate, control event→command) points at an earlier, causally
  *available* record;
* observations sit exactly on their sensor's declared ticks (same phase and
  period, declared latency), never on dropped ticks, one per tick;
* timestamps match the clock's declaration (the schedule is recomputed from
  the clock section, so a log edited to smooth a timeline fails);
* an embedded E12-01 mission roster shares the clock's epoch (the
  one-epoch rule, checked, not hoped for), and an E12-08 ``plan_digest`` may
  bind the run to a sealed plan;
* the canonical ``run_digest`` (same 8-significant-digit quantization policy
  as ``orbitopt.rendezvous.canonical``) recomputes from the document itself,
  so a byte edited anywhere changes what replay refuses to read.

Writing is append-only (``log.append_*``); :func:`write_run_log` emits the
digest. This is the record layer E13-02..E13-05 (filter design, safety
supervision, paired statistics against the E12-10 open-loop reference) all
read from.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema

from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import MultiSpacecraftMission, mission_from_dict
from orbitopt.rendezvous.canonical import DIGEST_SIGNIFICANT_DIGITS

from .clock import ClockSpec, clock_from_document
from .contracts import (CONTRACT_SCHEMA_VERSION, GNCContractError, RECORD_KINDS,
                        RECORD_TYPES, _bundle, _def_schema, _mapping_of_str,
                        _nonempty_str, command_from_document,
                        control_event_from_document, estimate_from_document,
                        observation_from_document, truth_from_document)

RUN_LOG_KIND = "orbitopt.gnc.run-log"
_TIME_TOL_S = 1.0e-6
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_DOC_PARSERS = {"truth": truth_from_document, "observation": observation_from_document,
                "estimate": estimate_from_document, "command": command_from_document,
                "control_event": control_event_from_document}


# ---------------------------------------------------------------------------
# canonical digest (same quantization policy as the E12-08 plan digests)
# ---------------------------------------------------------------------------


def _quantize(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.{DIGEST_SIGNIFICANT_DIGITS - 1}e}")
    if isinstance(value, (list, tuple)):
        return [_quantize(x) for x in value]
    if isinstance(value, dict):
        return {k: _quantize(v) for k, v in value.items()}
    return value


def run_digest_of(document: Mapping[str, Any]) -> str:
    """sha256 of the run-log document WITHOUT its ``run_digest`` field,
    canonical JSON, floats quantized like every other orbitopt digest."""
    body = {k: v for k, v in document.items() if k != "run_digest"}
    payload = json.dumps(_quantize(body), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GncRunLog:
    """One closed-loop run: clock, config, optional mission roster / sealed
    plan binding, and the five record streams. Construct empty (or via
    ``GncRunLog.new``), ``append_*`` records as the run proceeds,
    :func:`write_run_log` seals it. Every constructor argument is parsed and
    cross-validated (including :meth:`validate`) -- a hand-assembled log goes
    through the same gates as a written one."""

    run_id: str
    clock: ClockSpec
    config: Mapping[str, Any] = field(default_factory=dict)
    mission: Mapping[str, Any] | None = None
    plan_digest: str | None = None
    records: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "run_id", _nonempty_str(self.run_id, "GncRunLog.run_id"))
        if not isinstance(self.clock, ClockSpec):
            raise GNCContractError(f"GncRunLog.clock must be a ClockSpec, got {self.clock!r}")
        object.__setattr__(self, "config", _mapping_of_str(self.config, "GncRunLog.config"))
        mission_doc = None
        if self.mission is not None:
            try:
                parsed = mission_from_dict(self.mission)
            except (jsonschema.ValidationError, ValueError) as exc:
                raise GNCContractError(f"GncRunLog.mission: not a valid "
                                       f"multi-spacecraft-mission-1.0 document: {exc}") from exc
            mission_doc = dict(self.mission)
            skew = abs(parsed.epoch.delta_seconds(self.clock.t0))
            if skew > _TIME_TOL_S:
                raise GNCContractError(
                    f"GncRunLog: mission epoch and clock t0 differ by {skew:g} s -- E12-01's "
                    "single-shared-epoch rule: a run whose roster and clock start at "
                    "different instants has no shared timeline to replay")
            object.__setattr__(self, "_mission_parsed", parsed)
        else:
            object.__setattr__(self, "_mission_parsed", None)
        if mission_doc is not None:
            object.__setattr__(self, "mission", mission_doc)
        if self.plan_digest is not None and not _HEX64.fullmatch(str(self.plan_digest)):
            raise GNCContractError("GncRunLog.plan_digest must be 64 hex digits (an E12-08 "
                                   "canonical plan digest) or None")
        recs: dict = {}
        for kind in RECORD_KINDS:
            items = tuple(self.records.get(kind, ())) if isinstance(self.records, Mapping) else ()
            parsed_items = []
            for item in items:
                expected = RECORD_TYPES[kind]
                if not isinstance(item, expected):
                    raise GNCContractError(
                        f"GncRunLog.records[{kind!r}]: expected {expected.__name__}, got {item!r}")
                parsed_items.append(item)
            recs[kind] = tuple(parsed_items)
        object.__setattr__(self, "records", recs)
        self.validate()

    @classmethod
    def new(cls, run_id: str, clock: ClockSpec, config: Mapping[str, Any] | None = None,
            mission: Mapping[str, Any] | None = None, plan_digest: str | None = None) -> "GncRunLog":
        """An empty log to append into (config defaults to ``{}``)."""
        return cls(run_id=run_id, clock=clock, config=dict(config or {}), mission=mission,
                   plan_digest=plan_digest)

    def _push(self, kind: str, record) -> None:
        object.__setattr__(self, "records", {**self.records, kind: (*self.records[kind], record)})
        self.validate()

    def append_truth(self, sample) -> None:
        self._push("truth", sample)

    def append_observation(self, obs) -> None:
        self._push("observation", obs)

    def append_estimate(self, est) -> None:
        self._push("estimate", est)

    def append_command(self, cmd) -> None:
        self._push("command", cmd)

    def append_control_event(self, ev) -> None:
        self._push("control_event", ev)

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Every cross-record gate replay would check; cheap enough to run
        on each append, and :func:`read_run_log` runs it again on what came
        off disk (via :func:`run_log_from_document`)."""
        ids: dict = {}
        for kind in RECORD_KINDS:
            for rec in self.records[kind]:
                if rec.record_id in ids:
                    raise GNCContractError(
                        f"run log: record_id {rec.record_id!r} used twice ({ids[rec.record_id]} "
                        f"and {kind}) -- ids are the reference keys, duplicates make "
                        "provenance ambiguous")
                ids[rec.record_id] = kind

        truth_ids = set()
        for t in self.records["truth"]:
            truth_ids.add(t.record_id)
            if self._mission_parsed is not None:
                roster = {sc.spacecraft_id for sc in self._mission_parsed.spacecraft}
                if t.spacecraft_id not in roster:
                    raise GNCContractError(
                        f"truth record {t.record_id}: spacecraft {t.spacecraft_id!r} is not on "
                        f"the mission roster {sorted(roster)}")
                if self._mission_parsed.frame != t.frame:
                    raise GNCContractError(
                        f"truth record {t.record_id}: frame {t.frame} != mission frame "
                        f"{self._mission_parsed.frame}")

        observations: dict = {}
        seen_ticks: set = set()
        for o in self.records["observation"]:
            observations[o.record_id] = o
            try:
                spec = self.clock.sensor(o.sensor_id)
            except GNCContractError as exc:
                raise GNCContractError(f"observation {o.record_id}: {exc}") from exc
            if abs(o.latency_s - spec.latency_s) > _TIME_TOL_S:
                raise GNCContractError(
                    f"observation {o.record_id}: latency {o.latency_s} != the clock's declared "
                    f"{spec.latency_s} for sensor {o.sensor_id!r}")
            schedule = {tick.t_sample_s: tick for tick in
                        self.clock.sensor_schedule(o.sensor_id, o.t_sample_s + spec.period_s)}
            hit = [t for t in schedule if abs(t - o.t_sample_s) <= _TIME_TOL_S]
            if not hit:
                sched_head = ", ".join(f"{s:.3f}" for s in sorted(schedule)[:6])
                raise GNCContractError(
                    f"observation {o.record_id}: t_sample_s {o.t_sample_s} is not a declared "
                    f"tick of sensor {o.sensor_id!r} (schedule: {sched_head}...) -- the log "
                    "claims a measurement the clock never scheduled")
            tick = schedule[hit[0]]
            if tick.dropped:
                raise GNCContractError(
                    f"observation {o.record_id}: tick {tick.tick} of sensor {o.sensor_id!r} is "
                    "declared dropped -- a dropped frame is an absence, not a delayed record")
            if (o.sensor_id, tick.tick) in seen_ticks:
                raise GNCContractError(
                    f"observation {o.record_id}: duplicate delivery for sensor {o.sensor_id!r} "
                    f"tick {tick.tick}")
            seen_ticks.add((o.sensor_id, tick.tick))

        for e in self.records["estimate"]:
            for src in e.source_observation_ids:
                obs = observations.get(src)
                if obs is None:
                    raise GNCContractError(
                        f"estimate {e.record_id}: source observation {src!r} is not in the log "
                        "-- an update must cite measurements that exist")
                if obs.t_available_s > e.t_update_s + _TIME_TOL_S:
                    raise GNCContractError(
                        f"estimate {e.record_id}: source {src} became usable at "
                        f"{obs.t_available_s} s, after the update at {e.t_update_s} s -- "
                        "the filter used data before it arrived")

        commands: dict = {}
        estimates_by_id = {e.record_id: e for e in self.records["estimate"]}
        for c in self.records["command"]:
            commands[c.record_id] = c
            est = estimates_by_id.get(c.source_estimate_id)
            if est is None:
                raise GNCContractError(
                    f"command {c.record_id}: source estimate {c.source_estimate_id!r} is not "
                    "in the log -- commands are attributed to navigation estimates only")
            readable = est.t_update_s + self.clock.estimator.latency_s
            if readable > c.t_tick_s + _TIME_TOL_S:
                raise GNCContractError(
                    f"command {c.record_id}: its estimate became readable at {readable} s, "
                    f"after the control tick at {c.t_tick_s} s")

        for ev in self.records["control_event"]:
            cmd = commands.get(ev.source_command_id)
            if cmd is None:
                raise GNCContractError(
                    f"control event {ev.record_id}: source command {ev.source_command_id!r} "
                    "is not in the log")
            if ev.t_exec_s < cmd.t_execute_s - _TIME_TOL_S:
                raise GNCContractError(
                    f"control event {ev.record_id}: executed at {ev.t_exec_s} s, before the "
                    f"command's requested {cmd.t_execute_s} s")
            if not ev.executed and any(abs(x) > 0.0 for x in ev.realized_delta_v_m_s):
                raise GNCContractError(
                    f"control event {ev.record_id}: executed=False with non-zero realized "
                    "delta-v -- a burn that did not happen delivered nothing")

    def to_document(self, run_digest: str) -> dict:
        """The run-log document, digest included (what
        :func:`write_run_log` seals)."""
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION, "kind": RUN_LOG_KIND,
            "run_id": self.run_id, "mission": dict(self.mission) if self.mission else None,
            "plan_digest": self.plan_digest, "config": dict(self.config),
            "clock": self.clock.to_document(),
            "records": {kind: [r.to_document() for r in self.records[kind]]
                        for kind in RECORD_KINDS},
            "run_digest": run_digest,
        }

    def digest(self) -> str:
        """The digest of this log's document (computes it; writing seals the
        same value)."""
        return run_digest_of(self.to_document(""))


def run_log_from_document(doc: Mapping[str, Any]) -> GncRunLog:
    """Parse + replay-verify a run-log document: schema gate, digest
    recompute, then every cross-record gate. A tampered record, an edited
    clock or a re-ordered timeline all fail here -- replay means the log
    agrees with itself, not that it loads."""
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"run-log document must be a mapping, got {type(doc).__name__}")
    if doc.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise GNCContractError(
            f"no run-log schema for version {doc.get('schema_version')!r} (implements "
            f"{CONTRACT_SCHEMA_VERSION!r})")
    if doc.get("kind") != RUN_LOG_KIND:
        raise GNCContractError(f"run-log kind {doc.get('kind')!r} != {RUN_LOG_KIND!r}")
    try:
        jsonschema.validate(instance=dict(doc), schema=_def_schema("run_log"))
    except jsonschema.ValidationError as exc:
        raise GNCContractError(f"run-log document violates gnc-contracts-1.0: {exc.message}") from exc
    claimed = str(doc["run_digest"])
    recomputed = run_digest_of(doc)
    if claimed != recomputed:
        raise GNCContractError(
            f"run-log digest mismatch: document claims {claimed[:16]}..., the contents hash to "
            f"{recomputed[:16]}... -- something in this log was edited after it was written")
    records = {kind: tuple(_DOC_PARSERS[kind](r) for r in doc["records"][kind])
               for kind in RECORD_KINDS}
    return GncRunLog(run_id=doc["run_id"], clock=clock_from_document(doc["clock"]),
                     config=doc["config"], mission=doc.get("mission"),
                     plan_digest=doc.get("plan_digest"), records=records)


def write_run_log(log: GncRunLog, path: str | Path) -> Path:
    """Seal the log to disk: canonical document + run digest, deterministic
    (no timestamps -- the digest is the identity, like the E12-08 sealed
    plan)."""
    digest = log.digest()
    doc = log.to_document(digest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return out


def read_run_log(path: str | Path) -> GncRunLog:
    """Replay-read: :func:`run_log_from_document` over the file."""
    return run_log_from_document(json.loads(Path(path).read_text(encoding="utf-8")))


def replay_stream(log: GncRunLog, kind: str) -> Iterable:
    """One record stream of a replayed log, in recorded order (the loop's
    causal order)."""
    if kind not in RECORD_KINDS:
        raise GNCContractError(f"unknown record kind {kind!r}; expected one of {RECORD_KINDS}")
    return iter(log.records[kind])
