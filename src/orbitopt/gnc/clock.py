"""The simulation clock contract: four declared time domains, one epoch (E13-01).

A closed-loop run mixes four cadences that an open-loop replay never has to
distinguish: the truth **dynamics** step (how the plant is integrated), each
**sensor's** sampling (its own period, phase and latency), the navigation
**estimator's** update cadence, and the **controller's** tick. "One shared
epoch" is the E12-01 rule; *one shared period* would be a modelling straitjacket --
the sensor runs at 1 Hz, the filter at 2 Hz, the computer at 5 Hz and the
plant at 10 Hz, and every mismatch, phase offset and delay is part of the
scenario being tested, not noise to hide. So the clock *declares* each
domain and hands out exact times:

* every timestamp is a pure function of its tick index --
  ``phase + n*period`` -- never an accumulator, so a dropped frame, a paused
  sensor or a longer latency cannot shift any later sample;
* a measurement's availability time is ``sample + latency``, so causality
  ("the filter may use observation X at update T") is arithmetic, not
  judgement (:mod:`orbitopt.gnc.recording` checks it against the log);
* the clock's timescale is uniform (TT or TDB): a UTC label inside a loop
  silently jumps by a leap second. ``epoch_at(t_s)`` converts elapsed
  seconds back to an E12-01 ``Epoch`` on the clock's own label.

:func:`ClockSpec.from_document`/``to_document`` round-trip through the
``clock`` definition of ``orbitopt/schemas/gnc-contracts-1.0.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from orbitopt.mission.contract import Epoch, SECONDS_PER_DAY

from .contracts import (CLOCK_TIMESCALES, CONTRACT_SCHEMA_VERSION, GNCContractError,
                        _bundle, _def_schema, _finite, _mapping_of_str, _nonempty_str,
                        _nonneg, _pos)

import jsonschema

CLOCK_KIND = "orbitopt.gnc.clock"


def _last_tick(end_s: float, phase_s: float, period_s: float) -> int:
    """Largest tick index whose instant <= end_s. Plain ``floor`` on a float
    division is off by one at exact multiples (0.5 / 0.1 is 4.999999...),
    and a horizon bug in a clock is a lost frame nobody sees -- the epsilon
    absorbs float division error only, never a real half-step."""
    import math
    return max(-1, int(math.floor((end_s - phase_s) / period_s + 1.0e-9)))


@dataclass(frozen=True)
class SensorSpec:
    """Declared timing of one sensor: period, first-sample phase after t0,
    delivery latency, and the tick indices it drops (frame drop is scenario
    data -- the same clock must replay the same drop pattern)."""

    sensor_id: str
    period_s: float
    phase_s: float
    latency_s: float
    dropped_ticks: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "sensor_id", _nonempty_str(self.sensor_id, "SensorSpec.sensor_id"))
        object.__setattr__(self, "period_s", _pos(self.period_s, f"SensorSpec({self.sensor_id}).period_s"))
        object.__setattr__(self, "phase_s", _nonneg(self.phase_s, f"SensorSpec({self.sensor_id}).phase_s"))
        if self.phase_s >= self.period_s:
            raise GNCContractError(
                f"SensorSpec({self.sensor_id}): phase_s {self.phase_s} >= period_s "
                f"{self.period_s} -- phase is an offset within one period (use a t0 shift "
                "for a whole-period delay; that is the clock's job, not the phase's)")
        object.__setattr__(self, "latency_s", _nonneg(self.latency_s, f"SensorSpec({self.sensor_id}).latency_s"))
        drops = tuple(self.dropped_ticks)
        for d in drops:
            if isinstance(d, bool) or not isinstance(d, int) or d < 0:
                raise GNCContractError(f"SensorSpec({self.sensor_id}): dropped_ticks must be "
                                       f"non-negative ints, got {d!r}")
        if len(set(drops)) != len(drops):
            raise GNCContractError(f"SensorSpec({self.sensor_id}): duplicate dropped_ticks")
        object.__setattr__(self, "dropped_ticks", tuple(sorted(drops)))

    def sample_time(self, tick: int) -> float:
        """The shutter-open instant of tick ``tick`` (0-based): a pure
        function of the index, so drops never shift later samples."""
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise GNCContractError(f"SensorSpec({self.sensor_id}): tick must be a non-negative "
                                   f"int, got {tick!r}")
        return self.phase_s + tick * self.period_s

    def to_document(self) -> dict:
        return {"sensor_id": self.sensor_id, "period_s": self.period_s, "phase_s": self.phase_s,
                "latency_s": self.latency_s, "dropped_ticks": list(self.dropped_ticks)}


def sensor_spec_from_document(doc: Mapping[str, Any]) -> SensorSpec:
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"sensor spec must be a mapping, got {doc!r}")
    return SensorSpec(sensor_id=doc["sensor_id"], period_s=doc["period_s"], phase_s=doc["phase_s"],
                      latency_s=doc["latency_s"], dropped_ticks=tuple(doc["dropped_ticks"]))


@dataclass(frozen=True)
class _Cadence:
    """period + phase (phase within the period), for the filter and the
    controller."""

    period_s: float
    phase_s: float

    def _name(self) -> str:
        return type(self).__name__

    def _validated(self) -> None:
        object.__setattr__(self, "period_s", _pos(self.period_s, f"{self._name()}.period_s"))
        object.__setattr__(self, "phase_s", _nonneg(self.phase_s, f"{self._name()}.phase_s"))
        if self.phase_s >= self.period_s:
            raise GNCContractError(f"{self._name()}: phase_s {self.phase_s} >= period_s "
                                   f"{self.period_s} -- phase is an offset within one period")

    def time(self, tick: int) -> float:
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise GNCContractError(f"{self._name()}: tick must be a non-negative int, got {tick!r}")
        return self.phase_s + tick * self.period_s


@dataclass(frozen=True)
class EstimatorSpec(_Cadence):
    """Declared update cadence of the navigation filter plus its compute
    latency (update -> estimate readable)."""

    estimator_id: str = "nav"
    latency_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "estimator_id",
                           _nonempty_str(self.estimator_id, "EstimatorSpec.estimator_id"))
        self._validated()
        object.__setattr__(self, "latency_s", _nonneg(self.latency_s, "EstimatorSpec.latency_s"))

    def to_document(self) -> dict:
        return {"estimator_id": self.estimator_id, "period_s": self.period_s,
                "phase_s": self.phase_s, "latency_s": self.latency_s}


def estimator_spec_from_document(doc: Mapping[str, Any]) -> EstimatorSpec:
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"estimator spec must be a mapping, got {doc!r}")
    return EstimatorSpec(period_s=doc["period_s"], phase_s=doc["phase_s"],
                         estimator_id=doc["estimator_id"], latency_s=doc["latency_s"])


@dataclass(frozen=True)
class ControllerSpec(_Cadence):
    """Declared control tick cadence plus command-to-fire latency (the
    actuator records reality in its ControlEvent)."""

    actuator_id: str = "thruster"
    latency_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "actuator_id",
                           _nonempty_str(self.actuator_id, "ControllerSpec.actuator_id"))
        self._validated()
        object.__setattr__(self, "latency_s", _nonneg(self.latency_s, "ControllerSpec.latency_s"))

    def to_document(self) -> dict:
        return {"actuator_id": self.actuator_id, "period_s": self.period_s,
                "phase_s": self.phase_s, "latency_s": self.latency_s}


def controller_spec_from_document(doc: Mapping[str, Any]) -> ControllerSpec:
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"controller spec must be a mapping, got {doc!r}")
    return ControllerSpec(period_s=doc["period_s"], phase_s=doc["phase_s"],
                          actuator_id=doc["actuator_id"], latency_s=doc["latency_s"])


@dataclass(frozen=True)
class SensorTick:
    """One scheduled sensor tick: its shutter-open instant and when the data
    would be usable; ``dropped`` says the frame was lost (an honest absence,
    not a missing number)."""

    sensor_id: str
    tick: int
    t_sample_s: float
    t_available_s: float
    dropped: bool


@dataclass(frozen=True)
class ClockSpec:
    """The whole declared clock: dynamics step, sensor schedules, filter and
    controller cadences, all counted from one ``t0`` Epoch on a uniform
    timescale."""

    t0: Epoch
    dt_dynamics_s: float
    sensors: tuple
    estimator: EstimatorSpec
    controller: ControllerSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.t0, Epoch):
            raise GNCContractError(f"ClockSpec.t0 must be an Epoch, got {self.t0!r}")
        if self.t0.timescale not in CLOCK_TIMESCALES:
            raise GNCContractError(
                f"ClockSpec.t0 timescale {self.t0.timescale!r} not in {CLOCK_TIMESCALES} "
                "-- a simulation clock must be linear (no leap seconds mid-run)")
        object.__setattr__(self, "dt_dynamics_s", _pos(self.dt_dynamics_s, "ClockSpec.dt_dynamics_s"))
        specs = tuple(self.sensors)
        if not specs:
            raise GNCContractError("ClockSpec.sensors: at least one sensor -- a GNC sim "
                                   "without measurements has nothing to navigate on")
        for s in specs:
            if not isinstance(s, SensorSpec):
                raise GNCContractError(f"ClockSpec.sensors entries must be SensorSpec, got {s!r}")
        ids = [s.sensor_id for s in specs]
        if len(set(ids)) != len(ids):
            raise GNCContractError(f"ClockSpec.sensors: duplicate sensor ids {ids}")
        object.__setattr__(self, "sensors", specs)
        if not isinstance(self.estimator, EstimatorSpec):
            raise GNCContractError(f"ClockSpec.estimator must be an EstimatorSpec, got {self.estimator!r}")
        if not isinstance(self.controller, ControllerSpec):
            raise GNCContractError(f"ClockSpec.controller must be a ControllerSpec, got {self.controller!r}")
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "ClockSpec.metadata"))

    def sensor(self, sensor_id: str) -> SensorSpec:
        for s in self.sensors:
            if s.sensor_id == sensor_id:
                return s
        raise GNCContractError(f"clock has no sensor {sensor_id!r} (declared: "
                               f"{[s.sensor_id for s in self.sensors]})")

    def epoch_at(self, t_s: float) -> Epoch:
        """The E12-01 ``Epoch`` at ``t_s`` elapsed seconds after ``t0``, on
        the clock's own uniform label (exact for TT/TDB: those days count
        86400 SI seconds)."""
        t = _nonneg(t_s, "epoch_at(t_s)")
        return Epoch(mjd2000=self.t0.mjd2000 + t / SECONDS_PER_DAY, timescale=self.t0.timescale)

    def tdb_seconds_at(self, t_s: float) -> float:
        """TDB seconds since J2000 at ``t_s`` after t0 (the one comparable
        scale; ``Epoch.to_tdb_seconds`` of :meth:`epoch_at`)."""
        return self.epoch_at(t_s).to_tdb_seconds()

    def dynamics_times(self, t_end_s: float) -> tuple:
        """Fixed truth-integration grid t0, t0+dt, ... up to and including
        the last step <= ``t_end_s``."""
        end = _pos(t_end_s, "dynamics_times(t_end_s)")
        n = _last_tick(end, 0.0, self.dt_dynamics_s)
        # exact multiples, not a drifted accumulation; a float step would
        # put the 300th sample somewhere the declared grid has no point
        return tuple(round(k * self.dt_dynamics_s, 12) for k in range(n + 1))

    def sensor_schedule(self, sensor_id: str, t_end_s: float) -> tuple:
        """Every scheduled tick of one sensor up to ``t_end_s`` (inclusive
        of drops) -- the timeline the simulator must have *had* the chance to
        fill, used by the replay validator to check the log against the
        declaration."""
        spec = self.sensor(sensor_id)
        end = _pos(t_end_s, "sensor_schedule(t_end_s)")
        last = _last_tick(end, spec.phase_s, spec.period_s)
        if last < 0:
            return ()
        out = []
        for tick in range(last + 1):
            t = spec.sample_time(tick)
            out.append(SensorTick(sensor_id, tick, t, t + spec.latency_s,
                                  tick in set(spec.dropped_ticks)))
        return tuple(out)

    def estimator_times(self, t_end_s: float) -> tuple:
        end = _pos(t_end_s, "estimator_times(t_end_s)")
        last = _last_tick(end, self.estimator.phase_s, self.estimator.period_s)
        if last < 0:
            return ()
        return tuple(round(self.estimator.time(k), 12) for k in range(last + 1))

    def control_ticks(self, t_end_s: float) -> tuple:
        """(t_tick, t_fire) pairs: a command decided at the tick reaches the
        actuator one controller latency later."""
        end = _pos(t_end_s, "control_ticks(t_end_s)")
        last = _last_tick(end, self.controller.phase_s, self.controller.period_s)
        if last < 0:
            return ()
        return tuple((round(self.controller.time(k), 12),
                      round(self.controller.time(k) + self.controller.latency_s, 12))
                     for k in range(last + 1))

    def to_document(self) -> dict:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION, "kind": CLOCK_KIND,
            "t0": {"mjd2000": self.t0.mjd2000, "timescale": self.t0.timescale},
            "dt_dynamics_s": self.dt_dynamics_s,
            "sensors": [s.to_document() for s in self.sensors],
            "estimator": self.estimator.to_document(),
            "controller": self.controller.to_document(),
            "metadata": dict(self.metadata),
        }


def clock_from_document(doc: Mapping[str, Any]) -> ClockSpec:
    """Parse a gnc-contracts-1.0 clock document."""
    if not isinstance(doc, Mapping):
        raise GNCContractError(f"clock document must be a mapping, got {doc!r}")
    if doc.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise GNCContractError(
            f"no clock schema for version {doc.get('schema_version')!r} (implements "
            f"{CONTRACT_SCHEMA_VERSION!r})")
    if doc.get("kind") != CLOCK_KIND:
        raise GNCContractError(f"clock document kind {doc.get('kind')!r} != {CLOCK_KIND!r}")
    try:
        jsonschema.validate(instance=dict(doc), schema=_def_schema("clock"))
    except jsonschema.ValidationError as exc:
        raise GNCContractError(f"clock document violates gnc-contracts-1.0: {exc.message}") from exc
    return ClockSpec(
        t0=Epoch(mjd2000=_finite(doc["t0"]["mjd2000"], "clock.t0.mjd2000"),
                 timescale=doc["t0"]["timescale"]),
        dt_dynamics_s=doc["dt_dynamics_s"],
        sensors=tuple(sensor_spec_from_document(s) for s in doc["sensors"]),
        estimator=estimator_spec_from_document(doc["estimator"]),
        controller=controller_spec_from_document(doc["controller"]),
        metadata=doc.get("metadata", {}))
