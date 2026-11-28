"""Relative-measurement sensor models with declared errors (E13-03, task #420).

E13-01 fixed *what a measurement record is*; this module fixes *where the
measurement comes from and how it lies*. One relative-navigation sensor is
declared as a task-configuration document
(``orbitopt/schemas/gnc-sensor-config-1.0.json``, parsed by
:func:`sensor_task_config_from_document`) saying what it measures, how it
errs, and -- mandatorily, on every error parameter -- where the numbers come
from: a reference or ``DESIGN ASSUMPTION: <rationale>``. Nothing here
invents a number a reviewer cannot trace.

The domain has three tables, not a class zoo (extensibility = new rows):

* **CHANNELS** -- the measurement vocabulary: range, range rate, azimuth,
  elevation (line-of-sight direction in spherical form) and the six
  Cartesian relative components. Each channel knows its ideal measurement
  from the truth relative state and its Jacobians (needed to map measurement
  covariance into the Cartesian covariance the Observation contract carries).
* **SENSOR_KINDS** -- legal channel subsets and how a fix maps to the
  Cartesian Observation: ``polar_range_angle`` (range + LOS angles, optional
  Doppler -- a lidar/optical-tracker profile) and ``cartesian_relnav``
  (direct relative position, optional velocity -- a co-aligned GNSS-differenced
  profile).
* **Dropout schedule** -- random per-tick loss plus bursts of consecutive
  loss (geometric run lengths), generated as a pure function of
  (seed, n_ticks) and fed to the E13-01 clock as ``SensorSpec.dropped_ticks``
  so the clock stays the single authority over ticks and drops
  (a dropped frame is the *absence* of a record, never a retime).

Error model, per channel: additive constant **bias** (deterministic scenario
data -- the same every tick, so it is configured, not drawn), additive
**white noise** (Gaussian, seeded on (seed, tick) -- reproducible and
independent across ticks), and **latency**, which is *modelled as the clock
declares it*: constant per sensor, ``t_available_s == t_sample_s +
latency_s`` enforced by E13-01 (the reason a random-delay model is refused
here: a jittered arrival time makes the sealed replay's declared-schedule
check meaningless -- E13-02's filter consumes the honest constant).

Output is an E13-01 :class:`~orbitopt.gnc.contracts.Observation` (timestamps
+ covariance), or the honest absence of one on a dropped tick. The
:class:`SensorFix` layer below the Observation carries the explicit
validity flag the task asks for (the contract layer expresses validity as
presence -- ``valid=False`` never becomes a record);
:func:`measurement_nis` is the measurement-level innovation consistency
statistic E13-02's filter and the acceptance tests share.

Geometry convention (DESIGN ASSUMPTION): angles are spherical coordinates in
the mission frame with boresight along the frame +X axis
(``azimuth = atan2(y, x)``, ``elevation = atan2(z, hypot(x, y))``); a body
mount is a rotation the caller applies to the truth state before sampling,
which keeps this module frame-bookkeeping-free.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

import jsonschema
import numpy as np

from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState

from .clock import SensorSpec, SensorTick, _last_tick
from .contracts import GNCContractError, Observation

#: Schema file for the sensor task-configuration document.
SENSOR_CONFIG_SCHEMA_FILE = "gnc-sensor-config-1.0.json"
#: ``schema_version`` the document carries.
SENSOR_CONFIG_VERSION = "1.0"
SENSOR_CONFIG_KIND = "orbitopt.gnc.sensor-config"

#: Declared boresight singularity guard: measurements closer to the +Z axis
#: than this fraction of range have an ill-defined azimuth.
BORESIGHT_SINGULARITY = 1.0e-6


class SensorModelError(GNCContractError):
    """A sensor configuration or a measurement attempt violates the model
    (illegal channel subset, geometry at the boresight singularity, a tick
    that contradicts the sensor's declared timing)."""


# ---------------------------------------------------------------------------
# the measurement-channel table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementChannel:
    """One entry of the measurement vocabulary: how the ideal (noise-free,
    bias-free) value is produced from a truth relative state, and its
    Jacobians w.r.t. position and velocity (for the covariance map). Units
    live in the name (``*_m``, ``*_rad``, ``*_m_s``) like everywhere else in
    orbitopt; ``unit`` records them for provenance documents."""

    name: str
    unit: str
    ideal: Callable[[np.ndarray, np.ndarray], float]
    jac_pos: Callable[[np.ndarray, np.ndarray], np.ndarray]
    jac_vel: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _rng_vec(pos: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(pos))
    if n == 0.0:
        raise SensorModelError("zero relative range -- a bearing-free "
                               "geometry has no line of sight to measure")
    return pos / n


def _cyl(pos: np.ndarray) -> tuple[float, float]:
    """(d_xy, r) with d_xy = hypot(x, y); refuses the +Z boresight
    singularity where azimuth is undefined."""
    d_xy = float(math.hypot(float(pos[0]), float(pos[1])))
    r = float(np.linalg.norm(pos))
    if d_xy < BORESIGHT_SINGULARITY * max(r, 1.0):
        raise SensorModelError(f"LOS within {BORESIGHT_SINGULARITY:g} of the +Z boresight: "
                               "azimuth is undefined there; move the geometry or the mount")
    return d_xy, r


def _id_channel(idx: int, unit: str) -> MeasurementChannel:
    e = np.eye(3)[idx]
    return MeasurementChannel(
        name=f"rel_pos_{chr(ord('x') + idx)}_{unit}", unit=unit,
        ideal=lambda p, v, e=e: float(e @ p), jac_pos=lambda p, v, e=e: e.copy(),
        jac_vel=lambda p, v: np.zeros(3))


def _id_vel_channel(idx: int) -> MeasurementChannel:
    e = np.eye(3)[idx]
    return MeasurementChannel(
        name=f"rel_vel_{chr(ord('x') + idx)}_m_s", unit="m/s",
        ideal=lambda p, v, e=e: float(e @ v), jac_pos=lambda p, v: np.zeros(3),
        jac_vel=lambda p, v, e=e: e.copy())


CHANNELS: Mapping[str, MeasurementChannel] = {c.name: c for c in (
    MeasurementChannel(
        "range_m", "m",
        ideal=lambda p, v: float(np.linalg.norm(p)),
        jac_pos=lambda p, v: _rng_vec(p), jac_vel=lambda p, v: np.zeros(3)),
    MeasurementChannel(
        "range_rate_m_s", "m/s",
        ideal=lambda p, v: float(_rng_vec(p) @ v),
        jac_pos=lambda p, v: (np.asarray(v, dtype=np.float64)
                              - (float(np.dot(p, v)) / float(np.dot(p, p)) * np.asarray(p,
                                                                                        dtype=np.float64))
                              ) / float(np.linalg.norm(p)),
        jac_vel=lambda p, v: _rng_vec(p)),
    MeasurementChannel(
        "azimuth_rad", "rad",
        ideal=lambda p, v: float(math.atan2(float(p[1]), float(p[0]))),
        jac_pos=lambda p, v: np.array((-float(p[1]), float(p[0]), 0.0))
                             / (float(p[0]) ** 2 + float(p[1]) ** 2),
        jac_vel=lambda p, v: np.zeros(3)),
    MeasurementChannel(
        "elevation_rad", "rad",
        ideal=lambda p, v: float(math.atan2(float(p[2]), _cyl(p)[0])),
        # d = hypot(x, y), el = atan2(z, d): grad(el) = (-zx, -zy, d^2) / (r^2 d)
        jac_pos=lambda p, v: np.array([-float(p[2]) * float(p[0]), -float(p[2]) * float(p[1]),
                                       _cyl(p)[0] ** 2]) / (_cyl(p)[1] ** 2 * _cyl(p)[0]),
        jac_vel=lambda p, v: np.zeros(3)),
    _id_channel(0, "m"), _id_channel(1, "m"), _id_channel(2, "m"),
    _id_vel_channel(0), _id_vel_channel(1), _id_vel_channel(2),
)}

POSITION_BASES = (("range_m", "azimuth_rad", "elevation_rad"),
                  ("rel_pos_x_m", "rel_pos_y_m", "rel_pos_z_m"))
VELOCITY_BASES = (("rel_vel_x_m_s", "rel_vel_y_m_s", "rel_vel_z_m_s"),)
POLAR_POSITION = ("range_m", "azimuth_rad", "elevation_rad")


@dataclass(frozen=True)
class SensorKind:
    """A legal sensor profile: which channel subsets are accepted and how a
    fix maps to the Cartesian Observation (position base, optional velocity
    base, and whether range-rate-only velocity is allowed)."""

    kind: str
    allowed: tuple
    position_base: str  # name of POSITION_BASES entry
    velocity_bases: tuple  # names of VELOCITY_BASES entries, or "range_rate" alias
    allow_range_rate_velocity: bool


SENSOR_KINDS: Mapping[str, SensorKind] = {
    "polar_range_angle": SensorKind(
        kind="polar_range_angle",
        allowed=("range_m", "range_rate_m_s", "azimuth_rad", "elevation_rad"),
        position_base="polar", velocity_bases=(), allow_range_rate_velocity=True),
    "cartesian_relnav": SensorKind(
        kind="cartesian_relnav",
        allowed=("rel_pos_x_m", "rel_pos_y_m", "rel_pos_z_m",
                 "rel_vel_x_m_s", "rel_vel_y_m_s", "rel_vel_z_m_s"),
        position_base="cartesian", velocity_bases=("cartesian",),
        allow_range_rate_velocity=False),
}


# ---------------------------------------------------------------------------
# configuration dataclasses (parse at the boundary, validate on construction)
# ---------------------------------------------------------------------------


def _num(value: Any, where: str, *, minimum: float | None = None,
         exclusive: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise SensorModelError(f"{where}: {value!r} is not a number") from exc
    if not math.isfinite(v):
        raise SensorModelError(f"{where}: {value!r} is not finite")
    if minimum is not None and (v <= minimum if exclusive else v < minimum):
        raise SensorModelError(f"{where}: must be {'>' if exclusive else '>='} "
                               f"{minimum}, got {v}")
    return v


@dataclass(frozen=True)
class ChannelErrorConfig:
    """Declared error model of one channel: white-noise sigma (strictly
    positive -- a zero-variance claim is a filter trap, not a perfect
    sensor), constant bias, and the mandatory provenance string."""

    channel: str
    sigma: float
    bias: float
    source: str

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise SensorModelError(f"unknown measurement channel {self.channel!r}; the "
                                   f"registry implements {sorted(CHANNELS)}")
        object.__setattr__(self, "sigma", _num(self.sigma, f"channel({self.channel}).sigma",
                                               minimum=0.0, exclusive=True))
        object.__setattr__(self, "bias", _num(self.bias, f"channel({self.channel}).bias"))
        if not isinstance(self.source, str) or not self.source.strip():
            raise SensorModelError(f"channel({self.channel}): source is mandatory -- every error "
                                   "parameter states where its numbers come from (citation or "
                                   "DESIGN ASSUMPTION)")

    @property
    def unit(self) -> str:
        return CHANNELS[self.channel].unit


@dataclass(frozen=True)
class DropoutConfig:
    """Data-loss model: independent random per-tick loss, plus bursts of
    consecutive loss entered with per-clear-tick probability
    ``burst_enter_p`` and held for a geometric run length of mean
    ``burst_mean_ticks``. ``explicit_ticks`` are scenario-declared losses
    merged on top. The stochastic mask is generated by (seed, n_ticks) --
    see :meth:`RelativeSensorModel.dropout_schedule`."""

    random_drop_p: float = 0.0
    burst_enter_p: float = 0.0
    burst_mean_ticks: float = 1.0
    explicit_ticks: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "random_drop_p",
                           _num(self.random_drop_p, "dropout.random_drop_p", minimum=0.0))
        object.__setattr__(self, "random_drop_p",
                           min(float(self.random_drop_p), 1.0))
        object.__setattr__(self, "burst_enter_p",
                           _num(self.burst_enter_p, "dropout.burst_enter_p", minimum=0.0))
        object.__setattr__(self, "burst_enter_p", min(float(self.burst_enter_p), 1.0))
        object.__setattr__(self, "burst_mean_ticks",
                           _num(self.burst_mean_ticks, "dropout.burst_mean_ticks", minimum=1.0))
        ticks = tuple(self.explicit_ticks)
        for t in ticks:
            if isinstance(t, bool) or not isinstance(t, int) or t < 0:
                raise SensorModelError(f"dropout.dropped_ticks: non-negative ints required, "
                                       f"got {t!r}")
        if len(set(ticks)) != len(ticks):
            raise SensorModelError("dropout.dropped_ticks: duplicate ticks")
        object.__setattr__(self, "explicit_ticks", tuple(sorted(ticks)))


@dataclass(frozen=True)
class RelativeSensorModel:
    """One declared sensor. Timing fields feed ``SensorSpec`` (the clock stays
    the timing authority); ``channels`` + ``dropout`` declare the errors;
    ``seed`` roots every reproducible draw; ``bias_covariance_mode`` says
    whether the declared covariance conservatively covers the bias
    (``"bounded"`` -- bias sign unknown) or covers white noise only
    (``"none"`` -- the filter must model the bias; NIS inflation otherwise is
    demonstrated, not discovered late, in tests/test_e13_03_nis_consistency.py)."""

    sensor_id: str
    kind: str
    period_s: float
    phase_s: float
    latency_s: float
    seed: int
    channels: tuple
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    bias_covariance_mode: str = "none"
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise SensorModelError(f"sensor_id must be a non-empty string, got {self.sensor_id!r}")
        if self.kind not in SENSOR_KINDS:
            raise SensorModelError(f"sensor kind {self.kind!r} not implemented; SENSOR_KINDS "
                                   f"has {sorted(SENSOR_KINDS)}")
        prof = SENSOR_KINDS[self.kind]
        object.__setattr__(self, "period_s", _num(self.period_s, "period_s", minimum=0.0,
                                                  exclusive=True))
        object.__setattr__(self, "phase_s", _num(self.phase_s, "phase_s", minimum=0.0))
        if self.phase_s >= self.period_s:
            raise SensorModelError(f"sensor({self.sensor_id}): phase_s {self.phase_s} >= "
                                   f"period_s {self.period_s} -- phase is an offset within one "
                                   "period (the clock's rule, inherited)")
        object.__setattr__(self, "latency_s", _num(self.latency_s, "latency_s", minimum=0.0))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise SensorModelError(f"sensor({self.sensor_id}): seed must be a non-negative int, "
                                   f"got {self.seed!r}")
        chans = tuple(self.channels)
        if not chans:
            raise SensorModelError(f"sensor({self.sensor_id}): no measurement channels declared")
        names = tuple(c.channel for c in chans)
        if len(set(names)) != len(names):
            raise SensorModelError(f"sensor({self.sensor_id}): duplicate channels {names}")
        for n in names:
            if n not in prof.allowed:
                raise SensorModelError(f"sensor({self.sensor_id}) kind {self.kind!r} cannot "
                                       f"measure {n!r}; this profile takes {prof.allowed}")
        by_kind = set(names)
        pb = POLAR_POSITION if prof.position_base == "polar" else POSITION_BASES[1]
        missing = [c for c in pb if c not in by_kind]
        if missing:
            raise SensorModelError(f"sensor({self.sensor_id}): channels {missing} are required "
                                   f"to reconstruct a Cartesian position for kind {self.kind!r}")
        vel_names = tuple(n for n in names if n.startswith("rel_vel_"))
        if vel_names and len(vel_names) != 3:
            raise SensorModelError(f"sensor({self.sensor_id}): velocity channels are all three "
                                   f"axes or none, got {vel_names}")
        if self.bias_covariance_mode not in ("none", "bounded"):
            raise SensorModelError(f"sensor({self.sensor_id}): bias_covariance_mode "
                                   f"{self.bias_covariance_mode!r} not in ('none', 'bounded')")
        if not isinstance(self.dropout, DropoutConfig):
            raise SensorModelError(f"sensor({self.sensor_id}): dropout must be a DropoutConfig")
        if not isinstance(self.source, str) or not self.source.strip():
            raise SensorModelError(f"sensor({self.sensor_id}): source is mandatory for timing / "
                                   "profile / bias-mode provenance")
        object.__setattr__(self, "channels", chans)
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- timing ---------------------------------------------------------------

    def sensor_spec(self, n_ticks: int) -> SensorSpec:
        """The E13-01 clock declaration for this sensor, drops included:
        ``dropped_ticks`` = explicit scenario drops merged with the seeded
        stochastic mask for ``n_ticks`` scheduled samples. Build the
        ``ClockSpec`` with this, and the log's replay checks agree with the
        error model by construction."""
        if isinstance(n_ticks, bool) or not isinstance(n_ticks, int) or n_ticks < 0:
            raise SensorModelError(f"sensor({self.sensor_id}): n_ticks must be a non-negative "
                                   f"int, got {n_ticks!r}")
        return SensorSpec(sensor_id=self.sensor_id, period_s=self.period_s, phase_s=self.phase_s,
                          latency_s=self.latency_s,
                          dropped_ticks=sorted(set(self.dropout.explicit_ticks)
                                               | set(self.dropout_schedule(n_ticks))))

    def n_ticks_at(self, t_end_s: float) -> int:
        """Scheduled ticks up to ``t_end_s`` (the horizon the schedule is a
        function of -- declare it once, e.g. at ClockSpec build)."""
        end = _num(t_end_s, "n_ticks_at(t_end_s)", minimum=0.0, exclusive=True)
        return max(0, _last_tick(end, self.phase_s, self.period_s) + 1)

    def dropout_schedule(self, n_ticks: int) -> tuple:
        """Seeded random+burst drop ticks below ``n_ticks``. A pure function
        of (seed, n_ticks): reproducible per run, and NOT prefix-stable in
        n_ticks (the mask is drawn once against the declared horizon, like
        every other seeded draw in this package). Random bursts are drawn
        first (one uniform per clear tick), burst lengths geometric on
        support >= 1 (mean ``burst_mean_ticks``)."""
        if isinstance(n_ticks, bool) or not isinstance(n_ticks, int) or n_ticks < 0:
            raise SensorModelError(f"sensor({self.sensor_id}): n_ticks must be a non-negative "
                                   f"int, got {n_ticks!r}")
        d = self.dropout
        rng = np.random.default_rng([int(self.seed), int(n_ticks), 1])
        dropped: list = []
        tick = 0
        while tick < n_ticks:
            if d.burst_enter_p > 0.0 and rng.random() < d.burst_enter_p:
                length = int(rng.geometric(1.0 / d.burst_mean_ticks))
                dropped.extend(range(tick, min(tick + length, n_ticks)))
                tick += length
                continue
            if d.random_drop_p > 0.0 and rng.random() < d.random_drop_p:
                dropped.append(tick)
            tick += 1
        return tuple(dropped)

    # -- measurement ----------------------------------------------------------

    def measure(self, tick: SensorTick, truth: RelativeState) -> SensorFix:
        """Sample one scheduled tick against the truth relative state
        (plant-side call -- this module reads truth by design; only
        :meth:`observe` output crosses to the navigation side). The fix is
        explicit about validity: a dropped frame returns ``valid=False``
        with a reason; the Observation contract expresses the same fact as
        the absence of a record."""
        if not isinstance(tick, SensorTick):
            raise SensorModelError(f"measure() needs a clock SensorTick, got {tick!r}")
        if tick.sensor_id != self.sensor_id:
            raise SensorModelError(f"tick belongs to sensor {tick.sensor_id!r}, not "
                                   f"{self.sensor_id!r}")
        drift = tick.t_available_s - (tick.t_sample_s + self.latency_s)
        if abs(drift) > 1.0e-6:
            raise SensorModelError(f"sensor({self.sensor_id}): tick arrival {tick.t_available_s} "
                                   f"!= sample {tick.t_sample_s} + declared latency "
                                   f"{self.latency_s} -- the clock and the sensor disagree about "
                                   "the delay; fix the declaration, not the arithmetic")
        if not isinstance(truth, RelativeState):
            raise SensorModelError(f"measure() needs a RelativeState truth, got {truth!r}")
        pos = np.asarray(truth.r_rel_m, dtype=np.float64)
        vel = np.asarray(truth.v_rel_m_s, dtype=np.float64)
        if tick.dropped:
            return SensorFix(sensor_id=self.sensor_id, tick=tick.tick, t_sample_s=tick.t_sample_s,
                             t_available_s=tick.t_available_s, valid=False,
                             dropped_reason="declared_drop", measurements={},
                             measurement_variance={}, bias_covariance_mode=self.bias_covariance_mode)
        if SENSOR_KINDS[self.kind].position_base == "polar":
            _rng_vec(pos)  # range > 0 required before any bearing is computed
            _cyl(pos)  # azimuth is undefined at the +Z boresight: refuse, never guess
        rng = np.random.default_rng([int(self.seed), int(tick.tick)])
        measurements: dict = {}
        variance: dict = {}
        for ch in self.channels:
            chan = CHANNELS[ch.channel]
            noisy = chan.ideal(pos, vel) + ch.bias + rng.normal(0.0, ch.sigma)
            var = ch.sigma ** 2
            if self.bias_covariance_mode == "bounded":
                var += ch.bias ** 2
            measurements[ch.channel] = float(noisy)
            variance[ch.channel] = float(var)
        return SensorFix(sensor_id=self.sensor_id, tick=tick.tick, t_sample_s=tick.t_sample_s,
                         t_available_s=tick.t_available_s, valid=True, dropped_reason=None,
                         measurements=measurements, measurement_variance=variance,
                         bias_covariance_mode=self.bias_covariance_mode)

    def observe(self, tick: SensorTick, truth: RelativeState, *, target_id: str,
                chaser_id: str, epoch: Epoch) -> Observation | None:
        """The E13-01-facing output: an ``Observation`` contract record for a
        kept tick, ``None`` for a dropped one (absence, never a fake value).
        Cartesian position is reconstructed from the position base; its
        covariance is the linear map of the declared diagonal measurement
        covariance through the channel Jacobians. Velocity: from the
        velocity channels, or from range rate alone as
        ``rho_dot * r_hat`` with the (rank-1, positive-semi-definite)
        covariance that honestly claims along-LOS knowledge only; when the
        profile measures no velocity, the record carries no velocity
        covariance and the filter must propagate it (that is E13-02's job,
        not a number invented here)."""
        fix = self.measure(tick, truth)
        if not fix.valid:
            return None
        prof = SENSOR_KINDS[self.kind]
        pos, pos_cov = _reconstruct_position(fix, prof)
        vel, vel_cov = _reconstruct_velocity(fix, prof, pos)
        rs = RelativeState(target_id=target_id, chaser_id=chaser_id, epoch=epoch,
                           frame=truth.frame, r_rel_m=pos, v_rel_m_s=vel)
        metadata: dict = {"sensor_kind": self.kind,
                          "velocity_measured": "true" if vel_cov else "false"}
        metadata.update(self.metadata)
        return Observation(record_id=f"obs-{self.sensor_id}-{tick.tick:05d}",
                           sensor_id=self.sensor_id, t_sample_s=fix.t_sample_s,
                           t_available_s=fix.t_available_s, latency_s=self.latency_s,
                           relative_state=rs, covariance_position_m2=pos_cov,
                           covariance_velocity_m2_s2=vel_cov, metadata=metadata)


@dataclass(frozen=True)
class SensorFix:
    """One raw measurement set at one tick: native channel values plus the
    declared per-channel variances and the validity flag (the explicit
    ``valid`` flag; it is the input to :meth:`RelativeSensorModel.observe`,
    which is where the E13-01 presence/absence discipline applies)."""

    sensor_id: str
    tick: int
    t_sample_s: float
    t_available_s: float
    valid: bool
    dropped_reason: str | None
    measurements: Mapping[str, float]
    measurement_variance: Mapping[str, float]
    bias_covariance_mode: str = "none"

    def __post_init__(self):
        if not isinstance(self.valid, bool):
            raise SensorModelError("SensorFix.valid must be a bool")
        if self.valid and not self.measurements:
            raise SensorModelError("a valid SensorFix must carry measurements")
        for k, v in self.measurement_variance.items():
            if k not in self.measurements:
                raise SensorModelError(f"variance for {k!r} without a measurement")
            if not (v > 0.0) or not math.isfinite(v):
                raise SensorModelError(f"declared variance for {k!r} must be finite and > 0, "
                                       f"got {v!r}")


# ---------------------------------------------------------------------------
# fix -> Observation reconstruction (table-driven by profile, no branching zoo)
# ---------------------------------------------------------------------------


def _channel_var(fix: SensorFix, name: str) -> float:
    return float(fix.measurement_variance[name])


def _reconstruct_position(fix: SensorFix, prof: SensorKind) -> tuple:
    if prof.position_base == "cartesian":
        pos = tuple(float(fix.measurements[f"rel_pos_{c}_m"]) for c in "xyz")
        var = tuple(_channel_var(fix, f"rel_pos_{c}_m") for c in "xyz")
        return pos, tuple(_diag9(var))
    rho = float(fix.measurements["range_m"])
    az = float(fix.measurements["azimuth_rad"])
    el = float(fix.measurements["elevation_rad"])
    if rho <= 0.0:
        raise SensorModelError(f"reconstructed range {rho} <= 0 -- noise+bias swamped the range "
                               "measurement; this geometry is outside the sensor model")
    pos = (rho * math.cos(el) * math.cos(az),
           rho * math.cos(el) * math.sin(az),
           rho * math.sin(el))
    j = np.stack([CHANNELS[n].jac_pos(np.asarray(pos), np.zeros(3))
                  for n in POLAR_POSITION])
    w = np.diag([1.0 / _channel_var(fix, n) for n in POLAR_POSITION])
    cov = np.linalg.inv(j.T @ w @ j)
    return pos, tuple(float(x) for x in cov.reshape(9))


def _reconstruct_velocity(fix: SensorFix, prof: SensorKind, pos: tuple) -> tuple:
    vel_names = [n for n in fix.measurements if n.startswith("rel_vel_")]
    if vel_names:
        vel = tuple(float(fix.measurements[f"rel_vel_{c}_m_s"]) for c in "xyz")
        var = tuple(_channel_var(fix, f"rel_vel_{c}_m_s") for c in "xyz")
        return vel, tuple(_diag9(var))
    if prof.allow_range_rate_velocity and "range_rate_m_s" in fix.measurements:
        r = np.asarray(pos, dtype=np.float64)
        n = float(np.linalg.norm(r))
        if n == 0.0:
            raise SensorModelError("cannot project range rate onto a zero measured range")
        r_hat = r / n
        rdot = float(fix.measurements["range_rate_m_s"])
        var = _channel_var(fix, "range_rate_m_s")
        cov = var * np.outer(r_hat, r_hat)  # rank-1 PSD: along-LOS knowledge, honestly stated
        return tuple(float(x) for x in rdot * r_hat), tuple(float(x) for x in cov.reshape(9))
    return (0.0, 0.0, 0.0), ()


def _diag9(d: tuple) -> list:
    out = [0.0] * 9
    for i, v in enumerate(d):
        out[i * 4] = v
    return out


# ---------------------------------------------------------------------------
# measurement-level innovation consistency (the statistic E13-02 consumes)
# ---------------------------------------------------------------------------


def measurement_nis(fix: SensorFix, truth: RelativeState) -> float:
    """Normalized innovation sum of one fix against a reference state:
    ``sum_i (z_i - h_i(x))^2 / var_i`` over the fix's channels (chi-square
    with one degree of freedom per channel IF the declared variances are the
    truth and the reference state is what the sensor saw). Computed in
    measurement space -- no linearisation, no singularity -- so it is exactly
    the consistency gate for white noise, and it *shows* bias as inflation
    (the innovation carries the bias; the declared variance does not, unless
    ``bias_covariance_mode == 'bounded'``). A running filter adds its own
    prediction-residual terms; this is the measurement half of E13-02's NIS."""
    if not isinstance(fix, SensorFix):
        raise SensorModelError(f"measurement_nis needs a SensorFix, got {fix!r}")
    if not fix.valid:
        raise SensorModelError("NIS of a dropped frame is undefined -- the frame is absent")
    pos = np.asarray(truth.r_rel_m, dtype=np.float64)
    vel = np.asarray(truth.v_rel_m_s, dtype=np.float64)
    nis = 0.0
    for name, z in sorted(fix.measurements.items()):
        chan = CHANNELS[name]
        res = z - chan.ideal(pos, vel)
        nis += res * res / float(fix.measurement_variance[name])
    return float(nis)


# ---------------------------------------------------------------------------
# task-configuration documents (schema-gated, like every orbitopt boundary)
# ---------------------------------------------------------------------------


@cache
def _config_schema() -> dict:
    traversable = resources.files("orbitopt").joinpath("schemas", SENSOR_CONFIG_SCHEMA_FILE)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover -- packaging bug
        raise SensorModelError(f"No packaged schema orbitopt/schemas/{SENSOR_CONFIG_SCHEMA_FILE}")
    return json.loads(text)


def _sensor_def_schema() -> dict:
    schema = _config_schema()
    return {"$schema": schema["$schema"], "$id": schema["$id"], "$defs": schema["$defs"],
            "$ref": "#/$defs/sensor"}


def sensor_config_to_document(cfg: RelativeSensorModel) -> dict:
    """One sensor's task-configuration member (schema-valid by
    construction -- construction is the gate)."""
    return {"sensor_id": cfg.sensor_id, "kind": cfg.kind, "period_s": cfg.period_s,
            "phase_s": cfg.phase_s, "latency_s": cfg.latency_s,
            "bias_covariance_mode": cfg.bias_covariance_mode, "seed": cfg.seed,
            "channels": [{"channel": c.channel, "sigma": c.sigma, "bias": c.bias,
                          "source": c.source} for c in cfg.channels],
            "dropout": {"random_drop_p": cfg.dropout.random_drop_p,
                        "burst_enter_p": cfg.dropout.burst_enter_p,
                        "burst_mean_ticks": cfg.dropout.burst_mean_ticks,
                        "dropped_ticks": list(cfg.dropout.explicit_ticks),
                        "source": str(cfg.metadata.get("dropout_source", cfg.source))},
            "source": cfg.source}


def sensor_config_from_document(doc: Mapping[str, Any]) -> RelativeSensorModel:
    """Parse one sensor member: JSON-Schema gate first (shape, enums, units
    by field name), then the semantic walls (channel subsets, phase <
    period, positive sigmas, mandatory sources)."""
    try:
        jsonschema.validate(instance=dict(doc), schema=_sensor_def_schema())
    except jsonschema.ValidationError as exc:
        raise SensorModelError(f"sensor config violates {SENSOR_CONFIG_SCHEMA_FILE}: "
                               f"{exc.message}") from exc
    dd = doc["dropout"]
    meta = {"dropout_source": dd["source"]}
    return RelativeSensorModel(
        sensor_id=doc["sensor_id"], kind=doc["kind"], period_s=doc["period_s"],
        phase_s=doc["phase_s"], latency_s=doc["latency_s"], seed=doc["seed"],
        channels=tuple(ChannelErrorConfig(channel=c["channel"], sigma=c["sigma"], bias=c["bias"],
                                          source=c["source"]) for c in doc["channels"]),
        dropout=DropoutConfig(random_drop_p=dd["random_drop_p"],
                              burst_enter_p=dd["burst_enter_p"],
                              burst_mean_ticks=dd["burst_mean_ticks"],
                              explicit_ticks=tuple(dd["dropped_ticks"])),
        bias_covariance_mode=doc["bias_covariance_mode"], source=doc["source"], metadata=meta)


def sensor_task_config_to_document(configs: tuple, metadata: Mapping[str, Any] | None = None) -> dict:
    """The whole task-configuration document for a run's sensors."""
    doc = {"schema_version": SENSOR_CONFIG_VERSION, "kind": SENSOR_CONFIG_KIND,
           "sensors": [sensor_config_to_document(c) for c in configs]}
    if metadata:
        doc["metadata"] = dict(metadata)
    return doc


def sensor_task_config_from_document(doc: Mapping[str, Any]) -> tuple:
    """Parse + validate a task-configuration document, returning the sensor
    models in declared order."""
    if not isinstance(doc, Mapping):
        raise SensorModelError(f"task config must be a mapping, got {type(doc).__name__}")
    if doc.get("schema_version") != SENSOR_CONFIG_VERSION:
        raise SensorModelError(f"no sensor config schema for version "
                               f"{doc.get('schema_version')!r} (implements "
                               f"{SENSOR_CONFIG_VERSION!r})")
    if doc.get("kind") != SENSOR_CONFIG_KIND:
        raise SensorModelError(f"task config kind {doc.get('kind')!r} != {SENSOR_CONFIG_KIND!r}")
    try:
        jsonschema.validate(instance=dict(doc), schema=_config_schema())
    except jsonschema.ValidationError as exc:
        raise SensorModelError(f"task config violates {SENSOR_CONFIG_SCHEMA_FILE}: "
                               f"{exc.message}") from exc
    return tuple(sensor_config_from_document(s) for s in doc["sensors"])


def reference_task_config() -> dict:
    """The E13-03 reference sensor pair (documentation-quality defaults every
    value of which is a DESIGN ASSUMPTION anchored on typical proximity-
    operations hardware -- a lidar-class LOS sensor and a GNSS-differenced
    relative-nav unit at LEO rendezvous ranges). Use it as the starting
    declaration; re-source every number for a qualified mission."""
    return sensor_task_config_from_document(_REFERENCE_DOC)


#: Values are DESIGN ASSUMPTIONs (task acceptance: mark provenance): typical
#: spaceborne lidar at 100-1000 m ranges gives ~0.5 m range noise and tens of
#: micro-radians of LOS angle noise; Doppler gives ~cm/s range-rate noise
#: (the numbers are round engineering bounds, not a datasheet copy). Bias is
#: a calibrated-but-residual mount/oscillator offset, one order below noise.
#: Drop statistics model a bus-computing-delayed image pipeline: ~1% random
#: frame loss, rare ~5-frame processor stalls. Latency 0.2 s: exposure +
#: downlink + processing budget at 1 Hz. Cartesian profile: GNSS differential
#: relative navigation, metre-class position, cm/s-class velocity.
_REFERENCE_DOC = {
    "schema_version": "1.0", "kind": SENSOR_CONFIG_KIND,
    "sensors": [
        {"sensor_id": "lidar", "kind": "polar_range_angle",
         "period_s": 1.0, "phase_s": 0.0, "latency_s": 0.2,
         "bias_covariance_mode": "none", "seed": 42,
         "channels": [
             {"channel": "range_m", "sigma": 0.5, "bias": 0.05,
              "source": "DESIGN ASSUMPTION: lidar-class ranging noise at LEO-RDV ranges"},
             {"channel": "azimuth_rad", "sigma": 1.0e-4, "bias": 5.0e-5,
              "source": "DESIGN ASSUMPTION: optical LOS pointing error budget"},
             {"channel": "elevation_rad", "sigma": 1.0e-4, "bias": 5.0e-5,
              "source": "DESIGN ASSUMPTION: optical LOS pointing error budget"},
             {"channel": "range_rate_m_s", "sigma": 0.02, "bias": 0.005,
              "source": "DESIGN ASSUMPTION: Doppler front-end at 1 s integration"}],
         "dropout": {"random_drop_p": 0.01, "burst_enter_p": 0.005, "burst_mean_ticks": 5.0,
                     "dropped_ticks": [], "source": "DESIGN ASSUMPTION: bus image-pipeline loss"},
         "source": "DESIGN ASSUMPTION: 1 Hz LOS sensor, 0.2 s pipeline latency"},
        {"sensor_id": "relnav", "kind": "cartesian_relnav",
         "period_s": 2.0, "phase_s": 0.5, "latency_s": 0.5,
         "bias_covariance_mode": "none", "seed": 11,
         "channels": [
             {"channel": "rel_pos_x_m", "sigma": 1.0, "bias": 0.1,
              "source": "DESIGN ASSUMPTION: differential GNSS baseline at km baselines"},
             {"channel": "rel_pos_y_m", "sigma": 1.0, "bias": 0.1,
              "source": "DESIGN ASSUMPTION: differential GNSS baseline at km baselines"},
             {"channel": "rel_pos_z_m", "sigma": 1.0, "bias": 0.1,
              "source": "DESIGN ASSUMPTION: differential GNSS baseline at km baselines"},
             {"channel": "rel_vel_x_m_s", "sigma": 0.02, "bias": 0.002,
              "source": "DESIGN ASSUMPTION: Doppler-derived relative velocity"},
             {"channel": "rel_vel_y_m_s", "sigma": 0.02, "bias": 0.002,
              "source": "DESIGN ASSUMPTION: Doppler-derived relative velocity"},
             {"channel": "rel_vel_z_m_s", "sigma": 0.02, "bias": 0.002,
              "source": "DESIGN ASSUMPTION: Doppler-derived relative velocity"}],
         "dropout": {"random_drop_p": 0.0, "burst_enter_p": 0.0, "burst_mean_ticks": 1.0,
                     "dropped_ticks": [], "source": "DESIGN ASSUMPTION: GNSS link assumed clean"},
         "source": "DESIGN ASSUMPTION: 0.5 Hz GNSS-differenced relative nav, 0.5 s latency"},
    ],
}
