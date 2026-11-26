"""A reference closed-loop harness proving the E13-01 contracts end to end.

This is deliberately the *simplest* honest GNC loop, not a filter design
(that is E13-02..): two-body truth (E12-02's ``propagate_pair`` between
every pair of event instants), sensors = truth + seeded Gaussian noise at
their declared ticks, filter = "latest available measurement per sensor,
covariance = the measurement's" (no fusion, no propagation), guidance = any
:class:`~orbitopt.gnc.feed.GuidanceLaw`, actuator = commanded delta-v times
a seeded per-axis execution scale, recorded as a ControlEvent either way.
Even trivial, the loop keeps every discipline the contracts enforce:

* truth is written to the run log and never handed to the law -- the law is
  called through :func:`orbitopt.gnc.feed.run_guidance_step` with the
  :class:`~orbitopt.gnc.feed.GuidanceInput` as its only input;
* simultaneity has ONE priority order, recorded in the log's ``config``:
  integrate plant -> record truth -> sample sensors -> execute actuations
  -> filter update -> control tick;
* nothing is invented when data are missing: with no estimate yet a control
  tick records no command (the *absence* is the honest record), and a coast
  command is still executed (and recorded) as a ControlEvent with
  ``executed=False``.

The output is a sealed :class:`~orbitopt.gnc.recording.GncRunLog`:
measurements, estimates, control and configuration records replayable via
:func:`~orbitopt.gnc.recording.read_run_log` -- the E13-01 acceptance
surface. :class:`ProportionalLaw` is the shipped demo law (damped position
feedback, capped, coasts inside the hold-point radius).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState
from orbitopt.relative.propagation import EARTH_MU, propagate_pair

from .clock import ClockSpec
from .contracts import Command, ControlEvent, Estimate, GNCContractError, Observation, TruthSample
from .feed import GuidanceInput, GuidanceLaw, run_guidance_step
from .recording import GncRunLog

_NS = 1.0e-9  # event-key resolution: merge instants equal to within 1 ns


@dataclass(frozen=True)
class SensorNoise:
    """Seeded additive-Gaussian noise for one sensor (the run log's
    ``config`` section records what was used; the seed makes the run
    reproducible, the log makes it checkable)."""

    std_position_m: float
    std_velocity_m_s: float
    seed: int = 0

    def __post_init__(self):
        for name in ("std_position_m", "std_velocity_m_s"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"SensorNoise.{name} must be a finite >= 0 number, got {v}")
            object.__setattr__(self, name, v)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class ProportionalLaw:
    """Demo guidance: commanded delta-v = -(kp * rho + kd * rho_dot), capped
    at ``dv_max_m_s``, coasts (zero) inside ``hold_range_m``. Implements
    :class:`orbitopt.gnc.feed.GuidanceLaw` -- the estimate is its entire
    worldview."""

    kp_per_s: float
    kd: float
    dv_max_m_s: float = 0.5
    hold_range_m: float = 10.0

    def delta_v(self, estimate: Estimate) -> tuple:
        rho = np.asarray(estimate.relative_state.r_rel_m, dtype=np.float64)
        rho_dot = np.asarray(estimate.relative_state.v_rel_m_s, dtype=np.float64)
        if float(np.linalg.norm(rho)) <= self.hold_range_m:
            return (0.0, 0.0, 0.0)
        dv = -(self.kp_per_s * rho + self.kd * rho_dot)
        mag = float(np.linalg.norm(dv))
        if mag > self.dv_max_m_s:
            dv = dv * (self.dv_max_m_s / mag)
        return tuple(float(x) for x in dv)


def _pair(target_id: str, chaser_id: str, epoch: Epoch, frame: str,
          r_rel_m, v_rel_m_s) -> RelativeState:
    """A RelativeState formed at one shared instant in one shared frame (the
    sim holds both truth states, so the E12-01 alignment rule is satisfied
    by construction)."""
    return RelativeState(target_id=target_id, chaser_id=chaser_id, epoch=epoch, frame=frame,
                         r_rel_m=tuple(float(x) for x in r_rel_m),
                         v_rel_m_s=tuple(float(x) for x in v_rel_m_s))


def simulate(*, run_id: str, clock: ClockSpec, frame: str,
             target_id: str, chaser_id: str,
             chief_r_m, chief_v_m_s, deputy_r_m, deputy_v_m_s,
             laws: Mapping[str, GuidanceLaw],
             noise: Mapping[str, SensorNoise],
             t_end_s: float,
             mu: float = EARTH_MU, j2: float = 0.0,
             sub_dt_s: float | None = None,
             execution_scale: float = 1.0, execution_rtol: float = 0.0,
             execution_seed: int = 0,
             mission: Mapping[str, Any] | None = None,
             plan_digest: str | None = None,
             config_extra: Mapping[str, Any] | None = None) -> GncRunLog:
    """Fly the simplest honest closed loop and return the replay-sealed run
    log. ``laws`` maps chaser spacecraft_id -> GuidanceLaw; ``noise`` maps
    clock sensor_id -> SensorNoise (exactly the declared sensors). The plant
    is integrated between consecutive event instants with
    ``orbitopt.relative.propagate_pair`` (``sub_dt_s`` tightens the RK4
    step). All times are elapsed seconds on ``clock``; actuation events past
    ``t_end_s`` still run (every command gets its honest ControlEvent)."""
    for name, v in (("execution_scale", execution_scale), ("execution_rtol", execution_rtol)):
        if not math.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError(f"simulate(): {name} must be a finite >= 0 number, got {v!r}")
    declared = {s.sensor_id for s in clock.sensors}
    if set(noise) != declared:
        raise ValueError(f"simulate(): noise must be declared for exactly the clock's sensors "
                         f"{sorted(declared)}, got {sorted(noise)}")
    if not laws:
        raise ValueError("simulate(): at least one chaser needs a guidance law")

    r_c = np.asarray(chief_r_m, dtype=np.float64).reshape(3)
    v_c = np.asarray(chief_v_m_s, dtype=np.float64).reshape(3)
    r_d = np.asarray(deputy_r_m, dtype=np.float64).reshape(3)
    v_d = np.asarray(deputy_v_m_s, dtype=np.float64).reshape(3)

    config: dict = {
        "frame": frame, "target_id": target_id, "chaser_id": chaser_id,
        "mu_m3_s2": float(mu), "j2": float(j2), "t_end_s": float(t_end_s),
        "event_priority": ["plant", "truth", "observation", "control_event",
                           "estimate", "command"],
        "noise": {sid: {"std_position_m": n.std_position_m, "std_velocity_m_s": n.std_velocity_m_s,
                        "seed": n.seed} for sid, n in sorted(noise.items())},
        "execution": {"scale": float(execution_scale), "rtol": float(execution_rtol),
                      "seed": int(execution_seed)},
    }
    if config_extra:
        config.update(dict(config_extra))
    log = GncRunLog.new(run_id=run_id, clock=clock, config=config, mission=mission,
                        plan_digest=plan_digest)

    # ---- event timeline, every instant straight from the clock's schedules -
    truth: dict = {}
    for t in clock.dynamics_times(t_end_s):
        truth.setdefault(round(t / _NS), t)
    samples: dict = {}
    for spec in clock.sensors:
        for tick in clock.sensor_schedule(spec.sensor_id, t_end_s):
            samples.setdefault(round(tick.t_sample_s / _NS), []).append(tick)
    filter_keys = {round(t / _NS) for t in clock.estimator_times(t_end_s)}
    ticks: dict = {}
    actuations: dict = {}
    for t_tick, t_exec in clock.control_ticks(t_end_s):
        ticks.setdefault(round(t_tick / _NS), []).append((t_tick, t_exec))
        actuations.setdefault(round(t_exec / _NS), []).append((t_tick, t_exec))
    moment: dict = {}

    def _mark(key: int, t_rep: float) -> None:
        prev = moment.get(key)
        moment[key] = t_rep if prev is None else max(prev, t_rep)

    for key, t_rep in truth.items():
        _mark(key, t_rep)
    for key, tick_list in samples.items():
        _mark(key, tick_list[0].t_sample_s)
    for key, pair_list in ticks.items():
        _mark(key, pair_list[0][0])
    for key, pair_list in actuations.items():
        _mark(key, pair_list[0][1])
    for key in filter_keys:
        _mark(key, key * _NS)

    feed = GuidanceInput()
    pending: dict = {}
    est_n = cmd_n = truth_n = exec_n = 0

    def integrate_to(t_to: float, t_from: float) -> None:
        nonlocal r_c, v_c, r_d, v_d
        if t_to <= t_from:
            return
        traj = propagate_pair(r_c, v_c, r_d, v_d, [0.0, t_to - t_from], frame=frame, mu=mu,
                              j2=j2, sub_dt=sub_dt_s, where="gnc.simulate")
        r_c = traj.chief_r[0, -1, :].copy()
        v_c = traj.chief_v[0, -1, :].copy()
        r_d = traj.deputy_r[0, -1, :].copy()
        v_d = traj.deputy_v[0, -1, :].copy()

    t_prev = 0.0
    for key in sorted(moment):
        t_event = moment[key]
        integrate_to(t_event, t_prev)
        t_prev = t_event

        # 1. truth at dynamics-grid instants
        if key in truth:
            t_grid = truth[key]
            truth_n += 1
            for sc_id, (rr, vv) in ((target_id, (r_c, v_c)), (chaser_id, (r_d, v_d))):
                log.append_truth(TruthSample(
                    record_id=f"truth-{truth_n:05d}-{sc_id}", spacecraft_id=sc_id,
                    t_s=t_grid, epoch=clock.epoch_at(t_grid), frame=frame,
                    r_m=tuple(float(x) for x in rr), v_m_s=tuple(float(x) for x in vv)))

        # 3. sensors: the scheduled tick, minus declared drops
        for tick in samples.get(key, ()):
            if tick.dropped:
                continue
            nz = noise[tick.sensor_id]
            rng = np.random.default_rng([nz.seed, int(tick.tick)])
            measured = np.stack((r_d, v_d)) - np.stack((r_c, v_c))
            measured[0] += rng.normal(0.0, nz.std_position_m, 3)
            measured[1] += rng.normal(0.0, nz.std_velocity_m_s, 3)
            log.append_observation(Observation(
                record_id=f"obs-{tick.sensor_id}-{tick.tick:05d}", sensor_id=tick.sensor_id,
                t_sample_s=tick.t_sample_s, t_available_s=tick.t_available_s,
                latency_s=clock.sensor(tick.sensor_id).latency_s,
                relative_state=_pair(target_id, chaser_id,
                                     clock.epoch_at(tick.t_sample_s), frame,
                                     measured[0], measured[1]),
                covariance_position_m2=tuple(float(x) for x in
                                             np.eye(3).reshape(9) * nz.std_position_m ** 2),
                covariance_velocity_m2_s2=tuple(float(x) for x in
                                                np.eye(3).reshape(9) * nz.std_velocity_m_s ** 2)))

        # 4. actuations: what the actuator really delivered
        if key in actuations:
            for cmd in pending.pop(key, ()):
                if float(np.linalg.norm(cmd.delta_v_eci_m_s)) > 0.0:
                    rng = np.random.default_rng([int(execution_seed), exec_n])
                    scale = execution_scale + rng.normal(0.0, execution_rtol, 3)
                    realized = tuple(float(c) * float(s) for c, s in
                                     zip(cmd.delta_v_eci_m_s, scale, strict=True))
                    executed = True
                else:
                    realized = (0.0, 0.0, 0.0)
                    executed = False
                exec_n += 1
                log.append_control_event(ControlEvent(
                    record_id=f"cev-{cmd.record_id}", actuator_id=cmd.actuator_id,
                    t_exec_s=cmd.t_execute_s, executed=executed,
                    commanded_delta_v_m_s=cmd.delta_v_eci_m_s,
                    realized_delta_v_m_s=realized, source_command_id=cmd.record_id))
                if executed:
                    v_d = v_d + np.asarray(realized, dtype=np.float64)

        # 5. filter update: latest available measurement per sensor, trivial
        if key in filter_keys:
            newest_by_sensor: dict = {}
            for o in log.records["observation"]:
                if o.t_available_s <= t_event + 1.0e-9:
                    cur = newest_by_sensor.get(o.sensor_id)
                    if cur is None or o.t_sample_s > cur.t_sample_s:
                        newest_by_sensor[o.sensor_id] = o
            if newest_by_sensor:
                srcs = [newest_by_sensor[s] for s in sorted(newest_by_sensor)]
                newest = max(srcs, key=lambda o: o.t_sample_s)
                rs = newest.relative_state
                cov6 = [0.0] * 36
                cp = newest.covariance_position_m2
                cv = newest.covariance_velocity_m2_s2
                for i in range(3):
                    cov6[i * 6 + i] = cp[i * 3 + i]
                    cov6[(i + 3) * 6 + (i + 3)] = cv[i * 3 + i] if cv else 0.0
                est_n += 1
                est = Estimate(
                    record_id=f"est-{est_n:05d}", estimator_id=clock.estimator.estimator_id,
                    t_update_s=t_event,
                    relative_state=_pair(rs.target_id, rs.chaser_id, clock.epoch_at(t_event),
                                         rs.frame, rs.r_rel_m, rs.v_rel_m_s),
                    covariance_rel_m2=tuple(cov6),
                    source_observation_ids=tuple(o.record_id for o in srcs))
                log.append_estimate(est)
                feed.push(est)

        # 6. control tick: through the feed, the only view guidance has
        if key in ticks:
            t_tick, t_exec = ticks[key][0]
            for cid in sorted(laws):
                try:
                    est, dv = run_guidance_step(laws[cid], feed, cid, target_id)
                except GNCContractError:
                    continue  # no estimate yet: an honest absent tick, not a fake zero
                cmd_n += 1
                cmd = Command(record_id=f"cmd-{cmd_n:05d}", actuator_id=clock.controller.actuator_id,
                              t_tick_s=t_tick, t_execute_s=t_exec, delta_v_eci_m_s=dv,
                              source_estimate_id=est.record_id)
                log.append_command(cmd)
                pending.setdefault(round(t_exec / _NS), []).append(cmd)
    return log
