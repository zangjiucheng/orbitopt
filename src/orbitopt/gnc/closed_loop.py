"""The general closed-loop simulation: Truth -> Sensor -> Navigation ->
Guidance -> Command -> Execution error -> Truth (E13-04).

:func:`simulate_closed_loop` composes the E13 modules into one honest loop
and seals the result as a :class:`~orbitopt.gnc.recording.GncRunLog`:

* **Truth** -- an injectable plant (a callable advancing the 12-vector
  ``(r_chief, r_chaser, v_chief, v_chaser)`` between two instants; default:
  :func:`orbitopt.relative.propagation.propagate_pair`, the E12-02 model),
  sampled on the clock's dynamics grid into ``TruthSample`` records;
* **Sensor** -- any E13-03 :class:`~orbitopt.gnc.sensors.RelativeSensorModel`
  measures the truth relative state at its declared ticks (drops honoured);
* **Navigation** -- any E13-02 :class:`~orbitopt.gnc.ekf.RelativeNavEKF`
  predicts across each interval (executed impulses enter as LVLH controls at
  their fire instants), consumes every observation that has arrived, and
  publishes an ``Estimate`` on the chief reference the simulator (not the
  law) holds;
* **Guidance** -- driven through :func:`orbitopt.gnc.feed.run_guidance_step`
  exactly as in :func:`orbitopt.gnc.sim.simulate`: the feed is the law's only
  view, and a tick without an estimate is an honest absence, not a zero;
* **Command / Execution error** -- an injectable actuator
  ``f(Command) -> realized delta-v`` (default: the commanded vector, exact);
  the ControlEvent records what the plant then actually felt.

The simultaneity rule is the one ``sim.py`` records -- integrate, truth,
sensors, actuations, filter, control tick -- and every instant comes from the
declared clock. The offline readers (:func:`tracking_error`,
:func:`delta_v_ledger`, :func:`convergence_block`) turn a sealed log into
the deviation assessment this task prices; ``convergence_block`` is what the
E13-07 verifier expects under ``config["convergence"]``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from orbitopt.mission.multispacecraft import RelativeState
from orbitopt.relative.frames import eci_to_lvlh
from orbitopt.relative.propagation import EARTH_MU, propagate_pair

from .clock import ClockSpec
from .contracts import Command, ControlEvent, GNCContractError, TruthSample
from .ekf import RelativeNavEKF, Stacked
from .feed import GuidanceInput, GuidanceLaw, run_guidance_step
from .recording import GncRunLog
from .sensors import RelativeSensorModel

_NS = 1.0e-9  # event-key resolution, same as gnc.sim

#: Plant: advance the 12-vector (r_chief, r_chaser, v_chief, v_chaser) [m,
#: m/s, inertial frame of the run] from ``t_from`` to ``t_to``.
Plant = Callable[[np.ndarray, float, float], np.ndarray]

#: Actuator: one Command in, the realized inertial delta-v out (3-tuple).
Actuator = Callable[[Command], tuple]


def default_plant(*, frame: str, mu: float = EARTH_MU, j2: float = 0.0,
                  sub_dt_s: float | None = None) -> Plant:
    """The default plant: E12-02's two-spacecraft RK4 (``propagate_pair``)."""

    def advance(y: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
        if t_to <= t_from:
            return y
        traj = propagate_pair(y[0:3], y[6:9], y[3:6], y[9:12], [0.0, t_to - t_from],
                              frame=frame, mu=mu, j2=j2, sub_dt=sub_dt_s,
                              where="gnc.closed_loop")
        out = y.copy()
        out[0:3] = traj.chief_r[0, -1, :]
        out[3:6] = traj.deputy_r[0, -1, :]
        out[6:9] = traj.chief_v[0, -1, :]
        out[9:12] = traj.deputy_v[0, -1, :]
        return out

    return advance


def default_actuator(*, rtol: float = 0.0, seed: int = 0) -> Actuator:
    """Commanded delta-v times 1 + N(0, rtol) per axis, seeded by (seed, fire
    count) -- the E12-10-style scalar execution error. Campaigns replace it
    with the campaign's own execution model."""
    if not (math.isfinite(float(rtol)) and float(rtol) >= 0.0):
        raise ValueError(f"rtol must be >= 0, got {rtol!r}")
    state = {"n": 0}

    def act(cmd: Command) -> tuple:
        realized = tuple(float(x) for x in cmd.delta_v_eci_m_s)
        if rtol > 0.0 and any(x != 0.0 for x in realized):
            rng = np.random.default_rng([int(seed), int(state["n"])])
            realized = tuple(float(x) * float(s) for x, s in
                             zip(realized, rng.normal(1.0, rtol, 3), strict=True))
        state["n"] += 1
        return realized

    return act


@dataclass
class _ChiefHistory:
    """Truth chief states on the event grid, with linear interpolation --
    the simulator-side authority the Observation/Estimate frame conversions
    use (it lives with the plant; laws never see it)."""

    times: list = field(default_factory=list)
    states: list = field(default_factory=list)

    def push(self, t_s: float, r, v) -> None:
        self.times.append(float(t_s))
        self.states.append(np.concatenate((np.asarray(r, dtype=np.float64),
                                           np.asarray(v, dtype=np.float64))))

    def state_at(self, t_s: float) -> tuple:
        t = float(t_s)
        ts = np.asarray(self.times)
        if t <= ts[0]:
            s = self.states[0]
        elif t >= ts[-1]:
            s = self.states[-1]
        else:
            i = int(np.searchsorted(ts, t)) - 1
            w = (t - ts[i]) / (ts[i + 1] - ts[i])
            s = (1.0 - w) * self.states[i] + w * self.states[i + 1]
        return s[:3].copy(), s[3:].copy()


def simulate_closed_loop(*, run_id: str, clock: ClockSpec, frame: str,
                         target_id: str, chaser_id: str,
                         chief_r_m, chief_v_m_s, deputy_r_m, deputy_v_m_s,
                         sensors: Mapping[str, RelativeSensorModel],
                         measurement_models: Mapping[str, object],
                         filter: RelativeNavEKF,
                         laws: Mapping[str, GuidanceLaw],
                         t_end_s: float,
                         plant: Plant | None = None,
                         actuator: Actuator | None = None,
                         mission: Mapping | None = None,
                         plan_digest: str | None = None,
                         config_extra: Mapping | None = None,
                         observation_hook: Callable | None = None) -> GncRunLog:
    """Fly one closed-loop run and return the sealed run log.

    ``sensors`` maps clock sensor id -> E13-03 sensor model and must cover
    exactly the clock's declared sensors; ``measurement_models`` maps the
    same ids to the E13-02 measurement model the filter consumes for that
    sensor. ``laws`` maps chaser id -> GuidanceLaw. The plant defaults to the
    two-body + J2 propagator (E12-02); a campaign injects its own. The
    actuator defaults to delivering exactly what was commanded.
    ``config_extra`` is merged into the log config (campaign provenance).
    ``observation_hook``, when given, is applied to each arrived
    ``Observation`` *before* logging and filtering -- the environment seam
    for injecting a declared navigation fault (a sensor bias switching on);
    the sealed log records what the filter really consumed.
    """
    declared = {s.sensor_id for s in clock.sensors}
    if set(sensors) != declared:
        raise ValueError(f"simulate_closed_loop(): sensors must cover exactly the clock's "
                         f"{sorted(declared)}, got {sorted(sensors)}")
    if set(measurement_models) != declared:
        raise ValueError(f"simulate_closed_loop(): measurement_models must cover exactly "
                         f"{sorted(declared)}, got {sorted(measurement_models)}")
    if not laws:
        raise ValueError("simulate_closed_loop(): at least one chaser needs a guidance law")
    if not isinstance(filter, RelativeNavEKF):
        raise TypeError(f"filter must be a RelativeNavEKF, got {filter!r}")
    if not (math.isfinite(float(t_end_s)) and float(t_end_s) > 0.0):
        raise ValueError(f"t_end_s must be > 0, got {t_end_s!r}")
    if observation_hook is not None and not callable(observation_hook):
        raise ValueError(f"observation_hook must be callable or None, got {observation_hook!r}")
    plant_default = plant is None
    if plant_default:
        plant = default_plant(frame=frame)

    y = np.concatenate((np.asarray(chief_r_m, dtype=np.float64),
                        np.asarray(deputy_r_m, dtype=np.float64),
                        np.asarray(chief_v_m_s, dtype=np.float64),
                        np.asarray(deputy_v_m_s, dtype=np.float64)))
    if not np.all(np.isfinite(y)):
        raise ValueError("simulate_closed_loop(): initial states must be finite")

    config: dict = {
        "frame": frame, "target_id": target_id, "chaser_id": chaser_id,
        "t_end_s": float(t_end_s),
        "event_priority": ["plant", "truth", "observation", "control_event",
                           "estimate", "command"],
        "sensors": {sid: {"kind": s.kind, "period_s": s.period_s, "phase_s": s.phase_s,
                          "latency_s": s.latency_s, "seed": int(s.seed),
                          "bias_covariance_mode": s.bias_covariance_mode}
                    for sid, s in sorted(sensors.items())},
        "filter": {"estimator_id": clock.estimator.estimator_id,
                   "period_s": clock.estimator.period_s,
                   "process": type(filter.process).__name__,
                   "q_accel_m2_s3": float(filter.q)},
        "actuator": "commanded-exact" if actuator is None else "declared",
        "plant": "default-propagate_pair" if plant_default else "injected",
    }
    if config_extra:
        config.update(dict(config_extra))
    log = GncRunLog.new(run_id=run_id, clock=clock, config=config, mission=mission,
                        plan_digest=plan_digest)

    # ---- event timeline, every instant straight from the clock -------------
    truth: dict = {round(t / _NS): t for t in clock.dynamics_times(t_end_s)}
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
    chief = _ChiefHistory()
    chief.push(0.0, y[0:3], y[6:9])
    pending: dict = {}
    arrived: dict = {}   # sensor_id -> newest unconsumed Observation (None = consumed)
    fired: list = []     # (t_exec, realized delta-v in LVLH) since the last predict
    est_n = cmd_n = truth_n = 0

    def integrate_to(t_to: float, t_from: float) -> None:
        nonlocal y
        if t_to <= t_from:
            return
        y = plant(y, t_from, t_to)
        chief.push(t_to, y[0:3], y[6:9])

    def realized_lvlh(t_exec: float, realized_eci) -> np.ndarray:
        cr, cv = chief.state_at(t_exec)
        _rho, rho_dot = eci_to_lvlh(cr, cv, np.zeros(3),
                                     np.asarray(realized_eci, dtype=np.float64), frame,
                                     where="closed_loop actuation control input")
        return rho_dot  # an impulse on velocity: its LVLH part is the state jump

    def actuate_at(key: int) -> None:
        nonlocal y
        for cmd in pending.pop(key, ()):
            realized = tuple(float(x) for x in actuator(cmd)) if actuator is not None \
                else tuple(float(x) for x in cmd.delta_v_eci_m_s)
            executed = any(x != 0.0 for x in realized)
            log.append_control_event(ControlEvent(
                record_id=f"cev-{cmd.record_id}", actuator_id=cmd.actuator_id,
                t_exec_s=cmd.t_execute_s, executed=executed,
                commanded_delta_v_m_s=cmd.delta_v_eci_m_s,
                realized_delta_v_m_s=realized, source_command_id=cmd.record_id))
            if executed:
                y[9:12] += np.asarray(realized, dtype=np.float64)
                fired.append((cmd.t_execute_s,
                              realized_lvlh(cmd.t_execute_s, realized),
                              str(cmd.metadata.get("phase", ""))))

    t_prev = 0.0
    for key in sorted(moment):
        t_event = moment[key]
        integrate_to(t_event, t_prev)
        t_prev = t_event

        # 1. truth on the dynamics grid
        if key in truth:
            t_grid = truth[key]
            truth_n += 1
            for sc_id, (rr, vv) in ((target_id, (y[0:3], y[6:9])),
                                    (chaser_id, (y[3:6], y[9:12]))):
                log.append_truth(TruthSample(
                    record_id=f"truth-{truth_n:05d}-{sc_id}", spacecraft_id=sc_id,
                    t_s=t_grid, epoch=clock.epoch_at(t_grid), frame=frame,
                    r_m=tuple(float(x) for x in rr), v_m_s=tuple(float(x) for x in vv)))

        # 3. sensors measure the truth relative state (inertial, as the contract
        #    carries it); a dropped frame is the absence of a record
        for tick in samples.get(key, ()):
            if tick.dropped:
                continue
            model = sensors[tick.sensor_id]
            obs = model.observe(
                tick,
                RelativeState(target_id=target_id, chaser_id=chaser_id,
                              epoch=clock.epoch_at(tick.t_sample_s), frame=frame,
                              r_rel_m=tuple(float(x) for x in y[3:6] - y[0:3]),
                              v_rel_m_s=tuple(float(x) for x in y[9:12] - y[6:9])),
                target_id=target_id, chaser_id=chaser_id,
                epoch=clock.epoch_at(tick.t_sample_s))
            if obs is not None and observation_hook is not None:
                obs = observation_hook(obs)
            if obs is not None:
                log.append_observation(obs)
                arrived[tick.sensor_id] = obs

        # 4. actuations: the actuator decides what the plant really feels
        actuate_at(key)

        # 5. filter update: predict across the interval -- executed impulses
        #    enter at their instants -- then consume every arrived observation
        if key in filter_keys:
            due = [f for f in fired if f[0] <= t_event + 1.0e-6]
            fired = [f for f in fired if f[0] > t_event + 1.0e-6]
            prev = filter.t_s
            for j, (t_ex, u_lvlh, _phase) in enumerate(due):
                nxt = due[j + 1][0] if j + 1 < len(due) else t_event
                if t_ex > prev:
                    filter.predict(t_ex - prev, u=None)
                filter.predict(nxt - t_ex, u=u_lvlh)
                prev = nxt
            if filter.t_s < t_event:
                filter.predict(t_event - filter.t_s, u=None)
            for sid in sorted(arrived):
                obs = arrived[sid]
                if obs is None or obs.t_available_s > t_event + 1.0e-6:
                    continue
                cr, cv = chief.state_at(obs.t_sample_s)
                model = measurement_models[sid]
                full6 = (isinstance(model, Stacked)
                         and tuple(model.components) == ("rho_x", "rho_y", "rho_z",
                                                         "rho_dot_x", "rho_dot_y", "rho_dot_z"))
                if isinstance(model, Stacked):
                    filter.update_from_observations(
                        {type(m).__name__: obs for m in model.models}, model, cr, cv,
                        **({"compensate_to": t_event} if full6 else {}))
                else:
                    filter.update_from_observation(obs, model, cr, cv)
                arrived[sid] = None  # consumed once: no double-counting
            if filter.source_observation_ids:
                est_n += 1
                cr, cv = chief.state_at(t_event)
                est = filter.make_estimate(
                    record_id=f"est-{est_n:05d}", estimator_id=clock.estimator.estimator_id,
                    epoch=clock.epoch_at(t_event), target_id=target_id, chaser_id=chaser_id,
                    inertial_frame=frame, chief_r=cr, chief_v=cv)
                log.append_estimate(est)
                feed.push(est)

        # 6. control tick: through the feed -- the law's only view
        if key in ticks:
            t_tick, t_exec = ticks[key][0]
            for cid in sorted(laws):
                try:
                    est, dv = run_guidance_step(laws[cid], feed, cid, target_id)
                except GNCContractError:
                    continue  # no estimate yet: an honest absent tick
                cmd_n += 1
                phase = getattr(laws[cid], "phase_for", None)
                cmd = Command(record_id=f"cmd-{cmd_n:05d}",
                              actuator_id=clock.controller.actuator_id,
                              t_tick_s=t_tick, t_execute_s=t_exec, delta_v_eci_m_s=dv,
                              source_estimate_id=est.record_id,
                              metadata=({"phase": phase(est)} if callable(phase) else {}))
                log.append_command(cmd)
                pending.setdefault(round(t_exec / _NS), []).append(cmd)
            if ticks[key][0][1] <= ticks[key][0][0] + 1.0e-9:
                actuate_at(key)  # zero-latency controller: fire at the tick itself
    log.config["final_truth"] = {"t_s": float(t_prev), "y12": [float(x) for x in y]}
    log.config["convergence"] = convergence_block(log)
    return log


# ---------------------------------------------------------------------------
# offline assessment (reads the sealed log; nothing guidance-side does this)
# ---------------------------------------------------------------------------


def _truth_pairs(log: GncRunLog):
    """Truth-grid times with chief/chaser records aligned on both ids."""
    chaser = log.config["chaser_id"]
    target = log.config["target_id"]
    by_t: dict = {}
    for r in log.records["truth"]:
        by_t.setdefault(r.t_s, {})[r.spacecraft_id] = r
    out = []
    for t in sorted(by_t):
        row = by_t[t]
        if chaser in row and target in row:
            out.append((t, row[target], row[chaser]))
    return out


def tracking_error(log: GncRunLog, *, settle_after_s: float = 0.0) -> dict:
    """Estimate-minus-truth, sampled on the truth grid against the newest
    estimate at or before each epoch (never a future one): position/velocity
    RMSE (after an optional settling window) and maxima. Reads truth -- an
    offline evaluator only; guidance-side code never calls this."""
    est_t = np.array([e.t_update_s for e in log.records["estimate"]], dtype=np.float64)
    if len(est_t) == 0:
        raise GNCContractError("tracking_error: the log carries no estimates")
    cadence = float(np.median(np.diff(est_t))) if len(est_t) > 1 else 0.0
    pos, vel = [], []
    for t, tgt, chs in _truth_pairs(log):
        if t < settle_after_s:
            continue
        i = int(np.searchsorted(est_t, t + 1.0e-9)) - 1
        if i < 0 or (cadence > 0.0 and t - est_t[i] > 3.0 * cadence):
            continue
        e = log.records["estimate"][i]
        d = np.subtract(chs.r_m, tgt.r_m) - np.asarray(e.relative_state.r_rel_m, dtype=np.float64)
        dv = np.subtract(chs.v_m_s, tgt.v_m_s) - np.asarray(e.relative_state.v_rel_m_s,
                                                            dtype=np.float64)
        pos.append(float(np.linalg.norm(d)))
        vel.append(float(np.linalg.norm(dv)))
    p = np.asarray(pos)
    v = np.asarray(vel)

    def _rms(a):
        return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")

    return {"n_epochs": int(len(p)), "rmse_pos_m": _rms(p), "rmse_vel_m_s": _rms(v),
            "max_pos_m": float(p.max()) if len(p) else float("nan"),
            "max_vel_m_s": float(v.max()) if len(v) else float("nan")}


def delta_v_ledger(log: GncRunLog, *, budget_m_s: float | None = None) -> dict:
    """The fuel story from the log alone: commanded per phase, realized per
    phase, the commanded-minus-realized execution gap, and the margin a
    declared budget leaves."""
    cmd_by_phase: dict = {}
    real_by_phase: dict = {}
    gap_by_phase: dict = {}
    cmd_total = real_total = 0.0
    ev_by_cmd = {e.source_command_id: e for e in log.records["control_event"]}
    for c in log.records["command"]:
        mag = float(np.linalg.norm(c.delta_v_eci_m_s))
        ev = ev_by_cmd.get(c.record_id)
        rmag = float(np.linalg.norm(ev.realized_delta_v_m_s)) if ev is not None else 0.0
        phase = str(c.metadata.get("phase", "unphased"))
        cmd_by_phase[phase] = cmd_by_phase.get(phase, 0.0) + mag
        real_by_phase[phase] = real_by_phase.get(phase, 0.0) + rmag
        gap_by_phase[phase] = gap_by_phase.get(phase, 0.0) + abs(rmag - mag)
        cmd_total += mag
        real_total += rmag
    out = {"commanded_by_phase_m_s": cmd_by_phase, "realized_by_phase_m_s": real_by_phase,
           "execution_gap_by_phase_m_s": gap_by_phase,
           "commanded_total_m_s": cmd_total, "realized_total_m_s": real_total}
    if budget_m_s is not None:
        out["budget_m_s"] = float(budget_m_s)
        out["margin_m_s"] = float(budget_m_s) - real_total
    return out


def convergence_block(log: GncRunLog, *, settle_window_frac: float = 0.25) -> dict:
    """Producer-side settle/convergence evidence: every executed burn is
    complete and the command profile over the window after the final
    executed burn is zero. This is the block the E13-07 verifier reads under
    ``config[\"convergence\"][\"converged\"]``."""
    events = [e for e in log.records["control_event"] if e.executed]
    commands = log.records["command"]
    if not commands:
        return {"converged": False, "reason": "no command was ever issued",
                "n_executed_burns": 0, "settle_window_s": None,
                "max_command_dv_after_last_burn_m_s": None}
    t_last = max(e.t_exec_s for e in events) if events else None
    tail = [c for c in commands if t_last is None or c.t_tick_s > t_last + 1.0e-9]
    max_tail = max((float(np.linalg.norm(c.delta_v_eci_m_s)) for c in tail), default=0.0)
    span = max(c.t_tick_s for c in commands) - min(c.t_tick_s for c in commands)
    converged = bool(events) and max_tail <= 1.0e-9 and \
        t_last >= float(log.config["t_end_s"]) - settle_window_frac * span
    return {"converged": bool(converged),
            "reason": ("all scheduled burns executed and the command profile settled after the "
                       "last one" if converged else
                       "final burn incomplete or the command profile never settled"),
            "n_executed_burns": len(events),
            "settle_window_s": settle_window_frac * span,
            "max_command_dv_after_last_burn_m_s": max_tail}
