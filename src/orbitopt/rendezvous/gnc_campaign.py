"""Closed-loop campaign runner (E13-04): fly an E12-10 sample with feedback.

One function, :func:`fly_closed_loop_sample`, is the closed-loop counterpart
of :func:`orbitopt.rendezvous.robustness.fly_truth_sample`. The sample
identity is *unchanged* -- same cluster/index/seed stream from the same
``DisturbanceModel``, same planning view, same ``plan_sample`` gate and
screen (so a ``planning_infeasible`` sample is infeasible closed-loop too),
same perturbed truth physics (perturbed J2, co-rotating drag), same
execution-error draws for burn 1 and the arrival burn. What changes is the
commanding: the campaign's ground plan is fired as a planned maneuver at the
gate, then the E13-04 loop -- relative GNSS sensor (E13-03), relative-state
EKF (E13-02), CW mid-course law (E13-04) -- re-solves the remaining burns
from the *estimate*, at a mid-course epoch and again a declared lead time
before arrival.

Pairing rules that keep the open/closed comparison honest:

* burn 1 executes with ``draws.burn_factors[0]`` / ``draws.burn_angle_...[0]``
  -- the identical throttle/pointing error the open-loop campaign applied;
* the arrival burn is re-targeted, but executes with the sample's burn-2
  draws -- the same actuator, a different vector;
* the mid-course correction is a NEW burn, so its execution error is drawn
  from ``default_rng([sample seed, 2])`` in the campaign's execution model --
  deterministic, and consuming none of the sample's paired streams;
* passive releases: no-burn and gate-abort start from the same
  burn-free states as the open-loop campaign; the hold-point coast starts
  from where the closed loop actually arrived.

The declared sensing (a cartesian RelNAV-class measurement, 2 m / 3 mm/s
1-sigma, 20 s cadence, 2 s latency, no dropouts, zero bias) is a DESIGN
ASSUMPTION carried with provenance in the run-log config; it is not a
re-declaration of the campaign's perturbations, which stay bit-identical.

E13-05 adds one seam: an optional ``law_factory`` swaps the guidance law
for the sample (sensing, plant, actuator and grading untouched) and gives
any burn phase the law names its own execution draw from slot 101 upward.
The E13-05 arms (LQR, MPC) fly through this hook; the default law keeps the
baseline flight bit-identical.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from orbitopt.gnc.clock import ClockSpec, ControllerSpec, EstimatorSpec
from orbitopt.gnc.closed_loop import (
    delta_v_ledger,
    simulate_closed_loop,
    tracking_error,
)
from orbitopt.gnc.ekf import (
    CwProcess,
    RelativeNavEKF,
    RelativePosition,
    RelativeVelocity,
    Stacked,
)
from orbitopt.gnc.feed import GuidanceLaw
from orbitopt.gnc.guidance import CwMidCourseLaw, ManeuverPlan, ReferenceEphemeris
from orbitopt.gnc.recording import GncRunLog
from orbitopt.gnc.sensors import ChannelErrorConfig, DropoutConfig, RelativeSensorModel
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import (
    MultiSpacecraftMission,
    SpacecraftSpec,
    mission_to_dict,
)
from orbitopt.relative.frames import eci_to_lvlh
from orbitopt.relative.propagation import EARTH_J2, propagate_pair

from . import robustness as rb
from .config import RendezvousMissionConfig
from .planner import RendezvousPlanError, initial_states
from .robustness import (
    FlightOutcome,
    PassiveRelease,
    RobustnessCampaign,
)

#: Provenance of the declared sensing (mandatory on every channel).
SENSOR_SOURCE = ("DESIGN ASSUMPTION (E13-04): dual-constellation relative GNSS / RelNAV "
                 "at LEO rendezvous ranges -- 2 m 1-sigma position, 3 mm/s 1-sigma "
                 "relative velocity, 20 s cadence, 2 s latency, no dropouts, zero bias")

T0_EPOCH = Epoch(9000.0, "TDB")  # bookkeeping epoch; campaign times are seconds from t0


def _mission_doc(cfg: RendezvousMissionConfig) -> dict:
    """The contract-valid mission document for the sealed run log: the
    campaign's chief as the TARGET, the chaser as the CHASER, one shared
    frame and epoch."""
    return mission_to_dict(MultiSpacecraftMission(
        mission_id=f"{cfg.mission_id}-cl", title=f"{cfg.mission_id} closed-loop flight",
        frame=cfg.dynamics.frame, epoch=T0_EPOCH,
        spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")),
        metadata={"kind": "orbitopt.gnc.campaign-sample-1.0"}))


@dataclass(frozen=True)
class ClosedLoopSampleConfig:
    """The closed-loop overlay on the E12-10 campaign: sensing, cadences,
    guidance schedule, filter process noise. Perturbations are NOT here --
    they stay the campaign's, untouched."""

    pos_sigma_m: float = 2.0
    vel_sigma_m_s: float = 0.003
    sensor_period_s: float = 20.0
    sensor_latency_s: float = 2.0
    cadence_s: float = 10.0          # estimator and controller tick
    terminal_lead_s: float = 22.0    # arrival burn actuates in (t_f - 20,
    #   t_f - 10]: the law decides on the estimate (one cadence stale) and the
    #   actuator fires one cadence later, so the schedule needs 2 cadences of
    #   margin, and a velocity-nulling burn parked further out leaves the
    #   closing pass unfinished -- the final burn belongs within a cadence of t_f
    coast_s: float = 60.0            # run continues this long past t_f
    q_accel_m2_s3: float = 3.0e-8    # CW process noise covering the unmodelled
    #   differential J2/drag between planning and truth physics (DESIGN
    #   ASSUMPTION tuned on the drift leg: ~0.15 m/s of velocity growth)
    midcourse_fraction: float = 0.5  # correction at t1 + f * tof
    retarget_cap_m_s: float = 8.0    # terminal nulling burn is the campaign's
    #   arrival dv2 (~5.7 m/s nominal) plus correction margin
    pre_terminal_lead_s: float = 40.0   # position trim (2 cadences ahead of the
    #   terminal burn): fixes the truth-vs-plan offset a 7-s-out velocity burn
    #   cannot (radial error is what the passive hold-point coast drifts
    #   through the corridor)
    pre_terminal_velocity_weight: float = 0.25   # trim: position-focused
    trim_cap_m_s: float = 3.0
    terminal_velocity_weight: float = 100.0  # final burn: null the closing speed
    deadband_m_s: float = 0.01


@dataclass(frozen=True)
class GuidanceContext:
    """Everything a guidance law needs to build itself for one sample, handed
    to the optional ``law_factory`` of :func:`fly_closed_loop_sample` (E13-05):
    the plan and its gate/terminal epochs, the paired draws (for provenance
    only -- a law must never consume them; the actuator owns execution
    error), and the reference ephemeris. The factory replaces the law ONLY;
    sensing, filtering, plant, actuator and grading stay the campaign's."""

    cfg: Any
    campaign: Any
    clc: Any
    sol: Any
    draws: Any
    reference: ReferenceEphemeris
    t1_s: float
    tof_s: float
    t_f_s: float
    dv1_eci_m_s: tuple


def default_guidance_law(ctx: GuidanceContext):
    """The E13-04 baseline law (the schedule ``fly_closed_loop_sample`` used
    to hard-code): planned gate burn, mid-course retarget, position trim,
    arrival nulling."""
    cfg, clc = ctx.cfg, ctx.clc
    dv1 = ctx.dv1_eci_m_s
    return CwMidCourseLaw(
        maneuvers=(
            ManeuverPlan("departure", ctx.t1_s, "planned", dv_planned_eci_m_s=dv1,
                         dv_cap_m_s=max(1.5 * float(np.linalg.norm(dv1)),
                                        float(np.linalg.norm(dv1)) + 1.0)),
            ManeuverPlan("midcourse", ctx.t1_s + clc.midcourse_fraction * ctx.tof_s, "retarget",
                         dv_cap_m_s=clc.retarget_cap_m_s),
            ManeuverPlan("terminal_trim", ctx.t_f_s - clc.pre_terminal_lead_s, "retarget",
                         dv_cap_m_s=clc.trim_cap_m_s,
                         velocity_weight=clc.pre_terminal_velocity_weight),
            ManeuverPlan("arrival", ctx.t_f_s - clc.terminal_lead_s, "retarget",
                         dv_cap_m_s=clc.retarget_cap_m_s,
                         velocity_weight=clc.terminal_velocity_weight)),
        reference=ctx.reference, t_terminal_s=ctx.t_f_s,
        terminal_rho_lvlh_m=cfg.hold_point_rel_lvlh,
        budget_m_s=float(cfg.delta_v_budget_m_s), deadband_m_s=clc.deadband_m_s)


def _sensor(clc: ClosedLoopSampleConfig, seed: int, n_ticks: int):
    chans = tuple(ChannelErrorConfig(c, s, 0.0, SENSOR_SOURCE) for c, s in
                  (("rel_pos_x_m", clc.pos_sigma_m), ("rel_pos_y_m", clc.pos_sigma_m),
                   ("rel_pos_z_m", clc.pos_sigma_m),
                   ("rel_vel_x_m_s", clc.vel_sigma_m_s), ("rel_vel_y_m_s", clc.vel_sigma_m_s),
                   ("rel_vel_z_m_s", clc.vel_sigma_m_s)))
    model = RelativeSensorModel("relnav", "cartesian_relnav", clc.sensor_period_s, 0.0,
                                clc.sensor_latency_s, int(seed), chans,
                                dropout=DropoutConfig(), bias_covariance_mode="none",
                                source=SENSOR_SOURCE)
    return model, model.sensor_spec(max(2, int(n_ticks)))


def _reference(cfg: RendezvousMissionConfig, t_end_s: float, step_s: float = 60.0):
    """Nominal chief on the PLANNING dynamics (two-body + config J2, no
    campaign disturbances, no drag) -- the ephemeris a ground plan uplinks."""
    r0c, v0c, _r0d, _v0d = initial_states(cfg)
    tt = np.arange(0.0, t_end_s + 0.5 * step_s, step_s)
    tt[-1] = max(tt[-1], t_end_s)
    tr = propagate_pair(r0c, v0c, r0c, v0c, list(tt), frame=cfg.dynamics.frame,
                        mu=cfg.dynamics.mu, j2=(EARTH_J2 if cfg.dynamics.j2 else 0.0),
                        where="gnc_campaign: reference ephemeris")
    return ReferenceEphemeris(
        times_s=tuple(float(x) for x in tt),
        chief_r_m=tuple(tr.chief_r[0, i, :] for i in range(len(tt))),
        chief_v_m_s=tuple(tr.chief_v[0, i, :] for i in range(len(tt))),
        frame=cfg.dynamics.frame, mu_m3_s2=float(cfg.dynamics.mu))


def _execution_draws(model: rb.DisturbanceModel, seed: int, burn_slot: int) -> tuple:
    """The execution error of a NEW burn, in the campaign's own model and
    stream style (``default_rng([seed, slot])``, magnitude then rotation
    vector) -- never consuming the sample's paired streams."""
    exe = model.execution
    rng = np.random.default_rng([int(seed), int(burn_slot)])
    if exe.enabled and exe.sigma_magnitude_rel > 0.0:
        mag = float(1.0 + exe.sigma_magnitude_rel * rng.standard_normal())
    else:
        mag = 1.0
    if exe.enabled and exe.sigma_pointing_deg > 0.0:
        sig = math.radians(exe.sigma_pointing_deg) / math.sqrt(3.0)
        ang = tuple(float(sig * rng.standard_normal()) for _ in range(3))
    else:
        ang = (0.0, 0.0, 0.0)
    return mag, ang


def fly_closed_loop_sample(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                           tof_s: float, seed_base: int, cluster: int, index: int, *,
                           clc: ClosedLoopSampleConfig | None = None,
                           run_id: str | None = None,
                           law_factory: "Callable[[GuidanceContext], GuidanceLaw] | None" = None
                           ) -> tuple[FlightOutcome, dict, GncRunLog | None]:
    """Fly one campaign sample closed-loop. Returns ``(FlightOutcome,
    assessment, run_log)`` -- the FlightOutcome grades exactly like the
    campaign's own (same trajectory checks, budget and arrival tolerances,
    passive releases where the campaign grades them), and the assessment
    carries the closed-loop deviations: tracking error, the delta-v ledger by
    phase, refusals, and the settle/convergence evidence. A planning failure
    is the same ``planning_infeasible`` outcome the open-loop campaign
    reports (the plan is the identical ``plan_sample``), with ``None`` log.

    ``law_factory`` (E13-05) replaces the guidance law only: called once per
    sample with the :class:`GuidanceContext`, it returns any ``GuidanceLaw``.
    Default is :func:`default_guidance_law`, the E13-04 schedule, so the
    baseline flight is bit-identical to the pre-hook behaviour. Execution
    error for any burn phase the law names beyond the scheduled four draws
    ``default_rng([seed, slot])`` from slot 101 upward -- still the
    campaign's model, still never a paired stream."""
    clc = clc or ClosedLoopSampleConfig()
    model = campaign.disturbances
    draws = rb.draw_sample(model, int(seed_base), int(cluster), int(index))
    nan = float("nan")
    try:
        sol = rb.plan_sample(cfg, float(tof_s), draws)
    except (RendezvousPlanError, ValueError, np.linalg.LinAlgError) as exc:
        return (FlightOutcome(
            ok=False, failure_class="planning_infeasible", reason=str(exc)[:400],
            position_error_m=nan, relative_speed_m_s=nan, realized_delta_v_m_s=nan,
            planned_delta_v_m_s=nan, min_range_m=nan, min_range_epoch_s=nan,
            gate_epoch_s=nan, tof_s=float(tof_s), times_s=np.empty(0),
            rel_lvlh=np.empty((0, 6))),
            {"planning_error": str(exc)[:400]}, None)

    t1, tof = float(sol["t1"]), float(sol["tof"])
    t_f = t1 + tof
    t_end = t_f + clc.coast_s
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    r0c, v0c, r0d, v0d = initial_states(cfg)
    y0 = np.concatenate((r0c, r0d, v0c, v0d))

    # ---- truth plant: the campaign's physics, advanced between events -----
    drag_on = model.drag if (model.drag is not None and model.drag.enabled
                             and model.is_active("drag")) else None
    j2_truth = (EARTH_J2 * draws.j2_factor) if model.is_active("j2") \
        else (EARTH_J2 if cfg.dynamics.j2 else 0.0)
    dt_s = float(campaign.sample_dt_s)
    period = 2.0 * math.pi * math.sqrt(float(np.linalg.norm(r0c)) ** 3 / mu)
    sub_dt = min(dt_s, period / 500.0)
    snaps: dict = {}

    def deriv(y: np.ndarray) -> np.ndarray:
        return rb._truth_deriv(y, mu, j2_truth, drag_on, draws.drag_factor)

    def plant(y: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
        snaps[float(t_from)] = np.asarray(y, dtype=float).copy()
        if t_to <= t_from:
            return np.asarray(y, dtype=float).copy()
        out = rb._integrate_pair(np.asarray(y, dtype=float),
                                 rb._grid(t_from, t_to, dt_s), deriv,
                                 sub_dt=sub_dt, integrator="rk4")[-1]
        snaps[float(t_to)] = out.copy()
        return out

    # ---- actuator: the campaign's execution model, per phase ---------------
    # departure/arrival reuse the sample's paired burn draws; every NEW burn
    # (midcourse and anything later) gets the campaign's own execution-model
    # stream, default_rng([seed, slot]) -- never touching the paired streams.
    # The slot for a law-invented burn (E13-05 feedback pulses) is a counter
    # from 101, so it can never collide with the scheduled slots 2/3.
    phases = {"departure": (draws.burn_factors[0], draws.burn_angle_vectors_rad[0]),
              "arrival": (draws.burn_factors[1], draws.burn_angle_vectors_rad[1])}
    new_burn_slot = {"midcourse": 2, "terminal_trim": 3}
    slot_state = {"next": 100}

    def actuator(command) -> tuple:
        ph = str(command.metadata.get("phase", ""))
        if ph == "departure":
            mag, ang = phases["departure"]
        elif ph == "arrival":
            mag, ang = phases["arrival"]
        elif ph in new_burn_slot:
            mag, ang = _execution_draws(model, draws.seed, new_burn_slot[ph])
        elif ph and ph != "coast":
            slot_state["next"] += 1
            mag, ang = _execution_draws(model, draws.seed, slot_state["next"])
        else:
            return (0.0, 0.0, 0.0)  # coast: nothing to misfire
        realized = rb._rotate(np.asarray(command.delta_v_eci_m_s, dtype=float) * mag, ang)
        return tuple(float(x) for x in realized)

    # ---- sensing, filter, law ----------------------------------------------
    sensor, sensor_spec = _sensor(clc, draws.seed, math.ceil(t_end / clc.sensor_period_s) + 2)
    clock = ClockSpec(
        t0=T0_EPOCH, dt_dynamics_s=dt_s, sensors=(sensor_spec,),
        estimator=EstimatorSpec(period_s=clc.cadence_s, phase_s=0.0, latency_s=0.0),
        controller=ControllerSpec(period_s=clc.cadence_s, phase_s=0.0, latency_s=0.0))
    ref = _reference(cfg, t_end)
    r0c_b, v0c_b, r0d_b, v0d_b = rb._planning_view(cfg, draws)   # what nav believes
    rho0, rhod0 = eci_to_lvlh(r0c_b, v0c_b, r0d_b - r0c_b, v0d_b - v0c_b, frame,
                              where="gnc_campaign: filter initial belief")
    nav = model.initial_state
    p0_diag = np.array([s * s if s > 0.0 else 1.0e-6 for s in nav.sigma_pos_lvlh_m]
                       + [s * s if s > 0.0 else 1.0e-10 for s in nav.sigma_vel_lvlh_m_s])
    filt = RelativeNavEKF(CwProcess(ref.mean_motion), np.concatenate((rho0, rhod0)),
                          np.diag(p0_diag), q_accel_m2_s3=clc.q_accel_m2_s3)
    dv1 = tuple(float(x) for x in sol["dv1"])
    gctx = GuidanceContext(cfg=cfg, campaign=campaign, clc=clc, sol=sol, draws=draws,
                           reference=ref, t1_s=t1, tof_s=tof, t_f_s=t_f, dv1_eci_m_s=dv1)
    law = default_guidance_law(gctx) if law_factory is None else law_factory(gctx)

    # ---- fly ---------------------------------------------------------------
    hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
    log = simulate_closed_loop(
        run_id=run_id or f"cl-{cfg.mission_id}-{seed_base}-{cluster}-{index}", clock=clock,
        frame=frame, target_id="tgt", chaser_id="chaser",
        chief_r_m=r0c, chief_v_m_s=v0c, deputy_r_m=r0d, deputy_v_m_s=v0d,
        sensors={"relnav": sensor},
        measurement_models={"relnav": Stacked((RelativePosition(), RelativeVelocity()))},
        filter=filt, laws={"chaser": law}, t_end_s=t_end, plant=plant, actuator=actuator,
        mission=_mission_doc(cfg),
        config_extra={
            "closed_loop": {"campaign": campaign.mission_id, "seed_base": int(seed_base),
                            "cluster": int(cluster), "index": int(index), "seed": draws.seed,
                            "sensor": {"pos_sigma_m": clc.pos_sigma_m,
                                        "vel_sigma_m_s": clc.vel_sigma_m_s,
                                        "period_s": clc.sensor_period_s,
                                        "latency_s": clc.sensor_latency_s, "source": SENSOR_SOURCE},
                            "terminal_lead_s": clc.terminal_lead_s,
                            "midcourse_fraction": clc.midcourse_fraction,
                            "retarget_cap_m_s": clc.retarget_cap_m_s,
                            "q_accel_m2_s3": clc.q_accel_m2_s3,
                            "plant": "campaign truth (perturbed j2 + drag, rk4)",
                            "draws": draws.to_dict(), "planned_total_m_s": float(sol["total_m_s"])},
            "convergence_expected": True})

    # ---- grade on the campaign's own contract ------------------------------
    # arrival state at t_f: the last plant snapshot at or before t_f already
    # carries every executed impulse (the runner re-enters the plant at the
    # actuation epoch with the jumped state), so coasting it forward is exact.
    t_g = max(k for k in snaps if k <= t_f + 1.0e-9)
    y_arr = plant(snaps[t_g], t_g, t_f) if t_g < t_f - 1.0e-9 else snaps[t_g]
    gate = plant(y0, 0.0, t1)

    times = list(clock.dynamics_times(t_f))
    if times[-1] < t_f - 1.0e-9:
        times.append(t_f)
    by_t = {r.t_s: r for r in log.records["truth"] if r.spacecraft_id == "chaser"}
    tgt = {r.t_s: r for r in log.records["truth"] if r.spacecraft_id == "tgt"}
    rows = []
    for t in times:
        if t in by_t and t in tgt:
            rr_d, rr_c = np.asarray(by_t[t].r_m), np.asarray(tgt[t].r_m)
            vv_d, vv_c = np.asarray(by_t[t].v_m_s), np.asarray(tgt[t].v_m_s)
        else:
            rr_c, rr_d, vv_c, vv_d = y_arr[0:3], y_arr[3:6], y_arr[6:9], y_arr[9:12]
        rho, rho_dot = eci_to_lvlh(rr_c, vv_c, rr_d - rr_c, vv_d - vv_c, frame,
                                   where="gnc_campaign: flown rel_lvlh")
        rows.append((t, rho, rho_dot))
    times = np.array([r[0] for r in rows])
    rel = np.hstack((np.vstack([r[1] for r in rows]), np.vstack([r[2] for r in rows])))
    if not (np.all(np.isfinite(rel)) and np.all(np.isfinite(y_arr))):
        outcome = FlightOutcome(
            ok=True, failure_class="other", reason="closed-loop truth went non-finite",
            position_error_m=nan, relative_speed_m_s=nan, realized_delta_v_m_s=nan,
            planned_delta_v_m_s=float(sol["total_m_s"]), min_range_m=nan,
            min_range_epoch_s=nan, gate_epoch_s=t1, tof_s=tof, times_s=times,
            rel_lvlh=rel)
        return outcome, {"non_finite": True}, log

    ledger = delta_v_ledger(log, budget_m_s=float(cfg.delta_v_budget_m_s))
    tracking = tracking_error(log, settle_after_s=0.0)
    passive: tuple[PassiveRelease, ...] = ()
    if campaign.passive_contingency_in_success and cfg.safety is not None:
        passive = rb._truth_passive_releases(
            cfg, deriv, sub_dt=sub_dt, integrator="rk4", dt_s=dt_s, t1=t1, tof=tof,
            initial_pair=y0, gate_pair=gate,
            arrival_pair=y_arr)
    grade = rb.grade_flight(cfg, times, rel,
                            realized_delta_v_m_s=ledger["realized_total_m_s"],
                            planned_delta_v_m_s=ledger["commanded_total_m_s"],
                            passive=passive)
    rng = np.linalg.norm(rel[:, :3], axis=1)
    i = int(np.argmin(rng))
    outcome = FlightOutcome(
        ok=True, failure_class=grade["failure_class"], reason=grade["reason"],
        position_error_m=float(np.linalg.norm(rel[-1, :3] - hold)),
        relative_speed_m_s=float(np.linalg.norm(rel[-1, 3:])),
        realized_delta_v_m_s=ledger["realized_total_m_s"],
        planned_delta_v_m_s=ledger["commanded_total_m_s"],
        min_range_m=float(rng[i]), min_range_epoch_s=float(times[i]),
        gate_epoch_s=t1, tof_s=tof, times_s=times, rel_lvlh=rel, passive=passive)

    assessment = {"draws": draws.to_dict(), "t1_s": t1, "tof_s": tof,
                  "t_f_s": t_f, "grade": grade, "ledger": ledger, "tracking": tracking,
                  "convergence": dict(log.config["convergence"]),
                  "law_refusals": list(law.refusals),
                  "law_solve_time": dict(getattr(law, "solve_time_stats", {}) or {}),
                  "n_observations": len(log.records["observation"]),
                  "n_estimates": len(log.records["estimate"])}
    assessment["attribution"] = attribute_failure(outcome, assessment, cfg)
    return outcome, assessment, log


# ---------------------------------------------------------------------------
# failure attribution: navigation / guidance / execution / safety constraint
# ---------------------------------------------------------------------------

#: DESIGN ASSUMPTION thresholds (E13-04), used by :func:`attribute_failure`.
#: A terminal miss is charged to NAVIGATION when the estimate was already
#: wrong by more than this at any logged epoch; to EXECUTION when the
#: commanded-minus-realized gap, propagated through the run with the CW
#: position-drift gain below, could by itself account for the miss; to
#: GUIDANCE otherwise (navigation knew, the actuator delivered, the law
#: still aimed wrong). Safety and planning keep the mission's own precedence.
NAV_CHARGE_POS_M = 20.0
DRIFT_GAIN_M_PER_M_S = 6.0  # 1.5 * n * tof at LEO-RDV-004 geometry (n=0.00108, tof=3417 s)


def attribute_failure(outcome: FlightOutcome, assessment: Mapping[str, Any],
                      cfg: RendezvousMissionConfig) -> str | None:
    """One attribution class per outcome, from the run-log evidence only:
    ``navigation`` / ``guidance`` / ``execution`` / ``safety_constraint`` (plus
    ``planning`` and ``propagation`` for the two structural outcomes, and
    ``None`` on success). The mission classes in ``outcome.failure_class``
    stay the paired primary metric; this splits their causes."""
    cls = outcome.failure_class
    if cls is None:
        return None
    if cls == "planning_infeasible":
        return "planning"
    if cls == "other":
        return "propagation"
    if cls == "safety_violation":
        return "safety_constraint"
    ledger = assessment["ledger"]
    gap = float(sum(ledger["execution_gap_by_phase_m_s"].values()))
    if cls == "fuel_over_budget":
        return "guidance" if ledger["commanded_total_m_s"] > float(cfg.delta_v_budget_m_s) \
            else "execution"
    if float(assessment["tracking"].get("max_pos_m", 0.0)) > NAV_CHARGE_POS_M:
        return "navigation"
    if gap * DRIFT_GAIN_M_PER_M_S >= outcome.position_error_m:
        return "execution"
    return "guidance"
