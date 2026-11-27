"""Constrained-control research runner (E13-05): LQR / MPC vs the E13-04 law.

Thin campaign-side harness over :func:`orbitopt.rendezvous.gnc_campaign.fly_closed_loop_sample`'s
``law_factory`` hook. Nothing here redefines a mission quantity: the safety
families, budget and tolerances are read from the same mission config the
campaign freezes, the perturbations and execution draws are the campaign's,
and the flight is graded by the campaign's own contract. What the harness
adds per flight is the E13-05 research record:

* **arm identity** — ``baseline`` (the E13-04 schedule, flown through the
  identical code path with the default law), ``lqr``, or ``mpc``;
* **realized constraint violations** — the four families of
  :func:`orbitopt.rendezvous.safety.evaluate_trajectory_safety` evaluated on
  the FLOWN trajectory from the gate onward (the drift leg before the gate is
  the identical open-loop coast in every arm and is excluded; the sampling is
  the campaign truth grid, ``sample_dt_s``, coarser than the plan-verification
  grid — DESIGN NOTE, applied uniformly to all arms);
* **decision-compute accounting** — per-decision wall time inside the law
  (mean / p95 / max / total), refusals, and for MPC the solver-fallback count;
* an optional **stress cell**: law-side constraint overrides (tighter
  keep-out shell / cross-track corridor) evaluated against a matching stress
  safety set, for the question "what happens when the constraint binds" —
  the frozen mission safety set itself is never edited;
* **reference**: the arms do not regulate to a point — they track the
  sample's own planned transfer arc (:func:`_planned_arc`, the uplink a
  chaser flying this plan with feedback would carry), from the tick after
  departure to the trim lead, with the mission's own terminal pair
  (terminal_trim + arrival: same epochs, caps and weightings the baseline's
  ``clc`` schedule carries) fired by the law itself — the closing-speed
  arrest is a finite-maneuver job, not a tick's, and no arm gets to skip
  it. Point regulation from the gate would make
  the arms fight the mission's CW drift phasing, which is a property of the
  reference choice, not of LQR or MPC; tracking the arc keeps the pairing
  against the baseline's scheduled burns honest.

The paired layout (same clusters, same indices, same seed base, same
``draw_sample`` streams) is the benchmark's job; this module just flies one
sample under one arm.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from orbitopt.gnc.constraints import ApproachConstraints
from orbitopt.gnc.guidance import ManeuverPlan
from orbitopt.gnc.lqr import CwLqrLaw
from orbitopt.gnc.mpc import CwMpcLaw
from orbitopt.gnc.recording import GncRunLog
from orbitopt.relative.frames import eci_to_lvlh
from orbitopt.relative.propagation import EARTH_J2, propagate_pair

from .config import RendezvousMissionConfig, SafetyCfg
from .gnc_campaign import (
    ClosedLoopSampleConfig,
    GuidanceContext,
    fly_closed_loop_sample,
)
from .robustness import FlightOutcome, RobustnessCampaign
from .safety import evaluate_trajectory_safety

ARMS = ("baseline", "lqr", "mpc")


@dataclass(frozen=True)
class ConstrainedControlConfig:
    """Law-side tuning for the E13-05 arms. Weights are stated as tracking
    scales: position error of ``pos_scale_m`` and rate error of
    ``vel_scale_m_s`` each cost 1 per squared unit in the stage cost, the
    terminal cost carries ``terminal_gain`` times the stage weights (the
    arrival tolerances 50 m / 0.2 m/s set the scales; the terminal gain makes
    the final ticks pay for residual state, which is what replaces the
    baseline's scheduled arrival burn). ``u_max_m_s`` is the per-pulse thrust
    cap at the control cadence (DESIGN ASSUMPTION: 1 m/s per 10 s tick, the
    chemical RCS class of the mission's own departure burn compressed to one
    tick); ``min_impulse_m_s`` the minimum impulse bit below which the law
    coasts. The terminal pair (trim + arrival) is NOT a law knob: both arms
    fire it from the same ``clc`` schedule the baseline uses (names, epochs,
    caps, weightings), and the feedback treats their caps as budget
    reserve. The overrides (stress cell) tighten the law's CONSTRAINT view;
    mission grading is untouched."""

    pos_scale_m: float = 50.0
    vel_scale_m_s: float = 0.2
    terminal_gain: float = 25.0
    r_m_s: float = 1.0
    u_max_m_s: float = 1.0
    min_impulse_m_s: float = 0.002
    horizon_steps: int = 12
    keep_out_override_m: float | None = None
    cross_track_override_m: float | None = None

    def weights(self) -> tuple:
        q_pos = 1.0 / (self.pos_scale_m * self.pos_scale_m)
        q_vel = 1.0 / (self.vel_scale_m_s * self.vel_scale_m_s)
        q = np.array([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        return q, np.full(3, float(self.r_m_s)), self.terminal_gain * q


def approach_constraints(cfg: RendezvousMissionConfig, ccfg: ConstrainedControlConfig,
                         ) -> ApproachConstraints:
    """The law's constraint declaration: the mission's frozen safety set
    (from the config, not copied), budget and hold-point range, with the
    stress overrides applied to the law-side view only."""
    if cfg.safety is None:
        raise ValueError(f"{cfg.mission_id}: constrained arms need a safety block")
    s = cfg.safety
    ko = ccfg.keep_out_override_m if ccfg.keep_out_override_m is not None \
        else s.keep_out_radius_m
    xt = ccfg.cross_track_override_m if ccfg.cross_track_override_m is not None \
        else s.cross_track_max_m
    return ApproachConstraints.from_safety_values(
        keep_out_radius_m=ko, cross_track_max_m=xt,
        cone_half_angle_deg=s.cone_half_angle_deg, cone_range_m=s.cone_range_m,
        speed_range_knots_m=s.speed_range_knots_m, speed_cap_m_s=s.speed_cap_m_s,
        u_max_m_s=ccfg.u_max_m_s, budget_m_s=float(cfg.delta_v_budget_m_s),
        hold_point_rho_norm_m=float(np.linalg.norm(cfg.hold_point_rel_lvlh)))


def _planned_arc(ctx: GuidanceContext, step_s: float = 15.0) -> np.ndarray:
    """The uplinked reference the E13-05 arms track: the sample's own planned
    transfer arc (the gate burn 1 state, Newton-solved by ``plan_sample`` on
    the planning dynamics), sampled in LVLH at ``step_s`` from the gate to
    arrival. Rows are ``[t_s, rho, rho_dot]``; the last row is pinned to the
    hold point at rest, the hover arrival the transfer is aimed at. The
    arrival closing speed the arc still carries at the window's end is
    nulled by the arms' scheduled terminal burn, not reference-blended away:
    a ~6 m/s arrest is a finite-maneuver job (see the law docstring), and
    the baseline spends the same delta-v in its scheduled arrival burn."""
    sol = ctx.sol
    tt = np.arange(0.0, float(sol["tof"]) + 0.5 * step_s, step_s)
    tt[-1] = max(tt[-1], float(sol["tof"]))
    tr = propagate_pair(
        np.asarray(sol["chief_t1_r"], dtype=float).reshape(3),
        np.asarray(sol["chief_t1_v"], dtype=float).reshape(3),
        np.asarray(sol["deputy_t1_r"], dtype=float).reshape(3),
        np.asarray(sol["v1_abs"], dtype=float).reshape(3), tt,
        frame=ctx.cfg.dynamics.frame, mu=ctx.cfg.dynamics.mu,
        j2=(EARTH_J2 if ctx.cfg.dynamics.j2 else 0.0),
        where="constrained_campaign: planned arc reference")
    rho, rho_dot = eci_to_lvlh(tr.chief_r[0], tr.chief_v[0], tr.deputy_r[0] - tr.chief_r[0],
                               tr.deputy_v[0] - tr.chief_v[0], ctx.cfg.dynamics.frame,
                               where="constrained_campaign: planned arc LVLH")
    arc = np.column_stack([float(sol["t1"]) + tt, rho, rho_dot])
    arc[-1, 1:4] = np.asarray(ctx.cfg.hold_point_rel_lvlh, dtype=float)
    arc[-1, 4:7] = 0.0
    return arc


def make_law_factory(arm: str, ccfg: ConstrainedControlConfig,
                     laws_out: list | None = None):
    """The ``law_factory`` for :func:`fly_closed_loop_sample` implementing one
    E13-05 arm. ``laws_out`` (optional) collects the built law so the caller
    can read its research bookkeeping afterwards."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    def factory(ctx: GuidanceContext):
        q, r, qt = ccfg.weights()
        common = dict(
            reference=ctx.reference,
            planned_burn=ManeuverPlan(
                "departure", ctx.t1_s, "planned", dv_planned_eci_m_s=ctx.dv1_eci_m_s,
                dv_cap_m_s=max(1.5 * float(np.linalg.norm(ctx.dv1_eci_m_s)),
                               float(np.linalg.norm(ctx.dv1_eci_m_s)) + 1.0)),
            t_feedback_start_s=ctx.t1_s + ctx.clc.cadence_s,
            t_terminal_s=ctx.t_f_s, cadence_s=ctx.clc.cadence_s,
            terminal_lead_s=ctx.clc.pre_terminal_lead_s,
            terminal_rho_lvlh_m=ctx.cfg.hold_point_rel_lvlh,
            constraints=approach_constraints(ctx.cfg, ccfg),
            q_diag=q, r_diag=r, q_terminal_diag=qt,
            min_impulse_m_s=ccfg.min_impulse_m_s,
            trajectory_ref=_planned_arc(ctx),
            scheduled_burns=(
                ManeuverPlan("terminal_trim", ctx.t_f_s - ctx.clc.pre_terminal_lead_s,
                             "retarget", dv_cap_m_s=ctx.clc.trim_cap_m_s,
                             velocity_weight=ctx.clc.pre_terminal_velocity_weight),
                ManeuverPlan("arrival", ctx.t_f_s - ctx.clc.terminal_lead_s, "retarget",
                             dv_cap_m_s=ctx.clc.retarget_cap_m_s,
                             velocity_weight=ctx.clc.terminal_velocity_weight)))
        law = (CwLqrLaw(**common) if arm == "lqr"
               else CwMpcLaw(horizon_steps=ccfg.horizon_steps, **common))
        if laws_out is not None:
            laws_out.append(law)
        return law

    return factory


def _violation_block(cfg: RendezvousMissionConfig, outcome: FlightOutcome, t_from: float,
                     safety: SafetyCfg | None = None) -> dict:
    cfg_g = cfg if safety is None else dataclasses.replace(cfg, safety=safety)
    t = np.asarray(outcome.times_s, dtype=np.float64)
    rel = np.asarray(outcome.rel_lvlh, dtype=np.float64)
    if rel.size == 0:
        return {}
    m = t >= t_from - 1.0e-9
    checks = evaluate_trajectory_safety(cfg_g, t[m], rel[m, :3], rel[m, 3:])
    out = {}
    total = 0
    for name, c in checks.items():
        total += int(c.violation_count)
        out[name] = {"violated_samples": int(c.violation_count),
                     "min_margin": float(c.min_margin) if np.isfinite(c.min_margin) else None,
                     "worst_epoch_s": c.worst_epoch_s}
    out["total_violated_samples"] = total
    return out


def fly_constrained_sample(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                           tof_s: float, seed_base: int, cluster: int, index: int, *,
                           arm: str, ccfg: ConstrainedControlConfig | None = None,
                           clc: ClosedLoopSampleConfig | None = None,
                           run_id: str | None = None
                           ) -> tuple[FlightOutcome, dict, GncRunLog | None]:
    """Fly one E12-10-identical sample under one arm (``baseline`` = the E13-04
    schedule through the untouched default law; ``lqr`` / ``mpc`` = the E13-05
    arms) and extend the closed-loop assessment with the research record:
    ``arm``, ``violations`` (realized, gate onward, mission set),
    ``violations_stress`` (only when constraint overrides are active, against
    the matching stress set), and ``law_stats`` (decision timing / refusals /
    solver fallbacks; empty for the baseline, which schedules its burns)."""
    ccfg = ccfg or ConstrainedControlConfig()
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    built: list = []
    factory = None if arm == "baseline" else make_law_factory(arm, ccfg, built)
    outcome, assess, log = fly_closed_loop_sample(
        cfg, campaign, tof_s, seed_base, cluster, index, clc=clc,
        run_id=run_id or (f"e1305-{arm}-{cfg.mission_id}-{seed_base}-{cluster}-{index}"),
        law_factory=factory)
    assess["arm"] = arm
    if log is None:
        return outcome, assess, log
    t1 = float(assess["t1_s"])
    assess["violations"] = _violation_block(cfg, outcome, t1)
    if ccfg.keep_out_override_m is not None or ccfg.cross_track_override_m is not None:
        stress = dataclasses.replace(
            cfg.safety,
            **({"keep_out_radius_m": ccfg.keep_out_override_m}
               if ccfg.keep_out_override_m is not None else {}),
            **({"cross_track_max_m": ccfg.cross_track_override_m}
               if ccfg.cross_track_override_m is not None else {}))
        assess["violations_stress"] = _violation_block(cfg, outcome, t1, safety=stress)
    if built:
        law = built[0]
        stats = ({"arm": arm, **law.mpc_stats} if arm == "mpc"
                 else {"arm": arm, **law.solve_time_stats})
        stats["refusals"] = len(law.refusals)
        assess["law_stats"] = stats
    return outcome, assess, log
