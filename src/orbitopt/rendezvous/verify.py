"""Re-propagation gate for a :class:`orbitopt.rendezvous.planner.RendezvousPlan`
(E12-03): the acceptance criterion of the task is a maneuver sequence that
arrives at the Hold Point *on the nonlinear propagation*, so the verdict is
earned here, not by the planner's internal bookkeeping.

``verify_plan`` rebuilds the start states from the config (never from
anything the planner remembered), replays the drift leg and the transfer leg
with the plan's epochs and delta-v vectors through
:func:`orbitopt.relative.propagation.propagate_pair` -- the same model the
config selected -- and grades the post-insertion LVLH state against the
config's tolerances and budget. Every check carries a declared expectation
and the measured value; a check that fails says so, and the overall verdict
is the AND of them all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitopt.relative.frames import eci_to_lvlh
from orbitopt.relative.propagation import EARTH_J2, propagate_pair

from .config import RendezvousMissionConfig
from .planner import RendezvousPlan, initial_states


@dataclass(frozen=True)
class RendezvousVerification:
    """The acceptance record for one plan. ``checks`` names every gate;
    ``passed`` is true only when all of them hold."""

    mission_id: str
    frame: str
    drift_end_s: float
    arrival_epoch_s: float
    arrival_rel_lvlh_m: tuple[float, float, float]
    arrival_rel_lvlh_rate_m_s: tuple[float, float, float]
    position_error_m: float
    relative_speed_m_s: float
    total_delta_v_m_s: float
    delta_v_budget_m_s: float
    delta_v_margin_m_s: float
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "frame": self.frame,
            "time_note": "all epochs are elapsed seconds from the shared epoch t0",
            "drift_end_s": self.drift_end_s,
            "arrival_epoch_s": self.arrival_epoch_s,
            "arrival_rel_lvlh_m": list(self.arrival_rel_lvlh_m),
            "arrival_rel_lvlh_rate_m_s": list(self.arrival_rel_lvlh_rate_m_s),
            "position_error_m": self.position_error_m,
            "relative_speed_m_s": self.relative_speed_m_s,
            "maneuvers": [m.to_dict() for m in self._maneuvers],
            "total_delta_v_m_s": self.total_delta_v_m_s,
            "delta_v_budget_m_s": self.delta_v_budget_m_s,
            "delta_v_margin_m_s": self.delta_v_margin_m_s,
            "checks": dict(self.checks),
            "passed": self.passed,
        }

    _maneuvers: tuple = ()


def verify_plan(cfg: RendezvousMissionConfig, plan: RendezvousPlan) -> RendezvousVerification:
    """Replay ``plan`` from the config's t0 states on the configured model
    and grade the arrival. Never mutates ``plan``; never consults the
    planner's internal states."""
    if plan.mission_id != cfg.mission_id:
        raise ValueError(f"plan is for {plan.mission_id!r}, config declares {cfg.mission_id!r}")
    if plan.frame != cfg.dynamics.frame:
        raise ValueError(f"plan frame {plan.frame!r} disagrees with config {cfg.dynamics.frame!r}")
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    m1, m2 = plan.maneuvers
    if m1.epoch_s != plan.drift_end_s or m2.epoch_s != plan.drift_end_s + plan.tof_s:
        raise ValueError("plan maneuver epochs must be [drift end, drift end + tof]")

    r0_chief, v0_chief, r0_chaser, v0_chaser = initial_states(cfg)
    leg_a = propagate_pair(
        r0_chief, v0_chief, r0_chaser, v0_chaser, np.array([0.0, m1.epoch_s]),
        frame=frame, mu=mu, j2=j2, where="verify_plan: drift leg",
    )
    rt1, vt1 = leg_a.chief_r[0, 1], leg_a.chief_v[0, 1]
    rc1, vc1 = leg_a.deputy_r[0, 1], leg_a.deputy_v[0, 1]
    dv1 = np.asarray(m1.delta_v_m_s, dtype=float)

    leg_b = propagate_pair(
        rt1, vt1, rc1, vc1 + dv1, np.array([0.0, plan.tof_s]),
        frame=frame, mu=mu, j2=j2, where="verify_plan: transfer leg",
    )
    rt2, vt2 = leg_b.chief_r[0, 1], leg_b.chief_v[0, 1]
    rc2, vc2 = leg_b.deputy_r[0, 1], leg_b.deputy_v[0, 1]
    dv2 = np.asarray(m2.delta_v_m_s, dtype=float)

    rho, rho_dot = eci_to_lvlh(
        rt2, vt2, rc2 - rt2, (vc2 + dv2) - vt2, frame, where="verify_plan: arrival"
    )
    hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
    pos_err = float(np.linalg.norm(rho - hold))
    rel_speed = float(np.linalg.norm(rho_dot))
    total = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))

    checks = {
        "gate_within_drift_window": m1.epoch_s <= cfg.windows.drift_max_s,
        "tof_within_transfer_window": (
            cfg.windows.transfer_tof_min_s <= plan.tof_s <= cfg.windows.transfer_tof_max_s
        ),
        "position_error_within_tolerance": pos_err <= cfg.max_position_error_m,
        "relative_speed_within_tolerance": rel_speed <= cfg.max_relative_speed_m_s,
        "delta_v_within_budget": total <= cfg.delta_v_budget_m_s,
    }
    return RendezvousVerification(
        mission_id=cfg.mission_id,
        frame=frame,
        drift_end_s=float(m1.epoch_s),
        arrival_epoch_s=float(m2.epoch_s),
        arrival_rel_lvlh_m=tuple(float(x) for x in rho),
        arrival_rel_lvlh_rate_m_s=tuple(float(x) for x in rho_dot),
        position_error_m=pos_err,
        relative_speed_m_s=rel_speed,
        total_delta_v_m_s=total,
        delta_v_budget_m_s=cfg.delta_v_budget_m_s,
        delta_v_margin_m_s=cfg.delta_v_budget_m_s - total,
        checks=checks,
        _maneuvers=plan.maneuvers,
    )
