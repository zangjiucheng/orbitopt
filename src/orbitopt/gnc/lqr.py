"""Finite-horizon LQR tracking on the E13-04 seam (E13-05).

The comparison arm for MPC: the SAME receding-horizon picture -- CW impulse
model on the control cadence, regulation of the LVLH ERROR against an
uplinked reference (the planned transfer arc, or the hold-point equilibrium
``x_eq = (rho_f, 0)`` where no arc is given), stage cost ``e'Qe + u'Ru`` and
a terminal cost from a backward Riccati recursion over the remaining ticks --
but WITHOUT constraint-handling in the optimizer. The only concessions to
reality are honest actuator limits the baseline also respects: a per-pulse
magnitude cap (saturation, the consequences of which the research prices), a
minimum-impulse deadband (below it a pulse is noise, so the law coasts) and
the law-side commanded delta-v ledger (:class:`orbitopt.gnc.guidance.
CwMidCourseLaw` keeps the same kind; realized-versus-commanded accounting
stays offline).

Structure (shared with :mod:`orbitopt.gnc.mpc`, which overrides only the
single-decision hook):

* the scheduled **departure** burn is the open-loop commanded vector at the
  gate, exactly as every other arm fires it -- the arms differ only in how
  the REST of the transfer is flown, so the pairing against the E13-04
  baseline stays exact (same burn, same execution draw);
* feedback ticks run from ``t_feedback_start_s`` (the campaign harness opens
  the window at the tick after departure: the arms fly the WHOLE transfer
  closed-loop against the uplinked arc) until ``t_terminal_s -
  terminal_lead_s``; outside that window the law coasts (a decision it
  records as ``phase == "coast"``);
* gains are time-varying through the remaining-steps index of the Riccati
  sequence, so the terminal condition prices residual state at the horizon's
  end; the feedback replaces the baseline's MID-COURSE correction, not its
  terminal pair;
* ``scheduled_burns`` are terminal maneuvers the law fires itself with CW
  intercept solves (a position trim and a velocity-dominant arrival, the
  same names, epochs, caps and weightings the E13-04 schedule carries in
  ``clc``) -- honest seams: the planned arc arrives with a real closing
  speed and a single 3-DOF burn at the window's end cannot null position
  AND velocity, which is why the mission's own schedule is two terminal
  burns. The feedback horizon ENDS at the first scheduled burn (the harness
  sets ``terminal_lead_s`` to the trim's lead), and the feedback budget view
  -- ``budget_available_m_s``, what both arms' decisions consume against --
  holds the remaining caps in reserve. Mirroring the baseline's terminal
  structure is what makes the comparison about the feedback, not about who
  gets to skip the arrival burn.

State is per-law (fired flag, ledger, timing record): one instance flies one
sample, never reused. The module carries no plant handle and names the
reference motion "reference", never below (E13-01 truth isolation, mirrored
by ``tests/test_e13_05_constrained_control.py``).
"""
from __future__ import annotations

import math
import time

import numpy as np

from orbitopt.relative.cw import cw_state_transition
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci

from .contracts import Estimate, GNCContractError
from .constraints import ApproachConstraints
from .guidance import (GuidanceLawError, ManeuverPlan, ReferenceEphemeris,
                       solve_cw_intercept)


def cw_impulse_matrices(n: float, dt_s: float) -> tuple:
    """Discrete CW step for an impulsive delta-v applied at the START of the
    interval: ``x_next = A x + B u`` with ``A = Phi(dt)`` and ``B`` the
    velocity block of the same transition (an impulse enters as a velocity
    jump the coast then propagates)."""
    n = float(n)
    dt = float(dt_s)
    if not (math.isfinite(n) and n > 0.0 and math.isfinite(dt) and dt > 0.0):
        raise GuidanceLawError(f"cw_impulse_matrices needs finite n, dt > 0, got {n!r}, {dt!r}")
    phi = cw_state_transition(n, dt)
    return phi, phi[:, 3:]


def finite_horizon_cw_lqr_gains(n: float, dt_s: float, q_diag, r_diag, q_terminal_diag,
                                max_steps: int) -> tuple:
    """Backward Riccati over ``max_steps`` identical CW impulse steps.

    Returns ``(K, P)`` indexed by STEPS REMAINING: ``K[k]`` is the gain to
    apply with ``k`` steps left until the terminal epoch (``u = -K[k] e``),
    ``P[k]`` the cost-to-go with ``k`` steps left (``P[0] = diag
    (q_terminal_diag)``). Recursion: ``Ke = (R + B'P B)^-1 B'P A``,
    ``P_next = Q + A'P A - K'B'P A`` -- textbook discrete finite-horizon LQR,
    no stabilizing trick: the horizon IS finite, declared by the mission.
    """
    q = np.asarray(q_diag, dtype=np.float64).reshape(6)
    qt = np.asarray(q_terminal_diag, dtype=np.float64).reshape(6)
    r = np.diag(np.asarray(r_diag, dtype=np.float64).reshape(3))
    if not (np.all(q > 0.0) and np.all(qt > 0.0) and np.all(np.diag(r) > 0.0)):
        raise GuidanceLawError("LQR weights must be strictly positive")
    if int(max_steps) < 1:
        raise GuidanceLawError(f"max_steps must be >= 1, got {max_steps!r}")
    max_steps = int(max_steps)
    a, b = cw_impulse_matrices(n, dt_s)
    k = np.zeros((max_steps + 1, 3, 6))
    p = np.zeros((max_steps + 1, 6, 6))
    p[0] = np.diag(qt)
    q_mat = np.diag(q)
    for steps in range(1, max_steps + 1):
        p_next = p[steps - 1]
        gain = np.linalg.solve(r + b.T @ p_next @ b, b.T @ p_next @ a)
        k[steps] = gain
        p[steps] = q_mat + a.T @ p_next @ a - gain.T @ r @ gain
    return k, p


class FiniteHorizonCwLaw:
    """Shared bookkeeping of the two E13-05 arms: scheduled departure burn,
    feedback window, deadband, saturation, budget ledger, phase metadata and
    per-decision timing. Subclasses supply ``_feedback_pulse``, which sees a
    LVLH error state, the remaining-steps index and the decision epoch and
    returns the LVLH delta-v to command (or ``None`` to coast, having
    recorded why).

    The regulation target is an ERROR against a reference: either the point
    ``x_eq = (rho_f, 0)`` (no ``trajectory_ref``) or a planned transfer arc
    sampled as ``(n, 7)`` ``[t_s, rho, rho_dot]`` rows in LVLH
    (``trajectory_ref``) -- the uplink a chaser would actually fly against,
    which is what makes these arms fair counterparts of the baseline's
    schedule rather than point-bangers (CW phasing makes point regulation
    from far out burn fuel fighting the mission's own drift geometry). The
    arc's final row must sit on the hold point.

    ``planned_burn`` must be a ``ManeuverPlan`` in ``planned`` mode; the
    schedule convention is the baseline's: fire at the first tick at or after
    ``t_fire_s``, decided from the estimate that tick delivered.
    """

    _PHASE_COAST = "coast"

    def __init__(self, *, reference: ReferenceEphemeris, planned_burn: ManeuverPlan,
                 t_feedback_start_s: float, t_terminal_s: float, cadence_s: float,
                 terminal_rho_lvlh_m, constraints: ApproachConstraints,
                 q_diag, r_diag, q_terminal_diag,
                 min_impulse_m_s: float = 0.002, terminal_lead_s: float | None = None,
                 trajectory_ref=None, scheduled_burns: tuple = ()):
        if not isinstance(reference, ReferenceEphemeris):
            raise GuidanceLawError(f"reference must be a ReferenceEphemeris, got {reference!r}")
        if not isinstance(planned_burn, ManeuverPlan) or planned_burn.mode != "planned":
            raise GuidanceLawError("planned_burn must be a ManeuverPlan in 'planned' mode")
        if not isinstance(constraints, ApproachConstraints):
            raise GuidanceLawError(f"constraints must be an ApproachConstraints, got "
                                   f"{constraints!r}")
        self.reference = reference
        self.planned_burn = planned_burn
        self.t_terminal_s = _num("t_terminal_s", t_terminal_s)
        self.t_feedback_start_s = _num("t_feedback_start_s", t_feedback_start_s)
        if self.t_feedback_start_s < planned_burn.t_fire_s:
            raise GuidanceLawError("feedback window cannot open before the departure burn")
        self.cadence_s = _num("cadence_s", cadence_s)
        rho_f = np.asarray(terminal_rho_lvlh_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rho_f)):
            raise GuidanceLawError("terminal_rho_lvlh_m must be finite")
        if abs(float(np.linalg.norm(rho_f)) - constraints.hold_point_rho_norm_m) > 1.0:
            raise GuidanceLawError("terminal rho and the constraints' hold-point range disagree "
                                   "-- one target, two declarations")
        self.x_eq = np.concatenate((rho_f, np.zeros(3)))
        if trajectory_ref is None:
            self._ref_t = None
            self._ref_x = None
        else:
            tab = np.asarray(trajectory_ref, dtype=np.float64)
            if tab.ndim != 2 or tab.shape[1] != 7 or len(tab) < 3:
                raise GuidanceLawError("trajectory_ref must be an (n>=3, 7) [t, 6-state] table")
            if not np.all(np.isfinite(tab)):
                raise GuidanceLawError("trajectory_ref must be finite")
            if np.any(np.diff(tab[:, 0]) <= 0.0):
                raise GuidanceLawError("trajectory_ref times must strictly increase")
            if float(np.linalg.norm(tab[-1, 1:4] - rho_f)) > 1.0:
                raise GuidanceLawError("trajectory_ref must end on the terminal hold point")
            self._ref_t = tab[:, 0]
            self._ref_x = tab[:, 1:7]
        self.constraints = constraints
        self.min_impulse_m_s = _num("min_impulse_m_s", min_impulse_m_s, positive=False)
        self.scheduled_burns = tuple(scheduled_burns)
        for b in self.scheduled_burns:
            if not isinstance(b, ManeuverPlan) or b.mode != "retarget":
                raise GuidanceLawError("scheduled_burns entries must be ManeuverPlans in "
                                       "'retarget' mode")
            if b.t_fire_s < self.t_feedback_start_s + self.cadence_s:
                raise GuidanceLawError("scheduled burns must fire after the first feedback tick")
            if float(b.dv_cap_m_s) <= 0.0 or not math.isfinite(float(b.velocity_weight)):
                raise GuidanceLawError("scheduled burns need a positive cap and a finite "
                                       "velocity weight")
        if sorted(b.t_fire_s for b in self.scheduled_burns) != \
                [b.t_fire_s for b in self.scheduled_burns]:
            raise GuidanceLawError("scheduled_burns must be given in firing order")
        self._burn_fired = [False] * len(self.scheduled_burns)
        lead = 2.0 * self.cadence_s if terminal_lead_s is None else float(terminal_lead_s)
        self.terminal_lead_s = _num("terminal_lead_s", lead)
        if self.terminal_lead_s <= self.cadence_s:
            raise GuidanceLawError("terminal_lead_s must exceed one cadence: the last decision "
                                   "needs a tick to actuate")
        self.n = reference.mean_motion
        max_steps = int(math.ceil((self.t_terminal_s - self.t_feedback_start_s)
                                  / self.cadence_s)) + 2
        self._k, self._p = finite_horizon_cw_lqr_gains(self.n, self.cadence_s, q_diag, r_diag,
                                                       q_terminal_diag, max_steps)
        self._departure_fired = False
        self._commanded_m_s = 0.0
        self._phase_now = self._PHASE_COAST
        self.refusals: list = []
        self._solve_times_s: list = []

    # -- GuidanceLaw ----------------------------------------------------------

    def delta_v(self, estimate: Estimate) -> tuple:
        if not isinstance(estimate, Estimate):
            raise TypeError(f"{type(self).__name__}.delta_v takes an Estimate, got {estimate!r}")
        now = float(estimate.t_update_s)
        self._phase_now = self._PHASE_COAST
        if not self._departure_fired and now + 1.0e-9 >= self.planned_burn.t_fire_s:
            dv = self._planned_pulse(now)
            self._departure_fired = True
            return dv
        due = self._due_burn(now)
        if due is None and not self._in_feedback_window(now):
            return (0.0, 0.0, 0.0)
        t0 = time.perf_counter()
        try:
            cr, cv = self.reference.state_at(now)
            rho, rho_dot = eci_to_lvlh(cr, cv,
                                       np.asarray(estimate.relative_state.r_rel_m,
                                                 dtype=np.float64),
                                       np.asarray(estimate.relative_state.v_rel_m_s,
                                                  dtype=np.float64),
                                       estimate.relative_state.frame,
                                       where=f"{type(self).__name__}")
            if due is not None:
                i, due = due
                self._burn_fired[i] = True
                self._phase_now = due.label
                u_lvlh = self._intercept_command(due, now, rho, rho_dot)
                cap = float(due.dv_cap_m_s)
            else:
                steps_remaining = max(1, int(math.ceil((self.t_terminal_s - now)
                                                       / self.cadence_s)))
                steps_remaining = min(steps_remaining, len(self._k) - 1)
                self._phase_now = "feedback"
                u_lvlh = self._feedback_pulse(np.concatenate((rho, rho_dot)) - self._ref_at(now),
                                              steps_remaining, now)
        finally:
            self._solve_times_s.append(time.perf_counter() - t0)
        if u_lvlh is None:
            return (0.0, 0.0, 0.0)
        u_lvlh = np.asarray(u_lvlh, dtype=np.float64)
        mag = float(np.linalg.norm(u_lvlh))
        if due is not None:
            if mag > cap:
                u_lvlh = u_lvlh * (cap / mag)
                mag = cap
        else:
            if mag < self.min_impulse_m_s:
                return (0.0, 0.0, 0.0)  # below the minimum impulse bit: an ignition is worse
            u_lvlh = self._saturate(mag, u_lvlh)
            mag = float(np.linalg.norm(u_lvlh))
        avail = ((self.constraints.budget_m_s - self._commanded_m_s) if due is not None
                 else self.budget_available_m_s)
        if mag > avail:
            self.refusals.append(f"{self._phase_now} at t={now:.1f}: cumulative commanded "
                                 f"{self._commanded_m_s + mag:.3f} m/s over the available "
                                 f"{self.constraints.budget_m_s} m/s budget (terminal reserve "
                                 f"included for feedback ticks) -- coasted")
            return (0.0, 0.0, 0.0)
        _dr, dv_eci = lvlh_to_eci(cr, cv, np.zeros(3), u_lvlh,
                                  estimate.relative_state.frame,
                                  where=f"{type(self).__name__} egress")
        self._commanded_m_s += mag
        return tuple(float(x) for x in dv_eci)

    def _due_burn(self, now: float):
        """``(index, plan)`` for the next scheduled intercept burn due at or
        before ``now`` -- a peek; the fired flag is set where the command is
        issued."""
        for i, b in enumerate(self.scheduled_burns):
            if not self._burn_fired[i]:
                return (i, b) if now + 1.0e-9 >= b.t_fire_s else None
        return None

    def _intercept_command(self, burn, now: float, rho, rho_dot) -> np.ndarray | None:
        """CW intercept at the hold point at rest -- the same solve class (and
        weighting) the E13-04 schedule's terminal maneuvers use."""
        dv = solve_cw_intercept(self.n, self.t_terminal_s - now,
                                np.concatenate((rho, rho_dot)), self.x_eq,
                                velocity_weight=float(burn.velocity_weight))
        if dv is None:
            self.refusals.append(f"{burn.label} intercept ill-conditioned at t={now:.1f} "
                                 f"-- coasted")
        return dv

    @property
    def budget_available_m_s(self) -> float:
        """Budget the feedback decisions may consume: the ledger remainder,
        less the reserve held for scheduled intercept burns still to fire."""
        reserve = sum(float(b.dv_cap_m_s) for b, fired in zip(self.scheduled_burns,
                                                              self._burn_fired) if not fired)
        return max(0.0, self.constraints.budget_m_s - self._commanded_m_s - reserve)

    def phase_for(self, estimate: Estimate) -> str:
        return self._phase_now

    @property
    def commanded_total_m_s(self) -> float:
        return self._commanded_m_s

    @property
    def solve_time_stats(self) -> dict:
        """Decision-compute accounting for the benchmark's compute-time
        metric: wall seconds spent inside the decision hook, per call."""
        a = np.asarray(self._solve_times_s, dtype=np.float64) * 1000.0
        if a.size == 0:
            return {"n_decisions": 0}
        return {"n_decisions": int(a.size), "mean_ms": float(a.mean()),
                "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
                "total_s": float(a.sum() / 1000.0)}

    # -- subclass hook --------------------------------------------------------

    def _feedback_pulse(self, err_lvlh: np.ndarray, steps_remaining: int,
                        now: float) -> np.ndarray | None:
        raise NotImplementedError

    def _ref_at(self, t: float) -> np.ndarray:
        """Reference state at ``t`` (piecewise-linear between uplinked rows,
        held at the ends); without a table the point target ``x_eq``."""
        if self._ref_t is None:
            return self.x_eq
        return np.array([np.interp(t, self._ref_t, self._ref_x[:, j]) for j in range(6)])

    def _abs_from_err(self, err_lvlh, now: float) -> np.ndarray:
        """Absolute LVLH state implied by an error at epoch ``now``."""
        return np.asarray(err_lvlh, dtype=np.float64) + self._ref_at(now)

    def _ref_rows_at(self, times) -> np.ndarray:
        """Reference rows at several epochs (row 0 = point-target broadcast
        when no table): shape ``(len(times), 6)``."""
        if self._ref_t is None:
            return np.tile(self.x_eq, (len(times), 1))
        return np.stack([self._ref_at(t) for t in times])

    def _saturate(self, mag: float, u_lvlh: np.ndarray) -> np.ndarray:
        cap = self.constraints.u_max_m_s
        return u_lvlh * (cap / mag) if mag > cap else u_lvlh

    # -- internals ------------------------------------------------------------

    def _in_feedback_window(self, now: float) -> bool:
        return (self.t_feedback_start_s - 1.0e-9 <= now
                <= self.t_terminal_s - self.terminal_lead_s + 1.0e-9)

    def _planned_pulse(self, now: float) -> tuple:
        m = self.planned_burn
        dv = np.asarray(m.dv_planned_eci_m_s, dtype=np.float64)
        mag = float(np.linalg.norm(dv))
        if mag > m.dv_cap_m_s:
            self.refusals.append(f"{m.label}: |dv|={mag:.3f} m/s over cap {m.dv_cap_m_s} m/s "
                                 "-- coasted")
            return (0.0, 0.0, 0.0)
        if mag > self.constraints.budget_m_s:
            self.refusals.append(f"{m.label}: planned |dv|={mag:.3f} m/s over budget "
                                 f"{self.constraints.budget_m_s} m/s -- coasted")
            return (0.0, 0.0, 0.0)
        self._commanded_m_s += mag
        self._phase_now = m.label
        return tuple(float(x) for x in dv)

    def unconstrained_pulse(self, err_lvlh: np.ndarray, steps_remaining: int) -> np.ndarray:
        """The frozen-gain LQR step (also the MPC arm's failure fallback)."""
        return -self._k[steps_remaining] @ np.asarray(err_lvlh, dtype=np.float64)


class CwLqrLaw(FiniteHorizonCwLaw):
    """The unconstrained arm: u = -K[k] e with the finite-horizon gain of the
    remaining ticks, then deadband, saturation and the budget ledger."""

    def _feedback_pulse(self, err_lvlh: np.ndarray, steps_remaining: int,
                        now: float) -> np.ndarray:
        return self.unconstrained_pulse(err_lvlh, steps_remaining)


def _num(name: str, value, *, positive: bool = True) -> float:
    v = float(value)
    if not math.isfinite(v) or (v <= 0.0 if positive else v < 0.0):
        raise GuidanceLawError(f"{name} must be finite and {'> 0' if positive else '>= 0'}, "
                               f"got {value!r}")
    return v
