"""Two-impulse rendezvous planner: V-bar drift, then one targeted Lambert
transfer to a hovering arrival at the Hold Point (E12-03, LEO-RDV-001).

The plan is three phases on one propagation model -- the pair propagator
from E12-02 (two-body + optional J2), never the CW linearisation (its
measured validity envelope, ``docs/benchmarks/e12_02_cw_validity.json``,
does not cover kilometre-scale transfers; the planning-only rule from the
E12-02 ADR):

1. **Drift.** The chaser starts on a lower coplanar circular orbit, so it
   drifts forward along V-bar; the phase ends when the LVLH along-track
   offset reaches the configured departure gate.
2. **Transfer.** A 0-revolution Lambert arc (``solve_lambert_single``,
   invoked with an *explicit* Earth ``mu`` -- the solver's default is
   ``pk.MU_SUN``, and handing it LEO states with that default must stay a
   loud failure, which is what ``tests/test_e12_03_lambert_earth_mu.py``
   pins) connects the gate state to the arrival point at the chosen
   time-of-flight. The two-body Lambert solution is only the *initial
   guess*: with J2 on, the differential short-period between the transfer
   orbit and the hold orbit is kilometres, so a fixed inertial aim-point
   iteration cannot converge, and the departure burn is instead Newton-
   solved on the landing position under the configured nonlinear dynamics
   (finite-difference Jacobian, four batched pairs per iteration). The
   time-of-flight is chosen by golden-section search on total delta-v
   inside the config's fixed window.
3. **Insertion.** The second burn re-nulls the relative velocity at the
   Hold Point: arrival is *by construction* at zero relative speed at the
   predicted chief state, so the remaining error is purely the Newton
   landing residual in position.

Times are elapsed seconds from the shared epoch t0. The output
:class:`RendezvousPlan` carries the maneuver sequence (epoch, ECI delta-v);
budget margin comes from :func:`orbitopt.rendezvous.verify.verify_plan`,
which re-propagates the plan from the config and never trusts this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.relative.frames import lvlh_to_eci
from orbitopt.relative.propagation import EARTH_J2, EARTH_RE, propagate_pair

from .config import RendezvousMissionConfig

#: Departure delta-v above this is a branch/sense blunder, not a plan; the
#: golden-section loop treats such a time-of-flight as infeasible.
MAX_PLAUSIBLE_BURN_M_S = 500.0


class RendezvousPlanError(RuntimeError):
    """No feasible two-impulse plan inside the config's fixed windows."""


@dataclass(frozen=True)
class Maneuver:
    """One impulsive burn: epoch [s after t0], delta-v vector in the inertial
    ``frame`` of the config, m/s."""

    kind: str
    epoch_s: float
    delta_v_m_s: tuple[float, float, float]

    @property
    def magnitude_m_s(self) -> float:
        return float(np.linalg.norm(self.delta_v_m_s))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "epoch_s": self.epoch_s,
            "delta_v_eci_m_s": list(self.delta_v_m_s),
            "magnitude_m_s": self.magnitude_m_s,
        }


@dataclass(frozen=True)
class RendezvousPlan:
    """A planned drift + two-burn rendezvous. ``maneuvers`` is ordered by
    epoch; ``total_delta_v_m_s`` is the sum of their magnitudes (impulsive
    burns cannot be cheaper than that)."""

    mission_id: str
    frame: str
    drift_end_s: float
    tof_s: float
    maneuvers: tuple[Maneuver, Maneuver]
    total_delta_v_m_s: float
    delta_v_budget_m_s: float
    chief_at_gate_r_m: tuple[float, float, float]
    chief_at_gate_v_m_s: tuple[float, float, float]
    chief_at_arrival_r_m: tuple[float, float, float]
    chief_at_arrival_v_m_s: tuple[float, float, float]
    hold_point_rel_lvlh_m: tuple[float, float, float]

    @property
    def arrival_epoch_s(self) -> float:
        return self.drift_end_s + self.tof_s

    @property
    def budget_margin_m_s(self) -> float:
        return self.delta_v_budget_m_s - self.total_delta_v_m_s

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "frame": self.frame,
            "time_note": "all epochs are elapsed seconds from the shared epoch t0",
            "drift_end_s": self.drift_end_s,
            "transfer_tof_s": self.tof_s,
            "arrival_epoch_s": self.arrival_epoch_s,
            "maneuvers": [m.to_dict() for m in self.maneuvers],
            "total_delta_v_m_s": self.total_delta_v_m_s,
            "delta_v_budget_m_s": self.delta_v_budget_m_s,
            "budget_margin_m_s": self.budget_margin_m_s,
            "hold_point_rel_lvlh_m": list(self.hold_point_rel_lvlh_m),
        }


def _circular_state(a: float, inc: float, raan: float, u: float, mu: float):
    """Circular-orbit ECI state at argument of latitude ``u`` (rad): built
    in the orbital plane and rotated by RAAN/inc, the same construction the
    E12-02 tests use."""
    r_pq = np.array([a * math.cos(u), a * math.sin(u), 0.0])
    v_pq = math.sqrt(mu / a) * np.array([-math.sin(u), math.cos(u), 0.0])
    rot_z = np.array(
        [[math.cos(raan), -math.sin(raan), 0.0], [math.sin(raan), math.cos(raan), 0.0], [0.0, 0.0, 1.0]]
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(inc), -math.sin(inc)], [0.0, math.sin(inc), math.cos(inc)]]
    )
    rot = rot_z @ rot_x
    return rot @ r_pq, rot @ v_pq


def initial_states(cfg: RendezvousMissionConfig) -> tuple[np.ndarray, ...]:
    """The four ECI start vectors ``(r, v)`` for chief (target) and deputy
    (chaser) at the shared epoch t0, from the config. Coplanar pair (v1):
    the chaser inherits the target's plane, sits at its own circular radius,
    and is phased aft so the LVLH separation at t0 is exactly
    ``chaser.initial_separation_m`` (radial gap from the altitude difference,
    the rest along-track)."""
    mu = cfg.dynamics.mu
    inc = math.radians(cfg.target.inclination_deg)
    raan = math.radians(cfg.target.raan_deg)
    u_t = math.radians(cfg.target.argument_of_latitude_deg)
    a_t = EARTH_RE + cfg.target.altitude_m
    a_c = EARTH_RE + cfg.chaser.altitude_m
    r0_chief, v0_chief = _circular_state(a_t, inc, raan, u_t, mu)
    du = cfg.along_track_offset_m / a_c  # negative: aft on V-bar
    r0_chaser, v0_chaser = _circular_state(a_c, inc, raan, u_t + du, mu)
    return r0_chief, v0_chief, r0_chaser, v0_chaser


def _gate_crossing_at(cfg: RendezvousMissionConfig, r0_chief, v0_chief, r0_chaser, v0_chaser):
    """Drift phase from *explicit* t0 states: propagate the pair on a coarse
    grid, bracket the first along-track crossing of the departure gate, refine
    it linearly, then re-propagate exactly to that epoch. Returns
    ``(t1, states at t1)``. Split out of :func:`_gate_crossing` so a caller
    that flies the mission from a state other than the config's nominal one
    (the E12-07 dispersion campaign, where the onboard navigation estimate is
    perturbed) can open the transfer window the same way the planner does."""
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    step = cfg.phasing.gate_scan_step_s
    ts = np.arange(0.0, cfg.windows.drift_max_s + step, step, dtype=float)
    ts[-1] = min(ts[-1], cfg.windows.drift_max_s)
    traj = propagate_pair(
        r0_chief, v0_chief, r0_chaser, v0_chaser, ts,
        frame=cfg.dynamics.frame, mu=cfg.dynamics.mu, j2=j2, where="plan_rendezvous: drift scan",
    )
    y = traj.rel_lvlh[0, :, 1]
    hits = np.nonzero(y >= cfg.phasing.departure_gate_y_m)[0]
    if hits.size == 0 or hits[0] == 0:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: along-track offset never reaches the departure gate "
            f"{cfg.phasing.departure_gate_y_m:.0f} m within drift_max_s = {cfg.windows.drift_max_s:.0f} s "
            f"(y(t0) = {y[0]:.0f} m); the drift phase cannot open the transfer window"
        )
    i = int(hits[0])
    t1 = float(ts[i - 1] + step * (cfg.phasing.departure_gate_y_m - y[i - 1]) / (y[i] - y[i - 1]))
    fine = propagate_pair(
        r0_chief, v0_chief, r0_chaser, v0_chaser, np.array([0.0, t1]),
        frame=cfg.dynamics.frame, mu=cfg.dynamics.mu, j2=j2, where="plan_rendezvous: gate epoch",
    )
    return t1, fine.chief_r[0, 1], fine.chief_v[0, 1], fine.deputy_r[0, 1], fine.deputy_v[0, 1]


def _gate_crossing(cfg: RendezvousMissionConfig):
    """:func:`_gate_crossing_at` on the config's nominal t0 states."""
    return _gate_crossing_at(cfg, *initial_states(cfg))


def _hold_aim(cfg: RendezvousMissionConfig, chief_r_t2, chief_v_t2):
    """The inertial arrival aim point + velocity: the HOVER state at the
    Hold Point in the *predicted* chief LVLH frame at t2. Targeting this
    instead of a J2-drifted deputy state makes the post-insertion LVLH
    relative state exactly (hold, 0) up to the Newton landing residual."""
    dr, dv = lvlh_to_eci(
        chief_r_t2, chief_v_t2,
        np.asarray(cfg.hold_point_rel_lvlh), np.zeros(3),
        cfg.dynamics.frame, where="plan_rendezvous: hold aim",
    )
    return chief_r_t2 + dr, chief_v_t2 + dv


def _solve_transfer(cfg: RendezvousMissionConfig, t1, rt1, vt1, rc1, vc1, tof: float) -> dict:
    """Solve the two burns for one candidate time-of-flight, or raise.

    Predicts the chief at ``t1 + tof``, builds the hover arrival aim, takes
    the explicit-mu 0-rev Lambert solution as the initial guess, then Newtons
    the departure burn on the landing position under the configured
    dynamics. Returns the absolute burn velocities, both delta-v norms and
    the states the caller needs.
    """
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    h = np.cross(rt1, vt1)
    h = h / np.linalg.norm(h)

    pred = propagate_pair(
        rt1, vt1, rt1, vt1, np.array([0.0, tof]),
        frame=frame, mu=mu, j2=j2, where="plan_rendezvous: chief prediction",
    )
    rt2, vt2 = pred.chief_r[0, 1], pred.chief_v[0, 1]
    aim, va2 = _hold_aim(cfg, rt2, vt2)

    sols = solve_lambert_single(rc1, aim, tof, mu=mu, max_revs=0, plane_normal=h)
    v1 = np.asarray(sols[0][0], dtype=float)
    if float(np.linalg.norm(v1 - vc1)) > MAX_PLAUSIBLE_BURN_M_S:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: two-body Lambert initial guess needs "
            f"{np.linalg.norm(v1 - vc1):.1f} m/s at tof={tof:.0f} s -- branch or geometry blunder"
        )

    eps = cfg.optimizer.jacobian_eps_m_s
    eye = np.eye(3) * eps

    def land(v1_try: np.ndarray) -> np.ndarray:
        """Landing positions for [nominal, +eps in x, y, z] departure
        velocities, one batched pair propagation."""
        v_batch = np.vstack([v1_try, v1_try + eye[0], v1_try + eye[1], v1_try + eye[2]])
        tr = propagate_pair(
            np.broadcast_to(rt1, (4, 3)), np.broadcast_to(vt1, (4, 3)),
            np.broadcast_to(rc1, (4, 3)), v_batch, np.array([0.0, tof]),
            frame=frame, mu=mu, j2=j2, where="plan_rendezvous: Newton landing",
        )
        return tr.deputy_r[:, 1, :]

    residual = None
    for _ in range(cfg.optimizer.newton_max_iter):
        R = land(v1)
        residual = R[0] - aim
        if np.linalg.norm(residual) < cfg.optimizer.newton_tol_m:
            break
        J = (R[1:] - R[0]).T / eps
        v1 = v1 - np.linalg.solve(J, residual)
    else:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: burn-1 Newton did not reach {cfg.optimizer.newton_tol_m:g} m "
            f"at tof={tof:.0f} s (residual {np.linalg.norm(residual):.3g} m)"
        )
    if not np.all(np.isfinite(v1)) or float(np.linalg.norm(v1 - vc1)) > MAX_PLAUSIBLE_BURN_M_S:
        raise RendezvousPlanError(f"{cfg.mission_id}: Newton converged onto an implausible burn at tof={tof:.0f} s")

    dv1 = v1 - vc1
    leg = propagate_pair(
        rt1, vt1, rc1, v1, np.array([0.0, tof]),
        frame=frame, mu=mu, j2=j2, where="plan_rendezvous: transfer leg",
    )
    vc2 = leg.deputy_v[0, 1]
    dv2 = va2 - vc2
    return {
        "t1": t1, "tof": tof, "v1_abs": v1, "dv1": dv1, "dv2": dv2,
        "chief_t2": leg.chief_r[0, 1], "chief_t2_v": leg.chief_v[0, 1],
        "chief_t1_r": rt1, "chief_t1_v": vt1,
        "deputy_t1_r": rc1, "deputy_t1_v": vc1,
        "total_m_s": float(np.linalg.norm(dv1) + np.linalg.norm(dv2)),
    }


def _cost(
    cfg: RendezvousMissionConfig,
    gate,
    tof: float,
    feasible=None,
) -> tuple[float, dict | None]:
    try:
        sol = _solve_transfer(cfg, *gate, tof)
    except (RendezvousPlanError, ValueError, np.linalg.LinAlgError) as exc:
        return math.inf, {"reason": str(exc)[:200]}
    if feasible is not None:
        reason = feasible(sol)
        if reason:
            return math.inf, {"reason": str(reason)[:300]}
    return sol["total_m_s"], sol


def _golden_bracket(
    cfg: RendezvousMissionConfig,
    gate,
    lo: float,
    hi: float,
    feasible=None,
) -> tuple[float | None, dict | None, list[str]]:
    """Golden-section minimization of total delta-v on [lo, hi]. Returns
    ``(best_tof, best_sol, failure_reasons)``; the best pair is (None, None)
    when no evaluated candidate was solvable-and-feasible. Assumes a
    unimodal-ish cost curve -- the caller picks a bracket where that holds
    (plain ``plan_rendezvous``: dv-vs-tof is convex-ish with no feasibility
    hook; ``plan_rendezvous_safe``: one feasible basin from a coarse scan).
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, sol_c = _cost(cfg, gate, c, feasible)
    fd, sol_d = _cost(cfg, gate, d, feasible)
    best_tof, best_sol = (c, sol_c) if fc <= fd else (d, sol_d)
    failures: list[str] = []
    for sol in (sol_c, sol_d):
        if sol is not None and "reason" in sol:
            failures.append(sol["reason"])
    for _ in range(cfg.optimizer.golden_max_iter - 2):
        if b - a < 1.0:
            break
        if fc <= fd:
            b, d, fd, sol_d = d, c, fc, sol_c
            c = b - phi * (b - a)
            fc, sol_c = _cost(cfg, gate, c, feasible)
        else:
            a, c, fc, sol_c = c, d, fd, sol_d
            d = a + phi * (b - a)
            fd, sol_d = _cost(cfg, gate, d, feasible)
        if fc <= fd:
            best_tof, best_sol = c, sol_c
        else:
            best_tof, best_sol = d, sol_d
        if sol_c is not None and "reason" in sol_c:
            failures.append(sol_c["reason"])
        if sol_d is not None and "reason" in sol_d:
            failures.append(sol_d["reason"])
    if best_sol is None or "reason" in best_sol:
        return None, None, failures
    return float(best_tof), best_sol, failures


def _plan_from_solution(cfg: RendezvousMissionConfig, t1: float, tof: float, sol: dict) -> RendezvousPlan:
    m1 = Maneuver("transfer_injection", sol["t1"], tuple(float(x) for x in sol["dv1"]))
    m2 = Maneuver("hold_point_insertion", sol["t1"] + sol["tof"], tuple(float(x) for x in sol["dv2"]))
    return RendezvousPlan(
        mission_id=cfg.mission_id,
        frame=cfg.dynamics.frame,
        drift_end_s=t1,
        tof_s=float(tof),
        maneuvers=(m1, m2),
        total_delta_v_m_s=sol["total_m_s"],
        delta_v_budget_m_s=cfg.delta_v_budget_m_s,
        chief_at_gate_r_m=tuple(float(x) for x in sol["chief_t1_r"]),
        chief_at_gate_v_m_s=tuple(float(x) for x in sol["chief_t1_v"]),
        chief_at_arrival_r_m=tuple(float(x) for x in sol["chief_t2"]),
        chief_at_arrival_v_m_s=tuple(float(x) for x in sol["chief_t2_v"]),
        hold_point_rel_lvlh_m=tuple(cfg.hold_point_rel_lvlh),
    )


def plan_rendezvous(cfg: RendezvousMissionConfig, *, feasible=None) -> RendezvousPlan:
    """Optimize and return the two-impulse plan for ``cfg``. The golden-
    section search runs inside the config's fixed time-of-flight window;
    the delta-v budget is never consulted as a constraint here -- it is
    reported against, in :func:`orbitopt.rendezvous.verify.verify_plan`.

    ``feasible(sol) -> str | None`` (E12-04) is an optional hard-feasibility
    screen on a solved candidate transfer: a returned reason string makes
    that time-of-flight infeasible (cost +inf) and joins the failure list
    raised when the whole window screens out. Note golden-section search
    alone is not safe with a feasibility screen -- the feasible set can be
    two islands -- so safety-constrained planning goes through
    :func:`orbitopt.rendezvous.plan_rendezvous_safe`, which coarse-scans
    the window first."""
    t1, rt1, vt1, rc1, vc1 = _gate_crossing(cfg)
    gate = (t1, rt1, vt1, rc1, vc1)
    lo, hi = cfg.windows.transfer_tof_min_s, cfg.windows.transfer_tof_max_s
    best_tof, best_sol, failures = _golden_bracket(cfg, gate, lo, hi, feasible)
    if best_sol is None:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: no feasible transfer in the window [{lo:.0f}, {hi:.0f}] s; "
            f"first failures: {' | '.join(failures[:3])}"
        )
    return _plan_from_solution(cfg, t1, best_tof, best_sol)
