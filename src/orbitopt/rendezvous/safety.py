"""Full-trajectory safety constraints for a rendezvous plan (E12-04).

The E12-03 planner answers "how cheap can the transfer be"; this module
answers "is the whole mission safe", on the constraints fixed in the
config's ``safety`` block before any optimization runs:

* ``keep_out``    -- the sphere around the target, ``|rho| >=`` its radius
  on every sample of the trajectory, not just at the endpoints;
* ``terminal_cone`` -- within ``cone_range_m`` the line of sight from the
  target must stay within ``cone_half_angle_deg`` of ``-V-bar`` (90 deg
  reads "never in front"); the mid-range transfer geometry of a two-pulse
  family flies *over* the target, so the corridor is the terminal one plus
  the containment checks below, per ADR rendezvous-0002;
* ``cross_track`` -- ``|z|`` containment out of the V-bar plane;
* ``approach_speed`` -- ``|rho_dot|`` under a piecewise-linear ceiling that
  tightens as range closes;
* ``fuel_budget`` / ``time_window`` -- the E12-03 fixed budget and windows,
  re-stated as per-epoch safety margins;
* passive releases -- burn-abort coast from t0, from the gate, and from the
  Hold Point: with no thrust the keep-out sphere must stay clear for
  ``passive_contingency_s``.

Every constraint is evaluated on a dense re-propagated grid
(``check_sample_dt_s``, burn epochs and leg ends always included) over BOTH
legs -- drift coast, transfer coast between the burns, the post-insertion
hover sample -- so a violation anywhere, including between maneuvers,
reports its epoch and LVLH position with a signed margin.

``verify_plan_safety`` grades a finished plan;
``plan_rendezvous_safe`` optimizes under the same constraints, screening
every candidate transfer out of the golden-section search when it violates
them, and raises with the collected reasons when nothing inside the fixed
windows survives -- "plan safe, or say why not".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitopt.relative.frames import eci_to_lvlh
from orbitopt.relative.propagation import EARTH_J2, propagate_pair

from .canonical import canonical_config_sha256, rendezvous_plan_digest
from .config import RendezvousConfigError, RendezvousMissionConfig, SafetyCfg
from .planner import (
    RendezvousPlan,
    RendezvousPlanError,
    _cost,
    _gate_crossing,
    _golden_bracket,
    _plan_from_solution,
    initial_states,
)
from .verify import RendezvousVerification, verify_plan

#: Cap on the violation locations listed per check in a report dict.
MAX_VIOLATIONS_LISTED = 5

#: Intervals of the coarse time-of-flight scan that brackets the golden-
#: section refinement. The safety-screened feasible set is NOT convex (for
#: LEO-RDV-001 it is two islands), so a golden-section search alone can
#: discard the good basin on its first two infeasible samples; the scan
#: finds basins, the refinement polishes one.
SCAN_INTERVALS = 12

#: Checks that run on every flight sample (as opposed to event checks).
TRAJECTORY_CHECKS = ("keep_out", "terminal_cone", "cross_track", "approach_speed")

_EPS_M = 1e-9


@dataclass(frozen=True)
class SafetyCheck:
    """One constraint evaluated over its applicable samples. ``margins`` is
    signed per epoch (negative = violated), ``times_s`` the elapsed-seconds
    epochs aligned with it, ``positions_rel_lvlh_m`` the deputy LVLH position
    where the samples are spatial (None for event checks like the fuel
    budget). The worst epoch and position locate the binding sample."""

    name: str
    unit: str
    detail: str
    times_s: np.ndarray
    margins: np.ndarray
    positions_rel_lvlh_m: np.ndarray | None
    passed: bool
    min_margin: float
    worst_epoch_s: float | None
    worst_rel_lvlh_m: tuple[float, float, float] | None
    violation_count: int
    violations: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "detail": self.detail,
            "passed": self.passed,
            "min_margin": self.min_margin if np.isfinite(self.min_margin) else None,
            "worst_epoch_s": self.worst_epoch_s,
            "worst_rel_lvlh_m": list(self.worst_rel_lvlh_m) if self.worst_rel_lvlh_m else None,
            "violation_count": self.violation_count,
            "violations": list(self.violations),
            "epochs_s": self.times_s.tolist(),
            "margins": self.margins.tolist(),
            "rel_lvlh_m": None if self.positions_rel_lvlh_m is None else self.positions_rel_lvlh_m.tolist(),
        }


@dataclass(frozen=True)
class PassiveRelease:
    """One no-thrust coast contingency: where it releases, how long it
    coasts, and the closest the uncontrolled chaser came to the target."""

    kind: str
    release_epoch_s: float
    duration_s: float
    min_range_m: float
    worst_epoch_s: float
    worst_rel_lvlh_m: tuple[float, float, float]
    keep_out_margin_m: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "release_epoch_s": self.release_epoch_s,
            "duration_s": self.duration_s,
            "min_range_m": self.min_range_m,
            "worst_epoch_s": self.worst_epoch_s,
            "worst_rel_lvlh_m": list(self.worst_rel_lvlh_m),
            "keep_out_margin_m": self.keep_out_margin_m,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SafetyReport:
    """The safety verdict for one plan: ``checks`` names every constraint
    (trajectory checks plus the fuel-budget and time-window event checks),
    ``passive`` holds the contingency coasts. ``passed`` is the AND of all.
    ``plan_digest`` / ``config_sha256`` (E12-08) record the identity of the
    plan this report graded -- the same digest the sealed canonical plan
    carries, recomputed here from the config and plan actually consumed."""

    mission_id: str
    frame: str
    checks: dict[str, SafetyCheck]
    passive: tuple[PassiveRelease, ...]
    plan_digest: str = ""
    config_sha256: str = ""

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks.values()) and all(p.passed for p in self.passive)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, c in self.checks.items() if not c.passed))

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "frame": self.frame,
            "time_note": "all epochs are elapsed seconds from the shared epoch t0",
            "plan_digest": self.plan_digest,
            "config_sha256": self.config_sha256,
            "passed": self.passed,
            "checks": {name: c.to_dict() for name, c in self.checks.items()},
            "passive_contingency": [p.to_dict() for p in self.passive],
        }


def _mk_check(
    name: str,
    unit: str,
    detail: str,
    times_s: np.ndarray,
    margins: np.ndarray,
    positions: np.ndarray | None = None,
) -> SafetyCheck:
    times_s = np.asarray(times_s, dtype=float)
    margins = np.asarray(margins, dtype=float)
    if margins.size:
        i = int(np.argmin(margins))
        min_margin = float(margins[i])
        worst_epoch = float(times_s[i])
        worst_pos = None if positions is None else tuple(float(x) for x in positions[i])
    else:
        min_margin, worst_epoch, worst_pos = float("inf"), None, None
    bad = np.nonzero(margins < 0.0)[0]
    violations = tuple(
        {
            "epoch_s": float(times_s[j]),
            "rel_lvlh_m": None if positions is None else [float(x) for x in positions[j]],
            "margin": float(margins[j]),
        }
        for j in bad[:MAX_VIOLATIONS_LISTED]
    )
    return SafetyCheck(
        name=name,
        unit=unit,
        detail=detail,
        times_s=times_s,
        margins=margins,
        positions_rel_lvlh_m=None if positions is None else np.asarray(positions, dtype=float),
        passed=bool(bad.size == 0),
        min_margin=min_margin,
        worst_epoch_s=worst_epoch,
        worst_rel_lvlh_m=worst_pos,
        violation_count=int(bad.size),
        violations=violations,
    )


def approach_speed_cap_m_s(safety: SafetyCfg, ranges_m: np.ndarray) -> np.ndarray:
    """The configured relative-speed ceiling at each range: piecewise-linear
    through the knots, constant below the first and above the last."""
    return np.interp(np.asarray(ranges_m, dtype=float), np.asarray(safety.speed_range_knots_m), np.asarray(safety.speed_cap_m_s))


def evaluate_trajectory_safety(
    cfg: RendezvousMissionConfig,
    times_s: np.ndarray,
    rho_rel_lvlh_m: np.ndarray,
    rho_dot_rel_lvlh_m_s: np.ndarray,
) -> dict[str, SafetyCheck]:
    """Grade the four per-sample constraints on arbitrary LVLH samples --
    the same evaluation ``verify_plan_safety`` runs on the re-propagated
    flight, usable directly on any measured or constructed trajectory
    (that is what makes a constructed keep-out crossing provable)."""
    if cfg.safety is None:
        raise RendezvousConfigError(
            f"{cfg.mission_id}: config carries no 'safety' block; safety checks need one"
        )
    s = cfg.safety
    t = np.asarray(times_s, dtype=float)
    rho = np.asarray(rho_rel_lvlh_m, dtype=float)
    rhod = np.asarray(rho_dot_rel_lvlh_m_s, dtype=float)
    if rho.ndim != 2 or rho.shape[1] != 3 or rho.shape != rhod.shape or t.shape != (rho.shape[0],):
        raise ValueError(
            f"expected aligned (n, 3) LVLH positions/rates and (n,) epochs, "
            f"got {rho.shape}, {rhod.shape}, {t.shape}"
        )
    rng = np.linalg.norm(rho, axis=1)
    spd = np.linalg.norm(rhod, axis=1)
    checks: dict[str, SafetyCheck] = {}
    checks["keep_out"] = _mk_check(
        "keep_out", "m",
        f"|rho| - keep_out_radius over all {rho.shape[0]} samples",
        t, rng - s.keep_out_radius_m, rho,
    )
    los_deg = np.degrees(np.arccos(np.clip(-rho[:, 1] / np.maximum(rng, _EPS_M), -1.0, 1.0)))
    in_cone = rng <= s.cone_range_m
    checks["terminal_cone"] = _mk_check(
        "terminal_cone", "deg",
        f"{int(in_cone.sum())} of {rho.shape[0]} samples within {s.cone_range_m:.0f} m; "
        f"LOS half-angle from -V-bar must stay <= {s.cone_half_angle_deg:.0f} deg",
        t[in_cone], s.cone_half_angle_deg - los_deg[in_cone], rho[in_cone],
    )
    checks["cross_track"] = _mk_check(
        "cross_track", "m",
        f"cross_track_max_m - |z| over all {rho.shape[0]} samples",
        t, s.cross_track_max_m - np.abs(rho[:, 2]), rho,
    )
    checks["approach_speed"] = _mk_check(
        "approach_speed", "m/s",
        f"piecewise-linear speed cap(|rho|) - |rho_dot| over all {rho.shape[0]} samples",
        t, approach_speed_cap_m_s(s, rng) - spd, rho,
    )
    return checks


def _flight_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    ts = np.arange(t_start, t_end, dt, dtype=float)
    if ts.size == 0:
        return np.array([t_end], dtype=float)
    if ts[-1] != t_end:
        ts = np.append(ts, t_end)
    return ts


def _passive_release(
    cfg: RendezvousMissionConfig,
    kind: str,
    release_epoch_s: float,
    chief_r: np.ndarray,
    chief_v: np.ndarray,
    deputy_r: np.ndarray,
    deputy_v: np.ndarray,
) -> PassiveRelease:
    s = cfg.safety
    dur = s.passive_contingency_s
    step = max(s.check_sample_dt_s, dur / 600.0)
    traj = propagate_pair(
        chief_r, chief_v, deputy_r, deputy_v, _flight_grid(0.0, dur, step),
        frame=cfg.dynamics.frame, mu=cfg.dynamics.mu,
        j2=EARTH_J2 if cfg.dynamics.j2 else 0.0,
        where=f"verify_plan_safety: passive release {kind}",
    )
    rho = traj.rel_lvlh[0, :, :3]
    rng = np.linalg.norm(rho, axis=1)
    i = int(np.argmin(rng))
    margin = float(rng[i] - s.keep_out_radius_m)
    return PassiveRelease(
        kind=kind,
        release_epoch_s=float(release_epoch_s),
        duration_s=float(dur),
        min_range_m=float(rng[i]),
        worst_epoch_s=float(traj.times[i]),
        worst_rel_lvlh_m=tuple(float(x) for x in rho[i]),
        keep_out_margin_m=margin,
        passed=margin >= 0.0,
    )


def verify_plan_safety(cfg: RendezvousMissionConfig, plan: RendezvousPlan) -> SafetyReport:
    """Re-propagate ``plan`` from the config (like ``verify_plan``, and
    equally trusting nothing the planner remembered) on the dense
    ``check_sample_dt_s`` grid and grade every configured constraint on the
    FULL trajectory: drift leg, transfer leg -- coast between the burns
    included -- the post-insertion hover sample, and the three passive
    releases. The transfer leg's first sample is post-burn-1 and its last
    pre-burn-2, so both burn epochs appear with their before/after states."""
    if cfg.safety is None:
        raise RendezvousConfigError(
            f"{cfg.mission_id}: config carries no 'safety' block; verify_plan_safety needs one"
        )
    if plan.mission_id != cfg.mission_id:
        raise ValueError(f"plan is for {plan.mission_id!r}, config declares {cfg.mission_id!r}")
    if plan.frame != cfg.dynamics.frame:
        raise ValueError(f"plan frame {plan.frame!r} disagrees with config {cfg.dynamics.frame!r}")
    s = cfg.safety
    dt = s.check_sample_dt_s
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    m1, m2 = plan.maneuvers

    r0c, v0c, r0d, v0d = initial_states(cfg)
    ta = _flight_grid(0.0, m1.epoch_s, dt)
    leg_a = propagate_pair(
        r0c, v0c, r0d, v0d, ta, frame=frame, mu=mu, j2=j2,
        where="verify_plan_safety: drift leg",
    )
    rt1, vt1 = leg_a.chief_r[0, -1], leg_a.chief_v[0, -1]
    rc1, vc1 = leg_a.deputy_r[0, -1], leg_a.deputy_v[0, -1]
    dv1 = np.asarray(m1.delta_v_m_s, dtype=float)

    tb = _flight_grid(0.0, plan.tof_s, dt)
    leg_b = propagate_pair(
        rt1, vt1, rc1, vc1 + dv1, tb, frame=frame, mu=mu, j2=j2,
        where="verify_plan_safety: transfer leg",
    )
    rt2, vt2 = leg_b.chief_r[0, -1], leg_b.chief_v[0, -1]
    rc2, vc2 = leg_b.deputy_r[0, -1], leg_b.deputy_v[0, -1]
    dv2 = np.asarray(m2.delta_v_m_s, dtype=float)
    rho_post, rhod_post = eci_to_lvlh(
        rt2, vt2, rc2 - rt2, (vc2 + dv2) - vt2, frame, where="verify_plan_safety: post-insertion"
    )

    times = np.concatenate([ta, tb + m1.epoch_s, [m2.epoch_s]])
    rho = np.concatenate([leg_a.rel_lvlh[0, :, :3], leg_b.rel_lvlh[0, :, :3], rho_post[None, :]])
    rhod = np.concatenate([leg_a.rel_lvlh[0, :, 3:], leg_b.rel_lvlh[0, :, 3:], rhod_post[None, :]])
    checks = evaluate_trajectory_safety(cfg, times, rho, rhod)

    dv1_mag, dv2_mag = float(np.linalg.norm(dv1)), float(np.linalg.norm(dv2))
    checks["fuel_budget"] = _mk_check(
        "fuel_budget", "m/s",
        "cumulative delta-v margin against the fixed budget at t0 and after each burn",
        np.array([0.0, m1.epoch_s, m2.epoch_s]),
        np.array([cfg.delta_v_budget_m_s, cfg.delta_v_budget_m_s - dv1_mag,
                  cfg.delta_v_budget_m_s - dv1_mag - dv2_mag]),
    )
    w = cfg.windows
    checks["time_window"] = _mk_check(
        "time_window", "s",
        "gate within drift_max_s; tof within its fixed window (one margin per event)",
        np.array([m1.epoch_s, m2.epoch_s, m2.epoch_s]),
        np.array([w.drift_max_s - m1.epoch_s, w.transfer_tof_max_s - plan.tof_s,
                  plan.tof_s - w.transfer_tof_min_s]),
    )
    passive = (
        _passive_release(cfg, "no_burn_at_t0", 0.0, r0c, v0c, r0d, v0d),
        _passive_release(cfg, "gate_abort_no_burn", float(m1.epoch_s), rt1, vt1, rc1, vc1),
        _passive_release(cfg, "hold_point_coast", float(m2.epoch_s), rt2, vt2, rc2, vc2 + dv2),
    )
    return SafetyReport(mission_id=cfg.mission_id, frame=frame, checks=checks, passive=passive,
                        plan_digest=rendezvous_plan_digest(cfg, plan),
                        config_sha256=canonical_config_sha256(cfg))



def candidate_safety_reason(cfg: RendezvousMissionConfig, sol: dict) -> str | None:
    """Feasibility screen on one solved candidate transfer inside the
    golden-section search: the budget, the transfer-leg coast trajectory
    (the drift leg is mission-invariant and screened once around the plan).
    Returns a violation reason string, or None if the candidate is safe."""
    if sol["total_m_s"] > cfg.delta_v_budget_m_s:
        return (
            f"fuel_budget: candidate transfer costs {sol['total_m_s']:.2f} m/s, over the fixed "
            f"budget {cfg.delta_v_budget_m_s:.2f} m/s"
        )
    s = cfg.safety
    ts = _flight_grid(0.0, sol["tof"], s.check_sample_dt_s)
    leg = propagate_pair(
        sol["chief_t1_r"], sol["chief_t1_v"], sol["deputy_t1_r"], sol["v1_abs"], ts,
        frame=cfg.dynamics.frame, mu=cfg.dynamics.mu,
        j2=EARTH_J2 if cfg.dynamics.j2 else 0.0,
        where="plan_rendezvous_safe: candidate transfer safety screen",
    )
    checks = evaluate_trajectory_safety(
        cfg, ts + sol["t1"], leg.rel_lvlh[0, :, :3], leg.rel_lvlh[0, :, 3:]
    )
    for name in TRAJECTORY_CHECKS:
        c = checks[name]
        if not c.passed:
            return (
                f"{name}: margin {c.min_margin:+.2f} {c.unit} at t={c.worst_epoch_s:.1f} s "
                f"({c.violation_count} violated samples)"
            )
    return None


@dataclass(frozen=True)
class SafetyPlanResult:
    """The E12-04 acceptance record: the constrained-optimal plan, its
    E12-03 arrival verification, and the full-trajectory safety report."""

    plan: RendezvousPlan
    verification: RendezvousVerification
    safety: SafetyReport

    @property
    def passed(self) -> bool:
        return self.verification.passed and self.safety.passed

    @property
    def plan_digest(self) -> str:
        """The E12-08 identity of the plan this acceptance record qualified."""
        return self.safety.plan_digest

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "verification": self.verification.to_dict(),
            "safety": self.safety.to_dict(),
            "plan_digest": self.plan_digest,
            "passed": self.passed,
        }


def plan_rendezvous_safe(cfg: RendezvousMissionConfig) -> SafetyPlanResult:
    """Plan the transfer minimizing delta-v *subject to* every configured
    safety constraint holding on the whole trajectory. A coarse scan over
    the fixed time-of-flight window finds the feasible basins (the safety
    screen makes the feasible set non-convex), the best scan point is
    polished by golden-section search inside its bracket, and the chosen
    plan is re-verified independently -- drift leg and passive releases
    included. When every scanned time-of-flight violates a constraint the
    search raises with the collected per-constraint reasons: plan safe, or
    say exactly why not."""
    if cfg.safety is None:
        raise RendezvousConfigError(
            f"{cfg.mission_id}: plan_rendezvous_safe needs a config with a 'safety' block"
        )
    hook = lambda sol: candidate_safety_reason(cfg, sol)
    t1, rt1, vt1, rc1, vc1 = _gate_crossing(cfg)
    gate = (t1, rt1, vt1, rc1, vc1)
    lo, hi = cfg.windows.transfer_tof_min_s, cfg.windows.transfer_tof_max_s
    ts = np.linspace(lo, hi, SCAN_INTERVALS + 1)
    scanned: list[tuple[float, float, dict]] = []
    failures: list[str] = []
    for tv in ts:
        cost, sol = _cost(cfg, gate, float(tv), hook)
        if sol is not None and "reason" in sol:
            failures.append(sol["reason"])
        elif sol is not None:
            scanned.append((float(tv), cost, sol))
    if not scanned:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: no safe transfer in the window [{lo:.0f}, {hi:.0f}] s; every "
            f"candidate violates a fixed safety constraint. First reasons: "
            f"{' | '.join(failures[:3])}"
        )
    i = min(range(len(scanned)), key=lambda k: scanned[k][1])
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, SCAN_INTERVALS)]
    ref_tof, ref_sol, _ = _golden_bracket(cfg, gate, float(a), float(b), hook)
    if ref_sol is None:
        ref_tof, _, ref_sol = scanned[i]
    plan = _plan_from_solution(cfg, t1, ref_tof, ref_sol)
    verification = verify_plan(cfg, plan)
    report = verify_plan_safety(cfg, plan)
    if not verification.passed or not report.passed:
        bad = report.failed_checks + tuple(p.kind for p in report.passive if not p.passed)
        raise RendezvousPlanError(
            f"{cfg.mission_id}: the constrained optimum still violates {list(bad)} -- "
            f"the fixed windows/budget/gate geometry leaves no safe plan"
        )
    return SafetyPlanResult(plan=plan, verification=verification, safety=report)
