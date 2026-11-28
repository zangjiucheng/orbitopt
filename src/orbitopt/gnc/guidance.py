"""Feedback guidance on the E13-01 seam: CW mid-course correction and
re-targeting (E13-04).

The law implements :class:`orbitopt.gnc.feed.GuidanceLaw` -- one
``Estimate`` in, one commanded inertial delta-v out -- and answers the
closed-loop question an open-loop plan cannot: *given what navigation knows
now, what must the next burn be?* Three scheduled maneuvers, decided from
the estimate and the reference (planned) ephemeris only:

* a **planned** burn at the gate: the commanded vector of the open-loop
  solution, unchanged -- the closed loop starts from the same plan the
  open-loop campaign flew, and its errors are left for the correction;
* a **re-target** (mid-course correction) mid-transfer: the two-point CW
  intercept from the current estimate to the terminal condition;
* a final **re-target** a declared lead time before arrival: the same
  intercept, re-solved, so arrival burns the estimate, not the plan.

Everything the law knows beyond the estimate is STATIC PLAN DATA: the
nominal chief reference ephemeris (from which the LVLH frame and the
terminal hold-point are built), the schedule, the per-burn cap, the
correction budget and a deadband. No plant handle, no ledger, no clock --
:func:`orbitopt.gnc.feed.run_guidance_step` is the only call site, and this
module names the reference motion "reference", never the word below
(checked by ``tests/test_e13_04_closed_loop_guidance.py``, mirroring
``tests/test_e13_01_truth_isolation.py``): a law that cannot name the plant
state cannot read it.

The CW two-point solve is
``dv_lvlh = Phi_12(tau)^-1 (x_f - Phi_11(tau) x_hat)`` at the fire epoch,
with ``x_f = (rho_f, 0)`` the hold-point condition at the fixed terminal
epoch. ``Phi_12`` loses rank near integer half-revolutions, so the solve is
guarded: an ill-conditioned or budget-breaking solve commands the zero
vector -- coasting is a decision the law records by refusing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.relative.cw import cw_state_transition, mean_motion_circular
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci

from .contracts import Estimate, GNCContractError

#: Modes a scheduled maneuver can take. ``planned`` fires the open-loop
#: commanded vector unchanged; ``retarget`` re-solves from the estimate.
MANEUVER_MODES = ("planned", "retarget")


class GuidanceLawError(GNCContractError):
    """A guidance-law declaration is malformed, or a solve was refused for a
    declared reason (ill-conditioning, budget)."""


@dataclass(frozen=True)
class ReferenceEphemeris:
    """The nominal chief motion the guidance frame is built on: chief
    position/velocity sampled on a common time grid from the run's t0,
    propagated by the mission's PLANNING dynamics (the same force model the
    plan was solved under -- here two-body + J2, no campaign perturbations).

    :meth:`state_at` interpolates linearly between knots: the position
    error of a chord through a 10 s arc is ~a*dt^2/8 ~ 100 m at LEO, i.e. a
    frame tilt of ~15 urad and a terminal-condition error of a few
    centimetres at kilometre levers -- orders of magnitude below the
    declared navigation error, and honest: this is the ephemeris the ground
    plan would actually uplink.
    """

    times_s: tuple
    chief_r_m: tuple
    chief_v_m_s: tuple
    frame: str
    mu_m3_s2: float = 3.9860044150e14

    def __post_init__(self):
        mu = float(self.mu_m3_s2)
        if not math.isfinite(mu) or mu <= 0.0:
            raise GuidanceLawError(f"ReferenceEphemeris.mu_m3_s2 must be > 0, got {self.mu_m3_s2!r}")
        object.__setattr__(self, "mu_m3_s2", mu)
        t = tuple(float(x) for x in self.times_s)
        if len(t) < 2:
            raise GuidanceLawError("ReferenceEphemeris needs at least two knots")
        if t[0] != 0.0 or any(b <= a for a, b in zip(t, t[1:])):
            raise GuidanceLawError("ReferenceEphemeris times must start at 0 and increase")
        rr = np.asarray(self.chief_r_m, dtype=np.float64)
        vv = np.asarray(self.chief_v_m_s, dtype=np.float64)
        if rr.shape != (len(t), 3) or vv.shape != (len(t), 3):
            raise GuidanceLawError("ReferenceEphemeris needs (n_times, 3) position and velocity")
        if not (np.all(np.isfinite(rr)) and np.all(np.isfinite(vv))):
            raise GuidanceLawError("ReferenceEphemeris states must be finite")
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "chief_r_m", rr)
        object.__setattr__(self, "chief_v_m_s", vv)

    def state_at(self, t_s: float) -> tuple:
        """Chief position and velocity at ``t_s`` (clamped to the table)."""
        t = float(t_s)
        if not math.isfinite(t):
            raise GuidanceLawError(f"state_at({t_s!r}): not finite")
        t = min(max(t, self.times_s[0]), self.times_s[-1])
        i = int(np.searchsorted(self.times_s, t)) - 1
        i = min(max(i, 0), len(self.times_s) - 2)
        t0, t1 = self.times_s[i], self.times_s[i + 1]
        w = (t - t0) / (t1 - t0)
        r = (1.0 - w) * self.chief_r_m[i] + w * self.chief_r_m[i + 1]
        v = (1.0 - w) * self.chief_v_m_s[i] + w * self.chief_v_m_s[i + 1]
        return r, v

    def to_document(self) -> dict:
        return {"kind": "orbitopt.gnc.reference-ephemeris-1.0", "frame": self.frame,
                "times_s": [float(x) for x in self.times_s],
                "chief_r_m": [[float(x) for x in r] for r in self.chief_r_m],
                "chief_v_m_s": [[float(x) for x in v] for v in self.chief_v_m_s]}

    @property
    def mean_motion(self) -> float:
        """Circular mean motion at the table's reference radius (the CW ``n``
        the terminal condition and the intercept are priced with)."""
        return mean_motion_circular_from(self)


def mean_motion_circular_from(ephemeris: ReferenceEphemeris) -> float:
    return mean_motion_circular(ephemeris.mu_m3_s2, float(np.linalg.norm(ephemeris.chief_r_m[0])))


@dataclass(frozen=True)
class ManeuverPlan:
    """One scheduled maneuver: the earliest fire epoch (the law fires at the
    first guidance tick at or after it), its mode, the planned commanded
    vector (``planned`` mode only), and its magnitude cap."""

    label: str
    t_fire_s: float
    mode: str
    dv_planned_eci_m_s: tuple = (0.0, 0.0, 0.0)
    dv_cap_m_s: float = 12.0
    #: weight of the terminal-velocity block over the terminal-position block
    #: in ``retarget`` mode (0 = position-only aiming; a final burn wants >1)
    velocity_weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise GuidanceLawError(f"ManeuverPlan label must be non-empty, got {self.label!r}")
        t = float(self.t_fire_s)
        if not math.isfinite(t) or t < 0.0:
            raise GuidanceLawError(f"ManeuverPlan({self.label}).t_fire_s must be finite >= 0, "
                                   f"got {self.t_fire_s!r}")
        if self.mode not in MANEUVER_MODES:
            raise GuidanceLawError(f"ManeuverPlan({self.label}).mode {self.mode!r} not in "
                                   f"{MANEUVER_MODES}")
        dv = tuple(float(x) for x in self.dv_planned_eci_m_s)
        if len(dv) != 3 or not all(math.isfinite(x) for x in dv):
            raise GuidanceLawError(f"ManeuverPlan({self.label}).dv_planned_eci_m_s needs 3 "
                                   f"finite numbers, got {self.dv_planned_eci_m_s!r}")
        if self.mode == "planned" and not any(x != 0.0 for x in dv):
            raise GuidanceLawError(f"ManeuverPlan({self.label}): planned mode needs a non-zero "
                                   "commanded vector")
        cap = float(self.dv_cap_m_s)
        if not math.isfinite(cap) or cap <= 0.0:
            raise GuidanceLawError(f"ManeuverPlan({self.label}).dv_cap_m_s must be > 0, "
                                   f"got {self.dv_cap_m_s!r}")
        vw = float(self.velocity_weight)
        if not math.isfinite(vw) or vw < 0.0:
            raise GuidanceLawError(f"ManeuverPlan({self.label}).velocity_weight must be finite "
                                   f">= 0, got {self.velocity_weight!r}")
        object.__setattr__(self, "t_fire_s", t)
        object.__setattr__(self, "dv_planned_eci_m_s", dv)
        object.__setattr__(self, "dv_cap_m_s", cap)
        object.__setattr__(self, "velocity_weight", vw)


def _num(name: str, value, *, positive: bool = True) -> float:
    v = float(value)
    if not math.isfinite(v) or (v <= 0.0 if positive else v < 0.0):
        raise GuidanceLawError(f"{name} must be finite and {'> 0' if positive else '>= 0'}, "
                               f"got {value!r}")
    return v


class CwMidCourseLaw:
    """The E13-04 feedback law: coast outside the schedule; fire each
    scheduled maneuver once, at the first guidance tick at or after its fire
    epoch, using the estimate that tick delivered.

    ``terminal_rho_lvlh_m`` is the hold-point relative position at the fixed
    terminal epoch ``t_terminal_s`` (relative to the REFERENCE chief, LVLH),
    and the terminal relative velocity is zero. ``budget_m_s`` caps the
    cumulative commanded delta-v the law will issue (its own ledger: the law
    cannot see what the actuator delivered -- that honesty is the point of
    the seam; realized-versus-commanded accounting is offline).

    State is per-law (fired flags, the command ledger): one instance flies
    one sample, never reused.
    """

    def __init__(self, *, maneuvers: tuple, reference: ReferenceEphemeris,
                 t_terminal_s: float, terminal_rho_lvlh_m, budget_m_s: float,
                 deadband_m_s: float = 0.01, ill_condition_rcond: float = 1.0e6):
        self.maneuvers = tuple(maneuvers)
        if not self.maneuvers:
            raise GuidanceLawError("CwMidCourseLaw needs at least one scheduled maneuver")
        labels = [m.label for m in self.maneuvers]
        if len(set(labels)) != len(labels):
            raise GuidanceLawError(f"duplicate maneuver labels {labels}")
        self.maneuvers = tuple(sorted(self.maneuvers, key=lambda m: m.t_fire_s))
        if not isinstance(reference, ReferenceEphemeris):
            raise GuidanceLawError(f"reference must be a ReferenceEphemeris, got {reference!r}")
        self.reference = reference
        self.t_terminal_s = _num("t_terminal_s", t_terminal_s)
        rho_f = np.asarray(terminal_rho_lvlh_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rho_f)):
            raise GuidanceLawError("terminal_rho_lvlh_m must be finite")
        self.terminal_rho_lvlh_m = rho_f
        self.budget_m_s = _num("budget_m_s", budget_m_s)
        self.deadband_m_s = _num("deadband_m_s", deadband_m_s, positive=False)
        self.rcond_max = _num("ill_condition_rcond", ill_condition_rcond)
        self.n = reference.mean_motion
        self._fired: dict = {}
        self._phase_now = "coast"
        self._commanded_m_s = 0.0
        self.refusals: list = []

    # -- GuidanceLaw ----------------------------------------------------------

    def delta_v(self, estimate: Estimate) -> tuple:
        """The one call: coast unless this estimate has reached a scheduled
        maneuver that has not fired yet."""
        if not isinstance(estimate, Estimate):
            raise TypeError(f"CwMidCourseLaw.delta_v takes an Estimate, got {estimate!r}")
        now = float(estimate.t_update_s)
        self._phase_now = "coast"
        for m in self.maneuvers:
            if m.label in self._fired or now + 1.0e-9 < m.t_fire_s:
                continue
            dv = self._decide(m, estimate, now)
            self._fired[m.label] = True
            self._commanded_m_s += float(np.linalg.norm(dv))
            self._phase_now = m.label
            return dv
        return (0.0, 0.0, 0.0)

    def phase_for(self, estimate: Estimate) -> str:
        """The maneuver this tick decided (or ``coast``) -- command metadata
        for the ledger, read by the simulator, never by the law's input."""
        return self._phase_now

    @property
    def commanded_total_m_s(self) -> float:
        """What the law has ASKED for so far (the ledger the seam lets it
        keep; realized accounting happens offline from the run log)."""
        return self._commanded_m_s

    # -- internals ------------------------------------------------------------

    def _decide(self, m: ManeuverPlan, estimate: Estimate, now: float) -> tuple:
        if m.mode == "planned":
            return self._budgeted(m, np.asarray(m.dv_planned_eci_m_s, dtype=np.float64))
        cr, cv = self.reference.state_at(now)
        rho, rho_dot = eci_to_lvlh(cr, cv,
                                   np.asarray(estimate.relative_state.r_rel_m, dtype=np.float64),
                                   np.asarray(estimate.relative_state.v_rel_m_s, dtype=np.float64),
                                   estimate.relative_state.frame,
                                   where=f"CwMidCourseLaw({m.label})")
        tau = self.t_terminal_s - now
        if tau <= 0.0:
            self.refusals.append(f"{m.label}: terminal epoch {self.t_terminal_s} already past "
                                 f"at t={now}")
            return (0.0, 0.0, 0.0)
        dv_lvlh = solve_cw_intercept(self.n, tau, np.concatenate((rho, rho_dot)),
                                     np.concatenate((self.terminal_rho_lvlh_m,
                                                     np.zeros(3))),
                                     velocity_weight=m.velocity_weight,
                                     rcond_max=self.rcond_max)
        if dv_lvlh is None:
            self.refusals.append(f"{m.label}: intercept normal matrix ill-conditioned at "
                                 f"tau={tau:.1f}s -- coasted")
            return (0.0, 0.0, 0.0)
        _dr, dv_eci = lvlh_to_eci(cr, cv, np.zeros(3), dv_lvlh,
                                      estimate.relative_state.frame,
                                      where=f"CwMidCourseLaw({m.label}) egress")
        if float(np.linalg.norm(dv_eci)) < self.deadband_m_s:
            return (0.0, 0.0, 0.0)  # on solution already: a burn under the deadband is noise
        return self._budgeted(m, dv_eci)

    def _budgeted(self, m: ManeuverPlan, dv: np.ndarray) -> tuple:
        mag = float(np.linalg.norm(dv))
        if mag > m.dv_cap_m_s:
            self.refusals.append(f"{m.label}: |dv|={mag:.3f} m/s over cap {m.dv_cap_m_s} m/s "
                                 "-- coasted")
            return (0.0, 0.0, 0.0)
        if self._commanded_m_s + mag > self.budget_m_s:
            self.refusals.append(f"{m.label}: cumulative commanded {self._commanded_m_s + mag:.3f}"
                                 f" m/s over budget {self.budget_m_s} m/s -- coasted")
            return (0.0, 0.0, 0.0)
        return tuple(float(x) for x in dv)


def solve_cw_intercept(n: float, tau_s: float, x_now, x_terminal, *,
                       velocity_weight: float = 1.0,
                       rcond_max: float = 1.0e6) -> np.ndarray | None:
    """The CW delta-v that aims LVLH state ``x_now`` at ``x_terminal`` in
    ``tau_s`` seconds, in the terminal-weighted least-squares sense: the
    three-impulse unknown cannot meet six terminal conditions exactly at one
    epoch, so the burn minimizes ``|W (xf - Phi (x0 + B dv))|`` with the
    velocity block weighted ``velocity_weight`` over the position block.
    Position-only aiming (weight 0 on velocity rows) leaves the closing speed
    intact at arrival; a terminal burn wants the velocity rows to dominate --
    with a large ``velocity_weight`` and small ``tau_s`` this degenerates to
    the classic velocity-nulling final burn. ``None`` when the normal matrix
    is singular or ill-conditioned -- the caller decides what coasting means."""
    n = _num("n", n)
    tau = _num("tau_s", tau_s)
    vw = _num("velocity_weight", velocity_weight, positive=False)
    x0 = np.asarray(x_now, dtype=np.float64).reshape(6)
    xf = np.asarray(x_terminal, dtype=np.float64).reshape(6)
    if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(xf))):
        raise GuidanceLawError("intercept states must be finite")
    phi = cw_state_transition(n, tau)
    a = phi[:, 3:]
    resid = xf - phi @ x0
    w = np.array([1.0, 1.0, 1.0, vw, vw, vw])
    w2 = w * w
    m = a.T @ (w2[:, None] * a)
    cond = float(np.linalg.cond(m))
    if not math.isfinite(cond) or cond > rcond_max:
        return None
    return np.linalg.solve(m, a.T @ (w2 * resid))
