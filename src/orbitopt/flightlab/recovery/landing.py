"""FL-303: the G-FOLD landing burn -- convex min-fuel solve, honest verdict.

The landing-burn guidance of Flight Lab E3: a minimum-fuel powered-descent
trajectory from the FL-302 corridor-insertion state (pad-local ENU) to an
engine-cutoff point just above the pad, solved as a second-order cone
program (lossless convexification, Acikmese & Ploen) and *verified* the only
way E3 accepts: re-flown through the FL-301 3-DOF propagator with drag, J2
gravity, the rotating Earth and the throttle/ambient-thrust reality of the
vehicle model. The landing verdict comes from that re-integration, never
from the solver (G3: a convex model that cannot see drag must not sign its
own gradesheet).

The arc ends at *cutoff*, not at the pad: this vehicle's min-throttle thrust
(~520 kN at ambient) exceeds its weight at every mass, so no throttle
setting hovers it and a plan whose last action is thrusting down onto the
pad is unflyable at its lower end. The convex program steers to a slow
descent a few tenths of a metre up; the engine shuts down there and the
vehicle coasts the last step in ballistically (touchdown speed
``sqrt(v_cut^2 + 2 g h_cut)`` plus guidance error, graded by the checks).

Two solvers, one discretization:

``gfold`` (primary)
    ``gfold`` 0.3.2 (MIT; Rust core, Clarabel SOCP; pip wheel), pinned in
    ``envs/flightlab-landing.txt``. The wrapper below pins its exact 0.3.2
    semantics, verified against the shipped binary: first-order-hold (FOH)
    discretization on ``n`` nodes with ``h = tof / n`` (last sample at
    ``t0 + tof*(n-1)/n`` -- that node is the terminal state), acceleration
    command ``u = T / m_k`` against a fixed-point mass schedule ``m_k`` (so
    the *reported* thrusts respect the bounds while the dynamics stay linear),
    trapezoidal fuel accounting, and ``min_thrust_pct`` / ``max_thrust_pct``
    understood as *fractions* of ``real_max_thrust`` (0.4 / 1.0 -- passing
    percentages makes the Rust core report ``NumericalError``), and node
    times reported from 0 (the wrapper re-bases them to the mission clock).
    Caveat pinned by test: ``Environment.max_angle_deg`` is inert in 0.3.2
    (verified -- a 5 deg limit leaves the trajectory untouched), so the tilt
    limit is carried as a plan constraint enforced by the cvxpy reference and
    graded afterwards by the touchdown checks, never trusted to gfold.

``cvxpy`` + Clarabel (reference, cross-check)
    The same FOH discretization, the same fixed-point mass schedule, written
    out as an explicit SOCP (this module owns the model; no black box is its
    own cross-check). ``solve_landing_burn(..., solver="cvxpy")`` runs it;
    the agreement test asserts the two solvers land within the declared
    tolerances on three reference cases.

The G-FOLD model itself is the classical one -- constant gravity, flat
ground, no aero, point mass -- so its plan is re-integrated, never flown
blind: :func:`reintegrate_landing` maps the solved thrust vectors into the
FL-301 propagator (ENU -> ECEF through the pad basis, thrust magnitude ->
throttle through the *ambient* thrust at the flown altitude, min-throttle
clamping included), and :func:`landing_touchdown_checks` grades the
re-flown track: position error, vertical / lateral touchdown speed, thrust
tilt proxy, propellant above the vehicle reserve, max proper-g -- each an
independent :class:`~orbitopt.flightlab.contract.Check` with its own reason
code. A plan the SOCP cannot produce yields ``LANDING_INFEASIBLE`` and no
green anything; a plan that solves but lands hard or crooked after drag is
reported failed. ``fly_landing_burn`` ties it together and
:func:`landing_result_to_run` writes the booster body and the
``landing_start`` / ``touchdown`` / ``impact`` events into a FlightRun.

G3 note (acceptance criterion 2): the flat model and the real atmosphere
disagree exactly where the atmosphere lives. Drag decays horizontal velocity
faster than the flat plan expects, so a strongly-offset corridor state lands
short of the pad; and near the ground the ambient thrust is lower than the
insertion-altitude envelope the SOCP was solved against, so the last few
seconds under-thrust. Both failures show up in the re-integration verdict --
which is the point of re-integrating.

FL-305 (``docs/adr/flightlab-0002-closed-loop-landing-guidance.md``) adds
:func:`fly_landing_burn_closed_loop`: the same insertion SOCP as the
feasibility certificate, but the burn flown by :class:`TerminalGuidance`, a
throttle-limited hoverslam + ZEM/ZEV divert re-planned from the propagated
state every step, with the tilt limit in the loop and a final cutoff tuned
to the no-hover thrust-to-weight. The referee is unchanged:
:func:`landing_touchdown_checks` on the propagated track.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from orbitopt.entry import (G0_M_S2, OMEGA_EARTH_RAD_S, density_kg_m3,
                            height_m)
from orbitopt.flightlab.contract import Check, Event, FlightRun, Verification
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery.boostback import (
    LandingBurnEnu,
    RecoveryTarget,
    _enu_basis,
)
from orbitopt.flightlab.recovery.propagator import (
    BoosterModel,
    BoosterPhysics,
    BoosterState,
    BoosterTrack,
    RecoveryError,
    ThrustCmd,
    _gravity_ecef,
    pressure_pa,
    propagate,
)

#: The exact ``gfold`` release this wrapper's semantics were pinned against
#: (pip wheel; MIT; linux-x86_64 and macOS-arm64). ``envs/flightlab-landing.txt``
#: pins the same version; ``solve_landing_burn`` refuses a different one.
GFOLD_VERSION_PIN = "0.3.2"

# ---------------------------------------------------------------------------
# reason codes (frozen vocabulary, same style as boostback's)
# ---------------------------------------------------------------------------

#: The SOCP found no feasible powered-descent arc in the time-of-flight
#: bracket (or not at the fixed time): there is no plan to re-integrate, so
#: the result carries no green flags at all.
LANDING_INFEASIBLE = "LANDING_INFEASIBLE"
#: Re-integrated touchdown point outside the acceptance radius.
LANDING_POSITION_OUT_OF_TOLERANCE = "LANDING_POSITION_OUT_OF_TOLERANCE"
#: Vertical speed at touchdown above the soft-landing limit.
LANDING_VERTICAL_SPEED_EXCEEDED = "LANDING_VERTICAL_SPEED_EXCEEDED"
#: Horizontal (lateral) speed at touchdown above the limit.
LANDING_LATERAL_SPEED_EXCEEDED = "LANDING_LATERAL_SPEED_EXCEEDED"
#: Thrust vector tilted further from vertical (during the burn) than the
#: plan's tilt limit -- the tilt *proxy* (thrust direction), graded on the
#: re-integrated command history because gfold 0.3.2 does not enforce it.
LANDING_TILT_EXCEEDED = "LANDING_TILT_EXCEEDED"
#: Propellant left at touchdown below the vehicle's landing reserve.
LANDING_PROPELLANT_RESERVE_LOW = "LANDING_PROPELLANT_RESERVE_LOW"
#: Max proper-g on the re-flown burn above the vehicle limit.
LANDING_MAX_G_EXCEEDED = "LANDING_MAX_G_EXCEEDED"
#: The re-integration never reached the ground inside the burn window (the
#: plan hovered out / fell past the horizon without touching down).
LANDING_NO_TOUCHDOWN = "LANDING_NO_TOUCHDOWN"

TOUCHDOWN_CHECK_IDS = (
    "landing-position",
    "landing-vertical-speed",
    "landing-lateral-speed",
    "landing-tilt",
    "landing-propellant-reserve",
    "landing-max-g",
)

_CHECK_REASON = {
    "landing-position": LANDING_POSITION_OUT_OF_TOLERANCE,
    "landing-vertical-speed": LANDING_VERTICAL_SPEED_EXCEEDED,
    "landing-lateral-speed": LANDING_LATERAL_SPEED_EXCEEDED,
    "landing-tilt": LANDING_TILT_EXCEEDED,
    "landing-propellant-reserve": LANDING_PROPELLANT_RESERVE_LOW,
    "landing-max-g": LANDING_MAX_G_EXCEEDED,
}


class LandingInfeasibleError(RecoveryError):
    """Raised by the low-level solvers when the SOCP finds no feasible arc.
    The orchestrating :func:`fly_landing_burn` catches it and reports
    ``LANDING_INFEASIBLE`` instead; a RecoveryError subclass so callers
    already guarding the propagator guard this too."""


# ---------------------------------------------------------------------------
# limits and the solved-plan record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandingLimits:
    """Soft-landing acceptance, graded against the *re-integrated* track.
    Numbers illustrative for the synthetic vehicle (Falcon-class targets:
    ~2 m/s vertical, <1 m/s lateral, pads take a few tens of metres of
    placement error)."""

    position_tol_m: float = 25.0
    vertical_speed_max_m_s: float = 3.0
    lateral_speed_max_m_s: float = 1.5
    #: Max angle of the *commanded* thrust vector from vertical, deg, over
    #. the re-integrated burn samples.
    max_tilt_deg: float = 40.0
    max_proper_g: float = 5.0


#: Default acceptance for the synthetic RTLS case.
DEFAULT_LANDING_LIMITS = LandingLimits()


@dataclass(frozen=True)
class LandingPlan:
    """A solved landing-burn arc in pad-local ENU: the FOH node grid (mission
    clock), the *bounded physical* thrust vectors (N) at each node -- the
    command to fly, within [T_min, T_max] by construction -- the constant
    gravity and trapezoidal fuel accounting the solver used, and the
    losslessness diagnostic of the cvxpy reference (gfold reports no gap of
    its own; its ``Solved`` status is recorded in ``status``)."""

    solver: str
    status: str
    t0_s: float
    times_s: tuple[float, ...]
    position_enu_m: tuple[tuple[float, float, float], ...]
    velocity_enu_m: tuple[tuple[float, float, float], ...]
    thrust_enu_n: tuple[tuple[float, float, float], ...]
    mass_schedule_kg: tuple[float, ...]
    tof_commanded_s: float
    tof_flying_s: float
    gravity_m_s2: float
    glide_slope_deg: float
    tilt_limit_deg: float
    fuel_planned_kg: float
    final_mass_kg: float
    lossless_gap: float | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.times_s)


@dataclass(frozen=True)
class LandingResult:
    """The FL-303 verdict: the plan (``None`` when infeasible), the
    re-integrated booster track, the touchdown checks, and the two boolean
    gates -- ``feasible`` (the SOCP produced a plan) and ``landed`` (the
    re-flown track touched down inside every limit). ``reason_codes[0]`` is
    the governing reason of the failure; ``feasible and not landed`` is the
    G3 case: convex model happy, real atmosphere not."""

    feasible: bool
    landed: bool
    reason_codes: tuple[str, ...]
    checks: tuple[Check, ...]
    plan: LandingPlan | None = None
    track: BoosterTrack | None = None
    impact: Any | None = None
    touchdown_enu_m: tuple[float, float, float] | None = None
    vertical_speed_m_s: float | None = None
    lateral_speed_m_s: float | None = None
    position_error_m: float | None = None
    max_tilt_deg: float | None = None
    max_proper_g: float = 0.0
    ignitions_used: int = 0
    assumptions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# environment seen by the convex model
# ---------------------------------------------------------------------------


def _enu_axes(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e, n, u = _enu_basis(lat_deg, lon_deg)
    return np.array(e), np.array(n), np.array(u)


def _pad_of_target(target: RecoveryTarget) -> tuple[float, float, float, np.ndarray]:
    """(lat_deg, lon_deg, alt_m, ecef_m) of the pad. The pad reference is
    FL-302's ``RecoveryTarget``, never the insertion state's own geodetic
    site: the corridor state sits up to the corridor radius away from the
    pad, so its geodetic lat/lon is kilometres off and its height is
    pad-altitude + up-component."""
    ecef = np.asarray(geodetic_to_ecef(
        float(target.latitude_deg), float(target.longitude_deg),
        float(target.altitude_m)), dtype=np.float64)
    return (float(target.latitude_deg), float(target.longitude_deg),
            float(target.altitude_m), ecef)


def _ambient_thrust_envelope(vehicle: BoosterModel, h_m: float) -> tuple[float, float]:
    """(T_max, T_min) [N] of the landing engine group at geodetic altitude
    ``h_m`` -- the same pessimistic ambient-thrust floor (60 % of vacuum)
    :func:`orbitopt.flightlab.recovery.boostback.landing_burn_propellant_need_kg`
    uses, so corridor accounting and the convex plan agree."""
    eng = vehicle.engine
    t_max = eng.count * (eng.thrust_vacuum_n
                         - eng.nozzle_exit_area_m2 * pressure_pa(h_m))
    floor = 0.6 * eng.count * eng.thrust_vacuum_n
    if t_max < floor:
        t_max = max(t_max, floor)
    return t_max, eng.throttle_min * t_max


def _local_gravity(r_ecef: Sequence[float]) -> float:
    """Effective |g| [m/s^2] (J2 gravity + centrifugal) at the insertion
    point -- the constant gravity the flat model substitutes for the real
    field. The centrifugal term is included so the plan matches what the
    rotating FL-301 propagator feels (its ``-Omega x (Omega x r)``); at
    ~28.6 deg N it is a 0.026 m/s^2 vertical effect that a ~20 s burn
    integrates into metres of cutoff-altitude error."""
    r = np.asarray(r_ecef, dtype=np.float64)
    a = _gravity_ecef(r, "j2") + OMEGA_EARTH_RAD_S ** 2 * np.array([r[0], r[1], 0.0])
    return float(np.linalg.norm(a))


# ---------------------------------------------------------------------------
# solver front ends
# ---------------------------------------------------------------------------


def _require_gfold():
    try:
        import gfold
        from importlib.metadata import version

        got = version("gfold")
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise LandingInfeasibleError(
            "gfold is not installed: pip install -r envs/flightlab-landing.txt"
        ) from exc
    if got != GFOLD_VERSION_PIN:
        raise LandingInfeasibleError(
            f"gfold {got} != pinned {GFOLD_VERSION_PIN}: this wrapper pins the "
            "0.3.2 discretization/bound semantics; install the pinned version"
        )
    return gfold


def _gfold_plan(gfold, state: LandingBurnEnu, vehicle: BoosterModel, *,
                t_max_n: float, t_min_n: float, gravity: float, n: int,
                tof_s: float | None, tof_min_s: float, tof_max_s: float,
                glide_slope_deg: float, tilt_limit_deg: float,
                max_speed_m_s: float, cutoff_height_m: float,
                cutoff_descent_m_s: float) -> LandingPlan:
    eng = vehicle.engine
    fc = 1.0 / (eng.isp_sea_level_s * G0_M_S2)
    # Terminal state = engine *cutoff* point, not touchdown: with min-throttle
    # thrust above vehicle weight (this vehicle's floor is ~520 kN ambient vs
    # ~440 kN at wet mass) the engine cannot hover, so the convex arc ends at
    # a slow descent a short distance above the pad and the vehicle coasts the
    # rest of the way in ballistically (see module docstring).
    sc = gfold.Spacecraft(
        wet_mass=float(state.mass_kg),
        fuel=max(float(state.mass_kg) - float(vehicle.dry_mass_kg), 1e-6),
        real_max_thrust=t_max_n,
        min_thrust_pct=t_min_n / t_max_n,  # fractions, not percentages (0.3.2)
        max_thrust_pct=1.0,
        max_velocity=max_speed_m_s,
        initial_position=[float(c) for c in state.position_enu_m],
        initial_velocity=[float(c) for c in state.velocity_enu_m],
        target_position=[0.0, 0.0, float(cutoff_height_m)],
        target_velocity=[0.0, 0.0, -float(cutoff_descent_m_s)],
        fuel_consumption=fc,
    )
    env = gfold.Environment(
        gravity=[0.0, 0.0, -gravity],
        glide_slope_angle_deg=glide_slope_deg,
        max_angle_deg=tilt_limit_deg,  # accepted-but-inert in 0.3.2: see module docstring
    )
    if tof_s is not None:
        solver = gfold.Solver(n=n, time_of_flight=float(tof_s))
    else:
        solver = gfold.Solver(n=n, tof_min=float(tof_min_s), tof_max=float(tof_max_s))
    try:
        traj = gfold.solve(gfold.Config(spacecraft=sc, environment=env, solver=solver))
    except ValueError as exc:
        raise LandingInfeasibleError(f"gfold: {exc}") from exc
    # 0.3.2 reports node times from 0, not from the clock origin.
    times = float(state.t_s) + np.asarray(traj.time_points, dtype=np.float64)
    u = np.asarray(traj.u_values, dtype=np.float64)
    pos = np.asarray(traj.positions, dtype=np.float64)
    vel = np.asarray(traj.velocities, dtype=np.float64)
    # physical commanded thrust: the bounded reported magnitudes along the
    # command direction implied by u (the trajectory-consistent vector).
    th = np.asarray(traj.thrusts, dtype=np.float64)
    norm_u = np.linalg.norm(u, axis=1)
    dirv = u / np.where(norm_u > 0.0, norm_u, 1.0)[:, None]
    t_vec = th[:, None] * dirv
    dt = times[1] - times[0]
    m0 = float(state.mass_kg)
    final_mass = float(traj.final_mass)
    # mass schedule implied by the trapezoidal fuel accounting, for the
    # re-integration record (and the cvxpy reference's fixed-point start).
    mass = np.empty_like(times)
    mass[0] = m0
    for k in range(len(times) - 1):
        mass[k + 1] = mass[k] - fc * 0.5 * dt * (th[k] + th[k + 1])
    return LandingPlan(
        solver="gfold",
        status=str(traj.status),
        t0_s=float(state.t_s),
        times_s=tuple(float(t) for t in times),
        position_enu_m=tuple(tuple(float(c) for c in p) for p in pos),
        velocity_enu_m=tuple(tuple(float(c) for c in v) for v in vel),
        thrust_enu_n=tuple(tuple(float(c) for c in v) for v in t_vec),
        mass_schedule_kg=tuple(float(m) for m in mass),
        tof_commanded_s=float(dt * n),
        tof_flying_s=float(times[-1] - times[0]),
        gravity_m_s2=gravity,
        glide_slope_deg=glide_slope_deg,
        tilt_limit_deg=tilt_limit_deg,
        fuel_planned_kg=m0 - final_mass,
        final_mass_kg=final_mass,
    )


def _cvxpy_plan(state: LandingBurnEnu, vehicle: BoosterModel, *,
                t_max_n: float, t_min_n: float, gravity: float, n: int,
                tof_s: float, glide_slope_deg: float, tilt_limit_deg: float,
                cutoff_height_m: float, cutoff_descent_m_s: float,
                maxiter: int = 8, tol_kg: float = 0.05) -> LandingPlan:
    """FOH + lossless convexification written out explicitly -- the
    reference implementation of exactly what the gfold Rust core solves
    (module docstring), including its fixed-point mass schedule. Test /
    cross-check use."""
    import cvxpy as cp

    eng = vehicle.engine
    fc = 1.0 / (eng.isp_sea_level_s * G0_M_S2)
    m0 = float(state.mass_kg)
    p0 = np.asarray(state.position_enu_m, dtype=np.float64)
    v0 = np.asarray(state.velocity_enu_m, dtype=np.float64)
    g_vec = np.array([0.0, 0.0, -gravity])
    h = tof_s / n
    dt = h
    w = np.full(n, 1.0)
    w[0] = w[-1] = 0.5  # trapezoidal fuel weights

    t_g = math.tan(math.radians(glide_slope_deg))
    c_tilt = math.cos(math.radians(tilt_limit_deg))
    s_tilt = math.sin(math.radians(tilt_limit_deg))
    terminal_pos = np.array([0.0, 0.0, float(cutoff_height_m)])
    terminal_vel = np.array([0.0, 0.0, -float(cutoff_descent_m_s)])

    mass = np.full(n, m0)
    u_opt = s_opt = None
    status = "NotConverged"
    for _ in range(maxiter):
        u = cp.Variable((n, 3))
        s = cp.Variable(n)
        bounds = [cp.norm(u, axis=1) <= s,
                  s >= t_min_n / mass,
                  s <= t_max_n / mass,
                  u[:, 2] >= c_tilt * s,
                  cp.norm(u[:, :2], axis=1) <= s_tilt * s]
        cons = list(bounds)
        pos, vel = p0, v0
        posk, velk = [], []
        for k in range(n):
            posk.append(pos)
            velk.append(vel)
            if k < n - 1:
                a_k = g_vec + u[k]
                a_n = g_vec + u[k + 1]
                vel = vel + h * g_vec + 0.5 * h * (u[k] + u[k + 1])
                pos = pos + h * velk[k] + h * h * (a_k / 3.0 + a_n / 6.0)
        cons.append(velk[-1] == terminal_vel)
        cons.append(posk[-1] == terminal_pos)
        for k in range(n):
            cons.append(posk[k][2] >= t_g * cp.norm(posk[k][:2]))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(w * mass, s))), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ("optimal", "optimal_inaccurate") or u.value is None:
            raise LandingInfeasibleError(f"cvxpy: SOCP status {prob.status!r}")
        u_opt = np.asarray(u.value, dtype=np.float64)
        s_opt = np.asarray(s.value, dtype=np.float64)
        thrust = mass * s_opt  # physical commanded N (LCV tight at optimum)
        new_mass = np.empty(n)
        new_mass[0] = m0
        for k in range(n - 1):
            new_mass[k + 1] = new_mass[k] - fc * 0.5 * dt * (thrust[k] + thrust[k + 1])
        shift = float(np.max(np.abs(new_mass - mass)))
        mass = new_mass
        if shift <= tol_kg:
            status = "Solved"
            break
    if u_opt is None:  # pragma: no cover - loop body always assigns or raises
        raise LandingInfeasibleError("cvxpy: solver never returned")
    norm_u = np.linalg.norm(u_opt, axis=1)
    dirv = u_opt / np.where(norm_u > 0.0, norm_u, 1.0)[:, None]
    t_vec = thrust[:, None] * dirv
    # numpy replay of the FOH recursion at the optimum: the node trajectory
    # the plan flew (matches the cvxpy affine chain to roundoff).
    pos_n = np.empty((n, 3))
    vel_n = np.empty((n, 3))
    pos_n[0], vel_n[0] = p0, v0
    a = g_vec + u_opt
    for k in range(n - 1):
        vel_n[k + 1] = vel_n[k] + h * g_vec + 0.5 * h * (u_opt[k] + u_opt[k + 1])
        pos_n[k + 1] = pos_n[k] + h * vel_n[k] + h * h * (a[k] / 3.0 + a[k + 1] / 6.0)
    gap = float(np.max(np.abs(norm_u - s_opt) / np.maximum(s_opt, 1e-9)))
    fuel = fc * dt * float(np.dot(w, thrust))
    times = state.t_s + dt * np.arange(n)
    return LandingPlan(
        solver="cvxpy",
        status=status,
        t0_s=float(state.t_s),
        times_s=tuple(float(t) for t in times),
        position_enu_m=tuple(tuple(float(c) for c in p) for p in pos_n),
        velocity_enu_m=tuple(tuple(float(c) for c in v) for v in vel_n),
        thrust_enu_n=tuple(tuple(float(c) for c in v) for v in t_vec),
        mass_schedule_kg=tuple(float(m) for m in mass),
        tof_commanded_s=float(tof_s),
        tof_flying_s=float(times[-1] - times[0]),
        gravity_m_s2=gravity,
        glide_slope_deg=glide_slope_deg,
        tilt_limit_deg=tilt_limit_deg,
        fuel_planned_kg=fuel,
        final_mass_kg=m0 - fuel,
        lossless_gap=gap,
    )


def _cvxpy_tof_bracket(state: LandingBurnEnu, vehicle: BoosterModel, *,
                       t_max_n: float, t_min_n: float, gravity: float, n: int,
                       tof_min_s: float, tof_max_s: float,
                       glide_slope_deg: float, tilt_limit_deg: float,
                       cutoff_height_m: float, cutoff_descent_m_s: float,
                       iters: int = 10) -> LandingPlan:
    """Line search on time of flight for the reference solver: feasibility is
    monotone-ish in tof (more time buys slack down to the glide-slope limit),
    so bisect the first feasible tof between min (unconstrained fail-fast)
    and max (must solve or the problem is infeasible)."""
    try:
        plan = _cvxpy_plan(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                           gravity=gravity, n=n, tof_s=tof_min_s,
                           glide_slope_deg=glide_slope_deg,
                           tilt_limit_deg=tilt_limit_deg,
                           cutoff_height_m=cutoff_height_m,
                           cutoff_descent_m_s=cutoff_descent_m_s)
        return plan
    except LandingInfeasibleError:
        pass
    _cvxpy_plan(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                gravity=gravity, n=n, tof_s=tof_max_s,
                glide_slope_deg=glide_slope_deg, tilt_limit_deg=tilt_limit_deg,
                cutoff_height_m=cutoff_height_m,
                cutoff_descent_m_s=cutoff_descent_m_s)
    lo, hi = tof_min_s, tof_max_s
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        try:
            _cvxpy_plan(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                        gravity=gravity, n=n, tof_s=mid,
                        glide_slope_deg=glide_slope_deg,
                        tilt_limit_deg=tilt_limit_deg,
                        cutoff_height_m=cutoff_height_m,
                        cutoff_descent_m_s=cutoff_descent_m_s)
            hi = mid
        except LandingInfeasibleError:
            lo = mid
    return _cvxpy_plan(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                       gravity=gravity, n=n, tof_s=hi,
                       glide_slope_deg=glide_slope_deg,
                       tilt_limit_deg=tilt_limit_deg,
                       cutoff_height_m=cutoff_height_m,
                       cutoff_descent_m_s=cutoff_descent_m_s)


def solve_landing_burn(state: LandingBurnEnu, vehicle: BoosterModel,
                       *, gravity_m_s2: float | None = None,
                       pad_altitude_m: float = 0.0,
                       solver: str = "gfold",
                       tof_s: float | None = None,
                       tof_min_s: float = 8.0, tof_max_s: float = 120.0,
                       n: int = 16,
                       glide_slope_deg: float = 20.0,
                       tilt_limit_deg: float = 40.0,
                       max_speed_m_s: float = 800.0,
                       cutoff_height_m: float = 0.25,
                       cutoff_descent_m_s: float = 1.2) -> LandingPlan:
    """Solve the min-fuel landing SOCP from an FL-302 corridor state.

    ``state`` is the pad-local ENU insertion state (the pad is the target at
    the ENU origin); ``pad_altitude_m`` is the pad's own altitude (the
    up-component is measured from it). The thrust envelope is the engine
    group's ambient capability at the insertion altitude
    (:func:`_ambient_thrust_envelope`), min throttle included -- the flat
    model's other classic simplification (constant thrust) pinned to the
    corridor, not to vacuum. ``gravity_m_s2`` defaults to the local J2
    magnitude at the insertion point.

    The arc terminates at the engine *cutoff* state -- ``cutoff_height_m``
    above the pad, descending at ``cutoff_descent_m_s`` -- not at the pad
    itself: the vehicle's min-throttle thrust exceeds its weight at every
    mass (it can steer but never hover), so a convex arc ending at rest on
    the pad is unflyable at the lower end. Touchdown speed is then
    ``sqrt(cutoff_descent^2 + 2 g cutoff_height)`` plus guidance error,
    graded by the re-integration (module docstring).

    ``tof_s=None`` lets the solver pick the final time: gfold's built-in
    bracket search over ``[tof_min_s, tof_max_s]``, or bisection for the
    reference. ``solver="cvxpy"`` runs the reference implementation
    (fixed-point mass schedule, explicit FOH -- see module docstring); note
    the cvxpy reference additionally enforces the tilt cone that gfold
    0.3.2 ignores, so solver-agreement runs use ``tilt_limit_deg=90``
    (cone inactive) and a separate test pins that a real cone limit binds
    the reference. Raises :class:`LandingInfeasibleError` when no arc
    exists."""
    h0 = pad_altitude_m + float(state.position_enu_m[2])
    t_max_n, t_min_n = _ambient_thrust_envelope(vehicle, h0)
    if gravity_m_s2 is None:
        gravity = _gravity_pad_default(state, pad_altitude_m)
    else:
        gravity = float(gravity_m_s2)
    if solver == "gfold":
        gfold = _require_gfold()
        return _gfold_plan(gfold, state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                           gravity=gravity, n=n, tof_s=tof_s, tof_min_s=tof_min_s,
                           tof_max_s=tof_max_s, glide_slope_deg=glide_slope_deg,
                           tilt_limit_deg=tilt_limit_deg,
                           max_speed_m_s=max_speed_m_s,
                           cutoff_height_m=cutoff_height_m,
                           cutoff_descent_m_s=cutoff_descent_m_s)
    if solver == "cvxpy":
        if tof_s is not None:
            return _cvxpy_plan(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                               gravity=gravity, n=n, tof_s=float(tof_s),
                               glide_slope_deg=glide_slope_deg,
                               tilt_limit_deg=tilt_limit_deg,
                               cutoff_height_m=cutoff_height_m,
                               cutoff_descent_m_s=cutoff_descent_m_s)
        return _cvxpy_tof_bracket(state, vehicle, t_max_n=t_max_n, t_min_n=t_min_n,
                                  gravity=gravity, n=n, tof_min_s=tof_min_s,
                                  tof_max_s=tof_max_s,
                                  glide_slope_deg=glide_slope_deg,
                                  tilt_limit_deg=tilt_limit_deg,
                                  cutoff_height_m=cutoff_height_m,
                                  cutoff_descent_m_s=cutoff_descent_m_s)
    raise ValueError(f"solver must be 'gfold' or 'cvxpy', got {solver!r}")


def _gravity_pad_default(state: LandingBurnEnu, pad_altitude_m: float) -> float:
    """Standard-site gravity when no ECEF anchor is given: g0 (the ENU state
    alone does not say where on Earth it is; callers with the ECEF twin pass
    ``gravity_m_s2=`` computed from it)."""
    return float(G0_M_S2)


# ---------------------------------------------------------------------------
# re-integration through the FL-301 propagator (the G3 rule)
# ---------------------------------------------------------------------------


def landing_guidance(plan: LandingPlan, *, pad_lat_deg: float, pad_lon_deg: float,
                    vehicle: BoosterModel
                    ) -> tuple[Callable[[float, BoosterState], ThrustCmd | None],
                               list]:
    """The guidance callback :func:`reintegrate_landing` flies: the plan's
    node acceleration commands ``u_k = T_k / m_k`` interpolated
    piecewise-linearly in time (the exact replay of the FOH discretization,
    whose velocity update is linear in ``u``), re-mapped to physical thrust
    through the vehicle's *current* mass and rotated into ECEF through the
    pad basis. Mapping u through the current mass -- not the plan's mass
    schedule -- is deliberate: it reproduces the plan's acceleration command
    exactly while letting the propagator's engine model (ambient loss,
    min-throttle clamp) and aero disagree with the flat plan, which is the
    whole point. Thrust magnitude above ambient max near the ground clamps
    to full throttle (a deficit the checks then grade). Past the last node
    the engine is *commanded off* (``None``): the plan's terminal node is
    the cutoff state, and the vehicle coasts the last few tenths of a metre
    in ballistically -- holding a min-throttle command past cutoff would
    hover the vehicle up (the throttle floor exceeds its weight), which is
    not a landing.

    Returns ``(callback, samples)`` -- ``samples`` records every commanded
    ``(t, dir_enu, throttle_cmd, throttle_eff)`` for the tilt proxy."""
    e, n, u_axis = _enu_axes(pad_lat_deg, pad_lon_deg)
    basis = np.vstack([e, n, u_axis])       # rows: ENU unit vectors in ECEF
    times = np.asarray(plan.times_s)
    u_grid = (np.asarray(plan.thrust_enu_n)
              / np.asarray(plan.mass_schedule_kg)[:, None])
    eng = vehicle.engine
    samples: list = []

    def cmd(t_s: float, st: BoosterState) -> ThrustCmd | None:
        t = float(t_s)
        if t >= times[-1]:
            return None  # engine cutoff: coast to the pad
        if t <= times[0]:
            u_cmd = u_grid[0]
        else:
            k = int(np.searchsorted(times, t)) - 1
            frac = (t - times[k]) / (times[k + 1] - times[k])
            u_cmd = (1.0 - frac) * u_grid[k] + frac * u_grid[k + 1]
        u_mag = float(np.linalg.norm(u_cmd))
        if u_mag <= 0.0:
            return None
        d_enu = u_cmd / u_mag
        d_ecef = d_enu[0] * basis[0] + d_enu[1] * basis[1] + d_enu[2] * basis[2]
        t_mag = float(st.mass_kg) * u_mag
        h = height_m(np.asarray(st.position_m, dtype=np.float64))
        t_max_h, _ = _ambient_thrust_envelope(vehicle, h)
        throttle = t_mag / t_max_h
        eff = eng.effective_throttle(throttle)
        samples.append((t, tuple(float(c) for c in d_enu), float(throttle),
                        float(eff)))
        return ThrustCmd(throttle=throttle, dir_ecef=tuple(float(c) for c in d_ecef))

    return cmd, samples


def reintegrate_landing(plan: LandingPlan, state_ecef: BoosterState,
                        vehicle: BoosterModel, *,
                        pad_lat_deg: float, pad_lon_deg: float,
                        ground_height_m: float = 0.0,
                        dt_s: float = 0.1, sample_s: float = 0.5,
                         physics: BoosterPhysics | None = None,
                         coast_allowance_s: float = 45.0) -> tuple[BoosterTrack, list]:
    """Re-fly the solved plan through the FL-301 propagator with full physics
    (drag, J2, rotation) from the FL-302 insertion state. ``ground_height_m``
    = the height at which the ground event fires (pass the *propagator's*
    ``height_m`` of the pad point so the ground is anchored to the pad in the
    metric the event itself uses; note ``height_m`` and
    ``frames.geodetic_to_ecef`` geodesies differ by tens of metres, so the
    pad's nominal altitude is *not* the right number here).
    ``coast_allowance_s`` bounds the ballistic coast past the last node.
    Returns ``(track, samples)``; the caller grades the track -- the plan
    never grades itself."""
    cmd, samples = landing_guidance(plan, pad_lat_deg=pad_lat_deg,
                                    pad_lon_deg=pad_lon_deg, vehicle=vehicle)
    t_end = plan.t0_s + plan.tof_commanded_s + coast_allowance_s
    track = propagate(state_ecef, vehicle, cmd, t_end,
                      events=("burn", "ground"),
                      dt_s=dt_s, sample_s=sample_s,
                      ground_height_m=ground_height_m,
                      physics=physics)
    return track, samples


# ---------------------------------------------------------------------------
# touchdown checks (G3: every verdict from the re-integrated track)
# ---------------------------------------------------------------------------


def _enu_of(r_ecef: Sequence[float], v_ecef: Sequence[float],
            lat_deg: float, lon_deg: float,
            pad_ecef_m: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Pad-relative (position [m], velocity [m/s]) in the pad ENU basis: the
    position is measured *from the pad point* (an un-offset ECEF rotation
    would carry the ~6371 km Earth radius), the velocity is Earth-relative
    and rotates directly."""
    e, n, u = _enu_axes(lat_deg, lon_deg)
    basis = np.vstack([e, n, u])
    r = np.asarray(r_ecef, dtype=np.float64) - np.asarray(pad_ecef_m, dtype=np.float64)
    v = np.asarray(v_ecef, dtype=np.float64)
    return basis @ r, basis @ v


def _tilt_proxy_deg(samples: Sequence[tuple]) -> float | None:
    """Max commanded-thrust tilt from vertical over the samples that really
    burned (effective throttle above zero)."""
    worst = None
    for _t, d_enu, _th, eff in samples:
        if eff <= 0.0:
            continue
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, d_enu[2]))))
        worst = tilt if worst is None else max(worst, tilt)
    return worst


def landing_touchdown_checks(track: BoosterTrack, vehicle: BoosterModel,
                             limits: LandingLimits, *,
                             pad_lat_deg: float, pad_lon_deg: float,
                             pad_altitude_m: float,
                             samples: Sequence[tuple]) -> tuple[
                                 tuple[Check, ...], dict]:
    """Grade the re-integrated track: six independent checks with reason
    codes. Touchdown state comes from the propagator's own ground-event
    bisection (``track.impact``), never from the plan. Returns
    ``(checks, measured)`` with the raw numbers for the result record."""
    measured: dict[str, float | None] = {}
    checks: list[Check] = []

    def add(check_id: str, passed: bool, value: float | None, limit: float | None,
            unit: str, fail_code: str | None = None) -> None:
        code = "" if passed else (fail_code or _CHECK_REASON[check_id])
        checks.append(Check(id=check_id, passed=bool(passed), value=value,
                            limit=limit, unit=unit, reason_code=code))

    impact = track.impact
    pad_ecef = geodetic_to_ecef(pad_lat_deg, pad_lon_deg, pad_altitude_m)
    if impact is not None:
        rel, vel = _enu_of(impact.position_ecef_m, impact.velocity_ecef_m_s,
                           pad_lat_deg, pad_lon_deg, pad_ecef)
        pos_err = float(math.hypot(rel[0], rel[1]))
        # the touchdown point's height above the *pad plane* (ground event is
        # the pad-altitude cut by construction; allow sub-metre residuals)
        pos_err_3d = float(math.sqrt(pos_err ** 2 + (rel[2] - 0.0) ** 2))
        v_up = float(vel[2])
        v_lat = float(math.hypot(vel[0], vel[1]))
    else:
        pos_err_3d = v_up = v_lat = None

    measured.update(position_error_m=pos_err_3d, vertical_speed_m_s=v_up,
                    lateral_speed_m_s=v_lat)

    if impact is None:
        add("landing-position", False, None, limits.position_tol_m, "m",
            LANDING_NO_TOUCHDOWN)
        add("landing-vertical-speed", False, None, limits.vertical_speed_max_m_s,
            "m/s", LANDING_NO_TOUCHDOWN)
        add("landing-lateral-speed", False, None, limits.lateral_speed_max_m_s,
            "m/s", LANDING_NO_TOUCHDOWN)
    else:
        add("landing-position", pos_err_3d <= limits.position_tol_m,
            pos_err_3d, limits.position_tol_m, "m")
        add("landing-vertical-speed",
            v_up is not None and abs(v_up) <= limits.vertical_speed_max_m_s,
            v_up, limits.vertical_speed_max_m_s, "m/s")
        add("landing-lateral-speed",
            v_lat is not None and v_lat <= limits.lateral_speed_max_m_s,
            v_lat, limits.lateral_speed_max_m_s, "m/s")

    tilt = _tilt_proxy_deg(samples)
    measured["max_tilt_deg"] = tilt
    add("landing-tilt", tilt is not None and tilt <= limits.max_tilt_deg,
        tilt, limits.max_tilt_deg, "deg")

    margin = track.reserve_margin_kg()
    measured["propellant_margin_kg"] = margin
    add("landing-propellant-reserve", margin >= 0.0, margin,
        float(vehicle.landing_propellant_reserve_kg), "kg")

    g = float(track.max_proper_g)
    measured["max_proper_g"] = g
    add("landing-max-g", g <= limits.max_proper_g, g, limits.max_proper_g, "g")

    return tuple(checks), measured


# ---------------------------------------------------------------------------
# top level: solve -> re-integrate -> grade
# ---------------------------------------------------------------------------

_FL303_ASSUMPTIONS = (
    "G-FOLD model: constant gravity (local J2 magnitude at insertion), flat "
    "ground, no aero, thrust envelope = ambient engine capability at the "
    "insertion altitude; verdict from the FL-301 re-integration (drag, J2, "
    "rotation, min-throttle clamp), not the solver.",
    "gfold 0.3.2: Environment.max_angle_deg is accepted but inert (verified); "
    "the tilt limit binds the cvxpy reference and the touchdown tilt check.",
    "FOH discretization (n nodes, h = tof/n, terminal node = engine cutoff "
    "state a short distance above the pad; ballistic coast from there); "
    "trapezoidal fuel accounting against sea-level Isp (pessimistic end).",
)


def fly_landing_burn(state_enu: LandingBurnEnu, state_ecef: BoosterState,
                     target: RecoveryTarget, vehicle: BoosterModel, *,
                     limits: LandingLimits = DEFAULT_LANDING_LIMITS,
                     solver: str = "gfold",
                     tof_s: float | None = None,
                     tof_min_s: float = 8.0, tof_max_s: float = 120.0,
                     n: int = 16,
                     glide_slope_deg: float = 20.0,
                     tilt_limit_deg: float = 40.0,
                     gravity_m_s2: float | None = None,
                     dt_s: float = 0.1, sample_s: float = 0.5,
                     physics: BoosterPhysics | None = None,
                     max_speed_m_s: float = 800.0,
                     cutoff_height_m: float = 0.25,
                     cutoff_descent_m_s: float = 1.2,
                     drag_gravity_bias: bool = True) -> LandingResult:
    """The FL-303 flow: solve the SOCP from the corridor state, re-fly the
    solved thrust profile through the FL-301 propagator, grade the
    re-flown track. ``state_enu`` is FL-302's
    ``landing_burn_initial_state_enu``, ``state_ecef`` its ECEF twin
    (``landing_burn_initial_state``) and ``target`` the ``RecoveryTarget``
    whose geodetic site is the pad (ENU origin, cutoff target) -- the
    pad's own altitude anchors the thrust envelope, and the propagator's
    ground event fires at ``height_m(pad)``, both *not* the insertion
    state's height. The arc ends at engine cutoff
    (``cutoff_height_m`` / ``cutoff_descent_m_s``, see
    :func:`solve_landing_burn`) and the re-integration coasts in.
    ``feasible`` means the SOCP solved; ``landed`` means the re-integration
    touched down inside every limit."""
    lat, lon, pad_alt, pad_ecef = _pad_of_target(target)
    plan, infeasible = _insertion_plan(
        state_enu, state_ecef, vehicle, pad_alt, gravity_m_s2=gravity_m_s2,
        drag_gravity_bias=drag_gravity_bias,
        solver=solver, tof_s=tof_s, tof_min_s=tof_min_s, tof_max_s=tof_max_s,
        n=n, glide_slope_deg=glide_slope_deg, tilt_limit_deg=tilt_limit_deg,
        max_speed_m_s=max_speed_m_s,
        cutoff_height_m=cutoff_height_m,
        cutoff_descent_m_s=cutoff_descent_m_s)
    if infeasible is not None:
        return infeasible

    track, samples = reintegrate_landing(
        plan, state_ecef, vehicle, pad_lat_deg=lat, pad_lon_deg=lon,
        ground_height_m=float(height_m(pad_ecef)), dt_s=dt_s, sample_s=sample_s,
        physics=physics)
    return _graded_result(
        plan, track, samples, vehicle, limits, target,
        assumptions=_FL303_ASSUMPTIONS + (
            f"solver={plan.solver} status={plan.status} n={plan.n_nodes} "
            f"tof_flying={plan.tof_flying_s:.2f}s "
            f"fuel_planned={plan.fuel_planned_kg:.1f}kg",))


def _insertion_plan(state_enu: LandingBurnEnu, state_ecef: BoosterState,
                    vehicle: BoosterModel, pad_alt: float, *,
                    gravity_m_s2: float | None, drag_gravity_bias: bool,
                    **solve_kwargs: Any
                    ) -> tuple[LandingPlan | None, LandingResult | None]:
    """The SOCP at corridor insertion: ``(plan, None)``, or ``(None,
    result)`` with the ``LANDING_INFEASIBLE`` verdict when it does not
    solve. Shared by the single-plan flow and FL-305's closed loop, so an
    infeasible corridor state reads identically in both."""
    gravity = (gravity_m_s2 if gravity_m_s2 is not None
               else _local_gravity(state_ecef.position_m))
    if gravity_m_s2 is None and drag_gravity_bias:
        # First-order drag compensation: a convex flat model cannot carry
        # aero, but the burn "feels" the mean drag as reduced gravity --
        # the plan flies at (g - a_drag_mean) so the drag-laden reality
        # tracks it instead of arriving high at cutoff.
        v_mean = 0.9 * abs(state_enu.velocity_enu_m[2]) + 1.0
        rho = density_kg_m3(0.5 * (pad_alt + state_enu.position_enu_m[2]))
        a_drag = (0.5 * rho * v_mean ** 2 * vehicle.cd(0.02)
                  * vehicle.reference_area_m2 / max(state_enu.mass_kg, 1.0))
        gravity = max(gravity - a_drag, 1.0)
    try:
        plan = solve_landing_burn(state_enu, vehicle, gravity_m_s2=gravity,
                                  pad_altitude_m=pad_alt, **solve_kwargs)
    except LandingInfeasibleError as exc:
        return None, LandingResult(
            feasible=False, landed=False,
            reason_codes=(LANDING_INFEASIBLE,),
            checks=(Check(id="landing-solver", passed=False,
                          reason_code=LANDING_INFEASIBLE),),
            assumptions=_FL303_ASSUMPTIONS + (f"solver report: {exc}",),
        )
    return plan, None


def _graded_result(plan: LandingPlan, track: BoosterTrack, samples: Sequence[tuple],
                   vehicle: BoosterModel, limits: LandingLimits,
                   target: RecoveryTarget, *,
                   assumptions: tuple[str, ...]) -> LandingResult:
    """Grade a flown landing track with :func:`landing_touchdown_checks`
    (the referee both flows share) and pack the verdict."""
    lat, lon, pad_alt, pad_ecef = _pad_of_target(target)
    checks, measured = landing_touchdown_checks(
        track, vehicle, limits, pad_lat_deg=lat, pad_lon_deg=lon,
        pad_altitude_m=pad_alt, samples=samples)
    reasons = tuple(c.reason_code for c in checks if not c.passed)
    return LandingResult(
        feasible=True, landed=not reasons,
        reason_codes=reasons,
        checks=checks,
        plan=plan,
        track=track,
        impact=track.impact,
        touchdown_enu_m=None if track.impact is None else tuple(
            float(c) for c in _enu_of(track.impact.position_ecef_m,
                                      track.impact.velocity_ecef_m_s,
                                      lat, lon, pad_ecef)[0]),
        vertical_speed_m_s=measured.get("vertical_speed_m_s"),
        lateral_speed_m_s=measured.get("lateral_speed_m_s"),
        position_error_m=measured.get("position_error_m"),
        max_tilt_deg=measured.get("max_tilt_deg"),
        max_proper_g=measured.get("max_proper_g", 0.0),
        ignitions_used=int(track.ignitions_used),
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# FL-305: closed-loop terminal guidance (docs/adr/flightlab-0002)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalGuidance:
    """FL-305 closed-loop landing-burn guidance (ADR flightlab-0002). The
    convex plan made at corridor insertion is flown open-loop by FL-303; from
    FL-302's ~3 km / ~300 m/s handoff that lands at ~-100 m/s because the flat
    model sees neither drag nor the ambient-thrust drop near the ground. This
    law re-plans analytically from the *propagated* state at every guidance
    step instead:

    * **vertical -- throttle-limited hoverslam.** The constant deceleration
      that brings the descent rate to ``touchdown_speed_m_s`` exactly at pad
      level, ``a_req = (v_z^2 - v_td^2) / (2 h)``, recomputed every step. The
      engine stays shut while ``a_req`` is below what ``design_throttle`` of
      the *sea-level* thrust gives (minus gravity) and lights once it gets
      there, so the burn starts with ``1 - design_throttle`` of authority in
      hand for drag, ambient thrust and mass errors. Late and hard is also
      the min-gravity-loss burn. It lights earlier when the divert would
      otherwise need more than ``divert_ignition_fraction`` of the
      tilt-limited lateral authority (0 disables).
    * **lateral -- ZEM/ZEV divert.** ``a_h = 6 ZEM / t_h^2 - 2 ZEV / t_h``
      toward the pad with the vertical profile's time-to-go
      ``t_go = 2 h / (|v_z| + v_td)`` shortened by ``settle_s``, so the
      divert finishes before the last seconds; after that a PD loop
      (``lateral_damping_per_s``, ``lateral_stiffness_per_s2``) nulls the
      residual drift.
    * **attitude in the loop.** The lateral command is scaled so the thrust
      vector never tilts past ``tilt_limit_deg`` (below the 40 deg touchdown
      check), and the thrust is capped at ``max_thrust_accel_g``.
    * **cutoff tuned to the no-hover T/W.** With a throttle floor above the
      vehicle's weight the profile cannot be followed once ``a_req`` drops
      below the floor's net deceleration: below ``cutoff_height_m`` the
      engine shuts down for good as soon as the ballistic arrival speed
      ``sqrt(v_z^2 + 2 g h)`` is under ``cutoff_speed_m_s``. One ignition,
      one cutoff -- no restarts spent.

    The law commands; it never grades. The verdict is FL-303's
    :func:`landing_touchdown_checks` on the propagated track, unchanged."""

    design_throttle: float = 0.9
    touchdown_speed_m_s: float = 1.5
    tilt_limit_deg: float = 30.0
    max_thrust_accel_g: float = 4.5
    settle_s: float = 2.0
    lateral_damping_per_s: float = 2.0
    lateral_stiffness_per_s2: float = 0.2
    cutoff_height_m: float = 3.0
    cutoff_speed_m_s: float = 2.2
    min_time_to_go_s: float = 0.5
    divert_ignition_fraction: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 < self.design_throttle <= 1.0:
            raise ValueError("design_throttle must be in (0, 1]")
        if self.touchdown_speed_m_s <= 0.0:
            raise ValueError("touchdown_speed_m_s must be positive")
        if not 0.0 < self.tilt_limit_deg < 90.0:
            raise ValueError("tilt_limit_deg must be in (0, 90)")
        if self.max_thrust_accel_g <= 1.0:
            raise ValueError("max_thrust_accel_g must exceed 1 g")
        if min(self.settle_s, self.lateral_damping_per_s,
               self.lateral_stiffness_per_s2, self.cutoff_height_m) < 0.0:
            raise ValueError("settle/gain/cutoff-height terms cannot be negative")
        if self.cutoff_speed_m_s <= 0.0 or self.min_time_to_go_s <= 0.0:
            raise ValueError("cutoff_speed_m_s and min_time_to_go_s must be positive")


#: The FL-305 policy the recovery pipeline flies by default.
TERMINAL_GUIDANCE_DEFAULT = TerminalGuidance()


@dataclass(frozen=True)
class TerminalCommand:
    """One step of :func:`terminal_guidance_command`: ``phase`` is
    ``"coast"`` (engine not yet lit), ``"burn"`` or ``"cutoff"`` (final
    shutdown); ``thrust_enu_n`` is the commanded thrust vector in pad ENU
    (zero unless burning) and ``required_decel_m_s2`` the profile's
    ``a_req`` at this state."""

    phase: str
    thrust_enu_n: tuple[float, float, float]
    required_decel_m_s2: float

    @property
    def thrust_n(self) -> float:
        return math.sqrt(sum(c * c for c in self.thrust_enu_n))


def _divert_needs_ignition(r: np.ndarray, v: np.ndarray, a_design: float,
                           gravity_m_s2: float, g: TerminalGuidance) -> bool:
    """Divert-aware ignition: light early when a burn started now at the
    design deceleration would already need more than
    ``divert_ignition_fraction`` of the tilt-limited lateral authority to
    null the pad offset (a late hoverslam leaves too little time to divert)."""
    if a_design <= 0.0 or g.divert_ignition_fraction <= 0.0:
        return False
    t_h = abs(float(v[2])) / a_design - g.settle_s
    if t_h <= g.min_time_to_go_s:
        return False
    r_h, v_h = r[:2], v[:2]
    need = float(np.linalg.norm(6.0 * (-r_h - v_h * t_h) / t_h ** 2 + 2.0 * v_h / t_h))
    cap = math.tan(math.radians(g.tilt_limit_deg)) * (a_design + gravity_m_s2)
    return need > g.divert_ignition_fraction * cap


def terminal_guidance_command(position_enu_m: Sequence[float],
                              velocity_enu_m_s: Sequence[float],
                              mass_kg: float, *,
                              gravity_m_s2: float,
                              thrust_max_n: float,
                              thrust_min_n: float,
                              ignited: bool,
                              guidance: TerminalGuidance = TERMINAL_GUIDANCE_DEFAULT,
                              ) -> TerminalCommand:
    """The FL-305 law (see :class:`TerminalGuidance`) as a pure function of
    the pad-relative state. ``thrust_max_n`` is the engine's ambient maximum
    where the ignition decision should be judged (the pipeline passes the
    sea-level value: the burn ends there), ``thrust_min_n`` its floor at the
    current altitude, ``ignited`` whether the burn already started."""
    g = guidance
    r = np.asarray(position_enu_m, dtype=np.float64)
    v = np.asarray(velocity_enu_m_s, dtype=np.float64)
    m = float(mass_kg)
    h, vz = float(r[2]), float(v[2])
    h_eff = max(h, 0.05)
    v_td = g.touchdown_speed_m_s
    a_req = (vz * vz - v_td * v_td) / (2.0 * h_eff) if vz < -v_td else 0.0
    zero = (0.0, 0.0, 0.0)
    if not ignited:
        a_design = g.design_throttle * float(thrust_max_n) / m - gravity_m_s2
        if a_req < a_design and not _divert_needs_ignition(
                r, v, a_design, gravity_m_s2, g):
            return TerminalCommand("coast", zero, a_req)
    a_floor = float(thrust_min_n) / m - gravity_m_s2
    if (a_req < a_floor and h < g.cutoff_height_m
            and math.sqrt(vz * vz + 2.0 * gravity_m_s2 * max(h, 0.0))
            <= g.cutoff_speed_m_s):
        return TerminalCommand("cutoff", zero, a_req)

    a_up = a_req + gravity_m_s2
    t_go = max(2.0 * h_eff / (abs(vz) + v_td), g.min_time_to_go_s)
    t_h = t_go - g.settle_s
    r_h, v_h = r[:2], v[:2]
    if t_h > g.min_time_to_go_s:
        a_h = 6.0 * (-r_h - v_h * t_h) / t_h ** 2 + 2.0 * v_h / t_h
    else:
        a_h = -g.lateral_damping_per_s * v_h - g.lateral_stiffness_per_s2 * r_h
    a_h_mag = float(np.hypot(a_h[0], a_h[1]))
    a_h_cap = math.tan(math.radians(g.tilt_limit_deg)) * a_up
    if a_h_mag > a_h_cap:  # tilt limit enforced in the loop
        a_h = a_h * (a_h_cap / a_h_mag)
    acc = np.array([a_h[0], a_h[1], a_up])  # a_up >= g > 0: never a zero vector
    a_mag = float(np.linalg.norm(acc))
    a_cmd = min(a_mag, g.max_thrust_accel_g * G0_M_S2)
    thrust = acc / a_mag * (m * a_cmd)
    return TerminalCommand("burn", tuple(float(c) for c in thrust), a_req)


class _ClosedLoopLanding:
    """The guidance callback :func:`propagate` samples once per step: pad-ENU
    state in, :func:`terminal_guidance_command` out, mapped to a throttle
    through the *ambient* thrust at the flown altitude. Records the command
    history (the tilt proxy the touchdown checks grade), ignition and
    cutoff times."""

    def __init__(self, vehicle: BoosterModel, guidance: TerminalGuidance, *,
                 pad_lat_deg: float, pad_lon_deg: float, pad_altitude_m: float,
                 pad_ecef_m: np.ndarray, gravity_m_s2: float) -> None:
        self.vehicle, self.guidance = vehicle, guidance
        self.lat, self.lon = pad_lat_deg, pad_lon_deg
        self.pad_ecef = np.asarray(pad_ecef_m, dtype=np.float64)
        self.axes = _enu_axes(pad_lat_deg, pad_lon_deg)
        self.gravity = float(gravity_m_s2)
        self.thrust_max_pad, _ = _ambient_thrust_envelope(vehicle, pad_altitude_m)
        self.samples: list[tuple] = []
        self.ignition_t_s: float | None = None
        self.cutoff_t_s: float | None = None

    def __call__(self, t_s: float, st: BoosterState) -> ThrustCmd | None:
        if self.cutoff_t_s is not None:
            return None
        r, v = _enu_of(st.position_m, st.velocity_m_s, self.lat, self.lon,
                       self.pad_ecef)
        t_max, t_min = _ambient_thrust_envelope(
            self.vehicle, height_m(np.asarray(st.position_m, dtype=np.float64)))
        cmd = terminal_guidance_command(
            r, v, float(st.mass_kg), gravity_m_s2=self.gravity,
            thrust_max_n=min(t_max, self.thrust_max_pad), thrust_min_n=t_min,
            ignited=self.ignition_t_s is not None, guidance=self.guidance)
        if cmd.phase == "coast":
            return None
        if cmd.phase == "cutoff":
            self.cutoff_t_s = float(t_s)
            return None
        if self.ignition_t_s is None:
            self.ignition_t_s = float(t_s)
        f = np.asarray(cmd.thrust_enu_n, dtype=np.float64)
        f_mag = float(np.linalg.norm(f))
        d_enu = f / f_mag
        # never command a zero-throttle shutdown mid-burn (that would spend a
        # restart); the propagator clamps up to the engine floor instead.
        throttle = min(max(f_mag / t_max, 1e-6), self.vehicle.engine.throttle_max)
        e_, n_, u_ = self.axes
        d_ecef = d_enu[0] * e_ + d_enu[1] * n_ + d_enu[2] * u_
        self.samples.append((float(t_s), tuple(float(c) for c in d_enu),
                             float(throttle),
                             float(self.vehicle.engine.effective_throttle(throttle))))
        return ThrustCmd(throttle=float(throttle),
                         dir_ecef=tuple(float(c) for c in d_ecef))


def fly_landing_burn_closed_loop(state_enu: LandingBurnEnu, state_ecef: BoosterState,
                                 target: RecoveryTarget, vehicle: BoosterModel, *,
                                 limits: LandingLimits = DEFAULT_LANDING_LIMITS,
                                 guidance: TerminalGuidance = TERMINAL_GUIDANCE_DEFAULT,
                                 solver: str = "gfold",
                                 gravity_m_s2: float | None = None,
                                 dt_s: float = 0.1, sample_s: float = 0.5,
                                 physics: BoosterPhysics | None = None,
                                 flight_time_max_s: float = 300.0,
                                 ) -> LandingResult:
    """FL-305: the FL-303 landing flow with closed-loop terminal guidance.

    The SOCP is still solved at corridor insertion -- it is the feasibility
    certificate (an infeasible state reads ``LANDING_INFEASIBLE`` exactly as
    in :func:`fly_landing_burn`) and the recorded plan -- but the burn is
    flown by :class:`TerminalGuidance`, re-planned from the propagated state
    every integration step, instead of the plan's open-loop thrust schedule.
    One integration through the FL-301 propagator with the caller's
    ``physics``; :func:`landing_touchdown_checks` grades it (G3: the
    guidance never scores itself)."""
    lat, lon, pad_alt, pad_ecef = _pad_of_target(target)
    plan, infeasible = _insertion_plan(state_enu, state_ecef, vehicle, pad_alt,
                                       gravity_m_s2=gravity_m_s2,
                                       drag_gravity_bias=True, solver=solver,
                                       tilt_limit_deg=limits.max_tilt_deg)
    if infeasible is not None:
        return infeasible
    law = _ClosedLoopLanding(
        vehicle, guidance, pad_lat_deg=lat, pad_lon_deg=lon, pad_altitude_m=pad_alt,
        pad_ecef_m=pad_ecef,
        gravity_m_s2=(gravity_m_s2 if gravity_m_s2 is not None
                      else _local_gravity(state_ecef.position_m)))
    track = propagate(state_ecef, vehicle, law,
                      float(state_ecef.t_s) + float(flight_time_max_s),
                      events=("burn", "ground"), dt_s=dt_s, sample_s=sample_s,
                      ground_height_m=float(height_m(pad_ecef)), physics=physics)

    def _t(x: float | None) -> str:
        return "none" if x is None else f"{x - float(state_ecef.t_s):.1f}s"

    return _graded_result(
        plan, track, law.samples, vehicle, limits, target,
        assumptions=_FL303_ASSUMPTIONS + (
            "FL-305 closed-loop terminal guidance (docs/adr/flightlab-0002): "
            "the insertion SOCP is the feasibility certificate; the burn is "
            "re-planned from the propagated state every step (hoverslam "
            "vertical profile, ZEM/ZEV divert, tilt limit in the loop, "
            "no-hover ballistic cutoff); verdict from the same touchdown checks.",
            f"guidance: design_throttle={guidance.design_throttle:g} "
            f"v_td={guidance.touchdown_speed_m_s:g}m/s "
            f"tilt<={guidance.tilt_limit_deg:g}deg ignition={_t(law.ignition_t_s)} "
            f"cutoff={_t(law.cutoff_t_s)} after insertion; "
            f"insertion plan fuel_planned={plan.fuel_planned_kg:.1f}kg",))


# ---------------------------------------------------------------------------
# FlightRun plumbing: booster body + landing_start / touchdown / impact
# ---------------------------------------------------------------------------


def landing_events(result: LandingResult, *, body_id: str = "booster") -> tuple[Event, ...]:
    """The contract events of a landing attempt: ``landing_start`` at
    ignition, then ``touchdown`` (a soft, in-limits landing) or ``impact``
    (everything else that still reached the ground). An infeasible solve
    produced no burn and gets no events."""
    if not result.feasible or result.plan is None:
        return ()
    plan = result.plan
    events: list[Event] = [Event(
        t_s=plan.t0_s, kind="landing_start", body_id=body_id,
        data={"solver": plan.solver, "status": plan.status,
              "tof_flying_s": plan.tof_flying_s,
              "fuel_planned_kg": plan.fuel_planned_kg,
              "gravity_m_s2": plan.gravity_m_s2})]
    if result.track is not None and result.track.impact is not None:
        imp = result.track.impact
        kind = "touchdown" if result.landed else "impact"
        events.append(Event(
            t_s=float(imp.t_s), kind=kind, body_id=body_id,
            data={"position_error_m": result.position_error_m,
                  "vertical_speed_m_s": result.vertical_speed_m_s,
                  "lateral_speed_m_s": result.lateral_speed_m_s,
                  "reason_codes": list(result.reason_codes)}))
    return tuple(events)


def landing_result_to_run(result: LandingResult, *,
                          run_id: str, target_sha256: str, vehicle_sha256: str,
                          git_sha: str, created_utc: str,
                          body_id: str = "booster",
                          frame: str = "ECEF",
                          provenance: str = "illustrative",
                          parent_body_id: str | None = None,
                          ) -> FlightRun:
    """The landing attempt as a FlightResult v1 run: the booster body track
    (ECEF, with the flown throttle/Mach/q as extra columns), the
    ``landing_start`` / ``touchdown``|``impact`` events, and a verification
    whose green flags require both a converged SOCP *and* a re-integration
    inside every limit (an infeasible solve can never produce a green run)."""
    tracks: tuple = ()
    if result.track is not None:
        tracks = (result.track.to_body_track(
            frame, body_id=body_id, parent_body_id=parent_body_id,
            provenance=provenance),)
    checks = tuple(result.checks)
    verification = Verification(
        optimizer_converged=bool(result.feasible),
        physics_verified=bool(result.landed),
        mission_constraints_passed=bool(result.landed),
        checks=checks,
    )
    return FlightRun(
        run_id=run_id,
        target_sha256=target_sha256,
        vehicle_sha256=vehicle_sha256,
        git_sha=git_sha,
        created_utc=created_utc,
        engines={"gfold": GFOLD_VERSION_PIN if result.plan is not None
                 and result.plan.solver == "gfold" else "not-run",
                 "cvxpy": "reference",
                 "propagator": "orbitopt.flightlab.recovery.propagator"},
        assumptions=tuple(result.assumptions),
        tracks=tracks,
        events=landing_events(result, body_id=body_id),
        verification=verification,
    )
