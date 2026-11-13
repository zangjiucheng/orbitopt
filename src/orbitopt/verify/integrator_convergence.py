"""V-03 (task #242): integration timestep / tolerance numerical-error convergence.

Epic P1 asks every Mission Run to answer *is the numerical error under
control?*. This harness answers it for the four **key scenarios** that decide
the numbers the rest of orbitopt publishes, by sweeping each scenario's
integrator settings over a geometric grid and measuring the error against the
tightest run of the same grid:

* **tolerance axis** -- the adaptive DOP853 runs at
  ``(rtol, atol) = production default x factor``, ``factor`` over four decades
  (1000x looser ... 10x tighter), reference run at 100x tighter than the
  default. Error metric: max |delta r| / |delta v| along the common output
  grid of the window (a sup-norm over the arc, not just the endpoint).
* **step axis** -- the same model re-run by an explicit fixed-step RK4 with
  the step halved down a grid (the classical step-halving convergence study,
  and the audit of the *legacy fixed-step* integrator path,
  ``orbitopt.verify.tudat_propagate`` ``integrator="rk4"``, whose ``step_size``
  defaults to the same value the adaptive integrator uses as its *maximum*
  step). Error metric: final state at the window end (every step grid lands
  on the window end exactly; interpolating a coarse RK4 history onto a fine
  grid would measure the interpolation, not the integration).

From the sweep the harness derives, per scenario and axis:

* the **convergence curve** -- error vs setting for every grid point (stored
  verbatim in the report; ``--plot`` additionally writes it as an ASCII
  log-log curve);
* the **observed convergence order** -- median of the successive-pair
  log-log slopes (fixed-step: the expected value is the RK4 order 4, reduced
  by a burnout discontinuity or a C0 atmosphere-table kink; adaptive:
  global-error-vs-tolerance for an order-8 pair sits near 0.9);
* the **numerical error bound** -- at the production default setting: the
  measured error against the reference *plus* the fitted-power-law estimate
  of the reference run's own residual error. This is the number the task's
  acceptance asks for, and it is a bound in the standard Richardson sense:
  valid while the asymptotic regime (order gate) holds;
* a **recommendation** -- the loosest setting on the curve whose fitted error
  meets the budget with a safety factor, and a verdict on the production
  default ("justified", "conservative", "not justified"), with the realized
  step statistics showing whether the ``max_step`` cap is even active at the
  default tolerances.

The four scenarios, and the production setting each one audits:

``leo_j2_coast``
    500 km / 53 deg circular coast with J2 over one revolution -- the M-05
    machinery of :mod:`orbitopt.mission.propagation` (``rtol=1e-11``,
    ``atol=1e-6``, ``DEFAULT_MAX_STEP_S=300``), the propagation every mission
    coast report runs on.
``gto_coast``
    One full revolution of a 300 km x GEO transfer orbit with J2 -- the same
    M-05 defaults stretched to the most eccentric *bound* coast the mission
    layer propagates; along-track error growth over a 10.5 h arc is the worst
    case for coast tolerances.
``entry_descent``
    The FL-403-lineage lifting re-entry of :func:`orbitopt.entry.fly_entry`
    (150 t vehicle, flat-plate 65 deg, bank 0) from the 600 s before interface
    until t+2400 s (through peak deceleration, above the stop height), under
    the ``fly_entry`` defaults (``rtol=1e-9``, ``atol=1e-6``,
    ``max_step_s=60``) -- the stiff-drag scenario: the atmosphere table is a
    piecewise-exponential (C0) model and errors entering at 120 km get
    amplified through the whole entry.
``meco_ascent``
    The 3-DOF FL-403 spike ascent of :mod:`orbitopt.verify.cross_model`
    (V-02's reference backend: ``rtol=1e-10``, ``atol=1e-5``) from pad to
    MECO -- thrust + gravity + Coriolis with a mass singularity near burnout
    and a thrust cutoff, i.e. the powered-flight handoff state.

Gate semantics follow task #243 (this repo's cross_model module, V-02): every
residual is category ``numerical_consistency`` (same model, independent
numerics), a residual without a resolved budget is recorded but never gated,
and an absent check is never a pass. The budget per scenario and metric is the
acceptance line; the *justified* verdict additionally requires the safety
factor (default 3x) and the minimum observed order, so a report that merely
converges nowhere near the budget cannot read "passed" by luck. On the fixed
axis a step that would meet the budget only *below* the measured grid is
recorded as a power-law extrapolation, not a demonstrated pass; scenarios
without a production fixed-step setting carry no fixed-step budget (order
gate only -- the ascent arc's sub-millisecond RK4 requirement is precisely
the evidence that the *adaptive* default is the justified one).

    from orbitopt.verify.integrator_convergence import run_convergence_study
    study = run_convergence_study(budgets={"gto_coast.position_m": "20"})
    print(study.report["status"]["result"])    # gate-gradeable

Run from the command line (exit code 1 if a gated check fails):

    python -m orbitopt.verify.integrator_convergence [--scenarios ...]
        [--budget KEY=VALUE] [--safety M] [--plot] [-o PATH] [--json]
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from orbitopt.entry import (
    EntryVehicle,
    J2_EARTH,
    MU_EARTH_M3_S2,
    OMEGA_EARTH_RAD_S,
    R_EARTH_M,
    flat_plate_coefficients,
    fly_entry,
)
from orbitopt.flightlab.frames import EARTH_ROTATION_RATE
from orbitopt.verify.report import (
    CHECK_NUMERICAL,
    REPORT_DIR_ENV,
    build_report,
    default_report_path,
    finalize_report,
    format_status,
    status_result,
)

__all__ = [
    "BUDGET_METRICS",
    "Budgets",
    "ConvergenceScenario",
    "ConvergenceStudy",
    "DEFAULT_SAFETY_MARGIN",
    "SCENARIOS",
    "TOL_FACTORS",
    "parse_budgets",
    "run_convergence_study",
    "run_scenario",
]

#: Looser-than-default ... tighter-than-default factors of the tolerance
#: sweep; the reference run sits one decade beyond the tightest point.
TOL_FACTORS: tuple[float, ...] = (1e3, 1e2, 1e1, 1e0, 1e-1)
REFERENCE_FACTOR = 1e-2
#: scipy DOP853 refuses rtol below ~100*eps; the grid stays an order above it.
_MIN_RTOL = 1e-13

#: The "justified" verdict requires the measured bound to sit this far under
#: the budget (same spirit as the M-05 gate: 100 m / 0.01 m/s flags a broken
#: stack, not the last centimetre -- here the bound must be a factor SAFETY
#: under the line, so a 10x degradation still passes before the gate trips).
DEFAULT_SAFETY_MARGIN = 3.0

AXES = ("tolerance", "fixed_step")
BUDGET_METRICS = ("position_m", "velocity_m_s")


# ---------------------------------------------------------------------------
# Error model of one sweep
# ---------------------------------------------------------------------------

def _pair_orders(errs: Sequence[float], ratios: Sequence[float]) -> list[float]:
    """log-log slopes between successive grid points that converge
    (``errs[i+1] < errs[i]``); a growth between two points is recorded by the
    caller as curve noise and contributes no slope."""
    out = []
    for i in range(len(errs) - 1):
        e0, e1 = float(errs[i]), float(errs[i + 1])
        if e0 > 0.0 and e1 > 0.0 and e1 < e0:
            out.append(math.log(e0 / e1) / math.log(float(ratios[i])))
    return [o for o in out if math.isfinite(o)]


def observed_order(errs: Sequence[float], ratios: Sequence[float]) -> tuple[float, list]:
    """Median of :func:`_pair_orders` (robust to a single non-monotonic grid
    point, which real entry dynamics produce); 0.0 when the curve never
    decreases."""
    pairs = _pair_orders(errs, ratios)
    return (float(np.median(pairs)) if pairs else 0.0), pairs


def error_bound(err_default: float, err_tightest: float, order: float,
                ref_to_tightest_ratio: float) -> float:
    """Richardson-style bound: measured error at the setting under test plus
    the fitted power-law estimate of the *reference* run's own residual error
    (the tightest grid point extrapolated one grid interval finer)."""
    ref_residual = err_tightest * ref_to_tightest_ratio ** order if order > 0.0 else err_tightest
    return float(err_default + ref_residual)


# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceScenario:
    """One key scenario: the window, the initial state, both runner hooks and
    the production integrator settings + acceptance budget it audits.

    ``adaptive(rtol, atol, max_step_s)`` must return
    ``(t_samples, states, accepted_step_times_or_None)`` on a **fixed output
    grid** (identical for every factor, so the sweep's sup-norm contains no
    interpolation artefact); ``rhs(t, s)`` is the same ODE for the fixed-step
    axis."""

    name: str
    description: str
    audits: str
    production: Mapping
    budget: Mapping
    min_order: Mapping
    adaptive: Callable
    rhs: Callable
    rk4_grid: tuple
    duration_s: float
    state0: tuple
    safety: float = DEFAULT_SAFETY_MARGIN
    #: axes whose budget is recorded but not gated on that axis (a scenario
    #: without a production fixed-step setting cannot "fail" a fixed-step
    #: budget it never claims to meet; the order gate still applies)
    budget_exempt_axes: tuple = ()

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "audits": self.audits, "production": dict(self.production),
                "budget": dict(self.budget), "min_order": dict(self.min_order),
                "duration_s": self.duration_s,
                "rk4_grid": [float(h) for h in self.rk4_grid],
                "budget_exempt_axes": list(self.budget_exempt_axes)}


def _accel_gravity(t, s, j2: bool):
    """Point mass (+ J2 zonal about the z axis) -- the coast model of the
    M-05 machinery in the osculating-element convention (figure axis fixed
    to the inertial z over arc-length windows; the same formula
    :mod:`orbitopt.verify.benchmarks` uses for its J2 benchmark)."""
    r = np.asarray(s[:3], dtype=float)
    rn = float(np.linalg.norm(r))
    a = -MU_EARTH_M3_S2 * r / rn ** 3
    if j2:
        z2 = r[2] * r[2] / (rn * rn)
        k = -1.5 * J2_EARTH * MU_EARTH_M3_S2 * R_EARTH_M ** 2 / rn ** 5
        a = a + k * np.array([r[0] * (1.0 - 5.0 * z2), r[1] * (1.0 - 5.0 * z2),
                              r[2] * (3.0 - 5.0 * z2)])
    return np.concatenate([s[3:], a])


def _adaptive_ivp(rhs, state0, t_eval):
    """Adaptive DOP853 runner on a fixed output grid. Dense output lets one
    run serve both the sweep samples and the *accepted* step history (the
    realised-step statistics behind the max_step verdict)."""
    def run(rtol, atol, max_step_s):
        from scipy.integrate import solve_ivp

        te = np.asarray(t_eval, dtype=float)
        sol = solve_ivp(rhs, (float(te[0]), float(te[-1])),
                        np.asarray(state0, dtype=float),
                        method="DOP853", rtol=float(rtol), atol=float(atol),
                        max_step=float(max_step_s), dense_output=True)
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        states = np.asarray(sol.sol(te), dtype=float)[:6].T
        return te, states, np.asarray(sol.t, dtype=float)
    return run


def _m05_defaults() -> dict:
    """The M-05 propagation defaults, read off the result dataclass fields
    (orbitopt.mission.propagation is tudat-backed; import lazily)."""
    from orbitopt.mission.propagation import (DEFAULT_MAX_STEP_S,
                                              InjectedOrbitPropagation)

    f = InjectedOrbitPropagation.__dataclass_fields__
    return {"rtol": float(f["rtol"].default), "atol": float(f["atol"].default),
            "max_step_s": float(DEFAULT_MAX_STEP_S)}


def _leo_j2_coast() -> ConvergenceScenario:
    from orbitopt.verify.benchmarks import coe_to_ic

    m05 = _m05_defaults()

    state0 = coe_to_ic(R_EARTH_M + 500e3, 0.001, 53.0, 40.0, 30.0, 0.0)
    t_eval = np.linspace(0.0, 5400.0, 181)
    return ConvergenceScenario(
        name="leo_j2_coast",
        description="500 km / 53 deg circular coast with J2, one revolution "
                    "(5400 s): the M-05 coast arc every mission propagation runs",
        audits="orbitopt.mission.propagation M-05 defaults "
               f"(rtol={m05['rtol']:g}, atol={m05['atol']:g}, "
               f"max_step_s={m05['max_step_s']:g})",
        production=dict(m05),
        budget={"position_m": 1.0, "velocity_m_s": 1e-3},
        min_order={"tolerance": 0.3, "fixed_step": 3.0},
        adaptive=_adaptive_ivp(lambda t, s: _accel_gravity(t, s, True), state0, t_eval),
        rhs=lambda t, s: _accel_gravity(t, s, True),
        rk4_grid=(128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0),
        duration_s=5400.0, state0=tuple(float(x) for x in state0))


def _gto_coast() -> ConvergenceScenario:
    from orbitopt.verify.benchmarks import coe_to_ic

    m05 = _m05_defaults()

    rp, ra = R_EARTH_M + 300e3, 42_164_000.0
    a = 0.5 * (rp + ra)
    e = 1.0 - rp / a
    t_rev = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH_M3_S2)
    state0 = coe_to_ic(a, e, 28.0, 60.0, 180.0, 0.0)
    t_eval = np.linspace(0.0, t_rev, 361)
    return ConvergenceScenario(
        name="gto_coast",
        description="one full revolution of a 300 km x GEO transfer orbit "
                    "(e=0.727) with J2: the eccentric worst case for coast "
                    "along-track error growth",
        audits="orbitopt.mission.propagation M-05 defaults on a highly "
               "eccentric arc",
        production=dict(m05),
        budget={"position_m": 20.0, "velocity_m_s": 0.05},
        min_order={"tolerance": 0.3, "fixed_step": 3.0},
        adaptive=_adaptive_ivp(lambda t, s: _accel_gravity(t, s, True), state0, t_eval),
        rhs=lambda t, s: _accel_gravity(t, s, True),
        rk4_grid=(32.0, 16.0, 8.0, 4.0),
        duration_s=float(t_rev), state0=tuple(float(x) for x in state0))


# entry window: 600 s of coast-down + 1800 s through the atmosphere up to
# peak deceleration, ending at ~37 km altitude, above fly_entry's stop height
ENTRY_DURATION_S = 2400.0


def _entry_descent() -> ConvergenceScenario:
    import inspect

    from orbitopt.entry import _accelerations  # same physics fly_entry itself uses

    r0 = R_EARTH_M + 275e3
    state0 = np.array([r0, 0.0, 0.0, 0.0, math.sqrt(MU_EARTH_M3_S2 / r0) - 80.0, 0.0])
    vehicle = EntryVehicle(150_000.0, 400.0, *flat_plate_coefficients(65.0))
    sig = inspect.signature(fly_entry)

    def rhs(t, s):
        a, _aero = _accelerations(np.asarray(s[:3], float), np.asarray(s[3:], float),
                                  vehicle, 0.0)
        return np.concatenate([s[3:], a])

    def run(rtol, atol, max_step_s):
        traj = fly_entry(state0, vehicle, 0.0, stop_height_m=1000.0,
                         max_time_s=ENTRY_DURATION_S, sample_s=2.0,
                         rtol=float(rtol), atol=float(atol), max_step_s=float(max_step_s))
        if traj.ended_at_height:
            raise RuntimeError("entry window ended at the stop height; the sweep "
                               "needs the fixed window")
        return traj.times_s, traj.states[:, :6], None

    return ConvergenceScenario(
        name="entry_descent",
        description="275 km de-orbit + lifting entry (150 t, flat plate 65 deg, "
                    "bank 0) to t+2400 s through peak deceleration: the stiff "
                    "C0-atmosphere scenario (fly_entry's own production defaults)",
        audits="orbitopt.entry.fly_entry defaults "
               f"(rtol={sig.parameters['rtol'].default}, "
               f"atol={sig.parameters['atol'].default}, "
               f"max_step_s={sig.parameters['max_step_s'].default})",
        production={"rtol": float(sig.parameters["rtol"].default),
                    "atol": float(sig.parameters["atol"].default),
                    "max_step_s": float(sig.parameters["max_step_s"].default)},
        budget={"position_m": 5.0, "velocity_m_s": 0.05},
        min_order={"tolerance": 0.3, "fixed_step": 1.5},
        adaptive=run,
        rhs=rhs,
        rk4_grid=(4.0, 2.0, 1.0),
        duration_s=ENTRY_DURATION_S, state0=tuple(float(x) for x in state0))


def _meco_ascent() -> ConvergenceScenario:
    """The V-02 3-DOF reference backend's ascent arc, pad to MECO."""
    from orbitopt.flightlab.frames import geodetic_to_ecef
    from orbitopt.verify.cross_model import (CrossModelScenario, _gravity_point_mass,
                                             _thrust_direction_ecef)

    import inspect
    import orbitopt.verify.cross_model as cm

    sc = CrossModelScenario()
    r0 = np.asarray(geodetic_to_ecef(sc.latitude_deg, sc.longitude_deg,
                                     sc.altitude_m), dtype=float).reshape(3)
    state0 = np.concatenate([r0, np.zeros(3)])
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    b0, b1 = sc.burn_start_s, sc.burn_end_s

    def rhs(t, s):
        r, v = np.asarray(s[:3], float), np.asarray(s[3:], float)
        a = _gravity_point_mass(r, False) - 2.0 * np.cross(omega, v) \
            - np.cross(omega, np.cross(omega, r))
        if b0 <= t < b1:
            a = a + (sc.thrust_n / sc.mass_at(t)) * _thrust_direction_ecef(sc, r, t)
        return np.concatenate([v, a])

    t_eval = np.linspace(0.0, sc.meco_t_s, 211)
    run = _adaptive_ivp(rhs, state0, t_eval)

    # the tolerances cross_model's 3-DOF backend actually integrates with
    src = inspect.getsource(cm._produce_three_dof)
    rtol_m = src.index("rtol=1e-10")
    rtol = float(src[rtol_m + 5:src.index(",", rtol_m)].strip())
    atol_m = src.index("atol=1e-5")
    atol = float(src[atol_m + 5:src.index(")", atol_m)].strip())

    return ConvergenceScenario(
        name="meco_ascent",
        description="FL-403 spike-stage 3-DOF ascent, pad to MECO (105 s): the "
                    "powered-flight handoff state that V-02's cross-model pairs "
                    "start from; thrust cutoff + 1/m mass term",
        audits="orbitopt.verify.cross_model 3-DOF backend ascent run "
               f"(rtol={rtol}, atol={atol})",
        production={"rtol": rtol, "atol": atol, "max_step_s": float("inf")},
        budget={"position_m": 0.1, "velocity_m_s": 5e-3},
        min_order={"tolerance": 0.3, "fixed_step": 1.0},
        adaptive=run,
        rhs=rhs,
        rk4_grid=(0.5, 0.25, 0.125, 0.0625),
        duration_s=float(sc.meco_t_s), state0=tuple(float(x) for x in state0),
        budget_exempt_axes=("fixed_step",))


#: The key scenarios, keyed by name.
SCENARIOS: dict[str, ConvergenceScenario] = {
    s.name: s for s in (_leo_j2_coast(), _gto_coast(), _entry_descent(), _meco_ascent())
}

DEFAULT_SCENARIOS: tuple[str, ...] = tuple(SCENARIOS)


# ---------------------------------------------------------------------------
# Budgets (the acceptance line per scenario and metric)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Budgets:
    """Per-scenario per-metric budgets with dotted-key overrides:
    ``"<scenario>.<metric>"``. A resolved value of ``None`` (override string
    ``"none"``) records the metric without gating it (task #243: absent
    evidence is never a pass -- and never a fail either)."""

    overrides: Mapping[str, float | None] = field(default_factory=dict)

    def resolve(self, scenario: str, metric: str) -> float | None:
        key = f"{scenario}.{metric}"
        if key in self.overrides:
            return self.overrides[key]
        return float(SCENARIOS[scenario].budget[metric])

    def to_dict(self) -> dict:
        return {"defaults": {n: dict(SCENARIOS[n].budget) for n in sorted(SCENARIOS)},
                "overrides": {k: (None if v is None else float(v))
                              for k, v in sorted(self.overrides.items())}}

    @staticmethod
    def coerce(arg: "Budgets | Mapping | None") -> "Budgets":
        if arg is None:
            return Budgets()
        if isinstance(arg, Budgets):
            return arg
        return Budgets(overrides=parse_budgets(arg))


def parse_budgets(mapping: Mapping) -> dict:
    """Validate a ``{"<scenario>.<metric>": value}`` budget mapping (value
    may be ``None``/``"none"`` to record without gating)."""
    out: dict[str, float | None] = {}
    for key, value in dict(mapping).items():
        key = str(key)
        scenario, _, metric = key.partition(".")
        if metric not in BUDGET_METRICS or scenario not in SCENARIOS:
            raise ValueError(f"budget key {key!r}: expected <scenario>.<metric> with "
                             f"scenario in {sorted(SCENARIOS)} and metric in {BUDGET_METRICS}")
        if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
            out[key] = None
        else:
            f = float(value)
            if not math.isfinite(f) or f <= 0.0:
                raise ValueError(f"budget {key}={value!r}: must be a finite number > 0 (or None)")
            out[key] = f
    return out


# ---------------------------------------------------------------------------
# Fixed-step RK4
# ---------------------------------------------------------------------------

def rk4(rhs: Callable, state0, t_end: float, step_s: float):
    """Explicit fixed-step RK4 from 0 to ``t_end`` in ``round(t_end/step_s)``
    uniform steps (so the grid lands on ``t_end`` exactly). Returns
    ``(times, states)`` sampled at every step."""
    n = max(1, int(round(float(t_end) / float(step_s))))
    h = float(t_end) / n
    s = np.asarray(state0, dtype=float).copy()
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, s.size))
    ts[0], ys[0] = 0.0, s
    t = 0.0
    for i in range(n):
        k1 = np.asarray(rhs(t, s), dtype=float)
        k2 = np.asarray(rhs(t + 0.5 * h, s + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(rhs(t + 0.5 * h, s + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(rhs(t + h, s + h * k3), dtype=float)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"RK4 diverged at t={t:.1f}s with step {h:g}s")
        ts[i + 1], ys[i + 1] = t, s
    return ts, ys


def _final_state(t: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.asarray(states[-1], dtype=float)


def _max_window_error(ref_t, ref_states, t, states) -> tuple[float, float]:
    """Sup-norm position/velocity difference between a run and the reference
    on the reference's output grid (adaptive-axis runs share the grid by
    construction, so no interpolation is needed or done)."""
    ref_t = np.asarray(ref_t, dtype=float)
    t = np.asarray(t, dtype=float)
    if t.shape != ref_t.shape or not np.allclose(t, ref_t, rtol=0.0, atol=1e-9):
        raise ValueError("adaptive-axis runs must share the reference output grid")
    d = np.asarray(states, dtype=float)[:, :6] - np.asarray(ref_states, dtype=float)[:, :6]
    return float(np.linalg.norm(d[:, :3], axis=1).max()), \
        float(np.linalg.norm(d[:, 3:], axis=1).max())


# ---------------------------------------------------------------------------
# One scenario, both axes
# ---------------------------------------------------------------------------

def _axis_budgets(spec: ConvergenceScenario, budgets: Budgets, axis: str) -> tuple:
    """Budget resolution for one axis: override key wins, then the scenario's
    own budget field (works for spec objects built outside the registry),
    and axes in ``budget_exempt_axes`` record without gating."""
    if axis in spec.budget_exempt_axes:
        return None, None
    out = []
    for metric in BUDGET_METRICS:
        key = f"{spec.name}.{metric}"
        out.append(budgets.overrides[key] if key in budgets.overrides
                   else float(spec.budget[metric]))
    return tuple(out)


def _tolerance_axis(spec: ConvergenceScenario, budgets: Budgets) -> dict:
    rtol0, atol0 = float(spec.production["rtol"]), float(spec.production["atol"])
    max_step = float(spec.production["max_step_s"])
    step_cap = max_step if math.isfinite(max_step) else 1e9
    curve = []
    ref_t, ref_s, ref_steps = spec.adaptive(rtol0 * REFERENCE_FACTOR,
                                           max(atol0 * REFERENCE_FACTOR, 1e-10),
                                           step_cap)
    for f in TOL_FACTORS:
        rtol, atol = rtol0 * f, atol0 * f
        if rtol < _MIN_RTOL:
            raise ValueError(f"{spec.name}: tolerance grid below scipy's DOP853 floor")
        t, s, steps = spec.adaptive(rtol, atol, step_cap)
        e_pos, e_vel = _max_window_error(ref_t, ref_s, t, s)
        realised = None if steps is None else float(np.diff(steps).max())
        curve.append({"factor": f, "rtol": rtol, "atol": atol,
                      "error_position_m": e_pos, "error_velocity_m_s": e_vel,
                      "max_realised_step_s": realised})
    ratios = [TOL_FACTORS[i] / TOL_FACTORS[i + 1] for i in range(len(TOL_FACTORS) - 1)]
    pos_order, pos_pairs = observed_order([c["error_position_m"] for c in curve], ratios)
    vel_order, vel_pairs = observed_order([c["error_velocity_m_s"] for c in curve], ratios)
    order = min(pos_order, vel_order)
    default = curve[list(TOL_FACTORS).index(1.0)]
    tightest = curve[-1]
    budget_pos, budget_vel = _axis_budgets(spec, budgets, "tolerance")
    bound_pos = error_bound(default["error_position_m"], tightest["error_position_m"],
                            pos_order, REFERENCE_FACTOR / TOL_FACTORS[-1])
    bound_vel = error_bound(default["error_velocity_m_s"], tightest["error_velocity_m_s"],
                            vel_order, REFERENCE_FACTOR / TOL_FACTORS[-1])

    justified = order >= spec.min_order["tolerance"]
    if budget_pos is not None:
        justified &= bound_pos * spec.safety <= budget_pos
    if budget_vel is not None:
        justified &= bound_vel * spec.safety <= budget_vel
    passing = [c for c in curve
               if (budget_pos is None or c["error_position_m"] * spec.safety <= budget_pos)
               and (budget_vel is None or c["error_velocity_m_s"] * spec.safety <= budget_vel)]
    recommended = passing[0]["factor"] if passing else None
    if not justified:
        verdict = "not justified: order below minimum" if order < spec.min_order["tolerance"] \
            else "not justified: error bound exceeds budget/safety"
    elif recommended is not None and recommended > 1.0:
        verdict = f"justified; conservative ({recommended:g}x looser would still meet the budget)"
    elif recommended is not None and recommended < 1.0:
        verdict = "justified; default sits on the loosest budget-meeting grid point"
    else:
        verdict = "justified; default is the recommended setting"
    realised = default["max_realised_step_s"]
    max_step_active = None
    if realised is not None and math.isfinite(max_step):
        max_step_active = bool(realised >= 0.99 * max_step)

    residual = {
        "name": f"{spec.name}.tolerance",
        "reference": f"same model re-run by DOP853 at rtol={rtol0 * REFERENCE_FACTOR:g}, "
                     f"atol={atol0 * REFERENCE_FACTOR:g} (tightest sweep run)",
        "category": CHECK_NUMERICAL,
        "position_residual_m": bound_pos,
        "velocity_residual_m_s": bound_vel,
        "tolerance": {"position_m": budget_pos, "velocity_m_s": budget_vel},
    }
    if budget_pos is not None or budget_vel is not None or "tolerance" in spec.budget_exempt_axes:
        residual["within_tolerance"] = bool(justified)
    return {"axis": "tolerance", "curve": curve,
            "order": {"position": pos_order, "velocity": vel_order, "governing": order,
                      "pair_slopes": {"position": pos_pairs, "velocity": vel_pairs},
                      "minimum_required": float(spec.min_order["tolerance"])},
            "error_bound": {"at_default_position_m": bound_pos,
                            "at_default_velocity_m_s": bound_vel},
            "recommendation": {"justified_default": bool(justified), "verdict": verdict,
                               "recommended_factor": recommended,
                               "max_step_active_at_default": max_step_active},
            "nonmonotonic": len(pos_pairs) + len(vel_pairs) < 2 * (len(TOL_FACTORS) - 1),
            "residual": residual}


def _step_axis(spec: ConvergenceScenario, budgets: Budgets) -> dict:
    grid = [float(h) for h in spec.rk4_grid]
    ref_h = grid[-1] / 2.0
    ref_t, ref_s = rk4(spec.rhs, _initial_state(spec), spec.duration_s, ref_h)
    ref_final = _final_state(ref_t, ref_s)
    curve = []
    for h in grid:
        t, s = rk4(spec.rhs, _initial_state(spec), spec.duration_s, h)
        d = _final_state(t, s) - ref_final
        curve.append({"step_s": float(t[1] - t[0]), "nominal_step_s": h,
                      "n_steps": int(t.size - 1),
                      "error_position_m": float(np.linalg.norm(d[:3])),
                      "error_velocity_m_s": float(np.linalg.norm(d[3:]))})
    ratios = [grid[i] / grid[i + 1] for i in range(len(grid) - 1)]
    pos_order, pos_pairs = observed_order([c["error_position_m"] for c in curve], ratios)
    vel_order, vel_pairs = observed_order([c["error_velocity_m_s"] for c in curve], ratios)
    order = min(pos_order, vel_order)
    budget_pos, budget_vel = _axis_budgets(spec, budgets, "fixed_step")
    budgets_present = budget_pos is not None or budget_vel is not None

    def fits(err_pos, err_vel):
        return ((budget_pos is None or err_pos * spec.safety <= budget_pos)
                and (budget_vel is None or err_vel * spec.safety <= budget_vel))

    passing = [c for c in curve if fits(c["error_position_m"], c["error_velocity_m_s"])]
    fine = curve[-1]
    if passing and budgets_present:
        rec = passing[0]
        rec_step, measured = rec["step_s"], True
        bound_pos = rec["error_position_m"] * (1.0 + (ref_h / rec["step_s"]) ** pos_order)
        bound_vel = rec["error_velocity_m_s"] * (1.0 + (ref_h / rec["step_s"]) ** vel_order)
    elif order > 0.0 and (budget_pos is not None or budget_vel is not None):
        # power-law step that would meet budget/safety, anchored at the finest
        # measured grid point; reported as an extrapolation, not a measurement
        target_pos = budget_pos / spec.safety if budget_pos is not None else math.inf
        target_vel = budget_vel / spec.safety if budget_vel is not None else math.inf
        shrink = min((target_pos / fine["error_position_m"]) ** (1.0 / pos_order)
                     if math.isfinite(target_pos) else math.inf,
                     (target_vel / fine["error_velocity_m_s"]) ** (1.0 / vel_order)
                     if math.isfinite(target_vel) else math.inf)
        rec_step, measured = fine["step_s"] * shrink, False
        bound_pos = fine["error_position_m"] * shrink ** pos_order if pos_order > 0 else fine["error_position_m"]
        bound_vel = fine["error_velocity_m_s"] * shrink ** vel_order if vel_order > 0 else fine["error_velocity_m_s"]
    else:
        rec_step, measured = None, False
        bound_pos, bound_vel = fine["error_position_m"], fine["error_velocity_m_s"]

    demonstrated = measured or not budgets_present
    justified = (order >= spec.min_order["fixed_step"]
                 and (rec_step is not None or not budgets_present) and demonstrated)
    if order < spec.min_order["fixed_step"]:
        verdict = "order below minimum: no trustworthy step recommendation"
    elif not budgets_present:
        verdict = ("no fixed-step budget asserted on this axis (this scenario has "
                   "no production fixed-step setting): observed order recorded, "
                   "adaptive default justified by the sweep")
    elif rec_step is None:
        verdict = "no convergent step recommendation derivable from the curve"
    elif not measured:
        verdict = ("budget only reachable below the measured grid (power-law "
                   "extrapolation): recorded, not a demonstrated pass")
    else:
        verdict = "budget met inside the measured grid"

    legacy = float(spec.production["max_step_s"])
    legacy_audit = None
    if math.isfinite(legacy) and order > 0.0:
        legacy_audit = {
            "step_s": legacy,
            "extrapolated": bool(legacy > grid[0]),
            "error_position_m_est": fine["error_position_m"] * (legacy / fine["step_s"]) ** pos_order,
            "error_velocity_m_s_est": fine["error_velocity_m_s"] * (legacy / fine["step_s"]) ** vel_order,
            "note": "asymptotic power-law estimate of the legacy fixed-step "
                    "path (integrator='rk4', step_size = the adaptive max step); "
                    "outside the measured grid once 'extrapolated': true"}

    residual = {
        "name": f"{spec.name}.fixed_step",
        "reference": f"same model, fixed-step RK4 at h={ref_h:g}s (tightest sweep run)",
        "category": CHECK_NUMERICAL,
        "position_residual_m": bound_pos,
        "velocity_residual_m_s": bound_vel,
        "tolerance": {"position_m": budget_pos, "velocity_m_s": budget_vel},
    }
    if budget_pos is not None or budget_vel is not None or "fixed_step" in spec.budget_exempt_axes:
        residual["within_tolerance"] = bool(justified)
    return {"axis": "fixed_step", "curve": curve,
            "order": {"position": pos_order, "velocity": vel_order, "governing": order,
                      "pair_slopes": {"position": pos_pairs, "velocity": vel_pairs},
                      "minimum_required": float(spec.min_order["fixed_step"])},
            "recommendation": {"justified": bool(justified), "verdict": verdict,
                               "recommended_step_s": rec_step, "recommended_measured": measured,
                               "budget_exempt": not budgets_present,
                               "legacy_fixed_step_audit": legacy_audit},
            "nonmonotonic": len(pos_pairs) + len(vel_pairs) < 2 * (len(grid) - 1),
            "residual": residual}


def _initial_state(spec: ConvergenceScenario) -> np.ndarray:
    return np.asarray(spec.state0, dtype=float)


def run_scenario(spec: ConvergenceScenario, budgets: Budgets | Mapping | None = None,
                 axes: Sequence[str] | None = None) -> dict:
    """One key scenario, both sweep axes (or a named subset): the curve, the
    order, the error bound and the recommendation for each axis."""
    budgets = Budgets.coerce(budgets)
    axes = list(AXES) if axes is None else list(axes)
    for axis in axes:
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis!r}; have {AXES}")
    out = {"scenario": spec.to_dict(), "axes": {}}
    if "tolerance" in axes:
        out["axes"]["tolerance"] = _tolerance_axis(spec, budgets)
    if "fixed_step" in axes:
        out["axes"]["fixed_step"] = _step_axis(spec, budgets)
    out["residuals"] = [out["axes"][a]["residual"] for a in axes]
    return out


# ---------------------------------------------------------------------------
# Study / report
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceStudy:
    """One V-03 run: per-scenario sweeps and the digested
    verification-report-1.0 document (kind ``integrator-convergence``)."""

    scenarios: tuple
    report: dict
    report_path: object = None

    @property
    def result(self) -> str:
        return status_result(self.report["status"])

    def to_dict(self) -> dict:
        return {"scenarios": [dict(s) for s in self.scenarios],
                "status": self.report["status"],
                "report_file": str(self.report_path) if self.report_path else None}


def run_convergence_study(scenarios: Sequence[str] | None = None,
                          budgets: Budgets | Mapping | None = None,
                          safety: float | None = None,
                          report_path=None) -> ConvergenceStudy:
    """Sweep every requested key scenario (default: all four), grade the
    results against the budgets and build the ``integrator-convergence``
    verification report. Honours ``$ORBITOPT_VERIFICATION_REPORT_DIR`` like
    every other verify entry point (``report_path`` overrides it)."""
    import os

    if safety is not None and float(safety) <= 0.0:
        raise ValueError("safety margin must be > 0")
    names = list(DEFAULT_SCENARIOS) if scenarios is None else list(dict.fromkeys(scenarios))
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; have {sorted(SCENARIOS)}")
    budgets = Budgets.coerce(budgets)
    results = []
    for n in names:
        spec = SCENARIOS[n]
        if safety is not None:
            spec = _with_safety(spec, float(safety))
        results.append(run_scenario(spec, budgets))

    residuals = [r for res in results for r in res["residuals"]]
    label = "key-scenarios" if len(names) > 1 else names[0]
    report = build_report(
        "integrator-convergence", label, propagations=[], residuals=residuals,
        inputs={"safety_margin": float(safety or DEFAULT_SAFETY_MARGIN),
                "tolerance_grid_factors": list(TOL_FACTORS),
                "reference_factor": REFERENCE_FACTOR,
                "budgets": budgets.to_dict(),
                "scenarios": results})
    report = finalize_report(report, report_path)
    path = report_path
    if path is None and os.environ.get(REPORT_DIR_ENV):
        path = default_report_path(report, os.environ[REPORT_DIR_ENV])
    return ConvergenceStudy(scenarios=tuple(results), report=report, report_path=path)


def _with_safety(spec: ConvergenceScenario, safety: float) -> ConvergenceScenario:
    from dataclasses import replace

    if safety <= 0.0:
        raise ValueError("safety margin must be > 0")
    return replace(spec, safety=float(safety))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _ascii_curve(axis: dict) -> str:
    """Log-log text curve (the convergence plot for terminals)."""
    res = axis["curve"]
    key = "step_s" if axis["axis"] == "fixed_step" else "factor"
    lines = []
    for c in res:
        e = max(c["error_position_m"], 1e-30)
        bar = "#" * max(1, int(round(8 + 4 * math.log10(e))))
        lines.append(f"    {key}={c[key]:>10.4g}  err={e:>10.3g} m  {bar}")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Run the convergence study; print one curve + verdict block per
    scenario; exit 1 on a failed gate check."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.integrator_convergence",
        description="V-03: step/tolerance convergence sweeps for the key scenarios; "
                    "numerical error bounds and default-setting verdicts in a "
                    "verification report.")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                        help=f"comma list from {sorted(SCENARIOS)} (default: all)")
    parser.add_argument("--budget", action="append", default=[], metavar="KEY=VALUE",
                        help="override one budget: KEY = <scenario>.<position_m|velocity_m_s>; "
                             "VALUE 'none' records without gating")
    parser.add_argument("--safety", type=float, default=None, metavar="M",
                        help=f"safety factor for the 'justified' verdict "
                             f"(default {DEFAULT_SAFETY_MARGIN:g})")
    parser.add_argument("--plot", action="store_true", help="print the ASCII convergence curves")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="report path (default: $ORBITOPT_VERIFICATION_REPORT_DIR)")
    parser.add_argument("--json", action="store_true",
                        help="print the verification report document instead of the summary")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.budget:
        key, _, value = item.partition("=")
        overrides[key] = value
    study = run_convergence_study(scenarios=args.scenarios.split(","), budgets=overrides,
                                  safety=args.safety, report_path=args.output)
    if args.json:
        print(json.dumps(study.report, indent=2, sort_keys=True))
        return 0 if study.result != "failed" else 1

    for res in study.scenarios:
        spec = res["scenario"]
        print(f"\n=== {spec['name']} ===\n    {spec['description']}")
        print(f"    audits: {spec['audits']}")
        for axis_key, axis in res["axes"].items():
            o = axis["order"]
            rec = axis["recommendation"]
            print(f"    [{axis_key}] observed order {o['governing']:.2f} "
                  f"(min {o['minimum_required']:g}), curve monotonic: "
                  f"{'yes' if not axis['nonmonotonic'] else 'noise recorded'}")
            if axis_key == "tolerance":
                b = axis["error_bound"]
                print(f"        error bound at default: {b['at_default_position_m']:.3g} m / "
                      f"{b['at_default_velocity_m_s']:.3g} m/s")
                print(f"        verdict: {rec['verdict']}")
            else:
                print(f"        verdict: {rec['verdict']}"
                      + (f"; recommended fixed step {rec['recommended_step_s']:.4g} s"
                         if rec["recommended_step_s"] else ""))
                legacy = rec.get("legacy_fixed_step_audit")
                if legacy:
                    print(f"        legacy rk4 at {legacy['step_s']:g}s: est "
                          f"{legacy['error_position_m_est']:.3g} m / "
                          f"{legacy['error_velocity_m_s_est']:.3g} m/s"
                          f"{' (extrapolated)' if legacy['extrapolated'] else ''}")
            if args.plot:
                print(_ascii_curve(axis))
    print(f"\nverification {format_status(study.report['status'])}"
          + (f" -> {study.report_path}" if study.report_path else ""))
    return 1 if study.result == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
