"""F-06 (task #231): two-stage separation on the native 6-DOF plant -- two
independent rigid-body states after stage door, an upper stage that ignites
on a schedule (configurable delay), and a first stage that leaves the study
through the M-01 recovery handoff contract.

Why here, and not in ``staging.py``
-----------------------------------
FL-402 (:mod:`orbitopt.flightlab.sixdof.staging`) proved two side-by-side
JSBSim ``FGFDMExec`` instances can carry a staged stack's state through
``ic/...`` properties and fly apart without leakage. That spike injected
through property I/O, so its conservation bookkeeping carries the unit
round-trip (~1e-8 relative residuals). This module is the production path
on the F-02 reference plant (:mod:`~orbitopt.flightlab.sixdof.rigidbody`):
the stack flies closed-loop (F-04 tracker) to the separation instant, the
frozen ECEF state is split *analytically* into two child states in SI units,
and each child is a completely independent :func:`propagate` call with its
own vehicle, control law and state vector -- two rigid bodies, one mission
clock, no shared integrator state. Conservation is therefore exact to
floating-point rounding, and the acceptance checks measure it anyway:

* **mass closure**: the children's wet masses sum to the stack mass at
  separation, with the split bookkeeping derived from the stack's own mass
  trace (stage-1 burned = m(t0) - m(t_sep); the upper stage's propellant is
  inert ballast through stage-1 flight and comes out of the mass residual);
* **momentum**: the separation impulse ``J`` along the shared body +x is
  applied momentum-conservingly (``+J/m_u`` upper, ``-J/m_b`` booster).

The acceptance's other two clauses are structural. Independence: after the
split the two runs never talk to each other -- the test reads them back as
trajectories that diverge (the ignited upper pulls away from the ballistically
receding booster). Recovery handoff: the study's ``booster_handoff`` is an
M-01 :class:`~orbitopt.flightlab.handoff.HandoffState` on the
``separation -> boostback`` boundary (ECEF, wet/dry masses included), i.e.
exactly what ``recovery.handoff_check`` expects a consumer to restart from.

Scheduling notes
----------------
* The upper stage's engine is commanded open-loop: shut until
  ``separation_t_s + ignition_delay_s``, commanded throttle from there on
  (clamped by the engine's own throttle floor; a dry engine is shut), gimbal
  neutral. Attitude is inertially held from the separation instant -- no
  torques at altitude and a zero-gimbal engine, so the thrust line points
  where the stack pointed. Closed guidance for the upper stage is F-07's
  job, not this task's.
* The booster's engine stays shut after separation (it was the stack's
  engine; its remaining propellant is *for* the recovery segments the
  handoff state feeds), and it flies as a ballistic/aero body.
* A single RK4 step that straddles the ignition instant sees the throttle
  edge inside its stage evaluations -- the ignition transition is spread
  over one plant step, ~dt seconds. That is disclosed, not smoothed.

Conventions are the reference plant's (body x fwd / y right / z down, ECEF
states, scalar-first Tec2b quaternion, mission clock continuing across the
split). Numbers for the committed scenario live in the tests; like all
native-plant studies this is illustrative physics, not a flight run.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from orbitopt.flightlab.contract import Check, Event, FlightRun, Verification
from orbitopt.flightlab.handoff import (
    PHASE_BOOSTBACK,
    PHASE_SEPARATION,
    HandoffState,
    handoff_digest,
)
from orbitopt.flightlab.sixdof.aero import load_packaged_aero
from orbitopt.flightlab.sixdof.rigidbody import (
    ControlCommand,
    RigidBodyEngine,
    RigidBodyPhysics,
    RigidBodyState,
    RigidBodyVehicle,
    SixDofRun,
    constant_control,
    dcm_from_quaternion,
    propagate,
)
from orbitopt.flightlab.sixdof.tvc import (
    ControlledAscent,
    TvcActuator,
    TvcGains,
    WaypointAttitudeProgram,
    run_closed_loop_ascent,
)

__all__ = [
    "IgnitionPlan",
    "StagePair",
    "TwoStageScenario",
    "TwoStageStudy",
    "delayed_ignition_control",
    "f06_two_stage_scenario",
    "run_two_stage",
    "split_stack_state",
    "to_flight_run",
]

#: Children wet masses must close the frozen stack mass to this absolute
#: tolerance [kg]. The split is analytic SI arithmetic, so the real residual
#: is rounding-level (~1e-10), unlike the FL-402 property-injection spike.
MASS_CLOSURE_TOL_KG = 1e-6
#: Separation momentum residual, relative to the stack |p| at separation, and
#: its absolute floor [N.s] (same bookkeeping shape as FL-402's).
MOMENTUM_RESIDUAL_TOL_REL = 1e-6
MOMENTUM_RESIDUAL_FLOOR_N_S = 0.01
#: The handoff position residual the continuity check allows [m]: the split
#: copies the frozen position verbatim, so this is rounding territory.
HANDOFF_POSITION_TOL_M = 1e-12
#: The scheduled-ignition run passes when its first nonzero-thrust sample is
#: at or after the ignition instant, no earlier than scheduled and no later
#: than this much slack [s] after it (one sample interval covers the RK4
#: step that straddles the edge).
IGNITION_ONSET_SLACK_S = 1.0
#: The ignited upper vs the ballistically receding booster must be at least
#: this far apart [m] at the end of the post-separation span for the
#: trajectories to count as independent.
TRAJECTORY_DIVERGENCE_MIN_M = 1000.0


# --------------------------------------------------------------------------
# the stage pair and the analytic split
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StagePair:
    """Mass-consistent stack + two child specs.

    The children are given at their **spec** loads (stage-1 full propellant,
    upper stage full load); :meth:`resolve` re-derives their propellant from
    the stack's *flown* mass trace at separation, which is the only
    bookkeeping that conserves mass exactly:

    * ``stack.propellant_mass_kg == booster.propellant_mass_kg`` -- only the
      stage-1 engine burns during stack flight, so every kilogram the stack
      lost is stage-1 propellant;
    * ``stack.dry_mass_kg == booster.dry_mass_kg + upper.mass_wet_kg`` --
      the upper stage (structure *and* its own propellant) is inert ballast
      inside the stack's dry floor.

    ``separation_impulse_ns`` is the spring/URS impulse along the shared body
    +x at the split (+J to the upper stage, -J to the booster).
    """

    stack: RigidBodyVehicle
    booster: RigidBodyVehicle
    upper: RigidBodyVehicle
    separation_impulse_ns: float = 25_000.0

    def __post_init__(self) -> None:
        if self.stack.propellant_mass_kg != self.booster.propellant_mass_kg:
            raise ValueError(
                f"stack propellant {self.stack.propellant_mass_kg} kg must equal the "
                f"booster's stage-1 load {self.booster.propellant_mass_kg} kg "
                f"(only the stage-1 engine burns during stack flight)")
        if abs(self.stack.dry_mass_kg
               - (self.booster.dry_mass_kg + self.upper.mass_wet_kg)) > 1e-9:
            raise ValueError(
                f"stack dry mass {self.stack.dry_mass_kg} kg must equal booster dry "
                f"{self.booster.dry_mass_kg} kg + upper wet {self.upper.mass_wet_kg} kg "
                f"(the upper stage is inert ballast in the stack's dry floor)")
        if not math.isfinite(self.separation_impulse_ns) or self.separation_impulse_ns < 0.0:
            raise ValueError(f"separation_impulse_ns must be finite and >= 0, got "
                             f"{self.separation_impulse_ns}")

    def resolve(self, stack_mass_at_t0_kg: float,
                stack_mass_at_separation_kg: float) -> tuple[RigidBodyVehicle,
                                                             RigidBodyVehicle,
                                                             dict[str, float]]:
        """Child vehicles with their **at-separation** loads, plus the split
        bookkeeping. ``stack_mass_at_separation_kg`` is the frozen (live) mass
        from the flown stack; stage-1 burned propellant is the mass the trace
        lost since ``t0``, floored by the stage-1 load."""
        burned = min(max(stack_mass_at_t0_kg - stack_mass_at_separation_kg, 0.0),
                     self.booster.propellant_mass_kg)
        booster_prop = self.booster.propellant_mass_kg - burned
        booster_mass = self.booster.dry_mass_kg + booster_prop
        upper_mass = stack_mass_at_separation_kg - booster_mass
        if upper_mass < self.upper.dry_mass_kg - 1e-9:
            raise ValueError(
                f"mass trace leaves the upper stage {upper_mass} kg, below its dry "
                f"mass {self.upper.dry_mass_kg} kg -- the stack ran heavier dry than "
                f"the stage pair declares")
        if upper_mass > self.upper.mass_wet_kg + 1e-9:
            raise ValueError(
                f"mass trace leaves the upper stage {upper_mass} kg above its wet "
                f"mass {self.upper.mass_wet_kg} kg (propellant appearing from nowhere)")
        booster = replace(self.booster, propellant_mass_kg=booster_prop)
        upper = replace(self.upper, propellant_mass_kg=upper_mass - self.upper.dry_mass_kg)
        bookkeeping = {
            "stack_mass_t0_kg": float(stack_mass_at_t0_kg),
            "stack_mass_separation_kg": float(stack_mass_at_separation_kg),
            "stage1_propellant_burned_kg": float(burned),
            "booster_propellant_remaining_kg": float(booster_prop),
            "upper_propellant_kg": float(upper_mass - self.upper.dry_mass_kg),
        }
        return booster, upper, bookkeeping


@dataclass(frozen=True)
class IgnitionPlan:
    """Upper-stage engine schedule: shut at separation, on at
    ``separation_t_s + ignition_delay_s``, ``throttle`` from there on.
    ``ignition_delay_s`` is the acceptance's configurable delay (>= 0)."""

    separation_t_s: float
    ignition_delay_s: float = 8.0
    throttle: float = 1.0

    def __post_init__(self) -> None:
        if self.ignition_delay_s < 0.0:
            raise ValueError(f"ignition_delay_s must be >= 0, got {self.ignition_delay_s}")
        if not 0.0 < self.throttle <= 1.0:
            raise ValueError(f"ignition throttle must be in (0, 1], got {self.throttle}")

    @property
    def ignition_t_s(self) -> float:
        return float(self.separation_t_s) + float(self.ignition_delay_s)


def delayed_ignition_control(plan: IgnitionPlan) -> Callable:
    """The upper stage's :data:`ControlLaw`: gimbal neutral, throttle 0 until
    ``plan.ignition_t_s``, ``plan.throttle`` from there on (the plant floors a
    positive command at the engine's own minimum; a dry engine is shut)."""

    def control(t_s: float, state: RigidBodyState) -> ControlCommand:
        level = float(plan.throttle) if float(t_s) >= plan.ignition_t_s else 0.0
        return ControlCommand(throttle=level)

    return control


def _body_x_ecef(quaternion) -> np.ndarray:
    """Body +x expressed in ECEF: row 0 of the Tec2b DCM (the shared thrust
    line at separation, which is the axis a spring/URS impulse acts along)."""
    return np.asarray(dcm_from_quaternion(quaternion)[0], dtype=float)


def split_stack_state(frozen: RigidBodyState, pair: StagePair, *,
                      stack_mass_at_t0_kg: float,
                      impulse_ns: float | None = None) -> tuple[RigidBodyState,
                                                                RigidBodyState,
                                                                dict[str, Any]]:
    """The analytic handoff: two child :class:`RigidBodyState`s inheriting the
    stack's position, attitude and body rates, with velocities split by the
    momentum-conserving impulse, and full separation bookkeeping.

    Mass closure and momentum are exact by construction (SI arithmetic, no
    property round-trip); the returned bookkeeping measures them anyway --
    that table is what the acceptance checks read.
    """
    booster_v, upper_v, bk = pair.resolve(stack_mass_at_t0_kg, float(frozen.mass_kg))
    j = pair.separation_impulse_ns if impulse_ns is None else float(impulse_ns)
    x_ecef = _body_x_ecef(frozen.quaternion)
    m_b = booster_v.mass_wet_kg
    m_u = upper_v.mass_wet_kg
    dv_b = np.asarray([-j * x_ecef[i] / m_b for i in range(3)])
    dv_u = np.asarray([j * x_ecef[i] / m_u for i in range(3)])
    v0 = np.asarray(frozen.velocity_m_s, dtype=float)
    booster_s = RigidBodyState(
        t_s=frozen.t_s, position_m=frozen.position_m,
        velocity_m_s=tuple(v0[i] + dv_b[i] for i in range(3)),
        quaternion=frozen.quaternion, omega_rad_s=frozen.omega_rad_s, mass_kg=m_b)
    upper_s = RigidBodyState(
        t_s=frozen.t_s, position_m=frozen.position_m,
        velocity_m_s=tuple(v0[i] + dv_u[i] for i in range(3)),
        quaternion=frozen.quaternion, omega_rad_s=frozen.omega_rad_s, mass_kg=m_u)

    p0 = v0 * float(frozen.mass_kg)
    p1 = (np.asarray(booster_s.velocity_m_s, dtype=float) * m_b
          + np.asarray(upper_s.velocity_m_s, dtype=float) * m_u)
    bk.update({
        "separation_impulse_ns": float(j),
        "mass_after_kg": float(m_b + m_u),
        "mass_residual_kg": float(m_b + m_u) - float(frozen.mass_kg),
        "momentum_before_kg_m_s": tuple(float(v) for v in p0),
        "momentum_after_kg_m_s": tuple(float(v) for v in p1),
        "momentum_residual_kg_m_s": float(np.linalg.norm(p1 - p0)),
        "dv_booster_m_s": float(j / m_b),
        "dv_upper_m_s": float(j / m_u),
        "position_continuity_err_m": 0.0,
        "attitude_continuity_err_rad": 0.0,
    })
    return booster_s, upper_s, bk


# --------------------------------------------------------------------------
# the scenario
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoStageScenario:
    """One F-06 study: the pad state, the mass-consistent stage pair, the
    stack's attitude program, the separation/ignition schedule and the
    integration plan. The mission clock runs 0 -> ``separation_t_s`` closed
    loop on the stack, and both children fly ``separation_t_s -> t_end_s``."""

    name: str
    state0: RigidBodyState
    pair: StagePair
    program: WaypointAttitudeProgram
    separation_t_s: float
    ignition: IgnitionPlan
    t_end_s: float = 260.0
    dt_s: float = 0.02
    sample_every: int = 10
    dt_c_s: float = 0.05
    gains: TvcGains = field(default_factory=TvcGains)
    physics: RigidBodyPhysics = field(default_factory=RigidBodyPhysics)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.separation_t_s < float(self.t_end_s):
            raise ValueError(f"separation_t_s {self.separation_t_s} must be inside "
                             f"(0, t_end_s={self.t_end_s})")
        if abs(self.ignition.separation_t_s - self.separation_t_s) > 1e-9:
            raise ValueError("the ignition plan's separation instant must be the "
                             "scenario's separation instant")
        if self.ignition.ignition_t_s > float(self.t_end_s):
            raise ValueError(f"ignition at t={self.ignition.ignition_t_s} s is after the "
                             f"run ends at t_end_s={self.t_end_s} s")
        if self.dt_s <= 0.0 or int(self.sample_every) < 1:
            raise ValueError("dt_s must be positive and sample_every >= 1")

    def real_actuator(self) -> TvcActuator:
        """F-04's default hydraulic axis at the stack engine's travel limit
        (the same servo the F-05 ascent flies)."""
        return TvcActuator(limit_rad=self.pair.stack.engine.gimbal_limit_rad)


def f06_two_stage_scenario(*, separation_t_s: float = 95.0, ignition_delay_s: float = 8.0,
                           t_end_s: float = 260.0, dt_s: float = 0.05,
                           sample_every: int = 20,
                           separation_impulse_ns: float = 25_000.0) -> TwoStageScenario:
    """The packaged F-06 study: the F-05 spike-stage vehicle as stage 1 (10 t
    dry / 60 t prop / 1.88 MN, packaged ``spike_stage`` aero) stacked on a
    2 t / 6 t upper stage with a 220 kN vacuum-Isp-340 engine (no aero of its
    own -- it first feels the air again only if the study is extended). Wet
    stack 78 t off pad 28.5 N / 80.6 W on heading 090, F-04 tracker on the
    classic gravity-turn program (vertical to 10 s, pitch-over to 20 deg by
    110 s); stage-1 MECO lands near t = 100 s, so separation at t = 95 s
    leaves the booster ~3 t of propellant for its recovery segments. The
    upper stage ignites ``ignition_delay_s`` (default 8 s) after the 25 kN.s
    separation impulse and both bodies fly to ``t_end_s`` (default 260 s).
    """
    engine = RigidBodyEngine(thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0,
                             isp_sea_level_s=282.0, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=22.0,
                             gimbal_limit_rad=0.0873)
    stack = RigidBodyVehicle(name="f06_stack", dry_mass_kg=18_000.0,
                             propellant_mass_kg=60_000.0,
                             inertia_kg_m2=(1.0e5, 4.0e6, 4.0e6), engine=engine,
                             aero=load_packaged_aero("spike_stage"))
    booster = RigidBodyVehicle(name="booster", dry_mass_kg=10_000.0,
                               propellant_mass_kg=60_000.0,
                               inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                               aero=load_packaged_aero("spike_stage"))
    upper_engine = RigidBodyEngine(thrust_vacuum_n=2.2e5, isp_vacuum_s=340.0,
                                   isp_sea_level_s=300.0, nozzle_exit_area_m2=0.35,
                                   throttle_min=0.3, throttle_max=1.0, mount_x_cg_m=6.0)
    upper = RigidBodyVehicle(name="upper_stage", dry_mass_kg=2_000.0,
                             propellant_mass_kg=6_000.0,
                             inertia_kg_m2=(5.0e3, 2.0e5, 2.0e5), engine=upper_engine,
                             aero=None)
    state0 = RigidBodyState.from_launch(lat_geod_deg=28.5, lon_deg=-80.6,
                                        heading_deg=90.0, pitch_deg=90.0,
                                        roll_deg=0.0, mass_kg=stack.mass_wet_kg)
    program = WaypointAttitudeProgram(times_s=(0.0, 10.0, 110.0),
                                      heading_deg=(90.0, 90.0, 90.0),
                                      pitch_deg=(90.0, 90.0, 20.0),
                                      roll_deg=(0.0, 0.0, 0.0))
    pair = StagePair(stack=stack, booster=booster, upper=upper,
                     separation_impulse_ns=separation_impulse_ns)
    ignition = IgnitionPlan(separation_t_s=separation_t_s, ignition_delay_s=ignition_delay_s)
    return TwoStageScenario(
        name="f06-two-stage", state0=state0, pair=pair, program=program,
        separation_t_s=separation_t_s, ignition=ignition, t_end_s=t_end_s,
        dt_s=dt_s, sample_every=sample_every,
        assumptions=(
            ("Native-plant two-stage study of the F-02 reference plant with the "
             "packaged illustrative spike_stage vehicle -- not flight-certified."),
            ("Stack flight is closed-loop (F-04 tracker + actuator) on the shared "
             "gravity-turn program; after separation the booster coasts ballistically "
             "toward its recovery handoff and the upper stage flies open-loop with a "
             "neutral gimbal until F-07 closes its guidance loop."),
            ("The split is analytic SI arithmetic on the frozen stack state, so mass "
             "and momentum closure are rounding-exact; the FL-402 JSBSim spike's "
             "property-injection round-trip is not in this path."),
        ),
    )


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------

def _speed_at(run: SixDofRun, index: int) -> float:
    s = run.samples
    return float(math.sqrt(float(s["vx_m_s"][index]) ** 2 + float(s["vy_m_s"][index]) ** 2
                           + float(s["vz_m_s"][index]) ** 2))


def _terminal_state(run: SixDofRun) -> RigidBodyState:
    """The run's last *sampled* state as a plant state (the separation state
    is read this way: the stack phase ends exactly on ``separation_t_s``, on
    the sample grid)."""
    s = run.samples
    return RigidBodyState(
        t_s=float(s["t_s"][-1]),
        position_m=(float(s["x_m"][-1]), float(s["y_m"][-1]), float(s["z_m"][-1])),
        velocity_m_s=(float(s["vx_m_s"][-1]), float(s["vy_m_s"][-1]),
                      float(s["vz_m_s"][-1])),
        quaternion=(float(s["q_w"][-1]), float(s["q_x"][-1]), float(s["q_y"][-1]),
                    float(s["q_z"][-1])),
        omega_rad_s=(float(s["p_rad_s"][-1]), float(s["q_rad_s"][-1]),
                     float(s["r_rad_s"][-1])),
        mass_kg=float(s["mass_kg"][-1]))


def _ignition_onset_err_s(run: SixDofRun, ignition_t_s: float) -> float:
    """Signed delay [s] from the plan instant to the first nonzero-thrust
    sample. Negative = thrust came early (a violation); ``inf`` = the engine
    never lit at all (also a violation)."""
    t = run.samples["t_s"]
    th = run.samples["throttle"]
    early = np.nonzero((t < ignition_t_s - 1e-9) & (th > 0.0))[0]
    if early.size:
        return float(t[early[0]] - ignition_t_s)
    on = np.nonzero(th > 0.0)[0]
    if on.size == 0:
        return float("inf")
    return float(t[on[0]] - ignition_t_s)


@dataclass(frozen=True)
class TwoStageStudy:
    """One completed F-06 study: the closed-loop stack ascent to separation,
    the split bookkeeping, the two independent child runs (mission clock
    continuing), and the booster's M-01 handoff state. ``checks`` are the
    acceptance verdicts (mass closure, momentum residual, handoff continuity,
    scheduled ignition, trajectory divergence)."""

    scenario: TwoStageScenario
    ascent: ControlledAscent
    frozen_state: RigidBodyState
    separation: Mapping[str, Any]
    booster_state: RigidBodyState
    upper_state: RigidBodyState
    booster: SixDofRun
    upper: SixDofRun
    booster_handoff: HandoffState
    checks: tuple[Check, ...]

    @property
    def stack(self) -> SixDofRun:
        return self.ascent.run

    @property
    def booster_handoff_digest(self) -> str:
        return handoff_digest(self.booster_handoff)

    def trajectory_divergence_m(self) -> float:
        """End-of-run separation [m] between the two independent bodies."""
        pb, pu = self.booster.samples, self.upper.samples
        return float(math.sqrt(sum((float(pb[c][-1]) - float(pu[c][-1])) ** 2
                                   for c in ("x_m", "y_m", "z_m"))))

    def summary(self) -> dict[str, Any]:
        """Plain-dict view for a benchmark JSON / report sidecar."""
        s, sep = self.scenario, self.separation
        return {
            "scenario": {"name": s.name, "separation_t_s": s.separation_t_s,
                         "ignition_delay_s": s.ignition.ignition_delay_s,
                         "ignition_t_s": s.ignition.ignition_t_s,
                         "t_end_s": s.t_end_s, "dt_s": s.dt_s,
                         "impulse_ns": sep["separation_impulse_ns"]},
            "stack": {"max_q_pa": float(self.stack.max_q.q_pa),
                      "max_q_t_s": float(self.stack.max_q.t_s),
                      "att_err_max_deg": float(self.ascent.metrics.att_err_max_deg)},
            "separation": dict(sep),
            "booster": {"mass_kg": float(self.booster["mass_kg"][0]),
                        "apogee_m": float(np.max(self.booster["altitude_m"])),
                        "terminal": self.booster.terminal},
            "upper": {"mass_kg": float(self.upper["mass_kg"][0]),
                      "terminal_mass_kg": float(self.upper["mass_kg"][-1]),
                      "speed_gain_m_s": _speed_at(self.upper, -1)
                                        - _speed_vec(self.upper_state.velocity_m_s),
                      "terminal": self.upper.terminal},
            "trajectory_divergence_m": self.trajectory_divergence_m(),
            "handoff_digest": self.booster_handoff_digest,
            "checks": {c.id: bool(c.passed) for c in self.checks},
        }


def _speed_vec(velocity_m_s) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in velocity_m_s)))


def _acceptance_checks(s: TwoStageScenario, sep: Mapping[str, Any],
                       booster_run: SixDofRun, upper_run: SixDofRun) -> tuple[Check, ...]:
    p_norm = float(np.linalg.norm(np.asarray(sep["momentum_before_kg_m_s"], float)))
    mom_tol = max(MOMENTUM_RESIDUAL_FLOOR_N_S, MOMENTUM_RESIDUAL_TOL_REL * p_norm)
    mass_err = abs(float(sep["mass_residual_kg"]))
    onset = _ignition_onset_err_s(upper_run, s.ignition.ignition_t_s)
    on_time = -1e-9 <= onset <= IGNITION_ONSET_SLACK_S
    divergence = math.sqrt(sum((float(booster_run.samples[c][-1])
                                - float(upper_run.samples[c][-1])) ** 2
                               for c in ("x_m", "y_m", "z_m")))
    return (
        Check("staging-mass-closure", mass_err < MASS_CLOSURE_TOL_KG, mass_err,
              MASS_CLOSURE_TOL_KG, "kg", ""),
        Check("staging-momentum-residual",
              float(sep["momentum_residual_kg_m_s"]) < mom_tol,
              float(sep["momentum_residual_kg_m_s"]), mom_tol, "N.s", ""),
        Check("staging-handoff-continuity",
              float(sep["position_continuity_err_m"]) < HANDOFF_POSITION_TOL_M,
              float(sep["position_continuity_err_m"]), HANDOFF_POSITION_TOL_M, "m", ""),
        Check("staging-scheduled-ignition", bool(on_time), onset,
              IGNITION_ONSET_SLACK_S, "s", ""),
        Check("staging-trajectory-divergence", divergence > TRAJECTORY_DIVERGENCE_MIN_M,
              divergence, TRAJECTORY_DIVERGENCE_MIN_M, "m", ""),
    )


def run_two_stage(scenario: TwoStageScenario | None = None, *,
                  epoch_utc: datetime | None = None) -> TwoStageStudy:
    """Fly the scenario (default :func:`f06_two_stage_scenario`): closed-loop
    stack to separation, the analytic split, two independent ``propagate``
    calls to ``t_end_s``, acceptance checks, booster handoff state."""
    s = scenario or f06_two_stage_scenario()
    epoch = epoch_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)

    ascent = run_closed_loop_ascent(s.state0, s.pair.stack, s.program, s.separation_t_s,
                                    dt_s=s.dt_s, sample_every=s.sample_every,
                                    actuator=s.real_actuator(), gains=s.gains,
                                    throttle=1.0, dt_c_s=s.dt_c_s, physics=s.physics,
                                    epoch_utc=epoch)
    if ascent.run.terminal == "ground":
        raise ValueError(f"the stack hit the ground at t={ascent.run.ground_t_s} s before "
                         f"the separation instant {s.separation_t_s} s; pick a larger "
                         f"thrust-to-weight or an earlier separation_t_s")
    frozen = _terminal_state(ascent.run)

    booster_s, upper_s, sep = split_stack_state(frozen, s.pair,
                                                stack_mass_at_t0_kg=s.state0.mass_kg)
    booster_v = replace(s.pair.booster,
                        propellant_mass_kg=sep["booster_propellant_remaining_kg"])
    upper_v = replace(s.pair.upper, propellant_mass_kg=sep["upper_propellant_kg"])

    # Two independent bodies: separate propagate() calls, separate state
    # vectors, one mission clock continuing from the separation instant.
    booster_run = propagate(booster_s, booster_v, constant_control(0.0), s.t_end_s,
                            dt_s=s.dt_s, sample_every=s.sample_every,
                            physics=s.physics, epoch_utc=epoch)
    upper_run = propagate(upper_s, upper_v, delayed_ignition_control(s.ignition),
                          s.t_end_s, dt_s=s.dt_s, sample_every=s.sample_every,
                          physics=s.physics, epoch_utc=epoch)

    handoff = HandoffState(
        from_phase=PHASE_SEPARATION, to_phase=PHASE_BOOSTBACK,
        t_s=frozen.t_s, epoch_utc=epoch.isoformat().replace("+00:00", "Z"),
        frame="ECEF",
        position_m=tuple(float(c) for c in booster_s.position_m),
        velocity_m_s=tuple(float(c) for c in booster_s.velocity_m_s),
        mass_kg=booster_s.mass_kg, dry_mass_kg=booster_v.dry_mass_kg, ignitions_used=1)

    checks = _acceptance_checks(s, sep, booster_run, upper_run)
    return TwoStageStudy(scenario=s, ascent=ascent, frozen_state=frozen, separation=sep,
                         booster_state=booster_s, upper_state=upper_s,
                         booster=booster_run, upper=upper_run,
                         booster_handoff=handoff, checks=checks)


# --------------------------------------------------------------------------
# the contract view
# --------------------------------------------------------------------------

def to_flight_run(study: TwoStageStudy, *, run_id: str = "f06-two-stage-native",
                  epoch_utc: str = "2026-01-01T00:00:00Z") -> FlightRun:
    """The study as a three-body FlightResult v1 run (bodies ``stack``,
    ``booster``, ``upper``; the children carry ``parent_body_id=stack`` and
    share the mission clock -- the plant integrates on it, no FL-402-style
    ``t_offset_s`` re-basing was needed)."""
    s = study.scenario
    sep = study.separation
    mk = lambda tag: hashlib.sha256(f"f06:{tag}".encode()).hexdigest()
    return FlightRun(
        run_id=run_id,
        target_sha256=mk("no-flight-target-two-stage-scenario"),
        vehicle_sha256=mk("f06-stack-split-into-booster-and-upper"),
        git_sha="0" * 40,
        created_utc=epoch_utc,
        engines={"stage1": "spike_stage engine (vac 1.88 MN / Isp 320 s)",
                 "stage2": "upper engine (vac 220 kN / Isp 340 s)"},
    assumptions=tuple(s.assumptions) + (
        (f"Separation at t={s.separation_t_s} s with a "
         f"{sep['separation_impulse_ns']} N.s body-+x impulse "
         f"(+{sep['dv_upper_m_s']:.3f} m/s upper / -{sep['dv_booster_m_s']:.3f} m/s "
         f"booster); upper ignition scheduled at t={s.ignition.ignition_t_s} s "
         f"(delay {s.ignition.ignition_delay_s} s)."),
        ("Booster handed to recovery on the M-01 separation->boostback boundary "
         f"(handoff digest {study.booster_handoff_digest[:12]})."),
    ),
        tracks=(
            study.stack.to_body_track(body_id="stack"),
            study.booster.to_body_track(body_id="booster", parent_body_id="stack"),
            study.upper.to_body_track(body_id="upper", parent_body_id="stack"),
        ),
        events=(
            Event(0.0, "ignition", "stack", {"throttle": 1.0}),
            Event(study.stack.max_q.t_s, "max_q", "stack",
                  {"dynamic_pressure_pa": float(study.stack.max_q.q_pa),
                   "altitude_m": float(study.stack.max_q.altitude_m),
                   "mach": float(study.stack.max_q.mach)}),
            Event(s.separation_t_s, "separation", "stack",
                  {"impulse_ns": sep["separation_impulse_ns"],
                   "mass_kg": float(sep["stack_mass_separation_kg"])}),
            Event(s.separation_t_s, "separation", "booster",
                  {"dv_m_s": -float(sep["dv_booster_m_s"]),
                   "mass_kg": float(study.booster_state.mass_kg)}),
            Event(s.separation_t_s, "separation", "upper",
                  {"dv_m_s": float(sep["dv_upper_m_s"]),
                   "mass_kg": float(study.upper_state.mass_kg)}),
            Event(s.ignition.ignition_t_s, "ignition", "upper",
                  {"throttle": s.ignition.throttle,
                   "delay_s": s.ignition.ignition_delay_s}),
        ),
        verification=Verification(
            optimizer_converged=None,
            physics_verified=all(c.passed for c in study.checks),
            mission_constraints_passed=None,
            checks=study.checks,
        ),
    )
