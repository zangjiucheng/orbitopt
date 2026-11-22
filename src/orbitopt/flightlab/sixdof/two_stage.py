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
* E10-02 (task #299): a scenario carrying ``hot=HotStagingPlan(...)`` lights
  the upper stage's engine while the stages are still mated -- the overlap
  leg flies a plume-corrected COMPOSITE engine (ship thrust folded into the
  stack's, sharing its gimbal travel), the stack phase is two closed-loop
  legs concatenated at the ignition instant, and there is no coast anywhere:
  ship ignition -> separation -> child runs, with the child upper stage
  already alight at spawn (``IgnitionPlan`` delay 0). The split bookkeeping
  charges the ship's mated burn separately (``connected_upper_burn_kg``),
  and the composite engine's single-gimbal-pivot / vacuum-Isp bookkeeping is
  the declared simplification, mirroring ``orbitopt.flightlab.hot_staging``.

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

from orbitopt.flightlab.backends import BACKEND_NATIVE_RIGIDBODY
from orbitopt.flightlab.contract import Check, Event, FlightRun, Verification
from orbitopt.flightlab.guidance.upper_stage import (
    GuidedUpperStageControl,
    UpperStageGuidanceSpec,
)
from orbitopt.flightlab.parity import load_parity_scenario, native_stage_pair
from orbitopt.flightlab.handoff import (
    PHASE_BOOSTBACK,
    PHASE_SEPARATION,
    HandoffState,
    handoff_digest,
)
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
    "HotStagingPlan",
    "IgnitionPlan",
    "StagePair",
    "TwoStageScenario",
    "TwoStageStudy",
    "delayed_ignition_control",
    "f06_two_stage_scenario",
    "guided_upper_stage_pair",
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
                stack_mass_at_separation_kg: float, *,
                connected_upper_burn_kg: float = 0.0) -> tuple[RigidBodyVehicle,
                                                               RigidBodyVehicle,
                                                               dict[str, float]]:
        """Child vehicles with their **at-separation** loads, plus the split
        bookkeeping. ``stack_mass_at_separation_kg`` is the frozen (live) mass
        from the flown stack; stage-1 burned propellant is the mass the trace
        lost since ``t0``, floored by the stage-1 load.

        E10-02 (task #299): with hot staging the upper engine also burns while
        mated, so ``connected_upper_burn_kg`` (the ship's own mated-phase burn)
        is carved out of the trace loss before attributing the remainder to
        stage-1 -- the mass/momentum split itself is unchanged, only the
        attribution. The stage-1 load remains the floor."""
        burned = min(max(stack_mass_at_t0_kg - stack_mass_at_separation_kg, 0.0),
                     self.booster.propellant_mass_kg)
        connected = min(max(float(connected_upper_burn_kg), 0.0), burned)
        burned_stage1 = burned - connected
        booster_prop = self.booster.propellant_mass_kg - burned_stage1
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
            "stage1_propellant_burned_kg": float(burned_stage1),
            "connected_upper_burn_kg": float(connected),
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


#: Standard gravity [m/s^2], the mass-flow constant of every engine model here.
_G0 = 9.80665


@dataclass(frozen=True)
class HotStagingPlan:
    """E10-02 (task #299): hot staging on the native plant. The upper stage's
    engine lights at ``ship_ignition_t_s`` **while the stages are still
    mated**, so there is no unpowered gap before ``separation_t_s`` -- the
    stack flies one powered trace through the overlap, on a composite engine
    (:meth:`composite_stack_engine`: plume-corrected ship thrust folded into
    the stack engine, sharing its gimbal travel), and the child upper stage
    is already alight when the split spawns it (its ``IgnitionPlan`` delay is
    0). The contract-level twin of ``orbitopt.flightlab.hot_staging``, which
    carries the same timeline for the point-mass backends.

    ``plume_axial_loss_fraction`` is the (declared, assumed) axial fraction of
    the mated ship's thrust lost to the booster's plume/interstage interaction
    -- the only representation of the ship-on-booster interaction this plant
    models. The ship's mated-phase propellant use (:meth:`ship_burn_kg`) is
    charged separately at the split (``connected_upper_burn_kg``).
    """

    ship_ignition_t_s: float
    separation_t_s: float
    throttle: float = 1.0
    plume_axial_loss_fraction: float = 0.04

    def __post_init__(self) -> None:
        if not math.isfinite(self.ship_ignition_t_s) or self.ship_ignition_t_s <= 0.0:
            raise ValueError(f"ship_ignition_t_s must be finite and > 0, got "
                             f"{self.ship_ignition_t_s}")
        if not self.ship_ignition_t_s < self.separation_t_s:
            raise ValueError(f"hot staging needs ship ignition strictly BEFORE "
                             f"separation ({self.ship_ignition_t_s} s >= "
                             f"{self.separation_t_s} s)")
        if not 0.0 < self.throttle <= 1.0:
            raise ValueError(f"hot-staging ship throttle must be in (0, 1], got "
                             f"{self.throttle}")
        if not 0.0 <= self.plume_axial_loss_fraction < 1.0:
            raise ValueError(f"plume_axial_loss_fraction must be in [0, 1), got "
                             f"{self.plume_axial_loss_fraction}")

    @property
    def overlap_s(self) -> float:
        """The mated powered overlap [s] between ship ignition and separation."""
        return float(self.separation_t_s) - float(self.ship_ignition_t_s)

    def ship_burn_kg(self, upper: RigidBodyVehicle) -> float:
        """Ship propellant [kg] spent during the mated overlap at ``throttle``
        (vacuum Isp bookkeeping -- the overlap sits above the dense air; the
        sea-level correction over a one-to-two-second window is negligible)."""
        eng = upper.engine
        return (eng.thrust_vacuum_n * self.throttle / (eng.isp_vacuum_s * _G0)
                * self.overlap_s)

    def composite_stack_engine(self, stack: RigidBodyVehicle,
                               upper: RigidBodyVehicle) -> RigidBodyEngine:
        """The stack's engine over the overlap: full stack thrust plus the
        plume-corrected ship thrust, at the flow-consistent blended Isp. The
        stack's gimbal, mount and throttle envelope are kept -- the ship is
        bolted to the booster for the whole overlap (declared simplification,
        shared with the OT fold in :mod:`orbitopt.flightlab.hot_staging`)."""
        eta = 1.0 - self.plume_axial_loss_fraction
        e_b, e_s = stack.engine, upper.engine
        f_b = float(e_b.thrust_vacuum_n)
        f_s = float(e_s.thrust_vacuum_n) * self.throttle
        flow = f_b / float(e_b.isp_vacuum_s) + f_s / float(e_s.isp_vacuum_s)
        flow_sl = f_b / float(e_b.isp_sea_level_s) + f_s / float(e_s.isp_sea_level_s)
        f_tot = f_b + eta * f_s
        return replace(e_b, thrust_vacuum_n=f_tot,
                       isp_vacuum_s=f_tot / flow, isp_sea_level_s=f_tot / flow_sl,
                       nozzle_exit_area_m2=float(e_b.nozzle_exit_area_m2)
                                            + float(e_s.nozzle_exit_area_m2))


def _body_x_ecef(quaternion) -> np.ndarray:
    """Body +x expressed in ECEF: row 0 of the Tec2b DCM (the shared thrust
    line at separation, which is the axis a spring/URS impulse acts along)."""
    return np.asarray(dcm_from_quaternion(quaternion)[0], dtype=float)


def split_stack_state(frozen: RigidBodyState, pair: StagePair, *,
                      stack_mass_at_t0_kg: float,
                      connected_upper_burn_kg: float = 0.0,
                      impulse_ns: float | None = None) -> tuple[RigidBodyState,
                                                                RigidBodyState,
                                                                dict[str, Any]]:
    """The analytic handoff: two child :class:`RigidBodyState`s inheriting the
    stack's position, attitude and body rates, with velocities split by the
    momentum-conserving impulse, and full separation bookkeeping.

    Mass closure and momentum are exact by construction (SI arithmetic, no
    property round-trip); the returned bookkeeping measures them anyway --
    that table is what the acceptance checks read. ``connected_upper_burn_kg``
    attributes the mated-phase ship burn (E10-02 hot staging) before the
    remainder is charged to stage-1; the split arithmetic is unchanged.
    """
    booster_v, upper_v, bk = pair.resolve(
        stack_mass_at_t0_kg, float(frozen.mass_kg),
        connected_upper_burn_kg=connected_upper_burn_kg)
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
    #: E1-03 (task #252): the upper stage's state-feedback guidance. When
    #: set, BOTH backends fly it -- the point-mass twin through
    #: ``propagate_point_mass(guidance=...)``, the rigid body through
    #: :class:`GuidedUpperStageControl` around the F-04 tracker -- instead
    #: of the open-loop program attitude / neutral gimbal. ``None`` keeps
    #: the F-06 open-loop upper stage.
    upper_guidance: UpperStageGuidanceSpec | None = None
    #: The actuator :attr:`upper_guidance` steers with (default: the
    #: standard servo at the upper engine's own travel limit; the
    #: benchmark's ideal-actuator attribution rung sets it explicitly).
    upper_actuator: TvcActuator | None = None
    #: E10-02 (task #299): hot staging -- ship ignition while mated, no coast
    #: window (see :class:`HotStagingPlan`). ``None`` keeps the F-06 cold
    #: separation with its scheduled delayed ignition.
    hot: HotStagingPlan | None = None

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
        if self.hot is not None:
            if abs(self.hot.separation_t_s - self.separation_t_s) > 1e-9:
                raise ValueError("the hot-staging plan's separation instant must be the "
                                 "scenario's separation instant")
            if self.ignition.ignition_delay_s > 1e-9:
                raise ValueError("with hot staging the child ignition plan's delay must "
                                 "be 0: the ship is already alight at the split, so it "
                                 "ignites (again, in the record) at separation itself")
        if self.dt_s <= 0.0 or int(self.sample_every) < 1:
            raise ValueError("dt_s must be positive and sample_every >= 1")
        if self.upper_guidance is not None and (abs(self.upper_guidance.ignition_t_s
                                                    - self.ignition.ignition_t_s) > 1e-9):
            raise ValueError("the guidance's ignition gate must be the scenario's "
                             "scheduled ignition instant")

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
    # E1-01 (task #250): the vehicle figures come from the SHARED parity
    # source (data/flightlab/parity/spike_stage_parity.yaml via
    # orbitopt.flightlab.parity.native_stage_pair) -- the same document the
    # jsbsim XMLs are cross-checked against -- so this scenario and the
    # JSBSim spike cannot drift apart silently. ``separation_impulse_ns``
    # stays a call-level choice (it is a schedule parameter, not a vehicle
    # parameter).
    pair = replace(native_stage_pair(load_parity_scenario()),
                   separation_impulse_ns=separation_impulse_ns)
    state0 = RigidBodyState.from_launch(lat_geod_deg=28.5, lon_deg=-80.6,
                                        heading_deg=90.0, pitch_deg=90.0,
                                        roll_deg=0.0,
                                        mass_kg=pair.stack.mass_wet_kg)
    program = WaypointAttitudeProgram(times_s=(0.0, 10.0, 110.0),
                                      heading_deg=(90.0, 90.0, 90.0),
                                      pitch_deg=(90.0, 90.0, 20.0),
                                      roll_deg=(0.0, 0.0, 0.0))
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


def guided_upper_stage_pair(pair: StagePair, *, upper_dry_mass_kg: float,
                            upper_propellant_mass_kg: float,
                            gimbal_limit_rad: float) -> StagePair:
    """Mass- and actuator-consistent variant of ``pair`` with a
    TVC-capable, larger-tank upper stage (E1-03, task #252).

    The frozen parity pair flies an upper stage with a fixed engine
    (gimbal travel 0 -- the tracker would command what the plant clamps
    back to zero) and a 6 t tank whose ideal delta-v cannot reach any
    parking orbit. State-feedback guidance needs both an actuator and
    enough propellant to insert, so the guided F-07 mission flies this
    variant instead: stage 1, the split bookkeeping and the booster
    handoff are untouched, the upper engine and inertia are untouched,
    and the E1-01 parity document is untouched -- this is a scenario
    variant built *beside* the parity source, and the resulting pair
    still satisfies the invariants :meth:`StagePair.__post_init__`
    enforces (stage-1 propellant unchanged; the upper stage, structure
    *and* tank, is inert ballast inside the stack's dry floor). The F-07
    default keeps the upper stage's *wet* mass at the parity value (more
    propellant, less structure), so the stack's liftoff state and the
    whole stage-1 ascent up to separation stay bit-identical and only the
    upper stage's own flight changes.
    """
    if gimbal_limit_rad <= 0.0:
        raise ValueError(f"gimbal_limit_rad must be positive, got {gimbal_limit_rad}")
    if upper_dry_mass_kg <= 0.0 or upper_propellant_mass_kg <= 0.0:
        raise ValueError("the upper stage masses must be positive")
    upper = replace(pair.upper,
                    dry_mass_kg=float(upper_dry_mass_kg),
                    propellant_mass_kg=float(upper_propellant_mass_kg),
                    engine=replace(pair.upper.engine,
                                   gimbal_limit_rad=float(gimbal_limit_rad)))
    stack = replace(pair.stack,
                    dry_mass_kg=pair.booster.dry_mass_kg + upper.mass_wet_kg)
    return replace(pair, stack=stack, upper=upper)


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


def _concat_runs(a: SixDofRun, b: SixDofRun) -> SixDofRun:
    """One continuous stack trace out of two consecutive closed-loop legs
    (E10-02: booster leg + composite overlap leg). Shared sample grid, the
    seam sample counted once; ``max_q`` and every derived diagnostic then read
    the whole powered trace like on any single run."""
    if tuple(a.columns) != tuple(b.columns):
        raise ValueError("leg runs carry different columns and cannot be joined")
    samples = {c: np.concatenate([np.asarray(a.samples[c], dtype=float),
                                  np.asarray(b.samples[c], dtype=float)[1:]])
               for c in a.columns}
    return replace(b, samples=samples)


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
                         "impulse_ns": sep["separation_impulse_ns"],
                         **({} if s.hot is None else
                            {"hot_ignition_t_s": s.hot.ship_ignition_t_s,
                             "hot_overlap_s": s.hot.overlap_s})},
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
                       booster_run: SixDofRun, upper_run: SixDofRun,
                       stack_run: SixDofRun | None = None) -> tuple[Check, ...]:
    p_norm = float(np.linalg.norm(np.asarray(sep["momentum_before_kg_m_s"], float)))
    mom_tol = max(MOMENTUM_RESIDUAL_FLOOR_N_S, MOMENTUM_RESIDUAL_TOL_REL * p_norm)
    mass_err = abs(float(sep["mass_residual_kg"]))
    onset = _ignition_onset_err_s(upper_run, s.ignition.ignition_t_s)
    on_time = -1e-9 <= onset <= IGNITION_ONSET_SLACK_S
    divergence = math.sqrt(sum((float(booster_run.samples[c][-1])
                                - float(upper_run.samples[c][-1])) ** 2
                               for c in ("x_m", "y_m", "z_m")))
    checks = [
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
    ]
    if s.hot is not None and stack_run is not None:
        # E10-02: the whole [ship ignition, separation) window is on the stack's
        # powered trace -- the engine is never shut between the two events.
        t = np.asarray(stack_run.samples["t_s"], dtype=float)
        th = np.asarray(stack_run.samples["throttle"], dtype=float)
        win = (t >= s.hot.ship_ignition_t_s) & (t <= s.separation_t_s)
        checks.append(Check("staging-hot-window-powered",
                            bool(win.any()) and float(np.min(th[win])) > 0.0,
                            float(np.min(th[win])) if win.any() else float("nan"),
                            0.0, "", ""))
    return tuple(checks)


def run_two_stage(scenario: TwoStageScenario | None = None, *,
                  epoch_utc: datetime | None = None) -> TwoStageStudy:
    """Fly the scenario (default :func:`f06_two_stage_scenario`): closed-loop
    stack to separation, the analytic split, two independent ``propagate``
    calls to ``t_end_s``, acceptance checks, booster handoff state.

    With a hot-staging plan (E10-02) the stack phase is two closed-loop legs
    -- booster engine to ``ship_ignition_t_s``, composite engine to
    ``separation_t_s`` -- joined into one powered trace; every other step, and
    every conservation identity, is the cold-separation one."""
    s = scenario or f06_two_stage_scenario()
    epoch = epoch_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def _closed_loop(state: RigidBodyState, vehicle: RigidBodyVehicle,
                     t_end_s: float) -> ControlledAscent:
        asc = run_closed_loop_ascent(state, vehicle, s.program, t_end_s,
                                     dt_s=s.dt_s, sample_every=s.sample_every,
                                     actuator=s.real_actuator(), gains=s.gains,
                                     throttle=1.0, dt_c_s=s.dt_c_s, physics=s.physics,
                                     epoch_utc=epoch)
        if asc.run.terminal == "ground":
            raise ValueError(f"the stack hit the ground at t={asc.run.ground_t_s} s before "
                             f"the separation instant {s.separation_t_s} s; pick a larger "
                             f"thrust-to-weight or an earlier separation_t_s")
        return asc

    if s.hot is None:
        ascent = _closed_loop(s.state0, s.pair.stack, s.separation_t_s)
    else:
        leg_a = _closed_loop(s.state0, s.pair.stack, s.hot.ship_ignition_t_s)
        hot_stack = replace(s.pair.stack, engine=s.hot.composite_stack_engine(
            s.pair.stack, s.pair.upper))
        leg_b = _closed_loop(_terminal_state(leg_a.run), hot_stack, s.separation_t_s)
        ascent = replace(leg_b, run=_concat_runs(leg_a.run, leg_b.run))
    frozen = _terminal_state(ascent.run)

    booster_s, upper_s, sep = split_stack_state(
        frozen, s.pair, stack_mass_at_t0_kg=s.state0.mass_kg,
        connected_upper_burn_kg=(s.hot.ship_burn_kg(s.pair.upper)
                                 if s.hot is not None else 0.0))
    booster_v = replace(s.pair.booster,
                        propellant_mass_kg=sep["booster_propellant_remaining_kg"])
    upper_v = replace(s.pair.upper, propellant_mass_kg=sep["upper_propellant_kg"])

    # Two independent bodies: separate propagate() calls, separate state
    # vectors, one mission clock continuing from the separation instant.
    booster_run = propagate(booster_s, booster_v, constant_control(0.0), s.t_end_s,
                            dt_s=s.dt_s, sample_every=s.sample_every,
                            physics=s.physics, epoch_utc=epoch)
    # E1-03 (task #252): with a guidance spec the upper stage closes its own
    # loop (the same law the point-mass twin flies); without one it stays
    # the F-06 open-loop delayed ignition.
    if s.upper_guidance is not None:
        upper_law = GuidedUpperStageControl(s.upper_guidance, upper_v,
                                           actuator=s.upper_actuator,
                                           gains=s.gains, dt_c_s=s.dt_c_s)
    else:
        upper_law = delayed_ignition_control(s.ignition)
    upper_run = propagate(upper_s, upper_v, upper_law,
                          s.t_end_s, dt_s=s.dt_s, sample_every=s.sample_every,
                          physics=s.physics, epoch_utc=epoch)

    handoff = HandoffState(
        from_phase=PHASE_SEPARATION, to_phase=PHASE_BOOSTBACK,
        t_s=frozen.t_s, epoch_utc=epoch.isoformat().replace("+00:00", "Z"),
        frame="ECEF",
        position_m=tuple(float(c) for c in booster_s.position_m),
        velocity_m_s=tuple(float(c) for c in booster_s.velocity_m_s),
        mass_kg=booster_s.mass_kg, dry_mass_kg=booster_v.dry_mass_kg, ignitions_used=1)

    checks = _acceptance_checks(s, sep, booster_run, upper_run, stack_run=ascent.run)
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
    assumptions = tuple(s.assumptions) + (
        (f"Separation at t={s.separation_t_s} s with a "
         f"{sep['separation_impulse_ns']} N.s body-+x impulse "
         f"(+{sep['dv_upper_m_s']:.3f} m/s upper / -{sep['dv_booster_m_s']:.3f} m/s "
         f"booster); upper ignition scheduled at t={s.ignition.ignition_t_s} s "
         f"(delay {s.ignition.ignition_delay_s} s)."),
        ("Booster handed to recovery on the M-01 separation->boostback boundary "
         f"(handoff digest {study.booster_handoff_digest[:12]})."),
    )
    if s.hot is not None:
        assumptions += (
            (f"Hot staging (E10-02): upper ignition at t={s.hot.ship_ignition_t_s} s "
             f"while mated -- {s.hot.overlap_s} s of powered overlap, no coast window; "
             f"the overlap flies a plume-corrected composite engine "
             f"(axial loss {s.hot.plume_axial_loss_fraction}, declared assumption) on "
             f"the stack's gimbal, and the ship's mated burn "
             f"{sep['connected_upper_burn_kg']:.1f} kg is charged to it, not stage-1."),
        )
    return FlightRun(
        run_id=run_id,
        target_sha256=mk("no-flight-target-two-stage-scenario"),
        vehicle_sha256=mk("f06-stack-split-into-booster-and-upper"),
        git_sha="0" * 40,
        created_utc=epoch_utc,
        engines={"stage1": "spike_stage engine (vac 1.88 MN / Isp 320 s)",
                 "stage2": "upper engine (vac 220 kN / Isp 340 s)"},
        backend=BACKEND_NATIVE_RIGIDBODY,
        assumptions=assumptions,
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
            *(() if s.hot is None else (
                Event(s.hot.ship_ignition_t_s, "ignition", "upper",
                      {"throttle": s.hot.throttle, "mated": True,
                       "overlap_s": s.hot.overlap_s}),)),
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
                   "delay_s": s.ignition.ignition_delay_s,
                   **({} if s.hot is None else {"at_spawn": True})}),
        ),
        verification=Verification(
            optimizer_converged=None,
            physics_verified=None,
            backend_self_check=all(c.passed for c in study.checks),
            mission_constraints_passed=None,
            checks=study.checks,
        ),
    )
