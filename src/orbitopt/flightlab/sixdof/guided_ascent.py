"""E6-02 (task #274): a guided JSBSim ascent with staging on the production aero model.

The study
---------
The E1-04 comparison (``production_compare``) flew the production airframe
*open loop*, held vertical: it answered "do the two plants agree about one
body on one burn".  E6-02 asks the production-maturity question that flight
cannot: does the JSBSim path complete a **guided** ascent -- closed-loop
gravity turn, TVC, staging, delayed upper-stage ignition -- and do its
per-phase metrics (attitude error, angle of attack, max-Q, TVC margins) sit
inside declared constraints when powered and coasting are measured
*separately* (the E1-02 discipline)?

One mission, two backends, mirroring the F-06/F-07 pattern:

    JSBSim  the production airframe ``spike_stage_aero`` (ADR flightlab-0007)
            flown by the FL-403 PID gimbal controller on the F-06 gravity-turn
            program stretched to a 140 s pitch-over; commanded MECO (E6-01
            #273: the engine shuts down because
            the flight computer commands it, not because a tank runs dry) at
            the separation instant; the FL-402 split onto the
            ``spike_booster`` / ``spike_upper`` child aircraft with the
            momentum-conserving property injection, then a delayed ignition
            on the upper child.
    native  the same 70 t stack (the parity booster vehicle -- the same 10 t
            dry / 60 t propellant the production XML declares) flown by the
            F-04 tracker on the same program, split by the same declared
            fractions into two rigid-body children built *from the child
            aircraft XMLs themselves* (``parity.jsbsim_declared_params``),
            with the same impulse, the same ignition schedule and the same
            mission clock.

The scenario is deliberately NOT F-06's: F-06's stage pair is a 78 t stack
with a 220 kN upper stage, and no packaged JSBSim child carries that engine --
the production path's children are the FL-402 airframes (both on the stage-1
spike engine, zero aero, declared dry masses 4 t / 6 t).  E6-02 therefore
declares its own mass-consistent scenario (stack dry 10 t = booster 4 t +
upper 6 t, one 60 t propellant book split by fraction at separation) and the
native twin is built to match it.  What that costs -- the upper stage flies a
stage-1-sized engine and burns its 60 % propellant share within seconds -- is
disclosed in the scenario assumptions and in ADR flightlab-0010.

The comparison report
---------------------
``compare_guided_ascent`` classifies every measured disagreement between the
backends into exactly one of four channels -- the E6-01 residual
classification (ADR flightlab-0009) applied to a closed-loop mission:

* ``dynamics``    -- same physics, different machinery: the production FCS
  servo (lag 5.0 / rate 0.35 rad/s in the aircraft XML) vs the native
  law-internal actuator (tau 0.10 s / rate 0.10 rad/s), JSBSim's per-frame
  inertia recomputation and fixed-AERORP aero application vs the native
  constant tensor at the live CG (ADR flightlab-0007),
* ``guidance``    -- the two control laws (F-04 tracker vs FL-403 PID) doing
  the same job with different gain schedules and sampling,
* ``integration`` -- fixed-step RK4 vs JSBSim's internal sequence, measured
  by re-running the stack phase at half the step on each plant (the E6-04
  method, task #276),
* ``model``       -- shared laws implemented twice (two atmosphere models,
  central gravity, the E6-01 engine law): measured by max-Q, mass, pad
  thrust and separation-state agreement.

The numbers land in ``docs/benchmarks/e6_02_guided_ascent.json``
(``PYTHONPATH=src python benchmarks/bench_e6_02_guided_ascent.py``); the
regression thresholds in ``tests/test_flightlab_sixdof_guided_ascent_jsbsim.py``;
the reasoning in ``docs/adr/flightlab-0010-guided-jsbsim-ascent.md``.  Like
every native-plant study here it is illustrative physics, not a flight run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from orbitopt.flightlab.contract import Check
from orbitopt.flightlab.parity import (jsbsim_declared_params, load_parity_scenario,
                                       native_booster)
from orbitopt.flightlab.sixdof.jsbsim_aero import PRODUCTION_AIRCRAFT
from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions, SixDofSim
from orbitopt.flightlab.sixdof.production_compare import (
    ALT_MECO_REL_BAND, MASS_PROP_REL_BAND, MAXQ_REL_BAND, PAD_THRUST_RATIO_REL_BAND,
    PSF_TO_PA, SPEED_MECO_REL_BAND)
from orbitopt.flightlab.sixdof.rigidbody import (RigidBodyEngine, RigidBodyPhysics,
                                                 RigidBodyState, RigidBodyVehicle,
                                                 constant_control, dcm_from_quaternion,
                                                 propagate)
from orbitopt.flightlab.sixdof.staging import (HANDOFF_POSITION_TOL_M,
                                               HANDOFF_VELOCITY_TOL_M_S,
                                               MOMENTUM_RESIDUAL_FLOOR_N_S, StageSplit,
                                               _mat3_row, _quat_to_dcm,
                                               capture_frozen_state, spawn_stage)
from orbitopt.flightlab.sixdof.tracking import (AttitudeProgram, PidAttitudeController,
                                                PidGains, chart_error_deg)
from orbitopt.flightlab.sixdof.two_stage import IgnitionPlan, delayed_ignition_control
from orbitopt.flightlab.sixdof.tvc import (TvcActuator, WaypointAttitudeProgram,
                                           attitude_error_deg_series,
                                           run_closed_loop_ascent)

__all__ = ["E602Scenario", "e602_scenario", "run_guided_ascent_jsbsim",
           "run_guided_ascent_native", "compare_guided_ascent", "run_e602_study",
           "e602_phase_constraint_checks", "E602_PHASE_CONSTRAINTS", "PHASES"]

#: The FL-402 child aircraft the production staging path splits onto.
BOOSTER_MODEL = "spike_booster"
UPPER_MODEL = "spike_upper"

#: The phases the acceptance counts separately (report order).
STACK_POWERED = "stack.powered"
BOOSTER_COAST = "booster.coast"
UPPER_COAST_BEFORE = "upper.coast_before_ignition"
UPPER_POWERED = "upper.powered"
UPPER_COAST_AFTER = "upper.coast_after_burnout"
PHASES = (STACK_POWERED, BOOSTER_COAST, UPPER_COAST_BEFORE, UPPER_POWERED,
          UPPER_COAST_AFTER)

#: Gimbal travel, rad: the clipto of the production and child aircraft FCSs
#: (the parity gate cross-checks it against the stage-1 engine limit).
GIMBAL_LIMIT_RAD = 0.0873

#: Launch pad: the F-06/F-05 lineage pad (Cape), not FL-403's equatorial
#: reference point -- with the gravity-turn program the native tracker's
#: two-gimbal azimuth loop drifts tens of degrees when flown from lat 0
#: (an open-loop chart-frame property of the tracker lineage, observed at
#: the equatorial pad, not a JSBSim-vs-native residual), so the study
#: inherits the pad every closed-loop study of this vehicle has flown.
#: The JSBSim launch IC carries roll -90, which is what puts the body
#: exactly on the program chart at vertical (FL-403's convention); the
#: native ``from_launch`` convention is 90 deg off, so it launches at roll
#: 0 -- the same physical attitude.
PAD_LAT_DEG = 28.5
PAD_LON_DEG = -80.6
PAD_HEADING_DEG = 90.0
JSBSIM_LAUNCH_ROLL_DEG = -90.0

#: Cross-model agreement bands for the comparison report.  The three
#: trajectory bands are E1-04's own constants (imported above, same
#: rationale: what they absorb is two implementations of the shared laws,
#: not a shared truth); the closed-loop additions are declared here, before
#: measurement, from the per-plant constraint bands: the native tracker
#: lineage's own measured worst case for this vehicle on this program
#: (F-05/FL-605: 4.8 deg peak / 2.8 deg RMS powered) plus the production
#: PID's near-zero error bounds agreement at 5 / 3 deg -- law difference,
#: inside each law's own declared band.
ATT_PEAK_AGREEMENT_BAND_DEG = 5.0
ATT_RMS_AGREEMENT_BAND_DEG = 3.0
SEP_FPA_ABS_BAND_DEG = 3.0
TVC_MARGIN_AGREEMENT_BAND = 0.05
#: Property-injection round-trip bands for the production split (FL-402's
#: handoff tolerances are absolute because FL-402 hands off a ~150 m/s hop;
#: E6-02 hands off a ~4 km/s separation state and the round-trip --
#: m/s -> ft/s -> NED components -> back, and quaternion -> Euler-deg ->
#: back near pitch 90 -- scales with the state, not with the maneuver):
#: velocity and momentum carry a 5e-6 RELATIVE band on top of FL-402's
#: absolute floor, and the attitude band is 1e-5 rad, the Euler-reprojection
#: resolution near the pitch-90 lock.  This is a round-trip band, not a
#: conservation law: the native split closes it at 1e-12 and the difference
#: is the property-injection path's own accounting (F-06 assumption 3).
HANDOFF_ROUNDTRIP_REL = 5.0e-6
ATT_ROUNDTRIP_TOL_RAD = 1.0e-5
#: Half-step separation-state bands (E6-04's method on the stack phase):
#: JSBSim at dt/2 must move < 2 % (its dt=0.02 ascent is the coarse one);
#: the native RK4 must be grid-converged (E6-04 measured 2e-4 at these
#: steps), so 0.2 % is the declaration of that fact.
DT_HALF_JSBSIM_ALT_BAND_REL = 0.02
DT_HALF_NATIVE_ALT_BAND_REL = 0.002


# --------------------------------------------------------------------------
# the scenario
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class E602Scenario:
    """One E6-02 study: pad, program, schedule, split bookkeeping, plan.

    The mission shape follows F-06 (the same gravity-turn program, stretched
    to a 140 s pitch-over for the production servo's longer lag, separation at
    95 s, delayed ignition); the masses are the production path's own (the
    production airframe is 10 t dry / 60 t propellant, so the stack *is* the
    70 t stage; its structure splits 4 t / 6 t exactly as the FL-402 child
    aircraft declare)."""

    name: str = "e602-guided-ascent"
    model: str = PRODUCTION_AIRCRAFT
    booster_model: str = BOOSTER_MODEL
    upper_model: str = UPPER_MODEL
    dt_s: float = 0.02
    sample_every: int = 10
    separation_t_s: float = 95.0
    ignition_delay_s: float = 8.0
    t_end_s: float = 260.0
    separation_impulse_ns: float = 10_000.0
    booster_dry_fraction: float = 0.4
    booster_propellant_fraction: float = 0.4
    #: gravity turn: vertical through 10 s, pitch-over to 20 deg by 140 s --
    #: F-06's program stretched from 110 s to 140 s because the production
    #: FCS servo (lag 0.2 s, ADR flightlab-0007) lags the tracker it replaces
    #: and 110 s drove angle of attack against the +/-15 deg table envelope;
    #: a disclosed guidance design choice, not a fitted number
    program_times_s: tuple[float, ...] = (0.0, 10.0, 140.0)
    program_elevation_deg: tuple[float, ...] = (90.0, 90.0, 20.0)
    gains: PidGains = field(default_factory=PidGains)
    assumptions: tuple[str, ...] = (
        ("Production-path study of the packaged spike vehicle -- illustrative "
         "physics, not a flight run (ADR flightlab-0010)."),
        ("Commanded MECO at separation (E6-01 #273): the stack engine shuts "
         "down because the flight computer commands it, with ~3.1 t of "
         "propellant still in the tanks; the split is bookkeeping, not "
         "depletion."),
        ("The JSBSim children are the FL-402 airframes (both carry the stage-1 "
         "spike engine, zero aero); the native twin is built from those child "
         "XMLs' own declared masses/inertias, so the upper stage flies a "
         "stage-1-sized engine and burns its 60 % propellant share in ~3 s. "
         "This differs from F-06's 220 kN upper stage by construction."),
        ("Coast-phase attitude error and AoA are reported but not gated (the "
         "E1-02 discipline: an unactuated drift is geometry, not control); "
         "the structural q*alpha budget and the gimbal-off-its-stops checks "
         "still bind."),
    )

    @property
    def ignition_t_s(self) -> float:
        return self.separation_t_s + self.ignition_delay_s

    def attitude_program(self) -> AttitudeProgram:
        return AttitudeProgram(times=self.program_times_s,
                               azimuth_deg=(PAD_HEADING_DEG,) * len(self.program_times_s),
                               elevation_deg=self.program_elevation_deg)

    def native_program(self) -> WaypointAttitudeProgram:
        return WaypointAttitudeProgram(times_s=self.program_times_s,
                                       heading_deg=(PAD_HEADING_DEG,)
                                       * len(self.program_times_s),
                                       pitch_deg=self.program_elevation_deg,
                                       roll_deg=(0.0,) * len(self.program_times_s))


def e602_scenario(**overrides) -> E602Scenario:
    """The packaged E6-02 study, with call-level overrides (schedule numbers
    only -- the vehicles are what the packaged XMLs declare)."""
    return replace(E602Scenario(), **overrides) if overrides else E602Scenario()


# --------------------------------------------------------------------------
# declared phase constraints
# --------------------------------------------------------------------------

#: (phase, metric) -> (limit, sense, unit, rationale).  Declared before the
#: first measurement of this task, from the vehicle envelope and the
#: precedents named in the rationale -- never read off the results (the
#: E1-02/E1-04 discipline; ADR flightlab-0010).
E602_PHASE_CONSTRAINTS: Mapping[tuple[str, str], tuple[float, str, str, str]] = {
    (STACK_POWERED, "att_err_max_deg"): (6.0, "max", "deg",
        "F-05's closed-loop band for this vehicle on this program (E1-02): "
        "the production loop must hold the program at least as well as the "
        "tracker lineage did."),
    (STACK_POWERED, "att_err_rms_deg"): (3.5, "max", "deg",
        "F-05 powered RMS band, same rationale."),
    (STACK_POWERED, "alpha_max_deg"): (15.0, "max", "deg",
        "the aero document's tables end at +/-15 deg; beyond that the "
        "production model is extrapolating flat tables (ADR flightlab-0007)."),
    (STACK_POWERED, "q_alpha_max_pa_deg"): (5.0e5, "max", "Pa.deg",
        "E1-02's study-level side-load budget for this booster -- a property "
        "of the vehicle, so it binds in every phase."),
    (STACK_POWERED, "max_q_pa"): (1.0e5, "max", "Pa",
        "study-level Max-Q allowance: 1.4x the held-vertical production "
        "ascent this airframe already flies in ADR flightlab-0007 (71.3 kPa "
        "there); a gravity turn sheds vertical speed earlier, so it should "
        "not bind -- a declared allowance, not a measurement."),
    (STACK_POWERED, "deflection_margin_min"): (0.85, "min", "fraction",
        "F-05 demanded 0.90; the production servo is the disclosed "
        "slower-lagging one (lag parameter 5.0, i.e. tau 0.2 s, in the "
        "aircraft XML vs the native tau 0.10 s, ADR flightlab-0007), so the "
        "reserve is relaxed to 0.85 -- disclosed, and the TVC is still never "
        "the actuating limit."),
    (STACK_POWERED, "saturated_s"): (0.0, "max", "s",
        "a gimbal at its hard stop means the loop demanded authority the "
        "production actuator does not have (FL-403's saturation rule)."),
    (BOOSTER_COAST, "q_alpha_max_pa_deg"): (5.0e5, "max", "Pa.deg",
        "structural budget, same vehicle: uncontrolled side-load must not "
        "exceed it either."),
    (BOOSTER_COAST, "saturated_s"): (0.0, "max", "s",
        "the booster's gimbal must come back to rest and stay off its stops."),
    (UPPER_COAST_BEFORE, "q_alpha_max_pa_deg"): (5.0e5, "max", "Pa.deg",
        "structural budget; the pre-ignition coast is the first phase in "
        "which the upper airframe sees live air on its own."),
    (UPPER_POWERED, "att_err_max_deg"): (12.0, "max", "deg",
        "E6-05 (#292): the child is unactuated through its whole burn "
        "(assumption 3), so its powered attitude error is the drift the "
        "separation left it with, accrued over coast + burn at the rate it "
        "was separated with: the program this scenario ends with slews at "
        "70 deg over its last 130 s (0.54 deg/s at MECO), the ignition delay "
        "is 8 s and the disclosed upper burn 3 s, so the accrual bound is "
        "0.54 x 11 s = 5.9 deg on top of the stack's own closed-loop 6 deg "
        "band (F-05) the child inherits at separation; declared 12 deg."),
    (UPPER_POWERED, "att_err_rms_deg"): (7.0, "max", "deg",
        "E6-05 (#292): same rationale: a body drifting at a constant rate "
        "accumulates a "
        "linear ramp, whose RMS is its peak / sqrt(3); declared at the ramp "
        "ratio rounded up (12 / 1.7 ~= 7)."),
    (UPPER_POWERED, "alpha_max_deg"): (25.0, "max", "deg",
        "the child airframes carry no aerodynamics at all (FL-402), so no "
        "table envelope applies; this upper bound keeps the free-drift angle "
        "geometry (separation impulse + coast tumble at ~5 Pa of q) inside "
        "the attitude at which the FL-402 handoff frame re-projection stays "
        "meaningful -- the structural side-load gate q*alpha is the real one "
        "here, and it binds two orders of magnitude below."),
    (UPPER_POWERED, "q_alpha_max_pa_deg"): (5.0e5, "max", "Pa.deg",
        "structural budget."),
    (UPPER_POWERED, "deflection_margin_min"): (0.85, "min", "fraction",
        "the upper stage's gimbal is idle at neutral until F-07-class "
        "guidance uses it; the full travel must therefore be available."),
    (UPPER_POWERED, "saturated_s"): (0.0, "max", "s", "as the booster coast."),
    (UPPER_COAST_AFTER, "q_alpha_max_pa_deg"): (5.0e5, "max", "Pa.deg",
        "structural budget."),
}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


# --------------------------------------------------------------------------
# phase metrics (one column set, both plants)
# --------------------------------------------------------------------------

def _angle_between_quats(q1, q2) -> float:
    d = abs(sum(float(a) * float(b) for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def _phase_metrics(cols: Mapping[str, np.ndarray], t0: float, t1: float,
                   sample_dt_s: float, bounds: str = "[]") -> dict[str, float]:
    """Window metrics over ``[t0, t1]`` (``bounds`` selects which endpoints
    belong to this window -- an event frame, ignition or burn-out, belongs to
    the phase it starts, not the one it ends) of a recorded column set (see
    ``_jsbsim_columns`` / ``_native_columns`` for the shared column
    vocabulary): the acceptance's four families -- attitude error, AoA /
    side-load, max-Q, TVC margin -- measured inside the window only (the
    E1-02 discipline)."""
    t = np.asarray(cols["t_s"], dtype=float)
    keep = ((t >= t0 - 1e-9) if bounds[0] == "[" else (t > t0 + 1e-9)) & \
           ((t <= t1 + 1e-9) if bounds[1] == "]" else (t < t1 - 1e-9))
    if not bool(keep.any()):
        raise ValueError(f"phase window [{t0}, {t1}] keeps no samples")
    tw = t[keep]
    err = np.abs(np.asarray(cols["att_err_deg"], dtype=float)[keep])
    a = np.abs(np.asarray(cols["alpha_deg"], dtype=float)[keep])
    q = np.asarray(cols["q_bar_pa"], dtype=float)[keep]
    cmd_p = np.asarray(cols["gimbal_pitch_cmd_rad"], dtype=float)[keep]
    cmd_y = np.asarray(cols["gimbal_yaw_cmd_rad"], dtype=float)[keep]
    act_p = np.asarray(cols["gimbal_pitch_actual_rad"], dtype=float)[keep]
    act_y = np.asarray(cols["gimbal_yaw_actual_rad"], dtype=float)[keep]
    eng = np.asarray(cols["engine_on"], dtype=float)[keep] > 0.5
    qa = q * a
    i_q = int(np.argmax(q))
    peak_cmd = float(max(np.max(np.abs(cmd_p)), np.max(np.abs(cmd_y))))
    sat = (np.abs(act_p) >= 0.999 * GIMBAL_LIMIT_RAD) | \
          (np.abs(act_y) >= 0.999 * GIMBAL_LIMIT_RAD)
    lag = float(max(np.max(np.abs(cmd_p - act_p)), np.max(np.abs(cmd_y - act_y))))
    return {
        "t0_s": float(tw[0]), "t1_s": float(tw[-1]), "n_samples": float(keep.sum()),
        "engine_s": float(eng.sum()) * sample_dt_s,
        "att_err_max_deg": float(np.max(err)),
        "att_err_rms_deg": float(np.sqrt(np.mean(err ** 2))),
        "alpha_max_deg": float(np.max(a)),
        "q_alpha_max_pa_deg": float(np.max(qa)),
        "max_q_pa": float(q[i_q]), "max_q_t_s": float(tw[i_q]),
        "deflection_margin_min": float(1.0 - peak_cmd / GIMBAL_LIMIT_RAD),
        "command_peak_deg": math.degrees(peak_cmd),
        "saturated_s": float(sat.sum()) * sample_dt_s,
        "deflection_lag_max_deg": math.degrees(lag),
    }


def e602_phase_constraint_checks(table: Mapping[str, Mapping[str, float]]
                                 ) -> list[Check]:
    """Evaluate every :data:`E602_PHASE_CONSTRAINTS` row against a phase
    metrics table (``{phase: metrics}``); a missing phase or metric is a
    failed check, not a skipped one."""
    out: list[Check] = []
    for (phase, metric), (limit, sense, unit, _why) in E602_PHASE_CONSTRAINTS.items():
        value = table.get(phase, {}).get(metric)
        if value is None:
            out.append(Check(id=f"e602-{phase}-{metric}", passed=False,
                             reason_code="E602_PHASE_METRIC_MISSING"))
            continue
        ok = value >= limit if sense == "min" else value <= limit
        out.append(Check(id=f"e602-{phase}-{metric}", passed=bool(ok), value=float(value),
                         limit=float(limit), unit=unit,
                         reason_code="" if ok else "E602_PHASE_OUT_OF_BAND"))
    return out


# --------------------------------------------------------------------------
# the JSBSim path
# --------------------------------------------------------------------------

def _jsbsim_row(s: SixDofSim, att_err_deg: float) -> dict[str, float]:
    """One recorded frame of a live JSBSim instance in the shared column
    vocabulary (alpha and q-bar are read from properties; the driver's sample
    dict carries neither, per E1-04)."""
    smp = s.sample()
    sp = math.sqrt(smp["vx_m_s"] ** 2 + smp["vy_m_s"] ** 2 + smp["vz_m_s"] ** 2)
    fpa = math.degrees(math.asin(max(-1.0, min(1.0, -smp["v_down_m_s"] / max(sp, 1e-9)))))
    return {"t_s": smp["t_s"], "altitude_m": smp["altitude_m"], "speed_m_s": sp,
            "fpa_deg": fpa, "alpha_deg": float(s["aero/alpha-deg"]),
            "q_bar_pa": float(s["aero/qbar-psf"]) * PSF_TO_PA,
            "mach": float(s["velocities/mach"]), "mass_kg": smp["mass_kg"],
            "thrust_n": smp["thrust_n"],
            "att_err_deg": att_err_deg,
            "gimbal_pitch_cmd_rad": smp["gimbal_pitch_cmd_rad"],
            "gimbal_yaw_cmd_rad": smp["gimbal_yaw_cmd_rad"],
            "gimbal_pitch_actual_rad": smp["gimbal_pitch_actual_rad"],
            "gimbal_yaw_actual_rad": smp["gimbal_yaw_actual_rad"],
            # engine-on from the physical thrust, not the throttle command:
            # JSBSim floors the throttle position property at the engine's
            # minimum even after a commanded or depleted shutdown
            "engine_on": 1.0 if smp["thrust_n"] > 1e3 else 0.0}


def _columns(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def _child_columns(rows: list[dict[str, float]], t0_s: float) -> dict[str, np.ndarray]:
    """Spawned children run on their own JSBSim clocks; the study reports the
    mission clock, so the child trace is re-based onto the separation instant
    (the same re-basing ``SixDofSim.to_body_track``'s ``t_offset_s`` does)."""
    cols = _columns(rows)
    cols["t_s"] = cols["t_s"] + t0_s
    return cols


def _fly_stack_jsbsim(s: E602Scenario, dt_s: float):
    """Closed-loop production stack from the pad to the commanded-MECO
    separation instant; returns (sim, recorded rows)."""
    program = s.attitude_program()
    sim = SixDofSim(dt_s, model=s.model)
    sim.initialize(InitialConditions(lat_geod_deg=PAD_LAT_DEG, lon_deg=PAD_LON_DEG,
                                     alt_m=0.0, pitch_deg=90.0,
                                     heading_deg=PAD_HEADING_DEG,
                                     roll_deg=JSBSIM_LAUNCH_ROLL_DEG))
    sim.ignite(1.0)
    ctrl = PidAttitudeController(gains=s.gains)
    rows: list[dict[str, float]] = []
    frame = 0
    cut: list[bool] = []

    def control(t: float, st: SixDofSim) -> None:
        nonlocal frame
        cmd_p, cmd_y, _, _ = ctrl.command(st.sample(), program, st.dt_s)
        st.set_gimbal(cmd_p, cmd_y)
        if t + st.dt_s >= s.separation_t_s - 1e-6 and not cut:
            st.set_throttle(0.0)          # commanded MECO at separation (E6-01)
            cut.append(True)
        if frame % int(s.sample_every) == 0:
            rows.append(_jsbsim_row(st, chart_error_deg(st.sample(), program)))
        frame += 1

    sim.run_for(s.separation_t_s, control)
    rows.append(_jsbsim_row(sim, chart_error_deg(sim.sample(), program)))
    if not cut:
        raise RuntimeError("the stack never received its commanded MECO")
    return sim, rows


@dataclass
class JsbsimAscentStudy:
    scenario: E602Scenario
    traces: Mapping[str, Mapping[str, np.ndarray]]
    phases: Mapping[str, Mapping[str, float]]
    frozen: object
    bookkeeping: Mapping[str, float]
    ignition_onset_err_s: float
    upper_burnout_t_s: float
    checks: tuple[Check, ...] = field(default=())

    def phase_table(self) -> dict[str, dict[str, float]]:
        return {p: dict(self.phases[p]) for p in PHASES if p in self.phases}


def run_guided_ascent_jsbsim(scenario: E602Scenario | None = None,
                            *, dt_s: float | None = None) -> JsbsimAscentStudy:
    """Fly the E6-02 mission on the JSBSim production airframe: closed-loop
    gravity turn to the commanded-MECO separation instant, the FL-402 split,
    the two children interleaved to ``t_end_s`` (the upper one ignited on
    schedule and shut down when its tanks run dry)."""
    s = scenario or e602_scenario()
    dt = float(dt_s or s.dt_s)
    sim, stack_rows = _fly_stack_jsbsim(s, dt)
    frozen = capture_frozen_state(sim)
    sample_dt = dt * int(s.sample_every)

    alloc = StageSplit(s.booster_dry_fraction, s.booster_propellant_fraction)\
        .allocate(frozen)
    xb = _mat3_row(_quat_to_dcm(frozen.quaternion), 0)
    j = s.separation_impulse_ns
    masses = {side: sum(alloc[side][k] for k in ("dry_kg", "ox_kg", "fuel_kg"))
              for side in ("booster", "upper")}
    booster, hb = spawn_stage(s.booster_model, frozen, alloc["booster"], dt_s=dt,
                              impulse_dv_ecef_m_s=tuple(-j * xb[i] / masses["booster"]
                                                        for i in range(3)))
    upper, hu = spawn_stage(s.upper_model, frozen, alloc["upper"], dt_s=dt,
                            impulse_dv_ecef_m_s=tuple(j * xb[i] / masses["upper"]
                                                      for i in range(3)))
    sep_att = frozen.quaternion

    bookkeeping = {
        "mass_before_kg": frozen.mass_kg,
        "mass_after_kg": masses["booster"] + masses["upper"],
        "mass_residual_kg": (masses["booster"] + masses["upper"]) - frozen.mass_kg,
        "momentum_before_kg_m_s": tuple(frozen.mass_kg * v for v in frozen.velocity_m_s),
        "momentum_residual_kg_m_s": math.sqrt(sum(
            (masses["booster"] * booster.samples[0][k]
             + masses["upper"] * upper.samples[0][k]
             - frozen.mass_kg * v) ** 2
            for k, v in zip(("vx_m_s", "vy_m_s", "vz_m_s"), frozen.velocity_m_s))),
        "dv_booster_m_s": j / masses["booster"],
        "dv_upper_m_s": j / masses["upper"],
        "sep_speed_m_s": math.sqrt(sum(v * v for v in frozen.velocity_m_s)),
        "position_continuity_err_m": max(hb["position_err_m"], hu["position_err_m"]),
        "attitude_continuity_err_rad": max(hb["attitude_err_rad"], hu["attitude_err_rad"]),
        "velocity_continuity_err_m_s": max(hb["velocity_err_m_s"], hu["velocity_err_m_s"]),
    }

    booster_rows: list[dict[str, float]] = []
    upper_rows: list[dict[str, float]] = []
    ign: list[float] = []
    bur: list[float] = []
    frame = 0

    def children(t: float) -> None:
        nonlocal frame
        if frame % int(s.sample_every) == 0:
            for child, rows in ((booster, booster_rows), (upper, upper_rows)):
                smp = child.sample()
                err = math.degrees(_angle_between_quats(
                    (smp["q_w"], smp["q_x"], smp["q_y"], smp["q_z"]), sep_att))
                rows.append(_jsbsim_row(child, err))
        frame += 1
        if not ign and t + dt >= s.ignition_t_s - 1e-6:
            upper.ignite(1.0)
            ign.append(t + dt)
        elif ign and not bur and upper.propellant_mass_kg() <= 0.0:
            upper.set_throttle(0.0)       # the upper child's own depletion
            bur.append(t)

    n_child = round((s.t_end_s - s.separation_t_s) / dt)
    for k in range(n_child):
        children(s.separation_t_s + k * dt)
        booster.step()
        upper.step()
    # terminal frames at t_end_s
    for child, rows in ((booster, booster_rows), (upper, upper_rows)):
        smp = child.sample()
        err = math.degrees(_angle_between_quats(
            (smp["q_w"], smp["q_x"], smp["q_y"], smp["q_z"]), sep_att))
        rows.append(_jsbsim_row(child, err))

    if len(ign) != 1 or len(bur) != 1:
        raise RuntimeError(f"the upper child did not complete its ignition/burn-out "
                           f"sequence: ignition {ign}, burn-out {bur}")
    t_burnout, onset_err = bur[0], abs(ign[0] - s.ignition_t_s)

    stack_cols = _columns(stack_rows)
    booster_cols = _child_columns(booster_rows, s.separation_t_s)
    upper_cols = _child_columns(upper_rows, s.separation_t_s)
    phases = {
        STACK_POWERED: _phase_metrics(stack_cols, 0.0, s.separation_t_s, sample_dt),
        BOOSTER_COAST: _phase_metrics(booster_cols, s.separation_t_s,
                                      s.t_end_s, sample_dt),
        UPPER_COAST_BEFORE: _phase_metrics(upper_cols, s.separation_t_s,
                                           s.ignition_t_s, sample_dt, bounds="[)"),
        UPPER_POWERED: _phase_metrics(upper_cols, s.ignition_t_s,
                                      t_burnout, sample_dt),
        UPPER_COAST_AFTER: _phase_metrics(upper_cols, t_burnout,
                                          s.t_end_s, sample_dt, bounds="(]"),
    }
    return JsbsimAscentStudy(
        scenario=s,
        traces={"stack": stack_cols, "booster": booster_cols, "upper": upper_cols},
        phases=phases, frozen=frozen, bookkeeping=bookkeeping,
        ignition_onset_err_s=onset_err, upper_burnout_t_s=t_burnout,
        checks=tuple(_staging_checks(bookkeeping, onset_err, dt)))


def _staging_checks(bk: Mapping, onset_err_s: float, onset_period_s: float,
                    prefix: str = "e602") -> list[Check]:
    """The FL-402 handoff bookkeeping as checks: position continuity keeps
    FL-402's absolute tolerance (length scales don't round-trip), velocity /
    momentum / attitude use the scale-aware round-trip bands declared with
    the scenario constants, plus the upper stage's ignition schedule (the
    onset tolerance is 1.5x the plant's ignition-onset *observation* period:
    the JSBSim hook sees it on the integration step, the native study only
    on its sample grid)."""
    p_norm = math.sqrt(sum(v * v for v in bk["momentum_before_kg_m_s"]))
    v_norm = float(bk["sep_speed_m_s"])
    specs = (
        ("mass-closure", abs(bk["mass_residual_kg"]),
         1e-6 * max(1.0, bk["mass_before_kg"]), "kg", "E602_MASS_NOT_CONSERVED"),
        ("momentum-residual", bk["momentum_residual_kg_m_s"],
         max(MOMENTUM_RESIDUAL_FLOOR_N_S, HANDOFF_ROUNDTRIP_REL * p_norm), "N.s",
         "E602_MOMENTUM_NOT_CONSERVED"),
        ("position-continuity", bk["position_continuity_err_m"],
         HANDOFF_POSITION_TOL_M, "m", "E602_HANDOFF_POSITION"),
        ("attitude-continuity", bk["attitude_continuity_err_rad"],
         ATT_ROUNDTRIP_TOL_RAD, "rad", "E602_HANDOFF_ATTITUDE"),
        ("velocity-continuity", bk["velocity_continuity_err_m_s"],
         max(HANDOFF_VELOCITY_TOL_M_S, HANDOFF_ROUNDTRIP_REL * v_norm), "m/s",
         "E602_HANDOFF_VELOCITY"),
        ("ignition-onset-schedule", onset_err_s, 1.5 * onset_period_s, "s",
         "E602_IGNITION_NOT_ON_SCHEDULE"),
    )
    out = []
    for cid, value, limit, unit, fail in specs:
        ok = value <= limit
        out.append(Check(id=f"{prefix}-{cid}", passed=bool(ok), value=float(value), limit=float(limit),
                         unit=unit, reason_code="" if ok else fail))
    return out


# --------------------------------------------------------------------------
# the native twin
# --------------------------------------------------------------------------

@dataclass
class NativeAscentStudy:
    scenario: E602Scenario
    traces: Mapping[str, Mapping[str, np.ndarray]]
    phases: Mapping[str, Mapping[str, float]]
    bookkeeping: Mapping[str, float]
    ignition_onset_err_s: float
    upper_burnout_t_s: float

    def phase_table(self) -> dict[str, dict[str, float]]:
        return {p: dict(self.phases[p]) for p in PHASES if p in self.phases}


def _native_columns(run, att_err_deg: np.ndarray) -> dict[str, np.ndarray]:
    """A native ``SixDofRun`` in the shared column vocabulary (the gimbal
    *applied* deflection is the command column: the native law records the
    actuator state as the applied deflection, so the lag metric is 0 by
    construction there and only meaningful on the JSBSim side)."""
    cols = {c: np.asarray(run[c], dtype=float)
            for c in ("t_s", "altitude_m", "alpha_deg", "q_bar_pa", "mach", "mass_kg",
                      "thrust_n", "gimbal_pitch_cmd_rad", "gimbal_yaw_cmd_rad",
                      "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")}
    pos = np.stack([cols["x_m"], cols["y_m"], cols["z_m"]])
    vel = np.stack([cols["vx_m_s"], cols["vy_m_s"], cols["vz_m_s"]])
    cols["speed_m_s"] = np.linalg.norm(vel, axis=0)
    vr = np.sum(pos * vel, axis=0) / np.maximum(np.linalg.norm(pos, axis=0), 1e-9)
    cols["fpa_deg"] = np.degrees(np.arcsin(np.clip(vr / np.maximum(cols["speed_m_s"], 1e-9),
                                                  -1.0, 1.0)))
    cols["att_err_deg"] = att_err_deg
    cols["gimbal_pitch_actual_rad"] = cols["gimbal_pitch_cmd_rad"]
    cols["gimbal_yaw_actual_rad"] = cols["gimbal_yaw_cmd_rad"]
    cols["engine_on"] = (cols["thrust_n"] > 1e3).astype(float)
    return cols


def _e602_child_vehicles(scenario: E602Scenario, propellant_shares_kg: Mapping[str, float]
                         ) -> tuple[RigidBodyVehicle, RigidBodyVehicle]:
    """The native children, built from the FL-402 child aircraft XMLs' own
    declared masses / inertias / arm, so both plants agree on what a child
    is (the JSBSim side flies those XMLs; the native side reads them)."""
    stage1 = load_parity_scenario().document["stage1_engine"]
    out = {}
    for model, key in ((scenario.booster_model, "booster"),
                       (scenario.upper_model, "upper")):
        d = jsbsim_declared_params(model)
        eng = RigidBodyEngine(
            thrust_vacuum_n=float(stage1["thrust_vacuum_n"]),
            isp_vacuum_s=float(stage1["isp_vacuum_s"]),
            isp_sea_level_s=float(stage1["isp_sea_level_s"]),
            nozzle_exit_area_m2=float(stage1["nozzle_exit_area_m2"]),
            throttle_min=float(stage1["throttle_min"]),
            throttle_max=float(stage1["throttle_max"]),
            mount_x_cg_m=float(d["thruster_x_m"]) - float(d["cg_x_m"]),
            gimbal_limit_rad=float(d["gimbal_limit_rad"]))
        out[key] = RigidBodyVehicle(
            name=f"e602-{key}", dry_mass_kg=float(d["empty_mass_kg"]),
            propellant_mass_kg=float(propellant_shares_kg[key]),
            inertia_kg_m2=d["inertia_kg_m2"], engine=eng, aero=None)
    return out["booster"], out["upper"]


def _stack_only_native(s: E602Scenario, dt_s: float, physics: RigidBodyPhysics):
    """The native stack phase alone (the dt-halving integration probe)."""
    stack = native_booster(name="e602-stack")
    state0 = RigidBodyState.from_launch(lat_geod_deg=PAD_LAT_DEG, lon_deg=PAD_LON_DEG,
                                        alt_m=0.0, heading_deg=PAD_HEADING_DEG,
                                        pitch_deg=90.0, roll_deg=0.0,
                                        mass_kg=stack.mass_wet_kg)
    return run_closed_loop_ascent(state0, stack, s.native_program(), s.separation_t_s,
                                  dt_s=dt_s, sample_every=s.sample_every,
                                  actuator=TvcActuator(
                                      limit_rad=stack.engine.gimbal_limit_rad),
                                  throttle=1.0, dt_c_s=0.05, physics=physics)


def run_guided_ascent_native(scenario: E602Scenario | None = None, *,
                             dt_s: float | None = None,
                             physics: RigidBodyPhysics | None = None
                             ) -> NativeAscentStudy:
    """The same mission on the native rigid-body plant: the parity stack
    vehicle closed-loop to separation, the declared fraction split (exact SI
    arithmetic, momentum-conserving impulse along the thrust line), two
    ``propagate`` children to ``t_end_s``.  ``j2=False`` like E1-04: JSBSim's
    gravity is central, and folding a gravity-model difference into the
    closed-loop comparison is the report's ``model`` channel's job."""
    s = scenario or e602_scenario()
    dt = float(dt_s or s.dt_s)
    phys = RigidBodyPhysics(j2=False) if physics is None else physics
    ascent = _stack_only_native(s, dt, phys)
    if ascent.run.terminal == "ground":
        raise ValueError("the native stack hit the ground before separation")
    run = ascent.run
    sample_dt = dt * int(s.sample_every)

    prop_rem = max(float(run["mass_kg"][-1]) - native_booster().dry_mass_kg, 0.0)
    dp_b = s.booster_propellant_fraction * prop_rem
    booster_v, upper_v = _e602_child_vehicles(
        s, {"booster": dp_b, "upper": prop_rem - dp_b})

    frozen_q = (float(run["q_w"][-1]), float(run["q_x"][-1]), float(run["q_y"][-1]),
                float(run["q_z"][-1]))
    frozen_pos = (float(run["x_m"][-1]), float(run["y_m"][-1]), float(run["z_m"][-1]))
    v0 = np.array([float(run["vx_m_s"][-1]), float(run["vy_m_s"][-1]),
                   float(run["vz_m_s"][-1])])
    x_ecef = np.asarray(dcm_from_quaternion(frozen_q)[0], dtype=float)
    m_b, m_u = booster_v.mass_wet_kg, upper_v.mass_wet_kg
    j = s.separation_impulse_ns
    rates = (float(run["p_rad_s"][-1]), float(run["q_rad_s"][-1]), float(run["r_rad_s"][-1]))
    booster_s = RigidBodyState(t_s=float(run["t_s"][-1]), position_m=frozen_pos,
                               velocity_m_s=tuple(v0[i] - j * x_ecef[i] / m_b
                                                  for i in range(3)),
                               quaternion=frozen_q, omega_rad_s=rates, mass_kg=m_b)
    upper_s = RigidBodyState(t_s=float(run["t_s"][-1]), position_m=frozen_pos,
                             velocity_m_s=tuple(v0[i] + j * x_ecef[i] / m_u
                                                for i in range(3)),
                             quaternion=frozen_q, omega_rad_s=rates, mass_kg=m_u)

    plan = IgnitionPlan(separation_t_s=s.separation_t_s, ignition_delay_s=s.ignition_delay_s)
    booster_run = propagate(booster_s, booster_v, constant_control(0.0), s.t_end_s,
                            dt_s=dt, sample_every=s.sample_every, physics=phys)
    upper_run = propagate(upper_s, upper_v, delayed_ignition_control(plan), s.t_end_s,
                          dt_s=dt, sample_every=s.sample_every, physics=phys)

    stack_cols = _native_columns(run, attitude_error_deg_series(run, s.native_program()))
    booster_cols = _native_children_columns(booster_run, frozen_q)
    upper_cols = _native_children_columns(upper_run, frozen_q)

    on = upper_cols["engine_on"] > 0.5
    if not bool(on.any()):
        raise RuntimeError("the native upper child never ignited")
    t_ig = float(upper_cols["t_s"][on][0])
    t_burnout = float(upper_cols["t_s"][on][-1])

    ph = ascent.phases["powered"]
    phases = {
        STACK_POWERED: {
            "t0_s": ph.t0_s, "t1_s": ph.t1_s, "n_samples": ph.aero["n_samples"],
            "engine_s": ph.engine_s,
            "att_err_max_deg": ph.att_err_max_deg, "att_err_rms_deg": ph.att_err_rms_deg,
            "alpha_max_deg": ph.aero["alpha_max_deg"],
            "q_alpha_max_pa_deg": ph.aero["q_alpha_max_pa_deg"],
            "max_q_pa": ph.aero["max_q_pa"], "max_q_t_s": ph.aero["max_q_t_s"],
            "deflection_margin_min": min(ph.pitch_deflection_margin,
                                         ph.yaw_deflection_margin),
            "command_peak_deg": max(ph.pitch_command_peak_deg, ph.yaw_command_peak_deg),
            "saturated_s": ph.pitch_saturated_s + ph.yaw_saturated_s,
            # the native actuator state *is* the applied deflection; its lag is
            # inside the law, not observable as a cmd/actual pair
            "deflection_lag_max_deg": 0.0,
        },
        BOOSTER_COAST: _phase_metrics(booster_cols, s.separation_t_s, s.t_end_s, sample_dt),
        UPPER_COAST_BEFORE: _phase_metrics(upper_cols, s.separation_t_s, t_ig, sample_dt,
                                           bounds="[)"),
        UPPER_POWERED: _phase_metrics(upper_cols, t_ig, t_burnout, sample_dt),
        UPPER_COAST_AFTER: _phase_metrics(upper_cols, t_burnout, s.t_end_s, sample_dt,
                                          bounds="(]"),
    }
    p0 = v0 * float(run["mass_kg"][-1])
    p1 = (np.array(booster_s.velocity_m_s) * m_b + np.array(upper_s.velocity_m_s) * m_u)
    bookkeeping = {
        "mass_before_kg": float(run["mass_kg"][-1]), "mass_after_kg": m_b + m_u,
        "mass_residual_kg": (m_b + m_u) - float(run["mass_kg"][-1]),
        "momentum_before_kg_m_s": tuple(float(v) for v in p0),
        "momentum_residual_kg_m_s": float(np.linalg.norm(p1 - p0)),
        "dv_booster_m_s": j / m_b, "dv_upper_m_s": j / m_u,
        "sep_speed_m_s": float(np.linalg.norm(v0)),
        "position_continuity_err_m": 0.0, "attitude_continuity_err_rad": 0.0,
        "velocity_continuity_err_m_s": 0.0,
    }
    return NativeAscentStudy(
        scenario=s,
        traces={"stack": stack_cols, "booster": booster_cols, "upper": upper_cols},
        phases=phases, bookkeeping=bookkeeping,
        ignition_onset_err_s=abs(t_ig - s.ignition_t_s), upper_burnout_t_s=t_burnout)


def _native_children_columns(run, sep_att) -> dict[str, np.ndarray]:
    """Child runs in the shared columns, with the attitude error measured as
    the drift off the separation attitude (an unactuated body's only
    meaningful 'error' -- reported, not gated, per E1-02)."""
    n = int(np.asarray(run["t_s"]).size)
    err = np.array([_angle_between_quats(
        (float(run["q_w"][i]), float(run["q_x"][i]), float(run["q_y"][i]),
         float(run["q_z"][i])), sep_att) for i in range(n)])
    return _native_columns(run, np.degrees(err))


# --------------------------------------------------------------------------
# the comparison report, in four channels
# --------------------------------------------------------------------------

def compare_guided_ascent(jsb: JsbsimAscentStudy, nat: NativeAscentStudy, *,
                          dt_halving: bool = True) -> dict:
    """Classify the JSBSim-vs-native disagreement of the same mission into
    ``dynamics`` / ``guidance`` / ``integration`` / ``model`` rows (module
    docstring), each with its measurement, its explanation and -- where a
    band exists -- the verdict check."""
    checks: list[Check] = []

    def band(cid: str, value: float, limit: float, unit: str, why: str) -> None:
        ok = bool(value <= limit)
        checks.append(Check(id=cid, passed=ok, value=float(value), limit=float(limit),
                            unit=unit, reason_code="" if ok else "E602_AGREEMENT_OUT_OF_BAND"))
        return None

    j_st, n_st = jsb.phases[STACK_POWERED], nat.phases[STACK_POWERED]
    j_b, n_b = jsb.phases[BOOSTER_COAST], nat.phases[BOOSTER_COAST]

    # -- dynamics: same physics, different machinery -----------------------
    dynamics = [
        {"id": "stack.deflection_lag_jsbsim_deg", "value": j_st["deflection_lag_max_deg"],
         "explanation": "max |cmd - actual| on the JSBSim gimbals during the "
                        "powered phase: the production FCS servo (lag 5.0, "
                        "rate 0.35 rad/s) against the native law-internal "
                        "actuator (tau 0.10 s, rate 0.10 rad/s).  A delta of "
                        "the actuator implementation (ADR flightlab-0007), "
                        "not of the commanded loop."},
        {"id": "stack.command_peak_rel", "value": _rel(j_st["command_peak_deg"],
                                                       n_st["command_peak_deg"]),
         "explanation": "peak TVC deflection each plant had to apply for the "
                        "same program on the same airframe: different "
                        "actuators, and JSBSim recomputes mass/CG/inertia per "
                        "frame while the native tensor is constant."},
        {"id": "booster.coast_att_drift_diff_deg",
         "value": abs(j_b["att_err_max_deg"] - n_b["att_err_max_deg"]),
         "explanation": "the unactuated coast attitude drift differs on any "
                        "two plants whose gravity, frame integration and "
                        "CG/inertia history differ -- reported, not gated "
                        "(E1-02 discipline)."},
    ]
    band("e602-tvc-margin-agreement",
         abs(j_st["deflection_margin_min"] - n_st["deflection_margin_min"]),
         TVC_MARGIN_AGREEMENT_BAND, "fraction",
         "the two plants must report the same TVC reserve band")

    # -- guidance: two laws, one job ---------------------------------------
    guidance = [
        {"id": "stack.att_err_max_pair_deg",
         "value": [j_st["att_err_max_deg"], n_st["att_err_max_deg"]],
         "explanation": "the guidance channel *is* this pair: the FL-403 PID "
                        "samples every plant frame with a live instrumented "
                        "gain schedule; the F-04 tracker ticks at 0.05 s with "
                        "a rated-thrust schedule.  Both are separately inside "
                        "the declared 6 deg band."},
        {"id": "stack.att_err_rms_pair_deg",
         "value": [j_st["att_err_rms_deg"], n_st["att_err_rms_deg"]],
         "explanation": "same rationale as the peak."},
    ]
    band("e602-att-peak-agreement", abs(j_st["att_err_max_deg"] - n_st["att_err_max_deg"]),
         ATT_PEAK_AGREEMENT_BAND_DEG, "deg", "law difference inside the band")
    band("e602-att-rms-agreement", abs(j_st["att_err_rms_deg"] - n_st["att_err_rms_deg"]),
         ATT_RMS_AGREEMENT_BAND_DEG, "deg", "law difference inside the band")

    # -- integration: two fixed-step worlds ---------------------------------
    integration: list[dict] = []
    if dt_halving:
        _, half_rows = _fly_stack_jsbsim(replace(jsb.scenario), jsb.scenario.dt_s / 2.0)
        half = _columns(half_rows)
        j_shift = _rel(float(half["altitude_m"][-1]),
                       float(jsb.traces["stack"]["altitude_m"][-1]))
        j_speed = _rel(float(half["speed_m_s"][-1]),
                       float(jsb.traces["stack"]["speed_m_s"][-1]))
        n_ascent = _stack_only_native(nat.scenario, nat.scenario.dt_s / 2.0,
                                      RigidBodyPhysics(j2=False))
        n_shift = _rel(float(n_ascent.run["altitude_m"][-1]),
                       float(nat.traces["stack"]["altitude_m"][-1]))
        n_speed = _rel(float(np.linalg.norm(np.stack(
            [n_ascent.run["vx_m_s"][-1], n_ascent.run["vy_m_s"][-1],
             n_ascent.run["vz_m_s"][-1]]))), float(nat.traces["stack"]["speed_m_s"][-1]))
        integration.append(
            {"id": "stack.separation_shift_dt_half",
             "value": {"jsbsim_alt_rel": j_shift, "jsbsim_speed_rel": j_speed,
                       "native_alt_rel": n_shift, "native_speed_rel": n_speed},
             "explanation": "the separation state's response to halving the "
                            "step on each plant (E6-04's method, #276): what "
                            "the numerics -- JSBSim's internal sequence vs "
                            "the plant's fixed-step RK4 -- explain of the "
                            "backend gap."})
        band("e602-jsbsim-integration-stable", j_shift, DT_HALF_JSBSIM_ALT_BAND_REL,
             "fraction", "JSBSim's separation altitude must move < 2 % when "
                         "its step halves")
        band("e602-native-grid-converged", n_shift, DT_HALF_NATIVE_ALT_BAND_REL,
             "fraction", "the native RK4 is already grid-converged (E6-04)")

    # -- model: shared laws, implemented twice -------------------------------
    pad_j = float(jsb.traces["stack"]["thrust_n"][1])
    pad_n = float(nat.traces["stack"]["thrust_n"][1])
    pad_ratio = pad_j / max(pad_n, 1e-9)
    model = [
        {"id": "stack.max_q_pair_pa", "value": [n_st["max_q_pa"], j_st["max_q_pa"]],
         "explanation": "max-Q is one airframe in two atmosphere models "
                        "(US76 vs JSBSim's standard atmosphere) -- the "
                        "disclosed model delta E1-04 already bounded."},
        {"id": "stack.pad_thrust_ratio", "value": pad_ratio,
         "explanation": "both engines debit ambient over the same 2 m^2 exit "
                        "(E6-01, ADR flightlab-0009): the expected ratio is "
                        "exactly 1; drift is the two atmosphere models and "
                        "JSBSim's unit rounding."},
        {"id": "stack.sep_state_pair",
         "value": {"jsbsim": {"alt_m": float(jsb.traces["stack"]["altitude_m"][-1]),
                              "speed_m_s": float(jsb.traces["stack"]["speed_m_s"][-1]),
                              "fpa_deg": float(jsb.traces["stack"]["fpa_deg"][-1])},
                   "native": {"alt_m": float(nat.traces["stack"]["altitude_m"][-1]),
                              "speed_m_s": float(nat.traces["stack"]["speed_m_s"][-1]),
                              "fpa_deg": float(nat.traces["stack"]["fpa_deg"][-1])}},
         "explanation": "the separation state both paths handed to the split: "
                        "everything left after the numerics and the laws -- "
                        "the shared laws (engine, atmosphere, gravity) "
                        "implemented twice, per the E6-01 classification."},
    ]
    band("e602-max-q-agreement", _rel(j_st["max_q_pa"], n_st["max_q_pa"]), MAXQ_REL_BAND,
         "fraction", "atmosphere models may not move max-Q by more than 5 %")
    band("e602-pad-thrust-agreement", abs(pad_ratio - 1.0), PAD_THRUST_RATIO_REL_BAND,
         "fraction", "the shared engine law must reproduce itself at the pad")
    band("e602-sep-alt-agreement",
         _rel(float(jsb.traces["stack"]["altitude_m"][-1]),
              float(nat.traces["stack"]["altitude_m"][-1])), ALT_MECO_REL_BAND,
         "fraction", "E1-04's MECO-altitude band, same rationale")
    band("e602-sep-speed-agreement",
         _rel(float(jsb.traces["stack"]["speed_m_s"][-1]),
              float(nat.traces["stack"]["speed_m_s"][-1])), SPEED_MECO_REL_BAND,
         "fraction", "E1-04's MECO-speed band, same rationale")
    band("e602-sep-fpa-agreement",
         abs(float(jsb.traces["stack"]["fpa_deg"][-1])
             - float(nat.traces["stack"]["fpa_deg"][-1])), SEP_FPA_ABS_BAND_DEG, "deg",
         "two laws each inside the 6 deg band converge the path to a few deg")
    mass_j = np.asarray(jsb.traces["stack"]["mass_kg"])
    mass_n = np.asarray(nat.traces["stack"]["mass_kg"])
    n_common = min(len(mass_j), len(mass_n))
    band("e602-mass-agreement", float(np.max(np.abs(mass_j[:n_common] - mass_n[:n_common]))),
         MASS_PROP_REL_BAND * 60_000.0, "kg", "E1-04's propellant-drain band")

    return {"categories": {"dynamics": dynamics, "guidance": guidance,
                           "integration": integration, "model": model},
            "checks": tuple(checks)}


# --------------------------------------------------------------------------
# the whole study
# --------------------------------------------------------------------------

@dataclass
class E602Study:
    scenario: E602Scenario
    jsbsim: JsbsimAscentStudy
    native: NativeAscentStudy
    comparison: Mapping
    checks: tuple[Check, ...]

    def phase_table(self) -> dict[str, dict[str, dict[str, float]]]:
        return {"jsbsim": self.jsbsim.phase_table(), "native": self.native.phase_table()}

    @property
    def failed_checks(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)


def run_e602_study(scenario: E602Scenario | None = None, *, dt_s: float | None = None,
                   dt_halving: bool = True) -> E602Study:
    """Fly the mission on both plants, settle the acceptance: the declared
    phase constraints on the JSBSim table, the staging bookkeeping, the
    four-channel comparison.  Never raises on a failing check -- read
    :attr:`E602Study.failed_checks`."""
    s = scenario or e602_scenario()
    jsb = run_guided_ascent_jsbsim(s, dt_s=dt_s)
    nat = run_guided_ascent_native(s, dt_s=dt_s)
    comp = compare_guided_ascent(jsb, nat, dt_halving=dt_halving)
    checks = (tuple(e602_phase_constraint_checks(jsb.phase_table()))
              + jsb.checks
              + tuple(_staging_checks(nat.bookkeeping, nat.ignition_onset_err_s,
                                      jsb.scenario.dt_s * jsb.scenario.sample_every,
                                      prefix="native"))
              + tuple(comp["checks"]))
    return E602Study(scenario=s, jsbsim=jsb, native=nat, comparison=comp, checks=checks)
