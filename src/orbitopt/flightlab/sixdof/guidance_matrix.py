"""E6-03 (task #275): the guidance-generalisation matrix.

E1-03 (task #252) closed the upper-stage loop with ONE shared PEG-class law
(:mod:`orbitopt.flightlab.guidance.upper_stage`) and shrank the 6-DOF/3-DOF
altitude gap from 235 km to 1.65 km -- but only for the single disclosed
scenario point it declared: 220 km parking orbit, the F-06 pad on heading 090
(inclination inherited from the pad, never declared), 1 t dry / 7 t propellant
and a 3 deg TVC axis. "Shared law" is a statement about the law, not about
its accuracy envelope. This module flies the law over a declared *matrix* of
scenario points and reports, per cell, the insertion error of each backend,
the cross-backend difference, and -- where the declared tolerance is missed --
which of three things is actually at fault:

* ``vehicle-capability`` -- the stage cannot do the mission at all (no TVC
  travel at all, or a tank whose terminal orbit does not even have the energy
  of the target circle);
* ``guidance`` -- the law's own terminal conditions do not deliver this
  target: either it shuts the burn with propellant to spare and still misses,
  or it never terminates and empties the tank on an orbit WIDER than the
  target -- the case that proves the tank was not the limit;
* ``actuator-constraint`` -- the law's command is right and the point-mass
  twin flies it, but the servo's travel / slew rate / lag cannot realise it.

The matrix axes
---------------
* **target altitude** -- 180 / 220 / 260 / 300 / 400 km circular parking orbit
  (the envelope turns out to be bounded at BOTH ends: too low and the
  periapsis-only cutoff leaves apoapsis over the band, too high and the cutoff
  never fires and the stage burns itself empty over-shooting);
* **inclination** -- 28.5 / 45.0 / 51.6 deg, realised by the pad latitude with
  a due-east launch (see :data:`LAUNCH_HEADING_DEG`: the native plant steers
  with a single TVC axis and no bank, so it cannot dog-leg, and launch
  latitude is the inclination the ascent actually realises. The graded
  residual against the declared (geodetic) inclination is the geodetic to
  geocentric latitude conversion of the pad, ~0.19 deg, well inside the
  declared 1 deg band, and it is recorded per cell rather than hidden;
* **upper-stage vehicle** -- the disclosed E1-03 variant, a stretched tank, a
  heavy-payload variant, and the ORIGINAL F-06 parity stage (2 t dry / 6 t
  propellant, zero gimbal travel) flown open loop as F-06 flies it;
* **actuator set** -- the F-04 standard servo, a derated servo, a hard
  rate-limited servo, a wide-travel servo, and the no-TVC axis of the original
  stage, each applied consistently to the plant's engine travel *and* to the
  servo model (the point-mass twin has no servo at all, which is what turns
  this axis into a measurement instead of a claim).

All guided cells hold the upper stage's *wet* mass at the parity 8 t and the
stage-1 vehicle untouched, so the stack ascent, the frozen separation state
and the booster handoff are the same flight in every guided cell: the only
thing that varies is the target, the stage's own tank split and its servo.
Every cell flies BOTH native backends through
:func:`~orbitopt.flightlab.sixdof.mission_output.run_mission`, so each cell
carries its own 3-DOF/6-DOF difference; the point-mass twin is actuator-free,
which is exactly what makes the actuator axis measurable.

Committed numbers live in ``docs/benchmarks/e6_03_guidance_matrix.json``
(``PYTHONPATH=src python benchmarks/bench_e6_03_guidance_matrix.py``), the
regression locks in ``tests/test_flightlab_e6_03_guidance_matrix.py`` and the
reading of the result in ``docs/flightlab.md`` (E6-03). Like every native
plant study this is illustrative physics, not flight certification.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from orbitopt.entry import G0_M_S2, MU_EARTH_M3_S2, OMEGA_EARTH_RAD_S
from orbitopt.flightlab.guidance.upper_stage import (
    ParkingOrbitTarget, UpperStageGuidanceSpec, insertion_report,
)
from orbitopt.flightlab.sixdof.gravity_turn import compare_ascent
from orbitopt.flightlab.sixdof.mission_output import (
    BACKENDS, BACKEND_3DOF, BACKEND_6DOF, f07_scenario, run_mission,
)
from orbitopt.flightlab.sixdof.rigidbody import RigidBodyState
from orbitopt.flightlab.sixdof.tvc import TvcActuator, WaypointAttitudeProgram

__all__ = [
    "ACTUATOR_SETS",
    "BURN_ENDED_BY",
    "DEPLETION_MASS_KG",
    "BACKEND_PAIR",
    "FAILURE_REASONS",
    "GUIDANCE_MATRIX",
    "INCLINATION_TOLERANCE_DEG",
    "INSERTION_TOLERANCE_M",
    "LAUNCH_HEADING_DEG",
    "MATRIX_DT_S",
    "MATRIX_SAMPLE_EVERY",
    "PAD_LONGITUDE_DEG",
    "REASON_ACTUATOR_CONSTRAINT",
    "REASON_GUIDANCE",
    "REASON_VEHICLE_CAPABILITY",
    "UpperVehicle",
    "VEHICLE_CONFIGS",
    "failure_reasons",
    "ActuatorSet",
    "GuidanceMatrixCell",
    "matrix_coverage",
    "matrix_scenario",
    "run_guidance_matrix",
    "run_matrix_cell",
]

_OMEGA = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

#: Every cell is flown at the packaged F-07 integration plan.
MATRIX_DT_S = 0.05
MATRIX_SAMPLE_EVERY = 20
#: The pad longitude and the due-east launch heading every cell shares (the
#: pad *latitude* is the cell's inclination target -- see the module
#: docstring). 28.5 N / -80.6 W on heading 090 is the F-05/F-06 lineage pad.
PAD_LONGITUDE_DEG = -80.6
LAUNCH_HEADING_DEG = 90.0
#: Declared insertion tolerances of the matrix, common to every cell (the
#: E1-03 packaged mission's own band).
INSERTION_TOLERANCE_M = 25_000.0
INCLINATION_TOLERANCE_DEG = 1.0
#: E6-05 (task #292): every guided cell flies the *generalised* law -- the
#: energy-capped transfer target, the climb-hold feed and the apoapsis-side
#: cutoff (see the ``upper_stage`` module docstring).  One shared setting for
#: the whole matrix, declared here once; no cell carries its own gains.  The
#: climb-hold gain is unity -- the pure time-to-go climb profile (arrive at
#: the target radius exactly as the tank runs dry); the cutoff's apoapsis
#: margin sits inside the declared 25 km insertion tolerance so a burn that
#: cuts on both criteria lands inside the box.
E605_CLIMB_TIME_TO_GO_GAIN = 1.0
E605_CUTOFF_APOAPSIS_MARGIN_M = 10_000.0

BACKEND_PAIR: tuple[str, str] = BACKENDS

#: The three reason codes the acceptance asks a missed cell to name.
REASON_VEHICLE_CAPABILITY = "vehicle-capability"
REASON_GUIDANCE = "guidance"
REASON_ACTUATOR_CONSTRAINT = "actuator-constraint"
FAILURE_REASONS: tuple[str, ...] = (REASON_VEHICLE_CAPABILITY, REASON_GUIDANCE,
                                    REASON_ACTUATOR_CONSTRAINT)


# --------------------------------------------------------------------------
# the axes: vehicle configs and actuator sets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class UpperVehicle:
    """One upper-stage config: the tank split (at the parity 8 t wet mass)
    and whether the stage can fly the shared law at all.

    ``guided=False`` is the ORIGINAL F-06 parity stage: its engine has no TVC
    travel, so the shared law cannot be instantiated on it and the cell flies
    the F-06 open-loop upper stage."""

    vehicle_id: str
    dry_mass_kg: float
    propellant_mass_kg: float
    guided: bool
    note: str

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


@dataclass(frozen=True)
class ActuatorSet:
    """One TVC-axis constraint set: travel, slew-rate limit and first-order
    lag. Applied to BOTH the plant's engine travel and the servo model, so a
    cell's actuator axis is one thing, not two. ``travel_deg == 0`` states
    that the stage has no TVC axis at all."""

    actuator_id: str
    travel_deg: float
    slew_rate_deg_s: float
    lag_tau_s: float
    note: str

    def tvc_actuator(self) -> TvcActuator | None:
        if self.travel_deg <= 0.0:
            return None
        return TvcActuator(limit_rad=math.radians(self.travel_deg),
                           max_slew_rate_rad_s=math.radians(self.slew_rate_deg_s),
                           tau_s=self.lag_tau_s)


VEHICLE_CONFIGS: dict[str, UpperVehicle] = {
    cfg.vehicle_id: cfg for cfg in (
        UpperVehicle(
            "V1-e103-variant", 1_000.0, 7_000.0, True,
            "the disclosed E1-03 stage variant (1 t dry / 7 t propellant at the "
            "parity 8 t wet mass, TVC axis) -- the configuration E1-03 proved"),
        UpperVehicle(
            "V2-stretched-tank", 800.0, 7_200.0, True,
            "stretched tank at the same wet mass (0.8 t dry / 7.2 t propellant): "
            "the delta-v headroom test -- if a cell still misses with this tank, "
            "the tank is not what limited it"),
        UpperVehicle(
            "V3-heavy-payload", 1_500.0, 6_500.0, True,
            "heavy payload at the same wet mass (1.5 t dry / 6.5 t propellant)"),
        UpperVehicle(
            "V0-f06-original", 2_000.0, 6_000.0, False,
            "the ORIGINAL F-06 parity stage (2 t dry / 6 t propellant, zero "
            "gimbal travel): no TVC axis, so the shared law cannot be "
            "instantiated on it and it flies the F-06 open-loop upper stage"),
    )
}

ACTUATOR_SETS: dict[str, ActuatorSet] = {
    aset.actuator_id: aset for aset in (
        ActuatorSet("A0-no-tvc", 0.0, 0.0, 0.0,
                    "no TVC axis (the original F-06 stage)"),
        ActuatorSet("A1-f04-standard", 3.0, math.degrees(0.10), 0.10,
                    "the F-04 standard servo used by E1-03/E1-02 (3 deg travel, "
                    "0.10 rad/s slew, 0.10 s lag)"),
        ActuatorSet("A2-derated", 2.0, math.degrees(0.05), 0.15,
                    "derated servo: 2 deg travel, 0.05 rad/s slew, 0.15 s lag"),
        ActuatorSet("A3-rate-limited", 2.0, math.degrees(0.02), 0.25,
                    "hard rate limit: 2 deg travel, 0.02 rad/s slew (below the "
                    "law's own 3 deg/s command slew limit), 0.25 s lag"),
        ActuatorSet("A4-wide", 5.0, math.degrees(0.15), 0.08,
                    "wide-travel servo: 5 deg travel, 0.15 rad/s slew, 0.08 s lag"),
    )
}


# --------------------------------------------------------------------------
# the cells
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GuidanceMatrixCell:
    """One scenario point of the matrix. ``expected`` is the declared
    verdict (``"insert"`` = both backends inside the declared tolerances,
    ``"miss"`` = the cell is a documented failure whose cause the runner has
    to attribute); the regression test locks the committed study against it."""

    cell_id: str
    target_altitude_m: float
    inclination_deg: float
    vehicle_id: str
    actuator_id: str
    expected: str
    note: str = ""

    @property
    def pad_latitude_deg(self) -> float:
        return self.inclination_deg

    @property
    def vehicle(self) -> UpperVehicle:
        return VEHICLE_CONFIGS[self.vehicle_id]

    @property
    def actuator(self) -> ActuatorSet:
        return ACTUATOR_SETS[self.actuator_id]


def _cell(cell_id: str, altitude_km: float, inclination_deg: float,
          vehicle_id: str, actuator_id: str, expected: str, note: str,
          ) -> GuidanceMatrixCell:
    if vehicle_id not in VEHICLE_CONFIGS:
        raise ValueError(f"unknown vehicle config {vehicle_id!r}")
    if actuator_id not in ACTUATOR_SETS:
        raise ValueError(f"unknown actuator set {actuator_id!r}")
    if expected not in ("insert", "miss"):
        raise ValueError(f"unknown expected verdict {expected!r}")
    return GuidanceMatrixCell(cell_id=cell_id, target_altitude_m=altitude_km * 1e3,
                              inclination_deg=inclination_deg,
                              vehicle_id=vehicle_id, actuator_id=actuator_id,
                              expected=expected, note=note)


#: The matrix. 5 target altitudes x 3 inclinations x 4 vehicle configs x 5
#: actuator sets is 300 points; this is the declared 17-point slice that covers
#: every axis (>=3 altitudes, >=2 inclinations, >=2 payload/vehicle configs,
#: >=2 actuator constraint sets), crosses the payload axis against the
#: altitude axis, crosses the inclination axis against the altitude axis, and
#: includes the original F-06 stage.
GUIDANCE_MATRIX: tuple[GuidanceMatrixCell, ...] = (
    _cell("a220-i28.5-V1-A1", 220, 28.5, "V1-e103-variant", "A1-f04-standard",
          "insert", "the E1-03 packaged scenario point, restated as a matrix "
          "cell with the inclination now declared"),
    _cell("a180-i28.5-V1-A1", 180, 28.5, "V1-e103-variant", "A1-f04-standard",
          "miss", "the LOW end of the envelope: the cutoff fires on periapsis "
          "with propellant to spare and leaves apoapsis over the band -- the "
          "criterion grades one element, not the orbit"),
    _cell("a260-i28.5-V1-A1", 260, 28.5, "V1-e103-variant", "A1-f04-standard",
          "miss", "the cutoff criterion stops closing above ~240 km: the "
          "stage burns to depletion and over-shoots instead of under-shooting"),
    _cell("a300-i28.5-V1-A1", 300, 28.5, "V1-e103-variant", "A1-f04-standard",
          "miss", "the law does not terminate at this target"),
    _cell("a220-i28.5-V2-A1", 220, 28.5, "V2-stretched-tank", "A1-f04-standard",
          "insert", "the nominal target on the stretched tank"),
    _cell("a300-i28.5-V2-A1", 300, 28.5, "V2-stretched-tank", "A1-f04-standard",
          "miss", "the same target with 200 kg more propellant: separates a "
          "guidance failure from a tank limit"),
    _cell("a400-i28.5-V1-A1", 400, 28.5, "V1-e103-variant", "A1-f04-standard",
          "miss", "the far end of the altitude axis"),
    _cell("a220-i45.0-V1-A1", 220, 45.0, "V1-e103-variant", "A1-f04-standard",
          "insert", "inclination axis, mid point"),
    _cell("a220-i51.6-V1-A1", 220, 51.6, "V1-e103-variant", "A1-f04-standard",
          "insert", "inclination axis, high end"),
    _cell("a260-i45.0-V1-A1", 260, 45.0, "V1-e103-variant", "A1-f04-standard",
          "miss", "altitude x inclination crossing"),
    _cell("a300-i51.6-V1-A1", 300, 51.6, "V1-e103-variant", "A1-f04-standard",
          "miss", "altitude x inclination crossing at the high end of both"),
    _cell("a180-i28.5-V3-A1", 180, 28.5, "V3-heavy-payload", "A1-f04-standard",
          "miss", "payload axis: the heavy stage runs dry with its terminal "
          "orbit still tighter than even the low target"),
    _cell("a220-i28.5-V3-A1", 220, 28.5, "V3-heavy-payload", "A1-f04-standard",
          "miss", "payload axis at the nominal target"),
    _cell("a220-i28.5-V1-A2", 220, 28.5, "V1-e103-variant", "A2-derated",
          "insert", "actuator axis: derated servo, still inside the band"),
    _cell("a220-i28.5-V1-A3", 220, 28.5, "V1-e103-variant", "A3-rate-limited",
          "miss", "actuator axis: the servo no longer tracks the command"),
    _cell("a220-i28.5-V1-A4", 220, 28.5, "V1-e103-variant", "A4-wide",
          "insert", "actuator axis: wide travel and a faster servo"),
    _cell("a220-i28.5-V0-A0", 220, 28.5, "V0-f06-original", "A0-no-tvc",
          "miss", "the original F-06 stage, open loop: the pre-E1-03 baseline"),
)


def matrix_coverage() -> dict[str, Any]:
    """What the matrix actually covers, read off the declared cells (the
    acceptance's coverage clause, machine-checkable)."""
    return {
        "cells": len(GUIDANCE_MATRIX),
        "target_altitudes_m": sorted({c.target_altitude_m for c in GUIDANCE_MATRIX}),
        "inclinations_deg": sorted({c.inclination_deg for c in GUIDANCE_MATRIX}),
        "vehicle_ids": [v for v in VEHICLE_CONFIGS
                        if any(c.vehicle_id == v for c in GUIDANCE_MATRIX)],
        "actuator_ids": [a for a in ACTUATOR_SETS
                         if any(c.actuator_id == a for c in GUIDANCE_MATRIX)],
    }


# --------------------------------------------------------------------------
# scenario construction
# --------------------------------------------------------------------------

def matrix_target(cell: GuidanceMatrixCell) -> ParkingOrbitTarget:
    """The declared target of one cell: a circular parking orbit at the cell's
    altitude and inclination, with the matrix-wide insertion tolerances."""
    return ParkingOrbitTarget(
        altitude_m=float(cell.target_altitude_m),
        periapsis_tolerance_m=INSERTION_TOLERANCE_M,
        apoapsis_tolerance_m=INSERTION_TOLERANCE_M,
        inclination_deg=float(cell.inclination_deg),
        inclination_tolerance_deg=INCLINATION_TOLERANCE_DEG)


def matrix_scenario(cell: GuidanceMatrixCell, *, dt_s: float = MATRIX_DT_S,
                    sample_every: int = MATRIX_SAMPLE_EVERY):
    """The :class:`~orbitopt.flightlab.sixdof.two_stage.TwoStageScenario` of
    one matrix cell: the packaged F-07 guided mission, restated at the cell's
    target / vehicle / actuator point. The pad is the cell's inclination on
    the shared longitude with a due-east heading, the stage-1 vehicle and the
    attitude program are the packaged ones, and the upper stage's engine
    travel and servo are the cell's actuator set.

    Guided cells get a freshly built :class:`UpperStageGuidanceSpec` whose
    target carries the declared inclination (the law ignores that entry; it is
    what :func:`run_matrix_cell` grades against), so the scenario's own target
    and the grading target cannot drift apart."""
    from orbitopt.flightlab.sixdof.two_stage import TwoStageScenario

    vehicle, aset = cell.vehicle, cell.actuator
    travel_deg = aset.travel_deg if vehicle.guided else 0.0
    scenario: TwoStageScenario = f07_scenario(
        guided_upper_stage=vehicle.guided,
        parking_altitude_m=float(cell.target_altitude_m),
        insertion_periapsis_tolerance_m=INSERTION_TOLERANCE_M,
        insertion_apoapsis_tolerance_m=INSERTION_TOLERANCE_M,
        upper_dry_mass_kg=vehicle.dry_mass_kg,
        upper_propellant_mass_kg=vehicle.propellant_mass_kg,
        upper_gimbal_limit_rad=math.radians(travel_deg),
        dt_s=dt_s, sample_every=sample_every)
    state0 = RigidBodyState.from_launch(
        lat_geod_deg=cell.pad_latitude_deg, lon_deg=PAD_LONGITUDE_DEG,
        heading_deg=LAUNCH_HEADING_DEG, pitch_deg=90.0, roll_deg=0.0,
        mass_kg=scenario.pair.stack.mass_wet_kg)
    program = WaypointAttitudeProgram(times_s=(0.0, 10.0, 110.0),
                                      heading_deg=(LAUNCH_HEADING_DEG,) * 3,
                                      pitch_deg=(90.0, 90.0, 20.0),
                                      roll_deg=(0.0, 0.0, 0.0))
    scenario = replace(scenario, state0=state0, program=program)
    if vehicle.guided:
        servo = aset.tvc_actuator()
        assert servo is not None  # a guided config never names A0-no-tvc
        scenario = replace(
            scenario, upper_actuator=servo,
            upper_guidance=UpperStageGuidanceSpec.from_vehicle(
                scenario.pair.upper, target=matrix_target(cell),
                ignition_t_s=scenario.ignition.ignition_t_s,
                transfer_target=True,
                climb_time_to_go_gain=E605_CLIMB_TIME_TO_GO_GAIN,
                cutoff_apoapsis_margin_m=E605_CUTOFF_APOAPSIS_MARGIN_M))
    return scenario


# --------------------------------------------------------------------------
# running one cell
# --------------------------------------------------------------------------

#: Below this much propellant left there is nothing left to command with: the
#: trace of a throttle that closes here records a dry tank, not a cutoff.
DEPLETION_MASS_KG = 1.0

#: How a cell's upper-stage burn ended, read off the throttle trace and the
#: mass trace together (the throttle closing at the end of a dry tank is a
#: *depletion*, not a cutoff -- the two look identical in the throttle trace
#: alone, and they are the difference between a guidance failure and a
#: vehicle-capability failure).
BURN_ENDED_BY = ("no-law", "still-burning", "depletion", "law-cutoff")


def _cutoff_t_s_of(run) -> float | None:
    """First sampled instant after which a guided burn is found shut (``None``
    when the burn was still alight at the last sample). Same sampling-grid
    reading as the F-07 comparison report's; read it together with
    ``burn_ended_by``, which separates a commanded cutoff from a dry tank."""
    throttle = np.asarray(run.samples["throttle"], dtype=float)
    burning = np.nonzero(throttle > 0.0)[0]
    if burning.size == 0 or int(burning[-1]) >= throttle.size - 1:
        return None
    return round(float(run.samples["t_s"][int(burning[-1]) + 1]), 3)


def _terminal_state(run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The last sampled state: geocentric position, the plant's own ECEF
    velocity (what ``insertion_report`` expects -- it adds ``omega x r``
    itself), and the inertial velocity the speed / flight-path diagnostics are
    read from."""
    s = run.samples
    r = np.array([float(s["x_m"][-1]), float(s["y_m"][-1]), float(s["z_m"][-1])])
    v_ecef = np.array([float(s["vx_m_s"][-1]), float(s["vy_m_s"][-1]),
                       float(s["vz_m_s"][-1])])
    return r, v_ecef, v_ecef + np.cross(_OMEGA, r)


def _backend_record(study, cell: GuidanceMatrixCell) -> dict[str, Any]:
    """One backend of one cell: the terminal orbit graded against the declared
    target, the burn bookkeeping that lets a miss be attributed, and the
    cell's delta-v budget."""
    run = study.upper
    r, v_ecef, v_inertial = _terminal_state(run)
    graded = matrix_target(cell)
    insertion = insertion_report(graded, r, v_ecef, frame="ECEF")
    rmag = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v_inertial))
    fpa_deg = math.degrees(math.asin(max(-1.0, min(1.0, float(v_inertial @ r)
                                                   / (speed * rmag)))))
    v_circ = math.sqrt(MU_EARTH_M3_S2 / graded.radius_m)
    dry = float(cell.vehicle.dry_mass_kg)
    prop_at_ignition = float(study.separation["upper_propellant_kg"])
    mass_end = float(run.samples["mass_kg"][-1])
    prop_left = max(mass_end - dry, 0.0)
    burn_off = _cutoff_t_s_of(run) if cell.vehicle.guided else None
    if not cell.vehicle.guided:
        ended_by = "no-law"
    elif burn_off is None:
        ended_by = "still-burning"
    elif prop_left <= DEPLETION_MASS_KG:
        ended_by = "depletion"
    else:
        ended_by = "law-cutoff"
    # the terminal orbit's size, straight from the state (the energy margin is
    # what separates "the tank could not pay for this orbit" from "the law put
    # the available energy in the wrong shape")
    vm2 = float(v_inertial @ v_inertial)
    a_terminal = (1.0 / (2.0 / rmag - vm2 / MU_EARTH_M3_S2)
                  if 2.0 / rmag - vm2 / MU_EARTH_M3_S2 > 0.0 else float("inf"))
    return {
        "insertion": insertion,
        "inserted": bool(insertion["inserted"]),
        "burn_off_t_s": burn_off,
        "burn_ended_by": ended_by,
        "ignition_t_s": float(study.scenario.ignition.ignition_t_s),
        "terminal": str(run.terminal),
        "terminal_mass_kg": mass_end,
        "propellant_remaining_kg": prop_left,
        "propellant_burned_kg": prop_at_ignition - prop_left,
        "terminal_semi_major_axis_m": a_terminal,
        "energy_margin_vs_target_orbit_m": a_terminal - graded.radius_m,
        "propellant_at_ignition_kg": prop_at_ignition,
        "dv_ideal_m_s": (float(study.scenario.pair.upper.engine.isp_vacuum_s)
                         * G0_M_S2 * math.log((dry + prop_at_ignition) / dry)),
        "terminal_speed_m_s": speed,
        "target_circular_speed_m_s": v_circ,
        "speed_surplus_m_s": speed - v_circ,
        "terminal_flight_path_angle_deg": fpa_deg,
    }


def _cross_backend(studies: dict[str, Any]) -> dict[str, Any]:
    """The cell's cross-backend difference: the F-05 channel metrics on the
    upper body plus the difference of the two graded insertion errors."""
    i3, i6 = (studies[b][1]["insertion"] for b in (BACKEND_3DOF, BACKEND_6DOF))
    out: dict[str, Any] = {
        "d_periapsis_6dof_minus_3dof_m": (i6["periapsis_altitude_m"]
                                          - i3["periapsis_altitude_m"]),
        "d_apoapsis_6dof_minus_3dof_m": (i6["apoapsis_altitude_m"]
                                         - i3["apoapsis_altitude_m"]),
        "d_inclination_6dof_minus_3dof_deg": (i6["inclination_deg"]
                                              - i3["inclination_deg"]),
        "d_terminal_mass_6dof_minus_3dof_kg": (
            studies[BACKEND_6DOF][1]["terminal_mass_kg"]
            - studies[BACKEND_3DOF][1]["terminal_mass_kg"]),
    }
    try:
        out["ascent_diff"] = compare_ascent(studies[BACKEND_3DOF][0].upper,
                                            studies[BACKEND_6DOF][0].upper,
                                            label="6dof minus 3dof (upper)").summary()
    except ValueError as exc:  # no shared span: visible, never faked
        out["ascent_diff"] = {"error": str(exc)}
    return out


def failure_reasons(cell: GuidanceMatrixCell, backends: dict[str, Any]) -> list[dict]:
    """Attribute a missed cell to vehicle capability, guidance or actuator
    constraint. Every branch reads quantities the record itself carries (how
    the burn ended, how much propellant is left, whether the terminal orbit has
    the energy of the target orbit, whether the actuator-free twin made it), so
    the attribution is a rule over the data rather than a label."""
    recs = [backends[b] for b in BACKEND_PAIR]
    if not cell.vehicle.guided:
        return [{
            "code": REASON_VEHICLE_CAPABILITY,
            "detail": "the original F-06 upper stage has no TVC travel, so the "
                      "shared law cannot even be instantiated on it "
                      "(TvcActuator requires positive travel); it flies the "
                      "F-06 open-loop upper stage on a "
                      f"{recs[0]['dv_ideal_m_s']:.0f} m/s ideal delta-v budget. "
                      "This is the pre-E1-03 baseline, not a guidance result.",
        }]
    if sum(1 for b in recs if b["inserted"]) == 1:
        red = [b for b in BACKEND_PAIR if not backends[b]["inserted"]][0]
        return [{
            "code": REASON_ACTUATOR_CONSTRAINT,
            "detail": "the actuator-free point-mass twin inserts on the same law "
                      f"with {backends[BACKEND_3DOF]['propellant_remaining_kg']:.0f}"
                      " kg of margin while "
                      f"{red} does not "
                      f"({backends[red]['burn_ended_by']}), so the mission "
                      "was achievable and the tank paid for it: what failed is "
                      f"realising the command through the servo "
                      f"({cell.actuator.travel_deg:.1f} deg travel, "
                      f"{cell.actuator.slew_rate_deg_s:.2f} deg/s slew, "
                      f"{cell.actuator.lag_tau_s:.2f} s lag). Neither a law "
                      "failure nor a vehicle limit.",
        }]
    depleted = [b for b in recs if b["burn_ended_by"] == "depletion"]
    if len(depleted) == len(recs):
        margin = max(b["energy_margin_vs_target_orbit_m"] for b in recs)
        if not math.isfinite(margin) or margin > 0.0:
            return [{
                "code": REASON_GUIDANCE,
                "detail": "the tank emptied because the burn never terminated on "
                          "the osculating-periapsis criterion, and the terminal "
                          f"orbit is {abs(margin) / 1e3:.0f} km WIDER than the "
                          "target orbit: the stage had the energy for this "
                          "target and more, and the law flew past it. Propellant "
                          "was not the binding limit (the cutoff criterion is).",
            }]
        return [{
            "code": REASON_VEHICLE_CAPABILITY,
            "detail": "the tank ran dry with the terminal orbit still "
                      f"{abs(margin) / 1e3:.0f} km TIGHTER than the target orbit "
                      "(and its periapsis "
                      f"{min(b['insertion']['d_periapsis_m'] for b in recs) / 1e3:.0f}"
                      " km short of it): this stage's delta-v budget does not "
                      "reach this target orbit, so the cutoff criterion never "
                      "had a chance to close. Not a law failure.",
        }]
    return [{
        "code": REASON_GUIDANCE,
        "detail": "the law shut the burn with propellant to spare ("
                  f"{min(b['propellant_remaining_kg'] for b in recs):.0f} kg left)"
                  " and the terminal orbit is still outside the declared "
                  "tolerance: the law's own terminal conditions do not deliver "
                  "this target. Periapsis is graded, apoapsis is only a "
                  "consequence of where the cutoff fired, which is where this "
                  "miss sits.",
    }]


def run_matrix_cell(cell: GuidanceMatrixCell, *, dt_s: float = MATRIX_DT_S,
                    sample_every: int = MATRIX_SAMPLE_EVERY,
                    epoch_utc: datetime | None = None) -> dict[str, Any]:
    """Fly one cell on both native backends and return its record: declared
    scenario point, per-backend insertion grading and burn bookkeeping, the
    cross-backend difference, and the pass/miss verdict with the attributed
    cause of a miss."""
    scenario = matrix_scenario(cell, dt_s=dt_s, sample_every=sample_every)
    studies: dict[str, tuple] = {}
    backends: dict[str, Any] = {}
    for backend in BACKEND_PAIR:
        mission = run_mission(scenario, backend=backend, epoch_utc=epoch_utc)
        studies[backend] = (mission.study, _backend_record(mission.study, cell))
        backends[backend] = studies[backend][1]
    inserted = all(backends[b]["inserted"] for b in BACKEND_PAIR)
    record: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "expected": cell.expected,
        "t_end_s": float(scenario.t_end_s),
        "note": cell.note,
        "target": {
            "altitude_m": float(cell.target_altitude_m),
            "inclination_deg": float(cell.inclination_deg),
            "periapsis_tolerance_m": INSERTION_TOLERANCE_M,
            "apoapsis_tolerance_m": INSERTION_TOLERANCE_M,
            "inclination_tolerance_deg": INCLINATION_TOLERANCE_DEG,
            "pad_latitude_deg": cell.pad_latitude_deg,
            "pad_longitude_deg": PAD_LONGITUDE_DEG,
            "launch_heading_deg": LAUNCH_HEADING_DEG,
        },
        "vehicle": {
            "vehicle_id": cell.vehicle_id,
            "upper_dry_mass_kg": cell.vehicle.dry_mass_kg,
            "upper_propellant_mass_kg": cell.vehicle.propellant_mass_kg,
            "upper_wet_mass_kg": cell.vehicle.wet_mass_kg,
            "guided": cell.vehicle.guided,
            "note": cell.vehicle.note,
        },
        "actuator": {
            "actuator_id": cell.actuator_id,
            "travel_deg": cell.actuator.travel_deg,
            "slew_rate_deg_s": cell.actuator.slew_rate_deg_s,
            "lag_tau_s": cell.actuator.lag_tau_s,
            "note": cell.actuator.note,
        },
        "backends": backends,
        "cross_backend": _cross_backend(studies),
        "inserted_both": bool(inserted),
        "status": "pass" if inserted else "miss",
    }
    record["reasons"] = [] if inserted else failure_reasons(cell, backends)
    record["unexpected"] = bool((cell.expected == "insert") != inserted)
    return record


def run_guidance_matrix(cells: tuple[GuidanceMatrixCell, ...] = GUIDANCE_MATRIX,
                        *, dt_s: float = MATRIX_DT_S,
                        sample_every: int = MATRIX_SAMPLE_EVERY) -> dict[str, Any]:
    """Fly the whole matrix and return the study payload (the benchmark writes
    it; the regression test reads it)."""
    t0 = time.time()
    flown = [run_matrix_cell(cell, dt_s=dt_s, sample_every=sample_every)
             for cell in cells]
    return {
        "schema": "orbitopt-e603-1",
        "task": "#275 (E6-03, epic #265)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_clock_s": round(time.time() - t0, 1),
        "integration": {"dt_s": dt_s, "sample_every": sample_every},
        "law": "orbitopt.flightlab.guidance.upper_stage.UpperStageGuidance "
               "(the single shared PEG-class law of E1-03; every guided cell "
               "flies the identical spec, only the declared target changes)",
        "declared": {
            "insertion_tolerance_m": INSERTION_TOLERANCE_M,
            "inclination_tolerance_deg": INCLINATION_TOLERANCE_DEG,
            "pad_longitude_deg": PAD_LONGITUDE_DEG,
            "launch_heading_deg": LAUNCH_HEADING_DEG,
            "inclination_convention": "launch latitude = declared inclination, "
                                      "due-east heading: the native plant "
                                      "steers with one TVC axis and no bank, "
                                      "so it cannot dog-leg",
        },
        "coverage": matrix_coverage(),
        "cells": flown,
    }
