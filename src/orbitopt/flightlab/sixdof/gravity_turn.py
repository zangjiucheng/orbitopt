"""F-05 (task #230): a 6-DOF gravity-turn ascent flown against a 3-DOF
point-mass reference, with the difference curves, the difference metrics and
an attribution of where the differences come from.

What "the same flight, twice" means here
----------------------------------------
The acceptance asks for a 6-DOF gravity-turn ascent run "under the same
vehicle, guidance and near-identical physics assumptions" as the 3-DOF
comparison. Flight Lab's formal 3-DOF ascent engine is OpenTsiolkovsky
(:mod:`orbitopt.flightlab.adapters.opentsiolkovsky`), but the OT binary is a
pinned *external* artefact and its physics stack is its own (its atmosphere,
Isp law and integrator are not this plant's), so a live OT run compares two
vehicles and two physics stacks at once -- the fidelity question F-05 asks
would be buried. This module therefore flies the *same class* of model the OT
adapter drives -- a point mass, attitude kinematically following the same
attitude program, aerodynamics as ``q_bar S CA/CN(M, alpha)`` -- natively, on
the identical vehicle object, the identical
:class:`~orbitopt.flightlab.sixdof.rigidbody.RigidBodyPhysics` switches, the
identical ambient-corrected engine law and the identical RK4 core. The pair
then differs ONLY in the 6-DOF content: rigid-body attitude dynamics, the TVC
actuator, and the aero realised on a body with orientation states instead of
on a point. That is the isolation F-05's acceptance then quantifies, and
``tests/test_flightlab_sixdof_opentsiolkovsky.py`` remains where the real
pinned binary is cross-checked.

The five rungs and the attribution
---------------------------------
One scenario (:func:`f05_scenario`) is flown five times, each rung adding one
piece of 6-DOF physics on top of the rung below it::

    3dof_vacuum  point mass, aero off                -- shared-physics baseline
    3dof_aero    point mass, aero on                 -- + drag/lift (OT-style)
    6dof_ideal   rigid body, tracker, ideal actuator, aero off
    6dof_lag     rigid body, tracker, real actuator, aero off
    6dof_full    rigid body, tracker, real actuator, aero on   (the ascent)

The attribution table the acceptance asks for is four labelled rung
*differences*, chosen so each flips exactly one piece of physics (see
:data:`ATTRIBUTION_STEPS`): ``attitude_dynamics`` (rigid-body inertia and the
discrete tracker, aero off both sides), ``actuator_lag`` (the F-04 lag/slew
axis, aero still off), ``aero_rigid_body`` (body-oriented aero closed onto
the lagged loop) and ``aero_point_mass`` (the drag the 3-DOF baseline itself
carries, measured on the point mass). The headline difference -- the
acceptance's four difference channels, ``6dof_full`` against ``3dof_aero`` --
telescopes over them EXACTLY, sample-wise on the shared grid:
``headline = attitude_dynamics + actuator_lag + aero_rigid_body -
aero_point_mass``. Altitude, speed and mass use the contract's ECEF states,
and flight-path angle is the angle between the ECEF velocity and the local
horizontal (the ``speed_earth``/FPA pair the OT trajectory CSV reports).

The numbers themselves live in ``docs/benchmarks/fl605_gravity_turn_ascent.json``
(``PYTHONPATH=src python benchmarks/bench_fl605_gravity_turn.py``), the prose
in ``docs/flightlab.md`` (F-05 section), and the regression thresholds in
``tests/test_flightlab_sixdof_gravity_turn.py``.

Conventions are the reference plant's (body x fwd / y right / z down, ECEF
states, scalar-first Tec2b quaternion); no column here is smoothed or
filtered. Nothing in this module touches the JSBSim backend or the OT
adapter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from orbitopt.flightlab import frames
from orbitopt.flightlab.sixdof.aero import MaxQ, load_packaged_aero, max_q_of
from orbitopt.flightlab.sixdof.rigidbody import (
    _OMEGA_ECEF,
    _air_state,
    _gravity_ecef,
    _ned2ecef,
    RigidBodyEngine,
    RigidBodyPhysics,
    RigidBodyState,
    RigidBodyVehicle,
    SixDofRun,
)
from orbitopt.flightlab.sixdof.tvc import (
    ControlledAscent,
    TvcActuator,
    TvcGains,
    WaypointAttitudeProgram,
    run_closed_loop_ascent,
)

__all__ = [
    "AscentDiff",
    "F05Study",
    "GravityTurnScenario",
    "PointMassRun",
    "RUN_COLUMNS_3DOF",
    "RUNG_ORDER",
    "ascent_channels",
    "compare_ascent",
    "f05_scenario",
    "propagate_point_mass",
    "run_f05_study",
]

#: Sample columns of a :class:`PointMassRun` (the 6-DOF run's set minus the
#: attitude block, plus the two channels the comparison reports directly).
RUN_COLUMNS_3DOF: tuple[str, ...] = (
    "t_s", "mass_kg", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
    "altitude_m", "mach", "alpha_deg", "beta_deg", "q_bar_pa", "throttle",
    "thrust_n", "speed_earth_m_s", "fpa_deg",
    "aero_fx_N", "aero_fy_N", "aero_fz_N",
)

#: The attribution ladder, bottom to top. Each rung adds one piece of physics.
RUNG_ORDER: tuple[str, ...] = ("3dof_vacuum", "3dof_aero", "6dof_ideal",
                               "6dof_lag", "6dof_full")

#: The attribution terms, each a labelled pair of rungs ``(high, low)`` of
#: RUNG_ORDER that flips exactly one piece of physics between its two runs.
#: The headline difference telescopes over them EXACTLY (sample-wise, the
#: grids coincide):
#:
#:     6dof_full - 3dof_aero = attitude_dynamics + actuator_lag
#:                             + aero_rigid_body - aero_point_mass
#:
#: i.e. the three 6-DOF increments minus the drag the 3-DOF baseline
#: (``3dof_aero``) already carries but the aero-off rungs of the increments
#: do not.
ATTRIBUTION_STEPS: tuple[tuple[str, str], ...] = (
    ("attitude_dynamics", "rigid-body inertia + discrete tracker, aero off both "
     "sides (6dof_ideal - 3dof_vacuum)"),
    ("actuator_lag", "TVC first-order lag + slew limit on the aero-off loop "
     "(6dof_lag - 6dof_ideal)"),
    ("aero_rigid_body", "body-oriented aero on the lagged 6-DOF loop "
     "(6dof_full - 6dof_lag)"),
    ("aero_point_mass", "drag/lift carried by the 3-DOF baseline, the headline's "
     "counter-term (3dof_aero - 3dof_vacuum)"),
)

ATTRIBUTION_RUNG_PAIRS: dict[str, tuple[int, int]] = {
    "attitude_dynamics": (2, 0),
    "actuator_lag": (3, 2),
    "aero_rigid_body": (4, 3),
    "aero_point_mass": (1, 0),
}


# --------------------------------------------------------------------------
# the 3-DOF point-mass reference
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PointMassRun:
    """A flown 3-DOF point-mass ascent (module docstring for the model).
    Mirrors :class:`~orbitopt.flightlab.sixdof.rigidbody.SixDofRun` so one
    comparison code path reads both: ``samples`` holds
    :data:`RUN_COLUMNS_3DOF` as arrays, ``terminal`` is ``t_end`` / ``ground``
    / ``propellant`` and ``ground_t_s`` the interpolated crossing."""

    columns: tuple[str, ...]
    samples: Mapping[str, np.ndarray]
    vehicle: RigidBodyVehicle
    physics: RigidBodyPhysics
    epoch_utc: datetime
    terminal: str = "t_end"
    ground_t_s: float | None = None
    propellant_exhausted_t_s: float | None = None

    def __getitem__(self, column: str) -> np.ndarray:
        return self.samples[column]

    def __len__(self) -> int:
        return int(self.samples["t_s"].size)

    @property
    def max_q(self) -> MaxQ:
        s = self.samples
        return max_q_of(s["t_s"], s["q_bar_pa"], altitude_m=s["altitude_m"],
                        mach=s["mach"])

    @property
    def alpha_max_deg(self) -> float:
        """Largest |AoA| seen -- the same reading as the 6-DOF run's, and the
        reason the two run kinds drop into one record shape."""
        return float(np.max(np.abs(self.samples["alpha_deg"])))


def _point_mass_deriv(t_s: float, y: np.ndarray, vehicle: RigidBodyVehicle,
                      physics: RigidBodyPhysics, program: WaypointAttitudeProgram,
                      throttle_command: float, use_aero: bool, dry: float) -> np.ndarray:
    """RHS of the 7-component point-mass state ``(r, v, m)``: the plant's own
    gravity/J2/rotation and air model, thrust along the *program* axis with a
    zero gimbal deflection (kinematic attitude -- the OT ``free_flight:
    attitude_program`` assumption), aero as the same coefficient tables read
    at the velocity-vs-program angles, moment-free."""
    r, v = y[0:3], y[3:6]
    mass = max(float(y[6]), dry)
    lat_deg, lon_deg, _ = frames.ecef_to_geodetic(r)
    tec2b = program.dcm_at(t_s, float(lat_deg), float(lon_deg))
    air = _air_state(r, v, tec2b, physics)
    rho = float(air["density_kg_m3"])
    p_pa = float(air["pressure_pa"])
    if not use_aero or vehicle.aero is None or rho <= 0.0 or float(air["q_bar_pa"]) <= 0.0:
        force_aero = np.zeros(3)
    else:
        force_aero, _ = vehicle.aero.force_and_moment(
            float(air["q_bar_pa"]), float(air["mach"]), float(air["alpha_deg"]),
            float(air["beta_deg"]), np.zeros(3), float(air["speed_of_sound_m_s"]))
    throttle = vehicle.engine.throttle(throttle_command) if mass > dry else 0.0
    force_thrust, mdot, _ = vehicle.engine.force_body_n(throttle, 0.0, 0.0, p_pa)

    accel = tec2b.T @ ((force_aero + force_thrust) / mass)
    accel = accel + _gravity_ecef(r, physics.j2)
    if physics.rotation:
        accel = (accel - 2.0 * np.cross(_OMEGA_ECEF, v)
                 - np.cross(_OMEGA_ECEF, np.cross(_OMEGA_ECEF, r)))
    dydt = np.empty(7)
    dydt[0:3] = v
    dydt[3:6] = accel
    dydt[6] = -mdot if mass > dry else 0.0
    return dydt


def _speed_fpa_deg(r: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Earth-relative speed [m/s] and flight-path angle [deg] of one ECEF
    state: the angle between the velocity and the local horizontal of the
    vehicle's own geodetic site (``speed_earth_mps`` / FPA in the OT
    trajectory CSV sense; zero speed reports FPA 0)."""
    speed = float(np.linalg.norm(v))
    if speed <= 0.0:
        return 0.0, 0.0
    lat_deg, lon_deg, _ = frames.ecef_to_geodetic(r)
    v_ned = _ned2ecef(math.radians(float(lat_deg)), math.radians(float(lon_deg))).T @ v
    fpa = math.degrees(math.asin(max(-1.0, min(1.0, -float(v_ned[2]) / speed))))
    return speed, fpa


def _point_mass_sample(t_s: float, y: np.ndarray, vehicle: RigidBodyVehicle,
                       physics: RigidBodyPhysics, program: WaypointAttitudeProgram,
                       throttle_command: float, use_aero: bool, dry: float) -> dict[str, float]:
    r, v = y[0:3], y[3:6]
    lat_deg, lon_deg, _ = frames.ecef_to_geodetic(r)
    tec2b = program.dcm_at(t_s, float(lat_deg), float(lon_deg))
    air = _air_state(r, v, tec2b, physics)
    rho = float(air["density_kg_m3"])
    if not use_aero or vehicle.aero is None or rho <= 0.0 or float(air["q_bar_pa"]) <= 0.0:
        force_aero = np.zeros(3)
    else:
        force_aero, _ = vehicle.aero.force_and_moment(
            float(air["q_bar_pa"]), float(air["mach"]), float(air["alpha_deg"]),
            float(air["beta_deg"]), np.zeros(3), float(air["speed_of_sound_m_s"]))
    mass = float(y[6])
    throttle = vehicle.engine.throttle(throttle_command) if mass > dry else 0.0
    force_thrust, _, _ = vehicle.engine.force_body_n(throttle, 0.0, 0.0,
                                                    float(air["pressure_pa"]))
    speed, fpa = _speed_fpa_deg(r, v)
    row = {
        "t_s": float(t_s), "mass_kg": mass,
        "x_m": float(r[0]), "y_m": float(r[1]), "z_m": float(r[2]),
        "vx_m_s": float(v[0]), "vy_m_s": float(v[1]), "vz_m_s": float(v[2]),
        "altitude_m": float(air["altitude_m"]), "mach": float(air["mach"]),
        "alpha_deg": float(air["alpha_deg"]), "beta_deg": float(air["beta_deg"]),
        "q_bar_pa": float(air["q_bar_pa"]), "throttle": throttle,
        "thrust_n": float(np.linalg.norm(force_thrust)),
        "speed_earth_m_s": speed, "fpa_deg": fpa,
        "aero_fx_N": float(force_aero[0]), "aero_fy_N": float(force_aero[1]),
        "aero_fz_N": float(force_aero[2]),
    }
    return {key: float(value) for key, value in row.items()}


def propagate_point_mass(state: RigidBodyState, vehicle: RigidBodyVehicle,
                         program: WaypointAttitudeProgram, t_end_s: float,
                         *, dt_s: float = 0.02, sample_every: int = 1,
                         physics: RigidBodyPhysics | None = None,
                         throttle: float = 1.0, use_aero: bool = True,
                         epoch_utc: datetime | None = None) -> PointMassRun:
    """Fly the point-mass reference for ``t_end_s`` on the same core the
    6-DOF plant uses: fixed-step RK4 in ECEF, the same gravity/J2/rotation,
    the same atmosphere, the same engine law, the propellant floored at dry
    mass and a ground stop with interpolated crossing.

    Attitude is *kinematic*: ``Tec2b`` comes straight from ``program`` at the
    live site, so the trajectory never sees quaternion states or body rates,
    the gimbal is always zero, and aerodynamics act as forces only. This is
    the 3-DOF side of the F-05 comparison (module docstring for why a native
    reference, not the pinned OT binary)."""
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if int(sample_every) < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if float(t_end_s) <= state.t_s:
        raise ValueError(f"t_end_s ({t_end_s}) must be after state.t_s ({state.t_s})")
    physics = physics or RigidBodyPhysics()
    epoch = epoch_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
    dry = vehicle.dry_mass_kg
    y = np.zeros(7)
    y[0:3] = np.asarray(state.position_m, dtype=float)
    y[3:6] = np.asarray(state.velocity_m_s, dtype=float)
    y[6] = float(state.mass_kg)
    if y[6] < dry:
        raise ValueError(f"state mass {y[6]} kg is below the vehicle dry mass {dry} kg")

    samples: dict[str, list[float]] = {column: [] for column in RUN_COLUMNS_3DOF}

    def append(t_s: float, yy: np.ndarray) -> None:
        row = _point_mass_sample(t_s, yy, vehicle, physics, program, throttle, use_aero, dry)
        for column in RUN_COLUMNS_3DOF:
            samples[column].append(row[column])

    append(state.t_s, y)
    t = float(state.t_s)
    dt = float(dt_s)
    step = 0
    terminal = "t_end"
    ground_t_s: float | None = None
    exhausted_t_s: float | None = None
    if y[6] <= dry + 1e-9:
        exhausted_t_s = t
    prev_h = float(frames.ecef_to_geodetic(y[0:3])[2])

    while t < float(t_end_s) - 1e-12:
        h = min(dt, float(t_end_s) - t)
        k1 = _point_mass_deriv(t, y, vehicle, physics, program, throttle, use_aero, dry)
        k2 = _point_mass_deriv(t + 0.5 * h, y + 0.5 * h * k1, vehicle, physics, program,
                               throttle, use_aero, dry)
        k3 = _point_mass_deriv(t + 0.5 * h, y + 0.5 * h * k2, vehicle, physics, program,
                               throttle, use_aero, dry)
        k4 = _point_mass_deriv(t + h, y + h * k3, vehicle, physics, program,
                               throttle, use_aero, dry)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y[6] = max(float(y[6]), dry)
        t += h
        step += 1
        h_now = float(frames.ecef_to_geodetic(y[0:3])[2])
        if exhausted_t_s is None and y[6] <= dry + 1e-9:
            exhausted_t_s = t
        if h_now <= 0.0 < prev_h:
            terminal = "ground"
            ground_t_s = t - h * h_now / (h_now - prev_h) if h_now != prev_h else t
            break
        if step % int(sample_every) == 0:
            append(t, y)
        prev_h = h_now

    if terminal != "ground" and samples["t_s"][-1] != t:
        append(t, y)
    return PointMassRun(columns=RUN_COLUMNS_3DOF,
                        samples={key: np.asarray(values, dtype=float)
                                 for key, values in samples.items()},
                        vehicle=vehicle, physics=physics, epoch_utc=epoch,
                        terminal=terminal, ground_t_s=ground_t_s,
                        propellant_exhausted_t_s=exhausted_t_s)


# --------------------------------------------------------------------------
# comparison channels and the difference record
# --------------------------------------------------------------------------

#: The acceptance's four channels -> the diff-array key and the metric names.
DIFF_CHANNELS: tuple[str, ...] = ("altitude_m", "speed_m_s", "fpa_deg", "mass_kg")


def ascent_channels(run: PointMassRun | SixDofRun) -> dict[str, np.ndarray]:
    """The comparison channels of either run kind, derived from the raw ECEF
    sample columns only: ``t_s``, ``mass_kg``, ``altitude_m`` (ellipsoid
    height), ``speed_m_s`` (ECEF speed) and ``fpa_deg``. Recomputed rather
    than copied from the sampled extras so both sides of a comparison are
    measured by the same arithmetic."""
    s = run.samples
    t = np.asarray(s["t_s"], dtype=float)
    r = np.stack([np.asarray(s["x_m"], float), np.asarray(s["y_m"], float),
                  np.asarray(s["z_m"], float)], axis=-1)
    v = np.stack([np.asarray(s["vx_m_s"], float), np.asarray(s["vy_m_s"], float),
                  np.asarray(s["vz_m_s"], float)], axis=-1)
    lat_deg, lon_deg, h_m = frames.ecef_to_geodetic(r)
    speed = np.sqrt(np.sum(v * v, axis=1))
    fpa = np.zeros_like(speed)
    for i in range(t.size):
        if speed[i] > 0.0:
            v_ned = _ned2ecef(math.radians(float(lat_deg[i])),
                              math.radians(float(lon_deg[i]))).T @ v[i]
            fpa[i] = math.degrees(math.asin(max(-1.0, min(1.0, -float(v_ned[2]) / speed[i]))))
    return {"t_s": t, "mass_kg": np.asarray(s["mass_kg"], dtype=float),
            "altitude_m": np.asarray(h_m, dtype=float), "speed_m_s": speed,
            "fpa_deg": fpa}


def _interp_channels(ch: Mapping[str, np.ndarray], t_dst: np.ndarray) -> dict[str, np.ndarray]:
    out = {"t_s": t_dst}
    for key in DIFF_CHANNELS:
        out[key] = np.interp(t_dst, ch["t_s"], ch[key])
    return out


@dataclass(frozen=True)
class AscentDiff:
    """3-DOF/6-DOF (or rung-minus-rung) difference on one common time grid:
    the acceptance's four difference curves and their regression metrics
    (``max_abs`` / ``rms`` / ``terminal``, the difference at the last common
    sample). ``ref`` and ``six`` are the two runs' channels on the same grid
    -- the overlay the difference is read against."""

    label: str
    t_s: np.ndarray
    ref: Mapping[str, np.ndarray]
    six: Mapping[str, np.ndarray]
    diff: Mapping[str, np.ndarray]
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def curve(self, channel: str) -> np.ndarray:
        """The difference curve ``six - ref`` of one acceptance channel."""
        if channel not in DIFF_CHANNELS:
            raise KeyError(f"not a difference channel: {channel!r}")
        return self.diff[channel]

    def summary(self) -> dict[str, float]:
        """Flat ``d_<channel>_<metric>`` numbers -- what the regression tests
        and the benchmark JSON quote."""
        out: dict[str, float] = {"t_span_s": float(self.t_s[-1] - self.t_s[0])}
        for channel in DIFF_CHANNELS:
            m = self.metrics[channel]
            unit = {"altitude_m": "m", "speed_m_s": "m_s", "fpa_deg": "deg",
                    "mass_kg": "kg"}[channel]
            for metric, value in m.items():
                out[f"d_{channel}_{metric}_{unit}"] = float(value)
        return out


def compare_ascent(ref, six, *, label: str = "6dof-minus-3dof") -> AscentDiff:
    """Difference the two runs on the common time grid (``six - ref``): both
    :meth:`ascent_channels` interpolated onto the reference's own time stamps
    within the shared time span. Accepts either run kind (anything with the
    raw ECEF sample columns)."""
    ch_ref = ascent_channels(ref)
    ch_six = ascent_channels(six)
    t_lo = max(float(ch_ref["t_s"][0]), float(ch_six["t_s"][0]))
    t_hi = min(float(ch_ref["t_s"][-1]), float(ch_six["t_s"][-1]))
    if t_hi <= t_lo:
        raise ValueError("runs share no time span")
    grid = ch_ref["t_s"][(ch_ref["t_s"] >= t_lo) & (ch_ref["t_s"] <= t_hi)]
    a = _interp_channels(ch_ref, grid)
    b = _interp_channels(ch_six, grid)
    diff = {key: b[key] - a[key] for key in DIFF_CHANNELS}
    metrics = {}
    for key in DIFF_CHANNELS:
        d = diff[key]
        metrics[key] = {"max_abs": float(np.max(np.abs(d))),
                        "rms": float(np.sqrt(np.mean(d * d))),
                        "terminal": float(d[-1])}
    return AscentDiff(label=label, t_s=grid, ref=a, six=b, diff=diff, metrics=metrics)


# --------------------------------------------------------------------------
# the scenario
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GravityTurnScenario:
    """Everything one rung of the ladder varies against: the pad state, the
    vehicle (with its aero), the attitude program, the shared physics and the
    integration plan. All five rungs consume the *same* object."""

    name: str
    state0: RigidBodyState
    vehicle: RigidBodyVehicle
    program: WaypointAttitudeProgram
    t_end_s: float
    dt_s: float = 0.02
    sample_every: int = 25
    physics: RigidBodyPhysics = field(default_factory=RigidBodyPhysics)
    throttle: float = 1.0
    dt_c_s: float = 0.05
    gains: TvcGains = field(default_factory=TvcGains)
    assumptions: tuple[str, ...] = ()

    def ideal_actuator(self) -> TvcActuator:
        """A servo with the same travel but no lag and no rate limit (tau a
        tenth of a millisecond, slew a megarad/s) -- the ladder's rung
        separator, not a claim about hardware."""
        return TvcActuator(limit_rad=self.vehicle.engine.gimbal_limit_rad,
                           max_slew_rate_rad_s=1.0e6, tau_s=1.0e-4)

    def real_actuator(self) -> TvcActuator:
        """F-04's default hydraulic axis (tau 0.10 s, slew 0.10 rad/s) at the
        vehicle's own travel limit."""
        return TvcActuator(limit_rad=self.vehicle.engine.gimbal_limit_rad)


def f05_scenario(*, t_end_s: float = 180.0, dt_s: float = 0.05,
                 sample_every: int = 20) -> GravityTurnScenario:
    """The packaged F-05 study: the spike-stage booster (10 t dry / 60 t
    propellant / 1.88 MN, the packaged ``spike_stage`` coefficient tables)
    off the pad at 28.5 N / 80.6 W on heading 090, flying the classic
    gravity-turn program -- vertical rise to ``t = 10 s``, then a linear
    pitch-over to 18 deg by ``t = 120 s`` (the tanks run dry at ``t = 99.85 s``,
    so the last of the ramp is coasted), held to ``t = 180 s``. Full throttle
    until burn-out; US76, J2 and Earth rotation on for both sides."""
    engine = RigidBodyEngine(thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0,
                             isp_sea_level_s=282.0, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=22.0,
                             gimbal_limit_rad=0.0873)
    vehicle = RigidBodyVehicle(name="spike_stage", dry_mass_kg=10_000.0,
                               propellant_mass_kg=60_000.0,
                               inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                               aero=load_packaged_aero("spike_stage"))
    state0 = RigidBodyState.from_launch(lat_geod_deg=28.5, lon_deg=-80.6,
                                        heading_deg=90.0, pitch_deg=90.0,
                                        roll_deg=0.0, mass_kg=vehicle.mass_wet_kg)
    program = WaypointAttitudeProgram(times_s=(0.0, 10.0, 120.0),
                                      heading_deg=(90.0, 90.0, 90.0),
                                      pitch_deg=(90.0, 90.0, 18.0),
                                      roll_deg=(0.0, 0.0, 0.0))
    return GravityTurnScenario(
        name="f05-gravity-turn",
        state0=state0, vehicle=vehicle, program=program, t_end_s=t_end_s,
        dt_s=dt_s, sample_every=sample_every,
        assumptions=(
            "Single-stage gravity-turn study of the F-02 reference plant and "
            "the F-04 tracker with the packaged illustrative spike_stage "
            "aero -- illustrative vehicle, not flight-certified.",
            "Both models share the identical vehicle object, physics "
            "switches (US76 + J2 + rotation), ambient-corrected engine law, "
            "attitude program and RK4 core; the pair differs only in the "
            "6-DOF content the attribution ladder isolates.",
            "Constant full throttle until propellant exhaustion, then coast; "
            "throttle scheduling is guidance (F-06/F-07), not this study.",
        ),
    )


# --------------------------------------------------------------------------
# the five-rung study
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class F05Study:
    """One completed five-rung study. ``runs`` holds every rung's run object
    keyed by :data:`RUNG_ORDER` (the 6-DOF rungs as
    :class:`~orbitopt.flightlab.sixdof.tvc.ControlledAscent`), ``headline``
    is the acceptance's 6dof_full-vs-3dof_aero :class:`AscentDiff`, and
    ``attribution`` maps each :data:`ATTRIBUTION_STEPS` label to the
    consecutive-rung :class:`AscentDiff` that isolates it."""

    scenario: GravityTurnScenario
    runs: Mapping[str, Any]
    headline: AscentDiff
    attribution: Mapping[str, AscentDiff]

    def summary(self) -> dict[str, Any]:
        """Plain-dict view: scenario metadata, the headline metrics, the
        attribution table and the tracker diagnostics of the headline
        ascent."""
        s = self.scenario
        full = self.runs["6dof_full"]
        aero_ref = self.runs["3dof_aero"]
        out: dict[str, Any] = {
            "scenario": {
                "name": s.name,
                "vehicle_kg": {"dry": s.vehicle.dry_mass_kg,
                               "wet": s.vehicle.mass_wet_kg},
                "thrust_vac_n": s.vehicle.engine.thrust_vacuum_n,
                "program_pitch_deg": list(s.program.pitch_deg),
                "program_times_s": list(s.program.times_s),
                "t_end_s": s.t_end_s, "dt_s": s.dt_s, "dt_c_s": s.dt_c_s,
                "sample_every": s.sample_every,
            },
            "headline": self.headline.summary(),
            "headline_max_q": {"q_pa": aero_ref.max_q.q_pa,
                               "t_s": aero_ref.max_q.t_s,
                               "altitude_m": aero_ref.max_q.altitude_m},
            "headline_6dof": {
                "max_q_pa": float(full.run.max_q.q_pa),
                "alpha_max_deg": float(full.run.alpha_max_deg),
                "att_err_max_deg": float(full.metrics.att_err_max_deg),
                "att_err_rms_deg": float(full.metrics.att_err_rms_deg),
            },
            "attribution": {name: diff.summary() for name, diff in self.attribution.items()},
            "tvc": {k: v for k, v in full.summary().items() if not isinstance(v, tuple)},
        }
        return out


def run_f05_study(scenario: GravityTurnScenario | None = None) -> F05Study:
    """Fly the ladder: five propagations of ``scenario`` (default
    :func:`f05_scenario`), then compare. ``headline`` = 6dof_full - 3dof_aero;
    ``attribution`` isolates each rung's contribution (module docstring).
    Real RK4 physics -- a study is a handful of seconds of CPU, all rungs."""
    scn = scenario or f05_scenario()
    veh = scn.vehicle
    kw = dict(dt_s=scn.dt_s, sample_every=scn.sample_every, physics=scn.physics,
              throttle=scn.throttle)
    runs: dict[str, Any] = {}
    runs["3dof_vacuum"] = propagate_point_mass(scn.state0, veh, scn.program, scn.t_end_s,
                                               use_aero=False, **kw)
    runs["3dof_aero"] = propagate_point_mass(scn.state0, veh, scn.program, scn.t_end_s,
                                             use_aero=True, **kw)
    no_aero = replace(veh, aero=None)
    closed = dict(dt_c_s=scn.dt_c_s, gains=scn.gains, **kw)
    runs["6dof_ideal"] = run_closed_loop_ascent(scn.state0, no_aero, scn.program,
                                                scn.t_end_s, actuator=scn.ideal_actuator(),
                                                **closed)
    runs["6dof_lag"] = run_closed_loop_ascent(scn.state0, no_aero, scn.program,
                                              scn.t_end_s, actuator=scn.real_actuator(),
                                              **closed)
    runs["6dof_full"] = run_closed_loop_ascent(scn.state0, veh, scn.program,
                                               scn.t_end_s, actuator=scn.real_actuator(),
                                               **closed)
    ladder = [runs[key].run if isinstance(runs[key], ControlledAscent) else runs[key]
              for key in RUNG_ORDER]
    attribution = {name: compare_ascent(ladder[lo], ladder[hi],
                                        label=f"{RUNG_ORDER[hi]} minus {RUNG_ORDER[lo]}")
                   for name, (hi, lo) in ATTRIBUTION_RUNG_PAIRS.items()}
    headline = compare_ascent(runs["3dof_aero"], runs["6dof_full"].run,
                              label="6dof_full minus 3dof_aero")
    return F05Study(scenario=scn, runs=runs, headline=headline, attribution=attribution)
