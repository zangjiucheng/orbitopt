"""JSBSim 6-DOF attitude-tracking spike (FL-403, task #208).

Closes the loop of the E4 spike: a simple PID attitude controller on the
spike stage's TVC gimbals (FL-401, #206) tracks the 3-DOF attitude program
of an OpenTsiolkovsky ascent run (FL-004 adapter lineage) for stage-1 to
MECO, and measures what the 6-DOF flight did relative to the 3-DOF
reference:

* attitude error (max/RMS angle between body attitude and program),
* body rates, gimbal saturation time,
* trajectory deviation vs the 3-DOF run at MECO (altitude / velocity /
  flight-path angle),
* numerical stability at altitude (no NaN, dt-halving convergence).

The 3-DOF reference is a recorded OT run (fixture
``tests/fixtures/flightlab/opentsiolkovsky/fl403_spike/ot_out/``,
regenerable with ``scripts/fl403_record_reference.py``) flown with the
vehicle file mirrored from the JSBSim spike stage: same masses, Isp,
mdot, 100 s burn, token nozzle area, zero aerodynamics on both sides.

Conventions match :mod:`orbitopt.flightlab.sixdof.jsbsim_driver` (ECEF
states, scalar-first quaternion with ``R(q) = Tec2b``, body x forward /
y right / z down as JSBSim's Euler sequence leaves them, positive pitch
gimbal = nose-up moment, positive yaw gimbal = nose-right moment).

Controller
----------
:class:`PidAttitudeController` drives the two gimbals from the SO(3)
geodesic attitude error :func:`attitude_error_body` (components in the
actual body frame): ``a = kp*e + ki*I - kd*(omega - omega_ff)`` [rad/s^2]
with the pitch channel's velocity feed-forward ``omega_ff_y = el_ff`` (the
program's elevation slew; body pitch rate changes elevation directly, so
the 1.71 deg/s pitch-over is well within TVC authority). Whole-burn
tracking stays sub-degree (peak ~0.5 deg at the t=8 s pitch-over kick). An
earlier revision reported a ~24 deg early "program-infeasibility transient";
it was a transposed :func:`dcm_from_quat` plus a launch IC tuned against
it, not a property of the program (see the ADR).
The desired body angular acceleration is converted to a gimbal command
with the *instruments of the moment*: ``cmd = I * alpha / (T * arm)``,
using the live thrust, inertia and CG the model reports -- a gain schedule
that cancels the plant gain variation over the burn (thrust constant,
mass/CG/inertia changing). The integrator is clamped; the actuator's own
lag/slew/travel limits sit between command and physics (the saturation
metric watches the servo output, not the command).

Altitude comparison: JSBSim's ``h-sl`` is geoid-referenced while OT
reports ellipsoidal altitude, so deviations use the ECEF radius *climb
above the launch point* (``|r| - |r(0)|``) on both sides -- that cancels
the geoid undulation and needs no geopotential model.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, Vehicle
from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions, SixDofSim

#: Structural x of the spike nozzle in the XML (aft of the nose), meters.
THRUSTER_X_M = 42.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
G0_M_S2 = 9.80665

#: Below this fraction of rated thrust the gain scheduler reports "no
#: authority" and the controller commands neutral (post-MECO coast).
MIN_THRUST_FRACTION = 0.01
#: Rated thrust of the spike engine (Isp * g0 * mdot), N.
RATED_THRUST_N = 320.0 * G0_M_S2 * 600.0


# --------------------------------------------------------------------------
# quaternion / frame helpers (scalar-first quaternions as 4-tuples)
# --------------------------------------------------------------------------

def dcm_from_quat(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rows = body x/y/z axes expressed in ECEF, from the FL-401 convention
    ``R(q) = Tec2b`` (so this is just ``R(q)`` reconstructed)."""
    w, x, y, z = q
    # Exact inverse of jsbsim_driver.quat_from_dcm (standard scalar-first
    # R(q)); the transpose here would silently swap ECEF->body for
    # body->ECEF and mis-steer the controller.
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def geodetic_from_ecef(x: float, y: float, z: float) -> tuple[float, float]:
    """(geodetic latitude, longitude) [rad] of an ECEF point: Bowring's
    closed form plus one fixed-point refinement (< 1e-12 rad anywhere)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:
        return (math.copysign(math.pi / 2.0, z), lon)
    e2 = WGS84_F * (2.0 - WGS84_F)
    b = WGS84_A * (1.0 - WGS84_F)
    theta = math.atan2(z * WGS84_A, p * b)
    lat = math.atan2(z + (WGS84_A - b) * (math.sin(theta) ** 3),
                     p - (WGS84_A - b) * (math.cos(theta) ** 3))
    sin_lat = math.sin(lat)
    n_phi = WGS84_A / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    lat = math.atan2(z + e2 * n_phi * sin_lat, p)
    return (lat, lon)


def dcm_ecef_to_ned(lat: float, lon: float) -> np.ndarray:
    """Rows north/east/down: ``v_ned = T_e2n @ v_ecef`` (standard geodetic NED)."""
    s_la, c_la, s_lo, c_lo = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([
        [-s_la * c_lo, -s_la * s_lo, c_la],
        [-s_lo, c_lo, 0.0],
        [-c_la * c_lo, -c_la * s_lo, -s_la],
    ])


def dcm_ned_to_body(elevation_rad: float, azimuth_rad: float) -> np.ndarray:
    """NED->body rows (x forward, y right, z down) for an (az, el) program
    point under the FL-401/JSBSim Euler convention: the x-axis is
    ``[cos el cos az, cos el sin az, -sin el]`` in NED, y is horizontal
    right, z = x cross y."""
    ce, se, ca, sa = (math.cos(elevation_rad), math.sin(elevation_rad),
                      math.cos(azimuth_rad), math.sin(azimuth_rad))
    x = np.array([ce * ca, ce * sa, -se])
    y = np.array([-sa, ca, 0.0])
    z = np.cross(x, y)
    return np.vstack((x, y, z))


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Exact matrix logarithm of an SO(3) matrix: the angle-axis rotation
    vector. Unlike the half-antisymmetric (vee) approximation, the direction
    stays geodesically correct for large angles, up to the antipode."""
    antisym = np.array([rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1]])
    c = 0.5 * (rotation[0, 0] + rotation[1, 1] + rotation[2, 2] - 1.0)
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-9:
        return 0.5 * antisym
    s = math.sin(angle)
    if s < 1e-6:
        i = int(np.argmax(np.diag(rotation)))
        axis = rotation[:, i] + np.eye(3)[i]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return 0.5 * antisym * (angle / max(float(np.linalg.norm(antisym)), 1e-9))
        axis = axis / norm
        if float(axis @ antisym) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * s)) * antisym


def body_axis_ned(sample: Mapping[str, float]) -> np.ndarray:
    """Body x-axis (nose) in NED components, from the sampled quaternion and
    geodetic position."""
    lat, lon = geodetic_from_ecef(sample["x_m"], sample["y_m"], sample["z_m"])
    t_e2b = dcm_from_quat((sample["q_w"], sample["q_x"], sample["q_y"], sample["q_z"]))
    return dcm_ecef_to_ned(lat, lon) @ t_e2b[0]


def chart_attitude(sample: Mapping[str, float]) -> tuple[float, float]:
    """(azimuth, elevation) [rad] of the body x-axis: elevation is always
    well defined; azimuth is undefined at exactly vertical (returns 0 there,
    harmless because the program holds a constant azimuth until then)."""
    n, e, d = body_axis_ned(sample)
    el = math.asin(max(-1.0, min(1.0, -d)))
    if math.hypot(n, e) < 1e-9:
        return (0.0, el)
    return (math.atan2(e, n) % (2.0 * math.pi), el)


def flight_path_angle(sample: Mapping[str, float]) -> float:
    """Velocity flight-path angle [rad] from ECEF state: asin(v . rhat / |v|).
    ECEF (rotating-frame) velocity, comparable with OT's ``speed_earth``."""
    r = np.array((sample["x_m"], sample["y_m"], sample["z_m"]))
    v = np.array((sample["vx_m_s"], sample["vy_m_s"], sample["vz_m_s"]))
    speed = float(np.linalg.norm(v))
    if speed == 0.0:
        return 0.0
    return math.asin(max(-1.0, min(1.0, float(np.dot(v, r)) / speed / float(np.linalg.norm(r)))))


# --------------------------------------------------------------------------
# attitude program (piecewise-linear az/el waypoints from the vehicle file)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AttitudeProgram:
    """Piecewise-linear azimuth/elevation program (waypoints sorted by
    time, constant before the first / after the last)."""

    times: tuple[float, ...]
    azimuth_deg: tuple[float, ...]
    elevation_deg: tuple[float, ...]

    @staticmethod
    def from_points(points: Sequence[AttitudePoint]) -> "AttitudeProgram":
        pts = sorted(points, key=lambda p: p.t_s)
        return AttitudeProgram(tuple(p.t_s for p in pts),
                               tuple(p.azimuth_deg for p in pts),
                               tuple(p.elevation_deg for p in pts))

    @staticmethod
    def from_vehicle(vehicle: Vehicle) -> "AttitudeProgram":
        if not vehicle.attitude_program:
            raise ValueError(f"vehicle {vehicle.name!r} has no attitude_program section")
        return AttitudeProgram.from_points(vehicle.attitude_program)

    def __call__(self, t: float) -> tuple[float, float, float, float]:
        """(azimuth_rad, elevation_rad, az_dot_rad_s, el_dot_rad_s) at t."""
        ts = self.times
        if len(ts) == 1:
            return (math.radians(self.azimuth_deg[0]),
                    math.radians(self.elevation_deg[0]), 0.0, 0.0)
        if t <= ts[0]:
            i = 0
        elif t >= ts[-1]:
            i = len(ts) - 2
        else:
            i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
        span = ts[i + 1] - ts[i]
        az0, el0 = self.azimuth_deg[i], self.elevation_deg[i]
        az1, el1 = self.azimuth_deg[i + 1], self.elevation_deg[i + 1]
        if t <= ts[0] or t >= ts[-1] or span <= 0.0:
            az_d = el_d = 0.0
            frac = 0.0 if t <= ts[0] else 1.0
        else:
            az_d = math.radians((az1 - az0) / span)
            el_d = math.radians((el1 - el0) / span)
            frac = (t - ts[i]) / span
        return (math.radians(az0 + (az1 - az0) * frac),
                math.radians(el0 + (el1 - el0) * frac), az_d, el_d)


# --------------------------------------------------------------------------
# 3-DOF reference (recorded OT trajectory CSV)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceTrajectory:
    """The OT ``trajectory.csv`` of the 3-DOF reference run, sampled to
    arrays. ``elevation_deg``/``azimuth_deg`` are OT's *attitude* columns;
    ``fpa_deg`` is the *velocity* flight-path angle recomputed from
    ``vel_down/speed_earth`` (with zero normal force the velocity path
    lags the attitude while the thrust vector tilts)."""

    t_s: np.ndarray
    altitude_m: np.ndarray
    speed_earth_m_s: np.ndarray
    fpa_deg: np.ndarray
    elevation_deg: np.ndarray
    mach: np.ndarray
    mass_kg: np.ndarray

    @staticmethod
    def from_csv(path) -> "ReferenceTrajectory":
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path}: no trajectory rows")
        t = np.array([float(r["time_s"]) for r in rows])
        keep = np.concatenate(([True], np.diff(t) > 0.0))  # OT may repeat event times
        g = lambda col: np.array([float(r[col]) for r in rows])[keep]  # noqa: E731
        alt, sp = g("altitude_m"), g("speed_earth_mps")
        vd = g("vel_down_mps")
        fpa = np.degrees(np.arcsin(np.clip(np.divide(-vd, sp, out=np.zeros_like(sp),
                                                     where=sp > 0.0), -1.0, 1.0)))
        return ReferenceTrajectory(t[keep], alt, sp, fpa, g("elevation_deg"),
                                   g("mach"), g("mass_kg"))

    def at(self, t: float) -> dict[str, float]:
        i = {"t": float(t)}
        for name in ("altitude_m", "speed_earth_m_s", "fpa_deg", "elevation_deg",
                     "mach", "mass_kg"):
            i[name] = float(np.interp(t, self.t_s, getattr(self, name)))
        return i


# --------------------------------------------------------------------------
# PID attitude controller
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PidGains:
    """PID in ACCELERATION space on chart errors: ``alpha = kp*e + ki*I -
    kd*(omega - omega_ff)`` [rad/s^2] per channel, converted to gimbal
    deflection by the live plant gain ``T*arm/I``. Integrals clamped so
    ``ki*|I| <= i_max_alpha``."""

    kp: float = 3.0
    ki: float = 0.01
    kd: float = 4.0
    i_max_alpha: float = 0.05  # rad/s^2 equivalent


@dataclass
class PidAttitudeController:
    """Two-axis (pitch/yaw) gimbal controller tracking an
    :class:`AttitudeProgram`. Step-invariant: call :meth:`command` once per
    FGSimulation step with that step's dt."""

    gains: PidGains = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gains is None:
            self.gains = PidGains()
        self._i_pitch = 0.0
        self._i_yaw = 0.0

    def reset(self) -> None:
        self._i_pitch = 0.0
        self._i_yaw = 0.0

    def command(self, sample: Mapping[str, float], program: AttitudeProgram,
                dt_s: float) -> tuple[float, float, float, float]:
        """(cmd_pitch, cmd_yaw, e_y, e_z) from the current sample; also
        advances the clamped integrators by dt_s. ``e_y/e_z`` are the pitch/
        yaw components of :func:`attitude_error_body` [rad]."""
        _, _, az_ff, el_ff = program(sample["t_s"])
        e = attitude_error_body(sample, program)
        e_p, e_y = float(e[1]), float(e[2])
        a_p = self.gains.kp * e_p + self.gains.ki * self._i_pitch \
            - self.gains.kd * (sample["q_rad_s"] - el_ff)
        a_y = self.gains.kp * e_y + self.gains.ki * self._i_yaw \
            - self.gains.kd * (sample["r_rad_s"] - az_ff)
        cap = self.gains.i_max_alpha / max(self.gains.ki, 1e-12)
        self._i_pitch = max(-cap, min(cap, self._i_pitch + e_p * dt_s))
        self._i_yaw = max(-cap, min(cap, self._i_yaw + e_y * dt_s))
        # Instrumented gain schedule: commanded angular acceleration ->
        # gimbal angle through the live thrust, arm and inertia.
        arm = THRUSTER_X_M - sample["cg_x_m"]
        authority = sample["thrust_n"] * arm
        if authority < MIN_THRUST_FRACTION * RATED_THRUST_N * arm:
            return (0.0, 0.0, e_p, e_y)
        return (a_p * sample["iyy_kg_m2"] / authority,
                a_y * sample["izz_kg_m2"] / authority, e_p, e_y)


# --------------------------------------------------------------------------
# the closed-loop spike
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackingMetrics:
    """All FL-403 report numbers (units in the field names)."""

    dt_s: float
    meco_t_s: float
    att_err_max_deg: float
    att_err_rms_deg: float
    p_rate_max_deg_s: float
    q_rate_max_deg_s: float
    r_rate_max_deg_s: float
    gimbal_sat_fraction_of_burn: float
    gimbal_max_deg: float
    alt_climb_6dof_m: float
    alt_3dof_m: float
    d_alt_m: float
    speed_6dof_m_s: float
    speed_3dof_m_s: float
    d_speed_m_s: float
    fpa_6dof_deg: float
    fpa_3dof_deg: float
    d_fpa_deg: float
    mach_6dof: float
    finite: bool


def attitude_error_body(sample: Mapping[str, float], program: AttitudeProgram,
                        lat: float | None = None,
                        lon: float | None = None) -> np.ndarray:
    """Angle-axis attitude error (components in the *actual* body frame) =
    the geodesic rotation carrying the actual attitude onto the commanded
    one: ``T_act @ log(T_cmd^T T_act)`` (exact SO(3) matrix logarithm, not
    the small-angle half-antisymmetric approximation, so the direction stays
    correct through the tens-of-degrees errors the ascent ramp produces).
    Rotation-matrix composition end to end -- no quaternion convention
    ambiguity, and no chart singularity at vertical (a commanded azimuth at
    el=90 is only a roll convention, which matches the FL-401 launch
    convention exactly).
    Components: ``e_y`` positive = nose must rotate up (pitch channel),
    ``e_z`` positive = nose must rotate right (yaw channel)."""
    if lat is None or lon is None:
        lat, lon = geodetic_from_ecef(sample["x_m"], sample["y_m"], sample["z_m"])
    az_c, el_c, _, _ = program(sample["t_s"])
    t_a = dcm_from_quat((sample["q_w"], sample["q_x"], sample["q_y"], sample["q_z"]))
    t_c = dcm_ned_to_body(el_c, az_c) @ dcm_ecef_to_ned(lat, lon)
    return t_a @ so3_log(t_c.T @ t_a)


def chart_error_deg(sample: Mapping[str, float], program: AttitudeProgram) -> float:
    """Total attitude tracking error [deg] in the program chart:
    sqrt(d_el^2 + (d_az*cos(el))^2) (azimuth contribution weighted by the
    horizontal factor, so a heading error near vertical costs nothing)."""
    az_c, el_c, _, _ = program(sample["t_s"])
    az_a, el_a = chart_attitude(sample)
    e_el = el_c - el_a
    e_az = ((az_c - az_a + math.pi) % (2.0 * math.pi) - math.pi) * math.cos(el_a)
    return math.degrees(math.hypot(e_el, e_az))


class TrackingResult:
    """Closed-loop run: the raw 6-DOF samples, the metrics, and helpers."""

    def __init__(self, sim: SixDofSim, program: AttitudeProgram,
                 reference: ReferenceTrajectory, metrics: TrackingMetrics,
                 r0_m: float):
        self.sim = sim
        self.program = program
        self.reference = reference
        self.metrics = metrics
        self.r0_m = r0_m

    @property
    def samples(self) -> list[dict[str, float]]:
        return self.sim.samples

    def alt_climb_m(self, s: Mapping[str, float]) -> float:
        """ECEF radius climb above the launch point [m] -- the geoid-free
        altitude comparable with the OT reference."""
        r = math.sqrt(s["x_m"] ** 2 + s["y_m"] ** 2 + s["z_m"] ** 2)
        return r - self.r0_m


def run_tracking_spike(reference: ReferenceTrajectory, program: AttitudeProgram,
                       meco_t_s: float, *, dt_s: float = 0.01,
                       burn_t_s: float | None = None,
                       gains: PidGains | None = None,
                       ic: InitialConditions | None = None,
                       sim: SixDofSim | None = None) -> TrackingResult:
    """Fly the JSBSim spike stage closed-loop on the program to ``meco_t_s``
    (+ 5 s coast), return the :class:`TrackingResult`. ``burn_t_s`` bounds
    the metrics window (default: ``meco_t_s``)."""
    if sim is None:
        sim = SixDofSim(dt_s)
    if ic is None:
        # (theta=90, psi=90, phi=-90) puts the body exactly on the program
        # chart's (el=90, az=90) frame, roll included: Tec2b gives body
        # x = up, y = south, z = east at (0, 0), so a nose-down pitch tips
        # the nose east (checked against Tec2b directly, not through
        # dcm_from_quat -- see test_launch_ic_matches_program_frame).
        ic = InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=0.0,
                               pitch_deg=90.0, heading_deg=90.0, roll_deg=-90.0)
    sim.initialize(ic)
    r0 = math.sqrt(sim.samples[0]["x_m"] ** 2 + sim.samples[0]["y_m"] ** 2
                   + sim.samples[0]["z_m"] ** 2)
    sim.ignite(1.0)
    ctrl = PidAttitudeController(gains=gains)
    t_end = meco_t_s + 5.0
    meco_mach: list[float] = []

    def control(t: float, s: SixDofSim) -> None:
        cmd_p, cmd_y, _, _ = ctrl.command(s.sample(), program, s.dt_s)
        s.set_gimbal(cmd_p, cmd_y)
        if abs(t + s.dt_s - meco_t_s) <= 0.5 * s.dt_s:
            meco_mach.append(float(s["velocities/mach"]))

    sim.run_for(t_end - sim.t_s, control)
    metrics = _metrics(sim, program, reference, meco_mach[0] if meco_mach else float("nan"),
                       meco_t_s, burn_t_s if burn_t_s is not None else meco_t_s, r0, dt_s)
    return TrackingResult(sim, program, reference, metrics, r0)


def _metrics(sim: SixDofSim, program: AttitudeProgram, reference: ReferenceTrajectory,
             meco_mach: float, meco_t_s: float, burn_t_s: float,
             r0_m: float, dt_s: float) -> TrackingMetrics:
    travel = 0.0873  # XML clipto
    err_max = 0.0
    err_sq = 0.0
    n = 0
    p_max = q_max = r_max = 0.0
    sat_steps = 0
    burn_steps = 0
    g_max = 0.0
    for s in sim.samples:
        t = s["t_s"]
        if t <= burn_t_s:
            burn_steps += 1
            if abs(s["gimbal_pitch_actual_rad"]) >= travel - 1e-6 \
                    or abs(s["gimbal_yaw_actual_rad"]) >= travel - 1e-6:
                sat_steps += 1
            g_max = max(g_max, abs(s["gimbal_pitch_actual_rad"]), abs(s["gimbal_yaw_actual_rad"]))
        if 0.5 <= t <= burn_t_s:  # after ignition settles
            e = chart_error_deg(s, program)
            err_max = max(err_max, e)
            err_sq += e * e
            n += 1
            p_max = max(p_max, abs(s["p_rad_s"]))
            q_max = max(q_max, abs(s["q_rad_s"]))
            r_max = max(r_max, abs(s["r_rad_s"]))
    # MECO state: nearest sample (grid-aligned, dt divides meco_t_s)
    meco = min(sim.samples, key=lambda s: abs(s["t_s"] - meco_t_s))
    climb = math.sqrt(meco["x_m"] ** 2 + meco["y_m"] ** 2 + meco["z_m"] ** 2) - r0_m
    ref = reference.at(meco_t_s)
    fpa = flight_path_angle(meco)
    speed = math.sqrt(meco["vx_m_s"] ** 2 + meco["vy_m_s"] ** 2 + meco["vz_m_s"] ** 2)
    finite = all(math.isfinite(v) for s in sim.samples for v in s.values())
    return TrackingMetrics(
        dt_s=dt_s, meco_t_s=meco_t_s,
        att_err_max_deg=err_max,
        att_err_rms_deg=math.sqrt(err_sq / n) if n else 0.0,
        p_rate_max_deg_s=math.degrees(p_max),
        q_rate_max_deg_s=math.degrees(q_max),
        r_rate_max_deg_s=math.degrees(r_max),
        gimbal_sat_fraction_of_burn=sat_steps / burn_steps if burn_steps else 0.0,
        gimbal_max_deg=math.degrees(g_max),
        alt_climb_6dof_m=climb, alt_3dof_m=ref["altitude_m"],
        d_alt_m=climb - ref["altitude_m"],
        speed_6dof_m_s=speed, speed_3dof_m_s=ref["speed_earth_m_s"],
        d_speed_m_s=speed - ref["speed_earth_m_s"],
        fpa_6dof_deg=math.degrees(fpa), fpa_3dof_deg=ref["fpa_deg"],
        d_fpa_deg=math.degrees(fpa) - ref["fpa_deg"],
        mach_6dof=meco_mach, finite=finite)
