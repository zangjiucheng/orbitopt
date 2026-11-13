"""F-04 (task #229): TVC actuator model and a basic attitude tracker for the
native 6-DOF plant.

The reference plant (:mod:`orbitopt.flightlab.sixdof.rigidbody`) applies a
gimbal command directly (F-02's scope); the JSBSim backend has the lagged
servo chain inside its FCS XML, but the native plant had nothing between the
control law and ``RigidBodyEngine.force_body_n``. This module is that
something:

* :class:`TvcActuator` -- one hydraulic gimbal axis as a first-order lag
  (``tau_s``) with a slew-rate limit and a travel (deflection) limit. The
  model is the standard three-part actuator idealisation, discrete, with
  internal substepping so the lag is integrated at no coarser than a quarter
  time constant regardless of the control tick.
* :class:`WaypointAttitudeProgram` -- piecewise-linear heading/pitch/roll
  waypoints (the attitude-program idea of the FL-403 spike, in the native
  plant's own ``tec2b_from_attitude`` convention), with the 3-2-1 body-rate
  feed-forward that comes free from the waypoint slopes.
* :class:`AttitudeTvcLaw` -- a stateful :data:`ControlLaw` for
  :func:`~orbitopt.flightlab.sixdof.rigidbody.propagate`: a two-axis (pitch/
  yaw) PID in angular-acceleration space on the geodesic SO(3) attitude error
  (FL-403's :class:`~orbitopt.flightlab.sixdof.tracking.PidAttitudeController`
  lineage, not its JSBSim plumbing), updated at a fixed control tick with
  zero-order hold, converted to a gimbal deflection through the engine's
  moment arm and rated thrust, and then driven through the actuator. The
  plant sees the *post-actuator* deflection, so lag, slew and saturation are
  real in the dynamics, and the law records every tick for the diagnostics.
* :class:`TvcDiagnostics` / :func:`attitude_metrics` -- what the acceptance
  asks to record: max/commanded deflection, deflection margin, saturated and
  slew-limited seconds, and the closed-loop attitude error.

Sign conventions are the plant's (body x forward, y right, z down; positive
pitch gimbal ``dp`` -> ``+My`` nose-up, positive yaw gimbal ``dy`` -> ``-Mz``,
i.e. this plant's engine behind the CG yaws the nose *left* for a positive
yaw deflection -- the conversion here is derived from
:meth:`~orbitopt.flightlab.sixdof.rigidbody.RigidBodyEngine.force_body_n`,
so the tracker is correct by the plant's own geometry, whatever a docstring
elsewhere says).

Roll is not actuated (two gimbals, no RCS): the tracker steers pitch and yaw
only and leaves roll to aero damping. This is a *basic* tracker on purpose --
PID with rate damping and feed-forward, no gain scheduling beyond the rated
thrust, no LQR -- F-06/F-07 close the guidance loop for real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from orbitopt.flightlab import frames
from orbitopt.flightlab.sixdof.rigidbody import (
    ControlCommand,
    RigidBodyState,
    RigidBodyVehicle,
    SixDofRun,
    dcm_from_quaternion,
    propagate,
    tec2b_from_attitude,
)

__all__ = [
    "AttitudeTvcLaw",
    "ControlledAscent",
    "TvcActuator",
    "TvcDiagnostics",
    "TvcGains",
    "WaypointAttitudeProgram",
    "attitude_error_body",
    "attitude_metrics",
    "run_closed_loop_ascent",
    "so3_log",
]


# --------------------------------------------------------------------------
# SO(3) helpers
# --------------------------------------------------------------------------

def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Angle-axis rotation vector of an SO(3) matrix (exact matrix log, so the
    direction stays geodesically correct through tens-of-degrees errors, up to
    the antipode -- the FL-403 lesson, reimplemented here so this module does
    not import the JSBSim spike machinery)."""
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


# --------------------------------------------------------------------------
# actuator
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TvcActuator:
    """One TVC gimbal axis: first-order lag, slew-rate limit, travel limit.

    ``tau_s`` is the lag time constant, ``max_slew_rate_rad_s`` caps the
    angular rate of the *deflection* and ``limit_rad`` its travel. The command
    is saturated to ``+/-limit_rad`` first (the servo cannot ask the cylinder
    for more stroke than it has); the state then chases the saturated command
    through the lag at no more than the slew rate and is itself floored at the
    travel limit.

    :meth:`step` integrates one control tick of ``dt_s`` with internal
    substeps (at least four per lag time constant), so an explicit-Euler tick
    neither overshoots the lag nor under-resolves the lag/slew interaction
    even when the control tick is coarse against ``tau_s``.
    """

    limit_rad: float = 0.0873
    max_slew_rate_rad_s: float = 0.10
    tau_s: float = 0.10

    def __post_init__(self) -> None:
        if self.limit_rad <= 0.0:
            raise ValueError(f"limit_rad must be positive, got {self.limit_rad}")
        if self.max_slew_rate_rad_s <= 0.0:
            raise ValueError(f"max_slew_rate_rad_s must be positive, got "
                             f"{self.max_slew_rate_rad_s}")
        if self.tau_s <= 0.0:
            raise ValueError(f"tau_s must be positive, got {self.tau_s}")

    def saturate(self, cmd_rad: float) -> float:
        """The command as the actuator sees it: clamped into the travel."""
        return max(-self.limit_rad, min(float(cmd_rad), self.limit_rad))

    def step(self, delta_rad: float, cmd_rad: float, dt_s: float) -> tuple[float, bool]:
        """Advance the deflection state ``delta_rad`` one tick of ``dt_s``
        toward ``cmd_rad``; returns ``(new_delta, slew_limited)``.

        ``slew_limited`` is True when the unconstrained lag demanded more rate
        than ``max_slew_rate_rad_s`` at any substep (so the tick is recorded as
        slew-saturated even if the substep order hid the exact crossing).
        """
        if dt_s <= 0.0:
            return float(delta_rad), False
        cmd = self.saturate(cmd_rad)
        n_sub = max(1, int(math.ceil(dt_s / (self.tau_s / 4.0))))
        h = float(dt_s) / n_sub
        delta = float(delta_rad)
        slew_limited = False
        for _ in range(n_sub):
            rate = (cmd - delta) / self.tau_s
            if abs(rate) > self.max_slew_rate_rad_s:
                rate = math.copysign(self.max_slew_rate_rad_s, rate)
                slew_limited = True
            delta += rate * h
        return self.saturate(delta), slew_limited


# --------------------------------------------------------------------------
# attitude program
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointAttitudeProgram:
    """Piecewise-linear heading/pitch/roll waypoints (degrees, times sorted,
    constant before the first / after the last waypoint).

    The angles are the same ``(heading, pitch, roll)`` the plant's own
    :func:`~orbitopt.flightlab.sixdof.rigidbody.tec2b_from_attitude` takes, so
    a program waypoint and a :meth:`...RigidBodyState.from_launch` state agree
    by construction. :meth:`dcm_at` resolves the program into a commanded
    Tec2b at the *current* position (a heading is a direction, not a vector --
    re-resolving at the live site is what keeps a gravity-turn-style program
    Earth-relative).
    """

    times_s: tuple[float, ...]
    heading_deg: tuple[float, ...]
    pitch_deg: tuple[float, ...]
    roll_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.times_s)
        if n == 0 or not (len(self.heading_deg) == len(self.pitch_deg)
                          == len(self.roll_deg) == n):
            raise ValueError("the waypoint tuples must all be the same non-zero length")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError("times_s must be strictly increasing")

    @classmethod
    def vertical(cls, heading_deg: float = 90.0) -> WaypointAttitudeProgram:
        """Hold vertical: one waypoint (used with any launch heading -- at
        pitch 90 the heading is a roll convention, not a direction)."""
        return cls((0.0,), (float(heading_deg),), (90.0,), (0.0,))

    def angles_at(self, t_s: float) -> tuple[float, float, float, float, float, float]:
        """``(heading, pitch, roll, heading_dot, pitch_dot, roll_dot)`` [deg,
        deg/s] at ``t_s`` (piecewise linear, zero slopes off the endpoints)."""
        ts = self.times_s
        t = float(t_s)
        if len(ts) == 1 or t <= ts[0] or t >= ts[-1]:
            if len(ts) == 1 or t <= ts[0]:
                i, frac, slope = 0, 0.0, 0.0
            else:
                i, frac, slope = len(ts) - 2, 1.0, 0.0
        else:
            i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
            span = ts[i + 1] - ts[i]
            frac = (t - ts[i]) / span
            slope = 1.0 / span
        out = []
        slopes = []
        for values in (self.heading_deg, self.pitch_deg, self.roll_deg):
            a, b = float(values[i]), float(values[i + 1]) if len(ts) > 1 else float(values[i])
            out.append(a + (b - a) * frac)
            slopes.append((b - a) * slope)
        return (*out, *slopes)  # type: ignore[return-value]

    def dcm_at(self, t_s: float, lat_geod_deg: float, lon_deg: float) -> np.ndarray:
        """The commanded ECEF->body DCM at ``(t_s, lat, lon)``."""
        h, p, r = self.angles_at(t_s)[:3]
        return tec2b_from_attitude(lat_geod_deg, lon_deg, heading_deg=h, pitch_deg=p, roll_deg=r)

    def omega_ff_body(self, t_s: float) -> np.ndarray:
        """Body-rate feed-forward [rad/s] from the waypoint slopes under the
        3-2-1 kinematics at the interpolated angles: ``p = psidot_r sin(theta)``,
        ``q = thetadot cos(phi) + psidot_r cos(theta) sin(phi)``,
        ``r = -thetadot sin(phi) + psidot_r cos(theta) cos(phi)`` (roll/
        pitch/heading dot from the segment slope)."""
        h_d, p_d, r_d, h_dd, p_dd, r_dd = self.angles_at(t_s)
        psi, theta, phi = (math.radians(v) for v in (h_d, p_d, r_d))
        pd, qd, rd = (math.radians(v) for v in (h_dd, p_dd, r_dd))
        return np.array([rd - pd * math.sin(theta),
                         qd * math.cos(phi) + pd * math.cos(theta) * math.sin(phi),
                         -qd * math.sin(phi) + pd * math.cos(theta) * math.cos(phi)])


def attitude_error_body(state: RigidBodyState, program: WaypointAttitudeProgram) -> np.ndarray:
    """Angle-axis attitude error [rad] in the *actual* body frame: ``R_act @
    log(R_cmd^T R_act)`` (FL-403's convention -- ``e_y`` positive = nose must
    rotate up, ``e_z`` positive = nose must rotate right; near-vertical the
    azimuth is a roll convention, which the composition absorbs)."""
    lat_deg, lon_deg, _ = frames.ecef_to_geodetic(np.asarray(state.position_m, dtype=float))
    r_cmd = program.dcm_at(state.t_s, float(lat_deg), float(lon_deg))
    r_act = dcm_from_quaternion(state.quaternion)
    return r_act @ so3_log(r_cmd.T @ r_act)


# --------------------------------------------------------------------------
# controller gains and the closed-loop law
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TvcGains:
    """Two-channel PID in angular-acceleration space (FL-403's shape):
    ``alpha = kp*e + ki*I - kd*(omega_axis - omega_ff_axis)`` [rad/s^2],
    shared by pitch and yaw. ``i_max_alpha`` clamps the integral term
    itself (rad/s^2 equivalent)."""

    kp: float = 3.0
    ki: float = 0.02
    kd: float = 4.0
    i_max_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.kp < 0.0 or self.ki < 0.0 or self.kd < 0.0:
            raise ValueError("gains must be non-negative")
        if self.i_max_alpha <= 0.0:
            raise ValueError("i_max_alpha must be positive")


_HISTORY_KEYS = ("t_s", "dt_s", "err_pitch_rad", "err_yaw_rad",
                 "alpha_pitch_rad_s2", "alpha_yaw_rad_s2",
                 "cmd_pitch_rad", "cmd_yaw_rad",
                 "delta_pitch_rad", "delta_yaw_rad",
                 "pitch_saturated", "yaw_saturated",
                 "pitch_slew_limited", "yaw_slew_limited")


@dataclass
class AttitudeTvcLaw:
    """Stateful :data:`ControlLaw`: PID attitude tracker + :class:`TvcActuator`
    on each of the two gimbals, ticking at ``dt_c_s`` with zero-order hold.

    Feed the plant what the *servo delivers* (``ControlCommand.gimbal_*_rad``
    carries the post-actuator deflection, which is what the plant's own
    ``RigidBodyEngine`` clamp will re-clamp to its identical travel -- the
    column ``gimbal_pitch_cmd_rad`` in the recorded run is therefore the
    deflection the engine felt, the FL-403 lesson again: watch the servo, not
    the wish). The pre-saturation command and every diagnostic flag are
    recorded per tick in :meth:`history` / :attr:`diagnostics`.

    The tick fires at the first evaluation at or after ``next tick``; RK4
    evaluates the law four times per plant step, and a tick fired at a stage
    time uses that stage's state estimate -- a ZOH digital controller does
    not know better, and neither does this one. The throttle is fixed at
    ``throttle`` (guidance is F-06/F-07); after burnout (mass at dry) the
    tracker trims to neutral and freezes its integrators, because a shut
    engine steers nothing.
    """

    vehicle: RigidBodyVehicle
    program: WaypointAttitudeProgram
    actuator: TvcActuator = field(default_factory=TvcActuator)
    gains: TvcGains = field(default_factory=TvcGains)
    throttle: float = 1.0
    dt_c_s: float = 0.02

    def __post_init__(self) -> None:
        if self.dt_c_s <= 0.0:
            raise ValueError(f"dt_c_s must be positive, got {self.dt_c_s}")
        if self.vehicle.engine.mount_x_cg_m <= 0.0:
            raise ValueError("the tracker needs an engine with a moment arm behind the CG")
        if self.actuator.limit_rad > self.vehicle.engine.gimbal_limit_rad + 1e-12:
            raise ValueError(f"actuator travel {self.actuator.limit_rad} rad exceeds the "
                             f"engine gimbal limit {self.vehicle.engine.gimbal_limit_rad} rad "
                             f"(the plant would silently clamp what the tracker commands)")
        self.reset()

    def reset(self) -> None:
        """Clear the tick clock, actuator states, integrators and history."""
        self._primed = False
        self._t_last = 0.0
        self._next_tick = 0.0
        self._delta = [0.0, 0.0]
        self._integ = [0.0, 0.0]
        self._rows: list[dict[str, float]] = []

    # -- internals ---------------------------------------------------------
    @property
    def _rated_thrust_n(self) -> float:
        """Schedule thrust: the rated *vacuum* thrust at the law's throttle
        setting (ambient-corrected thrust is a few percent lower at the pad --
        a constant schedule on purpose; gain scheduling on live thrust is the
        JSBSim FCS's job, not this tracker's)."""
        engine = self.vehicle.engine
        return engine.throttle(self.throttle) * engine.count * engine.thrust_vacuum_n

    def _tick(self, t_s: float, state: RigidBodyState, dt_s: float) -> None:
        dry = self.vehicle.dry_mass_kg
        engine_off = self.throttle <= 0.0 or state.mass_kg <= dry
        if engine_off:
            a_p = a_y = 0.0
            e_p = e_y = 0.0
        else:
            e = attitude_error_body(state, self.program)
            ff = self.program.omega_ff_body(t_s)
            omega = np.asarray(state.omega_rad_s, dtype=float)
            e_p, e_y = float(e[1]), float(e[2])
            e_p, e_y = float(e[1]), float(e[2])
            i_cap = self.gains.i_max_alpha / max(self.gains.ki, 1e-12)
            a_p = (self.gains.kp * e_p + self.gains.ki * self._integ[0]
                   - self.gains.kd * (float(omega[1]) - float(ff[1])))
            a_y = (self.gains.kp * e_y + self.gains.ki * self._integ[1]
                   - self.gains.kd * (float(omega[2]) - float(ff[2])))
            self._integ[0] = max(-i_cap, min(i_cap, self._integ[0] + e_p * dt_s))
            self._integ[1] = max(-i_cap, min(i_cap, self._integ[1] + e_y * dt_s))
        # Angular acceleration -> gimbal deflection through the plant's own
        # moment geometry (arm * T * sin(delta), pitch +, yaw -): see the
        # module docstring. asin keeps the command inside the small-angle
        # inversion at large demand instead of pretending the moment is linear.
        authority = self._rated_thrust_n * self.vehicle.engine.mount_x_cg_m
        i_pitch, i_yaw = self.vehicle.inertia_kg_m2[1], self.vehicle.inertia_kg_m2[2]
        s_p = max(-1.0, min(1.0, a_p * i_pitch / authority))
        s_y = max(-1.0, min(1.0, -a_y * i_yaw / authority))
        cmd_p = math.asin(s_p)
        cmd_y = math.asin(s_y)
        limit = self.actuator.limit_rad
        sat_p, sat_y = abs(cmd_p) > limit, abs(cmd_y) > limit
        cmd_p = max(-limit, min(limit, cmd_p))
        cmd_y = max(-limit, min(limit, cmd_y))
        d_p, slew_p = self.actuator.step(self._delta[0], cmd_p, dt_s)
        d_y, slew_y = self.actuator.step(self._delta[1], cmd_y, dt_s)
        self._delta = [d_p, d_y]
        self._rows.append({"t_s": float(t_s), "dt_s": float(dt_s),
                           "err_pitch_rad": e_p, "err_yaw_rad": e_y,
                           "alpha_pitch_rad_s2": a_p, "alpha_yaw_rad_s2": a_y,
                           "cmd_pitch_rad": cmd_p, "cmd_yaw_rad": cmd_y,
                           "delta_pitch_rad": d_p, "delta_yaw_rad": d_y,
                           "pitch_saturated": 1.0 if sat_p else 0.0,
                           "yaw_saturated": 1.0 if sat_y else 0.0,
                           "pitch_slew_limited": 1.0 if slew_p else 0.0,
                           "yaw_slew_limited": 1.0 if slew_y else 0.0})

    # -- ControlLaw --------------------------------------------------------
    def __call__(self, t_s: float, state: RigidBodyState) -> ControlCommand:
        if not self._primed:
            self._primed = True
            self._t_last = float(t_s)
            self._next_tick = float(t_s) + self.dt_c_s
            self._rows.append({**{k: 0.0 for k in _HISTORY_KEYS}, "t_s": float(t_s)})
        elif float(t_s) >= self._next_tick - 1e-9:
            dt_s = float(t_s) - self._t_last
            self._t_last = float(t_s)
            self._next_tick = float(t_s) + self.dt_c_s
            if dt_s > 0.0:
                self._tick(float(t_s), state, dt_s)
        return ControlCommand(throttle=self.throttle,
                              gimbal_pitch_rad=self._delta[0],
                              gimbal_yaw_rad=self._delta[1])

    # -- output ------------------------------------------------------------
    def history(self) -> dict[str, np.ndarray]:
        """Per-tick columns, one row per recorded update (the row at ``t0`` is
        the prime: zeros). Keys are :data:`_HISTORY_KEYS`; the flags are 0.0/1.0
        floats so the whole history is one float array per column."""
        return {k: np.array([row[k] for row in self._rows], dtype=float) for k in _HISTORY_KEYS}

    @property
    def diagnostics(self) -> TvcDiagnostics:
        """The F-04 acceptance numbers as one frozen record."""
        return TvcDiagnostics.from_history(self.history(), actuator=self.actuator)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TvcDiagnostics:
    """Control-margin summary of one closed-loop run (units in the names; the
    *applied* deflection is what the actuator delivered, the *command* is the
    pre-saturation PID output -- their difference is the saturation story)."""

    n_ticks: int
    burn_s: float
    limit_rad: float
    pitch_command_peak_deg: float
    pitch_applied_peak_deg: float
    pitch_deflection_margin: float
    pitch_saturated_s: float
    pitch_slew_limited_s: float
    yaw_command_peak_deg: float
    yaw_applied_peak_deg: float
    yaw_deflection_margin: float
    yaw_saturated_s: float
    yaw_slew_limited_s: float

    @staticmethod
    def from_history(history: Mapping[str, np.ndarray], *,
                     actuator: TvcActuator) -> TvcDiagnostics:
        n = int(history["t_s"].size)
        if n == 0:
            raise ValueError("empty actuator/ticker history")
        dt = history["dt_s"]
        lim_deg = math.degrees(actuator.limit_rad)

        def axis(cmd_key: str, delta_key: str, sat_key: str, slew_key: str) -> tuple[float, ...]:
            cmd_peak = math.degrees(float(np.max(np.abs(history[cmd_key]))))
            app_peak = math.degrees(float(np.max(np.abs(history[delta_key]))))
            sat_s = float(np.sum(dt[history[sat_key] > 0.5]))
            slew_s = float(np.sum(dt[history[slew_key] > 0.5]))
            margin = 1.0 - min(1.0, app_peak / lim_deg)
            return (cmd_peak, app_peak, margin, sat_s, slew_s)

        p = axis("cmd_pitch_rad", "delta_pitch_rad", "pitch_saturated", "pitch_slew_limited")
        y = axis("cmd_yaw_rad", "delta_yaw_rad", "yaw_saturated", "yaw_slew_limited")
        burn = float(history["t_s"][-1] - history["t_s"][0])
        return TvcDiagnostics(n_ticks=n - 1, burn_s=burn, limit_rad=actuator.limit_rad,
                              pitch_command_peak_deg=p[0], pitch_applied_peak_deg=p[1],
                              pitch_deflection_margin=p[2], pitch_saturated_s=p[3],
                              pitch_slew_limited_s=p[4],
                              yaw_command_peak_deg=y[0], yaw_applied_peak_deg=y[1],
                              yaw_deflection_margin=y[2], yaw_saturated_s=y[3],
                              yaw_slew_limited_s=y[4])


@dataclass(frozen=True)
class AttitudeMetrics:
    """Closed-loop tracking quality read off the *recorded run* (not the tick
    history): the geodesic angle between flown attitude and program."""

    att_err_max_deg: float
    att_err_rms_deg: float
    alpha_max_deg: float
    rates_max_dps: tuple[float, float, float]


def attitude_metrics(run: SixDofRun, program: WaypointAttitudeProgram) -> AttitudeMetrics:
    """Per-sample geodesic attitude error against the program, from the run's
    quaternion columns (the actual flown attitude -- the same measurement a
    reader would make from the track, independent of the controller's own
    tick bookkeeping)."""
    errs: list[float] = []
    s = run.samples
    for i in range(s["t_s"].size):
        lat_deg, lon_deg, _ = frames.ecef_to_geodetic(
            np.array([s["x_m"][i], s["y_m"][i], s["z_m"][i]]))
        r_cmd = program.dcm_at(float(s["t_s"][i]), float(lat_deg), float(lon_deg))
        r_act = dcm_from_quaternion((s["q_w"][i], s["q_x"][i], s["q_y"][i], s["q_z"][i]))
        angle = so3_log(r_cmd.T @ r_act)
        errs.append(math.degrees(float(np.linalg.norm(angle))))
    errs_arr = np.asarray(errs)
    rates = [math.degrees(float(np.max(np.abs(s[c]))))
             for c in ("p_rad_s", "q_rad_s", "r_rad_s")]
    return AttitudeMetrics(att_err_max_deg=float(np.max(errs_arr)),
                           att_err_rms_deg=float(np.sqrt(np.mean(errs_arr ** 2))),
                           alpha_max_deg=run.alpha_max_deg,
                           rates_max_dps=(rates[0], rates[1], rates[2]))


@dataclass(frozen=True)
class ControlledAscent:
    """A closed-loop ascent: the plant's :class:`SixDofRun` (whose
    ``gimbal_*_cmd_rad`` columns carry the applied deflection), the tracker's
    :class:`TvcDiagnostics` and the :class:`AttitudeMetrics`."""

    run: SixDofRun
    diagnostics: TvcDiagnostics
    metrics: AttitudeMetrics

    def summary(self) -> dict[str, float | tuple[float, float, float]]:
        """Everything the F-04 acceptance asks to record, in one plain dict
        (deflection margin, max deflection, saturation/slew seconds, attitude
        error) -- suitable for a run-directory JSON sidecar."""
        d, m = self.diagnostics, self.metrics
        return {"att_err_max_deg": m.att_err_max_deg, "att_err_rms_deg": m.att_err_rms_deg,
                "alpha_max_deg": m.alpha_max_deg,
                "rates_max_dps": m.rates_max_dps,
                "pitch_command_peak_deg": d.pitch_command_peak_deg,
                "pitch_applied_peak_deg": d.pitch_applied_peak_deg,
                "pitch_deflection_margin": d.pitch_deflection_margin,
                "pitch_saturated_s": d.pitch_saturated_s,
                "pitch_slew_limited_s": d.pitch_slew_limited_s,
                "yaw_command_peak_deg": d.yaw_command_peak_deg,
                "yaw_applied_peak_deg": d.yaw_applied_peak_deg,
                "yaw_deflection_margin": d.yaw_deflection_margin,
                "yaw_saturated_s": d.yaw_saturated_s,
                "yaw_slew_limited_s": d.yaw_slew_limited_s,
                "tvc_limit_deg": math.degrees(d.limit_rad),
                "n_ticks": float(d.n_ticks), "burn_s": d.burn_s}


def run_closed_loop_ascent(state: RigidBodyState, vehicle: RigidBodyVehicle,
                           program: WaypointAttitudeProgram, t_end_s: float,
                           *, dt_s: float = 0.02, sample_every: int = 1,
                           actuator: TvcActuator | None = None,
                           gains: TvcGains | None = None,
                           throttle: float = 1.0, dt_c_s: float = 0.02,
                           physics=None, epoch_utc=None) -> ControlledAscent:
    """Fly the native plant closed-loop on ``program`` with the F-04 tracker +
    actuator (a thin :func:`~orbitopt.flightlab.sixdof.rigidbody.propagate`
    wrapper that builds the law, flies it and reads the diagnostics)."""
    law = AttitudeTvcLaw(vehicle=vehicle, program=program,
                         actuator=actuator or TvcActuator(), gains=gains or TvcGains(),
                         throttle=throttle, dt_c_s=dt_c_s)
    run = propagate(state, vehicle, law, t_end_s, dt_s=dt_s, sample_every=sample_every,
                    physics=physics, epoch_utc=epoch_utc)
    return ControlledAscent(run=run, diagnostics=law.diagnostics,
                            metrics=attitude_metrics(run, program))
