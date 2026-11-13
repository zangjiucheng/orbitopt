"""Newtonian 6-DOF rigid-body plant for the Flight Lab aero model (F-02, task
#227): flies :class:`~orbitopt.flightlab.sixdof.aero.AeroModel` in ECEF and
reports the dynamic-pressure history and max-Q that task acceptance asks for.

Why this exists next to the JSBSim backend
------------------------------------------
:func:`orbitopt.flightlab.adapters.sixdof.run_ascent` (F-01) is Flight Lab's
formal 6-DOF engine, and it stays that. But its aerodynamics live inside the
JSBSim aircraft XML, and the ``jsbsim==1.3.3`` Python bindings offer no hook
to inject an external force/moment per step (``FGFDMExec`` exposes
``get_aerodynamics().get_forces()``, not ``SetForceAndMoment``), so an
in-house coefficient model cannot be pushed into it. What F-02 needs is the
mirror image -- aero authored here, flying a 6-DOF plant here -- so this module
is that plant: the same physics a point-mass propagator has (gravity + J2,
rotation, US76, ambient-corrected throttleable thrust, propellant bookkeeping)
plus attitude.

That makes it the *reference* 6-DOF plant: it is what the aero model's
acceptance tests integrate, what F-05's 3-DOF/6-DOF comparison can use for the
ascent-side aero questions without a JSBSim install, and where a real
coefficient table from F-05 lands first. It is not a JSBSim replacement (no
aero-XML ecosystem, no engine model, no FCS lag chain, constant inertia tensor)
and it is not wired into ``orbitopt flight simulate``.

Conventions (identical to :mod:`~orbitopt.flightlab.sixdof.jsbsim_driver`)
-------------------------------------------------------------------------
* ``position_m`` / ``velocity_m_s``: ECEF [m, m/s] -- Earth-fixed, so a vehicle
  on the pad has zero velocity; the co-rotating air therefore has zero
  velocity too, and the air-relative velocity is the ECEF velocity (wind is
  still not modelled anywhere in Flight Lab).
* ``quaternion``: scalar-first ``(q_w, q_x, q_y, q_z)`` with ``R(q) == Tec2b``,
  the ECEF->body direction-cosine matrix, so a vertical vehicle has no gimbal
  lock. Built from geodetic + heading/pitch/roll by
  :func:`tec2b_from_attitude`, which reproduces JSBSim's 3-2-1 Euler sequence.
* ``omega_rad_s``: body rates ``(p, q, r)`` relative to ECEF about the body
  axes x forward, y right, z down. Positive ``q`` is nose-up.
* Integrator: fixed-step RK4 on the 13-component state ``(r, v, q, omega, m)``,
  quaternion renormalised after every stage and every step.

Sign conventions of the forces are the aero module's (see
:mod:`~orbitopt.flightlab.sixdof.aero`): drag ``-x``, ``cn > 0`` lifts toward
``-z``, and a positive pitch gimbal command gives a nose-up ``+My`` moment from
an engine mounted behind the CG.

Not modelled: inertia variation with propellant (constant tensor; the mass
still burns down), structural/TPS loads, slosh, wind, aero-elasticity, TVC
actuator lag (the command is applied directly; F-01's JSBSim model has the
lagged servo), and thrust-vector coupling to the engine's own gimbal dynamics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from orbitopt.entry import J2_EARTH, MU_EARTH_M3_S2, OMEGA_EARTH_RAD_S, R_EARTH_M
from orbitopt.flightlab import frames
from orbitopt.flightlab.atmosphere import (
    G0_M_S2,
    P_SEA_LEVEL_PA,
    STANDARD_ATMOSPHERE,
    Atmosphere,
    dynamic_pressure_pa,
)
from orbitopt.flightlab.contract import BodyTrack, Event
from orbitopt.flightlab.sixdof.aero import (
    AeroModel,
    MaxQ,
    angle_of_attack_deg,
    max_q_of,
    sideslip_deg,
)
from orbitopt.flightlab.sixdof.jsbsim_driver import quat_from_dcm

__all__ = [
    "ControlCommand",
    "MaxQ",
    "RigidBodyEngine",
    "RigidBodyPhysics",
    "RigidBodyState",
    "RigidBodyVehicle",
    "SixDofRun",
    "RUN_COLUMNS",
    "constant_control",
    "dcm_from_quaternion",
    "propagate",
    "quaternion_from_tec2b",
    "tec2b_from_attitude",
]

_OMEGA_ECEF = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

#: Sample columns a :class:`SixDofRun` records, in order. The first eight are
#: the contract's required track columns; the rest are the aero/attitude
#: diagnostics F-02 has to report (and what ``to_body_track`` keeps as extras).
RUN_COLUMNS: tuple[str, ...] = (
    "t_s", "mass_kg", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
    "q_w", "q_x", "q_y", "q_z", "p_rad_s", "q_rad_s", "r_rad_s",
    "altitude_m", "mach", "alpha_deg", "beta_deg", "q_bar_pa",
    "throttle", "gimbal_pitch_cmd_rad", "gimbal_yaw_cmd_rad", "thrust_n",
    "aero_fx_N", "aero_fy_N", "aero_fz_N",
    "roll_moment_n_m", "pitch_moment_n_m", "yaw_moment_n_m",
)


# --------------------------------------------------------------------------
# attitude
# --------------------------------------------------------------------------

def dcm_from_quaternion(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix ``R(q)`` from a scalar-first quaternion (body-from-ECEF
    when ``q`` is a Tec2b quaternion, i.e. ``v_body = R v_ecef``)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("a zero quaternion is not a rotation")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def quaternion_from_tec2b(tec2b: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Scalar-first quaternion of an ECEF->body direction-cosine matrix --
    re-exported from the FL-401 driver so this plant and the JSBSim backend
    spell attitude the same way."""
    return quat_from_dcm(np.asarray(tec2b, dtype=float))


def _ned2ecef(lat_rad: float, lon_rad: float) -> np.ndarray:
    """The NED->ECEF rotation of a geodetic site: columns are the north, east
    and down unit vectors, so ``ned2ecef @ v_ned`` is the ECEF vector ("down"
    is the inward geodetic direction, the negative of the ellipsoid normal)."""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    return np.array([
        [-sl * co, -so, -cl * co],
        [-sl * so, co, -cl * so],
        [cl, 0.0, -sl],
    ])


def tec2b_from_attitude(lat_geod_deg: float, lon_deg: float, *, heading_deg: float = 90.0,
                       pitch_deg: float = 90.0, roll_deg: float = 0.0) -> np.ndarray:
    """The 3x3 ECEF->body DCM of a vehicle at geodetic ``lat_geod_deg`` /
    ``lon_deg`` pointing at ``heading_deg`` (true, clockwise from north),
    ``pitch_deg`` (up from horizontal) and ``roll_deg``.

    The NED->ECEF part uses the geodetic (ellipsoid-normal) vertical, and the
    body->NED part the 3-2-1 yaw-pitch-roll sequence, so this reproduces
    JSBSim's ``Tec2b`` for the same ``(psi, theta, phi)`` -- which is what lets
    an F-05 study start this plant and that backend from the same launch state.
    ``pitch_deg=90`` points the body +x axis along the local vertical.
    """
    ned2ecef = _ned2ecef(math.radians(float(lat_geod_deg)), math.radians(float(lon_deg)))
    psi, theta, phi = (math.radians(float(v)) for v in (heading_deg, pitch_deg, roll_deg))
    cp, sp = math.cos(psi), math.sin(psi)
    ct, st = math.cos(theta), math.sin(theta)
    cr, sr = math.cos(phi), math.sin(phi)
    b2ned = np.array([
        [cp * ct, cp * st * sr - sp * cr, cp * st * cr + sp * sr],
        [sp * ct, sp * st * sr + cp * cr, sp * st * cr - cp * sr],
        [-st, ct * sr, ct * cr],
    ])
    return b2ned.T @ ned2ecef.T


# --------------------------------------------------------------------------
# vehicle
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RigidBodyEngine:
    """One relightable engine group, in the same ambient-corrected form the 3-DOF
    recovery propagator uses (so the same vehicle means the same thrust and Isp
    on both sides of a 3-DOF/6-DOF comparison).

    ``mount_x_cg_m`` is how far *behind* the CG the gimbal pivot sits (a
    positive number; the moment arm of an off-CG gimbal is what turns a gimbal
    command into the ``+My``/``+Mz`` pitch/yaw moment).
    """

    thrust_vacuum_n: float
    isp_vacuum_s: float
    isp_sea_level_s: float
    count: int = 1
    nozzle_exit_area_m2: float = 0.0
    throttle_min: float = 0.4
    throttle_max: float = 1.0
    mount_x_cg_m: float = 0.0
    gimbal_limit_rad: float = 0.0873

    def __post_init__(self) -> None:
        if self.thrust_vacuum_n <= 0.0 or self.isp_vacuum_s <= 0.0 or self.isp_sea_level_s <= 0.0:
            raise ValueError("thrust_vacuum_n and both Isp values must be positive")
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if not 0.0 <= self.throttle_min <= self.throttle_max <= 1.0:
            raise ValueError(f"need 0 <= throttle_min <= throttle_max <= 1, got "
                             f"{self.throttle_min}/{self.throttle_max}")
        if self.mount_x_cg_m < 0.0:
            raise ValueError("mount_x_cg_m is a distance behind the CG, so it must be >= 0")

    def throttle(self, commanded: float) -> float:
        """The commanded throttle clamped into ``[throttle_min, throttle_max]``
        (or zero -- an engine group that is shut down is shut down, it does not
        creep at minimum throttle)."""
        level = float(commanded)
        if level <= 0.0:
            return 0.0
        return min(max(level, self.throttle_min), self.throttle_max)

    def isp_s(self, p_ambient_pa: float) -> float:
        """Effective Isp at ambient pressure: linear between sea level and
        vacuum (the recovery propagator's law)."""
        return self.isp_sea_level_s + (self.isp_vacuum_s - self.isp_sea_level_s) * (
            1.0 - min(float(p_ambient_pa), P_SEA_LEVEL_PA) / P_SEA_LEVEL_PA)

    def force_body_n(self, throttle: float, gimbal_pitch_rad: float, gimbal_yaw_rad: float,
                     p_ambient_pa: float) -> tuple[np.ndarray, float, np.ndarray]:
        """Body-axis thrust force [N], mass flow [kg/s] and the moment [N m] of
        an engine group at ``throttle`` with gimbal angles ``(pitch, yaw)``.

        Positive ``gimbal_pitch_rad`` tips the thrust vector toward ``+z``
        (down) at the mount, which is a nose-up ``+My`` moment; positive
        ``gimbal_yaw_rad`` is nose-right ``+Mz``. Both are clamped into
        ``+/-gimbal_limit_rad``, and the ambient-corrected force never goes
        negative: an overexpanded nozzle is thrust-limited, not a propeller.
        """
        level = self.throttle(throttle)
        if level <= 0.0:
            return np.zeros(3), 0.0, np.zeros(3)
        force = self.count * level * (self.thrust_vacuum_n
                                      - self.nozzle_exit_area_m2 * float(p_ambient_pa))
        if force <= 0.0:
            return np.zeros(3), 0.0, np.zeros(3)
        dp = max(-self.gimbal_limit_rad, min(float(gimbal_pitch_rad), self.gimbal_limit_rad))
        dy = max(-self.gimbal_limit_rad, min(float(gimbal_yaw_rad), self.gimbal_limit_rad))
        direction = np.array([math.cos(dp) * math.cos(dy), math.sin(dy), math.sin(dp)])
        moment = np.cross(np.array([-self.mount_x_cg_m, 0.0, 0.0]), force * direction)
        return force * direction, force / (self.isp_s(p_ambient_pa) * G0_M_S2), moment


@dataclass(frozen=True)
class RigidBodyVehicle:
    """Mass, inertia and the two force producers of the plant.

    ``inertia_kg_m2`` is the roll/pitch/yaw diagonal, taken constant (the
    propellant bookkeeping changes ``mass``, not the tensor -- F-03's
    propellant-state-dependent inertia is the fix, and a 6-DOF ascent burn
    changes ``Iyy`` by tens of percent, so treat pitch response at max-Q as
    indicative and F-03's as the number).
    """

    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    inertia_kg_m2: Sequence[float]
    engine: RigidBodyEngine
    aero: AeroModel | None = None

    def __post_init__(self) -> None:
        if self.dry_mass_kg <= 0.0:
            raise ValueError("dry_mass_kg must be positive")
        if self.propellant_mass_kg < 0.0:
            raise ValueError("propellant_mass_kg must be >= 0")
        inertia = tuple(float(v) for v in self.inertia_kg_m2)
        if len(inertia) != 3 or any(v <= 0.0 for v in inertia):
            raise ValueError(f"inertia_kg_m2 needs three positive diagonals, got "
                             f"{self.inertia_kg_m2}")
        object.__setattr__(self, "inertia_kg_m2", inertia)

    @property
    def mass_wet_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


@dataclass(frozen=True)
class RigidBodyPhysics:
    """What the plant switches on. Mirrors the recovery propagator's
    :class:`~orbitopt.flightlab.recovery.propagator.BoosterPhysics` names so a
    3-DOF/6-DOF pair reads the same."""

    atmosphere: bool = True
    rotation: bool = True
    j2: bool = True
    #: The air model; defaults to US76, and an
    #: :class:`~orbitopt.flightlab.atmosphere.ExponentialAtmosphere` is what a
    #: closed-form max-Q test wants.
    atmosphere_model: Atmosphere = field(default=STANDARD_ATMOSPHERE)
    density_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.density_scale <= 0.0:
            raise ValueError("density_scale must be positive")


@dataclass(frozen=True)
class RigidBodyState:
    """The plant's state at one instant: ECEF position/velocity, Tec2b
    quaternion, body rates and flown mass."""

    t_s: float
    position_m: Sequence[float]
    velocity_m_s: Sequence[float]
    quaternion: Sequence[float]
    omega_rad_s: Sequence[float]
    mass_kg: float

    def __post_init__(self) -> None:
        for what, value in (("position_m", self.position_m), ("velocity_m_s", self.velocity_m_s),
                            ("omega_rad_s", self.omega_rad_s)):
            if np.asarray(value, dtype=float).shape != (3,):
                raise ValueError(f"{what} must have 3 components")
        if np.asarray(self.quaternion, dtype=float).shape != (4,):
            raise ValueError("quaternion must have 4 components (w, x, y, z)")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")

    @classmethod
    def from_launch(cls, *, lat_geod_deg: float = 0.0, lon_deg: float = 0.0, alt_m: float = 0.0,
                    heading_deg: float = 90.0, pitch_deg: float = 90.0, roll_deg: float = 0.0,
                    mass_kg: float, t_s: float = 0.0,
                    v_north_m_s: float = 0.0, v_east_m_s: float = 0.0,
                    v_down_m_s: float = 0.0) -> RigidBodyState:
        """A state on the launch pad (or anywhere, at any attitude), built from
        geodetic coordinates and heading/pitch/roll -- the ECEF equivalent of
        the FL-401 driver's :class:`~orbitopt.flightlab.sixdof.jsbsim_driver.InitialConditions`.

        Velocities are NED (``v_down`` negative means climbing), converted into
        the same ECEF vector the integrator carries; on a pad everything is
        zero, which is the ECEF convention (the pad is not moving in ECEF).
        """
        r = frames.geodetic_to_ecef(lat_geod_deg, lon_deg, alt_m).reshape(3)
        ned2ecef = _ned2ecef(math.radians(float(lat_geod_deg)), math.radians(float(lon_deg)))
        v = ned2ecef @ np.array([v_north_m_s, v_east_m_s, v_down_m_s], dtype=float)
        return cls(t_s=float(t_s), position_m=r, velocity_m_s=v,
                   quaternion=quaternion_from_tec2b(tec2b_from_attitude(
                       lat_geod_deg, lon_deg, heading_deg=heading_deg, pitch_deg=pitch_deg,
                       roll_deg=roll_deg)),
                   omega_rad_s=(0.0, 0.0, 0.0), mass_kg=float(mass_kg))


@dataclass(frozen=True)
class ControlCommand:
    """What the flight computer asks for at one instant."""

    throttle: float = 0.0
    gimbal_pitch_rad: float = 0.0
    gimbal_yaw_rad: float = 0.0


#: ``(t_s, state) -> ControlCommand``; the plant is open loop, guidance is the
#: caller's job (F-06/F-07 close that loop).
ControlLaw = Callable[[float, RigidBodyState], ControlCommand]


def constant_control(throttle: float = 1.0, gimbal_pitch_rad: float = 0.0,
                     gimbal_yaw_rad: float = 0.0) -> ControlLaw:
    """A fixed throttle/gimbal command -- the attitude-hold case F-02's tests
    need (a vehicle pinned at a small AoA through max-Q)."""

    def control(t_s: float, state: RigidBodyState) -> ControlCommand:
        return ControlCommand(throttle=throttle, gimbal_pitch_rad=gimbal_pitch_rad,
                              gimbal_yaw_rad=gimbal_yaw_rad)

    return control


# --------------------------------------------------------------------------
# run container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SixDofRun:
    """A flown 6-DOF trajectory: the sampled columns of :data:`RUN_COLUMNS`,
    the vehicle/physics that produced them, and the max-Q diagnostic.

    ``terminal`` is ``"t_end"`` (flew the requested span), ``"ground"`` (hit
    the WGS-84 ellipsoid -- the crossing time is reported by interpolation
    between the two bracketing samples, and the *samples* stop at the step
    boundary) or ``"propellant"`` (dry). ``propellant_exhausted_t_s`` is when
    the bookkeeping hit dry mass, if it did; the throttle goes to zero there
    and stays there, it does not fly negative mass.
    """

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
    def times_s(self) -> np.ndarray:
        return self.samples["t_s"]

    @property
    def q_bar_pa(self) -> np.ndarray:
        return self.samples["q_bar_pa"]

    @property
    def max_q(self) -> MaxQ:
        """Peak dynamic pressure of the flown samples, with its time refined by
        parabolic interpolation and the altitude/Mach of the peak sample."""
        return max_q_of(self.samples["t_s"], self.samples["q_bar_pa"],
                        altitude_m=self.samples["altitude_m"], mach=self.samples["mach"])

    @property
    def alpha_max_deg(self) -> float:
        """Largest ``|AoA|`` the vehicle saw -- the number that says whether the
        linear-slope tables were read inside their tabulated range."""
        return float(np.max(np.abs(self.samples["alpha_deg"])))

    def events(self, body_id: str = "booster", *, kinds: Sequence[str] = ("max_q",)
               ) -> list[Event]:
        """Contract :class:`~orbitopt.flightlab.contract.Event` list for this
        run: ``max_q`` (the F-02 deliverable) and ``impact`` when the run ended
        on the ground. Ignition/separation/etc. are the scenario's, not the
        plant's."""
        out: list[Event] = []
        if "max_q" in kinds:
            peak = self.max_q
            out.append(Event(t_s=peak.t_s, kind="max_q", body_id=body_id,
                             data={"dynamic_pressure_pa": peak.q_pa,
                                   "altitude_m": peak.altitude_m, "mach": peak.mach}))
        if self.terminal == "ground" and self.ground_t_s is not None:
            out.append(Event(t_s=float(self.ground_t_s), kind="impact", body_id=body_id, data={}))
        return out

    def to_body_track(self, body_id: str = "booster", *, epoch_utc: datetime | None = None,
                      parent_body_id: str | None = None, frame: str = "ECEF",
                      provenance: str = "illustrative",
                      columns: Sequence[str] | None = None) -> BodyTrack:
        """The run as a contract track, ready for a run directory.

        The frame is ``ECEF`` because that is what the plant integrates in; a
        GCRS track is the frame bridge's job (``frames.to_gcrs``), not this
        method's. ``columns`` picks which extras to keep (default: all)."""
        keep = tuple(columns) if columns is not None else tuple(
            c for c in RUN_COLUMNS if c not in ("t_s", "mass_kg", "x_m", "y_m", "z_m",
                                                "vx_m_s", "vy_m_s", "vz_m_s"))
        unknown = [c for c in keep if c not in self.columns]
        if unknown:
            raise KeyError(f"run has no column(s) {unknown}")
        s = self.samples
        return BodyTrack(
            body_id=body_id, parent_body_id=parent_body_id, frame=frame,
            epoch_utc=(epoch_utc or self.epoch_utc).isoformat().replace("+00:00", "Z"),
            provenance=provenance,
            t_s=tuple(float(v) for v in s["t_s"]),
            mass_kg=tuple(float(v) for v in s["mass_kg"]),
            position_m=tuple((float(a), float(b), float(c))
                             for a, b, c in zip(s["x_m"], s["y_m"], s["z_m"])),
            velocity_m_s=tuple((float(a), float(b), float(c))
                               for a, b, c in zip(s["vx_m_s"], s["vy_m_s"], s["vz_m_s"])),
            extra_columns={c: tuple(float(v) for v in s[c]) for c in keep})


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def _gravity_ecef(r: np.ndarray, j2: bool) -> np.ndarray:
    """Gravity at an ECEF position (point mass, optionally + J2) -- the same
    expression the recovery propagator uses, so a 3-DOF/6-DOF pair feels the
    same Earth."""
    x, y, z = (float(c) for c in r)
    rn2 = x * x + y * y + z * z
    f = -MU_EARTH_M3_S2 / (rn2 * math.sqrt(rn2))
    if not j2:
        return np.array([f * x, f * y, f * z])
    k = 1.5 * J2_EARTH * (R_EARTH_M * R_EARTH_M) / rn2
    zz = 5.0 * z * z / rn2
    return np.array([f * x * (1.0 + k * (1.0 - zz)),
                     f * y * (1.0 + k * (1.0 - zz)),
                     f * z * (1.0 + k * (3.0 - zz))])


def _altitude_m(r: np.ndarray) -> float:
    """Height above the WGS-84 ellipsoid [m] (``frames.ecef_to_geodetic``, the
    same ellipsoid the recovery propagator's ``entry.height_m`` approximates to
    within a metre or two at these latitudes)."""
    return float(frames.ecef_to_geodetic(np.asarray(r, dtype=float))[2])


def _air_state(r: np.ndarray, v: np.ndarray, tec2b: np.ndarray,
               physics: RigidBodyPhysics) -> dict[str, float | np.ndarray]:
    """The air-relative flight state the aero model is read at: height,
    density, pressure, speed of sound, air-relative body velocity, speed,
    Mach, dynamic pressure and the two angles.

    With ``rotation`` on, the atmosphere co-rotates with the Earth, so the
    air-relative velocity is the ECEF velocity (which is why a vehicle sitting
    on the pad reads Mach 0); with it off, the ground frame's own motion is
    subtracted so the vehicle still flies as if over a non-rotating Earth.
    """
    h = _altitude_m(r)
    out: dict[str, float | np.ndarray] = {"altitude_m": h}
    if not physics.atmosphere:
        out.update(density_kg_m3=0.0, pressure_pa=0.0, speed_of_sound_m_s=0.0,
                   v_air_body=np.zeros(3), air_speed_m_s=0.0, mach=0.0, q_bar_pa=0.0,
                   alpha_deg=0.0, beta_deg=0.0)
        return out
    atmosphere = physics.atmosphere_model
    rho = float(atmosphere.density_kg_m3(h)) * float(physics.density_scale)
    p_pa = float(atmosphere.pressure_pa(h))
    v_air = v if physics.rotation else v - np.cross(_OMEGA_ECEF, r)
    v_air_body = tec2b @ v_air
    speed = float(np.linalg.norm(v_air_body))
    sound = float(atmosphere.speed_of_sound_m_s(h))
    out.update(density_kg_m3=rho, pressure_pa=p_pa, speed_of_sound_m_s=sound,
               v_air_body=v_air_body, air_speed_m_s=speed,
               mach=speed / sound if sound > 0.0 else 0.0,
               q_bar_pa=dynamic_pressure_pa(rho, speed) if rho > 0.0 else 0.0,
               alpha_deg=angle_of_attack_deg(v_air_body) if speed > 1e-9 else 0.0,
               beta_deg=sideslip_deg(v_air_body) if speed > 1e-9 else 0.0)
    return out


# --------------------------------------------------------------------------
# integration core
# --------------------------------------------------------------------------

def _quat_rate(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``dq/dt = 1/2 q (x) (0, omega)`` for a scalar-first quaternion and body
    rates (Tec2b, so body-frame rates right-multiply)."""
    w, x, y, z = q
    p, qq, r = omega
    return 0.5 * np.array([-(x * p + y * qq + z * r),
                           w * p + y * r - z * qq,
                           w * qq + z * p - x * r,
                           w * r + x * qq - y * p])


def _state_from_y(y: np.ndarray, t_s: float = 0.0) -> RigidBodyState:
    return RigidBodyState(t_s=float(t_s), position_m=y[0:3], velocity_m_s=y[3:6],
                          quaternion=y[6:10], omega_rad_s=y[10:13], mass_kg=float(y[13]))


def _deriv(t_s: float, y: np.ndarray, vehicle: RigidBodyVehicle, physics: RigidBodyPhysics,
           control: ControlLaw, dry_mass_kg: float) -> np.ndarray:
    """RHS of the 13-component ECEF rigid-body equations."""
    r, v, q4, omega = y[0:3], y[3:6], y[6:10], y[10:13]
    mass = max(float(y[13]), dry_mass_kg)
    norm = float(np.linalg.norm(q4)) or 1.0
    tec2b = dcm_from_quaternion(q4 / norm)
    air = _air_state(r, v, tec2b, physics)
    rho = float(air["density_kg_m3"])
    p_pa = float(air["pressure_pa"])

    if vehicle.aero is None or rho <= 0.0 or float(air["q_bar_pa"]) <= 0.0:
        force_aero = np.zeros(3)
        moment_aero = np.zeros(3)
    else:
        force_aero, moment_aero = vehicle.aero.force_and_moment(
            float(air["q_bar_pa"]), float(air["mach"]), float(air["alpha_deg"]),
            float(air["beta_deg"]), omega, float(air["speed_of_sound_m_s"]))

    cmd = control(t_s, _state_from_y(y, t_s))
    throttle = vehicle.engine.throttle(cmd.throttle) if mass > dry_mass_kg else 0.0
    force_thrust, mdot, moment_thrust = vehicle.engine.force_body_n(
        throttle, cmd.gimbal_pitch_rad, cmd.gimbal_yaw_rad, p_pa)

    accel = tec2b.T @ ((force_aero + force_thrust) / mass)
    accel = accel + _gravity_ecef(r, physics.j2)
    if physics.rotation:
        accel = (accel - 2.0 * np.cross(_OMEGA_ECEF, v)
                 - np.cross(_OMEGA_ECEF, np.cross(_OMEGA_ECEF, r)))

    inertia = np.asarray(vehicle.inertia_kg_m2, dtype=float)
    moment_total = moment_aero + moment_thrust
    omega_dot = (moment_total - np.cross(omega, inertia * omega)) / inertia

    dydt = np.empty(14)
    dydt[0:3] = v
    dydt[3:6] = accel
    dydt[6:10] = _quat_rate(q4 / norm, omega)
    dydt[10:13] = omega_dot
    dydt[13] = -mdot if mass > dry_mass_kg else 0.0
    return dydt


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def _sample(t_s: float, y: np.ndarray, vehicle: RigidBodyVehicle, physics: RigidBodyPhysics,
            control: ControlLaw) -> dict[str, float]:
    """One row of :data:`RUN_COLUMNS`, evaluated at the sample instant (the
    control law and the aero read are re-evaluated here, not carried over from
    the integration stage that produced the state)."""
    r, v, q4, omega = y[0:3], y[3:6], y[6:10], y[10:13]
    mass = float(y[13])
    norm = float(np.linalg.norm(q4)) or 1.0
    tec2b = dcm_from_quaternion(q4 / norm)
    air = _air_state(r, v, tec2b, physics)
    if vehicle.aero is None or float(air["density_kg_m3"]) <= 0.0:
        force_aero = np.zeros(3)
        moment_aero = np.zeros(3)
    else:
        force_aero, moment_aero = vehicle.aero.force_and_moment(
            float(air["q_bar_pa"]), float(air["mach"]), float(air["alpha_deg"]),
            float(air["beta_deg"]), omega, float(air["speed_of_sound_m_s"]))
    cmd = control(t_s, _state_from_y(y, t_s))
    dry = vehicle.dry_mass_kg
    throttle = vehicle.engine.throttle(cmd.throttle) if mass > dry else 0.0
    force_thrust, _, _ = vehicle.engine.force_body_n(throttle, cmd.gimbal_pitch_rad,
                                                     cmd.gimbal_yaw_rad,
                                                     float(air["pressure_pa"]))
    row = {
        "t_s": t_s, "mass_kg": mass,
        "x_m": float(r[0]), "y_m": float(r[1]), "z_m": float(r[2]),
        "vx_m_s": float(v[0]), "vy_m_s": float(v[1]), "vz_m_s": float(v[2]),
        "q_w": float(q4[0] / norm), "q_x": float(q4[1] / norm), "q_y": float(q4[2] / norm),
        "q_z": float(q4[3] / norm),
        "p_rad_s": float(omega[0]), "q_rad_s": float(omega[1]), "r_rad_s": float(omega[2]),
        "altitude_m": float(air["altitude_m"]), "mach": float(air["mach"]),
        "alpha_deg": float(air["alpha_deg"]), "beta_deg": float(air["beta_deg"]),
        "q_bar_pa": float(air["q_bar_pa"]), "throttle": throttle,
        "gimbal_pitch_cmd_rad": float(cmd.gimbal_pitch_rad),
        "gimbal_yaw_cmd_rad": float(cmd.gimbal_yaw_rad),
        "thrust_n": float(np.linalg.norm(force_thrust)),
        "aero_fx_N": float(force_aero[0]), "aero_fy_N": float(force_aero[1]),
        "aero_fz_N": float(force_aero[2]),
        "roll_moment_n_m": float(moment_aero[0]), "pitch_moment_n_m": float(moment_aero[1]),
        "yaw_moment_n_m": float(moment_aero[2]),
    }
    return {key: float(value) for key, value in row.items()}


def _append(samples: dict[str, list[float]], row: Mapping[str, float]) -> None:
    for column in RUN_COLUMNS:
        samples[column].append(float(row[column]))


def _state0_vector(state: RigidBodyState) -> np.ndarray:
    y = np.zeros(14)
    y[0:3] = np.asarray(state.position_m, dtype=float)
    y[3:6] = np.asarray(state.velocity_m_s, dtype=float)
    q = np.asarray(state.quaternion, dtype=float)
    y[6:10] = q / float(np.linalg.norm(q))
    y[10:13] = np.asarray(state.omega_rad_s, dtype=float)
    y[13] = float(state.mass_kg)
    return y


def propagate(state0: RigidBodyState, vehicle: RigidBodyVehicle, control: ControlLaw,
              t_end_s: float, *, dt_s: float = 0.02, sample_every: int = 1,
              physics: RigidBodyPhysics | None = None,
              epoch_utc: datetime | None = None) -> SixDofRun:
    """Fly the plant for ``t_end_s`` seconds of mission clock from ``state0``.

    Fixed-step RK4 on ``(r, v, q, omega, m)`` in ECEF, the quaternion
    renormalised at every accepted step, the propellant bookkeeping floored at
    ``dry_mass_kg`` (a run that burns out gets ``throttle = 0`` from there on
    and ``propellant_exhausted_t_s``, it never flies a negative mass).

    ``sample_every`` records every n-th step (plus the first and the last), so
    a 0.01 s integration can hand back a 0.5 s track. Ground contact ends the
    run: the *samples* stop at the last step above the WGS-84 ellipsoid and
    :attr:`SixDofRun.ground_t_s` carries the crossing time interpolated between
    the two bracketing steps -- enough for a max-Q report, not enough for a
    landing verdict, which is what the recovery propagator's bisected touchdown
    is for.

    The default ``epoch_utc`` (2026-01-01T00:00:00Z) only labels the output:
    this plant has no launch date, it has a mission clock.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if int(sample_every) < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if float(t_end_s) <= state0.t_s:
        raise ValueError(f"t_end_s ({t_end_s}) must be after state0.t_s ({state0.t_s})")
    physics = physics or RigidBodyPhysics()
    epoch = epoch_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
    dry = vehicle.dry_mass_kg
    y = _state0_vector(state0)
    if y[13] < dry:
        raise ValueError(f"state0 mass {y[13]} kg is below the vehicle dry mass {dry} kg")
    samples: dict[str, list[float]] = {column: [] for column in RUN_COLUMNS}
    _append(samples, _sample(state0.t_s, y, vehicle, physics, control))

    t = float(state0.t_s)
    dt = float(dt_s)
    step = 0
    terminal = "t_end"
    ground_t_s: float | None = None
    exhausted_t_s: float | None = None
    if y[13] <= dry + 1e-9:
        exhausted_t_s = t
    prev_h = _altitude_m(y[0:3])

    while t < float(t_end_s) - 1e-12:
        h = min(dt, float(t_end_s) - t)
        k1 = _deriv(t, y, vehicle, physics, control, dry)
        k2 = _deriv(t + 0.5 * h, y + 0.5 * h * k1, vehicle, physics, control, dry)
        k3 = _deriv(t + 0.5 * h, y + 0.5 * h * k2, vehicle, physics, control, dry)
        k4 = _deriv(t + h, y + h * k3, vehicle, physics, control, dry)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y[6:10] = y[6:10] / float(np.linalg.norm(y[6:10]))
        y[13] = max(float(y[13]), dry)
        t += h
        step += 1
        h_now = _altitude_m(y[0:3])
        if exhausted_t_s is None and y[13] <= dry + 1e-9:
            exhausted_t_s = t
        if h_now <= 0.0 < prev_h:
            terminal = "ground"
            ground_t_s = t - h * h_now / (h_now - prev_h) if h_now != prev_h else t
            break
        if step % int(sample_every) == 0:
            _append(samples, _sample(t, y, vehicle, physics, control))
        prev_h = h_now

    if terminal != "ground" and samples["t_s"][-1] != t:
        _append(samples, _sample(t, y, vehicle, physics, control))
    return SixDofRun(columns=RUN_COLUMNS,
                     samples={key: np.asarray(values, dtype=float)
                              for key, values in samples.items()},
                     vehicle=vehicle, physics=physics, epoch_utc=epoch, terminal=terminal,
                     ground_t_s=ground_t_s, propellant_exhausted_t_s=exhausted_t_s)
