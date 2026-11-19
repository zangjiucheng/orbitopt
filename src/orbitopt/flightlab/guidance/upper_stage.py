"""E1-03 (task #252): upper-stage guidance shared by both F-07 backends.

The F-07 comparison (task #232) flew the upper stage open loop on both
sides: the rigid-body backend held its inertial attitude with a neutral
gimbal, the point-mass twin kinematically followed the stack's pitch
program. Two different implicit "guidance laws" is exactly why the upper
body's terminal altitude differed by ~235 km while the stack (closed
loop on both sides against the SAME program) agreed to ~330 m. This
module closes that loop the way the acceptance asks: ONE state-feedback
law -- a PEG-class implicit guidance (powered explicit guided steering,
the ILU family's standard ascent form) with a velocity-to-be-gained
cutoff -- evaluated from the live ``(r, v, m)`` state, so a point mass and
a rigid body feeding it the same state command the same attitude and the
same throttle.

The law
-------
``Lambda = V_c * vhat_t - v_inertial + gravity_term_gain * g(r) * tau *
rhat``: the velocity-to-be-gained between the current inertial velocity
and the circular velocity ``V_c = sqrt(mu / r_target)`` along the current
horizontal direction, plus an optional first-order gravity-loss feed
along vertical (``gravity_term_gain``, default off -- for this mission
family the closed loop of the perigee criterion closes the flight-path
angle by itself, and an unmodelled vertical feed only biases the corner;
with the gain at zero the law is velocity-targeting, the linear-tangent
cousin PEG descends from). Steering is "fly along Lambda"; the throttle is
full until the
cutoff fires. The cutoff is an osculating-periapsis check
(``r_peri >= r_target - margin``): at cutoff the coast orbit IS the
osculating orbit, so the criterion puts the *final* orbit's periapsis at
the target instead of just the burnout point -- a pitch-over-to-horizontal
or time-based cutoff would leave a deep (Earth-crossing) periapsis, and a
``|Lambda| = 0`` cutoff fires while the trajectory is still seconds from
its final orbit (the first-order ``tau`` feedforward cannot see the last
tank of forward ``dv``). The perigee kicker is the classical fix, and the
periapsis criterion is its closed-loop form.

The E6-05 extensions (task #292, ADR flightlab-0011)
----------------------------------------------------
Three optional spec fields generalise the law beyond the F-07 point without
changing it there (all default off):

* ``transfer_target``: below the target radius the tangential target is no
  longer the *circular* speed at ``r_target`` (which asks for more speed at
  low altitude than the remaining propellant can circularise, so the burn
  over-coasts to a huge apogee and depletes) but the speed of the *transfer
  ellipse whose apogee the remaining delta-v can still circularise*:
  ``v_t = sqrt(mu (2/r - 1/a))`` with ``a = (r + r_a)/2`` and ``r_a`` capped
  by the largest apogee whose apogee circularisation fits the remaining
  ``dv`` (with a 10 % reserve).  Above the target radius the target falls
  back to the circular speed, so the law self-selects: transfer while
  climbing, circular while inserting.
* ``climb_time_to_go_gain``: a proportional vertical (climb-hold) feed
  ``k (r_target - r) / tau_burn_left * rhat`` along vertical while below the
  target radius (time-to-go over the remaining burn, clamped by the same
  attitude limiter; 1.0 commands arrival at the target radius exactly as the
  tank runs dry).  With ``gravity_term_gain`` off, ``Lambda``'s
  vertical part is ``-v_r``, which commands ever-flatter flight as soon as
  the climb starts and collapses the flight-path angle before the target
  altitude is reached -- the mechanism behind the high-target misses.  The
  feed holds the climb open until ``r`` converges to ``r_target``.
* ``cutoff_apoapsis_margin_m``: the cutoff also waits for the osculating
  *apoapsis* to come down inside ``r_target + margin``.  The periapsis
  criterion alone ends the burn while the ellipse is still too big, which
  is why a low-target cell can coast out 50 km high on apoapsis with
  hundreds of kilograms of propellant still aboard (the energy/apoapsis
  termination ADR flightlab-0011 asks for).

The command limiter: ``Lambda``'s *direction* is undefined as its
magnitude vanishes (and its tangential part flips 180 deg when ``v_t``
overshoots ``V_c``), and between "command nearly spent" and "periapsis
reached" there are a few seconds of exactly that. The shared law therefore
clamps its own attitude command in the velocity frame
(``command_alpha_limit_deg`` pitch bias, ``command_heading_limit_deg``
yaw bias) and rate-limits it (``command_slew_rate_deg_s``) — the same
services a real autopilot's mode-attention limiter provides. Raw Lambda
demands tens of degrees of pitch-off-velocity while the gravity feed
still carries a long burn, and a slender body at that AoA lives in a
hypersonic normal-force regime the point-mass twin cannot even represent;
a tracker also follows a vanishing Lambda's wandering heading straight
into sideslip and a 180 deg flip. Both backends fly the *limited* command
— that is the point of a shared law: the same filter, the same trajectory
family, and the residual that survives is execution, not law.

Inertial velocity enters everywhere (``v_inertial = v_ECEF + omega x r``),
so the rotating-ECEF and inertial backends see the same law in the same
frame; the commanded attitude is converted to ``Tec2b`` at the live
geodetic site.

Both backends, one law
----------------------
The guidance is duck-typed for both propagators of the F-06/F-07
scenario:

* :meth:`UpperStageGuidance.command_from_state` (state in, ``(Tec2b,
  throttle)`` out) plugs into
  :func:`orbitopt.flightlab.sixdof.gravity_turn.propagate_point_mass`
  through its ``guidance=`` keyword -- the 3-DOF twin steers with it
  instead of ``program.dcm_at``;
* :class:`GuidedUpperStageControl` wraps the same object into a
  :data:`~orbitopt.flightlab.sixdof.rigidbody.ControlLaw` for the
  rigid-body plant: the PEG direction becomes the command attitude for
  the F-04 tracker (the tracker/actuator stay the plant's own, so what
  the comparison then measures is the *execution* of one shared law, not
  two laws). The ignition schedule stays the scenario's
  :class:`~orbitopt.flightlab.sixdof.two_stage.IgnitionPlan`: throttle
  stays 0 before ``ignition_t_s`` here too, so the coast leg is identical
  on both backends.

:class:`ParkingOrbitTarget` carries the declared insertion tolerances,
and :func:`insertion_report` grades a terminal state against them -- the
"insertion accuracy with a declared tolerance" artefact of the
acceptance. Like everything in Flight Lab this is illustrative native
physics, not flight certification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.entry import G0_M_S2, MU_EARTH_M3_S2, OMEGA_EARTH_RAD_S
from orbitopt.flightlab import frames
from orbitopt.flightlab.checks import orbital_elements
from orbitopt.flightlab.sixdof.rigidbody import (
    _OMEGA_ECEF,
    _ned2ecef,
    tec2b_from_attitude,
)

__all__ = [
    "GuidedUpperStageControl",
    "ParkingOrbitTarget",
    "UpperStageGuidance",
    "UpperStageGuidanceSpec",
    "insertion_report",
]

_OMEGA = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

#: "same instant" tolerance for the held-command re-read of
#: :meth:`UpperStageGuidance.dcm_from_state` (E6-04, task #276): the RK4 step
#: boundary times carry at most round-off of a few ulps at the 1e4 s scale,
#: far below a physical control interval.
_GUIDANCE_T_EPS_S = 1e-9

#: Fraction of the remaining delta-v the transfer-target apogey cap may
#: spend on circularisation (E6-05): the last tenth covers the gravity and
#: drag losses of the terminal shaping the cap's two-impulse estimate does
#: not carry.
_TRANSFER_TARGET_RESERVE = 0.9


# --------------------------------------------------------------------------
# the target orbit and its tolerance declaration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParkingOrbitTarget:
    """The parking orbit the guidance aims at, with the tolerances the
    insertion is *declared* against (the acceptance's "declared tolerance
    that passes"). ``altitude_m`` is the circular target's altitude over
    the WGS-84 equatorial radius used by the plant (``R_EARTH_M``); the
    inclination entry is optional -- a launch to a fixed pad and azimuth
    inherits its plane from the pad, so the inclination is reported by
    :func:`insertion_report` and graded only when ``inclination_deg`` is
    stated."""

    altitude_m: float
    periapsis_tolerance_m: float = 25_000.0
    apoapsis_tolerance_m: float = 25_000.0
    inclination_deg: float | None = None
    inclination_tolerance_deg: float = 1.0

    def __post_init__(self) -> None:
        if self.altitude_m <= 0.0:
            raise ValueError(f"altitude_m must be positive, got {self.altitude_m}")
        if self.periapsis_tolerance_m <= 0.0 or self.apoapsis_tolerance_m <= 0.0:
            raise ValueError("the apsides tolerances must be positive")
        if self.inclination_tolerance_deg <= 0.0:
            raise ValueError("inclination_tolerance_deg must be positive")

    @property
    def radius_m(self) -> float:
        from orbitopt.entry import R_EARTH_M

        return R_EARTH_M + float(self.altitude_m)


def insertion_report(target: ParkingOrbitTarget, position_m, velocity_m_s, *,
                     frame: str = "ECEF") -> dict:
    """Grade one state against ``target``'s declared tolerances. The state
    is read in ``frame`` (only ``"ECEF"``, the plant's own convention): the
    velocity is converted to inertial (``+ omega x r``) before the Kepler
    elements, so the numbers are the coast-orbit elements the run actually
    lands in, not an ECEF-relative artefact."""
    from orbitopt.entry import R_EARTH_M

    if frame != "ECEF":
        raise ValueError(f"only the plant's ECEF frame is supported, got {frame!r}")
    r = np.asarray(position_m, dtype=float)
    v = np.asarray(velocity_m_s, dtype=float)
    v_inertial = v + np.cross(_OMEGA, r)
    el = orbital_elements(r, v_inertial, mu=MU_EARTH_M3_S2)
    peri = float(el["periapsis_altitude_m"])
    apo = float(el["apoapsis_altitude_m"])
    inc = float(el["inclination_deg"])
    d_peri = peri - float(target.altitude_m)
    d_apo = apo - float(target.altitude_m)
    peri_ok = abs(d_peri) <= target.periapsis_tolerance_m
    apo_ok = abs(d_apo) <= target.apoapsis_tolerance_m
    if target.inclination_deg is None:
        inc_ok = True
        d_inc = 0.0
    else:
        d_inc = inc - float(target.inclination_deg)
        inc_ok = abs(d_inc) <= target.inclination_tolerance_deg
    return {
        "target_altitude_m": float(target.altitude_m),
        "target_radius_m": float(target.radius_m),
        "periapsis_altitude_m": peri,
        "apoapsis_altitude_m": apo,
        "inclination_deg": inc,
        "eccentricity": float(el["eccentricity"]),
        "d_periapsis_m": d_peri,
        "d_apoapsis_m": d_apo,
        "d_inclination_deg": d_inc,
        "periapsis_tolerance_m": float(target.periapsis_tolerance_m),
        "apoapsis_tolerance_m": float(target.apoapsis_tolerance_m),
        "periapsis_within_tolerance": bool(peri_ok),
        "apoapsis_within_tolerance": bool(apo_ok),
        "inclination_within_tolerance": bool(inc_ok),
        "inserted": bool(peri_ok and apo_ok and inc_ok),
    }


# --------------------------------------------------------------------------
# the guidance: spec (shared, frozen) and per-flight stateful instance
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class UpperStageGuidanceSpec:
    """Everything the law needs that is fixed at scenario-build time: the
    target, the engine's rated figures (for the ``tau`` estimate the law
     carries -- the spec is vehicle-agnostic numbers, not a plant object),
     the ignition gate, the cutoff margin and the command slew limit.
     ``new_guidance()`` mints one stateful :class:`UpperStageGuidance` per
     flight, so the same scenario object flown twice (once per backend, or
     twice on one backend at different dt) never shares a latch."""

    target: ParkingOrbitTarget
    dry_mass_kg: float
    rated_thrust_vacuum_n: float
    isp_vacuum_s: float
    ignition_t_s: float
    #: Cutoff fires when the osculating periapsis reaches
    #: ``r_target - cutoff_periapsis_margin_m``: a positive margin cuts the
    #: burn early so the final periapsis lands on the target.
    cutoff_periapsis_margin_m: float = 0.0
    #: Attitude-command slew limiter [deg/s] (see the module docstring):
    #: the direction of a vanishing Lambda is numerically meaningless (and
    #: flips 180 deg once ``v_t`` overshoots ``V_c``), so the shared law
    #: rate-limits its own command instead of asking the airframe for
    #: step reversals. The smooth phase of the burn demands < 1 deg/s, so
    #: the limiter is transparent until the terminal window.
    command_slew_rate_deg_s: float = 3.0
    #: Pitch bias limit in the velocity frame [deg]: the commanded attitude
    #: may deviate from the velocity vector's pitch by no more than this.
    #: First-order Lambda can demand tens of degrees of pitch-off-velocity
    #: while the gravity feed ``g * tau`` still carries a long remaining
    #: burn, and a slender body at tens of degrees AoA in hypersonic flow
    #: lives in a normal-force regime the point-mass twin cannot even
    #: represent -- so the shared law must not command it. This is the same
    #: AoA limiter a real PEG autopilot runs.
    command_alpha_limit_deg: float = 30.0
    #: Yaw/heading bias limit in the velocity frame [deg]: bounds sideslip
    #: the same way, because Lambda's heading component is free to wander
    #: and a state-feedback tracker follows it straight into beta.
    command_heading_limit_deg: float = 3.0
    gravity_term_gain: float = 0.0
    #: E6-05 (task #292): below the target radius the tangential target is
    #: the speed of the transfer ellipse whose apogee the remaining delta-v
    #: can still circularise (module docstring), instead of the circular
    #: speed at ``r_target``.  Default off: the F-07 mission the law was
    #: built for is one point on the envelope and does not need it.
    transfer_target: bool = False
    #: E6-05: proportional vertical (climb-hold) feed along vertical while
    #: below the target radius: ``k (r_target - r) / tau_burn_left * rhat``,
    #: a time-to-go profile over the remaining burn, clamped by
    #: ``command_alpha_limit_deg`` like every other command deviation
    #: (1.0 = arrive at the target radius exactly as the tank runs dry).
    #: Default 0 = off.
    climb_time_to_go_gain: float = 0.0
    #: E6-05: the cutoff also waits for the osculating apoapsis to come down
    #: inside ``r_target + cutoff_apoapsis_margin_m`` (the energy/apoapsis
    #: termination of ADR flightlab-0011).  Default 0 = periapsis-only, the
    #: F-07 behaviour.
    cutoff_apoapsis_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if self.dry_mass_kg <= 0.0:
            raise ValueError(f"dry_mass_kg must be positive, got {self.dry_mass_kg}")
        if self.rated_thrust_vacuum_n <= 0.0 or self.isp_vacuum_s <= 0.0:
            raise ValueError("rated_thrust_vacuum_n and isp_vacuum_s must be positive")
        if self.cutoff_periapsis_margin_m < 0.0:
            raise ValueError("cutoff_periapsis_margin_m must be >= 0")
        if not 0.0 < self.command_slew_rate_deg_s <= 90.0:
            raise ValueError(f"command_slew_rate_deg_s must be in (0, 90], got "
                             f"{self.command_slew_rate_deg_s}")
        if not 0.0 < self.command_alpha_limit_deg <= 90.0:
            raise ValueError(f"command_alpha_limit_deg must be in (0, 90], got "
                             f"{self.command_alpha_limit_deg}")
        if not 0.0 < self.command_heading_limit_deg <= 30.0:
            raise ValueError(f"command_heading_limit_deg must be in (0, 30], got "
                             f"{self.command_heading_limit_deg}")
        if not 0.0 <= self.gravity_term_gain <= 2.0:
            raise ValueError(f"gravity_term_gain must be in [0, 2], got "
                             f"{self.gravity_term_gain}")
        if self.climb_time_to_go_gain < 0.0:
            raise ValueError(f"climb_time_to_go_gain must be >= 0, got "
                             f"{self.climb_time_to_go_gain}")
        if self.cutoff_apoapsis_margin_m < 0.0:
            raise ValueError(f"cutoff_apoapsis_margin_m must be >= 0, got "
                             f"{self.cutoff_apoapsis_margin_m}")

    @classmethod
    def from_vehicle(cls, vehicle, *, target: ParkingOrbitTarget,
                     ignition_t_s: float, **kwargs) -> UpperStageGuidanceSpec:
        """Read the rated figures off a plant vehicle -- the scenario's
        :class:`~orbitopt.flightlab.sixdof.rigidbody.RigidBodyVehicle`
        stays the single source of the engine truth."""
        engine = vehicle.engine
        return cls(target=target, dry_mass_kg=float(vehicle.dry_mass_kg),
                   rated_thrust_vacuum_n=float(engine.count) * float(engine.thrust_vacuum_n),
                   isp_vacuum_s=float(engine.isp_vacuum_s),
                   ignition_t_s=float(ignition_t_s), **kwargs)

    def new_guidance(self) -> UpperStageGuidance:
        return UpperStageGuidance(self)


class UpperStageGuidance:
    """One flight's stateful PEG: steering direction from the live state,
    osculating-periapsis cutoff with a one-way latch, and a rate-limited
    attitude command.

    :meth:`command_from_state` is the duck-typed entry the 3-DOF twin
    calls; :class:`GuidedUpperStageControl` wraps the same method for the
    rigid-body plant. After the cutoff latches the throttle stays 0 for the
    rest of the flight (MECO is not a suggestion) and the attitude stays
    at its last limited value.
    """

    def __init__(self, spec: UpperStageGuidanceSpec) -> None:
        self.spec = spec
        self.cutoff_t_s: float | None = None
        self._cmd_hp: tuple[float, float] | None = None
        self._cmd_t_s: float | None = None

    @property
    def cutoff_engaged(self) -> bool:
        return self.cutoff_t_s is not None

    # -- PEG internals -----------------------------------------------------
    def _lambda_ecef(self, r: np.ndarray, v_inertial: np.ndarray,
                     mass_kg: float) -> tuple[np.ndarray, float]:
        """The velocity-to-be-gained vector (ECEF components) and the
        target circular speed its band is measured against."""
        rmag = float(np.linalg.norm(r))
        r_hat = r / rmag
        v_radial = float(v_inertial @ r_hat)
        v_tangent = v_inertial - v_radial * r_hat
        vt = float(np.linalg.norm(v_tangent))
        if vt > 1e-9:
            t_hat = v_tangent / vt
        else:
            t_hat = np.array([0.0, 1.0, 0.0])
        r_t = self.spec.target.radius_m
        v_c = math.sqrt(MU_EARTH_M3_S2 / r_t)
        burn_left = max(float(mass_kg) - self.spec.dry_mass_kg, 0.0)
        tau = burn_left * self.spec.isp_vacuum_s * G0_M_S2 / self.spec.rated_thrust_vacuum_n
        g_r = MU_EARTH_M3_S2 / (rmag * rmag)
        v_t_target = (self._capped_transfer_speed(rmag, float(mass_kg))
                      if self.spec.transfer_target and rmag < r_t else v_c)
        lam = (v_t_target * t_hat - v_inertial
               + self.spec.gravity_term_gain * g_r * tau * r_hat)
        if self.spec.climb_time_to_go_gain > 0.0 and rmag < r_t:
            v_climb_cap = v_c * math.tan(math.radians(self.spec.command_alpha_limit_deg))
            v_r_cmd = min(self.spec.climb_time_to_go_gain * (r_t - rmag) / max(tau, 1e-3),
                          v_climb_cap)
            lam = lam + v_r_cmd * r_hat
        return lam, v_c

    def _capped_transfer_speed(self, rmag: float, mass_kg: float) -> float:
        """E6-05: the tangential speed of the transfer ellipse whose apogee
        is the largest the remaining propellant can still circularise,
        capped at the target radius (module docstring).  With the reserve
        exhausted the target degenerates to the circular speed at the
        current radius -- the law then holds station, and the mission fails
        on capability with a clean energy bookkeeping, not on guidance."""
        mu = MU_EARTH_M3_S2
        r_t = self.spec.target.radius_m

        def dv_apo(r_a: float) -> float:
            a = 0.5 * (rmag + r_a)
            v_a = math.sqrt(max(mu * (2.0 / r_a - 1.0 / a), 0.0))
            return math.sqrt(mu / r_a) - v_a

        burn_left = max(mass_kg - self.spec.dry_mass_kg, 0.0)
        budget = (_TRANSFER_TARGET_RESERVE * self.spec.isp_vacuum_s * G0_M_S2
                  * math.log(mass_kg / (mass_kg - burn_left))
                  if burn_left > 0.0 else 0.0)
        if dv_apo(r_t) <= budget:
            r_a = r_t
        else:
            lo, hi = rmag, r_t
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if dv_apo(mid) <= budget:
                    lo = mid
                else:
                    hi = mid
            r_a = lo
        a = 0.5 * (rmag + r_a)
        return math.sqrt(max(mu * (2.0 / rmag - 1.0 / a), mu / rmag))

    def _periapsis_radius(self, r: np.ndarray, v_inertial: np.ndarray) -> float:
        """Osculating periapsis RADIUS [m] of the inertial state (+inf for
        escape trajectories -- an escape orbit clears any margin)."""
        rmag = float(np.linalg.norm(r))
        energy = float(v_inertial @ v_inertial) / 2.0 - MU_EARTH_M3_S2 / rmag
        if energy >= 0.0:
            return float("inf")
        a = -MU_EARTH_M3_S2 / (2.0 * energy)
        h = np.cross(r, v_inertial)
        e_sq = max(0.0, 1.0 - float(h @ h) / (a * MU_EARTH_M3_S2))
        return a * (1.0 - min(math.sqrt(e_sq), 1.0))

    def _apoapsis_radius(self, r: np.ndarray, v_inertial: np.ndarray) -> float:
        """Osculating apoapsis RADIUS [m] of the inertial state (``+inf`` for
        escape trajectories -- an escape orbit clears no margin)."""
        rmag = float(np.linalg.norm(r))
        energy = float(v_inertial @ v_inertial) / 2.0 - MU_EARTH_M3_S2 / rmag
        if energy >= 0.0:
            return float("inf")
        a = -MU_EARTH_M3_S2 / (2.0 * energy)
        h = np.cross(r, v_inertial)
        e_sq = max(0.0, 1.0 - float(h @ h) / (a * MU_EARTH_M3_S2))
        return a * (1.0 + min(math.sqrt(e_sq), 1.0))

    # -- command interface ---------------------------------------------------
    def _heading_pitch(self, direction_ecef: np.ndarray, r: np.ndarray) -> tuple[float, float]:
        """Heading/pitch [deg] of an ECEF direction at the state's geodetic
        site (the PEG plane is the velocity plane, so roll does no steering
        work and stays 0)."""
        lat_deg, lon_deg, _ = frames.ecef_to_geodetic(r)
        d_ned = (_ned2ecef(math.radians(float(lat_deg)),
                           math.radians(float(lon_deg))).T
                 @ np.asarray(direction_ecef, dtype=float))
        horiz = math.hypot(float(d_ned[0]), float(d_ned[1]))
        heading_deg = math.degrees(math.atan2(float(d_ned[1]), float(d_ned[0]))) % 360.0
        pitch_deg = math.degrees(math.atan2(-float(d_ned[2]), horiz))
        return heading_deg, pitch_deg

    def dcm_from_state(self, t_s: float, position_m, velocity_m_s,
                       mass_kg: float) -> np.ndarray:
        """Command ``Tec2b`` at the live state: fly along Lambda, with the
        command rate-limited (``command_slew_rate_deg_s``) -- the direction
        of a vanishing velocity-to-be-gained is numerically meaningless and
        a tracker must never be asked to flip 180 deg in one step."""
        r = np.asarray(position_m, dtype=float)
        v_inertial = (np.asarray(velocity_m_s, dtype=float)
                      + np.cross(_OMEGA, r))
        lam, _ = self._lambda_ecef(r, v_inertial, float(mass_kg))
        if float(np.linalg.norm(lam)) > 1e-9:
            heading_deg, pitch_deg = self._heading_pitch(lam, r)
        else:
            # |Lambda| exactly zero (a scenario that starts inserted): hold
            # along-track, level.
            r_hat = r / float(np.linalg.norm(r))
            v_t = v_inertial - float(v_inertial @ r_hat) * r_hat
            if float(np.linalg.norm(v_t)) > 1e-9:
                heading_deg, pitch_deg = self._heading_pitch(v_t, r)
            else:
                heading_deg, pitch_deg = self._heading_pitch(
                    np.array([0.0, 1.0, 0.0]), r)
        # Velocity-frame bias clamps (the law's AoA / sideslip limiters):
        # the command's deviation from the velocity vector is bounded, so
        # neither backend is ever asked to fly a normal-force regime the
        # other cannot represent.
        vel_h, vel_p = self._heading_pitch(v_inertial, r)
        pitch_deg = vel_p + min(max(pitch_deg - vel_p,
                                    -self.spec.command_alpha_limit_deg),
                                self.spec.command_alpha_limit_deg)
        dh = (heading_deg - vel_h + 180.0) % 360.0 - 180.0
        heading_deg = (vel_h + min(max(dh, -self.spec.command_heading_limit_deg),
                                  self.spec.command_heading_limit_deg)) % 360.0
        if self._cmd_hp is not None:
            if float(t_s) <= float(self._cmd_t_s) + _GUIDANCE_T_EPS_S:
                # E6-04 (task #276): the law already answered this instant --
                # a re-read at a held time (the plant's sampler of the sample
                # grid, or any caller that evaluates twice at one time) gets
                # the HELD command back; it must neither advance the slew
                # limiter by dt=0 nor re-seed it from the raw un-slewed
                # pointing.
                heading_deg, pitch_deg = self._cmd_hp
            else:
                dt = float(t_s) - float(self._cmd_t_s)
                max_step = self.spec.command_slew_rate_deg_s * dt
                last_h, last_p = self._cmd_hp
                dh = (heading_deg - last_h + 180.0) % 360.0 - 180.0
                heading_deg = (last_h + math.copysign(min(abs(dh), max_step), dh)) % 360.0
                pitch_deg = last_p + math.copysign(
                    min(abs(pitch_deg - last_p), max_step), pitch_deg - last_p)
        self._cmd_hp = (heading_deg, pitch_deg)
        self._cmd_t_s = float(t_s)
        lat_deg, lon_deg, _ = frames.ecef_to_geodetic(r)
        return tec2b_from_attitude(float(lat_deg), float(lon_deg),
                                   heading_deg=heading_deg, pitch_deg=pitch_deg,
                                   roll_deg=0.0)

    def throttle_from_state(self, t_s: float, position_m, velocity_m_s,
                            mass_kg: float) -> float:
        """0 before the scheduled ignition or once the osculating-periapsis
        cutoff latches, ``1.0`` (rated) in between -- the plant's own
        ``RigidBodyEngine.throttle`` floor still applies downstream, so this
        number is a *commanded level*, not a delivered one."""
        if float(t_s) < self.spec.ignition_t_s or self.cutoff_engaged:
            return 0.0
        r = np.asarray(position_m, dtype=float)
        v_inertial = np.asarray(velocity_m_s, dtype=float) + np.cross(_OMEGA, r)
        limit = self.spec.target.radius_m - float(self.spec.cutoff_periapsis_margin_m)
        if self._periapsis_radius(r, v_inertial) >= limit:
            if self.spec.cutoff_apoapsis_margin_m > 0.0:
                apo_limit = (self.spec.target.radius_m
                             + float(self.spec.cutoff_apoapsis_margin_m))
                if self._apoapsis_radius(r, v_inertial) > apo_limit:
                    return 1.0
            self.cutoff_t_s = float(t_s)
            return 0.0
        return 1.0

    def command_from_state(self, t_s: float, position_m, velocity_m_s,
                           mass_kg: float) -> tuple[np.ndarray, float]:
        """One ``(Tec2b, throttle)`` command evaluated at the live state."""
        return (self.dcm_from_state(t_s, position_m, velocity_m_s, mass_kg),
                self.throttle_from_state(t_s, position_m, velocity_m_s, mass_kg))


# --------------------------------------------------------------------------
# the rigid-body side: the same guidance as a ControlLaw
# --------------------------------------------------------------------------

class _GuidanceProgram:
    """Duck-typed ``WaypointAttitudeProgram`` view of a live guidance
    object for :class:`~orbitopt.flightlab.sixdof.tvc.AttitudeTvcLaw`:
    ``dcm_at`` replays the last commanded attitude, ``omega_ff_body`` is
    zero (a state-feedback law needs no programmed rate feed-forward --
    the rate command IS the tracking loop)."""

    def __init__(self, guidance: UpperStageGuidance) -> None:
        self._guidance = guidance
        self._tec2b: np.ndarray | None = None

    def update(self, tec2b: np.ndarray, t_s: float) -> None:
        self._tec2b = np.asarray(tec2b, dtype=float)
        self._t_s = float(t_s)

    def dcm_at(self, t_s: float, lat_geod_deg: float, lon_deg: float) -> np.ndarray:
        if self._tec2b is None:
            raise RuntimeError("the guidance program was queried before its "
                               "first command update")
        return self._tec2b

    def omega_ff_body(self, t_s: float) -> np.ndarray:
        return np.zeros(3)


class GuidedUpperStageControl:
    """The guidance as a plant :data:`ControlLaw`: the PEG command attitude
    fed to the F-04 tracker and the guidance's throttle. The tracker and
    its actuator are the plant's standard pair (same servo model, same
    gains as the stack ascent), so the 3-DOF/6-DOF comparison after this
    law is on measures the *execution* delta (attitude dynamics + actuator
    lag), never a guidance delta.

    Before ``ignition_t_s`` the throttle is 0, which puts the tracker in
    its engine-off freeze (neutral gimbals) -- the rigid-body coast leg is
    the same ballistic coast the point-mass twin flies.
    """

    def __init__(self, spec: UpperStageGuidanceSpec, vehicle, *,
                 actuator=None, gains=None, dt_c_s: float = 0.05) -> None:
        from orbitopt.flightlab.sixdof.tvc import AttitudeTvcLaw, TvcActuator, TvcGains

        self.guidance = spec.new_guidance()
        self._program = _GuidanceProgram(self.guidance)
        actuator = actuator or TvcActuator(limit_rad=float(vehicle.engine.gimbal_limit_rad))
        self.tracker = AttitudeTvcLaw(vehicle=vehicle, program=self._program,
                                      actuator=actuator,
                                      gains=gains or TvcGains(),
                                      throttle=0.0, dt_c_s=dt_c_s)

    def __call__(self, t_s, state) -> "ControlCommand":  # noqa: F821
        tec2b, throttle = self.guidance.command_from_state(
            float(t_s), state.position_m, state.velocity_m_s, state.mass_kg)
        self._program.update(tec2b, float(t_s))
        self.tracker.throttle = throttle
        return self.tracker(float(t_s), state)

    @property
    def diagnostics(self):
        """The F-04 tracker diagnostics of the guided burn."""
        return self.tracker.diagnostics

    def history(self) -> dict:
        return self.tracker.history()
