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
  ``dv`` (with a 10 % reserve) and by ``r_target - transfer_apogee_bias_m``.
  Above the target radius the target falls back to the circular speed, so
  the law self-selects: transfer while climbing, circular while inserting.
  The bias is the planning margin the matrix needs: on the low-target cells
  the perigee-kick phase overshoots the commanded apogee by ~50 km, so a
  plan capped at the target radius itself has its apogee out of the
  declared band before the periapsis criterion can close.
* ``climb_time_to_go_gain``: a proportional vertical (climb-hold) feed
  ``k (r_target - r) / tau_burn_left * rhat`` along vertical while below the
  target radius (time-to-go over the remaining burn, clamped by the same
  attitude limiter; 1.0 commands arrival at the target radius exactly as the
  tank runs dry).  With ``gravity_term_gain`` off, ``Lambda``'s
  vertical part is ``-v_r``, which commands ever-flatter flight as soon as
  the climb starts and collapses the flight-path angle before the target
  altitude is reached.  Measured on the E6-03 matrix the feed *holds the
  climb open past the apogee the cutoff can accept*: with the gain at 1.0
  every cell it touched ended in depletion with a 7000 km apogee, because
  the apoapsis-side termination then holds a throttle whose target radius
  recedes faster than the tank empties.  The knob stays available for a
  target whose altitude the vehicle cannot reach at all; the matrix flies
  it at zero.
* ``cutoff_insertion_box``: the cutoff fires at the first instant the
  osculating orbit lies inside the *declared insertion band* -- ``r_peri >=
  r_target - periapsis_tolerance_m`` and ``r_apo <= r_target +
  apoapsis_tolerance_m`` together -- instead of waiting for the strict
  circular periapsis.  The periapsis-only criterion ends the burn while the
  ellipse is still too big, which is why a low-target cell can coast out 50
  km high on apoapsis with hundreds of kilograms of propellant still
  aboard: the apogee leaves the band *before* the periapsis criterion
  fires, so the burn has to be judged against the whole declared box and
  not against its lower edge alone (the energy/apoapsis termination of ADR
  flightlab-0011).  Terminating inside the declared band is not a tolerance
   relaxation: the acceptance band is unchanged, the law simply stops where
   the mission is already satisfied.
* ``apogee_ceiling_gain``: terminal apogee hold, the control side of the
  box cutoff.  A cutoff can only *stop* a burn; on the low-target cells the
  trace shows the opposite failure -- the periapsis enters the band while
  the apogee rides ~50-70 km *over* its ceiling (the pull-up dumps a few
  tens of m/s of tangential over-speed at perigee), and a law whose
  tangential target is then still the transfer speed keeps pumping energy
  into an already-overshot apogee until the tank runs dry.  While the
  cutoff box is armed -- periapsis criterion satisfied, apoapsis criterion
  not -- the tangential target is trimmed proportionally below the planned
  speed, ``v_t -= k_apogee (r_apo - r_ceiling)`` (floored at near-circular
  at the current radius), so the burn bleeds the over-speed instead of
  adding to it; the apogee converges to the ceiling and the box closes.
  Proportional, self-clearing (the term vanishes as the apogee enters the
  band) and off by default.
  Measured on the matrix this hold is inert-but-correct, and the reason is
  a limit of the law family, not of the cutoff: the command is clamped to
  +/-``command_alpha_limit_deg`` off the velocity vector, so within 30 deg
  a retro correction is >= 87 % prograde and the apogee keeps growing
  anyway.  Widening that limiter to 45 deg *does* convert the last
  shape-limited cell -- but the same extra pitch authority dumps the two
  high-inclination reference cells' periapsis just under the band floor
  (they deplete 3-6 km short), so the 45 deg point is recorded as rejected:
  the envelope boundary is the finding, not a wider setting that trades
  cells.  No setting of the cutoff, the cap, the feeds or this hold turns
  the depletion-class misses either: their periapsis never reaches the band
  floor while the tank empties, so the box is never armed and their cause
  is the stage's delta-v budget, not the law's terminal conditions.

The E10-08 extension (task #305): the terminal-radius radial channel
--------------------------------------------------------------------
E10-08 re-read the E6-05 depletion class on the trace instead of on the
terminal elements, and the last paragraph above does not survive it: on
a260 the stage peaks at 227 km with ``v_r`` through zero and then empties
the tank horizontally -- the periapsis cannot exceed the current radius, and
the velocity-only law never *flew* to the target radius.  Nothing in
``Lambda`` asks for a radius: its vertical part is ``-v_r`` (plus feeds that
are not a position target), so the climb is whatever the handoff's 2 km/s of
vertical speed and the tangential target leave behind.  The energy rule was
right (the terminal orbit is thousands of km WIDER than the target), and the
fault is the law's -- the cutoff could not arm because the state never
reached the box.  Three optional fields fix that without touching the F-07
point (all default off):

* ``terminal_radius_guidance``: the steering direction becomes the current
  point of a **radial plan**.  The plan is the classic PEG radial channel of
  the linear-tangent family: a vertical thrust fraction ``s(t) = A + B t``
  that lands ``r(T) = r_target`` *and* ``v_r(T) = 0`` for a constant-thrust
  stage draining linearly (the closed-form thrust integrals ``L, J, S, Q``)
  under the net vertical gravity averaged between now and circular arrival;
  ``T`` is the burn time whose plan buys exactly the horizontal speed still
  missing (bisection on a Simpson integral of ``sqrt(1 - s^2) a(t)``), or the
  whole remaining burn when even that falls short.  Below
  ``terminal_radius_t_go_floor_s`` (12 s) the plan is ill-conditioned (it
  divides by ``T^2``) and the law latches the terminal velocity-to-be-gained
  steering ``(v_c - v_t) t_hat - v_r r_hat`` -- the direction that zeroes
  both residuals together.  12 s rather than a few: in the last seconds the
  stage accelerates at ~20 g, and a handover that late leaves the command
  slew no time to swing from the plan's terminal pitch-down onto the
  velocity-to-be-gained direction (measured: at a 5 s handover the rigid
  body sweeps through the band on a300 between two control samples).  For
  the same swing the matrix flies a 6 deg/s command slew instead of the
  E1-03 3 deg/s, which was sized for a vanishing-Lambda command that can
  flip; the plan's command is smooth and the limiter still bounds a flip
  to 30 s (10 deg/s inserts too, but over-runs the JSBSim tracker's E6-05
  attitude-arrest allowance; see ``guidance_matrix``).
  With the channel on, the command is also held **prograde until
  ignition** and the slew limiter is seeded there: the point-mass twin
  first consults the law at ignition, the rigid body and JSBSim from
  separation with their trackers frozen through the coast, so a command
  that jumped (or slewed during the coast) onto the radial plan handed the
  frozen bodies a ~70 deg step at ignition that the point mass flew for
  free -- a backend-dependent command history, measured as a 35 kg native
  / JSBSim propellant split on a180.
* **settle cutoff** (with ``cutoff_insertion_box`` and the radial channel on):
  inside the band the burn continues while the apsides error ``max(r_t -
  r_peri, r_apo - r_t)`` is shrinking by no more than its own size per
  control sample, i.e. while the next sample is predicted to land closer to
  the circle and on the same side; otherwise it cuts at once.  At ~20 g the
  band is crossed in a handful of 0.05 s samples, so "cut at the first inside
  sample" leaves the orbit on the band edge, and "cut when the error grows"
  can step clean through the band and never cut.
* ``command_q_alpha_limit_pa_deg``: the pitch-bias limiter becomes a
  **q-alpha load limit**: ``clip(q_alpha / q, command_alpha_limit_deg,
  vacuum_alpha_limit_deg)`` at the live US76 dynamic pressure on the
  air-relative speed.  The 30 deg bound exists because a slender body at tens
  of degrees AoA in hypersonic flow lives in a normal-force regime the
  point-mass twin cannot represent (above); the upper stage ignites at
  ~90 km, q ~ 10 Pa, where that regime does not exist.  Inside the 30 deg
  clamp the radial plan cannot point below the steep handoff velocity
  vector, which is the very authority it needs to kill the vertical speed.
  Wherever the air can load the airframe the 30 deg bound is unchanged.

With the three on, the matrix's guidance class converts (ADR
flightlab-0011, E10-08 follow-through).  The plan is also an honest
feasibility test: at 400 km from this handoff it needs ``|s| > 1`` (more
than all of the thrust vertical) for the whole burn, so no single
continuous full-thrust burn arrives level there -- a stage without
coast/relight cannot fly that profile, and the cell stays a miss.

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
    #: E6-05: the cutoff fires on the declared insertion band (module
    #: docstring): the osculating periapsis at or above ``r_target -
    #: periapsis_tolerance_m`` AND the osculating apoapsis at or below
    #: ``r_target + apoapsis_tolerance_m`` together, instead of the strict
    #: ``r_peri >= r_target - cutoff_periapsis_margin_m`` alone.  Default
    #: off: the F-07 mission cuts on the strict criterion already.
    cutoff_insertion_box: bool = False
    #: E6-05: planning margin [m] deducted from the transfer cap's apogee
    #: ceiling (``r_target - transfer_apogee_bias_m`` instead of
    #: ``r_target``): the perigee-kick phase of the burn overshoots the
    #: commanded apogee by a few tens of km, so a plan capped at the target
    #: radius itself arrives with the apogee already out of the declared
    #: band before the periapsis criterion can close.  Expressed in the
    #: declared insertion tolerance, not in per-cell numbers.  Default 0.
    transfer_apogee_bias_m: float = 0.0
    #: E6-05: terminal apogee hold gain [s^-1 ... m/s per m of apogee over
    #: the band ceiling]: while the cutoff box's periapsis side is satisfied
    #: but its apoapsis side is not, the tangential target is lowered by
    #: ``k (r_apo - r_target - apoapsis_tolerance_m)`` (module docstring) so
    #: the burn bleeds the perigee over-speed instead of amplifying it.
    #: Default 0 = off.
    apogee_ceiling_gain: float = 0.0
    #: E10-08: terminal-radius (E-guidance) radial channel, see
    #: :meth:`UpperStageGuidance._terminal_radius_direction`.  Default off.
    terminal_radius_guidance: bool = False
    #: E10-08: time-to-go [s] below which the radial plan hands over to the
    #: terminal velocity-to-be-gained steering (module docstring).
    terminal_radius_t_go_floor_s: float = 12.0
    #: E10-08: q-alpha load limit [Pa deg].  When positive, the pitch-bias
    #: limit becomes ``clip(q_alpha / q, command_alpha_limit_deg,
    #: vacuum_alpha_limit_deg)`` at the live dynamic pressure ``q``: the
    #: ``command_alpha_limit_deg`` bound is held wherever the air can load
    #: the airframe, and opens only where it cannot.  Default 0 = off.
    command_q_alpha_limit_pa_deg: float = 0.0
    #: E10-08: the pitch-bias ceiling of the q-alpha limiter [deg].
    vacuum_alpha_limit_deg: float = 90.0

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
        if not 0.0 <= self.transfer_apogee_bias_m < self.target.radius_m:
            raise ValueError(f"transfer_apogee_bias_m must be in [0, r_target), got "
                             f"{self.transfer_apogee_bias_m}")
        if self.apogee_ceiling_gain < 0.0:
            raise ValueError(f"apogee_ceiling_gain must be >= 0, got "
                             f"{self.apogee_ceiling_gain}")
        if self.terminal_radius_t_go_floor_s <= 0.0:
            raise ValueError(f"terminal_radius_t_go_floor_s must be > 0, got "
                             f"{self.terminal_radius_t_go_floor_s}")
        if self.command_q_alpha_limit_pa_deg < 0.0:
            raise ValueError(f"command_q_alpha_limit_pa_deg must be >= 0, got "
                             f"{self.command_q_alpha_limit_pa_deg}")
        if not self.command_alpha_limit_deg <= self.vacuum_alpha_limit_deg <= 90.0:
            raise ValueError(f"vacuum_alpha_limit_deg must be in "
                             f"[command_alpha_limit_deg, 90], got "
                             f"{self.vacuum_alpha_limit_deg}")

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
        self._settle: tuple[float, float] | None = None
        self._radial_terminal = False

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
        if self.spec.apogee_ceiling_gain > 0.0 and self.spec.cutoff_insertion_box:
            # E6-05 terminal apogee hold (module docstring): the periapsis
            # side of the cutoff box is already satisfied, so the burn is in
            # its terminal phase -- an apogee over the ceiling at this point
            # is over-speed the cutoff cannot fix by stopping, only by not
            # adding.  Trim the tangential target proportionally, floored at
            # near-circular so the command stays a passable orbit.
            target = self.spec.target
            r_ceil = target.radius_m + float(target.apoapsis_tolerance_m)
            r_apo = self._apoapsis_radius(r, v_inertial)
            if (r_apo > r_ceil
                    and self._periapsis_radius(r, v_inertial)
                    >= target.radius_m - float(target.periapsis_tolerance_m)):
                trim = self.spec.apogee_ceiling_gain * (r_apo - r_ceil)
                v_t_target = max(v_t_target - trim,
                                 0.85 * math.sqrt(MU_EARTH_M3_S2 / rmag))
        if self.spec.terminal_radius_guidance:
            return self._terminal_radius_direction(rmag, r_hat, t_hat, v_radial, vt,
                                                   float(mass_kg)), v_c
        lam = (v_t_target * t_hat - v_inertial
               + self.spec.gravity_term_gain * g_r * tau * r_hat)
        if self.spec.climb_time_to_go_gain > 0.0 and rmag < r_t:
            v_climb_cap = v_c * math.tan(math.radians(self.spec.command_alpha_limit_deg))
            v_r_cmd = min(self.spec.climb_time_to_go_gain * (r_t - rmag) / max(tau, 1e-3),
                          v_climb_cap)
            lam = lam + v_r_cmd * r_hat
        return lam, v_c

    #: Simpson nodes of the E10-08 time-to-go quadrature (even).
    _RADIAL_NODES = 16

    def _radial_plan(self, t_go: float, tau: float, ve: float, rmag: float,
                     v_radial: float, g_net: float) -> tuple[float, float]:
        """E10-08 radial plan: the coefficients ``(A, B)`` of the vertical
        thrust fraction ``s(t) = A + B t`` (the sine of the thrust
        direction's elevation over the local horizontal) that lands ``r(T) =
        r_target`` with ``v_r(T) = 0`` after a burn of ``T = t_go`` seconds,
        for a constant-thrust stage draining linearly under a constant net
        vertical gravity ``g_net`` -- the PEG thrust integrals ``L, J, S,
        Q`` of the linear-tangent family, closed form."""
        r_t = self.spec.target.radius_m
        big_l = -ve * math.log(1.0 - t_go / tau)
        big_j = big_l * tau - ve * t_go
        big_s = big_l * t_go - big_j
        big_q = big_s * tau - 0.5 * ve * t_go * t_go
        rhs_v = -v_radial - g_net * t_go
        rhs_r = r_t - rmag - v_radial * t_go - 0.5 * g_net * t_go * t_go
        det = big_l * big_q - big_j * big_s
        if abs(det) < 1e-12:
            return 0.0, 0.0
        return ((rhs_v * big_q - big_j * rhs_r) / det,
                (big_l * rhs_r - big_s * rhs_v) / det)

    def _horizontal_gain(self, t_go: float, tau: float, ve: float, rmag: float,
                         v_radial: float, g_net: float) -> tuple[float, float]:
        """Horizontal speed the radial plan over ``t_go`` buys (Simpson over
        ``sqrt(1 - s^2) a(t)``, ``s`` clipped to the realisable [-1, 1]) and
        the plan's current ``s``."""
        a_c, b_c = self._radial_plan(t_go, tau, ve, rmag, v_radial, g_net)
        n = self._RADIAL_NODES
        h = t_go / n
        acc = 0.0
        for k in range(n + 1):
            t = k * h
            w = 1.0 if k in (0, n) else (4.0 if k % 2 else 2.0)
            sk = min(max(a_c + b_c * t, -1.0), 1.0)
            acc += w * math.sqrt(1.0 - sk * sk) * ve / (tau - t)
        return acc * h / 3.0, a_c

    def _terminal_radius_direction(self, rmag: float, r_hat: np.ndarray,
                                   t_hat: np.ndarray, v_radial: float,
                                   v_tangent: float, mass_kg: float) -> np.ndarray:
        """E10-08 thrust direction (unit vector, ECEF components): the
        current point ``s = A`` of the radial plan (:meth:`_radial_plan`),
        its time-to-go solved (bisection) so that the plan's horizontal
        thrust integral buys exactly the missing horizontal speed ``v_c -
        v_t`` (the whole remaining burn when even that falls short).  Below
        ``terminal_radius_t_go_floor_s`` the plan is ill-conditioned (it
        divides by ``T^2``) and the law latches the terminal
        velocity-to-be-gained steering ``(v_c - v_t) t_hat - v_r r_hat``
        (module docstring)."""
        mu = MU_EARTH_M3_S2
        r_t = self.spec.target.radius_m
        v_c = math.sqrt(mu / r_t)
        if not self._radial_terminal:
            ve = self.spec.isp_vacuum_s * G0_M_S2
            thrust = self.spec.rated_thrust_vacuum_n
            mass = max(float(mass_kg), self.spec.dry_mass_kg * (1.0 + 1e-9))
            tau = mass * ve / thrust
            burn_left = min((mass - self.spec.dry_mass_kg) * ve / thrust,
                            0.999 * tau)
            # net vertical gravity averaged between now and a circular
            # arrival (where gravity and centripetal term cancel)
            g_net = -0.5 * (mu / (rmag * rmag) - v_tangent * v_tangent / rmag)
            dv_h = max(v_c - v_tangent, 0.0)
            lo, hi = 1e-3, max(burn_left, 2e-3)
            gain_hi, s_hi = self._horizontal_gain(hi, tau, ve, rmag, v_radial, g_net)
            if gain_hi <= dv_h:
                t_go, s = hi, s_hi
            else:
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if self._horizontal_gain(mid, tau, ve, rmag, v_radial, g_net)[0] < dv_h:
                        lo = mid
                    else:
                        hi = mid
                t_go = hi
                s = self._horizontal_gain(t_go, tau, ve, rmag, v_radial, g_net)[1]
            if t_go >= float(self.spec.terminal_radius_t_go_floor_s):
                s = min(max(s, -1.0), 1.0)
                return math.sqrt(1.0 - s * s) * t_hat + s * r_hat
            self._radial_terminal = True
        lam = (v_c - v_tangent) * t_hat - v_radial * r_hat
        norm = float(np.linalg.norm(lam))
        return lam / norm if norm > 1e-9 else t_hat

    def _capped_transfer_speed(self, rmag: float, mass_kg: float) -> float:
        """E6-05: the tangential speed of the transfer ellipse whose apogee
        is the largest the remaining propellant can still circularise,
        capped at ``r_target - transfer_apogee_bias_m`` (module docstring).
        With the reserve exhausted the target degenerates to the circular
        speed at the current radius -- the law then holds station, and the
        mission fails on capability with a clean energy bookkeeping, not on
        guidance."""
        mu = MU_EARTH_M3_S2
        r_t = self.spec.target.radius_m - float(self.spec.transfer_apogee_bias_m)

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

    def _alpha_limit_deg(self, r: np.ndarray, v_ecef: np.ndarray) -> float:
        """The pitch-bias limit at this state: the fixed
        ``command_alpha_limit_deg``, or -- with the E10-08 q-alpha limiter
        on -- ``q_alpha / q`` clipped to ``[command_alpha_limit_deg,
        vacuum_alpha_limit_deg]`` at the live US76 dynamic pressure on the
        air-relative (ECEF) speed."""
        base = float(self.spec.command_alpha_limit_deg)
        qa = float(self.spec.command_q_alpha_limit_pa_deg)
        if qa <= 0.0:
            return base
        from orbitopt.flightlab.atmosphere import density_us76_kg_m3, dynamic_pressure_pa

        _, _, alt_m = frames.ecef_to_geodetic(r)
        q = dynamic_pressure_pa(density_us76_kg_m3(max(float(alt_m), 0.0)),
                                float(np.linalg.norm(v_ecef)))
        ceiling = float(self.spec.vacuum_alpha_limit_deg)
        if q <= qa / ceiling:
            return ceiling
        return min(max(qa / q, base), ceiling)

    def dcm_from_state(self, t_s: float, position_m, velocity_m_s,
                       mass_kg: float) -> np.ndarray:
        """Command ``Tec2b`` at the live state: fly along Lambda, with the
        command rate-limited (``command_slew_rate_deg_s``) -- the direction
        of a vanishing velocity-to-be-gained is numerically meaningless and
        a tracker must never be asked to flip 180 deg in one step."""
        r = np.asarray(position_m, dtype=float)
        v_inertial = (np.asarray(velocity_m_s, dtype=float)
                      + np.cross(_OMEGA, r))
        if (self.spec.terminal_radius_guidance
                and float(t_s) < self.spec.ignition_t_s):
            # E10-08: nothing to steer through an unpowered coast (the
            # trackers are frozen through it); hold prograde so every backend
            # starts the slew-limited command from the same place at ignition.
            lam = v_inertial
        else:
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
        alpha_lim = self._alpha_limit_deg(r, np.asarray(velocity_m_s, dtype=float))
        pitch_deg = vel_p + min(max(pitch_deg - vel_p, -alpha_lim), alpha_lim)
        dh = (heading_deg - vel_h + 180.0) % 360.0 - 180.0
        heading_deg = (vel_h + min(max(dh, -self.spec.command_heading_limit_deg),
                                  self.spec.command_heading_limit_deg)) % 360.0
        if self._cmd_hp is None and self.spec.terminal_radius_guidance:
            # E10-08: seed the slew limiter on the velocity vector (where a
            # separated, frozen-tracker body points) instead of letting the
            # first command jump straight onto the radial plan -- a
            # tens-of-degrees step a point mass follows for free and a rigid
            # body does not.
            self._cmd_hp = (vel_h, vel_p)
            self._cmd_t_s = float(t_s)
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
        if self.spec.cutoff_insertion_box:
            target = self.spec.target
            r_peri = self._periapsis_radius(r, v_inertial)
            r_apo = self._apoapsis_radius(r, v_inertial)
            inside = (r_peri >= target.radius_m - target.periapsis_tolerance_m
                      and r_apo <= target.radius_m + target.apoapsis_tolerance_m)
            if self.spec.terminal_radius_guidance:
                # E10-08 settle (module docstring): inside the band, keep
                # burning while the next control sample is predicted to land
                # closer to the target circle than this one.
                err = max(target.radius_m - r_peri, r_apo - target.radius_m)
                last = self._settle
                if last is not None and float(t_s) <= last[0] + _GUIDANCE_T_EPS_S:
                    return 1.0
                self._settle = (float(t_s), err)
                if inside and last is not None:
                    shrink = last[1] - err
                    if 0.0 < shrink <= err:
                        return 1.0
            if inside:
                self.cutoff_t_s = float(t_s)
                return 0.0
            return 1.0
        if self._periapsis_radius(r, v_inertial) >= limit:
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
