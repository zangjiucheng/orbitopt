"""F-04 (#229): the TVC actuator model and the basic attitude tracker.

The acceptance's three lines each get their own kind of evidence: the
actuator's lag / slew / saturation are tested *as a unit* (exact ramps against
the analytic one-step values), saturation and slew-limiting are then covered
*in the loop* (a fast attitude program and an aggressive initial offset run
the command into the travel limit and the rate limit, and the recorded
seconds must be non-zero), and the aero ascent must track its program without
diverging while the deflection margin is recorded in the summary.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.flightlab.sixdof.aero import load_packaged_aero
from orbitopt.flightlab.sixdof.rigidbody import (
    RigidBodyEngine,
    RigidBodyState,
    RigidBodyVehicle,
    dcm_from_quaternion,
    propagate,
)
from orbitopt.flightlab.sixdof.tvc import (
    AttitudeTvcLaw,
    TvcActuator,
    TvcGains,
    WaypointAttitudeProgram,
    attitude_error_body,
    run_closed_loop_ascent,
)

pytestmark = pytest.mark.slow  # real RK4 closed-loop propagations


def vehicle(*, aero=True, dry_mass_kg=10_000.0, propellant_mass_kg=60_000.0,
            gimbal_limit_rad=0.0873) -> RigidBodyVehicle:
    engine = RigidBodyEngine(thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0,
                             isp_sea_level_s=282.0, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=22.0,
                             gimbal_limit_rad=gimbal_limit_rad)
    return RigidBodyVehicle(name="spike_stage", dry_mass_kg=dry_mass_kg,
                            propellant_mass_kg=propellant_mass_kg,
                            inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                            aero=load_packaged_aero("spike_stage") if aero else None)


def pad_state(**kw) -> RigidBodyState:
    kw.setdefault("lat_geod_deg", 28.5)
    kw.setdefault("lon_deg", -80.6)
    kw.setdefault("mass_kg", 70_000.0)
    return RigidBodyState.from_launch(**kw)


def nose_tilt_deg(samples, i: int = -1) -> float:
    """Angle between the body x axis and the local vertical, degrees."""
    up = np.array([samples["x_m"][i], samples["y_m"][i], samples["z_m"][i]])
    up = up / np.linalg.norm(up)
    q = np.array([samples["q_w"][i], samples["q_x"][i],
                  samples["q_y"][i], samples["q_z"][i]])
    nose = dcm_from_quaternion(q).T @ np.array([1.0, 0.0, 0.0])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(nose @ up)))))


# --------------------------------------------------------------------------
# the actuator itself
# --------------------------------------------------------------------------

def test_actuator_saturates_command_and_state_to_the_travel_limit():
    act = TvcActuator(limit_rad=0.05, max_slew_rate_rad_s=1.0, tau_s=0.02)
    assert act.saturate(9.9) == pytest.approx(0.05)
    assert act.saturate(-9.9) == pytest.approx(-0.05)
    # a state already beyond the travel (a gust of the imagination) is floored
    delta, _ = act.step(0.9, 0.05, 1e-6)
    assert delta <= 0.05 + 1e-12


def test_actuator_first_order_lag_chases_the_saturated_command():
    act = TvcActuator(max_slew_rate_rad_s=10.0, tau_s=0.10)  # slew never bites
    cmd = 0.02
    delta, _ = act.step(0.0, cmd, act.tau_s)
    # one time constant: e^-1 of the gap left (Euler substepping, so a few %)
    assert delta == pytest.approx(cmd * (1.0 - math.exp(-1.0)), rel=0.15)
    # and it converges to the command itself, not past it
    for _ in range(80):
        delta, _ = act.step(delta, cmd, act.tau_s)
    assert delta == pytest.approx(cmd, rel=1e-9)


def test_actuator_slew_rate_caps_the_ramp_exactly():
    act = TvcActuator(limit_rad=0.0873, max_slew_rate_rad_s=0.10, tau_s=0.10)
    # lag alone would demand 0.873 rad/s at the step; the rate must come out
    # at exactly the limit, so the deflection is the ramp slew*dt
    delta, slew_limited = act.step(0.0, 0.0873, 0.3)
    assert slew_limited
    assert delta == pytest.approx(0.10 * 0.3, abs=1e-12)
    # a demand the lag can service is not slew-limited
    delta, slew_limited = act.step(0.087, 0.0873, 0.02)
    assert not slew_limited


def test_actuator_rejects_unphysical_parameters():
    with pytest.raises(ValueError):
        TvcActuator(limit_rad=0.0)
    with pytest.raises(ValueError):
        TvcActuator(max_slew_rate_rad_s=-1.0)
    with pytest.raises(ValueError):
        TvcActuator(tau_s=0.0)


# --------------------------------------------------------------------------
# the attitude program
# --------------------------------------------------------------------------

def test_program_interpolates_linearly_and_freezes_off_the_endpoints():
    prog = WaypointAttitudeProgram(times_s=(0.0, 10.0), heading_deg=(90.0, 90.0),
                                   pitch_deg=(90.0, 45.0), roll_deg=(0.0, 0.0))
    h, p, r, hd, pd, rd = prog.angles_at(5.0)
    assert (h, p, r) == (90.0, 67.5, 0.0)
    assert (hd, pd, rd) == (0.0, -4.5, 0.0)
    assert prog.angles_at(-3.0)[1] == pytest.approx(90.0)      # before: hold first
    assert prog.angles_at(99.0)[1] == pytest.approx(45.0)      # after: hold last
    assert prog.angles_at(99.0)[4] == pytest.approx(0.0)       # no frozen slope
    with pytest.raises(ValueError):
        WaypointAttitudeProgram((0.0, 1.0), (90.0,), (90.0, 30.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        WaypointAttitudeProgram((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def test_program_feed_forward_is_the_321_body_rate_of_the_waypoint_slopes():
    # heading constant, roll zero: the 3-2-1 kinematics collapse to q = thetadot
    prog = WaypointAttitudeProgram(times_s=(0.0, 10.0), heading_deg=(0.0, 0.0),
                                   pitch_deg=(90.0, 45.0), roll_deg=(0.0, 0.0))
    ff = prog.omega_ff_body(5.0)
    assert ff[0] == pytest.approx(0.0, abs=1e-12)
    assert ff[2] == pytest.approx(0.0, abs=1e-12)
    assert ff[1] == pytest.approx(-math.radians(4.5), rel=1e-9)


def test_attitude_error_follows_the_fl403_body_convention():
    # launched 3 deg downrange of vertical: the tracker must say "nose UP"
    state = pad_state(pitch_deg=87.0)
    e = attitude_error_body(state, WaypointAttitudeProgram.vertical())
    assert math.degrees(float(e[1])) == pytest.approx(3.0, abs=1e-6)
    assert abs(float(e[0])) < 1e-6 and abs(float(e[2])) < 1e-6


# --------------------------------------------------------------------------
# the law's wiring
# --------------------------------------------------------------------------

def test_law_refuses_an_actuator_that_outruns_the_plants_own_gimbal_clamp():
    veh = vehicle(gimbal_limit_rad=0.05)
    with pytest.raises(ValueError, match="gimbal limit"):
        AttitudeTvcLaw(vehicle=veh, program=WaypointAttitudeProgram.vertical())
    with pytest.raises(ValueError):
        AttitudeTvcLaw(vehicle=vehicle(), program=WaypointAttitudeProgram.vertical(),
                       dt_c_s=0.0)


def test_law_freezes_and_neutralises_after_engine_cut_off():
    # 500 kg of propellant: burnout inside the first second, then the tracker
    # must trim its gimbals home instead of chasing an error no thrust can fix
    veh = vehicle(dry_mass_kg=10_000.0, propellant_mass_kg=500.0)
    law = AttitudeTvcLaw(vehicle=veh, program=WaypointAttitudeProgram.vertical(),
                         dt_c_s=0.02)
    run = propagate(pad_state(mass_kg=10_500.0, pitch_deg=85.0), veh, law, 8.0,
                    dt_s=0.05, sample_every=4)
    assert run.terminal == "t_end"
    hist = law.history()
    peak = float(np.max(np.abs(hist["delta_pitch_rad"])))
    assert peak > 1e-3                                       # it did steer first
    assert abs(float(hist["delta_pitch_rad"][-1])) < 1e-9    # then gave up politely
    assert abs(float(hist["delta_yaw_rad"][-1])) < 1e-9


# --------------------------------------------------------------------------
# closed loop: the acceptance's ascent, saturation and slew coverage
# --------------------------------------------------------------------------

def test_aero_ascent_tracks_the_vertical_program_with_margin_recorded():
    prog = WaypointAttitudeProgram.vertical()
    ascent = run_closed_loop_ascent(pad_state(pitch_deg=89.5), vehicle(), prog, 40.0,
                                    dt_s=0.05, sample_every=5)
    assert ascent.run.terminal == "t_end"
    # no unexpected divergence: the 0.5 deg pad offset is fought off and the
    # whole flight never swings more than a degree off the program
    assert ascent.metrics.att_err_max_deg < 1.0
    # the residual is the slow drift of the *unactuated* azimuth-about-vertical
    # degree of freedom (two gimbals steer pitch and pitch-plane yaw only);
    # it stays well under a degree over the whole burn
    assert ascent.metrics.att_err_rms_deg < 0.5
    assert nose_tilt_deg(ascent.run.samples) < 0.3           # thrust line back near vertical
    assert ascent.metrics.alpha_max_deg < 1.0
    # and the tracker did it far inside the travel: the margin is recorded
    s = ascent.summary()
    assert s["pitch_deflection_margin"] > 0.9
    assert s["yaw_deflection_margin"] > 0.9
    assert s["pitch_saturated_s"] == 0.0 and s["yaw_saturated_s"] == 0.0
    assert s["tvc_limit_deg"] == pytest.approx(math.degrees(0.0873))
    assert s["pitch_applied_peak_deg"] < s["tvc_limit_deg"]
    assert float(ascent.run["altitude_m"][-1]) > 8_000.0     # real ascent, not a hover
    assert s["burn_s"] == pytest.approx(40.0, rel=1e-6)


def test_fast_attitude_program_proves_the_slew_limit_bites_in_the_loop():
    # a 15 deg/s pitch program outruns a 5.73 deg/s actuator by design: the
    # command must be rate-limited for seconds, and the *applied* deflection
    # must never be seen moving faster than the ceiling
    prog = WaypointAttitudeProgram(times_s=(0.0, 4.0), heading_deg=(90.0, 90.0),
                                   pitch_deg=(90.0, 30.0), roll_deg=(0.0, 0.0))
    veh = vehicle()
    law = AttitudeTvcLaw(vehicle=veh, program=prog, dt_c_s=0.02)
    run = propagate(pad_state(pitch_deg=90.0), veh, law, 30.0, dt_s=0.02, sample_every=5)
    d = law.diagnostics
    assert d.pitch_slew_limited_s > 0.5
    assert d.pitch_saturated_s == 0.0                        # travel was never the binding one
    hist = law.history()
    rate = np.abs(np.diff(hist["delta_pitch_rad"]) / np.diff(hist["t_s"]))
    slew = law.actuator.max_slew_rate_rad_s
    assert float(np.max(rate)) <= slew * (1.0 + 1e-6)         # never exceeded, ...
    assert float(np.max(rate)) == pytest.approx(slew, rel=1e-6)  # ... and truly at the cap
    assert d.pitch_applied_peak_deg < math.degrees(law.actuator.limit_rad)


def test_aggressive_initial_offset_proves_saturation_in_the_loop():
    # big body-axis offset with hot gains: the PID demands more than the
    # travel has; the command pins at the limit, the applied deflection rides
    # it, and the margin records exactly that
    #
    # Two-gimbal TVC cannot hold the roll/azimuth-about-vertical axis, so a
    # hard pitch-over through vertical leaves the vehicle yawed in azimuth by
    # whatever the transient handed it (the vector error norm ends near 38
    # deg here). That spin is *not* flight divergence: the thrust line and the
    # pitch rate converge back onto the program, and that is what the
    # acceptance's "no unexpected divergence" asks for -- so the assertions
    # are on tilt, rates and the saturation bookkeeping, not on the spin.
    prog = WaypointAttitudeProgram.vertical()
    ascent = run_closed_loop_ascent(pad_state(pitch_deg=80.0), vehicle(),
                                    prog, 40.0, dt_s=0.01, sample_every=100,
                                    gains=TvcGains(kp=15.0, ki=0.02, kd=8.0))
    d = ascent.diagnostics
    lim_deg = math.degrees(0.0873)
    assert d.pitch_saturated_s > 0.0
    assert d.pitch_command_peak_deg == pytest.approx(lim_deg, abs=1e-9)
    # the applied deflection is eaten by the lag and the rate limit: the
    # command pins at the travel but the servo never gets the whole stroke
    assert d.pitch_applied_peak_deg < d.pitch_command_peak_deg - 1e-3
    assert 0.0 < d.pitch_deflection_margin < 0.9
    # saturated is not divergent: the 10 deg nose-off start is fought off,
    # the tilt settles on the vertical and the pitch rate dies out
    assert ascent.metrics.att_err_max_deg < 45.0
    s = ascent.run.samples
    tilts = np.array([nose_tilt_deg(s, i) for i in range(s["t_s"].size)])
    assert float(tilts.max()) < 12.0                         # never worse than the start
    assert float(tilts[-1]) < 1.0                            # converged onto vertical
    assert abs(math.degrees(float(s["q_rad_s"][-1]))) < 0.01  # pitch rate killed
    assert float(s["altitude_m"][-1]) > 8_000.0              # still flying, not tumbling
    assert ascent.summary()["burn_s"] == pytest.approx(40.0, rel=1e-6)
