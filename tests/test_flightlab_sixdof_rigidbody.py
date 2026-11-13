"""F-02 (#227): the packaged aero model flown through the native 6-DOF plant.

These are the acceptance's third line -- "the aero force/moment are actually
fed into the 6-DOF integration": drag that costs real altitude, a weathervane
moment that closes on the air-relative flow, damping that decays a body rate,
a gimbal command that pitches the airframe -- plus the max-Q baseline check
against the exponential-atmosphere closed form. Runs are short (a minute of
ascent at dt = 0.1 s) so the whole file stays fast.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.flightlab.atmosphere import ExponentialAtmosphere
from orbitopt.flightlab.contract import BodyTrack, FlightRun, validate_run, write_run
from orbitopt.flightlab.sixdof.aero import load_packaged_aero
from orbitopt.flightlab.sixdof.rigidbody import (
    RUN_COLUMNS,
    ControlCommand,
    RigidBodyEngine,
    RigidBodyPhysics,
    RigidBodyState,
    RigidBodyVehicle,
    constant_control,
    propagate,
)

pytestmark = pytest.mark.slow  # real RK4 propagations: the file takes minutes


def vehicle(*, aero=True, dry_mass_kg=10_000.0, propellant_mass_kg=60_000.0,
            thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0, isp_sea_level_s=282.0,
            mount_x_cg_m=22.0) -> RigidBodyVehicle:
    engine = RigidBodyEngine(thrust_vacuum_n=thrust_vacuum_n, isp_vacuum_s=isp_vacuum_s,
                             isp_sea_level_s=isp_sea_level_s, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=mount_x_cg_m)
    return RigidBodyVehicle(name="spike_stage", dry_mass_kg=dry_mass_kg,
                            propellant_mass_kg=propellant_mass_kg,
                            inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                            aero=load_packaged_aero("spike_stage") if aero else None)


def pad_state(**kw) -> RigidBodyState:
    kw.setdefault("lat_geod_deg", 28.5)
    kw.setdefault("lon_deg", -80.6)
    kw.setdefault("mass_kg", 70_000.0)
    return RigidBodyState.from_launch(**kw)


def test_run_reports_the_columns_and_an_energetic_ascent_with_a_max_q():
    run = propagate(pad_state(), vehicle(), constant_control(1.0), 60.0, dt_s=0.1,
                    sample_every=5)
    assert run.columns == RUN_COLUMNS
    assert len(run) == 121                                # 600 steps, one row in five
    assert run.terminal == "t_end"
    assert run["throttle"][0] == pytest.approx(1.0)
    assert run["altitude_m"][-1] > 15_000.0
    peak = run.max_q
    assert 5.0 < peak.t_s < 55.0                      # not pinned to the edges
    assert 20_000.0 < peak.q_pa < 100_000.0            # a booster's max-Q band
    assert 2_000.0 < peak.altitude_m < 20_000.0
    assert 1.0 < peak.mach < 2.2                          # transonic-to-supersonic
    # the reported max-Q is a sample, and it is the largest sample
    assert peak.q_pa == pytest.approx(float(np.max(run.q_bar_pa)))
    assert run.alpha_max_deg < 1.0                    # a clean vertical climb


def test_drag_comes_out_of_the_thrust_it_costs_altitude():
    kwargs = dict(dt_s=0.1, sample_every=5)
    with_aero = propagate(pad_state(), vehicle(), constant_control(1.0), 50.0, **kwargs)
    no_aero = propagate(pad_state(), vehicle(aero=False), constant_control(1.0), 50.0, **kwargs)
    assert float(np.max(np.abs(no_aero["aero_fx_N"]))) == 0.0
    # drag is body -x: retarding, and real by max-Q
    peak_idx = int(np.argmax(with_aero.q_bar_pa))
    assert with_aero["aero_fx_N"][peak_idx] < -200_000.0
    # and it is not bookkeeping-only: the aero run flies lower at every sample
    # after ignition (the same throttle line, one force term apart)
    assert np.all(with_aero["altitude_m"][1:] < no_aero["altitude_m"][1:])
    assert no_aero["altitude_m"][-1] - with_aero["altitude_m"][-1] > 1_000.0
    # same trajectory otherwise: the aero-free run still passes max-Q plumbing
    assert no_aero.max_q.q_pa > 20_000.0


def test_the_air_weathervanes_the_vehicle_and_the_moment_opposes_the_aoa():
    # nose pitched 5 deg off the climb vector: a small AoA to fight (at pitch
    # exactly 90 a horizontal offset is pure sideslip -- heading is degenerate
    # overhead -- so the pitch story needs a tilted attitude to live in)
    state = pad_state(pitch_deg=85.0, v_north_m_s=20.0, v_down_m_s=-300.0)
    run = propagate(state, vehicle(), constant_control(1.0), 20.0, dt_s=0.1, sample_every=5)
    a0 = run["alpha_deg"][0]
    assert abs(a0) > 1.0
    # a stable airframe (cm_alpha < 0) must produce a moment against the AoA...
    assert run["pitch_moment_n_m"][0] * a0 < 0.0
    # ...and the weathervane must turn the vehicle INTO the flow, shrinking it
    assert abs(run["alpha_deg"][-1]) < abs(a0)
    # normal force lifts toward body -z for a positive AoA
    assert run["aero_fz_N"][0] * a0 < 0.0


def test_pitch_damping_opposes_and_then_decays_the_rate():
    # attitude exactly into the flow (alpha = beta = 0), but spinning at q = 0.5 rad/s:
    # cm_q < 0 must push back, and the rate must decay.
    state = pad_state(v_down_m_s=-300.0)
    state = RigidBodyState(t_s=0.0, position_m=state.position_m, velocity_m_s=state.velocity_m_s,
                           quaternion=state.quaternion, omega_rad_s=(0.0, 0.5, 0.0),
                           mass_kg=state.mass_kg)

    def hold(t_s, s):  # pinned throttle, gimbal neutral: the moment is pure aero
        return ControlCommand(throttle=0.6, gimbal_pitch_rad=0.0, gimbal_yaw_rad=0.0)

    run = propagate(state, vehicle(), hold, 15.0, dt_s=0.05, sample_every=10)
    assert abs(run["alpha_deg"][0]) < 0.1 and abs(run["beta_deg"][0]) < 0.1
    assert run["pitch_moment_n_m"][0] < 0.0            # against the +q spin
    assert run["q_rad_s"][-1] < run["q_rad_s"][0]      # ...and the spin decays
    assert run["q_rad_s"][-1] < 0.3


def test_a_gimbal_command_pitches_the_airframe():
    state = pad_state()
    run = propagate(state, vehicle(aero=False),
                    constant_control(1.0, gimbal_pitch_rad=0.01), 5.0, dt_s=0.1,
                    sample_every=5)
    # +gimbal_pitch tips the thrust +z at a mount behind the CG -> nose-up +My
    assert run["q_rad_s"][-1] > 0.1
    assert run["q_rad_s"][-1] > run["q_rad_s"][0]      # and it keeps accelerating


def test_max_q_against_the_exponential_atmosphere_closed_form():
    # The exponential-atmosphere model has a closed form: constant net
    # acceleration a, h = a t^2 / 2, V = a t, so q = q0 t^2 exp(-a t^2 / 2H)
    # peaks at t* = sqrt(2H/a) with q* = rho0 a H / e. Compare the plant
    # against the formula evaluated at the run's OWN mean acceleration (the
    # burn-down makes a grow; the mean is the right a to hand the closed form).

    # baseline second: the recorded state, the recorded column, one force term
    light = vehicle(thrust_vacuum_n=1.05e6, dry_mass_kg=30_000.0, propellant_mass_kg=40_000.0)
    physics = RigidBodyPhysics(j2=False, atmosphere_model=ExponentialAtmosphere())
    run = propagate(pad_state(mass_kg=70_000.0), light, constant_control(1.0), 120.0,
                    dt_s=0.05, sample_every=20, physics=physics)
    peak = run.max_q
    # plumbing first: the column is 0.5 rho V^2 of the recorded state, every row
    # (rotation on: a co-rotating atmosphere makes the ECEF speed the air speed)
    airspeed = np.sqrt(run["vx_m_s"] ** 2 + run["vy_m_s"] ** 2 + run["vz_m_s"] ** 2)
    rho = np.array([physics.atmosphere_model.density_kg_m3(h) for h in run["altitude_m"]])
    assert run["q_bar_pa"] == pytest.approx(0.5 * rho * airspeed ** 2, rel=1e-9)
    # baseline second: mean acceleration over [0, t_q]; the burn-down makes a
    # grow, h(t) runs convex to the closed form, so the run's max-Q lands a bit
    # later and hotter than the constant-a formula -- 35 % is the honest band
    a_bar = float(airspeed[int(np.argmax(run.q_bar_pa))]) / peak.t_s
    h_scale = ExponentialAtmosphere().scale_height_m
    t_closed = math.sqrt(2.0 * h_scale / a_bar)
    q_closed = ExponentialAtmosphere().density_sea_level_kg_m3 * a_bar * h_scale / math.e
    assert peak.t_s == pytest.approx(t_closed, rel=0.35)
    assert peak.q_pa == pytest.approx(q_closed, rel=0.35)


def test_ground_contact_stops_the_run_at_the_surface():
    physics = RigidBodyPhysics(atmosphere=False, rotation=False, j2=False)
    state = pad_state(v_down_m_s=-800.0)
    run = propagate(state, vehicle(aero=False, propellant_mass_kg=0.0),
                    constant_control(0.0), 200.0, dt_s=0.2, sample_every=10, physics=physics)
    assert run.terminal == "ground"
    t_ballistic = 2.0 * 800.0 / 9.80665
    assert run.ground_t_s == pytest.approx(t_ballistic, rel=0.02)  # g droop lengthens it
    assert (run["altitude_m"] > 0.0).all()                          # never underground
    kinds = [event.kind for event in run.events()]
    assert kinds == ["max_q", "impact"]
    assert run.events()[1].t_s == pytest.approx(run.ground_t_s)


def test_burnout_floors_at_dry_mass_and_shuts_the_throttle():
    v = vehicle(propellant_mass_kg=6_000.0)
    run = propagate(pad_state(mass_kg=v.mass_wet_kg), v, constant_control(1.0),
                    30.0, dt_s=0.1, sample_every=5)
    dry = run.vehicle.dry_mass_kg
    t_dry = run["t_s"][run["mass_kg"] <= dry + 1e-6][0]
    assert 5.0 < run.propellant_exhausted_t_s < 30.0   # plausibly ~6 t at ~400 kg/s mean
    assert run.propellant_exhausted_t_s == pytest.approx(t_dry, abs=0.25)
    assert run["mass_kg"][-1] == pytest.approx(dry)    # it never flies a negative mass
    after = run["t_s"] > run.propellant_exhausted_t_s + 0.1
    assert np.all(run["throttle"][after] == 0.0)
    assert np.all(run["thrust_n"][after] == 0.0)


def test_the_quaternion_stays_a_rotation():
    run = propagate(pad_state(), vehicle(), constant_control(1.0), 60.0, dt_s=0.1,
                    sample_every=5)
    norm = np.sqrt(run["q_w"] ** 2 + run["q_x"] ** 2 + run["q_y"] ** 2 + run["q_z"] ** 2)
    assert norm == pytest.approx(np.ones_like(norm), abs=1e-9)


def test_to_body_track_round_trips_through_the_contract(tmp_path):
    run = propagate(pad_state(), vehicle(), constant_control(1.0), 30.0, dt_s=0.1,
                    sample_every=10)
    track = run.to_body_track("booster")
    assert isinstance(track, BodyTrack)
    assert track.frame == "ECEF"
    assert len(track.t_s) == len(track.mass_kg) == len(track.position_m) == len(run)
    assert {"q_bar_pa", "alpha_deg", "pitch_moment_n_m"} <= set(track.extra_columns)
    flight_run = FlightRun(run_id="fl602-smoke", target_sha256="0" * 64,
                           vehicle_sha256="1" * 64, git_sha="2" * 40,
                           created_utc=run.epoch_utc.isoformat().replace("+00:00", "Z"),
                           engines={"ascent": "orbitopt.sixdof.rigidbody"},
                           tracks=(track,), events=tuple(run.events()))
    write_run(tmp_path / "run", flight_run)
    validate_run(tmp_path / "run")
    assert (tmp_path / "run" / "manifest.json").exists()
