"""E10-08 (task #305): the terminal-radius radial channel, the settle cutoff
and the q-alpha limiter of the shared upper-stage law, unit by unit.  The
matrix-level result (13/17 inserted, no unexpected verdict) is locked by
``test_flightlab_e6_03_guidance_matrix.py`` against the committed study."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from orbitopt.entry import G0_M_S2, MU_EARTH_M3_S2, R_EARTH_M
from orbitopt.flightlab.guidance.upper_stage import (
    ParkingOrbitTarget,
    UpperStageGuidanceSpec,
)

THRUST_N = 220e3
ISP_S = 340.0
DRY_KG = 1_000.0


def _spec(alt_km: float = 260.0, **kw) -> UpperStageGuidanceSpec:
    return UpperStageGuidanceSpec(
        target=ParkingOrbitTarget(altitude_m=alt_km * 1e3), dry_mass_kg=DRY_KG,
        rated_thrust_vacuum_n=THRUST_N, isp_vacuum_s=ISP_S, ignition_t_s=0.0,
        terminal_radius_guidance=True, cutoff_insertion_box=True, **kw)


def _fly_plan_open_loop(g, t_go, tau, ve, r0, vr0, g_net, n=20_000):
    """Integrate the planned radial motion s(t) = A + B t with the plan's own
    thrust and gravity model, to check the closed-form integrals."""
    a_c, b_c = g._radial_plan(t_go, tau, ve, r0, vr0, g_net)
    dt = t_go / n
    r, vr = r0, vr0
    for k in range(n):
        t = (k + 0.5) * dt
        acc = (a_c + b_c * t) * ve / (tau - t) + g_net
        r += vr * dt + 0.5 * acc * dt * dt
        vr += acc * dt
    return r, vr


def test_radial_plan_lands_on_the_target_radius_with_zero_climb_rate():
    g = _spec(260.0).new_guidance()
    ve = ISP_S * G0_M_S2
    mass = 7_000.0
    tau = mass * ve / THRUST_N
    r0, vr0, g_net = R_EARTH_M + 120e3, 2_000.0, -4.0
    r, vr = _fly_plan_open_loop(g, 80.0, tau, ve, r0, vr0, g_net)
    assert r == pytest.approx(g.spec.target.radius_m, abs=50.0)
    assert vr == pytest.approx(0.0, abs=0.5)


def test_steep_handoff_is_steered_below_the_velocity_vector():
    # the E6-05 failure: 2 km/s of climb at ignition, ballistic apogee ~330 km.
    # For a 220 km target the plan must point below the 39 deg flight path.
    g = _spec(220.0).new_guidance()
    vr, v = 2_012.0, 3_211.0
    vh = math.sqrt(v * v - vr * vr)
    d = g._terminal_radius_direction(R_EARTH_M + 92.6e3, np.array([1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), vr, vh, 7_871.0)
    elevation = math.degrees(math.asin(d[0]))
    fpa = math.degrees(math.atan2(vr, vh))
    assert elevation < fpa - 5.0
    assert not g._radial_terminal


def test_terminal_handover_latches_and_points_along_velocity_to_be_gained():
    g = _spec(220.0).new_guidance()
    r_t = g.spec.target.radius_m
    v_c = math.sqrt(MU_EARTH_M3_S2 / r_t)
    # 20 m/s of horizontal speed missing at 6 g: well under the 12 s floor
    d = g._terminal_radius_direction(r_t, np.array([1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), -20.0, v_c - 20.0,
                                     1_200.0)
    assert g._radial_terminal
    assert d[0] == pytest.approx(math.sqrt(0.5), abs=1e-9)  # +20 up, +20 along
    assert d[1] == pytest.approx(math.sqrt(0.5), abs=1e-9)


def _state(alt_m, v_m_s, fpa_deg=0.0):
    r = np.array([R_EARTH_M + alt_m, 0.0, 0.0])
    v_in = v_m_s * np.array([math.sin(math.radians(fpa_deg)), 0.0,
                             math.cos(math.radians(fpa_deg))])
    # the law's input is ECEF velocity; omega x r is along +y here, so feed a
    # polar-plane inertial state with that removed
    v_ecef = v_in - np.cross(np.array([0.0, 0.0, 7.2921159e-5]), r)
    return r, v_ecef


def test_settle_cutoff_waits_inside_the_band_while_the_orbit_still_closes():
    spec = _spec(220.0)
    g = spec.new_guidance()
    r_t = spec.target.radius_m
    v_c = math.sqrt(MU_EARTH_M3_S2 / r_t)
    # horizontal at the target radius, approaching circular speed from below:
    # periapsis rises into the band and keeps closing
    # 6 m/s per control sample (~12 g at 0.05 s): -18 and -12 m/s are out of
    # band (periapsis -61 / -41 km), -6 m/s is in (-20 km) and the next step
    # is predicted to close further on the same side -> keep burning; at the
    # circle the error has stopped shrinking by less than itself -> cut there
    speeds = [v_c - 18.0, v_c - 12.0, v_c - 6.0, v_c, v_c + 6.0]
    throttles = []
    for k, sp in enumerate(speeds):
        r, v = _state(220e3, sp)
        throttles.append(g.throttle_from_state(1.0 + 0.05 * k, r, v, 1_500.0))
    assert throttles == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert g.cutoff_t_s == pytest.approx(1.15)


def test_settle_cutoff_cuts_at_once_when_the_next_step_would_overshoot():
    spec = _spec(220.0)
    g = spec.new_guidance()
    r_t = spec.target.radius_m
    v_c = math.sqrt(MU_EARTH_M3_S2 / r_t)
    r, v = _state(220e3, v_c - 60.0)
    assert g.throttle_from_state(1.0, r, v, 1_500.0) == 1.0
    # one sample later it is just inside the band, but the error fell by far
    # more than what is left: the next sample lands past the circle
    r, v = _state(220e3, v_c - 6.0)
    assert g.throttle_from_state(1.05, r, v, 1_500.0) == 0.0
    assert g.cutoff_engaged


def test_q_alpha_limiter_keeps_30_deg_in_the_air_and_opens_in_vacuum():
    spec = _spec(220.0, command_q_alpha_limit_pa_deg=30_000.0)
    g = spec.new_guidance()
    # 30 km at 1.2 km/s: q ~ 13 kPa -> the 30 deg floor holds
    r, v = _state(30e3, 1_200.0, 30.0)
    assert g._alpha_limit_deg(r, v) == pytest.approx(30.0)
    # 90 km at 3.2 km/s: q ~ 13 Pa -> fully open
    r, v = _state(90e3, 3_200.0, 30.0)
    assert g._alpha_limit_deg(r, v) == pytest.approx(90.0)
    # off by default: the fixed limit everywhere
    plain = replace(spec, command_q_alpha_limit_pa_deg=0.0).new_guidance()
    assert plain._alpha_limit_deg(r, v) == pytest.approx(30.0)


def test_defaults_leave_the_f07_law_untouched():
    spec = UpperStageGuidanceSpec(
        target=ParkingOrbitTarget(altitude_m=220e3), dry_mass_kg=DRY_KG,
        rated_thrust_vacuum_n=THRUST_N, isp_vacuum_s=ISP_S, ignition_t_s=0.0)
    assert spec.terminal_radius_guidance is False
    assert spec.command_q_alpha_limit_pa_deg == 0.0
    g = spec.new_guidance()
    r, v = _state(200e3, 7_000.0, 5.0)
    lam, _ = g._lambda_ecef(r, v + np.cross(np.array([0, 0, 7.2921159e-5]), r),
                            3_000.0)
    assert np.linalg.norm(lam) > 1.0  # a velocity-to-be-gained, not a unit vector


def test_command_holds_prograde_through_the_coast_and_slews_from_there():
    # every backend must see the same command history from ignition: the
    # point mass first asks at ignition, the rigid body / JSBSim from
    # separation with frozen trackers (module docstring)
    spec = replace(_spec(180.0, command_q_alpha_limit_pa_deg=30_000.0),
                   ignition_t_s=10.0, command_slew_rate_deg_s=10.0)
    r, v = _state(92e3, 3_200.0, 39.0)
    v_in = v + np.cross(np.array([0.0, 0.0, 7.2921159e-5]), r)
    for g in (spec.new_guidance(), spec.new_guidance()):
        g.dcm_from_state(0.0, r, v, 7_900.0)   # coast: held prograde
        g.dcm_from_state(9.95, r, v, 7_900.0)
        _, vel_p = g._heading_pitch(v_in, r)
        assert g._cmd_hp[1] == pytest.approx(vel_p, abs=1e-9)
    late = spec.new_guidance()           # a backend that first asks at ignition
    late.dcm_from_state(10.0, r, v, 7_900.0)
    assert late._cmd_hp[1] == pytest.approx(vel_p, abs=1e-9)
    late.dcm_from_state(10.5, r, v, 7_900.0)
    # then the slew-limited move onto the (pitch-down) radial plan
    assert late._cmd_hp[1] == pytest.approx(vel_p - 5.0, abs=1e-6)
