"""E1-03 (task #252) unit tests for the shared upper-stage guidance law:
target/tolerance validation, insertion grading against the declared
tolerances, the osculating-periapsis cutoff latch, and the command guards
(velocity-frame AoA/sideslip clamps and the attitude-command slew limiter).
Pure geometry -- no propagator runs.

    PYTHONPATH=src python -m pytest tests/test_flightlab_upper_stage_guidance.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.entry import MU_EARTH_M3_S2, OMEGA_EARTH_RAD_S, R_EARTH_M
from orbitopt.flightlab.guidance.upper_stage import (
    ParkingOrbitTarget,
    UpperStageGuidance,
    UpperStageGuidanceSpec,
    insertion_report,
)

ALT_M = 220_000.0
R_T = R_EARTH_M + ALT_M
V_C = math.sqrt(MU_EARTH_M3_S2 / R_T)
INC_DEG = 28.32
_OMEGA = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])


def _spec(**kwargs) -> UpperStageGuidanceSpec:
    kwargs.setdefault("target", ParkingOrbitTarget(altitude_m=ALT_M))
    kwargs.setdefault("dry_mass_kg", 1000.0)
    kwargs.setdefault("rated_thrust_vacuum_n", 440_000.0)
    kwargs.setdefault("isp_vacuum_s", 340.0)
    kwargs.setdefault("ignition_t_s", 100.0)
    return UpperStageGuidanceSpec(**kwargs)


def _state(r_inertial, v_inertial) -> tuple[np.ndarray, np.ndarray]:
    """(r, v_ECEF) for an inertial state: v_ECEF = v_inertial - omega x r."""
    r = np.asarray(r_inertial, dtype=float)
    return r, np.asarray(v_inertial, dtype=float) - np.cross(_OMEGA, r)


def _circular_inertial(alt_m: float = ALT_M, vt_scale: float = 1.0,
                       inc_deg: float = INC_DEG):
    """Inertial circular(-ish) state at the ascending node on +X: the
    tangential speed is ``vt_scale`` times the local circular speed."""
    rmag = R_EARTH_M + alt_m
    r = np.array([rmag, 0.0, 0.0])
    v_circ = math.sqrt(MU_EARTH_M3_S2 / rmag)
    v = vt_scale * v_circ * np.array([0.0, math.cos(math.radians(inc_deg)),
                                      math.sin(math.radians(inc_deg))])
    return r, v


# ---------------------------------------------------------------------------
# target and tolerance declaration
# ---------------------------------------------------------------------------

def test_target_requires_positive_altitude_and_tolerances():
    with pytest.raises(ValueError):
        ParkingOrbitTarget(altitude_m=0.0)
    with pytest.raises(ValueError):
        ParkingOrbitTarget(altitude_m=ALT_M, periapsis_tolerance_m=0.0)
    with pytest.raises(ValueError):
        ParkingOrbitTarget(altitude_m=ALT_M, apoapsis_tolerance_m=-1.0)
    assert ParkingOrbitTarget(altitude_m=ALT_M).radius_m == R_EARTH_M + ALT_M


def test_spec_validation():
    for kw in ({"command_slew_rate_deg_s": 0.0},
               {"command_slew_rate_deg_s": 91.0},
               {"command_alpha_limit_deg": 0.0},
               {"command_alpha_limit_deg": 90.5},
               {"command_heading_limit_deg": 0.0},
               {"command_heading_limit_deg": 30.5},
               {"gravity_term_gain": -0.1},
               {"gravity_term_gain": 2.5},
                {"cutoff_periapsis_margin_m": -1.0},
                {"apogee_ceiling_gain": -1.0e-4},
               {"dry_mass_kg": 0.0},
               {"rated_thrust_vacuum_n": -1.0}):
        with pytest.raises(ValueError):
            _spec(**kw)


# ---------------------------------------------------------------------------
# insertion grading against the declared tolerances
# ---------------------------------------------------------------------------

def test_insertion_report_grades_a_clean_circular_insertion():
    target = ParkingOrbitTarget(altitude_m=ALT_M, inclination_deg=INC_DEG)
    r, v_in = _circular_inertial()
    rep = insertion_report(target, *_state(r, v_in))
    assert rep["inserted"] is True
    assert abs(rep["d_periapsis_m"]) < 1.0 and abs(rep["d_apoapsis_m"]) < 1.0
    assert abs(rep["d_inclination_deg"]) < 1e-6


def test_insertion_report_rejects_an_underfired_orbit():
    target = ParkingOrbitTarget(altitude_m=ALT_M)
    r, v_in = _circular_inertial(vt_scale=0.9)  # deep suborbital perigee
    rep = insertion_report(target, *_state(r, v_in))
    assert rep["inserted"] is False
    assert rep["periapsis_within_tolerance"] is False
    assert rep["d_periapsis_m"] < -target.periapsis_tolerance_m


def test_insertion_report_grades_inclination_only_when_declared():
    r, v_in = _circular_inertial(inc_deg=15.0)
    rep = insertion_report(ParkingOrbitTarget(altitude_m=ALT_M), *_state(r, v_in))
    assert rep["inclination_within_tolerance"] is True and rep["inserted"] is True
    rep = insertion_report(ParkingOrbitTarget(altitude_m=ALT_M,
                                             inclination_deg=INC_DEG),
                          *_state(r, v_in))
    assert rep["inclination_within_tolerance"] is False
    assert rep["inserted"] is False


# ---------------------------------------------------------------------------
# cutoff: scheduled ignition, osculating-periapsis latch, one-way
# ---------------------------------------------------------------------------

def test_throttle_zero_before_ignition_one_during_burn():
    g = _spec().new_guidance()
    r, v = _state(*_circular_inertial(alt_m=150_000.0, vt_scale=0.6))
    assert g.throttle_from_state(95.0, r, v, 8000.0) == 0.0
    assert g.throttle_from_state(110.0, r, v, 8000.0) == 1.0
    assert not g.cutoff_engaged


def test_cutoff_latches_on_osculating_periapsis_and_never_releases():
    g = _spec().new_guidance()
    r, v = _state(*_circular_inertial())  # perigee == target radius
    assert g.throttle_from_state(120.0, r, v, 2000.0) == 0.0
    assert g.cutoff_engaged and g.cutoff_t_s == pytest.approx(120.0)
    # one-way: a later state with a deep perigee does not reopen the valve
    r2, v2 = _state(*_circular_inertial(vt_scale=0.6))
    assert g.throttle_from_state(130.0, r2, v2, 2000.0) == 0.0
    # the attitude holds at its last command after cutoff
    d1 = g.dcm_from_state(140.0, r2, v2, 2000.0)
    d2 = g.dcm_from_state(141.0, r2, v2, 2000.0)
    assert np.allclose(d1, d2, atol=1e-12)


def test_cutoff_margin_lets_the_latch_fire_early():
    g = _spec(cutoff_periapsis_margin_m=5_000.0).new_guidance()
    r, v = _state(*_circular_inertial(alt_m=ALT_M - 3_000.0))  # perigee 3 km short
    assert g.throttle_from_state(120.0, r, v, 2000.0) == 0.0  # >= r_t - 5 km
    assert g.cutoff_engaged
    # ... and a perigee 8 km short still burns
    g2 = _spec(cutoff_periapsis_margin_m=5_000.0).new_guidance()
    r2, v2 = _state(*_circular_inertial(alt_m=ALT_M - 8_000.0))
    assert g2.throttle_from_state(120.0, r2, v2, 2000.0) == 1.0


def test_each_flight_gets_its_own_latch():
    spec = _spec()
    g1, g2 = spec.new_guidance(), spec.new_guidance()
    r, v = _state(*_circular_inertial())
    g1.throttle_from_state(120.0, r, v, 2000.0)
    assert g1.cutoff_engaged and not g2.cutoff_engaged


def test_apogee_ceiling_hold_trims_only_while_the_box_is_armed():
    # perigee at the target radius (peri side satisfied), apogee 40 km over
    # the 25 km ceiling (apo side not): the box is armed and the hold bites.
    band = ParkingOrbitTarget(altitude_m=ALT_M, periapsis_tolerance_m=25_000.0,
                              apoapsis_tolerance_m=25_000.0)
    rp = R_T
    ra = R_T + 40_000.0
    r = np.array([rp, 0.0, 0.0])
    v = math.sqrt(2.0 * MU_EARTH_M3_S2 * ra / (rp * (rp + ra))) \
        * np.array([0.0, math.cos(math.radians(INC_DEG)),
                    math.sin(math.radians(INC_DEG))])
    lam_off, _ = _spec(target=band).new_guidance()._lambda_ecef(r, v, 8000.0)
    lam_on, _ = _spec(target=band, cutoff_insertion_box=True,
                      apogee_ceiling_gain=8.0e-4).new_guidance()._lambda_ecef(
                          r, v, 8000.0)
    trim = 8.0e-4 * (ra - (R_T + 25_000.0))
    dt = lam_on - lam_off
    assert float(np.linalg.norm(dt)) == pytest.approx(trim, rel=1e-6)
    # the correction is retrograde: it removes the tangential over-speed
    t_hat = v / float(np.linalg.norm(v))
    assert float(dt @ t_hat) == pytest.approx(-trim, rel=1e-6)
    # an apogee inside the ceiling (gain on) commands exactly what gain off
    # commands: the hold is inert outside its regime.
    v2 = math.sqrt(MU_EARTH_M3_S2 / rp) * np.array(
        [0.0, math.cos(math.radians(INC_DEG)), math.sin(math.radians(INC_DEG))])
    lam_a, _ = _spec(target=band, cutoff_insertion_box=True,
                     apogee_ceiling_gain=8.0e-4).new_guidance()._lambda_ecef(
                         r, v2, 8000.0)
    lam_b, _ = _spec(target=band).new_guidance()._lambda_ecef(r, v2, 8000.0)
    assert np.allclose(lam_a, lam_b)


# ---------------------------------------------------------------------------
# command guards: velocity-frame clamps and the slew limiter
# ---------------------------------------------------------------------------

def _pitch_heading_of(g, t_s, r, v):
    """Extract (heading, pitch) [deg] of the commanded body +X (thrust axis)
    in ECEF at the state's site from the returned Tec2b."""
    dcm = g.dcm_from_state(t_s, r, v, 8000.0)
    thr = dcm.T @ np.array([1.0, 0.0, 0.0])
    from orbitopt.flightlab import frames
    from orbitopt.flightlab.sixdof.rigidbody import _ned2ecef

    lat, lon, _ = frames.ecef_to_geodetic(np.asarray(r))
    d_ned = _ned2ecef(math.radians(float(lat)), math.radians(float(lon))).T @ thr
    return (math.degrees(math.atan2(float(d_ned[1]), float(d_ned[0]))) % 360.0,
            math.degrees(math.atan2(-float(d_ned[2]),
                                    math.hypot(float(d_ned[0]), float(d_ned[1])))))


def _burn_state(tangential_m_s: float, vertical_m_s: float, alt_m: float):
    """Ascending east-ish inertial state: (vertical, tangential*cos i,
    tangential*sin i) at the node on +X."""
    rmag = R_EARTH_M + alt_m
    r = np.array([rmag, 0.0, 0.0])
    v = np.array([vertical_m_s,
                  tangential_m_s * math.cos(math.radians(INC_DEG)),
                  tangential_m_s * math.sin(math.radians(INC_DEG))])
    return _state(r, v)


def test_command_is_clamped_to_the_velocity_frame():
    # tangential speed well past the target circular speed: the raw Lambda
    # flips backwards; without the clamp the command would demand >90 deg
    # of pitch off the velocity vector (a normal-force regime the point-mass
    # twin cannot represent)
    g = _spec(command_alpha_limit_deg=30.0, command_heading_limit_deg=3.0).new_guidance()
    r, v = _burn_state(7_900.0, 600.0, 80_000.0)
    h_cmd, p_cmd = _pitch_heading_of(g, 101.0, r, v)
    v_inertial = np.asarray(v) + np.cross(_OMEGA, r)
    h_vel, p_vel = g._heading_pitch(v_inertial, r)
    assert abs(p_cmd - p_vel) <= 30.0 + 1e-6
    assert abs((h_cmd - h_vel + 180.0) % 360.0 - 180.0) <= 3.0 + 1e-6


def test_command_respects_the_slew_rate():
    g = _spec(command_slew_rate_deg_s=3.0).new_guidance()
    r, v = _burn_state(7_900.0, 600.0, 80_000.0)
    d1 = g.dcm_from_state(101.0, r, v, 8000.0)
    # the trajectory flips 180 deg downrange in one second
    r2, v2 = _burn_state(7_900.0, 600.0, 80_000.0)
    v_flip = np.asarray(v2)
    t_hat = np.array([0.0, math.cos(math.radians(INC_DEG)),
                      math.sin(math.radians(INC_DEG))])
    radial = v_flip[1] * t_hat
    r2, v2 = _state(np.array([R_EARTH_M + 80_000.0, 0.0, 0.0]),
                    np.array([600.0, 0.0, 0.0]) - radial)
    d2 = g.dcm_from_state(102.0, r2, v2, 8000.0)
    rot = float(np.clip((np.trace(d1 @ d2.T) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(rot))
    # moved (the raw command flips ~180 deg) but rate-limited: the limiter
    # caps heading AND pitch at 3 deg/s, so one second allows at most
    # ~2x the per-axis step of 3D rotation
    assert 1.0 < angle_deg <= 2.0 * 3.0 + 1e-6


def test_first_command_is_the_clamped_lambda_not_a_ramp():
    g = _spec(command_slew_rate_deg_s=3.0).new_guidance()
    r, v = _burn_state(7_900.0, 600.0, 80_000.0)
    d1 = g.dcm_from_state(101.0, r, v, 8000.0)
    fresh = _spec(command_slew_rate_deg_s=3.0).new_guidance()
    assert np.allclose(d1, fresh.dcm_from_state(101.0, r, v, 8000.0))


# ---------------------------------------------------------------------------
# the packaged guided F-07 scenario wires the spec from the vehicle
# ---------------------------------------------------------------------------

def test_f07_scenario_carries_the_shared_spec():
    from orbitopt.flightlab.sixdof.mission_output import (
        F07_PARKING_ALTITUDE_M, F07_UPPER_DRY_MASS_KG, f07_scenario)

    spec = f07_scenario().upper_guidance
    assert isinstance(spec, UpperStageGuidanceSpec)
    assert spec.target.altitude_m == F07_PARKING_ALTITUDE_M
    assert spec.dry_mass_kg == F07_UPPER_DRY_MASS_KG
    assert spec.gravity_term_gain == 0.0
    assert isinstance(spec.new_guidance(), UpperStageGuidance)
