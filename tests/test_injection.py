"""Checks for orbitopt.injection: launch site -> LEO parking orbit -> TLI /
TMI burn from v-infinity -> numerically verified escape / translunar arc.
Pure numpy/scipy except the final tudatpy test (skipped without tudatpy)."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from orbitopt.injection import (  # noqa: E402
    MU_EARTH_M3_S2,
    InjectionTarget,
    NumpyEarthPropagator,
    ParkingOrbit,
    asymptote_unit_vector,
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    hyperbolic_time_to_radius,
    mjd2000_to_ephemeris_seconds,
    osculating_apoapsis_m,
    parking_orbit_for_asymptote,
    parking_orbit_from_launch,
    plan_departure,
    plan_injection,
    verify_injection,
)
from orbitopt.launch_constraints import LaunchConstraints  # noqa: E402
from orbitopt.launch_sites import get_launch_site  # noqa: E402
from orbitopt.launch_windows import _site_vectors, as_mjd2000, asymptote_from_v_inf  # noqa: E402
from orbitopt.verify import report as vr  # noqa: E402
from orbitopt.verify.b_plane import b_plane_state  # noqa: E402

START = "2026-11-08T00:00:00Z"
TMI = InjectionTarget.from_c3_rla_dla(9.0, 340.0, 20.0, name="TMI")
TLI = InjectionTarget.from_apogee(384_400.0, 120.0, 10.0, name="TLI")


# --------------------------------------------------------------------------
# Frames / time
# --------------------------------------------------------------------------

def test_frame_rotation_round_trip_and_matches_launch_windows():
    v = np.array([1.2, -2.3, 0.7])
    assert np.allclose(equatorial_to_ecliptic(ecliptic_to_equatorial(v)), v)
    state = np.arange(6.0) + 1
    assert np.allclose(ecliptic_to_equatorial(state)[3:], ecliptic_to_equatorial(state[3:]))
    # Same RLA/DLA convention as launch_windows.asymptote_from_v_inf.
    rla, dla, c3 = asymptote_from_v_inf(v, frame="ecliptic")
    t = InjectionTarget.from_v_inf(v, frame="ecliptic")
    assert t.rla_dla_deg == pytest.approx((rla, dla), abs=1e-9)
    assert t.c3_km2_s2(6.5e6) == pytest.approx(c3)


def test_ephemeris_seconds():
    assert mjd2000_to_ephemeris_seconds(0.5) == pytest.approx(69.184)


# --------------------------------------------------------------------------
# Parking orbit
# --------------------------------------------------------------------------

def test_parking_orbit_state_is_circular_and_in_plane():
    o = ParkingOrbit(200.0, 51.6, 100.0, 9000.0, arg_latitude_deg=30.0)
    for dt in (0.0, 0.01, 0.3):
        s = o.state_at(9000.0 + dt)
        assert np.linalg.norm(s[:3]) == pytest.approx(o.radius_m)
        assert np.linalg.norm(s[3:]) == pytest.approx(o.speed_m_s)
        assert np.dot(s[:3], s[3:]) == pytest.approx(0.0, abs=1e-3)
        h = np.cross(s[:3], s[3:])
        assert np.allclose(h / np.linalg.norm(h), o.normal_at(9000.0 + dt), atol=1e-12)
    # Prograde orbit regresses westward under J2; ~5 deg/day at ISS-like altitude.
    assert -6.0 < o.raan_rate_deg_s * 86400 < -4.0
    assert 85 * 60 < o.period_s < 90 * 60


def test_parking_orbit_from_due_east_launch():
    site = get_launch_site("cape-canaveral")
    lv = parking_orbit_from_launch(site, START, 90.0, altitude_km=185.0, ascent_time_s=0.0,
                                   ascent_downrange_deg=0.0)
    # Due east -> inclination equals the (geocentric) site latitude.
    assert lv.orbit.inclination_deg == pytest.approx(28.45, abs=0.02)
    assert lv.site_offplane_deg == pytest.approx(0.0, abs=1e-9)
    assert lv.azimuth_in_corridor and lv.inclination_reachable
    # Insertion point is above the site at lift-off.
    s, _, _ = _site_vectors(site, as_mjd2000(START))
    r = lv.orbit.state_at(lv.orbit.epoch_mjd2000)[:3]
    assert np.dot(r / np.linalg.norm(r), s[0]) == pytest.approx(1.0, abs=1e-9)


def test_out_of_corridor_azimuth_is_flagged():
    lv = parking_orbit_from_launch("cape-canaveral", START, 200.0)  # southbound over land
    assert not lv.azimuth_in_corridor


def test_parking_plane_contains_asymptote():
    lv = parking_orbit_for_asymptote("cape-canaveral", START, 340.0, 20.0)
    h = lv.orbit.normal_at(lv.orbit.epoch_mjd2000)  # RAAN holds at insertion
    assert abs(np.dot(h, asymptote_unit_vector(340.0, 20.0))) < 1e-9
    assert lv.orbit.inclination_deg >= 20.0 - 1e-9  # |DLA| <= i


# --------------------------------------------------------------------------
# Burn planning (exact in two-body dynamics)
# --------------------------------------------------------------------------

def _two_body_plan(target):
    lv = parking_orbit_for_asymptote("cape-canaveral", START, *target.rla_dla_deg, j2_secular=False)
    return lv.orbit, plan_injection(lv.orbit, target, min_coast_s=600.0)


def test_tmi_burn_reaches_target_v_inf_in_two_body():
    orbit, burn = _two_body_plan(TMI)
    assert burn.coast_s >= 600.0 and burn.coast_s < 600.0 + orbit.period_s
    assert burn.out_of_plane_deg == pytest.approx(0.0, abs=1e-9)
    # Tangential burn, magnitude v_p - v_circ.
    v_p = math.sqrt(9.0e6 + 2 * MU_EARTH_M3_S2 / orbit.radius_m)
    assert burn.delta_v_mag_m_s == pytest.approx(v_p - orbit.speed_m_s)
    assert 3500 < burn.delta_v_mag_m_s < 3700
    post = burn.pre_burn_state.copy()
    post[3:] += burn.delta_v_m_s
    _b, v_inf = b_plane_state(post[:3], post[3:], MU_EARTH_M3_S2)
    assert np.allclose(v_inf, TMI.v_inf_m_s, atol=1e-6)
    # ... and stays there after numerical two-body propagation to the SOI.
    end = NumpyEarthPropagator(j2=False).final_state(post, burn.flight_time_s)
    assert np.linalg.norm(end[:3]) == pytest.approx(925e6, rel=1e-6)
    _b, v_inf_end = b_plane_state(end[:3], end[3:], MU_EARTH_M3_S2)
    assert np.allclose(v_inf_end, TMI.v_inf_m_s, atol=1e-3)


def test_tli_burn_reaches_target_apogee_in_two_body():
    orbit, burn = _two_body_plan(TLI)
    assert 3000 < burn.delta_v_mag_m_s < 3200
    assert burn.c3_km2_s2 < 0
    post = burn.pre_burn_state.copy()
    post[3:] += burn.delta_v_m_s
    assert np.allclose(osculating_apoapsis_m(post), TLI.apogee_m, atol=1.0)
    end = NumpyEarthPropagator(j2=False).final_state(post, burn.flight_time_s)
    assert np.allclose(end[:3], TLI.apogee_m, atol=10.0)  # at apogee after half a period


def test_hyperbolic_time_to_radius_matches_propagation():
    r_p, v_inf = 6.6e6, 3000.0
    t = hyperbolic_time_to_radius(r_p, v_inf, 1e8)
    v_p = math.sqrt(v_inf**2 + 2 * MU_EARTH_M3_S2 / r_p)
    end = NumpyEarthPropagator(j2=False).final_state(np.array([r_p, 0, 0, 0, v_p, 0]), t)
    assert np.linalg.norm(end[:3]) == pytest.approx(1e8, rel=1e-7)


def test_plan_departure_uses_first_launch_window():
    plan = plan_departure("cape-canaveral", TMI, start=START, min_coast_s=600.0)
    assert plan.window is not None
    assert plan.launch.liftoff_mjd2000 == pytest.approx(plan.window.optimal_mjd2000)
    assert plan.launch.azimuth_deg == pytest.approx(plan.window.azimuth_optimal_deg, abs=1e-3)
    assert plan.burn.epoch_mjd2000 > plan.launch.insertion_mjd2000
    with pytest.raises(ValueError):
        plan_departure("cape-canaveral", TMI)


def test_plan_departure_liftoff_within_search_span():
    from orbitopt.launch_windows import asymptote_launch_windows

    rla, dla = TMI.rla_dla_deg
    (w0, *_) = asymptote_launch_windows("cape-canaveral", rla, dla, START, 1.0)
    # Search span that closes inside the first window, before its optimum.
    for start, end in ((w0.open_mjd2000 - 0.01, 0.5 * (w0.open_mjd2000 + w0.close_mjd2000)),
                       (as_mjd2000(START), as_mjd2000(START) + 1.0)):
        plan = plan_departure("cape-canaveral", TMI, start=start, days=end - start)
        assert start <= plan.launch.liftoff_mjd2000 <= end


def test_lunar_b_plane_targeter_inputs_are_ecliptic():
    plan = plan_departure("cape-canaveral", TLI, start=START)
    kw = plan.lunar_b_plane_targeter_inputs()
    assert set(kw) >= {"r0", "v0_pre_burn", "initial_epoch", "dv_guess", "coast_duration_guess",
                       "reference_epoch_ephemeris_seconds"}
    assert np.allclose(ecliptic_to_equatorial(kw["r0"]), plan.burn.pre_burn_state[:3])
    assert np.allclose(ecliptic_to_equatorial(kw["dv_guess"]), plan.burn.delta_v_m_s)
    assert kw["initial_epoch"] == pytest.approx(mjd2000_to_ephemeris_seconds(plan.burn.epoch_mjd2000))


# --------------------------------------------------------------------------
# End-to-end verification (J2 numerical propagation + differential correction)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmi_verification():
    plan = plan_departure("cape-canaveral", TMI, start=START, min_coast_s=600.0)
    lc = LaunchConstraints.from_names("falcon-heavy", 4000.0, "cape-canaveral", frame="equatorial")
    return verify_injection(plan, launch_constraints=lc)


def test_tmi_end_to_end_verification_passes(tmi_verification):
    v = tmi_verification
    assert v.passed and v.converged
    assert v.final_error < 0.5  # m/s of v-infinity
    # J2 is a small perturbation: the analytic burn is close, the correction modest.
    assert v.planned_error < 50.0
    assert v.correction_dv_m_s < 100.0
    assert v.coast_residual_m < 50_000.0
    assert v.achieved_c3_km2_s2 == pytest.approx(9.0, abs=0.01)
    checks = v.report["status"]["checks"]
    for key in ("launch-injection.step_convergence", "launch-injection.two_body_conservation",
                "residual.parking_coast_vs_analytic", "targeting.converged", "launch.site_in_plane",
                "launch.azimuth_in_corridor", "launch.inclination_reachable", "launch.c3_within_vehicle",
                "launch.dla_within_site"):
        assert checks[key], key


def test_tmi_report_is_schema_valid(tmi_verification):
    pytest.importorskip("jsonschema")
    report = tmi_verification.report
    assert report["kind"] == "launch-injection"
    vr.validate_report(report)
    # Corrected burn reproduces the reported trajectory end state.
    assert np.allclose(tmi_verification.propagation.final_state, report["propagations"][0]["final_state"])


def test_tli_end_to_end_verification_passes():
    plan = plan_departure("cape-canaveral", TLI, start=START)
    v = verify_injection(plan, convergence_check=False)
    assert v.passed and v.converged
    assert v.final_error < 10_000.0
    assert abs(np.linalg.norm(v.achieved_apogee_m) - 384_400e3) < 10_000.0
    assert abs(v.out_of_plane_deg) < 1.0
    assert v.correction_dv_m_s < 50.0


def test_c3_beyond_vehicle_fails_verification():
    target = InjectionTarget.from_c3_rla_dla(40.0, 340.0, 20.0)
    plan = plan_departure("cape-canaveral", target, start=START)
    lc = LaunchConstraints.from_names("falcon-9", 4000.0, "cape-canaveral", frame="equatorial")
    v = verify_injection(plan, launch_constraints=lc, convergence_check=False)
    assert v.converged
    assert not v.report["status"]["checks"]["launch.c3_within_vehicle"]
    assert not v.passed


def test_tli_beyond_vehicle_fails_verification():
    # Falcon 9's tabulated curve gives max C3 = -2 km^2/s^2 at 5200 kg; a TLI
    # to a 1.5 Mkm apogee needs C3 ~ -0.5, so it must not pass.
    target = InjectionTarget.from_apogee(1_500_000.0, 120.0, 10.0, name="high-TLI")
    plan = plan_departure("cape-canaveral", target, start=START)
    lc = LaunchConstraints.from_names("falcon-9", 5200.0, "cape-canaveral", frame="equatorial")
    v = verify_injection(plan, launch_constraints=lc, convergence_check=False)
    assert v.converged
    checks = v.report["status"]["checks"]
    assert v.report["targets"]["achieved_c3_km2_s2"] > lc.max_c3
    assert not checks["launch.c3_within_vehicle"]
    assert "launch.dla_within_site" not in checks  # no asymptote for an ellipse
    assert not v.passed


def test_tli_within_vehicle_passes_c3_check():
    plan = plan_departure("cape-canaveral", TLI, start=START)
    lc = LaunchConstraints.from_names("falcon-heavy", 10_000.0, "cape-canaveral", frame="equatorial")
    v = verify_injection(plan, launch_constraints=lc, convergence_check=False)
    assert -2.5 < v.report["targets"]["achieved_c3_km2_s2"] < -1.5
    assert v.report["status"]["checks"]["launch.c3_within_vehicle"]
    assert v.passed


def test_tudat_backend_uses_injection_earth_gravity(monkeypatch):
    """The tudat backend must give tudat the same Earth (GM, radius, J2) as
    the analytic plan and the report's conservation check (tudat's default
    SPICE / GOCO Earth differs by ~2e-7 in GM -- 20x the 1e-8 tolerance)."""
    import sys
    import types

    from orbitopt import injection

    seen = {}

    class FakeCoast:
        def __init__(self, **kw):
            seen.update(kw)

    fake = types.ModuleType("orbitopt.verify.tudat_propagate")
    fake.ReusableCoastPropagator = FakeCoast
    monkeypatch.setitem(sys.modules, "orbitopt.verify.tudat_propagate", fake)
    prop = injection.TudatEarthPropagator()
    g = seen["central_gravity"]
    assert g["gravitational_parameter"] == prop.mu == MU_EARTH_M3_S2
    assert g["reference_radius"] == injection.R_EARTH_M
    assert g["j2"] == injection.J2_EARTH
    seen.clear()
    prop.with_overrides(perturbing_bodies=("Earth",), earth_spherical_harmonic=None)
    assert seen["central_gravity"]["gravitational_parameter"] == MU_EARTH_M3_S2
    assert seen["earth_spherical_harmonic"] is None


def test_tudat_backend_end_to_end():
    pytest.importorskip("tudatpy")
    plan = plan_departure("cape-canaveral", TMI, start=START, min_coast_s=600.0)
    v = verify_injection(plan, backend="tudat", convergence_check=False)
    assert v.passed and v.final_error < 0.5
    assert v.report["status"]["checks"]["launch-injection.two_body_conservation"]
