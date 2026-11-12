"""FL-203 (#201): launch-window & RAAN targeting.

Epoch derivation reuses ``launch_windows`` geometry; the achieved RAAN is
graded from GCRS elements via the single Gate-B check -- never from
OpenTsiolkovsky's launch-aligned frame -- and the RAAN residual joins the
FL-201 fitness only when the target constrains RAAN. The epoch is calibrated
(probe flights, measured sidereal slope) so the *achieved* plane, not just
the geometric pass, is the target plane.
"""
from __future__ import annotations

import json
import math
import shutil
import types
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.flightlab import checks
from orbitopt.flightlab.contract import load_target
from orbitopt.flightlab.optimize.ascent_problem import (AscentProblem,
                                                        _EQ_RAAN_CHECK)
from orbitopt.flightlab.window_targeting import (EARTH_ROTATION_DEG_S,
                                                 calibrate_epoch,
                                                 derive_in_plane_liftoff,
                                                 raan_report,
                                                 signed_angle_diff_deg,
                                                 site_for_target,
                                                 window_sensitivity)

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml"
VEHICLE = REPO / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"
FIX = REPO / "tests/fixtures/flightlab/opentsiolkovsky"

STUB_RAAN_DEG = 125.74252437465294     # stub insertion state's GCRS RAAN


@pytest.fixture
def target():
    return load_target(TARGET)


@pytest.fixture
def raan_target(target):
    return replace(target, target_orbit=replace(
        target.target_orbit, raan_deg=round(STUB_RAAN_DEG, 4)))


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def stub_run(raan_target, stub, tmp_path):
    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

    return run_ascent(raan_target, VEHICLE, None, tmp_path / "flight")


# --- epoch derivation ---------------------------------------------------------

def test_derivation_requires_raan_and_window(target):
    with pytest.raises(ValueError, match="RAAN-constrained"):
        derive_in_plane_liftoff(target)
    raan = replace(target, target_orbit=replace(target.target_orbit,
                                                raan_deg=45.0))
    with pytest.raises(ValueError, match="window_days"):
        derive_in_plane_liftoff(raan, window_days=0.0)


def test_derivation_first_in_plane_pass_consistent(raan_target):
    der = derive_in_plane_liftoff(raan_target)
    assert der.epoch_raan_deg == pytest.approx(125.715, abs=0.05)
    assert der.target.mission.epoch_utc == der.target.target_orbit.epoch_utc
    # the gradeable RAAN is the plane RAAN *at the derived epoch*, not the
    # anchor RAAN (J2 regression moves the plane between the two)
    assert der.epoch_raan_deg != pytest.approx(der.requested_raan_deg)
    from orbitopt.launch_windows import OrbitPlane, as_mjd2000

    anchor = as_mjd2000(der.anchor_utc)
    plane = OrbitPlane.from_altitude(51.6, 125.7425, anchor, 470.0)
    assert plane.raan_at(as_mjd2000(der.target.mission.epoch_utc)) \
        == pytest.approx(der.epoch_raan_deg, abs=0.0015)
    assert der.window.duration_s == 0.0          # a pass, not a corridor
    assert any(a.startswith("FL-203: liftoff epoch")
               for a in der.target.assumptions)


def test_derivation_is_idempotent(raan_target):
    from orbitopt.launch_windows import SECONDS_PER_DAY, as_mjd2000

    once = derive_in_plane_liftoff(raan_target)
    twice = derive_in_plane_liftoff(once.target)
    drift = abs(as_mjd2000(twice.target.mission.epoch_utc)
                - as_mjd2000(once.target.mission.epoch_utc)) * SECONDS_PER_DAY
    assert drift < 2.0                  # a node pass is its own optimum
    assert abs(twice.epoch_raan_deg - once.epoch_raan_deg) < 0.01


def test_site_for_target_ad_hoc_open_corridors(target):
    from orbitopt.launch_windows import geocentric_latitude_deg

    site = site_for_target(target)      # "selected-site" is not a DB site
    assert site.azimuth_corridors_deg == ((0.0, 360.0),)
    assert site.rotation_speed_gain_m_s == pytest.approx(
        465.1 * math.cos(math.radians(
            geocentric_latitude_deg(site.latitude_deg))), rel=0.01)


# --- the discrimination test: GCRS RAAN, not OT's frame -----------------------

def test_raan_report_is_gcrs_not_ot_launch_aligned(stub_run, raan_target):
    report = raan_report(stub_run, raan_target)
    ins = checks.insertion_state(stub_run)
    el = checks.orbital_elements(ins["position_m"], ins["velocity_m_s"])
    assert report["achieved_raan_deg"] == pytest.approx(el["raan_deg"], abs=1e-9)
    ot = report["ot_launch_aligned_raan_deg"]
    # the discriminator: OT's own launch-aligned RAAN is >1 tolerance away
    # from the GCRS one, so grading the OT number would flip pass -> fail
    assert checks.angle_diff_deg(ot, el["raan_deg"]) > checks.RAAN_TOLERANCE_DEG
    assert ot != pytest.approx(report["achieved_raan_deg"], abs=0.5)
    assert report["passed"] is True
    assert report["error_deg"] == pytest.approx(
        signed_angle_diff_deg(el["raan_deg"], 125.7425), abs=1e-3)
    assert report["tolerance_deg"] == checks.RAAN_TOLERANCE_DEG


def test_raan_report_refuses_unconstrained(stub_run, target):
    with pytest.raises(ValueError, match="RAAN"):
        raan_report(stub_run, target)


def test_check_with_ot_raan_target_fails_grading(stub_run, target):
    # a target whose raan_deg equals OT's frame value must NOT pass: the
    # grading frame is GCRS, and this is exactly the wrong-number trap
    from orbitopt.flightlab.window_targeting import _ot_launch_aligned_raan

    wrong = replace(target, target_orbit=replace(
        target.target_orbit, raan_deg=round(_ot_launch_aligned_raan(stub_run), 3)))
    verdict = checks.evaluate(stub_run, wrong)
    check = next(c for c in verdict.checks if c.id == _EQ_RAAN_CHECK)
    assert check.passed is False
    assert checks.angle_diff_deg(check.value, check.limit) > \
        checks.RAAN_TOLERANCE_DEG


# --- AscentProblem gating -----------------------------------------------------

def test_raan_slot_only_when_target_constrains(raan_target, target):
    plain = AscentProblem(target, VEHICLE, "payload")
    assert plain.raan_constrained is False
    assert plain.eq_check_ids == ("insertion-periapsis-within-tolerance",
                                  "insertion-apoapsis-within-tolerance",
                                  "insertion-inclination-within-tolerance")
    assert len(plain.get_bounds_eq_ineq()) == 6     # FL-201 shape untouched
    constrained = AscentProblem(raan_target, VEHICLE, "payload")
    assert constrained.raan_constrained is True
    assert constrained.eq_check_ids[-1] == _EQ_RAAN_CHECK
    bands = constrained.get_bounds_eq_ineq()
    assert len(bands) == 7
    raan_band = bands[3]                        # residual order: rp, ra, inc, raan
    assert raan_band[:2] == pytest.approx((-checks.RAAN_TOLERANCE_DEG,
                                           checks.RAAN_TOLERANCE_DEG))


def test_raan_residual_signed_wrap():
    ap = AscentProblem(load_target(TARGET), VEHICLE, "payload")
    check = types.SimpleNamespace(value=1.0, limit=359.0)     # across the seam
    by_id = {_EQ_RAAN_CHECK: check}
    assert ap._residual(by_id, _EQ_RAAN_CHECK, signed=True) == pytest.approx(2.0)
    # without the wrap this would read -358 -- the optimizer would steer wrong
    assert ap._residual(by_id, "insertion-periapsis-within-tolerance",
                        signed=True) == 0.0     # absent check -> unconstrained


def test_raan_weight_and_slot_names(raan_target):
    from orbitopt.flightlab.optimize.ascent_problem import (_DEFAULT_WEIGHTS,
                                                            _SLOT_NAMES)
    from orbitopt.flightlab.optimize import run as _run

    assert _DEFAULT_WEIGHTS["eq_raan"] == 10.0
    assert _SLOT_NAMES[_EQ_RAAN_CHECK] == "eq_raan"
    from orbitopt.flightlab.optimize.ascent_problem import REPORT_NAMES
    assert REPORT_NAMES[_EQ_RAAN_CHECK][0] == "eq_raan_deg"
    # run.py scales the RAAN weight by the one tolerance the schema lacks
    udp = AscentProblem(raan_target, VEHICLE, "payload")
    weights = _run._scaled_weights(udp)
    inc_tol = float(udp.target.constraints.inclination_tolerance_deg
                    or checks.DEFAULT_INCLINATION_TOLERANCE_DEG)
    assert weights["eq_raan"] == pytest.approx(
        weights["eq_inc"] * inc_tol / checks.RAAN_TOLERANCE_DEG)


# --- window sensitivity ---------------------------------------------------------

def test_window_sensitivity_stub_shows_earth_rotation(raan_target, stub):
    # the stub replays a fixed LAUNCH_ECI insertion state, which the frame
    # bridge converts to GCRS at the launch epoch -- so even the stub shows
    # the physical slope: one sidereal day of Earth rotation per day of
    # liftoff shift (~0.25 deg per minute off-window)
    rows = window_sensitivity(raan_target, VEHICLE, None,
                              offsets_s=(-60.0, 60.0), timeout_s=60.0)
    assert [r["liftoff_offset_s"] for r in rows] == [0.0, -60.0, 60.0]
    assert rows[0]["liftoff_epoch_utc"] == raan_target.mission.epoch_utc
    assert rows[0]["achieved_raan_deg"] == pytest.approx(STUB_RAAN_DEG, abs=1e-4)
    slope = (rows[2]["achieved_raan_deg"] - rows[1]["achieved_raan_deg"]) / 120.0
    assert slope == pytest.approx(EARTH_ROTATION_DEG_S, rel=0.05)
    assert all(abs(r["error_deg"]) < checks.RAAN_TOLERANCE_DEG + 0.3
               for r in rows)


def test_calibrate_epoch_moves_pass_to_the_achieved_plane(raan_target, stub,
                                                          tmp_path):
    der = derive_in_plane_liftoff(raan_target)
    corrected, cal = calibrate_epoch(der, VEHICLE, timeout_s=60.0)
    assert cal["geometric_pass_utc"] == der.target.mission.epoch_utc
    assert cal["raan_error_at_pass_deg"] == pytest.approx(2.03, abs=0.02)
    assert cal["measured_slope_deg_s"] == pytest.approx(
        EARTH_ROTATION_DEG_S, rel=0.05)
    # the stub replays a state launched at the FILE epoch (12:00), so the
    # correction must pull the ~12:08 pass back to just before the file epoch
    assert cal["correction_s"] == pytest.approx(-2.03 / EARTH_ROTATION_DEG_S,
                                                abs=5.0)
    assert corrected.target.mission.epoch_utc != der.target.mission.epoch_utc
    assert corrected.target.assumptions[-1].startswith("FL-203: liftoff epoch corrected")
    assert corrected.window is der.window
    # verified by re-flying: the corrected target passes the graded check
    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

    run = run_ascent(corrected.target, VEHICLE, None, tmp_path / "cal-run",
                     timeout_s=60.0)
    verdict = checks.evaluate(run, corrected.target)
    raan = raan_report(run, corrected.target,
                       by_id={c.id: c for c in verdict.checks})
    assert raan["passed"] is True
    assert raan["error_deg"] == pytest.approx(0.0, abs=0.1)


def _has_pygmo_and_scipy():
    import importlib.util

    return all(importlib.util.find_spec(m) for m in ("pygmo", "scipy"))


def _real_binary():
    from orbitopt.flightlab.adapters.opentsiolkovsky import OTBinaryMissingError, \
        find_binary

    try:
        return find_binary()
    except OTBinaryMissingError:
        return None


@pytest.mark.slow
@pytest.mark.skipif(not _real_binary(), reason="OpenTsiolkovsky binary missing")
def test_window_sensitivity_slope_is_earth_rotation(raan_target, tmp_path):
    rows = {r["liftoff_offset_s"]: r
            for r in window_sensitivity(raan_target, VEHICLE, None,
                                        offsets_s=(-120.0, 120.0))}
    slope = (rows[120.0]["achieved_raan_deg"] - rows[-120.0]["achieved_raan_deg"]) \
        / 240.0
    assert slope == pytest.approx(EARTH_ROTATION_DEG_S, rel=0.15)


# --- handoff to the parking-orbit flows ----------------------------------------

def test_parking_export_gates_on_verification(stub_run):
    from orbitopt.flightlab.parking_export import parking_orbit_from_insertion

    with pytest.raises(ValueError, match="did not pass its mission constraints"):
        parking_orbit_from_insertion(stub_run)          # stub ra is 539 km


def test_parking_export_maps_gcrs_insertion_state(stub_run):
    from orbitopt.flightlab.parking_export import parking_orbit_from_insertion
    from orbitopt.injection import ParkingOrbit

    hand = parking_orbit_from_insertion(stub_run, require_verified=False)
    assert isinstance(hand.orbit, ParkingOrbit)
    ins = checks.insertion_state(stub_run)
    el = checks.orbital_elements(ins["position_m"], ins["velocity_m_s"])
    mean_alt_km = 0.5 * (el["periapsis_altitude_m"]
                         + el["apoapsis_altitude_m"]) / 1000.0
    assert hand.orbit.altitude_km == pytest.approx(mean_alt_km, rel=1e-9)
    assert hand.orbit.inclination_deg == pytest.approx(el["inclination_deg"])
    assert hand.orbit.raan_deg == pytest.approx(el["raan_deg"])
    # epoch = track epoch + SECO time, and it round-trips through injection's
    # own state model to the same in-plane direction as the flown state
    import numpy as np

    r = np.asarray(hand.orbit.state_at(hand.epoch_mjd2000))[:3]
    r_flown = np.asarray(ins["position_m"], dtype=float)
    cos = float(np.dot(r, r_flown) / (np.linalg.norm(r) * np.linalg.norm(r_flown)))
    assert cos > 1.0 - 1e-9
    assert hand.verification["mission_constraints_passed"] is False
    blob = hand.to_dict()
    assert blob["kind"] == "orbitopt.flightlab.ascent-handoff"
    assert blob["orbit"]["altitude_km"] == pytest.approx(hand.orbit.altitude_km)


# --- the optimizer feed-through (stub solve, RAAN-constrained) ------------------

def test_optimize_stub_derives_epoch_and_reports_raan(raan_target, stub, tmp_path):
    pytest.importorskip("pygmo")
    pytest.importorskip("scipy")
    from orbitopt.flightlab.optimize.run import optimize_ascent

    result = optimize_ascent(raan_target, VEHICLE, tmp_path / "run",
                             generations=2, population=8, workers=1,
                             local=False, seed=7, refly=False,
                             sensitivity_offsets_s=(-60.0, 60.0),
                             target_path=str(TARGET))
    assert result["raan"] is not None
    assert result["raan"]["tolerance_deg"] == checks.RAAN_TOLERANCE_DEG
    # the winner flew the CALIBRATED epoch: derived in-plane pass at 12:08
    # (the fixed stub trajectory lands ~8 min of Earth rotation -- 2.03 deg
    # -- off the plane at the pass), probe-measured slope, epoch pulled back
    # ~486 s to the pre-pass point where the flight's own plane hits the
    # target. Residual is only the plane's J2 regression across the shift.
    window = result["raan"]["liftoff_window"]
    assert window["geometric_pass_utc"] == "2026-11-10T12:08:00Z"
    assert window["derived_epoch_utc"] != window["geometric_pass_utc"]
    assert window["window"]["kind"] == "in-plane"
    cal = window["epoch_correction"]
    assert cal["raan_error_at_pass_deg"] == pytest.approx(2.03, abs=0.02)
    assert cal["measured_slope_deg_s"] == pytest.approx(
        EARTH_ROTATION_DEG_S, rel=0.05)
    assert cal["correction_s"] == pytest.approx(-2.03 / EARTH_ROTATION_DEG_S,
                                                abs=5.0)
    assert result["raan"]["error_deg"] == pytest.approx(0.0, abs=0.1)
    assert result["raan"]["passed"] is True
    assert "INFEASIBLE_RAAN_OUT_OF_TOLERANCE" not in result["reason_codes"]
    assert "INFEASIBLE_APOAPSIS_OUT_OF_TOLERANCE" in result["reason_codes"]
    offs = [r["liftoff_offset_s"] for r in result["raan"]["window_sensitivity"]]
    assert offs == [0.0, -60.0, 60.0]
    doc = json.loads((tmp_path / "run" / "optimization.json").read_text())
    assert doc["raan"]["error_deg"] == pytest.approx(result["raan"]["error_deg"])


def test_optimize_fixed_epoch_skips_derivation(raan_target, stub, tmp_path):
    pytest.importorskip("pygmo")
    pytest.importorskip("scipy")
    from orbitopt.flightlab.optimize.run import optimize_ascent

    result = optimize_ascent(raan_target, VEHICLE, tmp_path / "run",
                             generations=1, population=6, workers=1,
                             local=False, seed=3, refly=False,
                             derive_window=False, sensitivity_offsets_s=None,
                             target_path=str(TARGET))
    assert result["raan"].get("liftoff_window") is None
    assert "window_sensitivity" not in result["raan"]
    # 1e-4 deg like the sensitivity test: the frame bridge reads IERS Earth
    # orientation data, and an older astropy-iers-data release (the locked
    # env's) moves the result by ~5e-6 deg
    assert result["raan"]["achieved_raan_deg"] == pytest.approx(STUB_RAAN_DEG,
                                                                abs=1e-4)
