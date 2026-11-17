"""E5-05c: the closed-loop return guidance options of the FL-302 planner.

A scripted boostback burns a fixed time, so on the reference RTLS fixture
0.5 s of burn moves the impact ~10 km and a +-5 % thrust draw throws it tens
of kilometres off; a fixed-length entry burn turns the same draw into a
handoff-speed (and propellant) spread. ``boostback_cutoff="iip"`` ends and
steers the boostback on the vacuum instantaneous impact point;
``entry_cutoff="speed"`` ends the entry burn at a speed. These tests pin the
IIP against a numerical Kepler arc and the dispersion behaviour of both.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.recovery.boostback import (
    RecoveryTarget,
    plan_boostback_entry,
    vacuum_iip_ecef,
)
from orbitopt.flightlab.recovery.pipeline import separation_state_ecef
from orbitopt.flightlab.recovery.propagator import BoosterModel

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "orbitopt" / "data" / "flightlab"
VEHICLE = DATA / "vehicles" / "synthetic_two_stage_rtls.yaml"
SCENARIO = DATA / "scenarios" / "leo_two_stage_rtls.yaml"
ASCENT = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"
#: the +30 t one-burn script (scripts/e505c_recovery_search.py SEEDS[1])
SEED = [22.37, 49.76, 1.0, 0.0, -0.127, 20492.6, 18.54, 1.0]
#: the E5-05d separation mass + 30 t these scripts were tuned on, pinned
#: absolute: the re-baked fixture (E5-05c stage-1 resize) already carries
#: the +48 t reserve, at the same separation kinematics
SEP_MASS_KG = 165_433.8
MU = 3.986004418e14
OMEGA = 7.2921159e-5


def _kepler_impact(r_ecef, v_ecef, radius, dt=0.05):
    """Reference: RK4 point-mass arc in the inertial frame to ``radius``,
    rotated back to ECEF at the impact time."""
    r = np.asarray(r_ecef, float)
    v = np.asarray(v_ecef, float) + np.cross([0.0, 0.0, OMEGA], r)

    def acc(x):
        return -MU * x / np.linalg.norm(x) ** 3

    t = 0.0
    while True:
        k1v, k1r = acc(r), v
        k2v, k2r = acc(r + 0.5 * dt * k1r), v + 0.5 * dt * k1v
        k3v, k3r = acc(r + 0.5 * dt * k2r), v + 0.5 * dt * k2v
        k4v, k4r = acc(r + dt * k3r), v + dt * k3v
        r_new = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if np.linalg.norm(r_new) <= radius:
            f = (np.linalg.norm(r) - radius) / (np.linalg.norm(r) - np.linalg.norm(r_new))
            r, t = r + f * (r_new - r), t + f * dt
            break
        r, v, t = r_new, v_new, t + dt
    th = -OMEGA * t
    c, s = math.cos(th), math.sin(th)
    return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])


def test_vacuum_iip_matches_a_numerical_kepler_arc():
    radius = 6_372_000.0
    r = np.array([radius + 120_000.0, 30_000.0, 15_000.0])
    for v in ([-900.0, 400.0, 1200.0], [300.0, -1500.0, 200.0], [0.0, 0.0, 800.0]):
        iip = vacuum_iip_ecef(r, np.array(v), radius)
        ref = _kepler_impact(r, v, radius)
        assert np.linalg.norm(iip - ref) < 50.0, (v, np.linalg.norm(iip - ref))
    assert vacuum_iip_ecef(np.array([radius - 10.0, 0.0, 0.0]), np.zeros(3), radius) is None
    assert vacuum_iip_ecef(r, np.array([0.0, 11_500.0, 0.0]), radius) is None  # escape


@pytest.fixture(scope="module")
def inputs():
    booster = BoosterModel.load(VEHICLE)
    run = read_run(ASCENT)
    rt = RecoveryTarget.from_scenario(load_target(SCENARIO).to_dict())
    sep = separation_state_ecef(run)
    return booster, rt, dataclasses.replace(sep, mass_kg=SEP_MASS_KG)


def _plan(booster, rt, sep, *, thrust=1.0, **kw):
    eng = booster.engine
    b = dataclasses.replace(booster, engine=dataclasses.replace(
        eng, thrust_vacuum_n=eng.thrust_vacuum_n * thrust))
    x = list(SEED)
    if kw.get("boostback_cutoff") == "iip":
        x[1] = 2.5     # IIP aim [km] past the pad
    if kw.get("entry_cutoff") == "speed":
        x[6] = 162.0   # entry cutoff speed [m/s]
    return plan_boostback_entry(sep, b, rt, entry_burns=1, entry_throttle=True,
                                initial_guess=x, entry_q_max_pa=250_000.0,
                                entry_g_max=12.0, horizon_s=900.0, dt_opt=0.25,
                                dt_final=0.25, search=kw.pop("search", lambda obj, v: v),
                                landing_need_model="closed_loop", **kw)


@pytest.mark.slow
def test_iip_boostback_holds_the_aim_under_thrust_dispersion(inputs):
    booster, rt, sep = inputs
    for thrust in (0.95, 1.05):
        scripted = _plan(booster, rt, sep, thrust=thrust).corridor_report
        closed = _plan(booster, rt, sep, thrust=thrust, boostback_cutoff="iip").corridor_report
        assert scripted.miss_m is None or scripted.miss_m > 10_000.0, thrust
        assert closed.miss_m < 1_000.0, thrust


@pytest.mark.slow
def test_speed_cut_entry_burn_does_not_over_brake(inputs):
    booster, rt, sep = inputs
    fixed = _plan(booster, rt, sep, thrust=1.05, boostback_cutoff="iip").corridor_report
    cut = _plan(booster, rt, sep, thrust=1.05, boostback_cutoff="iip",
                entry_cutoff="speed").corridor_report
    # 5 % hot: brakes to ~90-102 m/s (E5-05d / E5-05c-resized fixture kinematics),
    # well under the 162 m/s cut the speed-cut burn stops at
    assert fixed.speed_m_s < 110.0
    assert cut.speed_m_s > 140.0                       # stops at the cut speed instead
    # what over-braking costs is what is left once the landing burn has flown
    left_fixed = fixed.propellant_available_kg - fixed.landing_need_kg
    left_cut = cut.propellant_available_kg - cut.landing_need_kg
    assert left_cut > left_fixed + 3000.0


def test_closed_loop_options_validate(inputs):
    booster, rt, sep = inputs
    with pytest.raises(ValueError, match="boostback_cutoff"):
        _plan(booster, rt, sep, boostback_cutoff="nope")
    with pytest.raises(ValueError, match="entry_cutoff"):
        _plan(booster, rt, sep, entry_cutoff="nope")
    with pytest.raises(ValueError, match="one-burn"):
        plan_boostback_entry(sep, booster, rt, entry_burns=2, entry_cutoff="speed",
                             search=lambda obj, v: v)


def test_landing_reserve_option_selects_the_graded_reserve(inputs):
    from orbitopt.flightlab.recovery.pipeline import landing_reserve_booster

    booster = inputs[0]
    target = load_target(SCENARIO)
    assert landing_reserve_booster(booster, target) is booster          # default: vehicle
    assert landing_reserve_booster(booster, target, "vehicle") is booster
    graded = landing_reserve_booster(booster, target, "target")
    assert graded.landing_propellant_reserve_kg == target.recovery.min_propellant_reserve_kg
    assert graded.landing_propellant_reserve_kg < booster.landing_propellant_reserve_kg
    assert landing_reserve_booster(booster, target, 12_000).landing_propellant_reserve_kg == 12_000.0
    for bad in ("nope", -1.0):
        with pytest.raises(ValueError, match="landing_reserve"):
            landing_reserve_booster(booster, target, bad)


@pytest.mark.slow
def test_speed_cut_below_h0_is_an_engine_on_handoff(inputs):
    """A speed-cut entry burn still lit at h0 does not hand over: the handoff
    is gated at the cutoff, so a cut below the corridor height is graded
    CORRIDOR_MISSED (it used to be accepted as an engine-on handoff), and a
    cut above it is not. The cutoff height is on the corridor report."""
    booster, rt, sep = inputs
    h0 = rt.corridor.h0_m
    for thrust in (0.95, 1.0):          # cold / nominal: this script cuts below h0
        plan = _plan(booster, rt, sep, thrust=thrust, boostback_cutoff="iip",
                     entry_cutoff="speed")
        assert plan.corridor_report.entry_cutoff_altitude_m < h0, thrust
        assert "CORRIDOR_MISSED" in plan.reason_codes, thrust
    hot = _plan(booster, rt, sep, thrust=1.05, boostback_cutoff="iip", entry_cutoff="speed")
    assert hot.corridor_report.entry_cutoff_altitude_m > h0
    assert "CORRIDOR_MISSED" not in hot.reason_codes
    timed = _plan(booster, rt, sep, boostback_cutoff="iip")
    assert timed.corridor_report.entry_cutoff_altitude_m is None    # speed mode only


@pytest.mark.slow
def test_search_keep_flies_the_initial_script(inputs):
    """search='keep' (the form a mission document pins a validated script in)
    is the identity search: the plan is the initial script, flown as is."""
    booster, rt, sep = inputs
    kept = _plan(booster, rt, sep, boostback_cutoff="iip", search="keep")
    ident = _plan(booster, rt, sep, boostback_cutoff="iip")
    assert kept.entry_burn == ident.entry_burn and kept.boostback == ident.boostback
    assert kept.corridor_report == ident.corridor_report


def test_search_and_throttle_layout_validate(inputs):
    booster, rt, sep = inputs
    with pytest.raises(ValueError, match="search"):
        plan_boostback_entry(sep, booster, rt, search="nope")
    with pytest.raises(ValueError, match="initial_guess"):
        plan_boostback_entry(sep, booster, rt, search="keep")
    with pytest.raises(ValueError, match="entry_throttle"):   # 7 values, throttled layout has 8
        plan_boostback_entry(sep, booster, rt, entry_throttle=True,
                             initial_guess=SEED[:7], search="keep")


def test_unpack_script_layouts():
    from orbitopt.flightlab.recovery.boostback import _unpack_script

    x7 = np.array(SEED[:7])
    bb, eb, eb2 = _unpack_script(x7)
    assert eb.throttle == 1.0 and eb2 is None and bb.duration_s == SEED[1]
    _bb, eb, _ = _unpack_script(np.array(SEED[:7] + [0.6]), 1, True)
    assert eb.throttle == 0.6
    x2 = np.array(SEED[:7] + [6000.0, 12.0, 0.9, 0.3])
    _bb, eb, eb2 = _unpack_script(x2, 2, True)
    assert (eb.throttle, eb2.trigger_altitude_m, eb2.duration_s, eb2.throttle) == (0.9, 6000.0, 12.0, 0.3)
    _bb, eb, eb2 = _unpack_script(x2[:9], 2, False)
    assert eb.throttle == 1.0 and eb2.throttle == 1.0
