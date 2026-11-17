"""E5-05c: the planner's closed-loop landing forecast.

The momentum landing need (a full-thrust vertical burn lit at the 3 km
handoff) read 5.2 t on the reference RTLS fixture where the flown FL-305
landing burn spent 7.9 t, so the planner chased handoffs the landing could
not afford. ``predict_closed_loop_landing`` flies the FL-305 law itself in a
flat pad frame; ``landing_need_model="closed_loop"`` makes the planner price
and grade the handoff with it. Fast tests pin the forecast's physics and the
planner switch; the slow test pins the forecast against the flown landing.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.recovery.boostback import (
    RecoveryTarget,
    landing_burn_propellant_need_kg,
    plan_boostback_entry,
)
from orbitopt.flightlab.recovery.landing import (
    DEFAULT_LANDING_LIMITS,
    predict_closed_loop_landing,
)
from orbitopt.flightlab.recovery.pipeline import separation_state_ecef
from orbitopt.flightlab.recovery.propagator import BoosterModel

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "orbitopt" / "data" / "flightlab"
VEHICLE = DATA / "vehicles" / "synthetic_two_stage_rtls.yaml"
SCENARIO = DATA / "scenarios" / "leo_two_stage_rtls.yaml"
ASCENT = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"
#: the +30 t one-burn script of scripts/e505c_recovery_search.py (SEEDS[1])
SEED = [22.37, 49.76, 1.0, 0.0, -0.127, 20492.6, 18.54, 1.0]


@pytest.fixture(scope="module")
def booster():
    return BoosterModel.load(VEHICLE)


def test_forecast_lands_a_vertical_handoff_and_costs_more_than_the_momentum_need(booster):
    m = booster.dry_mass_kg + 20_000.0
    pred = predict_closed_loop_landing((0.0, 0.0, 3000.0), (0.0, 0.0, -300.0), m, booster)
    assert pred.feasible
    assert pred.touchdown_vertical_m_s <= DEFAULT_LANDING_LIMITS.vertical_speed_max_m_s
    assert pred.touchdown_lateral_m_s < 0.1 and pred.touchdown_miss_m < 1.0
    assert 0.0 < pred.ignition_altitude_m <= 3000.0
    # the law coasts, accelerates and burns at design throttle: dearer than
    # the ideal full-thrust burn lit at the handoff
    assert pred.need_kg > landing_burn_propellant_need_kg(m, 300.0, 3000.0, booster)


def test_forecast_reports_an_unstoppable_handoff(booster):
    m = booster.dry_mass_kg + 20_000.0
    pred = predict_closed_loop_landing((0.0, 0.0, 300.0), (0.0, 0.0, -400.0), m, booster)
    assert not pred.feasible and pred.touchdown_vertical_m_s > 50.0


def test_forecast_sees_the_divert(booster):
    m = booster.dry_mass_kg + 20_000.0
    straight = predict_closed_loop_landing((0.0, 0.0, 3000.0), (0.0, 0.0, -300.0), m, booster)
    offset = predict_closed_loop_landing((600.0, 0.0, 3000.0), (-40.0, 0.0, -300.0), m, booster)
    assert offset.need_kg > straight.need_kg


@pytest.fixture(scope="module")
def handoff_inputs(booster):
    run = read_run(ASCENT)
    target = load_target(SCENARIO)
    sep = separation_state_ecef(run)
    sep = dataclasses.replace(sep, mass_kg=sep.mass_kg + 30_000.0)
    return sep, RecoveryTarget.from_scenario(target.to_dict()), target


def _plan(sep, booster, rt, **kw):
    return plan_boostback_entry(sep, booster, rt, entry_burns=1, entry_throttle=True,
                                initial_guess=SEED, entry_q_max_pa=250_000.0,
                                entry_g_max=12.0, horizon_s=900.0, dt_opt=0.5,
                                dt_final=0.5, search=lambda obj, x: x, **kw)


def test_planner_closed_loop_mode_prices_the_landing_it_will_fly(booster, handoff_inputs):
    sep, rt, _ = handoff_inputs
    momentum = _plan(sep, booster, rt)
    closed = _plan(sep, booster, rt, landing_need_model="closed_loop")
    rm, rc = momentum.corridor_report, closed.corridor_report
    assert rm.landing_ignition_altitude_m is None                 # default: untouched
    assert rc.landing_ignition_altitude_m is not None
    assert rc.landing_touchdown_vertical_m_s is not None
    assert rc.landing_need_kg > rm.landing_need_kg + 1000.0      # the momentum need is optimistic
    assert "INSUFFICIENT_PROPELLANT_RESERVE" in closed.reason_codes
    with pytest.raises(ValueError, match="landing_need_model"):
        _plan(sep, booster, rt, landing_need_model="nope")


@pytest.mark.slow
def test_forecast_matches_the_flown_closed_loop_landing(booster, handoff_inputs):
    """The forecast's propellant is the flown FL-305 burn's within 2 % on the
    reference fixture's +30 t handoff (0.6 % when recorded)."""
    pytest.importorskip("gfold")
    from orbitopt.flightlab.recovery.landing import fly_landing_burn_closed_loop
    from orbitopt.flightlab.recovery.pipeline import _landing_limits_for_target

    sep, rt, target = handoff_inputs
    plan = plan_boostback_entry(sep, booster, rt, entry_burns=1, entry_throttle=True,
                                initial_guess=SEED, entry_q_max_pa=250_000.0,
                                entry_g_max=12.0, horizon_s=900.0, dt_opt=0.1,
                                dt_final=0.1, search=lambda obj, x: x,
                                landing_need_model="closed_loop")
    state, enu = plan.landing_burn_initial_state, plan.landing_burn_initial_state_enu
    assert state is not None
    flown = fly_landing_burn_closed_loop(enu, state, rt, booster,
                                         limits=_landing_limits_for_target(target))
    used = state.mass_kg - float(flown.track.mass_kg[-1])
    predicted = plan.corridor_report.landing_need_kg
    assert predicted == pytest.approx(used, rel=0.02)
