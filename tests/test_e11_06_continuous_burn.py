"""E11-06: the optional pieces behind the continuous-burn Starbase recovery -- fin drag that scales with the
fin area, speed feedback on the fin schedule, engine-by-engine shutdown in the landing burn, the engine-lit
hand-over, and the planner guard. Every option is off by default; the defaults are asserted unchanged."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.recovery.grid_fins import GridFinGuidance, GridFinModel
from orbitopt.flightlab.recovery.landing import (TerminalGuidance, engine_stage_fraction,
                                                  predict_closed_loop_landing)
from orbitopt.flightlab.recovery.propagator import BoosterModel, ThrustCmd

VEH = Path(__file__).resolve().parents[1] / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase_fins.yaml"


def _model(**over) -> GridFinModel:
    node = {"count": 3, "fin_area_m2": 28.1, "cd_add_mach": [[0.3, 0.5], [1.0, 1.0], [5.0, 0.4]],
            "cn_alpha_mach": [[0.3, 1.0], [5.0, 0.5]], "provenance": "illustrative"}
    node.update(over)
    return GridFinModel.from_mapping(node, reference_area_m2=63.617)


# ---------------------------------------------------------------- fin drag follows the planform

def test_fin_drag_is_a_fixed_table_unless_a_reference_area_is_declared():
    legacy = _model()
    assert legacy.cd_area_scale == 1.0 and legacy.cd_add(1.0) == pytest.approx(1.0)
    scaled = _model(cd_add_reference_fin_area_m2=42.15)      # 3 x 28.1 = 84.3 m^2 against a 42.15 m^2 table
    assert scaled.cd_area_scale == pytest.approx(2.0, rel=1e-3)
    assert scaled.cd_add(1.0) == pytest.approx(2.0, rel=1e-3)
    bigger = _model(fin_area_m2=56.2, cd_add_reference_fin_area_m2=42.15)
    assert bigger.cd_add(1.0) == pytest.approx(2.0 * scaled.cd_add(1.0), rel=1e-3)


# ---------------------------------------------------------------- speed feedback on the schedule

def _law(**over):
    kw = dict(model=_model(), mode="schedule", alpha_deg=30.0, schedule=((70000.0, 0.0, 30.0),))
    kw.update(over)
    return GridFinGuidance(**kw)


def _cmd(law, h, speed):
    r = np.array([6.4e6, 0.0, 0.0]); v = np.array([speed, 0.0, 0.0])
    return law.command(h, r, v)


def test_speed_feedback_is_off_by_default_and_exact_when_off():
    open_loop = _law()
    assert _cmd(open_loop, 20000.0, 900.0).alpha_cmd_deg == 30.0
    zero_gain = _law(speed_ref=((20000.0, 600.0),), speed_gain_deg_per_mps=0.0)
    assert _cmd(zero_gain, 20000.0, 900.0).alpha_cmd_deg == 30.0


def test_speed_feedback_adds_alpha_when_too_fast_and_clips_to_authority():
    law = _law(speed_ref=((40000.0, 800.0), (10000.0, 400.0)), speed_gain_deg_per_mps=0.05)
    assert _cmd(law, 25000.0, 700.0).alpha_cmd_deg == pytest.approx(30.0 + 0.05 * (700.0 - 600.0))
    assert _cmd(law, 25000.0, 500.0).alpha_cmd_deg == pytest.approx(30.0 - 0.05 * 100.0)   # slower: less alpha
    assert _cmd(law, 25000.0, 5000.0).alpha_cmd_deg == law.model.alpha_authority_deg       # clipped
    assert _cmd(law, 25000.0, 0.0).alpha_cmd_deg == 0.0


def test_speed_feedback_needs_a_descending_reference_profile():
    with pytest.raises(ValueError):
        _law(speed_ref=((10000.0, 400.0), (40000.0, 800.0)), speed_gain_deg_per_mps=0.01)
    with pytest.raises(ValueError):
        _law(speed_gain_deg_per_mps=0.01)


# ---------------------------------------------------------------- engine staging in the landing burn

def test_engine_stage_is_the_smallest_set_with_thrust_headroom_and_never_relights():
    fr = (1.0, 5 / 13, 3 / 13)
    t_max, t_min = 30.0e6, 12.0e6
    assert engine_stage_fraction((), 1.0, 5e6, t_min, t_max) == 1.0                      # no staging: whole set
    assert engine_stage_fraction(fr, 1.0, 5e6, t_min, t_max) == pytest.approx(3 / 13)    # small demand: 3 engines
    assert engine_stage_fraction(fr, 1.0, 9e6, t_min, t_max) == pytest.approx(5 / 13)
    assert engine_stage_fraction(fr, 1.0, 25e6, t_min, t_max) == 1.0                     # big demand: all 13
    assert engine_stage_fraction(fr, 5 / 13, 25e6, t_min, t_max) == pytest.approx(5 / 13)  # a shut engine stays shut


def test_thrust_command_floor_scales_with_the_engines_still_lit():
    eng = BoosterModel.load(VEH).engine
    assert eng.effective_throttle(0.1) == eng.throttle_min                           # whole set: clamped up
    assert eng.effective_throttle(0.1, 3 / 13) == pytest.approx(0.1)                 # 3 engines: 0.4 * 3/13 = 0.092
    assert eng.effective_throttle(0.0, 3 / 13) == 0.0
    assert ThrustCmd(throttle=0.5, dir_ecef=(0, 0, 1)).min_throttle_scale == 1.0


def _predict(v_down, **guid):
    veh = BoosterModel.load(VEH)
    g = TerminalGuidance(max_thrust_accel_g=4.0, **guid)
    return predict_closed_loop_landing((0.0, 0.0, 3000.0), (0.0, 0.0, -v_down), 370e3, veh,
                                       pad_altitude_m=5.0, guidance=g)


def test_a_slow_arrival_cannot_land_on_the_whole_engine_set_but_can_on_a_staged_one():
    whole = _predict(250.0, lit_at_start=True)
    staged = _predict(250.0, lit_at_start=True, engine_stage_fractions=(1.0, 5 / 13, 3 / 13))
    assert not whole.feasible                                 # 13 engines at minimum throttle cannot hover down
    assert staged.feasible and staged.touchdown_vertical_m_s <= 3.0


# ---------------------------------------------------------------- the engine-lit hand-over

def test_lit_at_start_lights_at_the_hand_over_instead_of_late():
    late = _predict(350.0)
    lit = _predict(350.0, lit_at_start=True)
    assert lit.ignition_altitude_m == pytest.approx(3000.0)   # lit at the hand-over altitude
    assert late.ignition_altitude_m < 2500.0                  # the hoverslam waits for the deceleration it needs
    assert TerminalGuidance().lit_at_start is False and TerminalGuidance().engine_stage_fractions == ()


# ---------------------------------------------------------------- the planner guard

def test_fins_with_an_entry_burn_need_the_explicit_flag():
    import datetime as dt

    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget, plan_boostback_entry
    from orbitopt.flightlab.recovery.propagator import BoosterState

    scen = Path(__file__).resolve().parents[1] / "src/orbitopt/data/flightlab/scenarios/leo_starship_class_rtls_e508_starbase.yaml"
    rt = RecoveryTarget.from_scenario(load_target(scen).to_dict())
    sep = BoosterState(t_s=0.0, epoch_utc=dt.datetime(2026, 11, 10), frame="ECEF",
                       position_m=(6.4e6, 0.0, 0.0), velocity_m_s=(0.0, 1000.0, 0.0), mass_kg=1.0e6)
    aero = {"mode": "schedule", "alpha_deg": 30.0, "schedule": [[70000.0, 0.0, 30.0]]}
    with pytest.raises(ValueError, match="entry_aero_with_burn"):
        plan_boostback_entry(sep, BoosterModel.load(VEH), rt, entry_burns=1, entry_aero=aero)


# ---------------------------------------------------------------- fixed fins (Super Heavy) vs fold-away fins

def test_a_fold_away_fin_deploys_for_entry_and_stows_before_the_landing_burn_but_a_fixed_fin_never_folds():
    from orbitopt.flightlab.recovery.grid_fins import GridFinActuator, GridFinCommand

    heights = [120_000.0, 60_000.0, 20_000.0, 800.0]               # above the deploy line, in the fin window, low
    folding = GridFinActuator(_model())
    fixed = GridFinActuator(_model(stowable=False))
    assert not folding.deployed and fixed.deployed                 # a fixed fin is out from the start
    seen_fold, seen_fixed = [], []
    for h in heights:
        for _ in range(3):
            folding.step(0.5, GridFinCommand(30.0, 0.0), height_m=h, dynamic_pressure_pa=5000.0, mach=3.0)
            fixed.step(0.5, GridFinCommand(30.0, 0.0), height_m=h, dynamic_pressure_pa=5000.0, mach=3.0)
        seen_fold.append(folding.deployed)
        seen_fixed.append(fixed.deployed)
    assert seen_fold == [False, True, True, False]                 # 70 km down to 1.5 km, then folded away
    assert seen_fixed == [True, True, True, True]                  # never folds
    # present but not commanded: neutral, yet still a fin in the airstream (its profile drag stays)
    snap = fixed.step(0.5, None, height_m=120_000.0, dynamic_pressure_pa=1.0, mach=9.0)
    assert snap.deployed and snap.alpha_deg == 0.0 and snap.cd_increment > 0.0


def test_the_guidance_keeps_commanding_a_fixed_fin_at_any_height_and_a_folded_one_not():
    folding = _law()
    fixed = _law(model=_model(stowable=False))
    assert _cmd(folding, 100_000.0, 700.0) is None and _cmd(folding, 800.0, 100.0) is None     # stowed: no command
    assert _cmd(fixed, 100_000.0, 700.0) is not None and _cmd(fixed, 800.0, 100.0) is not None


def test_the_independent_twin_honours_the_fixed_fin_too():
    from orbitopt.verify.full_recovery_reflight import _TwinFins

    class Gf:                                                      # the attributes the twin reads
        alpha_authority_deg, bank_max_deg, stow_altitude_m, deploy_altitude_m, deploy_q_min_pa = 60.0, 60.0, 1500.0, 70000.0, 100.0

    fold, fixed = Gf(), Gf()
    fold.stowable, fixed.stowable = True, False
    assert not _TwinFins(fold).deployed and _TwinFins(fixed).deployed
    assert _TwinFins(type("G", (), {k: v for k, v in vars(Gf).items() if not k.startswith("_")})()).deployed is False   # no flag: folds


def test_the_super_heavy_fin_vehicle_declares_its_fins_fixed_and_the_dc8_legacy_vehicle_still_folds():
    fins = BoosterModel.load(VEH).grid_fins
    assert fins.stowable is False
    dc8 = Path(VEH).with_name("starship_class_rtls_dc8.yaml")
    assert BoosterModel.load(dc8).grid_fins.stowable is True       # the E11-01 results are unchanged
