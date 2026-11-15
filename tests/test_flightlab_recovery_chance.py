"""Checks for orbitopt.flightlab.recovery.chance (E3-02 joint recovery
chance constraints) and their FL-302 wiring.

Tolerances -- declared up front:

    PHI_*               phi/phi_inv round-trip to 1e-9 on probe levels.
    DETERMINISM         The Rayleigh quantile consistency: p_divert at the
                        derived divert bar equals scatter_quantile exactly
                        (the deterministic equivalent is an identity of the
                        Rayleigh model, not a tuned match).
    FL305_BIT_EXACT     With the joint constraints relaxed to never bind
                        (tiny probability levels, astronomical scatter
                        budget) the chance-mode planner must reproduce the
                        FL-305 plan bit-for-bit on the same warm start --
                        the "disabled state" convention of the R-01 switch.

The chance module itself is pure math (fast); the planner integration tests
pin the script with ``initial_guess`` + ``max_fevals=1`` like the boostback
reason-code tests, plus one slow full-search comparison.
"""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.recovery import BoosterModel, RecoveryTarget
from orbitopt.flightlab.recovery.boostback import plan_boostback_entry
from orbitopt.flightlab.recovery.chance import (
    DEFAULT_MIN_LANDING_PROBABILITY,
    DEFAULT_SCATTER_BUDGET_M,
    DEFAULT_SCATTER_QUANTILE,
    JOINT_PENALTY_SCALE,
    JointChanceSpec,
    evaluate_joint_chance,
    joint_penalty,
    phi,
    phi_inv,
    rayleigh_quantile_factor,
)

from test_flightlab_recovery_boostback import PAD, VEHICLE_YAML, _SEED_PLAN, _sep_state


# ---------------------------------------------------------------------------
# pure math of the deterministic equivalents
# ---------------------------------------------------------------------------


def test_phi_round_trip():
    for p in (0.001, 0.1, 0.5, 0.9, 0.999):
        assert abs(phi(phi_inv(p)) - p) < 1e-9
    assert phi(0.0) == 0.5
    with pytest.raises(ValueError):
        phi_inv(0.0)


def test_rayleigh_quantile_factor_values():
    assert rayleigh_quantile_factor(0.5) == pytest.approx(math.sqrt(2 * math.log(2)))
    assert rayleigh_quantile_factor(DEFAULT_SCATTER_QUANTILE) == pytest.approx(2.1460, abs=1e-3)
    with pytest.raises(ValueError):
        rayleigh_quantile_factor(1.0)


def test_loss_sigma_zero_bands_zero_scale():
    spec = JointChanceSpec(thrust_pct=0.0, isp_pct=0.0, density_pct=0.0, cd_pct=0.0)
    assert spec.propellant_loss_sigma_kg(20_000.0) == 0.0
    assert spec.required_margin_kg(20_000.0) == 0.0
    v = evaluate_joint_chance(spec, miss_m=100.0, margin_kg=-5.0, exposure_kg=20_000.0)
    assert v.p_fuel == 0.0  # zero scale, negative margin: the shortfall is certain
    v0 = evaluate_joint_chance(spec, miss_m=100.0, margin_kg=0.0, exposure_kg=20_000.0)
    assert v0.p_fuel == 0.5


def test_loss_sigma_scales_linearly_with_exposure():
    spec = JointChanceSpec()
    s1, s2 = spec.propellant_loss_sigma_kg(10_000.0), spec.propellant_loss_sigma_kg(25_000.0)
    assert s2 == pytest.approx(2.5 * s1)
    # calibration band: ~0.04 fractional sigma of the burn exposure
    assert 0.03 < s1 / 10_000.0 < 0.05


def test_divert_bar_is_rayleigh_identity():
    spec = JointChanceSpec()
    bar = spec.divert_budget_m()
    # the deterministic equivalent is an identity: P(divert closes) at the
    # derived bar reproduces the quantile level exactly
    assert spec.p_divert(bar) == pytest.approx(DEFAULT_SCATTER_QUANTILE, abs=1e-12)
    # and monotone decreasing in the miss
    assert spec.p_divert(bar * 2.0) < spec.p_divert(bar) < spec.p_divert(0.0)


def test_divert_bar_derived_not_asserted():
    # doubling the scatter budget scales the derived divert bar by exactly
    # the Rayleigh linearity (sigma floor aside)
    spec = JointChanceSpec(sigma_floor_m=0.0)
    assert spec.divert_budget_m() == pytest.approx(
        2.0 * JointChanceSpec(sigma_floor_m=0.0,
                              scatter_budget_m=DEFAULT_SCATTER_BUDGET_M / 2
                              ).divert_budget_m())
    # a budget the sigma floor alone busts derives a zero bar
    assert JointChanceSpec(sigma_floor_m=600.0).divert_budget_m() == 0.0


def test_accepted_handoff_satisfies_defaults():
    """The recorded accepted handoff (ADR flightlab-0003: 351 m off, +1662 kg
    above need + reserve, ~20 t burn exposure) sits inside every default
    constraint -- the calibration anchor."""
    spec = JointChanceSpec()
    v = evaluate_joint_chance(spec, miss_m=351.0, margin_kg=1662.0, exposure_kg=20_000.0)
    assert v.worst_violation == 0.0
    assert v.p_land > spec.min_landing_probability
    assert joint_penalty(v) == 0.0


def test_tail_handoff_violates_and_is_penalized_hard():
    """The dispersion tail R-07 measured (1.7 km divert, zero margin): all
    three constraints active, p_land far below the bound, and the exact
    penalty lands above the impact-branch floor (25.0) that it must dominate."""
    spec = JointChanceSpec()
    v = evaluate_joint_chance(spec, miss_m=1760.0, margin_kg=0.0, exposure_kg=20_000.0)
    assert all(g > 0.0 for g in v.violations)
    assert v.p_land < spec.min_landing_probability
    assert joint_penalty(v) > JOINT_PENALTY_SCALE


def test_worst_violation_not_tradable():
    """l-infinity aggregation: improving one violated constraint below the
    worst one changes nothing -- the constraint trade R-01's additive terms
    allowed is structurally gone."""
    spec = JointChanceSpec()
    a = evaluate_joint_chance(spec, miss_m=2000.0, margin_kg=-500.0, exposure_kg=20_000.0)
    b = evaluate_joint_chance(spec, miss_m=900.0, margin_kg=-500.0, exposure_kg=20_000.0)
    assert joint_penalty(b) == joint_penalty(a)  # fuel still governs both


def test_spec_validation():
    for kw in (dict(p_fuel_min=0.0), dict(p_fuel_min=1.0),
               dict(min_landing_probability=1.5), dict(scatter_quantile=0.0),
               dict(scatter_budget_m=-1.0), dict(sigma_floor_m=-1.0),
               dict(thrust_pct=-1.0)):
        with pytest.raises(ValueError):
            JointChanceSpec(**kw)


def test_spec_from_option_and_bands():
    assert JointChanceSpec.from_option(None) is None
    s = JointChanceSpec(p_fuel_min=0.8)
    assert JointChanceSpec.from_option(s) is s
    assert JointChanceSpec.from_option({"p_fuel_min": 0.7}).p_fuel_min == 0.7
    with pytest.raises(TypeError):
        JointChanceSpec.from_option(3.0)
    with pytest.raises(TypeError):
        JointChanceSpec.from_option({"bogus": 1.0})

    class Bands:
        thrust_pct, isp_pct, density_pct, cd_pct = 4.0, 2.0, 9.0, 6.0

    b = JointChanceSpec.from_dispersion_bands(Bands(), p_fuel_min=0.75)
    assert (b.thrust_pct, b.isp_pct, b.density_pct, b.cd_pct, b.p_fuel_min) \
        == (4.0, 2.0, 9.0, 6.0, 0.75)


# ---------------------------------------------------------------------------
# FL-302 wiring (pinned scripts like the boostback reason-code tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vehicle():
    return BoosterModel.load(str(VEHICLE_YAML))


@pytest.fixture(scope="module")
def pad_target():
    return RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                          longitude_deg=PAD[1], altitude_m=PAD[2])


SEP = dict(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0, mass=65_000.0)


def test_joint_chance_requires_r01_pinned_off(vehicle, pad_target):
    st = _sep_state(**SEP)
    with pytest.raises(ValueError, match="pin them off"):
        plan_boostback_entry(st, vehicle, pad_target, initial_guess=_SEED_PLAN,
                            max_fevals=1, joint_chance=JointChanceSpec())
    with pytest.raises(ValueError, match="pin them off"):
        plan_boostback_entry(st, vehicle, pad_target, initial_guess=_SEED_PLAN,
                            max_fevals=1, joint_chance={},
                            landing_reserve_margin_kg=1.0)


def test_chance_mode_reports_probabilities_at_the_handoff(vehicle, pad_target):
    """A pinned script whose raw handoff is the dispersion-tail one (miss
    ~1.8 km, thin margin): the chance-mode report must carry the probability
    and the positive violation the exact penalty is steering against -- and
    nothing of the R-01 soft terms, which the joint mode replaces."""
    st = _sep_state(**SEP)
    plan = plan_boostback_entry(
        st, vehicle, pad_target, initial_guess=_SEED_PLAN, max_fevals=1,
        landing_divert_budget_m=None, landing_reserve_margin_kg=0.0,
        joint_chance=JointChanceSpec())
    rep = plan.corridor_report
    assert rep.reached  # the pinned script does cross the corridor
    assert 0.0 < rep.landing_probability < 1.0
    assert rep.joint_violation > 0.0  # 1.8 km divert: outside the derived bar
    # FL-305 / R-01 objective fields stay None: the joint mode replaced the
    # soft terms (which must be pinned off for it)
    assert rep.landing_divert_budget_m is None
    assert rep.min_landing_reserve_margin_kg is None
    assert any("E3-02 joint chance" in a for a in plan.assumptions)
    assert not any("R-01 landing-aware" in a for a in plan.assumptions)


def test_chance_mode_violating_handoff_is_recorded(vehicle, pad_target):
    """A deliberately bad script (no boostback at all, entry burn from a low
    trigger) reaches the corridor far off and short: the verdict comes back
    with a positive violation and a depressed probability."""
    st = _sep_state(**SEP)
    plan = plan_boostback_entry(
        st, vehicle, pad_target,
        initial_guess=[150.0, 4.0, 0.3, 0.0, 0.0, 6_000.0, 3.0], max_fevals=1,
        landing_divert_budget_m=None, landing_reserve_margin_kg=0.0,
        joint_chance={"min_landing_probability": 0.5})
    rep = plan.corridor_report
    if rep.reached:  # crossed far away: the report must SEE that
        assert rep.joint_violation > 0.0
        assert rep.landing_probability < 0.9
    else:
        assert rep.landing_probability is None
        assert not plan.feasible


@pytest.mark.slow
def test_chance_mode_steers_to_a_constraint_satisfied_handoff(vehicle, pad_target):
    """At the MC convention (FL-302 warm start, 30 evaluations) the DEFAULT
    joint spec must deliver a feasible plan whose handoff satisfies every
    constraint with margin -- the accepted-handoff anchor of ADR
    flightlab-0003 (351 m, ~+1.7 t) sits inside the default bars, so the
    mode switch costs nothing where the baseline was already good."""
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS
    st = _sep_state(**SEP)
    plan = plan_boostback_entry(st, vehicle, pad_target,
                                initial_guess=NOMINAL_PLAN_GUESS, max_fevals=30,
                                landing_divert_budget_m=None,
                                landing_reserve_margin_kg=0.0,
                                joint_chance=JointChanceSpec())
    rep = plan.corridor_report
    assert plan.feasible
    assert rep.reached
    assert rep.landing_probability >= DEFAULT_MIN_LANDING_PROBABILITY
    assert rep.joint_violation == 0.0
    assert 0.0 <= rep.landing_probability <= 1.0


@pytest.mark.slow
def test_never_binding_joint_spec_reproduces_fl305(vehicle, pad_target):
    """Disabled-state convention: with p levels low enough to never bind and
    an astronomical scatter budget the exact penalty is an exact 0 addition
    at every rollout, so Nelder-Mead walks the identical path and the plan is
    bit-for-bit the FL-305 objective's."""
    kwargs = dict(initial_guess=_SEED_PLAN, max_fevals=18)
    fl305 = plan_boostback_entry(_sep_state(**SEP), vehicle, pad_target,
                                 landing_divert_budget_m=None,
                                 landing_reserve_margin_kg=0.0, **kwargs)
    spec = JointChanceSpec(p_fuel_min=1e-6, min_landing_probability=1e-6,
                           scatter_budget_m=1e12, scatter_quantile=0.99,
                           sigma_floor_m=1e9)
    # p levels a millimetre above the axis: no crossed rollout of this warm
    # start comes within 4.75 sigma of busting them, so the joint penalty is
    # an exact 0.0 addition at every evaluation.
    joint = plan_boostback_entry(_sep_state(**SEP), vehicle, pad_target,
                                 landing_divert_budget_m=None,
                                 landing_reserve_margin_kg=0.0,
                                 joint_chance=spec, **kwargs)
    assert joint.boostback == fl305.boostback
    assert joint.entry_burn == fl305.entry_burn


def test_penalty_additive_identity_when_satisfied():
    """A satisfied verdict adds exactly 0.0 -- the disabled/enabled states
    compose with the FL-305 core penalty without perturbation."""
    spec = JointChanceSpec()
    v = evaluate_joint_chance(spec, miss_m=0.0, margin_kg=5e4, exposure_kg=10_000.0)
    assert joint_penalty(v) == 0.0 and v.worst_violation == 0.0
