"""F-06 (task #231) regressions: the analytic two-stage split conserves mass
and momentum, the upper stage ignites exactly on its (configurable) schedule
and never before, the two children integrate independently and diverge, the
booster leaves through the M-01 separation->boostback handoff, and the whole
study serialises as a valid three-body FlightResult v1 run.

Real RK4 physics, same budget as the F-04/F-05 closed-loop tests:

    PYTHONPATH=src python -m pytest tests/test_flightlab_sixdof_two_stage.py
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from orbitopt.flightlab.contract import read_run, write_run
from orbitopt.flightlab.handoff import HandoffState, handoff_digest
from orbitopt.flightlab.sixdof.rigidbody import (
    RigidBodyState,
    dcm_from_quaternion,
)
from orbitopt.flightlab.sixdof.two_stage import (
    IGNITION_ONSET_SLACK_S,
    MASS_CLOSURE_TOL_KG,
    IgnitionPlan,
    StagePair,
    delayed_ignition_control,
    f06_two_stage_scenario,
    run_two_stage,
    split_stack_state,
    to_flight_run,
)

pytestmark = pytest.mark.slow  # real RK4 propagations


@pytest.fixture(scope="module")
def study():
    return run_two_stage(f06_two_stage_scenario())


# --------------------------------------------------------------------------
# the pair contract and the pure split (no integration)
# --------------------------------------------------------------------------

def test_stage_pair_rejects_inconsistent_masses(study):
    s = study.scenario
    with pytest.raises(ValueError, match="stage-1 engine"):
        StagePair(stack=s.pair.stack,
                  booster=replace(s.pair.booster, propellant_mass_kg=59_000.0),
                  upper=s.pair.upper)
    with pytest.raises(ValueError, match="inert ballast"):
        StagePair(stack=s.pair.stack, booster=s.pair.booster,
                  upper=replace(s.pair.upper, dry_mass_kg=2_500.0))


def test_split_conserves_mass_and_momentum_exactly():
    """Pure split arithmetic: a hand-built frozen state, analytic residuals."""
    s = f06_two_stage_scenario()
    frozen = RigidBodyState.from_launch(lat_geod_deg=28.5, lon_deg=-80.6, alt_m=70_000.0,
                                        heading_deg=90.0, pitch_deg=60.0,
                                        v_east_m_s=2400.0, v_down_m_s=-600.0,
                                        mass_kg=24_900.0, t_s=95.0)
    b_state, u_state, bk = split_stack_state(frozen, s.pair, stack_mass_at_t0_kg=78_000.0,
                                             impulse_ns=30_000.0)
    # Mass closes exactly, and every kilogram is accounted per stage.
    assert abs(bk["mass_residual_kg"]) < 1e-9
    assert b_state.mass_kg + u_state.mass_kg == pytest.approx(frozen.mass_kg, abs=1e-9)
    assert bk["booster_propellant_remaining_kg"] + bk["stage1_propellant_burned_kg"] \
        == pytest.approx(60_000.0, abs=1e-9)
    assert bk["upper_propellant_kg"] == pytest.approx(6_000.0, abs=1e-9)  # inert ballast
    assert b_state.mass_kg >= s.pair.booster.dry_mass_kg
    assert u_state.mass_kg >= s.pair.upper.dry_mass_kg
    # Momentum is conserved to rounding despite a 30 kN.s separation kick.
    p0 = np.asarray(frozen.velocity_m_s) * frozen.mass_kg
    p_norm = float(np.linalg.norm(p0))
    assert bk["momentum_residual_kg_m_s"] < 1e-6 * p_norm
    # The kick is along body +x: booster pushed aft, upper pushed forward.
    x_ecef = np.asarray(dcm_from_quaternion(frozen.quaternion)[0], dtype=float)
    v0 = np.asarray(frozen.velocity_m_s)
    assert float((np.asarray(b_state.velocity_m_s) - v0) @ x_ecef) < 0.0
    assert float((np.asarray(u_state.velocity_m_s) - v0) @ x_ecef) > 0.0
    # Attitude, rates and position hand over verbatim (analytic, no property I/O).
    assert tuple(b_state.position_m) == tuple(frozen.position_m)
    assert np.allclose(u_state.quaternion, frozen.quaternion)
    assert np.allclose(b_state.omega_rad_s, frozen.omega_rad_s)


# --------------------------------------------------------------------------
# the flown study
# --------------------------------------------------------------------------

def test_all_acceptance_checks_pass(study):
    by_id = {c.id: c for c in study.checks}
    assert set(by_id) == {"staging-mass-closure", "staging-momentum-residual",
                          "staging-handoff-continuity", "staging-scheduled-ignition",
                          "staging-trajectory-divergence"}
    for check in study.checks:
        assert check.passed, f"{check.id} failed (value={check.value}, limit={check.limit})"


def test_separation_conservation_on_the_flown_stack(study):
    sep = study.separation
    assert abs(sep["mass_residual_kg"]) < MASS_CLOSURE_TOL_KG
    p = np.asarray(sep["momentum_before_kg_m_s"], float)
    assert sep["momentum_residual_kg_m_s"] < max(0.01, 1e-6 * float(np.linalg.norm(p)))
    # The booster keeps stage-1 propellant for its recovery segments, and the
    # upper stage is exactly as full as it was inside the stack (inert).
    assert 0.0 < sep["booster_propellant_remaining_kg"] < 60_000.0
    assert sep["upper_propellant_kg"] == pytest.approx(6_000.0, abs=1e-9)
    assert study.booster_state.mass_kg + study.upper_state.mass_kg \
        == pytest.approx(study.frozen_state.mass_kg, abs=MASS_CLOSURE_TOL_KG)
    # Separation is before the stack runs stage-1 dry (MECO ~ t = 100 s).
    assert study.stack.propellant_exhausted_t_s is None


def test_upper_stage_ignites_exactly_on_the_schedule(study):
    t_ig = study.scenario.ignition.ignition_t_s
    t = study.upper.samples["t_s"]
    th = study.upper.samples["throttle"]
    early = t < t_ig - 1e-6
    assert early.any()
    assert not (th[early] > 0.0).any(), "the upper stage lit before its scheduled instant"
    after = t >= t_ig + 1e-6
    assert after.any()
    ex = study.upper.propellant_exhausted_t_s
    assert ex is not None and ex > t_ig
    lit = (t >= t_ig + 1e-6) & (t < ex - 1e-9)
    assert np.allclose(th[lit], 1.0)  # full throttle through the live burn
    assert not (th[t >= ex - 1e-9] > 0.0).any()  # engine shuts off at dry mass
    onset = {c.id: c.value for c in study.checks}["staging-scheduled-ignition"]
    assert 0.0 <= onset <= IGNITION_ONSET_SLACK_S  # on time, to the sample grid
    # And the burn is real: the mass trace burns down, ending at dry + residue.
    assert study.upper["mass_kg"][-1] < study.upper["mass_kg"][0] - 1_000.0
    assert study.upper.propellant_exhausted_t_s is not None
    # The booster never relights: its engine was the stack's.
    assert not (study.booster.samples["throttle"] > 0.0).any()


def test_ignition_delay_is_configurable_end_to_end():
    """A shorter delay on a shortened horizon: ignition onset follows the
    plan, not the scenario default."""
    s = f06_two_stage_scenario(ignition_delay_s=3.0, t_end_s=120.0)
    st = run_two_stage(s)
    assert st.scenario.ignition.ignition_t_s == pytest.approx(98.0)
    onset = {c.id: c for c in st.checks}["staging-scheduled-ignition"].value
    assert 0.0 <= onset <= IGNITION_ONSET_SLACK_S
    assert all(c.passed for c in st.checks)


def test_delayed_ignition_control_law_shape():
    plan = IgnitionPlan(separation_t_s=100.0, ignition_delay_s=12.0, throttle=0.7)
    assert plan.ignition_t_s == pytest.approx(112.0)
    law = delayed_ignition_control(plan)
    state = RigidBodyState.from_launch(mass_kg=1000.0)
    assert law(111.999, state).throttle == 0.0
    cmd = law(112.0, state)
    assert cmd.throttle == pytest.approx(0.7)
    assert cmd.gimbal_pitch_rad == 0.0 and cmd.gimbal_yaw_rad == 0.0
    with pytest.raises(ValueError):
        IgnitionPlan(separation_t_s=100.0, ignition_delay_s=-1.0)


def test_children_are_independent_diverging_bodies(study):
    # Both children start on the separation instant and fly the same clock.
    assert study.booster["t_s"][0] == pytest.approx(study.scenario.separation_t_s)
    assert study.upper["t_s"][0] == pytest.approx(study.scenario.separation_t_s)
    assert study.booster["t_s"][-1] == pytest.approx(study.scenario.t_end_s)
    assert study.upper["t_s"][-1] == pytest.approx(study.scenario.t_end_s)
    # Same frozen position at t_sep, then they pull apart for good: the
    # ignited upper gains real delta-v, the booster arcs ballistically.
    d0 = float(np.linalg.norm(np.asarray(study.booster["x_m"][0]
                                         - study.upper["x_m"][0], float)))
    assert d0 < 1e-9
    assert study.trajectory_divergence_m() > 1000.0
    v0 = math.sqrt(sum(v * v for v in study.upper_state.velocity_m_s))
    v1 = float(np.sqrt(study.upper["vx_m_s"][-1] ** 2 + study.upper["vy_m_s"][-1] ** 2
                       + study.upper["vz_m_s"][-1] ** 2))
    assert v1 - v0 > 500.0
    # The booster goes up-and-over (apogee above the separation altitude),
    # i.e. it is its own body on its own trajectory, not a copy of the upper.
    h_sep = float(np.interp(study.scenario.separation_t_s, study.stack["t_s"],
                            study.stack["altitude_m"]))
    assert float(np.max(study.booster["altitude_m"])) > h_sep


def test_booster_handoff_feeds_the_recovery_chain(study):
    h = study.booster_handoff
    assert h.boundary == ("separation", "boostback")
    assert h.frame == "ECEF"
    assert h.t_s == pytest.approx(study.scenario.separation_t_s)
    assert h.mass_kg == pytest.approx(study.booster_state.mass_kg)
    assert h.dry_mass_kg == pytest.approx(study.scenario.pair.booster.dry_mass_kg)
    assert h.propellant_kg > 0.0  # stage-1 residue is the recovery margin
    # The M-01 document round-trips and pins by digest.
    assert HandoffState.from_dict(h.to_dict()) == h
    assert handoff_digest(h) == study.booster_handoff_digest == handoff_digest(
        HandoffState.from_dict(h.to_dict()))


def test_to_flight_run_is_a_valid_three_body_run(study, tmp_path):
    run = to_flight_run(study)
    assert {t.body_id for t in run.tracks} == {"stack", "booster", "upper"}
    by_id = {t.body_id: t for t in run.tracks}
    assert by_id["stack"].parent_body_id is None
    assert by_id["booster"].parent_body_id == "stack"
    assert by_id["upper"].parent_body_id == "stack"
    assert by_id["booster"].t_start_s == pytest.approx(study.scenario.separation_t_s)
    kinds = {(e.kind, e.body_id) for e in run.events}
    assert ("separation", "stack") in kinds
    assert ("ignition", "upper") in kinds
    assert run.verification.physics_verified is True
    write_run(tmp_path / "run", run)
    back = read_run(tmp_path / "run")
    assert {t.body_id for t in back.tracks} == {"stack", "booster", "upper"}
    assert len(back.tracks[0].t_s) == len(run.tracks[0].t_s)
    assert back.verification.checks == run.verification.checks
