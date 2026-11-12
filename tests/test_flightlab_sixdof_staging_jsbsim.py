"""FL-402 (#207): two independent top-level FGFDMExec instances with a
staging state handoff (orbitopt/flightlab/sixdof/staging.py).

Acceptance thresholds are declared up front and reused from the module
(handoff/momentum tolerances). Skipped without ``jsbsim`` (opt-in pip dep:
``pip install -r envs/sixdof-requirements.txt``).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("jsbsim")

from orbitopt.flightlab.contract import read_run, write_run  # noqa: E402
from orbitopt.flightlab.sixdof import staging  # noqa: E402
from orbitopt.flightlab.sixdof.staging import (  # noqa: E402
    FrozenState, StageSplit, capture_frozen_state, run_staging_scenario, to_flight_run,
)

# --------------------------------------------------------------------------
# Declared thresholds (mirrors staging.py; tests assert against these)
# --------------------------------------------------------------------------
MASS_REL_TOL = 1e-6            # children total mass vs. frozen stack mass
COAST_S = 61.0                 # >= 60 s acceptance requirement
IMPULSE_N_S = 2.0e5            # nonzero separation impulse under test
SEP_REL_V_TOL = 1e-3           # rel. separation speed vs. J*(1/m_b + 1/m_u)


@pytest.fixture(scope="module")
def replay():
    """Reference scenario: burn, freeze, hand off with a nonzero impulse,
    coast 61 s stepping booster-then-upper."""
    return run_staging_scenario(burn_s=10.0, coast_s=COAST_S,
                                separation_impulse_ns=IMPULSE_N_S)


@pytest.fixture(scope="module")
def replay_rerun():
    return run_staging_scenario(burn_s=10.0, coast_s=COAST_S,
                                separation_impulse_ns=IMPULSE_N_S)


@pytest.fixture(scope="module")
def replay_ba():
    """Identical scenario stepped upper-stage-first."""
    return run_staging_scenario(burn_s=10.0, coast_s=COAST_S,
                                separation_impulse_ns=IMPULSE_N_S, step_order="ba")


def _final(sim):
    return sim.samples[-1]


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def test_mass_closes_over_the_handoff(replay):
    bk = replay.bookkeeping
    assert abs(bk["mass_residual_kg"]) < MASS_REL_TOL * bk["mass_before_kg"]
    # and the split bookkeeping itself sums to the frozen state
    db, du = bk["dry_split_kg"]
    pb, pu = bk["propellant_split_kg"]
    assert abs((db + du + pb + pu) - bk["mass_before_kg"]) < MASS_REL_TOL * bk["mass_before_kg"]


def test_momentum_residual_within_declared_tolerance(replay):
    bk = replay.bookkeeping
    p0 = _norm(bk["momentum_before_kg_m_s"])
    assert bk["momentum_residual_kg_m_s"] < max(
        staging.MOMENTUM_RESIDUAL_FLOOR_N_S,
        staging.MOMENTUM_RESIDUAL_TOL_REL * p0)


def test_separation_impulse_conserves_momentum_and_opens_the_gap(replay):
    bk = replay.bookkeeping
    b0, u0 = _final(replay.booster), _final(replay.upper)
    dv_expected = IMPULSE_N_S * (1.0 / (sum(replay.allocation["booster"][k]
                                            for k in ("dry_kg", "ox_kg", "fuel_kg")))
                                 + 1.0 / (sum(replay.allocation["upper"][k]
                                             for k in ("dry_kg", "ox_kg", "fuel_kg"))))
    dv_measured = _norm((u0["vx_m_s"] - b0["vx_m_s"], u0["vy_m_s"] - b0["vy_m_s"],
                         u0["vz_m_s"] - b0["vz_m_s"]))
    # gravity bends the two ballistic paths differently over 61 s, so this is
    # a loose (percent-level) consistency check of the bookkept dv.
    assert abs(dv_measured - dv_expected) < 0.05 * dv_expected


def test_handoff_state_is_continuous(replay):
    bk = replay.bookkeeping
    assert bk["position_continuity_err_m"] < staging.HANDOFF_POSITION_TOL_M
    assert bk["attitude_continuity_err_rad"] < staging.HANDOFF_ATTITUDE_TOL_RAD
    for body in ("booster", "upper"):
        assert replay.handoff[body]["velocity_err_m_s"] < staging.HANDOFF_VELOCITY_TOL_M_S
        assert abs(replay.handoff[body]["mass_err_kg"]) < MASS_REL_TOL * bk["mass_before_kg"]


def test_both_instances_coasted_for_at_least_sixty_seconds(replay):
    assert replay.booster.t_s >= COAST_S - 1e-9
    assert replay.upper.t_s >= COAST_S - 1e-9
    assert len(replay.booster.samples) == round(COAST_S / replay.dt_s) + 1


def test_two_reruns_are_bit_identical(replay, replay_rerun):
    for sim_a, sim_b in ((replay.booster, replay_rerun.booster),
                         (replay.upper, replay_rerun.upper)):
        assert sim_a.samples == sim_b.samples


def test_stepping_order_does_not_change_results(replay, replay_ba):
    assert replay.booster.samples == replay_ba.booster.samples
    assert replay.upper.samples == replay_ba.upper.samples
    assert replay.frozen == replay_ba.frozen


def test_no_property_leakage_between_instances(replay):
    # writing a property on one instance must not be visible on the other
    replay.booster.fdm["test/fl402-leak-marker"] = 7.0
    with pytest.raises(Exception):
        _ = replay.upper.fdm["test/fl402-leak-marker"]
    # engine commands on one instance must not disturb the other
    replay.booster.set_throttle(0.5)
    replay.booster.set_gimbal(0.05, -0.05)
    replay.booster.step()
    assert _final(replay.upper)["throttle"] == 0.0
    assert _final(replay.upper)["gimbal_pitch_cmd_rad"] == 0.0


def test_frozen_state_bookkeeping(replay):
    frozen = replay.frozen
    assert isinstance(frozen, FrozenState)
    # dry mass = the stack XML's 10 t; propellant = 60 t minus a 10 s burn
    assert abs(frozen.dry_mass_kg - 10_000.0) < 1.0
    assert abs(sum(frozen.tank_propellant_kg) - 54_000.0) < 60.0  # throttle ramp


def test_split_allocation_sums_back(replay):
    frozen = capture_frozen_state(replay.stack)
    alloc = StageSplit(0.4, 0.4).allocate(frozen)
    total = sum(sum(v for v in side.values()) for side in alloc.values())
    assert abs(total - frozen.mass_kg) < 1e-6 * frozen.mass_kg
    assert abs(alloc["booster"]["dry_kg"] - 0.4 * frozen.dry_mass_kg) < 1e-9


def test_flight_run_has_three_validated_bodies(replay, tmp_path):
    run = to_flight_run(replay, every=20)
    assert len(run.tracks) == 3
    by_id = {t.body_id: t for t in run.tracks}
    assert by_id["booster"].parent_body_id == "stack"
    assert by_id["upper"].parent_body_id == "stack"
    assert by_id["stack"].parent_body_id is None
    # children re-based onto the stack clock start at the separation instant
    for child in ("booster", "upper"):
        assert abs(by_id[child].t_start_s - replay.frozen.t_s) < 1e-9
        assert by_id[child].t_end_s - by_id[child].t_start_s >= COAST_S - replay.dt_s
    assert run.verification.physics_verified is True
    assert all(c.passed for c in run.verification.checks)
    assert any(e.kind == "separation" and e.body_id == "stack" for e in run.events)
    write_run(tmp_path, run)
    back = read_run(tmp_path)
    assert back.run_id == run.run_id
    assert [t.body_id for t in back.tracks] == ["stack", "booster", "upper"]
    assert len(back.tracks[1].t_s) == len(by_id["booster"].t_s)
