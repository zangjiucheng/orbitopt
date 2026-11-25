"""E12-04 (task #394): safety-constrained optimization, reference mission
LEO-RDV-001.

Gates the safety layer end to end: the optional ``safety`` block parses and
cross-checks (self-contradictory envelopes -- a keep-out sphere that swallows
the Hold Point, caps that loosen closer in -- are refused at the loader); the
whole trajectory of the constrained plan -- drift leg, coast between the two
burns, post-insertion hover, and three no-thrust passive releases -- passes
every configured constraint with measured positive margins; every check is
per-epoch with its binding sample located (epoch + LVLH position). The
negatives prove the checks bite: a constructed trajectory through the keep-out
sphere is failed with its crossing epoch reported, a real 1800 s transfer that
dips to ~95 m is failed by the safety gate while the E12-03 arrival gate still
passes it, a forward incursion inside the terminal cone and a fast approach
inside 1 km are failed, and windows/budgets that leave no safe arc raise with
the per-constraint reason.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbitopt.rendezvous import (
    RendezvousConfigError,
    RendezvousPlanError,
    evaluate_trajectory_safety,
    load_reference_config,
    parse_rendezvous_config,
    plan_rendezvous,
    plan_rendezvous_safe,
    verify_plan,
    verify_plan_safety,
)
from orbitopt.rendezvous.planner import _gate_crossing, _plan_from_solution, _solve_transfer
from orbitopt.rendezvous.safety import candidate_safety_reason


@pytest.fixture(scope="module")
def cfg():
    return load_reference_config()


@pytest.fixture(scope="module")
def result(cfg):
    return plan_rendezvous_safe(cfg)


@pytest.fixture(scope="module")
def safety(result):
    return result.safety


@pytest.fixture(scope="module")
def gate(cfg):
    return _gate_crossing(cfg)


# ---- the constrained reference plan ---------------------------------------


def test_safety_block_is_fixed_in_the_config(cfg):
    s = cfg.safety
    assert s is not None
    assert s.keep_out_radius_m == 500.0
    assert s.cross_track_max_m == 100.0
    assert s.cone_half_angle_deg == 90.0
    assert s.cone_range_m == 1100.0
    assert s.speed_range_knots_m == (500.0, 1000.0, 2500.0, 9500.0, 13000.0)
    assert s.speed_cap_m_s == (9.0, 9.0, 12.0, 22.0, 26.0)
    assert s.passive_contingency_s == 5400.0
    assert s.check_sample_dt_s == 2.0


def test_reference_mission_plans_safe(result, safety):
    assert result.passed
    assert safety.passed and safety.failed_checks == ()
    for name in ("keep_out", "terminal_cone", "cross_track", "approach_speed",
                 "fuel_budget", "time_window"):
        c = safety.checks[name]
        assert c.passed, name
        assert c.min_margin > 0.0, name
        assert c.violation_count == 0


def test_measured_margins_clear_the_reported_envelope(safety, result):
    """The packaged envelope was measured, not guessed: the constrained plan
    keeps at least these margins (grid-alias-independent floors from
    docs/adr/rendezvous-0002)."""
    assert safety.checks["keep_out"].min_margin > 380.0
    assert safety.checks["terminal_cone"].min_margin > 15.0
    assert safety.checks["cross_track"].min_margin > 70.0
    assert safety.checks["approach_speed"].min_margin > 2.0
    assert safety.checks["fuel_budget"].min_margin > 14.0
    assert result.plan.total_delta_v_m_s == pytest.approx(10.35, abs=0.05)


def test_full_trajectory_is_sampled_including_the_coast_between_burns(safety, cfg, result):
    c = safety.checks["keep_out"]
    dt = cfg.safety.check_sample_dt_s
    t = c.times_s
    assert t[0] == 0.0 and t[-1] == pytest.approx(result.plan.arrival_epoch_s)
    assert np.all(np.diff(t) <= dt + 1e-9)
    m1, m2 = result.plan.maneuvers
    assert np.any(t == m1.epoch_s)          # post-burn-1 coast start is a sample
    assert np.any(t == m2.epoch_s)          # pre-burn-2 and post-insertion too
    assert len(t) >= int(result.plan.arrival_epoch_s / dt)


def test_margins_are_per_epoch_and_the_binding_sample_is_located(safety, cfg):
    c = safety.checks["keep_out"]
    assert c.margins.shape == c.times_s.shape == (len(c.times_s),)
    assert c.positions_rel_lvlh_m.shape == (len(c.times_s), 3)
    assert c.worst_epoch_s is not None and 0.0 < c.worst_epoch_s
    rho_worst = np.linalg.norm(c.positions_rel_lvlh_m[np.argmin(c.margins)])
    assert rho_worst - cfg.safety.keep_out_radius_m == pytest.approx(c.min_margin, abs=1e-6)
    assert c.worst_rel_lvlh_m == tuple(c.positions_rel_lvlh_m[np.argmin(c.margins)])


def test_fuel_budget_check_is_cumulative_per_event(safety, cfg, result):
    c = safety.checks["fuel_budget"]
    m1, m2 = result.plan.maneuvers
    assert c.times_s == pytest.approx([0.0, m1.epoch_s, m2.epoch_s])
    assert c.margins[0] == pytest.approx(cfg.delta_v_budget_m_s)
    assert c.margins[-1] == pytest.approx(cfg.delta_v_budget_m_s - result.plan.total_delta_v_m_s)
    assert c.margins[0] > c.margins[1] > c.margins[2]


def test_passive_releases_stay_out_of_the_keep_out(safety, cfg):
    assert [p.kind for p in safety.passive] == [
        "no_burn_at_t0", "gate_abort_no_burn", "hold_point_coast",
    ]
    assert safety.passive[0].release_epoch_s == 0.0
    for p in safety.passive:
        assert p.passed and p.duration_s == cfg.safety.passive_contingency_s
        assert p.min_range_m - cfg.safety.keep_out_radius_m == pytest.approx(p.keep_out_margin_m)
    assert min(p.keep_out_margin_m for p in safety.passive) >= 500.0 - 1e-6
    assert max(p.keep_out_margin_m for p in safety.passive) > 9000.0


def test_report_serializes_for_review(safety):
    d = json.loads(json.dumps(safety.to_dict()))
    assert set(d["checks"]) == {"keep_out", "terminal_cone", "cross_track",
                                "approach_speed", "fuel_budget", "time_window"}
    k = d["checks"]["keep_out"]
    assert k["passed"] and len(k["margins"]) == len(k["epochs_s"]) == len(k["rel_lvlh_m"])
    assert d["passive_contingency"][0]["kind"] == "no_burn_at_t0"


# ---- negatives: the checks must bite ---------------------------------------


def test_constructed_keep_out_crossing_is_failed(cfg):
    """The acceptance negative: a trajectory hand-built straight through the
    500 m sphere must be judged failed, with the crossing located."""
    t = np.linspace(0.0, 100.0, 101)
    x = 2000.0 - 40.0 * t
    rho = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    rhod = np.tile(np.array([-40.0, 0.0, 0.0]), (t.size, 1))
    checks = evaluate_trajectory_safety(cfg, t, rho, rhod)
    c = checks["keep_out"]
    assert not c.passed
    assert c.violation_count >= 20
    assert c.violations[0]["epoch_s"] == pytest.approx(38.0)
    assert np.linalg.norm(c.violations[0]["rel_lvlh_m"]) < cfg.safety.keep_out_radius_m
    assert checks["cross_track"].passed          # stayed in plane: only what it broke failed
    assert not checks["approach_speed"].passed   # and 40 m/s busts every cap


def test_real_transfer_that_dips_into_the_sphere_is_a_safety_failure(cfg, gate):
    """tof = 1800 s solves and ARRIVES (E12-03 gate passes) -- yet its coast
    dips ~95 m from the target, so the safety gate must fail it, per epoch."""
    sol = _solve_transfer(cfg, *gate, 1800.0)
    plan = _plan_from_solution(cfg, gate[0], 1800.0, sol)
    assert verify_plan(cfg, plan).passed        # arrival is not the safety gate
    rep = verify_plan_safety(cfg, plan)
    c = rep.checks["keep_out"]
    assert not c.passed and not rep.passed
    assert c.min_margin < -300.0                # dips to ~95 m: margin ~ -405 m
    assert c.violation_count > 0
    first = c.violations[0]
    assert plan.drift_end_s < first["epoch_s"] < plan.arrival_epoch_s
    assert np.linalg.norm(first["rel_lvlh_m"]) < cfg.safety.keep_out_radius_m
    assert candidate_safety_reason(cfg, sol).startswith("keep_out")


def test_forward_incursion_inside_the_terminal_cone_is_failed(cfg):
    """One sample at 1 km with the deputy FORWARD of the target plane."""
    rho = np.array([[600.0, 800.0, 0.0]])
    rhod = np.zeros((1, 3))
    checks = evaluate_trajectory_safety(cfg, np.array([10.0]), rho, rhod)
    c = checks["terminal_cone"]
    assert not c.passed
    assert c.min_margin == pytest.approx(90.0 - np.degrees(np.arccos(-0.8)), abs=1e-9)
    assert checks["keep_out"].passed and checks["cross_track"].passed


def test_fast_approach_inside_a_kilometre_is_failed(cfg):
    rho = np.array([[0.0, -900.0, 0.0]])
    rhod = np.array([[5.0, -9.0, 0.0]])          # 10.3 m/s at 900 m, cap 9
    checks = evaluate_trajectory_safety(cfg, np.array([20.0]), rho, rhod)
    assert not checks["approach_speed"].passed
    assert checks["approach_speed"].min_margin < 0.0
    assert checks["keep_out"].passed and checks["terminal_cone"].passed


def test_window_with_only_unsafe_arcs_raises_with_the_reason(cfg):
    doc = _mutate(cfg, windows={"transfer_tof_min_s": 1600.0, "transfer_tof_max_s": 1900.0})
    with pytest.raises(RendezvousPlanError, match="keep_out"):
        plan_rendezvous_safe(parse_rendezvous_config(doc))
    # ... and it is the SAFETY layer refusing it, not the transfer: the plain
    # E12-03 planner still finds an arriving plan in the same window.
    unsafe = plan_rendezvous(parse_rendezvous_config(doc))
    assert 1600.0 <= unsafe.tof_s <= 1900.0


def test_budget_below_every_safe_arc_raises_with_the_reason(cfg):
    doc = _mutate(cfg, budget={"delta_v_budget_m_s": 5.0})
    with pytest.raises(RendezvousPlanError, match="fuel_budget"):
        plan_rendezvous_safe(parse_rendezvous_config(doc))


# ---- the config boundary ----------------------------------------------------


def test_loader_refuses_a_self_contradictory_envelope(cfg):
    doc = _mutate(cfg, safety={**_packaged_safety(cfg), "keep_out_radius_m": 1200.0})
    with pytest.raises(RendezvousConfigError, match="Hold Point"):
        parse_rendezvous_config(doc)

    doc = _mutate(cfg, safety={**_packaged_safety(cfg), "approach_speed": {
        "range_knots_m": [500.0, 1000.0, 2500.0, 9500.0, 13000.0],
        "cap_m_s": [22.0, 9.0, 12.0, 22.0, 26.0],
    }})
    with pytest.raises(RendezvousConfigError, match="non-decreasing"):
        parse_rendezvous_config(doc)

    doc = _mutate(cfg, safety={**_packaged_safety(cfg), "approach_speed": {
        "range_knots_m": [500.0, 1000.0, 1000.0, 9500.0, 13000.0],
        "cap_m_s": [9.0, 9.0, 12.0, 22.0, 26.0],
    }})
    with pytest.raises(RendezvousConfigError, match="strictly increase"):
        parse_rendezvous_config(doc)


def test_configs_without_the_safety_block_still_work(cfg):
    doc = _mutate(cfg)
    del doc["safety"]
    cfg2 = parse_rendezvous_config(doc)
    assert cfg2.safety is None
    plan = plan_rendezvous(cfg2)               # E12-03 pipeline unchanged
    assert verify_plan(cfg2, plan).passed
    with pytest.raises(RendezvousConfigError, match="safety"):
        plan_rendezvous_safe(cfg2)
    with pytest.raises(RendezvousConfigError, match="safety"):
        verify_plan_safety(cfg2, plan)


def test_constrained_planner_is_deterministic(cfg, result):
    assert plan_rendezvous_safe(cfg).plan.to_dict() == result.plan.to_dict()


# ---- helpers ----------------------------------------------------------------


def _packaged_safety(cfg) -> dict:
    s = cfg.safety
    return {
        "keep_out_radius_m": s.keep_out_radius_m,
        "cross_track_max_m": s.cross_track_max_m,
        "cone_half_angle_deg": s.cone_half_angle_deg,
        "cone_range_m": s.cone_range_m,
        "approach_speed": {"range_knots_m": list(s.speed_range_knots_m),
                           "cap_m_s": list(s.speed_cap_m_s)},
        "passive_contingency_s": s.passive_contingency_s,
        "check_sample_dt_s": s.check_sample_dt_s,
    }


def _mutate(cfg, **sections) -> dict:
    """Rebuild the packaged reference document and override whole sections
    (the schema is strict, so overrides are complete mappings)."""
    import yaml
    from importlib import resources

    doc = yaml.safe_load(
        (resources.files("orbitopt") / "data" / "leo_rdv_001.yaml").read_text(encoding="utf-8")
    )
    for key, patch in sections.items():
        doc[key] = {**doc[key], **patch}
    return doc
