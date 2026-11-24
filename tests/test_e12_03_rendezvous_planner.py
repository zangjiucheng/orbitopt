"""E12-03 (task #393): two-impulse rendezvous planner, reference mission
LEO-RDV-001.

Gates the whole pipeline end to end: the packaged config loads against its
schema with its cross-checks (mu must be the exact Earth value, the gate
ordering has to be drift-reachable), the planner's drift + Lambert + Newton
plan verifies -- via `verify_plan`, which re-propagates from the config
states on the nonlinear pair propagator -- inside the task acceptance
numbers (<= 50 m at the 1 km Hold Point, <= 0.2 m/s relative speed), with
the maneuver epochs inside the fixed windows and total delta-v under the
budget. The verdict-bearing Lambert mu regression itself lives in
tests/test_e12_03_lambert_earth_mu.py.
"""
from __future__ import annotations

import pytest

from orbitopt.rendezvous import (
    RendezvousConfigError,
    RendezvousPlanError,
    load_reference_config,
    parse_rendezvous_config,
    plan_rendezvous,
    verify_plan,
)


@pytest.fixture(scope="module")
def cfg():
    return load_reference_config()


@pytest.fixture(scope="module")
def plan(cfg):
    return plan_rendezvous(cfg)


@pytest.fixture(scope="module")
def result(cfg, plan):
    return verify_plan(cfg, plan)


def test_reference_config_loads(cfg):
    assert cfg.mission_id == "leo-rdv-001"
    assert cfg.dynamics.frame == "GCRS"
    assert cfg.dynamics.mu == 3.9860044150e14
    assert cfg.hold_point_rel_lvlh == (0.0, -1000.0, 0.0)
    assert cfg.max_position_error_m == 50.0
    assert cfg.max_relative_speed_m_s == 0.2
    assert cfg.delta_v_budget_m_s == 25.0
    assert cfg.windows.transfer_tof_min_s < cfg.windows.transfer_tof_max_s


def test_plan_lands_inside_hold_and_budget(result):
    """The task's acceptance, on the verifying propagation."""
    assert result.passed
    assert all(result.checks.values())
    assert result.position_error_m <= 50.0
    assert result.relative_speed_m_s <= 0.2
    assert result.total_delta_v_m_s <= result.delta_v_budget_m_s
    assert result.delta_v_margin_m_s > 0.0


def test_arrival_is_a_hover_at_the_hold_point(result):
    x, y, z = result.arrival_rel_lvlh_m
    vx, vy, vz = result.arrival_rel_lvlh_rate_m_s
    assert y == pytest.approx(-1000.0, abs=50.0)
    assert abs(x) <= 50.0 and abs(z) <= 50.0
    assert (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5 <= 0.2


def test_plan_reports_the_maneuver_sequence(plan):
    d = plan.to_dict()
    assert [m["kind"] for m in d["maneuvers"]] == ["transfer_injection", "hold_point_insertion"]
    e1, e2 = (m["epoch_s"] for m in d["maneuvers"])
    assert 0.0 < e1 < e2 == plan.arrival_epoch_s == plan.drift_end_s + plan.tof_s
    assert d["total_delta_v_m_s"] == pytest.approx(
        sum(m["magnitude_m_s"] for m in d["maneuvers"]), abs=1e-9
    )
    assert d["delta_v_budget_m_s"] == 25.0
    assert d["budget_margin_m_s"] == pytest.approx(
        d["delta_v_budget_m_s"] - d["total_delta_v_m_s"], abs=1e-9
    )
    assert all(len(m["delta_v_eci_m_s"]) == 3 for m in d["maneuvers"])


def test_windows_are_respected(cfg, plan):
    assert plan.drift_end_s <= cfg.windows.drift_max_s
    assert cfg.windows.transfer_tof_min_s <= plan.tof_s <= cfg.windows.transfer_tof_max_s


def test_plan_is_deterministic(cfg, plan):
    other = plan_rendezvous(cfg)
    assert other.to_dict() == plan.to_dict()


def test_narrowed_window_costs_at_least_as_much(cfg, plan):
    """The window is the user's, not the optimizer's: pinning it to the
    low-cost end of the reference window must not find a cheaper plan."""
    doc_narrow = _mutate(cfg, windows={"transfer_tof_min_s": 1400.0, "transfer_tof_max_s": 1600.0})
    narrow = plan_rendezvous(parse_rendezvous_config(doc_narrow))
    assert 1400.0 <= narrow.tof_s <= 1600.0
    assert narrow.total_delta_v_m_s > plan.total_delta_v_m_s


def test_gate_beyond_drift_window_raises(cfg):
    doc = _mutate(cfg, windows={"drift_max_s": 400.0})
    with pytest.raises(RendezvousPlanError, match="departure gate"):
        plan_rendezvous(parse_rendezvous_config(doc))


def test_budget_and_tolerance_verdicts_are_independent(cfg, plan):
    tight = _mutate(cfg, tolerances={"max_position_error_m": 1e-8})
    res = verify_plan(parse_rendezvous_config(tight), plan)
    assert not res.checks["position_error_within_tolerance"]
    assert res.checks["relative_speed_within_tolerance"]  # 5e-10 m/s, still fine
    assert not res.passed

    stingy = _mutate(cfg, budget={"delta_v_budget_m_s": 5.0})
    res2 = verify_plan(parse_rendezvous_config(stingy), plan)
    assert not res2.checks["delta_v_within_budget"]
    assert res2.checks["position_error_within_tolerance"]
    assert res2.delta_v_margin_m_s < 0.0
    assert not res2.passed


def test_verify_rejects_foreign_plan(cfg, plan):
    other = _mutate(cfg, mission={"id": "leo-rdv-999"})
    foreign = plan_rendezvous(parse_rendezvous_config(other))
    with pytest.raises(ValueError, match="leo-rdv-999"):
        verify_plan(cfg, foreign)


# ---- config contract -------------------------------------------------------


def test_solar_mu_is_rejected_by_the_loader(cfg):
    doc = _mutate(cfg, dynamics={"mu": 1.32712440018e20})
    with pytest.raises(RendezvousConfigError, match="Earth"):
        parse_rendezvous_config(doc)


def test_moon_mu_is_rejected_by_the_loader(cfg):
    doc = _mutate(cfg, dynamics={"mu": 4.9028001e12})
    with pytest.raises(RendezvousConfigError, match="Earth"):
        parse_rendezvous_config(doc)


def test_chaser_must_start_below_target(cfg):
    doc = _mutate(cfg, chaser={"altitude_m": 510000.0})
    with pytest.raises(RendezvousConfigError, match="FORWARD"):
        parse_rendezvous_config(doc)


def test_gate_must_be_reachable(cfg):
    doc = _mutate(cfg, phasing={"departure_gate_y_m": -19000.0})
    with pytest.raises(RendezvousConfigError, match="along-track"):
        parse_rendezvous_config(doc)


def test_unknown_key_is_rejected(cfg):
    doc = _mutate(cfg)
    doc["surprise"] = True
    with pytest.raises(RendezvousConfigError):
        parse_rendezvous_config(doc)


def test_wrong_kind_is_rejected(cfg):
    doc = _mutate(cfg, mission={"kind": "hohmann-only"})
    with pytest.raises(RendezvousConfigError):
        parse_rendezvous_config(doc)


def test_transfer_window_ordering(cfg):
    doc = _mutate(cfg, windows={"transfer_tof_min_s": 4000.0, "transfer_tof_max_s": 3600.0})
    with pytest.raises(RendezvousConfigError, match="must exceed"):
        parse_rendezvous_config(doc)


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
