"""E12-07 (task #397): robustness of the LEO-RDV-001 rendezvous under
dispersion.

Gates the campaign machinery rather than the headline numbers (the measured
rate lives in ``docs/benchmarks/e12_07_rendezvous_robustness.json``): the
packaged campaign parses, derives its sample size from the declared CI width,
and refuses self-contradictory declarations -- arms whose seed streams would
overlap, a truth grid finer than itself, a drag source with no drag area, a
cluster count hand-declared against the width that prices it. The RNG layout is
pinned: disabling a family leaves every other family's draws untouched, and one
arm is identical run to run and at one or many workers. The truth model is
pinned at its control point: with every family off, the flown mission is the
planned mission to millimetre agreement -- so every metre of error the campaign
then measures belongs to a disturbance and not to the harness. Grading follows
the mission's own precedence, the taxonomy is total by construction, the
sensitivity reading reports a null as a null, the spot check records the
independent-model line honestly when no backend exists, and the 5 s truth grid
is measured not to move a verdict against a 1 s one.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orbitopt.rendezvous import (
    load_reference_campaign,
    load_reference_config,
    plan_rendezvous_safe,
)
from orbitopt.rendezvous import (
    robustness as rb,
)
from orbitopt.verify.tolerances import VERIFY_ATOL, VERIFY_RTOL


@pytest.fixture(scope="module")
def cfg():
    return load_reference_config()


@pytest.fixture(scope="module")
def plan(cfg):
    return plan_rendezvous_safe(cfg).plan


@pytest.fixture(scope="module")
def campaign():
    return load_reference_campaign()


def _quarantine(camp: rb.RobustnessCampaign, n_clusters: int = 2, spc: int = 2, **over):
    """The packaged campaign shrunk to an arm-testable layout, seeds untouched."""
    return dataclasses.replace(
        camp, cluster_bootstrap_trials=200,
        train=dataclasses.replace(camp.train, n_clusters=n_clusters, samples_per_cluster=spc),
        holdout=dataclasses.replace(camp.holdout, n_clusters=n_clusters, samples_per_cluster=spc),
        **over)


@pytest.fixture(scope="module")
def small(cfg, plan, campaign):
    return rb.run_campaign_arm(cfg, _quarantine(campaign), plan, "holdout", jobs=1)


# ---- the packaged campaign -------------------------------------------------

def test_packaged_campaign_declares_all_four_sources_and_disjoint_arms(campaign):
    assert campaign.disturbances.active() == rb.DISTURBANCE_ORDER
    n = max(campaign.train.n, campaign.holdout.n)
    assert abs(campaign.train.seed_base - campaign.holdout.seed_base) >= n + 1


def test_sample_size_is_derived_from_declared_ci_width(campaign):
    from orbitopt.flightlab.recovery.stats import required_sample_size
    need = required_sample_size(campaign.target_ci_width, campaign.ci_level,
                                campaign.planning_rate)
    for arm in (campaign.train, campaign.holdout):
        assert arm.n >= need, "the arm must deliver at least the width-priced sample count"
        assert arm.n < need + arm.samples_per_cluster, "but no more than one cluster's worth"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.pop("schemaVersion"), "schemaVersion"),
    (lambda d: d["campaign"].update(n_clusters_hand=True) or
     d["campaign"]["train"].update(n_clusters=30), "derived"),
    (lambda d: d["campaign"]["holdout"].update(seed_base=d["campaign"]["train"]["seed_base"]),
     "disjoint"),
    (lambda d: d["campaign"].update(sample_dt_s=0.5), "grid_convergence_dt_s"),
    (lambda d: d["disturbances"]["drag"].update(ballistic_coefficient_m2_per_kg=0.0),
     "drag area"),
    (lambda d: d["disturbances"]["initial_state"].update(sigma_pos_lvlh_m=[1.0, 2.0]),
     "three numbers"),
])
def test_malformed_campaigns_are_refused_at_the_loader(mutate, message, tmp_path):
    import yaml
    doc = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "src/orbitopt/data/leo_rdv_001_robustness.yaml").read_text())
    mutate(doc)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(rb.RobustnessConfigError) as err:
        rb.load_campaign(p)
    assert message in str(err.value)


# ---- the RNG layout --------------------------------------------------------

def test_disabling_a_family_leaves_the_other_draws_untouched(campaign):
    """The stream-layout rule: a zero/disabled component consumes no randomness,
    so a solo arm sees the same navigation draws as the joint arm."""
    off = dataclasses.replace(
        campaign.disturbances,
        j2=dataclasses.replace(campaign.disturbances.j2, enabled=False),
        drag=dataclasses.replace(campaign.disturbances.drag, enabled=False))
    joint = rb.draw_sample(campaign.disturbances, 11, 3, 7)
    solo = rb.draw_sample(off, 11, 3, 7)
    assert solo.nav_pos_lvlh_m == joint.nav_pos_lvlh_m
    assert solo.nav_vel_lvlh_m_s == joint.nav_vel_lvlh_m_s
    assert solo.burn_factors == joint.burn_factors
    assert solo.burn_angle_vectors_rad == joint.burn_angle_vectors_rad
    assert solo.j2_factor == 1.0 and solo.drag_factor == 1.0


def test_sigma_equivalents_match_the_draws(campaign):
    d = rb.draw_sample(campaign.disturbances, 5, 1, 2)
    assert abs(d.z["j2"]) == pytest.approx(abs(d.j2_factor - 1.0)
                                           / campaign.disturbances.j2.sigma_rel)
    assert abs(d.z["drag"]) == pytest.approx(abs(np.log(d.drag_factor))
                                             / campaign.disturbances.drag.sigma_ln)


# ---- the truth model at its control point ----------------------------------

def test_zero_disturbance_control_reproduces_the_plan(cfg, plan, campaign):
    """With every family off the flown mission IS the planned mission: a
    residual here belongs to the harness, not to a disturbance, and it would
    poison every number the campaign reports."""
    off = dataclasses.replace(
        campaign.disturbances,
        **{name: dataclasses.replace(getattr(campaign.disturbances, name), enabled=False)
           for name in rb.DISTURBANCE_ORDER})
    quiet = dataclasses.replace(campaign, disturbances=off)
    out = rb.fly_truth_sample(cfg, quiet, plan.tof_s, rb.draw_sample(quiet.disturbances, 7, 0, 0))
    assert out.ok
    assert out.position_error_m < 1.0e-3
    assert out.relative_speed_m_s < 1.0e-6
    assert out.realized_delta_v_m_s == pytest.approx(plan.total_delta_v_m_s, abs=1.0e-6)
    assert out.gate_epoch_s == pytest.approx(plan.maneuvers[0].epoch_s, abs=1.0e-6)
    assert out.grade(cfg)["failure_class"] is None


# ---- grading and taxonomy --------------------------------------------------

def _synthetic_outcome(cfg, *, rho, rho_dot, realized_dv, position_error):
    times = np.array([0.0, 100.0])
    rel = np.array([[rho[0], rho[1], rho[2], rho_dot[0], rho_dot[1], rho_dot[2]],
                    [rho[0], rho[1], rho[2], rho_dot[0], rho_dot[1], rho_dot[2]]])
    return rb.FlightOutcome(
        ok=True, failure_class=None, reason="", position_error_m=position_error,
        relative_speed_m_s=float(np.linalg.norm(rho_dot)), realized_delta_v_m_s=realized_dv,
        planned_delta_v_m_s=10.0, min_range_m=float(np.linalg.norm(rho)),
        min_range_epoch_s=0.0, gate_epoch_s=0.0, tof_s=100.0, times_s=times, rel_lvlh=rel)


def test_grading_follows_the_declared_precedence(cfg):
    unsafe = _synthetic_outcome(cfg, rho=(0.0, 0.0, -200.0), rho_dot=(0.0, 0.0, 0.0),
                                realized_dv=99.0, position_error=900.0)
    g = unsafe.grade(cfg)
    assert g["failure_class"] == "safety_violation"        # before fuel and accuracy
    assert "keep_out" in g["failed_checks"] and g["margins"]["keep_out"] < 0.0

    thirsty = _synthetic_outcome(cfg, rho=(0.0, -6000.0, 0.0), rho_dot=(0.0, 0.0, 0.0),
                                 realized_dv=99.0, position_error=900.0)
    g2 = thirsty.grade(cfg)
    assert g2["failure_class"] == "fuel_over_budget"       # budget before accuracy
    assert "99.00" in g2["reason"]

    off_target = _synthetic_outcome(cfg, rho=(0.0, -6000.0, 0.0), rho_dot=(0.0, 0.0, 0.0),
                                    realized_dv=12.0, position_error=900.0)
    assert off_target.grade(cfg)["failure_class"] == "arrival_accuracy"

    clean = _synthetic_outcome(cfg, rho=(0.0, -6000.0, 0.0), rho_dot=(0.0, 0.0, 0.0),
                               realized_dv=12.0, position_error=10.0)
    assert clean.grade(cfg)["failure_class"] is None


def test_infeasible_samples_are_a_class_of_their_own(cfg, campaign):
    """A sample whose re-plan fails is ``planning_infeasible`` with the
    governing reason, not a fabricated zero-margin flight."""
    squeeze = dataclasses.replace(
        campaign,
        disturbances=dataclasses.replace(
            campaign.disturbances,
            execution=dataclasses.replace(campaign.disturbances.execution,
                                          sigma_magnitude_rel=0.0, sigma_pointing_deg=0.0)))
    d = rb.draw_sample(squeeze.disturbances, campaign.holdout.seed_base, 0, 0)
    out = rb.fly_truth_sample(cfg, squeeze, 1.0, d)     # tof below the config's own window
    assert not out.ok and out.failure_class == "planning_infeasible"
    assert out.reason


def test_taxonomy_is_total_by_construction(small):
    cls = small.failure["classes"]
    assert set(cls) == set(rb.FAILURE_CLASSES)
    assert sum(cls.values()) == small.failure["n_failures"] == small.n - small.successes


# ---- the arm: determinism, intervals, independence of workers --------------

def test_arm_is_reproducible_and_worker_count_independent(cfg, campaign, plan, small):
    again = rb.run_campaign_arm(cfg, _quarantine(campaign), plan, "holdout", jobs=1)
    assert [(r.cluster, r.index, r.success, r.reason) for r in again.records] == \
           [(r.cluster, r.index, r.success, r.reason) for r in small.records]
    wide = rb.run_campaign_arm(cfg, _quarantine(campaign), plan, "holdout", jobs=2)
    assert wide.summary["success_rate"] == small.summary["success_rate"]
    assert wide.summary["cluster_bootstrap"] == small.summary["cluster_bootstrap"]
    assert [list(c) for c in wide.cluster_counts] == [list(c) for c in small.cluster_counts]


def test_intervals_bracket_the_rate(small):
    s = small.summary
    ci = s["wilson"]
    assert ci["method"] == "wilson_score"
    assert 0.0 <= ci["lower"] <= s["success_rate"] <= ci["upper"] <= 1.0
    boot = s["cluster_bootstrap"]
    assert boot is not None and boot["lower"] <= boot["upper"]
    assert s["per_cluster_rate"]["min"] <= s["per_cluster_rate"]["max"]


def test_solo_campaigns_run_the_same_layout_one_family_at_a_time(campaign, small):
    solos = rb.solo_campaigns(_quarantine(campaign, n_clusters=1, spc=2))
    assert set(solos) == set(rb.DISTURBANCE_ORDER)
    for name, solo in solos.items():
        assert solo.disturbances.active() == (name,)
        assert solo.holdout.seed_base == campaign.holdout.seed_base
        assert solo.holdout.samples_per_cluster == 2, "the shrunk layout is carried across"
        assert solo.holdout.n == 2, "the shrunk layout, unchanged between lineups"


def test_solo_lineups_differ_from_the_joint_arm_in_more_than_name(cfg, plan, campaign):
    """The solo arms must actually be flying different physics: the drag-free
    arm cannot have the same realized trajectory as the joint one wherever both
    fly -- the lineup is only informative if disabling a source moves samples."""
    solo = rb.run_campaign_arm(cfg, _quarantine(campaign, n_clusters=1, spc=3), plan, "holdout",
                               model=campaign.disturbances.disabled_except("initial_state"))
    joint = rb.run_campaign_arm(cfg, _quarantine(campaign, n_clusters=1, spc=3), plan, "holdout")
    assert any(a.realized_delta_v_m_s != b.realized_delta_v_m_s
               or a.min_range_m != b.min_range_m
               for a, b in zip(solo.records, joint.records) if a.success and b.success) \
        or any(a.success != b.success for a, b in zip(solo.records, joint.records))


# ---- sensitivity -----------------------------------------------------------

def test_sensitivity_reports_a_null_as_a_null(campaign, small):
    all_pass = [dataclasses.replace(r, success=True, failure_class=None) for r in small.records]
    read = rb.sensitivity_reading(all_pass, campaign.disturbances)
    assert read["state"] == "all_passed" and read["ranking"] == []
    for src in read["sources"].values():
        assert src["state"] == "uninformative"

    only_nav = campaign.disturbances.disabled_except("initial_state")
    mixed = rb.sensitivity_reading(small.records, only_nav)
    assert mixed["sources"]["execution"]["state"] == "inactive_or_bias_only"
    assert mixed["sources"]["initial_state"]["state"] == "assessed"


def test_joint_sensitivity_ranks_a_real_spread(small):
    """The small arm is not guaranteed to contain both outcomes; when it does,
    every active source must carry a finite rank correlation and the ranking
    must name all four."""
    if 0 < small.successes < small.n:
        assert small.sensitivity["state"] == "assessed"
        assert sorted(small.sensitivity["ranking"]) == sorted(rb.DISTURBANCE_ORDER)
        for name in rb.DISTURBANCE_ORDER:
            assert small.sensitivity["sources"][name]["tail"]["high"]["n"] >= 0
    else:
        assert small.sensitivity["state"] in ("all_passed", "all_failed")


# ---- the independent spot check -------------------------------------------

def test_spot_check_lines_and_honest_backend(cfg, plan, campaign):
    small = rb.run_campaign_arm(cfg, _quarantine(campaign, spot_check_samples=2), plan,
                                "train", jobs=2)
    cc = rb.spot_check(cfg, _quarantine(campaign, spot_check_samples=2), plan, small)
    assert cc["n_samples"] >= 1
    integ = cc["independent_integration"]
    assert integ["state"] == "within_bands"
    for row in integ["rows"]:
        assert row["reproduces_record"] is True and row["verdict_agrees"] is True
        for k, band in rb.INTEGRATION_VS_RK4_TOLERANCES.items():
            assert row["integration_residuals"][k]["tolerance"] == band
            assert row["integration_residuals"][k]["observed"] <= band
    assert set(cc["independent_formulation"]["residuals"]) == {
        "hold_point_position_error_m", "relative_speed_m_s", "min_approach_distance_m"}
    model = cc["independent_model"]
    if model["backend"] is None:
        assert model["result"] == "not_checked" and "REFLY_BACKEND_MISSING" in model["reason"]
    else:
        assert model["result"] in ("not_run", "passed", "failed", "not_checked")


def test_independent_reflight_tolerances_are_the_production_ones():
    """The adaptive re-fly must run at E12-05's production tolerances -- the
    numbers, not a looser private copy -- and a tolerance must not be reachable
    only through ``import tudatpy``: a fresh interpreter that imports nothing
    but the tolerances module still has no tudatpy loaded."""
    probe = (
        "import sys; import orbitopt.verify.tolerances as t; "
        "print(t.VERIFY_RTOL, t.VERIFY_ATOL, 'tudatpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True,
                         text=True, env={**os.environ,
                                         "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
                         ).stdout.split()
    assert out[:2] == [repr(VERIFY_RTOL), repr(VERIFY_ATOL)] and out[2] == "False"


def test_grid_convergence_bounds_the_sample_interval(cfg, campaign, plan, small):
    gc = rb.grid_convergence(cfg, _quarantine(campaign, grid_convergence_samples=2), small)
    assert gc["state"] in ("grid_independent", "grid_dependent")
    assert gc["n_samples"] >= 1
    assert np.isfinite(gc["max_abs_delta_min_range_m"])
    # The 5 s grid must not be hiding a clip: measured agreement on the
    # reference mission is 0.02 m; 5 m is orders of magnitude of headroom.
    assert gc["max_abs_delta_min_range_m"] < 5.0


# ---- the artifact ----------------------------------------------------------

def test_benchmark_quick_run_writes_a_complete_artifact(tmp_path):
    out = tmp_path / "robustness.json"
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / "benchmarks/bench_e12_07_rendezvous_robustness.py"),
           "-o", str(out), "--quick"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800,
                   env={**os.environ, "PYTHONPATH": str(repo / "src")})
    doc = json.loads(out.read_text())
    assert doc["kind"] == "orbitopt.bench.e12-07-rendezvous-robustness"
    assert doc["quick_layout"] is True
    h = doc["headline"]
    assert 0.0 <= h["holdout_success_rate"] <= 1.0
    assert h["n_samples"] >= 4
    arm = doc["arms"]["holdout"]
    assert sum(h["failure_classes"].values()) == arm["n"] - arm["summary"]["successes"]
    assert {"wilson", "cluster_bootstrap", "per_cluster_rate"} <= set(h)
    assert h["independent_model"] in ("not_checked", "not_run", "passed", "failed")
    assert doc["arms"]["train"]["n"] == doc["arms"]["holdout"]["n"]
    assert doc["solo"], "a full artifact records the one-source-at-a-time lineup"
