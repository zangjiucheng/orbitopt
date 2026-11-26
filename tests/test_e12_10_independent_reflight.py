"""E12-10 (task #412): independent robustness re-flight -- the machinery that
runs WITHOUT a tudatpy backend.

The campaign re-grading (passive contingencies inside success), the deterministic
sample selection, the self-identifying spec, the per-sample comparison and the
honest no-backend line are all backend-free and tested here; the flight itself is
exercised by ``benchmarks/bench_e12_10_independent_reflight.py`` wherever a
tudatpy interpreter is configured (the artifact then records ``checked``, never
a locally faked one).
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from orbitopt.rendezvous import load_reference_config
from orbitopt.rendezvous import robustness as rb
from orbitopt.rendezvous.safety import PassiveRelease, evaluate_trajectory_safety
from orbitopt.verify.rendezvous_refly import (
    SAMPLE_REFLY_PASSIVE_MARGIN_TOLERANCE_M,
    SAMPLE_REFLY_SPEC_ENV,
    SAMPLE_REFLY_VS_MAIN_TOLERANCES,
    ReflyBackendError,
    run_sample_reflight_in_current_environment,
    sample_reflight_comparison,
    sample_reflight_stage_impl,
    verify_sample_reflight,
)


@pytest.fixture(scope="module")
def cfg():
    return load_reference_config()


@pytest.fixture(scope="module")
def campaign():
    return dataclasses.replace(rb.load_reference_campaign(), passive_contingency_in_success=True)


@pytest.fixture(scope="module")
def plan(cfg):
    from orbitopt.rendezvous.canonical import canonical_plan
    return canonical_plan(cfg).plan


def _record(cfg, *, cluster, index, success, failure_class, failed=()):
    draws = rb.draw_sample(rb.load_reference_campaign().disturbances, 1234, cluster, index)
    return rb.SampleRecord(
        cluster=cluster, index=index, seed=1234 * 1000 + cluster * 100 + index,
        success=success, failure_class=failure_class, reason="synthetic",
        position_error_m=0.1 if success else 3.0,
        relative_speed_m_s=0.001 if success else 0.02,
        realized_delta_v_m_s=1.5, planned_delta_v_m_s=1.5,
        min_range_m=9000.0, min_range_epoch_s=100.0, gate_epoch_s=200.0, tof_s=1000.0,
        margins={k: 5.0 for k in rb.TRAJECTORY_CHECKS},
        failed_checks=tuple(failed), draws=draws)


# --------------------------------------------------------------------------- #
# the new grading criterion
# --------------------------------------------------------------------------- #

def test_packaged_campaign_keeps_passive_out_of_success_until_asked():
    from importlib import resources

    import yaml
    base = rb.load_reference_campaign()
    assert base.passive_contingency_in_success is False
    assert base.to_dict()["statistics"]["passive_contingency_in_success"] is False
    doc = yaml.safe_load(
        resources.files("orbitopt").joinpath("data", "leo_rdv_001_robustness.yaml")
        .read_text(encoding="utf-8"))
    assert rb.parse_campaign(doc).passive_contingency_in_success is False
    flipped = rb.parse_campaign({**doc, "campaign": {**doc["campaign"],
                                                     "passive_contingency_in_success": True}})
    assert flipped.passive_contingency_in_success is True
    assert dataclasses.replace(base, passive_contingency_in_success=True) \
        .to_dict()["statistics"]["passive_contingency_in_success"] is True


def _nominal_checks(cfg):
    t = np.array([0.0, 60.0])
    rho = np.array([np.asarray(cfg.hold_point_rel_lvlh, float)] * 2) \
        + np.array([[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]])
    return evaluate_trajectory_safety(cfg, t, rho, np.zeros((2, 3)))


def _release(cfg, passed, margin):
    return PassiveRelease(kind="gate_abort_no_burn", release_epoch_s=200.0,
                          duration_s=cfg.safety.passive_contingency_s,
                          min_range_m=cfg.safety.keep_out_radius_m + margin,
                          worst_epoch_s=10.0, worst_rel_lvlh_m=(0.0, 0.0, 0.0),
                          keep_out_margin_m=margin, passed=passed)


def test_failing_passive_release_folds_into_safety_violation(cfg):
    checks = _nominal_checks(cfg)
    kw = dict(realized_delta_v_m_s=1.5, planned_delta_v_m_s=1.5,
              position_error_m=0.1, relative_speed_m_s=0.001)
    assert rb.grade_trajectory_checks(cfg, checks, **kw)["failure_class"] is None
    with_passing = rb.grade_trajectory_checks(cfg, checks, passive=(_release(cfg, True, 100.0),),
                                              **kw)
    assert with_passing["failure_class"] is None
    bad = _release(cfg, False, -10.0)
    v = rb.grade_trajectory_checks(cfg, checks, passive=(bad,), **kw)
    assert v["failure_class"] == "safety_violation"
    assert "passive_gate_abort_no_burn" in v["failed_checks"]
    # safety outranks fuel: a passive keep-out breach stays a safety violation even
    # when the same sample also blew the delta-v budget (FAILURE_PRECEDENCE)
    v2 = rb.grade_trajectory_checks(cfg, checks, realized_delta_v_m_s=9.9,
                                    planned_delta_v_m_s=1.5, position_error_m=0.1,
                                    relative_speed_m_s=0.001, passive=(bad,))
    assert v2["failure_class"] == "safety_violation"


# --------------------------------------------------------------------------- #
# selection and the self-identifying spec
# --------------------------------------------------------------------------- #

def _synthetic_arm():
    return [_record(cfg=None, cluster=c, index=i, success=(c == 0 and i == 0),
                    failure_class=None if (c == 0 and i == 0) else
                    ("safety_violation" if i == 1 else "arrival_accuracy"),
                    failed=(["passive_hold_point_coast"] if (c == 0 and i == 1) else []))
            for c in range(3) for i in range(4)] + \
           [_record(cfg=None, cluster=9, index=0, success=False,
                    failure_class="planning_infeasible")]


def test_selection_covers_every_class_deterministically(cfg):
    recs = _synthetic_arm()
    sel = rb.select_reflight_samples(cfg, recs)
    reasons = {why for _, why in sel}
    keys = [(r.cluster, r.index) for r, _ in sel]
    assert {"success", "passive_contingency_failure", "boundary_arrival_accuracy",
            "planning_infeasible"} <= reasons
    assert sorted(keys) == [(0, 0), (0, 1), (0, 2), (9, 0)]  # the safety boundary pick IS the
    # passive pick (0,1) -- deduplicated, so it carries only its first reason
    twice = rb.select_reflight_samples(cfg, recs)
    assert [(r.cluster, r.index, w) for r, w in twice] == \
        [(r.cluster, r.index, w) for r, w in sel]


def test_selection_respects_the_total_cap(cfg):
    recs = [_record(cfg=None, cluster=c, index=i, success=True, failure_class=None)
            for c in range(10) for i in range(10)]
    sel = rb.select_reflight_samples(cfg, recs)
    assert len(sel) <= rb.SAMPLE_REFLY_MAX_TOTAL
    assert sum(1 for _, w in sel if w == "success") <= rb.SAMPLE_REFLY_MAX_SUCCESSES


def test_spec_carries_the_sample_own_burns_and_digest(cfg, campaign, plan):
    recs = [_record(cfg=None, cluster=0, index=0, success=True, failure_class=None),
            _record(cfg=None, cluster=1, index=1, success=False,
                    failure_class="arrival_accuracy"),
            _record(cfg=None, cluster=9, index=0, success=False,
                    failure_class="planning_infeasible")]
    spec = rb.build_reflight_spec(cfg, campaign, recs, plan)
    assert spec["kind"] == rb.SAMPLE_REFLY_SPEC_KIND
    assert spec["campaign"]["statistics"]["passive_contingency_in_success"] is True
    assert len(spec["flights"]) == 2 and len(spec["skipped"]) == 1
    assert rb.reflight_spec_digest(spec) == spec["spec_digest"]
    fl = spec["flights"][0]
    assert "dv1_exec_eci_m_s" in fl and "dv2_exec_eci_m_s" in fl
    assert fl["j2_truth"] != 0.0 and np.isfinite(np.asarray(fl["dv1_exec_eci_m_s"])).all()
    tampered = {**spec, "flights": [dict(spec["flights"][0], t1_s=float("nan"))]}
    assert rb.reflight_spec_digest(tampered) != spec["spec_digest"]


def test_worker_refuses_a_tampered_spec_before_needing_a_backend(cfg, campaign, plan):
    spec = rb.build_reflight_spec(cfg, campaign,
                                  [_record(cfg=None, cluster=0, index=0, success=True,
                                           failure_class=None)], plan)
    with pytest.raises(ReflyBackendError, match="spec_digest"):
        run_sample_reflight_in_current_environment(cfg, {**spec, "spec_digest": "0" * 64})
    with pytest.raises(ReflyBackendError, match="sample-reflight spec"):
        run_sample_reflight_in_current_environment(cfg, dict(spec, kind="something.else"))


def test_stage_impl_demands_its_spec(tmp_path, cfg):
    os.environ.pop(SAMPLE_REFLY_SPEC_ENV, None)
    with pytest.raises(ReflyBackendError, match=SAMPLE_REFLY_SPEC_ENV):
        sample_reflight_stage_impl(cfg, None, tmp_path)


# --------------------------------------------------------------------------- #
# comparison and the honest no-backend line
# --------------------------------------------------------------------------- #

def _fake_evidence(spec, *, flip=False, position_shift=0.0, passive_shift=None):
    flights = []
    for fl in spec["flights"]:
        cls = (("arrival_accuracy" if fl["main_failure_class"] is None else None) if flip
               else fl["main_failure_class"])
        flights.append({
            "cluster": fl["cluster"], "index": fl["index"], "seed": fl["seed"],
            "selection": fl["selection"], "t1_s": fl["t1_s"], "tof_s": fl["tof_s"],
            "position_error_m": fl["main_position_error_m"] + position_shift,
            "relative_speed_m_s": fl["main_relative_speed_m_s"],
            "realized_delta_v_m_s": fl["main_realized_delta_v_m_s"],
            "min_approach_distance_m": fl["main_min_range_m"],
            "min_approach_epoch_s": 0.0,
            "arrival_rel_lvlh_m": [0.0] * 3, "arrival_rel_lvlh_rate_m_s": [0.0] * 3,
            "checks": {k: {"unit": "", "min_margin": 0.0, "passed": True}
                       for k in rb.TRAJECTORY_CHECKS},
            "passive": ([] if passive_shift is None else
                        [{"kind": "hold_point_coast", "keep_out_margin_m": 100.0 + passive_shift,
                          "passed": True}]),
            "verdict": {"failure_class": cls, "failed_checks": [], "reason": "fake",
                        "margins": {}},
        })
    return {"kind": rb.SAMPLE_REFLY_SPEC_KIND.replace("-spec", ""), "flights": flights,
            "skipped": []}


def test_bands_sit_far_below_the_contract_scales(cfg):
    """A band is an implementation-difference bound: it must be far tighter
    than anything the contract grades, or a model disagreement could hide
    inside it while still moving a verdict."""
    tol = SAMPLE_REFLY_VS_MAIN_TOLERANCES
    assert tol["position_error_m"] <= 1e-3 * cfg.max_position_error_m
    assert tol["relative_speed_m_s"] <= 1e-3 * cfg.max_relative_speed_m_s
    assert tol["min_approach_distance_m"] <= 1e-3 * cfg.safety.keep_out_radius_m
    assert SAMPLE_REFLY_PASSIVE_MARGIN_TOLERANCE_M <= 1e-3 * cfg.safety.keep_out_radius_m


def test_comparison_flags_flips_and_band_violations(cfg, campaign, plan):
    spec = rb.build_reflight_spec(cfg, campaign,
                                  [_record(cfg=None, cluster=0, index=0, success=True,
                                           failure_class=None)], plan)
    clean = sample_reflight_comparison(spec, _fake_evidence(spec))
    assert clean["n_reflied"] == 1 and clean["n_verdict_flips"] == 0
    assert clean["n_band_violations"] == 0 and clean["samples"][0]["all_within_bands"]
    assert clean["samples"][0]["residuals"]["min_approach_distance_m"]["within_band"] is True
    over = SAMPLE_REFLY_VS_MAIN_TOLERANCES["position_error_m"] * 2
    shifted = sample_reflight_comparison(spec, _fake_evidence(spec, position_shift=over))
    assert shifted["n_band_violations"] == 1
    assert shifted["samples"][0]["residuals"]["position_error_m"]["within_band"] is False
    flipped = sample_reflight_comparison(spec, _fake_evidence(spec, flip=True))
    assert flipped["n_verdict_flips"] == 1
    assert flipped["verdict_flips"][0]["main_model"] is None
    spec_p = dict(spec, flights=[dict(spec["flights"][0], main_passive=[
        {"kind": "hold_point_coast", "keep_out_margin_m": 100.0, "passed": True}])])
    moved = sample_reflight_comparison(
        spec_p, _fake_evidence(spec_p, passive_shift=2 * SAMPLE_REFLY_PASSIVE_MARGIN_TOLERANCE_M))
    assert moved["n_band_violations"] == 1 and moved["n_verdict_flips"] == 0
    assert moved["samples"][0]["passive_comparison"][0]["within_band"] is False
    with pytest.raises(ReflyBackendError, match="no main-model passive"):
        sample_reflight_comparison(spec, _fake_evidence(spec, passive_shift=0.0))
    missing = dict(_fake_evidence(spec), flights=[])
    with pytest.raises(ReflyBackendError, match="no re-flight"):
        sample_reflight_comparison(spec, missing)


def test_verify_reports_not_checked_without_a_backend(cfg, campaign, plan, tmp_path, monkeypatch):
    monkeypatch.delenv("ORBITOPT_TUDAT_PYTHON", raising=False)
    import importlib.util
    if importlib.util.find_spec("tudatpy") is not None:
        pytest.skip("this environment has a tudatpy backend; the honest line is 'checked'")
    spec = rb.build_reflight_spec(cfg, campaign,
                                  [_record(cfg=None, cluster=0, index=0, success=True,
                                           failure_class=None)], plan)
    res = verify_sample_reflight(cfg, spec, config_path=None, out_dir=tmp_path)
    assert res["status"] == "not_checked" and res["evidence"] is None
    assert "REFLY_BACKEND_MISSING" in res["reason"]


def test_selection_puts_passive_only_failures_first(cfg):
    mixed = _record(cfg, cluster=0, index=0, success=False, failure_class="safety_violation",
                    failed=("keep_out", "passive_hold_point_coast"))
    only = _record(cfg, cluster=5, index=0, success=False, failure_class="safety_violation",
                   failed=("passive_hold_point_coast",))
    sel = rb.select_reflight_samples(cfg, [mixed, only], max_passive=1)
    assert [(r.cluster, w) for r, w in sel][0] == (5, "passive_contingency_failure")


# --------------------------------------------------------------------------- #
# the flight itself -- only where tudatpy is importable next to orbitopt
# --------------------------------------------------------------------------- #

def test_tudat_reflight_of_a_drag_dispersed_sample_matches_the_main_model(cfg, campaign, plan):
    """Same per-sample physics in both models, so the residuals are the
    implementation's. Guards the two ways the comparison once measured
    something else: fixed-step RK4 legs overshooting the gate/arrival epoch
    (burn 1 applied 4.4 s late: ~150 m of false min-range and hold-point
    disagreement) and tudat's mean-radius Earth shifting the drag altitude by
    7.1 km (~11 m of false arrival disagreement)."""
    pytest.importorskip("tudatpy")
    draws = rb.draw_sample(campaign.disturbances, campaign.holdout.seed_base, 6, 7)
    out = rb.fly_truth_sample(cfg, campaign, float(plan.tof_s), draws)
    g = out.grade(cfg)
    rec = rb.SampleRecord(
        cluster=6, index=7, seed=draws.seed, success=g["failure_class"] is None,
        failure_class=g["failure_class"], reason=g["reason"],
        position_error_m=out.position_error_m, relative_speed_m_s=out.relative_speed_m_s,
        realized_delta_v_m_s=out.realized_delta_v_m_s,
        planned_delta_v_m_s=out.planned_delta_v_m_s, min_range_m=out.min_range_m,
        min_range_epoch_s=out.min_range_epoch_s, gate_epoch_s=out.gate_epoch_s,
        tof_s=out.tof_s, margins=dict(g["margins"]), failed_checks=tuple(g["failed_checks"]),
        draws=draws)
    spec = rb.build_reflight_spec(cfg, campaign, [rec], plan, selection=[(rec, "success")])
    assert spec["flights"][0]["drag"] is not None
    row = run_sample_reflight_in_current_environment(cfg, spec)["flights"][0]
    assert row["verdict"]["failure_class"] == rec.failure_class
    assert abs(row["position_error_m"] - rec.position_error_m) < 1.0
    assert abs(row["min_approach_distance_m"] - rec.min_range_m) < 1.0
    main_p = {p.kind: p.keep_out_margin_m for p in out.passive}
    for p in row["passive"]:
        assert abs(p["keep_out_margin_m"] - main_p[p["kind"]]) < 1.0, p["kind"]
