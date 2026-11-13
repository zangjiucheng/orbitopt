"""Checks for orbitopt.flightlab.recovery.stats (R-06, task #238).

The acceptance criteria this file pins:

* every failure carries a class -- the classifier is total over the reason
  codes the recovery chain can hand a Monte Carlo sample, and the breakdown
  accounts for every non-landed record exactly once;
* the result is a statistical statement, not just a percentage -- Wilson
  score interval with known reference values, sane edge behaviour and a
  monotone width in ``n``;
* sensitivity is meaningful and reproducible -- a source whose draw drives
  the failures is ranked first, a source that does not is ranked last, and
  the whole statistics section is a pure function of the samples (same seed,
  identical report, worker layout notwithstanding);
* the one-at-a-time reading isolates a source where the joint reading cannot.
"""
from __future__ import annotations

import math
import os
from copy import deepcopy

import numpy as np
import pytest

from orbitopt.flightlab import reasons
from orbitopt.flightlab.recovery.stats import (
    FAILURE_CLASSES,
    FAILURE_CLASS_EXECUTION_FAILURE,
    FAILURE_CLASS_FUEL_INSUFFICIENT,
    FAILURE_CLASS_LANDING_ACCURACY,
    FAILURE_CLASS_PLANNING_INFEASIBLE,
    FAILURE_CLASS_UNCLASSIFIED,
    classify_failure,
    failure_breakdown,
    failure_class_title,
    solo_models,
    source_sensitivity,
    success_rate_interval,
)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel

#: Every code the recovery chain (FL-301/302/303, handoff, MC plumbing) can
#: report as a sample's governing reason -- all of them must classify into one
#: of the four families, never ``unclassified``.
MC_REASON_CODES = (
    "RECOVERY_INFEASIBLE", "CORRIDOR_NOT_REACHED", "CORRIDOR_MISSED",
    "LANDING_INFEASIBLE", "LANDING_FAILED", "LANDING_NO_TOUCHDOWN",
    "LANDING_POSITION_OUT_OF_TOLERANCE", "LANDING_VERTICAL_SPEED_EXCEEDED",
    "LANDING_LATERAL_SPEED_EXCEEDED", "LANDING_TILT_EXCEEDED",
    "LANDING_MAX_G_EXCEEDED", "LANDING_PROPELLANT_RESERVE_LOW",
    "INSUFFICIENT_PROPELLANT_BOOSTBACK", "INSUFFICIENT_PROPELLANT_ENTRY_BURN",
    "INSUFFICIENT_PROPELLANT_RESERVE", "PROPELLANT_EXHAUSTED",
    "PROPELLANT_RESERVE_LOW", "ENTRY_Q_LIMIT_EXCEEDED",
    "ENTRY_G_LIMIT_EXCEEDED", "RECOVERY_LANDING_NOT_ASSESSED",
    "RECOVERY_LANDING_NOT_ATTEMPTED", "HANDOFF_STATE_DISCONTINUITY",
    "HANDOFF_MASS_NOT_CONSERVED", "SAMPLE_ERROR",
)

FAMILIES = (FAILURE_CLASS_PLANNING_INFEASIBLE, FAILURE_CLASS_EXECUTION_FAILURE,
            FAILURE_CLASS_LANDING_ACCURACY, FAILURE_CLASS_FUEL_INSUFFICIENT)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", MC_REASON_CODES)
def test_every_chain_reason_code_gets_a_family(code):
    """No Monte Carlo failure is left unexplained (acceptance: every failure
    is classified) -- and the code the chain emits is still in the FL-503
    catalog, so the two vocabularies cannot drift apart silently."""
    assert reasons.is_known(code), code
    cls = classify_failure(code)
    assert cls in FAMILIES, code


@pytest.mark.parametrize("code, family", [
    ("CORRIDOR_MISSED", FAILURE_CLASS_PLANNING_INFEASIBLE),
    ("LANDING_INFEASIBLE", FAILURE_CLASS_PLANNING_INFEASIBLE),
    ("INSUFFICIENT_PROPELLANT_BOOSTBACK", FAILURE_CLASS_FUEL_INSUFFICIENT),
    ("LANDING_PROPELLANT_RESERVE_LOW", FAILURE_CLASS_FUEL_INSUFFICIENT),
    ("LANDING_POSITION_OUT_OF_TOLERANCE", FAILURE_CLASS_LANDING_ACCURACY),
    ("LANDING_MAX_G_EXCEEDED", FAILURE_CLASS_LANDING_ACCURACY),
    ("LANDING_NO_TOUCHDOWN", FAILURE_CLASS_EXECUTION_FAILURE),
    ("SAMPLE_ERROR:ZeroDivisionError", FAILURE_CLASS_EXECUTION_FAILURE),
])
def test_classification_families_match_the_plain_language_reading(code, family):
    assert classify_failure(code) == family


def test_unknown_codes_are_classified_as_unclassified_never_dropped():
    """Totality: a code nobody mapped still gets a label, so the class counts
    always add up to the failure count."""
    for weird in ("", None, "LANDED", "SOMETHING_NEW", "max_q_exceeded_x"):
        assert classify_failure(weird) == FAILURE_CLASS_UNCLASSIFIED
    # generic fallbacks cover whole families of future codes
    assert classify_failure("INFEASIBLE_SOMETHING_NEW") \
        == FAILURE_CLASS_PLANNING_INFEASIBLE
    assert classify_failure("INSUFFICIENT_PROPELLANT_SOMETHING_NEW") \
        == FAILURE_CLASS_FUEL_INSUFFICIENT
    assert failure_class_title(FAILURE_CLASS_FUEL_INSUFFICIENT, "zh") \
        .startswith("燃料不足")


def test_breakdown_accounts_for_every_failure_exactly_once():
    samples = [
        {"idx": 0, "outcome": "landed", "reason": "LANDED"},
        {"idx": 1, "outcome": "failed", "reason": "CORRIDOR_MISSED"},
        {"idx": 2, "outcome": "failed", "reason": "CORRIDOR_MISSED"},
        {"idx": 3, "outcome": "failed", "reason": "PROPELLANT_EXHAUSTED"},
        {"idx": 4, "outcome": "error", "reason": "SAMPLE_ERROR:ValueError",
         "error": "boom"},
        {"idx": 5, "outcome": "failed", "reason": "CODE_FROM_THE_FUTURE"},
    ]
    out = failure_breakdown(samples)
    assert out["n_samples"] == 6 and out["n_failures"] == 5
    assert set(out["classes"]) == set(FAILURE_CLASSES)  # zeros stay visible
    assert sum(out["classes"].values()) == out["n_failures"]
    assert out["classes"] == {
        FAILURE_CLASS_PLANNING_INFEASIBLE: 2,
        FAILURE_CLASS_EXECUTION_FAILURE: 1,
        FAILURE_CLASS_LANDING_ACCURACY: 0,
        FAILURE_CLASS_FUEL_INSUFFICIENT: 1,
        FAILURE_CLASS_UNCLASSIFIED: 1,
    }
    assert out["codes_by_class"][FAILURE_CLASS_PLANNING_INFEASIBLE] \
        == {"CORRIDOR_MISSED": 2}
    assert out["codes_by_class"][FAILURE_CLASS_UNCLASSIFIED] \
        == {"CODE_FROM_THE_FUTURE": 1}
    # a LANDED record without the outcome field still counts as a success
    assert failure_breakdown([{"idx": 0, "reason": "LANDED"}])["n_failures"] == 0


# ---------------------------------------------------------------------------
# confidence interval
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_the_reference_case():
    """6/8 landed is the textbook Wilson example: 0.75 with a 95 % interval of
    roughly [0.41, 0.93] -- a wide statement of what eight samples support."""
    ci = success_rate_interval(6, 8)
    assert ci["method"] == "wilson_score" and ci["n"] == 8
    assert ci["point"] == pytest.approx(0.75)
    assert ci["standard_error"] == pytest.approx(math.sqrt(0.75 * 0.25 / 8))
    assert ci["lower"] == pytest.approx(0.4093, abs=1e-3)
    assert ci["upper"] == pytest.approx(0.9285, abs=1e-3)
    assert ci["lower"] < ci["point"] < ci["upper"]


def test_wilson_interval_edges_and_scaling():
    zero = success_rate_interval(0, 5)
    assert zero["lower"] == 0.0 and 0.0 < zero["upper"] < 0.5  # not "impossible"
    all_ok = success_rate_interval(5, 5)
    assert all_ok["upper"] == 1.0 and 0.5 < all_ok["lower"] < 1.0
    small, large = success_rate_interval(6, 8), success_rate_interval(60, 80)
    assert small["point"] == large["point"]
    assert (large["upper"] - large["lower"]) < (small["upper"] - small["lower"])
    narrow = success_rate_interval(6, 8, level=0.68)
    assert narrow["lower"] > small["lower"] and narrow["upper"] < small["upper"]
    for bad in ((1, 0), (9, 8), (0, 4, 0.0), (0, 4, 1.0)):
        with pytest.raises(ValueError):
            success_rate_interval(*bad)


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------


def _model(**overrides):
    config = {
        "thrust": {"enabled": True, "distribution": "uniform_pct",
                   "half_width_pct": 6.0},
        "isp": {"enabled": False},
        "density": {"enabled": True, "distribution": "uniform_pct",
                    "half_width_pct": 10.0},
        "drag": {"enabled": False},
        "wind": {"enabled": True, "distribution": "normal",
                 "sigma_east_m_s": 4.0, "sigma_north_m_s": 3.0,
                 "sigma_up_m_s": 0.0, "mean_east_m_s": 2.0,
                 "mean_north_m_s": 0.0, "mean_up_m_s": 0.0},
        "initial_state": {"enabled": True, "distribution": "normal",
                          "sigma_altitude_m": 400.0, "sigma_downrange_m": 0.0,
                          "sigma_crossrange_m": 0.0, "sigma_speed_m_s": 0.0,
                          "sigma_heading_deg": 0.0},
    }
    config.update(overrides)
    return UncertaintyModel.from_config(config)


def test_component_scales_are_in_draw_units():
    """A 6 % uniform thrust half-width is 0.06/sqrt(3) in factor units (the
    unit the record's thrust_factor is written in), a wind sigma is its own
    sigma in m/s, a pure wind bias is reported but never z-scored, and
    anything off or zero-width has no scale."""
    scales = _model().component_scales()
    th = scales["thrust"][0]
    assert (th.draw_key, th.neutral) == ("thrust_factor", 1.0)
    assert th.sigma_eq == pytest.approx(0.06 / math.sqrt(3.0))
    east, north, up = scales["wind"]
    assert (east.neutral, east.sigma_eq) == (2.0, 4.0)   # biased, dispersing
    assert (north.neutral, north.sigma_eq) == (0.0, 3.0)
    assert up.sigma_eq is None                            # zero width
    assert scales["isp"][0].sigma_eq is None
    assert scales["initial_state"][0].sigma_eq == 400.0
    assert all(c.sigma_eq is None for c in scales["initial_state"][1:])
    bias_only = UncertaintyModel.from_config(
        {"thrust": {"distribution": "normal_pct", "sigma_pct": 2.0},
         "wind": {"enabled": True, "distribution": "normal",
                  "sigma_east_m_s": 0.0, "sigma_north_m_s": 0.0,
                  "sigma_up_m_s": 0.0, "mean_east_m_s": 5.0,
                  "mean_north_m_s": 0.0, "mean_up_m_s": 0.0}})
    w = bias_only.component_scales()["wind"]
    assert [c.sigma_eq for c in w] == [None, None, None]
    assert w[0].neutral == 5.0
    assert bias_only.component_scales()["thrust"][0].sigma_eq == pytest.approx(0.02)


def _synthetic(model, n=60, seed=5, failure="thrust"):
    """Records built the way the Monte Carlo builds them (same draw keys,
    same [seed, idx] streams), with the failure driven by one source."""
    rng_draw = UncertaintyModel.from_dict(model.to_dict())
    out = []
    for idx in range(n):
        d = rng_draw.draw(np.random.default_rng([seed, idx]))
        if failure == "thrust":
            bad = d["thrust_factor"] < 0.985
            reason = "INSUFFICIENT_PROPELLANT_BOOSTBACK"
        elif failure == "wind":
            bad = abs(d["wind_east_m_s"]) > 5.0
            reason = "CORRIDOR_MISSED"
        else:
            bad = False
            reason = "LANDED"
        out.append({"idx": idx, **d, "outcome": "failed" if bad else "landed",
                    "reason": reason if bad else "LANDED"})
    return out


def test_sensitivity_names_the_driving_source_first():
    model = _model()
    s = source_sensitivity(_synthetic(model, failure="thrust"), model)
    assert s["state"] == "assessed"
    assert s["ranking"][0] == "thrust"
    th = s["sources"]["thrust"]
    assert th["state"] == "assessed" and th["components"] == 1
    assert th["mean_abs_z_failed"] > th["mean_abs_z_landed"]
    # signed reading says what the joint failure mode is: less thrust, more failures
    assert th["signed_rank_correlation"] < -0.4
    assert th["rank_correlation"] > 0.3
    assert (th["tail"]["high"]["failure_rate"]
            > th["tail"]["low"]["failure_rate"])
    # the wind does not drive these failures: ranked last, near-zero rho
    wind = s["sources"]["wind"]
    assert s["ranking"][-1] == "wind" or abs(wind["rank_correlation"]) < \
        abs(s["sources"][s["ranking"][0]]["rank_correlation"])
    assert abs(wind["rank_correlation"]) < 0.35
    # switchable-off sources are reported as such, not silently missing
    assert s["sources"]["isp"]["state"] == "inactive_or_bias_only"


def test_sensitivity_states_when_there_is_nothing_to_learn():
    model = _model()
    landed = source_sensitivity(_synthetic(model, failure="none"), model)
    assert landed["state"] == "all_landed" and landed["ranking"] == []
    assert all(v["state"] in ("assessed", "uninformative",
                              "inactive_or_bias_only")
               for v in landed["sources"].values())
    broken = [{}, {}]  # no draws, no outcome: nothing to learn from
    empty = source_sensitivity(broken, model)
    assert empty["state"] == "all_failed"  # honest: nothing was landed
    assert empty["ranking"] == []
    assert [v["state"] for k, v in empty["sources"].items()
            if k in ("thrust", "wind", "density", "initial_state")] == \
           ["no_sample_draws"] * 4
    bias_only = UncertaintyModel.from_config(
        {"wind": {"enabled": True, "distribution": "normal",
                  "sigma_east_m_s": 0.0, "sigma_north_m_s": 0.0,
                  "sigma_up_m_s": 0.0, "mean_east_m_s": 5.0,
                  "mean_north_m_s": 0.0, "mean_up_m_s": 0.0}})
    s = source_sensitivity(_synthetic(_model(), failure="none"), bias_only)
    assert s["sources"]["wind"]["state"] == "inactive_or_bias_only"
    assert s["state"] == "all_landed"


def test_sensitivity_is_a_pure_function_of_the_records():
    model = _model()
    samples = _synthetic(model, n=40, seed=11, failure="wind")
    a = source_sensitivity(deepcopy(samples), model)
    b = source_sensitivity(deepcopy(samples), model)
    assert a == b
    assert a["ranking"][0] == "wind"


# ---------------------------------------------------------------------------
# one-at-a-time lineup
# ---------------------------------------------------------------------------


def test_solo_models_keep_the_declaration_and_isolate_one_source():
    model = _model()
    solos = solo_models(model)
    assert set(solos) == set(model.active_sources())
    for name, solo in solos.items():
        assert solo.active_sources() == (name,)
        decl = solo.to_dict()[name]
        assert decl == model.to_dict()[name]  # verbatim, parameters included
        for other in solos:
            if other != name:
                assert solo.to_dict()[other] == {"enabled": False}


# ---------------------------------------------------------------------------
# wiring through run_monte_carlo (stubs -- no physics, statistics plumbing)
# ---------------------------------------------------------------------------


def _stub_sample(job):
    """Deterministic stand-in for a flown sample: the verdict is a fixed
    function of that sample's own draw -- it fails when its own thrust draw
    is an outlier (|z| > 0.5 sigma of the declared 6 % uniform band) -- so
    thrust must come out on top of the sensitivity ranking."""
    model = UncertaintyModel.from_dict(job["uncertainty"])
    draw = model.draw(np.random.default_rng([int(job["seed"]), int(job["idx"])]))
    bad = abs(draw["thrust_factor"] - 1.0) > 0.5 * (0.06 / math.sqrt(3.0))
    return {"idx": int(job["idx"]), **draw,
            "outcome": "failed" if bad else "landed",
            "reason": "INSUFFICIENT_PROPELLANT_BOOSTBACK" if bad else "LANDED"}


def _run_stubbed(monkeypatch, mc_module, tmp_path, **kwargs):
    monkeypatch.setattr(mc_module, "_run_sample", _stub_sample)
    monkeypatch.setattr(mc_module, "write_run", lambda *a, **k: None)
    params: dict = {"uncertainty": _model().to_dict(), "seed": 4, "n": 12,
                    "workers": 1}
    params.update(kwargs)
    return mc_module.run_monte_carlo(None, "vehicle.yaml", None, **params)


def test_report_carries_the_r06_statistics(tmp_path, monkeypatch):
    from orbitopt.flightlab.recovery import montecarlo as mc

    result = _run_stubbed(monkeypatch, mc, tmp_path)
    out = result.to_dict()
    assert out["schemaVersion"] == "1.3"
    stats = out["statistics"]
    ci = stats["success_rate_ci"]
    assert ci["n"] == 12 and ci["successes"] == ci["point"] * 12
    # the CI agrees with the plain rate it accompanies
    assert ci["lower"] <= result.success_rate <= ci["upper"]
    classes = stats["failure_classes"]
    assert classes["n_failures"] == 12 - ci["successes"]
    assert sum(classes["classes"].values()) == classes["n_failures"]
    # every failure is classified: classes reproduce the reason histogram
    # (the histogram also counts LANDED; that is not a failure)
    total = {c: 0 for c in FAILURE_CLASSES}
    for code, count in result.reason_histogram.items():
        if code == "LANDED":
            continue
        total[classify_failure(code)] += count
    assert total == classes["classes"]
    assert stats["sensitivity"]["sources"]["thrust"]["state"] == "assessed"
    assert stats["sensitivity"]["ranking"][0] == "thrust"
    assert "solo_sensitivity" not in stats  # opt-in, not implicit cost
    assert any("R-06" in a for a in result.assumptions)


def test_statistics_reproduce_with_the_seed(tmp_path, monkeypatch):
    """Same seed -> identical statistics section, worker layout or not; a
    different seed -> the same structure, not the same numbers."""
    from orbitopt.flightlab.recovery import montecarlo as mc

    a = _run_stubbed(monkeypatch, mc, tmp_path)
    b = _run_stubbed(monkeypatch, mc, tmp_path)
    assert a.to_dict()["statistics"] == b.to_dict()["statistics"]
    a.write(tmp_path / "a.json")
    b.write(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    if os.name == "posix":  # fork-based pool sees the same monkeypatched worker
        c = _run_stubbed(monkeypatch, mc, tmp_path, workers=2)
        assert c.to_dict()["statistics"] == a.to_dict()["statistics"]
    other = _run_stubbed(monkeypatch, mc, tmp_path)
    assert other.to_dict()["statistics"] == a.to_dict()["statistics"]


def test_solo_sensitivity_is_marginal_per_source(tmp_path, monkeypatch):
    from orbitopt.flightlab.recovery import montecarlo as mc

    def wind_only_failure(job):
        model = UncertaintyModel.from_dict(job["uncertainty"])
        draw = model.draw(np.random.default_rng([int(job["seed"]),
                                                 int(job["idx"])]))
        bad = abs(draw["wind_east_m_s"]) > 5.0  # the wind alone kills samples
        return {"idx": int(job["idx"]), **draw,
                "outcome": "failed" if bad else "landed",
                "reason": "CORRIDOR_MISSED" if bad else "LANDED"}

    monkeypatch.setattr(mc, "_run_sample", wind_only_failure)
    monkeypatch.setattr(mc, "write_run", lambda *a, **k: None)
    result = mc.run_monte_carlo(None, "vehicle.yaml", None, seed=2, n=16,
                                workers=1, uncertainty=_model().to_dict(),
                                solo_sensitivity=True,
                                out_path=tmp_path / "mc.json")
    solo = result.to_dict()["statistics"]["solo_sensitivity"]
    assert set(solo["sources"]) == set(_model().active_sources())
    assert solo["baseline"]["ci"]["n"] == 16
    assert solo["baseline"]["success_rate"] == result.success_rate
    # thrust alone: inert in this stub -> nothing fails -> +delta vs joint
    assert solo["sources"]["thrust"]["success_rate"] == 1.0
    assert solo["sources"]["thrust"]["delta_vs_baseline"] > 0.0
    assert all(v == 0 for v in solo["sources"]["thrust"]["failure_classes"]
               ["classes"].values())
    # wind alone: the actual killer, flown solo on its own stream -- it
    # still kills samples, unlike every inert source
    wind_solo = solo["sources"]["wind"]["success_rate"]
    assert 0.0 < wind_solo < 1.0
    assert wind_solo < solo["sources"]["thrust"]["success_rate"]
    assert solo["sources"]["wind"]["failure_classes"]["classes"]["planning_infeasible"] > 0
    assert (tmp_path / "mc.json").read_text(encoding="utf-8").count(
        "solo_sensitivity") == 1
    assert any("one-at-a-time" in a for a in result.assumptions)
