"""V-03 integrator step/tolerance convergence harness (task #242).

Fast suite: the gate plumbing (curve -> order -> bound -> recommendation ->
residual) runs on a **mini LEO scenario** (600 s arc, coarse grids) plus
unit checks of the estimators and budget parsing -- no entry tables, no
ascent. The real key scenarios (full sweeps over all four) run once as a
``slow`` test, mirroring how CI treats every other verify harness.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from orbitopt.verify.integrator_convergence import (
    AXES,
    BUDGET_METRICS,
    REFERENCE_FACTOR,
    SCENARIOS,
    TOL_FACTORS,
    Budgets,
    ConvergenceScenario,
    error_bound,
    main,
    observed_order,
    parse_budgets,
    rk4,
    run_convergence_study,
    run_scenario,
)
from orbitopt.verify.report import CHECK_NUMERICAL, RESULT_FAILED, RESULT_PASSED, validate_report

MU = 3.986004418e14


def _mini_leo(budget=None, rk4_grid=(32.0, 16.0, 8.0, 4.0), **kw):
    """A 2700 s J2 coast (half an orbit) -- the LEO key scenario shrunk so a
    full two-axis sweep runs in a couple of seconds. Long enough that the
    along-track tolerance error is real signal, not solver noise (a 600 s arc
    converges below tolerance sensitivity and measures nothing)."""
    from orbitopt.verify.integrator_convergence import _accel_gravity, _adaptive_ivp

    r0 = 6_778_137.0
    state0 = np.array([r0, 0.0, 0.0, 0.0, math.sqrt(MU / r0), 0.0])
    t_eval = np.linspace(0.0, 2700.0, 271)
    spec = ConvergenceScenario(
        name=kw.pop("name", "mini_leo"),
        description="miniature 2700 s J2 coast (fast test fixture)",
        audits="test fixture -- not a production setting",
        production={"rtol": 1e-11, "atol": 1e-6, "max_step_s": 300.0},
        budget=budget or {"position_m": 50.0, "velocity_m_s": 0.05},
        min_order={"tolerance": 0.3, "fixed_step": 3.0},
        adaptive=_adaptive_ivp(lambda t, s: _accel_gravity(t, s, True), state0, t_eval),
        rhs=lambda t, s: _accel_gravity(t, s, True),
        rk4_grid=rk4_grid, duration_s=2700.0,
        state0=tuple(float(x) for x in state0))
    return spec


# ---------------------------------------------------------------------------
# estimators (pure math, instant)
# ---------------------------------------------------------------------------

def test_observed_order_recovers_power_law():
    errs = [100.0, 25.0, 6.25, 1.5625, 0.390625]          # err ~ h^2
    ratios = [2.0, 2.0, 2.0, 2.0]
    order, pairs = observed_order(errs, ratios)
    assert order == pytest.approx(2.0, rel=1e-9)
    assert len(pairs) == 4


def test_observed_order_skips_growth_and_reports_nothing_when_flat():
    order, pairs = observed_order([1.0, 2.0, 0.5], [2.0, 2.0])  # one growth pair
    assert len(pairs) == 1
    assert order == pytest.approx(math.log2(4.0), rel=1e-9)
    assert observed_order([1.0, 1.0, 1.0], [2.0, 2.0])[0] == 0.0


def test_error_bound_adds_reference_residual_extrapolation():
    # err(h) = h^2 anchored so err(1)=1, ref at h=0.1 -> ref err = 0.01
    bound = error_bound(1.0, 0.01, 2.0, 0.1)
    assert bound == pytest.approx(1.0 + 0.01 * 0.1 ** 2, rel=1e-9)
    # a zero fitted order falls back to the tightest measured error as its
    # own reference residual (no bogus extrapolation)
    assert error_bound(2.0, 0.5, 0.0, 0.1) == pytest.approx(2.5)


def test_rk4_is_fourth_order_and_lands_on_t_end():
    rhs = lambda t, s: -np.asarray(s)                     # y' = -y, y(0)=1
    t, ys = rk4(rhs, [1.0], 1.0, 0.05)
    assert t[-1] == pytest.approx(1.0)
    assert ys.shape[0] == len(t) == int(round(1.0 / 0.05)) + 1
    assert abs(ys[-1, 0] - math.exp(-1.0)) < 3e-7
    t2, ys2 = rk4(rhs, [1.0], 1.0, 0.025)
    err1 = abs(ys[-1, 0] - math.exp(-1.0))
    err2 = abs(ys2[-1, 0] - math.exp(-1.0))
    assert 14.0 < err1 / err2 < 18.0                      # 2^4 = 16


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------

def test_parse_budgets_accepts_floats_and_none_strings():
    got = parse_budgets({"leo_j2_coast.position_m": "20", "gto_coast.velocity_m_s": None})
    assert got["leo_j2_coast.position_m"] == 20.0
    assert got["gto_coast.velocity_m_s"] is None
    b = Budgets(overrides=got)
    assert b.resolve("leo_j2_coast", "position_m") == 20.0
    assert b.resolve("gto_coast", "velocity_m_s") is None
    assert b.resolve("leo_j2_coast", "velocity_m_s") == 1e-3   # scenario default


def test_parse_budgets_rejects_garbage():
    with pytest.raises(ValueError, match="<scenario>.<metric>"):
        parse_budgets({"nonsense": 1.0})
    with pytest.raises(ValueError, match="> 0"):
        parse_budgets({"leo_j2_coast.position_m": -1})
    with pytest.raises(ValueError):
        parse_budgets({"entry_descent.position_m": math.nan})


# ---------------------------------------------------------------------------
# one scenario, both axes
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini_result():
    return run_scenario(_mini_leo())


def test_run_scenario_axes_and_curve_shape(mini_result):
    assert set(mini_result["axes"]) == set(AXES)
    tol = mini_result["axes"]["tolerance"]
    assert len(tol["curve"]) == len(TOL_FACTORS)
    assert [c["factor"] for c in tol["curve"]] == list(TOL_FACTORS)
    step = mini_result["axes"]["fixed_step"]
    assert [c["nominal_step_s"] for c in step["curve"]] == [32.0, 16.0, 8.0, 4.0]
    # every curve point carries the two error metrics and the reference is
    # never itself in the curve (it sits one grid interval finer)
    for c in step["curve"]:
        assert c["error_position_m"] > 0.0 and c["error_velocity_m_s"] > 0.0


def test_run_scenario_orders_and_monotonic_convergence(mini_result):
    tol, step = (mini_result["axes"][a] for a in AXES)
    assert step["order"]["governing"] > 3.5               # smooth Kepler: RK4 order 4
    assert not step["nonmonotonic"]
    assert not tol["nonmonotonic"]
    errs = [c["error_position_m"] for c in step["curve"]]
    assert all(a > b for a, b in zip(errs, errs[1:]))     # textbook halving


def test_run_scenario_default_bound_and_verdict(mini_result):
    tol = mini_result["axes"]["tolerance"]
    bound = tol["error_bound"]["at_default_position_m"]
    default_pt = tol["curve"][list(TOL_FACTORS).index(1.0)]
    assert bound >= default_pt["error_position_m"]        # bound >= measurement
    assert bound < 50.0 / 3.0                             # budget / safety
    rec = tol["recommendation"]
    assert rec["justified_default"]
    assert "justified" in rec["verdict"]
    # adaptive max-step audit exists for a finite production cap
    assert default_pt["max_realised_step_s"] is not None


def test_residual_shape_and_category(mini_result):
    for r in mini_result["residuals"]:
        assert r["category"] == CHECK_NUMERICAL
        assert r["name"] in {"mini_leo.tolerance", "mini_leo.fixed_step"}
        assert r["position_residual_m"] > 0.0 and "reference" in r
        assert r["within_tolerance"] is True


def test_budget_none_records_without_gating():
    res = run_scenario(_mini_leo(), Budgets(overrides={"mini_leo.position_m": None}))
    r = next(r for r in res["residuals"] if r["name"].endswith(".tolerance"))
    assert r["tolerance"]["position_m"] is None
    assert r["within_tolerance"] is True                  # gated on velocity only


def test_tight_budget_fails_the_check_and_hides_recommendation():
    res = run_scenario(_mini_leo(budget={"position_m": 1e-6, "velocity_m_s": 1e-9}))
    for r in res["residuals"]:
        assert r["within_tolerance"] is False
    step_rec = res["axes"]["fixed_step"]["recommendation"]
    assert step_rec["recommended_step_s"] is None or step_rec["recommended_measured"] is False
    assert "budget" in step_rec["verdict"] or "below" in step_rec["verdict"] \
        or "order" in step_rec["verdict"]


def test_legacy_fixed_step_audit_uses_production_cap():
    step = run_scenario(_mini_leo())["axes"]["fixed_step"]
    audit = step["recommendation"]["legacy_fixed_step_audit"]
    assert audit["step_s"] == 300.0                       # the mini production cap
    assert audit["extrapolated"] is True                  # beyond the measured grid
    assert audit["error_position_m_est"] > step["curve"][0]["error_position_m"]


def test_axes_subset_and_unknown_axis():
    res = run_scenario(_mini_leo(), axes=["fixed_step"])
    assert set(res["axes"]) == {"fixed_step"}
    with pytest.raises(ValueError, match="unknown axis"):
        run_scenario(_mini_leo(), axes=["teleport"])


# ---------------------------------------------------------------------------
# study / report / CLI
# ---------------------------------------------------------------------------

def test_study_report_validates_and_is_deterministic(tmp_path, monkeypatch):
    spec = _mini_leo()
    monkeypatch.setitem(SCENARIOS, "mini_leo", spec)
    a = run_convergence_study(["mini_leo"], report_path=tmp_path / "a.json")
    b = run_convergence_study(["mini_leo"], report_path=tmp_path / "b.json")
    validate_report(a.report)                             # schema + digest
    assert a.report["kind"] == "integrator-convergence"
    assert a.report["digest"] == b.report["digest"]       # reruns are stable
    assert a.result == RESULT_PASSED
    assert a.report["status"]["categories"]["numerical_consistency"]["checks_total"] == 2
    assert json.loads((tmp_path / "a.json").read_text())["inputs"]["safety_margin"] == 3.0


def test_study_fails_status_with_impossible_budget(tmp_path, monkeypatch):
    monkeypatch.setitem(SCENARIOS, "mini_leo", _mini_leo())
    study = run_convergence_study(["mini_leo"],
                                  budgets={"mini_leo.position_m": 1e-9},
                                  report_path=tmp_path / "r.json")
    assert study.result == RESULT_FAILED


def test_study_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown scenarios"):
        run_convergence_study(["nope"])
    with pytest.raises(ValueError, match="safety margin"):
        run_convergence_study(["mini_leo"], safety=0.0)


def test_cli_exit_code_and_json_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(SCENARIOS, "mini_leo", _mini_leo())
    out = tmp_path / "cli.json"
    assert main(["--scenarios", "mini_leo", "--budget", "mini_leo.velocity_m_s=none",
                 "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "=== mini_leo ===" in printed and "verification" in printed
    doc = json.loads(out.read_text())
    assert doc["status"]["result"] == RESULT_PASSED
    assert main(["--scenarios", "mini_leo", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    validate_report(doc)
    monkeypatch.setitem(SCENARIOS, "mini_leo",
                        _mini_leo(budget={"position_m": 1e-9, "velocity_m_s": 1e-9}))
    assert main(["--scenarios", "mini_leo", "-o", str(tmp_path / "f.json")]) == 1


# ---------------------------------------------------------------------------
# the real key scenarios (slow: ~1-2 min for the full four-way sweep)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_key_scenarios_converge_within_budget():
    study = run_convergence_study()                       # all four key scenarios
    validate_report(study.report)
    assert study.result == RESULT_PASSED
    assert study.report["status"]["categories"]["numerical_consistency"]["checks_total"] == 8
    for res in study.scenarios:
        for axis in res["axes"].values():
            o = axis["order"]
            assert o["governing"] >= o["minimum_required"]
    assert study.report["inputs"]["scenarios"][0]["scenario"]["name"] == "leo_j2_coast"


@pytest.mark.slow
def test_production_defaults_are_the_audited_settings():
    """The sweep only justifies anything if it sweeps the settings production
    actually uses."""
    import inspect

    from orbitopt.entry import fly_entry
    from orbitopt.mission.propagation import DEFAULT_MAX_STEP_S, InjectedOrbitPropagation

    leo = SCENARIOS["leo_j2_coast"].production
    f = InjectedOrbitPropagation.__dataclass_fields__
    assert leo == {"rtol": float(f["rtol"].default), "atol": float(f["atol"].default),
                   "max_step_s": float(DEFAULT_MAX_STEP_S)}
    ent = SCENARIOS["entry_descent"].production
    sig = inspect.signature(fly_entry).parameters
    assert ent["rtol"] == float(sig["rtol"].default)
    assert ent["atol"] == float(sig["atol"].default)
    assert ent["max_step_s"] == float(sig["max_step_s"].default)
    asc = SCENARIOS["meco_ascent"].production
    src = inspect.getsource(
        __import__("orbitopt.verify.cross_model", fromlist=["_produce_three_dof"])
        ._produce_three_dof)
    assert asc["rtol"] == 1e-10 and "rtol=1e-10" in src
    assert "atol=1e-5" in src and asc["atol"] == 1e-5
