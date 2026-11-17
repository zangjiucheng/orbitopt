"""E8-02 model discrepancy budget (task #282).

The structure tests (line/budget accounting, recipes, causes, gate mirroring)
run on the real always-available backends with the probes switched off and a
short coast -- no JSBSim, seconds each. The one decomposition test that flies
the 6-DOF probe flights is ``slow``.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from orbitopt.verify.cross_model import CrossModelScenario
from orbitopt.verify.discrepancy_budget import (
    BUDGET_CHANNELS,
    BUDGET_SOURCES,
    STATUS_DECLARED_ZERO,
    STATUS_MEASURED_TOTAL,
    STATUS_NOT_MEASURED,
    ChannelBudget,
    SourceLine,
    main,
    run_discrepancy_budget,
    _lines_for_pair,
    _Probes,
)
from orbitopt.verify.report import RESULT_FAILED, RESULT_PASSED, validate_report


def _short_coast() -> CrossModelScenario:
    return replace(CrossModelScenario(), coast_duration_s=60.0)


@pytest.fixture(scope="module")
def budget_run():
    """One structure run: real 3-DOF + analytic backends, probes off."""
    return run_discrepancy_budget(backends=["three_dof", "two_body_analytic"],
                                  scenario=_short_coast(), probes=False)


def _budget(res, pair, phase, channel):
    return next(b for b in res.budgets
                if (b.pair, b.phase, b.channel) == (pair, phase, channel))


# ---------------------------------------------------------------------------
# Line / budget accounting
# ---------------------------------------------------------------------------

def test_source_line_rules():
    with pytest.raises(ValueError):
        SourceLine("bogus_source", 0.0, "bound", "x")
    with pytest.raises(ValueError):
        SourceLine("engine", 1.0, "not_measured", "a not_measured line may not pay")
    with pytest.raises(ValueError):
        SourceLine("engine", 0.0, "magical", "x")
    line = SourceLine("engine", 2.0, "bound", "x")
    assert line.matrix_class == ""               # the class is set by the recipe
    assert line.to_dict()["probe"] == {}


def test_channel_budget_accounting():
    lines = tuple(
        SourceLine(s, {"engine": 3.0, "frame_time": 1.0}.get(s, 0.0),
                   "bound" if s == "engine" else "declared_zero", "x")
        for s in BUDGET_SOURCES)
    b = ChannelBudget("p_vs_q", "ascent", "position_m", "m", 10.0, lines)
    assert b.components["engine"] == 3.0
    assert b.component_sum == 4.0
    assert b.residual == 6.0           # recorded, not zeroed
    assert b.coverage == pytest.approx(0.4)
    over = ChannelBudget("p_vs_q", "ascent", "position_m", "m", 1.0, lines)
    assert over.residual == -3.0       # conservative bounds: legal, recorded
    with pytest.raises(ValueError):    # lines must be exactly the five sources
        ChannelBudget("p_vs_q", "ascent", "position_m", "m", 1.0, lines[:3])


# ---------------------------------------------------------------------------
# Structure run (probes off): every line accounted, residual = total
# ---------------------------------------------------------------------------

def test_every_flown_pair_has_all_five_source_lines(budget_run):
    flown = {(b.pair, b.phase) for b in budget_run.budgets if b.total is not None}
    assert flown == {("three_dof_vs_two_body_analytic", "coast")}
    for b in budget_run.budgets:
        assert tuple(l.source for l in b.lines) == BUDGET_SOURCES
        assert b.channel in BUDGET_CHANNELS
        if b.total is None:
            assert b.residual is None
            assert all(l.status == STATUS_NOT_MEASURED for l in b.lines)


def test_same_model_coast_numerics_carries_the_total(budget_run):
    for channel in BUDGET_CHANNELS:
        b = _budget(budget_run, "three_dof_vs_two_body_analytic", "coast", channel)
        num = next(l for l in b.lines if l.source == "numerical_integration")
        assert num.status == STATUS_MEASURED_TOTAL
        assert num.matrix_class == "numerical_integration"   # cause vocabulary
        assert num.contribution == pytest.approx(b.total)
        assert abs(b.residual) < 1e-12          # fully attributed
        assert b.matrix_result == "passed"      # real physics, real verdict
        assert b.matrix_limit is not None


def test_probe_off_and_missing_backends_never_fake_a_zero(budget_run):
    ascent = _budget(budget_run, "six_dof_vs_three_dof", "ascent", "position_m")
    assert ascent.total is None and ascent.matrix_result == "not_checked"
    causes = {l.cause for l in ascent.lines}
    assert causes == {"backend_not_requested"}   # the matrix's own cause code
    coast = _budget(budget_run, "three_dof_vs_tudatpy", "coast", "position_m")
    assert coast.total is None
    assert {l.cause for l in coast.lines} == {"backend_not_requested"}


def test_report_mirrors_the_matrix_and_validates(budget_run):
    validate_report(budget_run.report)
    assert budget_run.report["kind"] == "model-discrepancy-budget"
    assert budget_run.result == RESULT_PASSED
    by_name = {r["name"]: r for r in budget_run.report["residuals"]}
    b = _budget(budget_run, "three_dof_vs_two_body_analytic", "coast", "position_m")
    r = by_name["three_dof_vs_two_body_analytic.coast.position_m"]
    assert r["within_tolerance"] is True
    assert r["components"] == b.components
    assert r["component_sum"] == pytest.approx(b.component_sum)
    assert r["component_residual"] == pytest.approx(b.residual)
    assert r["observed"]["position_m"] == pytest.approx(b.total)
    assert r["tolerance"]["position_m"] == b.matrix_limit
    assert set(budget_run.report["inputs"]["sources"]) == set(BUDGET_SOURCES)


def test_engine_line_refuses_an_unavailable_parity_basis(budget_run):
    """Probes off means the E6-01 parity basis was not even read: the engine
    line is not_measured with the honest cause, never a silent zero."""
    ascent = _budget(budget_run, "six_dof_vs_three_dof", "ascent", "position_m")
    assert next(l for l in ascent.lines if l.source == "engine").cause == \
        "backend_not_requested"      # pair did not fly -> cause from the matrix
    P = _Probes(_short_coast(), {}, enabled=False)
    assert P.engine_residuals() is None


def test_unmatriculated_and_unknown_pairs_get_no_recipe():
    """A coast pair outside the same-model set gets five not_measured lines
    (full total stays in the residual) -- no bogus same-model claim."""
    lines = _lines_for_pair("three_dof_vs_skyrocket_vs_9000", "coast",
                            "position_m", 12.0, CrossModelScenario(),
                            _Probes(CrossModelScenario(), {}, False), None)
    assert len(lines) == 5 and all(l.status == STATUS_NOT_MEASURED for l in lines)
    assert {l.cause for l in lines} == {"no_decomposable_recipe"}


# ---------------------------------------------------------------------------
# CLI + gating behaviour
# ---------------------------------------------------------------------------

def test_cli_exit_zero_on_pass_and_one_on_a_gated_failure(tmp_path):
    out = tmp_path / "budget.json"
    rc = main(["--backends", "three_dof,two_body_analytic", "--no-probes",
               "-o", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    validate_report(report)
    assert report["status"]["result"] == RESULT_PASSED
    rc_fail = main(["--backends", "three_dof,two_body_analytic", "--no-probes",
                    "--threshold", "coast.position_m=1e-9", "-o", str(out)])
    assert rc_fail == 1
    failed = json.loads(out.read_text(encoding="utf-8"))
    assert failed["status"]["result"] == RESULT_FAILED
    row = next(r for r in failed["residuals"]
               if r["name"] == "three_dof_vs_two_body_analytic.coast.position_m")
    assert row["within_tolerance"] is False
    # the budget keeps recording even when the gate says no:
    assert row["components"]["numerical_integration"] == \
        pytest.approx(row["component_sum"])


# ---------------------------------------------------------------------------
# The decomposition with probes flying (needs jsbsim)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ascent_budget_decomposition_with_probes():
    pytest.importorskip("jsbsim", reason="the decomposition probes require the optional JSBSim backend")
    res = run_discrepancy_budget(backends=["three_dof", "six_dof"])
    assert res.result == RESULT_PASSED
    pos = _budget(res, "six_dof_vs_three_dof", "ascent", "position_m")
    lines = {l.source: l for l in pos.lines}
    assert pos.total > 100.0                       # the real 6v3 MECO spread
    assert lines["frame_time"].status == STATUS_DECLARED_ZERO
    assert lines["frame_time"].probe["dt_desync_s"] == 0.9
    assert lines["frame_time"].contribution == 0.0
    # numerics: the relaxed-reference probe, a true share (not the total):
    assert lines["numerical_integration"].status == "measured"
    assert 0.0 < lines["numerical_integration"].contribution < pos.total
    # engine: based on the E6-01 ALIGNED model -- the live parity residual,
    # which the alignment keeps at float-noise scale (acceptance criterion):
    eng = lines["engine"]
    assert eng.status == "measured"
    assert eng.probe["thrust_rel_residual"] < 1e-6
    assert "E6-01" in eng.probe["basis"]
    assert eng.contribution < lines["numerical_integration"].contribution
    # aero: zero by declaration, potential recorded but NOT summed:
    assert lines["aero"].status == STATUS_DECLARED_ZERO
    assert lines["aero"].contribution == 0.0
    assert lines["aero"].probe["potential"]["final_position_m"] > 1000.0
    # controller: a partial (probe) share, loop-stiffening method recorded:
    ctl = lines["controller"]
    assert ctl.status == "probe" and ctl.contribution > 0.0
    assert ctl.probe["stiff_pair_diff"] < pos.total
    # the residual is the *formulation* remainder: recorded, large, honest.
    assert pos.residual == pytest.approx(pos.total - pos.component_sum)
    assert 0.0 < pos.coverage < 0.5
    # mass channel: the disclosed one-frame bookkeeping bound covers the
    # total (ADR flightlab-0009; mdot * dt with the harness dt):
    mass = _budget(res, "six_dof_vs_three_dof", "ascent", "mass_kg")
    eng_m = next(l for l in mass.lines if l.source == "engine")
    assert eng_m.status == "bound" and eng_m.matrix_class == "propellant_accounting"
    assert eng_m.contribution == pytest.approx(res.harness.scenario.mdot_kg_s * 0.01)
    assert mass.coverage > 0.99
