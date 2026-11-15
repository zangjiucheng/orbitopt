"""V-06 cross-backend acceptance matrix (task #262).

The matrix machinery (config validation, grading, cause classification,
report, pipeline gate) runs on the V-02 **fake backends** in the fast suite;
only the real 3-DOF-vs-analytic regression needs physics and is ``slow``.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from orbitopt.verify.cross_model import (
    THRESHOLD_METRICS,
    ModelTrack,
    register_backend,
)
from orbitopt.verify.report import (
    CHECK_MODEL,
    CHECK_NUMERICAL,
    RESULT_NOT_CHECKED,
    RESULT_PASSED,
    validate_report,
)
from orbitopt.verify.acceptance_matrix import (
    ROW_FAILED,
    ROW_NOT_CHECKED,
    ROW_PASSED,
    ROW_RECORDED,
    ROW_UNMATRICULATED,
    FAILURE_CLASSES,
    default_matrix,
    default_matrix_path,
    evaluate_matrix,
    load_matrix,
    matrix_schema_path,
    main,
    run_acceptance_matrix,
)

MU = 3.986004418e14
FAKES = ("fake_a", "fake_b", "fake_c", "fake_off", "fake_boom")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _circular_track(label, t, radius=6_778_137.0, dr_m=0.0, mass_kg=None):
    """Synthetic Kepler-sampled track (fast fake, same trick as
    test_verify_cross_model) with an optional radial offset."""
    from orbitopt.verify.benchmarks import propagate_kepler_analytic

    i = math.radians(45.0)
    r0 = np.array([radius + dr_m, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU / radius) * math.cos(i),
                   math.sqrt(MU / radius) * math.sin(i)])
    states = np.array([propagate_kepler_analytic(np.concatenate([r0, v0]),
                                                 float(dt), MU) for dt in t])
    m = None if mass_kg is None else np.full(len(t), float(mass_kg))
    return ModelTrack(label, np.asarray(t, dtype=float), states, m)


@pytest.fixture()
def fake_backends():
    """fake_a vs fake_b: 10 m radial offset + 10 kg mass difference;
    fake_c: 5 m offset, NO mass channel; fake_off: not installed;
    fake_boom: raises at run time. Covers every cause-classification path."""
    t = np.arange(0.0, 100.0 + 1e-9, 0.5)
    t_b = np.arange(0.0, 100.0 + 1e-9, 0.4)  # different grid: force resampling
    a = _circular_track("fake_a", t, mass_kg=50_000.0)
    b = _circular_track("fake_b", t_b, dr_m=10.0, mass_kg=49_990.0)
    c = _circular_track("fake_c", t, dr_m=5.0)

    register_backend("fake_a", lambda s, ctx: (a, None))
    register_backend("fake_b", lambda s, ctx: (b, None))
    register_backend("fake_c", lambda s, ctx: (c, None))
    register_backend("fake_off", lambda s, ctx: (None, None), lambda: False)
    register_backend("fake_boom", lambda s, ctx: (_ for _ in ()).throw(
        RuntimeError("boom")))
    yield {}
    from orbitopt.verify import cross_model as cm

    for name in FAKES:
        cm.BACKENDS.pop(name, None)


BASIS = "synthetic fixture basis, long enough for the matrix schema"


def _row(limit=100.0, kind="measured_agreement", on_fail="model_formulation"):
    row = {"limit": limit, "basis_kind": kind, "basis": BASIS}
    if limit is not None and on_fail:
        row["on_fail"] = on_fail
    return row


def _cell(pair, phase, overrides=None, category="model_agreement",
          stages=("dynamics",)):
    tolerances = {m: _row() for m in THRESHOLD_METRICS}
    tolerances["raan_deg"] = _row(None, "ill_conditioned")
    tolerances["argument_of_perigee_deg"] = _row(None, "ill_conditioned")
    tolerances.update(overrides or {})
    return {"pair": pair, "phase": phase, "stages": list(stages),
            "category": category,
            "scope": "fixture cell covering one backend pair for one phase",
            "not_proven": "does not prove anything outside the fixture",
            "tolerances": tolerances}


def _doc(cells, name="fixture-matrix"):
    return {"schemaVersion": "1.0", "name": name,
            "stages": ["dynamics", "control", "propagation", "handoff"],
            "cells": list(cells)}


TMATRIX = _doc([
    _cell("fake_a_vs_fake_b", "ascent"),
    _cell("fake_a_vs_fake_c", "ascent",
          {"mass_kg": _row(1e-6, "physical_reasoning", "propellant_accounting")}),
    _cell("fake_off_vs_fake_a", "ascent", {"position_m": _row(1.0)}),
    _cell("fake_boom_vs_fake_a", "ascent", {"position_m": _row(1.0)}),
])


# ---------------------------------------------------------------------------
# shipped configuration: complete, justified, cause-coded (acceptance line)
# ---------------------------------------------------------------------------

def test_default_matrix_is_complete_and_every_row_judged():
    mat = default_matrix()
    assert mat.cells, "the shipped matrix must not be empty"
    for cell in mat.cells:
        assert set(cell.rows) == set(THRESHOLD_METRICS), \
            f"{cell.pair} @ {cell.phase} must declare a row for every metric"
        for row in cell.rows.values():
            assert len(row.basis.strip()) >= 24, f"{cell.pair}.{row.metric}: empty basis"
            assert row.basis_kind in ("convergence_analysis", "physical_reasoning",
                                      "measured_agreement", "ill_conditioned")
            if row.limit is None:
                assert row.basis_kind == "ill_conditioned", \
                    "a row without a limit must declare why it is ill-conditioned"
            else:
                assert row.limit > 0 and row.on_fail in FAILURE_CLASSES, \
                    f"{cell.pair}.{row.metric}: gated row needs an on_fail cause code"
        assert set(cell.stages) <= set(mat.stages)


def test_default_matrix_covers_the_default_harness_pairs():
    mat = default_matrix()
    assert {(c.pair, c.phase) for c in mat.cells} == {
        ("six_dof_vs_three_dof", "ascent"),
        ("three_dof_vs_two_body_analytic", "coast"),
        ("three_dof_vs_tudatpy", "coast"),
        ("tudatpy_vs_two_body_analytic", "coast"),
    }


def test_stage_budgets_match_their_declared_sources():
    """The YAML claims its tolerances hang off the V-03 convergence budgets,
    the FL-403 measured deviations and the M-05 gate -- verify, don't trust."""
    from orbitopt.mission.propagation import POINT_MASS_POS_TOL_M, POINT_MASS_VEL_TOL_M_S
    from orbitopt.verify.integrator_convergence import SCENARIOS

    budgets = default_matrix().stage_budgets
    for name in ("meco_ascent", "entry_descent", "leo_j2_coast", "gto_coast"):
        assert budgets["dynamics"]["scenarios"][name] == SCENARIOS[name].budget
    text = default_matrix_path().read_text(encoding="utf-8")
    assert "51 m MECO" in text and "2.0 m/s velocity" in text
    findings = (pathlib.Path(__file__).resolve().parent.parent
                    / "docs" / "flightlab-jsbsim-spike-findings.md").read_text(encoding="utf-8")
    assert "\u221251 m alt / \u22122.0 m/s" in findings and "0.51\u00b0 max" in findings
    assert budgets["control"]["measured"] == {"meco_position_m": 51.0,
                                              "meco_velocity_m_s": 2.0,
                                              "attitude_whole_burn_deg": 0.51}
    assert (POINT_MASS_POS_TOL_M, POINT_MASS_VEL_TOL_M_S) == (100.0, 0.01)
    assert "100 m / 0.01 m/s" in text


def test_failure_catalog_covers_every_schema_cause_code():
    schema = json.loads(matrix_schema_path().read_text(encoding="utf-8"))
    codes = set(schema["$defs"]["onFail"]["enum"])
    assert codes <= set(FAILURE_CLASSES)
    for code, fc in FAILURE_CLASSES.items():
        assert fc.title and fc.category in {
            "numerics", "physics", "control", "convention", "plumbing",
            "execution", "environment", "data", "matrix"}
        assert fc.title and fc.explanation and len(fc.next_step) > 12


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_load_matrix_rejects_documents_without_justification():
    bad = _doc([_cell("x_vs_y", "coast")])
    bad["cells"][0]["tolerances"]["position_m"].pop("basis")
    with pytest.raises(ValueError, match="basis"):
        load_matrix_from_doc(bad)


def load_matrix_from_doc(document):
    from orbitopt.verify.acceptance_matrix import load_matrix_from_dict

    return load_matrix_from_dict(document)


def _mut(pop_ecc=False, dup=False, ill_limit=False, unlimit=False,
         bad_category=False, bad_onfail=False):
    def mutate(d):
        if pop_ecc:
            d["cells"][0]["tolerances"].pop("eccentricity")       # incomplete cell
        if dup:
            d["cells"].append(d["cells"][0])                      # duplicate cell
        if ill_limit:
            d["cells"][0]["tolerances"]["raan_deg"].update({"limit": 0.1})  # ill + limit
        if unlimit:
            d["cells"][0]["tolerances"]["position_m"].update({"limit": None})  # no why
        if bad_category:
            d["cells"][0]["category"] = "bogus"
        if bad_onfail:
            d["cells"][0]["tolerances"]["position_m"]["on_fail"] = "bogus"
        return d
    return mutate


@pytest.mark.parametrize("mutate", [
    _mut(pop_ecc=True), _mut(dup=True), _mut(ill_limit=True), _mut(unlimit=True),
    _mut(bad_category=True), _mut(bad_onfail=True),
])
def test_load_matrix_rejects_structurally_bad_documents(mutate):
    with pytest.raises(ValueError):
        load_matrix_from_doc(mutate(_doc([_cell("x_vs_y", "coast")])))


def test_load_matrix_accepts_yaml_or_json_files(tmp_path):
    import yaml

    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(_doc([_cell("x_vs_y", "coast")])), encoding="utf-8")
    mat = load_matrix(path)
    assert mat.name == "fixture-matrix"
    assert mat.cells[0].pair == "x_vs_y"
    jpath = tmp_path / "matrix.json"
    jpath.write_text(json.dumps(_doc([_cell("x_vs_y", "coast")], name="json-matrix")),
                     encoding="utf-8")
    assert load_matrix(jpath).name == "json-matrix"


# ---------------------------------------------------------------------------
# grading on fake backends (fast)
# ---------------------------------------------------------------------------

def _run(matrix=TMATRIX, backends=("fake_a", "fake_b", "fake_off", "fake_boom"),
       thresholds=None):
    return run_acceptance_matrix(matrix, backends=list(backends), thresholds=thresholds)


def test_matrix_report_is_schema_valid_and_gates_the_pipeline(fake_backends):
    res = _run()
    assert res.result == RESULT_PASSED
    validate_report(res.report)
    assert res.report["kind"] == "acceptance-matrix"
    assert res.report["inputs"]["summary"]["by_result"][ROW_PASSED] >= 6
    causes = res.report["inputs"]["summary"]["by_failure_class"]
    assert causes.get("backend_environment") and causes.get("backend_runtime_error")

    from orbitopt.candidate_selection import VERDICT_ACCEPTED, VerificationGate

    gate = VerificationGate()
    gate.register(0, scenario="acceptance-matrix")
    assert gate.evaluate(0, [res.report])["verdict"] == VERDICT_ACCEPTED


def test_a_failed_row_carries_its_declared_cause(fake_backends):
    res = _run(thresholds={"fake_a_vs_fake_b.ascent.position_m": 1.0})
    bad = [r for r in res.rows if r.result == ROW_FAILED]
    assert [r.metric for r in bad] == ["position_m"]
    assert bad[0].failure_class == "model_formulation"
    assert bad[0].value == pytest.approx(res.harness.pair_metrics[0]["metrics"]
                                         ["final_position_m"], abs=1e-6)
    assert res.result == "failed"
    assert res.report["status"]["result"] == "failed"

    from orbitopt.candidate_selection import VERDICT_DEGRADED, VerificationGate

    gate = VerificationGate()
    gate.register(1)
    assert gate.evaluate(1, [res.report])["verdict"] == VERDICT_DEGRADED


def test_absent_backends_are_cause_classified_not_faked(fake_backends):
    res = _run()
    by = {(r.pair, r.phase, r.metric): r for r in res.rows}
    off = by[("fake_off_vs_fake_a", "ascent", "position_m")]
    assert (off.result, off.failure_class) == (ROW_NOT_CHECKED, "backend_environment")
    boom = by[("fake_boom_vs_fake_a", "ascent", "position_m")]
    assert (boom.result, boom.failure_class) == (ROW_NOT_CHECKED, "backend_runtime_error")
    missing = by[("fake_a_vs_fake_c", "ascent", "position_m")]
    assert missing.failure_class == "backend_not_requested"  # not in --backends
    # absence of evidence never votes
    assert res.result == RESULT_PASSED


def test_missing_mass_channel_is_a_named_cause_not_a_zero(fake_backends):
    res = _run(backends=("fake_a", "fake_c"),
               matrix=_doc([_cell("fake_a_vs_fake_c", "ascent",
                                  {"mass_kg": _row(1e-6, "physical_reasoning",
                                                   "propellant_accounting")})]))
    mass = next(r for r in res.rows if r.metric == "mass_kg")
    assert (mass.result, mass.failure_class, mass.value) == \
        (ROW_NOT_CHECKED, "channel_unavailable", None)
    assert res.report["status"]["result"] == RESULT_PASSED  # position still gates


def test_pairs_the_matrix_does_not_cover_are_recorded_not_ignored(fake_backends):
    res = _run(matrix=_doc([_cell("fake_a_vs_fake_b", "ascent")]),
               backends=("fake_a", "fake_b", "fake_c"))
    extra = [r for r in res.rows if r.pair == "fake_b_vs_fake_c"]
    assert len(extra) == len(THRESHOLD_METRICS)
    assert {r.result for r in extra} == {ROW_UNMATRICULATED}
    assert {r.failure_class for r in extra} == {"no_matrix_cell"}
    assert res.result == RESULT_PASSED  # unmatriculated never votes


def test_ill_conditioned_and_ungated_rows_are_recorded_only(fake_backends):
    res = _run()
    by = {(r.metric, r.pair): r for r in res.rows}
    raan = by[("raan_deg", "fake_a_vs_fake_b")]
    assert (raan.result, raan.limit, raan.failure_class) == (ROW_RECORDED, None, None)
    res2 = _run(thresholds={"fake_a_vs_fake_b.ascent.position_m": "none"})
    pos = next(r for r in res2.rows if r.pair == "fake_a_vs_fake_b" and r.metric == "position_m")
    assert (pos.result, pos.failure_class) == (ROW_RECORDED, "ungated_override")
    assert res2.result == RESULT_PASSED  # an un-gated row cannot fail the matrix


def test_user_override_wins_over_the_matrix_everywhere(fake_backends):
    loose = _cell("fake_a_vs_fake_b", "ascent", {"position_m": _row(50.0)})
    doc = _doc([loose, _cell("fake_a_vs_fake_c", "ascent")])
    assert _run(matrix=doc).result == RESULT_PASSED
    # phase-level override tightens the matrix row...
    assert _run(matrix=doc, thresholds={"ascent.position_m": 0.1}).result == "failed"
    # ...but the full-form override still outranks it (V-02 precedence)
    strict = _run(matrix=doc, thresholds={"ascent.position_m": 0.1,
                                          "fake_a_vs_fake_b.ascent.position_m": 50.0})
    assert strict.result == RESULT_PASSED
    row = next(r for r in strict.rows if r.pair == "fake_a_vs_fake_b"
               and r.metric == "position_m")
    assert row.limit == 50.0
    # and the harness saw the exact same resolved limit (no contradiction)
    assert strict.harness.pair_metrics[0]["tolerance"]["position_m"] == 50.0


def test_matrix_category_must_match_the_harness(fake_backends):
    doc = _doc([_cell("fake_a_vs_fake_b", "ascent", category="numerical_consistency")])
    with pytest.raises(ValueError, match="categorises this pair"):
        _run(matrix=doc)


def test_matrix_result_is_reproducible_and_digested(fake_backends):
    res = _run()
    assert res.report["inputs"]["matrix"]["config_digest"]
    res2 = _run()
    assert res2.report["digest"] == res.report["digest"]
    rows_again = evaluate_matrix(res.matrix, res.harness)
    assert [(r.pair, r.phase, r.metric, r.result) for r in rows_again] == \
        [(r.pair, r.phase, r.metric, r.result) for r in res.rows]


def test_cli_table_json_and_exit_codes(fake_backends, tmp_path, capsys):
    import yaml

    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(TMATRIX), encoding="utf-8")
    out = tmp_path / "report.json"
    assert main(["--matrix", str(path),
                 "--backends", "fake_a,fake_b,fake_off,fake_boom",
                 "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "fake_a_vs_fake_b @ ascent" in text and "cause backend_environment" in text
    assert out.read_text().strip() != ""
    assert main(["--matrix", str(path), "--backends", "fake_a,fake_b",
                 "--threshold", "fake_a_vs_fake_b.ascent.position_m=1e-9"]) == 1
    capsys.readouterr()
    assert main(["--matrix", str(path), "--backends", "fake_a,fake_b", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"]["result"] == RESULT_PASSED
    validate_report(doc)


# ---------------------------------------------------------------------------
# real physics regression (slow): the shipped numerical cell really holds
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_default_matrix_grades_the_real_numerical_pair_as_passed():
    res = run_acceptance_matrix(backends=["three_dof", "two_body_analytic"])
    assert res.result == RESULT_PASSED
    validate_report(res.report)
    cells = {(r.pair, r.phase): r for r in res.rows if r.pair == "three_dof_vs_two_body_analytic"}
    assert cells[("three_dof_vs_two_body_analytic", "coast")].category == CHECK_NUMERICAL
    for row in res.rows:
        if row.pair == "three_dof_vs_two_body_analytic" and row.result == ROW_PASSED:
            assert row.value <= row.limit
    causes = set(res.report["inputs"]["summary"]["by_failure_class"])
    assert causes == {"backend_not_requested"}
    assert res.report["status"]["result"] == RESULT_PASSED
