"""E12-02 (task #392): CW validity-envelope gates.

Fast gates on the committed benchmark JSON
(docs/benchmarks/e12_02_cw_validity.json): the envelope the docs promise is
the envelope the data shows, the linearisation scales as theory says it
must, eccentricity really does dominate, and the nonlinear baseline was
self-checked. Regenerate with

    PYTHONPATH=src python benchmarks/bench_e12_02_cw_validity.py \
        -o docs/benchmarks/e12_02_cw_validity.json

(~2 min, CPU only). The runtime twin of this gate -- CW against the
propagator on a live case -- lives in test_e12_02_relative_dynamics.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs/benchmarks/e12_02_cw_validity.json"


@pytest.fixture(scope="module")
def doc():
    assert DOC.exists(), f"run the benchmark first: {DOC}"
    return json.loads(DOC.read_text(encoding="utf-8"))


def case(doc, e, j2, sep, orbits):
    for r in doc["cases"]:
        if (r["e"] == e and r["j2"] is j2 and r["sep_m"] == sep
                and r["orbits"] == orbits):
            return r
    raise KeyError((e, j2, sep, orbits))


def envelope(doc, e, j2, sep):
    for r in doc["envelope_orbits"]:
        if r["e"] == e and r["j2"] is j2 and r["sep_m"] == sep:
            return r
    raise KeyError((e, j2, sep))


def test_baseline_is_not_the_error_being_measured(doc):
    # step-halving self-check: the integrator's own error is orders below
    # every CW error this benchmark reports
    chk = doc["setup"]["baseline_self_check_max_pos_diff_m"]
    assert chk < 1e-3
    assert chk < 0.01 * min(r["err_max_m"] for r in doc["cases"] if r["err_max_m"] > 0.1)


def test_linearisation_scales_linearly_with_separation(doc):
    # theory: error ~ C * rho * (rho/R), so error/rho grows linearly in rho.
    # e=0, J2 off isolates the pure linearisation error.
    errs = [case(doc, 0.0, False, sep, 1.0)["err_max_m"] / sep
            for sep in (100.0, 1000.0, 10000.0)]
    assert errs[0] < errs[1] < errs[2]
    assert 0.5 * errs[0] < errs[1] / 10 < 2 * errs[0]   # x10 sep -> x10 rel err
    assert 0.5 * errs[1] < errs[2] / 10 < 2 * errs[1]


def test_error_grows_superlinearly_with_duration(doc):
    errs = [case(doc, 0.0, False, 100.0, o)["err_max_m"]
            for o in (1.0, 2.0, 5.0)]
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] / errs[0] > 10  # ~t^2 growth: 5x duration -> >10x error


def test_eccentricity_breaks_cw_before_anything_else_does(doc):
    # 100 m, 1 orbit, J2 on: e = 0.01 is already worse than 1 orbit of
    # circular error by two orders of magnitude, and worse at every e above
    e0 = case(doc, 0.0, True, 100.0, 1.0)["err_max_m"]
    e01 = case(doc, 0.01, True, 100.0, 1.0)["err_max_m"]
    assert e01 > 10 * e0
    for e in (0.05, 0.1, 0.2):
        assert case(doc, e, True, 100.0, 1.0)["err_max_m"] > e01 * 0.5


def test_envelope_matches_the_recorded_data(doc):
    # 25 % cells for the circular chief: 2 orbits <=100 m, 1 orbit at 1 km,
    # half orbit at 10 km -- and nothing beyond a quarter orbit at e >= 0.01
    assert envelope(doc, 0.0, True, 100.0)["0.25"] == 2.0
    assert envelope(doc, 0.0, True, 1000.0)["0.25"] == 1.0
    assert envelope(doc, 0.0, True, 10000.0)["0.25"] == 0.5
    assert envelope(doc, 0.0, True, 100000.0)["0.25"] == 0.25
    for e in (0.01, 0.05, 0.1, 0.2):
        for sep in (10.0, 100.0, 1000.0):
            assert envelope(doc, e, True, sep)["0.25"] <= 0.25


def test_planner_claim_within_the_envelope(doc):
    # the acceptance claim: circular chief, <= 1 km, <= 1 orbit -> CW error
    # stays under 25 % of the separation even with J2 present
    for sep in (10.0, 100.0, 1000.0):
        assert case(doc, 0.0, True, sep, 1.0)["err_rel"] < 0.25
