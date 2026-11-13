"""Benchmark scenarios gate (task #245, epic P1 V-06).

Every benchmark must PASS inside its declared tolerance, on every push, in
the fast CI suite -- that is the regression gate the acceptance criteria ask
for. The gate is two-sided: the measurements must match the references, and
the *frozen reference constants* must keep matching their public/analytic
definitions, so drift of the Earth constants or of the reference numbers
themselves cannot silently move the goalposts.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.entry import J2_EARTH, MU_EARTH_M3_S2, R_EARTH_M
from orbitopt.verify import benchmarks as bm
from orbitopt.verify.report import validate_report

EXPECTED_BENCHMARKS = ["kepler_two_body", "j2_secular_drift", "starship_booster_catch"]


def _assert_declared(result):
    """Acceptance: every benchmark scenario has expected values and tolerances."""
    assert result.checks, f"{result.key} produced no checks"
    assert result.reference, f"{result.key} has no reference provenance"
    for c in result.checks:
        assert isinstance(c["expected"], float)
        assert isinstance(c["tolerance"], float) and c["tolerance"] > 0.0
        assert isinstance(c["actual"], float) and math.isfinite(c["actual"])
        assert c["passed"] == (c["abs_error"] <= c["tolerance"])
        assert c["passed"], (f"{result.key}.{c['name']}: actual {c['actual']:.6g} "
                             f"vs expected {c['expected']:.6g} +/- {c['tolerance']:g} {c['unit']}")


def test_all_three_reference_benchmarks_pass():
    for key in EXPECTED_BENCHMARKS:
        assert key in bm.BENCHMARKS
        _assert_declared(bm.run_benchmark(key))


def test_unknown_benchmark_key_is_refused():
    with pytest.raises(ValueError):
        bm.run_benchmark("nope")


# ---------------------------------------------------------------- constants
def test_frozen_kepler_state_is_the_analytic_solution():
    state0 = bm.coe_to_ic(**bm.KEPLER_COE)
    analytic = bm.propagate_kepler_analytic(state0, bm.KEPLER_DURATION_S)
    frozen = np.asarray(bm.KEPLER_EXPECTED_STATE)
    assert np.linalg.norm(analytic[:3] - frozen[:3]) < 0.1          # m
    assert np.linalg.norm(analytic[3:] - frozen[3:]) < 1e-3         # m/s


def test_elements_round_trip():
    coe = dict(bm.KEPLER_COE)
    state0 = bm.coe_to_ic(**coe)
    back = bm.ic_to_coe(state0)
    assert back["a"] == pytest.approx(coe["a"], rel=1e-10)
    assert back["e"] == pytest.approx(coe["e"], rel=1e-9)
    assert back["i_deg"] == pytest.approx(coe["i_deg"], abs=1e-8)
    assert back["raan_deg"] == pytest.approx(coe["raan_deg"], abs=1e-8)
    assert back["argp_deg"] == pytest.approx(coe["argp_deg"], abs=1e-8)
    assert back["M_deg"] == pytest.approx(coe["M0_deg"], abs=1e-8)


def test_frozen_j2_constants_match_their_analytic_definitions():
    # Frozen secular node drift of the critical-inclination scenario.
    n = math.sqrt(MU_EARTH_M3_S2 / bm.CRIT_SEMI_MAJOR_AXIS_M ** 3)
    p = bm.CRIT_SEMI_MAJOR_AXIS_M * (1.0 - bm.CRIT_ECCENTRICITY ** 2)
    i = math.radians(bm.CRITICAL_INCLINATION_DEG)
    formula = -1.5 * J2_EARTH * n * (R_EARTH_M / p) ** 2 * math.cos(i) * 86400.0 * 180.0 / math.pi
    assert formula == pytest.approx(bm.CRIT_EXPECTED_RAAN_DRIFT_DEG_PER_DAY, abs=1e-4)
    # The critical-inclination constant really is the frozen-perigee angle.
    assert 5.0 * math.cos(i) ** 2 - 1.0 == pytest.approx(0.0, abs=1e-6)
    # The sun-sync inclination solves the secular formula for the published rate.
    a = R_EARTH_M + bm.SSO_ALTITUDE_M
    p = a * (1.0 - bm.SSO_ECCENTRICITY ** 2)
    n = math.sqrt(MU_EARTH_M3_S2 / a ** 3)
    rate = -1.5 * J2_EARTH * n * (R_EARTH_M / p) ** 2 \
        * math.cos(math.radians(bm._sso_inclination_deg(a, bm.SSO_ECCENTRICITY)))
    published = math.radians(bm.SUNSYNC_RAAN_RATE_DEG_PER_DAY) / 86400.0
    assert rate == pytest.approx(published, rel=1e-10)
    assert bm.SUNSYNC_RAAN_RATE_DEG_PER_DAY == pytest.approx(360.0 / 365.2422, rel=1e-12)


# ------------------------------------------------------------------ report
def test_benchmark_report_is_a_verification_report():
    report = bm.benchmark_report()
    validate_report(report)                      # schema + digest
    assert report["kind"] == "benchmark_scenarios"
    assert report["status"]["result"] == "passed"
    assert sum(report["status"]["checks"].values()) == len(report["status"]["checks"]) == 8
    assert len(report["targets"]["metrics"]) == 8
    assert {r["key"] for r in report["inputs"]["benchmarks"]} == set(EXPECTED_BENCHMARKS)


def test_benchmark_report_is_deterministic():
    a = bm.benchmark_report()
    b = bm.benchmark_report([bm.run_benchmark(k) for k in EXPECTED_BENCHMARKS])
    assert a["digest"] == b["digest"]


def test_every_benchmark_check_is_categorised():
    """V-04 (task #243) refuses an uncategorised check; V-06's scenarios were
    written against the pre-V-04 report and failed to build after the merge.
    Each scenario declares what kind of evidence it is, and the roll-up shows
    all three kinds backed by a passing check."""
    from orbitopt.verify.report import CHECK_HISTORICAL, CHECK_MODEL, CHECK_NUMERICAL

    assert set(bm.BENCHMARK_CATEGORIES) == set(bm.BENCHMARKS)
    status = bm.benchmark_report()["status"]
    cats = status["check_categories"]
    assert set(cats) == set(status["checks"])
    assert {c for k, c in cats.items() if k.startswith("kepler_two_body.")} == {CHECK_NUMERICAL}
    assert {c for k, c in cats.items() if k.startswith("j2_secular_drift.")} == {CHECK_MODEL}
    assert {c for k, c in cats.items()
            if k.startswith("starship_booster_catch.")} == {CHECK_HISTORICAL}
    rollup = status["categories"]
    assert {c: (rollup[c]["result"], rollup[c]["checks_total"])
            for c in (CHECK_NUMERICAL, CHECK_MODEL, CHECK_HISTORICAL)} == {
        CHECK_NUMERICAL: ("passed", 3), CHECK_MODEL: ("passed", 3),
        CHECK_HISTORICAL: ("passed", 2)}


def test_cli_main_exits_zero_when_all_pass():
    assert bm.main([]) == 0
