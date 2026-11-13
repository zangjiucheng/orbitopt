"""V-02 cross-model comparison harness (task #241).

The gate plumbing (report shape, categories, threshold precedence, pipeline
gate verdict) runs on **fake backends** with synthetic tracks in the fast
suite -- no scipy integration. The physics pairs (3-DOF vs analytic, and
the real 6-DOF / TudatPy pairs where those stacks are installed) run as
``slow`` tests, matching the CI matrix (jsbsim opt-in, tudatpy conda-only).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint
from orbitopt.verify.cross_model import (
    DEFAULT_THRESHOLDS,
    STATUS_RAN,
    STATUS_UNAVAILABLE,
    CrossModelScenario,
    CrossModelThresholds,
    ModelTrack,
    compare_tracks,
    parse_thresholds,
    register_backend,
    run_cross_model,
)
from orbitopt.verify.report import (
    CHECK_MODEL,
    CHECK_NUMERICAL,
    RESULT_NOT_CHECKED,
    RESULT_PASSED,
    validate_report,
)

MU = 3.986004418e14


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def _circular_track(label, t, radius=6_778_137.0, dr_m=0.0, dv_m_s=0.0,
                    inclination_deg=45.0, mass_kg=None):
    """Synthetic circular-orbit track (fast fake: analytic Kepler sampling,
    no integration) with an optional radial/along-track offset."""
    from orbitopt.verify.benchmarks import propagate_kepler_analytic

    i = math.radians(inclination_deg)
    r0 = np.array([radius + dr_m, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU / radius) + dv_m_s, 0.0])
    v0[1], v0[2] = v0[1] * math.cos(i), v0[1] * math.sin(i)
    state0 = np.concatenate([r0, v0])
    states = np.array([propagate_kepler_analytic(state0, float(dt), MU) for dt in t])
    m = None if mass_kg is None else np.full(len(t), float(mass_kg))
    return ModelTrack(label, np.asarray(t, dtype=float), states, m)


@pytest.fixture()
def fake_pair(tmp_path, monkeypatch):
    """Two identical fake ascent backends (one radially offset by 10 m) plus
    a coast-only fake -- registered, then removed after the test."""
    t = np.arange(0.0, 100.0 + 1e-9, 0.5)
    t_b = np.arange(0.0, 100.0 + 1e-9, 0.4)  # different grid: force resampling
    a = _circular_track("fake_a", t, mass_kg=50_000.0)
    b = _circular_track("fake_b", t_b, dr_m=10.0, mass_kg=49_990.0)

    register_backend("fake_a", lambda s, c: (a, None))
    register_backend("fake_b", lambda s, c: (b, None))
    register_backend("fake_off", lambda s, c: (None, None), lambda: False)
    yield {"a": a, "b": b}
    from orbitopt.verify import cross_model as cm

    for name in ("fake_a", "fake_b", "fake_off"):
        cm.BACKENDS.pop(name, None)


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------

def test_scenario_defaults_are_the_spike_stage():
    s = CrossModelScenario()
    assert s.thrust_n == pytest.approx(600.0 * 320.0 * 9.80665)
    assert s.mass_at(-1.0) == pytest.approx(70_000.0)
    assert s.mass_at(50.0) == pytest.approx(40_000.0)
    assert s.mass_at(105.0) == pytest.approx(10_000.0)
    el, az = s.attitude_at(4.0)
    assert (math.degrees(el), math.degrees(az)) == pytest.approx((90.0, 90.0))
    el, az = s.attitude_at(29.0)  # mid pitch-over: halfway 90 -> 18
    assert math.degrees(el) == pytest.approx(54.0)
    assert s.program[2].elevation_deg == pytest.approx(18.0)  # keyword-built!
    assert s.epoch_utc().year == 2026


@pytest.mark.parametrize("bad", [
    {"burn_end_s": 200.0},                 # burns past MECO / propellant
    {"mdot_kg_s": 900.0},                  # mdot*burn > propellant load
    {"coast_duration_s": 0.0},
    {"dry_mass_kg": 0.0},
    {"program": (AttitudePoint(t_s=5.0, azimuth_deg=90.0, elevation_deg=90.0),
                 AttitudePoint(t_s=5.0, azimuth_deg=90.0, elevation_deg=18.0))},
])
def test_scenario_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        CrossModelScenario(**bad)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

def test_threshold_precedence_is_pair_phase_metric():
    th = CrossModelThresholds(overrides={"position_m": 1000.0, "ascent.position_m": 5.0,
                                         "fake_a_vs_fake_b.ascent.position_m": 20.0})
    assert th.resolve("fake_a_vs_fake_b", "ascent", "position_m") == 20.0
    assert th.resolve("other_pair", "ascent", "position_m") == 5.0
    assert th.resolve("other_pair", "coast", "position_m") == 1000.0
    assert th.resolve("x", "coast", "eccentricity") == DEFAULT_THRESHOLDS["coast"]["eccentricity"]


def test_threshold_parsing_accepts_none_and_rejects_junk():
    parsed = parse_thresholds({"position_m": "100", "coast.mass_kg": "none",
                               "coast.eccentricity": None})
    assert parsed["position_m"] == 100.0 and parsed["coast.mass_kg"] is None
    with pytest.raises(ValueError):
        parse_thresholds({"coast.nonsense": 1.0})
    with pytest.raises(ValueError):
        parse_thresholds({"ascent.bogus.phase": 1.0})
    with pytest.raises(ValueError):
        parse_thresholds({"position_m": 0.0})


# ---------------------------------------------------------------------------
# raw metrics
# ---------------------------------------------------------------------------

def test_compare_tracks_finds_the_injected_offset():
    t = np.linspace(0.0, 120.0, 49)
    a = _circular_track("a", t, mass_kg=50_000.0)
    b = _circular_track("b", t, dr_m=10.0, mass_kg=49_990.0)
    out = compare_tracks(a, b)
    # the offset state is a *different orbit*, so along-track divergence
    # grows on top of the injected 10 m radial gap
    assert 10.0 <= out["max_position_m"] < 10.5
    assert out["final_mass_kg"] == pytest.approx(10.0, rel=1e-9)
    assert abs(out["elements"]["inclination_deg"]) < 1e-9
    assert abs(out["elements"]["semi_major_axis_m"]) < 400.0


def test_compare_tracks_rejects_disjoint_windows():
    a = _circular_track("a", np.linspace(0.0, 10.0, 5))
    b = _circular_track("b", np.linspace(20.0, 30.0, 5))
    with pytest.raises(ValueError, match="share no time window"):
        compare_tracks(a, b)


# ---------------------------------------------------------------------------
# harness with fake backends (fast, no physics integration)
# ---------------------------------------------------------------------------

def test_fake_pair_report_is_schema_valid_and_categorised(fake_pair):
    res = run_cross_model(backends=["fake_a", "fake_b"], report_path=None)
    assert res.result == RESULT_PASSED
    validate_report(res.report)  # schema + digest
    assert res.report["kind"] == "cross-model"
    assert len(res.pair_metrics) == 1
    p = res.pair_metrics[0]
    assert p["pair"] == "fake_a_vs_fake_b" and p["phase"] == "ascent"
    assert p["category"] == CHECK_MODEL  # two different models
    assert 10.0 <= p["metrics"]["max_position_m"] < 10.5
    assert "mass_kg" in p["checks"]  # both sides track mass -> gated
    cats = res.report["status"]["check_categories"]
    assert cats["residual.fake_a_vs_fake_b.ascent"] == CHECK_MODEL


def test_threshold_override_moves_the_gate_and_back(fake_pair):
    ok = run_cross_model(backends=["fake_a", "fake_b"],
                         thresholds={"fake_a_vs_fake_b.ascent.position_m": 50.0})
    assert ok.result == RESULT_PASSED
    strict = run_cross_model(backends=["fake_a", "fake_b"],
                             thresholds={"fake_a_vs_fake_b.ascent.position_m": 1.0})
    assert strict.result == "failed"
    r = strict.report["residuals"][0]
    assert r["within_tolerance"] is False
    assert r["tolerance"]["position_m"] == 1.0
    # the failed residual keeps the full metric set (acceptance criteria)
    assert r["mass_residual_kg"] == pytest.approx(10.0, rel=1e-6)
    assert set(r["elements_residual"]) >= {"semi_major_axis_m", "eccentricity",
                                           "inclination_deg", "raan_deg",
                                           "argument_of_perigee_deg"}


def test_recorded_without_gating(fake_pair):
    res = run_cross_model(backends=["fake_a", "fake_b"],
                          thresholds={"mass_kg": "none", "position_m": "none"})
    p = res.pair_metrics[0]
    assert "mass_kg" not in p["tolerance"] and "position_m" not in p["tolerance"]
    assert p["recorded_only"]["mass_kg"] == pytest.approx(10.0, rel=1e-6)
    assert res.report["status"]["categories"][CHECK_MODEL]["result"] == RESULT_PASSED


def test_unavailable_and_failed_backends_are_recorded_not_faked(fake_pair):
    res = run_cross_model(backends=["fake_off", "fake_a"])
    by = {o.label: o for o in res.backends}
    assert by["fake_off"].status == STATUS_UNAVAILABLE
    assert     res.report["status"]["result"] == RESULT_NOT_CHECKED  # single backend: no pair

    def boom(scenario, context):
        raise RuntimeError("boom")

    register_backend("fake_boom", boom)
    try:
        res2 = run_cross_model(backends=["fake_a", "fake_b", "fake_boom"])
        by2 = {o.label: o for o in res2.backends}
        assert by2["fake_boom"].status == "failed" and "boom" in by2["fake_boom"].detail
        assert len(res2.pair_metrics) == 1  # the rest still compared
        assert res2.result == RESULT_PASSED
    finally:
        from orbitopt.verify.cross_model import BACKENDS

        BACKENDS.pop("fake_boom", None)


def test_pipeline_gate_consumes_the_report(fake_pair):
    from orbitopt.candidate_selection import VERDICT_ACCEPTED, VERDICT_DEGRADED, VerificationGate

    good = run_cross_model(backends=["fake_a", "fake_b"])
    gate = VerificationGate()
    gate.register(0, scenario="cross-model")
    c = gate.evaluate(0, [good.report])
    assert c["verdict"] == VERDICT_ACCEPTED

    bad = run_cross_model(backends=["fake_a", "fake_b"], thresholds={"ascent.position_m": 1.0})
    gate2 = VerificationGate()
    gate2.register(1)
    c2 = gate2.evaluate(1, [bad.report])
    assert c2["verdict"] == VERDICT_DEGRADED


def test_cli_reports_and_exit_codes(fake_pair, tmp_path, capsys):
    from orbitopt.verify.cross_model import main

    out = tmp_path / "report.json"
    assert main(["--backends", "fake_a,fake_b", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "fake_a_vs_fake_b (ascent)" in text
    assert out.read_text().strip() != ""
    assert main(["--backends", "fake_a,fake_b", "--threshold", "ascent.position_m=1e-9"]) == 1
    capsys.readouterr()  # drain, then --json output is exactly the report doc
    assert main(["--backends", "fake_a,fake_b", "--json"]) == 0
    payload = capsys.readouterr().out
    import json

    doc = json.loads(payload)
    assert doc["status"]["result"] == RESULT_PASSED
    validate_report(doc)


# ---------------------------------------------------------------------------
# real physics pairs (slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_dof_run():
    from orbitopt.verify.cross_model import run_cross_model as run

    return run(backends=["three_dof", "two_body_analytic"])


@pytest.mark.slow
def test_default_pair_passes_and_matches_the_fl403_reference(three_dof_run):
    res = three_dof_run
    assert all(o.status == STATUS_RAN for o in res.backends)
    assert res.result == RESULT_PASSED
    validate_report(res.report)
    p = res.pair_metrics[0]
    assert p["phase"] == "coast" and p["category"] == CHECK_NUMERICAL
    assert "coast.position_m" not in p["recorded_only"]  # point mass = same model -> gated
    # 3-DOF MECO state vs the FL-403 OT fixture (a=4416780.9 m, e=0.5383,
    # alt 90153 m, speed 5297 m/s Earth-relative)
    from orbitopt.flightlab.checks import orbital_elements

    asc = next(o for o in res.backends if o.label == "three_dof").ascent
    assert asc is not None
    import orbitopt.verify.cross_model as cm

    meco = asc.states_gcrs[-1]
    el = orbital_elements(meco[:3], meco[3:], 3.986004418e14)  # elements: inertial state
    assert el["semi_major_axis_m"] == pytest.approx(4_416_780.9, rel=3e-4)
    assert el["eccentricity"] == pytest.approx(0.5383, abs=5e-4)
    r_e, v_e = cm._from_gcrs_pair(meco[None, :3], meco[None, 3:], np.array([asc.t_end]),
                                  res.scenario.epoch_utc())
    assert float(np.linalg.norm(v_e[0])) == pytest.approx(5297.0, abs=20.0)  # Earth-relative


@pytest.mark.slow
def test_threshold_override_fails_the_real_coast_pair(three_dof_run):
    from orbitopt.verify.cross_model import run_cross_model as run

    res = run(backends=["three_dof", "two_body_analytic"],
              thresholds={"coast.position_m": 1e-6})
    assert res.result == "failed"
    res2 = run(backends=["three_dof", "two_body_analytic"], thresholds={"coast.position_m": None})
    assert "position_m" in res2.pair_metrics[0]["recorded_only"]


@pytest.mark.slow
def test_report_written_via_env_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOPT_VERIFICATION_REPORT_DIR", str(tmp_path))
    from orbitopt.verify.cross_model import run_cross_model as run

    res = run(backends=["three_dof", "two_body_analytic"])
    assert res.report_path is not None and str(tmp_path) in str(res.report_path)
    import json

    doc = json.loads(res.report_path.read_text())
    assert doc["digest"] == res.report["digest"]


@pytest.mark.slow
def test_six_dof_pair_agrees_with_three_dof():
    pytest.importorskip("jsbsim")
    res = run_cross_model(backends=["three_dof", "six_dof"])
    by = {o.label: o for o in res.backends}
    assert by["six_dof"].status == STATUS_RAN, by["six_dof"].detail
    p = next(p for p in res.pair_metrics if p["phase"] == "ascent")
    assert p["category"] == CHECK_MODEL
    # FL-403 measured 51 m / 2 m/s MECO deviation; defaults are far looser
    assert p["metrics"]["max_position_m"] < 500.0
    assert "mass_kg" in p["checks"]
    assert res.result == RESULT_PASSED


@pytest.mark.slow
def test_tudatpy_coast_pair_agrees_with_three_dof():
    pytest.importorskip("tudatpy")
    res = run_cross_model(backends=["three_dof", "tudatpy"])
    by = {o.label: o for o in res.backends}
    assert by["tudatpy"].status == STATUS_RAN, by["tudatpy"].detail
    p = next(p for p in res.pair_metrics if p["phase"] == "coast")
    assert p["category"] == CHECK_MODEL
    assert res.result == RESULT_PASSED
