"""F-07 (task #232) regressions: the two-stage mission is flyable on either
backend, both backends stay green on the shared checks, and every backend
produces the standard artefact bundle -- a valid three-body FlightResult v1
run directory, an M-01 handoff document that round-trips, a
verification-report-1.0 document, a Mission Control scene -- with an
auto-generated 3-DOF/6-DOF comparison report over the pair. The committed
F-07 benchmark JSON is checked against regression thresholds at the end.

Real RK4 physics (shortened scenarios where the test does not need the
packaged full-length mission), same budget as the F-05/F-06 tests:

    PYTHONPATH=src python -m pytest tests/test_flightlab_sixdof_mission_output.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import read_run, validate_path
from orbitopt.flightlab.handoff import HandoffState
from orbitopt.flightlab.sixdof.gravity_turn import RUN_COLUMNS_3DOF
from orbitopt.flightlab.sixdof.mission_output import (
    BACKENDS,
    BACKEND_3DOF,
    BACKEND_6DOF,
    f07_scenario,
    handoff_document,
    mission_comparison_report,
    run_mission,
    to_flight_run_mission,
    write_mission_outputs,
)
from orbitopt.scene_format import read_scene
from orbitopt.verify.report import validate_report

pytestmark = pytest.mark.slow  # real RK4 propagations

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "docs" / "benchmarks" / "fl607_mission_outputs.json"

SHORT = dict(separation_t_s=70.0, ignition_delay_s=8.0, t_end_s=150.0,
             dt_s=0.1, sample_every=10)


@pytest.fixture(scope="module")
def pair():
    """The short mission flown on both backends, with the comparison."""
    s = f07_scenario(**SHORT)
    missions = {b: run_mission(s, backend=b) for b in BACKENDS}
    return {"scenario": s, "missions": missions,
            "comparison": mission_comparison_report(missions[BACKEND_3DOF],
                                                    missions[BACKEND_6DOF])}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, pair):
    """The pair's full artefact bundle, written once for the artefact tests."""
    out = tmp_path_factory.mktemp("f07-bundle")
    paths = {b: write_mission_outputs(pair["missions"][b], out / b)
             for b in BACKENDS}
    return {"out": out, "paths": paths, **pair}


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------

def test_backend_names_are_case_insensitive_and_exhaustive():
    from orbitopt.flightlab.sixdof import mission_output as mo

    assert mo.BACKENDS == ("3DOF", "6DOF")
    assert mo.BACKEND_3DOF == "3DOF" and mo.BACKEND_6DOF == "6DOF"


def test_unknown_backend_is_rejected_by_name():
    with pytest.raises(ValueError, match="3DOF.*6DOF"):
        run_mission(f07_scenario(**SHORT), backend="8DOF")


def test_both_backends_green_on_the_shared_checks(pair):
    for backend in BACKENDS:
        checks = pair["missions"][backend].checks
        assert all(c.passed for c in checks), (backend,
                                               [c.id for c in checks if not c.passed])
        assert {c.id for c in checks} == {
            "staging-mass-closure", "staging-momentum-residual",
            "staging-handoff-continuity", "staging-scheduled-ignition",
            "staging-trajectory-divergence"}


def test_both_backends_consumed_the_identical_scenario(pair):
    assert pair["missions"][BACKEND_3DOF].scenario is pair["scenario"]
    assert pair["missions"][BACKEND_6DOF].scenario is pair["scenario"]


def test_3dof_twin_is_point_mass_and_ignites_on_schedule(pair):
    study = pair["missions"][BACKEND_3DOF].study
    for run in (study.stack, study.booster, study.upper):
        assert tuple(run.columns) == tuple(RUN_COLUMNS_3DOF)
        assert not {"q_w", "omega_x_rad_s"} & set(run.columns)
    ign = pair["scenario"].ignition.ignition_t_s
    up = study.upper.samples
    burning = up["t_s"][up["throttle"] > 0.0]
    assert burning.size > 0
    assert float(burning[0]) >= ign - 1e-6  # never before the plan
    assert float(burning[0]) <= ign + pair["scenario"].dt_s * pair["scenario"].sample_every


# --------------------------------------------------------------------------
# the artefact bundle
# --------------------------------------------------------------------------

def test_run_dirs_are_valid_three_body_flightresult_runs(bundle):
    for backend in BACKENDS:
        run_dir = bundle["paths"][backend]["run_dir"]
        summary = validate_path(run_dir)
        assert "3 bodies" in summary
        run = read_run(run_dir)
        assert [t.body_id for t in run.tracks] == ["stack", "booster", "upper"]
        assert all(t.frame == "ECEF" for t in run.tracks)
        assert next(t for t in run.tracks if t.body_id == "booster").parent_body_id == "stack"
        assert next(t for t in run.tracks if t.body_id == "upper").parent_body_id == "stack"
        assert all(c.passed for c in run.verification.checks)


def test_handoff_documents_round_trip_and_match_the_booster_state(bundle):
    for backend in BACKENDS:
        study = bundle["missions"][backend].study
        doc = json.loads(
            (bundle["paths"][backend]["handoff"]).read_text(encoding="utf-8"))
        assert doc["kind"] == "orbitopt.flightlab.handoff-state"
        assert doc["schema_version"] == "1.0"
        hand = HandoffState.from_dict(doc)
        assert hand.boundary == ("separation", "boostback")
        assert hand.frame == "ECEF"
        assert hand.mass_kg >= hand.dry_mass_kg > 0.0
        assert hand.t_s == pytest.approx(study.scenario.separation_t_s)
        assert hand.position_m == pytest.approx(
            tuple(study.booster_state.position_m), rel=0, abs=1e-6)
        assert hand.to_dict() == handoff_document(bundle["missions"][backend])


def test_verification_reports_are_report_1_and_passed(bundle):
    for backend in BACKENDS:
        report = json.loads(
            bundle["paths"][backend]["report"].read_text(encoding="utf-8"))
        validate_report(report)
        assert report["kind"] == "flightlab"
        assert report["status"]["result"] == "passed"
        assert report["inputs"]["run_id"].startswith("f07-mission")
        assert report["label"] == report["inputs"]["run_id"]
        run = read_run(bundle["paths"][backend]["run_dir"])
        assert len(run.verification.checks) == 5


def test_scenes_load_as_scene_1_and_carry_the_run_verdicts(bundle):
    for backend in BACKENDS:
        scene = read_scene(bundle["paths"][backend]["scene"])
        assert scene["schemaVersion"] == "1.0"
        assert [b["id"] for b in scene["bodies"]] == ["earth", "stack", "booster", "upper"]
        assert scene["flightRun"]["runId"].endswith(
            {"3DOF": "3dof", "6DOF": "6dof"}[backend])
        stack_body = next(b for b in scene["bodies"] if b["id"] == "stack")
        assert any(i["label"] == "Frame" and "ECEF (converted to GCRS)" == i["value"]
                   for i in stack_body["info"])
        assert "verification" in scene  # verdicts published for Mission Control


def test_to_flight_run_mission_matches_two_stage_shape_on_6dof(pair):
    run = to_flight_run_mission(pair["missions"][BACKEND_6DOF])
    assert run.run_id == "f07-mission-6dof"
    assert len(run.events) == 6  # ignition, max_q, separation x3, upper ignition
    kinds = {(e.kind, e.body_id) for e in run.events}
    assert ("separation", "booster") in kinds and ("ignition", "upper") in kinds


# --------------------------------------------------------------------------
# the comparison report
# --------------------------------------------------------------------------

def test_comparison_report_is_report_1_green_and_covers_all_bodies(bundle):
    report = bundle["comparison"]
    validate_report(report)
    assert report["kind"] == "flightlab-comparison"
    assert report["status"]["result"] == "passed"
    assert report["inputs"]["headline"] == "6dof_minus_3dof"
    assert set(report["inputs"]["bodies"]) == {"stack", "booster", "upper"}
    for doc in report["inputs"]["bodies"].values():
        assert "error" not in doc
        assert "d_altitude_m_max_abs_m" in doc["metrics"]
        assert doc["samples"] > 10


def test_comparison_checks_are_both_backends_own_checks(bundle):
    checks = bundle["comparison"]["inputs"]["checks"]
    expected = {"comparison-shared-spans"}
    expected |= {f"{b}.{c.id}" for b in BACKENDS
                 for c in bundle["missions"][b].checks}
    assert set(checks) == expected
    assert all(checks.values())


def test_comparison_orders_backends_correctly(pair):
    m3, m6 = pair["missions"][BACKEND_3DOF], pair["missions"][BACKEND_6DOF]
    with pytest.raises(ValueError, match="3DOF.*6DOF|that order"):
        mission_comparison_report(m6, m3)


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def test_cli_flight_mission_writes_bundle_and_exits_zero(tmp_path, monkeypatch, capsys):
    from orbitopt import cli
    from orbitopt.flightlab.sixdof import mission_output as mo

    monkeypatch.setattr(mo, "f07_scenario", lambda **kw: f07_scenario(**SHORT))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["flight", "mission", "--backend", "6DOF", "-o", str(tmp_path)])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "5/5 checks passed" in out
    run_dir = tmp_path / "6DOF"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "handoff.json").is_file()
    assert (run_dir / "scene.json").is_file()
    assert validate_path(run_dir)


def test_cli_rejects_unknown_backend():
    from orbitopt import cli

    with pytest.raises(SystemExit):
        cli.main(["flight", "mission", "--backend", "12DOF"])


# --------------------------------------------------------------------------
# the committed benchmark (regression thresholds)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not BENCHMARK.is_file(),
                    reason="run benchmarks/bench_fl607_mission_outputs.py first")
def test_committed_benchmark_satisfies_the_regression_thresholds():
    doc = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert doc["schema"] == "orbitopt-fl607-1"
    assert doc["comparison"]["status"] == "passed"
    assert all(doc["comparison"]["checks"].values())
    for backend in BACKENDS:
        assert all(doc["backends"][backend]["checks"].values())
        assert doc["backends"][backend]["handoff"]["frame"] == "ECEF"
    bodies = doc["comparison"]["bodies"]
    # Regression thresholds (2026-11-13 study, see docs/flightlab.md F-07):
    # the stack must separate within ~400 m of altitude between backends,
    # the booster handoff corridor within ~3.5 km, the (unguided) upper
    # stage is allowed to diverge hard -- its thrust direction is the 6-DOF
    # story -- and mass must agree to well under a kilogram everywhere.
    assert bodies["stack"]["metrics"]["d_altitude_m_max_abs_m"] < 400.0
    assert bodies["booster"]["metrics"]["d_altitude_m_max_abs_m"] < 3500.0
    assert bodies["upper"]["metrics"]["d_speed_m_s_max_abs_m_s"] > 100.0  # divergence is real
    for body in ("stack", "booster", "upper"):
        assert bodies[body]["metrics"]["d_mass_kg_max_abs_kg"] < 2.0
        assert bodies[body]["run_3dof"]["terminal"] == "t_end"
        assert bodies[body]["run_6dof"]["terminal"] == "t_end"
