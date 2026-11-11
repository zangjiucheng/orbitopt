"""Gate B (FL-006): `orbitopt flight simulate` -- pipeline, verdict, re-fly.

Fast tests reuse the FL-004 CLI stub (recorded v1.0.0 outputs replayed from
tests/fixtures/flightlab/opentsiolkovsky/synthetic_out) so the whole
simulate -> checks -> verdict pipeline runs without the real binary. The
synthetic vehicle deliberately inserts at 400 x 533 km, so the honest
Gate-B verdict is mission_constraints_passed=false with
INSERTION_APOAPSIS_OUT_OF_TOLERANCE -- these tests pin that honesty, the
determinism of two runs, the explicit reason codes of the failure variants,
and (via @slow, only where tudatpy exists) the physics re-fly.

Gate G5: the re-fly drift thresholds are asserted with their agreed values
*before* the slow test flies, and the slow test only ever checks the drift
against the same constants the pipeline used -- a red re-fly is never
whitewashed by loosening a limit.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
import yaml

from orbitopt.flightlab import checks
from orbitopt.flightlab.contract import (
    BodyTrack,
    Check,
    Event,
    FlightRun,
    Verification,
    load_target,
    read_run,
)
from orbitopt.flightlab.simulate import simulate
from orbitopt.flightlab.checks import evaluate

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def target():
    return load_target(TARGET)


def _run_id_map(run: FlightRun) -> dict:
    return {c.id: c for c in run.verification.checks}


def _hash_tree(root: Path) -> dict:
    """sha256 of every file in a run dir, keyed by relative path."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_simulate_nominal_verdict_is_honest(stub, target, tmp_path):
    run = simulate(target, VEHICLE, tmp_path / "run", timeout_s=60)
    v = run.verification
    # three independent booleans, never collapsed
    assert v.optimizer_converged is None          # no optimizer ran in E1
    assert v.physics_verified in (None, True)     # None only when tudatpy is absent
    assert v.mission_constraints_passed is False  # the tuned program overshoots apogee: honest red
    checks_by_id = _run_id_map(run)
    assert checks_by_id["insertion-apoapsis-within-tolerance"].reason_code == \
        "INSERTION_APOAPSIS_OUT_OF_TOLERANCE"
    # everything the hand-tuned program does satisfy must be green, not silent
    for cid in ("liftoff-thrust-to-weight", "propellant-reserve", "max-q-within-limit",
                "max-accel-within-limit", "insertion-above-atmosphere",
                "insertion-periapsis-above-minimum", "insertion-periapsis-within-tolerance",
                "insertion-inclination-within-tolerance"):
        assert cid in checks_by_id, cid
        assert checks_by_id[cid].passed, cid
    if v.physics_verified is None:  # tudatpy missing: re-fly not assessed, never faked
        assert any(c.reason_code == "REFLY_BACKEND_MISSING" for c in v.checks)
    # the written dir round-trips through the contract reader with the same verdict
    assert read_run(tmp_path / "run").verification == v


def test_simulate_two_runs_are_byte_identical(stub, target, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    simulate(target, VEHICLE, a, timeout_s=60)
    simulate(target, VEHICLE, b, timeout_s=60)
    ha, hb = _hash_tree(a), _hash_tree(b)
    assert ha and ha == hb


def test_underpowered_vehicle_fails_liftoff_with_reason(stub, target, tmp_path):
    doc = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    thrust = doc["stages"][0]["engine"]["thrust_vac_n"]
    doc["stages"][0]["engine"]["thrust_vac_n"] = (
        [[t, f / 2.0] for t, f in thrust] if isinstance(thrust, list) else thrust / 2.0)
    weak = tmp_path / "underpowered.yaml"
    weak.write_text(yaml.safe_dump(doc), encoding="utf-8")
    run = simulate(target, weak, tmp_path / "run", timeout_s=60)
    tw = _run_id_map(run)["liftoff-thrust-to-weight"]
    assert not tw.passed and tw.reason_code == "LIFTOFF_THRUST_TO_WEIGHT_LOW"
    assert tw.value < checks.MIN_LIFTOFF_THRUST_TO_WEIGHT
    assert run.verification.mission_constraints_passed is False


def test_propellant_reserve_shortfall_reported(stub, tmp_path):
    doc = yaml.safe_load(TARGET.read_text(encoding="utf-8"))
    doc["recovery"]["min_propellant_reserve_kg"] = 10000.0  # more than the fixture leaves
    target_path = tmp_path / "rich-reserve.yaml"
    target_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    run = simulate(load_target(target_path), VEHICLE, tmp_path / "run", timeout_s=60)
    res = _run_id_map(run)["propellant-reserve"]
    assert not res.passed and res.reason_code == "PROPELLANT_RESERVE_LOW"
    assert res.value == pytest.approx(13282.392560150622 - 8500.0, rel=1e-6)
    assert run.verification.mission_constraints_passed is False


def _clean_orbit_run(periapsis_m: float) -> FlightRun:
    """Hand-built run: one circular orbit at `periapsis_m` altitude, inclined
    51.6 deg about x, SECO event on the same body track."""
    r = checks.EARTH_RADIUS_M + periapsis_m
    v = math.sqrt(checks.EARTH_MU / r)
    inc = math.radians(51.6)
    return FlightRun(
        run_id="synthetic-clean", target_sha256="0" * 64, vehicle_sha256="0" * 64,
        git_sha="unknown", created_utc="2026-11-10T12:00:00Z",
        tracks=(BodyTrack(
            body_id="stage2", parent_body_id="stage1", frame="GCRS",
            epoch_utc="2026-11-10T12:00:00Z", provenance="illustrative",
            t_s=(620.0, 622.0), mass_kg=(9000.0, 9000.0),
            position_m=((r, 0.0, 0.0), (r, 0.0, 0.0)),
            velocity_m_s=((0.0, v * math.cos(inc), v * math.sin(inc)),
                          (0.0, v * math.cos(inc), v * math.sin(inc))),
        ),),
        events=(Event(t_s=620.0, kind="seco", body_id="stage2", data={}),),
    )


def test_evaluate_sets_mission_constraints_passed_true_on_a_clean_low_orbit(target):
    """A 400 x 400 @ 51.6 deg circular insertion state, flown on a stub-written
    run, must come back green on every insertion check (the atmosphere and
    tolerance reasons from the failing variants have to stay silent here)."""
    vdoc = evaluate(_clean_orbit_run(400000.0), target)
    assert vdoc.mission_constraints_passed is True
    assert vdoc.physics_verified is None and vdoc.optimizer_converged is None
    ids = {c.id for c in vdoc.checks}
    assert {"insertion-above-atmosphere", "insertion-periapsis-above-minimum",
            "insertion-apoapsis-within-tolerance", "insertion-apoapsis-within-tolerance",
            "insertion-inclination-within-tolerance"} <= ids


def test_periapsis_inside_atmosphere_is_a_hard_fail(target):
    """100 km periapsis is inside the atmosphere: mission_constraints_passed
    must be false with the exact reason code PERIAPSIS_IN_ATMOSPHERE (Gate G5:
    ~120 km ceiling) -- the atmosphere check stands alone."""
    assert checks.ATMOSPHERE_CEILING_M == 120000.0
    verdict = evaluate(_clean_orbit_run(100000.0), target)
    assert verdict.mission_constraints_passed is False
    assert "PERIAPSIS_IN_ATMOSPHERE" in checks.failed_reason_codes(verdict)
    atmo = next(c for c in verdict.checks if c.id == "insertion-above-atmosphere")
    assert not atmo.passed and atmo.limit == 120000.0


def test_cli_flight_simulate_writes_run_and_scene(stub, tmp_path, capsys):
    from orbitopt import cli

    out = tmp_path / "gateb-cli"
    rc = cli.main(["flight", "simulate", str(TARGET), "--vehicle", str(VEHICLE),
                   "-o", str(out), "--scene"])
    assert rc is None
    assert (out / "manifest.json").is_file()
    scene = json.loads((out / "scene.json").read_text(encoding="utf-8"))
    assert scene["schemaVersion"] == "1.0"
    assert len(scene["bodies"]) >= 2  # Earth aside, both live bodies (and debris) are attached
    printed = capsys.readouterr().out
    assert "mission_constraints_passed: FAILED" in printed
    assert "INSERTION_APOAPSIS_OUT_OF_TOLERANCE" in printed


def test_refly_drift_thresholds_pinned():
    """Gate G5: the re-fly drift limits are declared here, before anything
    flies, and are the same constants the pipeline (and the slow E2E below)
    checks against."""
    assert checks.REFLY_MAX_A_M == 100.0
    assert checks.REFLY_MAX_DE == 1e-6
    assert checks.REFLY_MAX_DI_DEG == 0.005
    assert checks.REFLY_MAX_DRAAN_DEG == 0.02


@pytest.mark.slow
def test_slow_e2e_tudatpy_refly_verifies_the_insertion_orbit(stub, target, tmp_path):
    pytest.importorskip("tudatpy")
    # Gate G5, declared before flying: this test passes only if the drift
    # stays inside the constants pinned by test_refly_drift_thresholds_pinned.
    assert checks.REFLY_MAX_A_M == 100.0 and checks.REFLY_MAX_DE == 1e-6
    run = simulate(target, VEHICLE, tmp_path / "run", timeout_s=120)
    assert run.verification.physics_verified is True
    for c in run.verification.checks:
        if c.id.startswith("orbit-refly-"):
            assert c.passed, f"{c.id}: drift {c.value} over limit {c.limit}"
    # the re-fly checks are recorded with the drifts, not just the booleans
    assert {c.id for c in run.verification.checks if c.id.startswith("orbit-refly-")} == {
        "orbit-refly-semimajor-axis", "orbit-refly-eccentricity",
        "orbit-refly-inclination", "orbit-refly-raan",
    }
