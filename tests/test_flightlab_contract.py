"""Checks for orbitopt.flightlab.contract (the Flight Lab v1 data contract):
packaged fixtures must validate, and every acceptance-criteria corruption
(missing body frame/epoch, unknown frame, negative mass, non-monotonic t_s,
event for an unknown body, apoapsis below periapsis) must be a raised,
catchable error -- never a silently-accepted run/target. Deliberately
compute-free (jsonschema + pyyaml only), matching test_mission_config.py's
convention: the contract's whole point is loading/validating without the
conda stack or any GUI import."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from orbitopt.cli import main as cli_main
from orbitopt.data_provenance import PROVENANCES
from orbitopt.flightlab.contract import (
    EVENT_KINDS,
    FLIGHT_PROVENANCE_TAGS,
    FRAMES,
    Check,
    BodyTrack,
    Event,
    FlightRun,
    Verification,
    is_valid_run,
    is_valid_target,
    load_target,
    read_run,
    target_digest,
    validate_path,
    validate_run,
    validate_target,
    write_run,
)
from orbitopt.flightlab.contract import Check

TARGET_FIXTURE = (Path(__file__).resolve().parents[1]
                  / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml")
RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"

MINIMAL_TARGET = {
    "schema_version": "1.0",
    "mission": {"id": "leo-min", "title": "minimal", "epoch_utc": "2026-01-01T00:00:00Z"},
    "launch_site": {"name": "site", "latitude_deg": 0.0, "longitude_deg": 0.0, "altitude_wgs84_m": 0.0},
    "target_orbit": {"central_body": "Earth", "periapsis_altitude_m": 400000,
                     "apoapsis_altitude_m": 400000, "inclination_deg": 51.6, "raan_deg": None},
    "vehicle": {"source": "vehicles/x.yaml", "provenance": "illustrative"},
}


def _broken_run(tmp_path: Path) -> Path:
    """A writable copy of the synthetic run fixture, for corruption tests."""
    dest = tmp_path / "run"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(RUN_FIXTURE, dest)
    return dest


def _read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------- targets


def test_minimal_target_validates():
    validate_target(MINIMAL_TARGET)  # must not raise
    assert is_valid_target(MINIMAL_TARGET)


def test_target_fixture_loads():
    target = load_target(TARGET_FIXTURE)
    assert target.mission.id == "leo-two-stage-400km-rtls"
    assert target.vehicle.provenance == "illustrative"
    assert target.target_orbit.raan_deg is None  # explicitly unconstrained, not missing
    assert set(target.simulation.enabled_phases) == {"ascent_3dof", "orbit_verify"}


def test_target_fixture_digest_is_stable_hex64():
    import yaml

    doc = yaml.safe_load(TARGET_FIXTURE.read_text(encoding="utf-8"))
    digest = target_digest(doc)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert target_digest(doc) == digest


def test_target_rejects_apoapsis_below_periapsis():
    broken = copy.deepcopy(MINIMAL_TARGET)
    broken["target_orbit"]["apoapsis_altitude_m"] = 300000
    broken["target_orbit"]["periapsis_altitude_m"] = 400000
    assert not is_valid_target(broken)
    with pytest.raises(ValueError, match="apoapsis"):
        validate_target(broken)


def test_target_rejects_unknown_schema_version_and_frame():
    broken = copy.deepcopy(MINIMAL_TARGET)
    broken["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema"):
        validate_target(broken)
    with_terminal = copy.deepcopy(MINIMAL_TARGET)
    with_terminal["terminal_state"] = {
        "frame": "J2000", "epoch_utc": "2026-01-01T00:00:00Z",
        "position_m": [0.0, 0.0, 0.0], "velocity_m_s": [0.0, 0.0, 0.0],
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_target(with_terminal)


def test_target_vocabulary_matches_contract():
    assert FRAMES == ("GCRS", "ECEF", "LAUNCH_ECI")
    assert "max_q" in EVENT_KINDS and "touchdown" in EVENT_KINDS
    assert set(FLIGHT_PROVENANCE_TAGS) == {"official", "public_estimate", "illustrative"}
    assert set(FLIGHT_PROVENANCE_TAGS) <= set(PROVENANCES)


# ------------------------------------------------------------------- runs


def test_run_fixture_reads_and_validates():
    run = read_run(RUN_FIXTURE)
    validate_run(RUN_FIXTURE)
    assert is_valid_run(RUN_FIXTURE)
    assert [t.body_id for t in run.tracks] == ["stack", "upper_stage", "booster"]
    assert run.tracks[1].parent_body_id == "stack"
    assert run.tracks[0].frame == "LAUNCH_ECI"
    assert run.tracks[0].extra_columns == {
        "q_pa": (0.0, 18000.0, 33000.0, 21000.0, 4000.0)}  # extra CSV columns preserved
    assert any(e.kind == "separation" for e in run.events)
    assert run.verification.physics_verified is False  # fixture claims nothing
    assert run.verification.optimizer_converged is True


def test_run_fixture_target_sha256_matches_target_fixture():
    import yaml

    from orbitopt.flightlab.contract import target_digest
    doc = yaml.safe_load(TARGET_FIXTURE.read_text(encoding="utf-8"))
    assert read_run(RUN_FIXTURE).target_sha256 == target_digest(doc)


def test_write_run_round_trips(tmp_path):
    run = read_run(RUN_FIXTURE)
    out = tmp_path / "written" / "run"
    write_run(out, run)
    assert read_run(out) == run  # frozen dataclasses compare by value, extras included


def test_written_run_survives_manifest_resort_and_extra(tmp_path):
    """write_run must emit a manifest that validates even after its bodies[]
    entry is built straight from a fresh BodyTrack (bounds from the CSV)."""
    run = read_run(RUN_FIXTURE)
    track = run.tracks[0]
    assert (track.t_start_s, track.t_end_s) == (track.t_s[0], track.t_s[-1])
    assert track.to_manifest_entry()["parent_body_id"] is None


def test_run_rejects_missing_frame_and_epoch(tmp_path):
    for missing in ("frame", "epoch_utc"):
        run_dir = _broken_run(tmp_path)
        manifest = _read_manifest(run_dir)
        del manifest["bodies"][0][missing]
        _write_manifest(run_dir, manifest)
        with pytest.raises(jsonschema.ValidationError):
            read_run(run_dir)
        assert not is_valid_run(run_dir)


def test_run_rejects_unknown_frame(tmp_path):
    run_dir = _broken_run(tmp_path)
    manifest = _read_manifest(run_dir)
    manifest["bodies"][0]["frame"] = "J2000"
    _write_manifest(run_dir, manifest)
    with pytest.raises(jsonschema.ValidationError):
        validate_run(run_dir)


def test_run_rejects_negative_mass(tmp_path):
    run_dir = _broken_run(tmp_path)
    csv_path = run_dir / "trajectories" / "booster.csv"
    text = csv_path.read_text(encoding="utf-8").replace("380000.0", "-380000.0")
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="mass"):
        read_run(run_dir)


def test_run_rejects_non_monotonic_time(tmp_path):
    run_dir = _broken_run(tmp_path)
    csv_path = run_dir / "trajectories" / "upper_stage.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[2], lines[3] = lines[3], lines[2]  # swap two sample rows
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        read_run(run_dir)


def test_run_rejects_unknown_body_in_events(tmp_path):
    run_dir = _broken_run(tmp_path)
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    events.append({"t_s": 100.0, "kind": "impact", "body_id": "fairing", "data": {}})
    (run_dir / "events.json").write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(ValueError, match="fairing"):
        validate_run(run_dir)


def test_run_rejects_unknown_event_kind_and_orphan_trajectory(tmp_path):
    run_dir = _broken_run(tmp_path)
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    events[0]["kind"] = "boom"
    (run_dir / "events.json").write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        read_run(run_dir)

    run_dir = _broken_run(tmp_path)
    shutil.copy(run_dir / "trajectories" / "booster.csv", run_dir / "trajectories" / "fairing.csv")
    with pytest.raises(ValueError, match="fairing"):
        read_run(run_dir)


def test_run_rejects_missing_trajectory_and_verification_nulls(tmp_path):
    run_dir = _broken_run(tmp_path)
    (run_dir / "trajectories" / "booster.csv").unlink()
    with pytest.raises(ValueError, match="booster"):
        read_run(run_dir)
    # null verification verdicts are contract-valid ("not assessed"), never a rejection
    run_dir = _broken_run(tmp_path)
    (run_dir / "verification.json").write_text(json.dumps({
        "optimizer_converged": None, "physics_verified": None,
        "mission_constraints_passed": None, "checks": [],
    }), encoding="utf-8")
    assert is_valid_run(run_dir)


# ------------------------------------------------------------------- CLI


def test_cli_flight_validate_target_fixture(capsys):
    cli_main(["flight", "validate", str(TARGET_FIXTURE)])
    assert "OK: flight target leo-two-stage-400km-rtls" in capsys.readouterr().out


def test_cli_flight_validate_run_fixture(capsys):
    cli_main(["flight", "validate", str(RUN_FIXTURE)])
    out = capsys.readouterr().out
    assert "OK: run synthetic-two-stage-0001" in out and "3 bodies" in out


def test_cli_flight_validate_reports_invalid(tmp_path, capsys):
    run_dir = _broken_run(tmp_path)
    manifest = _read_manifest(run_dir)
    manifest["bodies"][0]["frame"] = "J2000"
    _write_manifest(run_dir, manifest)
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["flight", "validate", str(run_dir)])
    assert excinfo.value.code == 1
    assert "INVALID" in capsys.readouterr().out


def test_validate_path_dispatches_by_kind(tmp_path):
    assert "flight target" in validate_path(TARGET_FIXTURE)
    assert "run synthetic-two-stage-0001" in validate_path(RUN_FIXTURE)


# ----------------------------------------------------- programmatic write


def test_write_run_from_scratch(tmp_path):
    import hashlib
    from datetime import datetime, timezone

    def track(body_id, parent, frame, ts, masses, r0):
        return BodyTrack(
            body_id=body_id, parent_body_id=parent, frame=frame,
            epoch_utc="2026-01-01T00:00:00Z", provenance="illustrative",
            t_s=tuple(float(t) for t in ts), mass_kg=tuple(float(m) for m in masses),
            position_m=tuple((r0 + i * 100.0, 0.0, 0.0) for i in range(len(ts))),
            velocity_m_s=tuple((0.0, 100.0 + i, 0.0) for i in range(len(ts))),
        )

    run = FlightRun(
        run_id="handmade-0001",
        target_sha256=hashlib.sha256(b"target").hexdigest(),
        vehicle_sha256=hashlib.sha256(b"vehicle").hexdigest(),
        git_sha="abc1234",
        created_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        engines={"synthetic-1a": "0.1"},
        assumptions=("handmade fixture",),
        tracks=(track("stack", None, "GCRS", [0, 10, 20], [100.0, 90.0, 80.0], 1000.0),
                track("pod", "stack", "GCRS", [20, 30], [10.0, 10.0], 5000.0)),
        events=(Event(t_s=0.0, kind="ignition", body_id="stack", data={}),
                Event(t_s=20.0, kind="separation", body_id="stack", data={"children": ["pod"]})),
        verification=Verification(optimizer_converged=None, physics_verified=True,
                                  mission_constraints_passed=True,
                                  checks=(Check(id="mass_positive", passed=True),)),
    )
    out = tmp_path / "handmade"
    write_run(out, run)
    assert sorted(p.name for p in (out / "trajectories").iterdir()) == ["pod.csv", "stack.csv"]
    assert read_run(out) == run
