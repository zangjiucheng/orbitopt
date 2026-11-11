"""Checks for orbitopt.flightlab.scene_export (FL-005, Gate A): the synthetic
run fixture must import as a scene-1.0 document the viewer can open (Earth +
one trail per body, lifespan bounds, event markers, run metadata), and every
un-renderable run (frame scene export cannot convert to GCRS, missing or
unparseable epoch, disagreeing epochs, non-monotonic time, mass
discontinuity) must raise before anything is produced -- the CLI must write
no file for a rejected run. Compute-free, GUI-free, like the contract tests."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from orbitopt.cli import main as cli_main
from orbitopt.flightlab.contract import BodyTrack, Event, FlightRun, read_run, write_run
from orbitopt.flightlab.scene_export import _gmst_deg, run_to_scene
from orbitopt.scene_format import read_scene, validate_scene
from orbitopt.timescales import utc_datetime_to_mjd2000

RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"
EPOCH = "2026-11-10T12:00:00Z"


def _copy_run(tmp_path):
    dest = tmp_path / "run"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(RUN_FIXTURE, dest)
    return dest


def _track(body_id, frame="GCRS", epoch=EPOCH, t=((0.0, 10.0)), mass=((1000.0, 900.0)),
           pos=((0.0, 0.0, 6.37e6), (0.0, 1e3, 6.4e6)), vel=((0.0, 0.0, 0.0), (0.0, 1e3, 1e3))):
    return BodyTrack(body_id=body_id, parent_body_id=None, frame=frame, epoch_utc=epoch,
                     provenance="illustrative", t_s=tuple(t), mass_kg=tuple(mass),
                     position_m=tuple(tuple(p) for p in pos),
                     velocity_m_s=tuple(tuple(v) for v in vel))


def _run(tracks, events=()):
    return FlightRun(run_id="scene-test", target_sha256="0" * 64, vehicle_sha256="0" * 64,
                     git_sha="x", created_utc=EPOCH, tracks=tuple(tracks), events=tuple(events))


@pytest.fixture(scope="module")
def fixture_scene():
    return run_to_scene(read_run(RUN_FIXTURE))


# ---------------------------------------------------------------- happy path


def test_fixture_scene_validates(fixture_scene):
    validate_scene(fixture_scene)  # must not raise (run_to_scene also self-checks)
    assert fixture_scene["schemaVersion"] == "1.0"
    assert fixture_scene["distanceUnit"] == "km"
    assert [b["id"] for b in fixture_scene["bodies"]] == ["earth", "stack", "upper_stage", "booster"]
    timeline = fixture_scene["timeline"]
    assert timeline["unitLabel"] == "seconds"
    assert (timeline["min"], timeline["max"]) == (0.0, 540.0)  # last event/trail end
    assert timeline["referenceEpochEt"] > 8.0e8  # 2026 in TDB seconds past J2000


def test_bodies_appear_and_disappear(fixture_scene):
    run = read_run(RUN_FIXTURE)
    by_id = {b["id"]: b for b in fixture_scene["bodies"]}
    for track in run.tracks:
        body = by_id[track.body_id]
        assert body["tStart"] == pytest.approx(track.t_start_s)
        assert body["tEnd"] == pytest.approx(track.t_end_s)
    # Booster landed and died at 420 s; upper stage flies to SECO at 540 s:
    # the two-body overlap window (160..420) and the divergence after it are
    # exactly what the viewer uses to gate visibility.
    assert by_id["booster"]["tEnd"] == 420.0
    assert by_id["upper_stage"]["tEnd"] == 540.0


def test_gcrs_tracks_render_verbatim_launch_eci_rotates():
    run = read_run(RUN_FIXTURE)
    by_id = {b["id"]: b for b in run_to_scene(run)["bodies"]}
    # GCRS tracks render as-is (m -> km, mm-rounded).
    booster = next(t for t in run.tracks if t.body_id == "booster")
    for raw, km in zip(booster.position_m, by_id["booster"]["trail"]["positions"]):
        assert km[0] == pytest.approx(raw[0] / 1000.0, abs=1e-3)
        assert km[1] == pytest.approx(raw[1] / 1000.0, abs=1e-3)
        assert km[2] == pytest.approx(raw[2] / 1000.0, abs=1e-3)
    # LAUNCH_ECI -> GCRS is a pure rotation: |r| and z preserved (Rz only),
    # and a sample off the z-axis moves -- so it really was rotated.
    stack = next(t for t in run.tracks if t.body_id == "stack")
    moved = False
    for (x, y, z), (kx, ky, kz) in zip(stack.position_m, by_id["stack"]["trail"]["positions"]):
        assert math.dist((0.0, 0.0, 0.0), (kx, ky, kz)) * 1000.0 == pytest.approx(
            math.dist((0.0, 0.0, 0.0), (x, y, z)), rel=2e-4)
        assert kz == pytest.approx(z / 1000.0, abs=1e-3)
        moved = moved or (kx, ky) != (pytest.approx(x / 1000.0, abs=1e-3),
                                      pytest.approx(y / 1000.0, abs=1e-3))
    assert moved


def test_launch_eci_rotation_matches_expectation():
    run = read_run(RUN_FIXTURE)
    stack = next(b for b in run_to_scene(run)["bodies"] if b["id"] == "stack")
    track = next(t for t in run.tracks if t.body_id == "stack")
    assert track.frame == "LAUNCH_ECI"
    epoch_dt = datetime.fromisoformat(EPOCH.replace("Z", "+00:00"))
    theta = math.radians(_gmst_deg(utc_datetime_to_mjd2000(epoch_dt)))
    c, s = math.cos(theta), math.sin(theta)
    for (x, y, z), km in zip(track.position_m, stack["trail"]["positions"]):
        assert km[0] == pytest.approx((c * x - s * y) / 1000.0, abs=1e-3)
        assert km[1] == pytest.approx((s * x + c * y) / 1000.0, abs=1e-3)
        assert km[2] == pytest.approx(z / 1000.0, abs=1e-3)


def test_events_become_timeline_markers(fixture_scene):
    events = fixture_scene["timeline"]["events"]
    assert len(events) == 8
    labels = {e["time"]: e["label"] for e in events}
    assert labels[68.0] == "Max-Q" and labels[0.0] == "Ignition" and labels[420.0] == "Touchdown"
    for e in events:
        assert e["bodyId"] in {"stack", "booster", "upper_stage"}
        assert e["time"] >= 0.0


def test_run_metadata_is_optional_block(fixture_scene):
    meta = fixture_scene["flightRun"]
    assert meta == {"runId": "synthetic-two-stage-0001", "optimizerConverged": True,
                    "physicsVerified": False, "missionConstraintsPassed": False}


def test_earth_texture_toggle():
    run = read_run(RUN_FIXTURE)
    earth = next(b for b in run_to_scene(run, earth_texture=True)["bodies"] if b["id"] == "earth")
    assert earth["texture"] == "earth_daymap.jpg" and earth["radius"] == pytest.approx(6378.137, abs=1e-3)
    assert len(earth["orientationBasis"]) == 3  # Rz(GMST at the mission epoch)
    bare = next(b for b in run_to_scene(run, earth_texture=False)["bodies"] if b["id"] == "earth")
    assert "texture" not in bare and "orientationBasis" not in bare
    validate_scene(run_to_scene(run, earth_texture=False))  # still a valid scene


# ---------------------------------------------------------------- rejections


def test_unknown_frame_rejected():
    with pytest.raises(ValueError, match="not convertible to GCRS"):
        run_to_scene(_run([_track("sc", frame="J2000")]))


def test_ecef_frame_rejected_as_not_convertible():
    # ECEF is a legal *contract* frame, but its velocity needs the FL-002
    # frame bridge's rotating-frame term -- scene export must refuse it.
    with pytest.raises(ValueError, match="frame bridge"):
        run_to_scene(_run([_track("sc", frame="ECEF")]))


def test_missing_epoch_rejected():
    with pytest.raises(ValueError, match="epoch_utc is missing"):
        run_to_scene(_run([_track("sc", epoch="")]))


def test_unparseable_epoch_rejected():
    with pytest.raises(ValueError, match="epoch_utc"):
        run_to_scene(_run([_track("sc", epoch="last tuesday")]))


def test_disagreeing_epochs_rejected():
    with pytest.raises(ValueError, match="one mission epoch"):
        run_to_scene(_run([_track("a"), _track("b", epoch="2026-11-11T12:00:00Z")]))


def test_non_monotonic_time_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        run_to_scene(_run([_track(
            "sc", t=(0.0, 5.0, 3.0), mass=(1e3, 9e2, 8e2),
            pos=((0.0, 0.0, 6.37e6), (0.0, 1e3, 6.4e6), (0.0, 2e3, 6.5e6)),
            vel=((0.0, 0.0, 0.0), (0.0, 1e3, 1e3), (0.0, 1e3, 1e3)))]))


def test_mass_discontinuity_rejected():
    # A rocket only sheds mass inside one track; a rise is a bookkeeping bug.
    with pytest.raises(ValueError, match="mass discontinuity"):
        run_to_scene(_run([_track("sc", mass=(1000.0, 1100.0))]))


def test_event_for_unknown_body_rejected():
    with pytest.raises(ValueError, match="not a body"):
        run_to_scene(_run([_track("sc")], events=[Event(t_s=1.0, kind="meco", body_id="ghost")]))


def test_missing_epoch_in_manifest_rejected_via_read(tmp_path):
    run_dir = _copy_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["bodies"][0].pop("epoch_utc")
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="epoch_utc"):
        read_run(run_dir)


def test_non_monotonic_csv_rejected_via_read(tmp_path):
    run_dir = _copy_run(tmp_path)
    csv_path = run_dir / "trajectories" / "stack.csv"
    lines = csv_path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]  # t=40 before t=0
    csv_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(Exception, match="strictly increasing"):
        read_run(run_dir)


# ---------------------------------------------------------------------- CLI


def test_cli_to_scene_writes_openable_scene(tmp_path, capsys):
    out = tmp_path / "scene.json"
    cli_main(["flight", "to-scene", str(RUN_FIXTURE), "-o", str(out)])
    assert out.exists()
    scene = read_scene(out)  # scene-1.0 round-trip, the way `orbitopt view` reads it
    assert scene["flightRun"]["runId"] == "synthetic-two-stage-0001"
    assert "open with: orbitopt view" in capsys.readouterr().out


def test_cli_rejects_ecef_run_and_writes_nothing(tmp_path, capsys):
    run_dir = _copy_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["bodies"][1]["frame"] = "ECEF"  # schema-legal, not renderable
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "scene.json"
    with pytest.raises(SystemExit):
        cli_main(["flight", "to-scene", str(run_dir), "-o", str(out)])
    assert not out.exists()
    assert "REJECTED" in capsys.readouterr().out


def test_cli_rejects_mass_discontinuity_run(tmp_path):
    run = read_run(RUN_FIXTURE)
    corrupted = [t if t.body_id != "booster" else BodyTrack(
        **{**t.__dict__, "mass_kg": (380000.0, 400000.0, 220000.0, 200000.0)}) for t in run.tracks]
    run_dir = tmp_path / "grown-mass"
    write_run(run_dir, FlightRun(**{**run.__dict__, "tracks": tuple(corrupted)}))
    with pytest.raises(SystemExit):
        cli_main(["flight", "to-scene", str(run_dir), "-o", str(tmp_path / "s.json")])
