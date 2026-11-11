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
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from orbitopt.cli import main as cli_main
from orbitopt.flightlab.contract import BodyTrack, Event, FlightRun, read_run, write_run
from orbitopt.flightlab.frames import itrs_to_gcrs_matrix, to_gcrs
from orbitopt.flightlab.scene_export import run_to_scene
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


def _epoch_dt():
    return datetime.fromisoformat(EPOCH.replace("Z", "+00:00"))


def _all_gcrs_copy_run(tmp_path):
    dest = tmp_path / "gcrs_run"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(RUN_FIXTURE, dest)
    run = read_run(dest)
    tracks = []
    for track in run.tracks:
        if track.frame != "GCRS":
            pos, vel = to_gcrs(track.position_m, track.velocity_m_s, track.frame,
                               _epoch_dt(), track.t_s)
            track = replace(track, frame="GCRS", position_m=tuple(pos.tolist()),
                            velocity_m_s=tuple(vel.tolist()))
        tracks.append(track)
    write_run(dest, replace(run, tracks=tuple(tracks)))
    return dest


def _gmst_only_basis(mjd2000_utc: float) -> list:
    d = mjd2000_utc - 0.5
    theta = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def test_base_install_flight_to_scene_default_texture_without_numpy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run_dir = _all_gcrs_copy_run(tmp_path)
    out = tmp_path / "scene.json"
    code = f"""
import builtins
BLOCK = {{'numpy', 'scipy', 'astropy'}}
real_import = builtins.__import__
def hook(name, *args, **kwargs):
    root = name.split('.', 1)[0]
    if root in BLOCK:
        raise ModuleNotFoundError(f"base-install simulation: import of {{name}} blocked")
    return real_import(name, *args, **kwargs)
builtins.__import__ = hook
import orbitopt.cli
orbitopt.cli.main(['flight', 'to-scene', {str(run_dir)!r}, '-o', {str(out)!r}])
print('ok')
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(root / "src")}, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "ok"
    assert out.is_file()
    scene = json.loads(out.read_text(encoding="utf-8"))
    assert "sceneAssumptions" in scene


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


def test_gcrs_tracks_render_verbatim_launch_eci_uses_frame_bridge():
    run = read_run(RUN_FIXTURE)
    by_id = {b["id"]: b for b in run_to_scene(run)["bodies"]}
    # GCRS tracks render as-is (m -> km, mm-rounded).
    booster = next(t for t in run.tracks if t.body_id == "booster")
    for raw, km in zip(booster.position_m, by_id["booster"]["trail"]["positions"]):
        assert km[0] == pytest.approx(raw[0] / 1000.0, abs=1e-3)
        assert km[1] == pytest.approx(raw[1] / 1000.0, abs=1e-3)
        assert km[2] == pytest.approx(raw[2] / 1000.0, abs=1e-3)
    # LAUNCH_ECI -> GCRS goes through the FL-002 frame bridge: exported km
    # positions must match frames.to_gcrs, not just a GMST-only Rz rotation.
    stack = next(t for t in run.tracks if t.body_id == "stack")
    stack_gcrs, _ = to_gcrs(stack.position_m, stack.velocity_m_s, stack.frame,
                           _epoch_dt(), stack.t_s)
    moved = False
    for raw, (kx, ky, kz), (gx, gy, gz) in zip(
            stack.position_m, by_id["stack"]["trail"]["positions"], stack_gcrs):
        assert kx == pytest.approx(gx / 1000.0, abs=1e-3)
        assert ky == pytest.approx(gy / 1000.0, abs=1e-3)
        assert kz == pytest.approx(gz / 1000.0, abs=1e-3)
        assert math.dist((0.0, 0.0, 0.0), (kx, ky, kz)) * 1000.0 == pytest.approx(
            math.dist((0.0, 0.0, 0.0), raw), rel=2e-4)
        moved = moved or math.dist((0.0, 0.0, 0.0), (kx - raw[0] / 1000.0,
                                                       ky - raw[1] / 1000.0,
                                                       kz - raw[2] / 1000.0)) > 1e-3
    assert moved


def test_launch_eci_rotation_uses_frame_bridge_not_gmst_only():
    run = read_run(RUN_FIXTURE)
    stack = next(b for b in run_to_scene(run)["bodies"] if b["id"] == "stack")
    track = next(t for t in run.tracks if t.body_id == "stack")
    assert track.frame == "LAUNCH_ECI"
    epoch_dt = _epoch_dt()
    bridge_r, _ = to_gcrs(track.position_m, track.velocity_m_s, track.frame,
                          epoch_dt, track.t_s)
    gmst_basis = _gmst_only_basis(utc_datetime_to_mjd2000(epoch_dt))
    c = gmst_basis[0][0]
    s = gmst_basis[1][0]
    gmst_moved = False
    for (x, y, z), km, (gx, gy, gz) in zip(track.position_m, stack["trail"]["positions"], bridge_r):
        assert km[0] == pytest.approx(gx / 1000.0, abs=1e-3)
        assert km[1] == pytest.approx(gy / 1000.0, abs=1e-3)
        assert km[2] == pytest.approx(gz / 1000.0, abs=1e-3)
        gmst_moved = gmst_moved or (
            abs(km[0] - (c * x - s * y) / 1000.0) > 1e-3
            or abs(km[1] - (s * x + c * y) / 1000.0) > 1e-3
            or abs(km[2] - z / 1000.0) > 1e-3
        )
    assert gmst_moved


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
    want = itrs_to_gcrs_matrix(_epoch_dt())
    assert earth["orientationBasis"] == [[pytest.approx(x) for x in row] for row in want]
    gmst_basis = _gmst_only_basis(utc_datetime_to_mjd2000(_epoch_dt()))
    assert max(abs(earth["orientationBasis"][i][j] - gmst_basis[i][j])
               for i in range(3) for j in range(3)) > 0.003
    bare = next(b for b in run_to_scene(run, earth_texture=False)["bodies"] if b["id"] == "earth")
    assert "texture" not in bare and "orientationBasis" not in bare
    validate_scene(run_to_scene(run, earth_texture=False))  # still a valid scene


def test_earth_texture_fallback_records_scene_assumption(monkeypatch):
    import orbitopt.flightlab.scene_export as scene_export

    monkeypatch.setattr(scene_export, "_earth_orientation_basis",
                        lambda epoch, mjd2000: (scene_export._gmst_only_basis(mjd2000), "fallback"))
    scene = run_to_scene(read_run(RUN_FIXTURE), earth_texture=True)
    assert scene["sceneAssumptions"] == ["fallback"]
    assert scene["bodies"][0]["orientationBasis"] == _gmst_only_basis(
        utc_datetime_to_mjd2000(_epoch_dt()))
    validate_scene(scene)


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
