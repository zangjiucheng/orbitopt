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


def test_base_install_flight_to_scene_launch_eci_fails_cleanly(tmp_path):
    """A LAUNCH_ECI track needs the numpy/astropy frame bridge: on a base
    install `flight to-scene` must exit 1 with a clear message, no traceback
    and no output file."""
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "scene.json"
    code = f"""
import builtins
BLOCK = {{'numpy', 'scipy', 'astropy'}}
real_import = builtins.__import__
def hook(name, *args, **kwargs):
    if name.split('.', 1)[0] in BLOCK:
        raise ModuleNotFoundError(f"base-install simulation: import of {{name}} blocked")
    return real_import(name, *args, **kwargs)
builtins.__import__ = hook
import orbitopt.cli
orbitopt.cli.main(['flight', 'to-scene', {str(RUN_FIXTURE)!r}, '-o', {str(out)!r}])
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(root / "src")}, timeout=20)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "frame bridge" in proc.stdout
    assert not out.exists()


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


def test_ecef_frame_converts_through_the_frame_bridge():
    # F-07 (task #232): ECEF is renderable -- the FL-002 frame bridge gained
    # the rotating-frame velocity term, so the same conversion the contract
    # pipeline uses now carries ECEF tracks into the scene.
    scene = run_to_scene(_run([_track("sc", frame="ECEF")]))
    validate_scene(scene)
    body = next(b for b in scene["bodies"] if b["id"] == "sc")
    assert any(i["label"] == "Frame" and "ECEF (converted to GCRS)" == i["value"]
               for i in body["info"])
    trail = body["trail"]["positions"]
    assert all(len(p) == 3 and all(math.isfinite(c) for c in p) for p in trail)


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


def test_cli_renders_ecef_tracks(tmp_path, capsys):
    # F-07: an ECEF-mutated fixture run is schema-legal AND renderable now;
    # the scene says so in the body's frame label. (The synthetic fixture's
    # ECEF numbers are not a physical pair with its LAUNCH_ECI parent, so a
    # separation gate may fail -- that rides along in the scene's checks, it
    # is not an export refusal.)
    run_dir = _copy_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["bodies"][1]["frame"] = "ECEF"  # schema-legal, now renderable
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "scene.json"
    cli_main(["flight", "to-scene", str(run_dir), "-o", str(out)])
    scene = read_scene(out)
    body = next(b for b in scene["bodies"] if b["id"] == "upper_stage")
    assert any(i["label"] == "Frame" and "converted to GCRS" in i["value"]
               for i in body["info"])
    assert "open with: orbitopt view" in capsys.readouterr().out


def test_cli_rejects_mass_discontinuity_run(tmp_path):
    run = read_run(RUN_FIXTURE)
    corrupted = [t if t.body_id != "booster" else BodyTrack(
        **{**t.__dict__, "mass_kg": (380000.0, 400000.0, 220000.0, 200000.0)}) for t in run.tracks]
    run_dir = tmp_path / "grown-mass"
    write_run(run_dir, FlightRun(**{**run.__dict__, "tracks": tuple(corrupted)}))
    with pytest.raises(SystemExit):
        cli_main(["flight", "to-scene", str(run_dir), "-o", str(tmp_path / "s.json")])


# ------------------------------------------------- M-06: phases + handoffs


def _phase_labels(scene):
    phases = scene["timeline"]["phases"]
    assert phases, "scene must carry a phase row"
    assert phases[0]["start"] == pytest.approx(scene["timeline"]["min"])
    assert phases[-1]["end"] == pytest.approx(scene["timeline"]["max"])
    for a, b in zip(phases, phases[1:]):
        assert a["end"] > a["start"]
        assert b["start"] == pytest.approx(a["end"]), "phase row must be continuous"
    return [p["label"] for p in phases]


def test_fixture_phases_tile_the_timeline_continuously(fixture_scene):
    assert _phase_labels(fixture_scene) == [
        "On the pad", "Boost", "Staging", "Upper-stage burn"]


def test_staging_and_seco_event_notes_carry_handoff_verdicts(fixture_scene):
    notes = {e["label"]: e["note"] for e in fixture_scene["timeline"]["events"]}
    assert "handoff" in notes["Separation"] and "FAILED" in notes["Separation"]
    assert "handoff FAILED" in notes["SECO"]


def test_separation_handoffs_are_m01_gated(fixture_scene):
    by_id = {h["id"]: h for h in fixture_scene["handoffs"]}
    gate = {"handoff.position_continuity", "handoff.velocity_continuity",
            "handoff.epoch_monotonic", "handoff.mass_conservation"}
    up = by_id["separation:upper_stage"]
    assert {c["id"] for c in up["checks"]} == gate
    assert up["from"]["frame"] == "GCRS" and up["to"]["frame"] == "GCRS"
    # the fabricated fixture ends the LAUNCH_ECI stack at the numbers its GCRS
    # child starts with -- through the frame bridge those are different states.
    assert up["status"] == "failed" and up["graded"] is True
    # The booster is off the ascent lineage (stack -> upper_stage reaches
    # orbit): recorded, never graded, as in the M-07 reference chain. Its
    # divergence from the stack (here even a fabricated mass gain) is the
    # recovered stage's own physics, not a lost handoff of the mission.
    booster = by_id["separation:booster"]
    assert booster["status"] == "not_checked" and booster["graded"] is False
    assert booster["checks"] == [] and "not graded" in booster["detail"]


def test_staging_gap_is_bridged_by_a_coast_before_the_gate():
    """A backend samples each stage on its own clock: the parent ends 2 s
    before the child begins (FL-004: 206 s vs 208 s). Graded raw, that gap
    is ~7 km and ~18 m/s of flight and every staging read FAILED; the parent
    is coasted (two-body + J2, the backend's own gravity) to the child's
    first sample first, as in the M-07 chain."""
    from orbitopt.flightlab.coast import j2_coast

    r0, v0 = (6.6e6, 0.0, 0.0), (0.0, 7771.0, 0.0)
    m = itrs_to_gcrs_matrix(datetime.fromisoformat(EPOCH))  # J2 about the true pole
    r_end, v_end = j2_coast(r0, v0, 2.0, pole=(m[0][2], m[1][2], m[2][2]))
    rs, vs = [r_end], [v_end]
    parent = _track("stack", t=(0.0, 100.0), mass=(300000.0, 120000.0),
                    pos=((6.5e6, 0.0, 0.0), r0), vel=((0.0, 7700.0, 0.0), v0))
    child = replace(_track("upper", t=(102.0, 200.0), mass=(100000.0, 60000.0),
                           pos=(tuple(rs[-1]), (6.6e6, 1.5e6, 0.0)),
                           vel=(tuple(vs[-1]), (-1700.0, 7600.0, 0.0))),
                    parent_body_id="stack")
    scene = run_to_scene(_run([parent, child], events=[
        Event(t_s=100.0, kind="separation", body_id="stack", data={"separation_impulse_ns": 0.0})]))
    handoff = next(h for h in scene["handoffs"] if h["id"] == "separation:upper")
    assert handoff["bridge"]["method"] == "j2_coast"
    assert handoff["bridge"]["pole"] == "itrs_z_in_gcrs"
    assert handoff["bridge"]["gap_s"] == pytest.approx(2.0)
    assert handoff["status"] == "passed", handoff["detail"]
    assert "coasted 2.0 s" in handoff["detail"]


def test_seco_handoff_carries_gate_b_checks(fixture_scene):
    seco = next(h for h in fixture_scene["handoffs"] if h["id"] == "seco")
    assert seco["status"] == "failed"  # insertion_periapsis fails by design
    assert [c["id"] for c in seco["checks"]] == ["insertion_periapsis"]
    assert seco["state"]["body"] == "upper_stage"
    assert seco["time"] == pytest.approx(540.0)


def test_continuous_staging_handoff_passes():
    parent = _track("stack", t=(0.0, 100.0), mass=(300000.0, 120000.0))
    child = replace(_track("upper", t=(100.0, 200.0), mass=(100000.0, 60000.0)),
                    parent_body_id="stack",
                    # the child starts exactly where the parent ends
                    position_m=((0.0, 1e3, 6.4e6), (0.0, 2e3, 6.5e6)),
                    velocity_m_s=((0.0, 1e3, 1e3), (0.0, 2e3, 1.1e3)))
    scene = run_to_scene(_run([parent, child], events=[
        Event(t_s=100.0, kind="separation", body_id="stack", data={"separation_impulse_ns": 0.0})]))
    handoff = next(h for h in scene["handoffs"] if h["id"] == "separation:upper")
    assert handoff["status"] == "passed" and all(c["passed"] for c in handoff["checks"])
    mass = next(c for c in handoff["checks"] if c["id"] == "handoff.mass_conservation")
    assert mass["passed"] and mass["reason_code"] == "staging_mass_drop"
    assert "handoff upper PASSED" in next(
        e["note"] for e in scene["timeline"]["events"] if e["label"] == "Separation")


def test_unassessed_refly_keeps_seco_handoff_not_checked():
    from orbitopt.flightlab.contract import Check, Verification

    run = FlightRun(run_id="not-checked", target_sha256="0" * 64, vehicle_sha256="0" * 64,
                    git_sha="x", created_utc=EPOCH, tracks=(_track("sc"),),
                    verification=Verification(mission_constraints_passed=True,
                                              checks=(Check("insertion-above-atmosphere", True),)))
    seco = next(h for h in run_to_scene(run)["handoffs"] if h["id"] == "seco")
    assert seco["status"] == "not_checked"
    assert "not assessed" in seco["detail"]


def test_single_track_run_without_insertion_evidence_has_no_handoffs():
    assert "handoffs" not in run_to_scene(_run([_track("sc")]))


def test_coast_extends_timeline_bodies_and_phases(tmp_path):
    scene = run_to_scene(read_run(_all_gcrs_copy_run(tmp_path)), earth_texture=False,
                         coast_orbits=2.0)
    assert _phase_labels(scene) == ["On the pad", "Boost", "Staging", "Upper-stage burn",
                                    "Parking orbit", "Propagation"]
    coast = next(b for b in scene["bodies"] if b["id"] == "parking_orbit")
    assert coast["tStart"] == pytest.approx(540.0)
    assert coast["trail"]["times"][0] == pytest.approx(540.0)
    upper = next(b for b in scene["bodies"] if b["id"] == "upper_stage")
    assert coast["trail"]["positions"][0] == upper["trail"]["positions"][-1]  # no jump at SECO
    assert any("two-body" in a for a in scene["sceneAssumptions"])
    one_period = scene["timeline"]["phases"][4]["end"] - 540.0
    assert scene["timeline"]["max"] == pytest.approx(540.0 + 2.0 * one_period)


def test_coast_disabled_by_default_and_negative_rejected():
    scene = run_to_scene(read_run(RUN_FIXTURE), earth_texture=False)
    assert "parking_orbit" not in {b["id"] for b in scene["bodies"]}
    with pytest.raises(ValueError, match="coast_orbits"):
        run_to_scene(read_run(RUN_FIXTURE), earth_texture=False, coast_orbits=-1.0)
