"""Run discovery and the launch summary behind Flight Lab mode (Qt-free)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from orbitopt.flightlab.compare import load_bundle
from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.run_index import (
    default_run_roots,
    discover_flight_runs,
    find_target,
    is_run_dir,
    key_events,
    launch_summary,
    run_label,
)
from orbitopt.flightlab.ui.telemetry import build_telemetry_channels

RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"
REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"


def test_discovery_finds_runs_at_any_depth_once_newest_first(tmp_path):
    root = tmp_path / "out"
    shutil.copytree(RUN_FIXTURE, root / "launch-iss" / "ascent")
    shutil.copytree(RUN_FIXTURE, root / "a" / "b" / "plain")
    _rename_run(root / "launch-iss" / "ascent", "other-run-id")   # distinct runs, not copies
    busy = root / "busy"
    shutil.copytree(RUN_FIXTURE, busy)
    (busy / "_designer-in-progress").write_text("1")
    (root / "empty").mkdir()
    newest = root / "a" / "b" / "plain" / "manifest.json"
    newest.touch()
    assert not is_run_dir(busy) and not is_run_dir(root / "empty")

    entries = discover_flight_runs([root, root / "launch-iss", tmp_path / "missing"])
    assert [e.label for e in entries] == ["plain", "launch-iss"]  # newest first, no duplicate
    assert entries[1].path == root / "launch-iss" / "ascent"
    assert entries[0].verdict is False                            # the fixture fails its mission
    assert read_run(RUN_FIXTURE).run_id in entries[0].detail
    # depth counts folders below the root: launch-iss/ascent is 2 deep, a/b/plain 3
    assert [e.label for e in discover_flight_runs([root], max_depth=1)] == []
    assert [e.label for e in discover_flight_runs([root], max_depth=2)] == ["launch-iss"]
    # a run directory is a root of its own
    assert [e.label for e in discover_flight_runs([root / "a" / "b" / "plain"])] == ["plain"]


def test_run_label_names_role_folders_by_their_parent():
    assert run_label(Path("/x/iss-cape/ascent")) == "iss-cape"
    assert run_label(Path("/x/gateb-0001")) == "gateb-0001"


def test_default_roots_include_the_env_folders(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ORBITOPT_FLIGHT_RUNS", os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]))
    monkeypatch.chdir(tmp_path)
    assert default_run_roots() == [tmp_path / "a", tmp_path / "b", tmp_path / "runs"]


def test_launch_summary_reads_site_target_vehicle_and_result(tmp_path):
    run_dir = tmp_path / "launch" / "ascent"
    shutil.copytree(RUN_FIXTURE, run_dir)
    assert find_target(run_dir) is None
    shutil.copy(TARGET, run_dir.parent / "target.yaml")   # the `orbitopt launch` layout
    target = find_target(run_dir)
    assert target is not None

    rows = {(r.section, r.label): r.value for r in launch_summary(load_bundle(run_dir), target)}
    assert rows[("Launch", "Site")] == target.launch_site.name
    assert "N" in rows[("Launch", "Coordinates")] and "W" in rows[("Launch", "Coordinates")]
    assert rows[("Target", "Orbit")].startswith("400 × 400 km @ 51.60°")
    assert ("Result", "Insertion orbit") in rows and ("Result", "Mission constraints") in rows
    assert rows[("Run", "Run id")] == read_run(RUN_FIXTURE).run_id
    # without the target the coordinates still come from the first sample
    rows = {(r.section, r.label): r.value for r in launch_summary(load_bundle(run_dir), None)}
    assert ("Launch", "Site") not in rows and ("Target", "Orbit") not in rows


def test_key_events_carry_altitude_and_speed():
    run = read_run(RUN_FIXTURE)
    events = key_events(run, build_telemetry_channels(run))
    assert [e.t_s for e in events] == sorted(e.t_s for e in run.events)
    assert all(e.title and e.title[0].isupper() for e in events)
    flown = [e for e in events if e.altitude_km is not None]
    assert flown and all(e.speed_km_s is not None for e in flown)
    assert json.dumps([e.title for e in events])  # plain strings


def _rename_run(path, run_id):
    import json

    manifest = path / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["run_id"] = run_id
    manifest.write_text(json.dumps(doc))


def test_copies_and_reflights_of_one_run_id_are_listed_once(tmp_path):
    """The same run_id is the same run: a copy pulled from a server and the
    run flown again both fold into the newest one, whose entry records the
    others. Runs with different ids stay separate."""
    import json
    import os
    import shutil

    from orbitopt.flightlab.run_index import discover_flight_runs

    def make(path, run_id, created, mtime):
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps(
            {"schema_version": "1.0", "run_id": run_id, "created_utc": created, "bodies": []}))
        os.utime(path / "manifest.json", (mtime, mtime))

    local, remote = tmp_path / "local", tmp_path / "remote" / "pulled"
    make(local / "ascent", "ot-1", "2026-11-14T00:00:00Z", 1_000)
    shutil.copytree(local / "ascent", remote / "deep" / "ascent")           # a copy
    make(local / "reflown", "ot-1", "2026-11-16T00:00:00Z", 2_000)            # the same run, flown again
    make(local / "other", "ot-2", "2026-11-14T00:00:00Z", 500)
    entries = discover_flight_runs([local, tmp_path / "remote"])
    assert sorted(e.path.name for e in entries) == ["other", "reflown"]
    newest = next(e for e in entries if e.path.name == "reflown")
    assert set(newest.copies) == {local / "ascent", remote / "deep" / "ascent"}
    assert next(e for e in entries if e.path.name == "other").copies == ()


def test_label_hint_tells_same_named_runs_apart(tmp_path):
    from orbitopt.flightlab.run_index import label_hint, run_label

    a = tmp_path / "e505-44e8" / "recovery" / "run"
    b = tmp_path / "remote" / "e204" / "recovery" / "run"
    assert run_label(a) == run_label(b) == "recovery"
    assert label_hint(a) == "e505-44e8" and label_hint(b) == "e204"
    assert label_hint(tmp_path / "iss" / "ascent") == tmp_path.name       # the label is "iss" already
