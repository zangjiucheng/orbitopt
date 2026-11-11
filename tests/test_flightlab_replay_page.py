"""FL-501 Flight Lab replay helpers and page tests.

The telemetry/cache helpers must stay Qt-free, so those tests run on a base
install with no display. The actual page test imports the GUI stack and is
marked slow like the Mission Control smoke test."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.ui.telemetry import (
    build_telemetry_channels,
    channel_values_at,
    interpolated_value,
    load_run_scene,
)

RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"
REPO = Path(__file__).resolve().parents[1]
OT_FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
REQUIRED_CHANNELS = {"altitude", "speed", "dynamic_pressure", "g_load", "mass", "throttle"}


def _copy_run(tmp_path):
    dest = tmp_path / "run"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(RUN_FIXTURE, dest)
    return dest


def _channels():
    run = read_run(RUN_FIXTURE)
    return {channel.id: channel for channel in build_telemetry_channels(run)}


def test_replay_telemetry_exposes_six_required_channels():
    channels = _channels()
    assert set(channels) == REQUIRED_CHANNELS
    for channel in channels.values():
        assert channel.label
        assert channel.unit
        assert channel.series or channel.missing_reason


def test_replay_telemetry_derives_launch_trajectory_channels():
    channels = _channels()
    for channel_id in ("altitude", "speed", "dynamic_pressure", "g_load", "mass"):
        channel = channels[channel_id]
        assert channel.has_data, channel.missing_reason
        assert all(series.unit == channel.unit for series in channel.series)
        assert all(series.color for series in channel.series)


def test_replay_telemetry_interpolates_channel_values_at_time():
    channels = _channels()
    values = channel_values_at(channels["altitude"], 0.0)
    assert values
    series = channels["altitude"].series[0]
    mid = series.points[len(series.points) // 2]
    assert interpolated_value(series.points, mid[0]) == pytest.approx(mid[1])


def test_load_run_scene_uses_cached_scene_and_invalidates_on_touch(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(cache_dir))
    run_dir = _copy_run(tmp_path)

    run, scene = load_run_scene(run_dir)
    assert run.run_id == scene["flightRun"]["runId"]
    assert scene["timeline"]["events"]
    assert scene["timeline"]["unitLabel"] == "seconds"
    first_files = list(cache_dir.rglob("*.json"))
    assert first_files

    manifest = run_dir / "manifest.json"
    stat = manifest.stat()
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    load_run_scene(run_dir)
    assert len(list(cache_dir.rglob("*.json"))) == 2


@pytest.fixture
def gate_b_run(monkeypatch, tmp_path):
    """A real Gate-B FlightRun from `simulate`, flown by the FL-004 CLI stub."""
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.simulate import simulate

    cli = tmp_path / "otcli"
    shutil.copy(OT_FIX / "stub_cli.py", cli)
    cli.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(cli))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(OT_FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "gate_b_run"
    simulate(load_target(TARGET), VEHICLE, out, timeout_s=60)
    return out


def test_gate_b_run_replays_telemetry_and_failed_checks(gate_b_run):
    run, scene = load_run_scene(gate_b_run)
    assert scene["timeline"]["events"]
    channels = {channel.id: channel for channel in build_telemetry_channels(run)}
    assert set(channels) == REQUIRED_CHANNELS
    for channel_id in ("altitude", "speed", "mass"):
        assert channels[channel_id].has_data, channels[channel_id].missing_reason
    # The stub vehicle inserts off-target by design: the page must have a
    # failed check with an explicit reason code to show in red.
    failed = [c for c in run.verification.checks if not c.passed]
    assert failed and all(c.reason_code for c in failed)


@pytest.mark.slow
def test_flightlab_replay_page_synthetic_run(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui import FlightLabReplayPage
    from orbitopt.viz.theme import DANGER

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    page = FlightLabReplayPage()
    page.show()
    app.processEvents()
    try:
        page.load_run(RUN_FIXTURE)
        app.processEvents()

        assert page._run is not None
        assert page._scene is not None
        assert page._events
        assert page.fidelity_label.text() == "fast-3dof"
        assert page.badge_optimizer._state == "verified"
        assert page.badge_physics._state == "failed"
        assert page.badge_constraints._state == "failed"

        assert page.event_list.count() == len(page._events)
        last = page.event_list.item(page.event_list.count() - 1)
        assert last is not None
        event_t = float(last.data(Qt.ItemDataRole.UserRole))
        page._event_clicked(last)  # click -> seek
        assert page._time == pytest.approx(event_t)
        assert all(chart._time == page._time for chart in page._charts.values())
        assert page.event_list.currentRow() == page.event_list.count() - 1

        # Scrubbing the timeline moves charts and the 3D time together.
        page.timeline_slider.setValue(page.timeline_slider.maximum() // 2)
        assert page._time < event_t or page._time == pytest.approx(event_t)
        assert all(chart._time == page._time for chart in page._charts.values())
        assert page._renderer is not None

        failed_check_ids = set()
        for row in range(page.check_table.rowCount()):
            result = page.check_table.item(row, 1)
            assert result is not None
            if result.text() == "FAIL":
                for col in range(page.check_table.columnCount()):
                    item = page.check_table.item(row, col)
                    assert item is not None
                    assert item.foreground().color() == QColor(DANGER)
                failed_check_ids.add(page.check_table.item(row, 0).text())
        assert "insertion_periapsis" in failed_check_ids
    finally:
        page.hide()
        app.processEvents()


@pytest.mark.slow
def test_flightlab_replay_page_gate_b_run(gate_b_run):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")

    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui import FlightLabReplayPage
    from orbitopt.viz.theme import DANGER

    app = QApplication.instance() or QApplication([])
    page = FlightLabReplayPage()
    page.show()
    app.processEvents()
    try:
        page.load_run(gate_b_run)
        app.processEvents()
        assert page._run is not None and page._renderer is not None
        assert page.badge_constraints._state == "failed"
        page.seek_relative_event(+1)
        assert all(chart._time == page._time for chart in page._charts.values())
        red_reasons = [
            page.check_table.item(row, 4).text()
            for row in range(page.check_table.rowCount())
            if page.check_table.item(row, 1).text() == "FAIL"
        ]
        assert red_reasons and all(red_reasons)
        row = next(r for r in range(page.check_table.rowCount())
                   if page.check_table.item(r, 1).text() == "FAIL")
        assert page.check_table.item(row, 4).foreground().color() == QColor(DANGER)
    finally:
        page.hide()
        app.processEvents()
