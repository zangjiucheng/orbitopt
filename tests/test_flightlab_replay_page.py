"""Flight Lab replay: telemetry helpers, the Flight / Telemetry panels and a
run rendered in Mission Control's main view.

The telemetry/cache helpers are Qt-free and run on a base install. The panel
tests need PySide6 only (offscreen); the Mission Control test needs the VTK
view and is marked slow like the other window smoke tests."""
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


def _qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_flight_panel_shows_launch_summary_events_and_red_checks(tmp_path, monkeypatch):
    """The Flight tab of a run: verdict badges, the launch summary, the key
    events with altitude/speed, and every failed check in red."""
    app = _qt_app()
    from PySide6.QtGui import QColor

    from orbitopt.flightlab.run_index import launch_summary
    from orbitopt.flightlab.ui.flight_panel import FlightRunPanel, load_flight_bundle
    from orbitopt.viz.theme import DANGER

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    run, scene = load_run_scene(RUN_FIXTURE)
    bundle, target, channels = load_flight_bundle(RUN_FIXTURE, run=run, scene=scene)
    panel = FlightRunPanel()
    panel.set_run(bundle, target, channels)
    app.processEvents()

    rows = {(r.section, r.label): r.value for r in launch_summary(bundle, target)}
    assert (rows[("Result", "Optimizer converged")], rows[("Result", "Physics")],
            rows[("Result", "Mission constraints")]) == ("PASS", "FAIL", "FAIL")
    assert DANGER.lower() in panel.summary.toHtml().lower()  # the FAIL verdicts are red
    text = panel.summary.toPlainText()
    for word in ("LAUNCH", "Liftoff (UTC)", "RESULT", "Insertion orbit", run.run_id):
        assert word in text, word
    assert panel.event_table.rowCount() == len(run.events) > 0
    titles = [panel.event_table.item(r, 1).text() for r in range(panel.event_table.rowCount())]
    assert "Liftoff" in titles or "Ignition" in titles

    seen = []
    panel.time_requested.connect(seen.append)
    last = panel.event_table.rowCount() - 1
    panel._event_clicked(last, 0)
    assert seen == [pytest.approx(panel.events[last].t_s)]
    panel.set_time(panel.events[last].t_s)
    assert panel.event_table.currentRow() == last

    failed = set()
    for row in range(panel.check_table.rowCount()):
        if panel.check_table.item(row, 1).text() == "FAIL":
            assert panel.check_table.item(row, 0).foreground().color() == QColor(DANGER)
            failed.add(panel.check_table.item(row, 0).text())
    assert "insertion_periapsis" in failed
    assert panel.reason_panel.entries

    panel.clear()
    assert panel.bundle is None and panel.event_table.rowCount() == 0


def test_telemetry_panel_follows_time_and_requests_seeks():
    _qt_app()
    from orbitopt.flightlab.ui.flight_panel import TELEMETRY_CHARTS, TelemetryPanel

    run = read_run(RUN_FIXTURE)
    panel = TelemetryPanel()
    assert set(panel.charts) == {cid for cid, _l, _u in TELEMETRY_CHARTS} == REQUIRED_CHANNELS
    panel.set_channels(build_telemetry_channels(run), 0.0, 500.0)
    panel.set_time(123.0)
    assert all(chart._time == 123.0 for chart in panel.charts.values())
    seen = []
    panel.time_requested.connect(seen.append)
    panel.charts["altitude"].time_requested.emit(42.0)
    assert seen == [42.0]


@pytest.mark.slow
def test_mission_control_renders_a_flight_run_like_a_mission(tmp_path, monkeypatch):
    """Flight Lab mode: a run found on disk loads into the main 3D view and
    timeline (no separate replay page), with the Flight and Telemetry tabs;
    back in Missions they go away."""
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    app = _qt_app()
    import time as _time

    from PySide6.QtCore import QSettings

    from orbitopt.viz.app import MissionControlWindow

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    runs = tmp_path / "runs"
    shutil.copytree(RUN_FIXTURE, runs / "demo-run")
    monkeypatch.setenv("ORBITOPT_FLIGHT_RUNS", str(runs))
    QSettings("orbitopt", "Mission Control").remove("flightlab/run_roots")

    def pump(cond, timeout=60.0):
        end = _time.monotonic() + timeout
        while not cond() and _time.monotonic() < end:
            app.processEvents()
            _time.sleep(0.01)
        assert cond()

    window = MissionControlWindow()
    window.show()
    try:
        pump(lambda: not window._loading)
        assert window.flightlab_nav.run_paths() == [str(runs / "demo-run")]
        window.sidebar.set_mode(window.sidebar.FLIGHTLAB)
        pump(lambda: not window._loading and window.flight_panel.bundle is not None)
        assert window.center_stack.currentIndex() == 0          # the mission view itself
        assert window.right_tabs.currentWidget() is window.flight_panel
        assert window.chart_panel.currentWidget() is window.telemetry_panel
        assert window.state.timeline is not None and window.timeline_bar.isVisible()
        assert window._lock_reference_body_id is None  # face lock stays off until picked
        t = window.flight_panel.events[-1].t_s
        window.flight_panel._event_clicked(window.flight_panel.event_table.rowCount() - 1, 0)
        assert window.state.time == pytest.approx(t)

        window.sidebar.set_mode(window.sidebar.MISSIONS)
        pump(lambda: not window._loading)
        assert window.right_tabs.indexOf(window.flight_panel) < 0
        assert window.chart_panel.indexOf(window.telemetry_panel) < 0
    finally:
        window.close()
        app.processEvents()
