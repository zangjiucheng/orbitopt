"""Replay input identity, stage fuel accounting, fixed sites and plot navigation."""
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.replay_context import replay_inputs
from orbitopt.flightlab.scene_export import run_to_scene
from orbitopt.flightlab.ui.telemetry import build_telemetry_channels

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tests/data/mission/reference/e5-08-new-vehicle-starbase/mission/recovery/run"


def test_recorded_stage_fuel_accounts_for_attached_upper_stack_and_booster():
    run = read_run(RUN)
    vehicle, target = replay_inputs(run, RUN)
    assert target.launch_site.name == "starbase"
    channel = next(c for c in build_telemetry_channels(run, vehicle=vehicle) if c.id == "propellant")
    series = {s.body_id: s for s in channel.series}
    # 4825 t stack includes the still-attached 1300 t ship and 275 t booster dry mass.
    assert series["stage1"].points[0][1] == pytest.approx(3250)
    assert series["stage2"].points[0][1] == pytest.approx(1200)
    assert series["stage1"].points[-1][1] == pytest.approx(series["booster"].points[0][1])
    assert series["booster"].points[-1][1] == pytest.approx(107.1107800696498)
    assert all(s.derived and "est." in s.label for s in series.values())


def test_payload_already_included_in_stage_dry_mass_is_not_subtracted_twice():
    run = read_run(RUN)
    track = replace(run.tracks[1], mass_kg=tuple(120 for _ in run.tracks[1].t_s))
    vehicle = {"payload_kg": 10, "stages": [
        {"mass_initial_kg": 500, "mass_dry_kg": 50},
        {"mass_initial_kg": 120, "mass_dry_kg": 30}]}  # dry includes the 10 kg payload
    channel = next(c for c in build_telemetry_channels(replace(run, tracks=(track,)), vehicle=vehicle)
                   if c.id == "propellant")
    assert channel.series[0].points[0][1] == pytest.approx(0.09)


def test_missing_input_identity_does_not_fabricate_fuel(tmp_path):
    run = replace(read_run(RUN), vehicle_sha256="0"*64, target_sha256="0"*64)
    (tmp_path / "vehicle.yaml").write_text("name: unrelated\nstages: []\n")
    assert replay_inputs(run, tmp_path) == (None, None)
    fuel = next(c for c in build_telemetry_channels(run) if c.id == "propellant")
    assert not fuel.has_data and "matching vehicle" in fuel.missing_reason


def test_rtls_site_stays_fixed_in_the_earth_frame_and_remains_after_touchdown():
    np = pytest.importorskip("numpy")
    from orbitopt.flightlab.frames import itrs_to_gcrs_matrix
    from orbitopt.flightlab.contract import _parse_utc
    from datetime import timedelta
    run = read_run(RUN)
    _, target = replay_inputs(run, RUN)
    scene = run_to_scene(run, target=target)
    sites = [b for b in scene["bodies"] if b["kind"] == "site"]
    assert len(sites) == 1 and "Launch / recovery" in sites[0]["name"]
    site = sites[0]
    assert site["hideTrail"] and "tEnd" not in site
    points, times = site["trail"]["positions"], site["trail"]["times"]
    epoch = _parse_utc(run.tracks[0].epoch_utc, "test epoch")
    first = np.array(itrs_to_gcrs_matrix(epoch)).T @ np.array(points[0])
    last = np.array(itrs_to_gcrs_matrix(epoch + timedelta(seconds=times[-1]))).T @ np.array(points[-1])
    assert np.linalg.norm(first - last) < 1e-6  # 1 mm in scene km
    assert np.linalg.norm(np.array(points[0]) - points[-1]) > 100


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_detail_plot_zoom_pan_reset_and_seeking_stay_linked(app):
    from PySide6.QtCore import QPointF, QPoint, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from orbitopt.flightlab.ui.charts import TelemetryChart
    run = read_run(RUN)
    channel = next(c for c in build_telemetry_channels(run) if c.id == "altitude")
    chart = TelemetryChart("altitude", "Altitude", "km")
    chart.set_channel(channel)
    chart.set_time_range(0, 700)
    chart.open_detail()
    detail = chart._detail_chart
    app.processEvents()
    event = QWheelEvent(QPointF(500, 200), QPointF(500, 200), QPoint(), QPoint(0, 240),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    detail.wheelEvent(event)
    before = detail._xmin
    assert detail._xmax - detail._xmin < 700
    QTest.mousePress(detail, Qt.RightButton, pos=QPoint(500, 200))
    QTest.mouseMove(detail, QPoint(400, 200))
    QTest.mouseRelease(detail, Qt.RightButton, pos=QPoint(400, 200))
    assert detail._xmin > before
    requested = []
    chart.time_requested.connect(requested.append)
    QTest.mouseClick(detail, Qt.LeftButton, pos=QPoint(500, 200))
    assert requested and detail._xmin < requested[-1] < detail._xmax
    chart.set_time(300)
    assert detail._time == 300
    detail.reset_view()
    assert (detail._xmin, detail._xmax) == (0, 700)
    chart._detail_dialog.close()
    chart.close()


def test_body_strip_keeps_focus_controls_separate_from_inspector(app):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from orbitopt.viz.body_strip import BodyStrip
    from orbitopt.viz.state import MissionState
    state = MissionState()
    strip = BodyStrip(state)
    state.set_scene({"bodies": [{"id": "earth", "name": "Earth"},
                                {"id": "stage2", "name": "Stage 2", "tStart": 122, "tEnd": 700}],
                     "timeline": {"min": 0, "max": 700}})
    assert not strip.buttons["stage2"].isEnabled()
    state.set_time(200)
    seen = []
    strip.focus_requested.connect(seen.append)
    QTest.mouseClick(strip.buttons["stage2"], Qt.LeftButton)
    assert seen == ["stage2"]
    state.select_body("stage2")
    assert strip.buttons["stage2"].isChecked()
    state.set_scene(None)
    assert not strip.buttons


def test_body_strip_never_exposes_a_light_scroll_viewport(app):
    from orbitopt.viz.body_strip import BodyStrip
    from orbitopt.viz.state import MissionState
    state = MissionState()
    strip = BodyStrip(state)
    strip.resize(900, 40)
    state.set_scene({"bodies": [{"id": "earth", "name": "Earth", "color": "#4c8eff"}],
                     "timeline": {"min": 0, "max": 1}})
    strip.show()
    app.processEvents()
    image = strip.grab().toImage()
    # Empty scroll-area space is where macOS previously painted pale grey.
    pixel = image.pixelColor(int(image.width() * 0.9), int(image.height() * 0.5))
    assert max(pixel.red(), pixel.green(), pixel.blue()) < 32
    assert strip.height() == 40
    strip.close()
