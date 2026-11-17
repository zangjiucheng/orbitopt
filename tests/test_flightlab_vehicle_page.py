"""Flight Lab's Vehicle page and its sidebar list (offscreen Qt).

The page is read-only like the rest of Flight Lab: nothing in it may be
editable, and it only draws what a vehicle file declares (plus, with a flown
run selected, the flown values beside the declared ones).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from orbitopt.flightlab import vehicle_catalog as vc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RTLS_RUN = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "nav.ini"), QSettings.IniFormat)


def _grab_has_ink(widget) -> bool:
    """True when the widget painted something other than one flat colour."""
    image = widget.grab().toImage()
    seen = {image.pixel(x, y) for x in range(0, image.width(), 5) for y in range(0, image.height(), 5)}
    return len(seen) > 4


@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_page_renders_every_catalog_vehicle_read_only(qapp, name):
    from PySide6.QtWidgets import QAbstractItemView

    from orbitopt.flightlab.ui.vehicle_page import VehiclePage
    from orbitopt.flightlab.vehicle_view import load_vehicle_view

    page = VehiclePage()
    page.resize(1100, 1200)
    page.show()
    qapp.processEvents()
    view = load_vehicle_view(name)
    page.set_vehicle(view)
    qapp.processEvents()
    assert page.title_label.text() == view.name
    assert page.numbers.rowCount() >= 10
    assert not page.overlay_table.isVisibleTo(page) and "Select a flown run" in page.overlay_header.text()
    for table in (page.numbers, page.overlay_table):
        assert table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert _grab_has_ink(page.profile) and _grab_has_ink(page.timeline)
    first_tag = page.numbers.item(2, 2).text() if page.numbers.rowCount() > 2 else ""
    assert isinstance(first_tag, str)
    if not view.geometry.declared:
        assert "schematic" in page.notes_label.text()
    page.clear()
    assert page.numbers.rowCount() == 0


def test_numbers_table_carries_the_files_provenance_with_its_source(qapp):
    from orbitopt.flightlab.ui.vehicle_page import VehiclePage
    from orbitopt.flightlab.vehicle_view import load_vehicle_view

    page = VehiclePage()
    page.set_vehicle(load_vehicle_view("f9_class_public_estimate"))
    rows = {page.numbers.item(r, 0).text(): page.numbers.item(r, 2)
            for r in range(page.numbers.rowCount())}
    isp = rows["  Isp (vac) · thrust (peak)"]
    assert isp.text().startswith("official") and "±1%" in isp.text()
    assert "Falcon 9 User Guide" in isp.toolTip()
    assert page.disclaimer_label.text().startswith("Public-estimate reconstruction")


def test_overlay_table_flags_what_differs_from_the_flown_run(qapp):
    from PySide6.QtGui import QColor

    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.ui.vehicle_page import VehiclePage
    from orbitopt.flightlab.vehicle_view import file_sha256, flown_overlay, load_vehicle_view
    from orbitopt.viz.theme import SUCCESS, WARNING

    view = load_vehicle_view("synthetic_two_stage_rtls")
    run = read_run(RTLS_RUN)
    page = VehiclePage()
    match = run.vehicle_sha256 == file_sha256(view.source_path)
    page.set_vehicle(view, flown_overlay(view, run), "e5-05a", match)
    assert match and "exact vehicle file" in page.overlay_header.text()
    assert page.overlay_table.isVisibleTo(page)
    rows = {page.overlay_table.item(r, 0).text(): r for r in range(page.overlay_table.rowCount())}
    meco = page.overlay_table.item(rows["Stage 1 cutoff (MECO)"], 3)
    seco = page.overlay_table.item(rows["Final cutoff (SECO)"], 3)
    assert meco.foreground().color() == QColor(SUCCESS)
    assert seco.foreground().color() == QColor(WARNING) and seco.text().startswith("-")
    assert page.overlay_table.item(rows["Max-Q"], 1).text() == "—"        # no declared value


def test_nav_lists_the_catalog_beside_the_runs(qapp, tmp_path):
    from orbitopt.flightlab.ui.nav import FlightLabNav

    nav = FlightLabNav(settings=_settings(tmp_path), default_roots=[tmp_path / "runs"])
    assert nav.page == nav.RUNS and nav.pages.currentIndex() == 0
    assert nav.vehicle_names() == sorted(vc.catalog_paths())
    pages, picked = [], []
    nav.page_changed.connect(pages.append)
    nav.vehicle_activated.connect(picked.append)
    nav.show_page(nav.VEHICLES)
    assert pages == ["vehicles"] and nav.pages.currentIndex() == 1
    assert nav.select_vehicle("f9_class_public_estimate")
    assert picked == ["f9_class_public_estimate"] and nav.current_vehicle() == "f9_class_public_estimate"
    assert not nav.select_vehicle("no-such-vehicle")
    nav.show_page(nav.RUNS)
    assert pages == ["vehicles", "runs"] and nav.pages.currentIndex() == 0
    nav.set_busy(True)
    assert not nav.vehicle_list.isEnabled() and not nav.page_buttons[nav.VEHICLES].isEnabled()
    nav.set_busy(False)
    assert nav.vehicle_list.isEnabled()


def test_mission_control_opens_the_vehicle_page_and_overlays_the_selected_run(
        qapp, tmp_path, monkeypatch):
    """Flight Lab sidebar -> Vehicles -> a vehicle opens the read-only page in
    the centre; with a flown run selected in Runs the page carries the
    declared-vs-flown rows; back to Runs the 3D view returns."""
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtCore import QSettings

    import orbitopt.flightlab.ui.nav as nav_mod
    from orbitopt.viz.app import MissionControlWindow

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    runs = tmp_path / "runs"
    shutil.copytree(RTLS_RUN, runs / "rtls-ascent")
    monkeypatch.setenv("ORBITOPT_FLIGHT_RUNS", str(runs))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(nav_mod, "QSettings", lambda *a, **k: QSettings(ini, QSettings.IniFormat))

    def pump(cond, timeout=60.0):
        end = time.monotonic() + timeout
        while not cond() and time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.01)
        assert cond()

    window = MissionControlWindow()
    window.show()
    try:
        pump(lambda: not window._loading)
        nav = window.flightlab_nav
        window.sidebar.set_mode(window.sidebar.FLIGHTLAB)
        pump(lambda: not window._loading and window.flight_panel.bundle is not None)
        assert window.center_stack.currentIndex() == 0
        assert nav.current_path() == runs / "rtls-ascent"

        nav.show_page(nav.VEHICLES)                         # first vehicle opens
        qapp.processEvents()
        assert window.center_stack.currentWidget() is window.vehicle_page
        assert window.scene_title_label.text().startswith("Vehicle · ")
        assert not window.timeline_bar.isVisible() and not window.chart_panel.isVisible()

        assert nav.select_vehicle("synthetic_two_stage_rtls")
        qapp.processEvents()
        page = window.vehicle_page
        assert page.view is not None and page.view.name == "synthetic-two-stage-rtls"
        assert page.overlay_table.isVisibleTo(page) and page.overlay
        assert "exact vehicle file" in page.overlay_header.text()

        assert nav.select_vehicle("f9_class_public_estimate")  # another vehicle, same run
        qapp.processEvents()
        assert window.vehicle_page.view.name == "f9-class-public-estimate"

        nav.show_page(nav.RUNS)
        qapp.processEvents()
        assert window.center_stack.currentIndex() == 0
        assert window.scene_title_label.text() != "Vehicle · f9-class-public-estimate"

        # Missions and back: Flight Lab returns to the page of the list that was showing
        nav.show_page(nav.VEHICLES)
        window.sidebar.mode_buttons[window.sidebar.MISSIONS].click()
        qapp.processEvents()
        assert window.center_stack.currentIndex() == 0 and not window._flightlab_active
        window.sidebar.mode_buttons[window.sidebar.FLIGHTLAB].click()
        qapp.processEvents()
        assert window.center_stack.currentWidget() is window.vehicle_page
    finally:
        window.plotter.close()
        window.close()
