"""Mission Control's layered navigation: the workspace bar, the breadcrumb and
the command palette (the match is Qt-free; the window flow runs offscreen)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from orbitopt.viz.workspace_nav import PaletteItem, filter_items  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RTLS_RUN = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"


def _items():
    return [PaletteItem("Workspace", "Runs"), PaletteItem("Run", "recovery", "e505-44e8c27 2026-11-10"),
            PaletteItem("Run", "recovery", "e204 2026-11-15"), PaletteItem("Vehicle", "starship_class_public_estimate"),
            PaletteItem("Vehicle", "f9_class_public_estimate"), PaletteItem("Mission", "Starship Flight 15")]


def test_filter_matches_every_word_in_kind_label_or_detail():
    items = _items()
    assert [i.label for i in filter_items(items, "")] == [i.label for i in items]    # order kept
    assert [i.detail for i in filter_items(items, "recovery e204")] == ["e204 2026-11-15"]
    assert [i.kind for i in filter_items(items, "vehicle")] == ["Vehicle", "Vehicle"]
    assert filter_items(items, "no-such-thing") == []
    assert filter_items(items, "RECOVERY")[0].label == "recovery"                    # case-insensitive


def test_filter_ranks_label_starts_above_later_matches():
    items = _items()
    got = [i.label for i in filter_items(items, "star")]
    assert got[0] == "Starship Flight 15" or got[0] == "starship_class_public_estimate"
    assert "f9_class_public_estimate" not in got
    ranked = filter_items([PaletteItem("Run", "xx recovery"), PaletteItem("Run", "recovery")], "recovery")
    assert [i.label for i in ranked] == ["recovery", "xx recovery"]


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_workspace_bar_emits_and_mirrors(qapp):
    from orbitopt.viz.workspace_nav import WORKSPACES, WorkspaceBar

    bar = WorkspaceBar()
    seen = []
    bar.workspace_changed.connect(seen.append)
    assert [k for k, _l, _t in WORKSPACES] == ["missions", "runs", "vehicles", "compare"]
    bar.buttons["missions"].setChecked(True)
    bar.buttons["runs"].click()
    assert seen == ["missions", "runs"] and bar.workspace == "runs"
    bar.set_workspace("vehicles")                               # mirroring never emits
    assert seen == ["missions", "runs"] and bar.workspace == "vehicles"


def test_palette_runs_the_chosen_action(qapp):
    from orbitopt.viz.workspace_nav import CommandPalette

    done = []
    palette = CommandPalette()
    palette.open_with([PaletteItem("Vehicle", "alpha", action=lambda: done.append("a")),
                       PaletteItem("Vehicle", "beta", action=lambda: done.append("b"))])
    assert palette.results.count() == 2
    palette.query.setText("bet")
    assert palette.results.count() == 1
    palette._activate()
    assert done == ["b"] and not palette.isVisible()


def test_window_navigates_through_workspaces_breadcrumb_and_palette(qapp, tmp_path, monkeypatch):
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
        bar, nav, sidebar = window.workspace_bar, window.flightlab_nav, window.sidebar
        assert bar.workspace == "missions" and window.breadcrumb_label.text().startswith("MISSIONS")
        # the old stacked switches are hidden: the bar is the navigation
        assert not any(b.isVisibleTo(window) for b in sidebar.mode_buttons.values())
        assert not nav.compare_button.isVisibleTo(window)

        bar.buttons["runs"].click()
        pump(lambda: not window._loading and window.flight_panel.bundle is not None)
        assert sidebar.mode == sidebar.FLIGHTLAB and nav.page == nav.RUNS
        assert window.center_stack.currentIndex() == 0
        assert window.breadcrumb_label.text() == "RUNS  ›  RTLS-ASCENT" or "RTLS" in window.breadcrumb_label.text()

        bar.buttons["vehicles"].click()
        qapp.processEvents()
        assert nav.page == nav.VEHICLES and window.center_stack.currentWidget() is window.vehicle_page
        assert bar.workspace == "vehicles" and window.breadcrumb_label.text().startswith("VEHICLES")

        bar.buttons["compare"].click()
        qapp.processEvents()
        assert window.center_stack.currentWidget() is window.flightlab_compare and bar.workspace == "compare"

        bar.buttons["runs"].click()                              # leaving Compare returns to the 3D view
        qapp.processEvents()
        assert window.center_stack.currentIndex() == 0 and bar.workspace == "runs"

        # the palette jumps straight to a vehicle from anywhere
        window._open_palette()
        titles = [window.palette.results.item(i).text() for i in range(window.palette.results.count())]
        assert any("starship_class_public_estimate" in t for t in titles)
        window.palette.query.setText("starship_class")
        window.palette._activate()
        qapp.processEvents()
        assert window.center_stack.currentWidget() is window.vehicle_page
        assert window.vehicle_page.view.name == "starship-class-public-estimate"
        assert bar.workspace == "vehicles" and "STARSHIP" in window.breadcrumb_label.text()

        bar.buttons["missions"].click()
        qapp.processEvents()
        assert sidebar.mode == sidebar.MISSIONS and window.center_stack.currentIndex() == 0
    finally:
        window.plotter.close()
        window.close()


def test_panel_toggles_live_in_the_top_bar(qapp, tmp_path, monkeypatch):
    """One toggle at each end of the bar collapses the list / the inspector;
    the in-panel chevrons and the rails' buttons are gone, and the bar's
    icons follow the real state."""
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QPushButton

    from orbitopt.viz.app import MissionControlWindow

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ORBITOPT_FLIGHT_RUNS", str(tmp_path / "none"))
    monkeypatch.chdir(tmp_path)
    window = MissionControlWindow()
    window.show()
    try:
        end = time.monotonic() + 60
        while window._loading and time.monotonic() < end:   # a workspace switch waits out a load
            qapp.processEvents()
            time.sleep(0.01)
        bar = window.workspace_bar
        assert set(bar.panel_toggles) == {"left", "right"}
        assert not window.sidebar.collapse_btn.isVisibleTo(window)
        assert not any(b.isVisibleTo(window) for rail in (window.left_rail, window.right_rail)
                       for b in rail.findChildren(QPushButton))
        assert window.left_stack.currentIndex() == 0 and window.right_stack.currentIndex() == 0
        bar.panel_toggles["left"].click()
        assert window.left_stack.currentIndex() == 1 and bar.panel_toggles["left"].property("open") is False
        bar.panel_toggles["left"].click()
        assert window.left_stack.currentIndex() == 0 and bar.panel_toggles["left"].property("open") is True
        bar.panel_toggles["right"].click()
        assert window.right_stack.currentIndex() == 1 and bar.panel_toggles["right"].property("open") is False
        window._on_workspace("vehicles")                   # a page has no inspector: its toggle is off
        assert not bar.panel_toggles["right"].isEnabled()
        window._on_workspace("missions")
        assert bar.panel_toggles["right"].isEnabled()
    finally:
        window.plotter.close()
        window.close()


def test_logo_mark_is_a_flat_single_colour_glyph(qapp):
    from PySide6.QtGui import QColor

    from orbitopt.viz.icon import render_logo_mark

    image = render_logo_mark(64, "#ffffff").toImage()
    opaque = {QColor(image.pixel(x, y)).rgb() for x in range(64) for y in range(64)
              if QColor.fromRgba(image.pixel(x, y)).alpha() == 255}
    assert opaque == {QColor("#ffffff").rgb()}                  # white ink only, no tile or gradient
    assert QColor.fromRgba(image.pixel(0, 0)).alpha() == 0       # transparent around the mark
