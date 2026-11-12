"""Flight Lab inside Mission Control: chart titles and legends, one-sided
diff rows, and the window chrome around the workbench (found in the E5
acceptance run on a real OpenTsiolkovsky flight)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _channel(n_series: int):
    from orbitopt.flightlab.ui.telemetry import TelemetryChannel, TelemetrySeries

    colors = ("#e8a23e", "#3ecf8e", "#ef5350", "#a78bfa")
    names = ("Stage1", "Stage2", "Stage2 fairing", "Booster")
    series = tuple(TelemetrySeries("altitude", names[i], "km", colors[i], names[i].lower(),
                                   ((0.0, 0.0), (100.0, 50.0 * (i + 1))))
                   for i in range(n_series))
    return TelemetryChannel("altitude", "Dynamic pressure", "kPa", "", series)


def _paint(chart):
    from PySide6.QtGui import QImage, QPainter

    image = QImage(chart.width(), chart.height(), QImage.Format_ARGB32)
    painter = QPainter(image)
    chart._paint_static(painter)
    painter.end()


def test_narrow_chart_keeps_its_title_and_a_complete_legend(qapp):
    """A narrow chart cut its title to "Dynam" (a fixed width-90 box) and
    painted only the legend entries that fitted beside it -- four bodies
    labelled "Booster" alone. The title now has the header row; the legend
    is all-or-nothing, moving inside the plot when the header is full."""
    from orbitopt.flightlab.ui.charts import _TelemetryChart

    chart = _TelemetryChart("altitude", "Dynamic pressure", "kPa")
    chart.resize(330, 170)
    chart.set_channel(_channel(4))
    _paint(chart)
    assert chart.legend_placement == "inside"
    chart.resize(900, 170)
    chart.set_channel(_channel(1))
    _paint(chart)
    assert chart.legend_placement == "header"
    # the title box is the widget's width, not the plot's minus 90 px
    assert chart.width() - 2 * chart._TITLE_X > chart.plot_rect().width() - 90


def test_one_sided_diff_rows_say_which_run_has_the_value():
    from orbitopt.flightlab.compare import DiffRow
    from orbitopt.flightlab.ui.compare_page import _fmt_delta

    assert _fmt_delta(DiffRow("k", "Vehicle", "Payload", "kg", 15000.0, None, None, True)) == "only in A"
    assert _fmt_delta(DiffRow("k", "Landing", "Prop", "kg", None, 1850.0, None, True)) == "only in B"
    assert _fmt_delta(DiffRow("k", "Mission", "Recovery", "", "none", "rtls", None, True)) == "changed"


def test_flight_lab_is_a_sidebar_mode_with_its_own_navigation(qapp, tmp_path, monkeypatch):
    """Flight Lab is not a mission: the sidebar's Missions | Flight Lab switch
    swaps the mission list for Flight Lab's own navigation (pages + runs).
    In Flight Lab mode the window shows Flight Lab's title, not the previous
    mission's ("Solar System"), its view buttons or its body inspector; the
    Missions switch brings the mission back as it was."""
    pytest.importorskip("pyvistaqt")
    import time

    from orbitopt.viz.app import FLIGHTLAB_SIDEBAR_LABEL, MissionControlWindow

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))  # no cached designer runs
    window = MissionControlWindow()
    window.resize(1500, 900)
    window.show()
    try:
        for _ in range(600):
            if not window._loading:
                break
            qapp.processEvents()
            time.sleep(0.05)
        sidebar, wb = window.sidebar, window.flightlab_page
        labels = [sidebar.mission_list.item(i).text() for i in range(sidebar.count())]
        assert FLIGHTLAB_SIDEBAR_LABEL not in labels
        mission_title = window.scene_title_label.text()
        mission_key = window._active_key

        sidebar.mode_buttons[sidebar.FLIGHTLAB].click()
        qapp.processEvents()
        assert window._flightlab_active and sidebar.pages.currentWidget() is wb.nav
        assert window.center_stack.currentWidget() is wb
        assert window.scene_title_label.text() == FLIGHTLAB_SIDEBAR_LABEL
        for widget in (window.view_strip, window._view_controls_widget, window.right_stack):
            assert not widget.isVisible()
        assert wb.tabs.tabBar().isHidden()  # the sidebar picks the page
        wb.nav.page_list.setCurrentRow(2)
        assert wb.tabs.currentWidget() is wb.compare_page
        wb.tabs.setCurrentWidget(wb.replay_page)  # e.g. a finished simulate opens Replay
        assert wb.nav.page_list.currentRow() == 1
        assert wb.nav.empty_label.isVisibleTo(wb.nav)

        sidebar.mode_buttons[sidebar.MISSIONS].click()
        qapp.processEvents()
        assert not window._flightlab_active and sidebar.pages.currentIndex() == 0
        assert window.scene_title_label.text() == mission_title
        assert window._active_key == mission_key
        for widget in (window.view_strip, window._view_controls_widget, window.right_stack):
            assert widget.isVisible()
    finally:
        window.plotter.close()
        window.close()


def test_nav_lists_runs_with_their_verdict(qapp, tmp_path):
    import json

    from orbitopt.flightlab.ui.nav import FlightLabNav, run_verdict

    passed, failed, unknown = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d, v in ((passed, True), (failed, False)):
        d.mkdir()
        (d / "verification.json").write_text(json.dumps({"mission_constraints_passed": v}))
    unknown.mkdir()
    assert (run_verdict(passed), run_verdict(failed), run_verdict(unknown)) == (True, False, None)
    nav = FlightLabNav()
    got = []
    nav.run_activated.connect(got.append)
    nav.set_runs([("run-a · recovery=none · 1234", passed), ("run-b", failed), ("run-c", unknown)], current=failed)
    assert nav.run_paths() == [str(passed), str(failed), str(unknown)]
    assert nav.run_list.item(0).text() == "run-a"  # the run id; the full label is the tooltip
    assert nav.run_list.currentRow() == 1 and not nav.empty_label.isVisibleTo(nav)
    nav._run_clicked(nav.run_list.item(2))
    assert got == [str(unknown)]


def test_flight_lab_pages_fit_a_laptop_width(qapp):
    """The Designer (Mission beside a one-row Vehicle header, unwrapped
    status lines) asked for ~1,070 px and Compare (two picker combos sized
    to their longest run label) ~1,000: the window could not shrink below
    ~1,250 px and ran past a laptop screen's edge. Each page now fits in
    under 760 px; the Designer stacks Mission over Vehicle when narrow."""
    from orbitopt.flightlab.ui import FlightLabWorkbench

    wb = FlightLabWorkbench()
    try:
        for page in (wb.designer_page, wb.replay_page, wb.compare_page):
            assert page.minimumSizeHint().width() <= 760, (type(page).__name__, page.minimumSizeHint().width())
        d = wb.designer_page
        wb.resize(700, 800)
        wb.show()
        for _ in range(5):
            qapp.processEvents()
        assert d._side_by_side is False  # stacked
        wb.resize(1700, 800)
        for _ in range(5):
            qapp.processEvents()
        assert d._side_by_side is True
    finally:
        wb.terminate_compute()
        wb.deleteLater()


def test_sidebar_mode_switch_is_never_squeezed(qapp):
    """The switch's "min-height: 0" let the layout squeeze it to 20 px,
    clipping "Missions" / "Flight Lab"."""
    from orbitopt.viz.sidebar import MissionSidebar
    from orbitopt.viz.theme import STYLESHEET

    bar = MissionSidebar()
    bar.setStyleSheet(STYLESHEET)
    bar.resize(220, 300)
    bar.show()
    qapp.processEvents()
    for button in bar.mode_buttons.values():
        # a real minimum, not 0: a short window squeezes down to it
        assert button.minimumHeight() >= button.fontMetrics().height() + 8
        assert button.height() >= button.minimumHeight()
    bar.close()
