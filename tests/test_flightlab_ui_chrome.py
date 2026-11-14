"""Flight Lab inside Mission Control: chart titles and legends, one-sided
diff rows, the run list and the window chrome around a flight run (found in
the E5 acceptance run on a real OpenTsiolkovsky flight)."""
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


def test_flight_lab_is_a_sidebar_mode_that_renders_runs_like_missions(qapp, tmp_path, monkeypatch):
    """The sidebar's Missions | Flight Lab switch swaps the mission list for
    the flight-run list. Flight Lab has no page of its own any more (no
    Designer, no separate replay): the 3D view, view buttons and right column
    stay; with no runs on disk the view keeps the mission and the list says
    how to make one. The Missions switch brings the mission back."""
    pytest.importorskip("pyvistaqt")
    import time

    from PySide6.QtCore import QSettings

    from orbitopt.viz.app import FLIGHTLAB_SIDEBAR_LABEL, MissionControlWindow

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ORBITOPT_FLIGHT_RUNS", str(tmp_path / "none"))
    monkeypatch.chdir(tmp_path)  # no ./runs
    QSettings("orbitopt", "Mission Control").remove("flightlab/run_roots")
    window = MissionControlWindow()
    window.resize(1500, 900)
    window.show()
    try:
        for _ in range(600):
            if not window._loading:
                break
            qapp.processEvents()
            time.sleep(0.05)
        sidebar, nav = window.sidebar, window.flightlab_nav
        labels = [sidebar.mission_list.item(i).text() for i in range(sidebar.count())]
        assert FLIGHTLAB_SIDEBAR_LABEL not in labels
        mission_title = window.scene_title_label.text()
        mission_key = window._active_key

        sidebar.mode_buttons[sidebar.FLIGHTLAB].click()
        qapp.processEvents()
        assert window._flightlab_active and sidebar.pages.currentWidget() is nav
        assert window.center_stack.currentIndex() == 0  # the 3D view, not a Flight Lab page
        for widget in (window.view_strip, window.right_stack):
            assert widget.isVisible()
        assert nav.run_paths() == [] and nav.empty_label.isVisibleTo(nav)
        assert window._active_key == mission_key  # nothing to load: the mission stays

        nav.compare_button.click()
        qapp.processEvents()
        assert window.center_stack.currentWidget() is window.flightlab_compare
        assert window.scene_title_label.text() == "Compare runs"

        sidebar.mode_buttons[sidebar.MISSIONS].click()
        qapp.processEvents()
        assert not window._flightlab_active and sidebar.pages.currentIndex() == 0
        assert window.center_stack.currentIndex() == 0
        assert window.scene_title_label.text() == mission_title
        assert window._active_key == mission_key
    finally:
        window.plotter.close()
        window.close()


def _fake_run(path, verdict, run_id):
    import json

    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(
        {"schema_version": "1.0", "run_id": run_id, "created_utc": "2026-11-14T00:00:00Z", "bodies": []}))
    if verdict is not None:
        (path / "verification.json").write_text(json.dumps({"mission_constraints_passed": verdict}))


def test_nav_discovers_runs_with_their_verdict(qapp, tmp_path):
    from PySide6.QtCore import QSettings

    from orbitopt.flightlab.ui.nav import FlightLabNav, run_verdict

    root = tmp_path / "runs"
    _fake_run(root / "iss" / "ascent", True, "ot-1")        # orbitopt launch layout
    _fake_run(root / "polar", False, "ot-2")
    _fake_run(root / "deep" / "x" / "unknown", None, "ot-3")
    (root / "not-a-run").mkdir()
    assert run_verdict(root / "polar") is False and run_verdict(root / "deep" / "x" / "unknown") is None
    settings = QSettings(str(tmp_path / "nav.ini"), QSettings.IniFormat)
    nav = FlightLabNav(settings=settings, default_roots=[root])
    labels = sorted(nav.run_list.item(i).text() for i in range(nav.run_list.count()))
    assert labels == ["iss", "polar", "unknown"]  # "ascent" is labelled by its folder
    assert not nav.empty_label.isVisibleTo(nav)
    got = []
    nav.run_activated.connect(lambda path, label: got.append((path, label)))
    assert nav.select_path(root / "polar")
    assert got == [(str(root / "polar"), "polar")]

    other = tmp_path / "elsewhere"
    _fake_run(other / "gto", True, "ot-4")
    nav.add_root(other)                     # remembered, and its newest run selected
    assert str(other / "gto") in nav.run_paths()
    assert nav.current_path() == other / "gto"
    again = FlightLabNav(settings=settings, default_roots=[root])
    assert str(other / "gto") in again.run_paths()


def test_compare_page_fits_a_laptop_width(qapp):
    """Compare (two picker combos sized to their longest run label) asked
    for ~1,000 px and ran past a laptop screen's edge; it fits in 760."""
    from orbitopt.flightlab.ui.compare_page import FlightLabComparePage

    page = FlightLabComparePage()
    try:
        assert page.minimumSizeHint().width() <= 760, page.minimumSizeHint().width()
    finally:
        page.deleteLater()


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
