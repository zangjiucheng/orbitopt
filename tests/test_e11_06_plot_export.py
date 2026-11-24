"""E11-06: every plot can be downloaded -- the picture (PNG) and, where it has data, the data (CSV)."""
from __future__ import annotations

import csv
import io

import pytest

from orbitopt.flightlab.ui.plot_export import CSV_HEADER, channel_to_csv, rows_to_csv, slug


def _channel():
    from orbitopt.flightlab.ui.telemetry import TelemetryChannel, TelemetrySeries

    a = TelemetrySeries("altitude", "Stage 1", "km", "#fff", "stage1", ((0.0, 0.0), (10.0, 5.5), (20.0, 12.25)))
    b = TelemetrySeries("altitude", "Booster (est.)", "km", "#0f0", "booster", ((5.0, 100.0), (15.0, 80.0)), derived=True)
    return TelemetryChannel("altitude", "Altitude", "km", "", (a, b))


def test_file_names_come_from_the_plot_title():
    assert slug("Dynamic pressure") == "dynamic-pressure"
    assert slug("SUPER HEAVY") == "super-heavy" and slug("  ?? ") == "plot" and slug("G-load (g)") == "g-load-g"


def test_a_channel_becomes_long_format_csv_with_every_sample_and_its_provenance():
    rows = list(csv.reader(io.StringIO(channel_to_csv(_channel()))))
    assert tuple(rows[0]) == CSV_HEADER and len(rows) == 1 + 3 + 2
    assert rows[1] == ["Stage 1", "stage1", "0", "0", "km", "0"]
    assert rows[3] == ["Stage 1", "stage1", "20", "12.25", "km", "0"]
    assert rows[4] == ["Booster (est.)", "booster", "5", "100", "km", "1"]          # a derived series is flagged
    from orbitopt.flightlab.ui.telemetry import TelemetryChannel

    assert channel_to_csv(TelemetryChannel("x", "X", "u", "no data", ())) is None    # nothing to download
    assert rows_to_csv(("a", "b"), [(1, 2)]) == "a,b\n1,2\n"


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _chooser(path):
    def choose(_parent, _caption, default_name, _filt):
        choose.asked = default_name
        return str(path) if path else ""
    return choose


def test_a_telemetry_chart_saves_its_picture_and_its_data(qapp, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage

    from orbitopt.flightlab.ui.charts import TelemetryChart
    from orbitopt.flightlab.ui.plot_export import ExportButton

    chart = TelemetryChart("altitude", "Altitude", "km")
    chart.resize(420, 220)
    chart.set_channel(_channel())
    chart.set_time_range(0.0, 20.0)
    chart.show()
    btn = chart.export_button
    # the button sits in the right margin, clear of the plot and so of the legend
    assert btn.x() >= chart.plot_rect().right() and btn.y() < chart.plot_rect().top()
    png = tmp_path / "chart.png"
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(png)))
    assert btn.save_png() == str(png) and png.is_file()
    img = QImage(str(png))
    assert img.width() == 840 and img.height() == 440                              # twice the screen size
    assert btn.isVisible()                                                          # the button is back after the grab
    out = tmp_path / "chart.csv"
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(out)))
    assert btn.save_csv() == str(out)
    assert out.read_text().splitlines()[0] == ",".join(CSV_HEADER) and len(out.read_text().splitlines()) == 6
    assert ExportButton.file_chooser.asked == "altitude.csv"                       # named after the plot


def test_cancelling_the_dialog_writes_nothing_and_a_missing_extension_is_added(qapp, tmp_path, monkeypatch):
    from orbitopt.flightlab.ui.charts import TelemetryChart
    from orbitopt.flightlab.ui.plot_export import ExportButton

    chart = TelemetryChart("altitude", "Altitude", "km")
    chart.resize(300, 200)
    chart.set_channel(_channel())
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(None)))
    assert chart.export_button.save_png() is None and chart.export_button.save_csv() is None
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(tmp_path / "noext")))
    saved = chart.export_button.save_csv()
    assert saved.endswith("noext.csv") and (tmp_path / "noext.csv").is_file()


def test_a_chart_with_no_data_offers_the_picture_but_not_the_data(qapp, tmp_path, monkeypatch):
    from orbitopt.flightlab.ui.charts import TelemetryChart
    from orbitopt.flightlab.ui.plot_export import ExportButton

    chart = TelemetryChart("throttle", "Throttle", "%")
    chart.resize(300, 200)
    btn = chart.export_button
    btn._refresh_menu()
    assert btn.png_action.isEnabled() and not btn.csv_action.isEnabled()
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(tmp_path / "x.csv")))
    assert btn.save_csv() is None and not (tmp_path / "x.csv").exists()


def test_every_plot_in_the_flight_panels_has_a_download_button(qapp):
    from orbitopt.flightlab.ui.engine_hud import EngineHud
    from orbitopt.flightlab.ui.flight_panel import TELEMETRY_CHARTS, TelemetryPanel
    from orbitopt.flightlab.ui.recovery_panel import DispersionPlot
    from orbitopt.flightlab.ui.plot_export import ExportButton

    panel = TelemetryPanel()
    assert set(panel.charts) == {c[0] for c in TELEMETRY_CHARTS}
    assert all(isinstance(c.export_button, ExportButton) for c in panel.charts.values())
    hud = EngineHud()
    for card in (hud.superheavy, hud.starship, hud.gridfins):
        card.resize(500, 200)
        assert isinstance(card.export_button, ExportButton)
        card.export_button._refresh_menu()
        assert card.export_button.png_action.isEnabled() and not card.export_button.csv_action.isEnabled()  # picture only
        assert card.export_button.x() >= 500 - 16 - 2 - 1                               # top-right corner
    scatter = DispersionPlot()
    scatter.resize(300, 240)
    scatter.export_button._refresh_menu()
    assert not scatter.export_button.csv_action.isEnabled()                              # no Monte Carlo yet
    scatter.set_data(((1.0, 2.0, True), (-3.5, 4.0, False)), None, "")
    scatter.export_button._refresh_menu()
    assert scatter.export_button.csv_action.isEnabled()


def test_the_touchdown_scatter_exports_its_points(qapp, tmp_path, monkeypatch):
    from orbitopt.flightlab.ui.plot_export import ExportButton
    from orbitopt.flightlab.ui.recovery_panel import DispersionPlot

    scatter = DispersionPlot()
    scatter.resize(300, 240)
    scatter.set_data(((1.0, 2.0, True), (-3.5, 4.0, False)), None, "")
    out = tmp_path / "td.csv"
    monkeypatch.setattr(ExportButton, "file_chooser", staticmethod(_chooser(out)))
    assert scatter.export_button.save_csv() == str(out)
    assert out.read_text() == "east_m,north_m,landed\n1,2,1\n-3.5,4,0\n"
