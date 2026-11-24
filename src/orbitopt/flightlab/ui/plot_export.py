"""Download for every plot: the picture as shown (PNG) and, where the plot has data behind it, the data (CSV).

A small button sits in the top-right corner of each plot (telemetry charts, the touchdown scatter, the engine and
grid-fin cards). Its menu saves the plot as a PNG at twice the screen resolution -- without the button itself in it
-- or, for plots with data, as a CSV in long format (one row per sample: series, body, time, value, unit,
estimated). The pure helpers are importable without Qt.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

CSV_HEADER = ("series", "body", "time_s", "value", "unit", "estimated")


def slug(text: str) -> str:
    """A file name stem from a plot title: 'Dynamic pressure' -> 'dynamic-pressure'."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return s or "plot"


def channel_to_csv(channel) -> str | None:
    """A telemetry channel as CSV text (long format), None when it has no data. ``estimated`` is 1 for a derived
    series (a value computed from other columns, labelled '(est.)' on the plot)."""
    series = [s for s in getattr(channel, "series", ()) if getattr(s, "points", None)]
    if not series:
        return None
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    for s in series:
        for t, v in s.points:
            w.writerow((s.label, s.body_id, f"{float(t):.6g}", f"{float(v):.10g}", s.unit, int(bool(getattr(s, "derived", False)))))
    return buf.getvalue()


def rows_to_csv(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    """Any table as CSV text (the touchdown scatter uses this)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


try:
    from PySide6.QtCore import QPoint, QStandardPaths, Qt
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QFileDialog, QMenu, QToolButton, QWidget
except Exception:                                          # pragma: no cover
    QToolButton = object  # type: ignore[assignment,misc]

BUTTON_SIZE = 16
BUTTON_MARGIN = 2


def default_directory() -> str:
    where = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
    return where or str(Path.home())


def default_chooser(parent, caption: str, default_name: str, filt: str) -> str:
    """The file dialog: a path, or '' when the user cancels."""
    path, _ = QFileDialog.getSaveFileName(parent, caption, str(Path(default_directory()) / default_name), filt)
    return path or ""


def render_png(widget, path, *, scale: float = 2.0, hide: Sequence = (), background: str | None = None) -> None:
    """Render ``widget`` as it is on screen to a PNG at ``scale`` times its size, with ``hide`` (the export
    button) left out."""
    from orbitopt.viz.theme import PANEL

    hidden = [w for w in hide if w.isVisible()]
    for w in hidden:
        w.hide()
    try:
        image = QImage(max(1, int(widget.width() * scale)), max(1, int(widget.height() * scale)), QImage.Format_ARGB32)
        image.setDevicePixelRatio(scale)
        image.fill(QColor(background or PANEL))
        widget.render(image)
    finally:
        for w in hidden:
            w.show()
    if not image.save(str(path), "PNG"):
        raise OSError(f"could not write {path}")


class ExportButton(QToolButton):                           # type: ignore[misc,valid-type]
    """The download button of one plot. ``title`` names the file; ``csv`` returns the plot's data as CSV text
    (None or absent: the plot has none, the item is greyed out)."""

    #: replaceable (tests, headless runs): (parent, caption, default name, filter) -> path or ''
    file_chooser: Callable[..., str] = staticmethod(default_chooser)

    def __init__(self, target, title: Callable[[], str], csv: Callable[[], str | None] | None = None):
        super().__init__(target)
        self._target, self._title, self._csv = target, title, csv
        self.setObjectName("flightLabPlotExport")
        self.setText("⤓")                              # a down arrow to a bar: download
        self.setToolTip("Download this plot")
        self.setCursor(Qt.PointingHandCursor)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
        self.setStyleSheet("QToolButton { background: transparent; color: #7a7a7a; border: none; font-size: 12px; }"
                           "QToolButton:hover { color: #ffffff; background: #262626; border-radius: 3px; }"
                           "QToolButton::menu-indicator { image: none; width: 0; }")
        menu = QMenu(self)
        self.png_action = menu.addAction("Save image (PNG)…", self.save_png)
        self.csv_action = menu.addAction("Save data (CSV)…", self.save_csv)
        menu.aboutToShow.connect(self._refresh_menu)
        self.setMenu(menu)
        self.reposition()
        self.raise_()

    def _refresh_menu(self) -> None:
        self.csv_action.setEnabled(self._csv is not None and bool(self._csv()))

    def reposition(self) -> None:
        """Top-right corner of the plot, inside the right margin the charts keep clear."""
        self.move(self._target.width() - BUTTON_SIZE - BUTTON_MARGIN, BUTTON_MARGIN)
        self.raise_()

    def save_png(self) -> str | None:
        name = f"{slug(self._title())}.png"
        path = self.file_chooser(self._target, "Save plot as image", name, "PNG image (*.png)")
        if not path:
            return None
        path = str(path if str(path).lower().endswith(".png") else f"{path}.png")
        render_png(self._target, path, hide=(self,))
        return path

    def save_csv(self) -> str | None:
        text = self._csv() if self._csv is not None else None
        if not text:
            return None
        name = f"{slug(self._title())}.csv"
        path = self.file_chooser(self._target, "Save plot data", name, "CSV (*.csv)")
        if not path:
            return None
        path = str(path if str(path).lower().endswith(".csv") else f"{path}.csv")
        Path(path).write_text(text, encoding="utf-8")
        return path
