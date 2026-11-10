"""Capture the README / docs screenshots of Mission Control.

    PYTHONPATH=src python scripts/capture_screenshots.py [--out docs/assets] [--width 1920]

Each shot opens the real app on a built-in mission (served from the scene
cache when it is warm -- a cold Mars compute takes many minutes), sets the
view up, and saves a PNG scaled to ``--width`` pixels.

A window ``grab()`` alone comes back black where the 3D view is: the VTK
surface is a native OpenGL context Qt's compositor does not capture (see
docs/engineering-notes.md). So each shot is a composite -- the Qt chrome from
``grab()``, the 3D view from ``plotter.screenshot()`` pasted into its rect.
An OS-level screen capture is deliberately not used: it can pick up whatever
window has focus instead of the app.

Needs a display (VTK's OpenGL context fails under QT_QPA_PLATFORM=offscreen).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

WINDOW_SIZE = (1600, 900)  # logical px; a 2x display doubles it


def _pump(app, predicate, timeout=1800.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("timed out waiting for the app")
        app.processEvents()


def _settle(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()


def _composite(window):
    """The window as an image, with the live 3D view pasted over the black
    rectangle grab() leaves for it."""
    import numpy as np
    from PySide6.QtCore import QPoint, QRectF
    from PySide6.QtGui import QImage, QPainter

    image = window.grab().toImage()  # keeps the device pixel ratio...
    pixels = np.ascontiguousarray(window.plotter.screenshot(return_img=True)[..., :3])
    h, w = pixels.shape[:2]
    view = QImage(pixels.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    origin = window.plotter.mapTo(window, QPoint(0, 0))
    painter = QPainter(image)  # ...so this paints in logical coordinates
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.drawImage(QRectF(origin.x(), origin.y(), window.plotter.width(), window.plotter.height()), view)
    painter.end()
    return image


def _open_mission(app, window, title_prefix):
    sidebar = window.sidebar
    row = next(i for i in range(sidebar.count()) if sidebar.mission_list.item(i).text().startswith(title_prefix))
    sidebar.select_row(row)
    _settle(app, 0.2)
    _pump(app, lambda: not window._loading)
    _settle(app, 1.5)


def shot_mars(app, w):
    """Heliocentric Mars transfer mid-cruise: candidate arcs, the launch-window
    chart with the candidates on it, the inspector's candidate card."""
    _open_mission(app, w, "Mars")
    w.view_strip.frame_button(None).click()
    w._view_top()
    w.plotter.reset_camera()
    w.plotter.camera.zoom(1.45)  # the orbits fill the view, not just its middle
    tl = w.state.timeline
    w.state.set_time(tl["min"] + 0.45 * (tl["max"] - tl["min"]))
    w.chart_panel.setCurrentWidget(w.chart_panel.porkchop)
    w.right_tabs.setCurrentWidget(w.inspector)
    w.plotter.render()
    _settle(app, 1.0)


def shot_artemis(app, w):
    """Artemis II centered on the Moon at closest approach: the flyby phase
    lit on the timeline, the distance chart, and the Verification tab."""
    _open_mission(app, w, "Artemis")
    closest = next(e for e in w.state.events if "approach" in e["label"].lower())
    w.state.set_time(closest["time"])
    _settle(app, 0.3)
    w.view_strip.frame_button("moon").click()
    _settle(app, 0.5)
    w.chart_panel.setCurrentWidget(w.chart_panel.distance)
    w._show_verification_tab()
    w.plotter.render()
    _settle(app, 1.0)


def shot_goes(app, w):
    """The GOES GTO-to-GEO raising campaign mid-way through its burns."""
    _open_mission(app, w, "GOES")
    burns = [e for e in w.state.events if "vector" in e]
    w.state.set_time(burns[len(burns) // 2]["time"] if burns else w.state.timeline["max"] * 0.5)
    w.chart_panel.setCurrentWidget(w.chart_panel.distance)
    w.right_tabs.setCurrentWidget(w.inspector)
    w.plotter.camera.zoom(1.8)
    w.plotter.render()
    _settle(app, 1.0)


def shot_starship(app, w):
    """Starship Flight 15 at the predicted return pass: the ship over the
    Gulf approach, Starbase on the rotating Earth, and the distance-to-
    Starbase chart with every pass as a dip."""
    _open_mission(app, w, "Starship")
    ret = next(e for e in w.state.events if e["label"] == "Return pass (predicted)")
    w.state.set_time(ret["time"])
    _settle(app, 0.3)
    # Look straight down on the ship (and so on its approach to Starbase)
    # rather than from the default iso corner, which faces the far side.
    import numpy as np

    pos = np.asarray(w.renderer.body_display_position("spacecraft", w.state.time), dtype=float)
    cam = w.plotter.camera
    cam.focal_point = tuple(0.55 * pos)
    cam.position = tuple(pos / np.linalg.norm(pos) * 6378.0 * 2.6)
    cam.up = (0.0, 0.0, 1.0)
    w.chart_panel.setCurrentWidget(w.chart_panel.distance)
    w.right_tabs.setCurrentWidget(w.inspector)
    w.plotter.render()
    _settle(app, 1.0)


SHOTS = {
    "mission-control-mars.png": shot_mars,
    "mission-control-artemis.png": shot_artemis,
    "mission-control-goes.png": shot_goes,
    "mission-control-starship.png": shot_starship,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="docs/assets", type=Path)
    parser.add_argument("--width", default=1920, type=int, help="output width in pixels")
    parser.add_argument("--only", nargs="*", choices=sorted(SHOTS), help="capture just these")
    args = parser.parse_args(argv)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MissionControlWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    _pump(app, lambda: not window._loading)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for name, setup in SHOTS.items():
            if args.only and name not in args.only:
                continue
            setup(app, window)
            image = _composite(window).scaledToWidth(args.width, Qt.SmoothTransformation)
            path = args.out / name
            image.save(str(path))
            print(f"wrote {path} ({image.width()}x{image.height()})", flush=True)
    finally:
        window.plotter.close()
        window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
