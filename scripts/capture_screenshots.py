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

Needs a display (VTK's OpenGL context fails under QT_QPA_PLATFORM=offscreen),
and the network for the Starship landing shot's Earth imagery (EOX tiles,
cached on disk after the first run).
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


def _look_down_at(w, body_ids, height_km, back_km):
    """Camera ``height_km`` above the mean of the bodies' positions, pulled
    ``back_km`` to the south so the view is oblique, north up."""
    import numpy as np

    w._set_tracking(False)
    pts = [np.asarray(w.renderer.body_display_position(b, w.state.time), dtype=float) for b in body_ids]
    c = np.mean(pts, axis=0)
    up = c / np.linalg.norm(c)
    north = np.array([0.0, 0.0, 1.0]) - up * up[2]
    north /= np.linalg.norm(north)
    cam = w.plotter.camera
    cam.focal_point = tuple(c)
    cam.position = tuple(c + up * height_km - north * back_km)
    cam.up = tuple(north)
    w._refresh_view()
    w._resize_maneuver_arrow()


def _wait_for_imagery(app, w, timeout=90.0):
    """Let the EOX tiles for this view download (or come from the disk
    cache) and draw."""
    t0 = time.time()
    _settle(app, 1.0)
    while time.time() - t0 < timeout:
        _settle(app, 0.5)
        if not w.renderer.tiles.cache.busy:
            _settle(app, 1.0)
            if not w.renderer.tiles.cache.busy:
                break
    w.plotter.render()
    _settle(app, 0.5)


def shot_starship_ascent(app, w):
    """Flight 15 four and a half minutes in: Super Heavy on its way back to
    the tower and the ship still climbing, each with its flight-data label,
    and the timeline zoomed to the ascent."""
    _open_mission(app, w, "Starship")
    w.labels_action.setChecked(True)
    w.state.set_time(270.0 / 86400.0)
    _settle(app, 0.3)
    _look_down_at(w, ["spacecraft", "booster"], 900.0, 630.0)
    w.timeline_bar.track.set_view(0.0, 480.0 / 86400.0)
    w.chart_panel.setCurrentWidget(w.chart_panel.distance)
    w.right_tabs.setCurrentWidget(w.inspector)
    _wait_for_imagery(app, w)


def shot_starship_landing(app, w):
    """The ship's landing burn over Starbase, on 10 m Sentinel-2 imagery,
    with its label (height, speed, delta-v, the burn in progress) and the
    timeline zoomed to the last two minutes."""
    _open_mission(app, w, "Starship")
    w.labels_action.setChecked(True)
    burn = next(e for e in w.state.events if e["label"].startswith("Flip"))
    w.state.set_time(burn["time"] + 3.0 / 86400.0)
    _settle(app, 0.3)
    _look_down_at(w, ["spacecraft"], 10.0, 14.0)
    w.timeline_bar.track.zoom_to_event(burn)
    w.chart_panel.setCurrentWidget(w.chart_panel.distance)
    w.right_tabs.setCurrentWidget(w.inspector)
    _wait_for_imagery(app, w)


def shot_shortcuts(app, w):
    """The keyboard-shortcut panel (F1), on its own."""
    w._show_shortcuts_dialog()
    _settle(app, 0.8)
    image = w._shortcuts_dialog.grab().toImage()
    w._shortcuts_dialog.close()
    return image


SHOTS = {
    "mission-control-mars.png": shot_mars,
    "mission-control-artemis.png": shot_artemis,
    "mission-control-goes.png": shot_goes,
    "mission-control-starship.png": shot_starship,
    "mission-control-starship-ascent.png": shot_starship_ascent,
    "mission-control-starship-landing.png": shot_starship_landing,
    "mission-control-shortcuts.png": shot_shortcuts,
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
            own = setup(app, window)  # a shot of another window returns its own image
            if own is not None:
                image = own
            else:
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
