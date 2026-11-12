"""The embedded PyVista/VTK viewport widget used by Mission Control."""
from __future__ import annotations

import math

from PySide6.QtCore import QTimer
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkRenderWindow


def _draws_into_qopenglwidget() -> bool:
    """pyvistaqt >= 0.13 renders into a QOpenGLWidget; 0.12 (the locked
    environment) embeds VTK's own native window in a plain QWidget."""
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:
        return False
    return issubclass(QtInteractor, QOpenGLWidget)


class DebouncedInteractor(QtInteractor):
    """QtInteractor with one draw per event-loop turn (see render()), a
    deferred repaint while a native window is being resized, and a near
    plane that follows the camera (see _apply_dynamic_clipping).

    Resize, native window (pyvistaqt 0.12): its resizeEvent renders
    synchronously every time. On Windows a live edge-drag delivers one per
    intermediate size inside the OS's modal resize loop, which does not hand
    control back until each repaint finishes -- measured ~45 ms per size
    (Qt/VTK resize bookkeeping, not the render), the window reading as frozen
    for the whole drag. So the VTK window's size/DPI follow every event
    (cheap) and the repaint waits until resizing has been idle briefly.

    Resize, QOpenGLWidget (pyvistaqt >= 0.13): left to Qt. Its resizeEvent
    resizes the widget's framebuffer; overriding it (as above) left the
    framebuffer at its startup size, and when the view shrank (opening
    Starship halves its height) frames landed in the bottom of the taller
    buffer and Qt scaled it onto the widget -- the scene squashed into the
    bottom half, the top half black. QOpenGLWidget repaints through update(),
    already at most once per event-loop turn.
    """

    _RESIZE_SETTLE_MS = 120
    _DEFER_RESIZE = not _draws_into_qopenglwidget()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resize_settle_timer = QTimer(self)
        self._resize_settle_timer.setSingleShot(True)
        self._resize_settle_timer.setInterval(self._RESIZE_SETTLE_MS)
        self._resize_settle_timer.timeout.connect(self._render_after_resize_settles)
        self._install_dynamic_clipping()

    def resizeEvent(self, event):  # noqa: N802 -- Qt override signature
        if not self._DEFER_RESIZE:
            super().resizeEvent(event)
            return
        if self._RenderWindow is None:
            return
        # Qt's own ratio: pyvistaqt 0.13 dropped the private _getPixelRatio
        # helper this used to call (0.12 computed the same screen ratio).
        scale = self.devicePixelRatioF()
        w = int(round(scale * self.width()))
        h = int(round(scale * self.height()))
        self._RenderWindow.SetDPI(int(round(72 * scale)))
        vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self._resize_settle_timer.start()

    def _render_after_resize_settles(self):
        # Guard against firing after the widget is closed: pyvistaqt swaps _Iren
        # for a stub with no Render() during teardown, so a still-pending settle
        # timer would otherwise raise on quit (Cmd+Q).
        iren = self._Iren
        if iren is not None and hasattr(iren, "Render"):
            iren.Render()

    # One draw per event-loop turn. pyvistaqt's render() draws once per call
    # (on macOS via a new thread each time), and a playback tick or a camera
    # step asks several times -- trails, labels, arrow, tracking, the
    # timeline. Each draw waits for the display's vsync (~8 ms at 120 Hz),
    # so a tick cost 3-4 refresh intervals: measured 29-42 ms per tick and up
    # to 46 ms per camera step before this, playback topping out near 25 fps.
    def render(self):
        self._rendered = True  # pyvista's own "has rendered" bookkeeping
        if getattr(self, "_render_pending", False):
            return
        self._render_pending = True
        QTimer.singleShot(0, self._flush_render)

    def _flush_render(self):
        self._render_pending = False
        if getattr(self, "_closed", False) or self.render_window is None:
            return
        try:
            self._render()
        except (RuntimeError, AttributeError):  # widget torn down while the request was queued
            pass

    # No keyboard for VTK. Its interactor style binds plain letters to
    # scene-wide changes: w turns every actor to wireframe (the textured
    # Earth became a see-through grid), s back to surfaces, a makes the mouse
    # drag the Earth itself instead of the camera, e/q close the render
    # window, 3 toggles stereo. Every Mission Control key is a window-level
    # Qt shortcut, and mouse modifiers (Shift-drag pan) come with the mouse
    # events, so the keys go back up to Qt instead.
    def keyPressEvent(self, event):  # noqa: N802 -- Qt override signature
        event.ignore()

    def keyReleaseEvent(self, event):  # noqa: N802 -- Qt override signature
        event.ignore()

    def render_now(self):
        """Draw immediately (screenshots, tests reading pixels)."""
        self._render_pending = False
        self._render()

    # KSP-style dynamic depth range. VTK's default fit-to-scene clipping can't
    # pull the near plane closer than a fixed fraction of the WHOLE scene's far
    # plane, so in a scene tens of thousands of km across (a GEO ring) or many
    # AU (the solar system), zooming in to inspect a local stretch of orbit
    # shoves the geometry inside the near plane and it just vanishes -- capping
    # how far you can usefully zoom. Instead we set the near plane proportional
    # to the live camera-to-focal distance, so you can keep zooming in
    # arbitrarily close (tracking-station style) with the focus staying visible,
    # while the far plane still reaches the scene's far corner so distant bodies
    # keep drawing. This has to catch BOTH ways the range gets reset: pyvista's
    # wheel-zoom calls reset_camera_clipping_range() (overridden below), while
    # track_body / zoom_to_body / reset_camera just move the camera and render,
    # so we also recompute on the camera's own ModifiedEvent -- setting the
    # range there, before the render, means VTK's fit never runs.
    _CLIP_NEAR_FRACTION = 1.0e-3

    def _install_dynamic_clipping(self) -> None:
        self._clipping_busy = False
        try:
            self.camera.AddObserver("ModifiedEvent", lambda *_a: self._apply_dynamic_clipping())
        except Exception:  # noqa: BLE001 -- no camera yet is fine; the override still covers zoom
            pass

    def _apply_dynamic_clipping(self) -> None:
        if getattr(self, "_clipping_busy", False):
            return  # our own SetClippingRange re-fires ModifiedEvent -- don't recurse
        self._clipping_busy = True
        try:
            camera = self.camera
            fx, fy, fz = camera.focal_point
            px, py, pz = camera.position
            distance = math.sqrt((px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2)
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            if distance <= 0.0 or not all(
                math.isfinite(b) for b in (xmin, xmax, ymin, ymax, zmin, zmax)
            ):
                super().reset_camera_clipping_range()
                return
            far_corner = max(
                math.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)
            )
            near = max(distance * self._CLIP_NEAR_FRACTION, 1.0e-9)
            far = far_corner * 1.05 + near
            if far <= near:
                super().reset_camera_clipping_range()
                return
            camera.SetClippingRange(near, far)
        except Exception:  # noqa: BLE001 -- clipping is cosmetic; never break a render over it
            super().reset_camera_clipping_range()
        finally:
            self._clipping_busy = False

    def reset_camera_clipping_range(self):  # noqa: N802 -- overrides pyvista's snake_case method
        self._apply_dynamic_clipping()
