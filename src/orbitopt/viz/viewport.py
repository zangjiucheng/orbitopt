"""The embedded PyVista/VTK viewport widget used by Mission Control."""
from __future__ import annotations

import math

from PySide6.QtCore import QTimer
from pyvistaqt import QtInteractor


class DebouncedInteractor(QtInteractor):
    """QtInteractor with one draw per event-loop turn (see render()) and a
    near plane that follows the camera (see _apply_dynamic_clipping).

    Resizing is left to pyvistaqt's QOpenGLWidget. An override from the
    older native-window interactor used to replace resizeEvent to defer
    repaints during a live Windows edge-drag. On QOpenGLWidget that skipped
    the framebuffer resize: after the view shrank (opening Starship halves
    its height) frames were copied into the old, taller framebuffer and
    shown scaled, squashed into the bottom half with the top half black.
    QOpenGLWidget repaints through update(), already at most once per
    event-loop turn.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_dynamic_clipping()

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
