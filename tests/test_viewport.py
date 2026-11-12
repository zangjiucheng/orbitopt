"""The Mission Control 3D view (orbitopt.viz.viewport) after a resize."""
from __future__ import annotations

import ctypes
import ctypes.util
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")

_GL_FRAMEBUFFER = 0x8D40
_GL_COLOR_ATTACHMENT0 = 0x8CE0
_GL_ATTACHMENT_TYPE, _GL_ATTACHMENT_NAME = 0x8CD0, 0x8CD1
_GL_TEXTURE, _GL_TEXTURE_2D = 0x1702, 0x0DE1
_GL_TEXTURE_WIDTH, _GL_TEXTURE_HEIGHT = 0x1000, 0x1001


def _libgl():
    path = ("/System/Library/Frameworks/OpenGL.framework/OpenGL" if sys.platform == "darwin"
            else ctypes.util.find_library("GL"))
    if not path:
        pytest.skip("no OpenGL library to query the framebuffer with")
    return ctypes.CDLL(path)


def _framebuffer_size(view):
    """Size of the texture behind the QOpenGLWidget's framebuffer -- what Qt
    scales onto the widget."""
    gl = _libgl()
    view.makeCurrent()
    try:
        gl.glBindFramebuffer(_GL_FRAMEBUFFER, view.defaultFramebufferObject())
        kind, name = ctypes.c_int(), ctypes.c_int()
        gl.glGetFramebufferAttachmentParameteriv(_GL_FRAMEBUFFER, _GL_COLOR_ATTACHMENT0,
                                                 _GL_ATTACHMENT_TYPE, ctypes.byref(kind))
        if kind.value != _GL_TEXTURE:
            pytest.skip("the widget's framebuffer is not texture-backed here")
        gl.glGetFramebufferAttachmentParameteriv(_GL_FRAMEBUFFER, _GL_COLOR_ATTACHMENT0,
                                                 _GL_ATTACHMENT_NAME, ctypes.byref(name))
        w, h = ctypes.c_int(), ctypes.c_int()
        gl.glBindTexture(_GL_TEXTURE_2D, name.value)
        gl.glGetTexLevelParameteriv(_GL_TEXTURE_2D, 0, _GL_TEXTURE_WIDTH, ctypes.byref(w))
        gl.glGetTexLevelParameteriv(_GL_TEXTURE_2D, 0, _GL_TEXTURE_HEIGHT, ctypes.byref(h))
        return w.value, h.value
    finally:
        view.doneCurrent()


def test_shrinking_the_view_shrinks_its_framebuffer():
    """Opening Starship halves the 3D view's height. A resizeEvent override
    (from the old native-window interactor) skipped QOpenGLWidget's own, so
    the framebuffer kept its old height: frames landed in its bottom half and
    Qt scaled the whole buffer onto the widget -- the scene squashed into the
    bottom half, the top half black."""
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz import viewport
    from orbitopt.viz.viewport import DebouncedInteractor

    if not viewport._draws_into_qopenglwidget():
        pytest.skip("pyvistaqt < 0.13: a native VTK window, no QOpenGLWidget framebuffer")
    app = QApplication.instance() or QApplication([])
    view = DebouncedInteractor()
    try:
        view.resize(400, 400)
        view.show()
        for _ in range(5):
            app.processEvents()
        if view.context() is None or not view.context().isValid():
            pytest.skip("no OpenGL context (headless / software GL)")
        view.resize(400, 200)
        for _ in range(5):
            app.processEvents()
        dpr = view.devicePixelRatioF()
        assert _framebuffer_size(view) == (round(400 * dpr), round(200 * dpr))
    finally:
        view.close()


def test_native_window_resize_defers_the_repaint(monkeypatch):
    """pyvistaqt 0.12 (the locked env): sizes follow every resize event, the
    repaint waits for the settle timer (Windows live edge-drags)."""
    from types import SimpleNamespace

    from orbitopt.viz import viewport

    calls = []
    rw = SimpleNamespace(SetDPI=lambda dpi: calls.append(("dpi", dpi)))
    monkeypatch.setattr(viewport, "vtkRenderWindow",
                        SimpleNamespace(SetSize=lambda _rw, w, h: calls.append(("size", w, h))))
    w = SimpleNamespace(
        _DEFER_RESIZE=True, _RenderWindow=rw, devicePixelRatioF=lambda: 2.0,
        width=lambda: 1067, height=lambda: 414,
        _Iren=SimpleNamespace(SetSize=lambda w, h: None, ConfigureEvent=lambda: None,
                              Render=lambda: calls.append(("render",))),
        _resize_settle_timer=SimpleNamespace(start=lambda: calls.append(("settle",))))
    viewport.DebouncedInteractor.resizeEvent(w, None)
    assert calls == [("dpi", 144), ("size", 2134, 828), ("settle",)]  # no draw yet
    viewport.DebouncedInteractor._render_after_resize_settles(w)
    assert calls[-1] == ("render",)


def test_vtk_letter_keys_do_not_reach_the_scene():
    """VTK's interactor style turned every actor to wireframe on "w" (the
    textured Earth became a see-through grid) and let "a" drag actors
    around; the 3D view now passes keys back to Qt instead."""
    import pyvista as pv
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.viewport import DebouncedInteractor

    app = QApplication.instance() or QApplication([])
    view = DebouncedInteractor()
    try:
        actor = view.add_mesh(pv.Sphere())
        view.show()
        app.processEvents()
        surface = actor.GetProperty().GetRepresentation()
        for key in (Qt.Key_W, Qt.Key_A, Qt.Key_3):
            QTest.keyClick(view, key)
        app.processEvents()
        assert actor.GetProperty().GetRepresentation() == surface  # not wireframe
        assert not view.render_window.GetStereoRender()
    finally:
        view.close()
