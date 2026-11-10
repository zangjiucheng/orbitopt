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

    from orbitopt.viz.viewport import DebouncedInteractor

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
