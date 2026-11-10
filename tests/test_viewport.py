"""DebouncedInteractor's resize path (orbitopt.viz.viewport), on stand-ins
for the Qt widget and VTK window so it runs without a display."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pyvistaqt")

from orbitopt.viz.viewport import DebouncedInteractor  # noqa: E402


class _CocoaWindow:
    """Records calls like a vtkCocoaRenderWindow (base-class SetSize, DPI,
    the context's UpdateContext)."""

    def __init__(self, context=1):
        self.calls, self.context = [], context

    def SetDPI(self, dpi):  # noqa: N802 -- VTK names
        self.calls.append(("dpi", dpi))

    def SetSize(self, w, h):  # noqa: N802
        self.calls.append(("size", w, h))

    def GetContextId(self):  # noqa: N802
        return self.context

    def UpdateContext(self):  # noqa: N802
        self.calls.append(("update",))


def _widget(rw):
    iren = SimpleNamespace(SetSize=lambda w, h: None, ConfigureEvent=lambda: None,
                           Render=lambda: rw.calls.append(("render",)))
    w = SimpleNamespace(_RenderWindow=rw, _Iren=iren, devicePixelRatioF=lambda: 2.0,
                        width=lambda: 1067, height=lambda: 414,
                        _resize_settle_timer=SimpleNamespace(start=lambda: rw.calls.append(("settle",))))
    w._update_gl_surface = lambda: DebouncedInteractor._update_gl_surface(w)
    return w


def test_resize_updates_the_gl_surface_before_any_draw(monkeypatch):
    """Opening Starship halves the view's height. Without UpdateContext the
    macOS context kept its old drawable and frames showed squashed into the
    bottom half, the top half black."""
    import orbitopt.viz.viewport as vp

    monkeypatch.setattr(vp, "vtkRenderWindow", SimpleNamespace(SetSize=lambda rw, w, h: rw.SetSize(w, h)))
    rw = _CocoaWindow()
    w = _widget(rw)
    DebouncedInteractor.resizeEvent(w, None)
    assert ("size", 2134, 828) in rw.calls
    assert rw.calls.index(("update",)) > rw.calls.index(("size", 2134, 828))
    DebouncedInteractor._render_after_resize_settles(w)
    assert rw.calls[-2:] == [("update",), ("render",)]


def test_no_context_yet_or_no_cocoa_window_is_left_alone():
    rw = _CocoaWindow(context=0)  # not drawn yet: no context to update
    DebouncedInteractor._update_gl_surface(SimpleNamespace(_RenderWindow=rw))
    assert rw.calls == []

    class _X11Window:  # no UpdateContext on other platforms
        def GetContextId(self):  # noqa: N802
            return 1

    DebouncedInteractor._update_gl_surface(SimpleNamespace(_RenderWindow=_X11Window()))
    DebouncedInteractor._update_gl_surface(SimpleNamespace(_RenderWindow=None))
