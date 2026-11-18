"""Programmatic app icon and logo mark for Mission Control.

One motif -- a bold tilted orbit, broken where the spacecraft crosses it, with
the body it circles -- drawn flat in the app's black/white theme (theme.py).
The Dock/window icon is that mark in white on a black rounded tile; the top-bar
logo is the same mark alone. Rendered with QPainter at several sizes into a
QIcon rather than shipping a hand-drawn binary asset, so it always tracks the
palette and stays crisp from the 16px menu-bar glyph up to the Dock tile.
``export_png`` writes a single high-res PNG for anywhere a file is needed.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from orbitopt.viz.theme import BG_VOID, INK_PRIMARY

_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)  # logical sizes; 2x bitmaps are added too


def _paint_mark(p: QPainter, s: float, ink: QColor) -> None:
    """The mark, centred on the painter's origin, filling a ``s`` x ``s`` box."""
    p.save()
    p.rotate(-38)
    rx, ry = s * 0.45, s * 0.3
    pen = QPen(ink, max(1.0, s * 0.1))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    # Qt arcs: 1/16 degree, counter-clockwise from 3 o'clock. Leave a gap
    # around the spacecraft at the right-hand end of the orbit.
    gap = 64
    p.drawArc(QRectF(-rx, -ry, 2 * rx, 2 * ry), int(gap / 2 * 16), int((360 - gap) * 16))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(ink))
    p.drawEllipse(QPointF(0, 0), s * 0.12, s * 0.12)                       # the body
    p.drawEllipse(QPointF(rx, 0), s * 0.085, s * 0.085)                    # the spacecraft
    p.restore()


def render_app_icon(size: int, dpr: float = 1.0) -> QPixmap:
    """The icon at ``size`` x ``size`` logical px (``size * dpr`` device
    px): the mark in white on a black rounded tile with a hairline edge."""
    s = float(size)
    side = max(1, round(size * dpr))
    px = QPixmap(side, side)
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)
    corner = s * 0.22
    tile = QPainterPath()
    inset = max(0.5, s * 0.004)
    tile.addRoundedRect(QRectF(inset, inset, s - 2 * inset, s - 2 * inset), corner, corner)
    p.setBrush(QColor(BG_VOID))
    edge = QColor(INK_PRIMARY)
    edge.setAlphaF(0.22)
    p.setPen(QPen(edge, max(1.0, s * 0.008)))
    p.drawPath(tile)
    p.translate(s / 2, s / 2)
    _paint_mark(p, s * 0.62, QColor(INK_PRIMARY))
    p.end()
    return px


def app_icon() -> QIcon:
    """A QIcon carrying the icon pre-rendered at every size Qt may ask for."""
    icon = QIcon()
    for size in _ICON_SIZES:
        # Both the 1x and the 2x bitmap of each size: without the 2x one the
        # Dock / a Retina title bar upscaled the 1x bitmap and looked soft.
        icon.addPixmap(render_app_icon(size))
        icon.addPixmap(render_app_icon(size, 2.0))
    return icon


def render_dock_icon(size: int = 1024) -> QPixmap:
    """The icon as macOS wants it for the Dock: the tile inset to ~80% of the
    canvas (Apple's icon grid leaves the rest as margin and shadow room), so
    it sits at the same visual size as the other Dock icons."""
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    inner = round(size * 0.8)
    off = (size - inner) // 2
    p.drawPixmap(off, off, render_app_icon(inner))
    p.end()
    return px


def set_macos_dock_icon() -> bool:
    """Hand the Dock a full-resolution 1024 px image directly. Qt's own
    QIcon -> NSImage conversion hands it a small bitmap that macOS upscales,
    which is what made the Dock icon soft. No-op (False) off macOS or without
    PyObjC."""
    import sys

    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSData
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice

        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.WriteOnly)
        render_dock_icon(1024).save(buf, "PNG")
        buf.close()
        image = NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(bytes(data), len(data)))
        if image is None:
            return False
        NSApplication.sharedApplication().setApplicationIconImage_(image)
        return True
    except Exception:  # noqa: BLE001 -- a cosmetic nicety must never stop the app starting
        return False


def export_png(path: str, size: int = 1024) -> str:
    """Write a single high-res PNG of the icon to ``path``. Requires a live
    QGuiApplication (QPixmap needs one)."""
    render_app_icon(size).save(path, "PNG")
    return path


def render_logo_mark(size: int, color: str = INK_PRIMARY, dpr: float = 1.0) -> QPixmap:
    """The flat, single-colour mark alone, for the top bar: no tile, gradient
    or glow, so it reads as type next to the wordmark. Drawn at ``size``
    logical pixels x ``dpr`` device pixels, so it stays crisp on HiDPI screens
    instead of being scaled down from a larger bitmap."""
    side = max(1, round(size * dpr))
    px = QPixmap(side, side)
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.translate(size / 2, size / 2)
    _paint_mark(p, float(size), QColor(color))
    p.end()
    return px
