"""Small, programmatically-drawn toolbar icons for Mission Control.

Same idea as icon.py's window/taskbar icon -- QPainter onto a transparent
QPixmap, not a shipped image and not a Unicode symbol used as button text
(a handful of buttons used to render "*" chevrons, a play/pause triangle,
a stopwatch, etc. as plain text; several of those code points have an emoji
presentation on some platforms/fonts, and font substitution in general makes
a text-glyph "icon" look different machine to machine). Drawing them here
means they always track this app's own palette and stay crisp at exactly
the pixel size the button asks for -- no multi-size iteration like icon.py's
app_icon() needs, since a toolbar icon is only ever shown at one fixed size.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from orbitopt.viz.theme import ACCENT, INK_MUTED, INK_SECONDARY


def _new_painter(size: int) -> tuple[QPixmap, QPainter]:
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)
    return px, p


def _icon(px: QPixmap, p: QPainter) -> QIcon:
    p.end()
    return QIcon(px)


def chevron_left_icon(color: str = INK_SECONDARY, size: int = 16) -> QIcon:
    return _chevron_icon(color, size, direction=-1)


def chevron_right_icon(color: str = INK_SECONDARY, size: int = 16) -> QIcon:
    return _chevron_icon(color, size, direction=1)


def _chevron_icon(color: str, size: int, direction: int) -> QIcon:
    px, p = _new_painter(size)
    s = float(size)
    pen = QPen(QColor(color), max(1.4, s * 0.11))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    tip_x = s * (0.62 if direction > 0 else 0.38)
    back_x = s * (0.38 if direction > 0 else 0.62)
    path = QPainterPath()
    path.moveTo(back_x, s * 0.28)
    path.lineTo(tip_x, s * 0.5)
    path.lineTo(back_x, s * 0.72)
    p.drawPath(path)
    return _icon(px, p)


def play_icon(color: str = ACCENT, size: int = 16) -> QIcon:
    px, p = _new_painter(size)
    s = float(size)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    tri = QPainterPath()
    tri.moveTo(s * 0.32, s * 0.22)
    tri.lineTo(s * 0.32, s * 0.78)
    tri.lineTo(s * 0.80, s * 0.5)
    tri.closeSubpath()
    p.drawPath(tri)
    return _icon(px, p)


def pause_icon(color: str = ACCENT, size: int = 16) -> QIcon:
    px, p = _new_painter(size)
    s = float(size)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    bar_w = s * 0.16
    radius = bar_w * 0.3
    p.drawRoundedRect(QRectF(s * 0.28, s * 0.22, bar_w, s * 0.56), radius, radius)
    p.drawRoundedRect(QRectF(s * 0.56, s * 0.22, bar_w, s * 0.56), radius, radius)
    return _icon(px, p)


def clock_icon(color: str = INK_MUTED, size: int = 14) -> QIcon:
    """Auto-slow-near-events toggle: a plain clock face, not a stopwatch
    emoji -- a thin ring plus hour/minute hands frozen at a "slowing down"
    ten-past reading."""
    px, p = _new_painter(size)
    s = float(size)
    cx, cy, r = s * 0.5, s * 0.52, s * 0.36
    ring = QPen(QColor(color), max(1.2, s * 0.09))
    ring.setCapStyle(Qt.RoundCap)
    p.setPen(ring)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r, r)

    hand = QPen(QColor(color), max(1.1, s * 0.08))
    hand.setCapStyle(Qt.RoundCap)
    p.setPen(hand)
    p.drawLine(QPointF(cx, cy), QPointF(cx, cy - r * 0.62))       # minute hand, up
    p.drawLine(QPointF(cx, cy), QPointF(cx + r * 0.42, cy + r * 0.24))  # hour hand
    return _icon(px, p)


def keyboard_icon(color: str = INK_MUTED, size: int = 14) -> QIcon:
    """Shortcuts-panel trigger: a simplified keyboard outline with a few
    key dots, not a "?" glyph or emoji."""
    px, p = _new_painter(size)
    s = float(size)
    outline = QPen(QColor(color), max(1.1, s * 0.08))
    outline.setJoinStyle(Qt.RoundJoin)
    p.setPen(outline)
    p.setBrush(Qt.NoBrush)
    rect = QRectF(s * 0.08, s * 0.26, s * 0.84, s * 0.5)
    p.drawRoundedRect(rect, s * 0.08, s * 0.08)

    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    key = s * 0.085
    for row, y in enumerate((rect.top() + s * 0.13, rect.top() + s * 0.32)):
        cols = 4 if row == 0 else 3
        start_x = rect.left() + s * (0.10 if row == 0 else 0.16)
        gap = s * 0.185
        for i in range(cols):
            p.drawRoundedRect(QRectF(start_x + i * gap, y, key, key), key * 0.25, key * 0.25)
    return _icon(px, p)
