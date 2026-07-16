"""Programmatic app icon for Mission Control.

An inclined orbit with a planet and a spacecraft marker, drawn in the app's
own deep-space palette (theme.py) so the Dock/window icon matches the window
chrome exactly. Rendered with QPainter at several sizes into a QIcon rather
than shipping a hand-drawn binary asset, so it always tracks the palette and
stays crisp from the 16px menu-bar glyph up to the Dock tile. ``export_png``
writes a single high-res PNG for anywhere a file is needed (README, packaging).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)

from orbitopt.viz.theme import ACCENT, BG_VOID, INK_PRIMARY, PANEL_RAISED

_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def render_app_icon(size: int) -> QPixmap:
    """Draw the icon at ``size`` x ``size`` px and return it as a QPixmap."""
    s = float(size)
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)

    # -- rounded-square tile: vertical gradient + a soft amber glow ----------
    corner = s * 0.22
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0, 0, s, s), corner, corner)
    p.setClipPath(clip)

    tile = QLinearGradient(0, 0, 0, s)
    tile.setColorAt(0.0, QColor(PANEL_RAISED))
    tile.setColorAt(1.0, QColor(BG_VOID))
    p.fillRect(QRectF(0, 0, s, s), QBrush(tile))

    glow = QRadialGradient(s * 0.5, s * 0.46, s * 0.62)
    c_in = QColor(ACCENT)
    c_in.setAlphaF(0.16)
    c_out = QColor(ACCENT)
    c_out.setAlphaF(0.0)
    glow.setColorAt(0.0, c_in)
    glow.setColorAt(1.0, c_out)
    p.fillRect(QRectF(0, 0, s, s), QBrush(glow))

    # -- a scatter of faint stars -------------------------------------------
    p.setPen(Qt.NoPen)
    for fx, fy, fa, fr in (
        (0.16, 0.18, 0.70, 0.010),
        (0.83, 0.15, 0.55, 0.008),
        (0.74, 0.73, 0.60, 0.009),
        (0.24, 0.82, 0.45, 0.007),
        (0.90, 0.50, 0.40, 0.006),
        (0.12, 0.56, 0.35, 0.006),
    ):
        star = QColor(INK_PRIMARY)
        star.setAlphaF(fa)
        p.setBrush(star)
        p.drawEllipse(QPointF(s * fx, s * fy), s * fr, s * fr)

    cx, cy = s * 0.5, s * 0.52
    a, b = s * 0.37, s * 0.155  # orbit semi-axes
    orbit_rect = QRectF(cx - a, cy - b, 2 * a, 2 * b)
    ring_w = max(1.0, s * 0.022)

    ring = QPen(QColor(ACCENT), ring_w)
    ring.setCapStyle(Qt.RoundCap)
    halo = QColor(ACCENT)
    halo.setAlphaF(0.18)
    ring_glow = QPen(halo, ring_w * 2.6)
    ring_glow.setCapStyle(Qt.RoundCap)

    # -- orbit ellipse (behind the planet) ----------------------------------
    p.save()
    p.translate(cx, cy)
    p.rotate(-26)
    p.translate(-cx, -cy)
    p.setBrush(Qt.NoBrush)
    p.setPen(ring_glow)
    p.drawEllipse(orbit_rect)
    p.setPen(ring)
    p.drawEllipse(orbit_rect)
    p.restore()

    # -- planet -------------------------------------------------------------
    pr = s * 0.145
    body = QRadialGradient(cx - pr * 0.4, cy - pr * 0.45, pr * 1.8)
    body.setColorAt(0.0, QColor(ACCENT).lighter(145))
    body.setColorAt(0.55, QColor(ACCENT))
    body.setColorAt(1.0, QColor(ACCENT).darker(180))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(body))
    p.drawEllipse(QPointF(cx, cy), pr, pr)

    rim = QColor(ACCENT).lighter(165)
    rim.setAlphaF(0.55)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(rim, max(1.0, s * 0.006)))
    p.drawEllipse(QPointF(cx, cy), pr, pr)

    # -- spacecraft marker + short trail, on the near arc (in front) --------
    p.save()
    p.translate(cx, cy)
    p.rotate(-26)
    p.translate(-cx, -cy)
    p.setPen(Qt.NoPen)
    for i, deg in enumerate((72.0, 84.0, 96.0)):
        t = math.radians(deg)
        tx, ty = cx + a * math.cos(t), cy + b * math.sin(t)
        trail = QColor(INK_PRIMARY)
        trail.setAlphaF(0.45 - i * 0.13)
        p.setBrush(trail)
        rr = s * (0.015 - i * 0.003)
        p.drawEllipse(QPointF(tx, ty), rr, rr)

    mt = math.radians(58.0)
    mx, my = cx + a * math.cos(mt), cy + b * math.sin(mt)
    halo_m = QColor(INK_PRIMARY)
    halo_m.setAlphaF(0.30)
    p.setBrush(halo_m)
    p.drawEllipse(QPointF(mx, my), s * 0.046, s * 0.046)
    p.setBrush(QColor(INK_PRIMARY))
    p.drawEllipse(QPointF(mx, my), s * 0.026, s * 0.026)
    p.restore()

    p.end()
    return px


def app_icon() -> QIcon:
    """A QIcon carrying the icon pre-rendered at every size Qt may ask for."""
    icon = QIcon()
    for size in _ICON_SIZES:
        icon.addPixmap(render_app_icon(size))
    return icon


def export_png(path: str, size: int = 1024) -> str:
    """Write a single high-res PNG of the icon to ``path``. Requires a live
    QGuiApplication (QPixmap needs one)."""
    render_app_icon(size).save(path, "PNG")
    return path
