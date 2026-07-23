"""Qt stylesheet for the Mission Control app -- "OrbitOpt Pixel Mission
Control": a dark, canvas-first tracking-station chrome (dense info panels,
a time-warp control strip, orbit-colored readouts) with a restrained retro
aerospace / pixel-console visual language layered on top -- hard 0-3px
edges instead of soft rounded cards, a monospace type system throughout,
and small semantic status badges instead of prose. The 3D trajectory view
itself stays untouched by any of this (see scene_renderer.py); this module
only ever styles the Qt chrome around it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Design tokens. Names are kept stable across the previous ("KSP tracking
# station") palette so icons.py/app.py's existing imports don't need to
# change -- only the *values* moved to the pixel-mission-control palette.
# New tokens (SURFACE_HOVER, BORDER_SOFT, SUCCESS/WARNING/DANGER/INFO) are
# additive.
# ---------------------------------------------------------------------------
BG_VOID = "#05070A"
PANEL = "#0B0F14"
PANEL_RAISED = "#10161D"
SURFACE_HOVER = "#151D26"
HAIRLINE = "#27313B"
BORDER_SOFT = "#18212A"
INK_PRIMARY = "#DCE7EE"
INK_SECONDARY = "#91A2AE"
INK_MUTED = "#56656F"
ACCENT = "#65D5C8"
ACCENT_DIM = "#17332F"
ACCENT_MID = "#2D746D"   # accent on a dark wash (e.g. selected-item text-on-fill contexts)

# Semantic status colors -- every state badge, mission-list status dot, and
# body/maneuver highlight maps to exactly one of these, never an arbitrary
# per-trajectory color (see spec: "颜色必须具有语义").
SUCCESS = "#79C267"   # verified / loaded
WARNING = "#D5A84D"   # partial / caution
DANGER = "#D46A6A"    # failed
INFO = "#70A5E8"      # running / neutral-active

MEASURE = INFO
MEASURE_DIM = "#16233A"

# Monospace stack: prefers a few widely-available programmer/geometric
# monospace faces (spec 4.3), falls back to the platform's own monospace --
# no font binaries are bundled with the app.
FONT_MONO = (
    '"JetBrains Mono", "IBM Plex Mono", "Space Mono", "SF Mono", '
    '"Cascadia Code", "Consolas", monospace'
)

STYLESHEET = f"""
* {{
    font-family: {FONT_MONO};
    color: {INK_PRIMARY};
}}

QMainWindow, QWidget#centralWidget {{
    background: {BG_VOID};
}}

QWidget#sidebar, QWidget#infoPanel, QWidget#timelineBar, QWidget#titleBar {{
    background: {PANEL};
    border: none;
}}

QWidget#sidebar {{
    border-right: 1px solid {HAIRLINE};
}}
QWidget#infoPanel {{
    border-left: 1px solid {HAIRLINE};
}}
QWidget#timelineBar {{
    border-top: 1px solid {HAIRLINE};
}}
QWidget#titleBar {{
    border-bottom: 1px solid {HAIRLINE};
}}

QLabel#appTitle {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: {ACCENT};
    padding: 2px 4px;
}}

QLabel#sceneTitle {{
    font-size: 15px;
    font-weight: 650;
    letter-spacing: 0.02em;
}}
QLabel#sceneSubtitle {{
    font-size: 11px;
    color: {INK_SECONDARY};
}}

QLabel#sectionHeader {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {INK_MUTED};
    padding: 10px 4px 4px 4px;
    text-transform: uppercase;
}}

QListWidget {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 12px;
}}
QListWidget::item {{
    padding: 8px 9px;
    border-radius: 0px;
    border-left: 2px solid transparent;
    margin: 1px 4px;
    color: {INK_SECONDARY};
}}
QListWidget::item:hover {{
    background: {SURFACE_HOVER};
    color: {INK_PRIMARY};
}}
QListWidget::item:selected {{
    background: {ACCENT_DIM};
    border-left: 2px solid {ACCENT};
    color: {ACCENT};
    font-weight: 650;
}}

QPushButton {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 600;
    color: {INK_SECONDARY};
}}
QPushButton:hover {{
    color: {INK_PRIMARY};
    border-color: {ACCENT};
}}
QPushButton:pressed, QPushButton:checked {{
    background: {ACCENT_DIM};
    color: {ACCENT};
    border-color: {ACCENT};
}}

QPushButton#playButton {{
    border-radius: 2px;
    min-width: 32px;
    max-width: 32px;
    min-height: 32px;
    max-height: 32px;
    font-size: 13px;
    padding: 0;
}}

QPushButton#panelCollapse {{
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    padding: 0;
    font-size: 11px;
    background: transparent;
    border: none;
    color: {INK_MUTED};
}}
QPushButton#panelCollapse:hover {{
    color: {ACCENT};
    background: {SURFACE_HOVER};
}}

QWidget#railBar {{
    background: {PANEL};
}}
QPushButton#railButton {{
    min-width: 22px;
    max-width: 22px;
    min-height: 30px;
    padding: 2px 0;
    font-size: 12px;
    border-radius: 0px;
    color: {INK_SECONDARY};
}}
QPushButton#railButton:hover {{
    color: {ACCENT};
    border-color: {ACCENT};
}}

QSplitter#mainSplitter::handle {{
    background: {HAIRLINE};
}}
QSplitter#mainSplitter::handle:hover {{
    background: {ACCENT};
}}

QLabel#viewLabel {{
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {INK_MUTED};
    padding-right: 2px;
}}
QPushButton#viewButton {{
    padding: 4px 9px;
    font-size: 10.5px;
    min-height: 0;
    border-radius: 2px;
}}
QFrame#viewSep {{
    color: {HAIRLINE};
    max-width: 1px;
    margin: 3px 2px;
}}

QPushButton#pixelIconButton {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
    margin-left: 2px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    color: {INK_MUTED};
}}
QPushButton#pixelIconButton:hover {{
    color: {ACCENT};
    border-color: {HAIRLINE};
    background: {SURFACE_HOVER};
}}
QPushButton#pixelIconButton:checked {{
    color: {ACCENT};
    border-color: {ACCENT};
    background: {ACCENT_DIM};
}}

QFrame#viewControlsPopover {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
}}
QLabel#menuSectionLabel {{
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {INK_MUTED};
    padding: 4px 10px 2px 10px;
}}

QComboBox#lockCombo {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    padding: 4px 8px;
    font-size: 10.5px;
    font-weight: 600;
    color: {INK_SECONDARY};
    min-width: 76px;
}}
QComboBox#lockCombo:hover {{
    color: {INK_PRIMARY};
    border-color: {ACCENT};
}}
QComboBox#lockCombo::drop-down {{
    border: none;
    width: 16px;
}}
QComboBox#lockCombo QAbstractItemView {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    color: {INK_SECONDARY};
    selection-background-color: {ACCENT_DIM};
    selection-color: {ACCENT};
    outline: none;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {HAIRLINE};
    border-radius: 0px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 0px;
}}
QSlider::handle:horizontal {{
    width: 12px;
    height: 14px;
    margin: -6px 0;
    border-radius: 1px;
    background: {ACCENT};
    border: 2px solid {PANEL};
}}

QFrame#bodyCard {{
    background: {PANEL_RAISED};
    border-radius: 0px;
    border: 1px solid {HAIRLINE};
    border-left: 2px solid transparent;
}}
QFrame#bodyCard[selected="true"] {{
    border-left: 2px solid {ACCENT};
    background: {ACCENT_DIM};
}}
QFrame#bodyCard[measure="true"] {{
    border-left: 2px solid {MEASURE};
    background: {MEASURE_DIM};
}}

QFrame#measureCard {{
    background: {MEASURE_DIM};
    border: 1px solid {MEASURE};
    border-radius: 0px;
}}
QLabel#measurePair {{
    font-size: 11px;
    font-weight: 650;
    color: {MEASURE};
}}
QLabel#measureDist {{
    font-size: 13px;
    color: {INK_PRIMARY};
}}
QLabel#bodyName {{
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.02em;
}}
QLabel#bodyStat {{
    font-size: 10.5px;
    color: {INK_SECONDARY};
}}
QLabel#bodyStatValue {{
    font-size: 10.5px;
    color: {INK_PRIMARY};
}}

QLabel#metReadout {{
    font-size: 13px;
    color: {INK_PRIMARY};
}}
QLabel#eventReadout {{
    font-size: 11px;
    font-weight: 650;
    color: {ACCENT};
}}
QLabel#speedValue {{
    font-size: 11px;
    font-weight: 650;
    color: {INK_PRIMARY};
}}
QLabel#speedValue[eased="true"] {{
    color: {ACCENT};
}}

QPushButton#autoSlowButton {{
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    padding: 0;
    margin-left: 6px;
    font-size: 11px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    color: {INK_MUTED};
}}
QPushButton#autoSlowButton:checked {{
    color: {ACCENT};
    border-color: {HAIRLINE};
    background: {ACCENT_DIM};
}}
QPushButton#autoSlowButton:hover {{
    color: {INK_PRIMARY};
}}

QPushButton#helpButton {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
    margin-left: 2px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    color: {INK_MUTED};
}}
QPushButton#helpButton:hover {{
    color: {ACCENT};
    border-color: {HAIRLINE};
}}

QDialog#shortcutsDialog {{
    background: {PANEL};
}}
QLabel#keyCap {{
    font-size: 11.5px;
    font-weight: 600;
    color: {INK_PRIMARY};
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    padding: 4px 10px;
    min-width: 70px;
}}
QLabel#shortcutDesc {{
    font-size: 13px;
    color: {INK_SECONDARY};
}}

QFrame#maneuverRow {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-left: 3px solid {ACCENT_DIM};
    border-radius: 0px;
}}
QFrame#maneuverRow:hover {{
    border-color: {ACCENT};
}}
QFrame#maneuverRow[active="true"] {{
    border-left: 3px solid {ACCENT};
    background: {ACCENT_DIM};
}}
QLabel#maneuverName {{
    font-size: 11.5px;
    font-weight: 650;
    color: {INK_PRIMARY};
}}
QLabel#maneuverTime {{
    font-size: 10px;
    color: {ACCENT};
}}
QLabel#maneuverNote {{
    font-size: 10px;
    color: {INK_SECONDARY};
}}
QLabel#dateReadout {{
    font-size: 10.5px;
    color: {INK_MUTED};
}}
QLabel#scaleReadout {{
    font-size: 9.5px;
    color: {INK_MUTED};
}}

QMenuBar {{
    background: {PANEL};
    color: {INK_SECONDARY};
    border-bottom: 1px solid {HAIRLINE};
}}
QMenuBar::item:selected {{
    background: {SURFACE_HOVER};
    color: {INK_PRIMARY};
}}
QMenu {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    color: {INK_SECONDARY};
}}
QMenu::item:selected {{
    background: {ACCENT_DIM};
    color: {ACCENT};
}}

QStatusBar {{
    background: {PANEL};
    color: {INK_MUTED};
    border-top: 1px solid {HAIRLINE};
    font-size: 10.5px;
}}

QScrollArea#infoScroll {{
    background: transparent;
    border: none;
}}
QScrollArea#infoScroll > QWidget > QWidget {{
    background: transparent;
}}
QWidget#infoCards {{
    background: transparent;
}}

QWidget#loadingPage {{
    background: {BG_VOID};
}}
QLabel#loadingLabel {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: {INK_SECONDARY};
    padding-top: 16px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {HAIRLINE};
    border-radius: 0px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QLabel#statusBadgeText {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
}}
QLabel#metricLabel {{
    font-size: 10px;
    color: {INK_MUTED};
    letter-spacing: 0.04em;
}}
QLabel#metricValue {{
    font-size: 11px;
    color: {INK_PRIMARY};
    font-weight: 600;
}}
QLabel#metricUnit {{
    font-size: 9.5px;
    color: {INK_MUTED};
}}
"""
