"""Qt stylesheet for the Mission Control app -- a dark "tracking station"
look in the spirit of KSP's map view (dense info panels, a time-warp
control strip, orbit-colored readouts) built from this project's own
already-validated deep-space palette, not KSP's literal assets/icon set.
"""
from __future__ import annotations

BG_VOID = "#06050c"
PANEL = "#12101f"
PANEL_RAISED = "#1a1730"
HAIRLINE = "#2a273f"
INK_PRIMARY = "#f4f2ea"
INK_SECONDARY = "#a9a6bb"
INK_MUTED = "#6f6c85"
ACCENT = "#e8a23e"
ACCENT_DIM = "#4a3a1c"
MEASURE = "#5ec8ff"      # distance-measurement line + readout (cool, to contrast the amber)
MEASURE_DIM = "#173040"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", -apple-system, sans-serif;
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
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {ACCENT};
    padding: 2px 4px;
}}

QLabel#sceneTitle {{
    font-size: 16px;
    font-weight: 650;
}}
QLabel#sceneSubtitle {{
    font-size: 11px;
    color: {INK_SECONDARY};
}}

QLabel#sectionHeader {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: {INK_MUTED};
    padding: 10px 4px 4px 4px;
    text-transform: uppercase;
}}

QListWidget {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 12.5px;
}}
QListWidget::item {{
    padding: 9px 10px;
    border-radius: 7px;
    margin: 1px 6px;
    color: {INK_SECONDARY};
}}
QListWidget::item:hover {{
    background: {PANEL_RAISED};
    color: {INK_PRIMARY};
}}
QListWidget::item:selected {{
    background: {ACCENT_DIM};
    color: {ACCENT};
    font-weight: 650;
}}

QPushButton {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 7px;
    padding: 6px 12px;
    font-size: 11.5px;
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
    border-radius: 16px;
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
    background: {PANEL_RAISED};
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
    letter-spacing: 0.08em;
    color: {INK_MUTED};
    padding-right: 2px;
}}
QPushButton#viewButton {{
    padding: 4px 9px;
    font-size: 10.5px;
    min-height: 0;
}}
QFrame#viewSep {{
    color: {HAIRLINE};
    max-width: 1px;
    margin: 3px 2px;
}}

QComboBox#lockCombo {{
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 7px;
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
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 7px;
    background: {ACCENT};
    border: 2px solid {PANEL};
}}

QFrame#bodyCard {{
    background: {PANEL_RAISED};
    border-radius: 8px;
    border: 1px solid {HAIRLINE};
}}
QFrame#bodyCard[selected="true"] {{
    border: 1px solid {ACCENT};
    background: {ACCENT_DIM};
}}
QFrame#bodyCard[measure="true"] {{
    border: 1px solid {MEASURE};
    background: {MEASURE_DIM};
}}

QFrame#measureCard {{
    background: {MEASURE_DIM};
    border: 1px solid {MEASURE};
    border-radius: 8px;
}}
QLabel#measurePair {{
    font-size: 11px;
    font-weight: 650;
    color: {MEASURE};
}}
QLabel#measureDist {{
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 13px;
    color: {INK_PRIMARY};
}}
QLabel#bodyName {{
    font-size: 12.5px;
    font-weight: 650;
}}
QLabel#bodyStat {{
    font-size: 10.5px;
    color: {INK_SECONDARY};
}}
QLabel#bodyStatValue {{
    font-size: 10.5px;
    font-family: "Cascadia Code", "Consolas", monospace;
    color: {INK_PRIMARY};
}}

QLabel#metReadout {{
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 13px;
    color: {INK_PRIMARY};
}}
QLabel#eventReadout {{
    font-size: 11px;
    font-weight: 650;
    color: {ACCENT};
}}
QLabel#speedValue {{
    font-family: "Cascadia Code", "Consolas", monospace;
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
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 11.5px;
    font-weight: 600;
    color: {INK_PRIMARY};
    background: {PANEL_RAISED};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
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
    border-radius: 6px;
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
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 10px;
    color: {ACCENT};
}}
QLabel#maneuverNote {{
    font-size: 10px;
    color: {INK_SECONDARY};
}}
QLabel#dateReadout {{
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 10.5px;
    color: {INK_MUTED};
}}
QLabel#scaleReadout {{
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9.5px;
    color: {INK_MUTED};
}}

QMenuBar {{
    background: {PANEL};
    color: {INK_SECONDARY};
    border-bottom: 1px solid {HAIRLINE};
}}
QMenuBar::item:selected {{
    background: {PANEL_RAISED};
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
    letter-spacing: 0.06em;
    color: {INK_SECONDARY};
    padding-top: 16px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {HAIRLINE};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
