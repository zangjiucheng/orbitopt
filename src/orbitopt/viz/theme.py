"""Qt stylesheet for the Mission Control app: a black-and-white "launch
control" chrome around the 3D view (mission rail, inspector, time-warp strip).

The palette follows SpaceX's own web and webcast graphics: pure black
surfaces, white type, neutral greys and thin hairlines, no tint and no
accent colour -- white *is* the accent (selected row, active tab, focus).
Colour is kept only where it carries meaning: the status colours below
(verified / caution / failed / running) and the body and trail colours of
the 3D scene. The shared tokens mirror docs/index.html's :root. Layout, hard
edges and status badges are unchanged; the type pairing stays a sans for
prose and a monospace for readouts. The 3D view itself is styled by
scene_renderer.py, not here.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Design tokens -- keep in sync with docs/index.html's :root.
# ---------------------------------------------------------------------------
BG_VOID = "#000000"        # --void: pure black
PANEL = "#0a0a0a"          # --panel
PANEL_RAISED = "#141414"   # --panel-raised
SURFACE_HOVER = "#1f1f1f"
HAIRLINE = "#2e2e2e"       # --hairline
BORDER_SOFT = "#1a1a1a"    # --hairline-soft
INK_PRIMARY = "#ffffff"    # --ink
INK_SECONDARY = "#b3b3b3"  # --ink-2
INK_MUTED = "#7a7a7a"      # --ink-muted
ACCENT = "#ffffff"         # white is the accent: selection, active tab, focus
ACCENT_DIM = "#262626"     # selected-row wash
ACCENT_MID = "#5c5c5c"     # accent on a dark wash (selected-item contexts)

# Semantic status colors -- every state badge, mission-list status dot and
# body/maneuver highlight maps to exactly one of these.
SUCCESS = "#35c48b"   # verified / loaded            (--saturn)
WARNING = "#f2c14e"   # partial / caution
DANGER = "#ff4d3d"    # failed                       (--burn)
INFO = "#a6c8ff"      # running / neutral-active     (--verify)

MEASURE = INFO          # distance-measurement line + readout (the one cool tone)
MEASURE_DIM = "#161c24"  # --verify-dim

# Type: SpaceX sets its interface in D-DIN, a DIN-style grotesque. D-DIN is
# open (SIL OFL) but not bundled; the stack takes it if installed (or dropped in
# viz/assets/fonts, see load_bundled_fonts), then the closest faces that ship
# with the OS: "DIN Alternate" on macOS, "Bahnschrift" on Windows 10+, then
# Barlow / Inter and the platform UI face. Readouts use a monospace.
FONT_SANS = ('"D-DIN", "DIN Alternate", "Bahnschrift", "Barlow", "Inter", '
             '"Helvetica Neue", "Segoe UI", "Noto Sans", "DejaVu Sans"')
FONT_MONO = ('"JetBrains Mono", "SF Mono", "IBM Plex Mono", "Cascadia Code", '
             '"Menlo", "Consolas", "DejaVu Sans Mono"')


def load_bundled_fonts() -> list[str]:
    """Register every font file in ``viz/assets/fonts`` with Qt (needs a
    QGuiApplication); returns the families added. Drop D-DIN (OFL) there to
    get the SpaceX face on every platform."""
    from pathlib import Path

    from PySide6.QtGui import QFontDatabase

    families: list[str] = []
    folder = Path(__file__).parent / "assets" / "fonts"
    for path in sorted(folder.glob("*")) if folder.is_dir() else []:
        if path.suffix.lower() in (".ttf", ".otf"):
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id >= 0:
                families += QFontDatabase.applicationFontFamilies(font_id)
    return families


def resolve_family(stack: str = FONT_SANS) -> str:
    """The first family of a CSS-style ``stack`` that Qt has installed (a
    QPainter font takes one family, not a list); the platform default when
    none is."""
    from PySide6.QtGui import QFontDatabase

    have = set(QFontDatabase.families())
    for name in (part.strip().strip('"') for part in stack.split(",")):
        if name in have:
            return name
    return ""


def installed_stack(stack: str) -> str:
    """Return only named Qt families. Cocoa can treat CSS generics as missing faces."""
    from PySide6.QtGui import QFontDatabase

    have = set(QFontDatabase.families())
    return ", ".join(part.strip() for part in stack.split(",") if part.strip().strip('"') in have)


def stylesheet() -> str:
    """:data:`STYLESHEET` with its two font stacks cut to the installed families (call it
    after ``load_bundled_fonts``, once a QApplication exists)."""
    css = STYLESHEET
    for stack in (FONT_SANS, FONT_MONO):
        family = installed_stack(stack)
        css = css.replace(f"font-family: {stack};", f"font-family: {family};" if family else "")
    return css


STYLESHEET = f"""
* {{
    font-family: {FONT_SANS};
    color: {INK_PRIMARY};
}}

/* Numeric readouts and small labels in monospace, as on the site. */
QLabel#appTitle, QLabel#sectionHeader, QLabel#viewLabel, QLabel#menuSectionLabel,
QLabel#measureDist, QLabel#bodyStatValue, QLabel#metReadout, QLabel#speedValue,
QLabel#keyCap, QLabel#maneuverTime, QLabel#dateReadout, QLabel#scaleReadout,
QLabel#statusBadgeText, QLabel#metricLabel, QLabel#metricValue, QLabel#metricUnit {{
    font-family: {FONT_MONO};
}}

/* Dialogs and tooltips get the dark panel explicitly: the global ink color
   above otherwise lands on the platform's default (light, on macOS) dialog
   background -- white text on white in QMessageBox. */
QDialog, QMessageBox {{
    background: {PANEL};
}}
QMessageBox QLabel {{
    color: {INK_PRIMARY};
    background: transparent;
}}
QMessageBox QPushButton {{
    min-width: 72px;
}}
QToolTip {{
    background: {PANEL_RAISED};
    color: {INK_PRIMARY};
    border: 1px solid {HAIRLINE};
    padding: 4px 6px;
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
QTabWidget#rightTabs, QTabWidget#chartTabs {{
    background: {PANEL};
}}
QTabWidget#rightTabs::pane {{
    border: none;
    border-left: 1px solid {HAIRLINE};
}}
QTabWidget#chartTabs::pane {{
    border: none;
    border-top: 1px solid {HAIRLINE};
}}
/* ">" (direct child): a tab widget nested in a tab -- the Flight tab's
   Checks / Why / Recovery -- keeps its own compact style; the descendant
   form gave it this letter-spaced font and Qt elided every label. */
QTabWidget#rightTabs > QTabBar, QTabWidget#chartTabs > QTabBar {{
    background: {PANEL};
}}
QTabWidget#rightTabs > QTabBar::tab, QTabWidget#chartTabs > QTabBar::tab {{
    background: {PANEL};
    color: {INK_MUTED};
    padding: 7px 12px 6px 12px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
}}
QTabWidget#rightTabs > QTabBar::tab:hover, QTabWidget#chartTabs > QTabBar::tab:hover {{
    color: {INK_SECONDARY};
}}
QTabWidget#rightTabs > QTabBar::tab:selected, QTabWidget#chartTabs > QTabBar::tab:selected {{
    color: {INK_PRIMARY};
    border-bottom: 2px solid {ACCENT};
}}
QPushButton#telemetryPlots {{
    margin: 2px 4px;
    padding: 2px 17px 2px 8px;
    font-size: 10.5px;
}}
QPushButton#telemetryPlots::menu-indicator {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    right: 5px;
}}
QWidget#viewStrip {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER_SOFT};
}}
QLabel#scaleNote {{
    font-size: 10.5px;
    color: {INK_MUTED};
}}
QLabel#scaleNote[warn="true"] {{
    color: {WARNING};
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

QPushButton#reloadBanner {{
    background: {ACCENT_DIM};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 2px;
    padding: 3px 10px;
    font-weight: 600;
}}
QPushButton#reloadBanner:hover {{
    background: {ACCENT};
    color: {PANEL};
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

/* Top workspace bar: black, type only, a white rule under the current workspace. */
QWidget#workspaceBar {{
    background: {BG_VOID};
    border-bottom: 1px solid {HAIRLINE};
}}
QLabel#workspaceMark {{
    font-family: {FONT_MONO};
    font-size: 12px; font-weight: 700; letter-spacing: 3px; color: {INK_PRIMARY};
}}
QPushButton#workspaceTab {{
    background: transparent; border: none; border-bottom: 2px solid transparent;
    color: {INK_MUTED}; padding: 0 16px; min-height: 38px;
    font-size: 11px; font-weight: 600;
}}
QPushButton#workspaceTab:hover {{ color: {INK_PRIMARY}; }}
QPushButton#workspaceTab:checked {{ color: {INK_PRIMARY}; border-bottom: 2px solid {INK_PRIMARY}; }}
QPushButton#panelToggle {{
    background: transparent; border: none; min-width: 30px; max-width: 30px;
    min-height: 30px; max-height: 30px; padding: 0;
}}
QPushButton#panelToggle:hover {{ background: {SURFACE_HOVER}; }}
QPushButton#panelToggle:disabled {{ background: transparent; }}
QLabel#workspaceLogo {{ background: transparent; }}
QPushButton#workspaceSearch {{
    background: transparent; border: 1px solid {HAIRLINE}; color: {INK_MUTED};
    padding: 4px 12px; font-family: {FONT_MONO}; font-size: 11px;
}}
QPushButton#workspaceSearch:hover {{ color: {INK_PRIMARY}; border-color: {INK_MUTED}; }}
QLabel#breadcrumb {{
    font-family: {FONT_MONO}; font-size: 10px; color: {INK_MUTED}; letter-spacing: 1px;
}}
QDialog#commandPalette {{ background: {PANEL}; border: 1px solid {INK_MUTED}; }}
QLineEdit#paletteQuery {{
    background: {BG_VOID}; border: none; border-bottom: 1px solid {HAIRLINE};
    padding: 12px 14px; font-size: 14px; color: {INK_PRIMARY};
}}
QListWidget#paletteResults {{
    background: {PANEL}; border: none; font-family: {FONT_MONO}; font-size: 12px;
    color: {INK_SECONDARY};
}}
QListWidget#paletteResults::item {{ padding: 7px 14px; }}
QListWidget#paletteResults::item:selected {{ background: {ACCENT_DIM}; color: {INK_PRIMARY}; }}

QPushButton#modeSwitch {{
    background: {PANEL};
    color: {INK_MUTED};
    border: 1px solid {HAIRLINE};
    padding: 5px 6px;
    min-height: 16px;  /* the label's own height: "min-height: 0" let the layout squeeze it to 20 px, clipping the text */
    font-size: 11px;
    font-weight: 600;
}}
QPushButton#modeSwitch[side="left"] {{
    border-top-left-radius: 3px; border-bottom-left-radius: 3px;
    border-top-right-radius: 0; border-bottom-right-radius: 0;
}}
QPushButton#modeSwitch[side="right"] {{
    border-left: none;
    border-top-right-radius: 3px; border-bottom-right-radius: 3px;
    border-top-left-radius: 0; border-bottom-left-radius: 0;
}}
QPushButton#modeSwitch:hover {{
    color: {INK_SECONDARY};
}}
QPushButton#modeSwitch:checked {{
    background: {PANEL_RAISED};
    color: {ACCENT};
}}
QPushButton#modeSwitch:disabled {{
    color: {BORDER_SOFT};
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
    min-height: 14px;  /* "min-height: 0" let a short window squeeze the CENTER strip's buttons, clipping Sun/Earth/Mars */
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

QLineEdit#metReadout {{
    font-family: {FONT_MONO};
    font-size: 13px;
    color: {INK_PRIMARY};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    padding: 2px 4px;
    selection-background-color: {ACCENT_MID};
}}
QLineEdit#metReadout:hover {{
    border-color: {HAIRLINE};
}}
QLineEdit#metReadout:focus {{
    background: {PANEL_RAISED};
    border-color: {ACCENT_MID};
}}
QLineEdit#metReadout[invalid="true"] {{
    border-color: {DANGER};
}}
QPushButton#timelineToolButton {{
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    padding: 0;
    font-family: {FONT_MONO};
    font-size: 14px;
    background: transparent;
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    color: {INK_SECONDARY};
}}
QPushButton#timelineTextButton {{
    min-height: 24px;
    max-height: 24px;
    padding: 0 8px;
    font-family: {FONT_MONO};
    font-size: 10.5px;
    background: transparent;
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    color: {INK_SECONDARY};
}}
QPushButton#realTimeButton {{
    min-width: 30px;
    max-width: 30px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
    font-family: {FONT_MONO};
    font-size: 10.5px;
    background: transparent;
    border: 1px solid {HAIRLINE};
    border-radius: 2px;
    color: {INK_SECONDARY};
}}
QPushButton#realTimeButton:checked {{
    color: {ACCENT};
    background: {ACCENT_DIM};
    border-color: {ACCENT_MID};
}}
QPushButton#realTimeButton:hover, QPushButton#timelineToolButton:hover, QPushButton#timelineTextButton:hover {{
    color: {INK_PRIMARY};
    background: {SURFACE_HOVER};
}}
QPushButton#timelineToolButton:disabled, QPushButton#timelineTextButton:disabled {{
    color: {INK_MUTED};
    border-color: {BORDER_SOFT};
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
QFrame#inspectorSection {{
    background: {PANEL};
    border: 1px solid {HAIRLINE};
    border-radius: 3px;
}}
QWidget#inspectorSectionHeader {{
    background: transparent;
    border-bottom: 1px solid {BORDER_SOFT};
}}
QLabel#sectionCount {{
    font-family: {FONT_MONO};
    font-size: 10px;
    color: {INK_MUTED};
}}
QSplitter#inspectorSplitter::handle {{
    background: transparent;
}}
QSplitter#inspectorSplitter::handle:hover {{
    background: {HAIRLINE};
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
