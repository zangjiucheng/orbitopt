"""The stylesheet only names fonts Qt has: a missing family in a QSS font stack made Qt
print ``Populating font family aliases took N ms ... missing font family "D-DIN"`` and rebuild
its alias table whenever it polished a widget."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _installed(monkeypatch, families):
    from PySide6.QtGui import QFontDatabase

    monkeypatch.setattr(QFontDatabase, "families", staticmethod(lambda *a, **k: list(families)))


def test_missing_families_are_dropped_and_generics_kept(qapp, monkeypatch):
    from orbitopt.viz.theme import FONT_MONO, FONT_SANS, installed_stack

    _installed(monkeypatch, ["DIN Alternate", "Inter", "Menlo"])
    assert installed_stack(FONT_SANS) == '"Inter", sans-serif' or installed_stack(FONT_SANS) == \
        '"DIN Alternate", "Inter", sans-serif'
    assert "D-DIN" not in installed_stack(FONT_SANS) and "-apple-system" not in installed_stack(FONT_SANS)
    assert installed_stack(FONT_MONO) == '"Menlo", monospace'


def test_the_first_installed_family_keeps_its_place(qapp, monkeypatch):
    from orbitopt.viz.theme import FONT_SANS, installed_stack

    _installed(monkeypatch, ["D-DIN", "DIN Alternate"])
    assert installed_stack(FONT_SANS).startswith('"D-DIN", "DIN Alternate"')      # D-DIN is used when it IS installed


def test_nothing_installed_falls_back_to_the_generic_family(qapp, monkeypatch):
    from orbitopt.viz.theme import FONT_SANS, installed_stack

    _installed(monkeypatch, [])
    assert installed_stack(FONT_SANS) == "sans-serif"


def test_the_stylesheet_names_no_missing_family(qapp, monkeypatch):
    from orbitopt.viz.theme import stylesheet

    _installed(monkeypatch, ["Menlo"])
    css = stylesheet()
    assert '"D-DIN"' not in css and '"Bahnschrift"' not in css and "apple-system" not in css
    assert 'font-family: sans-serif;' in css and '"Menlo", monospace' in css
