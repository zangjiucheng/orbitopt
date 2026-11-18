"""The toolchain record must not call an installed tool "absent".

A conda build of tudatpy ships no dist-info, so ``importlib.metadata`` cannot
see it; the cross-environment CI run sealed ``'tudatpy': 'absent'`` for the
very environment that flew the ascent, and the provenance test only checked
that the key existed.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json

import pytest

from orbitopt.flightlab import toolchain
from orbitopt.flightlab.toolchain import ABSENT, UNKNOWN_VERSION, toolchain_record


def _no_metadata(monkeypatch):
    def missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)


def _importable(monkeypatch, present: bool):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if (name == "tudatpy" and present)
                        else (None if name == "tudatpy" else real(name, *a, **k)))


def test_conda_record_gives_the_version_when_there_is_no_dist_info(tmp_path, monkeypatch):
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "tudatpy-extras-9.9-0.json").write_text(json.dumps({}))     # another package's record
    (meta / "tudatpy-3.4.5-py310h1234_0.json").write_text(json.dumps({}))
    monkeypatch.setattr(toolchain.sys, "prefix", str(tmp_path))
    _no_metadata(monkeypatch)
    _importable(monkeypatch, True)
    assert toolchain._dist_version("tudatpy") == "3.4.5"


def test_importable_but_unrecorded_is_not_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.sys, "prefix", str(tmp_path))          # no conda-meta here
    _no_metadata(monkeypatch)
    _importable(monkeypatch, True)
    assert toolchain._dist_version("tudatpy") == UNKNOWN_VERSION != ABSENT


def test_a_tool_that_is_not_installed_is_still_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.sys, "prefix", str(tmp_path))
    _no_metadata(monkeypatch)
    _importable(monkeypatch, False)
    assert toolchain._dist_version("tudatpy") == ABSENT
    assert toolchain._dist_version("gfold") == ABSENT                    # no fallback for the others


def test_the_real_environment_never_reports_an_importable_tudatpy_as_absent():
    if importlib.util.find_spec("tudatpy") is None:
        pytest.skip("tudatpy is not installed here")
    assert toolchain_record()["tudatpy"] != ABSENT
