"""The toolchain record must not call an installed tool "absent".

A conda build of tudatpy ships no dist-info, so ``importlib.metadata`` cannot
see it; the cross-environment CI run sealed ``'tudatpy': 'absent'`` for the
very environment that flew the ascent, and the provenance test only checked
that the key existed.

The missing dist-info is how a channel builds a package, not a tudatpy quirk,
so the lookup falls back for every recorded tool. The pip-installed ones
(gfold, cvxpy, jsbsim) always carry dist-info, so for them the first source
answers and the fallback never fires -- tested below by installing a fake
conda-meta record for a pip tool.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json

import pytest

from orbitopt.flightlab import toolchain
from orbitopt.flightlab.toolchain import (ABSENT, UNKNOWN_VERSION, installed_version,
                                         toolchain_record)


def _no_metadata(monkeypatch, *names: str):
    real = importlib.metadata.version

    def missing(name):
        if not names or name in names:
            raise importlib.metadata.PackageNotFoundError
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)


def _importable(monkeypatch, name: str, present: bool):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda mod, *a, **k: object() if (mod == name and present)
                        else (None if mod == name else real(mod, *a, **k)))


def _fake_prefix(tmp_path, *records: str):
    """An environment root whose ``conda-meta`` holds ``records``."""
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    for record in records:
        (meta / record).write_text(json.dumps({}))
    return str(tmp_path)


def test_conda_record_gives_the_version_when_there_is_no_dist_info(tmp_path, monkeypatch):
    prefix = _fake_prefix(tmp_path, "tudatpy-extras-9.9-0.json",        # other packages
                          "boost-1.84.0-h5e97a2a_0.json",
                          "tudatpy-3.4.5-py310h1234_0.json")
    monkeypatch.setattr(toolchain.sys, "prefix", prefix)
    _no_metadata(monkeypatch)
    _importable(monkeypatch, "tudatpy", True)
    assert installed_version("tudatpy") == "3.4.5"


def test_the_conda_record_is_used_for_any_tool_not_just_tudatpy(tmp_path, monkeypatch):
    """A conda channel that omits dist-info breaks every package it builds, so
    the fallback is keyed on nothing but the name."""
    prefix = _fake_prefix(tmp_path, "pygmo-2.20.0-py310habcdefg_1.json")
    monkeypatch.setattr(toolchain.sys, "prefix", prefix)
    _no_metadata(monkeypatch, "pygmo")
    _importable(monkeypatch, "pygmo", True)
    assert installed_version("pygmo") == "2.20.0"


def test_importable_but_unrecorded_is_not_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.sys, "prefix", _fake_prefix(tmp_path))  # no records here
    _no_metadata(monkeypatch)
    _importable(monkeypatch, "tudatpy", True)
    assert installed_version("tudatpy") == UNKNOWN_VERSION != ABSENT


def test_a_pip_tool_without_dist_info_is_not_absent_either(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.sys, "prefix", _fake_prefix(tmp_path))
    _no_metadata(monkeypatch, "gfold")
    _importable(monkeypatch, "gfold", True)
    assert installed_version("gfold") == UNKNOWN_VERSION != ABSENT


def test_the_import_name_is_not_always_the_distribution_name(tmp_path, monkeypatch):
    """``pyyaml`` is imported as ``yaml``: the fallback must look for the module
    the environment actually has."""
    monkeypatch.setattr(toolchain.sys, "prefix", _fake_prefix(tmp_path))
    _no_metadata(monkeypatch, "pyyaml")
    _importable(monkeypatch, "yaml", True)
    assert installed_version("pyyaml") == UNKNOWN_VERSION != ABSENT


def test_a_tool_that_is_not_installed_is_still_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.sys, "prefix", _fake_prefix(tmp_path))
    _no_metadata(monkeypatch)
    _importable(monkeypatch, "tudatpy", False)
    _importable(monkeypatch, "gfold", False)
    assert installed_version("tudatpy") == ABSENT
    assert installed_version("gfold") == ABSENT


def test_metadata_still_wins_when_it_is_there(tmp_path, monkeypatch):
    """The recorded version comes from ``importlib.metadata`` when the package
    ships dist-info, whatever conda-meta may also hold."""
    prefix = _fake_prefix(tmp_path, "numpy-1.0.0-py310h0_0.json")
    monkeypatch.setattr(toolchain.sys, "prefix", prefix)
    assert installed_version("numpy") == toolchain.importlib.metadata.version("numpy")


def test_the_real_environment_never_reports_an_importable_tudatpy_as_absent():
    if importlib.util.find_spec("tudatpy") is None:
        pytest.skip("tudatpy is not installed here")
    assert toolchain_record()["tudatpy"] != ABSENT


@pytest.mark.parametrize("dist", ["gfold", "cvxpy", "jsbsim"])
def test_the_pip_tools_report_their_real_versions_here(dist):
    """The pip layer of this environment (the check this task asked for): a
    tool installed from a wheel has dist-info, so its version is a version."""
    if importlib.util.find_spec(toolchain._IMPORT_NAMES.get(dist, dist)) is None:
        pytest.skip(f"{dist} is not installed here")
    version = installed_version(dist)
    assert version not in (ABSENT, UNKNOWN_VERSION), dist
    assert version[0].isdigit(), dist
