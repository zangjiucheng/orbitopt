"""E8-04: cross-platform reference consistency.

The Gate B frame bridge, frozen through the shared pinned-EOP mechanism,
must re-derive the committed GCRS tracks from the raw recorded
OpenTsiolkovsky outputs within the single declared cross-platform band on
every supported platform (Linux x86_64, macOS Intel, macOS ARM -- the CI
matrix runs this suite against the identical locked astropy 6.1.7 /
astropy-iers-data 0.2026.8.31.0.57.9 stack on all three). On stacks that
are not the recording stack the test skips with the mechanism's own
reason instead of pretending to verify a different Earth orientation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEDGER_PATH = Path(__file__).parent / "data" / "e804" / "cross-platform.json"
GATEB_REF = Path(__file__).parent / "data" / "flightlab" / "reference"


def _script():
    spec = importlib.util.spec_from_file_location(
        "fl601_reference", REPO / "scripts" / "fl601_reference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _band() -> dict:
    band = json.loads((GATEB_REF / "tolerances.json").read_text(encoding="utf-8"))
    return band["cross_platform"]


def test_ledger_declares_the_three_platforms_and_the_band_source():
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger["format"] == 1
    assert ledger["platforms"] == ["linux-x86_64", "darwin-x86_64", "darwin-arm64"]
    assert ledger["band_source"] == "tests/data/flightlab/reference/tolerances.json#/cross_platform"
    assert json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["observations"].keys() == {
        "linux-x86_64", "darwin-x86_64", "darwin-arm64"}
    for obs in ledger["observations"].values():
        if obs is None:
            continue
        assert obs["max_position_m"] <= _band()["position_m_abs"]
        assert obs["max_velocity_m_s"] <= _band()["velocity_m_s_abs"]


def test_cross_platform_reference_consistency():
    from orbitopt.reference_eop import ToolchainMismatch

    script = _script()
    try:
        obs = script.cross_platform_observation()
    except ToolchainMismatch as exc:
        pytest.skip(str(exc))
    band = _band()
    assert obs["id"] in json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["platforms"]
    assert obs["max_position_m"] <= band["position_m_abs"], (
        f"platform {obs['id']} re-derived the Gate B GCRS tracks {obs['max_position_m']} m "
        f"off the reference -- frame bridge is not platform-consistent under the pinned EOP")
    assert obs["max_velocity_m_s"] <= band["velocity_m_s_abs"]
    stored = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["observations"].get(obs["id"])
    if stored is not None:
        assert abs(obs["max_position_m"] - stored["max_position_m"]) <= band["position_m_abs"]
        assert abs(obs["max_velocity_m_s"] - stored["max_velocity_m_s"]) <= band["velocity_m_s_abs"]
