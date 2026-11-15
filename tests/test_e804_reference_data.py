"""E8-04: every reference regression pins Earth orientation through the one
shared mechanism (``orbitopt.reference_eop``), tolerance changes are audited,
and the audit ledger in docs/reference-tolerance-audit.md stays in sync with
the machine-readable tolerances.

The Gate B reference verifies its pinned table against a bundled-data
snapshot (mode ``pinned-bundled``: astropy keeps using its installed
``astropy-iers-data`` table, which is what the recorded tracks were flown
with -- an IERS-B default-table flight; the E2-04 reference keeps its
committed override snapshot of four IERS-A rows, mode ``override``). Both
go through the same snapshot format, the same freeze entry points and the
same audited ``update`` command.
"""
from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATEB_REF = Path(__file__).parent / "data" / "flightlab" / "reference"
E204_REF = Path(__file__).parent / "data" / "mission" / "reference" / "e2-04-frozen-mission"
AUDIT_DOC = REPO / "docs" / "reference-tolerance-audit.md"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_reference", REPO / "scripts" / f"{name}_reference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _lock_stack_or_skip():
    """Both frame-freeze guards below need the recording stack (astropy 6.1
    + astropy-iers-data 0.2026.8.31.0.57.9, per the snapshot); anything else
    skips with the mechanism's own reason."""
    from orbitopt.reference_eop import ToolchainMismatch

    script = _script("fl601")
    ctx = script.frozen_reference()
    try:
        ctx.__enter__()
    except ToolchainMismatch as exc:
        pytest.skip(str(exc))
    return ctx


def test_both_references_use_the_shared_snapshot_mechanism():
    from orbitopt.reference_eop import SNAPSHOT_UNITS, load_snapshot

    gateb = load_snapshot(GATEB_REF / "earth_orientation.json")
    e204 = load_snapshot(E204_REF / "earth_orientation.json")
    assert gateb["mode"] == "pinned-bundled"
    assert e204["mode"] == "override"
    for snap in (gateb, e204):
        assert snap.get("format", 1) == 1
        assert snap["units"] == SNAPSHOT_UNITS
        assert snap["rows"] and all(
            sorted(row) == ["MJD", "PM_x", "PM_y", "UT1_UTC"] for row in snap["rows"])
        assert "source_version" in snap
    assert "note" in gateb
    assert gateb["active_table"] == "IERS_B"
    assert gateb["astropy_series"] == "6.1"


def test_snapshot_epoch_matches_the_pure_utc_to_mjd_conversion():
    from orbitopt.reference_eop import load_snapshot, utc_to_mjd

    snap = load_snapshot(GATEB_REF / "earth_orientation.json")
    parsed = datetime.strptime(snap["epoch_utc"].replace("Z", ""),
                               "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    assert math.isclose(utc_to_mjd(parsed), snap["epoch_mjd_utc"], abs_tol=1e-9)


def test_reference_scripts_delegate_pinning_to_the_shared_module(monkeypatch):
    from orbitopt import reference_eop

    calls = {"n": 0}

    def sentinel():
        calls["n"] += 1
        return "0.0-test"

    monkeypatch.setattr(reference_eop, "pin_no_download", sentinel)
    for name in ("fl601", "e204"):
        calls["n"] = 0
        script = _script(name)
        assert script._pin_earth_orientation() == "0.0-test"
        assert calls["n"] == 1, f"{name} bypasses orbitopt.reference_eop"


def test_wrong_pinned_data_stack_refuses_not_falls_back():
    """The whole point of the E8-04 mechanism: an astropy 8 / newer
    astropy-iers-data stack must raise ToolchainMismatch with a readable
    reason, never silently fly a different Earth orientation (the astropy
    6 -> 8 default-table flip changed UT1-UTC by 0.071 s = ~28 m of track)."""
    from orbitopt.reference_eop import ToolchainMismatch, freeze_reference

    ctx = _lock_stack_or_skip()
    try:
        snap = json.loads((GATEB_REF / "earth_orientation.json").read_text(encoding="utf-8"))
        for field, bad, needle in (
                ("source_version", "0.2020.1.1.0.0.0", "astropy-iers-data"),
                ("astropy_series", "5.3", "astropy"),
                ("active_table", "IERS_A", "table")):
            broken = dict(snap, **{field: bad})
            with pytest.raises(ToolchainMismatch) as excinfo, freeze_reference(broken):
                pass
            assert needle in str(excinfo.value).lower()
    finally:
        ctx.__exit__(None, None, None)


def test_gateb_tolerance_change_is_audited_in_provenance():
    prov = json.loads((GATEB_REF / "PROVENANCE.json").read_text(encoding="utf-8"))
    entries = prov.get("tolerance_audit", [])
    entry = next((e for e in entries if any(
        k.startswith("cross_platform.") for k in e["changes"])), None)
    assert entry is not None, "E8-04 cross_platform band addition must be in tolerance_audit"
    assert entry["changes"]["cross_platform.position_m_abs"] == {"old": None, "new": 1e-4}
    assert entry["changes"]["cross_platform.velocity_m_s_abs"] == {"old": None, "new": 1e-6}
    assert entry["reason"].strip()
    import hashlib

    assert entry["tolerances_sha256"] == hashlib.sha256(
        (GATEB_REF / "tolerances.json").read_bytes()).hexdigest()


def test_audit_doc_current_values_match_the_machine_readable_tolerances():
    """The human ledger must not drift from the enforced bands."""
    import re

    text = AUDIT_DOC.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for section in text.split("```json"):
        body = section.split("```", 1)[0]
        lines = [line for line in body.splitlines() if line.strip()]
        head = lines[0].strip() if lines else ""
        found = re.search(r"tolerances:\s*([\w.-]+)", head)
        if found:
            blocks[found.group(1)] = body[body.index("-->") + 3:]
    for key, path in (("gateb-leo-two-stage", GATEB_REF / "tolerances.json"),
                      ("e2-04-frozen-mission", E204_REF / "tolerances.json")):
        assert key in blocks, f"docs ledger lacks a ```json block tagged tolerances:{key}"
        assert json.loads(blocks[key]) == json.loads(path.read_text(encoding="utf-8"))
