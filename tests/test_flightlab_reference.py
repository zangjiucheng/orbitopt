"""FL-601: the committed Gate B regression reference must stay exactly as
scripts/fl601_reference.py wrote it. Hand-editing any file under
tests/data/flightlab/reference/ -- a trajectory value, a check verdict, a
tolerance -- fails the provenance check here (in the fast CI job, no OT
binary needed); the numerical comparison against a fresh real-binary flight
is the slow `flightlab-gateb` job in .github/workflows/ci.yml. Together with
the [fl601-reference-update] commit-tag guard (G5) this makes a reference
change deliberate: regenerate with the script, tag the commit, CI agrees.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = Path(__file__).parent / "data" / "flightlab" / "reference"


def _script():
    spec = importlib.util.spec_from_file_location(
        "fl601_reference", REPO / "scripts" / "fl601_reference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_reference_matches_its_provenance():
    assert _script().cmd_verify() == 0, (
        "reference files changed without scripts/fl601_reference.py update "
        "-- see tests/test_flightlab_reference.py docstring")


def test_reference_manifest_records_the_recording_toolchain():
    manifest = json.loads((REF_ROOT / "gateb-leo-two-stage" / "manifest.json")
                          .read_text(encoding="utf-8"))
    toolchain = manifest["toolchain"]
    assert toolchain["opentsiolkovsky"] == "1.0.0"
    assert all(c in "0123456789abcdef" for c in toolchain["opentsiolkovsky_sha256"])


def test_declared_tolerances_are_complete():
    script = _script()
    declared = json.loads((REF_ROOT / "tolerances.json").read_text(encoding="utf-8"))
    assert set(declared) == set(script.DEFAULT_TOLERANCES)
    assert set(declared["extra_abs"]) == set(script.DEFAULT_TOLERANCES["extra_abs"])
    assert all(isinstance(v, (int, float)) and v >= 0 for v in declared["extra_abs"].values())


def test_reference_run_dir_is_a_valid_flight_result():
    pytest.importorskip("jsonschema")
    from orbitopt.flightlab.contract import is_valid_run

    assert is_valid_run(REF_ROOT / "gateb-leo-two-stage")
