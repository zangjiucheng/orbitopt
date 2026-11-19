"""E8-01 (task #281): full-chain reference validation of the E5-05 mission.

Guards the ``full-chain-reference`` report produced by
:mod:`orbitopt.verify.full_chain`:

* every graded check passes and the report is digest-bound;
* the E4-03 evidence map covers the five acceptance events (MECO,
  separation, SECO, insertion, landing) and every check carries a category
  and an evidence type;
* no check claims ``historical_telemetry`` -- the landing leg is honestly
  labelled design-input / numerical-consistency / analytical;
* every tolerance band has a non-empty basis and the report embeds the
  table (reviewer never needs the source);
* the report refuses a tampered reference tree (digests re-checked), and
  refuses to build at all when the reference is absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from orbitopt.verify import full_chain
from orbitopt.verify.full_chain import (
    FULL_CHAIN_REPORT_KIND,
    FULL_CHAIN_TOLERANCES,
    ReferenceNotFound,
    build_full_chain_report,
    verify_reference_digests,
)
from orbitopt.verify.report import EVIDENCE_TYPES, report_digest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = Path(__file__).parent / "data" / "mission" / "reference" / "e5-05-reference-mission"

NOT_RECORDED = pytest.mark.skipif(
    not REF_ROOT.is_dir(),
    reason="E5-05 reference not recorded yet (scripts/e505_reference.py update)")


@pytest.fixture(scope="module")
def report() -> dict:
    return build_full_chain_report()


@NOT_RECORDED
def test_report_kind_and_verdict(report: dict) -> None:
    assert report["kind"] == FULL_CHAIN_REPORT_KIND
    status = report["status"]
    assert status["result"] == "passed" and status["passed"] is True
    assert len(status["checks"]) >= 20
    assert all(status["checks"].values())
    # the ballistic-window pair really is the evidence base (E8-05 relabel:
    # OT-raw-output extraction checks no longer count as independent models)
    assert status["independent_validation"]["result"] == "passed"
    assert status["independent_validation"]["checks_passed"] >= 2


@NOT_RECORDED
def test_every_check_categorised_and_labelled(report: dict) -> None:
    status = report["status"]
    assert set(status["check_categories"]) == set(status["checks"])
    assert set(status["check_evidence"]) == set(status["checks"])
    assert set(status["check_evidence"].values()) <= set(EVIDENCE_TYPES)
    # E4-03 honesty: nothing here is telemetry of a real flight
    assert "historical_telemetry" not in status["check_evidence"].values()
    assert status["evidence"]["historical_telemetry"]["result"] == "not_checked"


@NOT_RECORDED
def test_acceptance_events_are_covered(report: dict) -> None:
    ev = report["status"]["check_evidence"]
    for key in ("ascent.event.meco", "ascent.event.separation", "ascent.event.seco",
                "insertion.semi_major_axis", "insertion.eccentricity",
                "recovery.event.touchdown", "recovery.touchdown_gates"):
        assert key in ev, f"acceptance coverage missing {key}"
    for key in ("ascent.event.meco", "ascent.event.separation", "ascent.event.seco"):
        assert ev[key] == "numerical_consistency"
    assert ev["insertion.semi_major_axis"] == "analytical"
    assert ev["recovery.touchdown_gates"] == "design_input"
    landing = [k for k in ev if k.startswith("recovery.")]
    assert any(ev[k] == "analytical" for k in landing), \
        "the recovery leg needs the closed-form coast-arc reference"
    assert "analytical" in ev["ascent.stage2_dv_ledger"]


@NOT_RECORDED
def test_tolerances_are_auditable_and_embedded_in_the_report(report: dict) -> None:
    details = report["inputs"]["check_details"]
    assert {d["id"] for d in details} == set(FULL_CHAIN_TOLERANCES)
    assert report["inputs"]["tolerances"] == json.loads(
        json.dumps({k: dict(v) for k, v in FULL_CHAIN_TOLERANCES.items()}))
    for band in FULL_CHAIN_TOLERANCES.values():
        assert isinstance(band["basis"], str) and len(band["basis"]) >= 40
        assert band["limit"] >= 0.0 and isinstance(band["unit"], str)
    for d in details:
        assert d["basis"] == FULL_CHAIN_TOLERANCES[d["id"]]["basis"]
        assert d["reference"]
        assert d["passed"] is True and abs(d["value"]) <= d["limit"]


@NOT_RECORDED
def test_report_is_digest_bound(report: dict) -> None:
    unsigned = {k: v for k, v in report.items() if k != "digest"}
    assert report["digest"] == report_digest(unsigned)


@NOT_RECORDED
def test_reference_digests_checked_and_required(tmp_path: Path, monkeypatch) -> None:
    digests = verify_reference_digests()
    assert digests["ok"] and digests["files_checked"] >= 20

    forged = tmp_path / "ref"
    forged.mkdir()
    (forged / "PROVENANCE.json").write_text(
        json.dumps({"files": {"one.json": "0" * 64}}), encoding="utf-8")
    (forged / "one.json").write_text("tampered", encoding="utf-8")
    assert verify_reference_digests(forged)["ok"] is False

    with pytest.raises(ReferenceNotFound):
        build_full_chain_report(tmp_path)


@NOT_RECORDED
def test_reference_root_default_is_the_committed_tree() -> None:
    assert full_chain.REFERENCE_ROOT == REPO / (
        "tests/data/mission/reference/e5-05-reference-mission")


@NOT_RECORDED
def test_observed_residuals_sit_well_inside_every_band(report: dict) -> None:
    # every documented band is loose enough to be a physics statement, yet
    # each recorded residual is finite (guards against inf/NaN drift)
    for d in report["inputs"]["check_details"]:
        assert math.isfinite(d["value"])
        assert abs(d["value"]) <= d["limit"]
