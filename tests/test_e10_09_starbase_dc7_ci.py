"""E10-09 (task #306): the final schemes' CI qualification gate, fast half.

The end-to-end half is the ``cross-env`` job's E10-09 steps in
``.github/workflows/ci.yml`` (DC7 and Starbase each fly their own frozen plan
across the tudatpy/gfold boundary and get re-graded by
``scripts/e1009_starbase_dc7.py crossenv-check``). What runs here without a
second interpreter:

* the frozen audit record itself -- tolerance digests and values match the
  sealed trees byte-for-byte, the audit trail records E10-04's no-band-change
  finding, and the tolerance-change guard fires on the right files;
* the audit gate's diff logic (``audit_required``);
* the evidence grader against synthetic sealed legs, positive and negative:
  a fabricated joint verdict, a tampered envelope, an unpinned evidence source
  or a doctored qualification block must each be refused -- and an honest
  run must report ``independently_verified`` False with exactly the one
  admissible blocker reason;
* the CI job shape itself: two E10-09 scheme steps, the retained expected-fail
  E5-08 negative regression, and the sealed continuity targets (both legs run
  under the sealed Earth-orientation snapshot; recovery is driven from the
  handoff the sealed mission records).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from orbitopt.crossenv.envelope import envelope_digest, write_envelope
from orbitopt.crossenv.qualification import (
    JOINT_KIND,
    RECOVERY_CHECKS_KIND,
    joint_qualification,
)

_spec = importlib.util.spec_from_file_location(
    "e1009", REPO / "scripts/e1009_starbase_dc7.py")
e1009 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1009)

AUDIT = REPO / "tests/data/mission/reference/e10-09-ci/tolerance-audit.json"
CI = REPO / ".github/workflows/ci.yml"
REFS = REPO / "tests/data/mission/reference"
FROZEN_ASCENT = REFS / "e2-04-frozen-mission" / "ascent"
SCENARIO = REPO / "src/orbitopt/data/mission_two_stage_leo_rtls_replay.yaml"

PHYSICAL_FIVE = ("orbit_insertion", "parking_propagation", "landing",
                 "mass_conservation", "epoch_frame_continuity")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------- audit record


def test_verify_passes_on_the_frozen_trees():
    assert e1009.cmd_verify() == 0


def test_audit_digests_and_values_match_the_sealed_trees():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["kind"] == e1009.AUDIT_KIND
    assert set(audit["schemes"]) == set(e1009.SCHEMES)
    for name, scheme in e1009.SCHEMES.items():
        entry = audit["schemes"][name]
        tol = scheme["reference"] / "tolerances.json"
        assert entry["tolerances_sha256"] == _sha(tol)
        assert entry["crossenv_plan_sha256"] == _sha(scheme["plan"])
        assert entry["values"] == json.loads(tol.read_text(encoding="utf-8"))
        assert entry["crossenv_plan"] in (
            scheme["plan"].relative_to(REPO).as_posix(),)


def test_both_schemes_share_one_tolerance_band_set():
    d = e1009.SCHEMES["dc7"]["reference"] / "tolerances.json"
    s = e1009.SCHEMES["starbase"]["reference"] / "tolerances.json"
    assert _sha(d) == _sha(s)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert (audit["schemes"]["dc7"]["tolerances_sha256"]
            == audit["schemes"]["starbase"]["tolerances_sha256"])
    vals = audit["schemes"]["dc7"]["values"]
    for key in ("maxq_pa", "maxg", "separation_mass_kg", "touchdown_pos_m",
                "touchdown_vert_m_s", "touchdown_lateral_m_s",
                "touchdown_propellant_kg", "plan_script_rel"):
        assert key in vals, f"{key} missing from the frozen band set"


def test_audit_trail_records_that_no_band_was_ever_relaxed():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    for name in e1009.SCHEMES:
        trail = audit["schemes"][name]["audit_trail"]
        assert trail, name
        assert trail[-1]["as_of"].startswith("E10-09"), "freeze entry must be last"
        e1004 = [e for e in trail if e["as_of"].startswith("E10-04")]
        assert e1004 and "NO band relaxed" in e1004[0]["change"], (
            "the E10-04 comparator-artifact finding must be on record")


def test_negative_regression_stays_sealed_as_a_gap():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    neg = audit["negative_regression"]
    ref = REPO / neg["reference"]
    assert ref == e1009.NEGATIVE_REFERENCE
    prov = json.loads((ref / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["reference_class"] == "new-vehicle-qualification-gap"
    script = json.loads((ref / "recovery-script.json").read_text(encoding="utf-8"))
    assert script["status"] == "not_closed"


def test_tolerances_change_requires_the_audit_file_in_the_same_diff():
    audit_rel = AUDIT.relative_to(REPO).as_posix()
    tol = (e1009.SCHEMES["dc7"]["reference"] / "tolerances.json"
           ).relative_to(REPO).as_posix()
    plan = (e1009.SCHEMES["starbase"]["plan"]).relative_to(REPO).as_posix()
    assert e1009.audit_required([tol], audit_rel) == [tol]
    assert e1009.audit_required([plan, "README.md"], audit_rel) == [plan]
    assert e1009.audit_required([tol, audit_rel], audit_rel) == []
    assert e1009.audit_required(["docs/verification.md"], audit_rel) == []


# ------------------------------------------------------------------- CI shape


@pytest.fixture(scope="module")
def cross_env_steps() -> dict:
    doc = yaml.safe_load(CI.read_text(encoding="utf-8"))
    return {s["name"]: s for s in doc["jobs"]["cross-env"]["steps"] if "name" in s}


def test_ci_runs_both_final_schemes_with_the_e1009_grader(cross_env_steps):
    for variant, plan in (("dc7", "crossenv_e508_dc7_plan.yaml"),
                          ("starbase", "crossenv_e508_starbase_plan.yaml")):
        step = next(s for n, s in cross_env_steps.items()
                    if n.startswith("E10-09 scheme") and variant in n.lower())
        run = step["run"]
        assert f"crossenv run --plan src/orbitopt/data/{plan}" in run
        assert "e1009_starbase_dc7.py crossenv-check" in run
        assert f"--variant {variant}" in run
        assert "ORBITOPT_TUDAT_PYTHON" in run
        assert "ORBITOPT_GFOLD_PYTHON" in (str(step.get("env", "")) + run)
    assert "crossenv_e508_plan.yaml" not in cross_env_steps[
        "E10-09 scheme DC7 crosses the environment boundary green"]["run"]


def test_ci_keeps_the_original_e508_as_the_negative_regression(cross_env_steps):
    step = cross_env_steps["E5-08 original (expected fail) retained as negative regression"]
    assert "crossenv_e508_plan.yaml" in step["run"]
    assert "scripts/e508_new_vehicle.py crossenv-check" in step["run"]


def test_ci_has_the_e1009_g5_pair(cross_env_steps):
    assert any(n.startswith("E10-09 final schemes") for n in cross_env_steps)
    tag = next(s for n, s in cross_env_steps.items()
               if n.startswith("E10-09 tolerance changes"))
    assert "e1009_starbase_dc7.py tag-check" in tag["run"]


# ------------------------------------------------- evidence grader, synthetic


def _c(check_id: str, passed: bool = True) -> dict:
    return {"id": check_id, "passed": bool(passed), "value": None, "limit": None,
            "unit": "", "reason_code": "" if passed else check_id}


ASCENT_CHECKS = [
    _c("station.separation_event"), _c("ascent.flight_verified"),
    _c("ascent.optimizer_converged"), _c("expected.apogee_altitude"),
    _c("expected.perigee_altitude"), _c("handoff.staging.epoch_monotonic"),
    _c("parking.epoch_monotonic"), _c("mass.bookkeeping"),
    _c("propagation.ran"), _c("propagation.epoch_forward"),
    _c("propagation.mass_conservation"), _c("propagation.report_passed"),
]
RECOVERY_CHECKS = [
    _c("recovery.landed"), _c("ascent.insertion_verified"),
    _c("recovery.propellant_reserve"), _c("recovery.touchdown_dispersion"),
    _c("recovery.handoff.epoch_monotonic"), _c("recovery.handoff.mass_conservation"),
]


def _joined(tmp_path: Path) -> dict:
    """A synthetic joined report over freshly sealed legs (the
    test_crossenv_qualification._seal_legs shape)."""
    asc = tmp_path / "ascent" / "exchange"
    import shutil
    shutil.copytree(FROZEN_ASCENT, asc / "ascent-run")
    report = {"kind": "orbitopt.crossenv.report", "schema_version": "1.0",
              "label": "wrapped mission report (synthetic)",
              "report": {"status": "ok", "checks": list(ASCENT_CHECKS),
                         "not_assessed": [], "propagation": {"status": "ran"}}}
    (asc / "reports").mkdir(parents=True, exist_ok=True)
    (asc / "reports" / "mission-report.json").write_text(json.dumps(report),
                                                         encoding="utf-8")
    write_envelope(asc, stage="ascent", runner="ascent", env_label="tudat", payloads=[
        {"name": "ascent-run", "type": "flight-run", "schema_version": "1.0",
         "path": "ascent-run", "digest_kind": "run-dir"},
        {"name": "mission-report", "type": "report", "schema_version": "1.0",
         "report_kind": "orbitopt.crossenv.report",
         "path": "reports/mission-report.json", "digest_kind": "file"},
    ])
    rec = tmp_path / "recovery" / "exchange"
    (rec / "reports").mkdir(parents=True, exist_ok=True)
    (rec / "reports" / "recovery-checks.json").write_text(json.dumps(
        {"kind": RECOVERY_CHECKS_KIND, "schema_version": "1.0",
         "label": "recovery-leg checks (synthetic)",
         "success": True, "landed": True, "checks": list(RECOVERY_CHECKS),
         "not_assessed": []}), encoding="utf-8")
    write_envelope(rec, stage="recovery", runner="recovery", env_label="gfold", payloads=[
        {"name": "recovery-checks", "type": "report", "schema_version": "1.0",
         "report_kind": RECOVERY_CHECKS_KIND,
         "path": "reports/recovery-checks.json", "digest_kind": "file"},
    ])
    stages = [
        {"id": "ascent", "runner": "ascent", "env": "tudat",
         "python_executable": sys.executable, "exchange": str(asc),
         "exchange_digest": envelope_digest(asc), "status": "completed"},
        {"id": "recovery", "runner": "recovery", "env": "gfold",
         "python_executable": sys.executable, "exchange": str(rec),
         "exchange_digest": envelope_digest(rec), "status": "completed"},
    ]
    joint = joint_qualification(scenario_path=SCENARIO, stages=stages,
                                plan_full_mission=True)
    assert joint["kind"] == JOINT_KIND
    return {"kind": "orbitopt.crossenv.mission-report", "schema_version": "1.0",
            "scenario": str(SCENARIO), "stages": stages,
            "joint_qualification": joint}


class TestEvidenceGrader:
    def test_honest_green_run_grades_clean(self, tmp_path):
        joined = _joined(tmp_path)
        doc = joined["joint_qualification"]
        assert doc["verdict"] == "passed"
        assert {c["check"] for c in doc["physical"]["checks"]} == set(PHYSICAL_FIVE)
        assert e1009.grade_joint_evidence(tmp_path, joined) == []

    def test_independently_verified_false_is_recorded_with_its_one_reason(
            self, tmp_path):
        joined = _joined(tmp_path)
        qual = joined["joint_qualification"]["qualification"]
        assert qual["independently_verified"] is False
        assert "no passing validation-grade independent evidence" in qual["reason"]
        assert e1009.grade_joint_evidence(tmp_path, joined) == []

    def test_evidence_turned_red_after_reporting_is_refused(self, tmp_path):
        """The report keeps its green joint document while the SEALED evidence
        is re-sealed red and the stage digest updated to match: the re-grade
        must catch the fabrication, never trust the report."""
        joined = _joined(tmp_path)
        rec = tmp_path / "recovery" / "exchange"
        doc = rec / "reports" / "recovery-checks.json"
        data = json.loads(doc.read_text(encoding="utf-8"))
        data["success"] = data["landed"] = False
        for c in data["checks"]:
            c["passed"] = False
            c["reason_code"] = c["id"]
        doc.write_text(json.dumps(data), encoding="utf-8")
        write_envelope(rec, stage="recovery", runner="recovery", env_label="gfold",
                       payloads=[{"name": "recovery-checks", "type": "report",
                                  "schema_version": "1.0",
                                  "report_kind": RECOVERY_CHECKS_KIND,
                                  "path": "reports/recovery-checks.json",
                                  "digest_kind": "file"}])
        joined["stages"][1]["exchange_digest"] = envelope_digest(rec)
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("verdict re-graded" in f for f in fails)
        assert any("landing" in f and "now hashes" in f for f in fails)

    def test_tampered_envelope_is_refused(self, tmp_path):
        joined = _joined(tmp_path)
        doc = tmp_path / "recovery" / "exchange" / "reports" / "recovery-checks.json"
        data = json.loads(doc.read_text(encoding="utf-8"))
        for c in data["checks"]:
            c["passed"] = False
        doc.write_text(json.dumps(data), encoding="utf-8")
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("did not re-verify" in f for f in fails)

    def test_unpinned_evidence_source_is_refused(self, tmp_path):
        joined = _joined(tmp_path)
        for check in joined["joint_qualification"]["physical"]["checks"]:
            if check["check"] == "landing":
                check["sources"] = [s for s in check["sources"]
                                    if not s.get("exchange_digest")]
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("landing" in f and "evidence source" in f for f in fails)

    def test_doctored_qualification_block_is_refused(self, tmp_path):
        joined = _joined(tmp_path)
        joined["joint_qualification"]["qualification"]["independently_verified"] = True
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("qualification block" in f for f in fails)

    def test_wrong_independent_blocker_reason_is_refused(self, tmp_path):
        joined = _joined(tmp_path)
        joined["joint_qualification"]["qualification"]["reason"] = "something else"
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("unexpected reason" in f for f in fails)

    def test_envelope_swapped_after_reporting_is_refused(self, tmp_path):
        joined = _joined(tmp_path)
        joined["stages"][1]["exchange_digest"] = "sha256:" + "0" * 64
        fails = e1009.grade_joint_evidence(tmp_path, joined)
        assert any("hashes" in f for f in fails)


# --------------------------------------------------- sealed continuity targets


def test_continuity_targets_are_frozen_in_both_sealed_trees():
    for scheme in e1009.SCHEMES.values():
        ref = scheme["reference"]
        assert (ref / "earth_orientation.json").is_file()
        sealed = json.loads((ref / "mission" / "mission.verification.json")
                            .read_text(encoding="utf-8"))
        handoff = (sealed.get("inputs", {}).get("handoffs", {})
                   .get("coupled_branches", {}) or {}).get("handoff_id")
        expect = "f86bc" if ref.name.endswith("dc7") else "9d3ac"
        assert handoff and handoff.startswith(expect), ref.name
        assert (ref / "mission" / "recovery" / "run" / "manifest.json").is_file()


def test_continuity_grader_refuses_an_empty_run(tmp_path):
    fails = e1009.grade_sealed_continuity(tmp_path, e1009.SCHEMES["dc7"])
    assert any("ascent" in f and "stage-report" in f for f in fails)
    assert any("recovery" in f and "stage-report" in f for f in fails)
