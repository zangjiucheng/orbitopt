"""Candidate Selection and the Verification Gate (orbitopt.candidate_selection,
task #171). Compute-free: candidates are plain tuples and verification
reports are minimal tri-state stand-ins."""
from __future__ import annotations

import copy
import json
import math

import pytest

from orbitopt import mission_config
from orbitopt.candidate_selection import (
    BEST_EFFORT_WARNING,
    VerificationGate,
    mark_scene_best_effort,
    select_candidates,
    verification_outcome,
    walk_candidates,
)
from orbitopt.pipeline import StageRecorder, run_pipeline, validate_pipeline_result
from orbitopt.verify import report as verification_report
from orbitopt.viz.scene import body_entry, scene_document

PASS = {"kind": "moi", "label": "ok", "status": verification_report.status_from_checks({"a": True})}
FAIL = {"kind": "tcm", "label": "bad", "status": verification_report.status_from_checks({"miss": False, "a": True})}
UNCHECKED = {"kind": "moi", "label": "off", "status": verification_report.status_from_checks({})}


def _select(items, **kw):
    kw.setdefault("tolerance", (1.0, 2.0))
    return select_candidates(items, cost=lambda it: it[2], coords=lambda it: it[:2], **kw)


# --- Candidate Selection --------------------------------------------------

def test_selection_dedupes_ranks_and_flags_infeasible():
    items = [(100.0, 200.0, 5.0), (100.5, 201.0, 4.0), (150.0, 250.0, 6.0), (10.0, 10.0, math.nan)]
    sel = _select(items)
    assert sel.selected_indices == [1, 2]  # #0 is a duplicate of the cheaper #1
    recs = {r["index"]: r for r in sel.records}
    assert recs[0]["status"] == "duplicate" and recs[0]["duplicate_of"] == 1
    assert "duplicate of candidate #1" in recs[0]["reason"]
    assert recs[3]["status"] == "infeasible" and recs[3]["cost"] is None
    assert [recs[i]["rank"] for i in (1, 2)] == [0, 1]
    assert sel.counts() == {"selected": 2, "duplicate": 1, "infeasible": 1, "truncated": 0}
    assert [r["status"] for r in sel.to_candidates()] == ["selected", "selected", "duplicate", "infeasible"]


def test_selection_keeps_diverse_candidates_when_truncating():
    # #1 is cheap but right next to #0; #2 is a distinct, costlier family.
    items = [(100.0, 200.0, 1.0), (105.0, 210.0, 2.0), (200.0, 300.0, 3.0)]
    sel = _select(items, min_separation=(20.0, 40.0), max_candidates=2)
    assert sel.selected_indices == [0, 2]
    trunc = next(r for r in sel.records if r["index"] == 1)
    assert trunc["status"] == "truncated" and "max_candidates=2" in trunc["reason"]
    assert "diversity" in trunc["reason"]
    # Without the diversity scale it's pure cost order.
    assert _select(items, max_candidates=2).selected_indices == [0, 1]


def test_selection_fills_with_non_diverse_when_too_few_diverse():
    items = [(100.0, 200.0, 1.0), (105.0, 210.0, 2.0)]
    sel = _select(items, min_separation=(20.0, 40.0), max_candidates=3)
    assert sel.selected_indices == [0, 1]
    assert "kept to fill the list" in sel.records[1]["reason"]


# --- Verification Gate ----------------------------------------------------

def test_verification_outcome_is_tri_state_with_reasons():
    assert verification_outcome([PASS]) == ("passed", [])
    result, reasons = verification_outcome([PASS, FAIL])
    assert result == "failed" and reasons == ["tcm:bad FAILED (miss)"]
    assert verification_outcome([PASS, UNCHECKED]) == ("not_checked", ["moi:off NOT CHECKED"])
    assert verification_outcome([])[0] == "not_checked"


@pytest.mark.parametrize("report, verdict", [(PASS, "accepted"), (FAIL, "degraded"), (UNCHECKED, "degraded")])
def test_only_passed_is_accepted(report, verdict):
    gate = VerificationGate()
    gate.evaluate(0, [report])
    gate.export(0)
    assert gate.candidates[0]["verdict"] == verdict
    assert gate.success is (verdict == "accepted")
    assert gate.best_effort is (verdict != "accepted")


def _flyer(outcomes):
    """fly() stand-in: outcome per index is a report, or an exception text."""
    calls = []

    def fly(index, payload, strict, rejected):
        calls.append((index, strict, tuple(rejected)))
        out = outcomes[index] if strict else outcomes.get(("relaxed", index), PASS)
        if isinstance(out, str):
            raise RuntimeError(out)
        return {"reports": [out], "index": index}
    return fly, calls


def _walk(outcomes, n=3, **kw):
    gate = VerificationGate()
    fly, calls = _flyer(outcomes)
    out = walk_candidates([(i, f"cand{i}") for i in range(n)], fly, gate=gate,
                          reports_of=lambda r: r["reports"], **kw)
    return gate, out, calls


def test_gate_skips_rejected_and_degraded_until_one_passes():
    gate, out, calls = _walk({0: "TCM did not converge", 1: FAIL, 2: PASS})
    assert out.index == 2 and not out.best_effort and gate.success
    assert [c["verdict"] for c in gate.candidates] == ["rejected", "degraded", "accepted"]
    assert gate.candidates[0]["reasons"] == ["TCM did not converge"]
    assert gate.candidates[1]["reasons"] == ["tcm:bad FAILED (miss)"]
    assert calls[-1] == (2, True, (0,))
    assert gate.warnings() == []


def test_gate_exports_best_degraded_best_effort_with_warning():
    gate, out, _ = _walk({0: UNCHECKED, 1: "boom", 2: FAIL})
    assert out.index == 0 and out.best_effort
    doc = gate.to_dict()
    assert doc["success"] is False and doc["best_effort"] is True and doc["result"] == "not_checked"
    assert doc["warnings"] == [BEST_EFFORT_WARNING]
    assert doc["counts"] == {"accepted": 0, "degraded": 2, "rejected": 1, "not_evaluated": 0}


def test_gate_refly_fallback_when_everything_rejected():
    gate, out, calls = _walk({0: "a", 1: "b", 2: "c", ("relaxed", 0): PASS})
    assert out.index == 0 and out.best_effort and not gate.success
    assert calls[-1] == (0, False, (0, 1, 2))
    # Even though the relaxed re-flight's reports pass, it is not a success.
    assert gate.to_dict()["warnings"] == [BEST_EFFORT_WARNING]
    with pytest.raises(RuntimeError, match="every selected candidate was rejected"):
        _walk({0: "a", 1: "b", 2: "c"}, fallback=False)


def test_mark_scene_best_effort_puts_warning_first():
    gate = VerificationGate()
    gate.evaluate(0, [FAIL])
    gate.export(0)
    scene = {"subtitle": "line 1"}
    mark_scene_best_effort(scene, gate)
    mark_scene_best_effort(scene, gate)  # idempotent
    assert scene["warnings"] == [BEST_EFFORT_WARNING]
    assert scene["subtitle"] == f"WARNING: {BEST_EFFORT_WARNING}\nline 1"


# --- pipeline integration -------------------------------------------------

CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "mars-gate", "title": "Mars gate", "kind": "mars-transfer"},
    "output": {"scene_file": "mars-gate.json"},
}
GATE_OUTCOMES = {}


def _scene(config):
    trail = {"times": [0.0, 1.0], "positions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    return scene_document(config["mission"]["id"], config["mission"]["title"],
                          [body_entry("spacecraft", "Probe", "#ffffff", "spacecraft", trail=trail)],
                          distance_unit="km")


def gating_builder(config, recorder=None):
    """Mimics mars_transfer: select stage + walk_candidates over the gate."""
    recorder = StageRecorder() if recorder is None else recorder
    items = [(100.0, 200.0, 1.0), (100.2, 200.5, 1.1), (200.0, 300.0, 2.0), (300.0, 150.0, 3.0)]
    with recorder.stage("select") as st:
        sel = _select(items, min_separation=(20.0, 40.0), max_candidates=3)
        st.candidates = sel.to_candidates()
    for rank, i in enumerate(sel.selected_indices):
        recorder.gate.register(i, selection_rank=rank)

    def fly(index, payload, strict, rejected):
        out = GATE_OUTCOMES[index]
        if isinstance(out, str):
            raise RuntimeError(out)
        scene = _scene(config)
        verification_report.attach_to_scene(scene, out)
        return scene

    outcome = walk_candidates([(i, items[i]) for i in sel.selected_indices], fly, gate=recorder.gate,
                              reports_of=lambda s: s.get(verification_report.SCENE_REPORTS_KEY, []))
    return mark_scene_best_effort(outcome.result, recorder.gate)


@pytest.fixture
def gating(monkeypatch):
    monkeypatch.setitem(mission_config._KIND_BUILDERS, "mars-transfer", (__name__, "gating_builder"))
    return GATE_OUTCOMES


def test_pipeline_traces_rejected_and_degraded_candidates(gating):
    gating.clear()
    gating.update({0: "TCM did not converge", 2: FAIL, 3: PASS})
    result = run_pipeline(copy.deepcopy(CONFIG))
    doc = result.to_dict()
    validate_pipeline_result(doc)
    assert doc["success"] is True and doc["verification"]["result"] == "passed"
    gate = doc["gate"]
    assert gate["exported_candidate"] == 3
    assert {c["index"]: c["verdict"] for c in gate["candidates"]} == {0: "rejected", 2: "degraded", 3: "accepted"}
    select = next(s for s in doc["stages"] if s["stage"] == "select")
    assert {c["index"]: c["status"] for c in select["candidates"]}[1] == "duplicate"
    assert "warnings" not in result.scene


def test_pipeline_best_effort_is_not_success_and_warns(gating, tmp_path, capsys):
    gating.clear()
    gating.update({0: FAIL, 2: "rejected", 3: UNCHECKED})
    result = run_pipeline(copy.deepcopy(CONFIG))
    doc = result.to_dict()
    validate_pipeline_result(doc)
    assert doc["status"] == "completed" and doc["success"] is False
    assert doc["gate"]["best_effort"] is True and doc["gate"]["exported_candidate"] == 0
    assert doc["verification"]["result"] == doc["gate"]["result"] == "failed"
    assert result.scene["warnings"] == [BEST_EFFORT_WARNING]
    assert result.scene["subtitle"].startswith("WARNING: BEST-EFFORT")
    assert "WARNING: " + BEST_EFFORT_WARNING in result.summary()

    from orbitopt.cli import main

    cfg = tmp_path / "mars.yaml"
    cfg.write_text(json.dumps(CONFIG))
    main(["run", str(cfg), "-o", str(tmp_path / "mars.json")])
    out = capsys.readouterr().out
    assert f"WARNING: {BEST_EFFORT_WARNING}" in out and "verification gate: NOT SUCCESSFUL" in out
    written = json.loads((tmp_path / "mars.pipeline.json").read_text())
    assert written["success"] is False and written["gate"]["warnings"] == [BEST_EFFORT_WARNING]


def minimal_builder(config, recorder=None):
    scene = _scene(config)
    if config.get("_report"):
        verification_report.attach_to_scene(scene, config.pop("_report"))
    return scene


@pytest.mark.parametrize("report, success, best_effort", [(PASS, True, False), (FAIL, False, True),
                                                           (None, False, False)])
def test_generic_gate_for_builders_that_dont_drive_it(monkeypatch, report, success, best_effort):
    monkeypatch.setitem(mission_config._KIND_BUILDERS, "geo-raising", (__name__, "minimal_builder"))
    config = {"schemaVersion": "1.0", "mission": {"id": "g", "title": "G", "kind": "geo-raising"},
              "output": {"scene_file": "g.json"}}
    monkeypatch.setattr(mission_config, "validate_mission_config", lambda c: None)
    if report is not None:
        config["_report"] = report
    result = run_pipeline(config)
    doc = result.to_dict()
    validate_pipeline_result(doc)
    assert doc["success"] is success and doc["gate"]["best_effort"] is best_effort
    assert bool(doc["gate"]["warnings"]) is (not success)
    assert doc["gate"]["exported_candidate"] == 0
