"""V-05 (epic #218): per-segment physics assumptions / fidelity / limitations.

orbitopt.model_fidelity derives one MissionSegment per mission phase from
the pipeline stage records and aggregates them into the ``model_fidelity``
section of the pipeline-result document. The tests are compute-free: they
drive ``run_pipeline`` with a stand-in builder the way test_pipeline does.
"""
from __future__ import annotations

import copy
import json

import pytest

from orbitopt import mission_config
from orbitopt.model_fidelity import (
    FIDELITIES,
    MissionSegment,
    format_coverage,
    model_coverage,
    segments_from_stages,
)
from orbitopt.pipeline import StageRecorder, run_pipeline, validate_pipeline_result
from orbitopt.viz.scene import body_entry, scene_document

GEO_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "geo-test", "title": "GEO test", "kind": "geo-raising"},
    "output": {"scene_file": "geo-test.json"},
}


def _scene(config):
    trail = {"times": [0.0, 1.0], "positions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    return scene_document(config["mission"]["id"], config["mission"]["title"],
                          [body_entry("spacecraft", "Probe", "#ffffff", "spacecraft", trail=trail)],
                          distance_unit="km")


def physics_builder(config, recorder=None):
    """Mimics a migrated builder that also records the V-05 metadata."""
    recorder = StageRecorder() if recorder is None else recorder
    with recorder.stage("search") as st:
        st.physical_model = {"name": "two-body screening", "fidelity": "two-body"}
        st.record_physics(assumptions=["impulsive burns"],
                          limitations=["screening only"],
                          not_modelled=["solar_radiation_pressure", "atmospheric_drag"])
        st.candidates = [{"n": 1, "selected": True}]
    with recorder.stage("refine") as st:
        st.physical_model = {"name": "N-body re-fly", "fidelity": "high-fidelity"}
        st.record_physics(assumptions=["point-mass bodies"],
                          not_modelled=["solar_radiation_pressure", "relativistic_correction"])
    with recorder.stage("verify") as st:
        st.physical_model = {"name": "skipped perturbed check", "fidelity": "high-fidelity"}
        st.record_physics(not_modelled=["solar_radiation_pressure"])
    recorder.skip("verify", "not run by this builder's story")  # demote the stage above
    return _scene(config)


@pytest.fixture
def physics_runs(monkeypatch):
    monkeypatch.setitem(mission_config._KIND_BUILDERS, "geo-raising", (__name__, "physics_builder"))


def _doc():
    return run_pipeline(copy.deepcopy(GEO_CONFIG)).to_dict()


# ---------------------------------------------------------------------------
# MissionSegment itself
# ---------------------------------------------------------------------------

def test_segment_from_stage_dict_carries_the_metadata():
    seg = MissionSegment.from_stage({
        "stage": "refine", "status": "completed",
        "physical_model": {"name": "N-body", "fidelity": "high-fidelity"},
        "assumptions": ["impulsive TLI"], "limitations": [], "not_modelled": ["SRP"],
    })
    assert seg.name == "refine" and seg.physical_model == "N-body"
    assert seg.fidelity == "high-fidelity" and seg.not_modelled == ("SRP",)
    assert seg.modelled and seg.has_physics
    assert seg.to_dict()["assumptions"] == ["impulsive TLI"]


def test_unknown_fidelity_is_refused():
    with pytest.raises(ValueError, match="fidelity"):
        MissionSegment(name="x", physical_model="m", fidelity="quantum", status="completed")


def test_stage_without_physics_is_a_segment_but_not_a_modelled_one():
    seg = MissionSegment.from_stage({"stage": "export", "status": "completed"})
    assert seg.fidelity == "none" and not seg.has_physics and not seg.modelled


def test_modelled_requires_completed_status():
    seg = MissionSegment.from_stage({"stage": "refine", "status": "failed",
                                     "physical_model": {"name": "m", "fidelity": "two-body"}})
    assert seg.has_physics and not seg.modelled


def test_stage_result_model_property_matches_to_dict():
    rec = StageRecorder()
    st = rec["search"]
    st.physical_model = {"name": "m", "fidelity": "analytic"}
    st.record_physics(assumptions=["a"], limitations=["l"], not_modelled=["n"])
    seg = st.model
    assert isinstance(seg, MissionSegment)
    assert seg.assumptions == ("a",) and seg.limitations == ("l",) and seg.not_modelled == ("n",)
    doc = st.to_dict()
    assert doc["assumptions"] == ["a"] and doc["limitations"] == ["l"] and doc["not_modelled"] == ["n"]


# ---------------------------------------------------------------------------
# Aggregation / mission report section
# ---------------------------------------------------------------------------

def test_pipeline_document_carries_a_valid_model_fidelity_section(physics_runs):
    doc = _doc()
    validate_pipeline_result(doc)
    json.dumps(doc, allow_nan=False)
    cov = doc["model_fidelity"]
    # Physics segments come first, ladder order: search (two-body) then refine.
    names = [s["name"] for s in cov["segments"]]
    assert names[:2] == ["search", "refine"]
    assert cov["fidelity_ladder"] == {"search": "two-body", "refine": "high-fidelity"}
    assert cov["highest_fidelity"] == "high-fidelity"
    # The union over completed physics segments answers "what is NOT modelled".
    assert cov["unmodelled_effects"] == ["atmospheric_drag", "relativistic_correction",
                                        "solar_radiation_pressure"]


def test_skipped_or_unrecorded_physics_is_not_in_the_union(physics_runs):
    # verify was skipped and never recorded -- its SRP note must not leak in...
    cov = _doc()["model_fidelity"]
    assert "verify" not in cov["fidelity_ladder"]
    # ...and a segment without a physics model is listed for completeness only.
    export = next(s for s in cov["segments"] if s["name"] == "export")
    assert export["fidelity"] == "none" and export["not_modelled"] == []


def test_coverage_accepts_the_plain_document(physics_runs):
    doc = _doc()
    assert model_coverage(doc) == doc["model_fidelity"]


def test_format_coverage_shows_per_segment_metadata_and_the_union(physics_runs):
    result = run_pipeline(copy.deepcopy(GEO_CONFIG))
    text = result.summary()
    assert "search: two-body screening" in text
    assert "assumes: impulsive burns" in text
    assert "limitation: screening only" in text
    assert "refine: N-body re-fly" in text
    assert "unmodelled physics (union over segments)" in text
    assert "atmospheric_drag" in text and "relativistic_correction" in text


def plain_builder(config, recorder=None):
    """A builder that records no physics metadata at all."""
    return _scene(config)


def test_no_physics_recorded_yields_an_empty_section(monkeypatch):
    monkeypatch.setitem(mission_config._KIND_BUILDERS, "geo-raising", (__name__, "plain_builder"))
    doc = _doc()
    validate_pipeline_result(doc)
    cov = doc["model_fidelity"]
    assert cov["fidelity_ladder"] == {} and cov["highest_fidelity"] == "none"
    assert cov["unmodelled_effects"] == []
    assert format_coverage(cov) == []


def test_segments_from_stages_accepts_stage_result_objects():
    rec = StageRecorder()
    rec["search"].physical_model = {"name": "m", "fidelity": "two-body"}
    segs = segments_from_stages(rec.results.values())
    assert len(segs) == len(rec.results)
    assert all(s.name in rec.results for s in segs)


def test_all_builders_record_physics_metadata():
    """The acceptance criterion: every mission kind's physics stages state
    their assumptions/limitations/not-modelled -- no silent gaps."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "orbitopt"
    for builder in ("viz/mars_transfer.py", "viz/geo_raising.py", "viz/mission_timeline.py"):
        text = (src / builder).read_text(encoding="utf-8")
        models = re.findall(r'"fidelity": "(?!none)([a-z-]+)"', text)
        records = text.count("record_physics(")
        assert models and records, builder
