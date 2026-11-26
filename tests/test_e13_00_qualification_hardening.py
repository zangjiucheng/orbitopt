"""E13-00 qualification hardening: the digests mean what they claim.

Three layers, each answering one question and nothing else:

* ``plan_digest`` (8 significant digits, unchanged) -- numerically the same
  plan, robust across machines;
* ``verification_digest`` / ``artifact_sha256`` -- the raw acceptance record
  and raw artifact are unedited (flipping a stored safety verdict, editing a
  margin, swapping a field or the provenance strings fails closed);
* the refly report's self-digest + the ``sealed_plan`` / ``worker`` identity
  inside ``inputs.main_model`` -- the independent evidence was produced for
  THIS config, THIS initial state and THIS plan, in a recorded environment.

These are the negative acceptance tests of the task: tampered safety
verdict, swapped independent re-flight report (same mission and maneuvers,
different config or initial state) and a modified raw artifact must all be
refused.
"""
from __future__ import annotations

import copy
import json

import pytest

@pytest.fixture(scope="module")
def reference_sealed():
    from orbitopt.rendezvous import canonical_plan, load_reference_config
    return canonical_plan(load_reference_config())


def _reference_doc():
    import yaml
    from importlib import resources
    return yaml.safe_load(resources.files("orbitopt").joinpath("data", "leo_rdv_001.yaml")
                          .read_text(encoding="utf-8"))


def _reconfig(doc):
    from orbitopt.rendezvous import parse_rendezvous_config
    return parse_rendezvous_config(doc)


def _roundtrip(tmp_path, sealed, cfg):
    from orbitopt.rendezvous import read_sealed_plan, write_sealed_plan
    path = write_sealed_plan(sealed, tmp_path / "sealed.json")
    return json.loads(path.read_text(encoding="utf-8")), path, read_sealed_plan


# ---------------------------------------------------------------------------
# tampered stored safety verdict
# ---------------------------------------------------------------------------

def test_flipped_safety_passed_is_rejected(tmp_path, reference_sealed):
    from orbitopt.rendezvous import PlanSealError, load_reference_config
    doc, path, reader = _roundtrip(tmp_path, reference_sealed, load_reference_config())
    assert doc["safety"]["passed"] is True
    for check in doc["safety"]["checks"].values():
        check["passed"] = False
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PlanSealError, match="contradicts its own checks"):
        reader(path, load_reference_config())


def test_swapped_safety_verdict_for_another_plan_is_rejected(reference_sealed):
    """A safety report carrying another plan's digest cannot be pasted in:
    the verification digest (and the artifact digest) refuse."""
    from orbitopt.rendezvous import PlanSealError, load_reference_config, sealed_from_document
    cfg = load_reference_config()
    doc = reference_sealed.to_document()
    other = copy.deepcopy(doc["safety"])
    other["plan_digest"] = "0" * 64
    doc["safety"] = other
    with pytest.raises(PlanSealError, match="safety"):
        sealed_from_document(doc, cfg)


# ---------------------------------------------------------------------------
# edited acceptance record / raw artifact
# ---------------------------------------------------------------------------

def test_edited_verification_record_is_rejected(reference_sealed):
    from orbitopt.rendezvous import PlanSealError, load_reference_config, sealed_from_document
    cfg = load_reference_config()
    doc = reference_sealed.to_document()
    doc["verification"]["position_error_m"] += 1e-3
    with pytest.raises(PlanSealError, match="verification digest mismatch"):
        sealed_from_document(doc, cfg)


def test_edited_provenance_is_rejected(reference_sealed):
    """Editing a field NO digest recomputes -- the planner provenance string
    -- is caught by the raw artifact digest, not assumed harmless."""
    from orbitopt.rendezvous import PlanSealError, load_reference_config, sealed_from_document
    cfg = load_reference_config()
    doc = reference_sealed.to_document()
    doc["canonical_planner"] = doc["canonical_planner"] + " (edited)"
    with pytest.raises(PlanSealError, match="artifact digest mismatch"):
        sealed_from_document(doc, cfg)


def test_missing_verification_digest_field_is_rejected(reference_sealed):
    from orbitopt.rendezvous import PlanSealError, load_reference_config, sealed_from_document
    doc = reference_sealed.to_document()
    del doc["verification_digest"]
    with pytest.raises(PlanSealError, match="missing its verification_digest"):
        sealed_from_document(doc, load_reference_config())


def test_sealed_document_binds_the_sealing_environment(reference_sealed):
    assert reference_sealed.environment_digest
    doc = reference_sealed.to_document()
    assert doc["environment_digest"] == reference_sealed.environment_digest
    assert doc["artifact_sha256"] == doc["artifact_sha256"]  # present; tamper-tested above


# ---------------------------------------------------------------------------
# swapped independent re-flight report (E12-09 evidence, E13-00 binding)
# ---------------------------------------------------------------------------

def _bound_refly_report(cfg, plan, result="passed", env_digest="e13-00-stub-env"):
    """A synthetic report carrying the FULL identity the hardened verdict
    requires -- sealed_plan triple + plan digest, maneuver digest, worker
    environment -- with its self-digest computed over exactly that content."""
    from orbitopt.rendezvous import (canonical_config_sha256, initial_states_sha256,
                                     maneuvers_sha256, rendezvous_plan_digest)
    from orbitopt.verify.rendezvous_refly import REPORT_KIND
    from orbitopt.verify.report import report_digest

    identity = {
        "config_sha256": canonical_config_sha256(cfg),
        "initial_states_sha256": initial_states_sha256(cfg),
        "maneuvers_sha256": maneuvers_sha256(plan),
        "plan_digest": rendezvous_plan_digest(cfg, plan),
    }
    doc = {"schemaVersion": "1.0", "kind": REPORT_KIND,
           "inputs": {"mission_id": plan.mission_id,
                      "main_model": {"maneuvers_sha256": identity["maneuvers_sha256"],
                                     "sealed_plan": dict(identity),
                                     "worker": {"environment_digest": env_digest}},
                      "error_sources": {
                          "independent_model_agreement": {"result": result}}}}
    doc["digest"] = report_digest(doc)
    return doc, identity


def _other_config():
    doc = _reference_doc()
    doc["chaser"]["initial_separation_m"] = doc["chaser"]["initial_separation_m"] + 50.0
    return _reconfig(doc)


def test_matching_bound_report_verifies(reference_sealed):
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.verify.rendezvous_refly import physics_verdict_from_refly_report
    cfg = load_reference_config()
    doc, _ = _bound_refly_report(cfg, reference_sealed.plan)
    assert physics_verdict_from_refly_report(doc, cfg, reference_sealed.plan) is True


def test_report_for_other_config_is_not_assessed(reference_sealed):
    """Swapped artifact: same mission, same maneuvers (identical burns),
    different CONFIG -- the report is internally coherent and self-digested,
    and still refuses to certify physics for this config."""
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.verify.rendezvous_refly import physics_verdict_from_refly_report
    cfg = load_reference_config()
    other = _other_config()
    assert other.mission_id == cfg.mission_id
    doc, _ = _bound_refly_report(other, reference_sealed.plan)  # honest report of the other run
    assert physics_verdict_from_refly_report(doc, cfg, reference_sealed.plan) is None


def test_report_for_other_initial_state_is_not_assessed(reference_sealed):
    """Same mission, same maneuvers, same claimed config -- but the report's
    initial-state digest is the one from a phased-apart run (honestly
    re-digested). The initial-states binding refuses it."""
    from orbitopt.rendezvous import (initial_states_sha256, load_reference_config,
                                     rendezvous_plan_digest)
    from orbitopt.verify.rendezvous_refly import physics_verdict_from_refly_report
    from orbitopt.verify.report import report_digest
    cfg = load_reference_config()
    doc, identity = _bound_refly_report(cfg, reference_sealed.plan)
    other = _other_config()
    doc["inputs"]["main_model"]["sealed_plan"]["initial_states_sha256"] = \
        initial_states_sha256(other)
    doc["inputs"]["main_model"]["sealed_plan"]["plan_digest"] = \
        rendezvous_plan_digest(cfg, reference_sealed.plan)
    doc["digest"] = report_digest(doc)  # internally coherent; ONLY the initial state lies
    assert initial_states_sha256(other) != initial_states_sha256(cfg)
    assert physics_verdict_from_refly_report(doc, cfg, reference_sealed.plan) is None


def test_modified_report_artifact_is_not_assessed(reference_sealed):
    """A swapped-in verdict on a real report: the self-digest no longer
    recomputes over the edited bytes, so the evidence is not assessed."""
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.verify.rendezvous_refly import physics_verdict_from_refly_report
    cfg = load_reference_config()
    doc, _ = _bound_refly_report(cfg, reference_sealed.plan, result="failed")
    doc["inputs"]["error_sources"]["independent_model_agreement"]["result"] = "passed"
    assert physics_verdict_from_refly_report(doc, cfg, reference_sealed.plan) is None


def test_report_without_environment_binding_is_not_assessed(reference_sealed):
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.verify.rendezvous_refly import physics_verdict_from_refly_report
    from orbitopt.verify.report import report_digest
    cfg = load_reference_config()
    doc, _ = _bound_refly_report(cfg, reference_sealed.plan, env_digest="")
    doc["digest"] = report_digest(doc)  # re-digest so ONLY the env binding is missing
    assert physics_verdict_from_refly_report(doc, cfg, reference_sealed.plan) is None


def test_sidecar_records_consumed_evidence(tmp_path, reference_sealed):
    from orbitopt.rendezvous import build_studio_run, load_reference_config
    cfg = load_reference_config()
    doc, _ = _bound_refly_report(cfg, reference_sealed.plan)
    path = tmp_path / "rendezvous-refly-report.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    studio = build_studio_run(cfg, sample_dt_s=60.0, sealed=reference_sealed,
                              refly_report=path)
    evidence = studio.sidecar["refly_evidence"]
    assert evidence["report_digest"] == doc["digest"]
    assert evidence["worker_environment_digest"] == "e13-00-stub-env"
    import hashlib
    assert evidence["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert studio.run.verification.physics_verified is True
