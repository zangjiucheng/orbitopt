"""E12-08 canonical plan qualification: one sealed plan, all consumers.

The task's acceptance is identity, not numbers: the safety report, the
TudatPy re-flight, the Mission Studio replay and the Monte Carlo campaign
must all record the SAME plan digest, and every consumer must fail loudly
when the digest it is handed does not match what it actually runs. These
tests pin the sealing, the guards, and the recorded identity -- the numbers
themselves belong to the E12-04/05/06/07 suites.
"""
from __future__ import annotations

import copy
import dataclasses
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


# ---------------------------------------------------------------------------
# the seal itself
# ---------------------------------------------------------------------------

def test_plan_digest_is_the_document_triple(reference_sealed):
    from orbitopt.rendezvous import (canonical_config_sha256, initial_states_sha256,
                                     maneuvers_sha256, rendezvous_plan_digest)
    from orbitopt.rendezvous import load_reference_config
    cfg = load_reference_config()
    sealed = reference_sealed
    assert sealed.config_sha256 == canonical_config_sha256(cfg)
    assert sealed.initial_states_sha256 == initial_states_sha256(cfg)
    assert sealed.maneuvers_sha256 == maneuvers_sha256(sealed.plan)
    assert sealed.plan_digest == rendezvous_plan_digest(cfg, sealed.plan)
    # the canonical digest covers the full config -- including the E12-04
    # safety block, which the pre-E12-08 studio subset digest omitted.
    doc = _reference_doc()
    doc["safety"]["keep_out_radius_m"] *= 1.0001
    assert canonical_config_sha256(_reconfig(doc)) != canonical_config_sha256(cfg)


def test_canonical_plan_is_deterministic():
    from orbitopt.rendezvous import canonical_plan, load_reference_config
    a = canonical_plan(load_reference_config())
    b = canonical_plan(load_reference_config())
    assert a.plan_digest == b.plan_digest
    assert a.maneuvers_sha256 == b.maneuvers_sha256


def test_canonical_plan_is_the_safety_constrained_optimum(reference_sealed):
    from orbitopt.rendezvous import load_reference_config, plan_rendezvous
    cfg = load_reference_config()
    unconstrained = plan_rendezvous(cfg)
    sealed = reference_sealed
    # the canonical plan is the E12-04 result: same tof as the constrained
    # search (the window edge here), never the bare unconstrained optimum.
    assert sealed.safety is not None and sealed.safety["passed"]
    assert sealed.plan.tof_s != unconstrained.tof_s
    assert sealed.passed


def test_sealed_document_roundtrips(tmp_path, reference_sealed):
    from orbitopt.rendezvous import load_reference_config, read_sealed_plan, write_sealed_plan
    path = write_sealed_plan(reference_sealed, tmp_path / "leo-rdv-001.sealed.json")
    back = read_sealed_plan(path, load_reference_config())
    assert back.plan_digest == reference_sealed.plan_digest
    assert [m.to_dict() for m in back.plan.maneuvers] == \
           [m.to_dict() for m in reference_sealed.plan.maneuvers]
    assert back.safety["passed"] is True


def test_two_writes_are_identical(tmp_path, reference_sealed):
    from orbitopt.rendezvous import write_sealed_plan
    a = write_sealed_plan(reference_sealed, tmp_path / "a.json")
    b = write_sealed_plan(reference_sealed, tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# the consumption guards: every mismatch is a hard PlanSealError
# ---------------------------------------------------------------------------

def test_config_change_breaks_the_seal(reference_sealed):
    from orbitopt.rendezvous import PlanSealError, load_reference_config, verify_seal
    import dataclasses
    cfg = dataclasses.replace(load_reference_config(),
                             delta_v_budget_m_s=load_reference_config().delta_v_budget_m_s + 1.0)
    with pytest.raises(PlanSealError, match="config digest mismatch"):
        verify_seal(cfg, reference_sealed)


@pytest.mark.parametrize("path,expected", [
    (("plan", "maneuvers", 0, "epoch_s"), "maneuver digest mismatch"),
    (("maneuvers_sha256",), "maneuver digest mismatch"),
    (("plan_digest",), "plan digest mismatch"),
    (("config_sha256",), "config digest mismatch"),
])
def test_tampered_document_is_rejected(tmp_path, reference_sealed, path, expected):
    from orbitopt.rendezvous import (PlanSealError, load_reference_config, read_sealed_plan,
                                    write_sealed_plan)
    sealed_path = write_sealed_plan(reference_sealed, tmp_path / "sealed.json")
    doc = json.loads(sealed_path.read_text(encoding="utf-8"))
    if len(path) == 1:
        doc[path[0]] = "0" * 64
    else:
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = float(node[path[-1]]) + 0.5
    (tmp_path / "sealed.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PlanSealError, match=expected):
        read_sealed_plan(sealed_path, load_reference_config())


def test_sealed_plan_for_other_config_cannot_be_consumed(tmp_path, reference_sealed):
    """A sealed document sealed under config A must be rejected by config B
    that parses fine -- the plan_digest guard, not a shrug."""
    from orbitopt.rendezvous import PlanSealError, read_sealed_plan, write_sealed_plan
    doc = _reference_doc()
    doc["chaser"]["initial_separation_m"] = doc["chaser"]["initial_separation_m"] + 50.0
    other = _reconfig(doc)
    path = write_sealed_plan(reference_sealed, tmp_path / "sealed.json")
    with pytest.raises(PlanSealError, match="config digest mismatch"):
        read_sealed_plan(path, other)


def test_plan_rebuilt_for_other_states_is_rejected(reference_sealed):
    """A hand-rebuilt document whose maneuver digest lies about its own
    maneuvers is caught by the maneuver digest check itself."""
    from orbitopt.rendezvous import PlanSealError, load_reference_config, sealed_from_document
    from orbitopt.rendezvous import verify_seal
    cfg = load_reference_config()
    doc = reference_sealed.to_document()
    doc_ok = copy.deepcopy(doc)
    doc["maneuvers_sha256"] = "f" * 64
    with pytest.raises(PlanSealError, match="maneuver digest mismatch"):
        sealed_from_document(doc, cfg)
    # and a plain sealed object under the wrong config fails the same way
    import dataclasses
    good = sealed_from_document(doc_ok, cfg)  # untouched -- verify_seal checks config
    other = dataclasses.replace(_reconfig(_reference_doc()), delta_v_budget_m_s=
                               _reconfig(_reference_doc()).delta_v_budget_m_s + 0.5)
    with pytest.raises(PlanSealError, match="config digest mismatch"):
        verify_seal(other, good)


# ---------------------------------------------------------------------------
# every consumer records the identity it consumed
# ---------------------------------------------------------------------------

def test_safety_report_records_the_plan_digest(reference_sealed):
    from orbitopt.rendezvous import load_reference_config, verify_plan_safety
    cfg = load_reference_config()
    report = verify_plan_safety(cfg, reference_sealed.plan)
    assert report.plan_digest == reference_sealed.plan_digest
    assert report.to_dict()["plan_digest"] == reference_sealed.plan_digest


def test_studio_run_records_the_consumed_digest(tmp_path, reference_sealed):
    from orbitopt.rendezvous import build_studio_run, load_reference_config, write_studio_run
    studio = build_studio_run(load_reference_config(), sealed=reference_sealed)
    assert studio.sidecar["plan_digest"] == reference_sealed.plan_digest
    root = write_studio_run(tmp_path / "studio", studio)
    sidecar = json.loads((root / "rendezvous.json").read_text(encoding="utf-8"))
    assert sidecar["plan_digest"] == reference_sealed.plan_digest
    assert sidecar["safety"]["plan_digest"] == reference_sealed.plan_digest
    assert all(check.passed for check in studio.run.verification.checks)


def test_campaign_records_the_ground_plan_digest(reference_sealed):
    import dataclasses

    from orbitopt.rendezvous import load_reference_config
    from orbitopt.rendezvous import robustness as rb
    cfg = load_reference_config()
    campaign = rb.load_reference_campaign()
    campaign = dataclasses.replace(
        campaign,
        train=dataclasses.replace(campaign.train, n_clusters=1, samples_per_cluster=1),
        holdout=dataclasses.replace(campaign.holdout, n_clusters=1, samples_per_cluster=1),
        cluster_bootstrap_trials=40, spot_check_samples=1, grid_convergence_samples=1)
    result = rb.run_campaign(cfg, campaign, reference_sealed,
                             cross_check=False, grid_check=False)
    gp = result.ground_plan
    assert gp["plan_digest"] == reference_sealed.plan_digest
    assert gp["config_sha256"] == reference_sealed.config_sha256
    assert gp["maneuvers_sha256"] == reference_sealed.maneuvers_sha256
    assert gp["sealed_consumed"] is True
    # a campaign fed the sealed plan under a changed config fails, not flies
    from orbitopt.rendezvous import PlanSealError
    mutated = dataclasses.replace(load_reference_config(), delta_v_budget_m_s=
                                 load_reference_config().delta_v_budget_m_s + 1.0)
    with pytest.raises(PlanSealError):
        rb.run_campaign(mutated, campaign, reference_sealed,
                        cross_check=False, grid_check=False)


def test_refly_report_records_the_plan_digest(tmp_path):
    from orbitopt.verify import rendezvous_refly as rr
    result = rr.verify_rendezvous_refly(out_dir=tmp_path / "e12-05")
    doc = json.loads((tmp_path / "e12-05" / "rendezvous-refly-report.json")
                     .read_text(encoding="utf-8"))
    sealed_q = doc["inputs"]["main_model"]["sealed_plan"]
    assert sealed_q["plan_digest"] == result["sealed"].plan_digest
    assert sealed_q["maneuvers_sha256"] == result["sealed"].maneuvers_sha256
    assert (tmp_path / "e12-05" / "plan.sealed.json").is_file()


def test_safety_plan_result_exposes_the_digest(reference_sealed):
    from orbitopt.rendezvous import load_reference_config, plan_rendezvous_safe
    result = plan_rendezvous_safe(load_reference_config())
    assert result.plan_digest == reference_sealed.plan_digest
    assert result.to_dict()["plan_digest"] == reference_sealed.plan_digest
