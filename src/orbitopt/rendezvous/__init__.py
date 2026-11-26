"""Rendezvous maneuver planning: two-impulse LEO rendezvous to a LVLH Hold
Point (E12-03, reference mission LEO-RDV-001), with full-trajectory safety
constraints (E12-04).

* :mod:`orbitopt.rendezvous.config` -- the versioned mission-config contract
  (budget, time windows, gate, tolerances and, from E12-04, the safety
  constraints -- all fixed before optimization).
* :mod:`orbitopt.rendezvous.planner` -- drift phase + one Lambert transfer,
  burn-1 Newton-targeted on the configured nonlinear pair propagation.
* :mod:`orbitopt.rendezvous.verify` -- replays a plan on that propagation
  and grades it; the acceptance verdict lives here.
* :mod:`orbitopt.rendezvous.safety` -- keep-out / corridor / approach-speed
  / passive-safety checks on the WHOLE trajectory, and the safety-constrained
  planner that screens unsafe candidates out of the search.
* :mod:`orbitopt.rendezvous.canonical` -- the canonical SEALED plan (E12-08):
  one safety-constrained plan per config, pinned by four digests, that every
  downstream component consumes so their reports describe one trajectory.
* :mod:`orbitopt.rendezvous.studio` -- replays the verified mission as a
  Flight Lab run directory for the Mission Studio (E12-06).
* :mod:`orbitopt.rendezvous.robustness` -- the Monte Carlo dispersion campaign:
  train and hold-out arms, success-rate intervals, failure taxonomy,
  sensitivity, and the independent spot check (E12-07).

The whole pipeline on the reference mission:

    from orbitopt.rendezvous import canonical_plan, load_reference_config
    cfg = load_reference_config()
    sealed = canonical_plan(cfg)          # plans, verifies, grades, seals
    assert sealed.passed
"""
from orbitopt.rendezvous.canonical import (
    PlanSealError,
    SealedRendezvousPlan,
    canonical_config_sha256,
    canonical_plan,
    consume_sealed_plan,
    initial_states_sha256,
    maneuvers_sha256,
    read_sealed_plan,
    rendezvous_plan_digest,
    sealed_from_document,
    seal_plan,
    verify_seal,
    write_sealed_plan,
)
from orbitopt.rendezvous.config import (
    RendezvousConfigError,
    RendezvousMissionConfig,
    SafetyCfg,
    load_reference_config,
    load_rendezvous_config,
    parse_rendezvous_config,
    validate_rendezvous_config,
)
from orbitopt.rendezvous.planner import (
    Maneuver,
    RendezvousPlan,
    RendezvousPlanError,
    initial_states,
    plan_rendezvous,
)
from orbitopt.rendezvous.robustness import (
    ArmResult,
    CampaignResult,
    DisturbanceModel,
    RobustnessCampaign,
    load_campaign,
    load_reference_campaign,
    run_campaign,
)
from orbitopt.rendezvous.safety import (
    PassiveRelease,
    SafetyCheck,
    SafetyPlanResult,
    SafetyReport,
    evaluate_trajectory_safety,
    plan_rendezvous_safe,
    verify_plan_safety,
)
from orbitopt.rendezvous.studio import StudioRun, build_studio_run, write_studio_run
from orbitopt.rendezvous.verify import RendezvousVerification, verify_plan

__all__ = [
    "ArmResult",
    "CampaignResult",
    "DisturbanceModel",
    "Maneuver",
    "PassiveRelease",
    "PlanSealError",
    "RendezvousConfigError",
    "RendezvousMissionConfig",
    "RendezvousPlan",
    "RendezvousPlanError",
    "RendezvousVerification",
    "RobustnessCampaign",
    "SafetyCfg",
    "SafetyCheck",
    "SafetyPlanResult",
    "SafetyReport",
    "SealedRendezvousPlan",
    "StudioRun",
    "build_studio_run",
    "canonical_config_sha256",
    "canonical_plan",
    "consume_sealed_plan",
    "evaluate_trajectory_safety",
    "initial_states",
    "initial_states_sha256",
    "load_campaign",
    "load_reference_campaign",
    "load_reference_config",
    "load_rendezvous_config",
    "maneuvers_sha256",
    "parse_rendezvous_config",
    "plan_rendezvous",
    "plan_rendezvous_safe",
    "read_sealed_plan",
    "rendezvous_plan_digest",
    "run_campaign",
    "sealed_from_document",
    "seal_plan",
    "validate_rendezvous_config",
    "verify_plan",
    "verify_plan_safety",
    "verify_seal",
    "write_sealed_plan",
    "write_studio_run",
]
