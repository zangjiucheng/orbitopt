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
* :mod:`orbitopt.rendezvous.studio` -- replays the verified mission as a
  Flight Lab run directory for the Mission Studio (E12-06).

The whole pipeline on the reference mission:

    from orbitopt.rendezvous import load_reference_config, plan_rendezvous_safe
    cfg = load_reference_config()
    result = plan_rendezvous_safe(cfg)   # plans, verifies, grades safety
    assert result.passed
"""
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
    "RendezvousConfigError",
    "RendezvousMissionConfig",
    "SafetyCfg",
    "load_reference_config",
    "load_rendezvous_config",
    "parse_rendezvous_config",
    "validate_rendezvous_config",
    "Maneuver",
    "RendezvousPlan",
    "RendezvousPlanError",
    "initial_states",
    "plan_rendezvous",
    "RendezvousVerification",
    "verify_plan",
    "PassiveRelease",
    "SafetyCheck",
    "SafetyPlanResult",
    "SafetyReport",
    "evaluate_trajectory_safety",
    "plan_rendezvous_safe",
    "verify_plan_safety",
    "StudioRun",
    "build_studio_run",
    "write_studio_run",
]
