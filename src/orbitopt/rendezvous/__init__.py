"""Rendezvous maneuver planning: two-impulse LEO rendezvous to a LVLH Hold
Point (E12-03, reference mission LEO-RDV-001).

* :mod:`orbitopt.rendezvous.config` -- the versioned mission-config contract
  (budget, time windows, gate, tolerances fixed before optimization).
* :mod:`orbitopt.rendezvous.planner` -- drift phase + one Lambert transfer,
  burn-1 Newton-targeted on the configured nonlinear pair propagation.
* :mod:`orbitopt.rendezvous.verify` -- replays a plan on that propagation
  and grades it; the acceptance verdict lives here.

The whole pipeline on the reference mission:

    from orbitopt.rendezvous import load_reference_config, plan_rendezvous, verify_plan
    cfg = load_reference_config()
    plan = plan_rendezvous(cfg)
    result = verify_plan(cfg, plan)
    assert result.passed
"""
from orbitopt.rendezvous.config import (
    RendezvousConfigError,
    RendezvousMissionConfig,
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
from orbitopt.rendezvous.verify import RendezvousVerification, verify_plan

__all__ = [
    "RendezvousConfigError",
    "RendezvousMissionConfig",
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
]
