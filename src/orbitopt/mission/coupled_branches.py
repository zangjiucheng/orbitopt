"""E5-02 (task #269): coupled branch execution.

E2-02 (task #255) graded the separation fork *after the fact*: two branch
tracks exist, each boundary is continuous, the masses sum. This task makes
the coupling operational and traceable. From separation on, the mission is
driven by exactly one actually-computed separation state -- the stack's
final tracked sample, lifted through the frame bridge into GCRS and frozen
as a contract :class:`~orbitopt.mission.contract.HandoffState`. Both branch
initial states descend from *that* state:

* the **booster branch** can be *flown from it*: :func:`booster_state`
  converts the canonical state (at the branch's own separation mass) into
  the ``BoosterState`` FL-304 propagates -- the mission no longer re-reads a
  separation state from the debris track; the recovery *starts* at the
  handoff;
* the **upper-stage branch** continues the same trajectory inside the ascent
  run; its first sample is gated against the same canonical state
  (:func:`coupled_branches`), exactly the E2-02 boundary gate, only now the
  shared source is named and hashed.

The canonical state carries a ``handoff_id`` -- :func:`~orbitopt.mission.
contract.state_digest`, the SHA-256 of its canonical handoff-state-1.0
document -- and every branch record carries the same id, so "both branches'
initial states came from the same separation ``HandoffState``" is a fact a
report reader can verify to the byte, and any fork that claims a different
source goes red (``handoff.coupled.<branch>.shared_source``).

The document :func:`coupled_branches` assembles re-gates each fork branch
against the shared state (four contract checks under
``handoff.coupled.<branch>.*``, velocity with the declared separation-
impulse headroom), proves the staging gates consumed exactly this state
(``shared_source``), and carries the cross-branch split
``mass.coupled.branch_mass_split`` -- the same conservation law E2-02's
``mass.separation_fork`` grades, now stated against the canonical state.
``passed`` is reported either way; whether the checks are *graded* into the
chain verdict is the caller's call (E5-02 grades them when the recovery
stage drives from the handoff, keeping the frozen E2-04 reference checks
byte-identical).

Pure grading over a recorded run, like :mod:`orbitopt.mission.branches`;
only :func:`booster_state` touches the recovery compute stack, lazily.

    from orbitopt.mission.coupled_branches import separation_handoff, coupled_branches

    handoff = separation_handoff(run)          # one state, one id
    doc = coupled_branches(run, tol, 0.01, staging_docs, handoff=handoff)
    assert doc["branches"][0]["handoff_id"] == handoff.handoff_id
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from orbitopt.mission.contract import (
    HandoffState,
    HandoffTolerances,
    check_handoff,
    state_digest,
    state_from_dict,
    state_to_dict,
)

#: Numeric-identity band of the ``shared_source`` check: the staging gate and
#: the coupled gate consume the same parent sample through the same bridge,
#: so their bridged states must agree to frame-bridge round-off, not to a
#: physics tolerance.
SHARED_SOURCE_TOLERANCES = {"position_m": 1e-4, "velocity_m_s": 1e-4,
                            "epoch_s": 1e-3, "mass_kg": 1e-3}


@dataclass(frozen=True)
class SeparationHandoff:
    """The one separation state both mission branches descend from, plus its
    traceability id. ``state`` is the stack's final tracked sample at
    separation in GCRS; ``handoff_id`` is its canonical-document digest;
    ``branch_bodies`` maps the E2-02 branch kinds to the track each branch
    flew as."""

    handoff_id: str
    state: HandoffState
    t_s: float
    epoch_utc: datetime
    parent_body: str
    parent_mass_kg: float
    separation_event_t_s: float
    branch_bodies: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"handoff_id": self.handoff_id,
                "separation_state": state_to_dict(self.state),
                "t_s": float(self.t_s), "epoch_utc": self.epoch_utc.isoformat(),
                "parent_body": self.parent_body,
                "parent_mass_kg": float(self.parent_mass_kg),
                "separation_event_t_s": float(self.separation_event_t_s),
                "branch_bodies": dict(self.branch_bodies)}

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "SeparationHandoff":
        """Rebuild the handoff from its ``to_dict`` document (E5-05, task #272:
        the recovery leg of a cross-environment run drives from the ascent
        leg's canonical handoff, so the document must round-trip across the
        environment boundary). ``handoff_id`` is taken from the document, not
        recomputed: it identifies the sealed state, and a reader that
        re-hashed the state could not tell tampering from a rebuild -- the
        caller's envelope digest validation does that, before this runs."""
        from datetime import datetime

        return cls(handoff_id=str(doc["handoff_id"]),
                   state=state_from_dict(doc["separation_state"]),
                   t_s=float(doc["t_s"]),
                   epoch_utc=datetime.fromisoformat(str(doc["epoch_utc"])),
                   parent_body=str(doc["parent_body"]),
                   parent_mass_kg=float(doc["parent_mass_kg"]),
                   separation_event_t_s=float(doc["separation_event_t_s"]),
                   branch_bodies={str(k): str(v)
                                  for k, v in (doc.get("branch_bodies") or {}).items()})


def separation_handoff(run) -> SeparationHandoff | None:
    """The run's separation fork as its single canonical handoff: the stack's
    final sample at separation, bridged to GCRS, frozen as a
    :class:`~orbitopt.mission.contract.HandoffState` and identified by the
    digest of its handoff-state-1.0 document. None when the mission never
    branched (:func:`orbitopt.mission.branches.separation_fork` says so)."""
    from orbitopt.mission.branches import separation_fork
    from orbitopt.mission.two_stage_leo import _sample_state
    from orbitopt.timescales import mjd2000_to_utc_datetime

    fork = separation_fork(run)
    if fork is None:
        return None
    parent = next((t for t in run.tracks if t.body_id == fork["parent_body"]), None)
    if parent is None:
        return None
    last = len(parent.t_s) - 1
    state = _sample_state(parent, last, {"body_id": parent.body_id,
                                         "parent_body_id": parent.parent_body_id or ""})
    # Only the separation facts go into the hashed state: the digest must pin
    # the physics of the split, so it stays stable under child relabeling
    # (debris -> booster) and under the merge's fresh run id.
    state = replace(state, metadata={**state.metadata, "event": "separation",
                                     "separation_event_t_s": float(fork["t_s"])})
    epoch_utc = mjd2000_to_utc_datetime(state.epoch.to_utc_mjd2000())
    return SeparationHandoff(
        handoff_id=state_digest(state), state=state,
        t_s=float(parent.t_s[last]), epoch_utc=epoch_utc,
        parent_body=str(fork["parent_body"]),
        parent_mass_kg=float(fork["parent_mass_kg"]),
        separation_event_t_s=float(fork["t_s"]),
        branch_bodies={str(b["kind"]): str(b["body_id"]) for b in fork["branches"]})


def branch_initial_mass(run, body_id: str) -> float:
    """The mass ``body_id``'s branch carried at separation: the first sample
    of its track (the mass the canonical stack state shed to it)."""
    track = next((t for t in run.tracks if t.body_id == body_id), None)
    if track is None or len(track.mass_kg) < 1:
        raise ValueError(f"run carries no track {body_id!r} to read a branch mass from")
    return float(track.mass_kg[0])


def booster_state(handoff: SeparationHandoff, *, mass_kg: float,
                  epoch_utc: datetime | None = None):
    """The canonical separation state as the ``BoosterState`` FL-304 flies
    (E5-02: the booster branch starts *at* the handoff, not at a separately
    read-out debris sample). The state stays in the handoff's frame (GCRS);
    ``epoch_utc`` is the mission-clock origin the propagator needs -- the
    run's reference epoch, not the handoff's absolute instant: FL-304's
    mission clock reads ``t_s`` seconds after ``epoch_utc`` (E5-05, task
    #272: driving with the separation instant itself double-counted the
    offset, sealing the booster track ~sep-time into the future and
    bridging continuity checks across the ghost gap)."""
    from datetime import timedelta

    from orbitopt.flightlab.recovery.propagator import BoosterState

    if epoch_utc is None:
        epoch_utc = handoff.epoch_utc.replace(tzinfo=None) - timedelta(
            seconds=float(handoff.t_s))
    return BoosterState(t_s=float(handoff.t_s),
                        epoch_utc=epoch_utc,
                        frame=handoff.state.frame,
                        position_m=tuple(handoff.state.r_m),
                        velocity_m_s=tuple(handoff.state.v_m_s),
                        mass_kg=float(mass_kg))


def booster_state_from_run(run, handoff: SeparationHandoff):
    """:func:`booster_state` with the booster branch's own separation mass
    (the debris track's first sample -- the bookkeeping half of the split)
    and the ascent run's own reference epoch for the propagator's mission
    clock (the handoff's absolute instant lands at t_s = ``handoff.t_s``)."""
    from orbitopt.mission.branches import BRANCH_BOOSTER

    body_id = handoff.branch_bodies.get(BRANCH_BOOSTER)
    if body_id is None:
        raise ValueError("the separation fork carries no booster branch to drive")
    parent = next(t for t in run.tracks if t.body_id == handoff.parent_body)
    epoch = datetime.fromisoformat(parent.epoch_utc.replace("Z", "+00:00"))
    return booster_state(handoff, mass_kg=branch_initial_mass(run, body_id),
                         epoch_utc=epoch.replace(tzinfo=None))


def _residual(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    pos = math.sqrt(sum((float(x) - float(y)) ** 2
                        for x, y in zip(a["position_m"], b["position_m"])))
    vel = math.sqrt(sum((float(x) - float(y)) ** 2
                        for x, y in zip(a["velocity_m_s"], b["velocity_m_s"])))
    return {"position_m": pos, "velocity_m_s": vel,
            "epoch_s": abs(float(a["epoch_mjd2000"]) - float(b["epoch_mjd2000"]))
            * 86400.0,
            "mass_kg": abs(float(a["mass_kg"]) - float(b["mass_kg"]))}


def coupled_branches(run, tolerances: HandoffTolerances,
                     staging_velocity_m_s: float,
                     staging_docs: Sequence[Mapping[str, Any]] = (),
                     handoff: SeparationHandoff | None = None) -> dict | None:
    """The coupled-branch document: one canonical separation
    :class:`~orbitopt.mission.contract.HandoffState` (with its digest) and
    one record per fork branch, each re-gated against *that* state --
    position/velocity/epoch/mass under the M-01 contract (velocity with the
    declared separation-impulse headroom, parent state bridged across the
    sample gap exactly as the staging gate does), a ``shared_source`` proof
    that the staging gate consumed the very same bridged state, and the
    cross-branch mass split. Returns None when the mission never forked.
    ``handoff`` lets a caller that already computed the canonical state (and
    *drove* a branch from it) hand it over; otherwise it is read off the
    run, so the document stands alone as pure grading."""
    from orbitopt.mission.branches import separation_fork
    from orbitopt.mission.two_stage_leo import (
        _bridge_across_gap,
        _check,
        _checks_doc,
        _sample_state,
        _state_doc,
    )

    handoff = handoff if handoff is not None else separation_handoff(run)
    if handoff is None:
        return None
    fork = separation_fork(run)
    boundary_tol = HandoffTolerances(
        position_m=tolerances.position_m, velocity_m_s=float(staging_velocity_m_s),
        epoch_s=tolerances.epoch_s, max_epoch_gap_s=tolerances.max_epoch_gap_s,
        mass_kg=tolerances.mass_kg)
    docs = {str(d["child_body"]): d for d in staging_docs}

    branches: list[dict] = []
    all_checks: list = []
    mass_sum = 0.0
    for branch in fork["branches"]:
        kind, body = str(branch["kind"]), str(branch["body_id"])
        child = next(t for t in run.tracks if t.body_id == body)
        after = _sample_state(child, 0, {"body_id": child.body_id,
                                         "parent_body_id": child.parent_body_id or ""})
        bridge = _bridge_across_gap(handoff.state, after)
        before = bridge["state"] if bridge is not None else handoff.state
        report = check_handoff(before, after, boundary_tol)
        checks = [replace(c, id=f"handoff.coupled.{kind}.{c.id.split('.', 1)[1]}")
                  for c in report.checks]
        doc = docs.get(body)
        if doc is None:
            checks.append(_check(f"handoff.coupled.{kind}.shared_source", False,
                                 reason_code="no_staging_handoff_document"))
        else:
            res = _residual(_state_doc(before), doc["before"])
            shared = (res["position_m"] <= SHARED_SOURCE_TOLERANCES["position_m"]
                      and res["velocity_m_s"] <= SHARED_SOURCE_TOLERANCES["velocity_m_s"]
                      and res["epoch_s"] <= SHARED_SOURCE_TOLERANCES["epoch_s"]
                      and res["mass_kg"] <= SHARED_SOURCE_TOLERANCES["mass_kg"])
            checks.append(_check(
                f"handoff.coupled.{kind}.shared_source", shared,
                value=res["position_m"], limit=SHARED_SOURCE_TOLERANCES["position_m"],
                unit="m", reason_code="" if shared else "staging_state_diverged_from_handoff"))
        all_checks.extend(checks)
        mass_sum += float(after.mass_kg)
        branches.append({"kind": kind, "body_id": body,
                         "handoff_id": handoff.handoff_id,
                         "state": _state_doc(after),
                         "bridge": (None if bridge is None else
                                    {k: bridge[k] for k in ("gap_s", "method", "pole")}),
                         "checks": _checks_doc(checks),
                         "passed": all(c.passed for c in checks),
                         "result": None})

    residual = mass_sum - handoff.parent_mass_kg
    tol_mass = float(tolerances.mass_kg)
    split = _check("mass.coupled.branch_mass_split", abs(residual) <= tol_mass,
                   value=residual, limit=tol_mass, unit="kg",
                   reason_code="mass_not_conserved_across_fork")
    all_checks.append(split)
    return {"handoff_id": handoff.handoff_id,
            "separation_state": state_to_dict(handoff.state),
            "t_s": handoff.t_s, "epoch_utc": handoff.epoch_utc.isoformat(),
            "parent_body": handoff.parent_body,
            "parent_mass_kg": handoff.parent_mass_kg,
            "separation_event_t_s": handoff.separation_event_t_s,
            "mass_sum_kg": float(mass_sum), "mass_residual_kg": float(residual),
            "driven_from_separation": False,
            "branches": branches,
            "checks": _checks_doc(all_checks),
            "passed": all(c.passed for c in all_checks)}


def record_branch_results(coupled: dict, *, upper_stage: Mapping[str, Any] | None = None,
                          booster: Mapping[str, Any] | None = None) -> dict:
    """Attach each branch's *outcome* to the coupled document, so one run's
    report carries both the upper-stage insertion result and the booster
    recovery result beside the shared handoff they descend from (E5-02
    acceptance: both branch results in one run). ``upper_stage`` is the
    insertion/parking evidence, ``booster`` the recovery block's; a branch
    that flew no executable outcome (debris only, no recovery stage) records
    why, never a silent pass."""
    for branch in coupled.get("branches", ()):
        if branch["kind"] == "upper-stage":
            branch["result"] = (dict(upper_stage) if upper_stage is not None else
                                {"reason": "the chain produced no parking-orbit result"})
        elif branch["kind"] == "booster":
            branch["result"] = (dict(booster) if booster is not None else
                                {"status": "ballistic-debris",
                                 "reason": "no recovery stage declared: the branch is the "
                                           "ascent run's debris arc, recorded not flown home"})
    return coupled
