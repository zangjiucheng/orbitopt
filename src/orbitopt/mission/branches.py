"""E2-02 (task #255): the mission's branched timeline.

From separation on, the mission is not one trajectory but two: the upper
stage climbs to the parking orbit while the first stage (the booster) flies
its own path down. Both branches *start* at the same handoff -- the stack's
state at separation -- and each branch's first sample must honour the M-01
handoff contract against it (epoch, frame, r, v), exactly as a segment-to-
segment handoff does. And the split must conserve mass: the two branches'
masses at separation sum to the stack's mass before the separation, within
the declared mass tolerance. A fork that fails either test is a bookkeeping
or wiring error, not physics.

:func:`separation_fork` finds the fork in a FlightRun (the ascent-lineage
root whose children are the mission's branches), and
:func:`branch_handoffs` assembles the fork document from the graded staging
handoff documents :mod:`orbitopt.mission.two_stage_leo` already produces:
one entry per branch (its kind, body and boundary handoff) plus the
cross-branch mass-conservation check ``mass.separation_fork``. A run that
does not branch at the stack (single-body, or a stack that shed exactly one
child) has no fork: the mission stayed on one trajectory.

Pure grading over a recorded run -- this module propagates nothing and
imports compute code lazily, like the rest of :mod:`orbitopt.mission`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The branch that flies to orbit (the ascent lineage's continuing child).
BRANCH_UPPER_STAGE = "upper-stage"
#: The branch the first stage took at separation: ballistic debris when only
#: the ascent was flown, the flown ``booster`` track after FL-304 replaced it.
BRANCH_BOOSTER = "booster"

#: Display labels for the scene timeline lanes (English-only UI).
BRANCH_LABELS = {BRANCH_UPPER_STAGE: "Upper stage", BRANCH_BOOSTER: "First stage"}


def separation_fork(run) -> dict | None:
    """The run's separation fork as a mission structure, or None when the
    mission never branched.

    The fork is the stack -- the root of the ascent lineage -- when it has at
    least two children: the lineage child (the :data:`BRANCH_UPPER_STAGE`
    branch the mission flies to orbit) and at least one more child of the
    stack (the :data:`BRANCH_BOOSTER` branch, which after E2-01 is a flown
    booster track, not a debris arc). Children of a *branch* (a fairing
    jettisoned off the upper stage) are not forks: they are the branch's own
    staging, not the mission splitting into two missions.

    Returns ``{t_s, parent_body, parent_mass_kg, branches: [{kind, body_id,
    label, ...}]}`` -- ``parent_mass_kg`` is the stack's final tracked mass
    (its mass at separation), ``t_s`` the separation event time (or the
    first child sample's time when the run carries no separation event)."""
    from orbitopt.mission.two_stage_leo import _ascent_lineage

    lineage = _ascent_lineage(run)
    if len(lineage) < 2:
        return None
    root, lineage_child = lineage[-1], lineage[-2]
    parent = next((t for t in run.tracks if t.body_id == root), None)
    if parent is None:
        return None
    children = [t for t in run.tracks if getattr(t, "parent_body_id", None) == root]
    child_ids = {t.body_id for t in children}
    if len(children) < 2 or lineage_child not in child_ids:
        return None
    t_s = next((float(e.t_s) for e in run.events if e.kind == "separation"
                and (e.body_id in child_ids or e.body_id == root
                     or (e.data or {}).get("separated_from") in child_ids)),
               None)
    if t_s is None:
        t_s = min(float(t.t_start_s) for t in children)
    branches = [{"kind": (BRANCH_UPPER_STAGE if t.body_id == lineage_child
                          else BRANCH_BOOSTER),
                 "body_id": t.body_id,
                 "label": BRANCH_LABELS[BRANCH_UPPER_STAGE if t.body_id == lineage_child
                                        else BRANCH_BOOSTER],
                 "t_start_s": float(t.t_start_s),
                 "t_end_s": float(t.t_end_s)} for t in children]
    from orbitopt.mission.powered_bridge import powered_parent_state

    powered = powered_parent_state(run, root, t_s)
    return {"t_s": t_s, "parent_body": root,
            "parent_mass_kg": float(powered["mass_kg"] if powered else parent.mass_kg[-1]),
            "powered_gap": (None if powered is None else
                            {k: powered[k] for k in ("gap_s", "method", "tolerances")}),
            "branches": branches}


def fork_lanes(fork: Mapping[str, Any] | None) -> list[dict]:
    """The fork's timeline lanes, in draw order (upper stage above the first
    stage), for the scene's ``timeline.branches`` block."""
    if not fork:
        return []
    order = (BRANCH_UPPER_STAGE, BRANCH_BOOSTER)
    return [{"id": kind, "label": BRANCH_LABELS[kind],
             "bodyId": next((b["body_id"] for b in fork["branches"] if b["kind"] == kind), "")}
            for kind in order
            if any(b["kind"] == kind for b in fork["branches"])]


def branch_handoffs(run, tolerances, staging_velocity_m_s: float,
                    staging_docs: Sequence[Mapping[str, Any]]) -> dict | None:
    """The fork document over a run and its graded staging handoff documents
    (:func:`orbitopt.mission.two_stage_leo.staging_handoffs`): one branch
    entry per separation child -- each carrying the M-01 boundary handoff the
    branch was graded under (epoch, frame, r, v continuity against the
    bridged stack state) -- plus ``mass.separation_fork``: the cross-branch
    conservation gate the per-boundary checks cannot see, that the branch
    masses at separation sum to the stack's mass before the split.

    Returns None when :func:`separation_fork` finds no fork. ``passed`` is
    None (not "passed") when a branch boundary was never graded; a fork with
    a missing branch handoff document is reported (``reason``) and fails,
    never skipped."""
    from orbitopt.mission.contract import HandoffTolerances
    from orbitopt.mission.two_stage_leo import _check, _checks_doc

    fork = separation_fork(run)
    if fork is None:
        return None
    tol = (tolerances.mass_kg if isinstance(tolerances, HandoffTolerances)
           else float(getattr(tolerances, "mass_kg", 1e-3)))
    if fork.get("powered_gap"):
        from orbitopt.mission.powered_bridge import powered_mass_tolerance_kg

        tol = powered_mass_tolerance_kg(fork["parent_mass_kg"], tol)
    docs = {d["child_body"]: d for d in staging_docs}
    branches: list[dict] = []
    missing: list[str] = []
    for branch in fork["branches"]:
        doc = docs.get(branch["body_id"])
        if doc is None:
            missing.append(branch["body_id"])
            branches.append({**branch, "handoff": None, "graded": False, "passed": None})
            continue
        branches.append({**branch, "handoff": doc, "graded": bool(doc["graded"]),
                         "passed": doc["passed"]})
    mass_sum = sum(float(d["after"]["mass_kg"]) for d in docs.values()
                   if d["child_body"] in {b["body_id"] for b in fork["branches"]})
    residual = mass_sum - float(fork["parent_mass_kg"])
    check = _check("mass.separation_fork", abs(residual) <= tol, value=residual,
                   limit=tol, unit="kg",
                   reason_code="" if abs(residual) <= tol else "mass_not_conserved_across_fork")
    graded = [b for b in branches if b["graded"]]
    passed = (bool(graded) and all(b["passed"] for b in graded)
              and check.passed and not missing)
    return {
        "t_s": float(fork["t_s"]), "parent_body": fork["parent_body"],
        "parent_mass_kg": float(fork["parent_mass_kg"]), "mass_sum_kg": float(mass_sum),
        "mass_residual_kg": float(residual),
        "branches": branches, "checks": _checks_doc([check]),
        "passed": bool(passed) if graded else None,
        **({"powered_gap": fork["powered_gap"]} if fork.get("powered_gap") else {}),
        **({"reason": f"no staging handoff document for {', '.join(missing)}"} if missing else {}),
    }
