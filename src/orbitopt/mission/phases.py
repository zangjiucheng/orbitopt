"""E2-03 (task #256): the full-mission verification gate.

The two-stage LEO chain (:mod:`orbitopt.mission.two_stage_leo`) grades one
frozen mission and diagnoses the *earliest failing link* of its station chain
(``STATUS_*``). That answers "which link broke?" but not the two questions
E2-03 has to answer for the epic's Full Mission Closure:

* **per phase** -- did orbit injection succeed, did the parking-orbit
  propagation succeed, did the first stage touch down? Each phase gets its own
  tri-state verdict (:data:`VERDICT_PASSED` / :data:`VERDICT_FAILED` /
  :data:`VERDICT_NOT_CHECKED`, the same tri-state as
  ``orbitopt.verify.report.RESULT_*``) and its own failure reason codes.
* **overall** -- is the *full mission* a success? A partial success is never
  an overall pass: an injected upper stage with a lost booster is an overall
  failure with a green ``orbit_injection`` phase -- the local success is
  visible in the phase verdict, the mission still did not close (task
  semantics: injection succeeded + recovery failed => overall failed, local
  injection passed).

Phase boundaries follow the separation fork (E2-02): at separation the mission
becomes two trajectories, and from there on each phase is graded by its own
checks. A check is attributed to a phase by id namespace:

======================  ======================================================
phase                   checks
======================  ======================================================
``orbit_injection``     ``station.*``, ``ascent.*``, ``handoff.*``,
                        ``parking.*``, ``mass.*``, ``expected.*``
``parking_propagation`` ``propagation.*``
``recovery_touchdown``  ``recovery.*``
======================  ======================================================

The run-level ``ascent.flight_verified`` verdict is the Gate-B conjunction over
the *whole* flight. Once a recovery stage has run over the combined run it
mixes touchdown into that conjunction, so the phase gate attributes it as
follows: with no recovery checks present it measures insertion and joins the
injection phase; with recovery checks present it is recorded but unattributed
(its halves are graded by ``ascent.insertion_verified`` and ``recovery.*``
instead, never re-used across a phase boundary).

The overall rule (:data:`OVERALL_RULE`) is the tri-state conjunction of the
*applicable* phases, with the evidence semantics task #243 unified (E4-01):
absent evidence is never a pass.

* any applicable phase ``failed``      -> full mission ``failed``
* else any applicable phase
  ``not_checked``                      -> full mission ``not_checked``
* else (every applicable phase
  ``passed``)                          -> full mission ``passed``

A phase is *applicable* when the mission flies it: orbit injection always;
parking-orbit propagation unless the scenario declares
``expected.propagation_result: not_checked`` (declared out of scope, not a
missing verdict); recovery when a ``recovery:`` stage is declared. A phase the
mission declares but that produced no verdict evidence (the ascent never flew,
propagation skipped by the caller) is ``not_checked``, which caps the overall
verdict at ``not_checked`` -- it never silently passes.

    from orbitopt.mission.phases import full_mission_gate

    gate = full_mission_gate(mission)
    print(gate["verdict"], [(p["phase"], p["verdict"]) for p in gate["phases"]])

Pure grading over the checks the chain already produced -- this module
simulates nothing and imports nothing beyond the standard library, like the
rest of :mod:`orbitopt.mission`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "INSERTION_ONLY_VERDICT_ID",
    "MIXED_RUN_VERDICT_ID",
    "OVERALL_RULE",
    "PHASES",
    "PHASE_LABELS",
    "PHASE_ORBIT_INJECTION",
    "PHASE_PARKING_PROPAGATION",
    "PHASE_RECOVERY_TOUCHDOWN",
    "PHASE_VERDICTS",
    "VERDICT_FAILED",
    "VERDICT_NOT_CHECKED",
    "VERDICT_PASSED",
    "combine_phase_verdicts",
    "full_mission_gate",
    "phase_of",
    "phase_verdicts",
]

#: The three phases of the full mission, in chain order (the order the mission
#: flies them, the order the gate reports them).
PHASE_ORBIT_INJECTION = "orbit_injection"
PHASE_PARKING_PROPAGATION = "parking_propagation"
PHASE_RECOVERY_TOUCHDOWN = "recovery_touchdown"
PHASES = (PHASE_ORBIT_INJECTION, PHASE_PARKING_PROPAGATION, PHASE_RECOVERY_TOUCHDOWN)

#: Display labels (English-only UI, like the scene vocabulary).
PHASE_LABELS = {
    PHASE_ORBIT_INJECTION: "Orbit injection",
    PHASE_PARKING_PROPAGATION: "Parking-orbit propagation",
    PHASE_RECOVERY_TOUCHDOWN: "Recovery touchdown",
}

#: Tri-state phase verdicts -- the same strings (and the same semantics) as
#: ``orbitopt.verify.report.RESULT_PASSED/RESULT_FAILED/RESULT_NOT_CHECKED``;
#: restated here so this module stays stdlib-only, like mission.contract
#: restates the frame vocabulary. A test asserts the two vocabularies agree.
VERDICT_PASSED = "passed"
VERDICT_FAILED = "failed"
VERDICT_NOT_CHECKED = "not_checked"
PHASE_VERDICTS = (VERDICT_PASSED, VERDICT_FAILED, VERDICT_NOT_CHECKED)

#: The overall rule, quoted verbatim into the report document so a consumer
#: reads the rule with the verdict it explains (the matrix of every
#: combination is exercised by tests/test_mission_phases.py).
OVERALL_RULE = (
    "full mission failed if any applicable phase failed; else not_checked if "
    "any applicable phase is not_checked (absent evidence is never a pass); "
    "passed only when every applicable phase passed; a phase the mission does "
    "not fly is not applicable and never blocks the verdict"
)

#: Check-id namespaces, longest-prefix first, mapped to their phase. Every
#: mission-chain check lives in exactly one namespace (:func:`phase_of`).
_PHASE_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("recovery.", PHASE_RECOVERY_TOUCHDOWN),
    ("propagation.", PHASE_PARKING_PROPAGATION),
    ("station.", PHASE_ORBIT_INJECTION),
    ("ascent.", PHASE_ORBIT_INJECTION),
    ("handoff.", PHASE_ORBIT_INJECTION),
    ("parking.", PHASE_ORBIT_INJECTION),
    ("mass.", PHASE_ORBIT_INJECTION),
    ("expected.", PHASE_ORBIT_INJECTION),
)

#: The mixed run-level Gate-B conjunction over the whole flight, and the
#: insertion-only verdict the recovery stage derives to replace it once the
#: combined run grades touchdown inside it (module docstring).
MIXED_RUN_VERDICT_ID = "ascent.flight_verified"
INSERTION_ONLY_VERDICT_ID = "ascent.insertion_verified"


def phase_of(check_id: str) -> str | None:
    """The phase the check with this id grades, or ``None`` when the id is
    outside the phase vocabulary (it is then recorded, never attributed)."""
    for prefix, phase in _PHASE_BY_PREFIX:
        if str(check_id).startswith(prefix):
            return phase
    return None


def _flag(check: Any) -> bool:
    return bool(check.passed if hasattr(check, "passed") else check["passed"])


def _check_id(check: Any) -> str:
    return str(check.id if hasattr(check, "id") else check["id"])


def _reason_code(check: Any) -> str:
    own = getattr(check, "reason_code", None)
    if own is None and isinstance(check, Mapping):
        own = check.get("reason_code")
    own = str(own or "")
    return own or _check_id(check)


def _attributed(checks: Sequence[Any]) -> dict[str, list[Any]]:
    """Group checks into phases (module docstring): the mixed run-level
    conjunction is attributed to injection only while it measures insertion
    alone (no recovery checks present); once recovery ran, its own phase-local
    signals grade the two halves and the mixed check is recorded, not
    attributed."""
    has_recovery = any(_check_id(c).startswith("recovery.") for c in checks)
    grouped: dict[str, list[Any]] = {phase: [] for phase in PHASES}
    for check in checks:
        check_id = _check_id(check)
        if check_id == MIXED_RUN_VERDICT_ID and has_recovery:
            continue  # insertion AND touchdown over the combined run: recorded, not attributed
        phase = phase_of(check_id)
        if phase is not None:
            grouped[phase].append(check)
    return grouped


def phase_verdicts(checks: Sequence[Any], *,
                   applicable: Mapping[str, Any],
                   reasons: Mapping[str, str] | None = None) -> dict[str, dict]:
    """One verdict document per phase (:data:`PHASES`, all three always
    present) over the mission-chain checks given.

    ``applicable`` maps each phase to ``True``, or to a string reason when the
    mission does not fly that phase (a false-like value uses the default
    reason). ``reasons`` optionally explains an applicable phase that produced
    no verdict evidence. The verdict of a phase with checks: ``failed`` if any
    failed, else ``passed``; with no checks: ``not_checked``. Failed checks
    contribute their reason codes (a failed check without its own code falls
    back to its check id -- never silence).
    """
    grouped = _attributed(checks)
    out: dict[str, dict] = {}
    for phase in PHASES:
        mine = grouped[phase]
        fail = [c for c in mine if not _flag(c)]
        verdict = (VERDICT_FAILED if fail
                   else VERDICT_PASSED if mine
                   else VERDICT_NOT_CHECKED)
        applic = applicable.get(phase, True)
        if isinstance(applic, str):
            applic, why_not_applicable = False, applic
        else:
            applic = bool(applic)
            why_not_applicable = ""
        doc: dict[str, Any] = {
            "phase": phase,
            "label": PHASE_LABELS[phase],
            "applicable": applic,
            "verdict": verdict,
            "checks_passed": sum(1 for c in mine if _flag(c)),
            "checks_total": len(mine),
            "reason_codes": sorted({_reason_code(c) for c in fail}),
            "checks": {_check_id(c): _flag(c) for c in mine},
            "reason": "",
        }
        if not applic:
            doc["reason"] = why_not_applicable or "phase not flown by this mission"
        elif verdict == VERDICT_NOT_CHECKED:
            doc["reason"] = (reasons or {}).get(
                phase, "phase declared but produced no verdict evidence")
        out[phase] = doc
    return out


def combine_phase_verdicts(docs: Sequence[Mapping[str, Any]]) -> str:
    """The overall verdict of the per-phase verdict documents
    (:data:`OVERALL_RULE`): failed beats not_checked beats passed, over the
    applicable phases only."""
    graded = [d for d in docs if d.get("applicable", True)]
    if any(d.get("verdict") == VERDICT_FAILED for d in graded):
        return VERDICT_FAILED
    if any(d.get("verdict") != VERDICT_PASSED for d in graded):
        return VERDICT_NOT_CHECKED
    return VERDICT_PASSED if graded else VERDICT_NOT_CHECKED


def full_mission_gate(mission: Any) -> dict:
    """The full-mission gate document of a ``TwoStageLeoMission`` (duck-typed:
    ``scenario``, ``checks``, ``not_assessed``, ``propagation``). Built into
    the unified task report under ``inputs.full_mission`` by
    :func:`orbitopt.mission.two_stage_leo.mission_report`.

    Applicability is read from the *scenario* (what the mission claims to fly,
    never from what the run happened to produce): recovery applies when a
    ``recovery:`` stage is declared, propagation unless the scenario declares
    ``expected.propagation_result: not_checked``. A phase the scenario declares
    but that produced no checks -- the ascent never flew, the caller skipped
    propagation -- is reported ``not_checked`` with the mission's own recorded
    reason."""
    scenario = mission.scenario
    recovery_declared = getattr(scenario, "recovery", None) is not None
    propagation_required = str(getattr(scenario.expected, "propagation_result",
                                       "passed")) != VERDICT_NOT_CHECKED
    applicable: dict[str, Any] = {
        PHASE_ORBIT_INJECTION: True,
        PHASE_PARKING_PROPAGATION: propagation_required or (
            "propagation declared out of scope (expected.propagation_result: "
            "not_checked): the parking orbit is not required to be propagated"),
        PHASE_RECOVERY_TOUCHDOWN: recovery_declared or (
            "no recovery stage declared: this mission flies no first-stage return, "
            "touchdown is out of scope (an expendable mission is a smaller mission, "
            "not a failed one)"),
    }
    reasons: dict[str, str] = {}
    ids = [_check_id(c) for c in mission.checks]
    if recovery_declared and not any(i.startswith("recovery.") for i in ids):
        reasons[PHASE_RECOVERY_TOUCHDOWN] = next(
            (str(n.get("reason", "")) for n in mission.not_assessed
             if n.get("id") == "recovery"),
            "recovery stage declared but never ran (the ascent produced no flown run)")
    if propagation_required and not any(i.startswith("propagation.") for i in ids):
        propagation = dict(getattr(mission, "propagation", None) or {})
        reasons[PHASE_PARKING_PROPAGATION] = str(propagation.get("reason")
                                                 or "propagation did not run")
    docs = phase_verdicts(mission.checks, applicable=applicable, reasons=reasons)
    phase_docs = [docs[phase] for phase in PHASES]
    verdict = combine_phase_verdicts(phase_docs)
    reason_codes = sorted({code for d in phase_docs
                           if d.get("verdict") == VERDICT_FAILED
                           for code in d["reason_codes"]})
    return {"rule": OVERALL_RULE, "verdict": verdict, "reason_codes": reason_codes,
            "phases": phase_docs}
