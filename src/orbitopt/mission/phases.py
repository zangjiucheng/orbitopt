"""E2-03 (task #256): the full-mission verification gate; E5-04 (task #271)
adds the physical gate over the same checks.

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

Verification coverage gate (E8-03, task #283)
---------------------------------------------
The out-of-scope declarations above let a *partial* run grade honestly on the
phases it flies. They must not let a run that *claims to be the full mission*
qualify on a subset. Every phase is a *required physical phase*
(:data:`REQUIRED_PHASES`): a scenario that declares ``full_mission: true``
forces all of them applicable -- an out-of-scope declaration of a required
phase is refused, not honoured -- and the gate document carries

* ``coverage`` -- required-phase accounting shown in every gate document: the
  per-phase coverage status (``checked`` / ``not_checked`` /
  ``declared_out_of_scope`` / ``not_declared``) and the ``checked/required``
  summary (e.g. ``2/3``) the report summary displays;
* ``qualification.independently_verified`` -- whether the overall description
  may say *independently verified*. It is true only for a run that declares
  ``full_mission``, whose gate verdict is ``passed`` with complete required
  coverage, and whose report carries passing validation-grade independent
  evidence (the caller passes ``independent_validation`` from the status
  rollup). An undeclared or unchecked required phase, or missing independent
  evidence, makes the claim false with a recorded reason -- never a silent
  pass.

Typical use::

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
    "COVERAGE_CHECKED",
    "COVERAGE_DECLARED_OUT_OF_SCOPE",
    "COVERAGE_NOT_CHECKED",
    "COVERAGE_NOT_DECLARED",
    "COVERAGE_STATUSES",
    "INSERTION_ONLY_VERDICT_ID",
    "MIXED_RUN_VERDICT_ID",
    "OVERALL_RULE",
    "PHASES",
    "PHASE_LABELS",
    "PHASE_ORBIT_INJECTION",
    "PHASE_PARKING_PROPAGATION",
    "PHASE_RECOVERY_TOUCHDOWN",
    "PHASE_VERDICTS",
    "PHYSICAL_CHECKS",
    "PHYSICAL_CHECK_EPOCH_FRAME_CONTINUITY",
    "PHYSICAL_CHECK_LANDING",
    "PHYSICAL_CHECK_MASS_CONSERVATION",
    "PHYSICAL_CHECK_ORBIT_INSERTION",
    "PHYSICAL_CHECK_PARKING_PROPAGATION",
    "PHYSICAL_LABELS",
    "PHYSICAL_NOT_EVIDENCED_CODE",
    "PHYSICAL_RULE",
    "REQUIRED_PHASES",
    "VERDICT_FAILED",
    "VERDICT_NOT_CHECKED",
    "VERDICT_PASSED",
    "combine_phase_verdicts",
    "full_mission_gate",
    "phase_coverage",
    "phase_of",
    "phase_verdicts",
    "physical_check_of",
    "physical_mission_gate",
    "qualification",
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

#: The required physical phases of a full mission (E8-03, task #283): every
#: phase of the chain. A run that declares itself the full mission
#: (``full_mission: true``) cannot take any of them out of scope -- the
#: out-of-scope declarations of the E2-03 rule are how a *partial* run grades
#: honestly, not how a complete claim skips its evidence.
REQUIRED_PHASES = PHASES

#: Required-phase coverage states (E8-03): how much of the required physical
#: inventory a run actually checked. ``checked`` graded the phase (the verdict
#: may still be red -- red evidence was still checked); ``not_checked`` leaves
#: it with no verdict evidence; ``declared_out_of_scope`` and ``not_declared``
#: distinguish an explicit out-of-scope declaration from a phase the scenario
#: simply never mentions.
COVERAGE_CHECKED = "checked"
COVERAGE_NOT_CHECKED = "not_checked"
COVERAGE_DECLARED_OUT_OF_SCOPE = "declared_out_of_scope"
COVERAGE_NOT_DECLARED = "not_declared"
COVERAGE_STATUSES = (COVERAGE_CHECKED, COVERAGE_NOT_CHECKED,
                     COVERAGE_DECLARED_OUT_OF_SCOPE, COVERAGE_NOT_DECLARED)

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
    reason), or to a mapping ``{"reason": ..., "kind": ...}`` naming the kind
    of inapplicability (``declared_out_of_scope`` for an explicit out-of-scope
    declaration, ``not_declared`` for a phase the scenario never mentions --
    the kinds E8-03 reports as required-phase coverage). ``reasons`` optionally
    explains an applicable phase that produced no verdict evidence. The verdict
    of a phase with checks: ``failed`` if any failed, else ``passed``; with no
    checks: ``not_checked``. Failed checks contribute their reason codes (a
    failed check without its own code falls back to its check id -- never
    silence).
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
            applic, why_not_applicable, how = False, applic, COVERAGE_NOT_DECLARED
        elif isinstance(applic, Mapping):
            how = str(applic.get("kind") or COVERAGE_NOT_DECLARED)
            applic, why_not_applicable = False, str(applic.get("reason") or "")
        else:
            applic = bool(applic)
            why_not_applicable, how = "", "applicable"
        doc: dict[str, Any] = {
            "phase": phase,
            "label": PHASE_LABELS[phase],
            "applicable": applic,
            "applicability": "applicable" if applic else how,
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


def phase_coverage(docs: Mapping[str, Mapping[str, Any]],
                   required: Sequence[str] = REQUIRED_PHASES) -> dict:
    """Required-phase coverage (E8-03, task #283) over per-phase verdict
    documents: per required phase the coverage status
    (:data:`COVERAGE_STATUSES`) with its recorded reason, and the
    ``checked/required`` summary the report displays. A phase graded and found
    red is still **checked** -- coverage measures what was verified, the
    verdict says how it turned out; an unchecked required phase is never
    counted as covered."""
    per_phase: dict[str, dict] = {}
    checked = 0
    for phase in required:
        doc = docs[phase]
        if not doc.get("applicable", True):
            kind = str(doc.get("applicability") or COVERAGE_NOT_DECLARED)
            status = (COVERAGE_DECLARED_OUT_OF_SCOPE
                      if kind == COVERAGE_DECLARED_OUT_OF_SCOPE else COVERAGE_NOT_DECLARED)
            reason = str(doc.get("reason") or "")
        elif doc.get("verdict") == VERDICT_NOT_CHECKED:
            status, reason = COVERAGE_NOT_CHECKED, str(doc.get("reason") or "")
        else:
            status, reason = COVERAGE_CHECKED, ""
            checked += 1
        per_phase[phase] = {"required": True, "status": status, "reason": reason}
    total = len(tuple(required))
    return {"required_phases": list(required), "phases_checked": checked,
            "phases_required": total, "summary": f"{checked}/{total}",
            "complete": checked == total, "phases": per_phase}


def qualification(*, full_mission_declared: bool, verdict: str,
                  coverage: Mapping[str, Any],
                  independent_validation: str | None = None) -> dict:
    """Whether the overall description may claim the mission *independently
    verified* (E8-03, task #283). True only when every blocker is absent:

    * the run declares itself the full mission (``full_mission: true``) -- a
      partial run describes the phases it flies, never a verified whole;
    * the full-mission verdict is ``passed`` -- an unchecked required phase
      caps it at ``not_checked`` and a red one fails it outright;
    * required-phase coverage is complete -- an out-of-scope or undeclared
      required phase never bypasses the claim;
    * the report's independent-evidence rollup is ``passed`` (the caller
      passes ``independent_validation`` from ``status.independent_validation``)
      -- internal consistency, however green, is not independent validation.

    Every blocker is recorded; the claim is false with a reason, never silent.
    """
    blockers: list[str] = []
    if not full_mission_declared:
        blockers.append("run does not declare full_mission: the verdict describes the phases "
                        "this mission flies, not an independently verified full mission")
    if verdict == VERDICT_FAILED:
        blockers.append("full-mission verdict failed")
    elif verdict != VERDICT_PASSED:
        ungraded = [p for p, entry in (coverage.get("phases") or {}).items()
                    if entry.get("status") != COVERAGE_CHECKED]
        blockers.append("required phase(s) not checked: "
                        + (", ".join(ungraded) or str(verdict))
                        + " (absent evidence is never a pass)")
    if not coverage.get("complete"):
        missed = [p for p, entry in (coverage.get("phases") or {}).items()
                  if entry.get("status") != COVERAGE_CHECKED]
        if missed:
            blockers.append("required-phase coverage incomplete ("
                            + str(coverage.get("summary", "?")) + "): " + ", ".join(missed))
    if independent_validation != VERDICT_PASSED:
        blockers.append("no passing validation-grade independent evidence "
                        f"(status.independent_validation: {independent_validation or 'absent'})")
    return {"independently_verified": not blockers,
            "independent_validation": independent_validation,
            "reason": "independently verified: full mission declared, every required physical "
                      "phase checked and passed, independent evidence passed" if not blockers
                      else "; ".join(blockers)}


def full_mission_gate(mission: Any, *,
                      independent_validation: str | None = None) -> dict:
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
    reason.

    A scenario declaring ``full_mission: true`` claims the complete physical
    mission, so every :data:`REQUIRED_PHASES` phase is forced applicable: an
    out-of-scope declaration of a required phase is refused (recorded in the
    phase reason, coverage shows it unchecked) instead of honoured. The
    document always carries the required-phase ``coverage`` (E8-03) and the
    ``qualification`` block; ``independent_validation`` is the caller's
    ``status.independent_validation.result``, so the qualification reads the
    report's own evidence rollup.

    The document also carries the E5-04 physical gate (:func:`physical_mission_gate`)
    under ``physical``: the same checks re-graded as five *required* checks,
    with no out-of-scope escape hatch -- a phase the E2-03 phase gate excuses
    (declared out of scope, or not flown) caps the physical gate at
    ``not_checked``, and the physical verdict is the verdict a computed
    mission must show to count as a full physical mission."""
    scenario = mission.scenario
    full_mission_declared = bool(getattr(scenario, "full_mission", False))
    recovery_declared = getattr(scenario, "recovery", None) is not None
    propagation_required = str(getattr(scenario.expected, "propagation_result",
                                       "passed")) != VERDICT_NOT_CHECKED
    applicable: dict[str, Any] = {
        PHASE_ORBIT_INJECTION: True,
        PHASE_PARKING_PROPAGATION: propagation_required or {
            "kind": COVERAGE_DECLARED_OUT_OF_SCOPE,
            "reason": "propagation declared out of scope (expected.propagation_result: "
                      "not_checked): the parking orbit is not required to be propagated"},
        PHASE_RECOVERY_TOUCHDOWN: recovery_declared or {
            "kind": COVERAGE_NOT_DECLARED,
            "reason": "no recovery stage declared: this mission flies no first-stage return, "
                      "touchdown is out of scope (an expendable mission is a smaller mission, "
                      "not a failed one)"},
    }
    refused: dict[str, str] = {}
    if full_mission_declared:
        for phase, entry in applicable.items():
            if isinstance(entry, Mapping):
                applicable[phase] = True
                refused[phase] = (
                    f"out-of-scope declaration refused: {phase} is a required physical phase "
                    f"of a run declared full_mission (declared: {entry.get('reason')})")
    reasons: dict[str, str] = dict(refused)
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
    coverage = phase_coverage(docs, REQUIRED_PHASES)
    return {"rule": OVERALL_RULE, "verdict": verdict, "reason_codes": reason_codes,
            "phases": phase_docs,
            "required_phases": list(REQUIRED_PHASES),
            "full_mission_declared": full_mission_declared,
            "coverage": coverage,
            "qualification": qualification(
                full_mission_declared=full_mission_declared, verdict=verdict,
                coverage=coverage, independent_validation=independent_validation),
            "physical": physical_mission_gate(mission)}


# ---------------------------------------------------------------------------
# E5-04 (task #271): the full physical mission gate
# ---------------------------------------------------------------------------

#: The five required checks of the physical gate, in mission order. Unlike
#: the E2-03 phases these are never conditionally applicable: a *computed*
#: mission (one whose trajectory was actually flown, not prescribed) must
#: show orbit insertion, parking-orbit propagation, a landing, mass
#: conservation across every split, and epoch/frame continuity across every
#: boundary. The E2-04 lesson (``docs/verification.md``): a phase declared
#: out of scope does not block the phase gate's verdict -- and that is
#: exactly why a declared-out-of-scope parking propagation cannot pass the
#: *physical* gate. Absent evidence is never a pass here either; a
#: declaration is not evidence.
PHYSICAL_CHECK_ORBIT_INSERTION = "orbit_insertion"
PHYSICAL_CHECK_PARKING_PROPAGATION = "parking_propagation"
PHYSICAL_CHECK_LANDING = "landing"
PHYSICAL_CHECK_MASS_CONSERVATION = "mass_conservation"
PHYSICAL_CHECK_EPOCH_FRAME_CONTINUITY = "epoch_frame_continuity"
PHYSICAL_CHECKS = (PHYSICAL_CHECK_ORBIT_INSERTION, PHYSICAL_CHECK_PARKING_PROPAGATION,
                   PHYSICAL_CHECK_LANDING, PHYSICAL_CHECK_MASS_CONSERVATION,
                   PHYSICAL_CHECK_EPOCH_FRAME_CONTINUITY)

#: Display labels (English-only UI, like the phase labels).
PHYSICAL_LABELS = {
    PHYSICAL_CHECK_ORBIT_INSERTION: "Orbit insertion",
    PHYSICAL_CHECK_PARKING_PROPAGATION: "Parking-orbit propagation",
    PHYSICAL_CHECK_LANDING: "Landing",
    PHYSICAL_CHECK_MASS_CONSERVATION: "Mass conservation",
    PHYSICAL_CHECK_EPOCH_FRAME_CONTINUITY: "Epoch/frame continuity",
}

#: The physical-gate rule, quoted verbatim into the report document. The
#: conjunction itself is :func:`combine_phase_verdicts` -- the same tri-state
#: rule as the phase gate -- but with every check required: no applicability
#: escape hatch, and an out-of-scope declaration (E2-03) never waives a
#: required check.
PHYSICAL_RULE = (
    "all five physical checks are required for a computed mission: failed if "
    "any check failed; else not_checked if any check carries no evidence "
    "(absent evidence is never a pass, and an out-of-scope declaration does "
    "not waive a required check); passed only when every check passed"
)

#: Reason code of a required check the run produced no evidence for -- each
#: check has its own, so "not checked" always names what is missing.
PHYSICAL_NOT_EVIDENCED_CODE = {name: f"{name}_not_evidenced" for name in PHYSICAL_CHECKS}


def physical_check_of(check_id: str) -> str | None:
    """The physical-gate check that grades the check with this id, or ``None``
    when the id is outside the physical vocabulary (recorded, never
    attributed). Attribution is by the law the check measures, not the phase
    it happened to run in:

    =====================  ====================================================
    physical check         checks (first matching rule wins)
    =====================  ====================================================
    ``mass_conservation``  ``mass.*`` and every boundary mass gate (any id
                           ending ``.mass_conservation``: staging, parking,
                           coupled, propagation, recovery handoff -- the law
                           is graded wherever it is measured)
    ``parking_propagation`` ``propagation.*`` (what is left after the mass law)
    ``landing``            ``recovery.*`` (touchdown, reserve, recovery handoff)
    ``epoch_frame_continuity`` ``handoff.*`` and ``parking.*`` -- the M-01
                           boundary gates (epoch monotonic, frame-identity
                           under every continuity gate, separation event,
                           coupled shared source)
    ``orbit_insertion``    ``station.*``, ``ascent.*``, ``expected.*``
    =====================  ====================================================

    The mixed run-level conjunction (:data:`MIXED_RUN_VERDICT_ID`) follows the
    E2-03 rule: attributed to insertion only while no recovery checks are
    present (handled by :func:`physical_mission_gate`)."""
    cid = str(check_id)
    if cid.startswith("mass.") or cid.endswith(".mass_conservation"):
        return PHYSICAL_CHECK_MASS_CONSERVATION
    if cid.startswith("propagation."):
        return PHYSICAL_CHECK_PARKING_PROPAGATION
    if cid.startswith("recovery."):
        return PHYSICAL_CHECK_LANDING
    if cid.startswith(("handoff.", "parking.")):
        return PHYSICAL_CHECK_EPOCH_FRAME_CONTINUITY
    if cid.startswith(("station.", "ascent.", "expected.")):
        return PHYSICAL_CHECK_ORBIT_INSERTION
    return None


def _physical_absent_reason(name: str, mission: Any) -> str:
    """Why a required check carries no evidence -- read from the scenario and
    the mission's own recorded notes, never invented."""
    scenario = mission.scenario
    if name == PHYSICAL_CHECK_PARKING_PROPAGATION:
        declared_out = str(getattr(scenario.expected, "propagation_result",
                                   "passed")) == VERDICT_NOT_CHECKED
        propagation = dict(getattr(mission, "propagation", None) or {})
        reason = str(propagation.get("reason") or "propagation did not run")
        if declared_out:
            reason += ("; the scenario declares propagation out of scope, but an "
                       "out-of-scope declaration never waives a required physical "
                       "check (E5-04: E2-04's declared-out-of-scope parking "
                       "propagation is exactly what this gate must not pass)")
        return reason
    if name == PHYSICAL_CHECK_LANDING:
        if getattr(scenario, "recovery", None) is not None:
            return next((str(n.get("reason", "")) for n in mission.not_assessed
                         if n.get("id") == "recovery"),
                        "recovery stage declared but never ran (the ascent "
                        "produced no flown run)")
        return ("no recovery stage declared: a computed physical mission must fly "
                "its first stage home -- an unflown landing is absent evidence, "
                "never a pass (an expendable mission is not a full physical mission)")
    if name == PHYSICAL_CHECK_ORBIT_INSERTION:
        return "no station, ascent or expected-orbit evidence: the ascent never flew"
    if name == PHYSICAL_CHECK_MASS_CONSERVATION:
        return "no mass bookkeeping or boundary mass gate was graded"
    return ("no boundary continuity gate was graded: the chain never crossed "
            "a staging, parking-orbit or coupled-separation handoff")


def physical_mission_gate(mission: Any) -> dict:
    """The full physical mission gate (E5-04, task #271) of a
    ``TwoStageLeoMission`` (same duck-typing as :func:`full_mission_gate`).
    Five required checks -- orbit insertion, parking propagation, landing,
    mass conservation, epoch/frame continuity -- each graded independently
    with its own tri-state verdict and its own reason codes over the checks
    the chain already produced (no new evidence, no new system), and one
    overall verdict: any check not ``passed`` (failed *or* merely
    ``not_checked``) keeps the whole gate from passing. Built into the
    full-mission gate document under ``physical`` by :func:`full_mission_gate`.
    """
    grouped = _physical_grouped(mission.checks)
    docs: list[dict] = []
    for name in PHYSICAL_CHECKS:
        mine = grouped[name]
        fail = [c for c in mine if not _flag(c)]
        verdict = (VERDICT_FAILED if fail
                   else VERDICT_PASSED if mine
                   else VERDICT_NOT_CHECKED)
        docs.append({
            "check": name,
            "label": PHYSICAL_LABELS[name],
            "required": True,
            "verdict": verdict,
            "checks_passed": sum(1 for c in mine if _flag(c)),
            "checks_total": len(mine),
            "reason_codes": (sorted({_reason_code(c) for c in fail}) if fail
                             else [PHYSICAL_NOT_EVIDENCED_CODE[name]]
                             if verdict == VERDICT_NOT_CHECKED else []),
            "checks": {_check_id(c): _flag(c) for c in mine},
            "reason": ("" if verdict != VERDICT_NOT_CHECKED
                       else _physical_absent_reason(name, mission)),
        })
    verdict = combine_phase_verdicts(docs)
    reason_codes = sorted({code for d in docs if d["verdict"] != VERDICT_PASSED
                           for code in d["reason_codes"]})
    return {"rule": PHYSICAL_RULE, "verdict": verdict, "reason_codes": reason_codes,
            "checks": docs}


def _physical_grouped(checks: Sequence[Any]) -> dict[str, list[Any]]:
    """Group mission-chain checks into the five physical checks
    (:func:`physical_check_of`). The mixed run-level conjunction follows the
    E2-03 rule: once recovery checks are present it mixes touchdown into the
    whole-flight conjunction, so it is recorded but not attributed (its
    halves are :data:`INSERTION_ONLY_VERDICT_ID` and the ``recovery.*``
    checks)."""
    has_recovery = any(_check_id(c).startswith("recovery.") for c in checks)
    grouped: dict[str, list[Any]] = {name: [] for name in PHYSICAL_CHECKS}
    for check in checks:
        check_id = _check_id(check)
        if check_id == MIXED_RUN_VERDICT_ID and has_recovery:
            continue
        name = physical_check_of(check_id)
        if name is not None:
            grouped[name].append(check)
    return grouped
