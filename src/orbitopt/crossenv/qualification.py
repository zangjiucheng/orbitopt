"""E5-06 (task #290): the joint full-mission qualification of a cross-env run.

A cross-environment plan grades its two legs in two different interpreters;
until now the joined report only AND-ed the stage ``mission_ok`` flags. The
epic's qualification closure needs the *same* conclusion the single-process
path produces -- one ``full_mission`` document (E2-03 phases, E5-04 physical
gate, E8-03 coverage/qualification) computed over the *union* of the sealed
evidence of both legs, with every physical check naming the environment and
envelope digest it rests on.

This module is that union -- nothing else. It re-reads the sealed exchanges
through :func:`orbitopt.crossenv.envelope.read_envelope` (schema + every
digest re-verified in this process, before a single byte is parsed), lifts
the ascent leg's mission report and the recovery leg's sealed checks into a
duck-typed mission, and hands it to
:func:`orbitopt.mission.phases.full_mission_gate`. No second gate system, no
new evidence: an environment boundary changes where a check was measured, not
what measuring it means.

Fail-closed semantics (the negative tests of task #290):

* a missing leg, a leg whose exchange no longer verifies (a broken shared
  separation state, any tampered artifact), or propagation evidence stripped
  from the ascent report all leave the joint verdict ``not_checked`` or
  ``failed`` -- absent or unverifiable evidence is never a pass;
* a red landing on the recovery leg fails the joint verdict;
* the document itself is always complete, so a consumer reads *why* the
  mission did not qualify, never a silent gap.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from orbitopt.crossenv.envelope import CrossEnvError, envelope_digest, read_envelope

JOINT_KIND = "orbitopt.crossenv.joint-qualification"
JOINT_SCHEMA_VERSION = "1.0"

#: The sealed payload each leg must carry for the joint gate to see it. The
#: ascent leg ships the wrapped mission report (the stage runner's own
#: ``reports/mission-report.json``); the recovery leg ships the graded
#: recovery checks in the wrapper below, so the landing leg's evidence crosses
#: the boundary as a first-class, digest-sealed document -- not as a summary
#: line in a stage report.
ASCENT_REPORT_PAYLOAD = "mission-report"
RECOVERY_CHECKS_PAYLOAD = "recovery-checks"
RECOVERY_CHECKS_KIND = "orbitopt.crossenv.recovery-checks"
REPORT_WRAPPER_KIND = "orbitopt.crossenv.report"

ASCENT_RUNNER = "ascent"
RECOVERY_RUNNER = "recovery"

#: Check ids the recovery leg measures (its own namespace plus the
#: insertion-only half it derives from the combined run); everything else in
#: the union was measured by the ascent leg.
_RECOVERY_OWNED = ("recovery.",)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _is_recovery_owned(check_id: str) -> bool:
    from orbitopt.mission.phases import INSERTION_ONLY_VERDICT_ID

    return str(check_id).startswith(_RECOVERY_OWNED) or str(check_id) == INSERTION_ONLY_VERDICT_ID


def _independent_rollup(checks: Sequence[Mapping[str, Any]]) -> str | None:
    """``status.independent_validation`` over the union of checks, computed
    exactly the way the single-process report rollups it (:func:`_check_
    category` / :func:`_check_evidence` + :func:`status_from_checks`) -- the
    joint qualification reads the evidence the legs produced, not a claim."""
    ids = {str(c["id"]): bool(c["passed"]) for c in checks}
    if not ids:
        return None
    from orbitopt.mission.two_stage_leo import _check_category, _check_evidence
    from orbitopt.verify.report import status_from_checks

    status = status_from_checks(ids, {k: _check_category(k) for k in ids},
                                {k: _check_evidence(k) for k in ids})
    return str(status["independent_validation"]["result"])


def _report_payload(exchange: Path, envelope: Mapping[str, Any], name: str) -> dict:
    """Load one ``report`` payload of an already-validated envelope (its
    digest was re-hashed by ``read_envelope``; the kind/version check mirrors
    the envelope's own reader without loading the heavy run payloads)."""
    entry = next((p for p in envelope["payloads"] if p["name"] == name), None)
    if entry is None:
        raise CrossEnvError(f"exchange seals no {name!r} payload")
    if entry.get("report_kind") not in (REPORT_WRAPPER_KIND, RECOVERY_CHECKS_KIND):
        raise CrossEnvError(f"payload {name!r}: unexpected report kind "
                            f"{entry.get('report_kind')!r}")
    doc = json.loads((exchange / Path(entry["path"])).read_text(encoding="utf-8"))
    if doc.get("kind") != entry["report_kind"]:
        raise CrossEnvError(f"payload {name!r}: document kind {doc.get('kind')!r} is not the "
                            f"sealed {entry['report_kind']!r}")
    return doc


def _leg(entry: Mapping[str, Any], problems: list[str]) -> dict | None:
    """One stage entry plus its freshly re-verified envelope, or ``None``
    with the reason recorded in ``problems``."""
    stage = str(entry.get("id"))
    ex_value = str(entry.get("exchange") or "")
    if not ex_value:
        problems.append(f"stage {stage!r}: no sealed exchange to grade (the leg never "
                        "completed or sealed nothing)")
        return None
    exchange = Path(ex_value)
    leg: dict[str, Any] = {"stage": stage, "runner": entry.get("runner"),
                           "env": entry.get("env"),
                           "python_executable": entry.get("python_executable"),
                           "exchange": str(exchange)}
    try:
        envelope = read_envelope(exchange)
    except CrossEnvError as exc:
        problems.append(f"stage {stage!r}: exchange {str(exchange)!r} did not re-verify: {exc}")
        leg["error"] = str(exc)
        return {"_failed": leg}
    digest = envelope_digest(exchange)
    recorded = entry.get("exchange_digest")
    if recorded and recorded != digest:
        problem = (f"stage {stage!r}: envelope digest mismatch -- the mission report pins "
                   f"{recorded}, the envelope on disk hashes {digest}")
        problems.append(problem)
        leg["error"] = problem
        return {"_failed": leg}
    leg["exchange_digest"] = digest
    leg["environment"] = dict(envelope["environment"])
    leg["payloads"] = {p["name"]: p["digest"] for p in envelope["payloads"]}
    leg["_envelope"] = envelope
    return leg


def joint_qualification(*, scenario_path, stages: Sequence[Mapping[str, Any]],
                        plan_full_mission: bool = False) -> dict:
    """The joint ``full_mission`` gate document of a finished cross-env run.

    ``stages`` are the stage entries of the mission report (runner, env,
    exchange dir and the digest the orchestrator recorded); ``scenario_path``
    is the shared scenario the plan declares -- applicability is read from
    the scenario, never from what the legs happened to produce, exactly like
    the single-process gate. ``plan_full_mission`` carries the plan's
    ``full_mission: true`` claim into the virtual scenario so a cross-env run
    cannot qualify on a subset of the required phases (E8-03).

    Never raises for evidence trouble: every missing/unverifiable leg becomes
    a recorded problem and a ``not_checked`` phase, so the verdict -- and the
    CI gate built on it -- fails closed instead of crashing open.
    """
    from dataclasses import replace

    from orbitopt.mission.phases import (PHASES, PHYSICAL_CHECKS, full_mission_gate)
    from orbitopt.mission.two_stage_leo import load_scenario

    scenario = load_scenario(scenario_path)
    declared = bool(plan_full_mission or getattr(scenario, "full_mission", False))
    if declared and not scenario.full_mission:
        scenario = replace(scenario, full_mission=True)

    problems: list[str] = []
    evidence: dict[str, dict] = {}
    checks: list[dict] = []
    not_assessed: list[dict] = []
    propagation: dict = {"status": "not_run",
                         "reason": "no ascent-leg exchange was graded"}

    legs: dict[str, dict] = {}
    for runner in (ASCENT_RUNNER, RECOVERY_RUNNER):
        entry = next((s for s in stages if s.get("runner") == runner), None)
        if entry is None:
            problems.append(f"the plan runs no {runner!r} stage: the joint gate has no "
                            f"{runner} evidence")
            continue
        leg = _leg(entry, problems)
        if leg is None:
            continue
        if "_failed" in leg:
            evidence[runner] = leg["_failed"]
            continue
        evidence[runner] = {k: v for k, v in leg.items() if not k.startswith("_")}
        legs[runner] = leg

    if ASCENT_RUNNER in legs:
        leg = legs[ASCENT_RUNNER]
        try:
            report_doc = _report_payload(Path(leg["exchange"]), leg["_envelope"],
                                        ASCENT_REPORT_PAYLOAD)
            mission = report_doc.get("report") or {}
            checks.extend(dict(c) for c in mission.get("checks") or [])
            not_assessed.extend(dict(n) for n in mission.get("not_assessed") or [])
            if mission.get("propagation"):
                propagation = dict(mission["propagation"])
        except (CrossEnvError, json.JSONDecodeError) as exc:
            problems.append(f"stage {leg['stage']!r}: unreadable sealed mission report: {exc}")
            checks_ascent = False
        else:
            checks_ascent = True
        if not checks_ascent:
            del legs[ASCENT_RUNNER]
            not_assessed.append({"id": "ascent",
                                 "reason": "ascent report unreadable (see problems)"})
    else:
        not_assessed.append({"id": "ascent",
                             "reason": "no ascent leg graded: the orbit was never injected "
                                       "in an environment this gate can see"})

    if RECOVERY_RUNNER in legs:
        leg = legs[RECOVERY_RUNNER]
        try:
            checks_doc = _report_payload(Path(leg["exchange"]), leg["_envelope"],
                                        RECOVERY_CHECKS_PAYLOAD)
            checks.extend(dict(c) for c in checks_doc.get("checks") or [])
            not_assessed.extend(dict(n) for n in checks_doc.get("not_assessed") or [])
        except (CrossEnvError, json.JSONDecodeError) as exc:
            problems.append(f"stage {leg['stage']!r}: unreadable sealed recovery checks: {exc}")
            del legs[RECOVERY_RUNNER]
            not_assessed.append({"id": "recovery",
                                 "reason": f"recovery evidence unreadable: {exc}"})
    else:
        not_assessed.append({"id": "recovery",
                             "reason": "no recovery leg graded: the first stage never came "
                                       "home in an environment this gate can see"})

    mission = SimpleNamespace(scenario=scenario, checks=checks,
                              not_assessed=not_assessed, propagation=propagation)
    gate = full_mission_gate(mission, independent_validation=_independent_rollup(checks))

    # Evidence provenance per physical check: which leg measured it, in which
    # environment, pinned by the envelope digest re-computed here.
    def _source_for(check_id: str) -> dict:
        runner = RECOVERY_RUNNER if _is_recovery_owned(check_id) else ASCENT_RUNNER
        leg = legs.get(runner)
        if leg is None:
            return {"check": str(check_id), "stage": None, "env": None,
                    "reason": "the leg that would own this check produced no verifiable "
                              "evidence"}
        return {"check": str(check_id), "stage": leg["stage"], "runner": runner,
                "env": leg["env"], "python_executable": leg["python_executable"],
                "exchange_digest": leg["exchange_digest"],
                "payload": RECOVERY_CHECKS_PAYLOAD if runner == RECOVERY_RUNNER
                else ASCENT_REPORT_PAYLOAD}

    for doc in gate["physical"]["checks"]:
        sources: dict[tuple, dict] = {}
        for check_id in sorted(doc["checks"]):
            src = _source_for(check_id)
            sources[(src.get("stage"), src.get("payload"), src.get("exchange_digest"))] = src
        doc["sources"] = list(sources.values()) or [
            {"stage": None, "reason": "no evidence contributes to this check"}]

    return {
        "kind": JOINT_KIND,
        "schema_version": JOINT_SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "rule": gate["rule"],
        "verdict": gate["verdict"],
        "physical_mission_verdict": gate["physical"]["verdict"],
        "full_mission_declared": declared,
        "plan_full_mission": bool(plan_full_mission),
        "reason_codes": gate["reason_codes"],
        "phases": {doc["phase"]: doc for doc in gate["phases"]},
        "phase_order": list(PHASES),
        "required_phases": gate["required_phases"],
        "coverage": gate["coverage"],
        "qualification": gate["qualification"],
        "physical": gate["physical"],
        "physical_order": list(PHYSICAL_CHECKS),
        "problems": problems,
        "evidence": evidence,
    }
