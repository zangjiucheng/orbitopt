"""What the verification guard returns: four separate status outcomes, findings and one verdict."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from ..problem import STATUSES


class Outcome(str, Enum):
    """The guard's finding for one status, or for the whole result."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ASSESSED = "not_assessed"


class FindingCode(str, Enum):
    """Why the guard objected or took note. Callers branch on this, never on messages.

    The nine codes ``objective_mismatch`` to ``forged_verification`` are the
    fault reasons of the E14-10 fault catalogue (``benchmarks/e14_spec/faults.py``).
    """

    OBJECTIVE_MISMATCH = "objective_mismatch"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DIGEST_MISMATCH = "digest_mismatch"
    MISSING_EVIDENCE = "missing_evidence"
    PROBLEM_MISMATCH = "problem_mismatch"
    UNAUTHORIZED_RELAXATION = "unauthorized_relaxation"
    FORMULATION_MISMATCH = "formulation_mismatch"
    STALE_CACHE = "stale_cache"
    FORGED_VERIFICATION = "forged_verification"
    RESIDUAL_MISMATCH = "residual_mismatch"
    DOMAIN_VIOLATION = "domain_violation"
    NON_FINITE = "non_finite"
    NOT_EVALUABLE = "not_evaluable"
    EVALUATION_ERROR = "evaluation_error"
    NOT_CONVERGED = "not_converged"
    CLAIM_CONTRADICTED = "claim_contradicted"
    SELF_ATTESTATION = "self_attestation"
    EVIDENCE_NOT_ALLOWED = "evidence_not_allowed"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    EVIDENCE_INVALID = "evidence_invalid"
    EVIDENCE_ERROR = "evidence_error"
    EVIDENCE_DISAGREES = "evidence_disagrees"
    EVIDENCE_INCONCLUSIVE = "evidence_inconclusive"
    CERTIFICATE_FAILED = "certificate_failed"
    CERTIFICATE_UNSUPPORTED = "certificate_unsupported"
    REVISED_PROBLEM = "revised_problem"
    INVALID_PROBLEM = "invalid_problem"
    MALFORMED_RESULT = "malformed_result"
    MALFORMED_ARTIFACT = "malformed_artifact"
    MISSING_FILE = "missing_file"
    NO_REQUIREMENT = "no_requirement"
    GUARD_ERROR = "guard_error"

    @property
    def fatal(self) -> bool:
        """A fatal finding makes the verdict ``failed`` whatever the statuses say."""
        return self not in _NOTES


_NOTES = frozenset({
    FindingCode.MISSING_EVIDENCE, FindingCode.NOT_EVALUABLE, FindingCode.NOT_CONVERGED,
    FindingCode.SELF_ATTESTATION, FindingCode.EVIDENCE_NOT_ALLOWED, FindingCode.EVIDENCE_MISMATCH,
    FindingCode.EVIDENCE_INVALID, FindingCode.EVIDENCE_ERROR, FindingCode.EVIDENCE_DISAGREES,
    FindingCode.EVIDENCE_INCONCLUSIVE, FindingCode.CERTIFICATE_FAILED, FindingCode.CERTIFICATE_UNSUPPORTED,
    FindingCode.REVISED_PROBLEM, FindingCode.NO_REQUIREMENT,
})
"""Codes that explain a status without condemning the result by themselves.

A note still blocks a pass whenever the status it explains is required:
missing evidence leaves ``independently_verified`` not assessed, and a
required not-assessed status is never a pass.
"""


@dataclass(frozen=True)
class Finding:
    """One objection or note: a code, a JSON-pointer-like path, a message."""

    code: FindingCode
    path: str
    message: str

    @property
    def fatal(self) -> bool:
        """Whether this finding alone fails the verdict."""
        return self.code.fatal


@dataclass(frozen=True)
class StatusOutcome:
    """The guard's own outcome for one status, why, and the evidence ids it rests on."""

    outcome: Outcome
    reason: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationReport:
    """The guard's grade of one result against one problem.

    ``statuses`` has exactly the four ``STATUSES``, each derived by the guard;
    the result's own booleans are claims it checked. ``verdict`` is
    ``passed`` only when every ``required`` status passed and no finding is
    fatal; it is ``failed`` when a required status failed or a finding is
    fatal, and ``not_assessed`` otherwise.
    """

    verdict: Outcome
    statuses: Mapping[str, StatusOutcome]
    required: tuple[str, ...]
    findings: tuple[Finding, ...]
    problem_digest: str | None = None
    result_digest: str | None = None

    @property
    def codes(self) -> frozenset[FindingCode]:
        """Every finding code in the report."""
        return frozenset(f.code for f in self.findings)

    def to_dict(self) -> dict:
        """The JSON-ready form."""
        return {
            "verdict": self.verdict.value,
            "required": list(self.required),
            "statuses": {name: {"outcome": s.outcome.value, "reason": s.reason, "evidence_ids": list(s.evidence_ids)}
                         for name, s in self.statuses.items()},
            "findings": [{"code": f.code.value, "path": f.path, "message": f.message, "fatal": f.fatal}
                         for f in self.findings],
            "problem_digest": self.problem_digest,
            "result_digest": self.result_digest,
        }


def decide(statuses: Mapping[str, StatusOutcome], required: tuple[str, ...], findings: tuple[Finding, ...]) -> Outcome:
    """The verdict rule: pass only on every required status passed and no fatal finding."""
    outcomes = [statuses[name].outcome for name in required]
    if any(f.fatal for f in findings) or Outcome.FAILED in outcomes:
        return Outcome.FAILED
    if outcomes and all(o is Outcome.PASSED for o in outcomes):
        return Outcome.PASSED
    return Outcome.NOT_ASSESSED


def unassessed_report(reason: str, findings: list[Finding], required: tuple[str, ...] = (),
                      problem_digest: str | None = None, result_digest: str | None = None) -> VerificationReport:
    """A report that graded nothing: every status not assessed for ``reason``."""
    statuses = {name: StatusOutcome(Outcome.NOT_ASSESSED, reason) for name in STATUSES}
    found = tuple(findings)
    return VerificationReport(decide(statuses, required, found), statuses, required, found,
                              problem_digest, result_digest)
