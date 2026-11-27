"""Independent verification guard (E14-06): grade optimization results from evidence, never from claims.

``verify_result`` derives the four result statuses (``solver_converged``,
``constraints_satisfied``, ``independently_verified``,
``optimality_certified``) separately and returns a ``VerificationReport``
whose verdict passes only when every status the policy requires passed and
nothing fatal was found. ``seal_run`` and ``verify_run`` write and re-grade a
sealed run directory. See ``docs/optimization/verification.md``.

Importing this package does not import numpy; the KKT certificate imports it
when it runs.
"""

from .artifact import RUN_FORMAT, canonical_json, file_digest, seal_run, verify_run
from .certificates import KKTResult, kkt_certificate, polynomial_degree, refute_lower_bound, unsupported_reason
from .evidence import (
    AlternateSolver,
    AlternateSolverCheck,
    AnalyticalOptimum,
    EvidenceItem,
    LowerBoundCertificate,
    ReferenceSolution,
    evidence_from_dict,
    evidence_to_dict,
    result_digest,
)
from .guard import compare_formulations, verify_result
from .report import Finding, FindingCode, Outcome, StatusOutcome, VerificationReport

__all__ = [
    "AlternateSolver", "AlternateSolverCheck", "AnalyticalOptimum", "EvidenceItem", "Finding", "FindingCode",
    "KKTResult", "LowerBoundCertificate", "Outcome", "RUN_FORMAT", "ReferenceSolution",
    "StatusOutcome", "VerificationReport", "canonical_json", "compare_formulations", "evidence_from_dict",
    "evidence_to_dict", "file_digest", "kkt_certificate", "polynomial_degree", "refute_lower_bound",
    "result_digest", "seal_run", "unsupported_reason", "verify_result", "verify_run",
]
