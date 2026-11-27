"""The shared output contract of every solver, tool chain and agent run.

The four statuses are four independent booleans that default to ``False``.
Nothing in this module derives one from another: a converged solver is not a
satisfied constraint set, and neither is an independent verification. Each is
set by whoever produced the evidence for it, and the verification guard
(E14-06) checks them against ``evidence`` and the problem's
``VerificationPolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .problem import SCHEMA_VERSION, _plain

RESULT_SCHEMA = "optimization-result-1.1.json"
RESULT_SCHEMAS = {"1.0": "optimization-result-1.0.json", SCHEMA_VERSION: RESULT_SCHEMA}
"""The result schema for each ``schemaVersion`` that still parses; ``to_dict`` writes the newest."""
EVIDENCE_KINDS = ("solver", "analytical", "reference_solution", "alternate_solver", "optimality_certificate")
"""``analytical``, ``reference_solution`` and ``alternate_solver`` match ``INDEPENDENT_METHODS``."""


class Termination(str, Enum):
    """Why the run stopped."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    MAX_EVALUATIONS = "max_evaluations"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence behind a status: what kind, a one-line summary, and its data."""

    kind: str
    summary: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class OptimizationResult:
    """The answer to one problem, identified by ``problem_digest``.

    ``x`` holds the best point found (empty when there is none), in each
    variable's unit. ``objective_values`` are in each objective's unit and
    sense as declared (a maximised objective is not negated).
    ``constraint_residuals`` are canonical (``Constraint.residual``). Values
    must be finite; omit a name whose value is not.
    """

    problem_digest: str
    solver_id: str
    termination: Termination
    solver_converged: bool = False
    constraints_satisfied: bool = False
    independently_verified: bool = False
    optimality_certified: bool = False
    x: Mapping[str, float] = field(default_factory=dict)
    objective_values: Mapping[str, float] = field(default_factory=dict)
    constraint_residuals: Mapping[str, float] = field(default_factory=dict)
    evaluations: int = 0
    iterations: int | None = None
    wall_time_s: float = 0.0
    message: str = ""
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "termination", Termination(self.termination))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict:
        """The JSON-ready dict form; every field is written."""
        return {"schemaVersion": SCHEMA_VERSION, **_plain(self)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "OptimizationResult":
        """Inverse of ``to_dict`` for schema-valid input of any version in ``RESULT_SCHEMAS``.

        Check it first with ``schema_errors(data, schema_for(data, RESULT_SCHEMAS))``.
        """
        fields = {k: v for k, v in data.items() if k != "schemaVersion"}
        fields["evidence"] = tuple(Evidence(**e) for e in data.get("evidence", ()))
        for key in ("x", "objective_values", "constraint_residuals"):
            fields[key] = dict(data.get(key, {}))
        return cls(**fields)
