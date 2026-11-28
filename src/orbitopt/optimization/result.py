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

from .problem import SCHEMA_VERSION, to_plain

RESULT_SCHEMA = "optimization-result-1.1.json"
RESULT_SCHEMAS = {"1.0": "optimization-result-1.0.json", SCHEMA_VERSION: RESULT_SCHEMA}
"""The result schema for each ``schemaVersion`` that still parses; ``to_dict`` writes the newest."""
VOLATILE_KEYS = frozenset({"wall_s", "wall_time_s", "latency_s"})
"""Keys that measure this machine's clock, not a choice or an answer. Identity digests (``result_digest``, the
agent trace digest) leave them out at any depth, so two identical runs share a digest."""
EVIDENCE_KINDS = ("solver", "analytical", "reference_solution", "alternate_solver", "optimality_certificate")
"""``analytical``, ``reference_solution`` and ``alternate_solver`` match ``INDEPENDENT_METHODS``."""


def stable(data: Any) -> Any:
    """``data`` without the ``VOLATILE_KEYS``, at any depth."""
    if isinstance(data, Mapping):
        return {k: stable(v) for k, v in data.items() if k not in VOLATILE_KEYS}
    if isinstance(data, (list, tuple)):
        return [stable(v) for v in data]
    return data


class Termination(str, Enum):
    """Why the run stopped."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    MAX_EVALUATIONS = "max_evaluations"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNBOUNDED = "unbounded"
    """The backend reports that the objective improves without bound on the feasible set."""
    STALLED = "stalled"
    """The backend stopped making progress before its own convergence test passed (a line search, stagnation)."""


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence behind a status: what kind, a one-line summary, its data and where it came from.

    ``producer`` names what produced it (a solver id, a tool id, a method).
    ``problem_digest`` names the problem it was produced for, so an item
    copied in from another problem's run is visible; it must equal the
    result's ``problem_digest``. An item inside a result cannot name that
    result's digest (the digest covers the item), so evidence pinned to a
    result travels beside it, as the guard's evidence types do.
    """

    kind: str
    summary: str
    data: Mapping[str, Any] = field(default_factory=dict)
    producer: str = ""
    problem_digest: str | None = None


@dataclass(frozen=True)
class ParetoPoint:
    """One member of a multi-objective front: the point and its objective values, as in the result."""

    x: Mapping[str, float]
    objective_values: Mapping[str, float]


@dataclass(frozen=True)
class Provenance:
    """How the result was obtained. It is part of the result, so ``result_digest`` pins it.

    ``cache_key`` is the problem digest a cache stored the result under when
    the result was served from a cache, and ``None`` for a fresh solve. A
    cache hit whose key is not the problem the result answers is a stale
    cache, which the guard reports.
    """

    cache_key: str | None = None


@dataclass(frozen=True, kw_only=True)
class OptimizationResult:
    """The answer to one problem, identified by ``problem_digest``.

    ``x`` holds the best point found (empty when there is none), in each
    variable's unit. ``objective_values`` are in each objective's unit and
    sense as declared (a maximised objective is not negated).
    ``constraint_residuals`` are canonical (``Constraint.residual``). Values
    must be finite; omit a name whose value is not. ``front`` holds the
    non-dominated feasible points of a multi-objective solve (``x`` is one of
    them) and is empty otherwise.
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
    front: tuple[ParetoPoint, ...] = ()
    provenance: Provenance = Provenance()

    def __post_init__(self):
        object.__setattr__(self, "termination", Termination(self.termination))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "front", tuple(self.front))

    def to_dict(self) -> dict:
        """The JSON-ready dict form; every field is written."""
        return {"schemaVersion": SCHEMA_VERSION, **to_plain(self)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "OptimizationResult":
        """Inverse of ``to_dict`` for schema-valid input of any version in ``RESULT_SCHEMAS``.

        Check it first with ``schema_errors(data, schema_for(data, RESULT_SCHEMAS))``.
        """
        fields = {k: v for k, v in data.items() if k != "schemaVersion"}
        fields["evidence"] = tuple(Evidence(**e) for e in data.get("evidence", ()))
        fields["front"] = tuple(ParetoPoint(dict(p["x"]), dict(p["objective_values"])) for p in data.get("front", ()))
        fields["provenance"] = Provenance(**data.get("provenance", {}))
        for key in ("x", "objective_values", "constraint_residuals"):
            fields[key] = dict(data.get(key, {}))
        return cls(**fields)
