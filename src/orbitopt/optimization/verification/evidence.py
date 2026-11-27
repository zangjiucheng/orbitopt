"""Evidence the guard can check itself, each item pinned to the problem and result it grades.

An evidence item is not a verdict. It is a claim that the guard re-checks
against the result in hand: an ``AnalyticalOptimum`` states a closed-form
optimum and its derivation, a ``ReferenceSolution`` states a stored optimum
and its tolerance, an ``AlternateSolverCheck`` hands the guard a solver to run
itself, and a ``LowerBoundCertificate`` states a lower bound the guard tries
to refute. Every item names ``problem_digest`` (``OptimizationProblem.digest()``)
and ``result_digest`` (``result_digest(result)``); an item whose digests are
missing or name another problem or result never counts.

Items travel through ``evidence.json`` in a sealed run. ``AlternateSolverCheck``
serialises without its callable, so ``verify_run`` needs the solver again by
``solver_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar, Mapping, Protocol, runtime_checkable

from ..problem import OptimizationProblem, canonical_digest
from ..result import OptimizationResult, stable

INDEPENDENT_KINDS = ("analytical", "reference_solution", "alternate_solver")
CERTIFICATE_KIND = "optimality_certificate"


def result_digest(result: OptimizationResult) -> str:
    """SHA-256 of the result's dict form, with the IR's 12-significant-digit float quantisation.

    This identifies a result the way ``OptimizationProblem.digest`` identifies
    a problem. The wall-clock keys of ``VOLATILE_KEYS`` (``wall_time_s``, and
    any such key inside evidence data) are left out, so two runs that found
    the same answer the same way share it. It is an identity, not a tamper
    seal: two points that agree to 12 significant digits share it.
    ``seal_run`` seals the exact bytes, wall clock included.
    """
    return canonical_digest(stable(result.to_dict()))


@runtime_checkable
class AlternateSolver(Protocol):
    """A solver the guard runs itself: takes the graded problem, returns an ``OptimizationResult``."""

    def __call__(self, problem: OptimizationProblem) -> OptimizationResult: ...


@dataclass(frozen=True, kw_only=True)
class AnalyticalOptimum:
    """A closed-form optimal value of one objective, with the derivation that proves it.

    ``x``, when given, is the closed-form optimiser: the guard checks that it
    lies in the problem's feasible set and attains ``value``, so a derivation
    that names a point can be checked numerically, not only read.
    """

    kind: ClassVar[str] = "analytical"
    id: str
    problem_digest: str
    result_digest: str
    value: float
    derivation: str
    objective: str | None = None
    x: Mapping[str, float] | None = None
    producer: str = "analytical"


@dataclass(frozen=True, kw_only=True)
class ReferenceSolution:
    """A stored optimal value for this problem, with where it came from and an absolute tolerance.

    The guard uses ``min(tolerance, objective_rtol * max(1, |value|))``, so a
    reference can tighten the policy but never loosen it.
    """

    kind: ClassVar[str] = "reference_solution"
    id: str
    problem_digest: str
    result_digest: str
    value: float
    source: str
    tolerance: float | None = None
    objective: str | None = None
    producer: str = "reference"


@dataclass(frozen=True, kw_only=True)
class AlternateSolverCheck:
    """Ask the guard to solve the problem again with ``solver`` and compare objective values.

    The guard calls ``solver`` itself; it never reads a result someone else
    obtained. ``solver_id`` must differ from the graded result's
    ``solver_id``, and so must the ``solver_id`` of what ``solver`` returns.
    """

    kind: ClassVar[str] = "alternate_solver"
    id: str
    problem_digest: str
    result_digest: str
    solver_id: str
    objective: str | None = None
    solver: AlternateSolver | None = field(default=None, compare=False)

    @property
    def producer(self) -> str:
        """The alternate solver's id."""
        return self.solver_id


@dataclass(frozen=True, kw_only=True)
class LowerBoundCertificate:
    """A claimed lower bound on one objective's minimum (upper bound when maximising), with its derivation.

    The guard checks that the result attains the bound within
    ``optimality_gap`` and tries to refute the bound on seeded samples of the
    feasible set. Sampling can refute a bound but never prove one, so this
    certificate can fail ``optimality_certified`` and cannot pass it on its
    own (see ``docs/optimization/verification.md``).
    """

    kind: ClassVar[str] = CERTIFICATE_KIND
    id: str
    problem_digest: str
    result_digest: str
    bound: float
    derivation: str
    objective: str | None = None
    samples: int = 256
    producer: str = "analytical"


EvidenceItem = AnalyticalOptimum | ReferenceSolution | AlternateSolverCheck | LowerBoundCertificate
_TYPES: dict[str, type] = {t.kind: t for t in (AnalyticalOptimum, ReferenceSolution, AlternateSolverCheck,
                                               LowerBoundCertificate)}


def evidence_to_dict(item: EvidenceItem) -> dict:
    """The JSON form of an evidence item; an ``AlternateSolverCheck`` drops its callable."""
    data = {f.name: getattr(item, f.name) for f in fields(item) if f.name != "solver"}
    if data.get("x") is not None:
        data["x"] = dict(data["x"])
    return {"kind": item.kind, **data}


def evidence_from_dict(data: Mapping[str, Any],
                       alternate_solvers: Mapping[str, Callable[[OptimizationProblem], OptimizationResult]] | None = None
                       ) -> EvidenceItem:
    """Inverse of ``evidence_to_dict``; an alternate solver is looked up by ``solver_id``.

    Raises ``ValueError`` for an unknown kind or fields, so a malformed item
    is reported by the caller instead of silently dropped.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"an evidence item must be an object, not {type(data).__name__}")
    kind = data.get("kind")
    if kind not in _TYPES:
        raise ValueError(f"unknown evidence kind {kind!r}")
    values = {k: v for k, v in data.items() if k != "kind"}
    if kind == AlternateSolverCheck.kind:
        values["solver"] = (alternate_solvers or {}).get(values.get("solver_id"))
    try:
        return _TYPES[kind](**values)
    except TypeError as exc:
        raise ValueError(f"malformed {kind} evidence: {exc}") from None
