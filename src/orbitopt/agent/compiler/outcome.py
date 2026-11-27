"""The closed set of results ``compile_problem`` returns.

A compile ends in exactly one of five outcomes. Refusals are values, not
exceptions: a missing parameter, an unsupported model, an unauthorised change
and an LLM that never produced a valid draft are all normal results that a UI
shows to the user. Only infrastructure failures (provider unreachable, budget
exhausted, replay divergence) propagate as exceptions.

``Compiled`` is the only outcome that carries an ``OptimizationProblem``, and
it refuses to hold one that ``validate_problem`` rejects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, Mapping, Union

from orbitopt.agent.providers import Usage
from orbitopt.optimization import OptimizationProblem, ProblemError, validate_problem

QuestionKind = Literal["missing_parameter", "ambiguity", "conflicting_objectives", "unsupported_feature", "other"]
UnsupportedCategory = Literal["unsupported_model", "stochastic", "dynamic", "soft_constraint", "variable_kind",
                              "semi_infinite", "other"]
ViolationKind = Literal["removed_locked_constraint", "changed_locked_constraint", "removed_user_objective",
                        "changed_user_objective", "added_user_objective", "changed_domain", "removed_variable",
                        "changed_parameter", "removed_parameter", "untrusted_authorization"]


class CompilerErrorCode(str, Enum):
    """Compiler-level failure codes; IR failures keep their ``ErrorCode``."""

    STRUCTURED_OUTPUT = "structured_output"
    MISSING_PROBLEM = "missing_problem"
    UNCOVERED_STATEMENT = "uncovered_statement"
    UNIT_HINT_MISMATCH = "unit_hint_mismatch"


@dataclass(frozen=True)
class CompilerIssue:
    """A compiler-level error, shaped like ``ProblemError``: a code, a JSON-pointer path, a message."""

    code: CompilerErrorCode
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: [{self.code.value}] {self.message}"


Issue = Union[ProblemError, CompilerIssue]


@dataclass(frozen=True)
class Question:
    """Something the user must answer before the problem can be compiled."""

    kind: QuestionKind
    text: str
    about: str = ""


@dataclass(frozen=True)
class Assumption:
    """Something the model added that the user did not state; ``about`` names the item, e.g. ``variables/x``."""

    about: str
    text: str


@dataclass(frozen=True)
class UnsupportedReason:
    """Why the request is outside what the IR and the known tools can express."""

    category: UnsupportedCategory
    explanation: str


@dataclass(frozen=True)
class Violation:
    """A change to what the user fixed that nothing in the user's own text authorises."""

    kind: ViolationKind
    item: str
    message: str


@dataclass(frozen=True)
class Compiled:
    """A valid IR problem, the assumptions behind it, and the cost of getting it.

    ``attempts`` counts compiler rounds (one ``generate_structured`` call
    each, which may itself include the provider's schema repair retry);
    ``usage`` sums every round.
    """

    problem: OptimizationProblem
    assumptions: tuple[Assumption, ...]
    usage: Usage
    attempts: int
    outcome: ClassVar[str] = "compile"

    def __post_init__(self):
        errors = validate_problem(self.problem)
        if errors:
            raise ValueError(f"Compiled must hold a valid problem, got {len(errors)} errors: {errors[0]}")


@dataclass(frozen=True)
class NeedsClarification:
    """Questions for the user; ``partial`` is the model's raw draft, untrusted and not an IR problem."""

    questions: tuple[Question, ...]
    partial: Mapping[str, Any] | None
    usage: Usage
    attempts: int
    outcome: ClassVar[str] = "needs_clarification"


@dataclass(frozen=True)
class Unsupported:
    """The request needs something the IR cannot express; no approximation is offered."""

    reason: UnsupportedReason
    usage: Usage
    attempts: int
    outcome: ClassVar[str] = "unsupported"


@dataclass(frozen=True)
class RejectedUnauthorizedChange:
    """The draft changes what the user fixed without the user's authorisation."""

    violations: tuple[Violation, ...]
    usage: Usage
    attempts: int
    outcome: ClassVar[str] = "rejected_unauthorized_change"


@dataclass(frozen=True)
class CompileFailed:
    """No valid draft within the repair budget; ``errors`` are the last round's, ``raw`` its reply text.

    When the provider itself gave up (``StructuredOutputError``), the tokens
    of that last call are not reported by the provider API and are missing
    from ``usage``.
    """

    errors: tuple[Issue, ...]
    raw: str | None
    usage: Usage
    attempts: int
    outcome: ClassVar[str] = "compile_failed"


CompileOutcome = Union[Compiled, NeedsClarification, Unsupported, RejectedUnauthorizedChange, CompileFailed]
