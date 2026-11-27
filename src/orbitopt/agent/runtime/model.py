"""The data shapes of an agent run: limits and usage, the policy, decisions, iteration records and the trace.

A run is a sequence of ``IterationRecord`` values, one per planner proposal.
A proposal is untrusted. It becomes a ``Decision`` only after the runtime has
checked it against the ``Policy``, the solver registry and the problem; a
proposal that fails becomes a ``rejected`` record with its ``Violation``
list and is never applied. The ``Trace`` holds every record, the final
verification and the outcome, and its digest is the IR's 12-significant-digit
``canonical_digest`` of the trace with wall-clock fields left out, so two
runs on two machines that made the same choices share a digest.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cache
from importlib import metadata
from typing import Any, Literal, Mapping

from orbitopt.optimization import Budget, OptimizationResult, canonical_digest, schema_errors, to_plain
from orbitopt.optimization.result import stable
from orbitopt.optimization.verification import VerificationReport

TRACE_SCHEMA_VERSION = "1.1"
TRACE_SCHEMA = "agent-trace-1.1.json"
TRACE_SCHEMAS = {"1.0": "agent-trace-1.0.json", TRACE_SCHEMA_VERSION: TRACE_SCHEMA}
"""The trace schema for each ``schemaVersion`` that still parses; ``to_dict`` writes the newest."""
TRACE_FORMAT = "orbitopt.agent.trace"

ENVIRONMENT_KEYS = frozenset({"orbitopt_version"})
"""Top-level trace keys that describe the machine that ran, not the run; the digest and replay leave them out."""
UNKNOWN_VERSION = "0+unknown"


@cache
def orbitopt_version() -> str:
    """The installed orbitopt version (read once: the metadata lookup costs about 2 ms)."""
    try:
        return metadata.version("orbitopt")
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


DEFAULT_MAX_ITERATIONS = 20
"""Planner proposals per run when neither the caller nor the problem names a limit (E14-10 arm E's call cap)."""
DEFAULT_EVALUATIONS_PER_DIMENSION = 2000
"""Evaluations per variable, at least two variables, when no limit is named (the E14-10 per-cell budget)."""
DEFAULT_MAX_WALL_S = 600.0
DEFAULT_MAX_TOKENS = 200_000

class RunStatus(str, Enum):
    """What the run delivered. Only ``verified`` means the guard passed the final answer."""

    VERIFIED = "verified"
    UNVERIFIED_BEST = "unverified_best"
    NO_FEASIBLE = "no_feasible"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    POLICY_VIOLATION = "policy_violation"
    REFUSED = "refused"


class StopReason(str, Enum):
    """Why the loop ended."""

    PLANNER_STOP = "planner_stop"
    MAX_ITERATIONS = "max_iterations"
    MAX_EVALUATIONS = "max_evaluations"
    MAX_WALL_S = "max_wall_s"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"
    POLICY_VIOLATION = "policy_violation"
    PLANNER_FAILED = "planner_failed"
    REFUSED = "refused"

    @property
    def budget(self) -> bool:
        """Whether a budget limit ended the loop."""
        return self in (StopReason.MAX_ITERATIONS, StopReason.MAX_EVALUATIONS, StopReason.MAX_WALL_S,
                        StopReason.MAX_TOKENS)


class CancelReason(str, Enum):
    """Which cancel check fired: the one before the next planner proposal, or the one before the final verification."""

    BEFORE_ITERATION = "before_iteration"
    BEFORE_VERIFICATION = "before_verification"


@dataclass(frozen=True)
class CancelPoint:
    """Where a cancellation fired: after ``iteration`` completed iterations, at the check ``reason`` names.

    Passed to ``solve`` as ``cancel``, it cancels a run at exactly that
    point, which is how ``replay`` reproduces a cancelled run.
    """

    iteration: int
    reason: CancelReason

    def to_dict(self) -> dict:
        return {"iteration": self.iteration, "reason": self.reason.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CancelPoint":
        return cls(int(data["iteration"]), CancelReason(data["reason"]))


class Action(str, Enum):
    SOLVE = "solve"
    STOP = "stop"


class RecordKind(str, Enum):
    """``solve``: a decision ran a solver. ``stop``: the planner ended the loop. ``rejected``: a proposal
    failed validation and was not applied."""

    SOLVE = "solve"
    STOP = "stop"
    REJECTED = "rejected"


class ViolationCode(str, Enum):
    """Why a proposal was refused. Callers branch on this, never on messages."""

    PROBLEM_EDIT = "problem_edit"
    """The proposal tried to change the objective, a constraint, a bound, a parameter, the budget or the policy."""
    UNKNOWN_FIELD = "unknown_field"
    MALFORMED = "malformed"
    UNKNOWN_SOLVER = "unknown_solver"
    SOLVER_NOT_ALLOWED = "solver_not_allowed"
    INVALID_OPTION = "invalid_option"
    X0_UNKNOWN_VARIABLE = "x0_unknown_variable"
    X0_OUT_OF_DOMAIN = "x0_out_of_domain"
    BUDGET_OVERREACH = "budget_overreach"


@dataclass(frozen=True)
class Limits:
    """The run's hard limits. Every one is finite: a run can never retry without bound.

    ``max_iterations`` counts planner proposals, accepted or rejected.
    ``max_evaluations`` counts every evaluation of the problem made for the
    run: by solvers, by the verification guard and by alternate solvers.
    """

    max_iterations: int
    max_evaluations: int
    max_wall_s: float
    max_tokens: int

    @classmethod
    def resolve(cls, problem_budget: Budget, override: Budget | None, dimension: int) -> "Limits":
        """Each limit from ``override``, else from the problem's budget, else the E14-10 default."""
        def pick(name: str, default):
            for source in (override, problem_budget):
                value = None if source is None else getattr(source, name)
                if value is not None:
                    return value
            return default

        return cls(int(pick("max_iterations", DEFAULT_MAX_ITERATIONS)),
                   int(pick("max_evaluations", DEFAULT_EVALUATIONS_PER_DIMENSION * max(dimension, 2))),
                   float(pick("max_wall_s", DEFAULT_MAX_WALL_S)), int(pick("max_tokens", DEFAULT_MAX_TOKENS)))


@dataclass(frozen=True)
class BudgetUse:
    """What a run has used so far."""

    iterations: int = 0
    evaluations: int = 0
    tokens: int = 0
    wall_s: float = 0.0

    def exhausted(self, limits: Limits) -> StopReason | None:
        """The first limit reached, or ``None``."""
        for reason, used, limit in ((StopReason.MAX_ITERATIONS, self.iterations, limits.max_iterations),
                                    (StopReason.MAX_EVALUATIONS, self.evaluations, limits.max_evaluations),
                                    (StopReason.MAX_TOKENS, self.tokens, limits.max_tokens),
                                    (StopReason.MAX_WALL_S, self.wall_s, limits.max_wall_s)):
            if used >= limit:
                return reason
        return None


@dataclass(frozen=True)
class Policy:
    """What the planner may change. The problem, its budget and its verification policy are never on the list.

    ``allowed_solvers`` limits the solver ids a decision may name (``None``:
    every available, compatible solver). ``allow_options`` and ``allow_x0``
    let a decision set solver options and a start point. A refused proposal
    is recorded; with ``on_violation="stop"`` it ends the run, otherwise the
    run goes on until ``max_violations`` proposals have been refused.
    """

    allowed_solvers: tuple[str, ...] | None = None
    allow_options: bool = True
    allow_x0: bool = True
    on_violation: Literal["continue", "stop"] = "continue"
    max_violations: int = 3

    def __post_init__(self):
        if self.allowed_solvers is not None:
            object.__setattr__(self, "allowed_solvers", tuple(self.allowed_solvers))
        if self.on_violation not in ("continue", "stop"):
            raise ValueError(f"on_violation must be 'continue' or 'stop', not {self.on_violation!r}")
        if self.max_violations < 1:
            raise ValueError("max_violations must be at least 1")

    def to_dict(self) -> dict:
        return {**asdict(self), "allowed_solvers": None if self.allowed_solvers is None else list(self.allowed_solvers)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Policy":
        return cls(**data)


@dataclass(frozen=True)
class Violation:
    """One reason a proposal was refused, with the JSON-pointer path of the offending field."""

    code: ViolationCode
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code.value, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class Decision:
    """A validated step: run ``solver_id`` with ``options`` from ``x0`` under ``seed``, within ``evaluations``.

    ``max_iterations`` is the backend's own iteration cap for this solve,
    not the run's limit. A ``stop`` decision carries only its rationale.
    """

    action: Action
    solver_id: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    x0: Mapping[str, float] | None = None
    seed: int = 0
    evaluations: int = 0
    max_iterations: int | None = None
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action.value, "solver_id": self.solver_id, "options": dict(self.options),
                "x0": None if self.x0 is None else dict(self.x0), "seed": self.seed,
                "evaluations": self.evaluations, "max_iterations": self.max_iterations, "rationale": self.rationale}


@dataclass(frozen=True)
class Proposal:
    """What a planner returned: the raw, untrusted decision object, or the reason it could not produce one.

    ``tokens`` are the LLM tokens the proposal cost (0 for a rule-based planner).
    """

    raw: Mapping[str, Any] | None
    tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {"raw": None if self.raw is None else to_plain(dict(self.raw)), "tokens": self.tokens,
                "error": self.error}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Proposal":
        return cls(data["raw"], data["tokens"], data["error"])


class FailureKind(str, Enum):
    """What a solve's outcome means for the next step (see ``diagnostics``)."""

    VERIFIED_LOCAL = "verified_local"
    """Converged with a certificate, after an exhaustive enumeration, or to a first-order point of a problem that is
    provably convex."""
    LOCAL_OPTIMUM = "local_optimum"
    """Converged to a first-order point of a problem that is not provably convex: possibly only local."""
    HEURISTIC_STOP = "heuristic_stop"
    """A heuristic converged (its population stagnated): no optimality claim at all."""
    NON_CONVERGENCE = "non_convergence"
    STALLED = "stalled"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    EVALUATION_ERROR = "evaluation_error"
    SOLVER_ERROR = "solver_error"
    BUDGET = "budget"
    REJECTED_BY_GUARD = "rejected_by_guard"
    """The guard found a dishonest claim (a mismatched objective, another problem, a stale cache)."""


class Remedy(str, Enum):
    """A bounded next step. None of them touches the problem: constraints, bounds and objective stay fixed."""

    STOP = "stop"
    REFINE_LOCALLY = "refine_locally"
    RESTART_GLOBAL = "restart_global"
    RESTART_PERTURBED = "restart_perturbed"
    SWITCH_SOLVER = "switch_solver"
    LOOSEN_SOLVER_TOLERANCE = "loosen_solver_tolerance"
    CHANGE_INITIAL_GUESS = "change_initial_guess"


@dataclass(frozen=True)
class Diagnosis:
    kind: FailureKind
    remedies: tuple[Remedy, ...]
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "remedies": [r.value for r in self.remedies], "detail": self.detail}


def result_summary(result: OptimizationResult) -> dict:
    """The parts of a result a trace keeps; the full result is saved beside the trace."""
    return {"solver_id": result.solver_id, "problem_digest": result.problem_digest,
            "termination": result.termination.value, "solver_converged": result.solver_converged,
            "constraints_satisfied": result.constraints_satisfied, "x": dict(result.x),
            "objective_values": dict(result.objective_values),
            "constraint_residuals": dict(result.constraint_residuals), "evaluations": result.evaluations,
            "iterations": result.iterations, "message": result.message, "wall_time_s": result.wall_time_s}


def report_summary(report: VerificationReport) -> dict:
    """The guard's verdict, the four statuses with reasons, and the finding codes, sorted."""
    return {"verdict": report.verdict.value,
            "statuses": {name: {"outcome": s.outcome.value, "reason": s.reason} for name, s in report.statuses.items()},
            "codes": sorted(c.value for c in report.codes)}


@dataclass(frozen=True)
class IterationRecord:
    """One planner proposal and what came of it.

    ``evaluations`` counts every problem evaluation made in this iteration
    (the solver's and the guard's screen); ``tokens`` the proposal's tokens.
    ``screen`` is the guard's report on the candidate, graded without
    independent evidence, so its verdict is never ``passed`` under a policy
    that requires ``independently_verified``.
    """

    index: int
    kind: RecordKind
    proposal: Proposal
    decision: Decision | None = None
    violations: tuple[Violation, ...] = ()
    result: OptimizationResult | None = None
    screen: VerificationReport | None = None
    diagnosis: Diagnosis | None = None
    evaluations: int = 0
    tokens: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return {"index": self.index, "kind": self.kind.value, "proposal": self.proposal.to_dict(),
                "decision": None if self.decision is None else self.decision.to_dict(),
                "violations": [v.to_dict() for v in self.violations],
                "result": None if self.result is None else result_summary(self.result),
                "screen": None if self.screen is None else report_summary(self.screen),
                "diagnosis": None if self.diagnosis is None else self.diagnosis.to_dict(),
                "evaluations": self.evaluations, "tokens": self.tokens, "wall_s": self.wall_s}


@dataclass(frozen=True)
class AlternateRun:
    """An alternate solver the guard ran during final verification, and what it found."""

    solver_id: str
    allocation: int
    seed: int
    options: Mapping[str, Any]
    result: OptimizationResult | None

    def to_dict(self) -> dict:
        return {"solver_id": self.solver_id, "allocation": self.allocation, "seed": self.seed,
                "options": dict(self.options), "result": None if self.result is None else result_summary(self.result)}


@dataclass(frozen=True)
class VerificationRecord:
    """One final-verification attempt on a candidate (``candidate`` is a record index, or an alternate's label)."""

    attempt: int
    candidate: str
    alternates: tuple[AlternateRun, ...]
    report: VerificationReport
    evaluations: int
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return {"attempt": self.attempt, "candidate": self.candidate,
                "alternates": [a.to_dict() for a in self.alternates], "report": report_summary(self.report),
                "evaluations": self.evaluations, "wall_s": self.wall_s}


@dataclass(frozen=True)
class Trace:
    """Everything a run did, in order, with the outcome. ``to_dict`` validates against ``agent-trace-1.1.json``.

    ``cancel`` is where a cancellation fired, ``None`` when none did.
    ``provider`` and ``model`` name the LLM behind the planner (``None``
    without one) and are part of the digest, since another model is another
    run. ``orbitopt_version`` is the version that ran it; it describes the
    machine, so the digest leaves it out (``ENVIRONMENT_KEYS``).
    """

    problem_id: str
    problem_digest: str
    fingerprint: str
    planner: str
    seed: int
    limits: Limits
    policy: Policy
    solvers: tuple[str, ...]
    iterations: tuple[IterationRecord, ...]
    verifications: tuple[VerificationRecord, ...]
    status: RunStatus
    stop_reason: StopReason
    best: str | None
    usage: BudgetUse
    refusal: tuple[str, ...] = ()
    cancel: CancelPoint | None = None
    provider: str | None = None
    model: str | None = None
    orbitopt_version: str = field(default_factory=orbitopt_version)

    def body(self) -> dict:
        """The trace as JSON-ready data, without its digest."""
        return {"schemaVersion": TRACE_SCHEMA_VERSION, "format": TRACE_FORMAT, "problem_id": self.problem_id,
                "problem_digest": self.problem_digest, "fingerprint": self.fingerprint, "planner": self.planner,
                "provider": self.provider, "model": self.model, "orbitopt_version": self.orbitopt_version,
                "seed": self.seed, "limits": asdict(self.limits), "policy": self.policy.to_dict(),
                "solvers": list(self.solvers), "iterations": [r.to_dict() for r in self.iterations],
                "verifications": [v.to_dict() for v in self.verifications],
                "outcome": {"status": self.status.value, "stop_reason": self.stop_reason.value, "best": self.best,
                            "refusal": list(self.refusal),
                            "cancel": None if self.cancel is None else self.cancel.to_dict()},
                "usage": asdict(self.usage)}

    def digest(self) -> str:
        return trace_digest(self.body())

    def to_dict(self) -> dict:
        data = {**self.body(), "digest": self.digest()}
        errors = schema_errors(data, TRACE_SCHEMA)
        if errors:  # the runtime built this itself; a mismatch is a bug, not bad input
            raise AssertionError(f"trace does not match {TRACE_SCHEMA}: {errors[:3]}")
        return data


def upgrade_trace(data: Mapping[str, Any]) -> dict:
    """A parsed trace of any version in the newest version's shape; ``digest`` stays the recorded one.

    A 1.0 trace has no cancel point: ``outcome.cancel`` is ``None``, so a
    cancelled 1.0 run replays as it did under 1.0, without a cancel signal.
    It names no provider, model or version either: all three are ``None``.
    """
    trace = copy.deepcopy(dict(data))
    if trace.get("schemaVersion") == "1.0":
        trace.update(schemaVersion=TRACE_SCHEMA_VERSION, provider=None, model=None, orbitopt_version=None)
        trace["outcome"] = {**trace["outcome"], "cancel": None}
    return trace


def trace_digest(body: Mapping[str, Any]) -> str:
    """``canonical_digest`` of a trace body without its digest, ``ENVIRONMENT_KEYS`` and wall-clock fields
    (``VOLATILE_KEYS``)."""
    return canonical_digest(stable({k: v for k, v in body.items() if k != "digest" and k not in ENVIRONMENT_KEYS}))


@dataclass(frozen=True)
class RunOutcome:
    """What ``solve`` returns: the status, the best answer with the guard's final report, and the trace.

    ``result`` is the best candidate by the guard's grading: a point the
    guard found feasible, best objective first. It is ``None`` only when no
    solve produced a point. ``report`` is the final verification of that
    candidate, or ``None`` when none ran; ``status`` is ``verified`` only when
    that report passed. ``evidence`` is what the final report was graded on.
    """

    status: RunStatus
    stop_reason: StopReason
    result: OptimizationResult | None
    report: VerificationReport | None
    evidence: tuple[Any, ...]
    trace: Trace
    problem: Any
    results: Mapping[str, OptimizationResult] = field(default_factory=dict)
    """Every result of the run by label (``iteration-0003``, ``verify-1-scipy.slsqp``), for persistence."""
    tool_log: Any = None

    @property
    def verified(self) -> bool:
        return self.status is RunStatus.VERIFIED

    @property
    def statuses(self) -> dict[str, str]:
        """The final report's four statuses (``passed``, ``failed`` or ``not_assessed``); empty without a report."""
        return {} if self.report is None else {k: s.outcome.value for k, s in self.report.statuses.items()}
