"""The data shapes of the experiment database: a record per run, the cache key, and the typed errors.

An ``ExperimentRecord`` describes one saved E14-05 run directory: what was
asked (the problem digest), what the planner chose (every solver step with
its options, seed, start point and allowance), what came out (the best
objective, its largest constraint violation and the guard's four statuses)
and what it cost (evaluations, wall clock and tokens). ``run_id`` is the
``sha256:`` digest of the run's ``manifest.json``, the value ``save_run``
returns, so the same run recorded twice is one record.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from orbitopt.agent.runtime import BudgetEvent, BudgetUse, Limits, RunStatus, StopReason
from orbitopt.optimization import canonical_digest, schema_errors

EXPERIMENT_SCHEMA = "agent-experiment-1.0.json"
EXPERIMENT_SCHEMA_VERSION = "1.0"
EXPERIMENT_FORMAT = "orbitopt.agent.experiment"
CACHE_FORMAT = "orbitopt.agent.cache/1"
STATUS_NAMES = ("solver_converged", "constraints_satisfied", "independently_verified", "optimality_certified")

RUN_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExperimentError(Exception):
    """Base of every error this package raises. Callers branch on the subclass, never on the message."""


class RunIntegrityError(ExperimentError):
    """A run directory failed its integrity check or its replay: it was changed, or it is not a run.

    ``problems`` lists what the check found, one line each.
    """

    def __init__(self, message: str, problems: tuple[str, ...] = ()):
        super().__init__(message + ("" if not problems else ": " + "; ".join(problems[:5])))
        self.problems = tuple(problems)


class InvalidIdError(ExperimentError, ValueError):
    """An id is not of the form ``sha256:<64 hex digits>``."""


class UnknownRunError(ExperimentError, LookupError):
    """No record has this run id."""


class ArtifactError(ExperimentError):
    """A stored artifact is missing or its bytes no longer match its digest."""


class SchemaVersionError(ExperimentError):
    """The database is newer than this code, or is not an experiment database."""


class MalformedRecordError(ExperimentError, ValueError):
    """A record does not match ``agent-experiment-1.0.json``."""


def check_run_id(value: Any) -> str:
    """``value`` when it is a run id, else ``InvalidIdError``."""
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise InvalidIdError(f"not a run id (sha256:<64 hex digits>): {value!r:.80}")
    return value


class Integrity(str, Enum):
    """How ingest verified the run.

    ``checked``: the runtime's ``check_run`` passed: every byte, the trace
    digest, the trace against the saved results and the guard's re-grade of
    the sealed run, with no solve run again. ``replayed``: ``replay``
    re-executed every proposal as well, a cancelled run up to its recorded
    cancel point, and matched the whole trace. Only a ``replayed`` record
    serves the cache, because only re-execution shows that the step it is
    keyed by produces its result.
    """

    CHECKED = "checked"
    REPLAYED = "replayed"


class VersionPolicy(str, Enum):
    """Which orbitopt versions may share a cache entry: the same version, the same ``major.minor``, or any."""

    EXACT = "exact"
    MINOR = "minor"
    ANY = "any"

    def token(self, version: str) -> str:
        if self is VersionPolicy.EXACT:
            return version
        if self is VersionPolicy.MINOR:
            return ".".join(version.split(".")[:2])
        return "*"


@dataclass(frozen=True)
class SolverStep:
    """One solver run in a run: a planner decision (``iteration-0003``) or a final-verification alternate
    (``verify-1-scipy.slsqp``). ``evaluations`` is the allowance it was given, not what it used."""

    label: str
    solver_id: str
    options: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    x0: Mapping[str, float] | None = None
    evaluations: int = 0
    max_iterations: int | None = None

    def to_dict(self) -> dict:
        return {"label": self.label, "solver_id": self.solver_id, "options": dict(self.options), "seed": self.seed,
                "x0": None if self.x0 is None else dict(self.x0), "evaluations": self.evaluations,
                "max_iterations": self.max_iterations}


def normalised(values: Mapping[str, Any] | None) -> dict | None:
    """Sorted keys, no ``None`` values, every number a float; an empty mapping is ``None``."""
    if not values:
        return None
    out = {}
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        out[str(key)] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    return out or None


def cache_key(problem_digest: str, solver_id: str, options: Mapping[str, Any] | None, seed: int,
              x0: Mapping[str, float] | None, evaluations: int, *, version: str,
              policy: VersionPolicy = VersionPolicy.EXACT) -> str:
    """The ``sha256:`` key of one solve: the problem, the solver, its normalised options, seed, start point and
    allowance, and the orbitopt version as ``policy`` reads it.

    The key's problem digest is what a hit carries as ``Provenance.cache_key``.
    Numbers count at the IR digest's 12 significant digits.
    """
    return "sha256:" + canonical_digest({
        "format": CACHE_FORMAT, "problem_digest": problem_digest, "solver_id": solver_id,
        "options": normalised(options), "seed": int(seed), "x0": normalised(x0), "evaluations": int(evaluations),
        "version_policy": policy.value, "version": policy.token(version)})


@dataclass(frozen=True)
class ExperimentRecord:
    """One recorded run. ``to_dict`` validates against ``agent-experiment-1.0.json``.

    ``objective`` is the best result's first objective in its declared sense,
    and ``max_violation`` the largest violation of a hard constraint at that
    point (0 when the problem has none), both ``None`` when the run has no
    best. ``statuses`` are the final report's four statuses, or the best
    candidate's screen when no final verification ran. ``created_at`` is when
    the run was recorded (UTC, ISO 8601); ``orbitopt_version`` is the version
    that ran it, from its trace (``0+unknown`` for a 1.0 trace). ``files`` maps each run file to its digest in
    the artifact store. ``budget`` is the trace's ``outcome.budget``; a record stored before it existed reads
    as an empty tuple.
    """

    run_id: str
    problem_id: str
    family: str
    problem_digest: str
    fingerprint: str
    trace_digest: str
    result_digest: str | None
    sealed_digest: str | None
    planner: str
    provider: str | None
    model: str | None
    seed: int
    steps: tuple[SolverStep, ...]
    best: str | None
    objective_name: str | None
    sense: str | None
    objective: float | None
    max_violation: float | None
    statuses: Mapping[str, str]
    status: RunStatus
    stop_reason: StopReason
    limits: Limits
    usage: BudgetUse
    created_at: str
    orbitopt_version: str
    integrity: Integrity
    files: Mapping[str, str]
    parent_id: str | None = None
    resumed_from: str | None = None
    budget: tuple[BudgetEvent, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status is RunStatus.VERIFIED

    @property
    def seeds(self) -> tuple[int, ...]:
        """The run seed, then every solver step's seed in order."""
        return (self.seed, *(s.seed for s in self.steps))

    @property
    def best_step(self) -> SolverStep | None:
        return next((s for s in self.steps if s.label == self.best), None)

    def to_dict(self) -> dict:
        data = {"schemaVersion": EXPERIMENT_SCHEMA_VERSION, "format": EXPERIMENT_FORMAT, "run_id": self.run_id,
                "problem_id": self.problem_id, "family": self.family, "problem_digest": self.problem_digest,
                "fingerprint": self.fingerprint, "trace_digest": self.trace_digest,
                "result_digest": self.result_digest, "sealed_digest": self.sealed_digest, "planner": self.planner,
                "provider": self.provider, "model": self.model, "seed": self.seed,
                "steps": [s.to_dict() for s in self.steps], "best": self.best,
                "objective_name": self.objective_name, "sense": self.sense, "objective": self.objective,
                "max_violation": self.max_violation, "statuses": dict(self.statuses), "status": self.status.value,
                "stop_reason": self.stop_reason.value, "limits": asdict(self.limits), "usage": asdict(self.usage),
                "created_at": self.created_at, "orbitopt_version": self.orbitopt_version,
                "integrity": self.integrity.value, "files": dict(self.files), "parent_id": self.parent_id,
                "resumed_from": self.resumed_from, "budget": [e.to_dict() for e in self.budget]}
        _validate(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        """Parse a stored record; ``MalformedRecordError`` when it does not match the schema."""
        _validate(data)
        fields = {k: v for k, v in data.items() if k not in ("schemaVersion", "format")}
        return cls(**{**fields, "steps": tuple(SolverStep(**s) for s in data["steps"]),
                      "status": RunStatus(data["status"]), "stop_reason": StopReason(data["stop_reason"]),
                      "limits": Limits(**data["limits"]), "usage": BudgetUse(**data["usage"]),
                      "integrity": Integrity(data["integrity"]),
                      "budget": tuple(BudgetEvent.from_dict(e) for e in data.get("budget", ()))})


def _validate(data: Mapping[str, Any]) -> None:
    errors = schema_errors(data, EXPERIMENT_SCHEMA)
    if errors:
        raise MalformedRecordError(f"record does not match {EXPERIMENT_SCHEMA}: "
                                   + "; ".join(f"{e.path}: {e.message}" for e in errors[:5]))
