"""Read the experiment database: resume an interrupted run, query records, compare runs, aggregate families.

Every answer has ``to_dict`` (JSON-ready) and ``to_text`` (a plain table).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from orbitopt.agent.runtime import BudgetUse, Limits, Policy, Proposal, RunStatus, StopReason
from orbitopt.optimization import OptimizationProblem, parse_problem

from .index import ExperimentDB
from .model import STATUS_NAMES, ExperimentRecord

RESUMABLE = frozenset({StopReason.CANCELLED, StopReason.MAX_ITERATIONS, StopReason.MAX_EVALUATIONS,
                       StopReason.MAX_WALL_S, StopReason.MAX_TOKENS, StopReason.PLANNER_FAILED})
"""Stop reasons that interrupted a run before its planner chose to stop."""
ORDER_BY = ("created_at", "objective", "max_violation", "evaluations", "wall_s", "tokens")
MAX_PAGE = 1000


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    body = [[cell(v) for v in row] for row in rows]
    widths = [max([len(h), *(len(r[i]) for r in body)]) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)), "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in body]
    return "\n".join(line.rstrip() for line in lines)


# ------------------------------------------------------------------------------------------------ resume
@dataclass(frozen=True)
class ResumePlan:
    """What an interrupted run completed and what it may still spend.

    ``proposals`` are the run's planner proposals in order, accepted or
    refused, for a ``RecordedPlanner`` to serve before a live planner takes
    over. ``remaining`` is ``limits`` minus ``used``, never below zero, so
    the limit that ended the run is at zero.
    """

    run_id: str
    problem: OptimizationProblem
    status: RunStatus
    stop_reason: StopReason
    seed: int
    planner: str
    policy: Policy
    proposals: tuple[Proposal, ...]
    limits: Limits
    used: BudgetUse
    remaining: Limits
    best: str | None
    objective: float | None

    @property
    def resumable(self) -> bool:
        """Whether a limit or a cancellation ended the run, not its planner or a refusal."""
        return self.stop_reason in RESUMABLE

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "problem_digest": self.problem.digest(), "status": self.status.value,
                "stop_reason": self.stop_reason.value, "resumable": self.resumable, "seed": self.seed,
                "planner": self.planner, "policy": self.policy.to_dict(),
                "proposals": [p.to_dict() for p in self.proposals], "limits": asdict(self.limits),
                "used": asdict(self.used), "remaining": asdict(self.remaining), "best": self.best,
                "objective": self.objective}

    def to_text(self) -> str:
        rows = [(name, getattr(self.limits, name), getattr(self.used, used), getattr(self.remaining, name))
                for name, used in (("max_iterations", "iterations"), ("max_evaluations", "evaluations"),
                                   ("max_wall_s", "wall_s"), ("max_tokens", "tokens"))]
        head = (f"run {self.run_id}: {self.status.value} ({self.stop_reason.value}), "
                f"{len(self.proposals)} proposals, {'resumable' if self.resumable else 'complete'}")
        return head + "\n" + _table(("limit", "limit value", "used", "remaining"), rows)


def resume_plan(db: ExperimentDB, run_id: str) -> ResumePlan:
    """The recorded proposals and the remaining limits of ``run_id``, read through the store's integrity check."""
    record = db.get(run_id)
    trace = db.store.get_json(record.files["trace.json"])
    problem = parse_problem(db.store.get_json(record.files["problem.json"]))
    used, limits = record.usage, record.limits
    remaining = Limits(max(0, limits.max_iterations - used.iterations),
                       max(0, limits.max_evaluations - used.evaluations),
                       max(0.0, limits.max_wall_s - used.wall_s), max(0, limits.max_tokens - used.tokens))
    return ResumePlan(record.run_id, problem, record.status, record.stop_reason, record.seed, record.planner,
                      Policy.from_dict(trace["policy"]),
                      tuple(Proposal.from_dict(item["proposal"]) for item in trace["iterations"]), limits, used,
                      remaining, record.best, record.objective)


# ------------------------------------------------------------------------------------------------ query
@dataclass(frozen=True)
class Page:
    """One page of a query: the records, and the number of records that match in all."""

    records: tuple[ExperimentRecord, ...]
    total: int
    limit: int
    offset: int

    def to_dict(self) -> dict:
        return {"total": self.total, "limit": self.limit, "offset": self.offset,
                "records": [r.to_dict() for r in self.records]}

    def to_text(self) -> str:
        return _table(("run_id", "problem", "planner", "status", "objective", "evals", "wall_s", "tokens"),
                      [(r.run_id[:19], r.problem_id, r.planner, r.status.value, r.objective, r.usage.evaluations,
                        r.usage.wall_s, r.usage.tokens) for r in self.records])


def _instant(value: datetime | str) -> str:
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(moment, datetime):
        raise TypeError(f"a date bound must be a datetime or an ISO 8601 string, not {type(value).__name__}")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def query(db: ExperimentDB, *, problem_digest: str | None = None, solver_id: str | None = None,
          status: RunStatus | str | None = None, verified: bool | None = None, family: str | None = None,
          seed: int | None = None, since: datetime | str | None = None, until: datetime | str | None = None,
          order_by: str = "created_at", descending: bool = True, limit: int = 50, offset: int = 0) -> Page:
    """Records that match every filter given, sorted and paginated.

    ``solver_id`` matches a run that ran that solver at any step. ``seed`` is
    the run seed. ``since`` and ``until`` bound ``created_at``, inclusive; a
    naive datetime is UTC. ``order_by`` is one of ``ORDER_BY``, with missing
    values last and the run id breaking ties.
    """
    if order_by not in ORDER_BY:
        raise ValueError(f"order_by must be one of {ORDER_BY}, not {order_by!r:.80}")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE or not isinstance(offset, int) or offset < 0:
        raise ValueError(f"limit must be 1..{MAX_PAGE} and offset at least 0")
    where, params = [], []
    for clause, value in (("problem_digest = ?", problem_digest), ("family = ?", family),
                          ("status = ?", None if status is None else RunStatus(status).value),
                          ("verified = ?", None if verified is None else int(bool(verified))),
                          ("seed = ?", None if seed is None else int(seed)),
                          ("created_at >= ?", None if since is None else _instant(since)),
                          ("created_at <= ?", None if until is None else _instant(until)),
                          ("run_id IN (SELECT run_id FROM steps WHERE solver_id = ?)", solver_id)):
        if value is not None:
            where.append(clause)
            params.append(value)
    condition = " WHERE " + " AND ".join(where) if where else ""
    total = db.rows("SELECT COUNT(*) FROM experiments" + condition, params)[0][0]
    direction = "DESC" if descending else "ASC"
    rows = db.rows(f"SELECT record FROM experiments{condition} ORDER BY ({order_by} IS NULL), {order_by} {direction}, "
                   "run_id LIMIT ? OFFSET ?", [*params, limit, offset])
    return Page(tuple(ExperimentRecord.from_dict(json.loads(r[0])) for r in rows), total, limit, offset)


# ------------------------------------------------------------------------------------------------ compare
@dataclass(frozen=True)
class ComparisonRow:
    """One run beside the others. ``gap`` is how much worse its objective is than ``best`` in the objective's
    sense (0 for the best); ``relative_gap`` is ``gap / max(1, |best|)``, the guard's objective scale."""

    run_id: str
    problem_id: str
    planner: str
    solvers: tuple[str, ...]
    status: str
    objective: float | None
    best: float | None
    gap: float | None
    relative_gap: float | None
    evaluations: int
    wall_s: float
    tokens: int
    statuses: Mapping[str, str]

    def to_dict(self) -> dict:
        return {**asdict(self), "solvers": list(self.solvers), "statuses": dict(self.statuses)}


@dataclass(frozen=True)
class Comparison:
    rows: tuple[ComparisonRow, ...]

    def to_dict(self) -> dict:
        return {"rows": [r.to_dict() for r in self.rows]}

    def to_text(self) -> str:
        short = {"solver_converged": "conv", "constraints_satisfied": "feas", "independently_verified": "indep",
                 "optimality_certified": "cert"}
        return _table(("run_id", "problem", "status", "objective", "gap", "rel_gap", "evals", "wall_s", "tokens",
                       "statuses"),
                      [(r.run_id[:19], r.problem_id, r.status, r.objective, r.gap, r.relative_gap, r.evaluations,
                        r.wall_s, r.tokens, " ".join(f"{short[k]}={r.statuses[k]}" for k in STATUS_NAMES if k in r.statuses))
                       for r in self.rows])


def compare(db: ExperimentDB, run_ids: Sequence[str]) -> Comparison:
    """``run_ids`` side by side, in the order given (repeats dropped).

    The best is taken per problem digest, among the runs whose final
    ``constraints_satisfied`` status passed, so a gap compares runs of one
    problem only. A run with no objective, or a problem with no feasible run
    among those compared, has no gap.
    """
    records = [db.get(i) for i in dict.fromkeys(run_ids)]
    best: dict[str, float] = {}
    for r in records:
        if r.objective is None or r.statuses.get("constraints_satisfied") != "passed":
            continue
        prior = best.get(r.problem_digest)
        better = prior is None or (r.objective > prior if r.sense == "maximize" else r.objective < prior)
        if better:
            best[r.problem_digest] = r.objective
    rows = []
    for r in records:
        ref = best.get(r.problem_digest)
        gap = None if ref is None or r.objective is None else (ref - r.objective if r.sense == "maximize"
                                                               else r.objective - ref)
        rows.append(ComparisonRow(r.run_id, r.problem_id, r.planner, tuple(dict.fromkeys(s.solver_id for s in r.steps)),
                                  r.status.value, r.objective, ref, gap,
                                  None if gap is None else gap / max(1.0, abs(ref)), r.usage.evaluations,
                                  r.usage.wall_s, r.usage.tokens, dict(r.statuses)))
    return Comparison(tuple(rows))


# ------------------------------------------------------------------------------------------------ aggregate
@dataclass(frozen=True)
class FamilyStats:
    family: str
    runs: int
    verified: int
    unverified_best: int
    other: int
    median_evaluations: float
    median_wall_s: float
    median_tokens: float


@dataclass(frozen=True)
class Aggregate:
    families: tuple[FamilyStats, ...]

    def to_dict(self) -> dict:
        return {"families": [asdict(f) for f in self.families]}

    def to_text(self) -> str:
        return _table(("family", "runs", "verified", "unverified_best", "other", "median_evals", "median_wall_s",
                       "median_tokens"), [tuple(asdict(f).values()) for f in self.families])


def aggregate(db: ExperimentDB) -> Aggregate:
    """Per problem family: how many runs, how many verified or ``unverified_best``, and the median costs."""
    by_family: dict[str, list[tuple]] = {}
    for family, status, evaluations, wall_s, tokens in db.rows(
            "SELECT family, status, evaluations, wall_s, tokens FROM experiments ORDER BY family"):
        by_family.setdefault(family, []).append((status, evaluations, wall_s, tokens))
    families = []
    for family, rows in by_family.items():
        statuses = [r[0] for r in rows]
        verified = statuses.count(RunStatus.VERIFIED.value)
        unverified = statuses.count(RunStatus.UNVERIFIED_BEST.value)
        families.append(FamilyStats(family, len(rows), verified, unverified, len(rows) - verified - unverified,
                                    float(statistics.median(r[1] for r in rows)),
                                    float(statistics.median(r[2] for r in rows)),
                                    float(statistics.median(r[3] for r in rows))))
    return Aggregate(tuple(families))
