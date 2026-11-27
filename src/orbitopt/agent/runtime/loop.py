"""The iteration loop: propose, validate, solve, screen, diagnose; then verify the best candidate independently.

``solve`` owns every limit. Planner proposals, accepted or refused, count as
iterations. Every evaluation of the problem made for the run goes through
one ``EvaluationMeter``: the solver adapters', the guard's re-evaluations
and the alternate solvers' alike, so the evaluation count is a measurement,
not a sum of what solvers report. The meter also refuses any evaluation past
the current allowance, so even a solver that ignores its ``Budget`` cannot
overspend. A quarter of the evaluation budget (``VERIFICATION_SHARE``) is
held back for the final verification, which runs the E14-06 guard with
alternate solvers as independent evidence.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Mapping

from orbitopt.agent.tools import Caller, ToolError, ToolRegistry, ToolView, bind_problem
from orbitopt.optimization import Budget, OptimizationProblem, OptimizationResult, ToolCall, validate_problem
from orbitopt.optimization.classify import classify
from orbitopt.optimization.solvers import SolverCapability, SolverRegistry, default_registry
from orbitopt.optimization.verification import (AlternateSolverCheck, Outcome, VerificationReport, result_digest,
                                                unsupported_reason, verify_result)

from .diagnostics import diagnose, distrust, feasible
from .model import (Action, AlternateRun, BudgetUse, CancelPoint, CancelReason, FailureKind, IterationRecord, Limits,
                    Policy, Proposal, RecordKind, RunOutcome, RunStatus, StopReason, Trace, VerificationRecord,
                    Violation, ViolationCode)
from .planners import Planner, PlannerState, PlannerUnavailable, default_options
from .validation import validate_proposal

VERIFICATION_SHARE = 0.25
"""Fraction of ``max_evaluations`` the planner can never spend: it is kept for the final verification."""
MIN_DECISION_EVALUATIONS = 8
"""The loop ends with ``max_evaluations`` once a decision could spend fewer evaluations than this."""
MAX_ALTERNATES = 2
MAX_VERIFICATION_ATTEMPTS = 2
"""A refuted candidate is replaced by the alternate's better point, which is verified once more."""
MIN_ALTERNATE_EVALUATIONS = 8
SCREEN_UNGRADED = frozenset({"independently_verified", "optimality_certified"})
"""Statuses a screen leaves to the final verification: it has no independent evidence and asks for no certificate."""

Tools = ToolView | Mapping[str, Callable[..., Any]]
Cancel = threading.Event | Callable[[], bool] | CancelPoint


class EvaluationBudgetExceeded(Exception):
    """Raised by the meter instead of an evaluation past the allowance. Not an ``EvaluationError``: a solver
    must stop on it, not skip the point."""


class EvaluationMeter:
    """Counts every evaluation of one problem and refuses those past ``allowance``.

    ``problem`` is a copy of the problem whose ``evaluate`` goes through the
    meter. The copy has the same fields, so the same digest and fingerprint,
    and solvers and the guard take it as the problem itself.
    """

    def __init__(self, problem: OptimizationProblem, allowance: int):
        self.count = 0
        self.allowance = allowance
        self.problem = copy.copy(problem)
        evaluate = OptimizationProblem.evaluate

        def metered(x, tools=None):
            if self.count >= self.allowance:
                raise EvaluationBudgetExceeded(f"evaluation {self.count + 1} is past the allowance of {self.allowance}")
            self.count += 1
            return evaluate(self.problem, x, tools)

        object.__setattr__(self.problem, "evaluate", metered)


def known_tool_ids(tools: ToolView | ToolRegistry | Mapping[str, Any] | None) -> frozenset[str]:
    """The tool ids a run may evaluate: a view's allowlist, every id of a registry, or a mapping's keys."""
    if tools is None:
        return frozenset()
    if isinstance(tools, ToolView):
        return tools.allowed
    if isinstance(tools, ToolRegistry):
        return frozenset(spec.tool_id for spec in tools.list())
    return frozenset(tools)


def screen_cost(problem: OptimizationProblem) -> int:
    """The most evaluations one guard call makes: the re-evaluation, plus the KKT stencil where it applies.

    The KKT certificate evaluates the point and ``2 n**2`` stencil points; it
    runs only on the class it can certify, and only when asked or claimed.
    """
    n = len(problem.variables)
    return 1 + (2 * n * n + 1 if unsupported_reason(problem) is None else 0)


def solve(problem: OptimizationProblem, *, planner: Planner, budget: Budget | None = None,
          policy: Policy | None = None, tools: Tools | None = None, seed: int = 0, cancel: Cancel | None = None,
          registry: SolverRegistry | None = None) -> RunOutcome:
    """Run ``planner`` on ``problem`` within the limits and return the verified best answer, or say why not.

    ``budget`` overrides ``problem.budget`` field by field; a limit named by
    neither takes the E14-10 default (``Limits.resolve``). ``policy`` says what
    the planner may change. ``tools`` is a ``ToolView`` (its allowlist is the
    set of known tool ids, and every call is logged) or a mapping of resolvers.
    ``cancel`` is checked before each planner proposal and before the final
    verification; the trace records where it fired. A ``CancelPoint`` cancels
    at exactly that point, as ``replay`` does. The verification policy is
    always ``problem.verification``. Never raises for a bad proposal or a
    failing solver; a malformed problem is refused, not raised.
    """
    run = _Run(problem, planner, Limits.resolve(problem.budget, budget, len(problem.variables)), policy or Policy(),
               tools, seed, _cancel_check(cancel), registry or default_registry())
    return run.execute()


def _cancel_check(cancel: Cancel | None) -> Callable[[int, CancelReason], bool]:
    """Whether to cancel after ``done`` iterations at the check ``reason`` names."""
    if cancel is None:
        return lambda done, reason: False
    if isinstance(cancel, CancelPoint):
        return lambda done, reason: (done, reason) == (cancel.iteration, cancel.reason)
    check = cancel.is_set if isinstance(cancel, threading.Event) else cancel
    return lambda done, reason: bool(check())


class _Run:
    """The mutable state of one ``solve`` call; nothing outlives it."""

    def __init__(self, problem, planner, limits, policy, tools, seed, cancelled, registry):
        self.asked, self.planner, self.limits, self.policy = problem, planner, limits, policy
        self.seed, self.cancel_check, self.registry, self.tools = seed, cancelled, registry, tools
        self.cancel_point: CancelPoint | None = None
        self.started = time.perf_counter()
        self.meter = EvaluationMeter(problem, limits.max_evaluations)
        self.problem = self.meter.problem
        self.records: list[IterationRecord] = []
        self.verifications: list[VerificationRecord] = []
        self.results: dict[str, OptimizationResult] = {}
        self.screens: dict[str, VerificationReport] = {}
        self.labels: dict[str, IterationRecord] = {}
        self.evidence: dict[str, tuple] = {}
        self.excluded: set[str] = set()
        self.capabilities: tuple[SolverCapability, ...] = ()
        self.resolvers = None if isinstance(tools, ToolView) else tools
        self.tool_log = tools.log if isinstance(tools, ToolView) else None
        self.screen_cost = 1
        self.tokens = 0
        self.violations = 0
        self.planning_cap = limits.max_evaluations - int(limits.max_evaluations * VERIFICATION_SHARE)

    # ------------------------------------------------------------------------------------------------- bookkeeping
    def usage(self) -> BudgetUse:
        return BudgetUse(len(self.records), self.meter.count, self.tokens, time.perf_counter() - self.started)

    def wall_left(self) -> float:
        return max(self.limits.max_wall_s - (time.perf_counter() - self.started), 1e-3)

    def choosable(self) -> tuple[SolverCapability, ...]:
        return tuple(c for c in self.capabilities if c.solver_id not in self.excluded
                     and (self.policy.allowed_solvers is None or c.solver_id in self.policy.allowed_solvers))

    def incumbent(self, *, skip: frozenset[str] = frozenset(),
                  verifiable: bool = False) -> tuple[str, OptimizationResult] | None:
        """The best result the guard found feasible and honest: objectives in their senses, then converged first,
        then earliest. ``verifiable`` keeps only results whose screen passed every required status that the
        screen can grade (all but ``independently_verified`` and ``optimality_certified``)."""
        signs = [-1.0 if o.sense == "maximize" else 1.0 for o in self.asked.objectives]
        graded = [s for s in self.asked.verification.required if s not in SCREEN_UNGRADED]
        eligible = [(label, r) for label, r in self.results.items() if label not in skip
                    and feasible(r, self.screens.get(label))
                    and (not verifiable or all(self.screens[label].statuses[s].outcome is Outcome.PASSED
                                               for s in graded))]
        if not eligible:
            return None

        def rank(item):
            label, result = item
            converged = self.screens[label].statuses["solver_converged"].outcome is Outcome.PASSED
            return [s * result.objective_values.get(o.name, float("inf"))
                    for s, o in zip(signs, self.asked.objectives)] + [not converged]

        return min(eligible, key=rank)

    def cancelled(self, reason: CancelReason) -> bool:
        """Whether the cancel check fires here; the point where it fired goes into the trace."""
        if not self.cancel_check(len(self.records), reason):
            return False
        self.cancel_point = CancelPoint(len(self.records), reason)
        return True

    # ------------------------------------------------------------------------------------------------- the run
    def execute(self) -> RunOutcome:
        refusal = self.admit()
        if refusal:
            return self.finish(StopReason.REFUSED, refusal)
        stop = self.iterate()
        if stop not in (StopReason.CANCELLED, StopReason.POLICY_VIOLATION, StopReason.MAX_WALL_S):
            if self.cancelled(CancelReason.BEFORE_VERIFICATION):
                stop = StopReason.CANCELLED
            else:
                self.verify_best()
        return self.finish(stop, ())

    def admit(self) -> tuple[str, ...]:
        """Refuse a problem the run cannot take faithfully, before anything is evaluated."""
        errors = validate_problem(self.asked)
        if errors:
            return tuple(f"{e.path}: [{e.code.value}] {e.message}" for e in errors)
        called = sorted({item.expression.tool_id for item in (*self.asked.objectives, *self.asked.constraints)
                         if isinstance(item.expression, ToolCall)})
        unknown = [t for t in called if t not in known_tool_ids(self.tools)]
        if unknown:
            return tuple(f"tool {t!r} is not among the known tool ids" for t in unknown)
        if isinstance(self.tools, ToolView):
            try:
                self.resolvers = bind_problem(self.asked, self.tools, Caller("agent-runtime"))
            except ToolError as exc:
                return (f"tool binding failed: [{exc.code.value}] {exc}",)
        self.structure = classify(self.asked)
        self.capabilities = tuple(s.capability for s in self.registry.compatible(self.asked))
        if not self.choosable():
            return (f"no available solver the policy allows can take this {self.structure.problem_class.value} "
                    "problem",)
        self.screen_cost = screen_cost(self.asked)
        return ()

    def iterate(self) -> StopReason:
        while True:
            if self.cancelled(CancelReason.BEFORE_ITERATION):
                return StopReason.CANCELLED
            usage = self.usage()
            reason = usage.exhausted(self.limits)
            if reason is not None:
                return reason
            spendable = self.planning_cap - self.meter.count - self.screen_cost
            if spendable < MIN_DECISION_EVALUATIONS:
                return StopReason.MAX_EVALUATIONS
            state = PlannerState(
                self.asked, self.structure, self.choosable(), tuple(self.records),
                self.incumbent_record(), BudgetUse(self.limits.max_iterations - usage.iterations, spendable,
                                                   self.limits.max_tokens - usage.tokens,
                                                   self.limits.max_wall_s - usage.wall_s), self.seed)
            started = time.perf_counter()
            try:
                proposal = self.planner.propose(state)
            except PlannerUnavailable:
                return StopReason.PLANNER_FAILED
            self.tokens += proposal.tokens
            stop = self.step(proposal, state, spendable, started)
            if stop is not None:
                return stop

    def incumbent_record(self) -> IterationRecord | None:
        best = self.incumbent()
        return None if best is None else self.labels[best[0]]

    def step(self, proposal: Proposal, state: PlannerState, spendable: int, started: float) -> StopReason | None:
        index = len(self.records) + 1
        if proposal.error is not None:
            decision = (Violation(ViolationCode.MALFORMED, "", proposal.error),)
        else:
            decision = validate_proposal(proposal.raw, problem=self.asked,
                                         solvers={c.solver_id: c for c in state.solvers},
                                         known_solvers=frozenset(c.solver_id for c in self.registry.capabilities()),
                                         policy=self.policy, available_evaluations=spendable,
                                         default_seed=self.seed)
        if isinstance(decision, tuple):
            self.records.append(IterationRecord(index, RecordKind.REJECTED, proposal, violations=decision,
                                                tokens=proposal.tokens, wall_s=time.perf_counter() - started))
            self.violations += 1
            if self.policy.on_violation == "stop" or self.violations >= self.policy.max_violations:
                return StopReason.POLICY_VIOLATION
            return None
        if decision.action is Action.STOP:
            self.records.append(IterationRecord(index, RecordKind.STOP, proposal, decision, tokens=proposal.tokens,
                                                wall_s=time.perf_counter() - started))
            return StopReason.PLANNER_STOP
        before = self.meter.count
        capability = next(c for c in state.solvers if c.solver_id == decision.solver_id)
        result, error = self.run_solver(capability.solver_id, decision.evaluations, seed=decision.seed,
                                        x0=decision.x0, options=decision.options,
                                        max_backend_iterations=decision.max_iterations)
        screen = None if result is None else self.screen(result)
        diagnosis = diagnose(result, screen, capability, self.structure, error)
        if diagnosis.kind in (FailureKind.REJECTED_BY_GUARD, FailureKind.SOLVER_ERROR):
            self.excluded.add(capability.solver_id)
        record = IterationRecord(index, RecordKind.SOLVE, proposal, decision, result=result, screen=screen,
                                 diagnosis=diagnosis, evaluations=self.meter.count - before, tokens=proposal.tokens,
                                 wall_s=time.perf_counter() - started)
        self.records.append(record)
        if result is not None:
            label = f"iteration-{index:04d}"
            self.results[label], self.screens[label], self.labels[label] = result, screen, record
        return None

    def run_solver(self, solver_id: str, allowance: int, *, seed: int, x0=None, options=None,
                   max_backend_iterations=None) -> tuple[OptimizationResult | None, str]:
        """One solve on the metered problem, with the meter's allowance set to this solve's share."""
        self.meter.allowance = self.meter.count + allowance
        budget = Budget(max_evaluations=allowance, max_wall_s=self.wall_left())
        try:
            result = self.registry.get(solver_id).solve(self.problem, budget=budget, seed=seed, x0=x0,
                                                        options=dict(options or {}) or None, tools=self.resolvers,
                                                        max_backend_iterations=max_backend_iterations)
        except Exception as exc:  # noqa: BLE001 - a solver is third-party code; its crash is a failed iteration
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            self.meter.allowance = self.limits.max_evaluations
        if not isinstance(result, OptimizationResult):
            return None, f"the solver returned {type(result).__name__}, not an OptimizationResult"
        return result, ""

    def screen(self, result: OptimizationResult) -> VerificationReport:
        """The guard's grade of a candidate without independent evidence: feasibility and honesty."""
        return verify_result(self.problem, result, solved_problem=self.problem, tools=self.resolvers, seed=self.seed)

    # ------------------------------------------------------------------------------------------------- verification
    def verify_best(self) -> None:
        tried: set[str] = set()
        for attempt in range(1, MAX_VERIFICATION_ATTEMPTS + 1):
            best = self.incumbent(skip=frozenset(tried), verifiable=True)
            if best is None:
                return
            label, result = best
            tried.add(label)
            remaining = self.limits.max_evaluations - self.meter.count
            if remaining < self.screen_cost:
                return
            before, started = self.meter.count, time.perf_counter()
            runs: list[AlternateRun] = []
            evidence = self.alternates(result, remaining, attempt, runs)
            report = verify_result(self.problem, result, evidence=evidence, solved_problem=self.problem,
                                   tools=self.resolvers, seed=self.seed)
            for run in runs:
                if run.result is not None:
                    alt_label = f"verify-{attempt}-{run.solver_id}"
                    self.results[alt_label] = run.result
                    self.screens[alt_label] = self.screen(run.result) if \
                        self.limits.max_evaluations - self.meter.count >= self.screen_cost else None
            self.verifications.append(VerificationRecord(attempt, label, tuple(runs), report,
                                                         self.meter.count - before, time.perf_counter() - started))
            self.evidence[label] = tuple(evidence)
            if report.verdict is Outcome.PASSED:
                return
            following = self.incumbent(skip=frozenset(tried), verifiable=True)
            if following is None or not following[0].startswith(f"verify-{attempt}-"):
                return  # only an alternate's better point is worth another attempt

    def alternates(self, result: OptimizationResult, remaining: int, attempt: int,
                   runs: list[AlternateRun]) -> list[AlternateSolverCheck]:
        """Alternate solvers for the guard to run, each with an equal share of what verification may spend.

        They come from every compatible solver except the candidate's own and
        those the run excluded for a dishonest report or a crash. The policy's
        solver list does not apply: it limits the planner, and choosing the
        verification is not the planner's job.
        """
        policy = self.asked.verification
        if "independently_verified" not in policy.required or "alternate_solver" not in policy.independent_methods:
            return []
        others = [c.solver_id for c in self.capabilities
                  if c.solver_id != result.solver_id and c.solver_id not in self.excluded][:MAX_ALTERNATES]
        if not others:
            return []
        share = (remaining - self.screen_cost * (1 + len(others))) // len(others)
        if share < MIN_ALTERNATE_EVALUATIONS:
            return []
        digest, rdigest = self.asked.digest(), result_digest(result)
        return [AlternateSolverCheck(id=f"alternate:{sid}", problem_digest=digest, result_digest=rdigest,
                                     solver_id=sid, solver=self.alternate(sid, share, self.seed + attempt + k, runs))
                for k, sid in enumerate(others)]

    def alternate(self, solver_id: str, share: int, seed: int, runs: list[AlternateRun]):
        capability = next(c for c in self.capabilities if c.solver_id == solver_id)
        options = default_options(capability, share, len(self.asked.variables))

        def run(problem: OptimizationProblem) -> OptimizationResult | None:
            result, _ = self.run_solver(solver_id, share, seed=seed, options=options)
            runs.append(AlternateRun(solver_id, share, seed, options, result))
            return result

        return run

    # ------------------------------------------------------------------------------------------------- outcome
    def finish(self, stop: StopReason, refusal: tuple[str, ...]) -> RunOutcome:
        passed = [v.candidate for v in self.verifications if v.report.verdict is Outcome.PASSED]
        best = (passed[-1], self.results[passed[-1]]) if passed else self.incumbent()
        report = None
        if best is not None:
            final = [v for v in self.verifications if v.candidate == best[0]]
            report = final[-1].report if final else self.screens[best[0]]
        else:
            best = self.best_effort()
            if best is not None:
                report = self.screens.get(best[0])
        status = _status(stop, bool(passed), best is not None and feasible(best[1], self.screens.get(best[0])))
        trace = Trace(self.asked.id, self.asked.digest(), self.asked.hard_constraint_fingerprint(), self.planner.name,
                      self.seed, self.limits, self.policy,
                      tuple(c.solver_id for c in self.capabilities), tuple(self.records),
                      tuple(self.verifications), status, stop, None if best is None else best[0], self.usage(),
                      refusal, self.cancel_point, getattr(self.planner, "provider", None),
                      getattr(self.planner, "model", None))
        return RunOutcome(status, stop, None if best is None else best[1], report,
                          self.evidence.get(best[0], ()) if best else (), trace, self.asked, dict(self.results),
                          self.tool_log)

    def best_effort(self) -> tuple[str, OptimizationResult] | None:
        """With nothing feasible: the honest result with a point and the least total violation, if any."""
        honest = [(label, r) for label, r in self.results.items() if r.x and self.screens.get(label) is not None
                  and not distrust(r, self.screens[label])]
        if not honest:
            return None
        by_name = {c.name: c for c in self.asked.constraints}

        def violation(result: OptimizationResult) -> float:
            return sum(max(by_name[n].violation(r), 0.0) / by_name[n].allowance(1.0)
                       for n, r in result.constraint_residuals.items() if n in by_name)

        return min(honest, key=lambda item: violation(item[1]))


def _status(stop: StopReason, verified: bool, feasible_best: bool) -> RunStatus:
    if stop is StopReason.REFUSED:
        return RunStatus.REFUSED
    if stop is StopReason.CANCELLED:
        return RunStatus.CANCELLED
    if stop is StopReason.POLICY_VIOLATION:
        return RunStatus.POLICY_VIOLATION
    if verified:
        return RunStatus.VERIFIED
    if stop.budget:
        return RunStatus.BUDGET_EXHAUSTED
    return RunStatus.UNVERIFIED_BEST if feasible_best else RunStatus.NO_FEASIBLE


