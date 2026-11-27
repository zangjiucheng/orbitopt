"""What every solver adapter shares: capabilities, the evaluation budget, and honest result mapping.

An adapter never trusts a backend with the budget. Every point the backend
asks about goes through one ``Evaluator``, which counts it, refuses the call
that would exceed ``max_evaluations`` or ``max_wall_s`` by raising
``BudgetExhausted`` before evaluating, and remembers the best point so far
among those the IR evaluated without error.
A maximised objective is negated here and nowhere else. The result always
reports the IR's own evaluation of the chosen point, never the backend's
numbers, and ``constraints_satisfied`` comes only from
``OptimizationProblem.constraints_satisfied`` at that point. When every
evaluation failed there is no point: ``x`` is empty and
``constraints_satisfied`` is false, even for a problem with no constraints.

Termination is decided in this order: an error (a backend exception, a
backend that reports failure, or every evaluation failing); a backend's
proof that the problem is unbounded; the adapter's budget; no feasible point found
(``INFEASIBLE``, which for a local or heuristic backend means "stopped
without one", not a proof); the backend's own convergence flag
(``CONVERGED``, and only then ``solver_converged``); the backend's own
iteration limit or stall (``STALLED``); otherwise ``ERROR`` carrying the
backend's message. A multi-objective solve reports its whole front in
``OptimizationResult.front``.

Malformed requests raise: an invalid problem raises ``InvalidProblemError``,
an unknown option or ``x0`` name raises ``ValueError``. A well-formed request
the solver cannot carry out returns ``Termination.ERROR`` with the reason:
the backend is not installed, or the problem's structure is outside the
solver's capability.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from ..classify import Form, ProblemStructure, classify
from ..errors import InvalidProblemError
from ..expressions import EvaluationError
from ..problem import Budget, OptimizationProblem, Variable, validate_problem
from ..result import Evidence, OptimizationResult, ParetoPoint, Termination

Vector = Sequence[float]
_BUDGET_STOPS = (Termination.MAX_EVALUATIONS, Termination.TIMEOUT, Termination.CANCELLED)
_DEFINITE_STOPS = (Termination.ERROR, Termination.UNBOUNDED, *_BUDGET_STOPS)
"""Stops reported as they are, whether or not a feasible point was found."""

Convergence = Literal["certificate", "first_order", "heuristic", "none"]
CONVERGENCE_MEANINGS: dict[str, str] = {
    "certificate": "the backend proved optimality within its gap tolerance (LP duality, MILP bound)",
    "first_order": "a first-order stationary (KKT) point; the global optimum only when the problem is convex",
    "heuristic": "the search stagnated; no optimality claim",
    "none": "the solver has no convergence test and never reports converged",
}
"""What ``solver_converged`` is worth for each ``SolverCapability.convergence``."""


@dataclass(frozen=True)
class SolverCapability:
    """What a solver can take, declared by the solver and matched against ``classify``.

    ``convergence`` says what the backend's convergence flag, and so
    ``solver_converged``, means (see ``CONVERGENCE_MEANINGS``).
    ``max_form`` is the most general expression form it accepts.
    ``multiobjective`` is exclusive: a multi-objective solver takes only
    problems with several objectives, any other solver only problems with one.
    ``honors`` lists the ``Budget`` fields the backend is configured from;
    ``max_evaluations`` and ``max_wall_s`` are enforced by the adapter for
    every solver regardless. ``options`` lists the option names ``solve``
    accepts. ``backend`` is the module the solver imports; when it cannot be
    found, ``unavailable_reason`` says so.
    """

    solver_id: str
    backend: str
    max_form: Form
    convergence: Convergence
    needs_gradients: bool = False
    supports_integers: bool = False
    supports_constraints: bool = False
    supports_equality: bool = False
    multiobjective: bool = False
    needs_finite_bounds: bool = False
    is_global: bool = False
    deterministic_given_seed: bool = True
    honors: frozenset[str] = frozenset({"max_evaluations", "max_wall_s"})
    options: frozenset[str] = frozenset()
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def rejection(self, structure: ProblemStructure) -> str | None:
        """Why this solver cannot take a problem of this structure; ``None`` when it can."""
        checks = (
            (structure.form > self.max_form,
             f"its expressions are {structure.form.name.lower()}, beyond {self.max_form.name.lower()}"),
            (structure.multiobjective != self.multiobjective,
             "it has one objective" if self.multiobjective else "it has several objectives"),
            (structure.integer and not self.supports_integers, "it has integer, binary or discrete variables"),
            (structure.constrained and not self.supports_constraints, "it has constraints besides variable bounds"),
            (structure.equality and not self.supports_equality, "it has equality constraints"),
            (self.needs_finite_bounds and not structure.bounded, "a variable has an open bound"),
        )
        reasons = [why for failed, why in checks if failed]
        return f"{self.solver_id} cannot take this {structure.problem_class.value} problem: " + "; ".join(reasons) \
            if reasons else None


class Solver(Protocol):
    """One solver behind the shared interface; see ``AdapterSolver`` for the contract every built-in keeps."""

    @property
    def capability(self) -> SolverCapability: ...

    def solve(self, problem: OptimizationProblem, *, budget: Budget | None = None, seed: int = 0,
              x0: Mapping[str, float] | None = None, options: Mapping[str, Any] | None = None,
              tools: Mapping[str, Callable[..., Any]] | None = None) -> OptimizationResult: ...


class BudgetExhausted(Exception):
    """Raised by the ``Evaluator`` instead of an evaluation the budget does not allow."""

    def __init__(self, termination: Termination, message: str):
        super().__init__(message)
        self.termination = termination


@dataclass(frozen=True)
class Space:
    """The search vector a backend sees: one coordinate per variable, in declaration order.

    A continuous or integer variable is its own value; a binary one is 0 or
    1; a discrete one is the index of its choice. ``integral`` marks the
    coordinates a backend must keep integer.
    """

    variables: tuple[Variable, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    integral: tuple[bool, ...]

    @classmethod
    def of(cls, problem: OptimizationProblem) -> "Space":
        def bounds(v: Variable) -> tuple[float, float]:
            if v.kind == "binary":
                return 0.0, 1.0
            if v.kind == "discrete":
                return 0.0, float(len(v.choices) - 1)
            return (-math.inf if v.lower is None else float(v.lower), math.inf if v.upper is None else float(v.upper))

        pairs = [bounds(v) for v in problem.variables]
        return cls(problem.variables, tuple(lo for lo, _ in pairs), tuple(hi for _, hi in pairs),
                   tuple(v.kind != "continuous" for v in problem.variables))

    def decode(self, vector: Vector) -> dict[str, float]:
        """The point ``x`` in each variable's unit; integral coordinates are rounded."""
        x = {}
        for v, value in zip(self.variables, vector):
            value = float(value)
            if v.kind == "discrete":
                x[v.name] = float(v.choices[min(max(round(value), 0), len(v.choices) - 1)])
            else:
                x[v.name] = float(round(value)) if v.kind != "continuous" else value
        return x

    def start(self, x0: Mapping[str, float] | None) -> list[float]:
        """A starting vector: ``x0``, else each ``initial``, else the bound midpoint, a finite bound, or 0."""
        x0 = dict(x0 or {})
        unknown = sorted(set(x0) - {v.name for v in self.variables})
        if unknown:
            raise ValueError(f"x0 names unknown variables {unknown}")
        out = []
        for v, lo, hi in zip(self.variables, self.lower, self.upper):
            value = x0.get(v.name, v.initial)
            if value is not None and v.kind == "discrete":
                value = min(range(len(v.choices)), key=lambda i: abs(v.choices[i] - value))
            elif value is None:
                value = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else (
                    lo if math.isfinite(lo) else hi if math.isfinite(hi) else 0.0)
            out.append(float(value))
        return out


@dataclass(frozen=True)
class Point:
    """One evaluated point: the IR's values plus the minimisation-sense vectors a backend needs.

    ``f`` holds each objective negated when maximised; ``g`` each canonical
    residual in constraint order. Both are ``inf`` where evaluation failed,
    and ``failed`` marks such a point, which never becomes the best point.
    """

    x: dict[str, float]
    objectives: dict[str, float]
    residuals: dict[str, float]
    f: tuple[float, ...]
    g: tuple[float, ...]
    feasible: bool
    violation: float
    failed: bool = False


class Evaluator:
    """The only path from a backend to the problem: counts, enforces the budget, tracks the best point.

    Recently evaluated points are cached (all of them when ``archive``), so a
    backend that asks for the objective and the constraints at one point pays
    for one evaluation, as the E14-10 budget defines it.
    """

    def __init__(self, problem: OptimizationProblem, space: Space, budget: Budget, started: float,
                 tools: Mapping[str, Callable[..., Any]] | None, archive: bool):
        self.problem, self.space, self.tools = problem, space, tools
        self.max_evaluations = budget.max_evaluations
        self.deadline = None if budget.max_wall_s is None else started + budget.max_wall_s
        self.count = 0
        self.failures = 0
        self.last_failure = ""
        self.best: Point | None = None
        self._best_key: tuple | None = None
        self._cache: OrderedDict[tuple[float, ...], Point] = OrderedDict()
        self._cache_size = None if archive else 4 * (len(space.variables) + 2)
        self._signs = tuple(-1.0 if o.sense == "maximize" else 1.0 for o in problem.objectives)

    def __call__(self, vector: Vector) -> Point:
        key = tuple(float(v) for v in vector)
        if key in self._cache:
            return self._cache[key]
        if self.max_evaluations is not None and self.count >= self.max_evaluations:
            raise BudgetExhausted(Termination.MAX_EVALUATIONS,
                                  f"evaluation budget of {self.max_evaluations} exhausted")
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise BudgetExhausted(Termination.TIMEOUT, "wall-clock budget exhausted")
        self.count += 1
        point = self._evaluate(key)
        self._cache[key] = point
        if self._cache_size is not None and len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        rank = (0, *point.f) if point.feasible else (1, point.violation)
        if not point.failed and (self._best_key is None or rank < self._best_key):
            self.best, self._best_key = point, rank
        return point

    def objective(self, vector: Vector) -> float:
        """The single objective in minimisation sense."""
        return self(vector).f[0]

    def residuals(self, vector: Vector) -> list[float]:
        return list(self(vector).g)

    def _evaluate(self, key: tuple[float, ...]) -> Point:
        problem = self.problem
        x = self.space.decode(key)
        try:
            ev = problem.evaluate(x, self.tools)
        except EvaluationError as exc:  # undefined here, or a tool failed (ToolCallError): the point is infeasible
            self.failures += 1
            self.last_failure = str(exc)
            inf = math.inf
            return Point(x, {}, {}, (inf,) * len(problem.objectives), (inf,) * len(problem.constraints), False, inf,
                         failed=True)
        objectives = {k: float(v) for k, v in ev.objectives.items()}
        residuals = {k: float(v) for k, v in ev.residuals.items()}
        f = tuple(_finite(s * objectives[o.name]) for s, o in zip(self._signs, problem.objectives))
        g = tuple(_finite(residuals[c.name]) for c in problem.constraints)
        violation = sum(max(c.violation(r), 0.0) / c.allowance(1.0)
                        for c, r in zip(problem.constraints, g)) if all(map(math.isfinite, g)) else math.inf
        return Point(x, objectives, residuals, f, g, problem.constraints_satisfied(x, residuals), violation)


def _finite(value: float) -> float:
    return value if math.isfinite(value) else math.inf


@dataclass(frozen=True)
class Run:
    """Everything an adapter's ``_run`` needs for one solve."""

    problem: OptimizationProblem
    structure: ProblemStructure
    space: Space
    evaluator: Evaluator
    budget: Budget
    seed: int
    start: list[float]
    x0_given: bool
    options: Mapping[str, Any]

    def remaining_wall_s(self) -> float | None:
        if self.evaluator.deadline is None:
            return None
        left = self.evaluator.deadline - time.perf_counter()
        if left <= 0:
            raise BudgetExhausted(Termination.TIMEOUT, "wall-clock budget exhausted")
        return left


@dataclass(frozen=True)
class Outcome:
    """What a backend reported: its own convergence flag, why it stopped otherwise, and points to report.

    ``points`` are search vectors the result is built from instead of the
    evaluator's best point: the optimum of a backend that does not evaluate
    the problem itself (HiGHS), or a multi-objective front.
    """

    converged: bool = False
    stop: Termination | None = None
    message: str = ""
    iterations: int | None = None
    points: tuple[tuple[float, ...], ...] = ()


def require_valid(problem: OptimizationProblem) -> None:
    errors = validate_problem(problem)
    if errors:
        raise InvalidProblemError(errors)


def backend_missing(module: str) -> str | None:
    """Why ``module`` cannot be imported, without importing it; ``None`` when it can be found."""
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    return None if found else f"backend module {module!r} is not installed"


class AdapterSolver:
    """Base of the built-in solvers: ``solve`` is the shared contract, ``_run`` drives one backend."""

    CAPABILITY: SolverCapability

    @property
    def capability(self) -> SolverCapability:
        """The declared capability, with availability probed now."""
        return replace(self.CAPABILITY, unavailable_reason=backend_missing(self.CAPABILITY.backend))

    def solve(self, problem: OptimizationProblem, *, budget: Budget | None = None, seed: int = 0,
              x0: Mapping[str, float] | None = None, options: Mapping[str, Any] | None = None,
              tools: Mapping[str, Callable[..., Any]] | None = None) -> OptimizationResult:
        """Solve ``problem`` within ``budget`` (default ``problem.budget``) and report an ``OptimizationResult``.

        ``seed`` drives every random choice, so a deterministic backend gives
        the identical result for the same problem, seed and budget. ``x0``
        maps variable names to a starting point. ``tools`` maps ``ToolCall``
        ids to callables, as ``OptimizationProblem.evaluate`` takes them.
        """
        started = time.perf_counter()
        require_valid(problem)
        budget = problem.budget if budget is None else budget
        options = dict(options or {})
        capability = self.capability
        unknown = sorted(set(options) - capability.options)
        if unknown:
            raise ValueError(f"{capability.solver_id} takes options {sorted(capability.options)}, not {unknown}")
        space = Space.of(problem)
        start = space.start(x0)
        structure = classify(problem)
        evaluator = Evaluator(problem, space, budget, started, tools, archive=capability.multiobjective)
        refusal = capability.unavailable_reason or capability.rejection(structure)
        if refusal:
            outcome = Outcome(stop=Termination.ERROR, message=refusal)
        else:
            run = Run(problem, structure, space, evaluator, budget, seed, start, x0 is not None, options)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outcome = self._run(run)
            except BudgetExhausted as stop:
                outcome = Outcome(stop=stop.termination, message=str(stop))
            except Exception as exc:  # noqa: BLE001 - any backend failure becomes an honest ERROR result
                outcome = Outcome(stop=Termination.ERROR, message=f"{type(exc).__name__}: {exc}")
        return _result(problem, capability, structure, evaluator, outcome, budget, seed, options, started)

    def _run(self, run: Run) -> Outcome:
        raise NotImplementedError


def _non_dominated(points: list[Point]) -> list[Point]:
    return [p for p in points
            if not any(all(a <= b for a, b in zip(q.f, p.f)) and q.f != p.f for q in points)]


def _result(problem: OptimizationProblem, capability: SolverCapability, structure: ProblemStructure,
            evaluator: Evaluator, outcome: Outcome, budget: Budget, seed: int, options: Mapping[str, Any],
            started: float) -> OptimizationResult:
    stop, message = outcome.stop, outcome.message
    try:
        reported = [evaluator(p) for p in outcome.points]
    except BudgetExhausted as exhausted:
        reported, stop, message = [], exhausted.termination, f"{message}; {exhausted}"
    if evaluator.count and evaluator.failures == evaluator.count:
        stop = Termination.ERROR
        message = f"all {evaluator.count} evaluations failed, the last with: {evaluator.last_failure}" + (
            f" ({message})" if message else "")
    front: list[Point] = []
    if capability.multiobjective:
        front = sorted(_non_dominated([p for p in reported if p.feasible]), key=lambda p: p.f)
        chosen = front[0] if front else None
    else:
        chosen = evaluator.best
    feasible = chosen is not None and chosen.feasible
    if stop in _DEFINITE_STOPS:
        termination = stop
    elif not feasible:
        termination = Termination.INFEASIBLE
        message = f"no feasible point found ({message})" if message else "no feasible point found"
    elif outcome.converged:
        termination = Termination.CONVERGED
    elif stop is not None:
        termination = stop
    else:
        termination = Termination.ERROR
    x = chosen.x if chosen else {}
    residuals = chosen.residuals if chosen else {}
    data: dict[str, Any] = {
        "backend": capability.backend,
        "backend_version": str(getattr(sys.modules.get(capability.backend), "__version__", "not imported")),
        "problem_class": structure.problem_class.value,
        "seed": seed,
        "budget": {k: v for k, v in asdict(budget).items() if v is not None},
        "options": dict(options),
    }
    digest = problem.digest()
    return OptimizationResult(
        problem_digest=digest,
        solver_id=capability.solver_id,
        termination=termination,
        solver_converged=termination == Termination.CONVERGED,
        constraints_satisfied=feasible,
        x=_finite_values(x),
        objective_values=_finite_values(chosen.objectives if chosen else {}),
        constraint_residuals=_finite_values(residuals),
        evaluations=evaluator.count,
        iterations=outcome.iterations,
        wall_time_s=time.perf_counter() - started,
        message=message,
        evidence=(Evidence("solver", f"{capability.solver_id} stopped: {termination.value}", data,
                           producer=capability.solver_id, problem_digest=digest),),
        front=tuple(ParetoPoint(_finite_values(p.x), _finite_values(p.objectives)) for p in front),
    )


def _finite_values(values: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if math.isfinite(v)}
