"""What every solver adapter shares: capabilities, the evaluation budget, and honest result mapping.

An adapter never trusts a backend with the budget. Every point the backend
asks about goes through one ``Evaluator``, which counts it, refuses the call
that would exceed ``max_evaluations`` or ``max_wall_s`` by raising
``BudgetExhausted`` before evaluating, and remembers the best point so far
among those the IR evaluated without error: clean feasible points first
(``SELECTION_FRACTION``), then the rest of the tolerance band, each by
objective. Before it reports, the adapter restores its best band point onto
the constraints (``Evaluator.restore``) with evaluations it kept back for that.
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
an unknown option, an option value outside its ``OptionSpec`` or an
unknown ``x0`` name raises ``ValueError``. A well-formed request
the solver cannot carry out returns ``Termination.ERROR`` with the reason:
the backend is not installed, or the problem's structure is outside the
solver's capability.
"""

from __future__ import annotations

import importlib.util
import math
import numbers
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from ..classify import Form, ProblemStructure, classify
from ..errors import InvalidProblemError
from ..expressions import EvaluationError, ToolCallError
from ..problem import Budget, OptimizationProblem, Variable, validate_problem
from ..result import Evidence, OptimizationResult, ParetoPoint, Termination

Vector = Sequence[float]
SELECTION_FRACTION = 1e-6
"""A feasible point is *clean* when no constraint uses more than this fraction of its tolerance band.

The verification tolerance (``constraint_tol``) decides what is feasible. It
does not make two feasible points comparable by objective: a point that uses
the band can beat the exact optimum by up to ``lambda * allowance``, where
``lambda`` is the constraint's Lagrange multiplier. A clean point can gain at
most ``lambda * SELECTION_FRACTION * allowance``, a millionth of that. The best
point ranks clean points before band points, and the band point is restored
onto the constraints before it competes (``RESTORE_STEPS``)."""


def band_fraction(problem: OptimizationProblem, residuals: Mapping[str, float]) -> float:
    """The largest fraction of a constraint's tolerance band ``residuals`` use: 0 when every constraint holds,
    above 1 when one is violated, ``inf`` when a residual is missing or not finite."""
    tol = problem.verification.constraint_tol
    worst = 0.0
    for c in problem.constraints:
        r = residuals.get(c.name)
        if not isinstance(r, (int, float)) or not math.isfinite(r):
            return math.inf
        worst = max(worst, max(c.violation(r), 0.0) / c.allowance(tol))
    return worst


RESTORE_STEPS = 3
"""Gauss-Newton steps that move the best band point onto its active constraints, each ``n + 1`` evaluations."""
_BUDGET_STOPS = (Termination.MAX_EVALUATIONS, Termination.TIMEOUT, Termination.CANCELLED)
_DEFINITE_STOPS = (Termination.ERROR, Termination.UNBOUNDED, *_BUDGET_STOPS)
"""Stops reported as they are, whether or not a feasible point was found."""

Convergence = Literal["certificate", "exhaustive", "first_order", "heuristic", "none"]
CONVERGENCE_MEANINGS: dict[str, str] = {
    "certificate": "the backend proved optimality within its gap tolerance (LP duality, MILP bound)",
    "exhaustive": "every point of the finite domain was evaluated: the global optimum within the constraint tolerance",
    "first_order": "a first-order stationary (KKT) point; the global optimum only when the problem is convex",
    "heuristic": "the search stagnated; no optimality claim",
    "none": "the solver has no convergence test and never reports converged",
}
"""What ``solver_converged`` is worth for each ``SolverCapability.convergence``."""


@dataclass(frozen=True)
class OptionSpec:
    """One solver option: its type, its closed range (``choices`` for a string), and whether it is a tolerance.

    ``tolerance`` marks a convergence tolerance of the solver, never a
    problem tolerance; it is the one kind of option a remedy may loosen.
    ``default`` is the value the backend runs with when the option is not
    given; a tolerance must state it, because loosening starts from it.
    """

    kind: type
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    tolerance: bool = False
    default: float | None = None

    def __post_init__(self):
        if self.kind not in (bool, int, float, str):
            raise ValueError(f"an option is bool, int, float or str, not {self.kind!r}")
        if self.kind in (int, float) and (self.low is None or self.high is None or self.low > self.high):
            raise ValueError("a numeric option needs a closed range low <= high")
        if self.kind is str and not self.choices:
            raise ValueError("a string option needs its choices")
        if self.tolerance and (self.kind is not float or self.default is None):
            raise ValueError("a tolerance is a float option with the default the backend uses")

    def errors(self, value: Any) -> str | None:
        """Why ``value`` is not a valid setting; ``None`` when it is."""
        if self.kind is bool:
            return None if isinstance(value, bool) else "must be true or false"
        if self.kind is str:
            return None if value in self.choices else f"must be one of {list(self.choices)}"
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            return "must be a finite number"
        if self.kind is int and value != int(value):
            return "must be an integer"
        if not (self.low <= value <= self.high):
            return f"must lie in [{self.low:g}, {self.high:g}]"
        return None


@dataclass(frozen=True)
class SolverCapability:
    """What a solver can take, declared by the solver and matched against ``classify``.

    ``convergence`` says what the backend's convergence flag, and so
    ``solver_converged``, means (see ``CONVERGENCE_MEANINGS``).
    ``max_form`` is the most general expression form it accepts.
    ``multiobjective`` is exclusive: a multi-objective solver takes only
    problems with several objectives, any other solver only problems with one.
    ``Budget.max_evaluations`` and ``Budget.max_wall_s`` are enforced by the
    adapter for every solver; ``Budget.max_iterations`` is the agent
    runtime's loop cap and no solver reads it. ``iteration_cap`` says the
    backend has its own iteration limit, which ``solve``'s
    ``max_backend_iterations`` sets (generations for NSGA-II). ``options`` maps each option name ``solve``
    accepts to its ``OptionSpec``. ``backend`` is the module the solver
    imports; when it cannot be found, ``unavailable_reason`` says so.
    ``max_domain_size`` is set by a solver that walks the whole domain: it
    takes a problem only when every variable has finitely many values and
    their product has at most this many points.
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
    iteration_cap: bool = False
    options: Mapping[str, OptionSpec] = field(default_factory=dict)
    max_domain_size: int | None = None
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
            (self.max_domain_size is not None and structure.domain_size is None,
             "a variable is continuous or has an open bound, so the domain is not finite"),
            (self.max_domain_size is not None and structure.domain_size is not None
             and structure.domain_size > self.max_domain_size,
             f"its domain has {structure.domain_size} points, more than the {self.max_domain_size} it enumerates"),
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
              tools: Mapping[str, Callable[..., Any]] | None = None,
              max_backend_iterations: int | None = None) -> OptimizationResult: ...


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
    vector: tuple[float, ...] = ()
    band: float = 0.0
    """The largest fraction of a constraint's tolerance band the point uses (0 when every constraint holds)."""

    @property
    def clean(self) -> bool:
        """Feasible, and no constraint uses more than ``SELECTION_FRACTION`` of its band."""
        return self.feasible and self.band <= SELECTION_FRACTION

    def rank(self) -> tuple:
        """Clean points by objective, then band points by objective, then infeasible points by violation."""
        if self.clean:
            return (0, *self.f)
        return (1, *self.f) if self.feasible else (2, self.violation)


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
        self.tool_failures = 0
        self.last_failure = ""
        self.best: Point | None = None
        self.band_best: Point | None = None
        """The best feasible point that is not clean, by objective: the one ``restore`` moves onto the constraints."""
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
        rank = point.rank()
        if not point.failed and (self._best_key is None or rank < self._best_key):
            self.best, self._best_key = point, rank
        if point.feasible and not point.clean and (self.band_best is None or point.f < self.band_best.f):
            self.band_best = point
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
            self.tool_failures += isinstance(exc, ToolCallError)
            self.last_failure = str(exc)
            inf = math.inf
            return Point(x, {}, {}, (inf,) * len(problem.objectives), (inf,) * len(problem.constraints), False, inf,
                         failed=True, vector=key, band=inf)
        objectives = {k: float(v) for k, v in ev.objectives.items()}
        residuals = {k: float(v) for k, v in ev.residuals.items()}
        f = tuple(_finite(s * objectives[o.name]) for s, o in zip(self._signs, problem.objectives))
        g = tuple(_finite(residuals[c.name]) for c in problem.constraints)
        violation = sum(max(c.violation(r), 0.0) / c.allowance(1.0)
                        for c, r in zip(problem.constraints, g)) if all(map(math.isfinite, g)) else math.inf
        return Point(x, objectives, residuals, f, g, problem.constraints_satisfied(x, residuals), violation,
                     vector=key, band=band_fraction(problem, residuals))

    def restore(self) -> None:
        """Move the best band point onto its constraints when it beats every clean point, so it competes on what
        it costs to be feasible, not on the band.

        Each step solves ``J dx = -g`` for the least-norm ``dx`` over the
        continuous coordinates, where ``g`` holds every equality residual and
        every inequality residual past ``SELECTION_FRACTION`` of its band, and
        ``J`` comes from forward differences (one evaluation per coordinate).
        A coordinate the step would push past a bound is clipped and held
        there. Every point goes through this evaluator, so ``best`` takes the
        restored point exactly when it is clean and beats the clean best.
        Stops after ``RESTORE_STEPS`` steps, on a clean point, on a failed
        evaluation, or when the budget runs out (``BudgetExhausted``).
        """
        point = self.band_best
        if point is None or (self.best is not None and self.best.clean and self.best.f <= point.f):
            return
        import numpy as np

        space, constraints = self.space, self.problem.constraints
        free = [i for i, integral in enumerate(space.integral) if not integral]
        tol = self.problem.verification.constraint_tol
        for _ in range(RESTORE_STEPS):
            rows = [j for j, c in enumerate(constraints) if c.sense == "==" or
                    max(point.g[j], 0.0) > SELECTION_FRACTION * c.allowance(tol)]
            if point.failed or point.clean or not rows or not free:
                return
            x = list(point.vector)
            jacobian = np.zeros((len(rows), len(free)))
            for k, i in enumerate(free):
                h = math.sqrt(2.220446049250313e-16) * max(1.0, abs(x[i]))
                h = -h if x[i] + h > space.upper[i] else h
                probe = self(x[:i] + [x[i] + h] + x[i + 1:])
                if probe.failed:
                    return
                jacobian[:, k] = [(probe.g[j] - point.g[j]) / h for j in rows]
            step = np.linalg.lstsq(jacobian, -np.array([point.g[j] for j in rows]), rcond=None)[0]
            for k, i in enumerate(list(free)):
                x[i] = min(max(x[i] + float(step[k]), space.lower[i]), space.upper[i])
                if x[i] in (space.lower[i], space.upper[i]):
                    free.remove(i)
            point = self(x)


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
    max_backend_iterations: int | None = None

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
              tools: Mapping[str, Callable[..., Any]] | None = None,
              max_backend_iterations: int | None = None) -> OptimizationResult:
        """Solve ``problem`` within ``budget`` (default ``problem.budget``) and report an ``OptimizationResult``.

        ``seed`` drives every random choice, so a deterministic backend gives
        the identical result for the same problem, seed and budget. ``x0``
        maps variable names to a starting point. ``tools`` maps ``ToolCall``
        ids to callables, as ``OptimizationProblem.evaluate`` takes them.
        ``max_backend_iterations`` caps the backend's own iterations, for a
        solver whose capability has ``iteration_cap``. ``budget.max_iterations``
        is the agent runtime's loop cap, not a solver limit, and is ignored here.
        """
        started = time.perf_counter()
        require_valid(problem)
        budget = problem.budget if budget is None else budget
        options = dict(options or {})
        capability = self.capability
        unknown = sorted(set(options) - capability.options.keys())
        if unknown:
            raise ValueError(f"{capability.solver_id} takes options {sorted(capability.options)}, not {unknown}")
        invalid = [f"{name} {why}" for name, value in sorted(options.items())
                   if (why := capability.options[name].errors(value)) is not None]
        if invalid:
            raise ValueError(f"{capability.solver_id}: " + "; ".join(invalid))
        if max_backend_iterations is not None:
            if not capability.iteration_cap:
                raise ValueError(f"{capability.solver_id} has no backend iteration limit")
            if isinstance(max_backend_iterations, bool) or not isinstance(max_backend_iterations, int) \
                    or max_backend_iterations < 1:
                raise ValueError(f"max_backend_iterations must be a positive integer, not {max_backend_iterations!r}")
        space = Space.of(problem)
        start = space.start(x0)
        structure = classify(problem)
        evaluator = Evaluator(problem, space, budget, started, tools, archive=capability.multiobjective)
        restorable = not capability.multiobjective and bool(problem.constraints) and not all(space.integral)
        reserve = restoration_reserve(space, budget) if restorable else 0
        if reserve:
            evaluator.max_evaluations -= reserve
        refusal = capability.unavailable_reason or capability.rejection(structure)
        if refusal:
            outcome = Outcome(stop=Termination.ERROR, message=refusal)
        else:
            run = Run(problem, structure, space, evaluator, budget, seed, start, x0 is not None, options,
                      max_backend_iterations)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outcome = self._run(run)
            except BudgetExhausted as stop:
                outcome = Outcome(stop=stop.termination, message=str(stop))
            except Exception as exc:  # noqa: BLE001 - any backend failure becomes an honest ERROR result
                outcome = Outcome(stop=Termination.ERROR, message=f"{type(exc).__name__}: {exc}")
        evaluator.max_evaluations = budget.max_evaluations
        return _result(problem, capability, structure, evaluator, outcome, budget, seed, options, started,
                       max_backend_iterations, restorable and not refusal)

    def _run(self, run: Run) -> Outcome:
        raise NotImplementedError


def _non_dominated(points: list[Point]) -> list[Point]:
    return [p for p in points
            if not any(all(a <= b for a, b in zip(q.f, p.f)) and q.f != p.f for q in points)]


def restoration_reserve(space: Space, budget: Budget) -> int:
    """Evaluations a solve keeps back from its backend for ``Evaluator.restore``: ``RESTORE_STEPS`` steps of
    one evaluation per continuous coordinate plus the step's point. Nothing is kept from a budget under four
    times that, where the restoration takes what the backend leaves."""
    need = RESTORE_STEPS * (sum(not integral for integral in space.integral) + 1)
    return need if budget.max_evaluations is not None and budget.max_evaluations >= 4 * need else 0


def _result(problem: OptimizationProblem, capability: SolverCapability, structure: ProblemStructure,
            evaluator: Evaluator, outcome: Outcome, budget: Budget, seed: int, options: Mapping[str, Any],
            started: float, max_backend_iterations: int | None, restorable: bool = False) -> OptimizationResult:
    stop, message = outcome.stop, outcome.message
    try:
        reported = [evaluator(p) for p in outcome.points]
    except BudgetExhausted as exhausted:
        reported, stop, message = [], exhausted.termination, f"{message}; {exhausted}"
    if restorable:
        before, source = evaluator.best, evaluator.band_best
        try:
            evaluator.restore()
        except BudgetExhausted:
            pass  # the restoration takes only what is left; the best point so far stands
        if evaluator.best is not before:
            message = (f"{message}; " if message else "") + \
                f"restored onto the constraints from a point using {source.band:.3g} of a tolerance band"
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
        "budget": {k: v for k, v in (("max_evaluations", budget.max_evaluations), ("max_wall_s", budget.max_wall_s),
                                     ("max_backend_iterations", max_backend_iterations)) if v is not None},
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
