"""E14-10 instance catalogue: the problems every arm is scored on.

Each runnable instance is stated three ways, and the test suite checks that they agree:

* ``objective_exprs`` / ``Constraint.expr``: expression strings in the neutral grammar below, the form a later
  runner maps onto the Optimization IR;
* ``objectives`` / ``Constraint.fn``: Python callables written separately from the strings, used for the reference
  check and for scoring returned points;
* ``statement``: the natural-language input the LLM compiler (arms D, E) and the direct proposer (arm C) receive.

Neutral expression grammar: a Python expression over ``x`` (the decision vector, indexed from 0), ``n`` (its
dimension), the builtins ``sum range abs min max``, the math names ``sqrt exp log cos sin atan pi e``, and the
instance's ``data`` mapping. Constraints are ``g(x) <= 0`` (kind ``le``) or ``h(x) == 0`` (kind ``eq``); bounds are
separate and always hard.

Nothing here imports orbitopt: the catalogue must stay usable by the IR and agent packages without depending on them.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

ProblemClass = Literal["lp", "convex_qp", "nlp", "milp", "global", "black_box", "multi_objective"]
PROBLEM_CLASSES: tuple[str, ...] = ("lp", "convex_qp", "nlp", "milp", "global", "black_box", "multi_objective")
REFERENCE_METHODS: tuple[str, ...] = ("analytical", "brute_force", "published")

Vector = Sequence[float]


@dataclass(frozen=True)
class Constraint:
    name: str
    expr: str
    kind: Literal["le", "eq"]
    fn: Callable[[Vector], float]


@dataclass(frozen=True)
class Reference:
    """The stored optimum and how it was obtained. ``value`` is the objective for single-objective instances and the
    hypervolume of the Pareto front w.r.t. ``reference_point`` for multi-objective ones."""
    value: float
    method: Literal["analytical", "brute_force", "published"]
    derivation: str
    x: tuple[float, ...] | None = None
    indicator: Literal["objective", "hypervolume"] = "objective"
    reference_point: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Tolerance:
    """A run succeeds when its worst hard-constraint violation (bounds included) is <= ``constraint_abs`` and its gap
    to the reference (see metrics.optimality_gap) is <= ``objective_abs + objective_rel * |reference|``."""
    objective_abs: float
    objective_rel: float = 0.0
    constraint_abs: float = 1e-6


@dataclass(frozen=True)
class Instance:
    id: str
    family: str
    problem_class: ProblemClass
    dimension: int
    variable_type: Literal["continuous", "integer", "binary"]
    bounds: tuple[tuple[float, float], ...]
    sense: Literal["minimize", "maximize"]
    objective_exprs: tuple[str, ...]
    objectives: tuple[Callable[[Vector], float], ...]
    constraints: tuple[Constraint, ...]
    reference: Reference
    tolerance: Tolerance
    statement: str
    split: Literal["tuning", "scoring"]
    well_known: bool  # instance or optimum is in public literature: arm C may recite it (memorisation risk)
    data: Mapping[str, object] = field(default_factory=dict)
    blackboxes: Mapping[str, Callable] = field(default_factory=dict)  # opaque names usable in expressions
    status: Literal["runnable"] = "runnable"

    def namespace(self) -> dict[str, object]:
        """The names an expression string may use, bound for this instance."""
        ns: dict[str, object] = {name: getattr(math, name) for name in ("sqrt", "exp", "log", "cos", "sin", "atan")}
        ns.update(pi=math.pi, e=math.e, sum=sum, range=range, abs=abs, min=min, max=max, n=self.dimension)
        ns.update(self.data)
        ns.update(self.blackboxes)
        return ns


@dataclass(frozen=True)
class ExternalInstance:
    """A named instance from an external suite. Not vendored and not run by this task; the later runner adds it only
    when the optional dependency is installed. ``published_value`` is quoted from the suite's documentation and is
    NOT verified here."""
    id: str
    suite: Literal["CUTEst", "COCO/BBOB"]
    name: str
    problem_class: ProblemClass
    dimension: str
    why: str
    published_value: str
    optional_dependency: str
    status: Literal["requires_optional_dependency"] = "requires_optional_dependency"


def evaluate_expr(instance: Instance, expr: str, x: Vector) -> float:
    """Evaluate one neutral expression string at ``x`` (no builtins beyond the grammar)."""
    ns = instance.namespace()
    ns["x"] = [float(v) for v in x]
    ns["__builtins__"] = {}  # generator expressions resolve names in globals, so the namespace IS the globals
    return float(eval(expr, ns))  # noqa: S307 -- expressions are this module's own literals


# ------------------------------------------------------------------------------------------------- objective callables
def _rosenbrock(x: Vector, shift: Vector | None = None) -> float:
    z = np.asarray(x, dtype=float)
    if shift is not None:
        z = z - np.asarray(shift, dtype=float) + 1.0
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


def _rastrigin(x: Vector, shift: Vector | None = None) -> float:
    z = np.asarray(x, dtype=float)
    if shift is not None:
        z = z - np.asarray(shift, dtype=float)
    return float(10.0 * z.size + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z)))


def _ackley(x: Vector, shift: Vector | None = None) -> float:
    z = np.asarray(x, dtype=float)
    if shift is not None:
        z = z - np.asarray(shift, dtype=float)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2))) - np.exp(np.mean(np.cos(2.0 * np.pi * z)))
                 + 20.0 + np.e)


#: shift for the contamination-control variants: the optimum sits at SHIFT, not at a textbook point
SHIFT = (0.7, -1.3, 2.1, -0.4, 1.8, -2.6, 0.3, 1.1, -0.9, 2.4)

_ROSENBROCK_EXPR = "sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(n-1))"
_ROSENBROCK_SHIFT_EXPR = ("sum(100*((x[i+1]-s[i+1]+1)-(x[i]-s[i]+1)**2)**2 + (1-(x[i]-s[i]+1))**2"
                          " for i in range(n-1))")
_RASTRIGIN_EXPR = "10*n + sum(x[i]**2 - 10*cos(2*pi*x[i]) for i in range(n))"
_RASTRIGIN_SHIFT_EXPR = "10*n + sum((x[i]-s[i])**2 - 10*cos(2*pi*(x[i]-s[i])) for i in range(n))"
_ACKLEY_EXPR = ("-20*exp(-0.2*sqrt(sum(x[i]**2 for i in range(n))/n))"
                " - exp(sum(cos(2*pi*x[i]) for i in range(n))/n) + 20 + e")
_ACKLEY_SHIFT_EXPR = ("-20*exp(-0.2*sqrt(sum((x[i]-s[i])**2 for i in range(n))/n))"
                      " - exp(sum(cos(2*pi*(x[i]-s[i])) for i in range(n))/n) + 20 + e")

_CONTINUOUS_TOL = Tolerance(objective_abs=1e-6)


def _box(n: int, lo: float, hi: float) -> tuple[tuple[float, float], ...]:
    return tuple((lo, hi) for _ in range(n))


# ----------------------------------------------------------------------------------------------------------- families
def _rosenbrock_instances() -> list[Instance]:
    proof = ("f is a sum of squares, so f >= 0; every square vanishes at x_i = 1 for all i, so f(1,...,1) = 0 is the "
             "global minimum.")
    out = []
    for n, split in ((2, "tuning"), (5, "scoring"), (20, "scoring")):
        out.append(Instance(
            id=f"rosenbrock-d{n}", family="rosenbrock", problem_class="nlp", dimension=n, variable_type="continuous",
            bounds=_box(n, -5.0, 10.0), sense="minimize", objective_exprs=(_ROSENBROCK_EXPR,),
            objectives=(_rosenbrock,), constraints=(),
            reference=Reference(0.0, "analytical", proof, x=(1.0,) * n), tolerance=_CONTINUOUS_TOL,
            statement=(f"Minimize the {n}-dimensional Rosenbrock function, the sum over i = 1..{n - 1} of "
                       "100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2, with every variable between -5 and 10."),
            split=split, well_known=True))
    n = 10
    out.append(Instance(
        id="rosenbrock-d10-shifted", family="rosenbrock", problem_class="nlp", dimension=n,
        variable_type="continuous", bounds=_box(n, -5.0, 10.0), sense="minimize",
        objective_exprs=(_ROSENBROCK_SHIFT_EXPR,), objectives=(functools.partial(_rosenbrock, shift=SHIFT),),
        constraints=(),
        reference=Reference(0.0, "analytical",
                            "Substitute z = x - s + 1: f is the Rosenbrock sum of squares in z, minimal (0) at z = 1, "
                            "i.e. x = s; s lies inside the box.", x=SHIFT),
        tolerance=_CONTINUOUS_TOL,
        statement=("Minimize a shifted 10-dimensional Rosenbrock function: with z_i = x_i - s_i + 1 and "
                   f"s = {list(SHIFT)}, minimize the sum over i = 1..9 of 100*(z_{{i+1}} - z_i^2)^2 + (1 - z_i)^2, "
                   "with every x_i between -5 and 10."),
        split="scoring", well_known=False, data={"s": SHIFT}))
    out.append(Instance(
        id="rosenbrock-disk-d2", family="rosenbrock", problem_class="nlp", dimension=2, variable_type="continuous",
        bounds=_box(2, -1.5, 1.5), sense="minimize", objective_exprs=(_ROSENBROCK_EXPR,), objectives=(_rosenbrock,),
        constraints=(Constraint("disk", "x[0]**2 + x[1]**2 - 2", "le", lambda x: x[0] ** 2 + x[1] ** 2 - 2.0),),
        reference=Reference(0.0, "analytical",
                            "f >= 0 everywhere and f(1, 1) = 0; (1, 1) satisfies x^2 + y^2 = 2 <= 2 (active), so the "
                            "constrained minimum is 0 at (1, 1). Also listed on the Wikipedia 'Test functions for "
                            "optimization' page (Rosenbrock constrained to a disk).", x=(1.0, 1.0)),
        tolerance=_CONTINUOUS_TOL,
        statement=("Minimize (1 - x)^2 + 100*(y - x^2)^2 over the disk x^2 + y^2 <= 2, with x and y each between "
                   "-1.5 and 1.5."),
        split="scoring", well_known=True))
    return out


def _rastrigin_ackley_instances() -> list[Instance]:
    rastrigin_proof = ("Each term x_i^2 - 10 cos(2 pi x_i) >= 0 - 10 = -10 with equality iff x_i = 0 (x_i^2 = 0 and "
                       "cos = 1 together), so f >= 10n - 10n = 0 with equality only at x = 0.")
    ackley_proof = ("-20 exp(-0.2 r) >= -20 with equality iff r = sqrt(mean x_i^2) = 0, and -exp(mean cos(2 pi x_i)) "
                    ">= -e with equality iff every x_i is an integer; both hold at x = 0, so f >= -20 - e + 20 + e = 0 "
                    "with equality only at x = 0.")
    out = []
    for fam, fn, expr, shift_expr, lo, proof, words in (
            ("rastrigin", _rastrigin, _RASTRIGIN_EXPR, _RASTRIGIN_SHIFT_EXPR, 5.12, rastrigin_proof,
             "the Rastrigin function 10n + sum_i (x_i^2 - 10 cos(2 pi x_i))"),
            ("ackley", _ackley, _ACKLEY_EXPR, _ACKLEY_SHIFT_EXPR, 32.768, ackley_proof,
             "the Ackley function -20 exp(-0.2 sqrt(mean_i x_i^2)) - exp(mean_i cos(2 pi x_i)) + 20 + e")):
        for n, split in ((2, "tuning"), (5, "scoring"), (20, "scoring")):
            out.append(Instance(
                id=f"{fam}-d{n}", family=fam, problem_class="global", dimension=n, variable_type="continuous",
                bounds=_box(n, -lo, lo), sense="minimize", objective_exprs=(expr,), objectives=(fn,), constraints=(),
                reference=Reference(0.0, "analytical", proof, x=(0.0,) * n), tolerance=_CONTINUOUS_TOL,
                statement=f"Find the global minimum of {words} in {n} dimensions, each x_i between -{lo} and {lo}.",
                split=split, well_known=True))
        out.append(Instance(
            id=f"{fam}-d10-shifted", family=fam, problem_class="global", dimension=10, variable_type="continuous",
            bounds=_box(10, -lo, lo), sense="minimize", objective_exprs=(shift_expr,),
            objectives=(functools.partial(fn, shift=SHIFT),), constraints=(),
            reference=Reference(0.0, "analytical", f"Substitute z = x - s: {proof} Hence x = s; s lies inside the box.",
                                x=SHIFT),
            tolerance=_CONTINUOUS_TOL,
            statement=(f"Find the global minimum of {words.replace('x_i', 'z_i')} in 10 dimensions, where "
                       f"z_i = x_i - s_i with s = {list(SHIFT)}, each x_i between -{lo} and {lo}."),
            split="scoring", well_known=False, data={"s": SHIFT}))
    return out


def _quadratic_instances() -> list[Instance]:
    out = [Instance(
        id="cqp-worked-d2", family="constrained_quadratic", problem_class="convex_qp", dimension=2,
        variable_type="continuous", bounds=((0.0, 10.0), (0.0, 10.0)), sense="minimize",
        objective_exprs=("x[0]**2 + x[1]**2",), objectives=(lambda x: x[0] ** 2 + x[1] ** 2,),
        constraints=(Constraint("sum_at_least_one", "1 - x[0] - x[1]", "le", lambda x: 1.0 - x[0] - x[1]),),
        reference=Reference(0.5, "analytical",
                            "x^2 + y^2 >= (x + y)^2 / 2 >= 1/2 (QM-AM, then x + y >= 1), with equality iff "
                            "x = y = 1/2, which is feasible.", x=(0.5, 0.5)),
        tolerance=Tolerance(objective_abs=1e-8),
        statement="Minimize x^2 + y^2 subject to x + y >= 1 and x, y >= 0 (and at most 10).",
        split="tuning", well_known=True)]
    for n in (5, 10):
        out.append(Instance(
            id=f"cqp-simplex-d{n}", family="constrained_quadratic", problem_class="convex_qp", dimension=n,
            variable_type="continuous", bounds=_box(n, 0.0, 1.0), sense="minimize",
            objective_exprs=("sum(x[i]**2 for i in range(n))",), objectives=(lambda x: float(np.sum(np.square(x))),),
            constraints=(Constraint("sum_at_least_one", "1 - sum(x[i] for i in range(n))", "le",
                                    lambda x: 1.0 - float(np.sum(x))),),
            reference=Reference(1.0 / n, "analytical",
                                f"Cauchy-Schwarz: sum x_i^2 >= (sum x_i)^2 / {n} >= 1/{n}, equality iff every "
                                f"x_i = 1/{n}, which is feasible.", x=(1.0 / n,) * n),
            tolerance=Tolerance(objective_abs=1e-8),
            statement=(f"Choose {n} numbers, each between 0 and 1, whose sum is at least 1, so that the sum of their "
                       "squares is as small as possible."),
            split="scoring", well_known=False))
    out.append(Instance(
        id="cqp-projection-d2", family="constrained_quadratic", problem_class="convex_qp", dimension=2,
        variable_type="continuous", bounds=((0.0, 5.0), (0.0, 5.0)), sense="minimize",
        objective_exprs=("(x[0]-2)**2 + (x[1]-1)**2",), objectives=(lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,),
        constraints=(Constraint("half_plane", "x[0] + x[1] - 1", "le", lambda x: x[0] + x[1] - 1.0),),
        reference=Reference(2.0, "analytical",
                            "Squared distance from (2, 1) to the half-plane x + y <= 1. The point is outside, so the "
                            "projection lies on x + y = 1: (2, 1) - ((3 - 1) / 2)(1, 1) = (1, 0), inside the bounds; "
                            "squared distance (3 - 1)^2 / 2 = 2.", x=(1.0, 0.0)),
        tolerance=Tolerance(objective_abs=1e-8),
        statement=("Find the point with x + y <= 1 and 0 <= x, y <= 5 closest to (2, 1); minimize the squared "
                   "distance (x - 2)^2 + (y - 1)^2."),
        split="scoring", well_known=False))
    out.append(Instance(
        id="cqp-equality-d3", family="constrained_quadratic", problem_class="convex_qp", dimension=3,
        variable_type="continuous", bounds=_box(3, -10.0, 10.0), sense="minimize",
        objective_exprs=("x[0]**2 + 2*x[1]**2 + 3*x[2]**2",),
        objectives=(lambda x: x[0] ** 2 + 2.0 * x[1] ** 2 + 3.0 * x[2] ** 2,),
        constraints=(Constraint("sum_is_one", "x[0] + x[1] + x[2] - 1", "eq", lambda x: x[0] + x[1] + x[2] - 1.0),),
        reference=Reference(6.0 / 11.0, "analytical",
                            "Lagrange: 2x = 4y = 6z = lambda, so (x, y, z) = lambda (1/2, 1/4, 1/6); the constraint "
                            "gives lambda = 12/11, x = (6/11, 3/11, 2/11) and f = (36 + 18 + 12)/121 = 6/11. The "
                            "objective is strictly convex, so this stationary point is the global minimum.",
                            x=(6.0 / 11.0, 3.0 / 11.0, 2.0 / 11.0)),
        tolerance=Tolerance(objective_abs=1e-8),
        statement="Minimize x^2 + 2y^2 + 3z^2 subject to x + y + z = 1, each variable between -10 and 10.",
        split="scoring", well_known=False))
    return out


def _linear_instances() -> list[Instance]:
    return [
        Instance(
            id="lp-diet-d2", family="linear_program", problem_class="lp", dimension=2, variable_type="continuous",
            bounds=((0.0, 10.0), (0.0, 10.0)), sense="minimize", objective_exprs=("2*x[0] + 3*x[1]",),
            objectives=(lambda x: 2.0 * x[0] + 3.0 * x[1],),
            constraints=(Constraint("nutrient_a", "4 - x[0] - x[1]", "le", lambda x: 4.0 - x[0] - x[1]),
                         Constraint("nutrient_b", "6 - x[0] - 3*x[1]", "le", lambda x: 6.0 - x[0] - 3.0 * x[1])),
            reference=Reference(9.0, "brute_force",
                                "Vertex enumeration of the 2-D polygon: (0, 4) -> 12, (3, 1) -> 9, (6, 0) -> 12, and "
                                "the box corners are dearer; minimum 9 at (3, 1).", x=(3.0, 1.0)),
            tolerance=Tolerance(objective_abs=1e-9),
            statement=("Food 1 costs 2 per unit and food 2 costs 3 per unit. A diet needs at least 4 units of "
                       "nutrient A (each unit of either food gives 1) and at least 6 units of nutrient B (food 1 "
                       "gives 1, food 2 gives 3). Buy at most 10 units of each. Minimize cost."),
            split="tuning", well_known=False),
        Instance(
            id="lp-wyndor-d2", family="linear_program", problem_class="lp", dimension=2, variable_type="continuous",
            bounds=((0.0, 100.0), (0.0, 100.0)), sense="maximize", objective_exprs=("3*x[0] + 5*x[1]",),
            objectives=(lambda x: 3.0 * x[0] + 5.0 * x[1],),
            constraints=(Constraint("plant_1", "x[0] - 4", "le", lambda x: x[0] - 4.0),
                         Constraint("plant_2", "2*x[1] - 12", "le", lambda x: 2.0 * x[1] - 12.0),
                         Constraint("plant_3", "3*x[0] + 2*x[1] - 18", "le", lambda x: 3.0 * x[0] + 2.0 * x[1] - 18.0)),
            reference=Reference(36.0, "published",
                                "Wyndor Glass Co. example (Hillier & Lieberman, Introduction to Operations Research): "
                                "Z = 36 at (2, 6). Independently re-derived by vertex enumeration in the tests.",
                                x=(2.0, 6.0)),
            tolerance=Tolerance(objective_abs=1e-9),
            statement=("A plant makes two products with profit 3 and 5 per batch. Plant 1 has 4 hours (product 1 "
                       "uses 1 hour), plant 2 has 12 hours (product 2 uses 2 hours), plant 3 has 18 hours (product 1 "
                       "uses 3, product 2 uses 2). Batches are continuous and nonnegative. Maximize profit."),
            split="scoring", well_known=True),
    ]


def _knapsack(id_, values, weight_rows, capacities, ref, x, method, derivation, statement, split, well_known,
              lp_relaxation=False) -> Instance:
    n = len(values)
    names = [f"w{k}" if len(weight_rows) > 1 else "w" for k in range(len(weight_rows))]
    caps = [f"C{k}" if len(weight_rows) > 1 else "C" for k in range(len(weight_rows))]
    data: dict[str, object] = {"v": tuple(values)}
    constraints = []
    for row, cap, wname, cname in zip(weight_rows, capacities, names, caps):
        data[wname], data[cname] = tuple(row), cap
        constraints.append(Constraint(f"capacity_{cname}", f"sum({wname}[i]*x[i] for i in range(n)) - {cname}", "le",
                                      lambda x, row=tuple(row), cap=cap: float(np.dot(row, x)) - cap))
    return Instance(
        id=id_, family="knapsack", problem_class="lp" if lp_relaxation else "milp", dimension=n,
        variable_type="continuous" if lp_relaxation else "binary", bounds=_box(n, 0.0, 1.0), sense="maximize",
        objective_exprs=("sum(v[i]*x[i] for i in range(n))",),
        objectives=(lambda x, v=tuple(values): float(np.dot(v, x)),),
        constraints=tuple(constraints), reference=Reference(ref, method, derivation, x=x),
        tolerance=Tolerance(objective_abs=1e-9), statement=statement, split=split, well_known=well_known, data=data)


_P01_W = (23, 31, 29, 44, 53, 38, 63, 85, 89, 82)
_P01_V = (92, 57, 49, 68, 60, 43, 67, 84, 87, 72)
_P07_W = (70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120)
_P07_V = (135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240)
_K20_W = (12, 7, 19, 25, 5, 31, 14, 9, 22, 17, 28, 11, 6, 35, 16, 20, 8, 27, 13, 24)
_K20_V = (24, 13, 35, 46, 9, 55, 27, 18, 40, 30, 50, 21, 10, 61, 29, 37, 15, 48, 25, 43)
_MD_W1 = (10, 14, 7, 21, 9, 16, 12, 5, 18, 11, 8, 15)
_MD_W2 = (6, 12, 15, 4, 11, 9, 17, 8, 5, 13, 10, 7)
_MD_V = (21, 30, 18, 38, 20, 33, 27, 11, 34, 24, 17, 29)


def _items(weights, values, cap, extra=""):
    pairs = "; ".join(f"item {i + 1}: weight {w}, value {v}" for i, (w, v) in enumerate(zip(weights, values)))
    return f"Pick a subset of items with total weight at most {cap}{extra} to maximize total value. {pairs}."


def _knapsack_instances() -> list[Instance]:
    bf = "Exhaustive enumeration of all 2^{n} subsets (re-run in the tests)."
    return [
        _knapsack("knapsack-p02-n5", (24, 13, 23, 15, 16), ((12, 7, 11, 8, 9),), (26,), 51.0, (0, 1, 1, 1, 0),
                  "brute_force", bf.format(n=5) + " Agrees with Burkardt's KNAPSACK_01 set P02.",
                  _items((12, 7, 11, 8, 9), (24, 13, 23, 15, 16), 26), "tuning", True),
        _knapsack("knapsack-p01-n10", _P01_V, (_P01_W,), (165,), 309.0, (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
                  "brute_force", bf.format(n=10) + " Agrees with Burkardt's KNAPSACK_01 set P01.",
                  _items(_P01_W, _P01_V, 165), "scoring", True),
        _knapsack("knapsack-p07-n15", _P07_V, (_P07_W,), (750,), 1458.0,
                  (1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1),
                  "brute_force", bf.format(n=15) + " Agrees with Burkardt's KNAPSACK_01 set P07.",
                  _items(_P07_W, _P07_V, 750), "scoring", True),
        _knapsack("knapsack-correlated-n20", _K20_V, (_K20_W,), (150,), 282.0, None,
                  "brute_force", bf.format(n=20) + " Instance authored for E14 (values within 1.67-2.0x weight, "
                  "so the greedy ratio order is a poor guide); two distinct subsets attain the optimum, so no x is "
                  "stored.",
                  _items(_K20_W, _K20_V, 150), "scoring", False),
        _knapsack("knapsack-2d-n12", _MD_V, (_MD_W1, _MD_W2), (70, 55), 147.0, (1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0),
                  "brute_force", bf.format(n=12) + " Two-constraint instance authored for E14.",
                  _items(_MD_W1, _MD_V, 70, extra=" and total volume at most 55 (volumes in item order: "
                         + ", ".join(map(str, _MD_W2)) + ")"), "scoring", False),
        _knapsack("knapsack-p01-lp-relaxation", _P01_V, (_P01_W,), (165,), 16378.0 / 53.0,
                  (1.0, 1.0, 1.0, 1.0, 38.0 / 53.0, 0.0, 0.0, 0.0, 0.0, 0.0), "analytical",
                  "Fractional knapsack is solved by Dantzig's greedy (exchange argument): take items by value/weight "
                  "ratio 4.00, 1.84, 1.69, 1.55 (weight 127, value 266), then 38/53 of item 5 (ratio 1.132), giving "
                  "266 + 60*38/53 = 16378/53 = 309.0189.",
                  _items(_P01_W, _P01_V, 165) + " Fractions of an item are allowed (each between 0 and 1).",
                  "scoring", False, lp_relaxation=True),
    ]


# ---------------------------------------------------------------------------------------------- integration family
#: the fixed menu of rules; a decision is (rule index, N) and costs N integrand evaluations
QUADRATURE_RULES = ("gauss_legendre", "trapezoid", "simpson", "gauss_legendre_t2")
N_MAX = 512


@functools.lru_cache(maxsize=None)
def _gl(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def quadrature(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, rule: int, n: int) -> float:
    """Rule ``rule`` with ``n`` integrand evaluations on [a, b]; NaN for a combination the rule does not define."""
    rule, n = int(round(rule)), int(round(n))
    if not 0 <= rule < len(QUADRATURE_RULES) or n < 1:
        return math.nan
    name = QUADRATURE_RULES[rule]
    if name == "gauss_legendre":
        t, w = _gl(n)
        return float(0.5 * (b - a) * np.dot(w, f(0.5 * (b - a) * t + 0.5 * (a + b))))
    if name == "gauss_legendre_t2":  # x = a + (b - a) s^2, dx = 2 (b - a) s ds, s in [0, 1]
        t, w = _gl(n)
        s = 0.5 * (t + 1.0)
        return float(0.5 * np.dot(w, f(a + (b - a) * s ** 2) * 2.0 * (b - a) * s))
    if name == "trapezoid" and n >= 2:
        y = f(np.linspace(a, b, n))
        return float((b - a) / (n - 1) * (np.sum(y) - 0.5 * (y[0] + y[-1])))
    if name == "simpson" and n >= 3 and n % 2 == 1:
        y = f(np.linspace(a, b, n))
        return float((b - a) / (n - 1) / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))
    return math.nan


def _integration(id_, integrand_expr, f, a, b, exact, exact_derivation, tau, n_star, rule_star, split, words):
    def quadrature_error(rule, n):
        q = quadrature(f, a, b, rule, n)
        return math.inf if math.isnan(q) else abs(q - exact)
    return Instance(
        id=id_, family="integration", problem_class="black_box", dimension=2, variable_type="integer",
        bounds=((0.0, len(QUADRATURE_RULES) - 1.0), (1.0, float(N_MAX))), sense="minimize",
        objective_exprs=("x[1]",), objectives=(lambda x: float(round(x[1])),),
        constraints=(Constraint("error_target", "quadrature_error(x[0], x[1]) - tau", "le",
                                lambda x: quadrature_error(x[0], x[1]) - tau),),
        reference=Reference(float(n_star), "brute_force",
                            f"Integral: {exact_derivation} Minimum cost by exhaustive search over N = 1, 2, ... and "
                            f"every rule, stopping at the first N whose error is <= {tau:g}: N = {n_star} with "
                            f"{QUADRATURE_RULES[rule_star]}.", x=(float(rule_star), float(n_star))),
        tolerance=Tolerance(objective_abs=0.0, constraint_abs=0.0),
        statement=(f"Estimate the integral of {words} from {a:g} to {b:g} to absolute error at most {tau:g}, using as "
                   "few integrand evaluations as possible. Choose one rule from Gauss-Legendre, composite trapezoid, "
                   "composite Simpson, or Gauss-Legendre after the substitution x = a + (b - a) s^2, and the number of "
                   f"evaluations N (at most {N_MAX})."),
        split=split, well_known=False,
        data={"tau": tau, "integrand": integrand_expr, "interval": (a, b), "rules": QUADRATURE_RULES},
        blackboxes={"quadrature_error": quadrature_error})


EPS_NEAR_SINGULAR = 1e-3


def _integration_instances() -> list[Instance]:
    exp_exact = math.e - 1.0
    runge_exact = 0.4 * math.atan(5.0)
    eps = EPS_NEAR_SINGULAR
    sing_exact = 2.0 * (math.sqrt(1.0 + eps) - math.sqrt(eps))
    sing = (lambda x: 1.0 / np.sqrt(x + eps))
    return [
        _integration("integ-exp-1e-10", "exp(t)", np.exp, 0.0, 1.0, exp_exact, "int_0^1 e^t dt = e - 1.",
                     1e-10, 5, 0, "tuning", "e^t"),
        _integration("integ-runge-1e-8", "1/(1+25*t**2)", lambda x: 1.0 / (1.0 + 25.0 * x ** 2), -1.0, 1.0,
                     runge_exact, "int_-1^1 dt/(1 + 25 t^2) = (2/5) atan(5).", 1e-8, 47, 0, "scoring",
                     "1/(1 + 25 t^2)"),
        _integration("integ-nearsing-1e-6", "1/sqrt(t+0.001)", sing, 0.0, 1.0, sing_exact,
                     "int_0^1 (t + eps)^(-1/2) dt = 2 (sqrt(1 + eps) - sqrt(eps)), eps = 1e-3.", 1e-6, 8, 3,
                     "scoring", "1/sqrt(t + 0.001)"),
        _integration("integ-nearsing-1e-10", "1/sqrt(t+0.001)", sing, 0.0, 1.0, sing_exact,
                     "int_0^1 (t + eps)^(-1/2) dt = 2 (sqrt(1 + eps) - sqrt(eps)), eps = 1e-3.", 1e-10, 39, 3,
                     "scoring", "1/sqrt(t + 0.001)"),
        _integration("integ-peak-1e-8", "1/((t-0.3)**2+0.01)", lambda x: 1.0 / ((x - 0.3) ** 2 + 0.01), 0.0, 1.0,
                     10.0 * (math.atan(7.0) + math.atan(3.0)),
                     "int_0^1 dt/((t - 0.3)^2 + 0.01) = 10 (atan(7) + atan(3)).", 1e-8, 52, 0, "scoring",
                     "1/((t - 0.3)^2 + 0.01)"),
        _integration("integ-oscillatory-1e-10", "cos(20*t)", lambda x: np.cos(20.0 * x), 0.0, 1.0,
                     math.sin(20.0) / 20.0, "int_0^1 cos(20 t) dt = sin(20)/20.", 1e-10, 14, 0, "scoring",
                     "cos(20 t)"),
        _integration("integ-gaussian-1e-12", "exp(-t**2)", lambda x: np.exp(-x ** 2), 0.0, 1.0,
                     0.5 * math.sqrt(math.pi) * math.erf(1.0), "int_0^1 e^(-t^2) dt = (sqrt(pi)/2) erf(1).", 1e-12,
                     7, 0, "scoring", "e^(-t^2)"),
    ]


# --------------------------------------------------------------------------------------------- multi-objective family
def _bi_objective_instances() -> list[Instance]:
    return [
        Instance(
            id="mo-schaffer-d1", family="bi_objective", problem_class="multi_objective", dimension=1,
            variable_type="continuous", bounds=((-10.0, 10.0),), sense="minimize",
            objective_exprs=("x[0]**2", "(x[0]-2)**2"), objectives=(lambda x: x[0] ** 2, lambda x: (x[0] - 2.0) ** 2),
            constraints=(),
            reference=Reference(40.0 / 3.0, "analytical",
                                "Pareto set x in [0, 2], front f2 = (sqrt(f1) - 2)^2 for f1 in [0, 4]. Hypervolume "
                                "w.r.t. (4, 4) = int_0^4 (4 - (sqrt(f1) - 2)^2) df1 = int_0^4 (4 sqrt(f1) - f1) df1 = "
                                "64/3 - 8 = 40/3.", indicator="hypervolume", reference_point=(4.0, 4.0)),
            tolerance=Tolerance(objective_abs=0.0, objective_rel=0.05),
            statement=("Find the trade-off between minimizing x^2 and minimizing (x - 2)^2 for x between -10 and 10; "
                       "return up to 50 non-dominated points."),
            split="tuning", well_known=True),
        Instance(
            id="mo-two-centres-d2", family="bi_objective", problem_class="multi_objective", dimension=2,
            variable_type="continuous", bounds=((-2.0, 2.0), (-2.0, 2.0)), sense="minimize",
            objective_exprs=("x[0]**2 + x[1]**2", "(x[0]-1)**2 + x[1]**2"),
            objectives=(lambda x: x[0] ** 2 + x[1] ** 2, lambda x: (x[0] - 1.0) ** 2 + x[1] ** 2),
            constraints=(),
            reference=Reference(5.0 / 6.0, "analytical",
                                "Pareto set is the segment y = 0, x in [0, 1]; front f2 = (1 - sqrt(f1))^2 for f1 in "
                                "[0, 1]. Hypervolume w.r.t. (1, 1) = int_0^1 (2 sqrt(f1) - f1) df1 = 4/3 - 1/2 = 5/6.",
                                indicator="hypervolume", reference_point=(1.0, 1.0)),
            tolerance=Tolerance(objective_abs=0.0, objective_rel=0.05),
            statement=("Find the trade-off between the squared distance to (0, 0) and the squared distance to (1, 0) "
                       "for a point in the square [-2, 2] x [-2, 2]; return up to 50 non-dominated points."),
            split="scoring", well_known=False),
        _zdt("mo-zdt1-d5", "1 - sqrt({r})", lambda r: 1.0 - math.sqrt(r), 2.0 / 3.0,
             "front f2 = 1 - sqrt(f1) (convex); hypervolume w.r.t. (1, 1) = int_0^1 sqrt(f1) df1 = 2/3."),
        _zdt("mo-zdt2-d5", "1 - ({r})**2", lambda r: 1.0 - r ** 2, 1.0 / 3.0,
             "front f2 = 1 - f1^2 (non-convex: a weighted sum finds only its end points); hypervolume w.r.t. "
             "(1, 1) = int_0^1 f1^2 df1 = 1/3."),
    ]


_ZDT_G = "(1 + 9*sum(x[i] for i in range(1, n))/(n-1))"


def _zdt(id_, h_expr, h, hv, front_words) -> Instance:
    """ZDT1/ZDT2 (Zitzler, Deb, Thiele 2000) with n = 5: f1 = x0, f2 = g h(f1 / g)."""
    n = 5

    def f2(x):
        g = 1.0 + 9.0 * float(np.sum(x[1:])) / (n - 1)
        return g * h(x[0] / g)
    return Instance(
        id=id_, family="bi_objective", problem_class="multi_objective", dimension=n, variable_type="continuous",
        bounds=_box(n, 0.0, 1.0), sense="minimize",
        objective_exprs=("x[0]", _ZDT_G + " * (" + h_expr.format(r="x[0]/" + _ZDT_G) + ")"),
        objectives=(lambda x: float(x[0]), f2), constraints=(),
        reference=Reference(hv, "analytical",
                            f"g >= 1 with equality iff x_1..x_4 = 0, and f2 grows with g, so the Pareto set is "
                            f"x_1..x_4 = 0, x_0 in [0, 1]; {front_words}", indicator="hypervolume",
                            reference_point=(1.0, 1.0)),
        tolerance=Tolerance(objective_abs=0.0, objective_rel=0.05),
        statement=(f"Five variables in [0, 1]. Minimize f1 = x0 and f2 = g * h(x0 / g) together, where "
                   f"g = 1 + 9 (x1 + x2 + x3 + x4) / 4 and h(r) = {h_expr.format(r='r')}; return up to 50 "
                   "non-dominated points."),
        split="scoring", well_known=True)


MO_MAX_POINTS = 50

INSTANCES: tuple[Instance, ...] = tuple(
    _rosenbrock_instances() + _rastrigin_ackley_instances() + _quadratic_instances() + _linear_instances()
    + _knapsack_instances() + _integration_instances() + _bi_objective_instances())

EXTERNAL: tuple[ExternalInstance, ...] = (
    ExternalInstance("cutest-hs035", "CUTEst", "HS35", "convex_qp", "3",
                     "Smallest Hock-Schittkowski convex QP with an active linear inequality: checks the compiler on a "
                     "classic QP whose optimum is interior to the bounds but on a constraint.",
                     "f* = 1/9 at (4/3, 7/9, 4/9) (Hock & Schittkowski 1981)", "pycutest + CUTEst/SIFDecode"),
    ExternalInstance("cutest-hs071", "CUTEst", "HS71", "nlp", "4",
                     "Mixed equality/inequality nonconvex NLP used as the IPOPT tutorial problem: the standard smoke "
                     "test for a general NLP path.",
                     "f* = 17.0140173 (Hock & Schittkowski 1981)", "pycutest + CUTEst/SIFDecode"),
    ExternalInstance("cutest-hs076", "CUTEst", "HS76", "convex_qp", "4",
                     "Convex QP with three linear inequalities and bounds: a second QP whose active set differs.",
                     "f* = -4.681818 (Hock & Schittkowski 1981)", "pycutest + CUTEst/SIFDecode"),
    ExternalInstance("cutest-dixmaana", "CUTEst", "DIXMAANA", "nlp", "3000 (scalable)",
                     "Scalable unconstrained NLP: checks that formulation and budget accounting survive a dimension "
                     "two orders above the runnable catalogue.",
                     "f* = 1 (CUTEst SIF header)", "pycutest + CUTEst/SIFDecode"),
    ExternalInstance("bbob-f08", "COCO/BBOB", "f8 Rosenbrock, original", "nlp", "2, 5, 10, 20; instances 1-15",
                     "Shifted, scaled Rosenbrock with per-instance optima: a non-memorisable twin of rosenbrock-d*.",
                     "f_opt per instance from cocoex; target f - f_opt <= 1e-8", "cocoex"),
    ExternalInstance("bbob-f15", "COCO/BBOB", "f15 Rastrigin, rotated", "global", "2, 5, 10, 20; instances 1-15",
                     "Non-separable Rastrigin: the runnable rastrigin-d* is separable, which a coordinate-wise "
                     "method exploits; f15 removes that.",
                     "f_opt per instance from cocoex; target f - f_opt <= 1e-8", "cocoex"),
    ExternalInstance("bbob-f21", "COCO/BBOB", "f21 Gallagher's Gaussian 101-me peaks", "global",
                     "2, 5, 10, 20; instances 1-15",
                     "Weak global structure with many unrelated local optima: no formula an LLM could exploit.",
                     "f_opt per instance from cocoex; target f - f_opt <= 1e-8", "cocoex"),
    ExternalInstance("bbob-f24", "COCO/BBOB", "f24 Lunacek bi-Rastrigin", "global", "2, 5, 10, 20; instances 1-15",
                     "Deceptive two-funnel landscape: the larger basin is not the one containing the optimum.",
                     "f_opt per instance from cocoex; target f - f_opt <= 1e-8", "cocoex"),
)


def by_id(instance_id: str) -> Instance:
    for inst in INSTANCES:
        if inst.id == instance_id:
            return inst
    raise KeyError(instance_id)


def neutral_form(instance: Instance) -> dict:
    """The JSON-able statement of an instance: what a runner maps onto the IR and what ``problem_digest`` covers.
    Callables are excluded; black boxes appear by name only."""
    return {
        "id": instance.id, "family": instance.family, "problem_class": instance.problem_class,
        "dimension": instance.dimension, "variable_type": instance.variable_type,
        "bounds": [list(b) for b in instance.bounds], "sense": instance.sense,
        "objective_exprs": list(instance.objective_exprs),
        "constraints": [{"name": c.name, "expr": c.expr, "kind": c.kind} for c in instance.constraints],
        "data": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(instance.data.items())},
        "blackboxes": sorted(instance.blackboxes),
    }
