"""Derive a problem's structure from its IR, exactly, so solvers can be matched to it.

Every fact here is decided from the expression syntax trees and the variable
declarations, never by sampling the functions. Each objective and constraint
expression is expanded into a polynomial with exact rational coefficients
(literals are read as the decimals they spell). A subtree that is not a
polynomial in the variables (``sin(x)``, ``x**0.5``, ``1/x``, ``abs(x)``) is
kept as an opaque atom, so ``sin(x) - sin(x)`` cancels to zero and
``(x + 1)**2 - x**2`` is linear. After expansion an expression is:

* ``LINEAR`` or ``QUADRATIC`` when no atom survives and the degree is at most 1 or 2;
* ``SMOOTH`` when it is a higher-degree polynomial or every surviving atom is
  built from functions that are infinitely differentiable on the interior of
  their domain (everything except ``abs``, ``min`` and ``max``);
* ``NONSMOOTH`` when a surviving atom applies ``abs``, ``min`` or ``max`` to a
  variable-dependent argument;
* ``BLACK_BOX`` when it is a ``ToolCall``.

What is decidable, and therefore reported: linearity and quadraticity, the
convexity of a quadratic objective (an exact rational positive-semidefinite
test of its Hessian), variable kinds, whether every variable has finite bounds,
constraint senses and the number of objectives. What is not decidable from the
IR, and therefore never claimed: whether a nonlinear problem is convex,
unimodal or multimodal (so there is no "global" class), and what a black-box
tool computes.

Two deliberate limits make the classification conservative, never wrong:
identities between transcendental functions are not used (``sin(x)**2 +
cos(x)**2`` is ``SMOOTH``, not constant), and a polynomial whose expansion
exceeds ``MAX_TERMS`` terms is treated as an opaque atom. ``SMOOTH`` does not
promise differentiability where an argument reaches a domain boundary:
``sqrt(x**2)`` is reported smooth although it equals ``abs(x)``.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from fractions import Fraction
from typing import Mapping

from .expressions import SCALAR_FUNCTIONS, ToolCall
from .problem import Constraint, Objective, OptimizationProblem
from .units import parse_unit

MAX_TERMS = 2000
"""Expansion budget per subtree; a subtree whose expansion is larger becomes an opaque atom."""

_NONSMOOTH_FUNCTIONS = frozenset({"abs", "min", "max"})


class Form(IntEnum):
    """The most general kind of function an expression is, ordered from most to least structured."""

    LINEAR = 0
    QUADRATIC = 1
    SMOOTH = 2
    NONSMOOTH = 3
    BLACK_BOX = 4


class ProblemClass(str, Enum):
    """The headline class of a problem, decided by ``classify`` in this precedence order.

    ``MULTI_OBJECTIVE``: more than one objective. ``BLACK_BOX``: a ``ToolCall``
    anywhere. ``MILP``/``MINLP``: an integer, binary or discrete variable, with
    every expression linear or not. Then, for continuous problems:
    ``NONSMOOTH`` (``abs``/``min``/``max`` survives), ``LP`` (linear objective
    and constraints), ``CONVEX_QP`` (quadratic objective that is convex in its
    sense, linear constraints), ``BOX_SMOOTH`` (no constraints besides the
    variable bounds) and ``NLP`` (everything else).
    """

    LP = "lp"
    CONVEX_QP = "convex_qp"
    BOX_SMOOTH = "box_smooth"
    NLP = "nlp"
    NONSMOOTH = "nonsmooth"
    MILP = "milp"
    MINLP = "minlp"
    BLACK_BOX = "black_box"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass(frozen=True)
class ProblemStructure:
    """The decidable facts about a problem that solver compatibility depends on.

    ``objective_forms`` has one entry per objective; ``constraint_form`` is the
    most general form among the constraints, ``None`` when there are none.
    ``integer`` means some variable is integer, binary or discrete;
    ``bounded`` that every variable's domain is finite. ``domain_size`` is
    the number of points of the variables' product domain when every
    variable takes finitely many values, ``None`` otherwise.
    """

    problem_class: ProblemClass
    objective_forms: tuple[Form, ...]
    constraint_form: Form | None
    integer: bool
    equality: bool
    bounded: bool
    domain_size: int | None = None

    @property
    def form(self) -> Form:
        """The most general form among every objective and constraint."""
        return max(self.objective_forms + (() if self.constraint_form is None else (self.constraint_form,)))

    @property
    def multiobjective(self) -> bool:
        return len(self.objective_forms) > 1

    @property
    def constrained(self) -> bool:
        return self.constraint_form is not None


class _Opaque(Exception):
    """The subtree is not a polynomial in the variables."""


Monomial = tuple[tuple[str, int], ...]
Polynomial = dict[Monomial, Fraction]


def _exact(value: float) -> Fraction:
    return Fraction(repr(float(value)))


def _add(a: Polynomial, b: Polynomial, sign: int = 1) -> Polynomial:
    out = dict(a)
    for mono, coef in b.items():
        out[mono] = out.get(mono, Fraction(0)) + sign * coef
    return {m: c for m, c in out.items() if c}


def _mul(a: Polynomial, b: Polynomial) -> Polynomial:
    if len(a) * len(b) > MAX_TERMS * 4:
        raise _Opaque
    out: Polynomial = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            powers = dict(ma)
            for name, p in mb:
                powers[name] = powers.get(name, 0) + p
            mono = tuple(sorted(powers.items()))
            out[mono] = out.get(mono, Fraction(0)) + ca * cb
    out = {m: c for m, c in out.items() if c}
    if len(out) > MAX_TERMS:
        raise _Opaque
    return out


def _constant(p: Polynomial) -> Fraction | None:
    if not p:
        return Fraction(0)
    if len(p) == 1 and () in p:
        return p[()]
    return None


class _Expander:
    """Expands one expression tree into a polynomial over variables and opaque atoms."""

    def __init__(self, variables: Mapping[str, float], parameters: Mapping[str, float]):
        self.variables = variables
        self.parameters = parameters
        self.atoms: dict[str, bool] = {}  # atom key -> smooth

    def expand(self, node: ast.AST) -> Polynomial:
        try:
            return self._poly(node)
        except _Opaque:
            return self._atom(node)

    def _atom(self, node: ast.AST) -> Polynomial:
        key = "@" + ast.dump(node)
        self.atoms[key] = self._smooth(node)
        return {((key, 1),): Fraction(1)}

    def _smooth(self, node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and sub.func.id in _NONSMOOTH_FUNCTIONS
                    and any(self._depends(arg) for arg in sub.args)):
                return False
        return True

    def _depends(self, node: ast.AST) -> bool:
        return any(isinstance(sub, ast.Name) and sub.id in self.variables for sub in ast.walk(node))

    def _poly(self, node: ast.AST) -> Polynomial:
        if isinstance(node, ast.Constant):
            return {(): _exact(node.value)} if node.value else {}
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return {((node.id, 1),): _exact(self.variables[node.id])}
            return {(): _exact(self.parameters[node.id])} if self.parameters[node.id] else {}
        if isinstance(node, ast.UnaryOp):
            inner = self.expand(node.operand)
            return inner if isinstance(node.op, ast.UAdd) else {m: -c for m, c in inner.items()}
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        return self._call(node)

    def _binop(self, node: ast.BinOp) -> Polynomial:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _add(self.expand(node.left), self.expand(node.right), 1 if isinstance(node.op, ast.Add) else -1)
        if isinstance(node.op, ast.Mult):
            return _mul(self.expand(node.left), self.expand(node.right))
        if isinstance(node.op, ast.Div):
            denominator = _constant(self.expand(node.right))
            if not denominator:
                raise _Opaque
            return {m: c / denominator for m, c in self.expand(node.left).items()}
        base, exponent = self.expand(node.left), _constant(self.expand(node.right))
        if exponent is None:
            raise _Opaque
        if _constant(base) is not None:
            return self._number(lambda: float(_constant(base)) ** float(exponent))
        if exponent.denominator != 1 or exponent < 0:
            raise _Opaque
        out: Polynomial = {(): Fraction(1)}
        for _ in range(int(exponent)):
            out = _mul(out, base)
        return out

    def _call(self, node: ast.Call) -> Polynomial:
        args = [_constant(self.expand(arg)) for arg in node.args]
        if any(a is None for a in args):
            raise _Opaque
        return self._number(lambda: SCALAR_FUNCTIONS[node.func.id](*(float(a) for a in args)))

    def _number(self, compute) -> Polynomial:
        """A constant subtree, folded the way the evaluator computes it; opaque if that fails."""
        try:
            value = compute()
        except (ArithmeticError, ValueError):
            raise _Opaque from None
        if isinstance(value, complex) or not math.isfinite(value):
            raise _Opaque
        return {(): _exact(value)} if value else {}


def _expand(problem: OptimizationProblem, item: Objective | Constraint) -> tuple[Polynomial, dict[str, bool]]:
    """The polynomial of ``item`` in the variables' own units (as ``evaluate`` reports it), and its atoms."""
    variables = {v.name: parse_unit(v.unit).scale for v in problem.variables}
    parameters = {p.name: p.value * parse_unit(p.unit).scale for p in problem.parameters}
    expander = _Expander(variables, parameters)
    scale = _exact(parse_unit(item.unit).scale)
    poly = expander.expand(item.expression.tree)
    return {m: c / scale for m, c in poly.items()}, expander.atoms


def _form(problem: OptimizationProblem, item: Objective | Constraint) -> tuple[Form, Polynomial]:
    if isinstance(item.expression, ToolCall):
        return Form.BLACK_BOX, {}
    poly, atoms = _expand(problem, item)
    surviving = {name for mono in poly for name, _ in mono if name in atoms}
    if surviving:
        return (Form.SMOOTH if all(atoms[a] for a in surviving) else Form.NONSMOOTH), poly
    degree = max((sum(p for _, p in mono) for mono in poly), default=0)
    if degree <= 1:
        return Form.LINEAR, poly
    return (Form.QUADRATIC if degree == 2 else Form.SMOOTH), poly


def _positive_semidefinite(matrix: list[list[Fraction]]) -> bool:
    """Exact test by symmetric Gaussian elimination: no negative pivot, and a zero pivot has a zero row."""
    a = [row[:] for row in matrix]
    n = len(a)
    for k in range(n):
        if a[k][k] < 0:
            return False
        if a[k][k] == 0:
            if any(a[k][j] for j in range(k + 1, n)):
                return False
            continue
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            for j in range(k + 1, n):
                a[i][j] -= factor * a[k][j]
    return True


def _convex_quadratic(problem: OptimizationProblem, poly: Polynomial, sense: str) -> bool:
    index = {v.name: i for i, v in enumerate(problem.variables)}
    hessian = [[Fraction(0)] * len(index) for _ in index]
    for mono, coef in poly.items():
        if sum(p for _, p in mono) != 2:
            continue
        if len(mono) == 1:
            i = index[mono[0][0]]
            hessian[i][i] += 2 * coef
        else:
            i, j = index[mono[0][0]], index[mono[1][0]]
            hessian[i][j] += coef
            hessian[j][i] += coef
    if sense == "maximize":
        hessian = [[-c for c in row] for row in hessian]
    return _positive_semidefinite(hessian)


def classify(problem: OptimizationProblem) -> ProblemStructure:
    """The structure of a valid problem, decided exactly from its expressions and declarations."""
    objectives = [_form(problem, o) for o in problem.objectives]
    constraint_forms = [_form(problem, c)[0] for c in problem.constraints]
    constraint_form = max(constraint_forms) if constraint_forms else None
    integer = any(v.kind != "continuous" for v in problem.variables)
    bounded = all(v.kind in ("binary", "discrete") or (v.lower is not None and v.upper is not None)
                  for v in problem.variables)
    forms = tuple(f for f, _ in objectives)
    worst = max(forms + tuple(constraint_forms))
    linear_constraints = constraint_form in (None, Form.LINEAR)
    if len(forms) > 1:
        cls = ProblemClass.MULTI_OBJECTIVE
    elif worst == Form.BLACK_BOX:
        cls = ProblemClass.BLACK_BOX
    elif integer:
        cls = ProblemClass.MILP if worst == Form.LINEAR else ProblemClass.MINLP
    elif worst == Form.NONSMOOTH:
        cls = ProblemClass.NONSMOOTH
    elif worst == Form.LINEAR:
        cls = ProblemClass.LP
    elif (forms[0] == Form.QUADRATIC and linear_constraints
          and _convex_quadratic(problem, objectives[0][1], problem.objectives[0].sense)):
        cls = ProblemClass.CONVEX_QP
    elif constraint_form is None:
        cls = ProblemClass.BOX_SMOOTH
    else:
        cls = ProblemClass.NLP
    return ProblemStructure(cls, forms, constraint_form, integer,
                            any(c.sense == "==" for c in problem.constraints), bounded, problem.domain_size())


def linear_coefficients(problem: OptimizationProblem, item: Objective | Constraint) -> tuple[dict[str, float], float]:
    """``(coefficient by variable name, constant)`` of a linear expression, in the units ``evaluate`` uses.

    Raises ``ValueError`` when the expression is not linear.
    """
    form, poly = _form(problem, item)
    if form != Form.LINEAR:
        raise ValueError(f"{item.name!r} is {form.name.lower()}, not linear")
    return ({mono[0][0]: float(c) for mono, c in poly.items() if mono},
            float(poly.get((), Fraction(0))))
