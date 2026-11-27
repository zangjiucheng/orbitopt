"""Optimality certificates: a KKT check for convex QPs and LPs, and a lower-bound refutation check.

``kkt_certificate`` is the only route to a passed ``optimality_certified``.
It applies to a problem whose single objective is structurally a polynomial
of degree at most 2 with a positive semidefinite Hessian (negative
semidefinite when maximising), whose constraints are all structurally affine
and whose variables are all continuous. For that class the KKT conditions are
sufficient for global optimality. Everything else (non-convex, mixed-integer,
multi-objective, tool-evaluated) is not assessed. Global optimality of a
non-convex problem is never certified here.

``refute_lower_bound`` re-checks a ``LowerBoundCertificate`` at the point and
on seeded samples of the feasible set. Sampling can refute a bound and never
prove one, so it can only fail the status.

numpy is imported when a certificate is computed, not when this module loads.
"""

from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..expressions import Expression, ToolCall
from ..problem import OptimizationProblem, VerificationPolicy
from ._point import check_point
from .evidence import LowerBoundCertificate
from .report import Outcome

ACTIVE_FACTOR = 10.0
"""An inequality counts as active when its residual is within this many constraint tolerances of zero."""
HESSIAN_RTOL = 1e-8
"""Smallest Hessian eigenvalue accepted as non-negative, relative to the largest Hessian entry."""


@dataclass(frozen=True)
class KKTResult:
    """A KKT check outcome with the multipliers it derived (by constraint name, bounds as ``name.lower``)."""

    outcome: Outcome
    reason: str
    multipliers: Mapping[str, float] = field(default_factory=dict)


def polynomial_degree(expression: Expression | ToolCall, variables: frozenset[str]) -> int | None:
    """Structural degree of ``expression`` in ``variables``; ``None`` when it is not a polynomial.

    Parameters and literals have degree 0, so ``a*x**2`` is quadratic. A
    function call is degree 0 when every argument is, and not a polynomial
    otherwise; division is allowed only by a degree-0 expression; a power
    needs a non-negative integer literal exponent.
    """
    if not isinstance(expression, Expression):
        return None
    return _degree(expression.tree, variables)  # raises ExpressionError for an unsafe source


def _degree(node: ast.AST, variables: frozenset[str]) -> int | None:
    if isinstance(node, ast.Constant):
        return 0
    if isinstance(node, ast.Name):
        return 1 if node.id in variables else 0
    if isinstance(node, ast.UnaryOp):
        return _degree(node.operand, variables)
    if isinstance(node, ast.Call):
        return 0 if all(_degree(a, variables) == 0 for a in node.args) else None
    if not isinstance(node, ast.BinOp):
        return None
    left, right = _degree(node.left, variables), _degree(node.right, variables)
    if left is None or right is None:
        return None
    if isinstance(node.op, (ast.Add, ast.Sub)):
        return max(left, right)
    if isinstance(node.op, ast.Mult):
        return left + right
    if isinstance(node.op, ast.Div):
        return left if right == 0 else None
    if left == 0 and right == 0:
        return 0
    power = node.right.value if isinstance(node.right, ast.Constant) else None
    if isinstance(power, (int, float)) and float(power).is_integer() and power >= 0:
        return left * int(power)
    return None


def unsupported_reason(problem: OptimizationProblem) -> str | None:
    """Why the KKT certificate cannot apply to ``problem``; ``None`` when the structure qualifies."""
    if len(problem.objectives) != 1:
        return "only single-objective problems can be certified"
    discrete = [v.name for v in problem.variables if v.kind != "continuous"]
    if discrete:
        return f"variables {discrete} are not continuous; KKT conditions do not certify mixed-integer problems"
    variables = frozenset(v.name for v in problem.variables)
    objective = problem.objectives[0]
    degree = polynomial_degree(objective.expression, variables)
    if degree is None or degree > 2:
        return f"objective {objective.name!r} is not structurally a polynomial of degree <= 2"
    for c in problem.constraints:
        degree = polynomial_degree(c.expression, variables)
        if degree is None or degree > 1:
            return f"constraint {c.name!r} is not structurally affine"
    return None


def kkt_certificate(problem: OptimizationProblem, x: Mapping[str, float], policy: VerificationPolicy) -> KKTResult:
    """Check the KKT conditions at a feasible ``x`` of a convex QP or LP.

    Derivatives come from central differences with step ``max(1, |x_i|)``,
    exact for quadratics up to rounding. Convexity is the Hessian's smallest
    eigenvalue being at least ``-HESSIAN_RTOL`` times its largest entry. The
    multipliers solve a least-squares problem on the active set; while one
    inequality multiplier is negative, that constraint leaves the set and the
    solve repeats. Stationarity must hold to ``optimality_gap * max(1,
    |grad f|)`` and complementarity to ``optimality_gap * max(1, |f|)``.
    The caller has already established primal feasibility.
    """
    reason = unsupported_reason(problem)
    if reason is not None:
        return KKTResult(Outcome.NOT_ASSESSED, reason)
    import numpy as np

    names = [v.name for v in problem.variables]
    objective = problem.objectives[0]
    sign = 1.0 if objective.sense == "minimize" else -1.0

    def evaluate(z):
        ev = problem.evaluate(dict(zip(names, map(float, z))))
        return sign * float(ev.objectives[objective.name]), {k: float(v) for k, v in ev.residuals.items()}

    x0 = np.array([float(x[n]) for n in names])
    n = len(names)
    step = np.maximum(1.0, np.abs(x0))
    basis = np.diag(step)
    f0, r0 = evaluate(x0)
    plus = [evaluate(x0 + basis[i]) for i in range(n)]
    minus = [evaluate(x0 - basis[i]) for i in range(n)]
    grad = np.array([(plus[i][0] - minus[i][0]) / (2 * step[i]) for i in range(n)])
    hess = np.diag([(plus[i][0] - 2 * f0 + minus[i][0]) / step[i] ** 2 for i in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            corners = [evaluate(x0 + si * basis[i] + sj * basis[j])[0] for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            hess[i, j] = hess[j, i] = (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * step[i] * step[j])
    scale = max(1.0, float(np.max(np.abs(hess))))
    smallest = float(np.min(np.linalg.eigvalsh(hess))) if n else 0.0
    if smallest < -HESSIAN_RTOL * scale:
        return KKTResult(Outcome.NOT_ASSESSED,
                         f"objective Hessian has eigenvalue {smallest:.3g} < 0: not convex, so KKT cannot certify")

    rows = []  # (name, residual, gradient, is_equality, rhs scale)
    for c in problem.constraints:
        g = np.array([(plus[i][1][c.name] - minus[i][1][c.name]) / (2 * step[i]) for i in range(n)])
        rows.append((c.name, r0[c.name], g, c.sense == "==", c.rhs))
    for i, v in enumerate(problem.variables):
        if v.lower is not None:
            rows.append((f"{v.name}.lower", v.lower - x0[i], -np.eye(n)[i], False, v.lower))
        if v.upper is not None:
            rows.append((f"{v.name}.upper", x0[i] - v.upper, np.eye(n)[i], False, v.upper))
    tol = policy.constraint_tol
    active = [k for k, (_, r, _, eq, rhs) in enumerate(rows)
              if eq or r >= -ACTIVE_FACTOR * tol * max(1.0, abs(rhs))]

    while True:
        if active:
            jac = np.column_stack([rows[k][2] for k in active])
            mult = np.linalg.lstsq(jac, -grad, rcond=None)[0]
        else:
            jac, mult = np.zeros((n, 0)), np.zeros(0)
        negative = [(m, pos) for pos, (k, m) in enumerate(zip(active, mult)) if not rows[k][3] and m < 0]
        if not negative:
            break
        active.pop(min(negative)[1])

    stationarity = float(np.max(np.abs(grad + jac @ mult))) if n else 0.0
    stat_tol = policy.optimality_gap * max(1.0, float(np.max(np.abs(grad))) if n else 1.0)
    multipliers = {rows[k][0]: float(m) for k, m in zip(active, mult)}
    if stationarity > stat_tol:
        return KKTResult(Outcome.FAILED, f"stationarity residual {stationarity:.3g} exceeds {stat_tol:.3g}: "
                                         "no non-negative multipliers on the active set make the gradient vanish",
                         multipliers)
    comp_tol = policy.optimality_gap * max(1.0, abs(f0))
    worst = max((abs(m * rows[k][1]) for k, m in zip(active, mult) if not rows[k][3]), default=0.0)
    if worst > comp_tol:
        return KKTResult(Outcome.FAILED, f"complementarity {worst:.3g} exceeds {comp_tol:.3g}", multipliers)
    return KKTResult(Outcome.PASSED, f"KKT conditions hold for a convex problem: stationarity {stationarity:.3g}, "
                                     f"{len(active)} active constraints, multipliers non-negative", multipliers)


def refute_lower_bound(problem: OptimizationProblem, certificate: LowerBoundCertificate, objective: str,
                       x: Mapping[str, float], value: float, policy: VerificationPolicy,
                       tools: Mapping[str, Callable[..., Any]] | None, seed: int) -> str | None:
    """Try to refute ``certificate`` at ``x`` (objective ``value``) and on seeded feasible samples.

    Returns why the bound is refuted or not attained, or ``None`` when it
    survived. A bound below ``value`` by more than ``optimality_gap`` is not
    attained; a feasible sample below the bound by more than
    ``objective_rtol`` refutes it (both mirrored when maximising).
    """
    bound = float(certificate.bound)
    if not math.isfinite(bound):
        return f"bound {certificate.bound!r} is not finite"
    sign = 1.0 if next(o for o in problem.objectives if o.name == objective).sense == "minimize" else -1.0
    gap = sign * (value - bound)
    if gap > policy.optimality_gap * max(1.0, abs(bound)):
        return f"the result's objective {value:.12g} does not attain the bound {bound:.12g}"
    tol = policy.objective_rtol * max(1.0, abs(bound))
    if gap < -tol:
        return f"the result's own feasible point has objective {value:.12g}, beyond the bound {bound:.12g}"
    rng = random.Random(seed)
    samples = max(16, min(int(certificate.samples), 4096))
    for k in range(samples):
        local = k % 2 == 1  # alternate global samples with perturbations of the result's point
        point = {v.name: _sample(v, float(x[v.name]), rng, local) for v in problem.variables}
        check = check_point(problem, point, policy, tools, "", lambda *_: None)
        if check.outcome is Outcome.PASSED:
            sample_value = check.objectives[objective]
            if math.isfinite(sample_value) and sign * (sample_value - bound) < -tol:
                return f"feasible sample {point} has objective {sample_value:.12g}, beyond the bound {bound:.12g}"
    return None


def _sample(v, at: float, rng: random.Random, local: bool) -> float:
    if v.kind == "binary":
        return float(rng.randint(0, 1))
    if v.kind == "discrete":
        return float(rng.choice(v.choices))
    span = max(1.0, abs(at)) * (1e-2 if local else 10.0)
    lo = v.lower if v.lower is not None and not local else at - span
    hi = v.upper if v.upper is not None and not local else at + span
    if v.lower is not None:
        lo = max(lo, v.lower)
    if v.upper is not None:
        hi = min(hi, v.upper)
    if v.kind == "integer":
        lo, hi = math.ceil(lo), math.floor(hi)
        return float(rng.randint(lo, hi)) if lo <= hi else float(round(at))
    return rng.uniform(lo, hi)
