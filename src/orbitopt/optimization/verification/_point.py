"""Re-evaluate one point against a problem: domains, objectives and hard constraints, with findings."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from ..expressions import EvaluationError, ToolCall, ToolCallError
from ..problem import Evaluation, OptimizationProblem, Variable, VerificationPolicy
from .report import FindingCode, Outcome

Add = Callable[[FindingCode, str, str], None]


@dataclass(frozen=True)
class PointCheck:
    """Whether a point is feasible by the guard's own evaluation, and the values it computed."""

    outcome: Outcome
    reason: str
    objectives: Mapping[str, float] | None = None
    residuals: Mapping[str, float] | None = None


def _number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _scaled(tol: float, reference: float) -> float:
    return tol * max(1.0, abs(reference))


def check_domain(v: Variable, value: float, tol: float, path: str, add: Add) -> bool:
    """Bounds (a binary variable is an integer in [0, 1]), integrality and discrete choices."""
    if v.kind == "discrete":
        if any(abs(value - c) <= _scaled(tol, c) for c in v.choices):
            return True
        add(FindingCode.DOMAIN_VIOLATION, path, f"{value!r} is not one of the choices {list(v.choices)}")
        return False
    lower, upper = (0.0, 1.0) if v.kind == "binary" else (v.lower, v.upper)
    ok = True
    if lower is not None and value < lower - _scaled(tol, lower):
        add(FindingCode.CONSTRAINT_VIOLATION, path, f"{value!r} is below the lower bound {lower!r}")
        ok = False
    if upper is not None and value > upper + _scaled(tol, upper):
        add(FindingCode.CONSTRAINT_VIOLATION, path, f"{value!r} is above the upper bound {upper!r}")
        ok = False
    if v.kind in ("integer", "binary") and abs(value - round(value)) > tol:
        add(FindingCode.DOMAIN_VIOLATION, path, f"{value!r} is not an integer")
        ok = False
    return ok


def missing_tools(problem: OptimizationProblem, tools: Mapping[str, Any] | None) -> list[str]:
    """Tool ids the problem references that ``tools`` does not resolve."""
    items = (*problem.objectives, *problem.constraints)
    return sorted({i.expression.tool_id for i in items
                   if isinstance(i.expression, ToolCall) and i.expression.tool_id not in (tools or {})})


def check_point(problem: OptimizationProblem, x: Mapping[str, Any], policy: VerificationPolicy,
                tools: Mapping[str, Callable[..., Any]] | None, path: str, add: Add) -> PointCheck:
    """Grade ``x`` against every variable domain and hard constraint of ``problem``, evaluating it afresh.

    ``failed`` when the point is malformed, outside a domain, outside an
    expression's domain or violates a constraint beyond
    ``policy.constraint_tol``; ``not_assessed`` only when the problem cannot
    be evaluated at all (an unresolved tool, or a tool that raised).
    """
    tol = policy.constraint_tol
    if not isinstance(x, Mapping) or not x:
        add(FindingCode.DOMAIN_VIOLATION, path, "the result has no point")
        return PointCheck(Outcome.FAILED, "no point to evaluate")
    names = [v.name for v in problem.variables]
    well_formed = in_domain = True
    for extra in sorted(set(x) - set(names), key=str):
        add(FindingCode.DOMAIN_VIOLATION, f"{path}/{extra}", "not a variable of the graded problem")
        in_domain = False
    for v in problem.variables:
        where, value = f"{path}/{v.name}", x.get(v.name)
        if v.name not in x:
            add(FindingCode.DOMAIN_VIOLATION, where, "no value for this variable")
            well_formed = False
        elif not _number(value):
            add(FindingCode.DOMAIN_VIOLATION, where, f"{value!r} is not a number")
            well_formed = False
        elif not math.isfinite(value):
            add(FindingCode.NON_FINITE, where, f"{value!r} is not finite")
            well_formed = False
        elif not check_domain(v, float(value), tol, where, add):
            in_domain = False
    if not well_formed:
        return PointCheck(Outcome.FAILED, "the point is malformed or not finite")
    unresolved = missing_tools(problem, tools)
    if unresolved:
        add(FindingCode.NOT_EVALUABLE, path, f"no resolver for tools {unresolved}")
        return PointCheck(Outcome.NOT_ASSESSED, f"cannot evaluate: no resolver for tools {unresolved}")
    point = {name: float(x[name]) for name in names}
    try:
        ev = problem.evaluate(point, tools)
    except ToolCallError as exc:  # the tool could not be asked: evaluation impossible, never a pass
        add(FindingCode.EVALUATION_ERROR, path, f"evaluation failed in a tool: {exc}")
        cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "ToolCallError"
        return PointCheck(Outcome.NOT_ASSESSED, f"evaluation raised {cause}")
    except EvaluationError as exc:
        add(FindingCode.EVALUATION_ERROR, path, f"the problem is not defined at this point: {exc}")
        return PointCheck(Outcome.FAILED, "the problem is not defined at this point")
    objectives, residuals, feasible = _floats(ev.objectives), _floats(ev.residuals), in_domain
    for name, value in objectives.items():
        if not math.isfinite(value):
            add(FindingCode.NON_FINITE, f"/objectives/{name}", f"objective evaluates to {value!r}")
    for c in problem.constraints:
        r = residuals[c.name]
        violation = abs(r) if c.sense == "==" else r
        if not math.isfinite(r):
            add(FindingCode.NON_FINITE, f"/constraints/{c.name}", f"residual evaluates to {r!r}")
            feasible = False
        elif violation > _scaled(tol, c.rhs):
            add(FindingCode.CONSTRAINT_VIOLATION, f"/constraints/{c.name}",
                f"residual {r:.6g} exceeds the tolerance {_scaled(tol, c.rhs):.3g}")
            feasible = False
    # The IR's own definition of the status must agree; a pass needs both.
    feasible = feasible and replace(problem, verification=policy).constraints_satisfied(point, residuals)
    if not feasible:
        return PointCheck(Outcome.FAILED, "a bound, domain or hard constraint is violated", objectives, residuals)
    return PointCheck(Outcome.PASSED, "every domain and hard constraint holds on re-evaluation", objectives, residuals)


def _floats(values: Evaluation | Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for name, value in values.items():
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = math.nan
    return out
