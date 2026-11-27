"""Turn an untrusted proposal into a ``Decision``, or into the list of reasons it is refused.

Every planner's output, the rule-based planner's included, goes through
``validate_proposal``. A proposal may name only the fields of
``PROPOSAL_FIELDS``. A field that names part of the problem (the objective,
a constraint, a bound, a parameter, the budget, the verification policy) is
a ``problem_edit`` violation; it is recorded and never applied, because no
field of a ``Decision`` can carry it.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping

from orbitopt.optimization import OptimizationProblem
from orbitopt.optimization.solvers import SolverCapability

from .model import Action, Decision, Policy, Violation, ViolationCode

PROPOSAL_FIELDS = frozenset({"action", "solver_id", "options", "x0", "seed", "evaluations", "max_iterations",
                             "rationale"})
PROBLEM_FIELDS = frozenset({
    "objective", "objectives", "sense", "constraint", "constraints", "rhs", "bound", "bounds", "lower", "upper",
    "variable", "variables", "domain", "parameter", "parameters", "budget", "limits", "max_tokens", "max_wall_s",
    "max_evaluations", "verification", "policy", "required", "constraint_tol", "objective_rtol", "optimality_gap",
    "independent_methods", "problem", "formulation", "relax", "relaxation", "tolerance",
})
"""Field names that would change what is asked, or how the answer is judged: always a ``problem_edit``."""

MAX_SEED = 2**31 - 1
MAX_BACKEND_ITERATIONS = 100_000


def validate_proposal(raw: Any, *, problem: OptimizationProblem, solvers: Mapping[str, SolverCapability],
                      known_solvers: frozenset[str], policy: Policy, available_evaluations: int,
                      default_seed: int) -> Decision | tuple[Violation, ...]:
    """The ``Decision`` a proposal states, or every reason to refuse it.

    ``solvers`` are the capabilities the planner may choose (available,
    compatible and allowed by ``policy``), by id; ``known_solvers`` every
    registered id, to tell an unknown solver from one that is not allowed.
    ``available_evaluations`` caps the evaluations one decision may ask for.
    """
    if not isinstance(raw, Mapping):
        return (Violation(ViolationCode.MALFORMED, "", f"a proposal is an object, not {type(raw).__name__}"),)
    out: list[Violation] = []

    def refuse(code: ViolationCode, path: str, message: str) -> None:
        out.append(Violation(code, path, message))

    for key in sorted(set(raw) - PROPOSAL_FIELDS, key=str):
        if str(key).lower() in PROBLEM_FIELDS:
            refuse(ViolationCode.PROBLEM_EDIT, f"/{key}", "a decision may not change the problem, its budget or "
                                                          "its verification policy")
        else:
            refuse(ViolationCode.UNKNOWN_FIELD, f"/{key}", f"not a decision field; fields: {sorted(PROPOSAL_FIELDS)}")
    action = raw.get("action")
    if action not in (Action.SOLVE.value, Action.STOP.value):
        refuse(ViolationCode.MALFORMED, "/action", "action must be 'solve' or 'stop'")
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        refuse(ViolationCode.MALFORMED, "/rationale", "rationale must be a string")
        rationale = ""
    if action == Action.STOP.value:
        return tuple(out) if out else Decision(Action.STOP, rationale=rationale)
    if action != Action.SOLVE.value:
        return tuple(out)

    solver_id = raw.get("solver_id")
    capability = solvers.get(solver_id) if isinstance(solver_id, str) else None
    if capability is None:
        if isinstance(solver_id, str) and solver_id in known_solvers:
            refuse(ViolationCode.SOLVER_NOT_ALLOWED, "/solver_id",
                   f"{solver_id!r} is not available, cannot take this problem or is not allowed by the policy")
        else:
            refuse(ViolationCode.UNKNOWN_SOLVER, "/solver_id",
                   f"no solver {solver_id!r}; choose from {sorted(solvers)}")
    options = _options(raw.get("options"), capability, policy, refuse)
    x0 = _x0(raw.get("x0"), problem, policy, refuse)
    seed = _integer(raw, "seed", default_seed, 0, MAX_SEED, ViolationCode.MALFORMED, refuse)
    evaluations = _integer(raw, "evaluations", max(1, available_evaluations // 2), 1, max(1, available_evaluations),
                           ViolationCode.BUDGET_OVERREACH, refuse)
    max_iterations = _integer(raw, "max_iterations", None, 1, MAX_BACKEND_ITERATIONS,
                              ViolationCode.BUDGET_OVERREACH, refuse)
    if max_iterations is not None and capability is not None and "max_iterations" not in capability.honors:
        refuse(ViolationCode.INVALID_OPTION, "/max_iterations", f"{capability.solver_id} has no iteration limit")
    if out:
        return tuple(out)
    return Decision(Action.SOLVE, solver_id, options, x0, seed, evaluations, max_iterations, rationale)


def _options(value: Any, capability: SolverCapability | None, policy: Policy, refuse) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        refuse(ViolationCode.INVALID_OPTION, "/options", "options must be an object")
        return {}
    if value and not policy.allow_options:
        refuse(ViolationCode.INVALID_OPTION, "/options", "the policy does not allow solver options")
        return {}
    options = {}
    for name, setting in sorted(value.items(), key=lambda kv: str(kv[0])):
        path = f"/options/{name}"
        spec = None if capability is None else capability.options.get(name)
        if str(name).lower() in PROBLEM_FIELDS:
            refuse(ViolationCode.PROBLEM_EDIT, path, "an option may not change the problem or its verification")
        elif capability is None:
            continue  # the solver is already refused; its options have no schema to check against
        elif spec is None:
            refuse(ViolationCode.INVALID_OPTION, path,
                   f"{capability.solver_id} takes options {sorted(capability.options)}")
        elif (why := spec.errors(setting)) is not None:
            refuse(ViolationCode.INVALID_OPTION, path, why)
        else:
            options[name] = int(setting) if spec.kind is int else float(setting) if spec.kind is float else setting
    return options


def _x0(value: Any, problem: OptimizationProblem, policy: Policy, refuse) -> dict | None:
    if value is None:
        return None
    if not policy.allow_x0:
        refuse(ViolationCode.X0_OUT_OF_DOMAIN, "/x0", "the policy does not allow a start point")
        return None
    if not isinstance(value, Mapping):
        refuse(ViolationCode.MALFORMED, "/x0", "x0 must map variable names to numbers")
        return None
    variables = {v.name: v for v in problem.variables}
    tol = problem.verification.constraint_tol
    x0 = {}
    for name, number in sorted(value.items(), key=lambda kv: str(kv[0])):
        path = f"/x0/{name}"
        variable = variables.get(name)
        if variable is None:
            refuse(ViolationCode.X0_UNKNOWN_VARIABLE, path, "not a variable of the problem")
        elif isinstance(number, bool) or not isinstance(number, numbers.Real) or not math.isfinite(number):
            refuse(ViolationCode.X0_OUT_OF_DOMAIN, path, "must be a finite number")
        elif reasons := variable.domain_violations(float(number), tol):
            refuse(ViolationCode.X0_OUT_OF_DOMAIN, path, f"{number!r} is outside the domain: {', '.join(reasons)}")
        else:
            x0[name] = float(number)
    return x0


def _integer(raw: Mapping, key: str, default, low: int, high: int, code: ViolationCode, refuse):
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) \
            or value != int(value):
        refuse(ViolationCode.MALFORMED, f"/{key}", f"{key} must be an integer")
        return default
    if not (low <= value <= high):
        refuse(code, f"/{key}", f"{key} {value!r} is outside [{low}, {high}]")
        return default
    return int(value)
