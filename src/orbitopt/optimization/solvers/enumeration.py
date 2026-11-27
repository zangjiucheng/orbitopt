"""Exhaustive enumeration of a finite domain: exact for any objective, evaluated through the IR.

The solver takes a problem whose every variable has finitely many values
(binary, discrete, or integer with both bounds) and at most
``MAX_ENUMERATION_POINTS`` points in all. It evaluates every point through
the shared ``Evaluator``, so each one counts against the budget, a
``ToolCall`` objective or constraint works as it does for any solver, and the
best point is chosen by the IR's own feasibility. Having seen every point,
its convergence is a proof: the best feasible point is the global optimum
within the constraint tolerance, and no feasible point is a proof of
infeasibility. A budget that cannot cover the domain is refused before the
first evaluation, since a partial enumeration proves nothing.
"""

from __future__ import annotations

import itertools

from ..classify import Form
from ..problem import MAX_ENUMERATION_POINTS
from ..result import Termination
from .base import AdapterSolver, OptionSpec, Outcome, Run, SolverCapability


class EnumerationSolver(AdapterSolver):
    """Evaluate every point of a finite domain and report the best feasible one, exactly.

    ``max_points`` lowers the cap for one solve; a domain larger than it, or
    than ``budget.max_evaluations``, is refused with no evaluation and
    termination ``max_evaluations``. A point whose tool failed leaves the
    enumeration unproven: the solve then stops ``stalled``, not
    ``converged``.
    """

    CAPABILITY = SolverCapability(
        solver_id="orbitopt.enumeration", backend="itertools", max_form=Form.BLACK_BOX, convergence="exhaustive",
        supports_integers=True, supports_constraints=True, supports_equality=True, is_global=True,
        max_domain_size=MAX_ENUMERATION_POINTS,
        options={"max_points": OptionSpec(int, 1, MAX_ENUMERATION_POINTS)})

    def _run(self, run: Run) -> Outcome:
        size = run.structure.domain_size
        cap = int(run.options.get("max_points", MAX_ENUMERATION_POINTS))
        allowed = run.budget.max_evaluations
        if size > cap:
            return Outcome(stop=Termination.MAX_EVALUATIONS,
                           message=f"the domain has {size} points, more than max_points {cap}; nothing evaluated")
        if allowed is not None and size > allowed:
            return Outcome(stop=Termination.MAX_EVALUATIONS,
                           message=f"the domain has {size} points, more than the {allowed} evaluations the budget "
                                   "allows; nothing evaluated")
        axes = [range(len(v.choices)) if v.kind == "discrete" else v.domain_values() for v in run.space.variables]
        for vector in itertools.product(*axes):  # a discrete coordinate is its choice's index, as in ``Space``
            run.evaluator(vector)
        failed = run.evaluator.tool_failures
        if failed:
            return Outcome(stop=Termination.STALLED, iterations=size,
                           message=f"{failed} of {size} points could not be evaluated (a tool failed), so the "
                                   "enumeration proves nothing")
        return Outcome(converged=True, iterations=size, message=f"evaluated all {size} points of the finite domain")
