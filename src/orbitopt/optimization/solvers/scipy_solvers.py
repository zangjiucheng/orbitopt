"""Solvers backed by scipy.optimize: HiGHS, SLSQP, L-BFGS-B and differential evolution.

scipy and numpy are imported inside ``_run`` only, so importing this module
stays numpy-free. Gradient-based methods use scipy's finite differences, and
each difference point is a counted evaluation.
"""

from __future__ import annotations

import math

from ..classify import Form, linear_coefficients
from ..result import Termination
from .base import AdapterSolver, OptionSpec, Outcome, Run, SolverCapability

DEFAULT_MIP_REL_GAP = 1e-9
"""HiGHS stops a MILP when the relative gap is below this; scipy's default (1e-4) can stop short of the optimum."""
DEFAULT_SLSQP_FTOL = 1e-10
"""SLSQP's precision goal on the objective; scipy's default (1e-6) is coarser than the E14-10 tolerances."""
LBFGSB_FTOL = 2.220446049250313e-09
LBFGSB_GTOL = 1e-5
"""scipy's own L-BFGS-B defaults, which the adapter keeps."""
DE_STRATEGIES = ("best1bin", "best1exp", "rand1bin", "rand1exp", "rand2bin", "rand2exp", "randtobest1bin",
                 "randtobest1exp", "currenttobest1bin", "currenttobest1exp", "best2exp", "best2bin")


def _bounds(run: Run) -> list[tuple[float | None, float | None]]:
    return [(lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None)
            for lo, hi in zip(run.space.lower, run.space.upper)]


def _constraint_band(run: Run) -> tuple[list[float], list[float]]:
    """Bounds on each canonical residual: ``g <= tol`` and ``|h| <= tol``, the IR's own feasibility band."""
    problem = run.problem
    band = [c.allowance(problem.verification.constraint_tol) for c in problem.constraints]
    lower = [-b if c.sense == "==" else -math.inf for c, b in zip(problem.constraints, band)]
    return lower, band


class HighsSolver(AdapterSolver):
    """LP and MILP through HiGHS (``scipy.optimize.milp``), with discrete variables expanded into binaries.

    HiGHS reads the coefficients, not the evaluator, so a run costs one
    evaluation: the IR's own evaluation of the point HiGHS returns.
    """

    CAPABILITY = SolverCapability(
        solver_id="scipy.highs", backend="scipy", max_form=Form.LINEAR, convergence="certificate",
        supports_integers=True, supports_constraints=True, supports_equality=True, is_global=True,
        options={"mip_rel_gap": OptionSpec(float, 0.0, 1e-4, tolerance=True, default=DEFAULT_MIP_REL_GAP),
                 "presolve": OptionSpec(bool)})

    def _run(self, run: Run) -> Outcome:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        problem, space = run.problem, run.space
        n = len(space.variables)
        lower, upper, integral = list(space.lower), list(space.upper), list(space.integral)
        rows: list[tuple[dict[int, float], float, float]] = []
        indicators: dict[int, list[int]] = {}
        for i, v in enumerate(space.variables):
            if v.kind != "discrete":
                continue
            lower[i], upper[i], integral[i] = -math.inf, math.inf, False
            first = len(lower)
            indicators[i] = list(range(first, first + len(v.choices)))
            lower += [0.0] * len(v.choices)
            upper += [1.0] * len(v.choices)
            integral += [True] * len(v.choices)
            rows.append(({i: 1.0, **{first + k: -float(c) for k, c in enumerate(v.choices)}}, 0.0, 0.0))
            rows.append(({first + k: 1.0 for k in range(len(v.choices))}, 1.0, 1.0))
        column = {v.name: i for i, v in enumerate(space.variables)}

        def dense(coefs: dict[str, float]) -> np.ndarray:
            out = np.zeros(len(lower))
            for name, value in coefs.items():
                out[column[name]] = value
            return out

        objective = problem.objectives[0]
        coefs, _ = linear_coefficients(problem, objective)
        c = dense(coefs) * (-1.0 if objective.sense == "maximize" else 1.0)
        matrix, lo, hi = [], [], []
        for entries, row_lo, row_hi in rows:
            row = np.zeros(len(lower))
            for j, value in entries.items():
                row[j] = value
            matrix.append(row)
            lo.append(row_lo)
            hi.append(row_hi)
        for con in problem.constraints:
            coefs, constant = linear_coefficients(problem, con)
            target = con.rhs - constant
            matrix.append(dense(coefs))
            lo.append(target if con.sense in (">=", "==") else -math.inf)
            hi.append(target if con.sense in ("<=", "==") else math.inf)
        options = {"presolve": run.options.get("presolve", True),
                   "mip_rel_gap": run.options.get("mip_rel_gap", DEFAULT_MIP_REL_GAP)}
        wall = run.remaining_wall_s()
        if wall is not None:
            options["time_limit"] = wall
        res = milp(c, integrality=np.array(integral, dtype=int), bounds=Bounds(lower, upper),
                   constraints=LinearConstraint(np.array(matrix), lo, hi) if matrix else None, options=options)
        points = ()
        if res.x is not None:
            vector = [float(round(res.x[i])) if space.integral[i] else float(res.x[i]) for i in range(n)]
            for i, cols in indicators.items():
                vector[i] = float(int(np.argmax(res.x[cols])))
            points = (tuple(vector),)
        if res.status == 1 and wall is not None and "time" in res.message.lower():
            stop = Termination.TIMEOUT
        else:
            stop = {0: None, 1: Termination.MAX_ITERATIONS, 2: Termination.INFEASIBLE,
                    3: Termination.UNBOUNDED}.get(res.status, Termination.ERROR)
        return Outcome(converged=res.status == 0, stop=stop, message=res.message, points=points)


class SlsqpSolver(AdapterSolver):
    """Sequential least squares programming (``scipy.optimize.minimize(method='SLSQP')``) from one start."""

    CAPABILITY = SolverCapability(
        solver_id="scipy.slsqp", backend="scipy", max_form=Form.SMOOTH, convergence="first_order", needs_gradients=True,
        supports_constraints=True, supports_equality=True,
        iteration_cap=True,
        options={"ftol": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=DEFAULT_SLSQP_FTOL)})

    def _run(self, run: Run) -> Outcome:
        from scipy.optimize import minimize

        ev, constraints = run.evaluator, run.problem.constraints
        ineq = [i for i, c in enumerate(constraints) if c.sense != "=="]
        eq = [i for i, c in enumerate(constraints) if c.sense == "=="]
        cons = []
        if ineq:
            cons.append({"type": "ineq", "fun": lambda v: [-ev(v).g[i] for i in ineq]})
        if eq:
            cons.append({"type": "eq", "fun": lambda v: [ev(v).g[i] for i in eq]})
        options = {"ftol": run.options.get("ftol", DEFAULT_SLSQP_FTOL)}
        if run.max_backend_iterations is not None:
            options["maxiter"] = run.max_backend_iterations
        res = minimize(ev.objective, run.start, method="SLSQP", bounds=_bounds(run), constraints=cons,
                       options=options)
        stop = {9: Termination.MAX_ITERATIONS, 8: Termination.STALLED}.get(res.status)  # 8: line search failed
        return Outcome(converged=bool(res.success), stop=stop, message=str(res.message), iterations=int(res.nit))


class LbfgsbSolver(AdapterSolver):
    """Limited-memory BFGS with bounds (``scipy.optimize.minimize(method='L-BFGS-B')``) from one start."""

    CAPABILITY = SolverCapability(
        solver_id="scipy.lbfgsb", backend="scipy", max_form=Form.SMOOTH, convergence="first_order",
        needs_gradients=True, iteration_cap=True,
        options={"ftol": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=LBFGSB_FTOL),
                 "gtol": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=LBFGSB_GTOL)})

    def _run(self, run: Run) -> Outcome:
        from scipy.optimize import minimize

        options = {"maxfun": 10**12, **{k: run.options[k] for k in ("ftol", "gtol") if k in run.options}}
        if run.max_backend_iterations is not None:
            options["maxiter"] = run.max_backend_iterations
        res = minimize(run.evaluator.objective, run.start, method="L-BFGS-B", bounds=_bounds(run), options=options)
        stop = {1: Termination.MAX_ITERATIONS, 2: Termination.STALLED}.get(res.status)  # 2: abnormal line search
        return Outcome(converged=bool(res.success), stop=stop, message=str(res.message), iterations=int(res.nit))


class DifferentialEvolutionSolver(AdapterSolver):
    """Differential evolution (``scipy.optimize.differential_evolution``): global, seeded, integer-aware.

    Constraints use the IR's feasibility band, so a point DE calls feasible
    is one ``constraints_satisfied`` accepts. ``polish`` (default on) refines
    the best member with a local method inside the same budget.
    """

    CAPABILITY = SolverCapability(
        solver_id="scipy.differential_evolution", backend="scipy", max_form=Form.BLACK_BOX,
        convergence="heuristic", supports_integers=True,
        supports_constraints=True, supports_equality=True, needs_finite_bounds=True, is_global=True,
        iteration_cap=True,
        options={"popsize": OptionSpec(int, 4, 100), "tol": OptionSpec(float, 0.0, 0.1, tolerance=True, default=0.01),
                 "atol": OptionSpec(float, 0.0, 0.1), "mutation": OptionSpec(float, 0.1, 1.9),
                 "recombination": OptionSpec(float, 0.0, 1.0), "polish": OptionSpec(bool),
                 "strategy": OptionSpec(str, choices=DE_STRATEGIES),
                 "init": OptionSpec(str, choices=("latinhypercube", "sobol", "halton", "random"))})

    def _run(self, run: Run) -> Outcome:
        from scipy.optimize import NonlinearConstraint, differential_evolution

        ev, space = run.evaluator, run.space
        kwargs = {k: run.options[k] for k in ("popsize", "tol", "atol", "mutation", "recombination", "polish",
                                               "strategy", "init") if k in run.options}
        if run.problem.constraints:
            lower, upper = _constraint_band(run)
            kwargs["constraints"] = NonlinearConstraint(ev.residuals, lower, upper)
        if any(space.integral):
            kwargs["integrality"] = list(space.integral)
        if run.x0_given:
            kwargs["x0"] = run.start
        if run.max_backend_iterations is not None:
            kwargs["maxiter"] = run.max_backend_iterations
        res = differential_evolution(ev.objective, list(zip(space.lower, space.upper)), rng=run.seed,
                                     updating="immediate", workers=1, **kwargs)
        limited = "maximum number of iterations" in str(res.message).lower()
        return Outcome(converged=bool(res.success), stop=Termination.MAX_ITERATIONS if limited else None,
                       message=str(res.message), iterations=int(res.nit))
