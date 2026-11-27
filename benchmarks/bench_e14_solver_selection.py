"""E14-10 arms A and B: the hand-defined problem with a fixed solver (A) or the rule-table solver (B).

Both arms call scipy directly on the catalogue's Python callables through ``e14_common.Evaluator``, which enforces the
cell's evaluation and wall-clock budget, so they share no code with the E14 runtime they are compared with. The
choices the spec leaves open are in ``e14_common.PROTOCOL["arm_A"]`` and ``["arm_B"]``.

    PYTHONPATH=src python benchmarks/bench_e14_solver_selection.py --arm B --instance lp-wyndor-d2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from typing import Callable

import numpy as np
from scipy.optimize import (BFGS, Bounds, LinearConstraint, NonlinearConstraint, differential_evolution, linprog, milp,
                            minimize)

import e14_common as C

N_WEIGHTS = 50
DE_POPSIZE = 15
#: set on the tuning instance cqp-worked-d2 only (PROTOCOL["tuning_adjustments"]); scipy's defaults stopped 1.6e-4 short
TRUST_CONSTR_OPTIONS = {"gtol": 1e-12, "xtol": 1e-12, "barrier_tol": 1e-12, "initial_barrier_parameter": 1e-6}


class Run:
    """What one baseline cell did: the point or front it returns, the solvers it called, and why it stopped."""

    def __init__(self, ev: C.Evaluator):
        self.ev = ev
        self.x: list[float] | None = None
        self.front: list[list[float]] | None = None
        self.solvers: list[str] = []
        self.stop = "completed"
        self.messages: list[str] = []


def _sign(inst) -> float:
    return 1.0 if inst.sense == "minimize" else -1.0


def _constraint(ev: C.Evaluator, inst) -> list:
    if not inst.constraints:
        return []
    lb = np.array([0.0 if c.kind == "eq" else -np.inf for c in inst.constraints])
    return [NonlinearConstraint(lambda x: np.array(ev.evaluate(x)[1]), lb, np.zeros(len(lb)))]


def _de(ev: C.Evaluator, inst, seed: int, budget: int, fun: Callable | None = None) -> np.ndarray | None:
    """The fixed solver: differential evolution with the cell seed and polish, its generations fitted to ``budget``."""
    d = inst.dimension
    fun = fun or (lambda x: _sign(inst) * ev.evaluate(x)[0][0])
    integral = inst.variable_type in ("integer", "binary")
    maxiter = max(1, math.floor(0.9 * budget / (DE_POPSIZE * d)) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = differential_evolution(fun, list(inst.bounds), rng=seed, popsize=DE_POPSIZE, tol=0.01, maxiter=maxiter,
                                     polish=True, init="latinhypercube", constraints=_constraint(ev, inst) or (),
                                     integrality=[integral] * d if integral else None)
    return np.asarray(res.x, dtype=float)


def _slsqp_multistart(ev: C.Evaluator, inst, rng: np.random.Generator, fun: Callable | None = None,
                      constraints: list | None = None, dim_extra: Callable | None = None) -> None:
    """SLSQP from seeded uniform starts until the evaluator's allocation runs out (BudgetExhausted ends it)."""
    lo, hi = np.array([b[0] for b in inst.bounds]), np.array([b[1] for b in inst.bounds])
    fun = fun or (lambda x: _sign(inst) * ev.evaluate(x)[0][0])
    if constraints is None:
        constraints = [{"type": "eq" if c.kind == "eq" else "ineq",
                        "fun": (lambda x, k=k, eq=(c.kind == "eq"): (1.0 if eq else -1.0) * ev.evaluate(x)[1][k])}
                       for k, c in enumerate(inst.constraints)]
    bounds = list(inst.bounds)
    while True:
        x0 = rng.uniform(lo, hi)
        if dim_extra is not None:
            x0, extra_bounds = dim_extra(x0)
            b = bounds + extra_bounds
        else:
            b = bounds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minimize(fun, x0, method="SLSQP", bounds=b, constraints=constraints)


def _probe_linear(ev: C.Evaluator, inst):
    """Coefficients of a linear objective and linear constraints, read from the callables at 0 and the unit vectors."""
    d = inst.dimension
    f0, g0 = ev.evaluate(np.zeros(d))
    c, rows = np.zeros(d), np.zeros((len(g0), d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        fi, gi = ev.evaluate(e)
        c[i] = fi[0] - f0[0]
        rows[:, i] = np.array(gi) - np.array(g0)
    return c, rows, -np.array(g0)


def _rule_lp(run: Run, inst, milp_: bool) -> None:
    ev = run.ev
    c, rows, rhs = _probe_linear(ev, inst)
    sign = _sign(inst)
    le = [k for k, con in enumerate(inst.constraints) if con.kind == "le"]
    eq = [k for k, con in enumerate(inst.constraints) if con.kind == "eq"]
    lo, hi = [b[0] for b in inst.bounds], [b[1] for b in inst.bounds]
    if milp_:
        run.solvers.append("scipy.milp")
        cons = []
        if le:
            cons.append(LinearConstraint(rows[le], -np.inf, rhs[le]))
        if eq:
            cons.append(LinearConstraint(rows[eq], rhs[eq], rhs[eq]))
        res = milp(sign * c, integrality=np.ones(inst.dimension), bounds=Bounds(lo, hi), constraints=cons)
    else:
        run.solvers.append("scipy.linprog.highs")
        res = linprog(sign * c, A_ub=rows[le] if le else None, b_ub=rhs[le] if le else None,
                      A_eq=rows[eq] if eq else None, b_eq=rhs[eq] if eq else None, bounds=list(zip(lo, hi)),
                      method="highs")
    run.messages.append(str(res.message))
    if res.x is not None:
        ev.evaluate(res.x)
        run.x = [float(v) for v in res.x]


def _rule_enumerate(run: Run, inst) -> bool:
    """Exhaustive enumeration in increasing stated cost; False when the integer grid does not fit the budget."""
    ev = run.ev
    axes = [range(int(math.ceil(lo)), int(math.floor(hi)) + 1) for lo, hi in inst.bounds]
    size = math.prod(len(a) for a in axes)
    if size > ev.max_evaluations or inst.variable_type == "continuous" or any(
            name in inst.objective_exprs[0] for name in inst.blackboxes):
        return False
    run.solvers.append("enumeration.increasing_cost")
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T.astype(float)
    cost = np.array([_sign(inst) * C.P.evaluate_expr(inst, inst.objective_exprs[0], p) for p in grid])
    for idx in np.lexsort((*grid.T[::-1], cost)):
        point = grid[idx]
        ev.evaluate(point)
        if C.violation(inst, point) <= inst.tolerance.constraint_abs:
            run.x = [float(v) for v in point]
            return True
    run.messages.append("no feasible point in the grid")
    return True


def _tchebycheff(run: Run, inst, seed: int, solve_weight: Callable) -> None:
    """50 weights, the two extremes first; each weight gets 1/50 of the budget and keeps its best scalarised point."""
    ev = run.ev
    share = ev.max_evaluations // N_WEIGHTS
    order = [0, N_WEIGHTS - 1] + list(range(1, N_WEIGHTS - 1))
    front = []
    for k in order:
        w = np.array([k / (N_WEIGHTS - 1), 1.0 - k / (N_WEIGHTS - 1)])
        seen = [ev.cache[key][0] for key in ev.cache]
        z = np.min(np.array(seen), axis=0) if seen else np.zeros(2)
        best: list = [math.inf, None]

        def scal(x):
            fs = np.array(ev.evaluate(x[:inst.dimension])[0])
            value = float(np.max(w * (fs - z)) + 1e-4 * np.sum(fs - z))
            if value < best[0] and C.violation(inst, x[:inst.dimension]) <= inst.tolerance.constraint_abs:
                best[0], best[1] = value, [float(v) for v in x[:inst.dimension]]
            return value

        ev.limit = min(ev.max_evaluations, ev.count + share)
        try:
            solve_weight(scal, w, z, np.random.default_rng([seed, k]))
        except C.BudgetExhausted:
            pass  # this weight's share, or the cell's budget, is spent
        finally:
            ev.limit = ev.max_evaluations
        if best[1] is not None:
            front.append(best[1])
        if ev.exhausted:
            break
    run.front = front
    if ev.exhausted:
        run.stop = f"budget_exhausted:{ev.exhausted}"


def arm_a(run: Run, inst, seed: int) -> None:
    run.solvers.append("scipy.differential_evolution")
    if len(inst.objectives) > 1:
        _tchebycheff(run, inst, seed, lambda scal, w, z, rng: _de(run.ev, inst, seed, run.ev.limit - run.ev.count,
                                                                    fun=scal))
        return
    x = _de(run.ev, inst, seed, run.ev.max_evaluations)
    run.x = [float(v) for v in x]


def arm_b(run: Run, inst, seed: int) -> None:
    ev, klass = run.ev, inst.problem_class
    if klass in ("lp", "milp"):
        _rule_lp(run, inst, milp_=klass == "milp")
    elif klass == "convex_qp":
        run.solvers.append("scipy.trust-constr")
        lo, hi = np.array([b[0] for b in inst.bounds]), np.array([b[1] for b in inst.bounds])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cons = [NonlinearConstraint(c.fun, c.lb, c.ub, jac="3-point", hess=BFGS()) for c in _constraint(ev, inst)]
            res = minimize(lambda x: _sign(inst) * ev.evaluate(x)[0][0], 0.5 * (lo + hi), method="trust-constr",
                           jac="3-point", hess=BFGS(), bounds=Bounds(lo, hi), constraints=cons,
                           options=TRUST_CONSTR_OPTIONS)
        run.x = [float(v) for v in res.x]
        run.messages.append(str(res.message))
    elif klass == "nlp":
        run.solvers.append("scipy.slsqp.multistart")
        try:
            _slsqp_multistart(ev, inst, np.random.default_rng(seed))
        finally:
            run.x = ev.best_x()
    elif klass == "global":
        arm_a(run, inst, seed)
    elif klass == "black_box":
        if not _rule_enumerate(run, inst):
            arm_a(run, inst, seed)
    elif klass == "multi_objective":
        run.solvers.append("scipy.slsqp.tchebycheff")
        d = inst.dimension

        def weight(scal, w, z, rng):
            def t_con(xt):  # w_i (f_i - z_i) <= t, from one evaluation
                fs = np.array(ev.evaluate(xt[:d])[0])
                scal(xt)
                return xt[d] - w * (fs - z)

            def start(x0):
                fs = np.array(ev.evaluate(x0)[0])
                return np.append(x0, float(np.max(w * (fs - z)))), [(None, None)]
            _slsqp_multistart(ev, inst, rng, fun=lambda xt: xt[d], constraints=[{"type": "ineq", "fun": t_con}],
                              dim_extra=start)
        _tchebycheff(run, inst, seed, weight)
    else:
        raise ValueError(f"no rule for class {klass}")


ARMS = {"A": arm_a, "B": arm_b}


def run_cell(arm: str, inst, seed: int) -> dict:
    """One scored cell of arm A or B, scored with the catalogue's callables; never raises for a solver failure."""
    counted, counter = C.counted_instance(inst)
    budget = C.A.BUDGET.evaluations(inst.dimension)
    ev = C.Evaluator(counted, budget, C.A.BUDGET.wall_clock_s)
    run = Run(ev)
    started = time.perf_counter()
    error = None
    try:
        ARMS[arm](run, counted, seed)
    except C.BudgetExhausted as exc:
        run.stop = f"budget_exhausted:{exc.kind}"
    except Exception as exc:  # a solver crash is a result, not a harness failure
        run.stop, error = "error", f"{type(exc).__name__}: {exc}"
    if run.x is None and run.front is None:
        run.x = ev.best_x()
        if len(inst.objectives) > 1:
            run.front = ev.points[-C.P.MO_MAX_POINTS:]
    if ev.exhausted and run.stop == "completed":  # an allocation inside the arm ran out the whole cell budget
        run.stop = f"budget_exhausted:{ev.exhausted}"
    wall = time.perf_counter() - started
    return C.baseline_row(arm, inst, seed, run=run, evaluations=ev.count, independent=counter.calls, budget=budget,
                          wall=wall, error=error)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    row = run_cell(args.arm, C.P.by_id(args.instance), args.seed)
    print(json.dumps(C.jsonable(row), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
