"""A point that buys objective with constraint violation inside the tolerance band never wins on that purchase.

The first E14-10 campaign had 19 E-rule cells on ``cqp-projection-d2`` and ``cqp-equality-d3`` whose returned
point was feasible only inside the 1e-6 band and "beat" the optimum by exactly multiplier x violation. Two places
ranked such points by raw objective: the solver adapters' best point and the runtime's incumbent. These tests pin
the two instances, the adapters' restoration onto the constraints, the runtime's clean-first ranking, and the
campaign scorer's reading of spec section 2 (a gap below minus the tolerance is an alarm, not a success).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

import bench_e14_agent_iteration as ERULE  # noqa: E402
import e14_common as C  # noqa: E402

from orbitopt.agent.runtime import Proposal, RecordedPlanner, solve  # noqa: E402
from orbitopt.optimization import (Budget, Constraint, Objective, OptimizationProblem, OptimizationResult,  # noqa: E402
                                   Parameter, Termination, Variable)
from orbitopt.optimization.solvers import (SELECTION_FRACTION, AdapterSolver, DifferentialEvolutionSolver,  # noqa: E402
                                           Outcome, Run, SlsqpSolver, SolverRegistry, band_fraction)

TOL = 1e-6


# ------------------------------------------------------------------------------------- the two campaign instances
@pytest.mark.parametrize("seed", [0, 4, 8])
def test_cqp_projection_d2_returns_the_point_on_the_half_plane(seed, tmp_path):
    """v1: SLSQP stopped 2**-26 outside x + y <= 1, and the cell "beat" f* = 2 by 2.98e-8 (multiplier 2)."""
    row = ERULE.run_cell("E-rule", C.P.by_id("cqp-projection-d2"), seed, tmp_path)
    assert not row["alarm"] and row["success"]
    assert row["max_violation"] <= 1e-12 and abs(row["gap"]) <= 1e-12
    assert row["replay"]["equal"]


@pytest.mark.parametrize("seed", [0, 2, 5, 9])
def test_cqp_equality_d3_keeps_the_exact_point_and_verifies_it(seed, tmp_path):
    """v1: the differential-evolution alternate found a point at the band edge, refuted SLSQP's exact (6, 3, 2)/11
    and became the run's best with gap -1.08e-6 = 12/11 x 9.9e-7. Restored onto x + y + z = 1 it agrees."""
    row = ERULE.run_cell("E-rule", C.P.by_id("cqp-equality-d3"), seed, tmp_path)
    assert not row["alarm"] and row["success"]
    assert row["max_violation"] <= 1e-12
    assert row["verification"]["run_status"] == "verified"
    assert row["replay"]["equal"]


def test_the_scorer_counts_an_alarm_as_no_success():
    """Spec section 2: a gap below minus the tolerance is not a win. The v1 scorer flagged it and still counted it."""
    inst = C.P.by_id("cqp-equality-d3")
    edge = [v - 9.9e-7 / 3 for v in (6 / 11, 3 / 11, 2 / 11)]  # x + y + z = 1 - 9.9e-7: inside the 1e-6 band
    sc = C.score(inst, x=edge)
    assert sc["feasible"] and sc["alarm"] and sc["gap"] < -C.allowed_gap(inst)
    assert sc["success"] is False
    exact = C.score(inst, x=[6 / 11, 3 / 11, 2 / 11])
    assert exact["success"] and not exact["alarm"]


# ------------------------------------------------------------------------------------- random linear QPs
def projection(rng, dimension: int, sense: str):
    """min |x - c|^2 subject to a.x <sense> b, with the constraint active: the exact optimum, its multiplier and
    the problem. ``b`` stays away from 0, so the band is ``TOL * |b|`` wide."""
    c = rng.uniform(-2.0, 2.0, dimension)
    a = rng.uniform(0.5, 2.0, dimension) * rng.choice([-1.0, 1.0], dimension)
    b = float(a @ c) - rng.uniform(0.5, 2.0) * float(np.sign(a @ c) or 1.0)
    if abs(b) < 0.5:
        b += 1.0 if b >= 0 else -1.0
    shift = (float(a @ c) - b) / float(a @ a)
    optimum = c - shift * a
    multiplier = 2.0 * abs(shift)
    if sense == "<=" and shift < 0:  # the optimum must lie on the boundary: c outside the half-space
        a, b = -a, -b
    names = [f"x{i}" for i in range(dimension)]
    problem = OptimizationProblem(
        id=f"projection-{dimension}-{sense}",
        variables=tuple(Variable(n, lower=-10.0, upper=10.0) for n in names),
        parameters=tuple(Parameter(f"c{i}", float(v)) for i, v in enumerate(c)),
        objectives=(Objective("f", " + ".join(f"({n} - c{i})**2" for i, n in enumerate(names))),),
        constraints=(Constraint("plane", " + ".join(f"{float(w)!r}*{n}" for w, n in zip(a, names)), sense, b),))
    return problem, dict(zip(names, map(float, optimum))), a, b, multiplier, c


class Fixed:
    """A solver that answers with one fixed point, evaluated honestly on the problem it is given."""

    def __init__(self, solver_id: str, point: dict[str, float]):
        self.solver_id, self.point = solver_id, point

    @property
    def capability(self):
        return replace(SlsqpSolver.CAPABILITY, solver_id=self.solver_id, options={})

    def solve(self, problem, *, budget=None, seed=0, x0=None, options=None, tools=None, max_backend_iterations=None):
        ev = problem.evaluate(self.point, tools)
        return OptimizationResult(
            problem_digest=problem.digest(), solver_id=self.solver_id, termination=Termination.CONVERGED,
            solver_converged=True, constraints_satisfied=problem.constraints_satisfied(self.point, ev.residuals),
            x=dict(self.point), objective_values=dict(ev.objectives), constraint_residuals=dict(ev.residuals),
            evaluations=1)


CASES = [(seed, dimension, sense) for seed in range(6) for dimension in (2, 3, 4) for sense in ("<=", "==")]


@pytest.mark.parametrize("seed,dimension,sense", CASES)
def test_a_band_point_whose_gain_its_violation_buys_never_replaces_a_clean_point(seed, dimension, sense):
    """Property: q uses a fraction u of the band, u in (SELECTION_FRACTION, 1], so its objective gain over the exact
    optimum p is at most multiplier x violation. Whatever order the planner meets them in, the run returns p."""
    rng = np.random.default_rng(1000 + seed)
    problem, p, a, b, multiplier, c = projection(rng, dimension, sense)
    u = float(rng.uniform(10 * SELECTION_FRACTION, 1.0))
    violation = u * TOL * abs(b)
    direction = float(np.sign(a @ c - b)) * a  # toward c, past the plane: the side that pays
    q = {n: v + violation * float(w) / float(a @ a) for (n, v), w in zip(p.items(), direction)}
    ev_p, ev_q = problem.evaluate(p), problem.evaluate(q)
    gain = ev_p.objectives["f"] - ev_q.objectives["f"]
    assert 0 < gain <= multiplier * violation * (1 + 1e-9)  # the purchase the band allows, and no more
    assert band_fraction(problem, ev_q.residuals) > SELECTION_FRACTION >= band_fraction(problem, ev_p.residuals)
    order = [("test.edge", q), ("test.exact", p)] if seed % 2 else [("test.exact", p), ("test.edge", q)]
    registry = SolverRegistry([Fixed(sid, point) for sid, point in order])
    planner = RecordedPlanner([Proposal({"action": "solve", "solver_id": sid}) for sid, _ in order]
                              + [Proposal({"action": "stop"})])
    outcome = solve(problem, planner=planner, budget=Budget(max_evaluations=400), registry=registry)
    assert outcome.result is not None and outcome.result.solver_id == "test.exact", outcome.trace.best
    assert outcome.result.x == p


@pytest.mark.parametrize("seed", range(4))
def test_between_clean_points_the_better_objective_still_wins(seed):
    rng = np.random.default_rng(2000 + seed)
    problem, p, a, b, _, _ = projection(rng, 3, "==")
    tangent = np.cross(a, rng.normal(size=3))  # stays on a.x = b
    worse = {n: v + 0.1 * float(t) for (n, v), t in zip(p.items(), tangent)}
    assert band_fraction(problem, problem.evaluate(worse).residuals) <= SELECTION_FRACTION
    registry = SolverRegistry([Fixed("test.worse", worse), Fixed("test.exact", p)])
    planner = RecordedPlanner([Proposal({"action": "solve", "solver_id": "test.worse"}),
                               Proposal({"action": "solve", "solver_id": "test.exact"}), Proposal({"action": "stop"})])
    outcome = solve(problem, planner=planner, budget=Budget(max_evaluations=400), registry=registry)
    assert outcome.result.solver_id == "test.exact" and outcome.result.x == p


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("solver", [DifferentialEvolutionSolver(), SlsqpSolver()], ids=lambda s: s.capability.solver_id)
def test_the_adapters_report_a_clean_point_that_does_not_beat_the_optimum(seed, solver):
    """Differential evolution sees the IR's band as its constraint and presses against its edge; SLSQP stops a
    finite-difference step outside. The adapter restores its best band point onto the plane before it reports."""
    rng = np.random.default_rng(3000 + seed)
    problem, p, _, _, _, _ = projection(rng, 3, "==")
    result = solver.solve(problem, budget=Budget(max_evaluations=6000), seed=seed)
    assert result.constraints_satisfied
    assert band_fraction(problem, result.constraint_residuals) <= SELECTION_FRACTION
    optimum = problem.evaluate(p).objectives["f"]
    assert result.objective_values["f"] >= optimum - 1e-12 * max(1.0, optimum)
    assert result.evaluations <= 6000


class Pinned(AdapterSolver):
    """An adapter whose backend evaluates one point and reports convergence: what SLSQP did on
    cqp-projection-d2 in v1, (1 + 2**-26, 0), 2**-26 outside x + y <= 1."""

    CAPABILITY = replace(SlsqpSolver.CAPABILITY, solver_id="test.pinned", options={})

    def _run(self, run: Run) -> Outcome:
        run.evaluator([1.0 + 2.0 ** -26, 0.0])
        return Outcome(converged=True)


def test_restoration_moves_a_band_point_onto_the_plane_and_holds_the_bound_it_meets():
    """The optimum (1, 0) sits on the bound y >= 0: the least-norm step would cross it, so the restoration holds
    y at its bound and moves x alone. Two steps: 3 + 2 evaluations after the backend's one."""
    problem = C.instance_ir(C.P.by_id("cqp-projection-d2"))
    result = Pinned().solve(problem, budget=Budget(max_evaluations=4000))
    assert result.x == {"x0": 1.0, "x1": 0.0} and result.objective_values["f0"] == 2.0
    assert result.constraint_residuals["half_plane"] == 0.0
    assert "restored onto the constraints" in result.message and result.evaluations == 1 + 3 + 2
    assert result.termination is Termination.CONVERGED


def test_without_evaluations_left_the_band_point_stands_and_ranks_behind_a_clean_one():
    problem = C.instance_ir(C.P.by_id("cqp-projection-d2"))
    result = Pinned().solve(problem, budget=Budget(max_evaluations=1))
    assert result.x == {"x0": 1.0 + 2.0 ** -26, "x1": 0.0}  # the only point: restoration had nothing to spend
    assert band_fraction(problem, result.constraint_residuals) > SELECTION_FRACTION
