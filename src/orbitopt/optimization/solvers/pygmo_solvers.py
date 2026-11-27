"""NSGA-II for multi-objective problems through pygmo, imported only when it runs.

pygmo deep-copies a problem object, so the adapter's problem returns itself
from ``__deepcopy__``: a copy with its own counter would escape the budget.
pygmo re-raises an exception from the problem's fitness but drops the
population with it, so the adapter evolves one generation at a time, plans
the generation count to fit ``max_evaluations``, and keeps the last complete
population when the wall clock runs out mid-generation. The result's ``x`` is
the front member with the lexicographically smallest objective vector, and
the whole front is in the solver evidence under ``front``.
"""

from __future__ import annotations

from ..classify import Form
from ..result import Termination
from .base import AdapterSolver, BudgetExhausted, Outcome, Run, SolverCapability

DEFAULT_POPULATION = 48
DEFAULT_GENERATIONS = 250
MIN_POPULATION = 8
"""pygmo's NSGA-II needs a population of at least 5 that is a multiple of 4."""
_HUGE = 1e300  # stands in for a failed evaluation so crowding distances stay finite


class _Problem:
    """A pygmo user-defined problem over the adapter's evaluator; pygmo wants integer coordinates last."""

    def __init__(self, run: Run):
        self.run = run
        space = run.space
        self.order = [i for i, f in enumerate(space.integral) if not f] + [i for i, f in enumerate(space.integral) if f]

    def __deepcopy__(self, memo):
        return self  # pygmo deep-copies its problem; every copy must share the one counting evaluator

    def to_vector(self, z):
        vector = [0.0] * len(self.order)
        for position, index in enumerate(self.order):
            vector[index] = float(z[position])
        return tuple(vector)

    def fitness(self, z):
        return [min(value, _HUGE) for value in self.run.evaluator(self.to_vector(z)).f]

    def get_bounds(self):
        space = self.run.space
        return [space.lower[i] for i in self.order], [space.upper[i] for i in self.order]

    def get_nobj(self):
        return len(self.run.problem.objectives)

    def get_nix(self):
        return sum(self.run.space.integral)


class Nsga2Solver(AdapterSolver):
    """NSGA-II (``pygmo.nsga2``): a seeded multi-objective evolutionary algorithm with no convergence test.

    It never reports ``CONVERGED``: it stops after its generations, on the
    evaluation budget, or on the wall clock.
    """

    CAPABILITY = SolverCapability(
        solver_id="pygmo.nsga2", backend="pygmo", max_form=Form.BLACK_BOX, convergence="none", supports_integers=True,
        multiobjective=True, needs_finite_bounds=True, is_global=True,
        honors=frozenset({"max_evaluations", "max_iterations", "max_wall_s"}),
        options=frozenset({"population", "cr", "eta_c", "m", "eta_m"}))

    def _run(self, run: Run) -> Outcome:
        import pygmo as pg

        size = run.options.get("population", DEFAULT_POPULATION)
        limit = run.budget.max_evaluations
        if limit is not None and limit < size:
            size = limit // 4 * 4
        if size < MIN_POPULATION:
            return Outcome(stop=Termination.MAX_EVALUATIONS,
                           message=f"an evaluation budget of {limit} is below NSGA-II's smallest population "
                                   f"({MIN_POPULATION})")
        generations = run.budget.max_iterations or DEFAULT_GENERATIONS
        stop = Termination.MAX_ITERATIONS
        if limit is not None and (limit - size) // size < generations:
            generations, stop = (limit - size) // size, Termination.MAX_EVALUATIONS
        udp = _Problem(run)
        population = pg.population(pg.problem(udp), size, seed=run.seed)
        params = {k: run.options[k] for k in ("cr", "eta_c", "m", "eta_m") if k in run.options}
        algorithm = pg.algorithm(pg.nsga2(gen=1, seed=run.seed, **params))
        message = f"completed {generations} generations of {size}"
        done = 0
        try:
            while done < generations:
                population = algorithm.evolve(population)
                done += 1
        except BudgetExhausted as exhausted:
            stop, message = exhausted.termination, f"{exhausted} after {done} generations"
        first_front = pg.fast_non_dominated_sorting(population.get_f())[0][0]
        xs = population.get_x()
        return Outcome(stop=stop, message=message, iterations=done,
                       points=tuple(udp.to_vector(xs[i]) for i in sorted(first_front)))
