"""The solver registry: solver ids to solvers, and which of them can take a given problem."""

from __future__ import annotations

from typing import Any, Iterable

from ..classify import classify
from ..problem import OptimizationProblem
from ..result import OptimizationResult
from .base import Solver, SolverCapability, require_valid
from .cma_solvers import CmaEsSolver
from .pygmo_solvers import Nsga2Solver
from .scipy_solvers import DifferentialEvolutionSolver, HighsSolver, LbfgsbSolver, SlsqpSolver


def _specialisation(capability: SolverCapability) -> tuple:
    """Sort key: the least general solver first, then the id, so the order never depends on registration."""
    return (capability.max_form, capability.supports_constraints, capability.supports_equality,
            capability.supports_integers, capability.is_global, capability.solver_id)


class SolverRegistry:
    """Solvers by ``capability.solver_id``; ``default_registry()`` holds every built-in."""

    def __init__(self, solvers: Iterable[Solver] = ()):
        self._solvers: dict[str, Solver] = {}
        for solver in solvers:
            self.register(solver)

    def register(self, solver: Solver) -> None:
        """Add a solver; its id must be new."""
        solver_id = solver.capability.solver_id
        if solver_id in self._solvers:
            raise ValueError(f"solver {solver_id!r} is already registered")
        self._solvers[solver_id] = solver

    def get(self, solver_id: str) -> Solver:
        """The solver with this id; ``KeyError`` naming the known ids otherwise."""
        try:
            return self._solvers[solver_id]
        except KeyError:
            raise KeyError(f"no solver {solver_id!r}; known: {sorted(self._solvers)}") from None

    def capabilities(self) -> tuple[SolverCapability, ...]:
        """Every registered solver's capability, available or not, by id."""
        return tuple(sorted((s.capability for s in self._solvers.values()), key=lambda c: c.solver_id))

    def compatible(self, problem: OptimizationProblem) -> tuple[Solver, ...]:
        """The available solvers whose capability takes ``problem``, most specialised first.

        Raises ``InvalidProblemError`` for an invalid problem.
        """
        require_valid(problem)
        structure = classify(problem)
        capable = [(s.capability, s) for s in self._solvers.values()]
        capable = [(c, s) for c, s in capable if c.available and c.rejection(structure) is None]
        return tuple(s for _, s in sorted(capable, key=lambda pair: _specialisation(pair[0])))

    def solve_with(self, problem: OptimizationProblem, solver_id: str, **kwargs: Any) -> OptimizationResult:
        """``get(solver_id).solve(problem, **kwargs)``: the same call for every solver."""
        return self.get(solver_id).solve(problem, **kwargs)


def default_registry() -> SolverRegistry:
    """A registry of every built-in solver, including those whose backend is not installed."""
    return SolverRegistry([HighsSolver(), SlsqpSolver(), LbfgsbSolver(), DifferentialEvolutionSolver(),
                           CmaEsSolver(), Nsga2Solver()])
