"""Solver registry and adapters (E14-03): one call solves an IR problem with any compatible solver.

``default_registry().compatible(problem)`` lists the installed solvers that can
take a problem, most specialised first, and every solver's ``solve`` returns
the shared ``OptimizationResult``. Backends (scipy, pygmo, cma) are imported
only when a solver runs. See ``docs/optimization/solvers.md``.
"""

from .base import CONVERGENCE_MEANINGS, AdapterSolver, BudgetExhausted, Outcome, Run, Solver, SolverCapability
from .cma_solvers import CmaEsSolver
from .pygmo_solvers import Nsga2Solver
from .registry import SolverRegistry, default_registry
from .scipy_solvers import DifferentialEvolutionSolver, HighsSolver, LbfgsbSolver, SlsqpSolver

__all__ = [
    "AdapterSolver", "BudgetExhausted", "CONVERGENCE_MEANINGS", "CmaEsSolver", "DifferentialEvolutionSolver",
    "HighsSolver", "LbfgsbSolver", "Nsga2Solver", "Outcome", "Run", "Solver", "SolverCapability", "SolverRegistry",
    "SlsqpSolver", "default_registry",
]
