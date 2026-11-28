"""Solver registry and adapters (E14-03): one call solves an IR problem with any compatible solver.

``default_registry().compatible(problem)`` lists the installed solvers that can
take a problem, most specialised first, and every solver's ``solve`` returns
the shared ``OptimizationResult``. Backends (scipy, pygmo, cma) are imported
only when a solver runs. See ``docs/optimization/solvers.md``.
"""

from .base import (CONVERGENCE_MEANINGS, RESTORE_STEPS, SELECTION_FRACTION, AdapterSolver, BudgetExhausted, OptionSpec,
                   Outcome, Run, Solver, SolverCapability, band_fraction, restoration_reserve)
from .cma_solvers import CmaEsSolver
from .enumeration import EnumerationSolver
from .pygmo_solvers import Nsga2Solver
from .registry import SolverRegistry, default_registry
from .scipy_solvers import DifferentialEvolutionSolver, HighsSolver, LbfgsbSolver, SlsqpSolver

__all__ = [
    "AdapterSolver", "BudgetExhausted", "CONVERGENCE_MEANINGS", "CmaEsSolver", "DifferentialEvolutionSolver",
    "EnumerationSolver", "HighsSolver", "LbfgsbSolver", "Nsga2Solver", "OptionSpec", "Outcome", "RESTORE_STEPS",
    "Run", "SELECTION_FRACTION", "SlsqpSolver", "Solver", "SolverCapability", "SolverRegistry", "band_fraction",
    "default_registry", "restoration_reserve",
]
