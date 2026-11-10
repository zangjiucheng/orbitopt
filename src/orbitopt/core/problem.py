"""Base classes for orbitopt optimization problems (pygmo UDP interface).

A pygmo "user defined problem" (UDP) only strictly needs get_bounds() and
fitness(). Problems in this framework additionally implement batch_fitness()
so that population-based algorithms can route candidate-solution evaluation
through a vectorized GPU/CPU kernel via GpuBatchFitnessEvaluator (see
orbitopt.optimize.gpu_bfe) instead of evaluating one candidate at a time.
"""
from __future__ import annotations

import abc

import numpy as np

from orbitopt.core.gpu import get_array_module


class BackendUsage:
    """Per-backend tally of the candidates a problem's batch_fitness() has
    evaluated, plus the backend of the most recent call.

    pygmo deep-copies a Python UDP (into ``pg.problem``, the population and
    each batch-evaluator call), so plain attributes set inside
    batch_fitness() never reach the caller's object. ``__deepcopy__``
    returns ``self``: every copy of the problem shares one tally, which is
    how run_optimization() reports the backend that actually ran. (Pickled
    copies, e.g. in multiprocessing islands, get their own tally.)
    """

    def __init__(self):
        self.candidates = {}
        self.calls = {}
        self.last = None

    def record(self, backend, n):
        self.candidates[backend] = self.candidates.get(backend, 0) + int(n)
        self.calls[backend] = self.calls.get(backend, 0) + 1
        self.last = backend

    def snapshot(self):
        return dict(self.candidates)

    def since(self, snapshot):
        """Candidates per backend evaluated since ``snapshot()``."""
        return {
            k: v - snapshot.get(k, 0)
            for k, v in self.candidates.items()
            if v - snapshot.get(k, 0) > 0
        }

    def __deepcopy__(self, memo):
        return self


class OrbitOptProblem(abc.ABC):
    """Common base for all trajectory-optimization UDPs in this framework."""

    def __init__(self, use_gpu: bool | None = None):
        self.use_gpu = use_gpu

    # --- pygmo UDP protocol -------------------------------------------------
    @abc.abstractmethod
    def get_bounds(self):
        """Return (lower_bounds, upper_bounds) as two equal-length sequences."""

    @abc.abstractmethod
    def fitness(self, dv):
        """Evaluate a single decision vector, returning [objective(s)]."""

    def batch_fitness(self, dvs):
        """Vectorized evaluation of many decision vectors at once.

        ``dvs`` is a flat, contiguous sequence following pygmo's batch_fitness
        convention: decision vector i occupies dvs[i*n:(i+1)*n]. The default
        implementation just loops over fitness() -- subclasses override this
        with a real vectorized (GPU-capable) implementation to get any
        speedup; see orbitopt.problems.transfer_2body for an example.
        """
        n = len(self.get_bounds()[0])
        dv_matrix = np.asarray(dvs, dtype=float).reshape(-1, n)
        out = [self.fitness(dv) for dv in dv_matrix]
        return np.asarray(out, dtype=float).ravel()

    def get_nobj(self) -> int:
        return 1

    def has_batch_fitness(self) -> bool:
        """True if a subclass overrides batch_fitness with a real (non-loop)
        implementation, i.e. it's worth routing through the GPU bfe."""
        return type(self).batch_fitness is not OrbitOptProblem.batch_fitness

    def array_module(self):
        """The array module (``numpy`` or ``cupy``) batch_fitness() actually
        runs on, resolved from ``self.use_gpu`` exactly as the vectorized
        kernels resolve it (see orbitopt.core.gpu.get_array_module)."""
        return get_array_module(self.use_gpu)
