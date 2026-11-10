"""Convenience helpers for driving pygmo optimizers with GPU-accelerated
batch fitness evaluation.

The population is both *seeded* (pg.population(..., b=bfe)) and, for the
algorithms that support it (pg.pso_gen, pg.cmaes, pg.nsga2 as of pygmo
2.19 -- notably *not* pg.sga/pg.sade/pg.de), *evolved* through the same
batch evaluator via ``uda.set_bfe(bfe)``. Every fitness evaluation in the
run then goes through the problem's vectorized batch_fitness() instead of
pygmo's default one-candidate-at-a-time loop.

Important pygmo quirk: set_bfe() only exists on the raw user-defined
algorithm (UDA) instance, never on the pg.algorithm wrapper -- so it must be
called *before* wrapping (``uda.set_bfe(bfe); pg.algorithm(uda)``). Calling
it on an already-wrapped pg.algorithm silently does nothing (AttributeError
if you check, but nothing stops you from swallowing it) and the run falls
back to evaluating one individual at a time -- 10x+ slower and defeating the
entire point of batching. ``algorithm`` here is therefore expected to be a
*raw UDA* (e.g. ``pg.pso_gen(gen=150)``), not a ``pg.algorithm``.
"""
from __future__ import annotations

import time

import pygmo as pg

from orbitopt.optimize.gpu_bfe import CPU, GPU, GpuBatchFitnessEvaluator


class OptimizationResult(tuple):
    """``(champion_x, champion_f, elapsed_seconds)`` -- still unpackable as
    the 3-tuple run_optimization() has always returned -- plus ``device``:
    the device ("CPU" or "GPU") fitness evaluation actually ran on;
    ``backend``: the kernel backend that evaluated the most candidates
    (e.g. "numpy", "numba-cpu", "cupy"); and ``backend_usage``: candidates
    evaluated per backend during the run (empty when the problem doesn't
    track it)."""

    def __new__(cls, champion_x, champion_f, elapsed, device, backend=None, backend_usage=None):
        self = super().__new__(cls, (champion_x, champion_f, elapsed))
        self.device = device
        self.backend = backend
        self.backend_usage = dict(backend_usage or {})
        return self

    @property
    def champion_x(self):
        return self[0]

    @property
    def champion_f(self):
        return self[1]

    @property
    def elapsed(self):
        return self[2]

    def __repr__(self):
        return (
            f"OptimizationResult(champion_x={self[0]!r}, champion_f={self[1]!r}, "
            f"elapsed={self[2]!r}, device={self.device!r}, backend={self.backend!r})"
        )


def _backend_used(problem, usage, device, batch_size):
    """Backend that evaluated the most candidates in the run: measured via
    the problem's ``backend_usage`` tally when it has one, else predicted
    from ``resolve_backend()``, else inferred from the device."""
    if usage:
        return max(usage, key=usage.get)
    if hasattr(problem, "resolve_backend"):
        return problem.resolve_backend(batch_size)
    return "cupy" if device == GPU else "numpy"


def run_optimization(
    problem,
    algorithm=None,
    pop_size=256,
    generations=100,
    seed=None,
    use_gpu_bfe=True,
    verbose=True,
):
    """Optimize an orbitopt.core.problem.OrbitOptProblem with a
    generation-based pygmo algorithm, routing evaluation through a batch
    fitness evaluator (GPU-vectorized when the problem supports it).

    ``algorithm``, if given, must be a raw UDA instance (not pg.algorithm-
    wrapped) so this function can attach the batch evaluator before wrapping
    it. Defaults to pg.pso_gen(gen=generations), which supports set_bfe.

    Returns an OptimizationResult: unpacks as (champion_x, champion_f,
    elapsed_seconds) and carries ``.device`` -- "GPU" only when the problem's
    batch_fitness() really runs on CuPy, "CPU" for the NumPy fallback, the
    per-item fitness() loop, or use_gpu_bfe=False -- plus ``.backend`` /
    ``.backend_usage``: the kernel backend(s) batch_fitness() actually used
    (see OptimizationResult; FreeReturnScreeningProblem's ``backend="auto"``
    picks one per call).
    """
    prob = pg.problem(problem)
    uda = algorithm if algorithm is not None else pg.pso_gen(gen=generations, seed=seed)

    want_gpu_bfe = use_gpu_bfe and problem.has_batch_fitness()
    bfe = pg.bfe(GpuBatchFitnessEvaluator()) if want_gpu_bfe else pg.bfe()
    device = GpuBatchFitnessEvaluator.device_for(problem) if want_gpu_bfe else CPU

    algo_uses_bfe = hasattr(uda, "set_bfe")
    if algo_uses_bfe:
        uda.set_bfe(bfe)
    algo = pg.algorithm(uda)

    tally = getattr(problem, "backend_usage", None)
    before = tally.snapshot() if tally is not None else None

    pop = pg.population(prob, size=pop_size, seed=seed, b=bfe)

    t0 = time.perf_counter()
    pop = algo.evolve(pop)
    elapsed = time.perf_counter() - t0

    usage = tally.since(before) if tally is not None else {}
    batch_size = pop_size if (want_gpu_bfe and algo_uses_bfe) else 1
    backend = _backend_used(problem, usage, device, batch_size)

    if verbose:
        print(
            f"[orbitopt] {generations} generations, pop={pop_size}, "
            f"{device} bfe, backend={backend} (algo consumes bfe each gen: {algo_uses_bfe}), "
            f"{elapsed:.2f}s -> best f = {pop.champion_f}"
        )

    return OptimizationResult(pop.champion_x, pop.champion_f, elapsed, device, backend, usage)
