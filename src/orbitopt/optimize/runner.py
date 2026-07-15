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

from orbitopt.optimize.gpu_bfe import GpuBatchFitnessEvaluator


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

    Returns (champion_x, champion_f, elapsed_seconds).
    """
    prob = pg.problem(problem)
    uda = algorithm if algorithm is not None else pg.pso_gen(gen=generations, seed=seed)

    want_gpu_bfe = use_gpu_bfe and problem.has_batch_fitness()
    bfe = pg.bfe(GpuBatchFitnessEvaluator()) if want_gpu_bfe else pg.bfe()

    algo_uses_bfe = hasattr(uda, "set_bfe")
    if algo_uses_bfe:
        uda.set_bfe(bfe)
    algo = pg.algorithm(uda)

    pop = pg.population(prob, size=pop_size, seed=seed, b=bfe)

    t0 = time.perf_counter()
    pop = algo.evolve(pop)
    elapsed = time.perf_counter() - t0

    if verbose:
        backend = "GPU" if want_gpu_bfe else "CPU"
        print(
            f"[orbitopt] {generations} generations, pop={pop_size}, "
            f"{backend} bfe (algo consumes bfe each gen: {algo_uses_bfe}), "
            f"{elapsed:.2f}s -> best f = {pop.champion_f}"
        )

    return pop.champion_x, pop.champion_f, elapsed
