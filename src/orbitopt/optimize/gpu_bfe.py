"""Custom pygmo batch-fitness evaluator (UDBFE) that dispatches to a
problem's own ``batch_fitness()`` -- which, for orbitopt problems, is
typically GPU-vectorized (see orbitopt.problems.transfer_2body).

pygmo's UDBFE protocol: a callable ``udbfe(prob, dvs) -> fitness_vector``
where ``dvs``/``fitness_vector`` are flat, contiguous 1D sequences (decision
vector i occupies dvs[i*n:(i+1)*n]). Wrapping this in ``pygmo.bfe(...)`` lets
population-based algorithms (pg.sga, pg.pso_gen, ...) evaluate an entire
generation in one call instead of one individual at a time.
"""
from __future__ import annotations

import numpy as np


class GpuBatchFitnessEvaluator:
    """UDBFE that routes evaluation through problem.batch_fitness() when the
    underlying orbitopt UDP provides one, falling back to a plain per-item
    loop over problem.fitness() otherwise (still correct, just not GPU-fast).
    """

    def __call__(self, prob, dvs):
        n = prob.get_nx()
        udp = prob.extract(object)

        if udp is not None and hasattr(udp, "batch_fitness"):
            fv = udp.batch_fitness(np.asarray(dvs, dtype=float))
            return np.asarray(fv, dtype=float).ravel()

        dv_matrix = np.asarray(dvs, dtype=float).reshape(-1, n)
        out = [prob.fitness(dv) for dv in dv_matrix]
        return np.asarray(out, dtype=float).ravel()

    def get_name(self):
        return "orbitopt GPU batch fitness evaluator"
