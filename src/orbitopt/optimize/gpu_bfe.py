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

CPU = "CPU"
GPU = "GPU"


def _module_is_numpy(xp) -> bool:
    return xp is np or getattr(xp, "__name__", "").split(".")[0] == "numpy"


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

    @staticmethod
    def device_for(udp) -> str:
        """Report the device ("CPU" or "GPU") this evaluator really runs
        ``udp``'s fitness on -- i.e. the array module behind the problem's
        batch_fitness(), not merely the fact that this (GPU-named) evaluator
        is in use.

        A problem without batch_fitness() goes through the per-item
        fitness() loop, which is CPU. Otherwise the backend is read from the
        problem's ``array_module()`` (orbitopt.core.problem.OrbitOptProblem
        provides it from ``use_gpu``); a third-party UDP that declares
        neither ``array_module()`` nor ``use_gpu`` is assumed to be NumPy.
        """
        if udp is None or not hasattr(udp, "batch_fitness"):
            return CPU
        if hasattr(udp, "array_module"):
            xp = udp.array_module()
        elif hasattr(udp, "use_gpu"):
            from orbitopt.core.gpu import get_array_module

            xp = get_array_module(udp.use_gpu)
        else:
            return CPU
        return CPU if _module_is_numpy(xp) else GPU

    def get_name(self):
        return "orbitopt GPU batch fitness evaluator"
