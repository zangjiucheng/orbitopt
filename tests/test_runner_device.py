"""run_optimization() must report the device fitness evaluation *actually*
ran on, not just "GPU" because the GPU-named batch evaluator was used."""
from __future__ import annotations

import numpy as np
import pytest

import orbitopt.core.gpu as gpu
from orbitopt.core.problem import OrbitOptProblem
from orbitopt.optimize.gpu_bfe import GpuBatchFitnessEvaluator
from orbitopt.optimize.runner import OptimizationResult, run_optimization


class Sphere(OrbitOptProblem):
    """Vectorized batch_fitness written against the problem's array module."""

    def get_bounds(self):
        return [-5.0, -5.0], [5.0, 5.0]

    def fitness(self, dv):
        return [float(np.sum(np.asarray(dv) ** 2))]

    def batch_fitness(self, dvs):
        xp = self.array_module()
        m = xp.asarray(np.asarray(dvs, dtype=float)).reshape(-1, 2)
        return gpu.to_numpy(xp.sum(m ** 2, axis=1))


class LoopOnlySphere(Sphere):
    batch_fitness = OrbitOptProblem.batch_fitness


class _FakeCupy:
    __name__ = "cupy"


def _run(problem, **kw):
    return run_optimization(problem, pop_size=16, generations=5, seed=1, verbose=False, **kw)


def test_result_still_unpacks_as_three_tuple():
    res = _run(Sphere(use_gpu=False))
    assert isinstance(res, OptimizationResult)
    x, f, elapsed = res
    assert res.champion_x is x and res.champion_f is f and res.elapsed == elapsed
    assert len(x) == 2 and f[0] < 1.0


def test_numpy_backend_reports_cpu_even_through_gpu_bfe():
    problem = Sphere(use_gpu=False)
    assert problem.has_batch_fitness()
    assert _run(problem).device == "CPU"


@pytest.mark.skipif(gpu.GPU_AVAILABLE, reason="checks the no-CUDA fallback path")
def test_auto_backend_without_gpu_falls_back_to_cpu():
    assert _run(Sphere(use_gpu=None)).device == "CPU"


def test_loop_only_problem_and_disabled_bfe_report_cpu():
    assert _run(LoopOnlySphere(use_gpu=None)).device == "CPU"
    assert _run(Sphere(use_gpu=None), use_gpu_bfe=False).device == "CPU"


def test_device_for_reads_the_evaluator_backend(monkeypatch):
    problem = Sphere(use_gpu=None)
    monkeypatch.setattr(problem, "array_module", lambda: np)
    assert GpuBatchFitnessEvaluator.device_for(problem) == "CPU"
    monkeypatch.setattr(problem, "array_module", lambda: _FakeCupy())
    assert GpuBatchFitnessEvaluator.device_for(problem) == "GPU"


def test_device_for_third_party_udps():
    class NoBackendDeclared:
        def batch_fitness(self, dvs):
            return dvs

    class NoBatch:
        use_gpu = True

    assert GpuBatchFitnessEvaluator.device_for(NoBackendDeclared()) == "CPU"
    assert GpuBatchFitnessEvaluator.device_for(NoBatch()) == "CPU"
    assert GpuBatchFitnessEvaluator.device_for(None) == "CPU"


@pytest.mark.skipif(not gpu.GPU_AVAILABLE, reason="needs CuPy + CUDA")
def test_cupy_backend_reports_gpu():
    assert _run(Sphere(use_gpu=True)).device == "GPU"
