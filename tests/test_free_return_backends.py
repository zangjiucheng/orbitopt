"""FreeReturnScreeningProblem(backend=...) (task #164): every backend gives
the same fitness within round-off, ``auto`` picks numba-cpu only when it can
and should, and run_optimization() reports the backend that actually ran.

Moon seed state injected into nbody_gpu's cache (no SPICE), as in
tests/test_nbody_closest_approach.py.
"""
from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pytest

import orbitopt.core.gpu as gpu
from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.problems import free_return as fr
from orbitopt.problems.free_return import FreeReturnScreeningProblem

HAS_NUMBA = importlib.util.find_spec("numba") is not None
HAS_PYGMO = importlib.util.find_spec("pygmo") is not None
needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

EPOCH0 = 838_000_000.0
MOON_R0_KM = np.array([-291_608.0, -266_716.0, 22_850.0])
MOON_V0_KM_S = np.array([0.6439, -0.6974, 0.0136])


@pytest.fixture(autouse=True)
def _seed_moon_cache(monkeypatch):
    monkeypatch.setitem(nb._MOON_STATE0_KM, round(EPOCH0, 3), (MOON_R0_KM.copy(), MOON_V0_KM_S.copy()))


def _problem(backend="auto", n_steps=80, **kw):
    r0 = np.array([6_563.0, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(nb.MU_EARTH_KM3_S2 / 6_563.0), 0.0])
    return FreeReturnScreeningProblem(
        r0, v0, EPOCH0, target_distance_km=8_000.0, coast_days=6.0, n_steps=n_steps,
        dv_bounds_km_s=(-0.5, 3.5), backend=backend, **kw,
    )


def _dvs(n, seed=0):
    dvs = np.random.default_rng(seed).normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    return dvs


def _agree(a, b):
    np.testing.assert_allclose(a.batch_fitness(_dvs(40).ravel()), b.batch_fitness(_dvs(40).ravel()),
                               rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(a._last_min_distance_epoch_s, b._last_min_distance_epoch_s,
                               rtol=0, atol=1e-6)


# --- validation --------------------------------------------------------------

def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        _problem(backend="jax")


def test_conflicting_use_gpu_rejected():
    with pytest.raises(ValueError):
        _problem(backend="numpy", use_gpu=True)
    if gpu.GPU_AVAILABLE:
        with pytest.raises(ValueError):
            _problem(backend="cupy", use_gpu=False)


@pytest.mark.skipif(gpu.GPU_AVAILABLE, reason="checks the no-CUDA error")
def test_cupy_without_gpu_rejected():
    with pytest.raises(RuntimeError):
        _problem(backend="cupy")


def test_numba_cpu_needs_numba_and_closest_approach(monkeypatch):
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", True)
    with pytest.raises(ValueError):
        _problem(backend="numba-cpu", propagation_output="history")
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        _problem(backend="numba-cpu")


# --- auto resolution -----------------------------------------------------------

def test_auto_falls_back_to_numpy_without_numba(monkeypatch):
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", False)
    p = _problem()
    assert p.resolve_backend(10_000) == "numpy"
    p.batch_fitness(_dvs(fr.AUTO_NUMBA_MIN_BATCH).ravel())
    assert p.last_backend == "numpy"
    assert p.array_module() is np


def test_auto_picks_numba_by_batch_size(monkeypatch):
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(nf, "_NUMBA_CPU_WARM", False)
    p = _problem()
    assert p.resolve_backend(fr.AUTO_NUMBA_MIN_BATCH) == "numba-cpu"
    assert p.resolve_backend(1) == "numpy"          # not worth a JIT load
    monkeypatch.setattr(nf, "_NUMBA_CPU_WARM", True)
    assert p.resolve_backend(1) == "numba-cpu"      # already loaded: always faster
    assert _problem(propagation_output="history").resolve_backend(10_000) == "numpy"
    assert p.array_module() is np                   # numba-cpu is a CPU backend


def test_auto_honours_explicit_gpu_request(monkeypatch):
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(gpu, "GPU_AVAILABLE", True)
    assert _problem(use_gpu=True).resolve_backend(10_000) == "cupy"
    assert _problem(use_gpu=False).resolve_backend(10_000) == "numba-cpu"


def test_auto_falls_back_when_numba_kernel_fails(monkeypatch):
    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", True)

    def boom(_):
        raise RuntimeError("simulated numba failure")

    monkeypatch.setattr(nf, "run_numba_cpu", boom)
    p = _problem()
    with pytest.warns(RuntimeWarning, match="falling back to numpy"):
        f = p.batch_fitness(_dvs(fr.AUTO_NUMBA_MIN_BATCH).ravel())
    assert p.last_backend == "numpy" and not nf.numba_available()
    np.testing.assert_allclose(f, _problem("numpy").batch_fitness(_dvs(fr.AUTO_NUMBA_MIN_BATCH).ravel()))

    monkeypatch.setattr(nf, "_NUMBA_AVAILABLE", True)
    explicit = _problem("numba-cpu")  # an explicit backend never falls back
    with pytest.raises(RuntimeError, match="simulated"):
        explicit.batch_fitness(_dvs(4).ravel())


def test_backend_usage_is_shared_across_deep_copies():
    p = _problem("numpy")
    q = copy.deepcopy(p)
    q.batch_fitness(_dvs(5).ravel())
    assert p.backend_usage.candidates == {"numpy": 5} and p.last_backend == "numpy"


# --- end-to-end fitness agreement ---------------------------------------------

@needs_numba
def test_numba_cpu_fitness_matches_numpy():
    _agree(_problem("numba-cpu"), _problem("numpy"))


@needs_numba
def test_numba_cpu_fitness_matches_history_path():
    _agree(_problem("numba-cpu"), _problem("numpy", propagation_output="history"))


@needs_numba
def test_auto_with_numba_matches_numpy_and_single_candidate_calls():
    auto, ref = _problem("auto"), _problem("numpy")
    _agree(auto, ref)
    assert auto.last_backend == "numba-cpu"
    dv = _dvs(3)[1]
    assert auto.fitness(dv)[0] == pytest.approx(ref.fitness(dv)[0], rel=1e-9, abs=1e-6)
    assert auto.time_to_closest_approach(dv) == pytest.approx(ref.time_to_closest_approach(dv), abs=1e-6)


@pytest.mark.gpu
def test_cupy_fitness_matches_numpy():
    _agree(_problem("cupy"), _problem("numpy"))


# --- run_optimization reports the backend --------------------------------------

def _optimize(problem, pop_size):
    from orbitopt.optimize.runner import run_optimization

    return run_optimization(problem, pop_size=pop_size, generations=3, seed=1, verbose=False)


@pytest.mark.skipif(not HAS_PYGMO, reason="pygmo not installed")
def test_run_optimization_reports_numpy_backend():
    res = _optimize(_problem("numpy", n_steps=40), 16)
    assert res.device == "CPU" and res.backend == "numpy"
    assert res.backend_usage == {"numpy": 16 * 4}  # initial pop + 3 generations


@needs_numba
@pytest.mark.skipif(not HAS_PYGMO, reason="pygmo not installed")
def test_run_optimization_reports_numba_backend():
    res = _optimize(_problem("auto", n_steps=40), fr.AUTO_NUMBA_MIN_BATCH)
    assert res.device == "CPU" and res.backend == "numba-cpu"
    assert set(res.backend_usage) == {"numba-cpu"}
