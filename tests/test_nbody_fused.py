"""Fused-kernel prototypes (task #143) must reproduce the production
``propagate_spacecraft_batch(output="closest_approach")`` result.

Moon/Sun seed states are injected into nbody_gpu's cache (no SPICE needed),
same as tests/test_nbody_closest_approach.py. Optional backends are skipped
when their package is missing; numba-cuda runs on the CUDA simulator.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb

EPOCH0 = 838_000_000.0
MOON_R0_KM = np.array([-291_608.0, -266_716.0, 22_850.0])
MOON_V0_KM_S = np.array([0.6439, -0.6974, 0.0136])
SUN_R0_KM = np.array([1.0e8, -1.1e8, 5.0e3])
SUN_V0_KM_S = np.array([22.0, 20.0, 0.0])


@pytest.fixture(autouse=True)
def _seed_cache(monkeypatch):
    key = round(EPOCH0, 3)
    monkeypatch.setitem(nb._MOON_STATE0_KM, key, (MOON_R0_KM.copy(), MOON_V0_KM_S.copy()))
    monkeypatch.setitem(nb._SUN_STATE0_KM, key, (SUN_R0_KM.copy(), SUN_V0_KM_S.copy()))


def _case(n, seed=0, mixed_spans=True):
    rng = np.random.default_rng(seed)
    r_park = np.array([6_563.0, 0.0, 0.0])
    v_park = np.array([0.0, np.sqrt(nb.MU_EARTH_KM3_S2 / 6_563.0), 0.0])
    dvs = rng.normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    t_span = np.linspace(4.0, 8.0, n) * 86400.0 if mixed_spans else 6.0 * 86400.0
    return np.broadcast_to(r_park, (n, 3)), v_park + dvs, t_span


def _check(got, ref):
    np.testing.assert_allclose(got.min_distance_km, ref.min_distance_km, rtol=1e-9, atol=1e-6)
    np.testing.assert_array_equal(got.min_distance_step, ref.min_distance_step)
    np.testing.assert_allclose(got.min_distance_epoch_s, ref.min_distance_epoch_s, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got.r, ref.r, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(got.v, ref.v, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got.t_abs, ref.t_abs, rtol=0, atol=1e-6)


def _run(backend, n, n_steps, include_sun, mixed_spans=True):
    r0, v0, t_span = _case(n, mixed_spans=mixed_spans)
    kw = dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
              include_sun=include_sun, epoch0=EPOCH0)
    ref = nb.propagate_spacecraft_batch(r0, v0, t_span, n_steps, use_gpu=False,
                                        output="closest_approach", **kw)
    got = nf.propagate_closest_approach_fused(r0, v0, t_span, n_steps, backend=backend, **kw)
    return got, ref


@pytest.mark.parametrize("include_sun", [False, True])
@pytest.mark.parametrize("mixed_spans", [False, True])
def test_python_reference_matches_production(include_sun, mixed_spans):
    _check(*_run("python", 12, 40, include_sun, mixed_spans))


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
@pytest.mark.parametrize("include_sun", [False, True])
def test_numba_cpu_matches_production(include_sun):
    _check(*_run("numba-cpu", 64, 120, include_sun))


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
@pytest.mark.parametrize("include_sun", [False, True])
def test_jax_matches_production(include_sun):
    _check(*_run("jax", 64, 120, include_sun))


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
def test_numba_cuda_simulator_matches_production():
    # The simulator flag must be set before numba.cuda is first imported,
    # so run in a fresh interpreter.
    code = (
        "import numpy as np, sys; sys.path.insert(0, 'tests');"
        "import test_nbody_fused as t; from orbitopt.dynamics import nbody_gpu as nb;"
        "k = round(t.EPOCH0, 3);"
        "nb._MOON_STATE0_KM[k] = (t.MOON_R0_KM, t.MOON_V0_KM_S);"
        "nb._SUN_STATE0_KM[k] = (t.SUN_R0_KM, t.SUN_V0_KM_S);"
        "t._check(*t._run('numba-cuda', 8, 10, True))"
    )
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_unknown_backend_rejected():
    r0, v0, t_span = _case(2)
    with pytest.raises(ValueError):
        nf.propagate_closest_approach_fused(r0, v0, t_span, 4, nb.MU_EARTH_KM3_S2,
                                            nb.MU_MOON_KM3_S2, epoch0=EPOCH0, backend="bogus")
