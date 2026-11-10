"""CUDA regression cases for the N-body screening propagator and its
closest-approach output: the CuPy path (``use_gpu=True``) and the fused
CUDA kernels must match the numpy reference.

Every test here is ``gpu``-marked, so it is skipped without a CUDA device
and is a hard error under ``ORBITOPT_REQUIRE_GPU=1`` (tests/conftest.py).
Run on a GPU host with ``python scripts/gpu_regression.py`` (see
docs/gpu-regression.md). The Moon/Sun seed states are injected into the
per-epoch cache like tests/test_nbody_closest_approach.py, so no SPICE kernels
are needed.

Tolerances: the CPU and GPU paths run the same float64 operations, but CUDA
fuses multiply-adds and its transcendental functions round differently, so
results agree to ~1e-9 relative rather than bit for bit. The closest-approach
step index is compared exactly. A mismatch here is a real backend divergence.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb

pytestmark = pytest.mark.gpu

EPOCH0 = 838_000_000.0
MOON_R0_KM = np.array([-291_608.0, -266_716.0, 22_850.0])
MOON_V0_KM_S = np.array([0.6439, -0.6974, 0.0136])
SUN_R0_KM = np.array([1.0e8, -1.1e8, 5.0e3])
SUN_V0_KM_S = np.array([22.0, 20.0, 0.0])


@pytest.fixture(autouse=True)
def _seed_body_cache(monkeypatch):
    key = round(EPOCH0, 3)
    monkeypatch.setitem(nb._MOON_STATE0_KM, key, (MOON_R0_KM.copy(), MOON_V0_KM_S.copy()))
    monkeypatch.setitem(nb._SUN_STATE0_KM, key, (SUN_R0_KM.copy(), SUN_V0_KM_S.copy()))


def _case(n, seed=0, mixed_spans=True):
    r_park = np.array([6_563.0, 0.0, 0.0])
    v_park = np.array([0.0, np.sqrt(nb.MU_EARTH_KM3_S2 / 6_563.0), 0.0])
    rng = np.random.default_rng(seed)
    dvs = rng.normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    t_span = np.linspace(4.0, 8.0, n) * 86400.0 if mixed_spans else 6.0 * 86400.0
    return np.broadcast_to(r_park, (n, 3)).copy(), v_park + dvs, t_span


def _kw(include_sun):
    return dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
                include_sun=include_sun, epoch0=EPOCH0)


def _assert_ca_close(got, ref):
    np.testing.assert_allclose(got.min_distance_km, ref.min_distance_km, rtol=1e-9, atol=1e-6)
    np.testing.assert_array_equal(got.min_distance_step, ref.min_distance_step)
    np.testing.assert_allclose(got.min_distance_epoch_s, ref.min_distance_epoch_s, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got.r, ref.r, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(got.v, ref.v, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got.t_abs, ref.t_abs, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# Two-body / ephemeris building blocks
# --------------------------------------------------------------------------

def test_kepler_batch_cpu_and_gpu_agree():
    r0, v0, _ = _case(256)
    dt = np.linspace(-3.0, 9.0, 256) * 86400.0
    r_c, v_c, ok_c, _ = nb.propagate_kepler_batch(r0, v0, dt, nb.MU_EARTH_KM3_S2,
                                                   use_gpu=False, return_info=True)
    r_g, v_g, ok_g, _ = nb.propagate_kepler_batch(r0, v0, dt, nb.MU_EARTH_KM3_S2,
                                                   use_gpu=True, return_info=True)
    assert isinstance(r_g, np.ndarray)  # results come back to host memory
    np.testing.assert_array_equal(ok_c, ok_g)
    np.testing.assert_allclose(r_g[ok_c], r_c[ok_c], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(v_g[ok_c], v_c[ok_c], rtol=1e-9, atol=1e-9)


def test_moon_and_sun_state_cpu_and_gpu_agree():
    dt = np.linspace(0.0, 10.0, 801) * 86400.0
    for fn in (nb.moon_state_batch, nb.sun_state_batch):
        r_c, v_c = fn(dt, use_gpu=False, seed_epoch_ephemeris_seconds=EPOCH0)
        r_g, v_g = fn(dt, use_gpu=True, seed_epoch_ephemeris_seconds=EPOCH0)
        np.testing.assert_allclose(r_g, r_c, rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(v_g, v_c, rtol=1e-10, atol=1e-9)


# --------------------------------------------------------------------------
# N-body RK4 propagation (history output)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("include_sun", [False, True])
@pytest.mark.parametrize("mixed_spans", [False, True])
def test_nbody_history_cpu_and_gpu_agree(include_sun, mixed_spans):
    r0, v0, t_span = _case(128, mixed_spans=mixed_spans)
    kw = _kw(include_sun)
    cpu = nb.propagate_spacecraft_batch(r0, v0, t_span, 120, use_gpu=False, **kw)
    gpu = nb.propagate_spacecraft_batch(r0, v0, t_span, 120, use_gpu=True, **kw)
    assert isinstance(gpu, nb.SpacecraftBatchResult)
    assert gpu.r.shape == cpu.r.shape == (121, 128, 3)
    np.testing.assert_allclose(gpu.r, cpu.r, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(gpu.v, cpu.v, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(gpu.t_abs, cpu.t_abs, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# Closest approach (O(N) output mode and the fused kernels)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("include_sun", [False, True])
def test_closest_approach_cpu_and_gpu_agree(include_sun):
    r0, v0, t_span = _case(256, seed=2)
    kw = _kw(include_sun)
    cpu = nb.propagate_spacecraft_batch(r0, v0, t_span, 150, use_gpu=False,
                                        output="closest_approach", **kw)
    gpu = nb.propagate_spacecraft_batch(r0, v0, t_span, 150, use_gpu=True,
                                        output="closest_approach", **kw)
    assert isinstance(gpu, nb.ClosestApproachBatchResult)
    _assert_ca_close(gpu, cpu)


def test_gpu_closest_approach_matches_gpu_history():
    r0, v0, t_span = _case(64, seed=3)
    kw = _kw(False)
    hist = nb.propagate_spacecraft_batch(r0, v0, t_span, 80, use_gpu=True, **kw)
    ca = nb.propagate_spacecraft_batch(r0, v0, t_span, 80, use_gpu=True,
                                       output="closest_approach", **kw)
    moon_r, _ = nb.moon_state_batch(hist.t_abs.ravel() - EPOCH0, use_gpu=False,
                                    seed_epoch_ephemeris_seconds=EPOCH0)
    dist = np.linalg.norm(hist.r - moon_r.reshape(hist.r.shape), axis=-1)
    np.testing.assert_allclose(ca.min_distance_km, dist.min(axis=0), rtol=1e-9, atol=1e-6)
    np.testing.assert_array_equal(ca.min_distance_step, np.argmin(dist, axis=0))
    np.testing.assert_allclose(ca.r, hist.r[-1], rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("include_sun", [False, True])
def test_fused_cupy_raw_matches_production(include_sun):
    r0, v0, t_span = _case(256, seed=4)
    kw = _kw(include_sun)
    ref = nb.propagate_spacecraft_batch(r0, v0, t_span, 120, use_gpu=False,
                                        output="closest_approach", **kw)
    got = nf.propagate_closest_approach_fused(r0, v0, t_span, 120, backend="cupy-raw", **kw)
    _assert_ca_close(got, ref)


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
@pytest.mark.parametrize("include_sun", [False, True])
def test_fused_numba_cuda_matches_production(include_sun):
    r0, v0, t_span = _case(256, seed=5)
    kw = _kw(include_sun)
    ref = nb.propagate_spacecraft_batch(r0, v0, t_span, 120, use_gpu=False,
                                        output="closest_approach", **kw)
    got = nf.propagate_closest_approach_fused(r0, v0, t_span, 120, backend="numba-cuda", **kw)
    _assert_ca_close(got, ref)
