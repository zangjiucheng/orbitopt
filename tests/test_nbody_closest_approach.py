"""propagate_spacecraft_batch(output="closest_approach") must give the same
fitness as the full-history path while keeping only O(N) state.

The Moon seed state is injected into nbody_gpu's per-epoch cache so these
tests don't need tudatpy/SPICE; the numbers below are a representative
Earth-relative ECLIPJ2000 Moon state (|r| ~ 385,000 km, |v| ~ 1.0 km/s).
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.problems.free_return import FreeReturnScreeningProblem

EPOCH0 = 838_000_000.0  # ~2026-07 ET; any value works, it only keys the seed cache
MOON_R0_KM = np.array([-291_608.0, -266_716.0, 22_850.0])
MOON_V0_KM_S = np.array([0.6439, -0.6974, 0.0136])
SUN_R0_KM = np.array([1.0e8, -1.1e8, 5.0e3])
SUN_V0_KM_S = np.array([22.0, 20.0, 0.0])


@pytest.fixture(autouse=True)
def _seed_moon_cache(monkeypatch):
    key = round(EPOCH0, 3)
    monkeypatch.setitem(nb._MOON_STATE0_KM, key, (MOON_R0_KM.copy(), MOON_V0_KM_S.copy()))
    monkeypatch.setitem(nb._SUN_STATE0_KM, key, (SUN_R0_KM.copy(), SUN_V0_KM_S.copy()))


def _parking_orbit_state():
    r0 = np.array([6_563.0, 0.0, 0.0])
    v_circ = np.sqrt(nb.MU_EARTH_KM3_S2 / np.linalg.norm(r0))
    return r0, np.array([0.0, v_circ, 0.0])


def _random_dvs(n, seed=0):
    rng = np.random.default_rng(seed)
    # Mostly prograde TLI-ish burns plus scatter, so some candidates actually
    # head toward the Moon and some don't.
    dvs = rng.normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    return dvs


def _history_closest(r0, v0, t_span, n_steps, **kw):
    hist = nb.propagate_spacecraft_batch(r0, v0, t_span, n_steps, output="history", **kw)
    moon_r, _ = nb.moon_state_batch(
        hist.t_abs.ravel() - EPOCH0, use_gpu=False, seed_epoch_ephemeris_seconds=EPOCH0,
    )
    moon_r = moon_r.reshape(hist.t_abs.shape[0], r0.shape[0], 3)
    dist = np.linalg.norm(hist.r - moon_r, axis=-1)
    idx = np.argmin(dist, axis=0)
    cols = np.arange(r0.shape[0])
    return hist, dist.min(axis=0), hist.t_abs[idx, cols], idx


@pytest.mark.parametrize("include_sun", [False, True])
def test_closest_approach_matches_history(include_sun):
    n, n_steps = 32, 60
    r_park, v_park = _parking_orbit_state()
    r0 = np.broadcast_to(r_park, (n, 3))
    v0 = v_park + _random_dvs(n)
    t_span = np.linspace(4.0, 8.0, n) * 86400.0  # per-candidate spans too
    kw = dict(
        mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
        include_sun=include_sun, epoch0=EPOCH0, use_gpu=False,
    )

    hist, min_d, min_t, min_idx = _history_closest(r0, v0, t_span, n_steps, **kw)
    ca = nb.propagate_spacecraft_batch(r0, v0, t_span, n_steps, output="closest_approach", **kw)

    assert isinstance(ca, nb.ClosestApproachBatchResult)
    np.testing.assert_allclose(ca.min_distance_km, min_d, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(ca.min_distance_step, min_idx)
    np.testing.assert_allclose(ca.min_distance_epoch_s, min_t, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ca.r, hist.r[-1], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(ca.v, hist.v[-1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ca.t_abs, hist.t_abs[-1], rtol=0, atol=1e-6)


def test_free_return_fitness_identical_between_modes():
    r_park, v_park = _parking_orbit_state()
    kw = dict(
        r0_km=r_park, v0_pre_burn_km_s=v_park, epoch0_seconds=EPOCH0,
        target_distance_km=8_000.0, coast_days=6.0, n_steps=80, use_gpu=False,
        backend="numpy",  # compares output modes; backends: test_free_return_backends.py
    )
    fast = FreeReturnScreeningProblem(**kw)  # default: closest_approach
    full = FreeReturnScreeningProblem(**kw, propagation_output="history")
    assert fast.propagation_output == "closest_approach"

    dvs = _random_dvs(24, seed=1)
    f_fast = fast.batch_fitness(dvs.ravel())
    f_full = full.batch_fitness(dvs.ravel())
    np.testing.assert_allclose(f_fast, f_full, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        fast._last_min_distance_epoch_s, full._last_min_distance_epoch_s, rtol=0, atol=1e-6,
    )
    assert fast.time_to_closest_approach(dvs[3]) == pytest.approx(
        full.time_to_closest_approach(dvs[3]), abs=1e-6,
    )


def test_invalid_output_mode_rejected():
    r_park, v_park = _parking_orbit_state()
    with pytest.raises(ValueError):
        nb.propagate_spacecraft_batch(
            r_park[None], v_park[None], 86400.0, 4, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
            epoch0=EPOCH0, use_gpu=False, output="bogus",
        )
    with pytest.raises(ValueError):
        FreeReturnScreeningProblem(r_park, v_park, EPOCH0, 1.0, 1.0, propagation_output="bogus")


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.slow
def test_closest_approach_memory_is_O_N_not_O_N_steps():
    """Measured with tracemalloc (it tracks numpy's host buffers): history
    mode stores (n_steps+1) * N * 7 float64 (~449 MB at the task's
    20,000 x 400 scale) while closest-approach peak memory must stay flat
    as n_steps grows.
    """
    n = 64
    r_park, v_park = _parking_orbit_state()
    r0 = np.broadcast_to(r_park, (n, 3))
    v0 = v_park + _random_dvs(n, seed=2)
    kw = dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2, epoch0=EPOCH0, use_gpu=False)

    def peak(n_steps, output):
        return _peak_bytes(lambda: nb.propagate_spacecraft_batch(
            r0, v0, 86400.0 * 5, n_steps, output=output, **kw))

    short_steps, long_steps = 10, 60
    history_array_bytes = (long_steps + 1) * n * 7 * 8
    peak_hist = peak(long_steps, "history")
    peak_ca_short = peak(short_steps, "closest_approach")
    peak_ca_long = peak(long_steps, "closest_approach")

    assert peak_hist >= history_array_bytes
    assert peak_ca_long < history_array_bytes / 4
    # O(N): 6x more steps must not grow the closest-approach footprint.
    assert peak_ca_long < 1.25 * peak_ca_short + 4096


@pytest.mark.parametrize("include_sun", [False, True])
def test_closest_approach_memory_flat_with_mixed_durations(include_sun):
    """Regression: with >1 distinct (epoch0, t_span) pair the ephemeris
    precompute used to expand the compact time x pair grid into a full
    (2*n_steps+1, N, 3) time x candidate array per body (~769 MB for the
    Moon alone at 20,000 x 400), silently restoring O(N x n_steps) memory
    in closest_approach mode. The grid must stay compact and be gathered
    one time slice at a time.
    """
    n = 1000
    r_park, v_park = _parking_orbit_state()
    r0 = np.broadcast_to(r_park, (n, 3))
    v0 = v_park + _random_dvs(n, seed=3)
    t_span = np.where(np.arange(n) % 2, 5.0, 6.0) * 86400.0  # two coast times
    epoch0 = EPOCH0 + 3600.0 * (np.arange(n) % 3 == 0)  # and two epochs
    kw = dict(
        mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2, include_sun=include_sun,
        epoch0=epoch0, use_gpu=False, moon_seed_epoch_ephemeris_seconds=EPOCH0,
    )

    def peak(n_steps):
        return _peak_bytes(lambda: nb.propagate_spacecraft_batch(
            r0, v0, t_span, n_steps, output="closest_approach", **kw))

    short_steps, long_steps = 20, 200
    # One expanded (position + indirect) body grid at long_steps.
    expanded_grid_bytes = 2 * (2 * long_steps + 1) * n * 3 * 8
    peak_short = peak(short_steps)
    peak_long = peak(long_steps)

    assert peak_long < expanded_grid_bytes / 4
    # 10x more steps must not grow the footprint beyond the tiny compact grid.
    assert peak_long < 1.25 * peak_short + 256 * 1024


def test_precomputed_body_grid_is_compact_for_mixed_pairs():
    n, n_steps = 50, 12
    dt = np.where(np.arange(n) % 2, 4.0, 5.0) * 86400.0 / n_steps
    epoch0 = np.full(n, EPOCH0)
    ephem, inverse = nb._precompute_body_grid(
        epoch0, dt, n_steps, np, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
        True, nb.MU_SUN_KM3_S2, 20, EPOCH0,
    )
    assert inverse.shape == (n,)
    assert set(inverse.tolist()) == {0, 1}
    for _, r_body, indirect in ephem:
        assert r_body.shape == (2 * n_steps + 1, 2, 3)
        assert indirect.shape == r_body.shape
    # Each candidate's column is the ephemeris at its own half-step epochs.
    moon_r = ephem[0][1]
    k = 7
    t_k = epoch0 + k * 0.5 * dt  # k odd -> t + 0.5*dt, matches grid accumulation to ~ulp
    ref, _ = nb.moon_state_batch(t_k - EPOCH0, use_gpu=False, seed_epoch_ephemeris_seconds=EPOCH0)
    np.testing.assert_allclose(moon_r[k][inverse], ref, rtol=1e-9, atol=1e-3)
