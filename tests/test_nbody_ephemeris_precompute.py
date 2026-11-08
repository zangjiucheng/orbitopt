"""propagate_spacecraft_batch precomputes Moon/Sun positions on the RK4
half-step grid once per call (task #139). Pin that the result is identical
to evaluating the ephemeris inside every RK4 stage, as the integrator used
to. SPICE is bypassed by seeding the module's state0 cache directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.dynamics import nbody_gpu as nb

EPOCH = 8.39e8
MOON0 = (np.array([-3.0e5, 2.4e5, 2.0e4]), np.array([-0.62, -0.78, 0.05]))
SUN0 = (np.array([1.2e8, -8.5e7, 3.0e3]), np.array([17.0, 24.3, 0.0]))


@pytest.fixture(autouse=True)
def _seed_cache(monkeypatch):
    monkeypatch.setattr(nb, "_MOON_STATE0_KM", {round(EPOCH, 3): MOON0})
    monkeypatch.setattr(nb, "_SUN_STATE0_KM", {round(EPOCH, 3): SUN0})


def _reference(r0, v0, t_span, n_steps, epoch0, include_sun):
    """Per-stage ephemeris evaluation (the pre-#139 integrator)."""
    n = r0.shape[0]
    dt = np.broadcast_to(np.atleast_1d(np.asarray(t_span, float)), (n,))
    dt = dt / n_steps
    t = np.broadcast_to(np.atleast_1d(np.asarray(epoch0, float)), (n,)).copy()

    def acc(r, t_abs):
        a = -nb.MU_EARTH_KM3_S2 * r / np.linalg.norm(r, axis=-1)[:, None] ** 3
        terms = [(nb.MU_MOON_KM3_S2, nb._moon_state_core(
            t_abs - EPOCH, np, nb.MU_EARTH_KM3_S2, 20, EPOCH, nb.MU_MOON_KM3_S2)[0])]
        if include_sun:
            terms.append((nb.MU_SUN_KM3_S2, nb._sun_state_core(
                t_abs - EPOCH, np, nb.MU_SUN_KM3_S2, 20, EPOCH)[0]))
        for mu, rb in terms:
            d = rb - r
            a = a + mu * (d / np.linalg.norm(d, axis=-1)[:, None] ** 3
                          - rb / np.linalg.norm(rb, axis=-1)[:, None] ** 3)
        return a

    r, v, c = r0, v0, dt[:, None]
    for _ in range(n_steps):
        k1r, k1v = v, acc(r, t)
        k2r, k2v = v + 0.5 * c * k1v, acc(r + 0.5 * c * k1r, t + 0.5 * dt)
        k3r, k3v = v + 0.5 * c * k2v, acc(r + 0.5 * c * k2r, t + 0.5 * dt)
        k4r, k4v = v + c * k3v, acc(r + c * k3r, t + dt)
        r = r + (c / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (c / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        t = t + dt
    return r, v


@pytest.mark.parametrize("include_sun", [False, True])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_precomputed_ephemeris_matches_per_stage_evaluation(include_sun, heterogeneous):
    rng = np.random.default_rng(1)
    n, n_steps = 40, 30
    r0 = np.broadcast_to([6563.0, 0.0, 0.0], (n, 3)).copy()
    v0 = np.array([0.0, 10.9, 0.3]) + rng.normal(0.0, 0.05, (n, 3))
    if heterogeneous:
        epoch0 = EPOCH + 3600.0 * (np.arange(n) % 2)
        t_span = np.where(np.arange(n) % 3, 6 * 86400.0, 5 * 86400.0)
    else:
        epoch0, t_span = EPOCH, 6 * 86400.0

    res = nb.propagate_spacecraft_batch(
        r0, v0, t_span, n_steps, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
        include_sun=include_sun, epoch0=epoch0, use_gpu=False,
        moon_seed_epoch_ephemeris_seconds=EPOCH,
    )
    r_ref, v_ref = _reference(r0, v0, t_span, n_steps, epoch0, include_sun)
    np.testing.assert_allclose(res.r[-1], r_ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.v[-1], v_ref, rtol=0, atol=1e-9)
