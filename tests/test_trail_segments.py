"""Spacecraft trails: Hermite playback split into segments at impulsive burns
(a burn is a pre-/post-burn sample pair at one time), and thinning that
never drops either half of such a pair."""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.viz.scene import thin_trail_indices

MU = 398600.4418 * 86400.0 ** 2  # km^3/day^2
R0 = 7000.0


def _kepler_circle(r, t, phase=0.0):
    w = np.sqrt(MU / r ** 3)
    a = w * np.asarray(t, dtype=float) + phase
    pos = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(a)], axis=-1)
    vel = np.stack([-r * w * np.sin(a), r * w * np.cos(a), np.zeros_like(a)], axis=-1)
    return pos, vel


def _burn_trail(n=30):
    """Circular coast, an instantaneous radial kick at t_b, then a straight
    line at the post-burn velocity -- exact positions are known everywhere."""
    period = 2.0 * np.pi / np.sqrt(MU / R0 ** 3)
    t_b = 0.5 * period
    t1 = np.linspace(0.0, t_b, n)
    p1, v1 = _kepler_circle(R0, t1)
    dv = 0.3 * np.linalg.norm(v1[-1]) * p1[-1] / np.linalg.norm(p1[-1])
    v_post = v1[-1] + dv
    t2 = np.linspace(t_b, t_b + 0.05 * period, n)
    p2 = p1[-1] + (t2 - t_b)[:, None] * v_post
    v2 = np.repeat(v_post[None, :], n, axis=0)
    times = np.concatenate([t1, t2])
    return times, np.vstack([p1, p2]), np.vstack([v1, v2]), t_b, p1[-1], v_post


def _truth(t, t_b, p_b, v_post):
    if t <= t_b:
        return _kepler_circle(R0, np.array([t]))[0][0]
    return p_b + (t - t_b) * v_post


def test_thin_keeps_both_halves_of_every_burn_and_the_ends():
    times = np.concatenate([np.linspace(0, 1, 500), np.linspace(1, 2, 500)[:-1], [2.0], np.linspace(2, 3, 500)])
    burns = np.flatnonzero(np.diff(times) == 0.0)
    assert len(burns) == 2
    keep = thin_trail_indices(times, 20, keep=[123])
    assert np.all(np.diff(keep) > 0)
    for b in burns:
        assert b in keep and b + 1 in keep
    assert {0, len(times) - 1, 123} <= set(keep.tolist())
    assert len(keep) <= 20 + 2 * len(burns) + 1 + 2


def test_thin_is_identity_when_short_enough():
    np.testing.assert_array_equal(thin_trail_indices([0.0, 1.0, 1.0, 2.0], 10), np.arange(4))


def test_hermite_never_interpolates_across_a_burn():
    pytest.importorskip("pyvista")
    from orbitopt.viz.scene_renderer import _sample_trail, _sample_trail_many

    times, pos, vel, t_b, p_b, v_post = _burn_trail()
    h = times[1] - times[0]
    ts = np.concatenate([
        np.linspace(times[0], times[-1], 301),
        [t_b, t_b - 0.3 * h, t_b + 0.3 * h, t_b - 1e-9, t_b + 1e-9],
    ])
    many = _sample_trail_many(times, pos, ts, vel)
    for t, m in zip(ts, many):
        one = _sample_trail(times, pos, float(t), vel)[1]
        np.testing.assert_allclose(one, m, atol=1e-6)
        # Within each segment the cubic Hermite tracks the true path closely;
        # a cubic spanning the burn would bow out by ~|dv| * h / 4 (hundreds
        # of km here). The post-burn segment is a straight line, i.e. exact.
        assert np.linalg.norm(one - _truth(float(t), t_b, p_b, v_post)) < 1.0
    np.testing.assert_allclose(_sample_trail(times, pos, t_b, vel)[1], p_b)


def test_burn_without_split_samples_would_be_wrong():
    """Guards the premise: dropping the pre-burn half of the pair makes the
    interval before the burn use the post-burn velocity, and misplaces the
    body by far more than the split trail does."""
    pytest.importorskip("pyvista")
    from orbitopt.viz.scene_renderer import _sample_trail

    times, pos, vel, t_b, p_b, v_post = _burn_trail()
    k = int(np.flatnonzero(np.diff(times) == 0.0)[0])
    merged = np.delete(np.arange(len(times)), k)  # keep only the post-burn sample
    t = t_b - 0.5 * (times[1] - times[0])
    err_merged = np.linalg.norm(_sample_trail(times[merged], pos[merged], t, vel[merged])[1] - _truth(t, t_b, p_b, v_post))
    err_split = np.linalg.norm(_sample_trail(times, pos, t, vel)[1] - _truth(t, t_b, p_b, v_post))
    assert err_merged > 50.0 * max(err_split, 1e-3)
