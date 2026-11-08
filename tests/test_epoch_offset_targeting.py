"""The lunar targeters' Moon lookups must use the propagation's *absolute*
epoch, ``reference_epoch_ephemeris_seconds + initial_epoch + t``.

The same physical trajectory written with a different epoch split (e.g.
``(ref=R, initial=0)`` vs ``(ref=R-T, initial=T)``) must give identical
miss vectors / B-plane states / closest-approach times. tudatpy is replaced
by a deterministic fake that only depends on absolute time, so the test runs
without SPICE kernels."""
from __future__ import annotations

import numpy as np
import pytest

dc = pytest.importorskip("orbitopt.verify.differential_correction")

from orbitopt.verify.tudat_propagate import ClosestApproachResult, PropagationResult  # noqa: E402

R = 8.0e8
SPLIT = 5.0e5


def _moon_pos(t_abs):
    return np.array([3.8e8 * np.cos(t_abs * 2.66e-6), 3.8e8 * np.sin(t_abs * 2.66e-6), 0.0])


def _moon_vel(t_abs):
    w = 2.66e-6
    return np.array([-3.8e8 * w * np.sin(t_abs * w), 3.8e8 * w * np.cos(t_abs * w), 0.0])


@pytest.fixture
def fake_tudat(monkeypatch):
    calls = {}

    def propagate(r0, v0, initial_epoch, arcs, **_kw):
        # Final state depends only on absolute time, like real dynamics in a
        # time-varying (ephemeris-driven) field. Epochs are relative.
        calls["initial_epoch"] = float(initial_epoch)
        dur = [a["duration"] for a in arcs if a["type"] == "coast"]
        edges = np.cumsum([0.0] + dur)
        epochs = np.concatenate([np.linspace(edges[i], edges[i + 1], 5) for i in range(len(dur))])
        t_end = float(initial_epoch) + epochs[-1]
        state = np.concatenate([_moon_pos(t_end) + 1.0e7, _moon_vel(t_end) + 100.0])
        states = np.tile(state, (len(epochs), 1))
        segs = [(5 * i, 5 * (i + 1)) for i in range(len(dur))]
        return PropagationResult(epochs=epochs, states=states, final_state=state,
                                 epoch_offset=float(initial_epoch), segments=segs)

    def closest(result, ref, body_name="Moon", **_kw):
        calls["closest_ref"] = float(ref)
        return ClosestApproachResult(epoch=float(result.epochs[-1]) * 0.9, distance_km=1.0e4,
                                     spacecraft_state=result.final_state)

    monkeypatch.setattr(dc, "propagate_multi_arc", propagate)
    monkeypatch.setattr(dc, "body_position_at_absolute_epoch", lambda b, t, **k: _moon_pos(t))
    monkeypatch.setattr(dc, "body_velocity_at_absolute_epoch", lambda b, t, **k: _moon_vel(t))
    monkeypatch.setattr(dc, "find_closest_approach", closest)
    return calls


ARGS = (np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
SPLITS = [(R, 0.0), (R - SPLIT, SPLIT)]


def test_miss_vector_independent_of_epoch_split(fake_tudat):
    out = [dc._miss_vector_m(*ARGS, ini, np.zeros(3), 3.0e5, ref, ("Earth",), 60.0)[0] for ref, ini in SPLITS]
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[0], [1.0e7, 1.0e7, 1.0e7], atol=1e-6)


def test_relative_state_independent_of_epoch_split(fake_tudat):
    out = [dc._relative_state_m(*ARGS, ini, np.zeros(3), 3.0e5, ref, ("Earth",), 60.0)[:2] for ref, ini in SPLITS]
    np.testing.assert_allclose(out[0][0], out[1][0], atol=1e-6)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-9)


def test_refine_coast_duration_uses_absolute_epoch(fake_tudat):
    for ref, ini in SPLITS:
        dc._refine_coast_duration(*ARGS, ini, np.zeros(3), 3.0e5, ref, ("Earth",), 60.0)
        assert fake_tudat["closest_ref"] == pytest.approx(R)
