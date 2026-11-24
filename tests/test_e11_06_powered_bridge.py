"""E11-06: the powered-gap bridge of the hot-staging handoff (mission chain)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbitopt.mission import powered_bridge as pb


def _run(meco_t, last_t=120.0, body="stack"):
    track = SimpleNamespace(body_id=body, t_s=[last_t - 2, last_t - 1, last_t])
    return SimpleNamespace(tracks=[track], events=[SimpleNamespace(kind="meco", body_id=body, t_s=meco_t, data={})])


def test_cold_separation_is_never_powered():
    assert pb.is_powered_across_gap(_run(120.0), "stack") is False
    assert pb.is_powered_across_gap(_run(119.0), "stack") is False


def test_still_firing_after_last_sample_is_powered():
    assert pb.is_powered_across_gap(_run(122.0), "stack") is True
    assert pb.is_powered_across_gap(_run(122.0), "other") is False


def test_extrapolation_exact_for_constant_jerk_and_linear_flow():
    t = [0.0, 1.0, 2.0]
    a0, j = 3.0, 0.5
    v = [[a0 * x + j * x * x / 2, 0.0, 0.0] for x in t]
    r = [[a0 * x * x / 2 + j * x ** 3 / 6, 0.0, 0.0] for x in t]
    m = [100.0 - 2.0 * x for x in t]
    r2, v2, m2 = pb.powered_extrapolate(t, r, v, m, 1.5)
    assert r2[0] == pytest.approx(a0 * 3.5 ** 2 / 2 + j * 3.5 ** 3 / 6)
    assert v2[0] == pytest.approx(a0 * 3.5 + j * 3.5 ** 2 / 2)
    assert m2 == pytest.approx(93.0)


def test_extrapolation_rejects_bad_input():
    with pytest.raises(ValueError):
        pb.powered_extrapolate([0, 1, 1], [[0] * 3] * 3, [[0] * 3] * 3, [1, 1, 1], 1.0)
    with pytest.raises(ValueError):
        pb.powered_extrapolate([0, 1, 2], [[0] * 3] * 3, [[0] * 3] * 3, [1, 1, 1], 0.0)


def test_tolerances_never_tighter_than_cold_and_declared():
    assert pb.powered_tolerances(0.5, 0.05) == (1.0, 1.0)
    assert pb.powered_tolerances(5.0, 3.0) == (5.0, 3.0)
    assert pb.powered_mass_tolerance_kg(2.26e6, 1e-3) == pytest.approx(1130.0)
    assert pb.powered_mass_tolerance_kg(1.0, 5.0) == 5.0
    assert "one sample" in pb.POWERED_BRIDGE_METHOD
