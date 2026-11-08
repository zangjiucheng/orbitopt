"""Event location (task #136): closest approach / SOI / entry-interface events
must come from root finding on event functions, not from the nearest saved
sample -- so the located close-flyby distance must not depend on the output
step. Pure numpy/scipy: a Keplerian hyperbolic flyby with an analytic
periapsis stands in for the tudatpy propagation, and scipy's DOP853 plays the
re-propagation hook.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbitopt.verify import events as ev

MU_MOON = 4.9048695e12  # m^3/s^2
RP = 1737.4e3 + 100e3  # 100 km perilune altitude, m
V_INF = 1000.0  # m/s


def _two_body(_t, y):
    r = y[:3]
    return np.concatenate([y[3:], -MU_MOON * r / np.linalg.norm(r) ** 3])


def _propagate(y0, t0, duration, max_step):
    sol = solve_ivp(
        _two_body, (t0, t0 + duration), y0, method="DOP853", rtol=1e-12, atol=1e-6,
        max_step=max_step, dense_output=False,
    )
    return sol.t, sol.y.T


def _sampled(step):
    """Hyperbolic flyby sampled on a fixed grid of ``step`` seconds, with the
    analytic periapsis placed at an epoch that is *not* on the grid."""
    vp = np.sqrt(V_INF**2 + 2 * MU_MOON / RP)
    yp = np.array([RP, 0.0, 0.0, 0.0, vp, 0.0])
    t_peri = 12_345.678
    span = 6 * 3600.0
    # integrate backward then forward from periapsis with tight tolerance
    back = solve_ivp(_two_body, (t_peri, t_peri - span), yp, method="DOP853", rtol=1e-13, atol=1e-8, dense_output=True)
    fwd = solve_ivp(_two_body, (t_peri, t_peri + span), yp, method="DOP853", rtol=1e-13, atol=1e-8, dense_output=True)
    grid = np.arange(t_peri - span + 0.37 * step, t_peri + span, step)
    states = np.array([(back if t < t_peri else fwd).sol(t) for t in grid])
    return grid, states, t_peri


@pytest.mark.parametrize("step", [60.0, 300.0, 900.0])
def test_closest_approach_independent_of_sampling(step):
    epochs, states, t_peri = _sampled(step)
    legacy = np.min(np.linalg.norm(states[:, :3], axis=1))
    located = ev.find_closest_approach_event(epochs, states, refine=_propagate)
    assert located.value == pytest.approx(RP / 1000.0, abs=1e-3)  # < 1 m
    assert located.epoch == pytest.approx(t_peri, abs=1e-2)
    # the legacy nearest-sample answer is strictly worse and grows with step
    assert legacy / 1000.0 - located.value > (0.5 if step >= 300 else 0.0)


def test_closest_approach_without_refine_uses_hermite_dense_output():
    epochs, states, _ = _sampled(60.0)
    located = ev.find_closest_approach_event(epochs, states)
    assert located.value == pytest.approx(RP / 1000.0, abs=0.05)


def test_periapsis_and_soi_crossings_with_moving_body():
    # Same flyby but expressed in a frame where the "body" moves at constant velocity.
    epochs, rel, t_peri = _sampled(300.0)
    vb = np.array([300.0, -200.0, 50.0])

    def body(t):
        return np.concatenate([vb * t, vb])

    states = rel + np.array([body(t) for t in epochs])
    peri = ev.find_periapsides(epochs, states, body_state_fn=body)
    assert len(peri) == 1
    assert peri[0].value == pytest.approx(RP / 1000.0, abs=0.5)

    soi_km = 20_000.0
    crossings = ev.find_sphere_crossings(epochs, states, soi_km, body_state_fn=body)
    assert [c.direction for c in crossings] == [-1, +1]
    for c in crossings:
        assert c.value == pytest.approx(soi_km, abs=1e-3)
    assert crossings[0].epoch < t_peri < crossings[1].epoch


def test_entry_interface_crossing_is_on_the_threshold():
    epochs, states, _ = _sampled(900.0)
    r_ei_km = 1737.4 + 500.0
    found = ev.find_sphere_crossings(epochs, states, r_ei_km, direction=-1, refine=_propagate, first_only=True)
    assert len(found) == 1
    assert np.linalg.norm(found[0].state[:3]) / 1000.0 == pytest.approx(r_ei_km, abs=1e-3)


def test_hermite_dense_output_handles_burn_seam_duplicates():
    epochs = np.array([0.0, 10.0, 10.0, 20.0])
    states = np.zeros((4, 6))
    states[:, 3] = [1.0, 1.0, 2.0, 2.0]
    states[:, 0] = [0.0, 10.0, 10.0, 30.0]
    dense = ev.HermiteDenseOutput(epochs, states)
    assert dense.epochs.tolist() == [0.0, 10.0, 20.0]
    assert dense(15.0)[0] == pytest.approx(20.0)
    assert dense(15.0)[3] == pytest.approx(2.0)


def test_soi_radius_formula():
    # Earth SOI ~ 924,000 km
    assert ev.soi_radius_km(149.598e6, 3.986004418e14, 1.32712440018e20) == pytest.approx(9.25e5, rel=0.01)
