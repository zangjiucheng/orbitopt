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


def _line(r0, v, t0, t1, n):
    t = np.linspace(t0, t1, n)
    return t, np.array([np.r_[r0 + v * (tt - t0), v] for tt in t])


@pytest.mark.parametrize("v_post", [(0.0, 200.0, 0.0), (0.0, 900.0, 0.0), (-300.0, 300.0, 0.0)])
@pytest.mark.parametrize("use_refine", [False, True])
def test_closest_approach_does_not_interpolate_across_burn_seam(v_post, use_refine):
    """Acceptance regression: a burn right after closest approach used to make
    the pre-seam interval interpolate toward the *post*-burn velocity (the
    pre-burn state was dropped), producing a spurious nearer periapsis. The
    straight-line coast passes the body at exactly 9 km."""
    t1, s1 = _line(np.array([-95e3, 9e3, 0.0]), np.array([100.0, 0.0, 0.0]), 0.0, 1000.0, 3)
    t2, s2 = _line(s1[-1, :3], np.array(v_post), 1000.0, 3000.0, 3)
    epochs, states = np.r_[t1, t2], np.vstack([s1, s2])

    def refine(state0, t0, duration, max_step):
        n = int(np.ceil(duration / max_step)) + 1
        return _line(state0[:3], state0[3:], t0, t0 + duration, n)

    e = ev.find_closest_approach_event(epochs, states, refine=refine if use_refine else None)
    assert e.value == pytest.approx(9.0, abs=1e-6)
    assert e.epoch == pytest.approx(950.0, abs=1e-2)


def test_dense_output_uses_pre_burn_state_at_seam():
    t1, s1 = _line(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0, 10.0, 2)
    t2, s2 = _line(s1[-1, :3], np.array([0.0, 5.0, 0.0]), 10.0, 20.0, 2)
    d = ev.HermiteDenseOutput(np.r_[t1, t2], np.vstack([s1, s2]))
    np.testing.assert_allclose(d(10.0, interval=0)[3:], [1.0, 0.0, 0.0])  # pre-burn
    np.testing.assert_allclose(d(10.0)[3:], [0.0, 5.0, 0.0])  # post-burn
    np.testing.assert_allclose(d(5.0), [5.0, 0, 0, 1.0, 0, 0], atol=1e-12)


# --------------------------------------------------------------------------
# Task #161: short-lived events hidden between two saved samples
# --------------------------------------------------------------------------

def _straddled(step, n_side=6):
    """Flyby sampled so the analytic periapsis sits exactly mid-interval: the
    two nearest saved samples are +-step/2 away and every sample is far above
    the periapsis radius."""
    vp = np.sqrt(V_INF**2 + 2 * MU_MOON / RP)
    yp = np.array([RP, 0.0, 0.0, 0.0, vp, 0.0])
    grid = (np.arange(-n_side, n_side) + 0.5) * step
    states = []
    for t in grid:
        sol = solve_ivp(_two_body, (0.0, t), yp, method="DOP853", rtol=1e-13, atol=1e-8)
        states.append(sol.y[:, -1])
    return grid, np.array(states)


GRAZE_STEP = 600.0
R_GRAZE_KM = RP / 1000.0 + 40.0  # true path dips 40 km below this sphere


def _analytic_crossing_epochs(radius_km):
    """|t| at which the hyperbola reaches radius_km, by bisection on a tight propagation."""
    vp = np.sqrt(V_INF**2 + 2 * MU_MOON / RP)
    yp = np.array([RP, 0.0, 0.0, 0.0, vp, 0.0])
    sol = solve_ivp(_two_body, (0.0, GRAZE_STEP), yp, method="DOP853", rtol=1e-13, atol=1e-8,
                    dense_output=True)
    from scipy.optimize import brentq
    t = brentq(lambda tt: np.linalg.norm(sol.sol(tt)[:3]) / 1000.0 - radius_km, 0.0, GRAZE_STEP / 2)
    return -t, t


def test_graze_fixture_samples_are_all_above_threshold():
    epochs, states = _straddled(GRAZE_STEP)
    radii_km = np.linalg.norm(states[:, :3], axis=1) / 1000.0
    assert np.all(radii_km > R_GRAZE_KM + 40.0)  # well clear: invisible to sign bracketing
    assert ev.find_sphere_crossings(epochs, states, R_GRAZE_KM, g_rate=None) == []  # legacy behaviour


@pytest.mark.parametrize("use_refine", [False, True])
def test_graze_between_samples_is_detected(use_refine):
    """Acceptance regression: all saved samples above the threshold, the true
    trajectory crosses below and back up within one output step."""
    epochs, states = _straddled(GRAZE_STEP)
    t_in, t_out = _analytic_crossing_epochs(R_GRAZE_KM)
    refine = _propagate if use_refine else None
    found = ev.find_sphere_crossings(epochs, states, R_GRAZE_KM, refine=refine)
    assert [c.direction for c in found] == [-1, +1]
    tol_t, tol_r = (1e-2, 1e-3) if use_refine else (5.0, 1.0)
    assert found[0].epoch == pytest.approx(t_in, abs=tol_t)
    assert found[1].epoch == pytest.approx(t_out, abs=tol_t)
    for c in found:
        assert np.linalg.norm(c.state[:3]) / 1000.0 == pytest.approx(R_GRAZE_KM, abs=tol_r)

    entry = ev.find_sphere_crossings(epochs, states, R_GRAZE_KM, direction=-1, refine=refine, first_only=True)
    assert len(entry) == 1 and entry[0].direction == -1
    assert entry[0].epoch == pytest.approx(t_in, abs=tol_t)


def test_near_miss_graze_is_resolved_by_refinement():
    """The extremum lies above the sphere: re-propagation must confirm that no
    crossing happens (no false positive), with or without refine."""
    epochs, states = _straddled(GRAZE_STEP)
    r_km = RP / 1000.0 - 5.0  # sphere 5 km below the true periapsis
    assert ev.find_sphere_crossings(epochs, states, r_km) == []
    assert ev.find_sphere_crossings(epochs, states, r_km, refine=_propagate) == []


def test_graze_found_by_refinement_when_hermite_misses_it():
    """Very coarse output: the Hermite dense output alone may not dip below
    the sphere, but the interval is suspicious, so it is re-propagated and
    the true crossings are found."""
    step = 2400.0
    epochs, states = _straddled(step, n_side=3)
    r_km = RP / 1000.0 + 2.0  # only 2 km of penetration
    t_in, t_out = _analytic_crossing_epochs(r_km)
    found = ev.find_sphere_crossings(epochs, states, r_km, refine=_propagate)
    assert [c.direction for c in found] == [-1, +1]
    assert found[0].epoch == pytest.approx(t_in, abs=1e-2)
    assert found[1].epoch == pytest.approx(t_out, abs=1e-2)


def test_find_altitude_crossing_detects_graze_between_samples():
    tp = pytest.importorskip("orbitopt.verify.tudat_propagate", exc_type=ImportError)
    epochs, states = _straddled(GRAZE_STEP)
    t_in, _ = _analytic_crossing_epochs(R_GRAZE_KM)
    body_r_km = 1737.4
    prop = tp.PropagationResult(epochs=epochs, states=states, final_state=states[-1])
    hit = tp.find_altitude_crossing(prop, body_r_km, R_GRAZE_KM - body_r_km, refine=_propagate)
    assert hit is not None
    assert hit.epoch == pytest.approx(t_in, abs=1e-2)
    assert hit.altitude_km == pytest.approx(R_GRAZE_KM - body_r_km, abs=1e-3)
    # the legacy linear method still cannot see it
    assert tp.find_altitude_crossing(prop, body_r_km, R_GRAZE_KM - body_r_km, method="linear") is None
