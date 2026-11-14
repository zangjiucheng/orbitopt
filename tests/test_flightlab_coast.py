"""Checks for orbitopt.flightlab.coast (M-06 display propagation): a
classical two-body Kepler arc. Display-quality, so the gates are the
physics invariants a coast must honour -- orbit closure, radius bounds,
element conservation -- not trajectory accuracy claims. Compute-free,
stdlib only, like the contract tests."""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.checks import orbital_elements
from orbitopt.flightlab.coast import EARTH_MU, coast_arc, j2_coast

RE = 6378137.0


def _circular_state(altitude_m=700e3):
    r = RE + altitude_m
    v = math.sqrt(EARTH_MU / r)
    return [r, 0.0, 0.0], [0.0, v, 0.0]


def test_circular_orbit_closes_after_one_period():
    r0, v0 = _circular_state()
    period = 2.0 * math.pi * math.sqrt((RE + 700e3) ** 3 / EARTH_MU)
    times, pos, vel = coast_arc(r0, v0, period, samples=721)
    assert times[0] == 0.0 and times[-1] == pytest.approx(period)
    for (x0, y0, z0), (x1, y1, z1) in zip([r0, v0], [pos[-1], vel[-1]]):
        assert x1 == pytest.approx(x0, abs=1e-3)
        assert y1 == pytest.approx(y0, abs=1e-3)
        assert z1 == pytest.approx(z0, abs=1e-3)


def test_circular_orbit_is_half_an_orbit_across_the_planet():
    r0, v0 = _circular_state()
    period = 2.0 * math.pi * math.sqrt((RE + 700e3) ** 3 / EARTH_MU)
    _, pos, vel = coast_arc(r0, v0, period / 2.0, samples=361)
    r = math.dist((0.0, 0.0, 0.0), pos[-1])
    assert pos[-1][0] == pytest.approx(-r, abs=1e-3 * r)
    speed = math.dist((0.0, 0.0, 0.0), vel[-1])
    assert speed == pytest.approx(math.dist((0.0, 0.0, 0.0), v0), rel=1e-9)


def test_elliptical_arc_stays_between_apsides():
    r0, v0 = _circular_state(400e3)
    v0 = [0.0, v0[1] * 0.98, 0.0]  # slightly below circular: an eccentric ellipse
    start = orbital_elements(r0, v0)
    _, pos, _ = coast_arc(r0, v0, start["period_s"], samples=500)
    radii = [math.dist((0.0, 0.0, 0.0), p) for p in pos]
    rp, ra = start["periapsis_altitude_m"] + RE, start["apoapsis_altitude_m"] + RE
    assert min(radii) == pytest.approx(rp, rel=1e-6)
    assert max(radii) == pytest.approx(ra, rel=1e-6)


def test_elements_are_invariant_along_the_arc():
    r0, v0 = _circular_state(400e3)
    v0 = [0.0, v0[1] * 0.97, v0[1] * 0.05]  # eccentric and inclined
    start = orbital_elements(r0, v0)
    times, pos, vel = coast_arc(r0, v0, start["period_s"], samples=240)
    assert times[0] == 0.0 and all(b > a for a, b in zip(times, times[1:]))
    end = orbital_elements(pos[-1], vel[-1])
    assert end["semi_major_axis_m"] == pytest.approx(start["semi_major_axis_m"], rel=1e-8)
    assert end["eccentricity"] == pytest.approx(start["eccentricity"], abs=1e-8)
    assert end["inclination_deg"] == pytest.approx(start["inclination_deg"], abs=1e-7)


def test_open_orbit_raises():
    # escape speed at 200 km: a parabolic state has no parking orbit to coast
    r = RE + 200e3
    v_esc = math.sqrt(2.0 * EARTH_MU / r)
    with pytest.raises(ValueError, match="elliptical"):
        coast_arc([r, 0.0, 0.0], [0.0, v_esc, 0.0], 1000.0)


def test_input_validation():
    r0, v0 = _circular_state()
    with pytest.raises(ValueError, match="duration"):
        coast_arc(r0, v0, 0.0)
    with pytest.raises(ValueError, match="samples"):
        coast_arc(r0, v0, 100.0, samples=1)
    with pytest.raises(ValueError, match="3-vector"):
        coast_arc([r0[0], r0[1]], v0, 100.0)


# --- j2_coast: the short staging-gap bridge ---------------------------------

def test_j2_coast_without_j2_matches_the_kepler_arc():
    r0, v0 = _circular_state(400e3)
    v0 = [v0[0], v0[1] * 0.98, 300.0]  # eccentric, inclined
    _, pos, vel = coast_arc(r0, v0, 30.0, samples=2)
    r, v = j2_coast(r0, v0, 30.0, j2=0.0)
    assert max(abs(a - b) for a, b in zip(r, pos[-1])) < 1e-6
    assert max(abs(a - b) for a, b in zip(v, vel[-1])) < 1e-9


def test_j2_coast_adds_the_oblateness_acceleration():
    """Over the 2 s FL-004 staging gap J2 moves velocity by centimetres per
    second -- the residual a Kepler bridge left above the 0.01 m/s gate."""
    r0, v0 = [RE + 185e3, 0.0, 0.0], [0.0, 7500.0, 1000.0]  # equatorial point
    _, _, vel = coast_arc(r0, v0, 2.0, samples=2)
    r, v = j2_coast(r0, v0, 2.0)
    dv = math.dist(v, vel[-1])
    # |a_J2| on the equator = 1.5 J2 mu Re^2 / r^4, radially inward
    a_j2 = 1.5 * 1.08262668e-3 * EARTH_MU * RE ** 2 / (RE + 185e3) ** 4
    assert dv == pytest.approx(2.0 * a_j2, rel=0.02)
    assert v[0] < vel[-1][0]  # extra inward pull on the equator


def test_j2_coast_follows_the_pole_axis():
    """J2 is symmetric about the pole: rotating the state and the pole
    together rotates the result the same way."""
    r0, v0 = [RE + 300e3, 1e5, 2e5], [10.0, 7600.0, 900.0]
    r, v = j2_coast(r0, v0, 5.0)
    rot = lambda p: [p[1], p[2], p[0]]  # cyclic axis permutation
    r2, v2 = j2_coast(rot(r0), rot(v0), 5.0, pole=(0.0, 1.0, 0.0))
    assert max(abs(a - b) for a, b in zip(r2, rot(r))) < 1e-6
    assert max(abs(a - b) for a, b in zip(v2, rot(v))) < 1e-9


def test_j2_coast_coasts_suborbital_states_and_rejects_bad_input():
    r, v = j2_coast([RE + 80e3, 0.0, 0.0], [500.0, 2000.0, 0.0], 2.0)
    assert math.hypot(*r) > RE
    with pytest.raises(ValueError):
        j2_coast([RE, 0.0, 0.0], [0.0, 7000.0, 0.0], 0.0)
    with pytest.raises(ValueError):
        j2_coast([RE, 0.0, 0.0], [0.0, 7000.0, 0.0], 1.0, pole=(0.0, 0.0, 0.0))
