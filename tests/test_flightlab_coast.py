"""Checks for orbitopt.flightlab.coast (M-06 display propagation): a
classical two-body Kepler arc. Display-quality, so the gates are the
physics invariants a coast must honour -- orbit closure, radius bounds,
element conservation -- not trajectory accuracy claims. Compute-free,
stdlib only, like the contract tests."""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.checks import orbital_elements
from orbitopt.flightlab.coast import EARTH_MU, coast_arc

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
