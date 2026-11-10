"""Starship Flight 15 ascent profile and Super Heavy return."""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.ground_track import ecef_to_geodetic, surface_distance_km
from orbitopt.launch_sites import get_launch_site
from orbitopt.viz import starship_ascent as sa


def test_booster_return_lands_on_the_tower_near_the_announced_time():
    site = get_launch_site("starbase")
    b = sa.solve_booster_return(site, 106.54)
    lat, lon, h = ecef_to_geodetic(b.r_ecef[-1][None, :])
    assert surface_distance_km(lat, lon, site.latitude_deg, site.longitude_deg)[0] < 0.01  # within 10 m
    assert h[0] == pytest.approx(sa.TOWER_CATCH_HEIGHT_M, abs=5.0)
    assert b.boostback_end_ground_speed < 0.0                     # heading back toward the pad
    assert 60e3 < b.apogee_m < 150e3
    # The catch time is a prediction, compared with SpaceX's plan (never an input).
    assert abs(b.catch_s - sa.BOOSTER_CATCH_ANNOUNCED_S) < 60.0
    # The landing burn: lit late (after drag has done its share), seconds long.
    assert b.landing_burn_start_s > sa.BOOSTBACK_END_S
    assert 10.0 < b.catch_s - b.landing_burn_start_s < 60.0
    times = np.asarray(b.times_s)
    assert np.all(np.diff(times) > 0)


def test_powered_descent_ends_at_rest_instead_of_hovering():
    # Thrust held against a vanishing velocity used to chatter in a hover
    # until the time limit; the burn must end once the booster is at rest.
    t, x = sa._burn_from(np.array([0.0, 14000.0, -300.0, -1150.0]), 0.0)
    assert t[-1] < 60.0 and np.hypot(x[-1][2], x[-1][3]) <= 1.0 + 1e-6


@pytest.mark.slow
def test_ascent_starts_on_the_pad_and_ends_on_the_orbit():
    pytest.importorskip("tudatpy")
    from orbitopt.viz.starship_flight import SECO_S, predict_return

    site = get_launch_site("starbase")
    pred = predict_return()
    a = sa.build_ascent(site, pred.azimuth_deg, pred.liftoff_mjd2000, SECO_S, pred.states_eq[0])
    _lat, _lon, h_stack = ecef_to_geodetic(a.stack_r_ecef)
    assert h_stack[0] == pytest.approx(50.0, abs=1.0)                     # on the pad
    assert h_stack[-1] == pytest.approx(sa.MECO_ALTITUDE_M, abs=1.0)
    assert np.all(np.diff(h_stack) > -1.0)                                 # climbs
    seco_r, _v = sa.eci_state_to_ecef(pred.states_eq[0], pred.liftoff_mjd2000 + SECO_S / 86400.0)
    assert np.linalg.norm(a.ship_r_ecef[-1] - seco_r) < 1.0                # meets the orbit exactly
