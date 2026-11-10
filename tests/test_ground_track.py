"""Ground-track conversions and site-pass detection (numpy only)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.ground_track import (
    MEAN_EARTH_RADIUS_KM,
    WGS84_A_M,
    ecef_to_eci,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    site_passes,
    surface_distance_km,
)
from orbitopt.launch_windows import gmst_deg


def test_geodetic_round_trip_on_and_above_the_surface():
    for lat, lon, h in [(0.0, 0.0, 0.0), (25.997, -97.157, 5.0), (-45.0, 170.0, 275e3), (89.0, 10.0, 1e3)]:
        glat, glon, gh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
        assert glat[0] == pytest.approx(lat, abs=1e-9)
        assert glon[0] == pytest.approx(lon, abs=1e-9)
        assert gh[0] == pytest.approx(h, abs=1e-3)
    assert geodetic_to_ecef(0.0, 0.0)[0] == pytest.approx(WGS84_A_M)


def test_eci_ecef_rotation_follows_gmst():
    t = np.array([9800.0, 9800.25])
    r = np.array([[7000e3, 0.0, 0.0], [7000e3, 0.0, 0.0]])
    ecef = eci_to_ecef(r, t)
    lon = np.degrees(np.arctan2(ecef[:, 1], ecef[:, 0]))
    # An inertial point on the x-axis sits at longitude -GMST.
    assert ((lon + gmst_deg(t) + 180.0) % 360.0 - 180.0) == pytest.approx([0.0, 0.0], abs=1e-9)
    np.testing.assert_allclose(ecef_to_eci(ecef, t), r, atol=1e-6)


def test_surface_distance():
    assert surface_distance_km(0.0, 1.0, 0.0, 0.0)[()] == pytest.approx(MEAN_EARTH_RADIUS_KM * math.pi / 180.0)
    assert surface_distance_km(10.0, 20.0, 10.0, 20.0)[()] == pytest.approx(0.0, abs=1e-9)


def test_site_passes_refines_minimum_and_reports_side_and_heading():
    # A track along the equator-parallel at latitude 1 deg, heading east,
    # sampled coarsely: it passes 1 deg (~111 km) north of a site at (0, 0.37).
    t = np.arange(0.0, 21.0)
    lon = np.linspace(-10.0, 10.0, 21)
    lat = np.full_like(lon, 1.0)
    (p,) = site_passes(t, lat, lon, 0.0, 0.37, max_distance_km=500.0)
    assert p.time_s == pytest.approx(10.37, abs=0.02)  # between samples, not snapped to one
    assert p.distance_km == pytest.approx(MEAN_EARTH_RADIUS_KM * math.radians(1.0), rel=2e-3)
    assert p.north_of_site and p.signed_miss_km > 0
    assert p.direction == "E" and p.heading_deg == pytest.approx(90.0, abs=1e-6)
    assert site_passes(t, lat, lon, 30.0, 0.0, max_distance_km=500.0) == []
