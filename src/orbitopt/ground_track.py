"""Ground tracks: inertial positions -> Earth-fixed -> WGS-84 latitude /
longitude, and the passes of a track near a ground site.

Earth rotation is the same GMST model the launch-site code uses
(:func:`orbitopt.launch_windows.gmst_deg`, UT1 ~= UTC, no polar motion or
precession/nutation), so a track computed here lines up with the orbit planes
:func:`orbitopt.injection.parking_orbit_from_launch` builds from a site.
That is planning-grade -- tens of metres to a few hundred metres on the
ground over a day -- not a precise ITRF conversion.

numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.launch_windows import gmst_deg

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
MEAN_EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius, for great-circle distances


def eci_to_ecef(r_eci, utc_mjd2000) -> np.ndarray:
    """Rotate (N, 3) Earth-centred inertial positions (J2000 equator) into the
    Earth-fixed frame at the matching UTC epochs (N,), by GMST about +Z."""
    r = np.atleast_2d(np.asarray(r_eci, dtype=float))
    g = np.radians(gmst_deg(np.atleast_1d(np.asarray(utc_mjd2000, dtype=float))))
    c, s = np.cos(g), np.sin(g)
    return np.column_stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1], r[:, 2]])


def ecef_to_geodetic(r_ecef_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS-84 geodetic latitude [deg], longitude [deg, -180..180] and height
    [m] of (N, 3) Earth-fixed positions [m] (Bowring's method, 3 iterations:
    sub-millimetre for anything near the surface or in LEO)."""
    r = np.atleast_2d(np.asarray(r_ecef_m, dtype=float))
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(3):
        n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + h_m) * math.cos(lat) * math.cos(lon),
                     (n + h_m) * math.cos(lat) * math.sin(lon),
                     (n * (1.0 - WGS84_E2) + h_m) * math.sin(lat)])


def ecef_to_eci(r_ecef, utc_mjd2000) -> np.ndarray:
    """Inverse of :func:`eci_to_ecef` (e.g. a ground site's inertial path)."""
    r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
    g = np.radians(gmst_deg(np.atleast_1d(np.asarray(utc_mjd2000, dtype=float))))
    c, s = np.cos(g), np.sin(g)
    return np.column_stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]])


def surface_distance_km(lat_deg, lon_deg, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Great-circle distance [km] on a sphere of the mean Earth radius from
    (lat0, lon0) to each (lat, lon) -- haversine, stable for small distances."""
    la, lo = np.radians(np.asarray(lat_deg, dtype=float)), np.radians(np.asarray(lon_deg, dtype=float))
    la0, lo0 = math.radians(lat0_deg), math.radians(lon0_deg)
    a = np.sin((la - la0) / 2.0) ** 2 + math.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2.0) ** 2
    return 2.0 * MEAN_EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class SitePass:
    """One local minimum of a ground track's distance to a site."""

    time_s: float          # in the caller's time base
    distance_km: float     # closest ground distance on this pass
    north_of_site: bool    # the track passes north (True) or south of the site
    heading_deg: float     # ground-track heading at closest approach, clockwise from north

    @property
    def signed_miss_km(self) -> float:
        """+ when the track passes north of the site, - when south."""
        return self.distance_km if self.north_of_site else -self.distance_km

    @property
    def direction(self) -> str:
        return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int(((self.heading_deg + 22.5) % 360.0) // 45.0)]


def site_passes(times_s, lat_deg, lon_deg, site_lat_deg: float, site_lon_deg: float,
                max_distance_km: float = 1500.0) -> list[SitePass]:
    """Passes of a sampled ground track within ``max_distance_km`` of a site:
    each local minimum of the ground distance, refined with a parabola
    through the three samples around it (so the result does not depend on
    where the samples happen to fall)."""
    t = np.asarray(times_s, dtype=float)
    lat, lon = np.asarray(lat_deg, dtype=float), np.asarray(lon_deg, dtype=float)
    d = surface_distance_km(lat, lon, site_lat_deg, site_lon_deg)
    passes = []
    for i in range(1, len(d) - 1):
        if not (d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] < max_distance_km):
            continue
        # Parabolic refinement on the *squared* distance: along a locally
        # straight track d^2 = d0^2 + (v (t - t0))^2 is exactly quadratic in
        # t, whereas d itself is not (it is a hyperbola near the minimum).
        q0, q1, q2 = d[i - 1] ** 2, d[i] ** 2, d[i + 1] ** 2
        denom = q0 - 2.0 * q1 + q2
        shift = 0.5 * (q0 - q2) / denom if denom > 0 else 0.0
        shift = max(-1.0, min(1.0, shift))
        t_min = t[i] + shift * (t[i + 1] - t[i - 1]) / 2.0
        d_min = math.sqrt(max(q1 - 0.25 * (q0 - q2) * shift, 0.0))
        # Heading from the neighbouring samples (east-north, small-angle).
        dlat = lat[i + 1] - lat[i - 1]
        dlon = ((lon[i + 1] - lon[i - 1] + 180.0) % 360.0 - 180.0) * math.cos(math.radians(lat[i]))
        heading = math.degrees(math.atan2(dlon, dlat)) % 360.0
        passes.append(SitePass(time_s=float(t_min), distance_km=float(max(d_min, 0.0)),
                               north_of_site=bool(lat[i] > site_lat_deg), heading_deg=heading))
    return passes
