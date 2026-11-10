"""EOX tile selection, cache and mesh (no network: a fake fetcher)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from orbitopt.viz import earth_tiles as et

STARBASE = (25.9966, -97.1555)


def _view(lat, lon, h_km, tilt_deg=0.0, fov_deg=30.0, px=1800):
    p = et.geodetic_points_km(np.array(lat), np.array(lon))
    up = et.geodetic_normals(np.array(lat), np.array(lon))
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    t = math.radians(tilt_deg)
    d = -up * math.cos(t) + north * math.sin(t)
    return et.CameraView(p + up * h_km, d / np.linalg.norm(d), px / math.radians(fov_deg), math.radians(fov_deg))


def test_tile_grid_matches_eox_wgs84_matrix_set():
    # Level 0 is two 180-degree tiles; rows from the north pole, columns from 180 W.
    assert et.tile_bounds((0, 0, 0)) == (-90.0, 90.0, -180.0, 0.0)
    assert et.tile_bounds((0, 0, 1)) == (-90.0, 90.0, 0.0, 180.0)
    # The z13 tile over Starbase is the one fetched from EOX by hand (row 2912, col 3770).
    assert et.tile_containing(*STARBASE, 13) == (13, 2912, 3770)
    s, n, w, e = et.tile_bounds((13, 2912, 3770))
    assert s <= STARBASE[0] <= n and w <= STARBASE[1] <= e
    kids = et.children((5, 3, 7))
    assert all(et.parent(k) == (5, 3, 7) for k in kids)
    assert {et.tile_bounds(k)[0] for k in kids} == {et.tile_bounds((5, 3, 7))[0], sum(et.tile_bounds((5, 3, 7))[:2]) / 2}


def test_selection_sharpens_near_the_camera_and_stays_in_budget():
    near = et.select_tiles(_view(*STARBASE, 12.0))
    assert near and {t[0] for t in near} == {et.MAX_LEVEL}
    assert et.tile_containing(*STARBASE, et.MAX_LEVEL) in near
    oblique = et.select_tiles(_view(*STARBASE, 12.0, tilt_deg=60.0))
    assert len(oblique) <= et.MAX_TILES and min(t[0] for t in oblique) < et.MAX_LEVEL  # coarser toward the horizon
    orbit = et.select_tiles(_view(*STARBASE, 400.0))
    assert 8 <= min(t[0] for t in orbit) and max(t[0] for t in orbit) <= 10
    # From GEO the packaged 8K map is already as sharp as the screen: nothing to fetch.
    assert et.select_tiles(_view(*STARBASE, 36000.0)) == []
    # Never the far side of the planet.
    assert all(et.tile_bounds(t)[2] < 0 for t in orbit)


def test_cache_downloads_once_and_backs_off_after_a_failure(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "/9/" in url:
            raise OSError("offline")
        return b"\xff\xd8 fake jpeg"

    cache = et.TileCache(tmp_path, fetch=fetch, workers=2)
    tile = (8, 10, 20)
    assert cache.path(tile) is None
    cache.request([tile, tile])
    deadline = time.time() + 5
    while cache.busy and time.time() < deadline:
        time.sleep(0.01)
    assert cache.take_new() == [tile] and cache.take_new() == []
    assert cache.path(tile) == tmp_path / et.LAYER / "8" / "10" / "20.jpg"
    assert calls == [et.URL_TEMPLATE.format(layer=et.LAYER, z=8, row=10, col=20)]
    cache.request([tile])  # cached: no second download
    bad = (9, 1, 1)
    cache.request([bad])
    while cache.busy and time.time() < deadline:
        time.sleep(0.01)
    cache.request([bad])  # failed a moment ago: not retried yet
    assert len(calls) == 2 and cache.path(bad) is None and not cache.busy
    cache.shutdown()


def test_tiles_are_off_in_the_test_suite():
    assert not et.tiles_enabled()


def test_tile_mesh_sits_on_the_ellipsoid_in_the_globe_frame():
    tile = (13, 2912, 3770)
    pts, faces, uv, normals = et.tile_mesh_arrays(tile, scale=1.0, n=4)
    assert pts.shape == (25, 3) and faces.size == 16 * 5 and uv.shape == (25, 2)
    from orbitopt.ground_track import ecef_to_geodetic

    ecef = pts * np.array([-1.0, -1.0, 1.0])  # globe mesh frame: Greenwich at local -X
    lat, lon, h = ecef_to_geodetic(ecef * 1000.0)
    s, n, w, e = et.tile_bounds(tile)
    assert np.allclose(h, 0.0, atol=1e-3)
    assert lat.min() == pytest.approx(s) and lat.max() == pytest.approx(n)
    assert lon.min() == pytest.approx(w) and lon.max() == pytest.approx(e)
    # v = 1 along the north edge, as the globe's texture rows run.
    assert np.allclose(uv[lat > n - 1e-9, 1], 1.0) and np.allclose(uv[lat < s + 1e-9, 1], 0.0)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
