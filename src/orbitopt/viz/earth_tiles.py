"""High-resolution Earth imagery streamed in tiles: EOX Sentinel-2 cloudless.

The packaged Earth texture is an 8192 x 4096 equirectangular map, ~5 km per
pixel -- fine from orbit, a blur at Starbase from 12 km up. This module
picks, for the current camera, which tiles of EOX's Sentinel-2 cloudless
mosaic (10 m per pixel at its finest) would add detail, and fetches them in
the background into a disk cache. scene_renderer.EarthTileLayer drapes them
over the globe.

The tile grid is EOX's WMTS "WGS84" matrix set (EPSG:4326): level z is
2^(z+1) x 2^z tiles of 256 px, each 180 / 2^z degrees square, rows from the
north pole, columns from 180 W -- the same latitude/longitude layout as the
base texture, so tiles need no reprojection.

Selection (select_tiles) is a screen-space-error quadtree, as map viewers
do: a tile splits into its four children while one of its texels would
cover more than ~1.5 screen pixels, largest first, under a fixed tile
budget, skipping tiles behind the horizon or outside the view. Tiles
coarser than MIN_LEVEL add nothing over the base texture and are not drawn.

Imagery: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH
(Contains modified Copernicus Sentinel data 2024)", CC BY-NC-SA 4.0 --
free for non-commercial use with that attribution (ATTRIBUTION, shown in
the view while tiles are drawn). Set ORBITOPT_EARTH_TILES=off to disable
all network access (the test suite does).
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbitopt.ground_track import WGS84_A_M, WGS84_E2

LAYER = "s2cloudless-2024"
URL_TEMPLATE = "https://tiles.maps.eox.at/wmts/1.0.0/{layer}/default/WGS84/{z}/{row}/{col}.jpg"
ATTRIBUTION = ("Sentinel-2 cloudless 2024 by EOX IT Services GmbH "
               "(contains modified Copernicus Sentinel data 2024), CC BY-NC-SA 4.0")
TILE_PX = 256
MIN_LEVEL = 5    # 5.6 deg tiles, ~2.4 km per pixel: just finer than the 8K base map
MAX_LEVEL = 13   # ~10 m per pixel, Sentinel-2's own resolution (deeper levels are upsampled)
MAX_TILES = 160  # drawn at once
SPLIT_PX = 1.5   # split while one texel would cover more screen pixels than this
_A_KM = WGS84_A_M / 1000.0

Tile = tuple[int, int, int]  # (z, row, col)


def tiles_enabled() -> bool:
    return os.environ.get("ORBITOPT_EARTH_TILES", "on").strip().lower() not in ("0", "off", "false", "no")


def tile_size_deg(z: int) -> float:
    return 180.0 / (1 << z)


def tile_bounds(tile: Tile) -> tuple[float, float, float, float]:
    """(south, north, west, east) in degrees."""
    z, row, col = tile
    d = tile_size_deg(z)
    north = 90.0 - row * d
    west = -180.0 + col * d
    return north - d, north, west, west + d


def tile_containing(lat: float, lon: float, z: int) -> Tile:
    d = tile_size_deg(z)
    row = min(int((90.0 - lat) // d), (1 << z) - 1)
    col = int(((lon + 180.0) % 360.0) // d)
    return z, row, col


def children(tile: Tile) -> list[Tile]:
    z, row, col = tile
    return [(z + 1, 2 * row + i, 2 * col + j) for i in (0, 1) for j in (0, 1)]


def parent(tile: Tile) -> Tile:
    z, row, col = tile
    return z - 1, row // 2, col // 2


def geodetic_points_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """WGS-84 Earth-fixed points [km] on the ellipsoid (h = 0)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A_KM / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon),
                     n * (1.0 - WGS84_E2) * np.sin(lat)], axis=-1)


def geodetic_normals(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


@dataclass(frozen=True)
class CameraView:
    """The camera in the Earth-fixed frame [km]: where it is, where it looks,
    and how many screen pixels one radian spans (viewport height / fov)."""

    position: np.ndarray
    direction: np.ndarray  # unit
    px_per_rad: float
    half_fov_rad: float    # half the *diagonal* field of view

    def subpoint(self) -> tuple[float, float]:
        from orbitopt.ground_track import ecef_to_geodetic

        lat, lon, _h = ecef_to_geodetic(self.position * 1000.0)
        return float(lat[0]), float(lon[0])


_SAMPLES = np.linspace(0.0, 1.0, 3)


def _tile_samples(tile: Tile) -> tuple[np.ndarray, np.ndarray]:
    s, n, w, e = tile_bounds(tile)
    lat = s + (n - s) * _SAMPLES[:, None] * np.ones((1, 3))
    lon = w + (e - w) * _SAMPLES[None, :] * np.ones((3, 1))
    return geodetic_points_km(lat.ravel(), lon.ravel()), geodetic_normals(lat.ravel(), lon.ravel())


def _contains(tile: Tile, lat: float, lon: float) -> bool:
    s, n, w, e = tile_bounds(tile)
    return s <= lat <= n and w <= lon <= e


def tile_screen_error(tile: Tile, view: CameraView, sub: tuple[float, float],
                      look: tuple[float, float] | None) -> float | None:
    """Screen pixels one texel of ``tile`` would cover at its nearest visible
    point, or None when the tile cannot be seen (behind the horizon or
    outside the view cone)."""
    points, normals = _tile_samples(tile)
    to_cam = view.position - points
    dist = np.linalg.norm(to_cam, axis=1)
    under_camera = _contains(tile, *sub)
    if tile[0] >= 3:
        facing = np.einsum("ij,ij->i", normals, to_cam) > -1e-9 * dist
        cos_off = np.einsum("ij,j->i", -to_cam, view.direction) / np.maximum(dist, 1e-12)
        in_cone = cos_off >= math.cos(min(view.half_fov_rad * 1.15, math.pi / 2))
        looked_at = look is not None and _contains(tile, *look)
        if not (under_camera or looked_at) and not np.any(facing & in_cone):
            return None
    nearest = float(np.min(dist[np.einsum("ij,ij->i", normals, to_cam) > 0]) if np.any(
        np.einsum("ij,ij->i", normals, to_cam) > 0) else np.min(dist))
    if under_camera:
        altitude = float(np.linalg.norm(view.position)) - _A_KM * (1.0 - 0.5 * WGS84_E2)
        nearest = min(nearest, max(altitude, 1e-3))
    texel_km = math.radians(tile_size_deg(tile[0])) * _A_KM / TILE_PX
    return texel_km / max(nearest, 1e-3) * view.px_per_rad


def _look_point(view: CameraView) -> tuple[float, float] | None:
    """Where the view direction meets the (spherical) Earth, as lat/lon."""
    p, d = view.position, view.direction
    b = float(np.dot(p, d))
    c = float(np.dot(p, p)) - _A_KM ** 2
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    if t < 0:
        return None
    q = p + t * d
    return math.degrees(math.asin(q[2] / np.linalg.norm(q))), math.degrees(math.atan2(q[1], q[0]))


def select_tiles(view: CameraView, max_tiles: int = MAX_TILES, min_level: int = MIN_LEVEL,
                 max_level: int = MAX_LEVEL, split_px: float = SPLIT_PX) -> list[Tile]:
    """The tiles to draw (at ``min_level`` or finer), coarsest-first order
    irrelevant: refine the visible tile with the largest screen error until
    every tile is sharp enough or the budget is spent."""
    sub, look = view.subpoint(), _look_point(view)
    heap: list[tuple[float, Tile]] = []
    for tile in ((0, 0, 0), (0, 0, 1)):
        err = tile_screen_error(tile, view, sub, look)
        if err is not None:
            heapq.heappush(heap, (-err, tile))
    done: list[Tile] = []
    drawn = 0  # tiles at min_level or finer, in `done` or still in the heap
    while heap:
        neg_err, tile = heapq.heappop(heap)
        own = 1 if tile[0] >= min_level else 0
        if -neg_err <= split_px or tile[0] >= max_level or drawn - own + 4 > max_tiles:
            done.append(tile)
            continue
        drawn -= own
        for child in children(tile):
            err = tile_screen_error(child, view, sub, look)
            if err is not None:
                heapq.heappush(heap, (-err, child))
                drawn += 1 if child[0] >= min_level else 0
    return [t for t in done if t[0] >= min_level]


def default_cache_dir() -> Path:
    env = os.environ.get("ORBITOPT_TILE_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "orbitopt" / "tiles"


class TileCache:
    """Tiles on disk, downloaded on demand by a small thread pool.

    ``path(tile)`` answers at once: the cached file, or None (and the tile is
    queued). Nothing here touches Qt or VTK, so the worker threads never do
    either; the renderer polls ``take_new()`` on its own thread. After a
    failed request (offline, server error) a tile is not retried for a
    minute, and while the network is down at most one probe runs at a time.
    """

    RETRY_AFTER_S = 60.0
    MAX_PENDING = 24

    def __init__(self, cache_dir: Path | None = None, layer: str = LAYER, workers: int = 6,
                 fetch=None):
        self.dir = (cache_dir or default_cache_dir()) / layer
        self.layer = layer
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eox-tiles")
        self._lock = threading.Lock()
        self._pending: set[Tile] = set()
        self._failed: dict[Tile, float] = {}
        self._new: list[Tile] = []
        self._offline_until = 0.0
        self._fetch = fetch or self._http_get

    def file(self, tile: Tile) -> Path:
        z, row, col = tile
        return self.dir / str(z) / str(row) / f"{col}.jpg"

    def path(self, tile: Tile) -> Path | None:
        f = self.file(tile)
        if f.exists():
            return f
        return None

    def request(self, tiles: list[Tile]) -> None:
        """Queue downloads for the missing ones, in the given (priority) order."""
        now = time.monotonic()
        with self._lock:
            for tile in tiles:
                if len(self._pending) >= self.MAX_PENDING:
                    break
                if tile in self._pending or self._failed.get(tile, -1e18) + self.RETRY_AFTER_S > now:
                    continue
                if self.file(tile).exists():
                    continue
                if now < self._offline_until and self._pending:
                    break  # one probe at a time while the network is down
                self._pending.add(tile)
                self._pool.submit(self._download, tile)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def take_new(self) -> list[Tile]:
        """Tiles that finished downloading since the last call."""
        with self._lock:
            new, self._new = self._new, []
        return new

    def _http_get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "orbitopt-mission-control/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                raise OSError(f"HTTP {resp.status}")
            return resp.read()

    def _download(self, tile: Tile) -> None:
        z, row, col = tile
        url = URL_TEMPLATE.format(layer=self.layer, z=z, row=row, col=col)
        try:
            data = self._fetch(url)
            if not data.startswith(b"\xff\xd8"):
                raise OSError("not a JPEG")
            f = self.file(tile)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".part{threading.get_ident()}")
            tmp.write_bytes(data)
            tmp.replace(f)  # atomic: a reader never sees half a file
            with self._lock:
                self._new.append(tile)
                self._offline_until = 0.0
        except Exception:  # noqa: BLE001 -- offline / HTTP error / disk: skip this tile for now
            with self._lock:
                self._failed[tile] = time.monotonic()
                self._offline_until = time.monotonic() + 10.0
        finally:
            with self._lock:
                self._pending.discard(tile)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def tile_mesh_arrays(tile: Tile, scale: float, n: int | None = None):
    """Points (mesh-local, scene units), triangle-strip quads, texture
    coordinates and normals for one tile on the WGS-84 ellipsoid.

    Mesh-local is the Earth sphere's own frame (see scene_renderer.
    _equirectangular_sphere): Earth-fixed rotated 180 deg about the pole
    (texture longitude 0 at local -X), times ``scale`` = drawn radius /
    WGS-84 equatorial radius. Texture v = 1 at the tile's north edge, as on
    the globe."""
    z = tile[0]
    n = n or (16 if z < 9 else 8)
    s, nn, w, e = tile_bounds(tile)
    lat = np.linspace(s, nn, n + 1)
    lon = np.linspace(w, e, n + 1)
    lon_g, lat_g = np.meshgrid(lon, lat)
    pts = geodetic_points_km(lat_g.ravel(), lon_g.ravel())
    nrm = geodetic_normals(lat_g.ravel(), lon_g.ravel())
    flip = np.array([-1.0, -1.0, 1.0])
    pts = pts * flip * scale
    nrm = nrm * flip
    u = (lon_g.ravel() - w) / (e - w)
    v = (lat_g.ravel() - s) / (nn - s)
    cols = n + 1
    r = np.arange(n)[:, None]
    c = np.arange(n)[None, :]
    tl = (r * cols + c).ravel()
    faces = np.column_stack([np.full(tl.size, 4), tl, tl + 1, tl + cols + 1, tl + cols]).ravel()
    return pts, faces, np.column_stack([u, v]).astype(np.float32), nrm.astype(np.float32)
