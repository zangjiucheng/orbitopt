"""Draws the streamed EOX imagery (earth_tiles) over the Earth sphere.

One small actor per tile: a curved patch on the WGS-84 ellipsoid in the
Earth sphere's own mesh frame, carrying the Earth actor's matrix as its
user matrix -- so the patches spin and move with the globe (time, frame
changes) without being rebuilt. The tiles lie on the true ellipsoid, above the
globe's flat facets, and the globe is pushed back in depth so it never
shows through; a parent tile stands in only for children that have not
arrived yet.

``update()`` is cheap when nothing changed (camera, Earth orientation and
downloads all the same as last time) and is polled by the window's timer.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pyvista as pv
import vtk

from orbitopt.viz import earth_tiles as et

_KEEP_ACTORS = 320  # hidden tile actors kept around (GPU textures) before eviction
_ATTRIBUTION_NAME = "eox-attribution"


def _np_matrix(m: vtk.vtkMatrix4x4) -> np.ndarray:
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


class EarthTileLayer:
    def __init__(self, plotter, cache: et.TileCache | None = None):
        self.plotter = plotter
        self._cache = cache
        self.enabled = et.tiles_enabled()
        self._earth = None
        self._scale = 1.0
        self._actors: OrderedDict[et.Tile, object] = OrderedDict()
        self._shown: set[et.Tile] = set()
        self._last_key = None
        self._attribution = None

    @property
    def cache(self) -> et.TileCache:
        if self._cache is None:
            self._cache = et.TileCache()
        return self._cache

    @property
    def shown(self) -> set[et.Tile]:
        return set(self._shown)

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget every actor (the plotter was just cleared)."""
        self._actors.clear()
        self._shown.clear()
        self._earth = None
        self._attribution = None
        self._last_key = None

    def attach(self, earth_actor, body: dict) -> None:
        self._earth = earth_actor
        # The globe gives way wherever a tile touches it (its facets mostly
        # sit below the tiles, which follow the true ellipsoid).
        mapper = earth_actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, 4.0)
        self._scale = float(body["radius"]) / (et.WGS84_A_M / 1000.0)
        self._last_key = None

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        self._last_key = None

    def shutdown(self) -> None:
        if self._cache is not None:
            self._cache.shutdown()

    # ------------------------------------------------------------ camera
    def camera_view(self) -> et.CameraView | None:
        if self._earth is None:
            return None
        m = _np_matrix(self._earth.GetMatrix())
        inv = np.linalg.inv(m)
        cam = self.plotter.camera
        pos = np.asarray(cam.GetPosition(), dtype=float)
        focal = np.asarray(cam.GetFocalPoint(), dtype=float)
        local = (inv @ np.append(pos, 1.0))[:3]
        direction = inv[:3, :3] @ (focal - pos)
        flip = np.array([-1.0, -1.0, 1.0])  # mesh-local -> Earth-fixed (180 deg about the pole)
        ecef = local * flip / self._scale
        direction = direction * flip
        norm = np.linalg.norm(direction)
        if norm == 0 or not np.all(np.isfinite(ecef)):
            return None
        width, height = self.plotter.renderer.GetSize()
        fov = math.radians(cam.GetViewAngle())
        aspect = width / max(height, 1)
        half_diag = math.atan(math.tan(fov / 2.0) * math.sqrt(1.0 + aspect * aspect))
        return et.CameraView(ecef, direction / norm, max(height, 1) / fov, half_diag)

    # ------------------------------------------------------------ update
    def _state_key(self):
        cam = self.plotter.camera
        m = _np_matrix(self._earth.GetMatrix()) if self._earth is not None else None
        return (tuple(np.round(cam.GetPosition(), 9)), tuple(np.round(cam.GetFocalPoint(), 9)),
                round(cam.GetViewAngle(), 6), tuple(self.plotter.renderer.GetSize()),
                None if m is None else m.round(12).tobytes(), self.enabled)

    def update(self) -> bool:
        """Re-select and show tiles for the current camera. Returns whether
        anything drawn changed (the caller renders)."""
        new = self.cache.take_new() if (self._cache is not None or self.enabled) else []
        key = self._state_key()
        if key == self._last_key and not new:
            return False
        self._last_key = key
        wanted: list[et.Tile] = []
        view = self.camera_view() if (self.enabled and self._earth is not None
                                      and self._earth.GetVisibility()) else None
        if view is not None and np.linalg.norm(view.position) < 8.0 * et.WGS84_A_M / 1000.0:
            wanted = et.select_tiles(view)
        wanted.sort(key=lambda t: t[0])  # coarse first: quick cover, then detail
        if wanted:
            self.cache.request(wanted)
        show: set[et.Tile] = set()
        for tile in wanted:
            t = tile
            while t[0] >= et.MIN_LEVEL:
                if self.cache.path(t) is not None:
                    show.add(t)
                    break
                t = et.parent(t)
        changed = show != self._shown or bool(new)
        for tile in self._shown - show:
            actor = self._actors.get(tile)
            if actor is not None:
                actor.SetVisibility(False)
        matrix = self._earth.GetMatrix() if self._earth is not None else None
        for tile in show:
            actor = self._actors.get(tile) or self._make_actor(tile)
            if actor is None:
                continue
            self._actors.move_to_end(tile)
            actor.SetUserMatrix(matrix)
            actor.SetVisibility(True)
        self._shown = {t for t in show if t in self._actors}
        self._evict()
        self._show_attribution(bool(self._shown))
        return changed

    def sync_transform(self) -> None:
        """The Earth turned or moved (time / frame change): carry the tiles along."""
        if self._earth is None or not self._shown:
            return
        matrix = self._earth.GetMatrix()
        for tile in self._shown:
            self._actors[tile].SetUserMatrix(matrix)

    # ------------------------------------------------------------ actors
    def _make_actor(self, tile: et.Tile):
        path = self.cache.path(tile)
        if path is None:
            return None
        try:
            texture = pv.Texture(str(path))
        except Exception:  # noqa: BLE001 -- a corrupt cache file: drop it, fetch again later
            path.unlink(missing_ok=True)
            return None
        texture.SetInterpolate(True)
        texture.SetMipmap(True)
        texture.SetWrap(vtk.vtkTexture.ClampToEdge)
        points, faces, tcoords, normals = et.tile_mesh_arrays(tile, self._scale)
        mesh = pv.PolyData(points, faces)
        mesh.active_texture_coordinates = tcoords
        mesh.point_data.active_normals = normals
        z, row, col = tile
        actor = self.plotter.add_mesh(mesh, texture=texture, name=f"eox-{z}-{row}-{col}", pickable=False,
                                      smooth_shading=True, reset_camera=False, render=False)
        # No polygon offset on the tiles: on macOS's OpenGL one offset
        # "unit" is far larger than the depth-buffer step, and any offset
        # put the imagery in front of trajectories hundreds of metres up.
        # The tiles follow the true ellipsoid, which lies above the coarse
        # globe's facets; the globe alone is pushed back (attach()).
        self._actors[tile] = actor
        return actor

    def _evict(self) -> None:
        while len(self._actors) > _KEEP_ACTORS + len(self._shown):
            for tile in self._actors:
                if tile not in self._shown:
                    self.plotter.remove_actor(self._actors.pop(tile), render=False)
                    break
            else:
                return

    def _show_attribution(self, on: bool) -> None:
        if on and self._attribution is None:
            self._attribution = self.plotter.add_text(
                "Imagery: " + et.ATTRIBUTION, position="lower_right", font_size=7,
                color="#a9a6bb", name=_ATTRIBUTION_NAME, render=False)
        if self._attribution is not None:
            self._attribution.SetVisibility(on)
