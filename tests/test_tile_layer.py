"""EarthTileLayer builds arrived tiles a few per tick (no network: a fake
cache handing out one small JPEG for every tile)."""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from orbitopt.viz import earth_tiles as et  # noqa: E402
from orbitopt.viz import tile_layer  # noqa: E402

STARBASE = (25.9966, -97.1555)


class _FakeCache:
    def __init__(self, jpg):
        self.jpg = jpg

    def path(self, tile):
        return self.jpg

    def request(self, tiles):
        pass

    def take_new(self):
        return []

    def shutdown(self):
        pass


@pytest.fixture
def jpg(tmp_path):
    import vtk

    img = vtk.vtkImageData()
    img.SetDimensions(64, 64, 1)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    writer = vtk.vtkJPEGWriter()
    path = tmp_path / "tile.jpg"
    writer.SetFileName(str(path))
    writer.SetInputData(img)
    writer.Write()
    return path


def test_tiles_arrive_over_several_ticks_within_the_budget(jpg, monkeypatch):
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    earth = plotter.add_mesh(pv.Sphere(radius=6378.137, theta_resolution=60, phi_resolution=60))
    layer = tile_layer.EarthTileLayer(plotter, cache=_FakeCache(jpg))
    layer.attach(earth, {"radius": 6378.137})
    layer.set_enabled(True)
    ground = et.geodetic_points_km(np.array(STARBASE[0]), np.array(STARBASE[1])) * np.array([-1.0, -1.0, 1.0])
    up = ground / np.linalg.norm(ground)
    plotter.camera.focal_point = tuple(ground)
    plotter.camera.position = tuple(ground + up * 30.0)
    plotter.camera.up = (0.0, 0.0, 1.0)

    wanted = len(et.select_tiles(layer.camera_view()))
    assert wanted > 20
    ticks, built = 0, []
    while True:
        layer.update()
        ticks += 1
        built.append(len(layer.shown))
        if not layer.pending or ticks > 200:
            break
    assert ticks > 1  # not all in one blocking tick
    assert built == sorted(built) and len(layer.shown) == wanted  # grows to the full set
    assert not layer.pending
    # Every tile shares the first tile's material.
    props = {id(a.GetProperty()) for a in layer._actors.values()}
    assert len(props) == 1
    plotter.close()
