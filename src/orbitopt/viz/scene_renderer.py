"""Populates a PyVista plotter (a plain ``pv.Plotter`` for the scripting
API in pv_viewer.py, or a ``pyvistaqt.QtInteractor`` embedded in the
Mission Control app in app.py -- both share the same drawing API, so one
renderer serves both) from a SceneData dict. This is the one place that
knows how to turn {bodies, orbits, trails} into VTK meshes; neither caller
duplicates that logic.

Deliberately has NO on-screen text/slider/legend widgets of its own (unlike
this package's simpler pv_viewer.show_scene, which draws those with VTK's
built-in 2D widgets for a quick, dependency-light script viewer) -- the app
wants real, richly-stylable Qt widgets for that chrome instead, so this
module's job stops at the 3D content and a small data API
(``set_time``/``distances_at``) for whatever HUD the caller builds around it.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

_MARKER_BASE_SIZE = 8.0
_MARKER_WEIGHT_SIZE = 1.1
# theta/phi mesh resolution. 48 left visible flat facets and a polygonal limb
# in a LEO close-up (Earth from a few hundred km); 128 (256 x 128 quads, ~33k
# points) keeps the limb smooth there and is still cheap at ~10 bodies.
_SPHERE_RESOLUTION = 128
# Texture filtering: mipmaps stop the 8K Earth from shimmering when it is
# small on screen; anisotropic filtering keeps it sharp at grazing angles
# (looking along the horizon from orbit).
_TEXTURE_ANISOTROPY = 16.0

# _equirectangular_sphere places texture longitude 0 (a body's prime meridian)
# at the mesh's local -X (azimuth 180); an IAU body-fixed frame puts it at +X.
# So orienting a mesh by a body's IAU->frame basis (orientationBasis) needs
# this extra 180-deg spin about the shared pole to reconcile the two
# prime-meridian conventions -- see viz.scene.earth_texture_mesh_azimuth_deg.
_MESH_PRIME_MERIDIAN_OFFSET_DEG = 180.0


def _rotation_z(angle_deg: float) -> np.ndarray:
    """3x3 right-handed rotation by ``angle_deg`` about +Z."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _actor_orientation_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """The (rx, ry, rz) Euler angles (degrees) a VTK Prop3D's ``orientation``
    needs to reproduce the 3x3 ``rotation``. Goes through vtkTransform so the
    decomposition uses VTK's own orientation convention exactly (rather than
    hand-rolling an Euler order that must match it), then the actor rotates
    about its origin and is translated by ``position`` as usual -- so this
    composes with set_time's per-tick ``actor.position`` the same way the
    previous +Z-only ``orientation`` did."""
    transform = vtk.vtkTransform()
    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, float(rotation[i, j]))
    transform.SetMatrix(matrix)
    return transform.GetOrientation()

MANEUVER_COLOR = "#ff5a3c"  # burn / delta-v arrows (distinct from amber trails)
# Optimizer candidates' screening arcs: the picked one in the app's "active"
# cyan, the rest a quiet neutral so they read as context, not as flights.
CANDIDATE_SELECTED_COLOR = "#5ec8ff"
CANDIDATE_COLOR = "#a9a6bb"


def _default_view_direction(scene: dict) -> np.ndarray | None:
    """Which direction the initial camera should look *from*, so a mission
    with a real launch/departure site (GOES's Cape Canaveral longitude, a
    lunar TLI burn, ...) opens already facing the hemisphere/region that
    burn actually happens in, instead of PyVista's canned "iso" preset --
    which has no idea where anything interesting in the scene is and, for
    GOES specifically, happened to face Eurasia while the real launch site
    (confirmed correct by direct SPICE-based position recovery, not
    assumed) sits on the opposite, unseen side of Earth. Every other
    Earth/Sun-relative direction in the scene (orbits, trails) is
    unaffected -- this only orients the DEFAULT camera angle, not any data.

    Uses the earliest timeline event with a ``vector`` (a burn's position)
    for its AZIMUTH only (which side of the central body to face) -- a burn
    site sits exactly in its orbit's own plane, so looking from that exact
    direction (elevation included) collapsed a multi-orbit campaign like
    GOES's edge-on into a thin line. The elevation is instead fixed at a
    pleasant isometric-like angle, matching the "iso" preset's own oblique
    look, so a mission with a real launch site opens facing the right
    hemisphere without losing the wide campaign shape a top-down or
    edge-on angle would flatten. Returns None (caller keeps whatever
    fallback it already has) if the scene has no such event, e.g. Solar
    System.
    """
    timeline = scene.get("timeline")
    if not timeline:
        return None
    vector_events = [e for e in timeline.get("events", []) if "vector" in e]
    if not vector_events:
        return None
    earliest = min(vector_events, key=lambda e: e.get("time", 0.0))
    position = np.asarray(earliest["vector"]["position"], dtype=float)
    if np.linalg.norm(position[:2]) == 0:
        return None  # burn sits exactly on the pole axis -- azimuth undefined
    azimuth = np.arctan2(position[1], position[0])
    elevation = np.radians(35.0)  # close to a true isometric's 35.26 degrees
    return np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])


def marker_size(body: dict) -> float:
    return _MARKER_BASE_SIZE + _MARKER_WEIGHT_SIZE * body.get("radiusDisplay", 4.0)


@lru_cache(maxsize=None)
def _load_texture(filename: str) -> pv.Texture | None:
    """Load and cache a packaged texture image by filename (see
    orbitopt.viz.scene.TEXTURE) from orbitopt/viz/assets/textures/.

    Returns None -- never raises -- if the asset is missing or fails to
    decode, so a body whose `texture` field doesn't resolve falls back to
    the point-marker rendering (see _sphere_actor) instead of taking down
    the whole scene load over one bad/missing image.
    """
    try:
        traversable = resources.files("orbitopt").joinpath("viz", "assets", "textures", filename)
        with resources.as_file(traversable) as path:
            texture = pv.Texture(str(path))
            texture.interpolate = True
            texture.mipmap = True
            texture.SetMaximumAnisotropicFiltering(_TEXTURE_ANISOTROPY)
            return texture
    except Exception:  # noqa: BLE001 -- any load/decode failure just means "no texture"
        return None


def _equirectangular_sphere(radius: float) -> pv.PolyData:
    """A UV sphere carrying per-vertex texture coordinates for an
    equirectangular (lat/long) image -- the projection every texture in
    assets/textures follows (row 0 = north pole, columns = longitude).

    Built explicitly rather than via ``pv.Sphere().texture_map_to_sphere()``
    because that filter's spherical UV assignment does not track azimuth
    linearly on the installed VTK: it collapses the whole image into a thin
    meridian band, rendering every textured body (Earth, Moon, Sun, planets)
    as vertical smears instead of a wrapped map. Here u is set directly from
    each vertex's azimuth and v from its polar angle, so the wrap is exact.

    Conventions, all so the existing calibration keeps working unchanged:
      * u = azimuth / 2pi, so real longitude 0 (Greenwich) lands at local
        -X and u increases with the same right-handed sense about +Z that
        SceneRenderer's rotation uses -- exactly what
        viz.scene.earth_texture_mesh_azimuth_deg = (longitude + 180) documents
        and viz.geo_raising's launch-site orientation relies on.
      * v = 1 - phi/pi, putting the +Z pole (this app's rotation axis, see
        advance_rotation) at the image's north edge -- so looking down +Z
        shows the Arctic, not Antarctica.
      * theta spans a full 0..2pi INCLUSIVE, duplicating the seam meridian, so
        the wrap column interpolates u 0.98->1.0 rather than 0.98->0.0 (which
        would smear the whole image across one seam triangle). The seam sits
        at local +X, i.e. the antimeridian -- open Pacific, where it's least
        visible.
    """
    n_theta, n_phi = _SPHERE_RESOLUTION * 2, _SPHERE_RESOLUTION
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)  # inclusive -> seam duplicated
    phi = np.linspace(0.0, np.pi, n_phi + 1)            # inclusive -> both poles
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    sin_phi = np.sin(phi_grid)
    points = np.column_stack([
        (radius * sin_phi * np.cos(theta_grid)).ravel(),
        (radius * sin_phi * np.sin(theta_grid)).ravel(),
        (radius * np.cos(phi_grid)).ravel(),
    ])
    tcoords = np.column_stack([
        (theta_grid / (2.0 * np.pi)).ravel(),
        (1.0 - phi_grid / np.pi).ravel(),
    ]).astype(np.float32)

    columns = n_theta + 1
    row = np.arange(n_phi)[:, None]
    col = np.arange(n_theta)[None, :]
    top_left = (row * columns + col).ravel()
    quads = np.column_stack([
        np.full(top_left.size, 4),
        top_left, top_left + 1, top_left + columns + 1, top_left + columns,
    ]).ravel()

    mesh = pv.PolyData(points, quads)
    mesh.active_texture_coordinates = tcoords
    # Explicit outward normals (radius direction) so smooth_shading lights the
    # sphere correctly without a normals recompute.
    mesh.point_data.active_normals = (points / radius).astype(np.float32)
    return mesh


def _sphere_actor(plotter, body: dict, name: str):
    """Build and add a real, textured 3D sphere for `body`, centered at the
    origin with the actor's own position transform left at (0,0,0) -- the
    caller (SceneRenderer.load/set_time) moves it via `actor.position`,
    which is a cheap rigid-body transform update, not a per-frame mesh
    rebuild. Returns None (not a raised exception) if `body` has no
    texture/radius or the texture fails to load, so the caller can fall
    back to the point-marker rendering used for every other body kind.
    """
    texture_name = body.get("texture")
    radius = body.get("radius")
    if texture_name is None or radius is None:
        return None
    texture = _load_texture(texture_name)
    if texture is None:
        return None

    sphere = _equirectangular_sphere(float(radius))
    flattening = float(body.get("flattening") or 0.0)
    if flattening > 0.0:
        # Oblate ellipsoid about the mesh's +Z -- which the actor orientation
        # (orientationBasis) turns onto the body's real pole. Normals of
        # x^2/a^2 + y^2/a^2 + z^2/c^2 = 1 are along (x, y, z a^2/c^2).
        squash = 1.0 - flattening
        points = np.asarray(sphere.points).copy()
        points[:, 2] *= squash
        sphere.points = points
        normals = points.copy()
        normals[:, 2] /= squash * squash
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        sphere.point_data.active_normals = normals.astype(np.float32)
        # Map texture rows to *geodetic* latitude, as equirectangular maps are
        # drawn: after the squash a vertex's mesh angle is the reduced
        # latitude beta, and tan(phi) = tan(beta) / (1 - f). Without this a
        # mid-latitude feature sat ~3 km (Earth, 45 deg) off its true spot.
        r_xy = np.hypot(points[:, 0], points[:, 1])
        beta = np.arctan2(points[:, 2] / squash, r_xy)
        phi = np.arctan(np.tan(beta) / squash)
        tcoords = np.asarray(sphere.active_texture_coordinates).copy()
        tcoords[:, 1] = (0.5 + phi / np.pi).astype(np.float32)
        sphere.active_texture_coordinates = tcoords
    # A star is its own light source -- lighting=False renders its raw
    # texture colors with no shading falloff, so it reads as uniformly
    # bright regardless of viewing angle instead of having an implausible
    # "dark side" like a lit planet.
    is_self_lit = body.get("kind") == "star"
    return plotter.add_mesh(
        sphere, texture=texture, name=name, pickable=True,
        lighting=not is_self_lit, smooth_shading=True,
    )


def _validated_trail(body: dict) -> tuple[list, list]:
    """Return (times, positions) for ``body["trail"]``, raising a clear,
    catchable ``ValueError`` instead of letting an empty or ragged trail
    reach the indexing below as an uncaught ``IndexError``.

    A scene document can be schema-valid (orbitopt/schemas/scene-1.0.json
    only requires ``times``/``positions`` to exist, not that they're
    non-empty or the same length -- draft 2020-12 has no clean way to
    express "same length as this other property") and still have an empty
    or mismatched trail, e.g. a producer bug that emits ``times: []``. Every
    trail consumer below (position_at_time, trail_index_at_time, load())
    goes through this so that case fails loudly here, at the one place that
    knows what a trail is supposed to look like, rather than as an
    IndexError several stack frames later that bypasses the app's normal
    QMessageBox error-dialog path.
    """
    trail = body["trail"]
    times = trail.get("times", [])
    positions = trail.get("positions", [])
    body_id = body.get("id", "<unknown>")
    if len(times) == 0 or len(positions) == 0:
        raise ValueError(
            f"body {body_id!r} has an empty trail (times has {len(times)} "
            f"entries, positions has {len(positions)}) -- a trail needs at "
            "least one times/positions pair to be rendered."
        )
    if len(times) != len(positions):
        raise ValueError(
            f"body {body_id!r} has a mismatched trail -- times has "
            f"{len(times)} entries but positions has {len(positions)}; "
            "they must be the same length (one position per timestamp)."
        )
    return times, positions


def _trail_velocities(body: dict) -> np.ndarray | None:
    """The trail's optional ``velocities`` as an (n, 3) array, or None."""
    velocities = body["trail"].get("velocities")
    return None if velocities is None else np.asarray(velocities, dtype=float)


def position_at_time(body: dict, t: float) -> np.ndarray:
    if "trail" not in body:
        return np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
    times, positions = _validated_trail(body)
    return _sample_trail(
        np.asarray(times, dtype=float), np.asarray(positions, dtype=float), t, _trail_velocities(body),
    )[1]


def trail_index_at_time(body: dict, t: float) -> int:
    times, _positions = _validated_trail(body)
    return int(np.searchsorted(times, t))


def _hermite(p0, v0, p1, v1, h, s):
    """Cubic Hermite between (p0, v0) and (p1, v1) over an interval of length
    ``h``, at normalized position ``s`` in [0, 1]. Works on scalars or on
    arrays broadcasting along a leading axis."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1


def _sample_trail(
    times: np.ndarray, positions: np.ndarray, t: float, velocities: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """``(trail_index_at_time, position_at_time)`` for already-converted
    numpy ``times`` (n,) / ``positions`` (n, 3), with one binary search.
    SceneRenderer caches those arrays per body at load() so the per-tick
    path never re-converts a (possibly many-thousand-sample) Python list.

    With ``velocities`` (n, 3) the position between samples is cubic Hermite;
    without, a straight chord. The chord is fine for a densely sampled path
    but not for a sparsely sampled one used as a moving frame origin -- see
    the scene schema's ``trail.velocities``.

    Interpolation is piecewise per segment. A repeated time (an impulsive
    burn exported as its pre-burn then post-burn sample) is a segment break:
    ``searchsorted`` puts a time before the burn in the interval ending at
    the pre-burn sample and a time after it in the interval starting at the
    post-burn sample, so no Hermite cubic ever mixes the velocities from the
    two sides of a burn -- the zero-length interval between the pair is
    never interpolated. Producers must keep both halves of the pair (see
    viz.scene.thin_trail_indices)."""
    idx = int(np.searchsorted(times, t))
    if t <= times[0]:
        return idx, positions[0].copy()
    if t >= times[-1]:
        return idx, positions[-1].copy()
    i = max(1, min(idx, len(times) - 1))
    t0, t1 = times[i - 1], times[i]
    h = t1 - t0
    if h <= 0.0:
        return idx, positions[i - 1].copy()
    frac = (t - t0) / h
    p0, p1 = positions[i - 1], positions[i]
    if velocities is not None:
        return idx, _hermite(p0, velocities[i - 1], p1, velocities[i], h, frac)
    return idx, p0 + frac * (p1 - p0)


def _sample_trail_many(
    times: np.ndarray, positions: np.ndarray, ts: np.ndarray, velocities: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized ``_sample_trail(...)[1]`` for every time in ``ts`` (m,) ->
    (m, 3). Used once per frame switch to move a whole trail into a new
    frame, not on the per-tick path."""
    ts = np.asarray(ts, dtype=float)
    if len(times) == 1:
        return np.repeat(positions[:1], len(ts), axis=0)
    i = np.clip(np.searchsorted(times, ts), 1, len(times) - 1)
    t0, t1 = times[i - 1], times[i]
    h = t1 - t0
    safe_h = np.where(h > 0.0, h, 1.0)
    frac = np.clip(np.where(h > 0.0, (ts - t0) / safe_h, 0.0), 0.0, 1.0)[:, None]
    p0, p1 = positions[i - 1], positions[i]
    if velocities is not None:
        out = _hermite(p0, velocities[i - 1], p1, velocities[i], safe_h[:, None], frac)
    else:
        out = p0 + frac * (p1 - p0)
    out[ts <= times[0]] = positions[0]
    out[ts >= times[-1]] = positions[-1]
    return out


class _PastTrail:
    """The "traveled so far" polyline of one trailed body, updated in place.

    ``set_time`` is the timeline's hot path (every 33 ms playback tick and
    every scrub event). It used to build a fresh ``pv.MultipleLines`` from
    ``full_positions[:idx] + [pos]`` and ``add_mesh`` it under the same name
    -- i.e. allocate a new mesh, tear down the old actor/mapper and upload
    everything to VTK again, every tick. This class keeps ONE PolyData (and
    so one actor/mapper) for the whole scene and edits it in O(1):

    * points: a copy of the trail's samples plus one spare slot at the end
      (N + 1 points). The polyline drawn is ``points[0 .. m]`` where
      ``m = max(idx, 1)`` and ``points[m]`` is temporarily overwritten with
      the body's current interpolated position -- exactly the old
      ``full_positions[:m] + [pos]`` vertex sequence.
    * cells: a single polyline whose connectivity is the static
      ``0 .. N``; only its offsets array ``[0, m + 1]`` changes.

    So an update restores the one previously-borrowed sample, writes the new
    head vertex, bumps one offset, and marks the arrays Modified -- no
    allocation, no actor churn, regardless of trail length or scrub distance.
    """

    def __init__(self, full_positions: np.ndarray):
        self._full = np.asarray(full_positions, dtype=float)
        n = len(self._full)
        self._points = np.empty((n + 1, 3), dtype=float)
        self._points[:n] = self._full
        self._points[n] = self._full[-1]
        # Offsets/connectivity backing arrays are kept alive on self: the VTK
        # arrays below are zero-copy views of them (deep=False).
        self._offsets = np.array([0, 2], dtype=_vtk_id_dtype())
        self._connectivity = np.arange(n + 1, dtype=_vtk_id_dtype())

        self._vtk_points_data = numpy_to_vtk(self._points, deep=False)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(self._vtk_points_data)
        self._vtk_offsets = numpy_to_vtkIdTypeArray(self._offsets, deep=False)
        self._vtk_connectivity = numpy_to_vtkIdTypeArray(self._connectivity, deep=False)
        self._cells = vtk.vtkCellArray()
        self._cells.SetData(self._vtk_offsets, self._vtk_connectivity)

        self.poly = pv.PolyData()
        self.poly.SetPoints(vtk_points)
        self.poly.SetLines(self._cells)
        self._head = 1  # index of the slot currently holding the live position
        self.update(1, self._full[0])

    def update(self, idx: int, pos: np.ndarray) -> None:
        n = len(self._full)
        m = min(max(int(idx), 1), n)
        old = self._head
        # Give the previously borrowed slot its real sample back (slot n is
        # the spare and has no sample of its own).
        if old < n:
            self._points[old] = self._full[old]
        self._points[m] = pos
        self._head = m
        self._offsets[1] = m + 1
        self._vtk_points_data.Modified()
        self._vtk_offsets.Modified()
        self._cells.Modified()
        self.poly.Modified()

    def set_positions(self, full_positions: np.ndarray) -> None:
        """Replace every sample (same count) -- e.g. the same trail seen from a
        different reference frame -- keeping the arrays VTK already holds, so
        no actor or mapper is rebuilt. The live head vertex is rewritten on
        the next ``update``."""
        full = np.asarray(full_positions, dtype=float)
        if full.shape != self._full.shape:
            raise ValueError(f"expected {self._full.shape} positions, got {full.shape}")
        self._full = full
        n = len(full)
        self._points[:n] = full
        self._points[n] = full[-1]
        self.update(self._head, full[min(self._head, n - 1)])


def _vtk_id_dtype():
    """numpy dtype matching this VTK build's vtkIdType (32 or 64 bit)."""
    return np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32


class SceneRenderer:
    """Owns the 3D content of one scene inside a plotter. ``load()`` tears
    down and rebuilds everything (switching missions is rare -- a handful
    of times per session -- so simplicity beats trying to diff old/new
    scenes); ``set_time()`` is the hot path, called every timeline tick or
    time-warp step, and only mutates existing mesh point arrays in place
    (body markers and the "traveled so far" trails, see _PastTrail), never
    rebuilds meshes or swaps actors.

    Reference frames: every position the scene stores is in the scene's own
    frame ("world"). ``set_frame(body_id)`` re-centers the *display* on a
    body -- r_display(t) = r_world(t) - r_body(t) -- so the neighborhood of a
    planet the spacecraft passes can be inspected at its own scale. The
    subtraction is done here in float64 before anything reaches VTK, so a
    185 km orbit around Earth is not lost in the ~1e8 km heliocentric
    magnitudes. World data never changes; distances reported by ``set_time``
    stay in the world frame, so a readout does not change meaning with the
    view.
    """

    def __init__(self, plotter):
        self.plotter = plotter
        self.scene: dict | None = None
        self._moving: dict[str, dict] = {}
        self._static_positions: dict[str, np.ndarray] = {}
        # Static geometry that has to be shifted when the frame origin moves:
        # body actors / marker polys, label polys, and orbit-ring actors.
        self._static_items: dict[str, dict] = {}
        self._orbit_actors: list = []
        self._frame_body_id: str | None = None
        self._time = 0.0
        self._candidate_actors: dict[str, object] = {}
        # Per candidate: world-frame screening arc, its own times, the bodies'
        # positions at those times, the drawn PolyData, and whether the arc can
        # be placed in the current frame (see _candidate_frame_origins).
        self._candidates: dict[str, dict] = {}
        self._candidate_id: str | None = None
        self._candidates_show_all = True
        self._rotating: list[dict] = []
        self._rotation_elapsed_hours = 0.0
        self._has_timeline = False
        # Keyed by the actor object itself, not id(actor) -- VTK's Python
        # wrapper objects are not guaranteed to be the same *Python* object
        # across two separate accesses to the same underlying C++ actor
        # (e.g. one obtained from add_mesh()'s return value, another later
        # from a picker's GetActor()), but they do implement __eq__/__hash__
        # against the underlying object identity, so a plain dict keyed on
        # the actor itself resolves correctly either way. id(actor) doesn't:
        # if nothing else kept the *original* wrapper object alive, it could
        # be garbage-collected and its id() reused by an unrelated actor,
        # silently mapping a later pick to the wrong body (caught by testing
        # -- see the click-to-focus feature this backs in app.py). Keeping
        # the actor itself as the key also keeps it alive, side-stepping the
        # problem entirely regardless of root cause.
        # Rebuilt fresh on every load() (see below); only body marker actors
        # (spheres/points) go in here -- orbit lines, trails, the measure
        # line, and the maneuver arrow all stay pickable=False (see this
        # module's other add_mesh calls), so click-to-select in the viewport
        # (app.py's viewport pick handler) can only ever land on a body.
        self._actor_body_ids: dict = {}

    def body_id_for_actor(self, actor) -> str | None:
        """Map a VTK actor (as returned by a picker's GetActor()) back to the
        body id it represents, or None if it isn't a pickable body marker."""
        return self._actor_body_ids.get(actor) if actor is not None else None

    def load(self, scene: dict) -> None:
        self.plotter.clear()
        self.scene = scene
        self._moving = {}
        self._static_positions = {}
        self._static_items = {}
        self._orbit_actors = []
        self._frame_body_id = None
        self._candidate_actors = {}
        self._candidates = {}
        self._candidate_id = None
        self._rotating = []
        self._rotation_elapsed_hours = 0.0
        self._actor_body_ids = {}
        # A scene with a timeline (a mission) has its own notion of "current
        # time" (scrubbable, playable at any warp speed) that axial rotation
        # should track exactly -- see _apply_rotation_for_time, driven from
        # set_time() below, not advance_rotation's independent wall-clock
        # accumulator (which stays reserved for the timeline-less case, e.g.
        # the static Solar System view, where there's no "current time" for
        # rotation to be in or out of sync with).
        self._has_timeline = bool(scene.get("timeline"))

        for body in scene["bodies"]:
            color = body["color"]

            if "orbit" in body:
                pts = np.asarray(body["orbit"], dtype=float)
                pts = np.vstack([pts, pts[0]])
                orbit_mesh = pv.MultipleLines(pts)
                opacity = 0.32 if body.get("orbitDashed") else 0.5
                self._orbit_actors.append(self.plotter.add_mesh(
                    orbit_mesh, color=color, opacity=opacity, line_width=1, pickable=False,
                ))

            if "trail" in body:
                trail_times, trail_positions = _validated_trail(body)
                t0 = scene.get("timeline", {}).get("min", trail_times[0])
                pos0 = position_at_time(body, t0)
                velocities = _trail_velocities(body)

                sphere_actor = _sphere_actor(self.plotter, body, name=f"marker-{body['id']}")
                point_poly = None
                if sphere_actor is not None:
                    sphere_actor.position = tuple(float(c) for c in pos0)
                    self._register_rotation(body, sphere_actor)
                    self._actor_body_ids[sphere_actor] = body["id"]
                else:
                    point_poly = pv.PolyData(pos0.reshape(1, 3))
                    marker_actor = self.plotter.add_mesh(
                        point_poly, color=color, point_size=marker_size(body),
                        render_points_as_spheres=True, name=f"marker-{body['id']}", pickable=True,
                    )
                    self._actor_body_ids[marker_actor] = body["id"]

                full_positions = np.asarray(trail_positions, dtype=float)
                full_poly = pv.MultipleLines(full_positions)
                self.plotter.add_mesh(full_poly, color=color, line_width=1.1, opacity=0.3, pickable=False)

                past_trail = _PastTrail(full_positions)
                self.plotter.add_mesh(
                    past_trail.poly, color=color, line_width=2.6, opacity=0.95,
                    name=f"past-{body['id']}", pickable=False,
                )

                self._moving[body["id"]] = {
                    "body": body,
                    "point_poly": point_poly,
                    "sphere_actor": sphere_actor,
                    "past_trail": past_trail,
                    "times": np.asarray(trail_times, dtype=float),
                    # World-frame samples (never modified) and the same
                    # samples in the current display frame (what is drawn).
                    "world_positions": full_positions,
                    "velocities": velocities,
                    "full_positions": full_positions,
                    "full_poly": full_poly,
                    "color": color,
                }
            else:
                pos = np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
                self._static_positions[body["id"]] = pos
                sphere_actor = _sphere_actor(self.plotter, body, name=f"marker-{body['id']}")
                marker = None
                if sphere_actor is not None:
                    sphere_actor.position = tuple(float(c) for c in pos)
                    self._register_rotation(body, sphere_actor)
                    self._actor_body_ids[sphere_actor] = body["id"]
                else:
                    marker = pv.PolyData(pos.reshape(1, 3))
                    marker_actor = self.plotter.add_mesh(
                        marker, color=color, point_size=marker_size(body),
                        render_points_as_spheres=True, name=f"marker-{body['id']}", pickable=True,
                    )
                    self._actor_body_ids[marker_actor] = body["id"]
                # Labels come from a named point-data array on a PolyData we
                # keep: given a list of strings instead, pyvista copies the
                # points into its own PolyData and the label could not follow
                # a frame change.
                label_poly = pv.PolyData(pos.reshape(1, 3))
                label_poly.point_data["label"] = np.array([body["name"]])
                self.plotter.add_point_labels(
                    label_poly, "label", font_size=12, text_color=color,
                    shape=None, always_visible=True, show_points=False,
                )
                self._static_items[body["id"]] = {
                    "sphere_actor": sphere_actor, "marker_poly": marker, "label_poly": label_poly,
                }

        for cand in scene.get("candidates", []):
            path = cand["screeningPath"]
            points = np.asarray(path["positions"], dtype=float)
            poly = pv.MultipleLines(points)
            self._candidate_actors[cand["id"]] = self.plotter.add_mesh(
                poly, color=CANDIDATE_COLOR, line_width=1.0, opacity=0.4,
                pickable=False, name=f"candidate-{cand['id']}",
            )
            self._candidates[cand["id"]] = {
                "times": np.asarray(path["times"], dtype=float),
                "world_positions": points,
                "body_positions": {
                    body_id: np.asarray(positions, dtype=float)
                    for body_id, positions in (path.get("bodyPositions") or {}).items()
                },
                "poly": poly,
                "in_frame": True,
            }

        timeline = scene.get("timeline")
        self.set_time(timeline["min"] if timeline else 0.0)

        view_direction = _default_view_direction(scene)
        if view_direction is None:
            self.plotter.camera_position = "iso"
        else:
            # Look from the direction of the earliest burn, keeping "world
            # +Z is up" unless that's degenerate (near-polar direction),
            # matching the up-vector convention _apply_axis_view's presets
            # already use elsewhere in this app.
            world_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(view_direction, world_up)) > 0.95:
                world_up = np.array([0.0, 1.0, 0.0])
            up = world_up - np.dot(world_up, view_direction) * view_direction
            up /= np.linalg.norm(up)
            self.plotter.camera_position = [
                tuple(view_direction), (0.0, 0.0, 0.0), tuple(up),
            ]
        self.plotter.reset_camera()

    def _register_rotation(self, body: dict, sphere_actor) -> None:
        period_hours = body.get("rotationPeriodHours")
        if period_hours:  # excludes None and 0 (a real period is never exactly 0)
            basis = body.get("orientationBasis")
            self._rotating.append({
                "actor": sphere_actor,
                "period_hours": float(period_hours),
                "phase_offset_deg": float(body.get("rotationPhaseOffsetDeg", 0.0)),
                # 3x3 IAU-body-fixed -> scene-frame orientation at t=0 (real
                # obliquity + pole + phase); None falls back to +Z spin.
                "basis": np.asarray(basis, dtype=float) if basis is not None else None,
            })

    def _apply_body_spin(self, entry: dict, spin_deg: float) -> None:
        """Orient one rotating body given ``spin_deg`` (degrees of axial spin
        accumulated so far). With an ``orientationBasis`` (its real IAU
        body-fixed -> scene-frame matrix at t=0), the body is tilted to its
        real obliquity and spun about its REAL pole: rotation = basis @
        Rz(prime-meridian offset + spin), applied as a full actor orientation.
        Without one it falls back to the old upright +Z spin (offset by
        rotationPhaseOffsetDeg). A negative period_hours (Venus, Uranus --
        real retrograde rotators) makes spin_deg negative and so spins the
        body the correct way through either path."""
        basis = entry["basis"]
        if basis is not None:
            rotation = basis @ _rotation_z(_MESH_PRIME_MERIDIAN_OFFSET_DEG + spin_deg)
            entry["actor"].orientation = _actor_orientation_from_matrix(rotation)
        else:
            angle_deg = (entry["phase_offset_deg"] + spin_deg) % 360.0
            entry["actor"].orientation = (0.0, 0.0, angle_deg)

    def advance_rotation(self, delta_hours: float) -> None:
        """Spin every real-sphere body with a rotationPeriodHours by
        ``delta_hours`` of wall-clock-driven time. A body with an
        orientationBasis spins about its real, tilted pole; one without spins
        upright about local +Z (see _apply_body_spin).

        A no-op for a scene with its own timeline: rotation there tracks the
        mission's actual current time exactly (see _apply_rotation_for_time,
        driven from set_time()), so this wall-clock accumulator -- which
        would otherwise drift out of sync with pausing, scrubbing, or
        playback speed -- only applies to a timeline-less scene (the static
        Solar System view), where continuous ambient motion is the only
        sensible notion of "current" rotation.
        """
        if self._has_timeline or not self._rotating:
            return
        self._rotation_elapsed_hours += delta_hours
        for entry in self._rotating:
            spin_deg = self._rotation_elapsed_hours / entry["period_hours"] * 360.0
            self._apply_body_spin(entry, spin_deg)
        self.plotter.render()

    def _apply_rotation_for_time(self, t: float) -> None:
        """Spin every real-sphere body to its orientation at mission time
        ``t`` (days, same units/origin as the scene's timeline), computed
        directly as a function of ``t`` rather than accumulated tick-by-tick.
        Deterministic and idempotent in ``t``, unlike advance_rotation's
        wall-clock accumulator, so scrubbing the timeline to the same point
        always yields the same orientation, pausing playback holds it exactly
        still, and any playback speed (see app.py's time-warp control) speeds
        up or slows down rotation in exact lockstep, because it's *t* driving
        the angle, not real time. Each body's real orientation comes from its
        orientationBasis at t=0 plus the t-driven spin about its real pole
        (see _apply_body_spin) -- so a body whose scene has a real epoch shows
        its actual surface orientation and axial tilt at t=0, not whatever the
        untouched mesh's default orientation happens to be.
        """
        if not self._rotating:
            return
        absolute_hours = t * 24.0
        for entry in self._rotating:
            spin_deg = absolute_hours / entry["period_hours"] * 360.0
            self._apply_body_spin(entry, spin_deg)

    def set_time(self, t: float, render: bool = True) -> dict[str, float]:
        """Move every trailed body to its position at time ``t``, regrow
        its "traveled so far" trail, and return {body_id: distance from
        origin} for every body (moving or static) -- the HUD's job to
        display, not this class's.

        ``render=False`` skips the final redraw so a caller that's about to
        make more scene changes this same tick (track_body, measure_line) can
        batch them into a single render -- see app.py's _on_tick, which used
        to trigger up to three separate renders per tick (one from each of
        these methods), inflating and destabilizing per-tick render cost
        enough to read as playback jitter/stutter.
        """
        self._time = t
        origin = self.frame_origin(t)
        distances = {}
        for body_id, entry in self._moving.items():
            idx, world = _sample_trail(entry["times"], entry["world_positions"], t, entry["velocities"])
            pos = world - origin
            if entry["sphere_actor"] is not None:
                entry["sphere_actor"].position = tuple(float(c) for c in pos)
            else:
                entry["point_poly"].points = pos.reshape(1, 3)

            entry["past_trail"].update(idx, pos)
            # World frame on purpose: the readout means "distance from the
            # scene's center" whatever frame the view is in.
            distances[body_id] = float(np.linalg.norm(world))

        for body_id, pos in self._static_positions.items():
            distances[body_id] = float(np.linalg.norm(pos))

        if self._frame_body_id is not None:
            self._place_static_geometry(origin)

        if self._has_timeline:
            self._apply_rotation_for_time(t)

        if render:
            self.plotter.render()
        return distances

    def body_world_position(self, body_id: str, t: float) -> np.ndarray:
        """Position in the scene's own (world) frame, whatever the view frame."""
        if body_id in self._moving:
            entry = self._moving[body_id]
            return _sample_trail(entry["times"], entry["world_positions"], t, entry["velocities"])[1]
        return self._static_positions.get(body_id, np.zeros(3))

    def body_display_position(self, body_id: str, t: float) -> np.ndarray:
        """Position as drawn: world position minus the frame origin at ``t``.
        Everything that places the camera or draws at a body uses this."""
        return self.body_world_position(body_id, t) - self.frame_origin(t)

    # ------------------------------------------------------------- frames
    # When the view is re-centered on a body, the camera frames the part of
    # the other trails within this many of the body's radii (see
    # local_extent): wide enough for an Earth parking orbit plus the start of
    # the escape hyperbola, tight enough that the orbit is not a dot.
    LOCAL_FRAME_RADII = 20.0

    @property
    def frame_body_id(self) -> str | None:
        return self._frame_body_id

    def frame_origin(self, t: float) -> np.ndarray:
        """World position of the current frame's origin at ``t`` (zero in the
        scene's own frame)."""
        if self._frame_body_id is None:
            return np.zeros(3)
        return self.body_world_position(self._frame_body_id, t)

    def set_frame(self, body_id: str | None, render: bool = True) -> None:
        """Re-center the display on ``body_id`` (None: the scene's own frame).

        Each trail is moved into the new frame sample by sample, r_i - r_F(t_i)
        with the frame body Hermite-interpolated at the trail's own times, so a
        trail drawn around a moving planet is the path *as seen from* that
        planet, not the world path shifted by one fixed offset. Unknown ids
        fall back to the scene's own frame.
        """
        if body_id is not None and body_id not in self._moving and body_id not in self._static_positions:
            body_id = None
        self._frame_body_id = body_id
        for entry in self._moving.values():
            if body_id is None:
                display = entry["world_positions"]
            else:
                display = entry["world_positions"] - self._frame_origin_many(entry["times"])
            entry["full_positions"] = display
            entry["full_poly"].points = display
            entry["past_trail"].set_positions(display)
        self._place_candidates()
        self._place_static_geometry(self.frame_origin(self._time))
        self._apply_candidate_style()
        self.set_time(self._time, render=render)

    # --------------------------------------------------------- candidates
    def highlight_candidate(self, candidate_id: str | None, show_all: bool = True, render: bool = True) -> None:
        """Emphasize one optimizer candidate's screening arc; with
        ``show_all`` False the others are hidden instead of dimmed."""
        self._candidate_id = candidate_id
        self._candidates_show_all = bool(show_all)
        self._apply_candidate_style()
        if render:
            self.plotter.render()

    def candidate_visible(self, candidate_id: str) -> bool:
        actor = self._candidate_actors.get(candidate_id)
        return bool(actor is not None and actor.GetVisibility())

    def candidate_in_frame(self, candidate_id: str) -> bool:
        """Whether the candidate's arc can be drawn in the current frame."""
        entry = self._candidates.get(candidate_id)
        return bool(entry is not None and entry["in_frame"])

    def candidate_display_positions(self, candidate_id: str) -> np.ndarray | None:
        """The candidate's screening arc as drawn in the current frame, or None
        when it cannot be placed in it (and so is hidden)."""
        entry = self._candidates.get(candidate_id)
        if entry is None or not entry["in_frame"]:
            return None
        return np.asarray(entry["poly"].points, dtype=float)

    def _candidate_frame_origins(self, entry: dict) -> np.ndarray | None:
        """The frame body's world position at each of a candidate's own sample
        times, or None if that is unknown.

        A candidate departs on its own date, so the frame body is taken at the
        candidate's times, never at the mission's: its exported
        ``bodyPositions`` first (same ephemeris as the arc); else a fixed body;
        else the body's trail, but only when the trail covers every candidate
        time -- extrapolating a planet's trail by months would misplace the arc
        by ~1e8 km, so such an arc is hidden instead."""
        body_id = self._frame_body_id
        times = entry["times"]
        if body_id is None:
            return np.zeros((len(times), 3))
        if body_id in entry["body_positions"]:
            return entry["body_positions"][body_id]
        if body_id in self._static_positions:
            return np.broadcast_to(self._static_positions[body_id], (len(times), 3))
        trail = self._moving.get(body_id)
        if trail is None:
            return None
        span = trail["times"]
        tol = 1e-9 * max(1.0, abs(float(span[-1])))
        if times.min() < span[0] - tol or times.max() > span[-1] + tol:
            return None
        return _sample_trail_many(span, trail["world_positions"], times, trail["velocities"])

    def _place_candidates(self) -> None:
        """Move every screening arc into the current frame point by point:
        r_display(t_i) = r_arc(t_i) - r_frame(t_i) at the candidate's own t_i.
        In float64 before VTK, like the trails. Runs once per frame switch."""
        for entry in self._candidates.values():
            origins = self._candidate_frame_origins(entry)
            entry["in_frame"] = origins is not None
            if origins is not None:
                entry["poly"].points = entry["world_positions"] - origins

    def _apply_candidate_style(self) -> None:
        # An arc the current frame cannot place (see _candidate_frame_origins)
        # is hidden rather than drawn against the wrong dates.
        for cid, actor in self._candidate_actors.items():
            selected = cid == self._candidate_id
            in_frame = self.candidate_in_frame(cid)
            actor.SetVisibility(in_frame and (selected or self._candidates_show_all))
            prop = actor.prop
            prop.color = CANDIDATE_SELECTED_COLOR if selected else CANDIDATE_COLOR
            # Kept under the mission trail's weight (2.6): the flown trajectory
            # stays the primary line, a candidate arc is a comparison overlay.
            prop.line_width = 1.6 if selected else 1.0
            prop.opacity = 0.8 if selected else 0.4

    def _frame_origin_many(self, ts: np.ndarray) -> np.ndarray:
        body_id = self._frame_body_id
        if body_id in self._moving:
            entry = self._moving[body_id]
            return _sample_trail_many(entry["times"], entry["world_positions"], ts, entry["velocities"])
        return np.broadcast_to(self._static_positions.get(body_id, np.zeros(3)), (len(ts), 3))

    def _place_static_geometry(self, origin: np.ndarray) -> None:
        """Shift every fixed-position body, label and orbit ring by -origin."""
        shift = tuple(float(-c) for c in origin)
        for actor in self._orbit_actors:
            actor.position = shift
        for body_id, item in self._static_items.items():
            pos = self._static_positions[body_id] - origin
            if item["sphere_actor"] is not None:
                item["sphere_actor"].position = tuple(float(c) for c in pos)
            if item["marker_poly"] is not None:
                item["marker_poly"].points = pos.reshape(1, 3)
            item["label_poly"].points = pos.reshape(1, 3)

    def local_extent(self, t: float) -> float:
        """Radius of the neighborhood worth framing around the frame body: the
        farthest point of any other trail within LOCAL_FRAME_RADII of it, so
        the camera fits the local part of the path rather than its whole
        interplanetary length. Falls back to the scene bounds when there is
        no frame body or no radius to scale by."""
        body = next((b for b in self.scene["bodies"] if b["id"] == self._frame_body_id), None) if self.scene else None
        radius = float(body.get("radius", 0.0)) if body else 0.0
        if radius <= 0.0:
            xmin, xmax, ymin, ymax, zmin, zmax = self.plotter.bounds
            return 0.5 * float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])) or 1.0
        limit = self.LOCAL_FRAME_RADII * radius
        nearest = 3.0 * radius
        for body_id, entry in self._moving.items():
            if body_id == self._frame_body_id:
                continue
            d = np.linalg.norm(entry["full_positions"], axis=1)
            d = d[d <= limit]
            if d.size:
                nearest = max(nearest, float(d.max()))
        return nearest

    def frame_camera_on_origin(self, extent: float) -> None:
        """Point the camera at the frame origin from its current direction, at
        a distance that fits a sphere of ``extent`` in the view."""
        cam = self.plotter.camera
        offset = np.asarray(cam.position, dtype=float) - np.asarray(cam.focal_point, dtype=float)
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 1e-12 else np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        half_angle = np.radians(cam.view_angle) / 2.0
        distance = 1.15 * extent / np.sin(half_angle)
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.position = tuple(direction * distance)
        self.plotter.render()

    def focus_on(self, body_id: str, t: float) -> None:
        pos = self.body_display_position(body_id, t)
        self.plotter.camera.focal_point = tuple(pos)
        self.plotter.render()

    def measure_line(self, id_a: str, id_b: str, t: float, color: str = "#5ec8ff", render: bool = True) -> float:
        """Draw (or redraw) a straight line between two bodies at time ``t`` and
        return the distance between them. Reuses a single named actor so it
        follows the bodies as the timeline advances, the same update-in-place
        trick set_time uses for the trails."""
        a = self.body_display_position(id_a, t)
        b = self.body_display_position(id_b, t)
        self.plotter.add_mesh(
            pv.Line(a, b), color=color, line_width=2.4,
            name="measure-line", pickable=False,
        )
        if render:
            self.plotter.render()
        return float(np.linalg.norm(a - b))

    def clear_measure_line(self, render: bool = True) -> None:
        self.plotter.remove_actor("measure-line", render=render)

    # The longest maneuver arrow is clamped to this fraction of the current
    # camera-to-focal distance, so it holds a roughly constant on-screen size
    # at any zoom. The caller's world-space length wins when zoomed out to the
    # whole orbit; this fraction wins when the camera is dollied in close on the
    # spacecraft during a burn -- where a purely world-space length (a fraction
    # of the orbit radius, ~9000 km at GEO) was many screen-widths long and
    # buried the very spacecraft it was drawn from. Kept modest (well under
    # half the ~distance-tall view) so the arrow reads as a marker on the
    # spacecraft, not a blade sweeping across the frame.
    _MANEUVER_ARROW_SCREEN_FRACTION = 0.14

    def set_maneuver_vector(self, position_km, delta_v, max_length_km: float,
                            magnitude_fraction: float = 1.0, t: float | None = None) -> None:
        """Draw an arrow at ``position_km`` along the ``delta_v`` direction -- a
        maneuver's burn vector. Physical delta-v (m/s) is tiny next to orbit
        radii (km), so the length is a display scale, not the true magnitude;
        only the direction is physical. The largest burn is drawn at
        ``max_length_km`` in a wide view but clamped down to a fixed fraction of
        the current camera distance when zoomed in (so it never swamps the
        tracked spacecraft); ``magnitude_fraction`` (0..1, this burn's |delta-v|
        over the campaign's largest) then scales it so smaller burns read as
        shorter arrows. Reuses one named actor so successive burns replace it.

        ``position_km`` is in the world frame; ``t`` (the burn's time) moves it
        into the current display frame."""
        pos = np.asarray(position_km, dtype=float)
        if t is not None:
            pos = pos - self.frame_origin(t)
        dv = np.asarray(delta_v, dtype=float)
        mag = float(np.linalg.norm(dv))
        if mag < 1e-12 or max_length_km <= 0.0 or magnitude_fraction <= 0.0:
            self.clear_maneuver_vector()
            return
        cam = self.plotter.camera
        distance = float(np.linalg.norm(
            np.asarray(cam.position, dtype=float)
            - np.asarray(cam.focal_point, dtype=float)))
        full_length = min(float(max_length_km),
                          self._MANEUVER_ARROW_SCREEN_FRACTION * distance)
        length_km = full_length * float(magnitude_fraction)
        if length_km <= 0.0:
            self.clear_maneuver_vector()
            return
        arrow = pv.Arrow(
            start=pos, direction=dv / mag, scale=length_km,
            tip_length=0.28, tip_radius=0.09, shaft_radius=0.032,
        )
        self.plotter.add_mesh(arrow, color=MANEUVER_COLOR, name="maneuver-vector",
                              pickable=False, specular=0.3)
        self.plotter.render()

    def clear_maneuver_vector(self) -> None:
        self.plotter.remove_actor("maneuver-vector", render=True)

    # Camera-follow smoothing rate (1/s) applied when track_body is called
    # with a ``dt`` -- i.e. every timeline tick during playback. Hard-snapping
    # the camera to a freshly computed position every tick made playback look
    # jittery, because _on_tick's dt is wall-clock-measured (see app.py) and
    # varies frame to frame with OS scheduler/render-cost noise, so each snap
    # was a slightly different size; exponential smoothing (independent of
    # dt's own jitter, since the decay is expressed per unit time) turns that
    # noisy step sequence into continuous motion. A manual jump -- selecting a
    # body, scrubbing the slider -- calls this with dt=None and still snaps
    # instantly, so only continuous playback is smoothed, not user input.
    _TRACK_SMOOTHING_RATE_HZ = 10.0

    def track_body(
        self,
        body_id: str,
        t: float,
        dt: float | None = None,
        reference_body_id: str | None = None,
        render: bool = True,
    ) -> None:
        """Center ``body_id`` by *translating* the camera to it, preserving the
        current view offset (direction + distance) -- so the body holds its
        apparent size and framing rather than the camera just swiveling to face
        it. Reading the offset live each call means the user can still orbit and
        zoom while tracking, KSP-tracking-station style. Called once to focus a
        selection and every timeline tick when tracking is on.

        ``reference_body_id``, if given (and different from ``body_id``), locks
        the *direction* of that offset instead of leaving it as whatever the
        user last dragged it to: the camera is kept on the side of ``body_id``
        opposite ``reference_body_id``, so the reference body stays lined up
        behind the tracked one in the camera's forward view -- e.g. tracking a
        spacecraft while locked to Earth keeps Earth framed in the background
        as the spacecraft moves, instead of it drifting out of frame. Only the
        direction is overridden; the existing offset *distance* (i.e. the
        current zoom level) is preserved, so scroll-zooming still works.

        ``dt`` (seconds since the last call) enables smoothing -- see
        _TRACK_SMOOTHING_RATE_HZ -- and is omitted for one-shot jumps.
        """
        pos = self.body_display_position(body_id, t)
        cam = self.plotter.camera
        focal = np.asarray(cam.focal_point, dtype=float)
        position = np.asarray(cam.position, dtype=float)
        offset = position - focal
        distance = float(np.linalg.norm(offset))

        if reference_body_id is not None and reference_body_id != body_id:
            ref_pos = self.body_display_position(reference_body_id, t)
            away = pos - ref_pos
            away_norm = float(np.linalg.norm(away))
            if away_norm > 1e-9 and distance > 1e-9:
                offset = (away / away_norm) * distance

        target_focal = pos
        target_position = pos + offset

        if dt is not None and dt > 0.0:
            alpha = 1.0 - np.exp(-self._TRACK_SMOOTHING_RATE_HZ * dt)
            target_focal = focal + (target_focal - focal) * alpha
            target_position = position + (target_position - position) * alpha

        cam.focal_point = tuple(target_focal)
        cam.position = tuple(target_position)
        if render:
            self.plotter.render()

    def zoom_to_body(self, body_id: str, t: float, distance_factor: float = 0.2,
                     distance_km: float | None = None) -> None:
        """Like track_body, but *dolly in* rather than preserve distance --
        the same direction/up as whatever the camera's current framing is,
        moved to a closer camera-to-focal-point distance, so the KSP-style
        M-key focus (see app.py's _focus_selected) visibly zooms in on the
        target instead of just recentering on it.

        The target distance is either ``distance_factor`` of the *current*
        distance (the default -- relative, so the M-key zoom-in works
        regardless of a scene's distance scale or how far the camera already
        was) or, if ``distance_km`` is given, that absolute distance. The
        absolute mode is for framing that must be repeatable rather than
        compounding: jumping to a maneuver event (see app.py's
        _seek_to_event) lands on the same close framing every time, instead
        of zooming in another 5x on each successive click.
        """
        pos = self.body_display_position(body_id, t)
        cam = self.plotter.camera
        focal = np.asarray(cam.focal_point, dtype=float)
        position = np.asarray(cam.position, dtype=float)
        offset = position - focal
        distance = float(np.linalg.norm(offset))
        direction = offset / distance if distance > 1e-12 else np.array([0.0, 0.0, 1.0])
        target_distance = float(distance_km) if distance_km is not None else distance * distance_factor
        cam.focal_point = tuple(pos)
        cam.position = tuple(pos + direction * target_distance)
        self.plotter.render()
