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

_MARKER_BASE_SIZE = 8.0
_MARKER_WEIGHT_SIZE = 1.1
_SPHERE_RESOLUTION = 48  # theta/phi mesh resolution -- smooth enough for a close-up, cheap at ~10 bodies

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
            return pv.Texture(str(path))
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
    # A star is its own light source -- lighting=False renders its raw
    # texture colors with no shading falloff, so it reads as uniformly
    # bright regardless of viewing angle instead of having an implausible
    # "dark side" like a lit planet.
    is_self_lit = body.get("kind") == "star"
    return plotter.add_mesh(
        sphere, texture=texture, name=name, pickable=False,
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


def position_at_time(body: dict, t: float) -> np.ndarray:
    if "trail" not in body:
        return np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
    times, positions = _validated_trail(body)
    if t <= times[0]:
        return np.asarray(positions[0], dtype=float)
    if t >= times[-1]:
        return np.asarray(positions[-1], dtype=float)
    idx = int(np.searchsorted(times, t))
    idx = max(1, min(idx, len(times) - 1))
    t0, t1 = times[idx - 1], times[idx]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0 = np.asarray(positions[idx - 1], dtype=float)
    p1 = np.asarray(positions[idx], dtype=float)
    return p0 + frac * (p1 - p0)


def trail_index_at_time(body: dict, t: float) -> int:
    times, _positions = _validated_trail(body)
    return int(np.searchsorted(times, t))


class SceneRenderer:
    """Owns the 3D content of one scene inside a plotter. ``load()`` tears
    down and rebuilds everything (switching missions is rare -- a handful
    of times per session -- so simplicity beats trying to diff old/new
    scenes); ``set_time()`` is the hot path, called every timeline tick or
    time-warp step, and only mutates existing mesh point arrays plus swaps
    the "traveled so far" trail actor, never rebuilds from scratch.
    """

    def __init__(self, plotter):
        self.plotter = plotter
        self.scene: dict | None = None
        self._moving: dict[str, dict] = {}
        self._static_positions: dict[str, np.ndarray] = {}
        self._rotating: list[dict] = []
        self._rotation_elapsed_hours = 0.0
        self._has_timeline = False

    def load(self, scene: dict) -> None:
        self.plotter.clear()
        self.scene = scene
        self._moving = {}
        self._static_positions = {}
        self._rotating = []
        self._rotation_elapsed_hours = 0.0
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
                self.plotter.add_mesh(orbit_mesh, color=color, opacity=opacity, line_width=1, pickable=False)

            if "trail" in body:
                trail_times, trail_positions = _validated_trail(body)
                t0 = scene.get("timeline", {}).get("min", trail_times[0])
                pos0 = position_at_time(body, t0)

                sphere_actor = _sphere_actor(self.plotter, body, name=f"marker-{body['id']}")
                point_poly = None
                if sphere_actor is not None:
                    sphere_actor.position = tuple(float(c) for c in pos0)
                    self._register_rotation(body, sphere_actor)
                else:
                    point_poly = pv.PolyData(pos0.reshape(1, 3))
                    self.plotter.add_mesh(
                        point_poly, color=color, point_size=marker_size(body),
                        render_points_as_spheres=True, name=f"marker-{body['id']}",
                    )

                full_positions = np.asarray(trail_positions, dtype=float)
                full_poly = pv.MultipleLines(full_positions)
                self.plotter.add_mesh(full_poly, color=color, line_width=1.1, opacity=0.3, pickable=False)

                past_poly = pv.MultipleLines(np.vstack([full_positions[0], pos0]))
                self.plotter.add_mesh(
                    past_poly, color=color, line_width=2.6, opacity=0.95,
                    name=f"past-{body['id']}", pickable=False,
                )

                self._moving[body["id"]] = {
                    "body": body,
                    "point_poly": point_poly,
                    "sphere_actor": sphere_actor,
                    "past_actor_name": f"past-{body['id']}",
                    "full_positions": full_positions,
                    "color": color,
                }
            else:
                pos = np.asarray(body.get("position", [0.0, 0.0, 0.0]), dtype=float)
                self._static_positions[body["id"]] = pos
                sphere_actor = _sphere_actor(self.plotter, body, name=f"marker-{body['id']}")
                if sphere_actor is not None:
                    sphere_actor.position = tuple(float(c) for c in pos)
                    self._register_rotation(body, sphere_actor)
                else:
                    marker = pv.PolyData(pos.reshape(1, 3))
                    self.plotter.add_mesh(
                        marker, color=color, point_size=marker_size(body),
                        render_points_as_spheres=True, name=f"marker-{body['id']}",
                    )
                self.plotter.add_point_labels(
                    pos.reshape(1, 3), [body["name"]], font_size=12, text_color=color,
                    shape=None, always_visible=True, show_points=False,
                )

        timeline = scene.get("timeline")
        self.set_time(timeline["min"] if timeline else 0.0)
        self.plotter.camera_position = "iso"
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
        distances = {}
        for body_id, entry in self._moving.items():
            body = entry["body"]
            pos = position_at_time(body, t)
            if entry["sphere_actor"] is not None:
                entry["sphere_actor"].position = tuple(float(c) for c in pos)
            else:
                entry["point_poly"].points = pos.reshape(1, 3)

            idx = trail_index_at_time(body, t)
            past_pts = np.vstack([entry["full_positions"][: max(idx, 1)], pos.reshape(1, 3)])
            new_past_poly = pv.MultipleLines(past_pts) if len(past_pts) > 1 else pv.MultipleLines(np.vstack([pos, pos]))
            self.plotter.add_mesh(
                new_past_poly, color=entry["color"], line_width=2.6, opacity=0.95,
                name=entry["past_actor_name"], pickable=False,
            )
            distances[body_id] = float(np.linalg.norm(pos))

        for body_id, pos in self._static_positions.items():
            distances[body_id] = float(np.linalg.norm(pos))

        if self._has_timeline:
            self._apply_rotation_for_time(t)

        if render:
            self.plotter.render()
        return distances

    def body_world_position(self, body_id: str, t: float) -> np.ndarray:
        if body_id in self._moving:
            return position_at_time(self._moving[body_id]["body"], t)
        return self._static_positions.get(body_id, np.zeros(3))

    def focus_on(self, body_id: str, t: float) -> None:
        pos = self.body_world_position(body_id, t)
        self.plotter.camera.focal_point = tuple(pos)
        self.plotter.render()

    def measure_line(self, id_a: str, id_b: str, t: float, color: str = "#5ec8ff", render: bool = True) -> float:
        """Draw (or redraw) a straight line between two bodies at time ``t`` and
        return the distance between them. Reuses a single named actor so it
        follows the bodies as the timeline advances, the same update-in-place
        trick set_time uses for the trails."""
        a = self.body_world_position(id_a, t)
        b = self.body_world_position(id_b, t)
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
    # buried the very spacecraft it was drawn from.
    _MANEUVER_ARROW_SCREEN_FRACTION = 0.32

    def set_maneuver_vector(self, position_km, delta_v, max_length_km: float,
                            magnitude_fraction: float = 1.0) -> None:
        """Draw an arrow at ``position_km`` along the ``delta_v`` direction -- a
        maneuver's burn vector. Physical delta-v (m/s) is tiny next to orbit
        radii (km), so the length is a display scale, not the true magnitude;
        only the direction is physical. The largest burn is drawn at
        ``max_length_km`` in a wide view but clamped down to a fixed fraction of
        the current camera distance when zoomed in (so it never swamps the
        tracked spacecraft); ``magnitude_fraction`` (0..1, this burn's |delta-v|
        over the campaign's largest) then scales it so smaller burns read as
        shorter arrows. Reuses one named actor so successive burns replace it."""
        pos = np.asarray(position_km, dtype=float)
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
        pos = self.body_world_position(body_id, t)
        cam = self.plotter.camera
        focal = np.asarray(cam.focal_point, dtype=float)
        position = np.asarray(cam.position, dtype=float)
        offset = position - focal
        distance = float(np.linalg.norm(offset))

        if reference_body_id is not None and reference_body_id != body_id:
            ref_pos = self.body_world_position(reference_body_id, t)
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
        pos = self.body_world_position(body_id, t)
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
