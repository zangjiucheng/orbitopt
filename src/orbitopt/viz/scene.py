"""Internal builder helpers over orbitopt's public scene file format
(orbitopt.scene_format / orbitopt/schemas/scene-1.0.json). Every
orbitopt.viz.* exporter (solar_system.py, mission_timeline.py,
geo_raising.py, and any future one) builds its output through
``body_entry``/``scene_document`` so every exporter produces the exact same
document shape -- the distinction between a static heliocentric snapshot and
a scrubbable mission timeline is just whether ``timeline``/``trail`` fields
are present, not a different document format.

This module is orbitopt's own *producer* convenience layer; it is not the
format's definition. The format itself -- what makes a document valid, and
what stays stable release to release -- lives in orbitopt.scene_format and
the JSON Schema it validates against, independent of whatever helpers this
module happens to offer. ``scene_document()`` stamps every document with the
current ``schemaVersion`` and validates it before returning, so a bug that
drifts one of orbitopt's own exporters out of sync with the published schema
fails immediately here, not silently in some downstream reader.

Colors are the same validated-categorical set used across every orbitopt
visualization (see the project README's dataviz notes): run
dataviz's scripts/validate_palette.js against any change here.
"""
from __future__ import annotations

from orbitopt.scene_format import SCHEMA_VERSION, validate_scene

COLOR = {
    "sun": "#fff4d6",
    "mercury": "#c4785a",
    "venus": "#ae8819",
    "earth": "#4c8eff",
    "mars": "#e14b3d",
    "jupiter": "#d9722e",
    "saturn": "#2fa97a",
    "uranus": "#2fa69e",
    "neptune": "#8c7fe0",
    "pluto": "#8c8496",
    "moon": "#928da3",
    "spacecraft": "#e8a23e",
}

RADIUS_DISPLAY = {
    "sun": 10.0,
    "mercury": 3.2, "venus": 4.4, "earth": 4.6, "mars": 3.6,
    "jupiter": 7.4, "saturn": 6.8, "uranus": 5.4, "neptune": 5.3, "pluto": 2.6,
    "moon": 3.4, "spacecraft": 3.0,
}

# Packaged texture filenames (see orbitopt/viz/assets/textures/, and
# scene_renderer._load_texture which resolves these) -- 2K equirectangular
# JPGs sourced from Solar System Scope (solarsystemscope.com/textures,
# themselves based on NASA imagery/elevation data), CC BY 4.0. No entry for
# pluto or spacecraft: pluto has no readily available free texture of this
# kind, and spacecraft is a synthetic marker with no real imagery to show --
# both fall back to the renderer's point-marker rendering, same as any body
# with no `texture` at all.
TEXTURE = {
    "sun": "sun.jpg",
    "mercury": "mercury.jpg",
    "venus": "venus_surface.jpg",
    "earth": "earth_daymap.jpg",
    "mars": "mars.jpg",
    "jupiter": "jupiter.jpg",
    "saturn": "saturn.jpg",
    "uranus": "uranus.jpg",
    "neptune": "neptune.jpg",
    "moon": "moon.jpg",
}

# Real sidereal rotation period (hours) -- one full 360-degree spin relative
# to the stars, not the (longer, for a prograde rotator) solar day. Negative
# = retrograde (spins backward relative to its orbit -- Venus and Uranus are
# the two real ones here; Uranus is additionally tipped ~98 degrees on its
# side, not modelled by this single scalar). The Sun's is its equatorial
# rate (it doesn't rotate as a rigid body -- differential rotation makes any
# single number a simplification, equatorial is the conventional one to
# quote). The Moon's equals its own orbital period around Earth (tidal
# locking -- same face always points at Earth), not an independent value.
ROTATION_PERIOD_HOURS = {
    "sun": 587.28,
    "mercury": 1407.6,
    "venus": -5832.6,
    "earth": 23.9345,
    "mars": 24.6229,
    "jupiter": 9.9250,
    "saturn": 10.656,
    "uranus": -17.24,
    "neptune": 16.11,
    "moon": 655.728,
}


def body_entry(
    body_id,
    name,
    color,
    kind,
    radius_display=4.0,
    orbit=None,
    orbit_dashed=False,
    position=None,
    trail=None,
    info=None,
    texture=None,
    radius=None,
    rotation_period_hours=None,
):
    """One entry in SceneData.bodies. ``orbit``/``position``/``trail`` are
    all optional and independent: a static planet has orbit+position, a
    time-animated spacecraft has trail (and usually no fixed position), a
    fixed reference body like the Sun or Earth-as-frame-origin has just
    position=[0,0,0].

    ``texture``/``radius`` are also optional and independent of the above --
    they ask the renderer for a real textured 3D sphere (see
    orbitopt.viz.scene_renderer) instead of the default point marker;
    passing only one of the two still renders as a point marker (see the
    schema's own description of these fields for why).

    ``rotation_period_hours`` only has a visible effect together with
    texture/radius (there's no marker-rendering equivalent of "spin") --
    see ROTATION_PERIOD_HOURS and scene_renderer.SceneRenderer.advance_rotation
    (timeline-less scenes) / SceneRenderer._apply_rotation_for_time (scenes
    with a timeline).
    """
    entry = {"id": body_id, "name": name, "color": color, "kind": kind, "radiusDisplay": radius_display}
    if orbit is not None:
        entry["orbit"] = orbit
    if orbit_dashed:
        entry["orbitDashed"] = True
    if position is not None:
        entry["position"] = position
    if trail is not None:
        entry["trail"] = trail
    if info is not None:
        entry["info"] = info
    if texture is not None:
        entry["texture"] = texture
    if radius is not None:
        entry["radius"] = radius
    if rotation_period_hours is not None:
        entry["rotationPeriodHours"] = rotation_period_hours
    return entry


def scene_document(scene_id, title, bodies, distance_unit="AU", subtitle=None, central_body_id=None, timeline=None):
    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "id": scene_id,
        "title": title,
        "distanceUnit": distance_unit,
        "bodies": bodies,
    }
    if subtitle is not None:
        doc["subtitle"] = subtitle
    if central_body_id is not None:
        doc["centralBodyId"] = central_body_id
    if timeline is not None:
        doc["timeline"] = timeline
    validate_scene(doc)  # every orbitopt exporter stays honest against the published schema
    return doc
