"""Shared "scene" document schema for the general-purpose 3D viewer in
viewer/ (React + three.js via @react-three/fiber). Every orbitopt.viz.*
exporter (solar_system.py, mission_timeline.py, and any future one) builds
its output through these helpers so the viewer's TypeScript SceneData type
(viewer/src/types.ts) has exactly one shape to render, regardless of
whether the scene is a static heliocentric snapshot or a scrubbable
mission timeline -- the distinction is just whether ``timeline`` /
``trail`` fields are present, not a different document format.

Colors are the same validated-categorical set used across every orbitopt
visualization (see the project README's dataviz notes): run
dataviz's scripts/validate_palette.js against any change here.
"""
from __future__ import annotations

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
):
    """One entry in SceneData.bodies. ``orbit``/``position``/``trail`` are
    all optional and independent: a static planet has orbit+position, a
    time-animated spacecraft has trail (and usually no fixed position), a
    fixed reference body like the Sun or Earth-as-frame-origin has just
    position=[0,0,0].
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
    return entry


def scene_document(scene_id, title, bodies, distance_unit="AU", subtitle=None, central_body_id=None, timeline=None):
    doc = {
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
    return doc
