"""orbitopt's scene file format: a stable, independently-versioned data
contract for 3D scenes (bodies, orbits, trails, an optional scrubbable
timeline), consumed by orbitopt's viewer (orbitopt.viz.pv_viewer / Mission
Control) and producible by orbitopt's own compute exporters -- or by anyone
else's code, in any language, since the authoritative definition is the
packaged JSON Schema (orbitopt/schemas/scene-{version}.json), not this
module.

This is the entire "read/write/validate a scene file" surface, and it
depends on nothing but the standard library + jsonschema -- no numpy, no
pyvista, no pykep/pygmo/tudatpy. That split matters: it means a scene file
someone else produced (or one orbitopt computed N releases ago) keeps
working with THIS version's reader regardless of what orbitopt's compute
code does, and someone who only wants to load/view scenes needs only the
`pip install orbitopt[viewer]` extras, not a conda environment.

Compatibility policy (see the schema's own description for the same text):
adding a new OPTIONAL field to a scene document is not a breaking change --
readers ignore fields they don't know about, so producers can add features
without bumping ``schemaVersion``. Removing, renaming, or narrowing an
existing field's type is breaking and gets a new schema file (scene-2.0.json,
etc.); a document's own ``schemaVersion`` field says which one it was written
against, so a reader can tell.

    from orbitopt.scene_format import read_scene, write_scene, validate_scene
    scene = read_scene("some_scene.json")
    validate_scene(scene)  # raises jsonschema.ValidationError on mismatch
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import jsonschema

SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=None)
def _schema(version: str = SCHEMA_VERSION) -> dict:
    """Load and cache the packaged JSON Schema for ``version`` (a string like
    ``'1.0'``). Uses importlib.resources, not a hand-built filesystem path, so
    this works from an installed wheel as well as a source checkout."""
    filename = f"scene-{version}.json"
    traversable = resources.files("orbitopt").joinpath("schemas", filename)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            f"No packaged schema for scene format version {version!r} "
            f"(expected orbitopt/schemas/{filename})"
        ) from exc
    return json.loads(text)


def validate_scene(scene: dict) -> None:
    """Validate ``scene`` against its own ``schemaVersion`` field's JSON
    Schema. Raises ``jsonschema.ValidationError`` on a mismatch, or
    ``ValueError`` if ``schemaVersion`` is missing or names a version this
    orbitopt install doesn't have a schema for, or if a body's ``trail`` has
    empty/mismatched-length ``times``/``positions`` arrays.

    That last check is deliberately not expressible in the packaged JSON
    Schema (draft 2020-12 has no clean "these two array properties must be
    the same length" constraint), but every renderer that walks a trail
    (orbitopt.viz.scene_renderer, orbitopt.viz.pv_viewer) indexes
    times[0]/positions[0]/times[-1]/... unconditionally, so a schema-valid
    document with an empty or ragged trail would otherwise pass validation
    and then crash deep in a renderer as an uncaught IndexError instead of
    surfacing as a normal, catchable load error here.
    """
    version = scene.get("schemaVersion")
    if version is None:
        raise ValueError(
            "scene document has no 'schemaVersion' field -- not a valid "
            "orbitopt SceneData document. If you're building one by hand, "
            f"set schemaVersion={SCHEMA_VERSION!r} (see orbitopt.scene_format.SCHEMA_VERSION)."
        )
    jsonschema.validate(instance=scene, schema=_schema(version))

    for body in scene.get("bodies", []):
        trail = body.get("trail")
        if trail is None:
            continue
        times = trail.get("times", [])
        positions = trail.get("positions", [])
        body_id = body.get("id", "<unknown>")
        if len(times) == 0 or len(positions) == 0:
            raise ValueError(
                f"scene document is invalid: body {body_id!r} has an empty "
                f"trail (times has {len(times)} entries, positions has "
                f"{len(positions)}) -- a trail needs at least one "
                "times/positions pair for the renderer to place the body."
            )
        if len(times) != len(positions):
            raise ValueError(
                f"scene document is invalid: body {body_id!r} has a "
                f"mismatched trail -- times has {len(times)} entries but "
                f"positions has {len(positions)}; they must be the same "
                "length (one position per timestamp)."
            )


def is_valid_scene(scene: dict) -> bool:
    """Non-raising form of validate_scene() -- True/False instead of an
    exception, for callers that just want a filter/check."""
    try:
        validate_scene(scene)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


def read_scene(path, *, validate: bool = True) -> dict:
    """Read a SceneData JSON document from disk. Validated by default; pass
    ``validate=False`` to load a document as-is (e.g. to inspect why it fails
    validation, or to read a newer/unknown schema version leniently)."""
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    if validate:
        validate_scene(scene)
    return scene


def write_scene(scene: dict, path, *, validate: bool = True) -> None:
    """Validate (by default) and write a SceneData dict to disk as JSON."""
    if validate:
        validate_scene(scene)
    Path(path).write_text(json.dumps(scene, indent=2), encoding="utf-8")
