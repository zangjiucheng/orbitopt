"""orbitopt's mission-config file format: a YAML/JSON, per-mission-kind data
contract letting a user tune a mission's parameters (orbit elements,
spacecraft mass, optimizer seed, ...) without editing source code.

Unlike orbitopt.scene_format (independent of orbitopt's compute code, usable
with just `pip install orbitopt[viewer]`), a mission config's whole purpose is
to drive that compute code: `build_scene_from_config` imports and calls the
mission kind's own builder, which needs whatever compute stack (pykep/pygmo/
tudatpy) that kind depends on. Loading and *validating* a config
(`load_mission_config`, `is_valid_mission_config`) needs nothing but pyyaml +
jsonschema, matching orbitopt.scene_format's split -- `orbitopt validate`
works from a lightweight pip install; `orbitopt run` needs the full conda
compute environment, same as the "goes"/"artemis2" built-in missions already
do.

    from orbitopt.mission_config import load_mission_config, build_scene_from_config
    config = load_mission_config("my_mission.yaml")  # raises on schema mismatch
    scene = build_scene_from_config(config)           # runs the mission kind's builder
"""
from __future__ import annotations

import importlib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import jsonschema
import yaml

# mission.kind -> packaged schema filename (orbitopt/schemas/<filename>).
# Adding a new kind means adding one entry here, one to _KIND_BUILDERS below,
# and one schema file -- no changes to the loading/validation machinery.
_KIND_SCHEMAS = {
    "geo-raising": "mission-geo-raising-1.0.json",
    "mars-transfer": "mission-mars-transfer-1.0.json",
}

# mission.kind -> (module, attribute) of a `build_from_config(config: dict)
# -> dict` callable. Imported lazily (only once a config of that kind is
# actually built, not merely validated) via importlib, not a module-level
# import, so validating a config never requires that kind's compute stack
# (pykep/pygmo/tudatpy) to be installed -- only jsonschema + pyyaml.
_KIND_BUILDERS = {
    "geo-raising": ("orbitopt.viz.geo_raising", "build_from_config"),
    "mars-transfer": ("orbitopt.viz.mars_transfer", "build_from_config"),
}


@lru_cache(maxsize=None)
def _schema(filename: str) -> dict:
    """Load and cache a packaged mission-config JSON Schema by filename. Uses
    importlib.resources, not a hand-built filesystem path, so this works from
    an installed wheel as well as a source checkout -- same approach as
    orbitopt.scene_format._schema."""
    traversable = resources.files("orbitopt").joinpath("schemas", filename)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"No packaged schema file orbitopt/schemas/{filename}") from exc
    return json.loads(text)


def known_kinds() -> list[str]:
    """The mission `kind` values orbitopt currently has a config schema (and
    builder) for."""
    return sorted(_KIND_SCHEMAS)


def validate_mission_config(config: dict) -> None:
    """Validate an already-parsed config dict against its own
    ``mission.kind``'s JSON Schema. Raises ``jsonschema.ValidationError`` on a
    mismatch, or ``ValueError`` if ``mission.kind`` is missing or names a kind
    orbitopt has no schema for."""
    kind = (config.get("mission") or {}).get("kind")
    if kind is None:
        raise ValueError(
            "mission config has no 'mission.kind' field -- not a valid "
            "orbitopt mission config. Known kinds: " + ", ".join(known_kinds())
        )
    schema_filename = _KIND_SCHEMAS.get(kind)
    if schema_filename is None:
        raise ValueError(f"unknown mission kind {kind!r} -- known kinds: {', '.join(known_kinds())}")
    jsonschema.validate(instance=config, schema=_schema(schema_filename))


def is_valid_mission_config(config: dict) -> bool:
    """Non-raising form of validate_mission_config() -- True/False instead of
    an exception, for callers that just want a filter/check."""
    try:
        validate_mission_config(config)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


def load_mission_config(path, *, validate: bool = True) -> dict:
    """Read a mission config YAML (or JSON -- YAML is a superset) document
    from disk. Validated by default against its own ``mission.kind``'s
    schema; pass ``validate=False`` to load a document as-is (e.g. to inspect
    why it fails validation)."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(config).__name__}")
    if validate:
        validate_mission_config(config)
    return config


def build_scene_from_config(config: dict) -> dict:
    """Validate ``config`` and run its mission kind's builder, returning a
    SceneData dict (see orbitopt.scene_format) ready for ``write_scene`` or
    the viewer. The kind's module -- and whatever compute stack it needs --
    is only imported here, at call time, not at import time of this module."""
    validate_mission_config(config)
    kind = config["mission"]["kind"]
    module_name, attr = _KIND_BUILDERS[kind]
    module = importlib.import_module(module_name)
    builder = getattr(module, attr)
    return builder(config)
