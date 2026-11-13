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

import hashlib
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
    "lunar-free-return": "mission-lunar-free-return-1.0.json",
    "flight-lab": "mission-flight-lab-1.0.json",
}

# mission.kind -> (module, attribute) of a `build_from_config(config: dict,
# recorder=None) -> dict` callable (``recorder``: an
# orbitopt.pipeline.StageRecorder the builder reports its stage results to). Imported lazily (only once a config of that kind is
# actually built, not merely validated) via importlib, not a module-level
# import, so validating a config never requires that kind's compute stack
# (pykep/pygmo/tudatpy) to be installed -- only jsonschema + pyyaml.
_KIND_BUILDERS = {
    "geo-raising": ("orbitopt.viz.geo_raising", "build_from_config"),
    "mars-transfer": ("orbitopt.viz.mars_transfer", "build_from_config"),
    "lunar-free-return": ("orbitopt.viz.mission_timeline", "build_from_config"),
    "flight-lab": ("orbitopt.flightlab.mission", "build_from_config"),
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
    """Validate ``config`` and run it through the unified mission pipeline
    (orbitopt.pipeline.run_pipeline), returning just the SceneData dict (see
    orbitopt.scene_format) ready for ``write_scene`` or the viewer, with any
    verification reports still attached under
    ``orbitopt.verify.report.SCENE_REPORTS_KEY`` (``pop_from_scene`` them).
    Use ``run_pipeline`` directly for the standardized per-stage result.

    The kind's module -- and whatever compute stack it needs -- is only
    imported at call time, not at import time of this module."""
    from orbitopt.pipeline import run_pipeline

    result = run_pipeline(config)
    scene = result.scene
    if result.verification_reports:
        from orbitopt.verify.report import SCENE_REPORTS_KEY

        scene[SCENE_REPORTS_KEY] = list(result.verification_reports)
    return scene


def config_digest(config: dict) -> str:
    """Short SHA-256 of a config's canonical JSON form -- two configs with
    the same content (regardless of key order / YAML formatting) share it."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def cached_scene_from_config(config: dict) -> dict:
    """``build_scene_from_config`` for viewing: verification reports are
    stripped (the viewer has no use for them) and the scene is served from
    orbitopt.scene_cache -- the same source-fingerprint cache the built-in
    missions use -- keyed by the mission id plus a digest of the config, so
    editing either the YAML or orbitopt's source recomputes."""
    from orbitopt.scene_cache import cached_scene
    from orbitopt.verify.report import pop_from_scene

    validate_mission_config(config)

    def _compute() -> dict:
        scene = build_scene_from_config(config)
        pop_from_scene(scene)
        return scene

    return cached_scene(f"config-{config['mission']['id']}-{config_digest(config)}", _compute)
