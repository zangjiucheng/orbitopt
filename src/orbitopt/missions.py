"""A registry of named missions -- (id, title, zero-arg loader) triples that
each produce a validated orbitopt.scene_format SceneData document -- shared
by Mission Control (orbitopt.viz.app) and the CLI (orbitopt.cli) so both
consume one list instead of maintaining their own copies (previously a
source of drift: a mission added to one and not the other).

Built-in missions are registered here, lazily -- each loader defers its
import of the actual compute module (which may need pykep/pygmo/tudatpy)
until called, so importing this module itself costs nothing beyond stdlib +
orbitopt.scene_format.

Third parties can add their own missions without touching orbitopt's source,
via a setuptools entry point in the ``orbitopt.missions`` group. Each entry
point resolves to a zero-arg callable returning a Mission (or an
``(id, title, load)`` tuple):

    # a third-party package's pyproject.toml
    [project.entry-points."orbitopt.missions"]
    my-mission = "my_package.missions:my_mission"

    # my_package/missions.py
    from orbitopt.missions import Mission

    def my_mission() -> Mission:
        return Mission("my-mission", "My Mission", _load)

    def _load() -> dict:
        ...  # build (or read_scene() a bundled file) and return a
        ...  # SceneData dict; orbitopt.scene_format.validate_scene it first
        return scene

Once that package is pip-installed alongside orbitopt, "my-mission" shows up
in list_missions() and works with ``orbitopt view my-mission`` with no
changes to orbitopt itself. A broken plugin (import error, wrong return
shape) is isolated -- skipped with a warning rather than breaking discovery
for everyone else.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable

SceneLoader = Callable[[], dict]

_ENTRY_POINT_GROUP = "orbitopt.missions"


@dataclass(frozen=True)
class Mission:
    """A named, lazily-loaded scene. ``load()`` computes/reads and returns a
    SceneData dict -- call it, don't just read the dataclass fields, to
    actually get the scene."""

    id: str
    title: str
    load: SceneLoader


_registry: dict[str, Mission] = {}
_plugins_loaded = False


def register(id: str, title: str, load: SceneLoader, *, overwrite: bool = False) -> Mission:
    """Add a mission to the registry. Raises ValueError on an id collision
    unless ``overwrite=True`` -- a second registration under the same id
    (e.g. a plugin colliding with a built-in, or with another plugin) is
    almost certainly a mistake worth surfacing, not silently shadowing."""
    if id in _registry and not overwrite:
        raise ValueError(f"a mission named {id!r} is already registered ({_registry[id].title!r})")
    mission = Mission(id=id, title=title, load=load)
    _registry[id] = mission
    return mission


def _load_plugins() -> None:
    """Discover third-party missions via the ``orbitopt.missions`` entry-point
    group, once per process. Each entry point is resolved and called in
    isolation: an exception from one broken/incompatible plugin is a warning,
    not a crash that takes down every other mission (built-in or not)."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            factory = ep.load()
            result = factory()
            mission = result if isinstance(result, Mission) else Mission(*result)
            register(mission.id, mission.title, mission.load)
        except Exception as exc:  # noqa: BLE001 -- isolate one bad plugin from the rest
            warnings.warn(
                f"orbitopt.missions: skipping plugin {ep.name!r} ({ep.value}): {exc!r}",
                stacklevel=2,
            )


def list_missions() -> list[Mission]:
    """All registered missions -- built-ins first, then discovered
    third-party plugins in entry-point iteration order."""
    _load_plugins()
    return list(_registry.values())


def get(id: str) -> Mission:
    """Look up a mission by id. Raises KeyError (listing what IS available)
    if not found among built-ins + discovered plugins."""
    _load_plugins()
    try:
        return _registry[id]
    except KeyError:
        available = ", ".join(sorted(_registry)) or "(none)"
        raise KeyError(f"no mission named {id!r}; available: {available}") from None


# --- built-in missions ---------------------------------------------------
# Each loader's import of the actual compute module is deferred to call time,
# so `import orbitopt.missions` itself never needs pykep/pygmo/tudatpy --
# only actually loading a compute-backed mission does.

# The compute-backed missions (everything but the near-instant Solar System
# data export) go through orbitopt.scene_cache.cached_scene: seconds of
# optimization + tudatpy propagation the first time, then a disk-cached JSON
# read on every later load (across app restarts) until the orbitopt source
# that produces the scene changes. See scene_cache for the fingerprint-based
# invalidation and why it's preferred over a committed static scene file.

def _load_solar_system() -> dict:
    from orbitopt.viz.solar_system import export_solar_system_data
    return export_solar_system_data()


# The config-driven built-ins run through the unified mission pipeline
# (orbitopt.pipeline) -- the same code path as `orbitopt run <config>` and the
# Python API -- from these minimal configs. Every omitted section falls back
# to the builder's own defaults, so the scenes are identical to the
# zero-argument compute_and_export_*_mission() calls. The GOES high-fidelity
# re-fly is off, matching what the GUI has always shown (it's a ~minute of
# extra tudatpy work); Mission Control says so in its Verification tab and
# offers a one-click re-run with it on (high_fidelity_loader), and
# `orbitopt run` a config gets it from the CLI. Artemis II keeps
# its verification report (the targeter's own TudatPy check -- no extra cost).
BUILTIN_MISSION_CONFIGS = {
    # Same scene id/title as compute_and_export_mission(), so the cached
    # scene and everything keyed on its id are unchanged.
    "artemis2": {
        "schemaVersion": "1.0",
        "mission": {"id": "artemis2-mission", "title": "Artemis II — Free Return", "kind": "lunar-free-return"},
        "output": {"scene_file": "artemis2.json"},
    },
    "goes": {
        "schemaVersion": "1.0",
        "mission": {"id": "goes-gto-geo", "title": "GOES — GTO to GEO Raising", "kind": "geo-raising"},
        "verification": {"enabled": False},
        "output": {"scene_file": "goes.json"},
    },
    "mars": {
        "schemaVersion": "1.0",
        "mission": {"id": "mars-transfer", "title": "Mars — Earth to Orbit Insertion", "kind": "mars-transfer"},
        "output": {"scene_file": "mars.json"},
    },
}


def builtin_mission_config(id: str) -> dict:
    """A fresh copy of the mission config a config-driven built-in mission
    ("artemis2", "goes", "mars") runs through orbitopt.pipeline.run_pipeline."""
    import copy

    return copy.deepcopy(BUILTIN_MISSION_CONFIGS[id])


def _run_builtin_pipeline(id: str) -> dict:
    from orbitopt.pipeline import run_pipeline

    return run_pipeline(builtin_mission_config(id)).scene


def _load_artemis2() -> dict:
    from orbitopt.scene_cache import cached_scene

    return cached_scene("artemis2", lambda: _run_builtin_pipeline("artemis2"))


def _load_goes() -> dict:
    from orbitopt.scene_cache import cached_scene

    return cached_scene("goes", lambda: _run_builtin_pipeline("goes"))


def _load_goes_verified() -> dict:
    """GOES with the TudatPy high-fidelity re-fly switched on -- the
    one-click "run verification" entry in Mission Control. Cached under its
    own id so the fast default scene stays cached alongside it."""
    from orbitopt.pipeline import run_pipeline
    from orbitopt.scene_cache import cached_scene

    def compute() -> dict:
        config = builtin_mission_config("goes")
        config["verification"] = {"enabled": True}
        return run_pipeline(config).scene

    return cached_scene("goes-verified", compute)


# Built-in missions whose default (GUI) load leaves high-fidelity verification
# off to keep loading cheap: the note shown in place of the reports, and the
# loader that re-runs the same mission with verification on. Module-level
# functions, so they pickle into Mission Control's compute worker process.
VERIFICATION_OFF_BY_DEFAULT = {
    "goes": (
        ("High-fidelity verification is OFF by default for GOES to keep loading fast. "
        "The animation shows the screening/optimization solution only -- a smooth "
        "animation does NOT mean it passed TudatPy high-fidelity verification "
        "(J2+J22 + Sun/Moon re-fly with the final-burn targeter)."),
        _load_goes_verified,
    ),
}


def verification_off_note(id: str) -> str | None:
    """Why mission ``id``'s default load carries no verification report, or
    None when its default load does verify (or it isn't a built-in)."""
    entry = VERIFICATION_OFF_BY_DEFAULT.get(id)
    return entry[0] if entry else None


def high_fidelity_loader(id: str) -> SceneLoader | None:
    """A loader for mission ``id`` with high-fidelity verification enabled,
    or None when its default load already verifies (or it isn't a built-in).
    ``orbitopt run`` a config with ``verification.enabled: true`` is the
    CLI equivalent."""
    entry = VERIFICATION_OFF_BY_DEFAULT.get(id)
    return entry[1] if entry else None


def _load_mars() -> dict:
    from orbitopt.scene_cache import cached_scene

    return cached_scene("mars", lambda: _run_builtin_pipeline("mars"))


register("solar-system", "Solar System", _load_solar_system)
register("artemis2", "Artemis II — Free Return", _load_artemis2)
register("goes", "GOES — GTO to GEO", _load_goes)
register("mars", "Mars — Earth to Orbit Insertion", _load_mars)
