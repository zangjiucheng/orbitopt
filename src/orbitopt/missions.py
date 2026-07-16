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

def _load_solar_system() -> dict:
    from orbitopt.viz.solar_system import export_solar_system_data
    return export_solar_system_data()


def _load_artemis2() -> dict:
    from orbitopt.viz.mission_timeline import compute_and_export_mission
    return compute_and_export_mission()


def _load_goes() -> dict:
    from orbitopt.viz.geo_raising import compute_and_export_geo_mission
    return compute_and_export_geo_mission()


register("solar-system", "Solar System", _load_solar_system)
register("artemis2", "Artemis II — Free Return", _load_artemis2)
register("goes", "GOES — GTO to GEO", _load_goes)
