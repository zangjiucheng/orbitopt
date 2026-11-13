"""orbitopt Flight Lab: the ascent/recovery side of orbitopt (Epic #187).

Everything here is contract-first (see orbitopt.flightlab.contract): frozen
dataclasses over the flight-target-1.0 / flight-result-1.0 JSON Schemas,
needing nothing but stdlib + jsonschema + pyyaml (+ numpy-free math in
scene_export). No module in this package
(sub-package ``flightlab.ui`` aside, when it exists) may import Qt, pyvista
or orbitopt.viz -- physics/contract code and rendering must stay separable,
so a produced run dir keeps loading regardless of what the viewer does.
"""
from orbitopt.flightlab.contract import (
    BodyTrack,
    Event,
    FlightRun,
    FlightTarget,
    Verification,
    is_valid_run,
    is_valid_target,
    load_target,
    read_run,
    target_digest,
    validate_path,
    validate_run,
    validate_target,
    write_run,
)
from orbitopt.flightlab.scene_export import run_to_scene

__all__ = [
    "BodyTrack",
    "Event",
    "FlightRun",
    "FlightTarget",
    "Verification",
    "is_valid_run",
    "is_valid_target",
    "load_target",
    "read_run",
    "run_to_scene",
    "target_digest",
    "validate_path",
    "validate_run",
    "validate_target",
    "write_run",
]

# target_export needs numpy (it asks orbitopt.launch_windows for node passes),
# and the Gate-B pipeline (simulate/evaluate) pulls the adapter + frame bridge
# (numpy), so their names resolve lazily: `import orbitopt.flightlab` stays
# stdlib-only, and `from orbitopt.flightlab import build_target` / `simulate`
# / `evaluate` work either way.
_LAZY = ("build_target", "feasibility_report", "parse_orbit_spec", "parse_window_spec", "save_target")
_SIM_LAZY = ("simulate", "evaluate", "orbit_refly")
_SEG_LAZY = ("FlightLabSegment", "FlightLabSegmentResult", "flight_run_report")

__all__ += list(_LAZY) + list(_SIM_LAZY) + list(_SEG_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from orbitopt.flightlab import target_export

        return getattr(target_export, name)
    if name in _SIM_LAZY:
        import importlib

        # importlib, not `from . import simulate`: the function name would
        # re-enter this __getattr__ (module and function share the name).
        return getattr(importlib.import_module("orbitopt.flightlab.simulate"), name)
    if name in _SEG_LAZY:
        from orbitopt.flightlab import segment

        return getattr(segment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
