"""orbitopt Flight Lab: the ascent/recovery side of orbitopt (Epic #187).

Everything here is contract-first (see orbitopt.flightlab.contract): frozen
dataclasses over the flight-target-1.0 / flight-result-1.0 JSON Schemas,
needing nothing but stdlib + jsonschema + pyyaml. No module in this package
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
    "target_digest",
    "validate_path",
    "validate_run",
    "validate_target",
    "write_run",
]

# target_export needs numpy (it asks orbitopt.launch_windows for node passes),
# so its names resolve lazily: `import orbitopt.flightlab` stays stdlib-only,
# and `from orbitopt.flightlab import build_target` works either way.
_LAZY = ("build_target", "feasibility_report", "parse_orbit_spec", "parse_window_spec", "save_target")

__all__ += list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from orbitopt.flightlab import target_export

        return getattr(target_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
