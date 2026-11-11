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
