"""orbitopt mission integration (Epic #215, End-to-End Mission Integration).

The chain ``Launch Site -> Liftoff -> MECO -> Separation -> SECO -> Parking
Orbit -> Orbit Propagation`` stitched together through one contract: every
segment (Ascent, Flight Lab / 6-DOF, TudatPy propagation, ...) consumes and
produces a :class:`~orbitopt.mission.contract.HandoffState` and implements
:class:`~orbitopt.mission.contract.MissionSegment`, and every boundary is
gated by the continuity/conservation checks of
:mod:`orbitopt.mission.contract`.

Like orbitopt.flightlab.contract, this package is contract-first and needs
nothing but the standard library + jsonschema (+ orbitopt.timescales): no
numpy, no Qt, no compute stack, so producers and consumers on either side
of a segment boundary never import each other. The one exception is the
opt-in M-04 unified runner :mod:`orbitopt.mission.launch_pipeline` (import
it directly, never through this package): it drives the real ascent
optimizer and therefore needs the compute stack.
"""
from orbitopt.mission.contract import (
    EPOCH_TIMESCALES,
    MISSION_FRAMES,
    SCHEMA_VERSION,
    Epoch,
    HandoffCheck,
    HandoffReport,
    HandoffState,
    HandoffTolerances,
    MissionSegment,
    SegmentResult,
    check_epoch_continuity,
    check_handoff,
    check_handoff_chain,
    check_mass_conservation,
    check_position_continuity,
    check_velocity_continuity,
    is_valid_state_document,
    run_checked,
    state_from_dict,
    state_to_dict,
    validate_state,
)

__all__ = [
    "EPOCH_TIMESCALES",
    "MISSION_FRAMES",
    "SCHEMA_VERSION",
    "Epoch",
    "HandoffCheck",
    "HandoffReport",
    "HandoffState",
    "HandoffTolerances",
    "MissionSegment",
    "SegmentResult",
    "check_epoch_continuity",
    "check_handoff",
    "check_handoff_chain",
    "check_mass_conservation",
    "check_position_continuity",
    "check_velocity_continuity",
    "is_valid_state_document",
    "run_checked",
    "state_from_dict",
    "state_to_dict",
    "validate_state",
]
