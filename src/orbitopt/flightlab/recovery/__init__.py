"""Flight Lab E3 booster recovery (3-DOF): the propagator of
``orbitopt.flightlab.recovery.propagator`` -- separation state in, booster
track with events out (J2 + rotating Earth + US76 atmosphere + Cd(Mach)
drag + relightable throttle-limited thrust).

Deliberately *not* imported from ``orbitopt.flightlab.__init__``: this
subpackage needs numpy (and, for non-ECEF frames, astropy), while the base
package and the contract stay stdlib-only. Import it directly:

    from orbitopt.flightlab.recovery import propagate, BoosterModel
"""
from orbitopt.flightlab.recovery.propagator import (  # noqa: F401
    AltitudeThreshold,
    BoosterEngine,
    BoosterModel,
    BoosterPhysics,
    BoosterState,
    BoosterTrack,
    ImpactEllipse,
    ImpactSite,
    RecoveryError,
    RecoveryEvent,
    ThrustCmd,
    ballistic_return,
    density_us76_kg_m3,
    parse_event_specs,
    pressure_pa,
    propagate,
    separation_state_from_run,
    speed_of_sound_m_s,
    temperature_k,
)

__all__ = [
    "AltitudeThreshold",
    "BoosterEngine",
    "BoosterModel",
    "BoosterPhysics",
    "BoosterState",
    "BoosterTrack",
    "ImpactEllipse",
    "ImpactSite",
    "RecoveryError",
    "RecoveryEvent",
    "ThrustCmd",
    "ballistic_return",
    "density_us76_kg_m3",
    "parse_event_specs",
    "pressure_pa",
    "propagate",
    "separation_state_from_run",
    "speed_of_sound_m_s",
    "temperature_k",
]
