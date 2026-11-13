"""Flight Lab M-01 handoff contract: ``HandoffState`` (R-02, task #234).

One recovery segment hands the vehicle to the next by *state*, not by
narrative: a :class:`HandoffState` freezes the boundary instant -- mission
clock, frame-tagged position/velocity, wet mass and dry mass -- with the two
phases it sits between. Boostback hands over at the entry-burn cutoff, the
entry phase hands over at the corridor insertion (FL-302's
``landing_burn_initial_state`` is literally this object's payload), the
landing burn hands over at touchdown. A consumer that restarts its segment
from anything other than the handed-over state is discontinuous, and
``orbitopt.flightlab.recovery.handoff_check`` exists to catch exactly that.

Like :mod:`orbitopt.flightlab.contract` this module is stdlib-only (no
numpy, no propagator): a producer (a 3-DOF planner, a 6-DOF truth run --
F-07) and a consumer (a continuity checker, a report writer) can stay on
either side of this file without importing each other. States serialize as
JSON documents (``to_dict`` / ``from_dict``) with the same
``schema_version`` compatibility policy as the FlightTarget/FlightResult
documents: unknown OPTIONAL fields are ignored on read; removing, renaming
or narrowing one is a breaking change and gets a new version.

The phase vocabulary is fixed: the recovery chain is
``separation -> boostback -> entry -> landing -> touchdown`` and every
boundary state names the phase it came from and the phase it feeds
(:data:`RECOVERY_HANDOFF_BOUNDARIES`). A state whose mass drops below its
dry mass is rejected at construction -- mass bookkeeping is part of the
contract, not the consumer's problem.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

HANDOFF_KIND = "orbitopt.flightlab.handoff-state"
HANDOFF_SCHEMA_VERSION = "1.0"

#: The boundary vocabulary. ``separation`` names the producer of the first
#: boundary (the ascent / stage separation), ``touchdown`` the consumer of
#: the last one (the ground).
PHASE_SEPARATION = "separation"
PHASE_BOOSTBACK = "boostback"
PHASE_ENTRY = "entry"
PHASE_LANDING = "landing"
PHASE_TOUCHDOWN = "touchdown"

#: The three recovery segments, in flight order (module M-01 scope: the
#: boostback / entry / landing chain).
RECOVERY_PHASES = (PHASE_BOOSTBACK, PHASE_ENTRY, PHASE_LANDING)

#: Every legal ``(from_phase, to_phase)`` pair of the recovery chain.
RECOVERY_HANDOFF_BOUNDARIES = (
    (PHASE_SEPARATION, PHASE_BOOSTBACK),
    (PHASE_BOOSTBACK, PHASE_ENTRY),
    (PHASE_ENTRY, PHASE_LANDING),
    (PHASE_LANDING, PHASE_TOUCHDOWN),
)

#: Frames a handoff state may declare -- the FlightResult frame vocabulary.
#: The producer converts to a contract frame before handing over (the frame
#: bridge, FL-002), never the consumer.
HANDOFF_FRAMES = ("GCRS", "ECEF", "LAUNCH_ECI")


def _iso_z(dt) -> str:
    """A contract-format ISO-8601 UTC stamp from a datetime (naive means UTC
    by the FlightLab convention; an aware one is converted to UTC)."""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    return dt.astimezone(tz=None).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class HandoffState:
    """The M-01 handoff payload: the state one recovery phase hands to the
    next at their shared boundary instant. ``t_s`` is the mission clock at
    the boundary; ``mass_kg`` is the wet mass the consumer inherits and
    ``dry_mass_kg`` the vehicle floor that makes the mass (and therefore any
    conservation check) evaluable from the state alone."""

    from_phase: str
    to_phase: str
    t_s: float
    epoch_utc: str
    frame: str
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    mass_kg: float
    dry_mass_kg: float
    ignitions_used: int = 0

    def __post_init__(self) -> None:
        if (self.from_phase, self.to_phase) not in RECOVERY_HANDOFF_BOUNDARIES:
            raise ValueError(
                f"not an M-01 recovery boundary: {self.from_phase!r} -> {self.to_phase!r} "
                f"(legal: {', '.join(f'{a} -> {b}' for a, b in RECOVERY_HANDOFF_BOUNDARIES)})")
        if self.frame not in HANDOFF_FRAMES:
            raise ValueError(f"handoff frame {self.frame!r} is not one of {HANDOFF_FRAMES}")
        if not _is_utc_stamp(self.epoch_utc):
            raise ValueError(f"handoff epoch_utc {self.epoch_utc!r} is not an ISO-8601 UTC stamp")
        pos = tuple(float(c) for c in self.position_m)
        vel = tuple(float(c) for c in self.velocity_m_s)
        if len(pos) != 3 or len(vel) != 3:
            raise ValueError("handoff position_m / velocity_m_s need exactly three components")
        if not all(math.isfinite(c) for c in (*pos, *vel, self.t_s)):
            raise ValueError("handoff t_s / position_m / velocity_m_s must be finite")
        if not (math.isfinite(self.mass_kg) and math.isfinite(self.dry_mass_kg)) \
                or self.dry_mass_kg <= 0.0 or self.mass_kg < self.dry_mass_kg:
            raise ValueError(
                f"handoff mass {self.mass_kg} kg is below the dry mass {self.dry_mass_kg} kg "
                "or not finite -- a negative propellant load is a bookkeeping bug, "
                "never a state")
        # Freeze the tuples (a frozen dataclass still hands out mutable ones).
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "velocity_m_s", vel)

    @property
    def propellant_kg(self) -> float:
        """Propellant the consumer inherits: wet mass above the dry mass."""
        return self.mass_kg - self.dry_mass_kg

    @property
    def boundary(self) -> tuple[str, str]:
        return (self.from_phase, self.to_phase)

    def to_dict(self) -> dict:
        """The JSON document form (key order of the dataclass fields plus
        the kind/version envelope)."""
        doc = dataclasses.asdict(self)
        doc["kind"] = HANDOFF_KIND
        doc["schema_version"] = HANDOFF_SCHEMA_VERSION
        return doc

    @staticmethod
    def from_dict(doc: Mapping[str, Any]) -> "HandoffState":
        """Parse a :meth:`to_dict` document. Unknown OPTIONAL keys are
        ignored (compatibility policy); a missing/foreign ``kind``, a wrong
        ``schema_version`` or an unparseable state raises ``ValueError``."""
        if doc.get("kind") != HANDOFF_KIND:
            raise ValueError(f"not a {HANDOFF_KIND!r} document (kind: {doc.get('kind')!r})")
        version = doc.get("schema_version")
        if version != HANDOFF_SCHEMA_VERSION:
            raise ValueError(
                f"no handoff schema for version {version!r} (this orbitopt "
                f"implements {HANDOFF_SCHEMA_VERSION!r})")
        known = {f.name for f in dataclasses.fields(HandoffState)}
        return HandoffState(**{k: v for k, v in doc.items() if k in known})

    @staticmethod
    def of(state: Any, *, from_phase: str, to_phase: str,
           dry_mass_kg: float, ignitions_used: int = 0) -> "HandoffState":
        """Build from any duck-typed booster state (the propagator's
        ``BoosterState`` or another :class:`HandoffState`): the M-01 shape
        only needs ``t_s``, ``epoch_utc``, ``frame``, ``position_m``,
        ``velocity_m_s`` and ``mass_kg``, so producers that carry more (or
        less) can hand over without an adapter class. ``epoch_utc`` may be a
        datetime (naive = UTC) or a contract ISO-8601 string."""
        return HandoffState(
            from_phase=from_phase, to_phase=to_phase,
            t_s=float(state.t_s), epoch_utc=_iso_z(state.epoch_utc),
            frame=str(state.frame),
            position_m=tuple(float(c) for c in state.position_m),
            velocity_m_s=tuple(float(c) for c in state.velocity_m_s),
            mass_kg=float(state.mass_kg), dry_mass_kg=float(dry_mass_kg),
            ignitions_used=int(ignitions_used))


def _is_utc_stamp(text: str) -> bool:
    """True for an ISO-8601 timestamp with an explicit UTC zone -- the same
    rule ``orbitopt.flightlab.contract._parse_utc`` enforces, without
    importing the private helper."""
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt.tzinfo is not None


def handoff_digest(state: HandoffState) -> str:
    """SHA-256 of the state's canonical JSON -- what a report records so a
    cited handoff can be pinned to the exact bytes it checked."""
    text = json.dumps(state.to_dict(), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
