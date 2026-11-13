"""orbitopt mission integration contract (M-01, Epic #215): the state a
pipeline segment consumes and produces, the interface a segment implements,
and the checks every handoff between two segments must pass.

The end-to-end mission (``Launch Site -> Liftoff -> MECO -> Separation ->
SECO -> Parking Orbit -> Orbit Propagation``) is a chain of segments --
Ascent (M-02), Flight Lab / 6-DOF (F-07), TudatPy propagation -- written by
different teams against different physics stacks. This module is the seam
they all share:

* :class:`HandoffState` -- the only state allowed to cross a segment
  boundary: epoch (with timescale), frame, r, v, mass, stage config and
  metadata. Serialized as the ``handoff-state-1.0.json`` JSON Schema
  document (:func:`state_to_dict` / :func:`state_from_dict`).
* :class:`MissionSegment` -- the interface every segment implements:
  ``run(initial_state) -> SegmentResult``.
* :func:`check_handoff` / :func:`check_handoff_chain` / :func:`run_checked`
  -- the continuity/conservation gate between consecutive states, with
  configurable :class:`HandoffTolerances`.

Frame and time conventions
--------------------------
A state's ``frame`` uses the Flight Lab v1 vocabulary (``GCRS``, ``ECEF``,
``LAUNCH_ECI``; same tuple as ``orbitopt.flightlab.contract.FRAMES``, so
there is exactly one frame vocabulary in the package). ``GCRS`` is the
recommended interchange frame; converting LAUNCH_ECI/ECEF <-> GCRS is the
frame bridge's job (:mod:`orbitopt.flightlab.frames`), never a segment's
private business, and must happen *inside* the producing segment so the
handoff check always compares like with like.

A state's epoch is MJD2000 day-labels plus an explicit ``timescale`` in
``("UTC", "TT", "TDB")`` -- the three scales :mod:`orbitopt.timescales`
converts between, with the same "day label counts exactly 86400 SI seconds"
convention that module documents for UTC. Comparison of two epochs always
goes through :meth:`Epoch.to_tdb_seconds` (leap-second and TDB-TT aware),
so a UTC epoch and the equivalent TT/TDB epoch compare to within
microseconds, never within a leap second.

Handoff semantics
-----------------
``check_handoff(before, after, tolerances)`` runs four checks and reports
each separately (mirroring ``orbitopt.flightlab.contract.Check``):

===========================  ====================================================
check                        rule (all tolerances configurable)
===========================  ====================================================
``handoff.position_continuity``  ``|r_after - r_before| <= position_m``, and both
                                 states declare the *same* frame (mismatch fails
                                 as ``frame_mismatch``; convert first, see above)
``handoff.velocity_continuity``  ``|v_after - v_before| <= velocity_m_s`` (same frame rule)
``handoff.epoch_monotonic``      ``t_after - t_before >= -epoch_s`` (no backwards time beyond
                                 clock skew) and, if ``max_epoch_gap_s`` is set, the forward
                                 gap must not exceed it
``handoff.mass_conservation``    mass may never *increase* beyond ``mass_kg``
                                 (``mass_gain``); a decrease beyond ``mass_kg`` is legal
                                 only when ``stage_config`` changed (staging/separation
                                 shed it, ``staging_mass_drop``), otherwise it is
                                 unexplained (``mass_loss_without_staging``)
===========================  ====================================================

Compatibility policy: same as the Flight Lab contract -- the document's own
``schema_version`` picks the schema; adding optional fields is not breaking,
removing/renaming/narrowing is and gets ``handoff-state-2.0.json``.

    from orbitopt.mission.contract import (
        HandoffState, Epoch, HandoffTolerances, check_handoff, run_checked)

    result, report = run_checked(my_ascent_segment, pad_state)   # gate it
    assert report.passed, [c for c in report.checks if not c.passed]

Like ``orbitopt.flightlab.contract``, this module needs nothing but the
standard library + jsonschema (+ orbitopt.timescales, pure math): no numpy,
no Qt, no pykep/pygmo/tudatpy -- so a producer and a consumer can live on
either side of the seam without importing each other's compute stack.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import jsonschema

from orbitopt.flightlab.contract import FRAMES
from orbitopt.timescales import (
    SECONDS_PER_DAY,
    tdb_minus_tt,
    tdb_seconds_to_utc_mjd2000,
    utc_mjd2000_to_tdb_seconds,
)

SCHEMA_VERSION = "1.0"

#: The timescales a :class:`Epoch` may declare -- exactly the ones
#: :mod:`orbitopt.timescales` converts between, upper-cased to match the
#: ``time_scale`` enum of the Flight Lab v1 terminal-state block.
EPOCH_TIMESCALES = ("UTC", "TT", "TDB")

#: Handoff states use the one frame vocabulary of the package (Flight Lab
#: v1: GCRS / ECEF / LAUNCH_ECI). GCRS is the recommended interchange frame.
MISSION_FRAMES = FRAMES


@lru_cache(maxsize=None)
def _schema(name: str) -> dict:
    """Load and cache a packaged JSON Schema by filename -- same
    importlib.resources approach as orbitopt.flightlab.contract._schema."""
    traversable = resources.files("orbitopt").joinpath("schemas", name)
    try:
        text = traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"No packaged schema file orbitopt/schemas/{name}") from exc
    return json.loads(text)


def _finite(value: Any, where: str) -> float:
    """Coerce to float and reject NaN/inf -- handoff vectors must be plain
    finite numbers; a NaN position propagating into a TudatPy propagation is
    a crash buried three layers down, so reject it at the seam."""
    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {value!r} is not a number") from exc
    if not math.isfinite(t):
        raise ValueError(f"{where}: {value!r} is not finite")
    return t


def _mapping_of_str(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{where}: keys must be strings, got {key!r}")
        out[key] = item
    return out


# --------------------------------------------------------------------------
# Epoch and HandoffState
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Epoch:
    """A mission instant: MJD2000 day-labels (days since 2000-01-01
    00:00:00, each day counting exactly 86400 SI seconds like a calendar
    date) plus the ``timescale`` they are counted on, one of
    ``EPOCH_TIMESCALES``. The timescale is never guessed -- a state whose
    epoch says ``TT`` is interpreted on TT and compared via
    :meth:`to_tdb_seconds`, never treated as UTC."""

    mjd2000: float
    timescale: str = "UTC"

    def __post_init__(self):
        if self.timescale not in EPOCH_TIMESCALES:
            raise ValueError(
                f"epoch timescale {self.timescale!r} not in {EPOCH_TIMESCALES} "
                "(see orbitopt.timescales for what the package converts)"
            )
        object.__setattr__(self, "mjd2000", _finite(self.mjd2000, "epoch.mjd2000"))

    def to_tdb_seconds(self) -> float:
        """TDB seconds since J2000.0 -- the single comparable scale for this
        epoch (the one tudatpy/SPICE take, via orbitopt.timescales)."""
        if self.timescale == "UTC":
            return utc_mjd2000_to_tdb_seconds(self.mjd2000)
        label_s = (self.mjd2000 - 0.5) * SECONDS_PER_DAY  # seconds since J2000 on own label
        if self.timescale == "TDB":
            return label_s
        return label_s + tdb_minus_tt(label_s)  # TT

    def to_utc_mjd2000(self) -> float:
        """The same instant as MJD2000 UTC day-labels."""
        if self.timescale == "UTC":
            return self.mjd2000
        return tdb_seconds_to_utc_mjd2000(self.to_tdb_seconds())

    def delta_seconds(self, later: "Epoch") -> float:
        """``later`` minus ``self`` in TDB seconds (negative if ``later`` is
        actually earlier)."""
        return later.to_tdb_seconds() - self.to_tdb_seconds()


@dataclass(frozen=True)
class HandoffState:
    """The state that crosses a segment boundary -- the only thing one
    mission segment is allowed to tell the next one. Positions in metres,
    velocities in m/s, mass in kg (wet mass of the remaining stack, incl.
    the stage config still attached)."""

    epoch: Epoch
    frame: str
    r_m: tuple[float, float, float]
    v_m_s: tuple[float, float, float]
    mass_kg: float
    stage_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.epoch, Epoch):
            raise ValueError(f"HandoffState.epoch must be an Epoch, got {self.epoch!r}")
        if self.frame not in MISSION_FRAMES:
            raise ValueError(
                f"frame {self.frame!r} not in {MISSION_FRAMES} -- a segment must hand off in "
                "a contract frame (convert pseudo-ECI backends via orbitopt.flightlab.frames)"
            )
        for name, vec in (("r_m", self.r_m), ("v_m_s", self.v_m_s)):
            if isinstance(vec, (str, bytes)) or not isinstance(vec, Iterable):
                raise ValueError(f"HandoffState.{name}: expected a 3-vector, got {vec!r}")
            parts = tuple(vec)
            if len(parts) != 3:
                raise ValueError(f"HandoffState.{name}: expected 3 components, got {len(parts)}")
            object.__setattr__(
                self, name, tuple(_finite(c, f"HandoffState.{name}[{i}]") for i, c in enumerate(parts))
            )
        mass = _finite(self.mass_kg, "HandoffState.mass_kg")
        if mass <= 0.0:
            raise ValueError(f"HandoffState.mass_kg must be > 0, got {mass}")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "stage_config", _mapping_of_str(self.stage_config, "HandoffState.stage_config"))
        object.__setattr__(self, "metadata", _mapping_of_str(self.metadata, "HandoffState.metadata"))

    def to_dict(self) -> dict:
        """The handoff-state-1.0 JSON document for this state."""
        return {
            "schema_version": SCHEMA_VERSION,
            "epoch": {"mjd2000": self.epoch.mjd2000, "timescale": self.epoch.timescale},
            "frame": self.frame,
            "position_m": list(self.r_m),
            "velocity_m_s": list(self.v_m_s),
            "mass_kg": self.mass_kg,
            "stage_config": dict(self.stage_config),
            "metadata": dict(self.metadata),
        }


def validate_state(state: HandoffState) -> None:
    """Re-validate a constructed HandoffState. Construction already checks
    everything here (:meth:`HandoffState.__post_init__`); this exists so
    callers holding a duck-typed state, or mutating a copy through
    ``dataclasses.replace``, have an explicit gate to call. Raises
    ``ValueError``; there is no non-raising form because there is nothing to
    recover -- see :func:`is_valid_state_document` for documents."""
    if not isinstance(state, HandoffState):
        raise ValueError(f"not a HandoffState: {state!r}")
    HandoffState(  # noqa: F841 -- constructing re-runs every check
        epoch=state.epoch,
        frame=state.frame,
        r_m=state.r_m,
        v_m_s=state.v_m_s,
        mass_kg=state.mass_kg,
        stage_config=state.stage_config,
        metadata=state.metadata,
    )


def state_to_dict(state: HandoffState) -> dict:
    """Serialize a validated HandoffState to its handoff-state-1.0 document."""
    validate_state(state)
    return state.to_dict()


def state_from_dict(doc: Mapping[str, Any]) -> HandoffState:
    """Parse a handoff-state-1.0 document into a frozen HandoffState.
    Raises ``jsonschema.ValidationError`` on a schema mismatch or
    ``ValueError`` on a missing/unknown schema_version or a non-finite /
    non-positive value -- the same schema/code split as
    orbitopt.flightlab.contract.validate_target."""
    version = doc.get("schema_version") if isinstance(doc, Mapping) else None
    if version is None:
        raise ValueError(
            "handoff state document has no 'schema_version' field -- set "
            f"schema_version={SCHEMA_VERSION!r} (see orbitopt/schemas/handoff-state-1.0.json)."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"no HandoffState schema for version {version!r} (this orbitopt "
            f"implements {SCHEMA_VERSION!r}; see orbitopt/schemas/)"
        )
    jsonschema.validate(instance=dict(doc), schema=_schema("handoff-state-1.0.json"))
    for key in ("position_m", "velocity_m_s"):
        for i, component in enumerate(doc[key]):
            _finite(component, f"{key}[{i}]")
    mass = _finite(doc["mass_kg"], "mass_kg")
    if mass <= 0.0:
        raise ValueError(f"mass_kg must be > 0, got {mass}")
    epoch_doc = doc["epoch"]
    epoch = Epoch(mjd2000=_finite(epoch_doc["mjd2000"], "epoch.mjd2000"), timescale=epoch_doc["timescale"])
    if epoch.timescale == "UTC":
        utc_mjd2000_to_tdb_seconds(epoch.mjd2000)  # reject UTC labels before 1972 early
    else:
        epoch.to_tdb_seconds()  # the UTC image must remain representable (>= 1972)
    return HandoffState(
        epoch=epoch,
        frame=doc["frame"],
        r_m=tuple(doc["position_m"]),
        v_m_s=tuple(doc["velocity_m_s"]),
        mass_kg=mass,
        stage_config=doc["stage_config"],
        metadata=doc.get("metadata", {}),
    )


def is_valid_state_document(doc: Mapping[str, Any]) -> bool:
    """Non-raising form of the state_from_dict() validation -- True/False
    instead of an exception, for callers that just want a filter/check."""
    try:
        state_from_dict(doc)
    except (jsonschema.ValidationError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------
# Segment interface and result
# --------------------------------------------------------------------------


@runtime_checkable
class MissionSegment(Protocol):
    """The interface every mission-pipeline segment implements: take the
    state the previous segment handed off, simulate/transform it, return a
    :class:`SegmentResult` whose ``final_state`` is the next handoff.

    Contract a conforming segment must honour (checked by
    :func:`run_checked`):

    * ``run`` returns a :class:`SegmentResult` naming this segment.
    * ``final_state`` is a contract-frame state that passes
      :func:`check_handoff` against ``initial_state`` -- position/velocity
      continuous within the tolerances, epoch never going backwards, mass
      never appearing from nowhere, any mass drop beyond tolerance matched
      by a ``stage_config`` change.
    * frame conversion into the contract frame happens inside the segment
      (see orbitopt.flightlab.frames); the handoff check compares same
      frame with same frame.
    """

    @property
    def name(self) -> str:
        """Unique, stable, human-readable segment name (used as the
        SegmentResult's ``segment`` and in check evidence)."""
        ...

    def run(self, initial_state: HandoffState) -> SegmentResult:
        """Propagate/transform ``initial_state`` and return the segment's
        result. Must not mutate ``initial_state``."""
        ...


@dataclass(frozen=True)
class SegmentResult:
    """What one segment run produced: the end-of-segment handoff state, the
    mission events observed inside the segment (free-form mappings, each
    carrying a non-empty string ``kind`` -- e.g. ``meco``, ``separation``,
    ``seco``; see orbitopt.flightlab.contract.EVENT_KINDS for the Flight Lab
    vocabulary), and free-form run diagnostics (solver stats, step counts,
    provenance)."""

    segment: str
    final_state: HandoffState
    events: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError(f"SegmentResult.segment must be a non-empty string, got {self.segment!r}")
        if not isinstance(self.final_state, HandoffState):
            raise ValueError(f"SegmentResult.final_state must be a HandoffState, got {self.final_state!r}")
        if isinstance(self.events, (str, bytes)) or not isinstance(self.events, Iterable):
            raise ValueError(f"SegmentResult.events must be an iterable of mappings, got {self.events!r}")
        events = tuple(self.events)
        for i, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise ValueError(f"SegmentResult.events[{i}]: expected a mapping, got {event!r}")
            if not isinstance(event.get("kind"), str) or not event["kind"]:
                raise ValueError(f"SegmentResult.events[{i}]: needs a non-empty string 'kind'")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "diagnostics", _mapping_of_str(self.diagnostics, "SegmentResult.diagnostics"))


# --------------------------------------------------------------------------
# Handoff validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoffTolerances:
    """The configurable gates of :func:`check_handoff`. Distances in m,
    speeds in m/s, times in TDB seconds, mass in kg; ``max_epoch_gap_s`` of
    ``None`` means the forward time gap between segments is unbounded (a
    coast phase may start days after the previous segment ended)."""

    position_m: float = 1.0
    velocity_m_s: float = 0.01
    epoch_s: float = 1.0
    max_epoch_gap_s: float | None = None
    mass_kg: float = 1e-3

    def __post_init__(self):
        for name in ("position_m", "velocity_m_s", "epoch_s", "mass_kg"):
            value = _finite(getattr(self, name), f"HandoffTolerances.{name}")
            if value < 0.0:
                raise ValueError(f"HandoffTolerances.{name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)
        if self.max_epoch_gap_s is not None:
            gap = _finite(self.max_epoch_gap_s, "HandoffTolerances.max_epoch_gap_s")
            if gap < 0.0:
                raise ValueError(f"HandoffTolerances.max_epoch_gap_s must be >= 0, got {gap}")
            object.__setattr__(self, "max_epoch_gap_s", gap)


@dataclass(frozen=True)
class HandoffCheck:
    """One handoff gate: what was tested, the verdict, the measured number
    behind it and the limit it was judged against (value/limit may be None
    when the check failed before measuring, e.g. a frame mismatch), plus a
    machine-readable ``reason_code`` for reports and CI gates."""

    id: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    reason_code: str = ""


@dataclass(frozen=True)
class HandoffReport:
    """The verdict of one handoff: the four gate checks between ``before``
    and ``after``. ``passed`` is the AND of the checks -- a report with zero
    checks cannot occur (check_handoff always emits four)."""

    before: HandoffState
    after: HandoffState
    checks: tuple[HandoffCheck, ...]

    @property
    def passed(self) -> bool:
        """True only if every gate check passed -- never a defaulted True."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[HandoffCheck, ...]:
        """The checks that failed (empty when ``passed``)."""
        return tuple(check for check in self.checks if not check.passed)


def _vector_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def check_position_continuity(
    before: HandoffState, after: HandoffState, tolerances: HandoffTolerances
) -> HandoffCheck:
    """End-to-end position gate: ``|r_after - r_before| <= tolerances.position_m``.
    A frame mismatch between the two states fails the check as
    ``frame_mismatch`` without measuring -- r in LAUNCH_ECI and r in GCRS
    are different vectors for the same physical state, so converting first
    (orbitopt.flightlab.frames) is mandatory, not a hint."""
    limit = tolerances.position_m
    if before.frame != after.frame:
        return HandoffCheck(
            id="handoff.position_continuity", passed=False, limit=limit, unit="m",
            reason_code=f"frame_mismatch:{before.frame}!={after.frame}",
        )
    delta = _vector_delta(before.r_m, after.r_m)
    return HandoffCheck(
        id="handoff.position_continuity", passed=delta <= limit, value=delta, limit=limit,
        unit="m", reason_code="" if delta <= limit else "position_discontinuity",
    )


def check_velocity_continuity(
    before: HandoffState, after: HandoffState, tolerances: HandoffTolerances
) -> HandoffCheck:
    """End-to-end velocity gate: ``|v_after - v_before| <= tolerances.velocity_m_s``
    with the same same-frame rule as :func:`check_position_continuity`
    (frame conversion carries the omega x r term for a rotating frame, so
    comparing across frames is meaningless)."""
    limit = tolerances.velocity_m_s
    if before.frame != after.frame:
        return HandoffCheck(
            id="handoff.velocity_continuity", passed=False, limit=limit, unit="m/s",
            reason_code=f"frame_mismatch:{before.frame}!={after.frame}",
        )
    delta = _vector_delta(before.v_m_s, after.v_m_s)
    return HandoffCheck(
        id="handoff.velocity_continuity", passed=delta <= limit, value=delta, limit=limit,
        unit="m/s", reason_code="" if delta <= limit else "velocity_discontinuity",
    )


def check_epoch_continuity(
    before: HandoffState, after: HandoffState, tolerances: HandoffTolerances
) -> HandoffCheck:
    """Time gate: the epoch gap (TDB seconds; each state converted from its
    own declared timescale) must not go backwards by more than
    ``epoch_s`` (clock skew between independent physics stacks), and must
    not exceed ``max_epoch_gap_s`` when that limit is set."""
    delta = before.epoch.delta_seconds(after.epoch)
    if delta < -tolerances.epoch_s:
        return HandoffCheck(
            id="handoff.epoch_monotonic", passed=False, value=delta, limit=-tolerances.epoch_s,
            unit="s", reason_code="epoch_backwards",
        )
    if tolerances.max_epoch_gap_s is not None and delta > tolerances.max_epoch_gap_s:
        return HandoffCheck(
            id="handoff.epoch_monotonic", passed=False, value=delta, limit=tolerances.max_epoch_gap_s,
            unit="s", reason_code="epoch_gap",
        )
    return HandoffCheck(
        id="handoff.epoch_monotonic", passed=True, value=delta,
        limit=tolerances.max_epoch_gap_s, unit="s", reason_code="",
    )


def check_mass_conservation(
    before: HandoffState, after: HandoffState, tolerances: HandoffTolerances
) -> HandoffCheck:
    """Mass gate: mass may never *increase* beyond ``tolerances.mass_kg``
    (``mass_gain`` -- nothing is refuelling this stack). A decrease beyond
    tolerance is legal only when ``stage_config`` changed between the two
    states (``staging_mass_drop``: staging/separation is exactly when a
    dry+propellant mass is expected to vanish); an unexplained decrease is
    ``mass_loss_without_staging``."""
    limit = tolerances.mass_kg
    delta = after.mass_kg - before.mass_kg
    if delta > limit:
        return HandoffCheck(
            id="handoff.mass_conservation", passed=False, value=delta, limit=limit,
            unit="kg", reason_code="mass_gain",
        )
    if abs(delta) <= limit:
        return HandoffCheck(
            id="handoff.mass_conservation", passed=True, value=delta, limit=limit,
            unit="kg", reason_code="",
        )
    # A drop beyond tolerance: legal iff the stage configuration changed.
    reconfigured = dict(before.stage_config) != dict(after.stage_config)
    return HandoffCheck(
        id="handoff.mass_conservation", passed=reconfigured, value=delta, limit=limit,
        unit="kg", reason_code="staging_mass_drop" if reconfigured else "mass_loss_without_staging",
    )


def check_handoff(
    before: HandoffState,
    after: HandoffState,
    tolerances: HandoffTolerances | None = None,
) -> HandoffReport:
    """Run the full handoff gate (position, velocity, epoch, mass) between
    two consecutive states and return the :class:`HandoffReport` with all
    four checks. Never raises for a bad handoff -- it reports it; invalid
    *states* raise at construction."""
    tol = tolerances if tolerances is not None else HandoffTolerances()
    return HandoffReport(
        before=before,
        after=after,
        checks=(
            check_position_continuity(before, after, tol),
            check_velocity_continuity(before, after, tol),
            check_epoch_continuity(before, after, tol),
            check_mass_conservation(before, after, tol),
        ),
    )


def check_handoff_chain(
    states: Sequence[HandoffState], tolerances: HandoffTolerances | None = None
) -> tuple[HandoffReport, ...]:
    """Gate a whole chain of consecutive handoff states (one per segment
    boundary, in mission order): ``len(states) - 1`` pairwise reports, same
    tolerances for every boundary. Raises ``ValueError`` for fewer than two
    states or a non-list/tuple chain."""
    if isinstance(states, (str, bytes)) or not isinstance(states, (list, tuple)):
        raise ValueError(f"check_handoff_chain needs a list/tuple of HandoffState, got {states!r}")
    if len(states) < 2:
        raise ValueError(f"check_handoff_chain needs at least 2 states, got {len(states)}")
    return tuple(check_handoff(before, after, tolerances) for before, after in zip(states, states[1:]))


def run_checked(
    segment: MissionSegment,
    initial_state: HandoffState,
    tolerances: HandoffTolerances | None = None,
) -> tuple[SegmentResult, HandoffReport]:
    """Run ``segment.run(initial_state)`` and immediately gate the handoff
    (this is how an integrator is meant to drive a chain of segments).
    Raises ``ValueError`` if the result does not name the segment that
    produced it; the handoff verdict itself comes back as the report, not
    an exception, so callers can decide gate-or-collect."""
    result = segment.run(initial_state)
    if not isinstance(result, SegmentResult):
        raise ValueError(f"segment {segment.name!r} returned {type(result).__name__}, not a SegmentResult")
    if result.segment != segment.name:
        raise ValueError(
            f"segment {segment.name!r} returned a SegmentResult naming {result.segment!r}"
        )
    return result, check_handoff(initial_state, result.final_state, tolerances)
