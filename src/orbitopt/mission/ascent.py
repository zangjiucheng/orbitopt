"""M-02 (task #220): the ascent run's output as the parking orbit's incoming
:class:`~orbitopt.mission.contract.HandoffState` -- and the gate that says the
handoff lost nothing.

The end-to-end chain (#215) ends the ascent segment at SECO and starts the
parking-orbit / propagation segment at exactly that state. The ascent side is
OpenTsiolkovsky's 3-DOF point-mass flight (FL-004 adapter lineage), whose
output is a FlightResult v1 :class:`~orbitopt.flightlab.contract.FlightRun`:
per-body GCRS tracks on a UTC clock, insertion at the ``seco`` event. This
module is the transform and the gate between that run and the contract state
the next segment consumes:

* :func:`ascent_handoff_state` -- the SECO insertion state of the flown upper
  stage, as a :class:`~orbitopt.mission.contract.HandoffState`: frame
  **GCRS** (the ECI/J2000 interchange frame; a track still labelled
  ``LAUNCH_ECI``/``ECEF`` is bridged through
  :func:`orbitopt.flightlab.frames.to_gcrs` right here, never downstream),
  epoch as MJD2000 day-labels on an **explicit timescale** (``UTC`` by
  default, ``TT``/``TDB`` on request -- the same instant, converted by
  :mod:`orbitopt.timescales`, never a silent re-labelling), r, v, mass.
* :func:`parking_orbit_elements` -- the classical Keplerian elements
  recomputed from a handoff state (the parking orbit's roots), same
  convention as :func:`orbitopt.flightlab.checks.orbital_elements` /
  OpenTsiolkovsky's ``summary.final_state.orbit``.
* :func:`check_ascent_to_parking_orbit` -- the acceptance gate: the four
  contract checks of :func:`~orbitopt.mission.contract.check_handoff`
  (position, velocity, epoch across declared timescales, mass) between the
  ascent run's own terminal state and the handoff state, plus element
  agreement (a, e, i, RAAN, apses) between the recomputed parking orbit and
  the elements of the ascent terminal point.

Handoff semantics note: the ascent end and the parking-orbit start are the
*same physical instant* -- this handoff is a transform, not a propagation, so
the epoch gate measures a ~0 s gap (UTC vs TDB declared epochs still compare
to microseconds, via ``Epoch.to_tdb_seconds``) and the mass gate measures an
exact copy (SECO-to-coast sheds nothing until a stage_config change).

    from orbitopt.flightlab.contract import read_run
    from orbitopt.mission.ascent import (
        ascent_handoff_state, check_ascent_to_parking_orbit, parking_orbit_elements)

    run = read_run("runs/fl004-0001")
    state = ascent_handoff_state(run)                    # GCRS, UTC epoch
    report = check_ascent_to_parking_orbit(run, state)
    assert report.passed, report.failures
    parking_orbit_elements(state)["semi_major_axis_m"]   # the roots to propagate

Like the contract itself this module needs nothing but the standard library
plus the flightlab contract/checks (pure stdlib); the frame bridge (numpy +
astropy) is imported lazily and only when a track actually needs converting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from orbitopt.flightlab import checks as gate_checks
from orbitopt.flightlab.contract import BodyTrack, FlightRun
from orbitopt.mission.contract import (
    EPOCH_TIMESCALES,
    Epoch,
    HandoffCheck,
    HandoffReport,
    HandoffState,
    HandoffTolerances,
    check_handoff,
)
from orbitopt.timescales import (
    SECONDS_PER_DAY,
    utc_datetime_to_mjd2000,
    utc_mjd2000_to_tdb_seconds,
    utc_mjd2000_to_tt_seconds,
)

__all__ = [
    "AscentHandoffReport",
    "ElementTolerances",
    "ascent_handoff_state",
    "check_ascent_to_parking_orbit",
    "epoch_at_ascent_time",
    "parking_orbit_elements",
]

#: The interchange frame the handoff is produced in -- GCRS is the package's
#: ECI/J2000 ("true equinox, true equator of date"); the contract's other
#: frames are produced by the bridge on request, never silently.
HANDBACK_FRAME = "GCRS"


def _parse_epoch_utc(text: str, where: str) -> datetime:
    """ISO-8601 UTC timestamp (``Z`` or explicit offset required) -> aware
    datetime; same rule as the Flight Lab contract's parser, kept local
    because it is private there."""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: {text!r} is not an ISO-8601 timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{where}: timestamp {text!r} needs an explicit UTC zone (e.g. trailing 'Z')")
    return dt


def epoch_at_ascent_time(track_epoch_utc: str, t_s: float, timescale: str = "UTC") -> Epoch:
    """The ascent clock (track ``epoch_utc`` + per-sample ``t_s`` on the UTC
    label) as a contract :class:`~orbitopt.mission.contract.Epoch` on the
    requested ``timescale`` in ``EPOCH_TIMESCALES``. The instant never
    moves: a ``TDB`` (or ``TT``) epoch here and the ``UTC`` one for the same
    sample compare to sub-microsecond through
    :meth:`Epoch.to_tdb_seconds` -- the timescale declares how the label is
    counted, it is not a time shift."""
    if timescale not in EPOCH_TIMESCALES:
        raise ValueError(
            f"epoch timescale {timescale!r} not in {EPOCH_TIMESCALES} (see orbitopt.timescales)"
        )
    utc_label = utc_datetime_to_mjd2000(_parse_epoch_utc(track_epoch_utc, "track epoch_utc")) \
        + float(t_s) / SECONDS_PER_DAY
    if timescale == "UTC":
        return Epoch(utc_label, "UTC")
    if timescale == "TDB":
        tdb_s = utc_mjd2000_to_tdb_seconds(utc_label)
        return Epoch(tdb_s / SECONDS_PER_DAY + 0.5, "TDB")
    tt_s = utc_mjd2000_to_tt_seconds(utc_label)
    return Epoch(tt_s / SECONDS_PER_DAY + 0.5, "TT")


def _track_for(run: FlightRun, body_id: str) -> BodyTrack:
    track = next((t for t in run.tracks if t.body_id == body_id), None)
    if track is None:  # insertion_state can only name a track's body_id, but stay honest
        raise ValueError(f"run {run.run_id!r}: insertion body {body_id!r} has no track")
    return track


def _bridge_to_gcrs(track: BodyTrack, r_m, v_m_s, t_s: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """One (r, v) sample from the track's declared frame to GCRS. A GCRS
    track is the contract frame already (the FL-004 adapter bridges at
    write time); a hand-built or raw-backend track in ``LAUNCH_ECI``/``ECEF``
    goes through the frame bridge here, inside the producing segment, as the
    M-01 contract demands -- so the handoff check always compares GCRS with
    GCRS."""
    if track.frame == HANDBACK_FRAME:
        return tuple(float(x) for x in r_m), tuple(float(x) for x in v_m_s)
    from orbitopt.flightlab.frames import to_gcrs  # lazy: numpy + astropy

    epoch = _parse_epoch_utc(track.epoch_utc, f"track {track.body_id!r} epoch_utc")
    r_g, v_g = to_gcrs([tuple(float(x) for x in r_m)], [tuple(float(x) for x in v_m_s)],
                       track.frame, epoch, [float(t_s)])
    return tuple(float(x) for x in r_g[0]), tuple(float(x) for x in v_g[0])


def ascent_handoff_state(
    run: FlightRun,
    timescale: str = "UTC",
    stage_config: Mapping[str, Any] | None = None,
) -> HandoffState:
    """The run's outgoing handoff state: the upper stage's SECO insertion
    state, unified to GCRS (frame bridge applied if the track still carries
    a pseudo-inertial/Earth-fixed label) and stamped with an explicitly
    declared epoch timescale (``UTC``/``TT``/``TDB`` -- the same instant).
    ``stage_config`` defaults to what the flown stack says (the upper-stage
    body and its parent); metadata records the source run, the track's
    original frame and the raw insertion time, so the handoff is traceable
    back to the ascent run it came from."""
    ins = gate_checks.insertion_state(run)
    track = _track_for(run, ins["body_id"])
    r_gcrs, v_gcrs = _bridge_to_gcrs(track, ins["position_m"], ins["velocity_m_s"], ins["t_s"])
    return HandoffState(
        epoch=epoch_at_ascent_time(ins["epoch_utc"], ins["t_s"], timescale),
        frame=HANDBACK_FRAME,
        r_m=r_gcrs,
        v_m_s=v_gcrs,
        mass_kg=float(ins["mass_kg"]),
        stage_config=dict(stage_config) if stage_config is not None else {
            "body_id": track.body_id,
            "parent_body_id": track.parent_body_id if track.parent_body_id is not None else "",
        },
        metadata={
            "source": "flightlab-ascent",
            "run_id": run.run_id,
            "ascent_frame": track.frame,
            "ascent_t_s": float(ins["t_s"]),
            "event_t_s": float(ins["event_t_s"]),
            "epoch_utc": ins["epoch_utc"],
        },
    )


def parking_orbit_elements(state: HandoffState, mu: float = gate_checks.EARTH_MU) -> dict:
    """The parking orbit's classical Keplerian elements (SI, angles in deg)
    recomputed from a handoff state -- the acceptance path "the next segment
    can recompute the orbit from the handoff alone". Same convention as
    :func:`orbitopt.flightlab.checks.orbital_elements` (matching
    OpenTsiolkovsky's summary and tudat's ``cartesian_to_keplerian`` up to
    the sub-metre noise their cross-checks allow). Raises for an ``ECEF``
    state: elements of a ground-fixed vector are not orbital elements."""
    if state.frame == "ECEF":
        raise ValueError(
            "parking orbit elements need an inertial state (GCRS/LAUNCH_ECI); "
            "an ECEF handoff must be frame-bridged first (orbitopt.flightlab.frames)"
        )
    return gate_checks.orbital_elements(state.r_m, state.v_m_s, mu)


@dataclass(frozen=True)
class ElementTolerances:
    """Declared tolerances for the element agreement between the ascent
    terminal point and the parking orbit recomputed from the handoff state:
    distances in m, eccentricity dimensionless, angles in deg."""

    semi_major_axis_m: float = 10.0
    eccentricity: float = 1e-6
    inclination_deg: float = 1e-3
    raan_deg: float = 1e-2
    apsides_m: float = 10.0

    def __post_init__(self):
        for name in ("semi_major_axis_m", "eccentricity", "inclination_deg", "raan_deg", "apsides_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"ElementTolerances.{name} must be a finite number >= 0, got {value}")
            object.__setattr__(self, name, value)


def _angle_delta(a_deg: float, b_deg: float) -> float:
    d = (a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _element_checks(before: dict, after: dict, tol: ElementTolerances) -> tuple[HandoffCheck, ...]:
    """One HandoffCheck per agreed element (a, e, i, RAAN, periapsis,
    apoapsis). Argument of perigee / true anomaly are deliberately not
    gated: at near-circular insertion e -> 0 makes them numerically
    meaningless (the periapsis/apoapsis altitudes carry the size of the
    orbit instead)."""
    checks: list[HandoffCheck] = []

    def add(check_id: str, measured: float, limit: float, unit: str):
        ok = math.isfinite(measured) and measured <= limit
        checks.append(HandoffCheck(
            id=f"ascent.{check_id}", passed=ok, value=measured, limit=limit, unit=unit,
            reason_code="" if ok else f"element_discontinuity:{check_id}",
        ))

    add("semi_major_axis", abs(after["semi_major_axis_m"] - before["semi_major_axis_m"]),
        tol.semi_major_axis_m, "m")
    add("eccentricity", abs(after["eccentricity"] - before["eccentricity"]), tol.eccentricity, "")
    add("inclination", _angle_delta(after["inclination_deg"], before["inclination_deg"]),
        tol.inclination_deg, "deg")
    add("raan", _angle_delta(after["raan_deg"], before["raan_deg"]), tol.raan_deg, "deg")
    add("periapsis", abs(after["periapsis_altitude_m"] - before["periapsis_altitude_m"]),
        tol.apsides_m, "m")
    add("apoapsis", abs(after["apoapsis_altitude_m"] - before["apoapsis_altitude_m"]),
        tol.apsides_m, "m")
    return tuple(checks)


@dataclass(frozen=True)
class AscentHandoffReport:
    """The M-02 verdict for one ascent -> parking-orbit handoff: the four
    contract checks (``handoff``), the element agreement checks against the
    ascent terminal point (``element_checks``) and both element sets.
    ``passed`` is the AND of all of them."""

    before: HandoffState
    after: HandoffState
    handoff: HandoffReport
    element_checks: tuple[HandoffCheck, ...]
    ascent_elements: Mapping[str, float] = field(default_factory=dict)
    parking_elements: Mapping[str, float] = field(default_factory=dict)

    @property
    def checks(self) -> tuple[HandoffCheck, ...]:
        return self.handoff.checks + self.element_checks

    @property
    def passed(self) -> bool:
        return self.handoff.passed and all(check.passed for check in self.element_checks)

    @property
    def failures(self) -> tuple[HandoffCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def check_ascent_to_parking_orbit(
    run: FlightRun,
    state: HandoffState,
    tolerances: HandoffTolerances | None = None,
    element_tolerances: ElementTolerances | None = None,
) -> AscentHandoffReport:
    """Gate ``state`` as the handoff of ``run``'s ascent: the contract's four
    checks (:func:`~orbitopt.mission.contract.check_handoff`) between the
    run's terminal state (always derived fresh, in GCRS on the UTC label) and
    ``state`` -- so a ``TDB``/``TT`` handoff epoch passes by declaring the
    same instant, not by matching labels -- plus element agreement between
    the ascent terminal point and the parking orbit recomputed from the
    handoff. Never raises for a bad handoff; it reports it."""
    tol = tolerances if tolerances is not None else HandoffTolerances()
    elem_tol = element_tolerances if element_tolerances is not None else ElementTolerances()
    before = ascent_handoff_state(run, timescale="UTC")
    ascent_elements = gate_checks.orbital_elements(before.r_m, before.v_m_s)
    report = check_handoff(before, state, tol)
    if report.passed:  # frames agree and r/v are continuous -> elements are meaningful
        parking_elements = parking_orbit_elements(state)
        element_checks = _element_checks(ascent_elements, parking_elements, elem_tol)
    else:
        parking_elements = {}
        element_checks = (HandoffCheck(
            id="ascent.elements_agree", passed=False,
            reason_code=f"frame_mismatch:{before.frame}!={state.frame}"
            if before.frame != state.frame else "elements_not_assessed_handoff_failed",
        ),)
    return AscentHandoffReport(
        before=before, after=state, handoff=report, element_checks=element_checks,
        ascent_elements=ascent_elements, parking_elements=parking_elements,
    )
