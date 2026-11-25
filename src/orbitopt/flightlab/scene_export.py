"""FlightRun -> SceneData v1.0 document: the Flight Lab scene importer
(FL-005, Gate A). Turns a run directory's tracks/events/verification into an
Earth-scene document ``orbitopt view`` (or the E5 Flight Lab page) can open,
with every body appearing/disappearing at its ``t_start_s``/``t_end_s`` and
every contract event becoming a timeline marker.

The contract-to-scene frame rule: a run may only be rendered once every
sample sits in the scene's GCRS. ``GCRS`` tracks render as-is; ``LAUNCH_ECI``
and ``ECEF`` tracks are converted through the FL-002 frame bridge
(``frames.to_gcrs``), so the exported trail uses the same authoritative
astropy/ITRS conversion as the contract pipeline rather than a GMST-only
approximation. The ECEF conversion (F-07, task #232 -- what lets Mission
Control render the native-plant 6-DOF runs) converts every sample at its own
epoch through the bridge's full rotating-frame transform, omega-cross-r
velocity term included; ``frames.to_gcrs`` growing that support is what
superseded FL-005's "ECEF is not convertible here" refusal. Anything failing the run contract (bad frame enum,
unparseable or missing epoch, non-monotonic time, mass discontinuity, ...)
raises before a single actor is drawn -- a bad run must never render.

Earth-texture orientation uses the same authoritative frame-bridge matrix when
its optional numpy/astropy dependencies are available. A base install may still
export default-texture GCRS runs through the stdlib GMST fallback, and records
that approximation as a scene assumption.

Scene-side conventions mirror orbitopt.viz.geo_raising / starship_flight
(km-unit Earth scene) but this module re-implements the handful of Earth
constants it needs rather than importing ``orbitopt.viz``: no module in this
package may depend on Qt/pyvista (see this package's ``__init__``), so a run
stays exportable on a headless, viewer-less install.

The scene document stays scene-1.0 with OPTIONAL fields only (no schema
bump, foreign members are legal): a top-level ``flightRun`` metadata block
(run id + the three never-collapsed success verdicts), per-body
``tStart``/``tEnd`` lifespan bounds in timeline units, which the reference
viewer uses to make a body appear/disappear mid-mission, a continuous
``timeline.phases`` row covering the whole mission (M-06: Pad -> Boost ->
Staging -> Upper-stage burn -> Parking orbit -> Propagation, derived from
the run's own events and track lifespans), and a top-level ``handoffs``
block (M-06): one entry per stage boundary -- a staging handoff gated with
the M-01 ``check_handoff`` continuity/conservation checks, and the SECO ->
parking-orbit handoff carrying the run's insertion / orbit re-fly checks --
each with a ``passed`` / ``failed`` / ``not_checked`` status. The same
verdict is appended to the matching timeline event's note, so the pass/fail
of every handoff is visible wherever events are drawn.

``coast_orbits`` (default 0 = off) extends the scene past the last flown
sample with a two-body display coast of the SECO state
(:func:`orbitopt.flightlab.coast.coast_arc`): a ``parking_orbit`` body with
the propagated arc, timeline max pushed to the coast end, and the
Parking-orbit / Propagation phases spanning it. The coast is display data,
recorded as a scene assumption -- Gate-B's tudatpy re-fly stays the
verification authority.

    from orbitopt.flightlab import read_run, run_to_scene
    scene = run_to_scene(read_run("runs/demo-0001"))
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from orbitopt.scene_format import validate_scene
from orbitopt.timescales import utc_datetime_to_mjd2000, utc_mjd2000_to_tdb_seconds

from orbitopt.flightlab.contract import FlightRun, _parse_utc

# orbitopt.mission.contract imports orbitopt.flightlab.contract, and this
# module is imported by this package's __init__ -- importing the M-01
# contract at module scope would close that import cycle, so the handoff
# helpers below reach for it lazily (a per-scene-export cost of one
# cached-module lookup).
if TYPE_CHECKING:
    from orbitopt.mission.contract import HandoffState
    from orbitopt.mission.two_stage_leo import TwoStageLeoScenario

#: Earth rendering constants (the scene's distanceUnit is always km here).
EARTH_RADIUS_KM = 6378.137  # WGS-84 equatorial radius
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SIDEREAL_DAY_HOURS = 23.9344696
EARTH_TEXTURE = "earth_daymap.jpg"  # the packaged equirectangular day map
EARTH_COLOR = "#4c8eff"
EARTH_RADIUS_DISPLAY = 4.6
CRAFT_RADIUS_DISPLAY = 3.0

#: Trail colours cycled per track, in contract (manifest) body order.
TRAIL_PALETTE = ("#ffffff", "#35c48b", "#d64550", "#b58cf0", "#4cc9f0", "#f25f5c")

#: Frames a track may be rendered in, after conversion if needed. Every
#: non-GCRS frame goes through the FL-002 frame bridge -- see this module's
#: docstring.
CONVERTIBLE_FRAMES = ("GCRS", "LAUNCH_ECI", "ECEF")

_EVENT_LABELS = {
    "ignition": "Ignition",
    "liftoff": "Liftoff",
    "max_q": "Max-Q",
    "meco": "MECO",
    "separation": "Separation",
    "seco": "SECO",
    "boostback_start": "Boostback start",
    "boostback_end": "Boostback end",
    "entry_start": "Entry start",
    "entry_end": "Entry end",
    "landing_start": "Landing burn",
    "touchdown": "Touchdown",
    "impact": "Impact",
}


def _gmst_deg(mjd2000_utc: float) -> float:
    """Greenwich mean sidereal time [deg] at UT1 ~= UTC (IAU 1982 linear
    term) -- the same model as orbitopt.launch_windows.gmst_deg, restated
    with ``math`` so this package stays dependency-light."""
    d = mjd2000_utc - 0.5  # days since J2000.0 (12:00 TT ~= 12:00 UTC)
    return (280.46061837 + 360.98564736629 * d) % 360.0


def _gmst_only_basis(mjd2000_utc: float) -> list:
    th = math.radians(_gmst_deg(mjd2000_utc))
    c, s = math.cos(th), math.sin(th)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _earth_orientation_basis(epoch_utc: datetime, mjd2000_utc: float) -> tuple[list, str]:
    """Earth-texture basis, with provenance. The full FL-002 frame-bridge
    matrix is preferred; a base install without its optional numpy/astropy
    dependencies falls back to the stdlib GMST approximation so a GCRS run can
    still render, while the fallback is recorded as a scene assumption."""
    try:
        from orbitopt.flightlab.frames import itrs_to_gcrs_matrix

        return itrs_to_gcrs_matrix(epoch_utc), ""
    except ImportError as exc:
        return _gmst_only_basis(mjd2000_utc), (
            "Earth texture orientation is the stdlib GMST Rz approximation because the "
            f"optional frame bridge is unavailable ({exc})"
        )


def _check_track(track) -> None:
    """Per-track render-readiness checks (the frame rule, parseable epoch,
    strictly increasing time, non-increasing mass). read_run already enforces
    most of these on a directory; run_to_scene must still enforce them on a
    hand-built FlightRun, and the checks are cheap."""
    where = f"track {track.body_id!r}"
    if track.frame not in CONVERTIBLE_FRAMES:
        raise ValueError(f"{where}: frame {track.frame!r} is not convertible to GCRS "
                         f"(scene export renders {', '.join(CONVERTIBLE_FRAMES)} only)")
    if not track.epoch_utc:
        raise ValueError(f"{where}: epoch_utc is missing -- a frame-bridge conversion and a "
                         "calendar-anchored timeline both need the track's epoch")
    _parse_utc(track.epoch_utc, f"{where}: epoch_utc")  # raises on unparseable
    n = len(track.t_s)
    if n == 0:
        raise ValueError(f"{where}: track has no samples")
    if not (n == len(track.mass_kg) == len(track.position_m) == len(track.velocity_m_s)):
        raise ValueError(f"{where}: t_s/mass_kg/position_m/velocity_m_s differ in length "
                         f"({n}/{len(track.mass_kg)}/{len(track.position_m)}/{len(track.velocity_m_s)})")
    for i in range(1, n):
        if track.t_s[i] <= track.t_s[i - 1]:
            raise ValueError(f"{where}: t_s is not strictly increasing "
                             f"({track.t_s[i - 1]} s followed by {track.t_s[i]} s) -- "
                             "a trajectory must be sampled in strictly increasing time")
        if track.mass_kg[i] > track.mass_kg[i - 1]:
            raise ValueError(f"{where}: mass discontinuity at t_s={track.t_s[i]} s -- mass_kg rises "
                             f"{track.mass_kg[i - 1]} -> {track.mass_kg[i]} kg within one track; "
                             "a track only ever depletes (staging ends a track and starts a new one)")
    if track.mass_kg[0] < 0:
        raise ValueError(f"{where}: mass_kg is negative ({track.mass_kg[0]})")


def _check_run(run: FlightRun) -> datetime:
    """Validate the whole run for rendering; return the single mission epoch
    (every track must agree -- one timeline needs one t=0)."""
    if not run.tracks:
        raise ValueError(f"run {run.run_id!r}: no body tracks to render")
    epochs: set[datetime] = set()
    for track in run.tracks:
        _check_track(track)
        epochs.add(_parse_utc(track.epoch_utc, f"track {track.body_id!r}: epoch_utc"))
    if len(epochs) != 1:
        raise ValueError(
            f"run {run.run_id!r}: body tracks disagree on epoch_utc -- one scene timeline "
            "needs one mission epoch (t=0); convert/normalise epochs before export")
    epoch = epochs.pop()
    known = {track.body_id for track in run.tracks}
    for event in run.events:
        if event.t_s < 0:
            raise ValueError(f"event {event.kind!r} at t_s={event.t_s}: events cannot precede "
                             "the mission epoch (t=0)")
        if event.body_id not in known:
            raise ValueError(f"event {event.kind!r}: body_id {event.body_id!r} is not a body "
                             "of this run -- a marker must attach to a rendered body")
    return epoch


def _km(v: float) -> float:
    """m -> km, rounded to millimetre precision, normalising -0.0 to 0.0 so
    a trail's JSON never contains the negative-zero sign."""
    return round(v / 1000.0, 3) + 0.0


def _gcrs_m(track, epoch: datetime) -> tuple[list, list]:
    """A track's samples in metres / m/s in GCRS -- the frame every handoff
    comparison and every coast-arc seed is taken in (FL-002 frame bridge
    for every non-GCRS frame, as-is for GCRS)."""
    if track.frame == "GCRS":
        return ([[float(x) for x in r] for r in track.position_m],
                [[float(v) for v in v] for v in track.velocity_m_s])
    from orbitopt.flightlab.frames import to_gcrs

    r_arr, v_arr = to_gcrs(track.position_m, track.velocity_m_s, track.frame, epoch, track.t_s)
    return ([[float(x) for x in rr] for rr in r_arr],
            [[float(vv) for vv in vv_row] for vv_row in v_arr])


def _fmt_t(seconds: float) -> str:
    return f"T+{seconds:,.0f} s"


def _event_note(event) -> str:
    parts = [event.body_id]
    scalars = [(k, v) for k, v in event.data.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
    for key, value in scalars[:3]:
        if isinstance(value, str):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={value:,.0f}" if abs(value) >= 100 else f"{key}={value:g}")
    return " \u00b7 ".join(parts)


# ---------------------------------------------------------------------------
# M-06: mission phase row, handoff verdicts, parking-orbit display coast
# ---------------------------------------------------------------------------

def _first_event(run: FlightRun, kind: str):
    return next((e for e in run.events if e.kind == kind), None)


def _iso_at(epoch: datetime, t_s: float) -> str:
    return (epoch + timedelta(seconds=float(t_s))).strftime("%Y-%m-%dT%H:%M:%SZ")


def _handoff_state(track, idx: int, gcrs: dict, epoch: datetime, mjd2000: float,
                   run_id: str) -> HandoffState:
    """The M-01 HandoffState at one sample of a track: GCRS (every frame was
    converted for rendering, and GCRS is the contract's single source of
    truth), epoch = mission epoch + t_s on UTC, stage_config keyed by body id
    so a stack -> child mass drop reads as the staging it is."""
    from orbitopt.mission.contract import Epoch, HandoffState

    r_m, v_m_s = gcrs[track.body_id]
    t = float(track.t_s[idx])
    return HandoffState(
        epoch=Epoch(mjd2000=mjd2000 + t / 86400.0, timescale="UTC"),
        frame="GCRS",
        r_m=tuple(r_m[idx]), v_m_s=tuple(v_m_s[idx]),  # type: ignore[arg-type]
        mass_kg=float(track.mass_kg[idx]),
        stage_config={"body_id": track.body_id},
        metadata={"run_id": run_id},
    )


def _check_doc(check) -> dict:
    return {"id": check.id, "passed": bool(check.passed), "value": check.value,
            "limit": check.limit, "unit": check.unit, "reason_code": check.reason_code}


def _state_doc(body_id: str, t_s: float, state: HandoffState, epoch: datetime) -> dict:
    return {"body": body_id, "time_s": float(t_s), "frame": state.frame,
            "epoch_utc": _iso_at(epoch, t_s),
            "position_m": [float(x) for x in state.r_m],
            "velocity_m_s": [float(x) for x in state.v_m_s],
            "mass_kg": float(state.mass_kg)}


def _tri_state(checks: list, assessed: bool) -> str:
    """Tri-state verdict for a handoff, mirroring the verification-report
    convention: any failing check fails it; nothing checked -- or a part of
    the gate that was never assessed -- is "not_checked", never "passed"."""
    if any(not c["passed"] for c in checks):
        return "failed"
    if checks and assessed:
        return "passed"
    return "not_checked"


def _check_summary(checks: list) -> str:
    n_ok = sum(1 for c in checks if c["passed"])
    reasons = sorted({c["reason_code"] for c in checks if not c["passed"] and c["reason_code"]})
    text = f"{n_ok}/{len(checks)} checks passed" if checks else "no checks"
    return text + (f" ({', '.join(reasons)})" if reasons else "")


def _staging_handoffs(run: FlightRun, epoch: datetime, mjd2000: float, gcrs: dict,
                      mission_scenario: TwoStageLeoScenario | None = None) -> list:
    """One handoff per child body: the parent track's final GCRS state vs the
    child track's initial GCRS state (position, velocity, epoch, mass), graded
    the way the M-07 reference chain grades staging
    (:func:`orbitopt.mission.two_stage_leo.staging_handoffs`):

    * the parent state is first coasted (two-body + J2) to the child's first
      sample -- a backend samples each body on its own clock, and the 2 s
      between the FL-004 stage-1 end and stage-2 start is ~7 km of flight,
      physics, not a lost handoff;
    * boundaries on the ascent lineage (first stage -> insertion body) AND on
      the E2-02 separation fork (the first stage's own branch) are graded --
      from separation the mission is two continuous trajectories, each
      inheriting the stack state. A fairing jettisoned off a branch is
      recorded as not graded: it diverges on purpose.

    Mission replays use the scenario's staging tolerances. Standalone runs
    use the default handoff tolerances plus any declared separation impulse."""
    from orbitopt.mission.branches import separation_fork
    from orbitopt.mission.contract import HandoffTolerances, check_handoff
    from orbitopt.mission.two_stage_leo import (BRIDGE_METHOD, _ascent_lineage,
                                                _bridge_across_gap)

    by_id = {track.body_id: track for track in run.tracks}
    lineage = _ascent_lineage(run)
    graded_pairs = set(zip(lineage[1:], lineage[:-1]))  # (parent, child) up the chain
    fork = separation_fork(run)
    if fork is not None:
        graded_pairs |= {(fork["parent_body"], b["body_id"]) for b in fork["branches"]}
    out = []
    for child in run.tracks:
        parent_id = getattr(child, "parent_body_id", None)
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        before = _handoff_state(parent, len(parent.t_s) - 1, gcrs, epoch, mjd2000, run.run_id)
        after = _handoff_state(child, 0, gcrs, epoch, mjd2000, run.run_id)
        base = {
            "id": f"separation:{child.body_id}",
            "label": f"Separation \u2014 {parent_id} \u2192 {child.body_id}",
            "time": float(child.t_start_s),
            "from_body": parent_id,
            "to": _state_doc(child.body_id, float(child.t_start_s), after, epoch),
        }
        if (parent_id, child.body_id) not in graded_pairs:
            out.append({**base, "status": "not_checked", "graded": False,
                        "detail": "off the ascent lineage and off the separation fork "
                                  "(a fairing jettisoned off a branch): recorded, not graded",
                        "from": _state_doc(parent_id, float(parent.t_end_s), before, epoch),
                        "checks": []})
            continue
        bridge = _bridge_across_gap(before, after)
        from_time = float(parent.t_end_s)
        if bridge is not None and bridge["method"] == BRIDGE_METHOD:
            before, from_time = bridge["state"], float(child.t_start_s)
        impulse_ns = next((float(ev.data.get("separation_impulse_ns")
                                or ev.data.get("impulse_ns") or 0.0)
                           for ev in run.events
                           if ev.kind == "separation" and ev.body_id == parent_id), 0.0)
        if mission_scenario is None:
            tolerances = HandoffTolerances(
                velocity_m_s=0.01 + abs(impulse_ns) / max(float(child.mass_kg[0]), 1.0))
            basis = "standalone staging check"
        else:
            tolerances = replace(mission_scenario.handoff,
                                 velocity_m_s=mission_scenario.staging_velocity_m_s)
            basis = "mission staging tolerance"
        report = check_handoff(before, after, tolerances)
        checks = [_check_doc(c) for c in report.checks]
        detail = _check_summary(checks)
        velocity = next(c for c in checks if c["id"] == "handoff.velocity_continuity")
        detail += (f" · velocity difference {velocity['value']:.5g} m/s"
                   f" (limit {velocity['limit']:.5g} m/s, {basis})")
        if bridge is not None:
            detail += (f" \u00b7 parent coasted {bridge['gap_s']:.1f} s to the child's first sample"
                       if bridge["method"] == BRIDGE_METHOD else f" \u00b7 {bridge['method']}")
        out.append({**base, "status": _tri_state(checks, True), "graded": True,
                    "detail": detail,
                    "bridge": None if bridge is None else {"gap_s": bridge["gap_s"],
                                                           "method": bridge["method"],
                                                           "pole": bridge["pole"]},
                    "from": _state_doc(parent_id, from_time, report.before, epoch),
                    "checks": checks})
    return out


def _seco_handoff(run: FlightRun, epoch: datetime, mjd2000: float, gcrs: dict):
    """The SECO -> parking-orbit handoff: the insertion state plus the run's
    own Gate-B evidence for it (insertion checks, and the tudatpy orbit
    re-fly when one ran). A re-fly that never ran keeps the verdict
    ``not_checked`` -- the parking orbit is only verified once it propagated."""
    if _first_event(run, "seco") is None and not any(
            c.id.startswith("insertion") for c in run.verification.checks):
        return None
    event = _first_event(run, "seco")
    track = (next((t for t in run.tracks if t.body_id == event.body_id), None) if event
             else None) or max(run.tracks, key=lambda t: t.t_end_s)
    idx = (min(range(len(track.t_s)), key=lambda i: abs(track.t_s[i] - event.t_s)) if event
           else len(track.t_s) - 1)
    state = _handoff_state(track, idx, gcrs, epoch, mjd2000, run.run_id)
    relevant = [c for c in run.verification.checks
                if (c.id.startswith("insertion") or c.id.startswith("orbit-refly"))
                and c.id != "orbit-refly-not-assessed"]
    checks = [_check_doc(c) for c in relevant]
    assessed = run.verification.physics_verified is not None
    detail = _check_summary(checks) + ("" if assessed else " \u00b7 orbit re-fly not assessed")
    return {
        "id": "seco", "label": "SECO \u2014 parking orbit insertion",
        "time": float(track.t_s[idx]),
        "status": _tri_state(checks, assessed),
        "detail": detail,
        "state": _state_doc(track.body_id, float(track.t_s[idx]), state, epoch),
        "checks": checks,
    }


def _fork_handoff(run: FlightRun, epoch: datetime, mjd2000: float, gcrs: dict,
                  fork, mission_scenario: TwoStageLeoScenario | None = None) -> dict | None:
    """The E2-02 separation-fork card: the split itself, graded as the mass
    gate the per-boundary cards cannot see -- the branches' initial masses sum
    to the stack's mass at separation within the contract's mass tolerance
    (``mass.separation_fork``). Epoch/frame/r/v continuity across the split is
    each branch's own ``separation:<body>`` card; this card is the one the
    timeline's fork marker and the branched phase row refer to."""
    if fork is None:
        return None
    from orbitopt.mission.contract import HandoffTolerances

    by_id = {track.body_id: track for track in run.tracks}
    parent = by_id[fork["parent_body"]]
    branch_mass = sum(float(by_id[b["body_id"]].mass_kg[0]) for b in fork["branches"]
                      if b["body_id"] in by_id)
    residual = branch_mass - float(fork["parent_mass_kg"])
    limit = (mission_scenario.handoff if mission_scenario else HandoffTolerances()).mass_kg
    if fork.get("powered_gap"):
        from orbitopt.mission.powered_bridge import powered_mass_tolerance_kg

        limit = powered_mass_tolerance_kg(fork["parent_mass_kg"], limit)
    check = {"id": "mass.separation_fork", "passed": bool(abs(residual) <= limit),
             "value": float(residual), "limit": float(limit), "unit": "kg",
             "reason_code": "" if abs(residual) <= limit else "mass_not_conserved_across_fork"}
    state = _handoff_state(parent, len(parent.t_s) - 1, gcrs, epoch, mjd2000, run.run_id)
    return {
        "id": "separation:fork",
        "label": ("Separation fork \u2014 {} \u2192 {}".format(
            fork["parent_body"], " + ".join(b["label"] for b in fork["branches"]))),
        "time": float(fork["t_s"]), "from_body": fork["parent_body"],
        "status": _tri_state([check], True), "graded": True,
        "detail": (f"branches {branch_mass:,.1f} kg vs stack {fork['parent_mass_kg']:,.1f} kg "
                   f"at separation (residual {residual:+.3g} kg) \u00b7 both branches inherit the "
                   "stack state under the M-01 contract"),
        "from": _state_doc(fork["parent_body"], float(parent.t_end_s), state, epoch),
        "checks": [check],
    }


def _mission_phases(run: FlightRun, total: float, coast, fork=None) -> list:
    """Contiguous phase row covering [0, total] from the run's own event
    boundaries: Pad -> Boost (liftoff to MECO) -> Staging coast (MECO to
    separation) -> Upper-stage burn (separation to SECO) -> Parking orbit /
    Propagation over the coast, when one was exported. Missing events merge
    neighbouring phases rather than inventing boundaries.

    E2-02: when the run has a separation fork, the phases after the split are
    per-branch *lanes* -- everything the upper stage does carries
    ``lane: "upper-stage"``, the first stage gets its own lane phase from
    separation to the end of its track, and the base row stops at the fork.
    A phase without a ``lane`` key belongs to the common (pre-split) row."""
    from orbitopt.mission.branches import BRANCH_BOOSTER, BRANCH_UPPER_STAGE

    liftoff = _first_event(run, "liftoff")
    meco = _first_event(run, "meco")
    separation = _first_event(run, "separation")
    seco = _first_event(run, "seco")
    t_liftoff = float(liftoff.t_s) if liftoff else None
    t_meco = float(meco.t_s) if meco else None
    t_separation = float(separation.t_s) if separation else None
    t_seco = float(seco.t_s) if seco else None
    child_starts = [float(t.t_start_s) for t in run.tracks
                    if getattr(t, "parent_body_id", None)]
    t_fork = float(fork["t_s"]) if fork else None
    if t_meco is None:
        t_meco = t_separation if t_separation is not None else t_fork
    if t_separation is None:
        t_separation = t_fork if t_fork is not None else (min(child_starts) if child_starts else None)

    phases: list[dict] = []
    upper_lane = BRANCH_UPPER_STAGE if fork is not None else ""

    def add(label: str, start: float, end: float, note: str = "", lane: str = "") -> None:
        start, end = max(0.0, min(start, total)), min(max(end, 0.0), total)
        if end > start:
            phase = {"label": label, "start": start, "end": end}
            if note:
                phase["note"] = note
            if lane:
                phase["lane"] = lane
            phases.append(phase)

    if t_liftoff is not None and t_liftoff > 0.0:
        add("On the pad", 0.0, t_liftoff, "on the pad until liftoff")
    burn_end = t_meco if t_meco is not None else (t_separation if t_separation is not None else total)
    add("Boost", t_liftoff if t_liftoff is not None else 0.0, burn_end, "first stage to MECO")
    if t_meco is not None and t_separation is not None:
        add("Staging", t_meco, t_separation, "coast from MECO to stage separation")
    if t_seco is not None:
        upper_start = t_separation if t_separation is not None else (
            t_meco if t_meco is not None else (t_liftoff or 0.0))
        add("Upper-stage burn", upper_start, t_seco, "second stage to SECO", upper_lane)
        if coast is not None:
            t0, period, coast_end = coast
            add("Parking orbit", t0, min(t0 + period, coast_end),
                "one revolution of the inserted parking orbit", upper_lane)
            add("Propagation", t0 + period, coast_end, "two-body display coast", upper_lane)
        elif total > t_seco:
            add("Parking orbit", t_seco, total, "coast after SECO", upper_lane)
    covered = max((p["end"] for p in phases), default=0.0)
    if covered < total:  # the run ends before any further event boundary
        add("Coast", covered, total, "no further event boundary in the run data")
    if not phases:
        add("Flight", 0.0, total)
    if fork is not None:  # E2-02: the first stage's lane, from the split to its track end
        by_id = {track.body_id: track for track in run.tracks}
        booster = next((b for b in fork["branches"] if b["kind"] == BRANCH_BOOSTER), None)
        track = by_id.get(booster["body_id"]) if booster else None
        if track is not None and float(track.t_end_s) > float(fork["t_s"]):
            note = ("recovered booster (FL-304) after separation"
                    if booster["body_id"] == "booster"
                    else "first-stage branch after separation (ballistic debris track)")
            add("First stage", float(fork["t_s"]), float(track.t_end_s), note, BRANCH_BOOSTER)
    return phases


def _parking_coast(run: FlightRun, gcrs: dict, coast_orbits: float, coast_samples: int):
    """Two-body display coast of the SECO state: dict with the seed time,
    orbital period and the propagated arc, or None when the insertion state
    has no closed two-body orbit to coast along. The seed is the run's own
    GCRS insertion state (frame-converted), so the arc continues exactly
    where the flown upper-stage trail ends."""
    from orbitopt.flightlab.checks import orbital_elements
    from orbitopt.flightlab.coast import coast_arc

    event = _first_event(run, "seco")
    track = (next((t for t in run.tracks if t.body_id == event.body_id), None) if event
             else None) or max(run.tracks, key=lambda t: t.t_end_s)
    idx = (min(range(len(track.t_s)), key=lambda i: abs(track.t_s[i] - event.t_s)) if event
           else len(track.t_s) - 1)
    r = gcrs[track.body_id][0][idx]
    v = gcrs[track.body_id][1][idx]
    try:
        elements = orbital_elements(r, v)
        period = elements["period_s"]
        if not (math.isfinite(period) and period > 0.0):
            return None
        times, r_m, v_m_s = coast_arc(r, v, coast_orbits * period, samples=coast_samples)
    except (ValueError, KeyError):
        return None
    return {"t0": float(track.t_s[idx]), "period": float(period),
            "duration": coast_orbits * period, "times": times, "r_m": r_m, "v_m_s": v_m_s}


def _coast_body(coast: dict, color: str) -> dict:
    t0, end = coast["t0"], coast["t0"] + coast["duration"]
    return {
        "id": "parking_orbit", "name": "Parking orbit (propagated)", "color": color,
        "kind": "spacecraft", "radiusDisplay": CRAFT_RADIUS_DISPLAY,
        "tStart": t0, "tEnd": end,
        "trail": {
            "times": [t0 + float(t) for t in coast["times"]],
            "positions": [[_km(x), _km(y), _km(z)] for x, y, z in coast["r_m"]],
            "velocities": [[_km(vx), _km(vy), _km(vz)] for vx, vy, vz in coast["v_m_s"]],
        },
        "info": [
            {"label": "Tracked", "value": f"{_fmt_t(t0)} \u2013 {_fmt_t(end)}"},
            {"label": "Arc", "value": f"two-body coast from the SECO state \u00b7 {coast['duration'] / 60.0:,.1f} min"},
            {"label": "Provenance", "value": "display propagation (orbitopt.flightlab.coast) \u2014 not flown data"},
        ],
    }


def _track_body_entry(track, color: str, positions_km: list, velocities_km_s: list) -> dict:
    name = track.body_id.replace("_", " ").strip().capitalize()
    return {
        "id": track.body_id,
        "name": name,
        "color": color,
        "kind": "spacecraft",
        "radiusDisplay": CRAFT_RADIUS_DISPLAY,
        # Optional scene-1.0 lifespan bounds (timeline units): the viewer
        # shows the body only inside [tStart, tEnd] -- it "appears" at
        # t_start_s and disappears at t_end_s.
        "tStart": float(track.t_start_s),
        "tEnd": float(track.t_end_s),
        "trail": {
            "times": [float(t) for t in track.t_s],
            "positions": positions_km,
            "velocities": velocities_km_s,
        },
        "info": [
            {"label": "Tracked", "value": f"{_fmt_t(track.t_start_s)} \u2013 {_fmt_t(track.t_end_s)}"},
            {"label": "Mass", "value": f"{track.mass_kg[0]:,.0f} \u2192 {track.mass_kg[-1]:,.0f} kg"},
            {"label": "Frame", "value": track.frame + (" (converted to GCRS)" if track.frame != "GCRS" else "")},
        ],
    }


def _earth_body(earth_texture: bool, orientation_basis: list | None) -> dict:
    earth: dict[str, Any] = {
        "id": "earth", "name": "Earth", "color": EARTH_COLOR, "kind": "planet",
        "radiusDisplay": EARTH_RADIUS_DISPLAY, "position": [0.0, 0.0, 0.0],
    }
    if earth_texture:
        # orientationBasis is normally the full FL-002 ITRS -> GCRS position
        # rotation used for non-GCRS tracks. A base install may instead get
        # the stdlib GMST Rz fallback; run_to_scene records that assumption.
        if orientation_basis is None:
            raise ValueError("earth texture needs a frame-bridge orientation basis")
        earth.update({
            "texture": EARTH_TEXTURE,
            "radius": EARTH_RADIUS_KM,
            "flattening": EARTH_FLATTENING,
            "rotationPeriodHours": EARTH_SIDEREAL_DAY_HOURS,
            "orientationBasis": [[float(x) for x in row] for row in orientation_basis],
        })
    return earth


def run_to_scene(run: FlightRun, *, earth_texture: bool = True,
                 coast_orbits: float = 0.0, coast_samples: int = 120, target=None,
                 mission_scenario: TwoStageLeoScenario | None = None,
                 rendezvous: dict | None = None) -> dict:
    """Convert a FlightResult v1 run into a scene-1.0 document (Earth scene,
    one trailed body per track, contract events as timeline markers, a
    continuous phase row and a per-boundary ``handoffs`` verdict block).

    Raises ``ValueError`` -- without producing anything renderable -- when the
    run fails the contract checks scene export depends on (frame not
    convertible to GCRS, missing/unparseable epoch, disagreeing epochs,
    non-monotonic time, mass discontinuity) or references an unknown body.
    ``earth_texture=False`` renders Earth as a bare marker (for headless
    diffs); the run data is unaffected either way.

    ``coast_orbits`` (>= 0, default 0) extends the scene past SECO with a
    two-body display coast of the insertion state spanning that many
    orbital periods, which is what gives the Parking-orbit and Propagation
    phases (and their trail) a span to occupy; the coast is recorded as a
    scene assumption, never as flown data.

    ``target`` optionally supplies launch/recovery coordinates for Earth-fixed
    markers. Replay resolves it by the run's target digest; unavailable named
    recovery coordinates are not replaced with an assumed launch-site target.

    ``rendezvous`` optionally is a ``rendezvous-studio-1.0`` sidecar document
    (E12-06): its events join the timeline and its markers become Earth-orbit
    site-kind bodies (keep-out centre, hold point). Sidecar data is display
    metadata only; it never alters tracks or verification.
    """
    if coast_orbits < 0.0:
        raise ValueError(f"coast_orbits must be >= 0, got {coast_orbits!r}")
    epoch = _check_run(run)
    mjd2000 = utc_datetime_to_mjd2000(epoch)
    orientation_basis = None
    orientation_note = ""
    if earth_texture:
        orientation_basis, orientation_note = _earth_orientation_basis(epoch, mjd2000)

    bodies = [_earth_body(earth_texture, orientation_basis)]
    gcrs: dict[str, tuple[list, list]] = {}
    for i, track in enumerate(run.tracks):
        # Convert once per track: GCRS metres for the handoff gates, km/km-s
        # for the trail (non-GCRS goes through the FL-002 frame bridge).
        r_m, v_m_s = _gcrs_m(track, epoch)
        gcrs[track.body_id] = (r_m, v_m_s)
        positions_km = [[_km(x), _km(y), _km(z)] for x, y, z in r_m]
        velocities_km_s = [[_km(vx), _km(vy), _km(vz)] for vx, vy, vz in v_m_s]
        bodies.append(_track_body_entry(track, TRAIL_PALETTE[i % len(TRAIL_PALETTE)],
                                        positions_km, velocities_km_s))

    handoffs = _staging_handoffs(run, epoch, mjd2000, gcrs, mission_scenario)
    from orbitopt.mission.branches import fork_lanes, separation_fork
    fork = separation_fork(run)
    fork_card = _fork_handoff(run, epoch, mjd2000, gcrs, fork, mission_scenario)
    if fork_card is not None:
        handoffs.append(fork_card)
    seco = _seco_handoff(run, epoch, mjd2000, gcrs)
    if seco is not None:
        handoffs.append(seco)

    coast = (_parking_coast(run, gcrs, coast_orbits, coast_samples)
             if coast_orbits > 0.0 else None)
    if coast is not None:
        bodies.append(_coast_body(coast, TRAIL_PALETTE[len(run.tracks) % len(TRAIL_PALETTE)]))

    events = []
    for event in run.events:
        entry: dict[str, Any] = {
            "label": _EVENT_LABELS.get(event.kind, event.kind.replace("_", " ").capitalize()),
            "time": float(event.t_s),
            "note": _event_note(event),
            "bodyId": event.body_id,
        }
        # M-06: put the handoff verdicts where events are drawn (event row,
        # tooltip), so a staging or insertion boundary shows its pass/fail.
        if event.kind == "separation":
            mine = [h for h in handoffs if h["id"].startswith("separation:")
                    and (h.get("from_body") == event.body_id
                         or (h.get("to", {}).get("body") == event.body_id
                             and math.isclose(h["time"], event.t_s, rel_tol=0, abs_tol=1e-6)))]
            if mine:
                entry["note"] += " \u00b7 handoff " + " | ".join(
                    f"{h.get('to', {}).get('body', 'fork')} {h['status'].upper()} ({h['detail']})"
                    for h in mine)
        elif event.kind == "seco" and seco is not None:
            entry["note"] += f" \u00b7 handoff {seco['status'].upper()} ({seco['detail']})"
        events.append(entry)
    if rendezvous:
        for sidecar_event in rendezvous.get("events", ()):
            try:
                t_s = float(sidecar_event["t_s"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append({"label": str(sidecar_event.get("title")
                                             or sidecar_event.get("kind") or "rendezvous"),
                           "time": t_s,
                           "note": "E12-06 mission studio sidecar",
                           "bodyId": str(sidecar_event.get("body_id", ""))})

    t_max = max([float(track.t_end_s) for track in run.tracks]
                + [float(event.t_s) for event in run.events]
                + ([coast["t0"] + coast["duration"]] if coast else []))
    if target is not None:
        bodies.extend(_site_bodies(target, epoch, t_max))
    if rendezvous:
        bodies.extend(_rendezvous_marker_bodies(rendezvous))
    timeline = {
        "unitLabel": "seconds",
        "min": 0.0,
        "max": t_max,
        # Calendar-anchored timeline: t=0 is the mission epoch (UTC -> TDB
        # seconds past J2000, so the viewer's date readout is exact).
        "referenceEpochEt": utc_mjd2000_to_tdb_seconds(mjd2000),
        "phases": _mission_phases(run, t_max,
                                  (coast["t0"], coast["period"], t_max) if coast else None,
                                  fork=fork),
    }
    if fork is not None:
        # E2-02: the branched timeline -- where the mission split, and the
        # lane each branch occupies in the phase row after the fork.
        timeline["branches"] = {"forkT": float(fork["t_s"]),
                                "parentBody": fork["parent_body"],
                                "lanes": fork_lanes(fork)}
    if events:
        timeline["events"] = events

    verification = run.verification
    scene: dict[str, Any] = {
        "schemaVersion": "1.0",
        "id": f"flight-run-{run.run_id}",
        "title": f"Flight Lab \u2014 {run.run_id}",
        "subtitle": f"created {run.created_utc} \u00b7 {len(run.tracks)} bodies \u00b7 {len(run.events)} events",
        "distanceUnit": "km",
        "bodies": bodies,
        "timeline": timeline,
        "initialView": {"focusBodyId": run.tracks[0].body_id, "faceLockBodyId": "earth", "distance": 2000.0},
        # Optional scene-1.0 metadata (foreign members): the run's identity
        # and its three verdicts, kept independent -- null means "not
        # assessed", never "passed".
        "flightRun": {
            "runId": run.run_id,
            **({"backend": run.backend} if run.backend else {}),
            "optimizerConverged": verification.optimizer_converged,
            "physicsVerified": verification.physics_verified,
            "missionConstraintsPassed": verification.mission_constraints_passed,
        },
    }
    if handoffs:
        scene["handoffs"] = handoffs
    assumptions = [orientation_note] if orientation_note else []
    if coast is not None:
        assumptions.append(
            "timeline past SECO is a two-body display coast (orbitopt.flightlab.coast) "
            "seeded at the SECO state -- not verification data; Gate-B's re-fly remains "
            "the authority on the insertion orbit")
    if orientation_note or coast is not None:
        scene["sceneAssumptions"] = assumptions
    validate_scene(scene)
    return scene


def _site_bodies(target, epoch, end: float) -> list[dict]:
    """Earth-fixed launch/recovery markers, rotating with the mission clock.

    The optional target must be the run's resolved input, not a nearest-name
    catalog guess. The 300 m lift is display-only, as in the Starship scene.
    """
    site = target.launch_site
    locations = [("launch_site", "Launch · " + site.name, site.latitude_deg,
                  site.longitude_deg, site.altitude_wgs84_m, "#f2c66d")]
    if target.recovery.kind != "none":
        from orbitopt.flightlab.recovery.boostback import RecoveryTarget
        try:
            recovery = RecoveryTarget.from_scenario(target.to_dict())
        except ValueError:
            recovery = None  # No coordinates for a named offshore target: never assume RTLS.
        if recovery:
            if (recovery.latitude_deg, recovery.longitude_deg) == (site.latitude_deg, site.longitude_deg):
                locations[0] = ("launch_site", "Launch / recovery · " + site.name,
                                site.latitude_deg, site.longitude_deg, site.altitude_wgs84_m, "#f2c66d")
            else:
                locations.append(("recovery_site", "Recovery · " + recovery.name,
                                  recovery.latitude_deg, recovery.longitude_deg, recovery.altitude_m, "#35c48b"))
    count = max(2, min(8192, math.ceil(end / 5) + 1))
    times = [end * i / (count - 1) for i in range(count)]
    basis, _ = _earth_orientation_basis(epoch, utc_datetime_to_mjd2000(epoch))
    bodies = []
    for body_id, name, lat, lon, altitude, color in locations:
        phi, lam = math.radians(lat), math.radians(lon)
        e2 = EARTH_FLATTENING * (2 - EARTH_FLATTENING)
        n = EARTH_RADIUS_KM * 1000 / math.sqrt(1 - e2 * math.sin(phi)**2)
        h = altitude + 300
        xyz = ((n + h) * math.cos(phi) * math.cos(lam),
               (n + h) * math.cos(phi) * math.sin(lam),
               (n * (1 - e2) + h) * math.sin(phi))
        try:
            from orbitopt.flightlab.frames import to_gcrs
            positions, _ = to_gcrs([xyz] * count, [(0, 0, 0)] * count, "ECEF", epoch, times)
            points = [[float(v) / 1000 for v in row] for row in positions]
        except ImportError:
            points = []
            for t in times:
                angle = 2 * math.pi * t / (EARTH_SIDEREAL_DAY_HOURS * 3600)
                c, s = math.cos(angle), math.sin(angle)
                rotated = (c * xyz[0] - s * xyz[1], s * xyz[0] + c * xyz[1], xyz[2])
                points.append([sum(a*b for a, b in zip(row, rotated)) / 1000 for row in basis])
        bodies.append({"id": body_id, "name": name, "color": color, "kind": "site",
                       "radiusDisplay": 4.0, "hideTrail": True,
                       "trail": {"times": times, "positions": points},
                       "info": [{"label": "Coordinates", "value": f"{lat:.4f}°, {lon:.4f}°"},
                                {"label": "Marker", "value": "Earth-fixed target; display lift 300 m"}]})
    return bodies


def _rendezvous_marker_bodies(rendezvous: dict) -> list[dict]:
    """Keep-out centre and hold point markers from a ``rendezvous-studio-1.0``
    sidecar (E12-06), as static site-kind bodies in the scene's position unit.

    The sidecar is producer-computed display metadata: malformed markers are
    skipped, and the run's tracks and verification block are untouched.
    """
    keepout = rendezvous.get("keepout") or {}
    hold = rendezvous.get("hold_point") or {}
    bodies = []
    for marker in rendezvous.get("markers", ()):
        try:
            times = [float(t) for t in marker["times_s"]]
            positions_m = [[float(c) for c in row] for row in marker["positions_m"]]
            body_id = str(marker["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(times) < 2 or len(positions_m) != len(times):
            continue
        info = [{"label": "Marker", "value": str(marker.get("name", body_id))}]
        if body_id == "keepout_center" and keepout:
            info.append({"label": "Keep-out sphere",
                         "value": f"radius {float(keepout['radius_m']):.0f} m"})
            info.append({"label": "Closest approach",
                         "value": f"{float(keepout['min_range_m']):.0f} m — "
                                  + ("clear" if keepout.get("respected") else "INTRUSION")})
        if body_id == "hold_point" and hold:
            info.append({"label": "Hold point (LVLH)",
                         "value": "(" + ", ".join(f"{v:+.0f}" for v in hold["rel_lvlh_m"]) + ") m"})
            info.append({"label": "Position error",
                         "value": f"{float(hold['position_error_m']):.2e} m "
                                  f"(tol {float(hold['tolerance_m']):.0f} m)"})
        bodies.append({"id": body_id, "name": str(marker.get("name", body_id)),
                       "color": str(marker.get("color", "#e05d5d")), "kind": "site",
                       "radiusDisplay": 4.0, "hideTrail": True,
                       "trail": {"times": times,
                                 "positions": [[_km(x), _km(y), _km(z)] for x, y, z in positions_m]},
                       "info": info})
    return bodies
