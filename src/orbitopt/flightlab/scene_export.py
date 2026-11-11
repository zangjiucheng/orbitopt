"""FlightRun -> SceneData v1.0 document: the Flight Lab scene importer
(FL-005, Gate A). Turns a run directory's tracks/events/verification into an
Earth-scene document ``orbitopt view`` (or the E5 Flight Lab page) can open,
with every body appearing/disappearing at its ``t_start_s``/``t_end_s`` and
every contract event becoming a timeline marker.

The contract-to-scene frame rule: a run may only be rendered once every
sample sits in the scene's GCRS. ``GCRS`` tracks render as-is; ``LAUNCH_ECI``
tracks are converted through the FL-002 frame bridge (``frames.to_gcrs``), so
the exported trail uses the same authoritative astropy/ITRS conversion as the
contract pipeline rather than a GMST-only approximation. ``ECEF`` is NOT
convertible here: its position/velocity pair needs the FL-002 frame bridge's
omega-cross-r velocity term, so an ECEF track is rejected with a clear error
rather than rendered wrong. Anything failing the run contract (bad frame enum,
unparseable or missing epoch, non-monotonic time, mass discontinuity, ...)
raises before a single actor is drawn -- a bad run must never render.

Scene-side conventions mirror orbitopt.viz.geo_raising / starship_flight
(km-unit Earth scene) but this module re-implements the handful of Earth
constants it needs rather than importing ``orbitopt.viz``: no module in this
package may depend on Qt/pyvista (see this package's ``__init__``), so a run
stays exportable on a headless, viewer-less install.

The scene document stays scene-1.0 with OPTIONAL fields only (no schema
bump, foreign members are legal): a top-level ``flightRun`` metadata block
(run id + the three never-collapsed success verdicts) and per-body
``tStart``/``tEnd`` lifespan bounds in timeline units, which the reference
viewer uses to make a body appear/disappear mid-mission.

    from orbitopt.flightlab import read_run, run_to_scene
    scene = run_to_scene(read_run("runs/demo-0001"))
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from orbitopt.scene_format import validate_scene
from orbitopt.timescales import utc_datetime_to_mjd2000, utc_mjd2000_to_tdb_seconds

from orbitopt.flightlab.contract import FlightRun, _parse_utc
from orbitopt.flightlab.frames import to_gcrs

#: Earth rendering constants (the scene's distanceUnit is always km here).
EARTH_RADIUS_KM = 6378.137  # WGS-84 equatorial radius
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SIDEREAL_DAY_HOURS = 23.9344696
EARTH_TEXTURE = "earth_daymap.jpg"  # the packaged equirectangular day map
EARTH_COLOR = "#4c8eff"
EARTH_RADIUS_DISPLAY = 4.6
CRAFT_RADIUS_DISPLAY = 3.0

#: Trail colours cycled per track, in contract (manifest) body order.
TRAIL_PALETTE = ("#e8a23e", "#2fa97a", "#d64550", "#b58cf0", "#4cc9f0", "#f25f5c")

#: Frames a track may be rendered in, after conversion if needed. ECEF is
#: deliberately absent -- see this module's docstring.
CONVERTIBLE_FRAMES = ("GCRS", "LAUNCH_ECI")

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


def _check_track(track) -> None:
    """Per-track render-readiness checks (the frame rule, parseable epoch,
    strictly increasing time, non-increasing mass). read_run already enforces
    most of these on a directory; run_to_scene must still enforce them on a
    hand-built FlightRun, and the checks are cheap."""
    where = f"track {track.body_id!r}"
    if track.frame not in CONVERTIBLE_FRAMES:
        hint = (" (ECEF needs the FL-002 frame bridge's rotating-frame "
                "velocity term; it is not renderable by scene export alone)"
                if track.frame == "ECEF" else
                f" (scene export renders {', '.join(CONVERTIBLE_FRAMES)} only)")
        raise ValueError(f"{where}: frame {track.frame!r} is not convertible to GCRS{hint}")
    if not track.epoch_utc:
        raise ValueError(f"{where}: epoch_utc is missing -- a LAUNCH_ECI conversion and a "
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


def _gcrs_track_state(track, epoch: datetime) -> tuple[list, list]:
    r, v = to_gcrs(track.position_m, track.velocity_m_s, track.frame, epoch, track.t_s)
    return (
        [[_km(float(x)), _km(float(y)), _km(float(z))] for x, y, z in r],
        [[_km(float(vx)), _km(float(vy)), _km(float(vz))] for vx, vy, vz in v],
    )


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
            {"label": "Frame", "value": track.frame + (" (converted to GCRS)" if track.frame == "LAUNCH_ECI" else "")},
        ],
    }


def _earth_body(earth_texture: bool, gmst_deg: float) -> dict:
    earth: dict[str, Any] = {
        "id": "earth", "name": "Earth", "color": EARTH_COLOR, "kind": "planet",
        "radiusDisplay": EARTH_RADIUS_DISPLAY, "position": [0.0, 0.0, 0.0],
    }
    if earth_texture:
        # orientationBasis = Rz(GMST at the mission epoch): the equirectangular
        # mesh's prime meridian lands on the Greenwich meridian at t=0, exactly
        # as _apply_rotation_for_time then expects (spin from the timeline).
        th = math.radians(gmst_deg)
        c, s = math.cos(th), math.sin(th)
        earth.update({
            "texture": EARTH_TEXTURE,
            "radius": EARTH_RADIUS_KM,
            "flattening": EARTH_FLATTENING,
            "rotationPeriodHours": EARTH_SIDEREAL_DAY_HOURS,
            "orientationBasis": [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        })
    return earth


def run_to_scene(run: FlightRun, *, earth_texture: bool = True) -> dict:
    """Convert a FlightResult v1 run into a scene-1.0 document (Earth scene,
    one trailed body per track, contract events as timeline markers).

    Raises ``ValueError`` -- without producing anything renderable -- when the
    run fails the contract checks scene export depends on (frame not
    convertible to GCRS, missing/unparseable epoch, disagreeing epochs,
    non-monotonic time, mass discontinuity) or references an unknown body.
    ``earth_texture=False`` renders Earth as a bare marker (for headless
    diffs); the run data is unaffected either way.
    """
    epoch = _check_run(run)
    mjd2000 = utc_datetime_to_mjd2000(epoch)
    gmst = _gmst_deg(mjd2000)

    bodies = [_earth_body(earth_texture, gmst)]
    for i, track in enumerate(run.tracks):
        # Convert once per track: km positions + km/s velocities, both in
        # GCRS (LAUNCH_ECI goes through the FL-002 frame bridge).
        positions_km, velocities_km_s = _gcrs_track_state(track, epoch)
        entry = _track_body_entry(track, TRAIL_PALETTE[i % len(TRAIL_PALETTE)],
                                  positions_km, velocities_km_s)
        bodies.append(entry)

    events = []
    for event in run.events:
        entry: dict[str, Any] = {
            "label": _EVENT_LABELS.get(event.kind, event.kind.replace("_", " ").capitalize()),
            "time": float(event.t_s),
            "note": _event_note(event),
            "bodyId": event.body_id,
        }
        events.append(entry)

    t_max = max([float(track.t_end_s) for track in run.tracks]
                + [float(event.t_s) for event in run.events])
    timeline = {
        "unitLabel": "seconds",
        "min": 0.0,
        "max": t_max,
        # Calendar-anchored timeline: t=0 is the mission epoch (UTC -> TDB
        # seconds past J2000, so the viewer's date readout is exact).
        "referenceEpochEt": utc_mjd2000_to_tdb_seconds(mjd2000),
    }
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
            "optimizerConverged": verification.optimizer_converged,
            "physicsVerified": verification.physics_verified,
            "missionConstraintsPassed": verification.mission_constraints_passed,
        },
    }
    validate_scene(scene)
    return scene
