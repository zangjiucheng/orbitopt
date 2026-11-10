"""Shared visualization state for Mission Control.

Every panel (the 3D viewport, the mission timeline, the inspector, and the
linked views still to come) reads the same small set of facts: which scene is
loaded, what the current timeline time is, and which bodies are selected.
Before this module those facts were private attributes on the main window and
each panel was updated by the window calling into it directly, so adding a
view meant threading another call through every code path that changes time
or selection. MissionState holds them in one place and announces changes with
Qt signals; panels subscribe instead of being poked.

The timeline helpers below are plain functions over the scene's ``timeline``
dict so they can be unit-tested without a display or a QApplication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal

# Two event times closer than this (in timeline units) count as "the same
# time". Matches the tolerance the window used before this module existed.
_TIME_EPS = 1e-9


# --------------------------------------------------------------------------
# Pure timeline helpers
# --------------------------------------------------------------------------
def timeline_span(timeline: dict) -> float:
    """``max - min`` of a timeline, never zero (so it is safe to divide by)."""
    return (timeline["max"] - timeline["min"]) or 1.0


def time_to_fraction(timeline: dict, t: float) -> float:
    """Position of ``t`` along the timeline, clamped to [0, 1]."""
    return min(1.0, max(0.0, (t - timeline["min"]) / timeline_span(timeline)))


def fraction_to_time(timeline: dict, frac: float) -> float:
    return timeline["min"] + frac * (timeline["max"] - timeline["min"])


def clamp_time(timeline: dict, t: float) -> float:
    return min(timeline["max"], max(timeline["min"], t))


def active_event(events: list[dict], t: float) -> dict | None:
    """The last event (in list order) whose time is at or before ``t`` -- the
    maneuver the spacecraft is currently "on"."""
    current = None
    for e in events:
        if e.get("time", 0.0) <= t + _TIME_EPS:
            current = e
    return current


def adjacent_event(events: list[dict], t: float, direction: int) -> dict | None:
    """The first event strictly after ``t`` (``direction > 0``) or the last
    one strictly before it (``direction <= 0``), or None at either end."""
    ordered = sorted(events, key=lambda e: e.get("time", 0.0))
    if direction > 0:
        later = [e for e in ordered if e.get("time", 0.0) > t + _TIME_EPS]
        return later[0] if later else None
    earlier = [e for e in ordered if e.get("time", 0.0) < t - _TIME_EPS]
    return earlier[-1] if earlier else None


def nearest_maneuver(events: list[dict], t: float) -> tuple[dict | None, float | None]:
    """The burn event (one carrying a ``vector``) closest in time to ``t``,
    and its absolute time distance. ``(None, None)`` if there are no burns."""
    best, best_dt = None, None
    for e in events:
        if "vector" not in e:
            continue
        dt = abs(e.get("time", 0.0) - t)
        if best_dt is None or dt < best_dt:
            best, best_dt = e, dt
    return best, best_dt


def burn_owner(scene: dict | None, event: dict) -> str | None:
    """The body a burn event belongs to: its ``bodyId``, else the scene's
    first spacecraft (older scenes carry one vehicle)."""
    if event.get("bodyId"):
        return event["bodyId"]
    return next((b["id"] for b in (scene or {}).get("bodies", []) if b.get("kind") == "spacecraft"), None)


def maneuver_windows(events: list[dict], span: float, fraction: float = 0.05) -> dict[int, float]:
    """``{id(event): window}`` for each burn event (one with a ``vector``):
    how close in time the scrubber must be for its arrow to show. Half the
    gap to the nearest *other* event (events at the very same time don't
    count), capped at ``fraction`` of the timeline span -- so a burn days
    from anything else keeps a wide window, while a landing burn seconds
    after peak deceleration shows only around the landing. (A fixed 5% of a
    22-hour timeline showed Starship's landing-burn arrow for over an hour.)"""
    cap = fraction * span
    windows = {}
    for e in events:
        if "vector" not in e:
            continue
        t = e.get("time", 0.0)
        gaps = [abs(o.get("time", 0.0) - t) for o in events if o is not e and abs(o.get("time", 0.0) - t) > 1e-9]
        windows[id(e)] = min(cap, 0.5 * min(gaps)) if gaps else cap
    return windows


def eased_speed(
    speed: float,
    events: list[dict],
    t: float,
    span: float,
    window_fraction: float,
    target_speed: float,
) -> float:
    """``speed`` eased toward the absolute ``target_speed`` as ``t`` nears the
    closest event, over ``window_fraction`` of the timeline ``span``.

    The easing is a clamp to an absolute pace, not a ratio of ``speed``: 15% of
    1000x is still 150x, which crosses a sub-day window in milliseconds and is
    not a visible slowdown. ``min()`` means it only ever slows playback down,
    never speeds up a deliberately slow ``speed``. A smoothstep ramp avoids a
    visible snap at the window edge.
    """
    if not events:
        return speed
    window = window_fraction * span
    if window <= 0.0:
        return speed
    nearest_dt = min(abs(e.get("time", 0.0) - t) for e in events)
    if nearest_dt >= window:
        return speed
    near_speed = min(speed, target_speed)
    x = nearest_dt / window
    smoothstep = x * x * (3.0 - 2.0 * x)
    return near_speed + (speed - near_speed) * smoothstep


# Body kinds that make sense as a view-frame center. A spacecraft is excluded
# on purpose: centering on it is what Track already does, and its own trail
# would collapse to a point.
_FRAME_KINDS = {"sun", "star", "planet", "dwarf-planet", "moon"}


def frame_options(scene: dict | None) -> list[tuple[str | None, str]]:
    """``(frame_body_id, label)`` choices for re-centering the view: first the
    scene's own frame (``None``), then every *moving* natural body, in scene
    order. Only scenes with a timeline get more than the native option --
    re-centering a static snapshot on a body is just a camera move, which the
    Focus control already does."""
    if not scene:
        return []
    bodies = scene.get("bodies", [])
    central = next((b for b in bodies if b["id"] == scene.get("centralBodyId")), None)
    options: list[tuple[str | None, str]] = [(None, central["name"] if central else "Scene origin")]
    if not scene.get("timeline"):
        return options
    for body in bodies:
        if body is central or body.get("kind") not in _FRAME_KINDS or "trail" not in body:
            continue
        options.append((body["id"], body["name"]))
    return options


_ALTITUDE_KINDS = {"planet", "dwarf-planet", "moon"}


@dataclass(frozen=True)
class AltitudeReference:
    """The planet (or moon) a scene is centered on, as drawn: equatorial
    ``radius``, polar ``flattening`` and unit ``pole`` in the scene frame."""

    body_id: str
    radius: float
    flattening: float
    pole: tuple

    def altitude(self, position) -> float:
        """Height of a scene-frame ``position`` above the drawn ellipsoid,
        to first order in the flattening: |r| - a (1 - f sin^2(lat)), with
        the geocentric latitude from the real pole (metres-level for Earth)."""
        r = np.asarray(position, dtype=float)
        rn = float(np.linalg.norm(r))
        if rn == 0.0:
            return -self.radius
        sin_lat = float(np.dot(r, self.pole)) / rn
        return rn - self.radius * (1.0 - self.flattening * sin_lat * sin_lat)

    def altitudes(self, positions) -> np.ndarray:
        """altitude() for an (N, 3) array of positions, vectorised."""
        r = np.asarray(positions, dtype=float)
        rn = np.linalg.norm(r, axis=1)
        safe = np.where(rn == 0.0, 1.0, rn)
        sin_lat = (r @ np.asarray(self.pole, dtype=float)) / safe
        return np.where(rn == 0.0, -self.radius, rn - self.radius * (1.0 - self.flattening * sin_lat * sin_lat))


def altitude_reference(scene: dict | None) -> AltitudeReference | None:
    """The scene's central planet or moon, when it is drawn with a known
    ``radius`` -- then a spacecraft's distance from the origin reads better
    as an altitude above that body's surface (the ellipsoid drawn). None for
    a Sun-centered or radius-less scene."""
    if not scene:
        return None
    central = next((b for b in scene.get("bodies", []) if b["id"] == scene.get("centralBodyId")), None)
    if central is None or central.get("kind") not in _ALTITUDE_KINDS or not central.get("radius"):
        return None
    basis = central.get("orientationBasis")
    pole = np.asarray(basis, dtype=float)[:, 2] if basis is not None else np.array([0.0, 0.0, 1.0])
    pole = pole / np.linalg.norm(pole)
    return AltitudeReference(central["id"], float(central["radius"]), float(central.get("flattening") or 0.0),
                             tuple(float(c) for c in pole))


def size_exaggeration_range(scene: dict | None) -> tuple[float, float] | None:
    """``(min, max)`` of the ``radiusExaggeration`` declared by bodies drawn as
    real spheres, if any is above 1 -- i.e. whether the view must say that
    sizes are not to scale."""
    if not scene:
        return None
    factors = [
        float(b["radiusExaggeration"]) for b in scene.get("bodies", [])
        if b.get("texture") and b.get("radius") and b.get("radiusExaggeration", 1.0) > 1.0
    ]
    return (min(factors), max(factors)) if factors else None


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------
class MissionState(QObject):
    """The single source of truth for what Mission Control is showing.

    Signals:
      * ``scene_changed(scene)`` -- a new scene was loaded. Time has already
        been reset to the timeline start and the selection cleared, so a
        subscriber can read ``time`` / ``selected_body_id`` directly and does
        not need to wait for separate time/selection signals.
      * ``time_changed(t)`` -- the current timeline time moved.
      * ``selection_changed()`` -- the selected body or the measure partner
        changed (read ``selected_body_id`` / ``measure_body_id``).
      * ``frame_changed(body_id)`` -- the view was re-centered on a body
        (None: the scene's own frame). Display only: world data, times and
        distance readouts do not change with it.
      * ``candidate_changed(candidate_id)`` -- a different optimizer
        candidate was picked for comparison (see the scene's ``candidates``).
        A new scene starts on its exported candidate.

    Camera state (tracking, face-lock, focus zoom) is deliberately *not* here:
    it belongs to the 3D view, not to the mission, and a 2D chart has no use
    for it.
    """

    scene_changed = Signal(object)
    time_changed = Signal(float)
    selection_changed = Signal()
    frame_changed = Signal(object)
    candidate_changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._scene: dict | None = None
        self._bodies: dict[str, dict] = {}
        self._time = 0.0
        self._selected_body_id: str | None = None
        self._measure_body_id: str | None = None
        self._frame_body_id: str | None = None
        self._candidates: dict[str, dict] = {}
        self._candidate_id: str | None = None

    # ------------------------------------------------------------ scene
    @property
    def scene(self) -> dict | None:
        return self._scene

    @property
    def timeline(self) -> dict | None:
        return self._scene.get("timeline") if self._scene else None

    @property
    def events(self) -> list[dict]:
        timeline = self.timeline
        return timeline.get("events", []) if timeline else []

    def body(self, body_id: str | None) -> dict | None:
        """The scene's body dict for ``body_id``, or None if there is no such
        body in the current scene."""
        return self._bodies.get(body_id) if body_id is not None else None

    def primary_spacecraft_id(self) -> str | None:
        """The scene's main spacecraft (first body with kind 'spacecraft'):
        GOES, Orion, the Mars probe -- the body every burn is applied to."""
        for body in self._bodies.values():
            if body.get("kind") == "spacecraft":
                return body["id"]
        return None

    def set_scene(self, scene: dict | None) -> None:
        self._scene = scene
        self._bodies = {b["id"]: b for b in scene.get("bodies", [])} if scene else {}
        timeline = self.timeline
        self._time = timeline["min"] if timeline else 0.0
        # A new scene has different bodies -- never keep an id that may not
        # exist any more.
        self._selected_body_id = None
        self._measure_body_id = None
        self._frame_body_id = None
        candidates = scene.get("candidates", []) if scene else []
        self._candidates = {c["id"]: c for c in candidates}
        exported = next((c for c in candidates if c.get("exported")), None)
        self._candidate_id = (exported or (candidates[0] if candidates else {})).get("id")
        self.scene_changed.emit(scene)

    # ------------------------------------------------------------- time
    @property
    def time(self) -> float:
        return self._time

    def set_time(self, t: float) -> None:
        """Move the current time, clamped to the timeline. A no-op for a scene
        without a timeline (a static snapshot has nothing to scrub)."""
        timeline = self.timeline
        if timeline is None:
            return
        self._time = clamp_time(timeline, float(t))
        self.time_changed.emit(self._time)

    def set_fraction(self, frac: float) -> None:
        timeline = self.timeline
        if timeline is not None:
            self.set_time(fraction_to_time(timeline, frac))

    def active_event(self) -> dict | None:
        return active_event(self.events, self._time)

    # -------------------------------------------------------- selection
    @property
    def selected_body_id(self) -> str | None:
        return self._selected_body_id

    @property
    def measure_body_id(self) -> str | None:
        return self._measure_body_id

    def select_body(self, body_id: str | None) -> None:
        """Select a body (or clear with None). A plain select also drops any
        measure partner. Ids not in the current scene are ignored."""
        if body_id is not None and body_id not in self._bodies:
            return
        self._selected_body_id = body_id
        self._measure_body_id = None
        self.selection_changed.emit()

    def toggle_measure_partner(self, body_id: str) -> bool:
        """Pick ``body_id`` as the second body of a distance measurement, or
        unpick it if it already is. Returns False (and changes nothing) for
        the selected body itself or an unknown id."""
        if body_id not in self._bodies or body_id == self._selected_body_id:
            return False
        self._measure_body_id = None if body_id == self._measure_body_id else body_id
        self.selection_changed.emit()
        return True

    # ------------------------------------------------------------ frame
    @property
    def frame_body_id(self) -> str | None:
        return self._frame_body_id

    def set_frame(self, body_id: str | None) -> None:
        """Re-center the view on ``body_id`` (None: the scene's own frame).
        Ignores ids that are not one of ``frame_options(scene)``."""
        if body_id not in {fid for fid, _ in frame_options(self._scene)}:
            return
        if body_id == self._frame_body_id:
            return
        self._frame_body_id = body_id
        self.frame_changed.emit(body_id)

    # ------------------------------------------------------- candidates
    @property
    def candidates(self) -> list[dict]:
        """The scene's optimizer candidates, best rank first."""
        return list(self._candidates.values())

    @property
    def candidate_id(self) -> str | None:
        return self._candidate_id

    def candidate(self, candidate_id: str | None = None) -> dict | None:
        """A candidate by id, or the selected one."""
        cid = self._candidate_id if candidate_id is None else candidate_id
        return self._candidates.get(cid) if cid is not None else None

    def select_candidate(self, candidate_id: str) -> None:
        if candidate_id not in self._candidates or candidate_id == self._candidate_id:
            return
        self._candidate_id = candidate_id
        self.candidate_changed.emit(candidate_id)
