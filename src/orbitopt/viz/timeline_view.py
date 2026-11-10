"""The zoomable timeline's view math, without Qt: which slice of the mission
the track shows, how zoom / pan / follow move it, the ruler's ticks, and MET
formatting and parsing.

The old scrubber was a fixed 10,000-step slider over the whole mission: on
Starship Flight 15's 22.5 h timeline one pixel was about a minute, so a 30 s
landing burn was narrower than a pixel, and on a 300-day Mars cruise a pixel
was hours. The track instead shows a window [start, end] of the timeline
that zooms from the whole mission down to a couple of seconds, with
continuous (float) time throughout.

Times here are in the timeline's own unit (``timeline.unitLabel``, days for
every orbitopt exporter) unless a name says ``_s`` (seconds). MET is the
timeline time itself: T+0 is t = 0.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

MIN_VIEW_SECONDS = 2.0  # the deepest zoom: two seconds across the track


@dataclass
class TimeWindow:
    """The visible slice [start, end] of a timeline [lo, hi]."""

    lo: float
    hi: float
    start: float
    end: float
    min_span: float  # the deepest zoom, in timeline units

    @classmethod
    def fit(cls, lo: float, hi: float, unit_s: float) -> TimeWindow:
        hi = hi if hi > lo else lo + 1.0
        return cls(lo, hi, lo, hi, min(hi - lo, MIN_VIEW_SECONDS / unit_s))

    @property
    def span(self) -> float:
        return self.end - self.start

    @property
    def full_span(self) -> float:
        return self.hi - self.lo

    @property
    def is_fit(self) -> bool:
        return self.span >= self.full_span * (1.0 - 1e-9)

    def zoom_level(self) -> float:
        """How far in: 1 at fit, 2 at half the mission, ..."""
        return self.full_span / self.span

    def set(self, start: float, end: float) -> None:
        """Show [start, end], clamped to the timeline and to the zoom limits
        (keeping its center where possible)."""
        span = min(max(end - start, self.min_span), self.full_span)
        center = 0.5 * (start + end)
        start = min(max(center - 0.5 * span, self.lo), self.hi - span)
        self.start, self.end = start, start + span

    def reset(self) -> None:
        self.start, self.end = self.lo, self.hi

    def zoom(self, factor: float, anchor: float | None = None) -> None:
        """Zoom in (``factor`` > 1) or out about ``anchor``, which stays at the
        same place on screen (the cursor, for a wheel zoom)."""
        anchor = 0.5 * (self.start + self.end) if anchor is None else anchor
        span = min(max(self.span / factor, self.min_span), self.full_span)
        frac = (anchor - self.start) / self.span if self.span else 0.5
        start = anchor - frac * span
        start = min(max(start, self.lo), self.hi - span)
        self.start, self.end = start, start + span

    def pan(self, dt: float) -> None:
        start = min(max(self.start + dt, self.lo), self.hi - self.span)
        self.start, self.end = start, start + self.span

    def center_on(self, t: float) -> None:
        self.pan(t - 0.5 * (self.start + self.end))

    def follow(self, t: float, margin: float = 0.08) -> bool:
        """Keep ``t`` (the playhead) in view: when it leaves the window, page
        so it sits ``margin`` in from the edge it was heading to -- the view
        jumps once per page, like an editor's timeline, instead of scrolling
        every frame. Returns whether the window moved."""
        if self.start <= t <= self.end:
            return False
        before = self.start
        if t > self.end + self.span or t < self.start - self.span:
            self.center_on(t)  # a jump, not playback running off the edge
        elif t > self.end:
            self.pan(t - (self.start + margin * self.span))
        else:
            self.pan(t - (self.end - margin * self.span))
        return self.start != before

    def to_fraction(self, t: float) -> float:
        return (t - self.start) / self.span if self.span else 0.0

    def from_fraction(self, f: float) -> float:
        return self.start + f * self.span


# Ruler steps in seconds: 1-2-5 below a minute, then the clock's own steps.
_TICK_STEPS_S = (
    0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
    60, 120, 300, 600, 900, 1800,
    3600, 2 * 3600, 3 * 3600, 6 * 3600, 12 * 3600,
    86400, 2 * 86400, 5 * 86400, 10 * 86400, 20 * 86400, 50 * 86400, 100 * 86400, 200 * 86400,
)


def tick_step_s(span_s: float, width_px: float, min_px: float = 90.0) -> float:
    """The smallest ruler step (seconds) whose labels stay ``min_px`` apart."""
    per_px = span_s / max(width_px, 1.0)
    return next((s for s in _TICK_STEPS_S if s / per_px >= min_px), _TICK_STEPS_S[-1])


def minor_step_s(major_s: float) -> float:
    """Unlabelled ticks between the labelled ones."""
    for big, small in ((86400 * 10, 86400), (86400, 6 * 3600), (6 * 3600, 3600), (3600, 900),
                       (900, 300), (300, 60), (60, 15), (30, 5), (10, 2), (5, 1), (1, 0.2)):
        if major_s >= big:
            return min(small, major_s / 2)
    return major_s / 5


def ticks(start: float, end: float, unit_s: float, step_s: float) -> list[float]:
    """Timeline times (in units) of the multiples of ``step_s`` in [start, end]."""
    a, b = start * unit_s, end * unit_s
    k0, k1 = math.ceil(a / step_s - 1e-9), math.floor(b / step_s + 1e-9)
    if k1 - k0 > 2000:
        return []
    return [k * step_s / unit_s for k in range(k0, k1 + 1)]


def format_met(seconds: float, resolution_s: float = 1.0, sign: bool = True) -> str:
    """Mission elapsed time: 'T+0:06:35', 'T-0:00:05', 'T+22:32:34',
    'T+12d 04:12:00'; with tenths when ``resolution_s`` is below a second
    ('T+0:02:19.4'). ``sign=False`` drops the 'T+' (ruler labels)."""
    neg = seconds < 0
    tenths = resolution_s < 1.0
    total = round(abs(seconds) * (10 if tenths else 1))
    frac, total = (total % 10, total // 10) if tenths else (None, total)
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    body = f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h}:{m:02d}:{s:02d}"
    if frac is not None:
        body += f".{frac}"
    if not sign:
        return ("-" if neg else "") + body
    return ("T-" if neg else "T+") + body


def format_ruler_label(seconds: float, step_s: float) -> str:
    """A ruler label, just precise enough for its step and never ambiguous:
    '12d' for whole days, '22h' / '1d 06h' for whole hours, '0:06:30' below
    an hour (not '0:06', which reads as either six minutes or six seconds),
    '2:19.4' at tenths."""
    neg, a = seconds < 0, abs(seconds)
    if step_s >= 86400 and abs(a / 86400 - round(a / 86400)) < 1e-9:
        text = f"{round(a / 86400)}d"
    elif step_s >= 3600 and abs(a / 3600 - round(a / 3600)) < 1e-9:
        d, h = divmod(round(a / 3600), 24)
        text = f"{d}d {h:02d}h" if d else f"{h}h"
    elif step_s < 1:
        m, s = divmod(a, 60)
        h, m = divmod(int(m), 60)
        text = (f"{h}:{m:02d}:" if h else f"{m}:") + f"{s:04.1f}"
    else:
        text = format_met(a, sign=False)
    return ("-" if neg else "") + text


_UNIT_WORDS = {"d": 86400.0, "day": 86400.0, "days": 86400.0, "h": 3600.0, "hr": 3600.0, "hrs": 3600.0,
               "hour": 3600.0, "hours": 3600.0, "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0,
               "minutes": 60.0, "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0}
_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)\s*([a-z]+)")


def _parse_duration_s(text: str) -> float | None:
    text = text.strip().lower()
    if not text:
        return None
    # Clock form, optionally with a day count: '22:32:34', '6:35', '1d 02:00:00', '2d 3:00'.
    m = re.fullmatch(r"(?:(\d+)\s*d\s*)?(\d+(?:\.\d*)?)(?::(\d+(?:\.\d*)?))?(?::(\d+(?:\.\d*)?))?", text)
    if m and (m.group(3) is not None or m.group(1) is not None):
        days = float(m.group(1) or 0)
        fields = [float(g) for g in m.groups()[1:] if g is not None]
        if len(fields) == 2:
            fields = [0.0] + fields if m.group(1) is None else fields + [0.0]  # m:s, or d h:m
        hours, minutes, secs = fields
        return days * 86400 + hours * 3600 + minutes * 60 + secs
    # Unit form: '6m35s', '1d 2h', '395 s', '1.5h'.
    parts = _PART.findall(text)
    if parts and re.fullmatch(r"(?:\s*(?:\d+(?:\.\d*)?|\.\d+)\s*[a-z]+)+\s*", text):
        if all(u in _UNIT_WORDS for _, u in parts):
            return sum(float(v) * _UNIT_WORDS[u] for v, u in parts)
        return None
    return None


def parse_time_input(text: str, now_s: float, unit_s: float, reference_et: float | None = None) -> float | None:
    """What someone typed into the time box, as MET seconds, or None.

    * ``T+22:32:34``, ``T+6m35s``, ``T-10s`` -- mission elapsed time;
    * ``+30s``, ``-5m`` -- relative to the current time (``now_s``);
    * ``22:32:34``, ``6m35s``, ``1d 2h``, ``395s`` -- MET without the T;
    * a bare number -- in the timeline's own unit (``unit_s`` seconds);
    * ``2026-11-14 03:12[:00]`` -- UTC, when the timeline has a calendar epoch.
    """
    raw = text.strip()
    if not raw:
        return None
    if reference_et is not None:
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ t](\d{1,2}:\d{2}(?::\d{2}(?:\.\d*)?)?)\s*(?:utc|z)?", raw.lower())
        if m:
            from datetime import datetime, timezone

            from orbitopt.timescales import utc_datetime_to_tdb_seconds

            clock = m.group(2) if m.group(2).count(":") == 2 else m.group(2) + ":00"
            try:
                dt = datetime.fromisoformat(f"{m.group(1)}T{clock}").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
            return utc_datetime_to_tdb_seconds(dt) - reference_et
    low = raw.lower().replace("−", "-").replace(" ", " ")
    base, sign = 0.0, 1.0
    if low.startswith("t"):
        low = low[1:].lstrip()
        if low[:1] in "+-":
            sign = -1.0 if low[0] == "-" else 1.0
            low = low[1:]
    elif low[:1] in "+-":
        base, sign = now_s, (-1.0 if low[0] == "-" else 1.0)
        low = low[1:]
    try:
        value = float(low) * unit_s  # a bare number: the timeline's unit
    except ValueError:
        value = _parse_duration_s(low)
    if value is None:
        return None
    return base + sign * value
