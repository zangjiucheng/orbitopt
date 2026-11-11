"""Pure telemetry and cached-scene helpers for Flight Lab replay.

No Qt, pyvista, or VTK import appears here. The Flight Lab replay page needs
them, but a run's numeric playback data should remain inspectable on a base
install, and this split lets the derivation tests run without a display.
"""
from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from orbitopt.flightlab.contract import BodyTrack, FlightRun, read_run
from orbitopt.flightlab.scene_export import TRAIL_PALETTE, run_to_scene

EARTH_RADIUS_KM = 6378.137
GRAVITY_M_S2 = 9.80665

ALTITUDE_ALIASES = ("altitude_m", "altitude", "alt_m", "altitude_km")
DYNAMIC_PRESSURE_ALIASES = ("dynamic_pressure_pa", "q_pa", "q")
G_LOAD_ALIASES = ("axial_accel_g", "accel_g", "g_load", "gload", "acceleration_g")
THROTTLE_ALIASES = ("throttle", "throttle_pct", "throttle_level", "engine_throttle",
                    "thrust_level", "thrust_fraction")


@dataclass(frozen=True)
class TelemetrySeries:
    channel_id: str
    label: str
    unit: str
    color: str
    body_id: str
    points: tuple[tuple[float, float], ...]
    derived: bool = False


@dataclass(frozen=True)
class TelemetryChannel:
    id: str
    label: str
    unit: str
    missing_reason: str = ""
    series: tuple[TelemetrySeries, ...] = ()

    @property
    def has_data(self) -> bool:
        return bool(self.series) and any(s.points for s in self.series)


def _norm3(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _finite_pairs(times: Iterable[float], values: Iterable[float]) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for t, value in zip(times, values):
        tf, vf = float(t), float(value)
        if math.isfinite(tf) and math.isfinite(vf):
            pairs.append((tf, vf))
    return tuple(pairs)


def _extra_values(track: BodyTrack, aliases: Iterable[str]) -> tuple[str, tuple[float, ...]] | None:
    wanted = {name.lower() for name in aliases}
    for name, values in track.extra_columns.items():
        if name.lower() in wanted:
            return name, values
    return None


def _track_color(track_colors: Mapping[str, str], track: BodyTrack, index: int) -> str:
    return track_colors.get(track.body_id) or TRAIL_PALETTE[index % len(TRAIL_PALETTE)]


def _altitude_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, ALTITUDE_ALIASES)
    if extra is not None:
        name, values = extra
        factor = 1.0 if name.lower().endswith("_km") else 0.001
        pairs = _finite_pairs(track.t_s, (float(v) * factor for v in values))
    else:
        pairs = _finite_pairs(
            track.t_s,
            (_norm3(pos) / 1000.0 - EARTH_RADIUS_KM for pos in track.position_m),
        )
    if not pairs:
        return None
    return TelemetrySeries("altitude", track.body_id.replace("_", " ").capitalize(), "km",
                           color, track.body_id, pairs, derived=extra is None)


def _speed_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    pairs = _finite_pairs(track.t_s, (_norm3(v) / 1000.0 for v in track.velocity_m_s))
    if not pairs:
        return None
    return TelemetrySeries("speed", track.body_id.replace("_", " ").capitalize(), "km/s",
                           color, track.body_id, pairs)


def _dynamic_pressure_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, DYNAMIC_PRESSURE_ALIASES)
    if extra is None:
        return None
    name, values = extra
    factor = 1.0 if "kpa" in name.lower() else 0.001
    pairs = _finite_pairs(track.t_s, (float(v) * factor for v in values))
    if not pairs:
        return None
    return TelemetrySeries("dynamic_pressure", track.body_id.replace("_", " ").capitalize(), "kPa",
                           color, track.body_id, pairs)


def _g_load_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, G_LOAD_ALIASES)
    if extra is not None:
        pairs = _finite_pairs(track.t_s, extra[1])
    else:
        n = len(track.t_s)
        values = [0.0] * n
        for i in range(n):
            if n == 1:
                acceleration = 0.0
            elif i == 0:
                dt = track.t_s[1] - track.t_s[0]
                acceleration = _norm3(_velocity_delta(track.velocity_m_s[0], track.velocity_m_s[1])) / dt
            elif i == n - 1:
                dt = track.t_s[-1] - track.t_s[-2]
                acceleration = _norm3(_velocity_delta(track.velocity_m_s[-2], track.velocity_m_s[-1])) / dt
            else:
                dt = track.t_s[i + 1] - track.t_s[i - 1]
                acceleration = _norm3(_velocity_delta(track.velocity_m_s[i - 1], track.velocity_m_s[i + 1])) / dt
            values[i] = acceleration / GRAVITY_M_S2
        pairs = _finite_pairs(track.t_s, values)
    if not pairs:
        return None
    return TelemetrySeries("g_load", track.body_id.replace("_", " ").capitalize(), "g",
                           color, track.body_id, pairs, derived=extra is None)


def _mass_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    pairs = _finite_pairs(track.t_s, track.mass_kg)
    if not pairs:
        return None
    return TelemetrySeries("mass", track.body_id.replace("_", " ").capitalize(), "kg",
                           color, track.body_id, pairs)


def _throttle_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, THROTTLE_ALIASES)
    if extra is None:
        return None
    _, values = extra
    numeric = [float(v) for v in values if math.isfinite(float(v))]
    factor = 100.0 if numeric and max(abs(v) for v in numeric) <= 1.05 else 1.0
    pairs = _finite_pairs(track.t_s, (float(v) * factor for v in values))
    if not pairs:
        return None
    return TelemetrySeries("throttle", track.body_id.replace("_", " ").capitalize(), "%",
                           color, track.body_id, pairs)


def _velocity_delta(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(bx) - float(ax) for ax, bx in zip(a, b)]


def build_telemetry_channels(
    run: FlightRun,
    track_colors: Mapping[str, str] | None = None,
) -> tuple[TelemetryChannel, ...]:
    """Build the six replay telemetry channels from a ``FlightRun``.

    Channels are returned in display order even when empty so the page layout
    stays stable and an absent measurement gets an explicit reason instead of
    silently moving next to a neighbouring series.
    """
    track_colors = track_colors or {}
    builders = {
        "altitude": _altitude_series,
        "speed": _speed_series,
        "dynamic_pressure": _dynamic_pressure_series,
        "g_load": _g_load_series,
        "mass": _mass_series,
        "throttle": _throttle_series,
    }
    specs = (
        ("altitude", "Altitude", "km", "no altitude column or launch position"),
        ("speed", "Speed", "km/s", "no velocity samples"),
        ("dynamic_pressure", "Dynamic pressure", "kPa", "no dynamic-pressure column"),
        ("g_load", "G-load", "g", "no acceleration column"),
        ("mass", "Mass", "kg", "no mass samples"),
        ("throttle", "Throttle", "%", "no throttle column"),
    )
    channels: list[TelemetryChannel] = []
    for index, (channel_id, label, unit, missing_reason) in enumerate(specs):
        builder = builders[channel_id]
        series: list[TelemetrySeries] = []
        for body_index, track in enumerate(run.tracks):
            built = builder(track, _track_color(track_colors, track, body_index))
            if built is not None:
                series.append(built)
        missing = "" if series else missing_reason
        channels.append(TelemetryChannel(channel_id, label, unit, missing, tuple(series)))
    return tuple(channels)


def interpolated_value(points: Sequence[tuple[float, float]], t: float) -> float | None:
    """Linear interpolate a telemetry series, clamped to its first/last sample."""
    if not points:
        return None
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    times = [point[0] for point in points]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        return points[0][1]
    if i >= len(points) - 1:
        return points[-1][1]
    t0, v0 = points[i]
    t1, v1 = points[i + 1]
    if t1 == t0:
        return v1
    f = (t - t0) / (t1 - t0)
    return v0 + f * (v1 - v0)


def channel_values_at(channel: TelemetryChannel, t: float) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for series in channel.series:
        value = interpolated_value(series.points, t)
        if value is not None:
            out.append((series.label, value))
    return tuple(out)


def _run_scene_cache_id(run_dir: Path, run: FlightRun) -> str:
    digest = hashlib.sha256()
    digest.update(str(run_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(b"\0")
    return f"flightlab-{run.run_id}-{digest.hexdigest()[:12]}"


def load_run_scene(run_dir: Path | str, *, earth_texture: bool = True) -> tuple[FlightRun, dict]:
    """Read one Flight Lab run directory and return it with its cached scene.

    ``scene_cache.cached_scene`` is deliberately used here: a Flight Lab replay
    is only a playback of a precomputed run, and converting the run dir into
    the scene document is pure but not free. The cache key combines the run id
    with the run-directory file metadata, so editing a CSV or manifest produces
    a new key.
    """
    from orbitopt.scene_cache import cached_scene

    path = Path(run_dir).expanduser()
    run = read_run(path)

    def compute() -> dict:
        return run_to_scene(run, earth_texture=earth_texture)

    return run, cached_scene(_run_scene_cache_id(path, run), compute)
