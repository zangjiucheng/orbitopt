"""Pure telemetry and cached-scene helpers for Flight Lab replay.

No Qt, pyvista, or VTK import appears here. The Flight Lab replay page needs
them, but a run's numeric playback data should remain inspectable on a base
install, and this split lets the derivation tests run without a display.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from orbitopt.flightlab.contract import BodyTrack, FlightRun, read_run
from orbitopt.flightlab.scene_export import TRAIL_PALETTE, run_to_scene

EARTH_RADIUS_KM = 6378.137
GRAVITY_M_S2 = 9.80665

ALTITUDE_ALIASES = ("altitude_m", "altitude", "alt_m", "altitude_km")
#: E12-06: (channel id, CSV aliases, label, unit) of the rendezvous
#: relative-state series a studio run carries as chaser extra columns.
RENDEZVOUS_SERIES = (
    ("rel_range", ("rel_range_m",), "Range to target", "m"),
    ("rel_speed", ("rel_speed_m_s",), "Relative speed", "m/s"),
    ("rel_x_lvlh", ("rel_x_lvlh_m",), "LVLH x (radial)", "m"),
    ("rel_y_lvlh", ("rel_y_lvlh_m",), "LVLH y (along-track)", "m"),
    ("rel_z_lvlh", ("rel_z_lvlh_m",), "LVLH z (cross-track)", "m"),
)
DYNAMIC_PRESSURE_ALIASES = ("dynamic_pressure_pa", "q_pa", "q")
G_LOAD_ALIASES = ("axial_accel_g", "accel_g", "g_load", "gload", "acceleration_g")
ENGINES_LIT_ALIASES = ("engines_lit", "engines_on", "n_engines")
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
    #: Line style for overlays ("solid" | "dash"); the compare view draws
    #: run B dashed over run A (FL-503).
    style: str = "solid"


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


def _body_label(body_id: str) -> str:
    if body_id.startswith("stage") and body_id[5:].isdigit():
        return "Stage " + body_id[5:]
    return body_id.replace("_", " ").capitalize()


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


def _estimated_throttle_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    """A track that records no throttle (OpenTsiolkovsky stage tracks) still burns propellant: the throttle is
    estimated from the mass flow, as a percentage of the track's peak burn rate -- which is the full-throttle
    rate whenever the stage reaches full throttle (a stage that never does reads high). Labelled "(est.)"."""
    t, m = [float(x) for x in track.t_s], [float(x) for x in track.mass_kg]
    if len(t) < 2:
        return None
    flow = [0.0] + [max(0.0, (m[k - 1] - m[k]) / (t[k] - t[k - 1])) if t[k] > t[k - 1] else 0.0
                    for k in range(1, len(t))]
    top = max(flow)
    if top <= 0.0:
        return None                                              # never burned: nothing to show
    # the flow at sample k is the burn over the interval ENDING at k; plot it at the interval's start so a burn
    # starts where the mass starts to fall (and a coast starts where it stops)
    pairs = _finite_pairs(t[:-1], (100.0 * flow[k + 1] / top for k in range(len(t) - 1)))
    if not pairs:
        return None
    return TelemetrySeries("throttle", _body_label(track.body_id) + " (est.)", "%", color, track.body_id, pairs,
                           derived=True)


def _throttle_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, THROTTLE_ALIASES)
    if extra is None:
        return _estimated_throttle_series(track, color)
    _, values = extra
    numeric = [float(v) for v in values if math.isfinite(float(v))]
    factor = 100.0 if numeric and max(abs(v) for v in numeric) <= 1.05 else 1.0
    pairs = _finite_pairs(track.t_s, (float(v) * factor for v in values))
    if not pairs:
        return None
    return TelemetrySeries("throttle", track.body_id.replace("_", " ").capitalize(), "%",
                           color, track.body_id, pairs)


def _engines_series(track: BodyTrack, color: str) -> TelemetrySeries | None:
    extra = _extra_values(track, ENGINES_LIT_ALIASES)
    if extra is None:
        return None
    _, values = extra
    pairs = _finite_pairs(track.t_s, (float(v) for v in values))
    if not pairs:
        return None
    return TelemetrySeries("engines_lit", track.body_id.replace("_", " ").capitalize(), "engines",
                           color, track.body_id, pairs)


def _velocity_delta(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(bx) - float(ax) for ax, bx in zip(a, b)]


def build_telemetry_channels(
    run: FlightRun,
    track_colors: Mapping[str, str] | None = None,
    *, vehicle: Mapping | None = None,
) -> tuple[TelemetryChannel, ...]:
    """Build replay telemetry, including explicitly labelled mass-based fuel estimates.

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
        "engines_lit": _engines_series,
    }
    specs = (
        ("altitude", "Altitude", "km", "no altitude column or launch position"),
        ("speed", "Speed", "km/s", "no velocity samples"),
        ("dynamic_pressure", "Dynamic pressure", "kPa", "no dynamic-pressure column"),
        ("g_load", "G-load", "g", "no acceleration column"),
        ("mass", "Mass", "kg", "no mass samples"),
        ("throttle", "Throttle", "%", "no throttle column"),
        ("engines_lit", "Engines lit", "engines", "no engine-count column"),
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
    fuel = []
    for i, track in enumerate(run.tracks):
        color = _track_color(track_colors, track, i)
        extra = _extra_values(track, ("propellant_kg", "fuel_remaining_kg"))
        if extra is not None:
            fuel.append(TelemetrySeries("propellant", _body_label(track.body_id) + " (recorded)", "t",
                        color, track.body_id, _finite_pairs(track.t_s, (v / 1000 for v in extra[1]))))
        elif vehicle:
            dry = _non_propellant_mass(vehicle, track.body_id)
            if dry is not None:
                # Unknown/mismatched mass conventions must not become zero-fuel curves.
                values = [m - dry for m in track.mass_kg]
                if math.isfinite(dry) and dry >= 0 and all(math.isfinite(v) and v >= -1e-3 for v in values):
                    points = _finite_pairs(track.t_s, (max(0, v) / 1000 for v in values))
                    # An attached upper stage carries its declared fuel before
                    # its own track starts. Extend only a verified unburned
                    # initial stack mass; never extend an already-burning arc.
                    stages = vehicle.get("stages", [])
                    if track.body_id.startswith("stage") and track.body_id[5:].isdigit():
                        index = int(track.body_id[5:]) - 1
                        initial = sum(float(s["mass_initial_kg"]) for s in stages[index:])
                        start = min(t.t_start_s for t in run.tracks)
                        if start < track.t_start_s and abs(track.mass_kg[0] - initial) < 1e-3:
                            points = ((start, points[0][1]), *points)
                    fuel.append(TelemetrySeries("propellant", _body_label(track.body_id) + " (est.)", "t",
                                color, track.body_id, points, derived=True))
    channels.append(TelemetryChannel("propellant", "Propellant remaining", "t",
                    "no recorded propellant or matching vehicle dry masses" if not fuel else "", tuple(fuel)))
    vertical, downrange = [], []
    for i, track in enumerate(run.tracks):
        color = _track_color(track_colors, track, i)
        label = _body_label(track.body_id)
        pairs = _finite_pairs(track.t_s, (sum(r * v for r, v in zip(pos, vel)) / _norm3(pos)
                              if _norm3(pos) else float("nan")
                              for pos, vel in zip(track.position_m, track.velocity_m_s)))
        vertical.append(TelemetrySeries("vertical_speed", label + " (est.)", "m/s",
                                       color, track.body_id, pairs, derived=True))
        extra = _extra_values(track, ("downrange_m",))
        if extra:
            downrange.append(TelemetrySeries("downrange", label, "km", color, track.body_id,
                                            _finite_pairs(track.t_s, (v / 1000 for v in extra[1]))))
    channels.extend((TelemetryChannel("vertical_speed", "Vertical speed (radial)", "m/s", "no state samples", tuple(vertical)),
                     TelemetryChannel("downrange", "Downrange", "km", "no recorded downrange", tuple(downrange))))
    fins = []
    for track in run.tracks:
        valid = _extra_values(track, ("fin_telemetry_valid",))
        for column, label, color in (("fin_delta_deg", "Fin deflection (mean abs.)", "#35c48b"),
                                     ("fin_alpha_deg", "Angle of attack", "#4cc9f0"),
                                     ("fin_bank_deg", "Bank", "#b58cf0")):
            extra = _extra_values(track, (column,))
            if extra is None:
                continue
            values = extra[1] if valid is None else (
                value if recorded else math.nan for value, recorded in zip(extra[1], valid[1]))
            points = _finite_pairs(track.t_s, values)
            if points:
                fins.append(TelemetrySeries("grid_fins", label, "deg", color, track.body_id, points))
    channels.append(TelemetryChannel("grid_fins", "Grid Fins", "deg",
                                     "no recorded grid-fin telemetry" if not fins else "", tuple(fins)))
    for channel_id, aliases, label, unit in RENDEZVOUS_SERIES:
        series = []
        for index, track in enumerate(run.tracks):
            extra = _extra_values(track, aliases)
            if extra is not None:
                series.append(TelemetrySeries(channel_id, _body_label(track.body_id), unit,
                                              _track_color(track_colors, track, index),
                                              track.body_id, _finite_pairs(track.t_s, extra[1])))
        if series:
            channels.append(TelemetryChannel(channel_id, label, unit, "", tuple(series)))
    return tuple(channels)


def _non_propellant_mass(vehicle: Mapping, body_id: str) -> float | None:
    """Stage track mass includes its dry mass and the attached upper stack.

    Recovery/debris tracks carry only the detached first stage. The vehicle
    designer already folds payload into the final stage's wet and dry mass.
    """
    def number(x):
        return float(x.get("value") if isinstance(x, Mapping) else x)
    try:
        stages = vehicle.get("stages", [])
        if body_id in ("booster", "stage1_debris"):
            dry = (vehicle.get("booster") or {}).get("dry_mass_kg")
            return number(dry if dry is not None else stages[0]["mass_dry_kg"])
        if body_id.startswith("stage") and body_id[5:].isdigit():
            index = int(body_id[5:]) - 1
            if index < 0 or index >= len(stages):
                return None
            return (number(stages[index]["mass_dry_kg"])
                    + sum(number(s["mass_initial_kg"]) for s in stages[index + 1:]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


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


def _run_scene_cache_id(run_dir: Path, run: FlightRun, *, earth_texture: bool, scenario) -> str:
    digest = hashlib.sha256()
    staging = (scenario.handoff, scenario.staging_velocity_m_s) if scenario else None
    digest.update(f"replay-staging-v3-verification:{earth_texture}:{staging!r}\0".encode())
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
    from orbitopt.flightlab.replay_context import replay_inputs, replay_mission_scenario
    _, target = replay_inputs(run, path)
    scenario = replay_mission_scenario(run)

    def compute() -> dict:
        from orbitopt.flightlab.segment import flight_run_report
        from orbitopt.verify.report import publish_to_scene

        rendezvous = None
        side = path / "rendezvous.json"
        if side.is_file():
            try:
                doc = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = None
            if isinstance(doc, dict) and doc.get("schema_version") == "rendezvous-studio-1.0":
                rendezvous = doc
        scene = run_to_scene(run, earth_texture=earth_texture, target=target,
                             mission_scenario=scenario, rendezvous=rendezvous)
        # The run's own verification (checks recorded in verification.json, incl. the orbit re-fly) as a
        # verification-report-1.0 on the scene: without it the Verification panel says NOT CHECKED for a run
        # that was in fact verified. A run with no assessed verdict still reports not_checked.
        return publish_to_scene(scene, [flight_run_report(run, target)])

    return run, cached_scene(_run_scene_cache_id(path, run, earth_texture=earth_texture,
                                                scenario=scenario), compute)
