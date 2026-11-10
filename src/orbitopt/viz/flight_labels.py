"""The flight-data label next to a vehicle in the 3D view (KSP-style):
height, speed and delta-v at the current time.

* H -- altitude above the central body's drawn ellipsoid in a scene centered
  on a planet; distance from the scene's center otherwise.
* V -- inertial speed; below 120 km over a planet also the surface-relative
  speed (minus the body's rotation), the one that matters for entry and
  landing.
* dv -- delta-v the vehicle has spent in burns so far (the sum of its burn
  events' vectors; shown from its first burn on), and the burn it is in,
  within that burn's arrow window.

Pure functions over the scene and a position lookup, so they test without a
window.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from orbitopt.viz.state import altitude_reference, burn_owner
from orbitopt.viz.timeline_bar import timeline_unit_seconds

AU_KM = 149_597_870.7
SURFACE_SPEED_BELOW_KM = 120.0
LABEL_KINDS = {"spacecraft"}


def _km_per_unit(scene: dict) -> float:
    return AU_KM if scene.get("distanceUnit", "km") == "AU" else 1.0


def _fmt_km(km: float) -> str:
    if abs(km) >= 1.0e6:
        return f"{km / 1.0e6:,.2f} M km"
    if abs(km) >= 1000.0:
        return f"{km:,.0f} km"
    return f"{km:,.1f} km"


def _fmt_speed(km_s: float) -> str:
    return f"{km_s * 1000.0:,.0f} m/s" if km_s < 1.0 else f"{km_s:,.2f} km/s"


def velocity_km_s(scene: dict, position: Callable[[float], np.ndarray], t: float) -> np.ndarray:
    """Inertial velocity [km/s] at timeline time ``t`` by a central difference
    over one second of the drawn trajectory (``position(t)`` in scene units)."""
    timeline = scene.get("timeline") or {}
    unit_s = timeline_unit_seconds(timeline.get("unitLabel"))
    h = 1.0 / unit_s
    lo, hi = timeline.get("min", t - h), timeline.get("max", t + h)
    t0, t1 = max(lo, t - h), min(hi, t + h)
    if t1 <= t0:
        return np.zeros(3)
    dr = (np.asarray(position(t1), dtype=float) - np.asarray(position(t0), dtype=float)) * _km_per_unit(scene)
    return dr / ((t1 - t0) * unit_s)


def label_text(scene: dict, body_id: str, t: float, position: Callable[[float], np.ndarray],
               burn_windows: dict[int, float] | None = None) -> str:
    body = next(b for b in scene["bodies"] if b["id"] == body_id)
    r = np.asarray(position(t), dtype=float)
    lines = [body["name"]]

    ref = altitude_reference(scene)
    km = _km_per_unit(scene)
    altitude = None
    if ref is not None:
        altitude = ref.altitude(r) * km
        lines.append(f"H {_fmt_km(altitude)}")
    else:
        lines.append(f"R {_fmt_km(float(np.linalg.norm(r)) * km)}")

    v = velocity_km_s(scene, position, t)
    speed = f"V {_fmt_speed(float(np.linalg.norm(v)))}"
    if ref is not None and altitude is not None and altitude < SURFACE_SPEED_BELOW_KM:
        central = next(b for b in scene["bodies"] if b["id"] == ref.body_id)
        period_h = central.get("rotationPeriodHours")
        if period_h:
            omega = 2.0 * math.pi / (float(period_h) * 3600.0) * np.asarray(ref.pole)
            surface = v - np.cross(omega, r * km)
            speed += f" · surf {_fmt_speed(float(np.linalg.norm(surface)))}"
    lines.append(speed)

    timeline = scene.get("timeline") or {}
    total, done, current = 0.0, 0, None
    windows = burn_windows or {}
    for e in timeline.get("events", []):
        if "vector" not in e or burn_owner(scene, e) != body_id:
            continue
        dv = float(np.linalg.norm(e["vector"]["deltaV"]))
        if e.get("time", 0.0) <= t + 1e-12:
            total += dv
            done += 1
        if abs(e.get("time", 0.0) - t) <= windows.get(id(e), 0.0):
            current = (e, dv)
    # Only once a burn has happened: "0 m/s" during a powered ascent drawn
    # as a profile (no burn events) would read as "no engine firing".
    if done:
        lines.append(f"Δv {total:,.0f} m/s")
    if current is not None:
        e, dv = current
        lines.append(f"» {e['label']} {dv:,.0f} m/s")
    return "\n".join(lines)


def labelled_bodies(scene: dict | None) -> list[str]:
    """The bodies that get a flight-data label: the vehicles."""
    return [b["id"] for b in (scene or {}).get("bodies", []) if b.get("kind") in LABEL_KINDS]
