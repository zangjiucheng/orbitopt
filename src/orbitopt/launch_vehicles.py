"""Launch-vehicle performance model: C3-vs-payload curves, LEO/GTO/TLI
capability, and upper-stage restart/coast limits, read from a configurable
YAML data file (the packaged default is ``orbitopt/data/launch_vehicles.yaml``).

Every value is tagged ``official`` or ``public_estimate``. Queries never
extrapolate: a C3 or payload outside the tabulated curve raises
:class:`OutOfPerformanceRange` (``max_payload`` / ``max_c3``) or comes back
flagged ``in_range=False`` (``query_max_payload`` / ``query_max_c3``).

    from orbitopt.launch_vehicles import get_launch_vehicle
    f9 = get_launch_vehicle("falcon-9")
    f9.max_payload(12.0)     # kg deliverable at C3 = 12 km^2/s^2
    f9.max_c3(3000.0)        # highest C3 reachable with a 3000 kg payload

Needs only pyyaml + jsonschema, like orbitopt.mission_config.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional

import jsonschema
import yaml

PROVENANCES = ("official", "public_estimate")
ORBITS = ("LEO", "GTO", "TLI")
_SCHEMA_FILE = "launch-vehicles-1.0.json"


class OutOfPerformanceRange(ValueError):
    """A query fell outside a vehicle's tabulated performance data."""


@dataclass(frozen=True)
class OrbitCapability:
    payload_kg: float
    provenance: str
    source: str = ""


@dataclass(frozen=True)
class UpperStage:
    max_restarts: int
    max_coast_s: float
    provenance: str
    name: str = ""
    notes: str = ""

    @property
    def max_burns(self) -> int:
        return self.max_restarts + 1


@dataclass(frozen=True)
class PerformanceResult:
    """Non-raising query result. ``value`` is None when ``in_range`` is False."""

    value: Optional[float]
    in_range: bool
    provenance: str
    message: str = ""


@dataclass(frozen=True)
class LaunchVehicle:
    key: str
    name: str
    c3_points: tuple  # ((c3, payload_kg), ...), c3 increasing, payload decreasing
    c3_provenance: str
    upper_stage: UpperStage
    orbit_capability: dict = field(default_factory=dict)  # orbit -> OrbitCapability
    provider: str = ""
    family: str = ""
    c3_source: str = ""

    @property
    def c3_range(self) -> tuple[float, float]:
        return self.c3_points[0][0], self.c3_points[-1][0]

    @property
    def payload_range(self) -> tuple[float, float]:
        return self.c3_points[-1][1], self.c3_points[0][1]

    def max_payload(self, c3: float) -> float:
        """Max payload [kg] at departure energy ``c3`` [km^2/s^2]. Linear
        interpolation between tabulated points; raises
        OutOfPerformanceRange outside the tabulated C3 range."""
        lo, hi = self.c3_range
        c3 = float(c3)
        if not (lo <= c3 <= hi):
            raise OutOfPerformanceRange(
                f"{self.key}: C3 = {c3:g} km^2/s^2 outside tabulated range [{lo:g}, {hi:g}]; "
                "refusing to extrapolate"
            )
        xs = [p[0] for p in self.c3_points]
        i = min(max(bisect.bisect_right(xs, c3) - 1, 0), len(xs) - 2)
        (x0, y0), (x1, y1) = self.c3_points[i], self.c3_points[i + 1]
        return y0 + (y1 - y0) * (c3 - x0) / (x1 - x0)

    def max_c3(self, payload_kg: float) -> float:
        """Max C3 [km^2/s^2] reachable with ``payload_kg``. Inverse of
        max_payload; raises OutOfPerformanceRange outside the tabulated
        payload range."""
        lo, hi = self.payload_range
        m = float(payload_kg)
        if not (lo <= m <= hi):
            raise OutOfPerformanceRange(
                f"{self.key}: payload {m:g} kg outside tabulated range [{lo:g}, {hi:g}] kg; "
                "refusing to extrapolate"
            )
        pts = self.c3_points
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y1 <= m <= y0:
                return x0 + (x1 - x0) * (y0 - m) / (y0 - y1)
        raise AssertionError("unreachable: payload within range but no bracketing segment")

    def query_max_payload(self, c3: float) -> PerformanceResult:
        try:
            return PerformanceResult(self.max_payload(c3), True, self.c3_provenance)
        except OutOfPerformanceRange as exc:
            return PerformanceResult(None, False, self.c3_provenance, str(exc))

    def query_max_c3(self, payload_kg: float) -> PerformanceResult:
        try:
            return PerformanceResult(self.max_c3(payload_kg), True, self.c3_provenance)
        except OutOfPerformanceRange as exc:
            return PerformanceResult(None, False, self.c3_provenance, str(exc))

    def payload_to(self, orbit: str) -> OrbitCapability:
        """LEO/GTO/TLI capability; raises OutOfPerformanceRange when the data
        file has no value for that orbit on this vehicle."""
        orbit = orbit.upper()
        try:
            return self.orbit_capability[orbit]
        except KeyError:
            raise OutOfPerformanceRange(
                f"{self.key}: no {orbit} capability recorded (have: {', '.join(sorted(self.orbit_capability)) or 'none'})"
            ) from None


@lru_cache(maxsize=None)
def _schema() -> dict:
    text = resources.files("orbitopt").joinpath("schemas", _SCHEMA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def _build_vehicle(key: str, d: dict) -> LaunchVehicle:
    pts = tuple((float(c3), float(m)) for c3, m in d["c3_curve"]["points"])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if not x1 > x0:
            raise ValueError(f"{key}: c3_curve C3 values must be strictly increasing ({x0:g} -> {x1:g})")
        if not y1 < y0:
            raise ValueError(f"{key}: c3_curve payloads must be strictly decreasing ({y0:g} -> {y1:g})")
    us = d["upper_stage"]
    return LaunchVehicle(
        key=key,
        name=d["name"],
        provider=d.get("provider", ""),
        family=d.get("family", ""),
        c3_points=pts,
        c3_provenance=d["c3_curve"]["provenance"],
        c3_source=d["c3_curve"].get("source", ""),
        upper_stage=UpperStage(
            max_restarts=int(us["max_restarts"]),
            max_coast_s=float(us["max_coast_s"]),
            provenance=us["provenance"],
            name=us.get("name", ""),
            notes=us.get("notes", ""),
        ),
        orbit_capability={
            orbit: OrbitCapability(float(v["payload_kg"]), v["provenance"], v.get("source", ""))
            for orbit, v in (d.get("orbit_capability") or {}).items()
        },
    )


def parse_launch_vehicles(data: dict) -> dict[str, LaunchVehicle]:
    """Validate a parsed database document and build its vehicles."""
    jsonschema.validate(instance=data, schema=_schema())
    return {key: _build_vehicle(key, v) for key, v in data["vehicles"].items()}


def load_launch_vehicles(path=None) -> dict[str, LaunchVehicle]:
    """Load a launch-vehicle database. ``path=None`` loads the packaged
    default; pass a YAML/JSON file path to use your own data."""
    if path is None:
        return dict(_default_vehicles())
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return parse_launch_vehicles(data)


@lru_cache(maxsize=1)
def _default_vehicles() -> dict[str, LaunchVehicle]:
    text = resources.files("orbitopt").joinpath("data", "launch_vehicles.yaml").read_text(encoding="utf-8")
    return parse_launch_vehicles(yaml.safe_load(text))


def known_vehicles() -> list[str]:
    return sorted(_default_vehicles())


def get_launch_vehicle(key: str) -> LaunchVehicle:
    """Look up a vehicle from the packaged default database."""
    try:
        return _default_vehicles()[key]
    except KeyError:
        raise KeyError(f"unknown launch vehicle {key!r}; known: {', '.join(known_vehicles())}") from None
