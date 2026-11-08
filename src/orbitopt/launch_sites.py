"""Launch-site database: location, Earth-rotation velocity gain, range-safety
azimuth corridors and the orbit inclinations they make reachable.

The data lives in a configurable YAML/JSON file (default: the packaged
``orbitopt/data/launch_sites.yaml``) validated against
``orbitopt/schemas/launch-sites-1.0.json``. Every entry cites its sources.

    from orbitopt.launch_sites import get_launch_site
    site = get_launch_site("Kourou")          # name, id or alias, case-insensitive
    site.reachable_inclination_range()        # (5.24, 100.4) deg
    site.can_reach_inclination(51.6)          # True

Inclinations are for a *direct* ascent (no dog-leg / plane change):
``cos i = cos(phi) * sin(Az)`` with geodetic latitude ``phi`` and launch
azimuth ``Az``. Like orbitopt.mission_config this module needs only pyyaml +
jsonschema -- no compute stack.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import jsonschema
import yaml

# IERS conventional mean angular velocity of the Earth [rad/s].
OMEGA_EARTH = 7.2921150e-5
# WGS-84 ellipsoid.
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3
# Stored rotation gains may differ from the recomputed value by at most this.
_GAIN_TOLERANCE_M_S = 1.0


def rotation_speed_gain(latitude_deg: float, altitude_m: float = 0.0) -> float:
    """Eastward inertial speed [m/s] of a point fixed on the rotating Earth at
    the given geodetic latitude and height (WGS-84 ellipsoid)."""
    phi = math.radians(latitude_deg)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(phi) ** 2)
    return OMEGA_EARTH * (n + altitude_m) * math.cos(phi)


def inclination_from_azimuth(latitude_deg: float, azimuth_deg: float) -> float:
    """Orbit inclination [deg] of a direct ascent at launch azimuth
    ``azimuth_deg`` (clockwise from north) from latitude ``latitude_deg``."""
    c = math.cos(math.radians(latitude_deg)) * math.sin(math.radians(azimuth_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _corridor_inclination_range(latitude_deg: float, start: float, end: float) -> tuple[float, float]:
    """Min/max inclination over a clockwise azimuth sweep start -> end.
    i(Az) is continuous, so the image is an interval bounded by the endpoint
    values and the interior extrema at Az=90 (i=|phi|) / Az=270 (i=180-|phi|)."""
    sweep = (end - start) % 360.0
    if sweep == 0.0 and end != start:
        sweep = 360.0
    candidates = [start, end]
    for extremum in (90.0, 270.0):
        if (extremum - start) % 360.0 <= sweep:
            candidates.append(extremum)
    incs = [inclination_from_azimuth(latitude_deg, az) for az in candidates]
    return min(incs), max(incs)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


@dataclass(frozen=True)
class SourceRef:
    covers: tuple[str, ...]
    ref: str


@dataclass(frozen=True)
class LaunchSite:
    id: str
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    rotation_speed_gain_m_s: float
    azimuth_corridors_deg: tuple[tuple[float, float], ...]
    sources: tuple[SourceRef, ...]
    aliases: tuple[str, ...] = ()
    country: str = ""
    reference_pad: str = ""
    confidence: str = "published"

    @classmethod
    def from_dict(cls, d: dict) -> "LaunchSite":
        return cls(
            id=d["id"],
            name=d["name"],
            latitude_deg=float(d["latitude_deg"]),
            longitude_deg=float(d["longitude_deg"]),
            altitude_m=float(d["altitude_m"]),
            rotation_speed_gain_m_s=float(d["rotation_speed_gain_m_s"]),
            azimuth_corridors_deg=tuple((float(a), float(b)) for a, b in d["azimuth_corridors_deg"]),
            sources=tuple(SourceRef(tuple(s["covers"]), s["ref"]) for s in d["sources"]),
            aliases=tuple(d.get("aliases", ())),
            country=d.get("country", ""),
            reference_pad=d.get("reference_pad", ""),
            confidence=d.get("confidence", "published"),
        )

    def rotation_gain_along(self, azimuth_deg: float) -> float:
        """Component [m/s] of the Earth-rotation velocity along the launch
        azimuth (negative for westward launches)."""
        return self.rotation_speed_gain_m_s * math.sin(math.radians(azimuth_deg))

    def inclination_for_azimuth(self, azimuth_deg: float) -> float:
        return inclination_from_azimuth(self.latitude_deg, azimuth_deg)

    def reachable_inclination_intervals(self) -> list[tuple[float, float]]:
        """Disjoint, sorted [min, max] inclination intervals [deg] reachable
        by direct ascent within the site's azimuth corridors."""
        return _merge_intervals([
            _corridor_inclination_range(self.latitude_deg, a, b)
            for a, b in self.azimuth_corridors_deg
        ])

    def reachable_inclination_range(self) -> tuple[float, float]:
        """Overall (min, max) reachable inclination [deg]. May span gaps if
        corridors are disjoint -- see reachable_inclination_intervals()."""
        intervals = self.reachable_inclination_intervals()
        return intervals[0][0], intervals[-1][1]

    def can_reach_inclination(self, inclination_deg: float, tol_deg: float = 1e-6) -> bool:
        return any(lo - tol_deg <= inclination_deg <= hi + tol_deg
                   for lo, hi in self.reachable_inclination_intervals())

    def sources_for(self, field: str) -> list[str]:
        """References cited for a field group (location, altitude,
        azimuth_corridors, rotation_speed_gain)."""
        return [s.ref for s in self.sources if field in s.covers]


@lru_cache(maxsize=None)
def _schema() -> dict:
    text = resources.files("orbitopt").joinpath("schemas", "launch-sites-1.0.json").read_text(encoding="utf-8")
    return json.loads(text)


def validate_launch_site_data(data: dict) -> None:
    """Validate a parsed database document. Raises jsonschema.ValidationError
    on a schema mismatch, ValueError on semantic problems (duplicate names,
    a stored rotation gain inconsistent with latitude/altitude)."""
    jsonschema.validate(instance=data, schema=_schema())
    seen: dict[str, str] = {}
    for site in data["sites"]:
        for key in (site["id"], site["name"], *site.get("aliases", ())):
            norm = _normalize(key)
            if norm in seen and seen[norm] != site["id"]:
                raise ValueError(f"launch-site name {key!r} used by both {seen[norm]!r} and {site['id']!r}")
            seen[norm] = site["id"]
        expected = rotation_speed_gain(site["latitude_deg"], site["altitude_m"])
        if abs(site["rotation_speed_gain_m_s"] - expected) > _GAIN_TOLERANCE_M_S:
            raise ValueError(
                f"{site['id']}: rotation_speed_gain_m_s={site['rotation_speed_gain_m_s']} "
                f"inconsistent with latitude/altitude (expected {expected:.1f} m/s)"
            )


def _normalize(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").split()).casefold()


class LaunchSiteDatabase:
    """Name-indexed collection of LaunchSite records."""

    def __init__(self, sites: list[LaunchSite]):
        self._sites = {s.id: s for s in sites}
        self._index: dict[str, str] = {}
        for s in sites:
            for key in (s.id, s.name, *s.aliases):
                self._index[_normalize(key)] = s.id

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self):
        return iter(self._sites.values())

    def __contains__(self, name: str) -> bool:
        return _normalize(name) in self._index

    def names(self) -> list[str]:
        return list(self._sites)

    def get(self, name: str) -> LaunchSite:
        """Look up a site by id, name or alias (case/dash/underscore-insensitive)."""
        site_id = self._index.get(_normalize(name))
        if site_id is None:
            raise KeyError(f"unknown launch site {name!r} -- known: {', '.join(self._sites)}")
        return self._sites[site_id]

    __getitem__ = get


def load_launch_sites(path=None) -> LaunchSiteDatabase:
    """Load and validate a launch-site database from a YAML/JSON file, or the
    packaged default when ``path`` is None."""
    if path is None:
        text = resources.files("orbitopt").joinpath("data", "launch_sites.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'launch_sites.yaml'}: expected a mapping at the top level")
    validate_launch_site_data(data)
    return LaunchSiteDatabase([LaunchSite.from_dict(s) for s in data["sites"]])


@lru_cache(maxsize=1)
def default_launch_sites() -> LaunchSiteDatabase:
    return load_launch_sites()


def get_launch_site(name: str) -> LaunchSite:
    """Look up a site in the packaged database by id, name or alias."""
    return default_launch_sites().get(name)


def reachable_inclination_range(name: str) -> tuple[float, float]:
    """(min, max) direct-ascent inclination [deg] for a packaged site."""
    return get_launch_site(name).reachable_inclination_range()
