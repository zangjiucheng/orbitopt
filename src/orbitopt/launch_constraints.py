"""Launch constraints for optimizer fitness: rocket C3 capability and
launch-site reachability of the departure asymptote declination (DLA).

A candidate's departure hyperbolic-excess vector ``v_inf`` (heliocentric,
m/s) is checked against

* **C3**: ``C3 = |v_inf|^2`` [km^2/s^2] must not exceed the vehicle's
  ``max_c3(payload_kg)`` (orbitopt.launch_vehicles; never extrapolated).
* **DLA**: the declination of ``v_inf`` in the Earth equatorial (J2000)
  frame. A departure from a circular parking orbit of inclination ``i``
  can only point its asymptote at ``|DLA| <= min(i, 180 - i)``, so the
  constraint is ``|DLA| <= max_abs_dla_deg`` where the limit is the best
  ``min(i, 180 - i)`` over the site's direct-ascent reachable inclinations
  (orbitopt.launch_sites).

Violating candidates are either marked invalid (fitness replaced by
``invalid_fitness`` -- the sentinel the problems already use for failed
Lambert solves -- plus the raw violation, so the search still has a slope) or
penalized (``objective + penalty_weight * excess`` with
excess in km^2/s^2 for C3 and deg for DLA).

``evaluate`` / ``apply`` are written against the numpy/cupy array-module
abstraction so the same code serves the CPU and GPU batch-fitness paths.

    from orbitopt.launch_constraints import LaunchConstraints
    lc = LaunchConstraints.from_names("falcon-heavy", 4000.0, "cape-canaveral")
    problem = Transfer2BodyProblem(earth, mars, t0b, tofb, launch_constraints=lc)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from orbitopt.data_provenance import DataSource, data_sources_report
from orbitopt.launch_sites import LaunchSite, get_launch_site
from orbitopt.launch_vehicles import LaunchVehicle, get_launch_vehicle

# IAU 1976 / J2000 mean obliquity of the ecliptic [deg].
OBLIQUITY_J2000_DEG = 23.4392911
MODES = ("invalid", "penalty")
FRAMES = ("ecliptic", "equatorial")


def max_abs_dla_for_site(site: LaunchSite) -> float:
    """Largest |DLA| [deg] reachable from ``site``: max over the site's
    reachable inclinations i of min(i, 180 - i)."""
    best = 0.0
    for lo, hi in site.reachable_inclination_intervals():
        if lo <= 90.0 <= hi:
            return 90.0
        best = max(best, min(lo, 180.0 - lo), min(hi, 180.0 - hi))
    return best


def _xp_of(arr):
    """numpy, or cupy when ``arr`` is a cupy array."""
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy

        return cupy
    return np


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Per-candidate constraint diagnostics (arrays of shape (N,))."""

    c3: object  # km^2/s^2
    dla_deg: object  # deg, Earth equatorial frame
    c3_excess: object  # km^2/s^2 above the vehicle limit (0 if satisfied)
    dla_excess_deg: object  # deg above the site limit (0 if satisfied)
    feasible: object  # bool


@dataclass(frozen=True)
class LaunchConstraints:
    """Launch constraint set attached to a problem's fitness.

    ``enabled=False`` turns every check into a no-op so a run can be
    A/B-compared with the same problem object configuration. Either limit
    may be disabled individually by leaving ``vehicle`` or ``site`` None.
    ``frame`` names the frame of the incoming v_inf vectors: ``"ecliptic"``
    (pykep's heliocentric J2000 ecliptic ephemerides, the default) or
    ``"equatorial"``.
    """

    vehicle: LaunchVehicle | None = None
    payload_kg: float = 0.0
    site: LaunchSite | None = None
    mode: str = "invalid"
    # m/s of objective per km^2/s^2 (C3) or deg (DLA) of excess. Near a Mars-
    # window optimum d(dv)/d(C3) is ~300 m/s per km^2/s^2, so 100 was
    # measured to leave the optimum slightly infeasible; 1000 does not.
    penalty_weight: float = 1000.0
    invalid_fitness: float = 1e6
    enabled: bool = True
    frame: str = "ecliptic"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {self.frame!r}")
        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be >= 0")
        if self.vehicle is not None:
            # Validate up front: raises OutOfPerformanceRange for a payload
            # the vehicle's tabulated curve does not cover.
            self.vehicle.max_c3(self.payload_kg)

    @classmethod
    def from_names(cls, vehicle: str | None, payload_kg: float, site: str | None, **kw) -> "LaunchConstraints":
        """Build from packaged database keys (vehicle key, site id/name/alias)."""
        return cls(
            vehicle=get_launch_vehicle(vehicle) if vehicle else None,
            payload_kg=float(payload_kg),
            site=get_launch_site(site) if site else None,
            **kw,
        )

    def with_enabled(self, enabled: bool) -> "LaunchConstraints":
        return replace(self, enabled=bool(enabled))

    @property
    def max_c3(self) -> float:
        """Vehicle C3 limit [km^2/s^2] at ``payload_kg`` (inf without a vehicle)."""
        return math.inf if self.vehicle is None else self.vehicle.max_c3(self.payload_kg)

    @property
    def max_abs_dla_deg(self) -> float:
        """Site |DLA| limit [deg] (90 = unconstrained without a site)."""
        return 90.0 if self.site is None else max_abs_dla_for_site(self.site)

    def data_sources(self) -> list[DataSource]:
        """Provenance of the data the limits are computed from: the vehicle's
        C3 curve (C3 limit) and the site's location + azimuth corridors
        (DLA limit)."""
        out: list[DataSource] = []
        if self.vehicle is not None:
            out += self.vehicle.data_sources(["c3_curve"])
        if self.site is not None:
            out += self.site.data_sources(["location", "azimuth_corridors"])
        return out

    def describe(self) -> dict:
        """JSON-ready summary of the constraint set, including the source and
        confidence of the data behind each limit (``data_sources``)."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "vehicle": self.vehicle.key if self.vehicle else None,
            "payload_kg": self.payload_kg,
            "max_c3_km2_s2": self.max_c3,
            "site": self.site.id if self.site else None,
            "max_abs_dla_deg": self.max_abs_dla_deg,
            "data_sources": data_sources_report(self.data_sources()),
        }

    # --- vectorized evaluation ------------------------------------------
    def evaluate(self, v_inf) -> ConstraintEvaluation:
        """Evaluate departure v_inf vectors (N, 3) [m/s] (numpy or cupy)."""
        xp = _xp_of(v_inf)
        v = xp.asarray(v_inf, dtype=float).reshape(-1, 3)
        if self.frame == "ecliptic":
            eps = math.radians(OBLIQUITY_J2000_DEG)
            z_eq = math.sin(eps) * v[:, 1] + math.cos(eps) * v[:, 2]
        else:
            z_eq = v[:, 2]
        vmag = xp.sqrt(xp.sum(v * v, axis=1))
        c3 = (vmag / 1000.0) ** 2
        safe = xp.where(vmag > 0, vmag, 1.0)
        dla = xp.degrees(xp.arcsin(xp.clip(z_eq / safe, -1.0, 1.0)))
        c3_excess = xp.maximum(c3 - self.max_c3, 0.0) if self.vehicle is not None else xp.zeros_like(c3)
        dla_excess = xp.maximum(xp.abs(dla) - self.max_abs_dla_deg, 0.0)
        feasible = (c3_excess <= 0.0) & (dla_excess <= 0.0)
        return ConstraintEvaluation(c3, dla, c3_excess, dla_excess, feasible)

    def apply(self, objective, v_inf):
        """Return ``objective`` (N,) with constraint handling applied.
        No-op when disabled. Candidates already at/above ``invalid_fitness``
        (e.g. failed Lambert solves) are left untouched."""
        if not self.enabled or (self.vehicle is None and self.site is None):
            return objective
        xp = _xp_of(objective)
        obj = xp.asarray(objective, dtype=float)
        ev = self.evaluate(v_inf)
        already_bad = obj >= self.invalid_fitness
        if self.mode == "invalid":
            # Graded sentinel: still >= invalid_fitness (clearly invalid), but
            # increasing with the violation so a population that starts fully
            # infeasible keeps a direction to search in.
            out = xp.where(ev.feasible, obj, self.invalid_fitness + ev.c3_excess + ev.dla_excess_deg)
        else:
            out = obj + self.penalty_weight * (ev.c3_excess + ev.dla_excess_deg)
        return xp.where(already_bad, obj, out)
