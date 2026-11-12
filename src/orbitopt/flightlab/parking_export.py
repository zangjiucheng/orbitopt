"""FL-203: hand a verified Flight Lab ascent to the patched-conic flows.

:func:`parking_orbit_from_insertion` reads the insertion (SECO) state out of
a Gate-B-verified :class:`~orbitopt.flightlab.contract.FlightRun` -- GCRS
state, the frame-bridged single source of truth -- and re-expresses it as the
:data:`~orbitopt.injection.ParkingOrbit` input type the existing orbitopt
flows consume (:mod:`orbitopt.injection` / :mod:`orbitopt.launch_injection`:
``plan_injection``, ``verify_injection``, ``parking_orbit_from_launch``'s
own return type). From there a Flight Lab ascent feeds a Mars/GEO/Artemis
plan exactly like a launch-window parking orbit does.

Mapping and its documented limit: a ``ParkingOrbit`` is *circular* by model,
so the achieved (slightly eccentric) insertion maps to the mean-altitude
circle at the insertion epoch carrying the GCRS inclination, RAAN and
argument of latitude; the true eccentricity stays in the returned ``elements``
dict for whoever needs it. The gate is honest: unless
``require_verified=False``, an insertion whose mission-constraint checks did
not pass raises -- a failed ascent is never laundered into a parking orbit.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from orbitopt.flightlab import checks as _checks
from orbitopt.injection import ParkingOrbit
from orbitopt.launch_windows import SECONDS_PER_DAY, as_mjd2000

__all__ = ["AscentHandoff", "parking_orbit_from_insertion"]


@dataclass(frozen=True)
class AscentHandoff:
    """Everything a downstream flow needs from an ascent: the circular
    parking-orbit model (``orbit``, the ``injection``/``launch_injection``
    input type), the *as-flown* GCRS elements and state at insertion (the
    truth the circle approximates), and the verification gate it passed."""

    orbit: ParkingOrbit
    elements: dict[str, float]
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    epoch_mjd2000: float
    epoch_utc: str
    verification: dict[str, Any]

    def to_dict(self) -> dict:
        return {"kind": "orbitopt.flightlab.ascent-handoff",
                "schema_version": "1.0",
                "orbit": self.orbit.to_dict(),
                "elements": dict(self.elements),
                "insertion_state": {"frame": "GCRS", "epoch_utc": self.epoch_utc,
                                    "epoch_mjd2000": self.epoch_mjd2000,
                                    "position_m": list(self.position_m),
                                    "velocity_m_s": list(self.velocity_m_s)},
                "verification": dict(self.verification)}

    def save(self, path) -> Any:
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        return out


def _argument_of_latitude(r_hat: Sequence[float], inc_deg: float,
                          raan_deg: float) -> float:
    """Angle from the ascending node along the orbit to the position vector
    (the ``ParkingOrbit.arg_latitude_deg`` convention; mirrors
    ``injection._argument_of_latitude`` without crossing the private border)."""
    n = np.array([math.cos(math.radians(raan_deg)),
                  math.sin(math.radians(raan_deg)), 0.0])
    h = np.array([math.sin(math.radians(inc_deg)) * math.sin(math.radians(raan_deg)),
                  -math.sin(math.radians(inc_deg)) * math.cos(math.radians(raan_deg)),
                  math.cos(math.radians(inc_deg))])
    m = np.cross(h, n)
    r = np.asarray(r_hat, dtype=float)
    return math.degrees(math.atan2(float(np.dot(r, m)), float(np.dot(r, n)))) % 360.0


def parking_orbit_from_insertion(run: Any, *, require_verified: bool = True,
                                 j2_secular: bool = True) -> AscentHandoff:
    """The run's insertion state (GCRS, via :func:`checks.insertion_state` --
    never the backend's own frame/RAAN) as an :class:`AscentHandoff`.
    ``require_verified`` gates on the run's own Gate-B verdict:
    ``mission_constraints_passed`` must be True. The orbit epoch is the
    insertion instant (mission epoch + SECO time)."""
    if require_verified and not bool(run.verification.mission_constraints_passed):
        failed = [c.id for c in run.verification.checks if not c.passed]
        raise ValueError(
            f"run {run.run_id!r} did not pass its mission constraints "
            f"(failing checks: {', '.join(failed) or 'none -- not assessed'}) -- "
            "a failed ascent is not a parking orbit (pass require_verified=False "
            "to export the state anyway)")
    ins = _checks.insertion_state(run)
    r = np.asarray(ins["position_m"], dtype=float)
    v = np.asarray(ins["velocity_m_s"], dtype=float)
    el = _checks.orbital_elements(r, v)
    alt_mean_m = 0.5 * (float(el["periapsis_altitude_m"])
                        + float(el["apoapsis_altitude_m"]))
    epoch_mjd2000 = (as_mjd2000(_parse_utc(ins["epoch_utc"]))
                     + float(ins["t_s"]) / SECONDS_PER_DAY)
    orbit = ParkingOrbit(
        altitude_km=alt_mean_m / 1000.0,
        inclination_deg=float(el["inclination_deg"]),
        raan_deg=float(el["raan_deg"]),
        epoch_mjd2000=epoch_mjd2000,
        arg_latitude_deg=_argument_of_latitude(
            r / np.linalg.norm(r), float(el["inclination_deg"]),
            float(el["raan_deg"])),
        j2_secular=j2_secular)
    verdict = run.verification
    return AscentHandoff(
        orbit=orbit,
        elements={k: float(val) for k, val in el.items()},
        position_m=tuple(float(x) for x in r),
        velocity_m_s=tuple(float(x) for x in v),
        epoch_mjd2000=epoch_mjd2000,
        epoch_utc=ins["epoch_utc"],
        verification={
            "run_id": run.run_id,
            "optimizer_converged": verdict.optimizer_converged,
            "physics_verified": verdict.physics_verified,
            "mission_constraints_passed": verdict.mission_constraints_passed,
            "checks": [{"id": c.id, "passed": bool(c.passed)} for c in verdict.checks],
        })


def _parse_utc(s: str):
    from orbitopt.flightlab.contract import _parse_utc as impl
    return impl(s, "insertion track epoch_utc")
