"""Launch site + rocket -> parking orbit -> injection, for a departure
asymptote picked by an interplanetary search: the one launch-to-injection
step shared by the config-driven Mars transfer (``launch`` section of a
mission-mars-transfer-1.0 YAML, :func:`orbitopt.viz.mars_transfer.
build_from_config`) and ``examples/14_mars2020_replay.py``.

Given the heliocentric departure excess velocity of a Lambert transfer and
its departure epoch, this

1. converts it to the departure asymptote (C3, RLA, DLA, Earth equatorial),
2. finds that UTC day's launch windows from the site
   (:func:`orbitopt.launch_windows.asymptote_launch_windows`) and picks the
   one whose optimum is nearest ``preferred_liftoff_utc`` (or the first),
3. plans site -> parking orbit -> TMI (:func:`orbitopt.injection.plan_departure`)
   and verifies it numerically, with the vehicle C3 / site DLA checks
   (:func:`orbitopt.injection.verify_injection`).

Every epoch here is MJD2000 UTC; ISO strings / datetimes are converted with
:func:`orbitopt.launch_windows.as_mjd2000` and printed with
:func:`orbitopt.timescales.mjd2000_to_utc_datetime` -- the unified time API,
no ad-hoc conversions. Needs numpy/scipy only (no pykep / tudatpy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from orbitopt.injection import InjectionPlan, InjectionTarget, InjectionVerification, plan_departure, verify_injection
from orbitopt.launch_constraints import LaunchConstraints
from orbitopt.launch_windows import LaunchWindow, as_mjd2000, asymptote_from_v_inf, asymptote_launch_windows
from orbitopt.timescales import mjd2000_to_utc_datetime

# Defaults match examples/14_mars2020_replay.py.
DEFAULT_PARKING_ALTITUDE_KM = 185.0
DEFAULT_MIN_COAST_S = 600.0
DEFAULT_ASCENT_TIME_S = 600.0
DEFAULT_ASCENT_DOWNRANGE_DEG = 20.0
DEFAULT_WINDOW_DAYS = 1.0
DEFAULT_WINDOW_STEP_S = 30.0


@dataclass(frozen=True)
class LaunchSpec:
    """The ``launch`` section of a mission config: where, on what, how much,
    into which parking orbit."""

    site: str
    vehicle: Optional[str]
    payload_kg: float = 0.0
    constraint_mode: str = "invalid"
    parking_altitude_km: float = DEFAULT_PARKING_ALTITUDE_KM
    min_coast_s: float = DEFAULT_MIN_COAST_S
    ascent_time_s: float = DEFAULT_ASCENT_TIME_S
    ascent_downrange_deg: float = DEFAULT_ASCENT_DOWNRANGE_DEG
    window_days: float = DEFAULT_WINDOW_DAYS
    preferred_liftoff_mjd2000: Optional[float] = None
    # Pinned transfer (skip the window search): departure MJD2000 UTC + TOF.
    departure_mjd2000: Optional[float] = None
    tof_days: Optional[float] = None

    @classmethod
    def from_config(cls, section: Optional[dict], default_altitude_km: float = DEFAULT_PARKING_ALTITUDE_KM
                    ) -> Optional["LaunchSpec"]:
        """``LaunchSpec`` from a config ``launch`` mapping, or None when the
        section is absent or ``enabled: false`` (the default, unconstrained
        pipeline). ``default_altitude_km`` is used when the section has no
        ``parking_orbit.altitude_km`` (e.g. the config's top-level
        ``parking_orbit``)."""
        if not section or not section.get("enabled", True):
            return None
        parking = section.get("parking_orbit", {})
        departure = section.get("departure", {})
        window = section.get("window", {})
        return cls(
            site=section["site"],
            vehicle=section.get("vehicle"),
            payload_kg=float(section.get("payload_kg", 0.0)),
            constraint_mode=section.get("constraint_mode", "invalid"),
            parking_altitude_km=float(parking.get("altitude_km", default_altitude_km)),
            min_coast_s=float(parking.get("min_coast_s", DEFAULT_MIN_COAST_S)),
            ascent_time_s=float(parking.get("ascent_time_s", DEFAULT_ASCENT_TIME_S)),
            ascent_downrange_deg=float(parking.get("ascent_downrange_deg", DEFAULT_ASCENT_DOWNRANGE_DEG)),
            window_days=float(window.get("days", DEFAULT_WINDOW_DAYS)),
            preferred_liftoff_mjd2000=(as_mjd2000(window["preferred_liftoff_utc"])
                                       if "preferred_liftoff_utc" in window else None),
            departure_mjd2000=as_mjd2000(departure["utc"]) if "utc" in departure else None,
            tof_days=float(departure["tof_days"]) if "tof_days" in departure else None,
        )

    @property
    def pinned(self) -> bool:
        return self.departure_mjd2000 is not None and self.tof_days is not None

    def constraints(self, frame: str = "ecliptic") -> LaunchConstraints:
        """Vehicle C3 + site DLA constraint set (``frame`` of the v_inf it
        will be fed: "ecliptic" for the Lambert search, "equatorial" for
        :func:`verify_injection`)."""
        return LaunchConstraints.from_names(self.vehicle, self.payload_kg, self.site,
                                            mode=self.constraint_mode, frame=frame)


@dataclass
class LaunchInjection:
    """Result of :func:`launch_to_injection`."""

    spec: LaunchSpec
    c3_km2_s2: float
    rla_deg: float
    dla_deg: float
    windows: list[LaunchWindow]
    window: LaunchWindow
    plan: InjectionPlan
    verification: InjectionVerification

    @property
    def liftoff_mjd2000(self) -> float:
        return self.plan.launch.liftoff_mjd2000

    @property
    def report(self) -> dict:
        return self.verification.report

    def summary(self) -> dict:
        """JSON-ready digest (same fields example 14 prints)."""
        v, plan = self.verification, self.plan
        checks = v.report["status"]["checks"]
        return {
            "site": self.spec.site, "vehicle": self.spec.vehicle, "payload_kg": self.spec.payload_kg,
            "c3_km2_s2": round(self.c3_km2_s2, 3), "rla_deg": round(self.rla_deg, 2),
            "dla_deg": round(self.dla_deg, 2),
            "liftoff_utc": f"{mjd2000_to_utc_datetime(self.liftoff_mjd2000):%Y-%m-%dT%H:%M:%SZ}",
            "parking_inclination_deg": round(plan.orbit.inclination_deg, 2),
            "coast_min": round(plan.burn.coast_s / 60, 1), "tmi_dv_m_s": round(plan.burn.delta_v_mag_m_s, 1),
            "residual_before": round(v.planned_error, 2), "residual_after": round(v.final_error, 3),
            "residual_units": v.error_units,
            "checks_passed": f"{sum(checks.values())}/{len(checks)}",
            "passed": v.passed,
        }


def _pick_window(windows: list[LaunchWindow], preferred_mjd2000: Optional[float]) -> LaunchWindow:
    if preferred_mjd2000 is None:
        return windows[0]
    return min(windows, key=lambda w: abs(w.optimal_mjd2000 - preferred_mjd2000))


def launch_to_injection(
    spec: LaunchSpec,
    v_inf_ecliptic_m_s,
    departure_mjd2000: float,
    *,
    window_step_s: float = DEFAULT_WINDOW_STEP_S,
    report_path=None,
    propagator=None,
) -> LaunchInjection:
    """Launch windows, site -> parking orbit -> TMI plan and its numerical
    verification for a departure excess velocity (heliocentric J2000
    ecliptic, m/s) leaving on ``departure_mjd2000`` (UTC).

    Windows are searched over ``spec.window_days`` from 00:00 UTC of the
    departure day. Raises ``ValueError`` if the site has no window for the
    asymptote in that span."""
    v_inf_km_s = np.asarray(v_inf_ecliptic_m_s, dtype=float).reshape(3) / 1000.0
    rla, dla, c3 = asymptote_from_v_inf(v_inf_km_s, frame="ecliptic")
    day_start = float(math.floor(departure_mjd2000))
    windows = asymptote_launch_windows(spec.site, rla, dla, day_start, spec.window_days, None, window_step_s)
    if not windows:
        raise ValueError(
            f"no launch window from {spec.site} for RLA={rla:.2f} DLA={dla:.2f} deg within "
            f"{spec.window_days:g} day(s) of {mjd2000_to_utc_datetime(day_start):%Y-%m-%d} UTC")
    window = _pick_window(windows, spec.preferred_liftoff_mjd2000)
    target = InjectionTarget.from_c3_rla_dla(c3, rla, dla, name="TMI")
    plan = plan_departure(spec.site, target, liftoff=window.optimal_mjd2000,
                          altitude_km=spec.parking_altitude_km, ascent_time_s=spec.ascent_time_s,
                          ascent_downrange_deg=spec.ascent_downrange_deg, min_coast_s=spec.min_coast_s)
    plan.window = window
    verification = verify_injection(plan, launch_constraints=spec.constraints(frame="equatorial"),
                                    propagator=propagator, report_path=report_path)
    return LaunchInjection(spec=spec, c3_km2_s2=float(c3), rla_deg=float(rla), dla_deg=float(dla),
                           windows=windows, window=window, plan=plan, verification=verification)
