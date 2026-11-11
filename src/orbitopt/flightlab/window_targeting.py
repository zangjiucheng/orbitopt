"""FL-203: launch-window & RAAN targeting for Flight Lab ascents.

The bridge between a RAAN-constrained ``FlightTarget`` and the FL-201/202
ascent optimizer:

* :func:`derive_in_plane_liftoff` turns a RAAN-constrained target into a
  target whose liftoff epoch is the first in-plane node pass computed by
  :func:`orbitopt.launch_windows.plane_launch_windows` (the same planning
  model ``build_target`` uses -- linear GMST, J2 nodal regression at the
  mean orbit altitude), so the epoch the optimizer flies at is the epoch
  that makes the target plane reachable from the site;
* :func:`calibrate_epoch` flies two ±probe_s probe flights and shifts that
  epoch so the *achieved* RAAN (not just the geometric pass) lands on the
  target plane -- the geometric pass alone leaves a degree or two of
  ascent-vs-model plane tilt that attitude cannot pay down but the sidereal
  epoch sensitivity can;
* :func:`raan_report` reads the achieved RAAN of a flown run out of the
  Gate-B verdict -- the GCRS elements of ``checks.py``, never OpenTsiolkovsky's
  own launch-aligned number -- and reports the signed error against
  ``checks.RAAN_TOLERANCE_DEG``;
* :func:`window_sensitivity` re-flies a program at liftoff epochs offset by
  ±N seconds and reports the achieved RAAN drift, the window-sensitivity
  figure the task asks for.

Everything here reuses the two existing single sources of truth:
``orbitopt.launch_windows`` for windows, ``orbitopt.flightlab.checks`` for
grading. Nothing is re-implemented, and no tolerance is invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from orbitopt.flightlab import checks as _checks
from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, run_ascent
from orbitopt.flightlab.contract import FlightTarget
from orbitopt.flightlab import frames
from orbitopt.launch_sites import LaunchSite
from orbitopt.launch_windows import (
    SECONDS_PER_DAY,
    OrbitPlane,
    as_mjd2000,
    geocentric_latitude_deg,
    plane_launch_windows,
)
__all__ = ["LaunchEpochDerivation", "derive_in_plane_liftoff", "calibrate_epoch",
           "raan_report", "window_sensitivity"]

#: Nodal-regression window search horizon used when a target carries a RAAN
#: but no derived liftoff epoch (same knob as ``build_target``'s window_days).
DEFAULT_WINDOW_DAYS = 14.0

#: Earth rotation rate [deg/s] (sidereal) -- the expected slope of the
#: achieved-RAAN-vs-liftoff-epoch sensitivity curve.
EARTH_ROTATION_DEG_S = 360.0 / 86164.0905


@dataclass(frozen=True)
class LaunchEpochDerivation:
    """The derived liftoff epoch and the window it came from: ``target`` is
    the RAAN-constrained target rewritten to liftoff at the window optimum
    (mission epoch, target-orbit epoch/RAAN all consistent at that instant)."""

    target: FlightTarget
    window: Any               # orbitopt.launch_windows.LaunchWindow
    anchor_utc: str           # the epoch the search started from, ISO-Z
    requested_raan_deg: float
    epoch_raan_deg: float     # plane RAAN at the derived epoch (what is graded)


def signed_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """a - b wrapped to (-180, 180] -- the signed counterpart of
    ``checks.angle_diff_deg`` (which stays the pass/fail source of truth)."""
    d = (float(a_deg) - float(b_deg)) % 360.0
    return d - 360.0 if d > 180.0 else d


def site_for_target(target: FlightTarget) -> LaunchSite:
    """The target's launch site as a ``launch_windows.LaunchSite``: the
    packaged database record when the target's site name resolves to one,
    else an ad-hoc record from the target's own lat/lon/altitude with fully
    open azimuth corridors (a FlightTarget carries no corridor data)."""
    from orbitopt.launch_sites import get_launch_site

    try:
        return get_launch_site(target.launch_site.name)
    except KeyError:
        pass
    lat = float(target.launch_site.latitude_deg)
    gain = 465.1 * math.cos(math.radians(geocentric_latitude_deg(lat)))
    return LaunchSite(
        id=f"target-{target.launch_site.name.lower().replace(' ', '-')}",
        name=target.launch_site.name, latitude_deg=lat,
        longitude_deg=float(target.launch_site.longitude_deg),
        altitude_m=float(target.launch_site.altitude_wgs84_m),
        rotation_speed_gain_m_s=gain,
        azimuth_corridors_deg=((0.0, 360.0),), sources=())


def _iso_utc_z(mjd2000: float) -> str:
    from orbitopt.flightlab.target_export import _iso_utc_z as impl
    return impl(mjd2000)


def derive_in_plane_liftoff(target: FlightTarget, *,
                            window_days: float = DEFAULT_WINDOW_DAYS
                            ) -> LaunchEpochDerivation:
    """Rewrite a RAAN-constrained ``target`` to liftoff at its first in-plane
    node pass: the epoch search is exactly ``build_target``'s (same plane
    model -- J2 regression at the mean target altitude -- same window
    function, same corridor rules), applied to an already-built target.

    The returned target has ``mission.epoch_utc`` at the window optimum and
    ``target_orbit.raan_deg``/``epoch_utc`` = the plane RAAN at that instant,
    so everything downstream (OT launch time, the Gate-B RAAN check, the
    optimizer) sees one consistent epoch. Raises ValueError when no in-plane
    pass exists within ``window_days``. Idempotent: passing an already
    derived target yields the same epoch (a node pass is its own optimum
    within its own window to well under a second)."""
    if target.target_orbit.raan_deg is None:
        raise ValueError("derive_in_plane_liftoff needs a RAAN-constrained "
                         "target (target_orbit.raan_deg set); an unconstrained "
                         "plane liftoff at any epoch")
    if not window_days > 0.0:
        raise ValueError(f"window_days must be > 0, got {window_days}")
    anchor = as_mjd2000(target.target_orbit.epoch_utc or target.mission.epoch_utc)
    raan = float(target.target_orbit.raan_deg) % 360.0
    inc = float(target.target_orbit.inclination_deg)
    mean_alt_km = 0.5 * (float(target.target_orbit.periapsis_altitude_m)
                         + float(target.target_orbit.apoapsis_altitude_m)) / 1000.0
    plane = OrbitPlane.from_altitude(inc, raan, anchor, mean_alt_km)
    site = site_for_target(target)
    windows = plane_launch_windows(site, plane, anchor, window_days)
    if not windows:
        raise ValueError(
            f"no in-plane launch window from {site.name!r} into the plane "
            f"i={inc:g} deg, raan={raan:g} deg at {_iso_utc_z(anchor)} within "
            f"{window_days:g} days (corridor-constrained sites: the target "
            f"carries no azimuth corridors, so the ad-hoc site assumed open "
            f"corridors -- an empty result means the geometry, not the corridor)")
    window = min(windows, key=lambda w: w.optimal_mjd2000)
    epoch_mjd = window.optimal_mjd2000
    epoch_raan = round(float(plane.raan_at(epoch_mjd)) % 360.0, 3)
    if epoch_raan >= 360.0:  # rounding 359.999x up to the schema's exclusive maximum
        epoch_raan = 0.0
    epoch_iso = _iso_utc_z(epoch_mjd)
    derived = replace(
        target,
        mission=replace(target.mission, epoch_utc=epoch_iso),
        target_orbit=replace(target.target_orbit, raan_deg=epoch_raan,
                             epoch_utc=epoch_iso),
        assumptions=tuple(target.assumptions) + (
            f"FL-203: liftoff epoch {epoch_iso} = modelled {window.direction} "
            f"in-plane node pass at azimuth {window.azimuth_optimal_deg:.2f} deg, "
            f"first pass on or after {_iso_utc_z(anchor)} within {window_days:g} d "
            f"(orbitopt.launch_windows, J2 regression at {mean_alt_km:g} km mean "
            f"altitude); target raan_deg {epoch_raan:g} applies at that epoch",
        ),
    )
    return LaunchEpochDerivation(target=derived, window=window,
                                 anchor_utc=_iso_utc_z(anchor),
                                 requested_raan_deg=raan,
                                 epoch_raan_deg=epoch_raan)


def _plane_for(target: FlightTarget, raan_deg: float, anchor_mjd: float) -> OrbitPlane:
    inc = float(target.target_orbit.inclination_deg)
    mean_alt_km = 0.5 * (float(target.target_orbit.periapsis_altitude_m)
                         + float(target.target_orbit.apoapsis_altitude_m)) / 1000.0
    return OrbitPlane.from_altitude(inc, raan_deg, anchor_mjd, mean_alt_km)


def _wrap_epoch_raan(raan_at: float) -> float:
    v = round(float(raan_at) % 360.0, 3)
    return 0.0 if v >= 360.0 else v  # rounding 359.999x up past the schema max


def calibrate_epoch(derivation: LaunchEpochDerivation,
                    vehicle_path: str | Path,
                    program: Sequence[AttitudePoint] | None = None, *,
                    probe_s: float = 30.0, max_shift_s: float = 600.0,
                    timeout_s: float = 300.0
                    ) -> tuple[LaunchEpochDerivation, dict]:
    """Shift the derived liftoff epoch so the *achieved* RAAN, not just the
    geometric node pass, hits the target plane -- returned as a rewritten
    ``LaunchEpochDerivation`` plus a metadata dict.

    Why this exists: the window model treats the rocket as instantly riding
    the great-circle through the site, but a real ascent tilts the achieved
    plane by a degree or two of out-of-plane velocity (vertical rise +
    gravity turn). At a fixed epoch that tilt is a constant RAAN error no
    attitude change can pay down (azimuth trades RAAN against inclination
    too tightly for it); the *epoch* can: the sensitivity report shows the
    achieved RAAN moves at the sidereal rate with liftoff time. So fly two
    probe flights at ±``probe_s`` around the geometric pass, measure the
    slope, shift the epoch by -error/slope, and grade the shifted plane RAAN
    (the plane itself keeps regressing under J2 at the mean altitude, exactly
    as ``derive_in_plane_liftoff`` models). This is a schedule change, not a
    tolerance change: ``RAAN_TOLERANCE_DEG`` is never touched.

    The probe uses ``program`` (default: the vehicle's built-in program);
    the tilt is dominated by the mission's own ascent, so a tuned program
    starts near zero error from here and the optimizer keeps it in band.
    No shift (and a note in the metadata) when the measured slope is
    non-positive or the required shift exceeds ``max_shift_s``."""
    rows = window_sensitivity(derivation.target, vehicle_path, program,
                              (-probe_s, probe_s), refly_target_epoch=True,
                              timeout_s=timeout_s)
    e0 = float(rows[0]["error_deg"])
    slope = (float(rows[2]["achieved_raan_deg"])
             - float(rows[1]["achieved_raan_deg"])) / (2.0 * probe_s)
    meta: dict[str, Any] = {
        "probe_offset_s": float(probe_s),
        "geometric_pass_utc": derivation.target.mission.epoch_utc,
        "raan_error_at_pass_deg": e0,
        "measured_slope_deg_s": slope,
    }
    if slope <= 0.0:
        return derivation, {**meta, "correction_s": 0.0,
                            "note": "measured RAAN-vs-epoch slope is "
                                    "non-positive -- epoch left at the pass"}
    dt = -e0 / slope
    if abs(dt) > max_shift_s:
        return derivation, {**meta, "correction_s": 0.0,
                            "note": f"required epoch shift {dt:+.1f} s exceeds "
                                    f"the {max_shift_s:.0f} s cap -- epoch left "
                                    "at the geometric pass"}
    epoch0 = as_mjd2000(derivation.target.target_orbit.epoch_utc
                        or derivation.target.mission.epoch_utc)
    new_mjd = epoch0 + dt / SECONDS_PER_DAY
    plane = _plane_for(derivation.target, derivation.requested_raan_deg,
                       as_mjd2000(derivation.anchor_utc))
    epoch_raan = _wrap_epoch_raan(plane.raan_at(new_mjd))
    epoch_iso = _iso_utc_z(new_mjd)
    corrected = replace(
        derivation.target,
        mission=replace(derivation.target.mission, epoch_utc=epoch_iso),
        target_orbit=replace(derivation.target.target_orbit,
                             raan_deg=epoch_raan, epoch_utc=epoch_iso),
        assumptions=tuple(derivation.target.assumptions) + (
            f"FL-203: liftoff epoch corrected {dt:+.1f} s from the modelled "
            f"node pass ({meta['geometric_pass_utc']}) so the ACHIEVED plane "
            f"matches the target plane -- measured achieved-RAAN slope "
            f"{slope:.6f} deg/s from +/-{probe_s:g} s probe flights, RAAN error "
            f"at the pass was {e0:+.3f} deg (window model vs 3-DOF ascent "
            f"tilt); grading tolerance unchanged "
            f"(checks.RAAN_TOLERANCE_DEG = {_checks.RAAN_TOLERANCE_DEG} deg)",
        ),
    )
    meta = {**meta, "correction_s": dt, "corrected_epoch_utc": epoch_iso,
            "corrected_raan_deg": epoch_raan}
    return LaunchEpochDerivation(target=corrected, window=derivation.window,
                                 anchor_utc=derivation.anchor_utc,
                                 requested_raan_deg=derivation.requested_raan_deg,
                                 epoch_raan_deg=epoch_raan), meta


def raan_report(run: Any, target: FlightTarget,
                by_id: Mapping[str, Any] | None = None) -> dict:
    """The achieved-vs-target RAAN of a flown run, straight off the Gate-B
    ``insertion-raan-within-tolerance`` check (achieved RAAN = GCRS elements
    of the frame-bridged insertion state -- OpenTsiolkovsky's own RAAN is in
    its launch-aligned pseudo-inertial frame and is NOT comparable; it is
    reported separately as a diagnostic under ``ot_launch_aligned_raan_deg``).
    Raises ValueError when the target constrains no RAAN or the run was
    never graded against it (no check)."""
    if target.target_orbit.raan_deg is None:
        raise ValueError("raan_report needs a RAAN-constrained target")
    if by_id is None:
        by_id = {c.id: c for c in run.verification.checks}
    check = by_id.get("insertion-raan-within-tolerance")
    if check is None or check.value is None:
        raise ValueError(
            f"run {getattr(run, 'run_id', '?')!r} carries no graded "
            "insertion-raan-within-tolerance value -- the run was not graded "
            "against a RAAN-constrained target")
    error = signed_angle_diff_deg(float(check.value), float(target.target_orbit.raan_deg))
    report: dict[str, Any] = {
        "achieved_raan_deg": float(check.value),
        "target_raan_deg": float(target.target_orbit.raan_deg),
        "error_deg": error,
        "abs_error_deg": abs(error),
        "tolerance_deg": float(_checks.RAAN_TOLERANCE_DEG),
        "passed": bool(check.passed),
        "source": "orbitopt.flightlab.checks (GCRS elements of the frame-"
                  "bridged insertion state) -- NOT the OT launch-aligned RAAN",
    }
    diagnostic = _ot_launch_aligned_raan(run)
    if diagnostic is not None:
        report["ot_launch_aligned_raan_deg"] = diagnostic
    return report


def _ot_launch_aligned_raan(run: Any) -> float | None:
    """Diagnostic only: OT's own-style RAAN, i.e. the elements of the same
    insertion state expressed back in the launch-aligned pseudo-inertial
    frame the binary reports. Never used for pass/fail."""
    ins = _checks.insertion_state(run)
    r_l, v_l = frames.from_gcrs(
        [ins["position_m"]], [ins["velocity_m_s"]], "LAUNCH_ECI",
        _as_datetime(ins["epoch_utc"]), [ins["t_s"]])
    el = _checks.orbital_elements(r_l[0], v_l[0])
    return float(el["raan_deg"])


def _parse_utc(s: str):
    from orbitopt.flightlab.contract import _parse_utc as impl
    return impl(s, "track epoch_utc")


def _as_datetime(s: str):
    return _parse_utc(s)


def window_sensitivity(target: FlightTarget, vehicle_path: str | Path,
                       program: Sequence[AttitudePoint],
                       offsets_s: Sequence[float] = (-60.0, 60.0), *,
                       refly_target_epoch: bool = True,
                       timeout_s: float = 300.0) -> list[dict]:
    """Re-fly ``program`` with the liftoff epoch offset by each ±N seconds in
    ``offsets_s`` and report the achieved RAAN (Gate-B ``evaluate`` on the
    re-flown 3-DOF run -- no tudatpy re-fly; sensitivity is an ascent-grade
    figure) at every offset. ``refly_target_epoch=True`` prepends the
    zero-offset flight (the un-shifted epoch itself). The expected slope is
    ``EARTH_ROTATION_DEG_S`` deg/s: a liftoff that misses the node pass by
    dt puts the orbit plane dt of Earth rotation away from the target plane,
    so RAAN error grows roughly 0.25 deg per minute off-window."""
    if target.target_orbit.raan_deg is None:
        raise ValueError("window_sensitivity needs a RAAN-constrained target")
    epoch0 = as_mjd2000(target.target_orbit.epoch_utc or target.mission.epoch_utc)
    offsets = ([0.0] + list(map(float, offsets_s))) if refly_target_epoch \
        else list(map(float, offsets_s))
    out: list[dict] = []
    for off in offsets:
        shifted = replace(
            target,
            mission=replace(target.mission, epoch_utc=_iso_utc_z(epoch0 + off / SECONDS_PER_DAY)),
            target_orbit=replace(target.target_orbit,
                                 epoch_utc=_iso_utc_z(epoch0 + off / SECONDS_PER_DAY)))
        run = run_ascent(shifted, Path(vehicle_path), program,
                         Path(_scratch_dir()) / "run", timeout_s=timeout_s)
        verdict = _checks.evaluate(run, shifted)
        report = raan_report(run, shifted,
                             by_id={c.id: c for c in verdict.checks})
        out.append({"liftoff_offset_s": off,
                    "liftoff_epoch_utc": shifted.mission.epoch_utc,
                    "achieved_raan_deg": report["achieved_raan_deg"],
                    "error_deg": report["error_deg"],
                    "passed_within_tolerance": report["passed"]})
    return out


def _scratch_dir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="fl203-")
