"""FlightTarget export (FL-003, #190): build a schema-valid FlightTarget v1
document from a launch site, a target orbit and a launch date/window --
``orbitopt flight export-target --site kourou --orbit 400x400@51.6
--date 2026-11-10 -o target.yaml``, or programmatically (E2/E5):

    from orbitopt.flightlab.target_export import build_target, save_target
    target = build_target("kourou", 400e3, 400e3, 51.6, date_utc="2026-11-10")
    save_target(target, "target.yaml")   # validates with flight-target-1.0.json

Everything the exported document claims is *derived or declared, never
invented*:

* the site block comes from the packaged launch-site database
  (:mod:`orbitopt.launch_sites`) with its cited sources;
* inclination feasibility (direct ascent, range-safety azimuth corridors)
  is checked before anything is written -- an inclination below |latitude|
  or outside every corridor-reachable band is rejected with a reason;
* when RAAN is constrained the liftoff epoch is *derived*, not typed: the
  first in-plane (node-pass) opportunity on or after the requested date
  within the search span, from :mod:`orbitopt.launch_windows` (J2 nodal
  regression of the target plane applied at the orbit's mean altitude), and
  the exported ``raan_deg`` is that plane's RAAN at the derived epoch;
* every field the call cannot derive (vehicle, backends, terminal state,
  recovery, the geoid/WGS-84 altitude subtlety, the window model's own
  fidelity) becomes an explicit ``assumptions`` entry instead of a made-up
  number, per the Flight Lab contract.

The generator's provenance (orbitopt version + git SHA of the checkout that
produced the document) is recorded as the first ``assumptions`` entry
(``build_target(..., record_generator=False)`` opts out, for golden tests).
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import yaml

from orbitopt.flightlab.contract import FlightTarget, validate_target
from orbitopt.launch_constraints import max_abs_dla_for_site
from orbitopt.launch_sites import LaunchSite, get_launch_site
from orbitopt.launch_windows import OrbitPlane, as_mjd2000, plane_launch_windows
from orbitopt.timescales import MJD2000_EPOCH_UTC, SECONDS_PER_DAY

#: Default forward search span [days] for the first feasible liftoff when
#: RAAN is constrained (one sidereal day of rotation misses a node pass when
#: the corridor admits only one pass/day, so the span covers a little more).
DEFAULT_WINDOW_DAYS = 2.0

#: The vehicle block of an export without a vehicle spec -- a placeholder the
#: consumer must replace (recorded as such in `assumptions`), never numbers.
UNSPECIFIED_VEHICLE = {"source": "unspecified:ascent-backend-default-vehicle",
                       "provenance": "illustrative"}

_ORBIT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)x([0-9]*\.?[0-9]+)@([0-9]*\.?[0-9]+)\s*$")


def parse_orbit_spec(spec: str) -> tuple[float, float, float]:
    """Parse ``<peri_km>x<apo_km>@<inc_deg>`` (e.g. ``400x400@51.6``) into
    ``(periapsis_m, apoapsis_m, inclination_deg)``. Raises ValueError with the
    expected shape on anything else, and on apoapsis below periapsis."""
    m = _ORBIT_RE.match(spec or "")
    if m is None:
        raise ValueError(f"invalid --orbit {spec!r}: expected <peri_km>x<apo_km>@<inc_deg>, e.g. 400x400@51.6")
    peri_m, apo_m, inc = float(m.group(1)) * 1000.0, float(m.group(2)) * 1000.0, float(m.group(3))
    if apo_m < peri_m:
        raise ValueError(f"--orbit {spec!r}: apoapsis {float(m.group(2)):g} km is below periapsis "
                         f"{float(m.group(1)):g} km -- swap them")
    if inc > 180.0:
        raise ValueError(f"--orbit {spec!r}: inclination {inc:g} deg is outside [0, 180]")
    return peri_m, apo_m, inc


def parse_window_spec(spec: str) -> tuple[str, float]:
    """Parse ``--window START[/DAYS]`` into ``(start_iso, days)``. START is any
    epoch :func:`orbitopt.launch_windows.as_mjd2000` accepts (ISO-8601 UTC,
    date-only = 00:00Z, or MJD2000 days); DAYS defaults to
    :data:`DEFAULT_WINDOW_DAYS`."""
    start, _, days_text = (spec or "").partition("/")
    if not start:
        raise ValueError(f"invalid --window {spec!r}: expected START[/DAYS], e.g. 2026-11-10/3")
    days = DEFAULT_WINDOW_DAYS if days_text == "" else float(days_text)
    if days <= 0.0:
        raise ValueError(f"invalid --window {spec!r}: DAYS must be > 0, got {days_text!r}")
    return start, days


# --------------------------------------------------------------------------
# feasibility
# --------------------------------------------------------------------------

def _inclination_infeasibility(site: LaunchSite, inclination_deg: float) -> str | None:
    """Why ``inclination_deg`` cannot be flown from ``site`` by direct ascent
    (no dogleg budget exists in the FlightTarget v1 contract), or None if the
    inclination is reachable. The latitude limit and the range-safety azimuth
    corridors are the two independent reasons; both cite the numbers behind
    them so the caller can fix the request instead of guessing."""
    phi = abs(site.latitude_deg)
    tol = 1e-6
    if inclination_deg < phi - tol:
        why = (f"below the site latitude {site.latitude_deg:+.4f} deg: a direct ascent cannot reach an "
               f"inclination lower than |latitude| (launching due east is the floor), and the "
               f"FlightTarget v1 contract has no dogleg budget")
    elif inclination_deg > 180.0 - phi + tol:
        why = (f"above the retrograde limit 180 - |latitude| = {180.0 - phi:.4f} deg of this site")
    elif not site.can_reach_inclination(inclination_deg):
        bands = "; ".join(f"{lo:.3f}..{hi:.3f}" for lo, hi in site.reachable_inclination_intervals())
        why = (f"outside every inclination band reachable inside the site's range-safety azimuth "
               f"corridors {[[a, b] for a, b in site.azimuth_corridors_deg]} (direct ascent reaches "
               f"i = {bands} deg there)")
    else:
        return None
    return (f"inclination {inclination_deg:g} deg is infeasible from launch site {site.id!r} "
            f"({site.name!r}): {why}")


def feasibility_report(site_id: str, inclination_deg: float) -> dict:
    """Azimuth/inclination feasibility summary for a site + inclination: the
    direct-ascent inclination bands the range-safety corridors admit, whether
    ``inclination_deg`` lies in one, and (from orbitopt.launch_constraints)
    the largest |DLA| any direct ascent from this site could ever deliver a
    departure asymptote to -- the interplanetary counterpart of the same
    azimuth geometry. Raises KeyError for an unknown site."""
    site = get_launch_site(site_id)
    return {
        "site": site.id,
        "latitude_deg": site.latitude_deg,
        "inclination_deg": float(inclination_deg),
        "reachable_inclination_intervals": [list(iv) for iv in site.reachable_inclination_intervals()],
        "feasible": site.can_reach_inclination(float(inclination_deg)),
        "max_abs_dla_deg": max_abs_dla_for_site(site),
        "azimuth_corridors_deg": [[a, b] for a, b in site.azimuth_corridors_deg],
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _git_sha() -> str:
    """Full HEAD commit of the checkout this module runs from, or
    'unknown' (installed wheel without git, git missing, shallow fetch)."""
    root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"


@lru_cache(maxsize=1)
def _orbitopt_version() -> str:
    try:
        return metadata.version("orbitopt")
    except metadata.PackageNotFoundError:
        return "unknown"


def generator_assumption() -> str:
    """The provenance line every export records: which orbitopt (version +
    git SHA of the code that generated the document) wrote it."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (f"generator: orbitopt {_orbitopt_version()} (git {_git_sha()}), "
            f"orbitopt.flightlab.target_export.build_target on {today}")


# --------------------------------------------------------------------------
# build / save
# --------------------------------------------------------------------------

def _iso_utc_z(mjd2000: float) -> str:
    """Whole-second ISO-8601 UTC with explicit Z (the contract's timestamp
    form) from an MJD2000 epoch, rounded to the nearest second."""
    dt = MJD2000_EPOCH_UTC + timedelta(seconds=round(mjd2000 * SECONDS_PER_DAY))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_target(site_id: str, periapsis_alt_m, apoapsis_alt_m, inclination_deg,
                 raan_deg=None, date_utc=None, constraints: dict | None = None,
                 recovery: dict | None = None, *,
                 window_days: float = DEFAULT_WINDOW_DAYS,
                 record_generator: bool = True) -> FlightTarget:
    """Build a validated, frozen FlightTarget from launch site + orbit +
    launch date (FL-003 interface, frozen for E2/E5).

    * ``site_id``: id/name/alias in the packaged launch-site database
      (KeyError with the known ids if absent).
    * ``raan_deg``: None = plane unconstrained (epoch used as-is); a number
      = RAAN at ``date_utc``, and the liftoff epoch becomes the first
      in-plane node pass on or after it within ``window_days`` days
      (ValueError with the reason when the corridor blocks every pass).
    * ``date_utc``: required -- a launch epoch cannot be invented; it comes
      from the caller (ISO-8601, date-only = 00:00Z, datetime or MJD2000).
    * ``constraints`` / ``recovery``: passed through verbatim (the schema
      checks the keys); absent = the matching assumption is recorded instead.
    * ``record_generator=False`` drops the generator provenance line (golden
      tests); real exports must keep it.

    Raises ValueError with a readable reason on infeasible inclination or
    apoapsis-below-periapsis, and jsonschema.ValidationError if the composed
    document somehow violates flight-target-1.0.json."""
    site = get_launch_site(site_id)
    peri = float(periapsis_alt_m)
    apo = float(apoapsis_alt_m)
    inc = float(inclination_deg)
    if peri < 0.0 or apo < 0.0:
        raise ValueError(f"orbit altitudes must be >= 0, got periapsis {peri} m / apoapsis {apo} m")
    if apo < peri:
        raise ValueError(f"apoapsis {apo} m is below periapsis {peri} m -- swap them or widen the orbit")
    reason = _inclination_infeasibility(site, inc)
    if reason is not None:
        raise ValueError(reason)
    if date_utc is None:
        raise ValueError("build_target needs date_utc: a launch epoch is never invented here "
                         "(give --date/--window on the CLI)")
    anchor = as_mjd2000(date_utc)
    if raan_deg is None:
        epoch_mjd, epoch_raan, window = anchor, None, None
    else:
        raan = float(raan_deg) % 360.0
        if not window_days > 0.0:
            raise ValueError(f"window_days must be > 0, got {window_days}")
        mean_alt_km = 0.5 * (peri + apo) / 1000.0
        plane = OrbitPlane.from_altitude(inc, raan, anchor, mean_alt_km)
        windows = plane_launch_windows(site, plane, anchor, window_days)
        if not windows:
            raise ValueError(
                f"no in-plane launch window from {site.id!r} into the plane i={inc:g} deg, "
                f"raan={raan:g} deg at {_iso_utc_z(anchor)} within {window_days:g} days: every node pass "
                f"requires a launch azimuth outside the site's range-safety corridors "
                f"{[[a, b] for a, b in site.azimuth_corridors_deg]} (feasible bands: "
                f"{'; '.join(f'{lo:.3f}..{hi:.3f}' for lo, hi in site.reachable_inclination_intervals())} deg)")
        window = min(windows, key=lambda w: w.optimal_mjd2000)
        epoch_mjd = window.optimal_mjd2000
        epoch_raan = round(float(plane.raan_at(epoch_mjd)) % 360.0, 3)
        if epoch_raan >= 360.0:  # rounding 359.999x up to the schema's exclusive maximum
            epoch_raan = 0.0

    peri_km, apo_km = peri / 1000.0, apo / 1000.0
    mission_id = f"{site.id}-{peri_km:g}x{apo_km:g}km-i{inc:g}"
    if epoch_raan is not None:
        mission_id += f"-raan{epoch_raan:g}"
    epoch_iso = _iso_utc_z(epoch_mjd)

    location_refs = sorted(set(site.sources_for("location")) | set(site.sources_for("altitude")))
    corridor_refs = sorted(set(site.sources_for("azimuth_corridors")))
    assumptions: list[str] = []
    if record_generator:
        assumptions.append(generator_assumption())
    assumptions.append(
        f"launch_site block (name, latitude, longitude, altitude) taken from the packaged "
        f"launch-site database entry {site.id!r}; sources: {'; '.join(location_refs)}")
    assumptions.append(
        "launch_site.altitude_wgs84_m uses the database ground elevation as WGS-84 ellipsoid "
        "height without geoid-undulation correction (< ~100 m at every catalogued site, "
        "irrelevant to ascent trajectory design)")
    assumptions.append(
        f"inclination {inc:g} deg checked feasible for direct ascent from {site.id!r} against its "
        f"range-safety azimuth corridors {[[a, b] for a, b in site.azimuth_corridors_deg]}; "
        f"sources: {'; '.join(corridor_refs)}")
    if window is not None:
        assumptions.append(
            f"liftoff epoch {epoch_iso} = modelled {window.direction} in-plane node pass at launch "
            f"azimuth {window.azimuth_optimal_deg:.2f} deg, first pass on or after {_iso_utc_z(anchor)} "
            f"within {window_days:g} d; target raan_deg {epoch_raan:g} applies at that epoch "
            f"(plane propagated at the J2 nodal-regression rate for {0.5 * (peri_km + apo_km):g} km mean altitude)")
        assumptions.append(
            "launch-window model is planning-grade: linear GMST (IAU-82), J2000 treated as the "
            "true-of-date equator/equinox, no precession/nutation, no ascent lead time "
            "(~1 min of window timing error by 2026) -- re-verify against an ephemeris before flight")
    else:
        assumptions.append(
            f"liftoff epoch {epoch_iso} exactly as requested; target plane unconstrained "
            "(raan_deg null): any corridor-admissible azimuth reaching this inclination yields "
            "a valid plane, so RAAN is set by the chosen epoch and ascent azimuth")
    assumptions.append(
        "vehicle block is a placeholder: no vehicle was specified at export time, the ascent "
        "backend's own default vehicle model applies and no performance figure here was verified")
    if not constraints:
        assumptions.append(
            "no mission constraints specified: the run uses backend defaults, no load/aero/"
            "insertion limits were asserted at export time")
    if not recovery:
        assumptions.append("no recovery plan specified: recovery.kind = none (expendable first stage)")
    assumptions.append(
        "no simulation backends/phases specified: the consumer (E2 ascent optimizer, Mission "
        "Control) selects them and records the choice in the run manifest")
    assumptions.append(
        "no terminal_state: the run is graded against the target orbit elements at insertion, "
        "not one fixed end-of-ascent state vector")

    doc: dict = {
        "schema_version": "1.0",
        "mission": {"id": mission_id,
                    "title": f"{peri_km:g}x{apo_km:g} km @ {inc:g} deg from {site.name}",
                    "epoch_utc": epoch_iso},
        "launch_site": {"name": site.name, "latitude_deg": site.latitude_deg,
                        "longitude_deg": site.longitude_deg, "altitude_wgs84_m": site.altitude_m},
        "target_orbit": {"central_body": "Earth", "periapsis_altitude_m": peri,
                         "apoapsis_altitude_m": apo, "inclination_deg": inc,
                         "raan_deg": epoch_raan},
        "vehicle": dict(UNSPECIFIED_VEHICLE),
        "assumptions": assumptions,
    }
    if epoch_raan is not None:
        doc["target_orbit"]["epoch_utc"] = epoch_iso
    if constraints:
        doc["constraints"] = dict(constraints)
    if recovery:
        doc["recovery"] = dict(recovery)
    validate_target(doc)
    return FlightTarget.from_dict(doc)


def save_target(target: FlightTarget, path) -> Path:
    """Validate a FlightTarget against flight-target-1.0.json, then write it
    as deterministic YAML (schema field order, no key sort). The file a
    Flight Lab run manifest will record as its target -- ``orbitopt flight
    validate <path>`` must accept it."""
    doc = target.to_dict()
    validate_target(doc)
    out = Path(path)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
