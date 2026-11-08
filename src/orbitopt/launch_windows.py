"""Daily launch windows from a launch site into a target orbit plane, and the
link from porkchop departure dates to launchable dates/times.

Two kinds of target are supported:

* **A fixed (or precessing) orbit plane** -- inclination + RAAN, e.g. the ISS
  or a sun-synchronous plane. The site is carried through the plane by Earth
  rotation twice a sidereal day (once northbound, once southbound) when
  ``|latitude| <= inclination``; those are the *in-plane* launch instants.
  With a plane-change budget ``max_dogleg_deg`` the window opens/closes when
  the site's angular distance from the plane equals that budget (a *dogleg*
  ascent removes the residual wedge angle). Sites whose latitude exceeds the
  inclination get dogleg-only windows if the budget covers ``|phi| - i``.

* **An interplanetary departure asymptote** (right ascension RLA and
  declination DLA of V-infinity, see :func:`asymptote_from_v_inf`). Any plane
  through the site and the asymptote direction works, so the window is the
  span of the day over which that plane's launch azimuth lies inside the
  allowed azimuth range (the site's range-safety corridors by default).

Every window reports open / close / optimal instants (MJD2000, UTC) and the
launch azimuth at each. Azimuths are degrees clockwise from true north and
are the azimuth of the plane through the site closest to the target plane
(equal to the in-plane azimuth ``sin Az = cos i / cos phi`` at an in-plane
instant).

Model fidelity: spherical Earth direction vectors (geocentric latitude), GMST
from the IAU-1982 linear term, no precession/nutation (the J2000 equatorial
frame is treated as the true-of-date frame -- ~0.3 deg of RA by 2026, i.e.
~1 min in window timing) and no ascent-duration lead time. Planning-level:
windows agree with flown missions to a few minutes, see
tests/test_launch_windows.py. Needs only numpy (pykep optional, for the
porkchop link only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

import numpy as np

from orbitopt.launch_sites import LaunchSite, get_launch_site

MJD2000_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
SECONDS_PER_DAY = 86400.0
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
J2_EARTH = 1.08262668e-3
OBLIQUITY_J2000_DEG = 23.4392911
# Rate of the mean sun in right ascension [deg/day]; a sun-synchronous plane
# precesses at this rate.
MEAN_SUN_RATE_DEG_DAY = 0.98564736
_WGS84_E2 = 6.69437999014e-3
# |n.s| below this counts as "in plane" (~0.02 arcsec, ~1 ms of rotation).
_IN_PLANE_EPS = 1e-7
_TIME_TOL_DAYS = 0.05 / SECONDS_PER_DAY
# Extra scan span either side of the requested interval (windows that open
# inside it are reported whole even if they close after it).
_SCAN_MARGIN_DAYS = 0.5


# --------------------------------------------------------------------------
# Time and frames
# --------------------------------------------------------------------------

def datetime_to_mjd2000(dt: datetime) -> float:
    """UTC datetime (naive = UTC) -> MJD2000 days (0 = 2000-01-01 00:00 UTC,
    the pykep convention)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - MJD2000_EPOCH).total_seconds() / SECONDS_PER_DAY


def mjd2000_to_datetime(mjd2000: float) -> datetime:
    return MJD2000_EPOCH + timedelta(seconds=round(float(mjd2000) * SECONDS_PER_DAY, 3))


def as_mjd2000(t) -> float:
    """MJD2000 from an MJD2000 number, UTC datetime or ISO-8601 string."""
    if isinstance(t, datetime):
        return datetime_to_mjd2000(t)
    if isinstance(t, str):
        return datetime_to_mjd2000(datetime.fromisoformat(t.replace("Z", "+00:00")))
    return float(t)


def gmst_deg(mjd2000):
    """Greenwich mean sidereal time [deg] at UT1 ~= UTC (IAU 1982, linear
    term; the quadratic term is < 0.01 s per century)."""
    d = np.asarray(mjd2000, dtype=float) - 0.5  # days since J2000.0 (12:00)
    return np.mod(280.46061837 + 360.98564736629 * d, 360.0)


def mean_sun_ra_deg(mjd2000):
    """Right ascension of the fictitious mean sun [deg] (= sun mean longitude)."""
    d = np.asarray(mjd2000, dtype=float) - 0.5
    return np.mod(280.460 + MEAN_SUN_RATE_DEG_DAY * d, 360.0)


def geocentric_latitude_deg(geodetic_deg: float) -> float:
    return math.degrees(math.atan((1.0 - _WGS84_E2) * math.tan(math.radians(geodetic_deg))))


def _site_vectors(site: LaunchSite, mjd2000):
    """Inertial unit vectors (position s, local east e, local north n) of the
    site at the given epochs; each (N, 3)."""
    t = np.atleast_1d(np.asarray(mjd2000, dtype=float))
    lat = math.radians(geocentric_latitude_deg(site.latitude_deg))
    lst = np.radians(gmst_deg(t) + site.longitude_deg)
    cl, sl = math.cos(lat), math.sin(lat)
    ca, sa = np.cos(lst), np.sin(lst)
    s = np.stack([cl * ca, cl * sa, np.full_like(ca, sl)], axis=-1)
    e = np.stack([-sa, ca, np.zeros_like(ca)], axis=-1)
    n = np.stack([-sl * ca, -sl * sa, np.full_like(ca, cl)], axis=-1)
    return s, e, n


def _azimuth_of_motion(normal, s, e, n):
    """Launch azimuth [deg] of motion along the plane with unit normal(s)
    ``normal`` at site position ``s`` (direction of motion = normal x s)."""
    v = np.cross(normal, s)
    return np.mod(np.degrees(np.arctan2(np.sum(v * e, -1), np.sum(v * n, -1))), 360.0)


def _direction(azimuth_deg: float) -> str:
    c = math.cos(math.radians(azimuth_deg))
    if abs(c) < 1e-4:  # within ~0.006 deg of due east/west
        return "east" if math.sin(math.radians(azimuth_deg)) > 0 else "west"
    return "northbound" if c > 0 else "southbound"


def azimuth_allowed(azimuth_deg: float, corridors: Sequence[Sequence[float]]) -> bool:
    """True if the azimuth lies in any clockwise [start, end] corridor."""
    return bool(_azimuth_mask(np.array([azimuth_deg]), corridors)[0])


def _azimuth_mask(azimuth_deg, corridors: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorized :func:`azimuth_allowed` (NaN azimuths are never allowed)."""
    az = np.asarray(azimuth_deg, dtype=float)
    mask = np.zeros(az.shape, dtype=bool)
    for start, end in corridors:
        sweep = (end - start) % 360.0
        if sweep == 0.0 and end != start:
            sweep = 360.0
        mask |= np.mod(az - start, 360.0) <= sweep + 1e-9
    return mask & ~np.isnan(az)


# --------------------------------------------------------------------------
# Target planes
# --------------------------------------------------------------------------

def j2_raan_rate_deg_day(a_km: float, inclination_deg: float, e: float = 0.0) -> float:
    """Secular nodal regression dOmega/dt [deg/day] due to Earth J2."""
    n = math.sqrt(MU_EARTH_KM3_S2 / a_km**3)
    p = a_km * (1.0 - e * e)
    rate = -1.5 * n * J2_EARTH * (R_EARTH_KM / p) ** 2 * math.cos(math.radians(inclination_deg))
    return math.degrees(rate) * SECONDS_PER_DAY


@dataclass(frozen=True)
class OrbitPlane:
    """Target orbit plane in the (J2000 ~ of-date) equatorial frame.

    ``raan_deg`` holds at ``epoch_mjd2000`` and drifts linearly at
    ``raan_rate_deg_day`` (e.g. :func:`j2_raan_rate_deg_day`)."""
    inclination_deg: float
    raan_deg: float = 0.0
    epoch_mjd2000: float = 0.0
    raan_rate_deg_day: float = 0.0
    name: str = ""

    @classmethod
    def from_altitude(cls, inclination_deg: float, raan_deg: float, epoch, altitude_km: float,
                      name: str = "") -> "OrbitPlane":
        """Circular orbit plane whose RAAN regresses at the J2 rate."""
        rate = j2_raan_rate_deg_day(R_EARTH_KM + altitude_km, inclination_deg)
        return cls(inclination_deg, raan_deg, as_mjd2000(epoch), rate, name)

    def raan_at(self, mjd2000):
        return self.raan_deg + self.raan_rate_deg_day * (np.asarray(mjd2000, dtype=float) - self.epoch_mjd2000)

    def normal_at(self, mjd2000) -> np.ndarray:
        """Unit angular-momentum vector(s), shape (N, 3)."""
        t = np.atleast_1d(np.asarray(mjd2000, dtype=float))
        i = math.radians(self.inclination_deg)
        raan = np.radians(self.raan_at(t))
        return np.stack([math.sin(i) * np.sin(raan), -math.sin(i) * np.cos(raan),
                         np.full_like(raan, math.cos(i))], axis=-1)


def sun_synchronous_plane(ltan_hours: float, epoch, inclination_deg: float,
                          name: str = "") -> OrbitPlane:
    """Sun-synchronous plane with mean local time of the ascending node
    ``ltan_hours`` (a 10:30 descending node = 22.5 LTAN)."""
    t = as_mjd2000(epoch)
    raan = float(mean_sun_ra_deg(t)) + (ltan_hours - 12.0) * 15.0
    return OrbitPlane(inclination_deg, raan % 360.0, t, MEAN_SUN_RATE_DEG_DAY, name)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchWindow:
    site_id: str
    kind: str                 # "in-plane", "dogleg" or "asymptote"
    open_mjd2000: float
    close_mjd2000: float
    optimal_mjd2000: float
    azimuth_open_deg: float
    azimuth_close_deg: float
    azimuth_optimal_deg: float
    inclination_deg: float    # inclination of the launch plane at the optimum
    dogleg_deg: float = 0.0   # plane change still needed at the optimum
    direction: str = ""       # "northbound" / "southbound" / "east" / "west" at the optimum

    @property
    def open_utc(self) -> datetime:
        return mjd2000_to_datetime(self.open_mjd2000)

    @property
    def close_utc(self) -> datetime:
        return mjd2000_to_datetime(self.close_mjd2000)

    @property
    def optimal_utc(self) -> datetime:
        return mjd2000_to_datetime(self.optimal_mjd2000)

    @property
    def duration_s(self) -> float:
        return (self.close_mjd2000 - self.open_mjd2000) * SECONDS_PER_DAY

    def to_dict(self) -> dict:
        iso = lambda d: d.isoformat(timespec="seconds").replace("+00:00", "Z")  # noqa: E731
        return {
            "site": self.site_id, "kind": self.kind, "direction": self.direction,
            "open_utc": iso(self.open_utc), "close_utc": iso(self.close_utc),
            "optimal_utc": iso(self.optimal_utc), "duration_s": round(self.duration_s, 1),
            "azimuth_open_deg": round(self.azimuth_open_deg, 3),
            "azimuth_close_deg": round(self.azimuth_close_deg, 3),
            "azimuth_optimal_deg": round(self.azimuth_optimal_deg, 3),
            "inclination_deg": round(self.inclination_deg, 3),
            "dogleg_deg": round(self.dogleg_deg, 4),
        }


def _bisect(fn: Callable[[float], bool], t_true: float, t_false: float) -> float:
    """Boundary between a point where fn is True and one where it is False."""
    while abs(t_false - t_true) > _TIME_TOL_DAYS:
        mid = 0.5 * (t_true + t_false)
        if fn(mid):
            t_true = mid
        else:
            t_false = mid
    return t_true


def _root(fn: Callable[[float], float], a: float, b: float, fa: float) -> float:
    for _ in range(80):
        if b - a <= _TIME_TOL_DAYS * 1e-2:
            break
        m = 0.5 * (a + b)
        fm = fn(m)
        if fm == 0.0:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _golden_min(fn: Callable[[float], float], a: float, b: float) -> float:
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fn(c), fn(d)
    while b - a > _TIME_TOL_DAYS:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def _scan(t0: float, t1: float, step_days: float, ok, seeds: list[float],
          score) -> list[tuple[float, float, float]]:
    """Maximal intervals where ``ok`` holds that *open* in [t0, t1). The scan
    runs over [t0 - _SCAN_MARGIN_DAYS, t1 + _SCAN_MARGIN_DAYS] so a window
    straddling midnight is reported whole, once. Intervals are found on a
    sample grid augmented with ``seeds`` (instants that may be narrower than
    the grid), boundaries refined by bisection. ``ok`` and ``score`` are vectorized
    (epoch array -> array). Returns (open, close, optimal); the optimum is the
    seed in the interval minimizing ``score``, else the best sample refined by
    golden-section search."""
    ok1 = lambda t: bool(ok(np.array([t]))[0])  # noqa: E731
    score1 = lambda t: float(score(np.array([t]))[0])  # noqa: E731
    lo, hi = t0 - _SCAN_MARGIN_DAYS, t1 + _SCAN_MARGIN_DAYS
    n = max(int(math.ceil((hi - lo) / step_days)), 1)
    ts = np.unique(np.concatenate([np.linspace(lo, hi, n + 1), np.asarray(seeds, dtype=float)]))
    flags = ok(ts)
    seed_set = set(float(s) for s in seeds)
    out = []
    k = 0
    while k < len(ts):
        if not flags[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(ts) and flags[j + 1]:
            j += 1
        t_open = float(ts[k]) if k == 0 else _bisect(ok1, float(ts[k]), float(ts[k - 1]))
        t_close = float(ts[j]) if j == len(ts) - 1 else _bisect(ok1, float(ts[j]), float(ts[j + 1]))
        inside = ts[k:j + 1]
        seeded = [float(t) for t in inside if float(t) in seed_set]
        if seeded:
            t_opt = min(seeded, key=score1)
        else:
            t_best = float(inside[int(np.argmin(score(inside)))])
            t_opt = _golden_min(score1, max(t_open, t_best - step_days), min(t_close, t_best + step_days))
        if t0 <= t_open < t1 and k > 0 and j < len(ts) - 1:
            out.append((t_open, t_close, t_opt))
        k = j + 1
    return out


def plane_launch_windows(
    site: LaunchSite | str,
    plane: OrbitPlane,
    start,
    days: float = 1.0,
    max_dogleg_deg: float = 0.0,
    respect_corridors: bool = True,
    azimuth_range_deg: Sequence[Sequence[float]] | None = None,
    step_s: float = 60.0,
) -> list[LaunchWindow]:
    """Launch windows from ``site`` into ``plane`` opening in [start, start+days)
    (``start``: MJD2000, datetime or ISO-8601 string, UTC).

    ``max_dogleg_deg = 0`` gives the instantaneous in-plane opportunities
    (open == close); a positive budget widens each into a window bounded by
    that out-of-plane angle. Windows whose launch azimuth leaves the allowed
    azimuths (``azimuth_range_deg`` corridors, else the site's range-safety
    corridors when ``respect_corridors``) are clipped/dropped.
    """
    site = get_launch_site(site) if isinstance(site, str) else site
    t0 = as_mjd2000(start)
    t1 = t0 + days
    step = step_s / SECONDS_PER_DAY
    corridors = azimuth_range_deg if azimuth_range_deg is not None else (
        site.azimuth_corridors_deg if respect_corridors else None)
    c = max(math.sin(math.radians(max_dogleg_deg)), _IN_PLANE_EPS)

    def offplane(t):  # n.s = sin(angular distance of the site from the plane)
        sv, _, _ = _site_vectors(site, t)
        return np.sum(plane.normal_at(t) * sv, -1)

    def geometry(t):
        """(azimuth, inclination) arrays of the plane through the site
        closest to the target plane."""
        sv, e, n = _site_vectors(site, t)
        h = plane.normal_at(t)
        hp = h - np.sum(h * sv, -1)[:, None] * sv
        norm = np.linalg.norm(hp, axis=-1)
        hp = hp / np.where(norm < 1e-12, np.nan, norm)[:, None]  # NaN: site at the plane's pole
        az = _azimuth_of_motion(hp, sv, e, n)
        inc = np.degrees(np.arccos(np.clip(hp[:, 2], -1.0, 1.0)))
        return az, inc

    def ok(t):
        good = np.abs(offplane(t)) <= c
        if corridors is not None:
            good &= _azimuth_mask(geometry(t)[0], corridors)
        return good

    def score(t):
        return np.abs(offplane(t))

    def score1(t: float) -> float:
        return float(score(np.array([t]))[0])

    def f1(t: float) -> float:
        return float(offplane(np.array([t]))[0])

    # Seeds: exact in-plane roots and local minima of |n.s| (dogleg-only windows).
    lo, hi = t0 - _SCAN_MARGIN_DAYS, t1 + _SCAN_MARGIN_DAYS
    ts = np.linspace(lo, hi, max(int(math.ceil((hi - lo) / step)), 1) + 1)
    f = offplane(ts)
    seeds = [float(ts[k]) for k in np.flatnonzero(f == 0.0)]
    for k in np.flatnonzero(f[:-1] * f[1:] < 0):
        seeds.append(_root(f1, float(ts[k]), float(ts[k + 1]), float(f[k])))
    af = np.abs(f)
    for k in np.flatnonzero((af[1:-1] <= af[:-2]) & (af[1:-1] <= af[2:]) & (f[:-2] * f[2:] > 0)) + 1:
        seeds.append(_golden_min(score1, float(ts[k - 1]), float(ts[k + 1])))

    windows = []
    for t_open, t_close, t_opt in _scan(t0, t1, step, ok, seeds, score):
        az, inc = geometry(np.array([t_open, t_close, t_opt]))
        dog = math.degrees(math.asin(min(1.0, score1(t_opt))))
        windows.append(LaunchWindow(
            site_id=site.id, kind="in-plane" if dog < 1e-3 else "dogleg",
            open_mjd2000=t_open, close_mjd2000=t_close, optimal_mjd2000=t_opt,
            azimuth_open_deg=float(az[0]), azimuth_close_deg=float(az[1]),
            azimuth_optimal_deg=float(az[2]), inclination_deg=float(inc[2]), dogleg_deg=dog,
            direction=_direction(float(az[2])),
        ))
    return windows


# --------------------------------------------------------------------------
# Interplanetary departures (porkchop link)
# --------------------------------------------------------------------------

def asymptote_from_v_inf(v_inf, frame: str = "ecliptic") -> tuple[float, float, float]:
    """(RLA deg, DLA deg, C3 km^2/s^2) of a departure excess-velocity vector
    [km/s]. ``frame``: "ecliptic" (heliocentric ecliptic J2000, pykep's
    jpl_lp frame) or "equatorial"."""
    v = np.asarray(v_inf, dtype=float).reshape(3)
    if frame == "ecliptic":
        eps = math.radians(OBLIQUITY_J2000_DEG)
        v = np.array([v[0], math.cos(eps) * v[1] - math.sin(eps) * v[2],
                      math.sin(eps) * v[1] + math.cos(eps) * v[2]])
    elif frame != "equatorial":
        raise ValueError(f"unknown frame {frame!r}")
    vmag = float(np.linalg.norm(v))
    if vmag == 0.0:
        raise ValueError("zero V-infinity has no asymptote direction")
    rla = math.degrees(math.atan2(v[1], v[0])) % 360.0
    dla = math.degrees(math.asin(v[2] / vmag))
    return rla, dla, vmag**2


def asymptote_launch_windows(
    site: LaunchSite | str,
    rla_deg: float,
    dla_deg: float,
    start,
    days: float = 1.0,
    azimuth_range_deg: Sequence[Sequence[float]] | None = None,
    step_s: float = 60.0,
) -> list[LaunchWindow]:
    """Daily windows for a departure along asymptote (RLA, DLA): the instants
    at which the prograde plane through the site and the asymptote has a
    launch azimuth inside ``azimuth_range_deg`` (default: the site's
    corridors). The optimum is the most easterly azimuth (least inclination,
    most rotation gain)."""
    site = get_launch_site(site) if isinstance(site, str) else site
    t0 = as_mjd2000(start)
    corridors = azimuth_range_deg if azimuth_range_deg is not None else site.azimuth_corridors_deg
    ra, de = math.radians(rla_deg), math.radians(dla_deg)
    u = np.array([math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)])

    def geometry(t):
        """(azimuth, inclination) arrays of the prograde plane through the
        site and the asymptote (NaN if degenerate)."""
        sv, e, n = _site_vectors(site, t)
        h = np.cross(sv, u)
        norm = np.linalg.norm(h, axis=-1)
        h = h / np.where(norm < 1e-9, np.nan, norm)[:, None]
        h = np.where(h[:, 2:3] < 0, -h, h)  # prograde orientation
        az = _azimuth_of_motion(h, sv, e, n)
        inc = np.degrees(np.arccos(np.clip(h[:, 2], -1.0, 1.0)))
        return az, inc

    def ok(t):
        return _azimuth_mask(geometry(t)[0], corridors)

    def score(t):  # distance of the azimuth from due east
        az = geometry(t)[0]
        return np.where(np.isnan(az), 1e9, np.abs(np.mod(az - 90.0 + 180.0, 360.0) - 180.0))

    windows = []
    for t_open, t_close, t_opt in _scan(t0, t0 + days, step_s / SECONDS_PER_DAY, ok, [], score):
        az, inc = geometry(np.array([t_open, t_close, t_opt]))
        windows.append(LaunchWindow(
            site_id=site.id, kind="asymptote",
            open_mjd2000=t_open, close_mjd2000=t_close, optimal_mjd2000=t_opt,
            azimuth_open_deg=float(az[0]), azimuth_close_deg=float(az[1]),
            azimuth_optimal_deg=float(az[2]), inclination_deg=float(inc[2]),
            direction=_direction(float(az[2])),
        ))
    return windows


@dataclass
class DepartureDay:
    """Launch opportunities on one porkchop departure date."""
    t0_mjd2000: float
    tof_days: float
    c3_km2_s2: float
    rla_deg: float
    dla_deg: float
    windows: list[LaunchWindow] = field(default_factory=list)

    @property
    def launchable(self) -> bool:
        return bool(self.windows)


def porkchop_launch_windows(
    grid,
    site: LaunchSite | str,
    azimuth_range_deg: Sequence[Sequence[float]] | None = None,
    max_c3: float | None = None,
    step_s: float = 60.0,
) -> list[DepartureDay]:
    """For every departure date of a porkchop grid (``compute_porkchop``
    result carrying ``v_inf_departure``), take the minimum-C3 converged
    transfer, derive its asymptote and compute that UTC day's launch windows
    from ``site``. Columns with no converged cell, or whose minimum C3
    exceeds ``max_c3``, get no windows."""
    if getattr(grid, "v_inf_departure", None) is None:
        raise ValueError("porkchop grid has no v_inf_departure vectors -- recompute with compute_porkchop()")
    site = get_launch_site(site) if isinstance(site, str) else site
    c3 = np.where(grid.converged, grid.c3_departure, np.inf)
    days = []
    for j, t0 in enumerate(grid.t0_grid):
        i = int(np.argmin(c3[:, j]))
        c3_ij = float(c3[i, j])
        if not np.isfinite(c3_ij):
            days.append(DepartureDay(float(t0), float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        rla, dla, _ = asymptote_from_v_inf(grid.v_inf_departure[i, j], frame="ecliptic")
        day = DepartureDay(float(t0), float(grid.tof_grid[i]), c3_ij, rla, dla)
        if max_c3 is None or c3_ij <= max_c3:
            day.windows = asymptote_launch_windows(
                site, rla, dla, math.floor(t0), 1.0, azimuth_range_deg, step_s)
        days.append(day)
    return days


def format_windows(windows: Sequence[LaunchWindow]) -> str:
    """Human-readable table of windows."""
    lines = [f"{'open (UTC)':<20} {'close (UTC)':<20} {'optimal (UTC)':<20} "
             f"{'Az open':>8} {'Az close':>8} {'Az opt':>8} {'incl':>7} {'dogleg':>7}  kind"]
    for w in windows:
        f = lambda d: d.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
        lines.append(
            f"{f(w.open_utc):<20} {f(w.close_utc):<20} {f(w.optimal_utc):<20} "
            f"{w.azimuth_open_deg:8.2f} {w.azimuth_close_deg:8.2f} {w.azimuth_optimal_deg:8.2f} "
            f"{w.inclination_deg:7.2f} {w.dogleg_deg:7.3f}  {w.kind} {w.direction}")
    return "\n".join(lines)
