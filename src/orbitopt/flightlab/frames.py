"""Flight Lab frame bridge (FL-002): LAUNCH_ECI / ECEF / GCRS and WGS84 geodetic.

The Flight Lab contract (orbitopt.flightlab.contract) allows exactly three
frame labels on a body track -- ``GCRS``, ``ECEF`` and ``LAUNCH_ECI`` -- and
this module is the ONLY place a track is allowed to move between them. A
physics backend never labels its own output "GCRS" unless it was really
propagated in GCRS; a pseudo-ECI backend's vectors arrive as ``LAUNCH_ECI``
and pass through here before the run manifest may say GCRS.

The three frames
----------------
* ``ECEF`` -- the Earth-fixed rotating frame (ITRF/WGS84 axes, the same axes
  as astropy's ``ITRS`` at the sample's epoch). Velocity is *relative to the
  rotating Earth*; converting to or from GCRS therefore carries the ω×r term,
  which is where the velocity transformation of the full IAU2006 pipeline
  (precession/nutation + polar motion + Earth rotation, astropy/ERFA) puts
  it in -- including the small CIP-motion part a bare ω×r about the ITRS
  z-axis misses.
* ``LAUNCH_ECI`` -- the launch-aligned pseudo-inertial frame of
  OpenTsiolkovsky: *axes equal to ECEF at the launch epoch t0, then held
  fixed in space* (non-rotating, no precession/nutation). At t0 it coincides
  with ECEF; afterwards it stays at launch orientation while the Earth turns
  underneath. Because it is inertial, converting it to/from GCRS is a pure
  constant rotation -- the ITRS→GCRS rotation *evaluated once at t0*, applied
  to both r and v, with no ω×r (the sim's velocity is already inertial).
* ``GCRS`` -- the Geocentric Celestial Reference System: the interchange
  frame of the Flight Lab v1 contract. Converting a ``LAUNCH_ECI`` track is
  exactly "identity rotation to ECEF(t0), then astropy ITRS→GCRS at the
  launch epoch" -- astropy's full conversion, not a GMST rotation.

Residual of OpenTsiolkovsky's own GMST-only RAAN
------------------------------------------------
OpenTsiolkovsky reports the orbit's RAAN by rotating its ``LAUNCH_ECI``
vectors with a GMST (IAU 1982) angle only -- no precession, no nutation, no
equation of equinoxes. Compared against a full conversion this GMST-only
RAAN misses the precession–nutation accumulated between launch and read-out:
general precession in longitude moves the equinox ≈ 50.3″/yr (≈ 0.14″/day;
≈ 5 m of cross-track position at 7000 km per day after launch), nutation in
longitude adds ±9″ short-period, and GMST vs GAST differs by the equation of
the equinoxes (≤ 0.009″). Inside a launch-day ascent window the residual is
sub-metre; anything holding the pseudo-inertial frame for days is not a
GCRS-contract frame any more. Hence the bridge rotates the vectors
themselves with the full astropy conversion and lets the contract's
``GCRS`` label mean what it says.

Time
----
Epochs are UTC datetimes plus per-sample ``t_s`` offsets on the UTC label
(same convention as the Flight Lab contract and Mission Control clock).
UTC→TT is done by :mod:`orbitopt.timescales` -- the package's single
leap-second-aware converter -- never by astropy; astropy is fed ``Time(...,
scale="tt")`` and takes care of UT1 (and hence the Earth rotation angle)
from IERS data internally. TDB−TT (< 1.7 ms of light-time noise) is not
needed by the frame rotation itself; astropy cross-checks of the time scales
live in the tests.

astropy is a lazy import: the base orbitopt package (and this module's
geodetic helpers and frame validation) works without it; only a call that
actually needs the ITRS→GCRS matrix raises a clear error if astropy is
missing. It is in ``envs/cpu.yml`` / ``envs/gpu.yml`` for the locked
environments.

    from orbitopt.flightlab.frames import to_gcrs, from_gcrs, geodetic_to_ecef
    r_gcrs, v_gcrs = to_gcrs(r_m, v_m_s, "LAUNCH_ECI", t0, t_s)   # batch
    r_ecef, v_ecef = from_gcrs(r_gcrs, v_gcrs, "ECEF", t0, t_s)
    site = geodetic_to_ecef(28.6083, -80.6042, 3.0)               # KSC LC-39A
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

import numpy as np

from orbitopt.flightlab.contract import FRAMES
from orbitopt.timescales import utc_datetime_to_mjd2000, utc_mjd2000_to_tt_seconds

#: WGS84 semi-major axis [m] (identical in GRS80, which astropy/ITRF uses).
WGS84_A = 6378137.0
#: WGS84 flattening.
WGS84_F = 1.0 / 298.257223563
#: IAU 2000 Earth rotation rate ω_E [rad/s] (the ω of the ω×r term).
EARTH_ROTATION_RATE = 7.2921159e-5

#: Astropy's TT origin J2000.0 = MJD 51544.5 in the TT scale, added to the
#: TT seconds computed by orbitopt.timescales to hand astropy an MJD(TT).
_MJD_TT_J2000 = 51544.5

#: Frame labels a caller may use with to_gcrs / from_gcrs -- exactly the
#: Flight Lab contract's vocabulary; there is no "ECI" default.
KNOWN_FRAMES: tuple[str, ...] = FRAMES

_FRAME_LIKE_ALIASES = (
    "ECI", "J2000", "TEME", "MOD", "TDME", "GCRF", "ITRF", "ECEF(0)", "NED", "VVLH", "LVLH",
)


class UnknownFrameError(ValueError):
    """A track/frame carried no Flight Lab frame label, or one this bridge
    does not know. Raised instead of guessing a frame: an unlabelled
    "ECI" is ambiguous in a way that quietly corrupts orbits (OpenTsiolkovsky
    pseudo-ECI vs J2000 vs TEME differ by arcminutes to degrees), so the
    caller must pick one of ``KNOWN_FRAMES``."""


def _check_frame(frame: str) -> None:
    if not isinstance(frame, str) or frame not in KNOWN_FRAMES:
        hint = ""
        if isinstance(frame, str) and frame.strip().upper() in _FRAME_LIKE_ALIASES:
            hint = (
                f" {frame!r} names a frame family, not a Flight Lab frame: a "
                "launch-aligned pseudo-inertial backend (e.g. OpenTsiolkovsky) "
                "is 'LAUNCH_ECI'; a true GCRS propagation is 'GCRS'; Earth-fixed "
                "is 'ECEF'."
            )
        raise UnknownFrameError(
            f"unknown Flight Lab frame {frame!r} -- allowed frames are exactly "
            f"{', '.join(KNOWN_FRAMES)} and there is no silent default.{hint}"
        )


def _as_batch(r_m, v_m_s, t_s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the (r, v, t) batch shapes and finiteness; returns
    float64 copies as (N,3), (N,3), (N,)."""
    r = np.asarray(r_m, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must be a [N,3] array of metres, got shape {r.shape}")
    if v.shape != r.shape:
        raise ValueError(f"velocities must have the same shape as positions, got {v.shape} vs {r.shape}")
    if t.ndim != 1 or t.shape[0] != r.shape[0]:
        raise ValueError(f"t_s must be a [N] array matching {r.shape[0]} samples, got shape {t.shape}")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(t))):
        raise ValueError("r/v/t contain non-finite values -- a nan in a trajectory is an error, not a sample")
    return r, v, t


def _as_utc_datetime(epoch_utc: datetime) -> datetime:
    if not isinstance(epoch_utc, datetime):
        raise TypeError(f"epoch_utc must be a datetime, got {type(epoch_utc).__name__}")
    return epoch_utc  # naive = UTC, same convention as orbitopt.timescales


def _require_astropy():
    """Lazy import of the astropy pieces the bridge needs (see module
    docstring): orbitopt works without astropy installed until a real frame
    conversion is requested."""
    try:
        from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
        from astropy.time import Time
        import astropy.units as u
    except ImportError as exc:
        raise RuntimeError(
            "the Flight Lab frame bridge needs astropy (ITRS<->GCRS at the launch "
            "epoch); install it (it is part of envs/cpu.yml / envs/gpu.yml) or "
            "keep the trajectory in its declared frame"
        ) from exc
    return CartesianDifferential, CartesianRepresentation, GCRS, ITRS, Time, u


def _tt_time(epoch_utc: datetime, t_s: Sequence[float] | None):
    """UTC epoch (+ optional per-sample UTC t_s offsets) -> astropy ``Time``
    in the TT scale. The UTC→TT conversion itself is orbitopt.timescales'
    (leap-second aware); astropy only receives the TT label and handles
    UT1/ERA internally via IERS data."""
    _Cd, _Cr, _GCRS, _ITRS, Time, _u = _require_astropy()
    mjd0 = utc_datetime_to_mjd2000(_as_utc_datetime(epoch_utc))
    if t_s is None:
        tt = utc_mjd2000_to_tt_seconds(mjd0)
        return Time(_MJD_TT_J2000 + tt / 86400.0, format="mjd", scale="tt")
    # t_s rides the UTC label (like every Flight Lab clock): offset the MJD2000
    # UTC day number, then convert each epoch through the single converter.
    tt = np.array([
        utc_mjd2000_to_tt_seconds(mjd0 + float(dt) / 86400.0) for dt in np.asarray(t_s, dtype=np.float64)
    ])
    return Time(_MJD_TT_J2000 + tt / 86400.0, format="mjd", scale="tt")


def _itrs_to_gcrs_matrix(obstime) -> np.ndarray:
    """The 3x3 position rotation ITRS(t)->GCRS at one TT epoch, obtained by
    pushing the three unit vectors through the astropy position pipeline (the
    matrix-free equivalent of the removed Frame.transform_matrix)."""
    Cd, Cr, GCRS, ITRS, _Time, u = _require_astropy()
    unit = Cr(np.eye(3) * u.m)
    out = ITRS(unit, obstime=obstime).transform_to(GCRS(obstime=obstime))
    return np.asarray(out.cartesian.xyz.to_value(u.m), dtype=np.float64)


def _transform_states(r: np.ndarray, v: np.ndarray, obstime, ecef_to_gcrs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Batch state-vector rotation between the rotating Earth-fixed frame and
    GCRS through the full astropy ITRS<->GCRS pipeline (position *and*
    velocity, including ω×r and the precession/nutation-rate terms).
    ``obstime`` is an astropy Time, scalar or one instant per sample."""
    Cd, Cr, GCRS, ITRS, _Time, u = _require_astropy()
    rep = Cr(
        r.T * u.m,
        differentials={"s": Cd(v.T * (u.m / u.s))},
    )
    frame = ITRS(rep, obstime=obstime) if ecef_to_gcrs else GCRS(rep, obstime=obstime)
    out = frame.transform_to(GCRS(obstime=obstime) if ecef_to_gcrs else ITRS(obstime=obstime))
    xyz = np.asarray(out.cartesian.xyz.to_value(u.m), dtype=np.float64).T
    diff = out.cartesian.differentials["s"]
    vel = np.stack(
        [np.asarray(diff.d_x.to(u.m / u.s).value),
         np.asarray(diff.d_y.to(u.m / u.s).value),
         np.asarray(diff.d_z.to(u.m / u.s).value)],
        axis=-1,
    ).reshape(-1, 3)
    return xyz, vel


def to_gcrs(r_m, v_m_s, frame: str, epoch_utc: datetime, t_s) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of Earth-centered state vectors to GCRS.

    Parameters follow the Flight Lab track convention: ``r_m`` [N,3] metres,
    ``v_m_s`` [N,3] m/s, both expressed in ``frame``; ``epoch_utc`` is the
    track's epoch (mission t=0) as a UTC datetime (naive = UTC); ``t_s`` [N]
    is each sample's seconds after the epoch on the UTC label. For
    ``frame="ECEF"`` every sample is converted at its own epoch
    (epoch_utc + t_s); for ``frame="LAUNCH_ECI"`` the rotation is the constant
    one evaluated at ``epoch_utc`` (an inertial frame does not care when the
    sample was taken, so ``t_s`` is validated but not used);
    ``frame="GCRS"`` passes through (a validated copy).

    Raises ``UnknownFrameError`` for unlabelled or unknown frames -- there is
    no silent "ECI" default -- and ``ValueError`` for malformed batches.
    """
    _check_frame(frame)
    r, v, t = _as_batch(r_m, v_m_s, t_s)
    if frame == "GCRS":
        return r.copy(), v.copy()
    if frame == "LAUNCH_ECI":
        mat = _itrs_to_gcrs_matrix(_tt_time(epoch_utc, None))
        return r @ mat.T, v @ mat.T
    return _transform_states(r, v, _tt_time(epoch_utc, t), ecef_to_gcrs=True)


def from_gcrs(r_m, v_m_s, frame: str, epoch_utc: datetime, t_s) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`to_gcrs`: GCRS state vectors back into ``frame``
    (``"GCRS"`` returns a validated copy). Same argument shapes and the same
    ``UnknownFrameError`` policy."""
    _check_frame(frame)
    r, v, t = _as_batch(r_m, v_m_s, t_s)
    if frame == "GCRS":
        return r.copy(), v.copy()
    if frame == "LAUNCH_ECI":
        mat = _itrs_to_gcrs_matrix(_tt_time(epoch_utc, None))
        return r @ mat, v @ mat
    return _transform_states(r, v, _tt_time(epoch_utc, t), ecef_to_gcrs=False)


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """WGS84 geodetic latitude/longitude [deg] + height above the ellipsoid
    [m] -> ECEF position [m]. Accepts scalars or numpy-broadcastable arrays;
    returns a ``(...,3)`` array (``(3,)`` for scalar input)."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n_rm = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)
    x = (n_rm + h) * np.cos(lat_r) * np.cos(lon_r)
    y = (n_rm + h) * np.cos(lat_r) * np.sin(lon_r)
    z = (n_rm * (1.0 - e2) + h) * np.sin(lat_r)
    return np.stack([np.asarray(c, dtype=np.float64) for c in np.broadcast_arrays(x, y, z)], axis=-1)


def ecef_to_geodetic(r_ecef_m):
    """ECEF position [m] -> ``(lat_deg, lon_deg, h_m)`` on WGS84, the inverse
    of :func:`geodetic_to_ecef` (Bowring seed + fixed-point refinement,
    converging to sub-nanometre). Accepts ``(3,)`` or ``(...,3)``."""
    r = np.asarray(r_ecef_m, dtype=np.float64)
    if r.shape[-1:] != (3,):
        raise ValueError(f"positions must be a [...,3] array of metres, got shape {r.shape}")
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    e2 = WGS84_F * (2.0 - WGS84_F)
    b = WGS84_A * math.sqrt(1.0 - e2)
    ep2 = e2 / (1.0 - e2)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    # reduced-latitude seed (Bowring): always within a degree of the answer
    theta = np.arctan2(p, z * (1.0 - WGS84_F))
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    lat = np.arctan2(
        z + ep2 * (WGS84_A / np.sqrt(1.0 + ep2**2 * sin_t**2)) * sin_t**3,
        np.where(p > 0.0, p, 1.0) - e2 * (WGS84_A / np.sqrt(1.0 + ep2**2 * sin_t**2)) * cos_t**3,
    )
    for _ in range(6):  # fixed point on (N + h): < 1e-13 rad in a few sweeps
        n_rm = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * n_rm * np.sin(lat), np.where(p > 0.0, p, 1.0))
    # exactly on the rotation axis the fixed point above is taken on a
    # substituted denominator: the answer is the pole itself.
    lat = np.where(p > 0.0, lat, np.where(z >= 0.0, np.pi / 2.0, -np.pi / 2.0))
    n_rm = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h_equator = p / np.cos(lat) - n_rm
    h_pole = np.abs(z) - b  # on the axis: z = sign(z) * (b + h)
    h = np.where(p > 1.0, h_equator, h_pole)
    lat_deg = np.degrees(lat)
    return np.asarray(lat_deg), np.asarray(np.degrees(lon)), np.asarray(h)
