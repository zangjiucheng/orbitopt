"""Thin helpers around pykep.planet for solar-system ephemerides, plus
epoch conversion utilities shared across problems/examples.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pykep as pk
from tudatpy.astro import time_conversion
from tudatpy.interface import spice

_JPL_LP_BODIES = (
    "mercury", "venus", "earth", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
)

_STD_MJD_AT_MJD2000_EPOCH = 51544.0

_SPICE_LOADED = False


def _ensure_spice_loaded() -> None:
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        spice.load_standard_kernels()
        _SPICE_LOADED = True


def planet(name: str) -> pk.planet._base:
    """Return a pykep low-precision JPL ephemeris planet by name, e.g.
    ``planet("earth")``. Raises ValueError for unknown names.
    """
    key = name.strip().lower()
    if key not in _JPL_LP_BODIES:
        raise ValueError(f"Unknown body '{name}'. Known: {_JPL_LP_BODIES}")
    return pk.planet.jpl_lp(key)


def mjd2000_from_date(year: int, month: int, day: int) -> float:
    """Convert a calendar date to MJD2000 (days since 2000-01-01), the
    epoch convention pykep uses by default.
    """
    return pk.epoch_from_string(f"{date(year, month, day).isoformat()} 00:00:00").mjd2000


def state(body: pk.planet._base, mjd2000: float):
    """Position (m) and velocity (m/s) of ``body`` at the given MJD2000 epoch,
    in the body's reference frame (heliocentric ecliptic J2000 for jpl_lp).
    """
    r, v = body.eph(pk.epoch(mjd2000))
    return r, v


def mjd2000_to_ephemeris_seconds(mjd2000: float) -> float:
    """Convert an MJD2000 epoch (days since 2000-01-01 00:00:00, pykep's
    convention) to ephemeris seconds since J2000 (2000-01-01 12:00:00 TDB),
    the time format tudatpy/SPICE functions expect. The 12-hour offset
    between the two epoch origins is why this isn't a plain days-to-seconds
    scaling.
    """
    julian_day = time_conversion.modified_julian_day_to_julian_day(
        mjd2000 + _STD_MJD_AT_MJD2000_EPOCH
    )
    return time_conversion.julian_day_to_seconds_since_epoch(julian_day)


def moon_state(mjd2000: float) -> tuple[np.ndarray, np.ndarray]:
    """Position (m) and velocity (m/s) of the Moon relative to Earth's
    center at the given MJD2000 epoch, in the ECLIPJ2000 frame. pykep's
    jpl_lp has no Moon entry, so this sources the real DE ephemeris via
    tudatpy's SPICE interface instead.
    """
    _ensure_spice_loaded()
    ephemeris_time = mjd2000_to_ephemeris_seconds(mjd2000)
    cartesian_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Moon",
        observer_body_name="Earth",
        reference_frame_name="ECLIPJ2000",
        aberration_corrections="NONE",
        ephemeris_time=ephemeris_time,
    )
    return cartesian_state[:3], cartesian_state[3:]


def body_fixed_to_eclipj2000_matrix(body_name: str, mjd2000: float) -> np.ndarray:
    """The 3x3 rotation matrix from a body's IAU body-fixed frame
    (``IAU_<BODY>`` -- x toward the prime meridian on the equator, z toward
    the north pole) to ECLIPJ2000, at the given MJD2000 epoch: the body's
    real inertial orientation (pole/axial tilt AND prime-meridian rotation
    phase) then.

    Works for any body with IAU rotation elements in the standard PCK
    kernels -- every Sun/planet/Moon rendered here. ``body_name`` is the
    SPICE body name (e.g. "Earth", "Mars", "Sun", "Moon"); the frame queried
    is ``IAU_<body_name uppercased>``. Note the giant planets' IAU frames use
    System III (deep-interior) rotation and the Sun's a rigid mean rate, so
    the *pole/tilt* is physical for every body but the absolute prime-meridian
    *phase* is only meaningful where the texture's longitude 0 is registered
    to that frame (today: Earth only).
    """
    _ensure_spice_loaded()
    ephemeris_time = mjd2000_to_ephemeris_seconds(mjd2000)
    frame = f"IAU_{body_name.upper()}"
    return np.asarray(
        spice.compute_rotation_matrix_between_frames(frame, "ECLIPJ2000", ephemeris_time),
        dtype=float,
    )


def earth_body_fixed_to_eclipj2000(vector_ecef, mjd2000: float) -> np.ndarray:
    """Rotate a (3,) vector from Earth body-fixed coordinates (IAU_Earth --
    x toward the Greenwich meridian on the equator, z toward the north
    pole) into the ECLIPJ2000 inertial frame, at the given MJD2000 epoch.

    For a fixed direction/point on Earth's surface (e.g. a real launch
    site's geodetic longitude/latitude), this is exactly what's needed to
    find where that point actually is in inertial space at a specific real
    time -- e.g. so a launch trajectory can start at the real launch site's
    real inertial direction at liftoff, not an arbitrary fixed axis.
    IAU_Earth's own pole is Earth's real rotation axis, not the ecliptic
    pole (~23.4 degrees away) -- also relevant for a "north" (z) reference
    that should mean Earth's true north, not the ecliptic normal.
    """
    return body_fixed_to_eclipj2000_matrix("Earth", mjd2000) @ np.asarray(vector_ecef, dtype=float)
