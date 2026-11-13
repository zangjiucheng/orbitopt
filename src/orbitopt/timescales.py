"""The one place orbitopt converts between epoch representations.

Time scales and conventions used across the package:

* **MJD2000 (UTC)** -- days since 2000-01-01 00:00:00 UTC, the convention of
  :mod:`orbitopt.launch_windows`, :mod:`orbitopt.injection` and
  :func:`orbitopt.bodies.mjd2000_from_date`. Like a calendar date / ISO
  string it is a UTC *label*: every day is counted as 86400 s and the
  fraction is the UTC time of day (leap seconds are not counted inside the
  day number, exactly like :class:`datetime.datetime`).
* **TT** -- Terrestrial Time, ``TT = TAI + 32.184 s`` and ``TAI = UTC +
  DeltaAT`` where DeltaAT is the leap-second count (10 s in 1972, 37 s since
  2017-01-01). TT - UTC is therefore *not* a constant 69.184 s.
* **TDB ephemeris seconds** ("ET") -- seconds of TDB since J2000.0
  (2000-01-01 12:00:00 TDB), the time argument of SPICE and tudatpy. TDB
  differs from TT by a periodic term below 1.7 ms, modelled here with the
  same formula SPICE's ``DELTET`` uses.

Every epoch conversion in orbitopt (``bodies.mjd2000_to_ephemeris_seconds``,
``injection.mjd2000_to_ephemeris_seconds``, the Mission Control date labels)
goes through this module so the same input gives the same epoch everywhere.
Pure Python / math: no pykep or tudatpy needed.

The leap-second table is current through IERS Bulletin C 70 (no leap second
before 2026-06-30 at least; none has been announced since 2017-01-01). UTC
before 1972-01-01 (the "rubber-second" era) is rejected rather than guessed.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

SECONDS_PER_DAY = 86400.0
#: TT - TAI, exact by definition.
TT_MINUS_TAI_S = 32.184
#: 2000-01-01 00:00:00 UTC, the MJD2000 origin.
MJD2000_EPOCH_UTC = datetime(2000, 1, 1, tzinfo=timezone.utc)

# (first MJD2000 UTC day the value applies, TAI - UTC [s]) -- IERS leap seconds.
_LEAP_SECOND_TABLE_DATES = (
    (1972, 1, 1, 10), (1972, 7, 1, 11), (1973, 1, 1, 12), (1974, 1, 1, 13),
    (1975, 1, 1, 14), (1976, 1, 1, 15), (1977, 1, 1, 16), (1978, 1, 1, 17),
    (1979, 1, 1, 18), (1980, 1, 1, 19), (1981, 7, 1, 20), (1982, 7, 1, 21),
    (1983, 7, 1, 22), (1985, 7, 1, 23), (1988, 1, 1, 24), (1990, 1, 1, 25),
    (1991, 1, 1, 26), (1992, 7, 1, 27), (1993, 7, 1, 28), (1994, 7, 1, 29),
    (1996, 1, 1, 30), (1997, 7, 1, 31), (1999, 1, 1, 32), (2006, 1, 1, 33),
    (2009, 1, 1, 34), (2012, 7, 1, 35), (2015, 7, 1, 36), (2017, 1, 1, 37),
)
_LEAP_MJD2000 = tuple(
    (datetime(y, m, d, tzinfo=timezone.utc) - MJD2000_EPOCH_UTC).days
    for y, m, d, _ in _LEAP_SECOND_TABLE_DATES
)
_LEAP_DELTA_AT = tuple(float(s) for *_, s in _LEAP_SECOND_TABLE_DATES)

# SPICE DELTET model of TDB - TT (kernel naif0012.tls values).
_DELTET_K = 1.657e-3
_DELTET_EB = 1.671e-2
_DELTET_M0 = 6.239996
_DELTET_M1 = 1.99096871e-7


#: Supported MJD2000 UTC epoch range, in days since 2000-01-01: the leap-second
#: table starts 1972-01-01, and 2110-01-01 is where trusting its last entry
#: stops being defensible. A value far outside this range is not an epoch in
#: this format at all -- ephemeris (TDB) seconds handed to an MJD2000 argument is
#: the classic case -- so the conversions reject it instead of returning a
#: silently wrong DeltaAT; see :mod:`orbitopt.physics_io` for the I/O layer that
#: turns the same magnitude check into a diagnosis.
MJD2000_MIN_UTC_DAYS = _LEAP_MJD2000[0]
MJD2000_MAX_UTC_DAYS = int((datetime(2110, 1, 1, tzinfo=timezone.utc) - MJD2000_EPOCH_UTC).total_seconds() // SECONDS_PER_DAY)


def tai_minus_utc(mjd2000_utc: float) -> float:
    """Leap-second count DeltaAT = TAI - UTC [s] in effect at the given
    MJD2000 UTC epoch. Raises ValueError outside the supported range
    (1972-01-01 .. 2110-01-01)."""
    t = float(mjd2000_utc)
    if not math.isfinite(t):
        raise ValueError(f"epoch must be finite, got {mjd2000_utc!r}")
    i = bisect_right(_LEAP_MJD2000, math.floor(t)) - 1
    if i < 0:
        raise ValueError("UTC before 1972-01-01 has no integer leap-second offset; not supported")
    if t > MJD2000_MAX_UTC_DAYS:
        raise ValueError(
            f"MJD2000 UTC epoch {mjd2000_utc!r} is beyond the supported range (up to "
            f"{MJD2000_MAX_UTC_DAYS} days = 2110-01-01); a value around 1e9 is TDB seconds "
            "since J2000, not MJD2000 days -- convert it with tdb_seconds_to_utc_mjd2000()"
        )
    return _LEAP_DELTA_AT[i]


def tt_minus_utc(mjd2000_utc: float) -> float:
    """TT - UTC [s] at the given MJD2000 UTC epoch (69.184 s since 2017)."""
    return tai_minus_utc(mjd2000_utc) + TT_MINUS_TAI_S


def tdb_minus_tt(tt_seconds_since_j2000: float) -> float:
    """TDB - TT [s] (|value| < 1.7 ms) at the given TT seconds past J2000,
    using SPICE's DELTET formula. Accurate to ~10 microseconds -- far below
    anything orbitopt resolves."""
    m = _DELTET_M0 + _DELTET_M1 * float(tt_seconds_since_j2000)
    return _DELTET_K * math.sin(m + _DELTET_EB * math.sin(m))


def utc_mjd2000_to_tt_seconds(mjd2000_utc: float) -> float:
    """MJD2000 UTC days -> TT seconds since J2000.0 (2000-01-01 12:00 TT)."""
    t = float(mjd2000_utc)
    return (t - 0.5) * SECONDS_PER_DAY + tt_minus_utc(t)


def utc_mjd2000_to_tdb_seconds(mjd2000_utc: float) -> float:
    """MJD2000 UTC days -> TDB seconds since J2000.0: the SPICE / tudatpy
    ephemeris time. Leap-second aware."""
    tt = utc_mjd2000_to_tt_seconds(mjd2000_utc)
    return tt + tdb_minus_tt(tt)


def tdb_seconds_to_utc_mjd2000(tdb_seconds_since_j2000: float) -> float:
    """Inverse of :func:`utc_mjd2000_to_tdb_seconds` (round-trips to well
    below a microsecond away from the leap second itself)."""
    et = float(tdb_seconds_since_j2000)
    tt = et
    for _ in range(3):  # TDB-TT varies by < 1e-9 s per ms: converges at once
        tt = et - tdb_minus_tt(tt)
    guess = tt / SECONDS_PER_DAY + 0.5  # TT label as MJD2000 days
    for _ in range(3):  # the leap-second offset is piecewise constant
        guess = (tt - tt_minus_utc(guess)) / SECONDS_PER_DAY + 0.5
    return guess


def utc_datetime_to_mjd2000(dt: datetime) -> float:
    """UTC datetime (naive = UTC) -> MJD2000 UTC days."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - MJD2000_EPOCH_UTC).total_seconds() / SECONDS_PER_DAY


def mjd2000_to_utc_datetime(mjd2000_utc: float) -> datetime:
    """MJD2000 UTC days -> aware UTC datetime (millisecond resolution)."""
    return MJD2000_EPOCH_UTC + timedelta(seconds=round(float(mjd2000_utc) * SECONDS_PER_DAY, 3))


def utc_datetime_to_tdb_seconds(dt: datetime) -> float:
    """UTC datetime -> TDB seconds since J2000.0."""
    return utc_mjd2000_to_tdb_seconds(utc_datetime_to_mjd2000(dt))


def tdb_seconds_to_utc_datetime(tdb_seconds_since_j2000: float) -> datetime:
    """TDB seconds since J2000.0 -> aware UTC datetime."""
    return mjd2000_to_utc_datetime(tdb_seconds_to_utc_mjd2000(tdb_seconds_since_j2000))
