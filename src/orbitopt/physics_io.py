"""The unified frame / epoch / timescale / unit check layer for physics I/O (V-01).

Every physics module in orbitopt speaks one of a handful of conventions --
metres or kilometres, seconds or days, Earth-centred or Sun-centred,
``GCRS``/``ECEF``/``LAUNCH_ECI``/``ECLIPJ2000``, an MJD2000 UTC label or TDB
ephemeris seconds -- and the epic rule for this package is that a convention
is always *declared*, never inferred from a bare array. This module is the one
place those declarations live and the one place they are enforced: physics
entry points state their :class:`PhysicsIOSpec` and check their arguments
against it, so a kilometre handed to a metre interface, a UTC label handed to a
TDB interface, or an ``ECEF`` track handed to something that wanted ``GCRS``
fails loudly at the boundary instead of turning into a plausible-looking wrong
trajectory.

What is checked, and how
------------------------
* **Frame** (:func:`check_frame`): the label must be one the API declares it
  accepts. There is no ``"ECI"`` default anywhere in orbitopt -- ``ECI``/
  ``J2000``/``ITRF``/``TEME`` name a *family*, and the members of a family
  differ by arcminutes to degrees, so :data:`AMBIGUOUS_FRAMES` turns them into
  an actionable error instead of a guess. :func:`check_frame_family` further
  rejects a rotating frame where an inertial one is required.
* **Timescale** (:func:`check_timescale`): ``utc`` / ``tai`` / ``tt`` / ``tdb``.
  A UTC label reaching an interface that wants ephemeris time is a
  leap-second-sized error (69.184 s since 2017 -- ~17 km of light-time along a
  Moon transfer), so the error says to convert through
  :mod:`orbitopt.timescales`, the package's single converter.
* **Epoch** (:func:`check_epoch`): every epoch argument also has to *be* what
  its format says. The supported formats live on disjoint magnitude ranges
  (MJD2000 days for the supported era: ~1e4; TDB/TT seconds since J2000: ~1e9),
  so handing an MJD2000 day count to an ephemeris-seconds argument (or the
  reverse) is caught by magnitude even when the label is missing.
* **Unit** (:func:`check_grav_parameter`, :func:`check_positions`,
  :func:`check_velocities`): a gravitational parameter identifies its central
  body and its unit system (:data:`KNOWN_MU`), so ``mu`` can be checked against
  the units an API declares -- and, when an API is unit-agnostic (the Lambert
  solvers), the units *inferred from* ``mu`` tell the caller whether the
  position vectors are in the same system. Positions and velocities are checked
  against a generous plausibility band for the declared unit; being out of band
  by a factor of ~1000 (metres vs kilometres) or ~1.5e11 (AU vs metres) is
  reported as exactly that mistake.

All errors derive from :class:`PhysicsIOError` (itself a ``ValueError``), so
existing ``pytest.raises(ValueError)`` around physics entry points keep
working, and every message names the argument, the expected convention and the
conversion that would fix it -- never just "invalid".

Deliberately *not* done here
----------------------------
The band checks are order-of-magnitude guards against unit confusion, not
physics validation: they accept an unphysical but mathematically valid Lambert
endpoint 2400 km from the Earth's centre, and they cannot tell an ``ECEF``
track labelled ``GCRS`` from a real one (only the frame label and the frame
bridge :mod:`orbitopt.flightlab.frames` can). Mission-level physics checks
belong to :mod:`orbitopt.flightlab.checks` and :mod:`orbitopt.verify.report`;
this layer only guarantees that what reaches them is expressed in the units,
frames and epochs the caller intended.

    from orbitopt.physics_io import NBODY_SCREENING
    NBODY_SCREENING.check_state(r0, v0, epoch=epoch0, where="propagate_spacecraft_batch")
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from orbitopt.timescales import (
    MJD2000_MAX_UTC_DAYS,
    MJD2000_MIN_UTC_DAYS,
    utc_mjd2000_to_tdb_seconds,
    utc_mjd2000_to_tt_seconds,
    utc_datetime_to_mjd2000,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PhysicsIOError(ValueError):
    """Base class of every frame / epoch / timescale / unit mismatch raised by
    the physics I/O layer. A ``ValueError`` subclass, because a wrong convention
    *is* a bad argument value."""


class FrameMismatchError(PhysicsIOError):
    """A frame label is missing, unknown, ambiguous (``ECI``), or from the
    wrong family for this interface. Nothing in orbitopt picks a frame on the
    caller's behalf."""


class TimescaleMismatchError(PhysicsIOError):
    """The declared timescale (UTC/TAI/TT/TDB) is not one this interface
    accepts. The fix is a conversion through :mod:`orbitopt.timescales`, never
    a relabelled number."""


class EpochMismatchError(PhysicsIOError):
    """An epoch is not an epoch of the format the interface declares -- either
    the wrong type, or a value whose magnitude belongs to a different epoch
    format (the classic MJD2000-days vs ephemeris-seconds swap)."""


class UnitMismatchError(PhysicsIOError):
    """A value is not in the unit system the interface declares (or ``mu`` and
    the state vectors disagree). Raised with the factor between the declared
    unit and the unit the values actually look like."""


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

#: Every frame label orbitopt knows, with its family. ``inertial`` frames do
#: not rotate with the Earth; ``rotating`` frames do (velocity is Earth-fixed);
#: ``pseudo-inertial`` means "aligned with the Earth-fixed frame at one epoch,
#: then held fixed" (no precession/nutation), so it may be rotated into an
#: inertial frame by a *constant* matrix -- see
#: :mod:`orbitopt.flightlab.frames`.
FRAME_FAMILIES = ("inertial", "pseudo-inertial", "rotating")


@dataclass(frozen=True)
class FrameSpec:
    """One known frame: its label, its rotation family and the plane it is
    defined by (``equatorial``, ``ecliptic`` or ``earth-fixed``)."""

    name: str
    family: str
    plane: str
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover - repr helper
        return f"{self.name} ({self.family}, {self.plane} plane)"


FRAMES: dict[str, FrameSpec] = {
    spec.name: spec
    for spec in (
        FrameSpec("GCRS", "inertial", "equatorial",
                  "Geocentric Celestial Reference System -- the Flight Lab v1 interchange frame."),
        FrameSpec("EME2000", "inertial", "equatorial",
                  "J2000 mean equator/equinox; tudatpy's Earth-centred equatorial frame."),
        FrameSpec("ECLIPJ2000", "inertial", "ecliptic",
                  "J2000 ecliptic: pykep's and orbitopt.dynamics' screening frame."),
        FrameSpec("TEME", "inertial", "equatorial",
                  "True equator/equinox of date (SGP4/satrec); not interchangeable with GCRS."),
        FrameSpec("LAUNCH_ECI", "pseudo-inertial", "equatorial",
                  "OpenTsiolkovsky's launch-aligned frame: ECEF axes at the launch epoch, then fixed."),
        FrameSpec("ECEF", "rotating", "earth-fixed",
                  "Earth-fixed (WGS84/ITRF axes); velocity is relative to the rotating Earth."),
    )
}

#: Labels that name a *family* or a near-miss rather than one of
#: :data:`FRAMES`, with the hint appended to the error. Keeping this table here
#: (instead of silently accepting the alias) is the point: ``ECI`` alone cannot
#: tell a caller whether they hold GCRS, EME2000, ECLIPJ2000 or a launch-aligned
#: pseudo-ECI, and those differ by arcminutes to degrees.
AMBIGUOUS_FRAMES: dict[str, str] = {
    "ECI": "'ECI' is a family, not a frame: a true inertial propagation is 'GCRS' "
           "(or 'EME2000'/'ECLIPJ2000'); a launch-aligned pseudo-inertial backend "
           "(e.g. OpenTsiolkovsky) is 'LAUNCH_ECI'.",
    "J2000": "'J2000' is ambiguous between the equatorial 'EME2000' and the ecliptic "
             "'ECLIPJ2000'; say which one.",
    "GCRF": "'GCRF' is the realization of 'GCRS' used by IERS products; orbitopt labels it 'GCRS'.",
    "ITRF": "'ITRF' is the realization of the Earth-fixed frame orbitopt labels 'ECEF'.",
    "WGS84": "'WGS84' is the ellipsoid, not a frame; the Earth-fixed frame is 'ECEF'.",
    "TEME": "'TEME' (SGP4's true-of-date frame) is not 'GCRS'/'EME2000'; convert before mixing them.",
    "NED": "'NED' is a local topographic frame; orbitopt physics I/O is Earth-centred.",
    "VVLH": "'VVLH'/'LVLH' is a body-fixed orbital frame, not an Earth-centred one.",
    "LVLH": "'VVLH'/'LVLH' is a body-fixed orbital frame, not an Earth-centred one.",
    "MOD": "'MOD' (mean-of-date) needs a precession/nutation conversion to 'GCRS'/'EME2000'.",
    "TDME": "'TDME' (true-of-date) needs a nutation conversion to 'GCRS'/'EME2000'.",
    "ECEF(0)": "'ECEF(0)' (fixed at one epoch) is what 'LAUNCH_ECI' means; say 'LAUNCH_ECI'.",
    "NONE": "an unlabelled track has no frame; every physics interface here requires one of its declared frames.",
}

#: Earth-fixed labels, i.e. the frames whose velocity carries the omega x r term.
ROTATING_FRAMES = tuple(s.name for s in FRAMES.values() if s.family == "rotating")


def _where(prefix: str, where: str) -> str:
    return f"{where}: " if where else ""


def check_frame(frame, allowed: Sequence[str], *, where: str = "") -> str:
    """Return ``frame`` if it is one of ``allowed``, else raise
    :class:`FrameMismatchError`.

    ``allowed`` is the frame vocabulary of the calling interface (usually its
    spec's :attr:`PhysicsIOSpec.frames`), so the same label can be fine in one
    module and rejected in another. Aliases in :data:`AMBIGUOUS_FRAMES` are
    rejected with a hint naming the label to use -- there is no silent default,
    and never a guess from ``"ECI"``.
    """
    names = tuple(allowed)
    for name in names:
        if name not in FRAMES:  # an interface declaring a frame this layer doesn't know
            raise FrameMismatchError(f"{_where('frame', where)}{name!r} is not a frame orbitopt knows; "
                                     f"known frames: {', '.join(sorted(FRAMES))}")
    if isinstance(frame, str) and frame in names:
        return frame
    hint = ""
    if isinstance(frame, str):
        key = frame.strip().upper()
        hint = AMBIGUOUS_FRAMES.get(key, "")
        if not hint and key in FRAMES:
            if key != frame:
                hint = f"{frame!r} is a known frame with a different spelling; use {key}."
            else:
                hint = (f"{key} is a known orbitopt frame ({FRAMES[key].family} family) that "
                        "this interface does not take.")
    raise FrameMismatchError(
        f"{_where('frame', where)}unknown frame {frame!r} -- this interface accepts exactly "
        f"{', '.join(names) or '(nothing declared)'} and there is no silent default."
        + (f" {hint}" if hint else "")
    )


def check_frame_family(frame, family: str, *, where: str = "") -> str:
    """Like :func:`check_frame` but against a rotation *family*
    (``"inertial"``, ``"pseudo-inertial"`` or ``"rotating"``): the check an API
    needs when it converts between frames and must not be handed an Earth-fixed
    vector where it expects an inertial one (or vice versa)."""
    if family not in FRAME_FAMILIES:  # pragma: no cover - developer error
        raise ValueError(f"unknown frame family {family!r}; expected one of {FRAME_FAMILIES}")
    allowed = tuple(name for name, spec in FRAMES.items() if spec.family == family)
    if isinstance(frame, str) and frame in allowed:
        return frame
    if isinstance(frame, str) and frame in FRAMES:
        raise FrameMismatchError(
            f"{_where('frame', where)}{frame!r} is a {FRAMES[frame].family}-family frame; this "
            f"interface takes one of the {family}-family frames: {', '.join(allowed)}."
        )
    return check_frame(frame, allowed, where=where)


# ---------------------------------------------------------------------------
# Timescales and epochs
# ---------------------------------------------------------------------------

#: The timescales orbitopt names in its I/O contracts.
TIMESCALES = ("utc", "tai", "tt", "tdb")

#: Why a rejected timescale is wrong, in the terms of the fix.
_TIMESCALE_HINTS = {
    "utc": "UTC is a calendar label, not ephemeris time: TT - UTC is 69.184 s since "
           "2017-01-01 (leap seconds), and TDB - TT is another sub-ms periodic term. "
           "Convert with orbitopt.timescales.utc_mjd2000_to_tdb_seconds() / "
           "utc_datetime_to_tdb_seconds() -- the package's leap-second-aware converter.",
    "tai": "TAI is not ephemeris time either: TT = TAI + 32.184 s. Convert through "
           "orbitopt.timescales.",
    "tt": "TT is not TDB: they differ by a periodic term below 1.7 ms (SPICE DELTET). "
          "For ephemeris-time interfaces use orbitopt.timescales utc_mjd2000_to_tdb_seconds().",
    "tdb": "TDB seconds since J2000 are ephemeris time; a UTC label (MJD2000 days or a "
           "datetime) is not. Convert the other way with "
           "orbitopt.timescales.tdb_seconds_to_utc_mjd2000().",
}

#: MJD2000 days is the only epoch *format* orbitopt's planning layer uses, and
#: it is only meaningful where the leap-second table is (>= 1972-01-01) and
#: within the era this project plans for.
_MIN_MJD2000 = float(MJD2000_MIN_UTC_DAYS)   # the bands live in orbitopt.timescales
_MAX_MJD2000 = float(MJD2000_MAX_UTC_DAYS)   # so they cannot drift apart

#: Formats accepted by :func:`check_epoch`, each with its magnitude band. The
#: bands are disjoint by orders of magnitude, which is what makes a mislabelled
#: epoch detectable without a label.
EPOCH_FORMATS: dict[str, tuple[float, float, str]] = {
    "mjd2000": (_MIN_MJD2000, _MAX_MJD2000,
                "MJD2000 UTC days (days since 2000-01-01 00:00:00 UTC)"),
    "tdb_s_j2000": (utc_mjd2000_to_tdb_seconds(_MIN_MJD2000),
                    utc_mjd2000_to_tdb_seconds(_MAX_MJD2000),
                    "TDB seconds since J2000.0 (SPICE/tudatpy ephemeris seconds)"),
    "tt_s_j2000": (utc_mjd2000_to_tt_seconds(_MIN_MJD2000),
                   utc_mjd2000_to_tt_seconds(_MAX_MJD2000),
                   "TT seconds since J2000.0"),
}

#: Below this magnitude an "epoch in seconds since J2000" is really a day count
#: (MJD2000 epochs of the supported era are ~1e4; ephemeris seconds are ~1e8).
_EPOCH_FORMAT_SPLIT_S = 1.0e5


def check_timescale(scale, allowed: Iterable[str] = TIMESCALES, *, where: str = "") -> str:
    """Return the normalised timescale label if ``scale`` is one of
    ``allowed``, else raise :class:`TimescaleMismatchError` with the conversion
    that would fix it."""
    names = tuple(s.lower() for s in allowed)
    if isinstance(scale, str) and scale.strip().lower() in names:
        return scale.strip().lower()
    hint = _TIMESCALE_HINTS.get(scale.strip().lower(), "") if isinstance(scale, str) else ""
    raise TimescaleMismatchError(
        f"{_where('timescale', where)}timescale {scale!r} is not one of {names} accepted here."
        + (f" {hint}" if hint else "")
    )


def check_epoch(epoch, fmt: str = "mjd2000", *, where: str = ""):
    """Check an epoch argument against the format the interface declares and
    return it (``float``, or the ``datetime`` for ``fmt="utc_datetime"``).

    ``fmt`` is a key of :data:`EPOCH_FORMATS` or ``"utc_datetime"``. Beyond the
    type check the value's magnitude has to fall in the format's band, so a
    value that is really another format's epoch -- an MJD2000 day count handed
    to an ephemeris-seconds argument, or ephemeris seconds handed to an MJD2000
    argument -- is rejected with the diagnosis, even when the caller labelled
    nothing. ``float("nan")``/``inf`` are never a valid epoch.
    """
    if fmt == "utc_datetime":
        if isinstance(epoch, datetime):
            return epoch
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch must be a UTC datetime (naive means UTC), got "
            f"{type(epoch).__name__} {epoch!r} -- a number is an MJD2000 day count or "
            "ephemeris seconds; convert with orbitopt.timescales."
        )
    if fmt not in EPOCH_FORMATS:
        raise ValueError(f"unknown epoch format {fmt!r}; expected one of "
                         f"{', '.join((*EPOCH_FORMATS, 'utc_datetime'))}")
    lo, hi, described = EPOCH_FORMATS[fmt]
    try:
        value = float(epoch)
    except (TypeError, ValueError) as exc:
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch {epoch!r} is not a number; this interface takes "
            f"{described}."
        ) from exc
    if not math.isfinite(value):
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch must be finite, got {epoch!r}."
        )
    if fmt == "mjd2000" and abs(value) > _EPOCH_FORMAT_SPLIT_S:
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch {epoch!r} is {described}, but its magnitude looks "
            "like seconds since J2000 (ephemeris/TDB seconds), not days: convert with "
            "orbitopt.timescales.tdb_seconds_to_utc_mjd2000() -- UTC and TDB are also not the "
            "same scale (TT - UTC is 69.184 s since 2017)."
        )
    if fmt != "mjd2000" and 0.0 < abs(value) < _EPOCH_FORMAT_SPLIT_S:
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch {epoch!r} is {described}, but a value of order 1e4 "
            "or less is an MJD2000 *day count* (or a duration in seconds), not seconds since "
            "J2000: convert with orbitopt.timescales.utc_mjd2000_to_tdb_seconds()."
        )
    if lo <= value <= hi:
        return value
    raise EpochMismatchError(
        f"{_where('epoch', where)}epoch {epoch!r} is outside the supported range "
        f"[{lo}, {hi}] of {described} (1972-01-01 .. 2110-01-01: before 1972 UTC has no "
        "integer leap-second offset)."
    )


def check_epochs(epochs, fmt: str = "mjd2000", *, where: str = "") -> np.ndarray:
    """Vectorised :func:`check_epoch` over an array of epochs: every element is
    checked against ``fmt`` (the message names the index of the first offender),
    and the array is returned as float64.

    A time *grid* is where a timescale mix-up actually happens in practice --
    one column of a propagation table filled with TDB seconds instead of MJD2000
    UTC days -- so the check has to run on the whole column, not only on a
    scalar mission epoch."""
    if fmt == "utc_datetime":
        out = np.asarray([check_epoch(e, fmt, where=where) for e in np.asarray(epochs).ravel()],
                         dtype=object).reshape(np.shape(epochs))
        return out
    e = np.asarray(epochs, dtype=np.float64)
    if e.ndim == 0:
        return np.asarray(check_epoch(e, fmt, where=where), dtype=np.float64)
    if not np.all(np.isfinite(e)):
        i = int(np.flatnonzero(~np.isfinite(e))[0])
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch[{i}] = {float(e.ravel()[i])!r} is not finite; an epoch must "
            "be a finite number."
        )
    lo, hi, described = EPOCH_FORMATS[fmt]
    if fmt == "mjd2000":
        wrong = np.abs(e) > _EPOCH_FORMAT_SPLIT_S
        if np.any(wrong):
            i = int(np.flatnonzero(wrong.ravel())[0])
            raise EpochMismatchError(
                f"{_where('epoch', where)}epoch[{i}] = {float(e.ravel()[i])!r} is {described}, but its "
                "magnitude looks like seconds since J2000: convert the column with "
                "orbitopt.timescales.tdb_seconds_to_utc_mjd2000() -- UTC and TDB are also not "
                "the same scale (TT - UTC is 69.184 s since 2017)."
            )
    else:
        wrong = (0.0 < np.abs(e)) & (np.abs(e) < _EPOCH_FORMAT_SPLIT_S)
        if np.any(wrong):
            i = int(np.flatnonzero(wrong.ravel())[0])
            raise EpochMismatchError(
                f"{_where('epoch', where)}epoch[{i}] = {float(e.ravel()[i])!r} is {described}, but its "
                "magnitude is not in the range this format can mean: values of order 1e4 or "
                "less are MJD2000 day counts, values above 1e9 are past the supported era. "
                "Convert with orbitopt.timescales."
            )
    bad = (e < lo) | (e > hi)
    if np.any(bad):
        i = int(np.flatnonzero(bad.ravel())[0])
        raise EpochMismatchError(
            f"{_where('epoch', where)}epoch[{i}] = {float(e.ravel()[i])!r} is outside the supported "
            f"range [{lo}, {hi}] of {described} (1972-01-01 .. 2110-01-01: before 1972 UTC has "
            "no integer leap-second offset)."
        )
    return e


def epoch_to_tdb_seconds(epoch, fmt: str = "mjd2000", *, where: str = "") -> float:
    """Validate an epoch with :func:`check_epoch` and return it as TDB seconds
    since J2000.0 -- the one conversion path (via :mod:`orbitopt.timescales`)
    every physics module shares."""
    value = check_epoch(epoch, fmt, where=where)
    if fmt == "tdb_s_j2000":
        return float(value)
    if fmt == "utc_datetime":
        return utc_mjd2000_to_tdb_seconds(utc_datetime_to_mjd2000(value))
    if fmt == "tt_s_j2000":
        return float(value)  # TDB - TT is < 1.7 ms; callers wanting it ask timescales directly
    return utc_mjd2000_to_tdb_seconds(float(value))


#: Durations longer than this (3000 years) are a unit/format mistake, not a
#: transfer.
MAX_DURATION_S = 1.0e11


def check_duration(duration, *, where: str = "", allow_negative: bool = True) -> float:
    """Check an elapsed-time argument (seconds) and return it as a float.
    Non-finite and beyond-the-universe values are rejected; by default negative
    durations are allowed because backward propagation is a real use case."""
    try:
        value = float(duration)
    except (TypeError, ValueError) as exc:
        raise UnitMismatchError(
            f"{_where('duration', where)}duration {duration!r} is not a number of seconds."
        ) from exc
    if not math.isfinite(value):
        raise UnitMismatchError(
            f"{_where('duration', where)}duration must be finite, got {duration!r}."
        )
    if not allow_negative and value < 0.0:
        raise UnitMismatchError(
            f"{_where('duration', where)}duration {duration!r} s is negative; time of flight "
            "is forward."
        )
    if abs(value) > MAX_DURATION_S:
        raise UnitMismatchError(
            f"{_where('duration', where)}duration {duration!r} s exceeds {MAX_DURATION_S:.0e} s "
            "(3000 years) -- a days/hertz/radians mix-up, not a transfer."
        )
    return value


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#: Length units orbitopt accepts in a physics I/O contract, [m] per unit.
LENGTH_UNITS: dict[str, float] = {"m": 1.0, "km": 1.0e3, "au": 1.495978707e11}

#: Time units, [s] per unit.
TIME_UNITS: dict[str, float] = {"s": 1.0, "day": 86400.0}

#: Central bodies whose gravitational parameter orbitopt uses, in SI (m^3/s^2).
#: ``mu`` is the one input that identifies both the body and the unit system,
#: so it is what the unit checks key on.
KNOWN_MU: dict[str, float] = {
    "sun": 1.32712440018e20,
    "earth": 3.9860044150e14,
    "moon": 4.9028001e12,
    "mars": 4.282837e13,
    "jupiter": 1.26686534e17,
    "mercury": 2.2032e13,
    "venus": 3.24859e14,
    "saturn": 3.793131e16,
    "uranus": 5.793939e15,
    "neptune": 6.836529e15,
}

#: Relative tolerance with which a ``mu`` must match a tabulated value for the
#: match to count (DE/EGM96 spellings differ by less than 1e-6).
_KNOWN_MU_RTOL = 1e-5


@dataclass(frozen=True)
class UnitSystem:
    """A length/time unit pair, e.g. ``(km, s)`` or ``(au, day)`` -- the units a
    physics interface declares, or the ones inferred from its ``mu``."""

    length: str
    time: str

    @property
    def metres_per_length_unit(self) -> float:
        """Metres per one :attr:`length` unit."""
        return LENGTH_UNITS[self.length]

    @property
    def seconds_per_time_unit(self) -> float:
        """Seconds per one :attr:`time` unit."""
        return TIME_UNITS[self.time]

    @property
    def label(self) -> str:
        """Human-readable unit label for messages, e.g. ``km / km/s``."""
        return f"{self.length} / {self.length} per {self.time}"

    def grav_parameter(self, mu_si: float) -> float:
        """An SI ``mu`` [m^3/s^2] expressed in this system."""
        return mu_si * self.seconds_per_time_unit ** 2 / self.metres_per_length_unit ** 3

    def to_metres(self, values):
        """Lengths in :attr:`length` -> metres."""
        return np.asarray(values, dtype=np.float64) * self.metres_per_length_unit

    def to_seconds(self, values):
        """Durations in :attr:`time` -> seconds."""
        return np.asarray(values, dtype=np.float64) * self.seconds_per_time_unit


def _unit_diagnosis(values: np.ndarray, unit: str, band: tuple[float, float],
                    scale_table: dict[str, float]) -> str:
    """Diagnose an out-of-band magnitude by naming the unit the values look like
    instead: ``|r|`` read as kilometres landing inside the band that metres are
    required to hit is reported as the ``km`` -> ``m`` error it is."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    lo, hi = band
    for name, scale in scale_table.items():
        if name == unit or scale == scale_table[unit]:
            continue
        factor = scale / scale_table[unit]
        rescaled = finite * factor
        if np.all(rescaled > 0.0) and np.all(rescaled >= lo) and np.all(rescaled <= hi):
            return (f"read as {name} instead of {unit} (factor {factor:g}) the same values span "
                    f"[{float(np.min(rescaled)):g}, {float(np.max(rescaled)):g}] and do fit the "
                    f"band -- this looks like a {name}/{unit} unit mix-up.")
    return ""


def _magnitudes(points, n_dims: int = 3) -> np.ndarray:
    """Magnitudes of a ``(3,)`` / ``(N, 3)`` / ``(..., 3)`` array of vectors."""
    a = np.asarray(points, dtype=np.float64)
    if a.ndim == 0 or a.shape[-1:] != (n_dims,):
        raise PhysicsIOError(
            f"expected a vector or [...,{n_dims}] array, got shape {a.shape}"
        )
    return np.linalg.norm(a, axis=-1)


def _identify_mu(value: float) -> list[tuple[str, UnitSystem]] | None:
    """Every (body, unit system) whose tabulated ``mu`` equals ``value``."""
    matches: list[tuple[str, UnitSystem]] = []
    for body, mu_si in KNOWN_MU.items():
        for length in LENGTH_UNITS:
            for time in TIME_UNITS:
                candidate = UnitSystem(length, time)
                if abs(candidate.grav_parameter(mu_si) - value) <= _KNOWN_MU_RTOL * value:
                    matches.append((body, candidate))
    return matches or None


def identify_grav_parameter(mu, *, where: str = "") -> tuple[str, UnitSystem] | None:
    """Name the central body a ``mu`` belongs to and the unit system it is
    written in -- the check a unit-agnostic interface (the Lambert solvers) uses
    to find out which units *its state vectors* must be in. ``None`` for a
    gravitational parameter of a body this table does not know (a custom
    ``mu``, an artificial central mass, ``mu_earth + mu_moon``)."""
    matches = check_grav_parameter(mu, where=where)
    body_unit = _identify_mu(float(mu)) if matches is not None else None
    if body_unit is None:
        return None
    for body, system in body_unit:
        if system == matches:
            return body, system
    return body_unit[0]


def check_grav_parameter(mu, *, where: str = "", units: UnitSystem | None = None) -> UnitSystem | None:
    """Check a gravitational parameter and, if it is a known body's tabulated
    ``mu``, return the :class:`UnitSystem` it is expressed in.

    ``units`` (the interface's declared length/time pair) makes the check
    strict: a ``mu`` that matches a tabulated body in a *different* system than
    the declared one raises :class:`UnitMismatchError` naming both -- that is
    the ``mu`` in km^3/s^2 handed to a metres interface. Without ``units``
    (unit-agnostic APIs such as the Lambert solvers) the function only reports
    the inferred system, and returns ``None`` for a central body it does not
    know. Always raises for a non-finite or non-positive ``mu``.
    """
    try:
        value = float(mu)
    except (TypeError, ValueError) as exc:
        raise UnitMismatchError(
            f"{_where('mu', where)}gravitational parameter {mu!r} is not a number."
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise UnitMismatchError(
            f"{_where('mu', where)}gravitational parameter must be finite and > 0, got {mu!r}."
        )
    matches = _identify_mu(value)
    if matches is None:
        return None
    if units is not None:
        for body, system in matches:
            if system == units:
                return system
        body, system = matches[0]
        raise UnitMismatchError(
            f"{_where('mu', where)}mu {value:g} is {body}'s gravitational parameter in "
            f"{system.length}^3/{system.time}^2, but this interface takes "
            f"{units.length}^3/{units.time}^2 (mu = "
            f"{units.grav_parameter(KNOWN_MU[body]):.9g}). Convert mu, not the states: "
            "a mu in one unit system with states in another is a unit error that no "
            "downstream check can undo."
        )
    return matches[0][1]


#: Plausible distance from each body's centre, in metres: generous bands whose
#: job is to catch a unit-system mix-up (a ``6600`` where ``6.6e6`` belongs),
#: not to judge a trajectory. The lower bounds sit below the surface on purpose
#: -- a Lambert endpoint inside the atmosphere, or an impact trajectory, is a
#: legal maneuver problem -- and the upper bound of the Sun's band is ~66 AU.
BODY_DISTANCE_RANGE_M: dict[str, tuple[float, float]] = {
    "sun": (1.0e7, 1.0e13),
    "earth": (1.0e5, 2.0e9),
    "moon": (1.0e6, 1.0e9),
    "mars": (3.0e6, 1.0e9),
    "mercury": (2.0e6, 1.0e9),
    "venus": (6.0e6, 1.0e9),
    "jupiter": (7.0e7, 1.0e10),
    "saturn": (6.0e7, 1.0e10),
    "uranus": (2.5e7, 1.0e10),
    "neptune": (2.4e7, 1.0e10),
}

_DEFAULT_DISTANCE_RANGE_M = (1.0e3, 1.0e14)


def distance_band(body: str, units: UnitSystem) -> tuple[float, float]:
    """:data:`BODY_DISTANCE_RANGE_M` expressed in ``units`` -- what a radius
    check needs when the interface's unit system came from ``mu``."""
    lo, hi = BODY_DISTANCE_RANGE_M.get(body, _DEFAULT_DISTANCE_RANGE_M)
    scale = LENGTH_UNITS[units.length]
    return lo / scale, hi / scale


def check_lambert_pair(r1, r2, mu, *, where: str = "") -> None:
    """Check a single Lambert transfer: ``mu`` names the central body and so the
    unit system its endpoints are written in, and both endpoints have to be
    finite vectors within that body's distance band.

    This is the check for the *single* solvers. The batch solvers deliberately
    do not run it: a batch marks per-row legality itself (an illegal row gets a
    ``NaN`` and ``valid=False``) rather than raising, so a row-level magnitude
    check there would change the documented contract of
    :func:`orbitopt.lambert.gpu_batch.solve_lambert_batch`."""
    body_units = identify_grav_parameter(mu, where=where)
    if body_units is None:  # custom mu: nothing to infer the units from
        return
    body, units = body_units
    band = distance_band(body, units)
    check_positions(r1, units=units, radius_range=band, centre=body,
                    where=f"{where}: r1" if where else "r1")
    check_positions(r2, units=units, radius_range=band, centre=body,
                    where=f"{where}: r2" if where else "r2")


def _as_float_array(values, what: str, where: str) -> np.ndarray:
    """``np.asarray(..., float64)`` with a layer error instead of numpy's."""
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PhysicsIOError(
            f"{_where(what, where)}{what} must be an array of numbers, got "
            f"{type(values).__name__}: {exc}"
        ) from exc


def check_positions(positions, *, units: UnitSystem, radius_range: tuple[float, float],
                    centre: str = "earth", where: str = "") -> np.ndarray:
    """Validate a batch of position vectors against a physics interface's
    declared unit system and return it as float64.

    Shape (last axis 3), finiteness and a plausibility band for the *radius* in
    the declared unit are checked. Out-of-band values get a
    diagnosis naming the unit they look like: a LEO state written as ``6600``
    reaches a metres interface as a ``km -> m`` error, and that is what the
    message says.
    """
    r = _as_float_array(positions, "positions", where)
    if r.ndim == 0 or r.shape[-1:] != (3,):
        raise PhysicsIOError(
            f"{_where('positions', where)}positions must be a (3,) or [...,3] array of "
            f"{units.length}, got shape {r.shape}"
        )
    radii = _magnitudes(r)
    finite = np.isfinite(radii)
    if not np.all(finite):
        raise PhysicsIOError(
            f"{_where('positions', where)}positions contain non-finite values -- a nan/inf in "
            "a state vector is an error, not a sample."
        )
    lo, hi = radius_range
    bad = (radii < lo) | (radii > hi)
    if np.any(bad):
        worst = float(np.max(radii[bad])) if np.any(radii[bad] > hi) else float(np.min(radii[bad]))
        detail = _unit_diagnosis(radii[bad], units.length, radius_range, LENGTH_UNITS)
        raise UnitMismatchError(
            f"{_where('positions', where)}|r| = {worst:g} {units.length} is outside the "
            f"plausible band [{lo:g}, {hi:g}] for {centre}-centred positions in "
            f"{units.length}. {detail}"
        )
    return r


def check_velocities(velocities, *, units: UnitSystem, speed_range: tuple[float, float],
                     centre: str = "earth", where: str = "") -> np.ndarray:
    """Validate a batch of velocity vectors against the interface's unit system
    (``units``) and its plausible speed band, expressed in
    :attr:`UnitSystem.length` per :attr:`UnitSystem.time`."""
    v = _as_float_array(velocities, "velocities", where)
    if v.ndim == 0 or v.shape[-1:] != (3,):
        raise PhysicsIOError(
            f"{_where('velocities', where)}velocities must be a (3,) or [...,3] array of "
            f"{units.length} per {units.time}, got shape {v.shape}"
        )
    speeds = _magnitudes(v)
    if not np.all(np.isfinite(speeds)):
        raise PhysicsIOError(
            f"{_where('velocities', where)}velocities contain non-finite values -- a nan/inf "
            "in a state vector is an error, not a sample."
        )
    lo, hi = speed_range
    bad = (speeds < lo) | (speeds > hi)
    if np.any(bad):
        worst = float(np.max(speeds[bad])) if np.any(speeds[bad] > hi) else float(np.min(speeds[bad]))
        detail = _unit_diagnosis(speeds[bad], units.length, speed_range, LENGTH_UNITS)
        raise UnitMismatchError(
            f"{_where('velocities', where)}|v| = {worst:g} {units.length}/{units.time} is "
            f"outside the plausible band [{lo:g}, {hi:g}] for {centre}-centred velocities. "
            f"{detail}"
        )
    return v


def check_matching_shapes(positions, velocities, *, where: str = "") -> None:
    """Require positions and velocities to describe the same set of samples."""
    r = _as_float_array(positions, "positions", where)
    v = _as_float_array(velocities, "velocities", where)
    if r.shape != v.shape:
        raise PhysicsIOError(
            f"{_where('state', where)}positions and velocities must have the same shape, got "
            f"{r.shape} vs {v.shape}"
        )


# ---------------------------------------------------------------------------
# The per-module declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsIOSpec:
    """What one physics interface accepts: its frame vocabulary, its epoch
    format, its unit system, the body at the origin, and the plausibility bands
    for positions and velocities in those units.

    An interface declares exactly one of these (module-level constant), checks
    its arguments with :meth:`check_state` / :meth:`check_epoch`, and can be
    described to a run manifest or report with :meth:`describe` -- so "which
    conventions did this module speak" is answerable from the artifact instead
    of from source archaeology.
    """

    name: str
    frames: tuple[str, ...]
    epoch_format: str
    units: UnitSystem
    centre: str
    position_range: tuple[float, float]
    velocity_range: tuple[float, float]
    mu: float | None = None

    def check_frame(self, frame=None, *, where: str = "") -> str:
        """Check a caller-supplied frame against this spec's vocabulary; with
        ``frame=None`` check the spec's own single declared frame (so a module
        that only speaks one frame still routes its convention through the
        layer)."""
        if frame is None:
            if len(self.frames) != 1:
                raise FrameMismatchError(
                    f"{_where('frame', where)}{self.name} accepts {len(self.frames)} frames "
                    f"{self.frames}; pass the label."
                )
            frame = self.frames[0]
        return check_frame(frame, self.frames, where=where or self.name)

    def check_epoch(self, epoch, *, where: str = ""):
        """Check an epoch against this spec's declared format."""
        return check_epoch(epoch, self.epoch_format, where=where or self.name)

    def check_epochs(self, epochs, *, where: str = "") -> np.ndarray:
        """Check an array of epochs against this spec's declared format."""
        return check_epochs(epochs, self.epoch_format, where=where or self.name)

    def check_mu(self, *, where: str = "") -> UnitSystem | None:
        """Check this spec's own declared central-body parameter (if it has
        one) against its declared units."""
        if self.mu is None:
            return None
        return check_grav_parameter(self.mu, where=where or self.name, units=self.units)

    def check_positions(self, positions, *, where: str = "") -> np.ndarray:
        """Check positions against this spec's units and radius band."""
        return check_positions(positions, units=self.units, radius_range=self.position_range,
                              centre=self.centre, where=where or self.name)

    def check_velocities(self, velocities, *, where: str = "") -> np.ndarray:
        """Check velocities against this spec's units and speed band."""
        return check_velocities(velocities, units=self.units, speed_range=self.velocity_range,
                               centre=self.centre, where=where or self.name)

    def check_state(self, r, v, *, frame: str | None = None, epoch=None,
                    where: str = "") -> tuple[np.ndarray, np.ndarray]:
        """Check a state vector (or batch) pair, and optionally its frame label
        and epoch, against this spec. Returns the ``(r, v)`` pair as float64."""
        if frame is not None or len(self.frames) == 1:
            self.check_frame(frame, where=where)
        if epoch is not None:
            self.check_epoch(epoch, where=where)
        check_matching_shapes(r, v, where=where or self.name)
        rr = check_positions(r, units=self.units, radius_range=self.position_range,
                             centre=self.centre, where=where or self.name)
        vv = check_velocities(v, units=self.units, speed_range=self.velocity_range,
                              centre=self.centre, where=where or self.name)
        return rr, vv

    def describe(self) -> dict:
        """This spec as a JSON-friendly dict, for a run manifest or a
        verification report's ``dynamics`` section."""
        return {
            "spec": self.name,
            "frames": list(self.frames),
            "epoch_format": self.epoch_format,
            "units": {"length": self.units.length, "time": self.units.time},
            "central_body": self.centre,
            "position_range": list(self.position_range),
            "velocity_range": list(self.velocity_range),
        }


# --- the declared conventions of orbitopt's physics modules -----------------

#: :func:`orbitopt.dynamics.nbody_gpu.propagate_spacecraft_batch` and the fused
#: backends: Earth-centred ``ECLIPJ2000`` states in km / km per s, epochs as TDB
#: ephemeris seconds since J2000.
NBODY_SCREENING = PhysicsIOSpec(
    name="orbitopt.dynamics.nbody_gpu",
    frames=("ECLIPJ2000",),
    epoch_format="tdb_s_j2000",
    units=UnitSystem("km", "s"),
    centre="earth",
    position_range=(1.0e2, 1.5e8),
    velocity_range=(1.0e-2, 2.0e2),
    mu=398600.4418,
)

#: The two-body/Keppler batch propagator of the same module: same units, and
#: the ``mu`` argument decides which body is meant, so positions are checked in
#: the unit system inferred from it. The position band is an order-of-magnitude
#: guard, not a bound-orbit test: universal-variable propagation is valid for
#: hyperbolic escape states too, so a 4.2e6 km geocentric radius is legal.
KEPLER_BATCH = PhysicsIOSpec(
    name="orbitopt.dynamics.nbody_gpu.propagate_kepler_batch",
    frames=("ECLIPJ2000",),
    epoch_format="tdb_s_j2000",
    units=UnitSystem("km", "s"),
    centre="earth",
    position_range=(1.0e2, 1.5e8),
    velocity_range=(1.0e-2, 2.0e2),
)

#: Flight Lab tracks and the frame bridge: exactly the contract's three frames,
#: metres / m per s, Earth-centred, track epoch a UTC datetime.
FLIGHT_TRACK = PhysicsIOSpec(
    name="orbitopt.flightlab.frames",
    frames=("GCRS", "ECEF", "LAUNCH_ECI"),
    epoch_format="utc_datetime",
    units=UnitSystem("m", "s"),
    centre="earth",
    position_range=(1.0e5, 2.0e9),
    velocity_range=(0.0, 2.0e5),
    mu=3.9860044150e14,
)

#: :mod:`orbitopt.injection` -- Earth-centred J2000 equatorial, metres and m/s,
#: epochs as MJD2000 UTC days.
EARTH_CENTRED_METRES = PhysicsIOSpec(
    name="orbitopt.injection",
    frames=("EME2000", "ECLIPJ2000"),
    epoch_format="mjd2000",
    units=UnitSystem("m", "s"),
    centre="earth",
    position_range=(1.0e5, 2.0e9),
    velocity_range=(0.0, 3.0e5),
    mu=3.9860044150e14,
)

#: :mod:`orbitopt.ground_track` -- Earth-centred inertial positions in metres
#: and MJD2000 UTC epochs; the inertial label is a caller-supplied argument.
GROUND_TRACK = PhysicsIOSpec(
    name="orbitopt.ground_track",
    frames=("ECLIPJ2000", "EME2000", "GCRS", "TEME", "LAUNCH_ECI"),
    epoch_format="mjd2000",
    units=UnitSystem("m", "s"),
    centre="earth",
    position_range=(1.0e5, 2.0e9),
    velocity_range=(0.0, 3.0e5),
    mu=3.9860044150e14,
)

#: The Lambert solvers are unit-agnostic (``mu`` decides), so the spec declares
#: no unit system: the checks infer it from ``mu`` and use it for the position
#: band. Positions/velocities are not band-checked row by row because a batch
#: marks per-row legality itself.
LAMBERT = PhysicsIOSpec(
    name="orbitopt.lambert",
    frames=("ECLIPJ2000", "EME2000", "GCRS"),
    epoch_format="tdb_s_j2000",
    units=UnitSystem("km", "s"),
    centre="sun",
    position_range=(1.0e2, 1.0e7),
    velocity_range=(1.0e-4, 1.0e3),
)

#: The spec a caller can look up by module name (the physics-layer registry).
PHYSICS_IO: dict[str, PhysicsIOSpec] = {
    spec.name: spec for spec in (
        NBODY_SCREENING, KEPLER_BATCH, FLIGHT_TRACK, EARTH_CENTRED_METRES,
        GROUND_TRACK, LAMBERT,
    )
}


def spec_for(module_or_tool: str) -> PhysicsIOSpec:
    """Look up a module's declared :class:`PhysicsIOSpec` by the dotted name it
    registers under (e.g. ``"orbitopt.dynamics.nbody_gpu"``); the way a
    verification harness asks what conventions an interface speaks."""
    try:
        return PHYSICS_IO[module_or_tool]
    except KeyError as exc:
        raise KeyError(
            f"no physics I/O spec registered for {module_or_tool!r}; registered: "
            f"{', '.join(sorted(PHYSICS_IO))}"
        ) from exc


def describe_all() -> dict[str, dict]:
    """Every registered spec as JSON-friendly dicts -- the inventory of frames,
    epochs, timescales and units orbitopt's physics modules exchange in."""
    return {name: spec.describe() for name, spec in sorted(PHYSICS_IO.items())}
