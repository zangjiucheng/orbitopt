"""orbitopt.timescales: leap-second-aware UTC <-> TT <-> TDB, and the
cross-module check that every epoch API in orbitopt agrees on it."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orbitopt import timescales as ts
from orbitopt.launch_windows import as_mjd2000, datetime_to_mjd2000, mjd2000_to_datetime


def _mjd(iso: str) -> float:
    return ts.utc_datetime_to_mjd2000(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc))


@pytest.mark.parametrize("iso, tt_minus_utc", [
    ("1972-01-01T00:00:00", 42.184),
    ("1999-12-31T23:59:59", 64.184),
    ("2000-01-01T12:00:00", 64.184),
    ("2005-12-31T23:59:59", 64.184),
    ("2006-01-01T00:00:00", 65.184),
    ("2016-12-31T23:59:59", 68.184),
    ("2017-01-01T00:00:00", 69.184),
    ("2026-11-08T00:00:00", 69.184),
])
def test_tt_minus_utc_follows_leap_seconds(iso, tt_minus_utc):
    assert ts.tt_minus_utc(_mjd(iso)) == pytest.approx(tt_minus_utc)


def test_rejects_pre_1972_and_non_finite():
    with pytest.raises(ValueError):
        ts.tai_minus_utc(_mjd("1971-12-31T00:00:00"))
    with pytest.raises(ValueError):
        ts.tai_minus_utc(float("nan"))


def test_tdb_seconds_match_spice_reference_values():
    # SPICE str2et (naif0012.tls) reference values.
    assert ts.utc_mjd2000_to_tdb_seconds(0.5) == pytest.approx(64.183927, abs=1e-6)
    # 2017-01-01T00:00:00 UTC = 6210 days after MJD2000 0.
    assert ts.utc_mjd2000_to_tdb_seconds(6210.0) == pytest.approx(
        (6210.0 - 0.5) * 86400.0 + 69.184, abs=2e-3)
    assert abs(ts.tdb_minus_tt(0.0)) < 1.7e-3


@pytest.mark.parametrize("mjd", [0.5, 1234.567, 6209.9, 6210.1, 9808.25])
def test_tdb_round_trip(mjd):
    et = ts.utc_mjd2000_to_tdb_seconds(mjd)
    assert ts.tdb_seconds_to_utc_mjd2000(et) == pytest.approx(mjd, abs=1e-6 / 86400.0)


def test_datetime_round_trip_and_launch_windows_delegates():
    dt = datetime(2020, 7, 30, 11, 50, tzinfo=timezone.utc)
    mjd = datetime_to_mjd2000(dt)
    assert mjd == ts.utc_datetime_to_mjd2000(dt) == as_mjd2000("2020-07-30T11:50:00Z")
    assert mjd2000_to_datetime(mjd) == dt
    assert ts.tdb_seconds_to_utc_datetime(ts.utc_datetime_to_tdb_seconds(dt)) == dt


# --------------------------------------------------------------------------
# Cross-module consistency: the same input gives the same epoch everywhere.
# --------------------------------------------------------------------------

EPOCHS = [0.5, 2402.0, 7516.75, _mjd("2026-11-08T00:00:00")]


@pytest.mark.parametrize("mjd", EPOCHS)
def test_injection_matches_timescales(mjd):
    pytest.importorskip("scipy")
    from orbitopt.injection import mjd2000_to_ephemeris_seconds

    assert mjd2000_to_ephemeris_seconds(mjd) == ts.utc_mjd2000_to_tdb_seconds(mjd)


@pytest.mark.parametrize("mjd", EPOCHS)
def test_bodies_matches_injection(mjd):
    pytest.importorskip("scipy")
    pytest.importorskip("tudatpy")
    pytest.importorskip("pykep")
    from orbitopt import bodies, injection

    assert bodies.mjd2000_to_ephemeris_seconds(mjd) == injection.mjd2000_to_ephemeris_seconds(mjd)


def _tudat_utc_to_tdb(mjd: float) -> float:
    tc = pytest.importorskip("tudatpy.astro.time_conversion")
    try:
        converter = tc.default_time_scale_converter()
        utc_seconds = (mjd - 0.5) * 86400.0  # UTC seconds past 2000-01-01 12:00 UTC
        return converter.convert_time(
            input_scale=tc.TimeScales.utc_scale, output_scale=tc.TimeScales.tdb_scale,
            input_value=utc_seconds)
    except AttributeError as exc:  # older tudatpy without the scale converter
        pytest.skip(f"tudatpy has no time-scale converter: {exc}")


@pytest.mark.parametrize("mjd", EPOCHS)
def test_matches_tudat_leap_second_aware_conversion(mjd):
    # tudat uses the full SOFA TDB-TT series; SPICE's DELTET model differs by
    # tens of microseconds at most.
    assert ts.utc_mjd2000_to_tdb_seconds(mjd) == pytest.approx(_tudat_utc_to_tdb(mjd), abs=1e-4)


@pytest.mark.parametrize("mjd", EPOCHS)
def test_matches_erfa(mjd):
    erfa = pytest.importorskip("erfa")
    tai1, tai2 = erfa.utctai(2451544.5, mjd)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    tt_seconds = ((tt1 - 2451545.0) + tt2) * 86400.0
    assert ts.utc_mjd2000_to_tt_seconds(mjd) == pytest.approx(tt_seconds, abs=1e-5)
