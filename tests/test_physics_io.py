"""The unified physics I/O validation layer (V-01, task #240).

Every physics module hands state vectors, epochs and gravitational parameters to
every other one, and the ways that goes wrong are finite and well known: a
LEO state written in kilometres handed to a metres interface, TDB ephemeris
seconds handed to an MJD2000 UTC argument, an Earth-fixed vector labelled (or
not labelled) "ECI". :mod:`orbitopt.physics_io` is the one place that turns
those into explicit errors, and this file is the contract:

* the common mismatches (km/m, UTC/TDB, ECEF/ECI) raise, with a message naming
  the unit/frame to use instead,
* every registered physics module declares its conventions and routes its I/O
  through the same checks,
* nothing is ever guessed: an ambiguous label is an error, not a default.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitopt import physics_io as pio
from orbitopt.physics_io import (
    EARTH_CENTRED_METRES,
    EPOCH_FORMATS,
    FLIGHT_TRACK,
    GROUND_TRACK,
    KNOWN_MU,
    LAMBERT,
    NBODY_SCREENING,
    PHYSICS_IO,
    EpochMismatchError,
    FrameMismatchError,
    PhysicsIOError,
    TimescaleMismatchError,
    UnitMismatchError,
    UnitSystem,
    check_duration,
    check_epoch,
    check_epochs,
    check_frame,
    check_frame_family,
    check_grav_parameter,
    check_lambert_pair,
    check_matching_shapes,
    check_positions,
    check_timescale,
    check_velocities,
    describe_all,
    distance_band,
    epoch_to_tdb_seconds,
    identify_grav_parameter,
    spec_for,
)

KM_S = UnitSystem("km", "s")
M_S = UnitSystem("m", "s")

#: A 600 km circular-orbit state, in the units the Flight Lab / injection
#: conventions use (metres, m/s).
LEO_R_M = np.array([[6978137.0, 0.0, 0.0]])
LEO_V_M_S = np.array([[0.0, 7515.0, 0.0]])

#: The same state in the units the screening propagators use.
LEO_R_KM = LEO_R_M / 1000.0
LEO_V_KM_S = LEO_V_M_S / 1000.0

EPOCH_MJD2000 = 9804.5           # 2026-11-04 12:00 UTC, a normal mission epoch
EPOCH_TDB_S = 8.38e8             # the same era, ephemeris seconds since J2000


# ---------------------------------------------------------------------------
# error hierarchy: the layer's errors are the module's ValueErrors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc_cls", [FrameMismatchError, TimescaleMismatchError,
                                     EpochMismatchError, UnitMismatchError])
def test_errors_are_value_errors_with_a_message(exc_cls):
    assert issubclass(exc_cls, PhysicsIOError)
    assert issubclass(exc_cls, ValueError)
    assert exc_cls.__doc__.strip()


def test_every_public_check_raises_a_layer_error_not_a_bare_value_error():
    """A caller must be able to catch the whole layer with one class."""
    with pytest.raises(PhysicsIOError):
        check_frame("ECI", ("GCRS",))
    with pytest.raises(PhysicsIOError):
        check_epoch(8.0e8, "mjd2000")
    with pytest.raises(PhysicsIOError):
        check_grav_parameter(-1.0)
    with pytest.raises(PhysicsIOError):
        check_positions(LEO_R_M, units=KM_S, radius_range=(1.0e2, 2.0e6))


# ---------------------------------------------------------------------------
# frames: ECEF vs ECI vs inertial, and the no-silent-default policy
# ---------------------------------------------------------------------------

def test_declared_frames_are_known_and_families_are_populated():
    assert set(pio.FRAMES) >= {"GCRS", "EME2000", "ECLIPJ2000", "TEME", "ECEF", "LAUNCH_ECI"}
    assert pio.FRAMES["GCRS"].family == "inertial"
    assert pio.FRAMES["ECEF"].family == "rotating"
    assert set(pio.ROTATING_FRAMES) == {"ECEF"}
    # An alias can also be a real frame (TEME is both: a frame orbitopt knows,
    # and a near-miss for interfaces whose vocabulary is GCRS/EME2000).
    for name, hint in pio.AMBIGUOUS_FRAMES.items():
        assert hint.strip()
    assert "ECI" in pio.AMBIGUOUS_FRAMES and "ECI" not in pio.FRAMES


def test_frame_check_accepts_exactly_the_declared_vocabulary():
    assert check_frame("ECEF", ("GCRS", "ECEF")) == "ECEF"
    assert GROUND_TRACK.check_frame("TEME") == "TEME"


@pytest.mark.parametrize("bad", ["ECI", "J2000", "gcrs", "Ecef", "", "QWERTY", None, 42])
def test_unknown_frame_raises_and_names_the_allowed_ones(bad):
    with pytest.raises(FrameMismatchError) as exc:
        check_frame(bad, ("GCRS", "ECEF"), where="test")
    msg = str(exc.value)
    assert msg.startswith("test: ")
    assert "no silent default" in msg
    assert "GCRS" in msg


def test_eci_is_rejected_as_ambiguous_and_says_what_to_use():
    with pytest.raises(FrameMismatchError) as exc:
        check_frame("ECI", FLIGHT_TRACK.frames, where="to_gcrs")
    msg = str(exc.value)
    assert "LAUNCH_ECI" in msg and "GCRS" in msg
    assert "family" in msg


def test_frame_family_check_catches_ecef_where_inertial_is_required():
    for good in ("GCRS", "EME2000", "ECLIPJ2000", "TEME"):
        assert check_frame_family(good, "inertial") == good
    with pytest.raises(FrameMismatchError) as exc:
        check_frame_family("ECEF", "inertial", where="eci_to_ecef")
    assert "rotating" in str(exc.value) and "inertial" in str(exc.value)
    with pytest.raises(FrameMismatchError):
        check_frame_family("GCRS", "rotating")


def test_check_frame_rejects_a_frame_the_layer_does_not_know():
    """An interface cannot declare a vocabulary this layer cannot check."""
    with pytest.raises(FrameMismatchError):
        check_frame("GCRS", ("GCRS", "NOT_A_FRAME"))


def test_nbody_frame_vocabulary_is_ecliptic_only():
    with pytest.raises(FrameMismatchError) as exc:
        NBODY_SCREENING.check_frame("GCRS")
    assert "ECLIPJ2000" in str(exc.value)


# ---------------------------------------------------------------------------
# timescales and epochs: UTC vs TDB
# ---------------------------------------------------------------------------

def test_timescale_vocabulary_and_mismatch():
    assert check_timescale("TDB") == "tdb"
    assert check_timescale(" utc ", allowed=("utc", "tai")) == "utc"
    with pytest.raises(TimescaleMismatchError) as exc:
        check_timescale("UTC", allowed=("tdb",), where="epoch")
    msg = str(exc.value)
    assert "utc" in msg and "tdb" in msg
    assert "timescales" in msg  # names the module that converts
    with pytest.raises(TimescaleMismatchError):
        check_timescale("solar_time")


def test_epoch_formats_have_disjoint_magnitude_bands():
    from orbitopt.timescales import utc_mjd2000_to_tt_seconds, utc_mjd2000_to_tdb_seconds
    assert set(EPOCH_FORMATS) == {"mjd2000", "tdb_s_j2000", "tt_s_j2000"}
    assert "utc_datetime" not in EPOCH_FORMATS  # checked by type instead
    hi_days = EPOCH_FORMATS["mjd2000"][1]
    assert hi_days < pio._EPOCH_FORMAT_SPLIT_S  # a day count never reaches 1e5
    for name in ("tdb_s_j2000", "tt_s_j2000"):
        lo, hi, _ = EPOCH_FORMATS[name]
        # What check_epoch accepts for a seconds format is band *and* magnitude,
        # and the two never meet the day-count band.
        assert lo <= -pio._EPOCH_FORMAT_SPLIT_S and hi >= pio._EPOCH_FORMAT_SPLIT_S
    # TDB and TT seconds differ by under 2 ms and cannot be told apart by
    # magnitude -- that is exactly why check_timescale exists.
    assert abs(utc_mjd2000_to_tt_seconds(EPOCH_MJD2000)
               - utc_mjd2000_to_tdb_seconds(EPOCH_MJD2000)) < 2.0e-3


def test_mjd2000_and_tdb_seconds_are_not_interchangeable():
    assert check_epoch(EPOCH_MJD2000) == EPOCH_MJD2000
    assert check_epoch(0.0) == 0.0
    # TDB seconds in an MJD2000 slot: 8.38e8 "days" is year ~2.3 million.
    with pytest.raises(EpochMismatchError) as exc:
        check_epoch(EPOCH_TDB_S, "mjd2000", where="liftoff")
    msg = str(exc.value)
    assert msg.startswith("liftoff: ")
    assert "seconds since J2000" in msg
    assert "utc_mjd2000_to_tdb_seconds" in msg or "tdb_seconds_to_utc_mjd2000" in msg
    # ... and the other direction.
    with pytest.raises(EpochMismatchError) as exc:
        check_epoch(EPOCH_MJD2000, "tdb_s_j2000", where="epoch0")
    assert "day count" in str(exc.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "yesterday", None])
def test_non_numeric_or_non_finite_epochs_raise(bad):
    with pytest.raises(EpochMismatchError):
        check_epoch(bad, "mjd2000")


def test_utc_datetime_format_rejects_numbers():
    epoch = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    assert check_epoch(epoch, "utc_datetime") is epoch
    for bad in (EPOCH_MJD2000, "2026-11-04"):
        with pytest.raises(EpochMismatchError) as exc:
            check_epoch(bad, "utc_datetime", where="to_gcrs")
        assert "datetime" in str(exc.value)


def test_epoch_arrays_are_checked_elementwise():
    assert np.allclose(check_epochs(np.array([0.0, 9800.0, 9813.99])), [0.0, 9800.0, 9813.99])
    grid = np.array([9800.0, 8.2e8, 9801.0])  # one column filled with ephemeris seconds
    with pytest.raises(EpochMismatchError) as exc:
        check_epochs(grid, "mjd2000", where="t_grid")
    assert "epoch[1]" in str(exc.value)
    with pytest.raises(EpochMismatchError):
        check_epochs(np.array([np.nan]))
    # A seconds-format grid may legitimately start at 0 (the propagators' default).
    assert np.allclose(check_epochs(np.array([0.0, EPOCH_TDB_S]), "tdb_s_j2000"),
                       [0.0, EPOCH_TDB_S])


def test_epoch_to_tdb_seconds_agrees_with_timescales():
    from orbitopt import timescales as ts
    assert epoch_to_tdb_seconds(EPOCH_MJD2000) == pytest.approx(
        ts.utc_mjd2000_to_tdb_seconds(EPOCH_MJD2000))
    assert epoch_to_tdb_seconds(EPOCH_TDB_S, "tdb_s_j2000") == EPOCH_TDB_S
    assert epoch_to_tdb_seconds(datetime(2000, 1, 1, 12, tzinfo=timezone.utc), "utc_datetime") \
        == pytest.approx(ts.utc_datetime_to_tdb_seconds(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)))


def test_timescales_reject_an_out_of_range_mjd2000_label():
    """The conversion itself is the last line of defence: a TDB-seconds value
    handed to utc_mjd2000_to_tdb_seconds must not come back as a DeltaAT."""
    from orbitopt import timescales as ts
    with pytest.raises(ValueError) as exc:
        ts.utc_mjd2000_to_tdb_seconds(EPOCH_TDB_S)
    assert "TDB seconds" in str(exc.value)
    assert ts.MJD2000_MIN_UTC_DAYS == EPOCH_FORMATS["mjd2000"][0]
    assert ts.MJD2000_MAX_UTC_DAYS == EPOCH_FORMATS["mjd2000"][1]


def test_as_mjd2000_validates_a_bare_number():
    from orbitopt.launch_windows import as_mjd2000
    assert as_mjd2000(EPOCH_MJD2000) == EPOCH_MJD2000
    assert as_mjd2000("2026-11-04T12:00:00Z") == pytest.approx(EPOCH_MJD2000, abs=1e-4)
    with pytest.raises(EpochMismatchError):
        as_mjd2000(EPOCH_TDB_S)


def test_durations_are_seconds_and_not_astronomical():
    assert check_duration(3600.0) == 3600.0
    assert check_duration(-86400.0) == -86400.0  # backward propagation is real
    assert check_duration(0.0, allow_negative=False) == 0.0
    with pytest.raises(UnitMismatchError):
        check_duration(1.0e13, allow_negative=False)
    with pytest.raises(UnitMismatchError):
        check_duration(-1.0, allow_negative=False)
    with pytest.raises(UnitMismatchError):
        check_duration(float("nan"))


# ---------------------------------------------------------------------------
# units: km vs m, and the mu that decides which system a state is in
# ---------------------------------------------------------------------------

def test_unit_system_conversions_round_trip():
    assert KM_S.grav_parameter(KNOWN_MU["earth"]) == pytest.approx(398600.4418, rel=1e-9)
    assert M_S.grav_parameter(KNOWN_MU["earth"]) == pytest.approx(3.986004418e14, rel=1e-9)
    assert UnitSystem("au", "day").grav_parameter(KNOWN_MU["sun"]) == pytest.approx(
        2.9591220828e-4, rel=1e-6)


def test_positions_in_band_pass_and_km_metres_confusion_is_diagnosed():
    assert check_positions(LEO_R_KM, units=KM_S, radius_range=(1.0e2, 2.0e6)).shape == (1, 3)
    assert check_positions(LEO_R_M, units=M_S, radius_range=(1.0e5, 2.0e9)).shape == (1, 3)
    # A LEO state written in km, handed to a metres interface.
    with pytest.raises(UnitMismatchError) as exc:
        check_positions(LEO_R_KM, units=M_S, radius_range=(1.0e5, 2.0e9), where="r0")
    msg = str(exc.value)
    assert msg.startswith("r0: ")
    assert "km" in msg and "m" in msg
    assert "unit mix-up" in msg and "1000" in msg
    # ... and the same state the other way round (metres into a km interface).
    with pytest.raises(UnitMismatchError) as exc:
        check_positions(LEO_R_M, units=KM_S, radius_range=(1.0e2, 2.0e6), where="r0")
    assert "unit mix-up" in str(exc.value)


def test_a_state_handed_in_au_is_reported_as_au():
    one_au_in_m = np.array([[1.496e11, 0.0, 0.0]])
    with pytest.raises(UnitMismatchError) as exc:
        check_positions(one_au_in_m, units=KM_S, radius_range=(1.0e2, 2.0e6), where="r1")
    assert "au" in str(exc.value)


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], "r0", [[float("nan"), 0.0, 0.0]],
                                 [[float("inf"), 0.0, 0.0]]])
def test_positions_shape_and_finiteness(bad):
    with pytest.raises(PhysicsIOError):
        check_positions(bad, units=M_S, radius_range=(1.0e5, 2.0e9))


def test_velocities_are_checked_in_the_same_unit_system():
    assert check_velocities(LEO_V_KM_S, units=KM_S, speed_range=(1.0e-2, 2.0e2)).shape == (1, 3)
    with pytest.raises(UnitMismatchError) as exc:
        check_velocities(LEO_V_M_S, units=KM_S, speed_range=(1.0e-2, 2.0e2), where="v0")
    assert "m" in str(exc.value) and "km" in str(exc.value)


def test_matching_shapes():
    with pytest.raises(PhysicsIOError) as exc:
        check_matching_shapes(np.zeros((4, 3)), np.zeros((3, 3)), where="state")
    assert "shape" in str(exc.value)


def test_grav_parameter_identifies_body_and_units():
    assert identify_grav_parameter(398600.4418) == ("earth", KM_S)
    assert identify_grav_parameter(3.986004418e14) == ("earth", M_S)
    assert identify_grav_parameter(KNOWN_MU["sun"]) == ("sun", M_S)
    assert identify_grav_parameter(403503.2418) is None  # mu_earth + mu_moon
    assert check_grav_parameter(1.0) is None  # custom mu: no units inferred


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_grav_parameter_must_be_finite_and_positive(bad):
    with pytest.raises(UnitMismatchError):
        check_grav_parameter(bad)


def test_mu_in_the_wrong_unit_system_raises_naming_both():
    with pytest.raises(UnitMismatchError) as exc:
        check_grav_parameter(3.986004418e14, units=KM_S, where="mu_earth")
    msg = str(exc.value)
    assert msg.startswith("mu_earth: ")
    assert "m^3/s^2" in msg and "km^3/s^2" in msg
    assert "398600.442" in msg  # the value this interface actually wants
    assert check_grav_parameter(398600.4418, units=KM_S) == KM_S


def test_distance_band_is_expressed_in_the_inferred_units():
    lo_km, hi_km = distance_band("earth", KM_S)
    lo_m, hi_m = distance_band("earth", M_S)
    assert (lo_m, hi_m) == pytest.approx((lo_km * 1000.0, hi_km * 1000.0))
    # A Lambert endpoint below the surface is still a legal maneuver problem.
    assert lo_km < 2400.0 < hi_km
    assert distance_band("sun", KM_S)[0] < 1.0e8 < distance_band("sun", KM_S)[1]


# ---------------------------------------------------------------------------
# the declared conventions, and that each module actually enforces them
# ---------------------------------------------------------------------------

def test_specs_are_registered_and_serialisable():
    assert set(PHYSICS_IO) == {
        "orbitopt.dynamics.nbody_gpu",
        "orbitopt.dynamics.nbody_gpu.propagate_kepler_batch",
        "orbitopt.flightlab.frames", "orbitopt.injection",
        "orbitopt.ground_track", "orbitopt.lambert",
    }
    assert spec_for("orbitopt.dynamics.nbody_gpu") is NBODY_SCREENING
    with pytest.raises(KeyError) as exc:
        spec_for("orbitopt.mystery")
    assert "orbitopt.lambert" in str(exc.value)
    dumped = json.dumps(describe_all())
    assert "ECLIPJ2000" in dumped and "mjd2000" in dumped
    for name, spec in PHYSICS_IO.items():
        d = json.loads(dumped)[name]
        assert d["spec"] == name
        assert d["units"] == {"length": spec.units.length, "time": spec.units.time}
        assert set(spec.frames) <= set(pio.FRAMES)


def test_flightlab_spec_matches_the_track_contract():
    from orbitopt.flightlab.contract import FRAMES as CONTRACT_FRAMES
    assert tuple(FLIGHT_TRACK.frames) == tuple(CONTRACT_FRAMES)
    assert FLIGHT_TRACK.units == M_S
    assert FLIGHT_TRACK.epoch_format == "utc_datetime"


def test_flightlab_bridge_routes_frames_through_the_layer():
    from orbitopt.flightlab.frames import UnknownFrameError, to_gcrs
    t = np.zeros(2)
    r = np.array([[6.678e6, 0.0, 0.0], [6.678e6, 0.0, 0.0]])
    v = np.array([[0.0, 7.8e3, 0.0], [0.0, 7.8e3, 0.0]])
    epoch = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(UnknownFrameError) as exc:
        to_gcrs(r, v, "ECI", epoch, t)
    assert "no silent default" in str(exc.value) and "LAUNCH_ECI" in str(exc.value)
    # UnknownFrameError is the layer's frame error, so one except catches both.
    assert issubclass(UnknownFrameError, FrameMismatchError)


def test_flightlab_bridge_rejects_a_track_in_kilometres():
    from orbitopt.flightlab.frames import to_gcrs
    t = np.zeros(2)
    r_km = LEO_R_M / 1000.0
    r_km = np.repeat(r_km, 2, axis=0)
    v_km = np.repeat(LEO_V_M_S / 1000.0, 2, axis=0)
    epoch = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(UnitMismatchError) as exc:
        to_gcrs(r_km, v_km, "GCRS", epoch, t)
    assert "unit mix-up" in str(exc.value)


def test_flightlab_bridge_still_checks_shapes_before_units():
    from orbitopt.flightlab.frames import to_gcrs
    epoch = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError) as exc:
        to_gcrs(np.zeros((3, 2)), np.zeros((3, 2)), "GCRS", epoch, np.zeros(3))
    assert "[N,3]" in str(exc.value)


def test_ground_track_rejects_an_ephemeris_seconds_epoch_column():
    from orbitopt.ground_track import eci_to_ecef
    r = np.repeat(LEO_R_M, 3, axis=0)
    with pytest.raises(EpochMismatchError):
        eci_to_ecef(r, np.full(3, EPOCH_TDB_S))
    with pytest.raises(EpochMismatchError):
        eci_to_ecef(r, [9800.0, EPOCH_TDB_S, 9802.0])
    out = eci_to_ecef(r, np.full(3, EPOCH_MJD2000))
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(r, axis=1))


def test_nbody_screening_rejects_si_mu_and_a_mislabelled_epoch():
    from orbitopt.dynamics import nbody_gpu as nb
    kw = dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2)
    with pytest.raises(UnitMismatchError) as exc:
        nb.propagate_spacecraft_batch(LEO_R_KM, LEO_V_KM_S, 3600.0, 10,
                                      mu_earth=3.986004418e14, mu_moon=nb.MU_MOON_KM3_S2)
    assert "mu_earth" in str(exc.value) and "km^3/s^2" in str(exc.value)
    with pytest.raises(EpochMismatchError):
        nb.propagate_spacecraft_batch(LEO_R_KM, LEO_V_KM_S, 3600.0, 10, epoch0=EPOCH_MJD2000, **kw)
    # A batch of states in metres with a km/s mu: caught, not propagated.
    with pytest.raises(UnitMismatchError) as exc:
        nb.propagate_spacecraft_batch(LEO_R_M, LEO_V_M_S, 3600.0, 10, **kw)
    assert "km" in str(exc.value)


def test_fused_screening_inputs_share_the_nbody_contract():
    """The fused kernel backend takes the same arguments as the vectorised
    screening propagator, so it must enforce the same I/O contract -- and it
    must do it in ``prepare_inputs``, before any kernel or SPICE work."""
    from orbitopt.dynamics import nbody_fused as fused
    from orbitopt.dynamics import nbody_gpu as nb
    kw = dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2)
    with pytest.raises(UnitMismatchError) as exc:
        fused.prepare_inputs(LEO_R_M, LEO_V_M_S, 3600.0, 10, **kw)
    assert "prepare_inputs" in str(exc.value) and "unit mix-up" in str(exc.value)
    with pytest.raises(UnitMismatchError) as exc:
        fused.prepare_inputs(LEO_R_KM, LEO_V_KM_S, 3600.0, 10,
                             mu_earth=3.986004418e14, mu_moon=nb.MU_MOON_KM3_S2)
    assert "mu_earth" in str(exc.value)
    with pytest.raises(EpochMismatchError):
        fused.prepare_inputs(LEO_R_KM, LEO_V_KM_S, 3600.0, 10,
                             epoch0=np.array([EPOCH_MJD2000]), **kw)
    with pytest.raises(EpochMismatchError):
        fused.prepare_inputs(LEO_R_KM, LEO_V_KM_S, 3600.0, 10,
                             moon_seed_epoch_ephemeris_seconds=EPOCH_MJD2000, **kw)


def test_kepler_batch_rejects_states_that_disagree_with_mu():
    from orbitopt.dynamics import nbody_gpu as nb
    with pytest.raises(UnitMismatchError):
        nb.propagate_kepler_batch(LEO_R_M, LEO_V_M_S, 3600.0, nb.MU_EARTH_KM3_S2, use_gpu=False)
    r, v = nb.propagate_kepler_batch(LEO_R_KM, LEO_V_KM_S, 3600.0, nb.MU_EARTH_KM3_S2, use_gpu=False)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))
    # An unknown mu (a custom central mass) means no units can be inferred, so
    # the state check is skipped rather than guessed.
    r2, _ = nb.propagate_kepler_batch(LEO_R_M, LEO_V_M_S, 3600.0, 1.0e5, use_gpu=False)
    assert r2.shape == (1, 3)


def test_moon_seed_epoch_is_an_epoch_not_a_day_count():
    from orbitopt.dynamics import nbody_gpu as nb
    with pytest.raises(EpochMismatchError) as exc:
        nb.get_moon_state0(EPOCH_MJD2000)
    assert "seed_epoch_ephemeris_seconds" in str(exc.value)


def test_single_lambert_solve_checks_its_endpoints_against_mu():
    r1 = np.array([1.0e7, 0.0, 0.0])          # metres, matching pk.MU_SUN
    r2 = np.array([0.0, 1.5e11, 0.0])
    check_lambert_pair(r1, r2, KNOWN_MU["sun"])   # explicit: the layer accepts it
    check_lambert_pair(r1, r2, 1.0)               # custom mu: nothing to check
    with pytest.raises(UnitMismatchError) as exc:
        check_lambert_pair(r1 / 1000.0, r2 / 1000.0, KNOWN_MU["sun"], where="solve_lambert_single")
    assert "unit mix-up" in str(exc.value)
    with pytest.raises(UnitMismatchError):
        check_lambert_pair(np.zeros(3), r2, KNOWN_MU["sun"])


def test_lambert_solver_entry_points_raise_for_a_bad_mu():
    from orbitopt.lambert import solve_lambert_batch, solve_lambert_single
    r1 = np.array([[1.0e7, 0.0, 0.0]])
    r2 = np.array([[0.0, 1.5e11, 0.0]])
    for mu in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            solve_lambert_batch(r1, r2, np.array([3.0e7]), mu, use_gpu=False)
        with pytest.raises(ValueError):
            solve_lambert_single(r1[0], r2[0], 3.0e7, mu=mu)


def test_lambert_batch_contract_unchanged_per_row():
    """The batch solver marks illegal rows itself; the I/O layer must not turn a
    bad row into an exception (documented contract of solve_lambert_batch)."""
    from orbitopt.lambert import solve_lambert_batch
    r1 = np.array([[1.0e7, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]])
    r2 = np.array([[0.0, 1.5e11, 0.0], [0.0, 1.5e11, 0.0], [0.0, 1.5e11, 0.0]])
    res = solve_lambert_batch(r1, r2, np.array([3.0e7, 3.0e7, -5.0]), KNOWN_MU["sun"],
                              use_gpu=False, cpu_fallback=False)
    assert res.valid.tolist() == [True, False, False]


def test_injection_parking_orbit_validates_epoch_and_mu():
    from orbitopt.injection import ParkingOrbit
    ok = ParkingOrbit(altitude_km=200.0, inclination_deg=28.5, raan_deg=40.0,
                      epoch_mjd2000=EPOCH_MJD2000)
    state = ok.state_at(EPOCH_MJD2000)
    assert np.allclose(np.linalg.norm(state[:3]), 6578137.0, rtol=1e-6)
    with pytest.raises(EpochMismatchError):
        ParkingOrbit(altitude_km=200.0, inclination_deg=28.5, raan_deg=40.0, epoch_mjd2000=EPOCH_TDB_S)
    with pytest.raises(UnitMismatchError):
        ParkingOrbit(altitude_km=200.0, inclination_deg=28.5, raan_deg=40.0,
                     epoch_mjd2000=EPOCH_MJD2000, mu_m3_s2=398600.4418)


def test_specs_describe_their_own_mu_consistently():
    assert NBODY_SCREENING.check_mu() == KM_S
    assert EARTH_CENTRED_METRES.check_mu() == M_S
    assert LAMBERT.mu is None and LAMBERT.check_mu() is None
