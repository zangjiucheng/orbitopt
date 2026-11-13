"""M-02 (task #220): ascent -> parking-orbit handoff (orbitopt.mission.ascent).

The fast tests gate the transform on a hand-built GCRS insertion run (exact
values, every declared tolerance exercised on both sides), the timescale
declarations (UTC/TT/TDB compare to the same instant via TDB seconds), and
the LAUNCH_ECI bridge on astropy (skipped when astropy is not installed,
like the frame-bridge reference tests). The recorded-OT-stub test flies the
real FL-004 adapter through the contract reader and checks the handoff
against the ascent's own terminal state end to end."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import (
    BodyTrack,
    Event,
    FlightRun,
    load_target,
    read_run,
)
from orbitopt.mission.ascent import (
    ElementTolerances,
    ascent_handoff_state,
    check_ascent_to_parking_orbit,
    epoch_at_ascent_time,
    parking_orbit_elements,
)
from orbitopt.mission.contract import (
    Epoch,
    HandoffState,
    HandoffTolerances,
    state_from_dict,
    state_to_dict,
)
from orbitopt.mission import (
    AscentHandoffReport,
    ascent_handoff_state as pkg_ascent_handoff_state,
    check_ascent_to_parking_orbit as pkg_check,
    parking_orbit_elements as pkg_elements,
)
from orbitopt.timescales import SECONDS_PER_DAY

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"

EPOCH_UTC = "2026-02-14T10:00:00Z"
SECO_T_S = 540.0
INCLINATION_DEG = 51.6
R_SEC_M = 7_000_000.0
MU = 3.9860044150e14  # same EARTH_MU the insertion elements use
V_CIRC = math.sqrt(MU / R_SEC_M)


def _ascent_run(frame: str = "GCRS", r=(R_SEC_M, 0.0, 0.0), v=None, mass_kg: float = 4200.0) -> FlightRun:
    """A hand-built upper-stage insertion: circular orbit, `frame` track with
    a seco event at the only sample."""
    if v is None:
        inc = math.radians(INCLINATION_DEG)
        v = (0.0, V_CIRC * math.cos(inc), V_CIRC * math.sin(inc))
    track = BodyTrack(
        body_id="stage2", parent_body_id="stage1", frame=frame, epoch_utc=EPOCH_UTC,
        provenance="handbuilt", t_s=(SECO_T_S,), mass_kg=(mass_kg,),
        position_m=(tuple(r),), velocity_m_s=(tuple(v),),
    )
    return FlightRun(
        run_id="m02-hand", target_sha256="0" * 64, vehicle_sha256="0" * 64,
        git_sha="deadbeef", created_utc=EPOCH_UTC,
        engines={"ascent": "opentsiolkovsky-1.0.0"},
        tracks=(track,),
        events=(Event(t_s=SECO_T_S, kind="seco", body_id="stage2"),),
    )


def _check(report, check_id):
    return next(c for c in report.checks if c.id == check_id)


class TestHandoffStateConversion:
    def test_handoff_state_is_the_seco_state_gcrs_utc(self):
        state = ascent_handoff_state(_ascent_run())
        assert isinstance(state, HandoffState)
        assert state.frame == "GCRS"
        assert state.epoch.timescale == "UTC"
        assert state.r_m == (R_SEC_M, 0.0, 0.0)
        assert state.v_m_s == pytest.approx((0.0, V_CIRC * math.cos(math.radians(INCLINATION_DEG)),
                                             V_CIRC * math.sin(math.radians(INCLINATION_DEG))))
        assert state.mass_kg == 4200.0
        assert state.metadata["run_id"] == "m02-hand"
        assert state.metadata["ascent_frame"] == "GCRS"
        assert state.stage_config["body_id"] == "stage2"

    def test_package_reexports_are_the_same_functions(self):
        assert pkg_ascent_handoff_state is ascent_handoff_state
        assert pkg_check is check_ascent_to_parking_orbit
        assert pkg_elements is parking_orbit_elements

    def test_state_survives_the_handoff_schema_round_trip(self):
        state = ascent_handoff_state(_ascent_run())
        parsed = state_from_dict(state_to_dict(state))
        assert parsed.r_m == state.r_m
        assert parsed.v_m_s == state.v_m_s
        assert parsed.mass_kg == state.mass_kg
        assert parsed.epoch.timescale == "UTC"
        assert parsed.epoch.delta_seconds(state.epoch) == 0.0

    def test_handoff_epoch_is_the_track_epoch_plus_t_on_the_utc_label(self):
        from orbitopt.timescales import utc_datetime_to_mjd2000

        epoch = epoch_at_ascent_time(EPOCH_UTC, SECO_T_S)
        assert epoch.mjd2000 == pytest.approx(utc_datetime_to_mjd2000(
            __import__("datetime").datetime(2026, 2, 14, 10, 0, 0)) + SECO_T_S / SECONDS_PER_DAY)

    def test_epoch_timescale_declares_the_label_not_a_time_shift(self):
        utc = epoch_at_ascent_time(EPOCH_UTC, SECO_T_S, "UTC")
        tdb = epoch_at_ascent_time(EPOCH_UTC, SECO_T_S, "TDB")
        tt = epoch_at_ascent_time(EPOCH_UTC, SECO_T_S, "TT")
        assert utc.timescale == "UTC" and tdb.timescale == "TDB" and tt.timescale == "TT"
        # Same instant: cross-scale comparisons (via TDB seconds) to the
        # microsecond, even though the labels differ by ~69 s (leap seconds)
        # plus the TDB-TT light-time term (< 1.7 ms).
        assert abs(utc.delta_seconds(tdb)) < 1e-6
        assert abs(utc.delta_seconds(tt)) < 1e-6
        assert abs(tdb.delta_seconds(tt)) < 1.7e-3
        assert tdb.to_utc_mjd2000() == pytest.approx(utc.mjd2000, abs=1e-6)

    def test_unknown_timescale_is_rejected(self):
        with pytest.raises(ValueError, match="timescale"):
            epoch_at_ascent_time(EPOCH_UTC, SECO_T_S, "TAI")

    def test_track_without_seco_falls_back_to_the_run_end(self):
        run = _ascent_run()
        no_seco = FlightRun(**{**run.__dict__, "events": ()})
        state = ascent_handoff_state(no_seco)
        assert state.r_m == (R_SEC_M, 0.0, 0.0)


class TestLaunchEciBridge:
    def test_launch_eci_track_is_bridged_to_gcrs(self):
        pytest.importorskip("astropy", reason="the frame bridge needs astropy")
        from datetime import datetime, timezone

        import numpy as np

        from orbitopt.flightlab.frames import from_gcrs

        inc = math.radians(INCLINATION_DEG)
        v = (0.0, V_CIRC * math.cos(inc), V_CIRC * math.sin(inc))
        epoch = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        r_le, v_le = from_gcrs([tuple(v0 for v0 in (R_SEC_M, 0.0, 0.0))], [v],
                               "LAUNCH_ECI", epoch, [SECO_T_S])
        run = _ascent_run(frame="LAUNCH_ECI", r=tuple(r_le[0]), v=tuple(v_le[0]))
        state = ascent_handoff_state(run)
        assert state.frame == "GCRS"
        assert state.metadata["ascent_frame"] == "LAUNCH_ECI"
        assert np.allclose(state.r_m, (R_SEC_M, 0.0, 0.0), atol=1e-3)
        assert np.allclose(state.v_m_s, v, atol=1e-3)
        report = check_ascent_to_parking_orbit(run, state)
        assert report.passed, report.failures


class TestParkingOrbitElements:
    def test_elements_recomputed_from_the_handoff(self):
        state = ascent_handoff_state(_ascent_run())
        elements = parking_orbit_elements(state)
        assert elements["semi_major_axis_m"] == pytest.approx(R_SEC_M, rel=1e-9)
        assert elements["eccentricity"] < 1e-9
        assert elements["inclination_deg"] == pytest.approx(INCLINATION_DEG, abs=1e-9)
        assert elements["periapsis_altitude_m"] == pytest.approx(R_SEC_M - 6_378_137.0, rel=1e-6)

    def test_ecef_state_is_refused(self):
        run = _ascent_run()
        state = ascent_handoff_state(run)
        ecef = HandoffState(**{**state.__dict__, "frame": "ECEF"})
        with pytest.raises(ValueError, match="ECEF"):
            parking_orbit_elements(ecef)


class TestAscentToParkingOrbitGate:
    def test_clean_handoff_passes_every_declared_tolerance(self):
        run = _ascent_run()
        state = ascent_handoff_state(run)
        report = check_ascent_to_parking_orbit(run, state)
        assert isinstance(report, AscentHandoffReport)
        assert report.passed and not report.failures
        epoch = _check(report, "handoff.epoch_monotonic")
        assert abs(epoch.value) < 1e-6  # same instant, UTC vs UTC
        assert _check(report, "handoff.position_continuity").value == 0.0
        assert _check(report, "handoff.mass_conservation").value == 0.0
        for cid in ("ascent.semi_major_axis", "ascent.eccentricity", "ascent.inclination",
                    "ascent.raan", "ascent.periapsis", "ascent.apoapsis"):
            assert _check(report, cid).passed

    def test_tdb_declared_handoff_passes_the_epoch_gate(self):
        run = _ascent_run()
        state = ascent_handoff_state(run, timescale="TDB")
        report = check_ascent_to_parking_orbit(run, state)
        assert report.passed, report.failures
        epoch = _check(report, "handoff.epoch_monotonic")
        assert abs(epoch.value) < 1e-6  # UTC vs TDB labels of one instant

    def test_altered_velocity_fails_the_gate_and_elements(self):
        run = _ascent_run()
        state = ascent_handoff_state(run)
        bad = HandoffState(**{**state.__dict__,
                              "v_m_s": tuple(c + 0.05 / 3 for c in state.v_m_s)})
        report = check_ascent_to_parking_orbit(run, bad)
        assert not report.passed
        assert not _check(report, "handoff.velocity_continuity").passed
        agree = _check(report, "ascent.elements_agree")
        assert not agree.passed
        assert agree.reason_code == "elements_not_assessed_handoff_failed"
        # and the elements themselves disagree once recomputed:
        e_bad = parking_orbit_elements(bad)["eccentricity"]
        assert abs(e_bad - report.ascent_elements["eccentricity"]) > 1e-6

    def test_frame_mismatch_fails_without_measuring(self):
        run = _ascent_run()
        state = ascent_handoff_state(run)
        bad = HandoffState(**{**state.__dict__, "frame": "LAUNCH_ECI"})
        report = check_ascent_to_parking_orbit(run, bad)
        assert not report.passed
        assert _check(report, "handoff.position_continuity").reason_code.startswith("frame_mismatch")
        assert _check(report, "ascent.elements_agree").reason_code.startswith("frame_mismatch")

    def test_mass_gain_fails_the_gate(self):
        run = _ascent_run()
        state = ascent_handoff_state(run)
        bad = HandoffState(**{**state.__dict__, "mass_kg": state.mass_kg + 5.0})
        report = check_ascent_to_parking_orbit(run, bad, HandoffTolerances())
        assert not report.passed
        assert _check(report, "handoff.mass_conservation").reason_code == "mass_gain"

    def test_element_tolerance_is_declared_not_fudged(self):
        tol = ElementTolerances(semi_major_axis_m=1e-9)
        run = _ascent_run()
        state = ascent_handoff_state(run)
        # exact copy: even the absurd tolerance holds for a transform that
        # carries the identical vectors...
        assert _check(check_ascent_to_parking_orbit(run, state,
                                                    element_tolerances=tol),
                      "ascent.semi_major_axis").passed
        # ...and a state nudged 1 mm radially trips it while the position
        # gate (1 m) still passes.
        nudged = HandoffState(**{**state.__dict__, "r_m": (state.r_m[0] + 1e-3, *state.r_m[1:])})
        report = check_ascent_to_parking_orbit(run, nudged, element_tolerances=tol)
        assert _check(report, "handoff.position_continuity").passed
        assert not _check(report, "ascent.semi_major_axis").passed


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


class TestRecordedAscentRegression:
    """The regression: fly the recorded OT run through the real adapter,
    hand it off, and verify epoch/frame/r/v/mass and the orbit elements
    against the ascent's own terminal state."""

    def test_stub_ascent_handoff_round_trips(self, stub, tmp_path):
        from orbitopt.flightlab import checks as gate_checks
        from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

        run = run_ascent(load_target(TARGET), VEHICLE, None, tmp_path / "fl004")
        state = ascent_handoff_state(run)

        assert state.frame == "GCRS" and state.epoch.timescale == "UTC"
        assert state.metadata["run_id"] == run.run_id

        ins = gate_checks.insertion_state(run)
        assert state.mass_kg == pytest.approx(ins["mass_kg"], rel=0, abs=1e-9)
        # r/v continuous to the micrometre against the ascent's own state
        # (the same GCRS vectors, no frame re-rotation anywhere).
        drift = math.dist(state.r_m, tuple(ins["position_m"]))
        assert drift < 1e-6
        assert math.dist(state.v_m_s, tuple(ins["velocity_m_s"])) < 1e-6

        report = check_ascent_to_parking_orbit(run, state)
        assert report.passed, report.failures
        assert abs(_check(report, "handoff.epoch_monotonic").value) < 1e-6

        # Parking orbit recomputed from the handoff == ascent terminal point.
        ref = gate_checks.orbital_elements(ins["position_m"], ins["velocity_m_s"])
        elements = parking_orbit_elements(state)
        assert elements["semi_major_axis_m"] == pytest.approx(
            ref["semi_major_axis_m"], abs=1e-3)
        assert elements["inclination_deg"] == pytest.approx(ref["inclination_deg"], abs=1e-9)

        # ...and the on-disk FlightResult re-reads to the identical handoff.
        again = ascent_handoff_state(read_run(tmp_path / "fl004"))
        assert again == state
