"""Joint lunar B-plane + Earth-return entry-corridor targeting (task #144):
the Artemis free return must come back through the entry interface inside
the safe lunar-return flight-path-angle corridor, verified by TudatPy.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

from orbitopt.verify.differential_correction import (
    EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
    EARTH_RADIUS_KM,
    LUNAR_RETURN_ENTRY_CORRIDOR_DEG,
    LUNAR_RETURN_ENTRY_FPA_DEG,
    flight_path_angle_deg,
    perigee_radius_for_entry_angle_m,
)

MU_EARTH = 398600.4418e9


@pytest.mark.parametrize("specific_energy", [-6.0e5, -1.0e5, 0.0, 2.0e5])
@pytest.mark.parametrize("fpa_deg", [-5.0, -6.0, -7.5, -20.0])
def test_perigee_radius_for_entry_angle_round_trips_through_the_conic(specific_energy, fpa_deg):
    r_entry = (EARTH_RADIUS_KM + EARTH_ENTRY_INTERFACE_ALTITUDE_KM) * 1000.0
    r_p = perigee_radius_for_entry_angle_m(specific_energy, r_entry, fpa_deg, MU_EARTH)
    assert 0.0 < r_p < r_entry

    # Rebuild the conic from (energy, perigee) and measure its FPA at r_entry.
    v_p = np.sqrt(2.0 * (specific_energy + MU_EARTH / r_p))
    h = r_p * v_p
    v_entry = np.sqrt(2.0 * (specific_energy + MU_EARTH / r_entry))
    measured = -np.degrees(np.arccos(h / (r_entry * v_entry)))  # descending branch
    assert measured == pytest.approx(fpa_deg, abs=1e-9)


def test_flight_path_angle_sign_convention():
    r = np.array([7.0e6, 0.0, 0.0])
    descending = np.concatenate([r, [-np.sin(np.radians(6.0)), np.cos(np.radians(6.0)), 0.0]])
    assert flight_path_angle_deg(descending) == pytest.approx(-6.0)
    assert flight_path_angle_deg(np.concatenate([r, [0.0, 1.0, 0.0]])) == pytest.approx(0.0)


@pytest.mark.slow
def test_artemis_free_return_enters_inside_the_safe_corridor():
    """End-to-end on the Mission Timeline's own default Artemis geometry:
    the joint targeter must hit the Artemis II perilune AND bring the return
    leg through the 121.92 km entry interface inside the corridor -- the
    flight-path angle checked here is measured on an independent TudatPy
    propagation (find_altitude_crossing), not the conic prediction."""
    from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, moon_state
    from orbitopt.lambert.cpu import solve_lambert_single
    from orbitopt.verify import report as rpt
    from orbitopt.verify.differential_correction import target_free_return_entry_corridor
    from orbitopt.viz.mission_timeline import (
        ARTEMIS_II_PERILUNE_ALTITUDE_KM,
        LAMBERT_AIM_OFFSET_KM,
        MOON_RADIUS_KM,
        _plane_aligned_parking_orbit,
    )

    departure = mjd2000_from_date(2026, 8, 1)
    reference_et = mjd2000_to_ephemeris_seconds(departure)
    coast_days = 3.5
    r_moon, v_moon = map(np.asarray, moon_state(departure + coast_days))
    r0, v0 = _plane_aligned_parking_orbit(r_moon, v_moon, 185.0, pk.MU_EARTH)
    h_hat = np.cross(r_moon, v_moon)
    h_hat /= np.linalg.norm(h_hat)
    aim = r_moon + LAMBERT_AIM_OFFSET_KM * 1000.0 * np.cross(h_hat, r_moon / np.linalg.norm(r_moon))
    (v1, _), = solve_lambert_single(r0, aim, coast_days * 86400.0, mu=pk.MU_EARTH, max_revs=0)[:1]

    target_periapsis_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM
    result = target_free_return_entry_corridor(
        r0, v0, 0.0, v1 - v0, coast_days * 86400.0, target_periapsis_km, reference_et,
    )

    assert result.converged
    assert result.entry_flight_path_angle_deg is not None, "return leg never reached the entry interface"
    steepest, shallowest = LUNAR_RETURN_ENTRY_CORRIDOR_DEG
    assert steepest <= result.entry_flight_path_angle_deg <= shallowest
    assert result.in_corridor
    assert result.entry_flight_path_angle_deg == pytest.approx(LUNAR_RETURN_ENTRY_FPA_DEG, abs=0.5)
    assert result.perilune_distance_km == pytest.approx(target_periapsis_km, abs=50.0)
    assert 10_500.0 < result.entry_speed_m_s < 11_500.0  # lunar-return entry speed
    assert result.return_perigee_km < EARTH_RADIUS_KM + EARTH_ENTRY_INTERFACE_ALTITUDE_KM

    rpt.validate_report(result.report)
    checks = result.report["status"]["checks"]
    assert checks["residual.entry_flight_path_angle_in_corridor"] is True
    assert checks["targeting.converged"] is True


def _fake_targeting(*, converged, in_corridor, entry_fpa_deg):
    from orbitopt.verify.differential_correction import FreeReturnTargetingResult

    return FreeReturnTargetingResult(
        delta_v=np.array([3100.0, 0.0, 0.0]), converged=converged, in_corridor=in_corridor,
        iterations=4, coast_duration=3.5 * 86400.0, perilune_epoch=3.5 * 86400.0,
        perilune_distance_km=8282.4, b_plane_angle_deg=0.0,
        return_perigee_km=EARTH_RADIUS_KM + 30.0, target_return_perigee_km=EARTH_RADIUS_KM + 30.0,
        entry_epoch=9.0 * 86400.0, entry_state=None, entry_speed_m_s=11_000.0,
        entry_flight_path_angle_deg=entry_fpa_deg,
    )


def _stub_mission_geometry(monkeypatch, mt):
    # Keep the gate tests independent of SPICE kernels: a fixed Moon state is
    # enough to build the parking orbit and the Lambert seed.
    monkeypatch.setattr(mt, "mjd2000_to_ephemeris_seconds", lambda mjd: 0.0)
    monkeypatch.setattr(mt, "moon_state", lambda mjd: (np.array([3.84e8, 0.0, 0.0]), np.array([0.0, 1022.0, 0.0])))


@pytest.mark.parametrize("converged, in_corridor, fpa", [
    (True, False, -9.5),   # residuals converged, but the entry angle is too steep
    (True, False, None),   # converged, but never reached the entry interface
    (False, True, -6.0),
])
def test_mission_timeline_rejects_targeting_that_is_not_converged_and_in_corridor(monkeypatch, converged, in_corridor, fpa):
    from orbitopt.viz import mission_timeline as mt

    _stub_mission_geometry(monkeypatch, mt)
    monkeypatch.setattr(mt, "target_free_return_entry_corridor",
                        lambda **kw: _fake_targeting(converged=converged, in_corridor=in_corridor, entry_fpa_deg=fpa))

    def must_not_propagate(*a, **kw):
        raise AssertionError("a rejected targeting result must not be propagated/exported")

    monkeypatch.setattr(mt, "propagate_multi_arc", must_not_propagate)
    with pytest.raises(RuntimeError, match="rejected"):
        mt.compute_and_export_mission(departure_mjd2000=9709.0)


def _entry_crossing(fpa_deg):
    from orbitopt.verify.tudat_propagate import AltitudeCrossingResult

    r = (EARTH_RADIUS_KM + EARTH_ENTRY_INTERFACE_ALTITUDE_KM) * 1000.0
    v = 11_000.0
    state = np.array([r, 0.0, 0.0, v * np.sin(np.radians(fpa_deg)), v * np.cos(np.radians(fpa_deg)), 0.0])
    return AltitudeCrossingResult(epoch=9.0 * 86400.0, altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
                                  spacecraft_state=state)


def test_mission_timeline_independently_rejects_an_out_of_corridor_export(monkeypatch):
    """The targeter claims converged + in_corridor, but the full re-propagated
    trajectory enters too steeply: the export must still be refused."""
    from orbitopt.verify.tudat_propagate import PropagationResult
    from orbitopt.viz import mission_timeline as mt

    _stub_mission_geometry(monkeypatch, mt)
    monkeypatch.setattr(mt, "target_free_return_entry_corridor",
                        lambda **kw: _fake_targeting(converged=True, in_corridor=True, entry_fpa_deg=-6.0))
    epochs = np.linspace(0.0, 9.5 * 86400.0, 20)
    states = np.tile([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0], (20, 1))
    monkeypatch.setattr(mt, "propagate_multi_arc",
                        lambda *a, **kw: PropagationResult(epochs=epochs, states=states, final_state=states[-1]))
    monkeypatch.setattr(mt, "find_altitude_crossing", lambda *a, **kw: _entry_crossing(-9.0))
    with pytest.raises(RuntimeError, match="outside the .* entry corridor"):
        mt.compute_and_export_mission(departure_mjd2000=9709.0)

    monkeypatch.setattr(mt, "find_altitude_crossing", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="never reaches"):
        mt.compute_and_export_mission(departure_mjd2000=9709.0)


@pytest.mark.parametrize("fpa, ok", [(-6.0, True), (-5.05, True), (-6.95, True), (-4.5, False), (-7.5, False)])
def test_verify_entry_corridor_bounds(monkeypatch, fpa, ok):
    from orbitopt.viz import mission_timeline as mt

    monkeypatch.setattr(mt, "find_altitude_crossing", lambda *a, **kw: _entry_crossing(fpa))
    if ok:
        assert mt._verify_entry_corridor(object()).spacecraft_state is not None
    else:
        with pytest.raises(RuntimeError, match="outside"):
            mt._verify_entry_corridor(object())
