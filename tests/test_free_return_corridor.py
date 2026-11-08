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
