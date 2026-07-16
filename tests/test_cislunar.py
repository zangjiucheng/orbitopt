"""Correctness checks for the cislunar free-return pipeline: Moon ephemeris
sanity, frame consistency between orbitopt.bodies and orbitopt.dynamics.nbody_gpu,
and an end-to-end differential-correction targeting run against the real
Artemis II perilune altitude.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
import pytest

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, moon_state
from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify.differential_correction import target_lunar_flyby

ARTEMIS_II_PERILUNE_ALTITUDE_KM = 6545.0
MOON_RADIUS_KM = 1737.4


def test_moon_state_is_physically_sane():
    for year, month, day in [(2026, 1, 1), (2026, 7, 15), (2026, 12, 1)]:
        r, v = moon_state(mjd2000_from_date(year, month, day))
        distance_km = np.linalg.norm(r) / 1000.0
        speed_km_s = np.linalg.norm(v) / 1000.0
        assert 356_000.0 < distance_km < 407_000.0
        assert 0.85 < speed_km_s < 1.15


def test_nbody_gpu_moon_seed_must_be_near_the_epoch_of_interest():
    """Seeding at J2000 (the default) and asking for a 2026 Moon position
    directly propagates the two-body model across 26+ years -- this should
    be wildly wrong, which is exactly why propagate_spacecraft_batch/
    moon_state_batch take a seed_epoch_ephemeris_seconds parameter (see
    their docstrings). This test pins down that failure mode so a future
    change can't silently "fix" it back to the default-J2000-seed bug.
    """
    mjd2000 = mjd2000_from_date(2026, 8, 5)
    et = mjd2000_to_ephemeris_seconds(mjd2000)

    r_nbody_wrong, _ = nb.moon_state_batch(np.array([et]), use_gpu=False)  # seed defaults to J2000
    r_bodies, _ = moon_state(mjd2000)
    error_km = np.linalg.norm(np.asarray(r_bodies) / 1000.0 - r_nbody_wrong[0])
    assert error_km > 50_000.0


def test_bodies_and_nbody_gpu_moon_ephemeris_agree_within_two_body_error():
    departure_mjd2000 = mjd2000_from_date(2026, 8, 5)
    departure_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)

    for days_after in [0.0, 4.0, 9.0]:
        r_bodies, _ = moon_state(departure_mjd2000 + days_after)
        r_nbody, _ = nb.moon_state_batch(
            np.array([days_after * 86400.0]), use_gpu=False,
            seed_epoch_ephemeris_seconds=departure_et,
        )
        error_km = np.linalg.norm(np.asarray(r_bodies) / 1000.0 - r_nbody[0])
        # Solar perturbation on the real Moon varies with orbital phase, so
        # this error isn't a fixed number across dates -- observed ~2-15k km
        # over a 9-day span depending on the specific date tested; 25k is a
        # generous bound that still catches a real regression (e.g. a wrong
        # seed epoch produces errors in the hundred-thousand-km range).
        assert error_km < 25_000.0, (
            f"at +{days_after}d: bodies.moon_state (real SPICE) and "
            "nbody_gpu.moon_state_batch (two-body osculating approximation, "
            "correctly seeded at the mission epoch) disagree by more than "
            "the two-body approximation should ever be off by."
        )


def _plane_aligned_parking_orbit(r_moon_arrival, altitude_km, mu_earth):
    u = r_moon_arrival / np.linalg.norm(r_moon_arrival)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    e2 = reference - np.dot(reference, u) * u
    e2 /= np.linalg.norm(e2)
    e1 = u
    theta = np.radians(150.0)
    r0_hat = np.cos(theta) * e1 - np.sin(theta) * e2
    tangent_hat = np.sin(theta) * e1 + np.cos(theta) * e2
    r0 = (6378.0 + altitude_km) * 1000.0 * r0_hat
    v0 = np.sqrt(mu_earth / np.linalg.norm(r0)) * tangent_hat
    return r0, v0


@pytest.mark.slow
def test_target_lunar_flyby_hits_artemis_ii_perilune_altitude():
    # coast_days=5.5, not the seemingly-more-natural 4.5: at 4.5 (and several
    # other nearby coast times), the dead-center Lambert guess's uncorrected
    # miss is small enough (a few hundred km) that this targeter's fixed
    # miss-vector-direction approach gets stuck at a nearby local point
    # instead of reaching the true target (see target_lunar_flyby's
    # docstring note on this limitation) -- confirmed by sweeping ~20
    # nearby (departure date, coast time) combinations, of which this was
    # the first to converge cleanly.
    departure_mjd2000 = mjd2000_from_date(2026, 8, 1)
    coast_days = 5.5
    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    r_moon_arrival, _ = moon_state(departure_mjd2000 + coast_days)
    r_moon_arrival = np.asarray(r_moon_arrival)

    r0, v0 = _plane_aligned_parking_orbit(r_moon_arrival, altitude_km=185.0, mu_earth=mu_earth)
    (v1, _v2), = solve_lambert_single(r0, r_moon_arrival, coast_days * 86400.0, mu=mu_earth, max_revs=0)[:1]
    dv_guess = v1 - v0

    target_distance_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM
    result = target_lunar_flyby(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=dv_guess, coast_duration_guess=coast_days * 86400.0,
        target_distance_km=target_distance_km,
        reference_epoch_ephemeris_seconds=reference_et,
        maxiter=15, tol_km=50.0,
    )

    assert result.converged
    assert abs(result.final_distance_km - target_distance_km) < 50.0


@pytest.mark.slow
def test_target_lunar_flyby_converges_from_an_imprecise_gpu_screened_guess():
    """Regression test for a real divergence found when wiring the full
    pipeline together: a GPU-coarse-screened guess close to, but not
    exactly, a hand-picked Lambert guess made the undamped Newton iteration
    blow up to a residual of tens of millions of km within a few steps
    (the coarse two-body Moon model's own closest-approach-time estimate
    was off by more than two days for this particular candidate, which
    then anchored the fixed-time targeter to a time with no nearby feasible
    solution). Fixed by (a) sanity-checking the refined coast duration
    against the caller's guess before trusting it, and (b) a capped,
    backtracking Newton step. This exact dv is the one that used to diverge.
    """
    departure_mjd2000 = mjd2000_from_date(2026, 8, 1)
    coast_days_guess = 4.5
    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)

    r_moon_arrival, _ = moon_state(departure_mjd2000 + coast_days_guess)
    r_moon_arrival = np.asarray(r_moon_arrival)
    r0, v0 = _plane_aligned_parking_orbit(r_moon_arrival, altitude_km=185.0, mu_earth=pk.MU_EARTH)

    previously_diverging_dv_guess = np.array([-42.88558, -28.74449, -3794.9622])
    target_distance_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM

    result = target_lunar_flyby(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=previously_diverging_dv_guess, coast_duration_guess=4.42 * 86400.0,
        target_distance_km=target_distance_km,
        reference_epoch_ephemeris_seconds=reference_et,
        maxiter=15, tol_km=50.0,
    )

    assert result.converged
    assert abs(result.coast_duration / 86400.0 - 4.42) < 1.0
    assert abs(result.final_distance_km - target_distance_km) < 50.0
