"""Checks for the GTO->GEO orbit-raising UDP: the pygmo interface, batch/loop
agreement, closed-form sanity (GEO constants, single-impulse dv), the encoding's
terminal-orbit-is-GEO invariant, and that the optimizer finds a finite-burn-
feasible insertion whose burn count matches the ~5 of a real GOES campaign.
"""
from __future__ import annotations

import numpy as np
import pygmo as pg
import pytest

from orbitopt.problems.geo_raising import (
    GeoRaisingProblem,
    geostationary_radius_km,
    single_impulse_geo_insertion_ms,
)

# GOES-16-like injection (see docs/goes_gto_geo_mission_plan.md)
GOES_GTO = dict(gto_perigee_km=8108.0, gto_apogee_km=35286.0, gto_inclination_deg=10.6)


def test_geostationary_constants():
    r_geo = geostationary_radius_km()
    assert abs(r_geo - 42164.17) < 1.0
    v_geo = np.sqrt(398600.4418 / r_geo)
    assert abs(v_geo - 3.0747) < 1e-3


def test_pygmo_accepts_the_problem():
    prob = pg.problem(GeoRaisingProblem(n_burns=4, **GOES_GTO))
    assert prob.get_nx() == 2 * (4 - 1)
    assert prob.get_nobj() == 1


def test_requires_at_least_two_burns():
    with pytest.raises(ValueError):
        GeoRaisingProblem(n_burns=1, **GOES_GTO)


def test_batch_fitness_matches_looped_fitness():
    problem = GeoRaisingProblem(n_burns=5, max_dv_per_burn_ms=217.0, **GOES_GTO)
    rng = np.random.default_rng(0)
    lo, hi = problem.get_bounds()
    dvs = rng.uniform(lo, hi, size=(64, len(lo)))

    looped = np.array([problem.fitness(dv)[0] for dv in dvs])
    batched = problem.batch_fitness(dvs.ravel())
    assert np.allclose(looped, batched, atol=1e-6)


def test_single_impulse_dv_is_physical():
    dv = single_impulse_geo_insertion_ms(**GOES_GTO)
    # ~1.0 km/s for this low-inclination, near-GEO injection (vs ~1.5 km/s from
    # a standard 27 deg GTO) -- and well under the propellant ceiling.
    assert 900.0 < dv < 1100.0
    cap = GeoRaisingProblem(n_burns=2, **GOES_GTO).tsiolkovsky_capacity_ms()
    assert dv < cap < 2100.0


def test_encoding_terminal_orbit_is_circular_equatorial_geo():
    problem = GeoRaisingProblem(n_burns=5, **GOES_GTO)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.random(len(problem.get_bounds()[0]))
        d = problem.decode(x)
        # last burn always circularizes at GEO and zeroes inclination
        assert abs(d["perigee_after_km"][-1] - d["r_geo_km"]) < 1e-6
        assert abs(d["inclination_after_deg"][-1]) < 1e-9
        # plane changes sum to the initial inclination; dv bookkeeping consistent
        assert abs(d["plane_change_deg"].sum() - 10.6) < 1e-6
        assert abs(d["dv_per_burn_ms"].sum() - d["dv_total_ms"]) < 1e-6


@pytest.mark.slow
def test_optimizer_finds_finite_burn_feasible_geo_insertion():
    from orbitopt.optimize.runner import run_optimization

    # Per-pass cap from thrust/mass * a 41-min apogee dwell (~217 m/s). With a
    # ~999 m/s total, ceil(999/217) = 5 apogee burns are the minimum feasible --
    # 4 or fewer cannot fit under the cap for any split (min total is fixed).
    cap = 217.0
    problem = GeoRaisingProblem(n_burns=5, max_dv_per_burn_ms=cap, **GOES_GTO)
    x, f, _ = run_optimization(problem, pop_size=120, generations=150, seed=1, verbose=False)
    d = problem.decode(x)

    assert d["feasible"]
    assert d["max_burn_ms"] <= cap + 1.0
    assert 950.0 < d["dv_total_ms"] < 1100.0
    assert d["dv_total_ms"] < problem.tsiolkovsky_capacity_ms()

    # 4 burns cannot be made feasible: min achievable total (~999) over 4 burns
    # forces a max burn >= ~250 m/s > cap.
    four = GeoRaisingProblem(n_burns=4, max_dv_per_burn_ms=cap, **GOES_GTO)
    x4, _, _ = run_optimization(four, pop_size=120, generations=150, seed=1, verbose=False)
    assert not four.decode(x4)["feasible"]


@pytest.mark.slow
def test_tudatpy_verification_reaches_geo():
    """Re-fly the screening champion under J2+J22+Sun+Moon and refine the final
    burn: it should land on circular GEO (a, e) with only a small correction and
    a small residual plane error."""
    from orbitopt.optimize.runner import run_optimization
    from orbitopt.verify.geo_insertion import verify_geo_raising

    problem = GeoRaisingProblem(n_burns=5, max_dv_per_burn_ms=217.0, **GOES_GTO)
    x, _, _ = run_optimization(problem, pop_size=120, generations=150, seed=1, verbose=False)

    result = verify_geo_raising(problem, x, step_size=150.0, stationkeeping_days=1.0)

    assert result.converged  # reachable circular-GEO targets (a, e) hit
    assert abs(result.achieved_corrected["a_km"] - result.r_geo_km) < 2.0
    assert result.achieved_corrected["e"] < 2e-4
    assert result.achieved_corrected["i_deg"] < 0.5  # residual plane error is small
    # the high-fidelity refinement of the final burn is a small nudge, not a resolve
    assert np.linalg.norm(result.corrected_final_burn_ms - result.ideal_final_burn_ms) < 30.0
