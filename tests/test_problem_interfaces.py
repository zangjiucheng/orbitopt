"""Sanity checks for the pygmo UDP interface: get_bounds()/fitness() shapes,
and that batch_fitness() agrees with looped fitness() calls.
"""
from __future__ import annotations

import numpy as np
import pygmo as pg
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.problems.transfer_2body import Transfer2BodyProblem


def _make_problem():
    return Transfer2BodyProblem(
        departure_body=planet("earth"),
        arrival_body=planet("mars"),
        t0_bounds=(mjd2000_from_date(2025, 1, 1), mjd2000_from_date(2027, 1, 1)),
        tof_bounds=(100.0, 400.0),
    )


def test_pygmo_accepts_the_problem():
    prob = pg.problem(_make_problem())
    assert prob.get_nx() == 2
    assert prob.get_nobj() == 1


def test_batch_fitness_matches_looped_fitness():
    problem = _make_problem()
    rng = np.random.default_rng(0)
    lo, hi = problem.get_bounds()
    dvs = rng.uniform(lo, hi, size=(64, 2))

    looped = np.array([problem.fitness(dv)[0] for dv in dvs])
    batched = problem.batch_fitness(dvs.ravel())

    finite = np.isfinite(looped) & (looped < 1e6) & (batched < 1e6)
    assert finite.sum() > 0
    assert np.allclose(looped[finite], batched[finite], atol=1.0)
