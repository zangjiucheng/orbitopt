"""Optimize departure epoch + time of flight for an Earth->Mars Lambert
transfer using pygmo's SGA, with every candidate in every generation
evaluated through the GPU-batched Lambert solver.

Run: python examples/01_earth_mars_lambert_gpu.py
"""
from __future__ import annotations

import pygmo as pg

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.transfer_2body import Transfer2BodyProblem


def main():
    earth = planet("earth")
    mars = planet("mars")

    t0_lo = mjd2000_from_date(2025, 1, 1)
    t0_hi = mjd2000_from_date(2027, 1, 1)

    problem = Transfer2BodyProblem(
        departure_body=earth,
        arrival_body=mars,
        t0_bounds=(t0_lo, t0_hi),
        tof_bounds=(100.0, 400.0),
    )

    x, f, elapsed = run_optimization(
        problem,
        algorithm=pg.pso_gen(gen=150),  # supports set_bfe -> every generation is GPU-batched
        pop_size=512,
        generations=150,
        seed=42,
    )

    print(f"\nBest departure t0 = {x[0]:.2f} MJD2000, tof = {x[1]:.2f} days")
    print(f"Total delta-v      = {f[0]:.3f} m/s")
    print(f"Wall time          = {elapsed:.2f}s")


if __name__ == "__main__":
    main()
