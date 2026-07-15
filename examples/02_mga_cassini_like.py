"""GPU-screen Earth->Venus launch windows, then optimize a full
Earth-Venus-Venus-Earth-Jupiter (Cassini-like) MGA-1DSM trajectory with
pygmo, seeding the optimizer's t0 range from the GPU-screened windows.

Run: python examples/02_mga_cassini_like.py
"""
from __future__ import annotations

import pygmo as pg

from orbitopt.bodies import mjd2000_from_date
from orbitopt.problems.mga import build_mga_1dsm, screen_departure_windows


def main():
    t0_range = (mjd2000_from_date(2025, 1, 1), mjd2000_from_date(2027, 1, 1))

    windows = screen_departure_windows(
        "earth", "venus", t0_range, tof_range=(80.0, 250.0),
        n_t0=300, n_tof=300, top_k=5,
    )
    print("GPU-screened Earth->Venus launch windows (lowest first-leg dv):")
    for w in windows:
        print(f"  t0={w['t0_mjd2000']:.1f} MJD2000  tof={w['tof_days']:.1f}d  "
              f"dv={w['total_dv_km_s']:.3f} km/s")

    best_t0 = windows[0]["t0_mjd2000"]
    narrowed_t0_range = (best_t0 - 30.0, best_t0 + 30.0)

    udp = build_mga_1dsm(
        sequence=["earth", "venus", "venus", "earth", "jupiter"],
        t0_range=narrowed_t0_range,
        tof_range=[[50, 400], [50, 400], [50, 400], [500, 2000]],
        vinf_range=(0.5, 4.5),
    )

    prob = pg.problem(udp)
    pop = pg.population(prob, size=200, seed=42)
    algo = pg.algorithm(pg.sade(gen=200))
    algo.set_verbosity(20)
    pop = algo.evolve(pop)

    print(f"\nBest total dv: {pop.champion_f[0]:.3f} m/s")
    print(f"Best decision vector: {pop.champion_x}")


if __name__ == "__main__":
    main()
