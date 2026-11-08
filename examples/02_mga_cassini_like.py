"""GPU-screen Earth->Venus launch windows, then optimize a full
Earth-Venus-Venus-Earth-Jupiter (Cassini-like) MGA-1DSM trajectory with
pygmo, seeding the optimizer's t0 range from the GPU-screened windows.

Run: python examples/02_mga_cassini_like.py
"""
from __future__ import annotations

import pygmo as pg

from orbitopt.bodies import mjd2000_from_date
from orbitopt.problems.mga import build_mga_1dsm, rank_windows_by_full_mga, screen_departure_windows


def main():
    t0_range = (mjd2000_from_date(2025, 1, 1), mjd2000_from_date(2027, 1, 1))

    windows = screen_departure_windows(
        "earth", "venus", t0_range, tof_range=(80.0, 250.0),
        n_t0=300, n_tof=300, top_k=5,
    )
    print("GPU-screened Earth->Venus launch windows (distinct first-leg basins):")
    for w in windows:
        print(f"  t0={w['t0_mjd2000']:.1f} MJD2000  tof={w['tof_days']:.1f}d  "
              f"dv={w['total_dv_km_s']:.3f} km/s")

    sequence = ["earth", "venus", "venus", "earth", "jupiter"]
    tof_range = [[50, 400], [50, 400], [50, 400], [500, 2000]]

    # Don't commit to the lowest first-leg dv: give every screened window a
    # short, independent full-trajectory run and pick by the full objective.
    ranked = rank_windows_by_full_mga(sequence, windows, tof_range, pop_size=40, generations=60, seed=42)
    print("\nWindows re-ranked by full MGA-1DSM objective (short independent runs):")
    for r in ranked:
        print(f"  t0={r['t0_mjd2000']:.1f}  first-leg rank #{r['first_leg_rank']}  "
              f"full objective={r['full_objective']:.1f} m/s")

    best_t0 = ranked[0]["t0_mjd2000"]
    narrowed_t0_range = (best_t0 - 30.0, best_t0 + 30.0)

    udp = build_mga_1dsm(
        sequence=sequence,
        t0_range=narrowed_t0_range,
        tof_range=tof_range,
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
