"""Generate an Earth->Mars porkchop plot from a 300x300 = 90,000-cell grid,
each cell a full Lambert solve, evaluated in one batched GPU call.

Run: python examples/04_porkchop_gpu.py
Writes: porkchop_earth_mars.png
"""
from __future__ import annotations

import time

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.core.gpu import GPU_AVAILABLE, device_info
from orbitopt.viz.porkchop import compute_porkchop, plot_porkchop


def main():
    print("GPU:", device_info() if GPU_AVAILABLE else "not available, using CPU")

    earth = planet("earth")
    mars = planet("mars")

    t0_range = (mjd2000_from_date(2025, 1, 1), mjd2000_from_date(2027, 1, 1))
    tof_range = (100.0, 400.0)

    t0 = time.perf_counter()
    grid = compute_porkchop(earth, mars, t0_range, tof_range, n_t0=300, n_tof=300)
    elapsed = time.perf_counter() - t0

    n_cells = grid.total_dv.size
    print(f"{n_cells:,} Lambert solves in {elapsed:.2f}s ({n_cells/elapsed:,.0f} solves/s)")
    print(f"Best total dv in grid: {grid.total_dv[grid.converged].min():.3f} km/s")

    ax = plot_porkchop(grid, title="Earth -> Mars porkchop (GPU-batched Lambert)")
    ax.figure.savefig("porkchop_earth_mars.png", dpi=150)
    print("Saved porkchop_earth_mars.png")


if __name__ == "__main__":
    main()
