"""GPU-accelerated porkchop-plot grid generation.

A porkchop plot sweeps a 2D grid of (departure epoch, time of flight) and
evaluates one Lambert problem per grid cell -- classically the single most
expensive routine step in mission-design screening, and an ideal fit for
batched GPU evaluation since every cell is independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pykep as pk

from orbitopt.lambert.gpu_batch import solve_lambert_batch


@dataclass
class PorkchopGrid:
    t0_grid: np.ndarray  # (n_t0,) MJD2000
    tof_grid: np.ndarray  # (n_tof,) days
    c3_departure: np.ndarray  # (n_tof, n_t0) km^2/s^2
    v_inf_arrival: np.ndarray  # (n_tof, n_t0) km/s
    total_dv: np.ndarray  # (n_tof, n_t0) km/s
    converged: np.ndarray  # (n_tof, n_t0) bool


def compute_porkchop(
    departure_body,
    arrival_body,
    t0_range,
    tof_range,
    n_t0=200,
    n_tof=200,
    mu=pk.MU_SUN,
    use_gpu=None,
) -> PorkchopGrid:
    """Evaluate a full (t0 x tof) porkchop grid with one batched GPU/CPU
    Lambert solve of n_t0 * n_tof independent problems.
    """
    t0_grid = np.linspace(t0_range[0], t0_range[1], n_t0)
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)
    tt0, ttof = np.meshgrid(t0_grid, tof_grid)  # both (n_tof, n_t0)
    n = tt0.size

    t0_flat = tt0.ravel()
    tof_flat = ttof.ravel()

    r1 = np.empty((n, 3))
    v1p = np.empty((n, 3))
    r2 = np.empty((n, 3))
    v2p = np.empty((n, 3))
    for i in range(n):
        r1[i], v1p[i] = departure_body.eph(pk.epoch(float(t0_flat[i])))
        r2[i], v2p[i] = arrival_body.eph(pk.epoch(float(t0_flat[i] + tof_flat[i])))

    result = solve_lambert_batch(
        r1, r2, tof_flat * pk.DAY2SEC, mu, use_gpu=use_gpu,
    )

    v_dep_excess = (result.v1 - v1p) / 1000.0  # km/s
    v_arr_excess = (result.v2 - v2p) / 1000.0  # km/s

    c3 = np.sum(v_dep_excess**2, axis=1)
    v_inf_arr = np.linalg.norm(v_arr_excess, axis=1)
    total_dv = np.sqrt(c3) + v_inf_arr

    shape = tt0.shape
    return PorkchopGrid(
        t0_grid=t0_grid,
        tof_grid=tof_grid,
        c3_departure=c3.reshape(shape),
        v_inf_arrival=v_inf_arr.reshape(shape),
        total_dv=total_dv.reshape(shape),
        converged=result.converged.reshape(shape),
    )


def plot_porkchop(grid: PorkchopGrid, title="Porkchop plot", levels=40, ax=None):
    """Contour-plot total delta-v (km/s) over the (t0, tof) grid. Returns the
    matplotlib Axes. Import of matplotlib is local so headless/GPU-only use
    of this module doesn't require a display backend.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    dv = np.where(grid.converged, grid.total_dv, np.nan)
    cs = ax.contourf(grid.t0_grid, grid.tof_grid, dv, levels=levels, cmap="viridis")
    ax.set_xlabel("Departure epoch [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title(title)
    plt.colorbar(cs, ax=ax, label="Total delta-v [km/s]")
    return ax
