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


def _eph_batch(body, epochs_mjd2000):
    """Query ``body.eph`` once per entry of ``epochs_mjd2000`` (1-D array).
    Returns ``(r, v)`` arrays of shape (len(epochs), 3) in SI units."""
    m = len(epochs_mjd2000)
    r = np.empty((m, 3))
    v = np.empty((m, 3))
    for k in range(m):
        r[k], v[k] = body.eph(pk.epoch(float(epochs_mjd2000[k])))
    return r, v


def _eph_unique(body, epochs_mjd2000):
    """Exact ephemeris lookup for an arbitrary array of epochs, querying the
    body only once per *distinct* epoch and scattering back. Returns arrays
    of shape ``epochs.shape + (3,)``."""
    flat = np.asarray(epochs_mjd2000, dtype=float).ravel()
    uniq, inverse = np.unique(flat, return_inverse=True)
    r_u, v_u = _eph_batch(body, uniq)
    return r_u[inverse], v_u[inverse]


def _eph_cubic(body, epochs_mjd2000, step_days):
    """Approximate ephemeris lookup by pre-sampling ``body`` on a uniform
    epoch grid of spacing <= ``step_days`` covering ``epochs`` and evaluating
    4-point (cubic) Lagrange interpolation of position and velocity
    *independently*.

    Velocity is deliberately not used as the derivative of position (no
    Hermite): pykep's ``jpl_lp`` model returns the osculating Keplerian
    velocity, which differs from the true d(r)/dt by ~0.5 m/s because the
    secular element rates are ignored. A Hermite fit then has O(h) position
    error (~5 km at h = 1 day for Mars) and ~1 m/s velocity error. Value-only
    cubic interpolation is O(h^4) in both: measured at h = 1 day, Mars is
    ~80 m / ~1.5e-5 m/s and Earth ~470 m / ~1e-4 m/s (h = 0.5 day: ~5 m /
    ~1e-6 m/s for Mars). Position errors of that size are ~1e-9 relative and
    negligible for Lambert (tests/test_porkchop.py enforces the budget).

    Returns ``(r, v, n_samples)``.
    """
    flat = np.asarray(epochs_mjd2000, dtype=float).ravel()
    t_min, t_max = float(flat.min()), float(flat.max())
    span = t_max - t_min
    n_samples = max(int(np.ceil(span / step_days)) + 1, 4)
    nodes = np.linspace(t_min, t_max, n_samples)
    r_s, v_s = _eph_batch(body, nodes)

    h_days = span / (n_samples - 1) if span > 0 else 1.0
    # Stencil nodes k-1, k, k+1, k+2; clipped at the ends (s then falls in
    # [-1, 0) or (1, 2], still a valid cubic interpolant through 4 samples).
    k = np.clip(np.floor((flat - t_min) / h_days).astype(int), 1, n_samples - 3)
    s = ((flat - nodes[k]) / h_days)[:, None]
    w_m1 = -s * (s - 1) * (s - 2) / 6.0
    w_0 = (s + 1) * (s - 1) * (s - 2) / 2.0
    w_1 = -(s + 1) * s * (s - 2) / 2.0
    w_2 = (s + 1) * s * (s - 1) / 6.0

    def interp(y):
        return w_m1 * y[k - 1] + w_0 * y[k] + w_1 * y[k + 1] + w_2 * y[k + 2]

    return interp(r_s), interp(v_s), n_samples


def compute_porkchop(
    departure_body,
    arrival_body,
    t0_range,
    tof_range,
    n_t0=200,
    n_tof=200,
    mu=pk.MU_SUN,
    use_gpu=None,
    arrival_interp_step=None,
) -> PorkchopGrid:
    """Evaluate a full (t0 x tof) porkchop grid with one batched GPU/CPU
    Lambert solve of n_t0 * n_tof independent problems.

    Ephemeris lookups are deduplicated: the departure body is queried once
    per departure epoch (n_t0 calls, broadcast over all TOFs) instead of once
    per grid cell, and the arrival body once per distinct arrival epoch.

    ``arrival_interp_step`` (days, optional): if given, the arrival body is
    instead pre-sampled every <= ``arrival_interp_step`` days across the
    arrival window and evaluated by cubic Lagrange interpolation (position
    and velocity independently), cutting arrival calls to ~window/step.
    ~1 day keeps the error at the ~100 m / ~1e-4 m/s level for planets. The
    default (None) keeps arrival ephemerides exact.
    """
    t0_grid = np.linspace(t0_range[0], t0_range[1], n_t0)
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)
    tt0, ttof = np.meshgrid(t0_grid, tof_grid)  # both (n_tof, n_t0)
    shape = tt0.shape

    t0_flat = tt0.ravel()
    tof_flat = ttof.ravel()
    t_arr_flat = t0_flat + tof_flat

    # Departure: one query per departure epoch, broadcast across the TOF axis
    # (meshgrid row-major ravel => column j of every row is t0_grid[j]).
    r1_dep, v1_dep = _eph_batch(departure_body, t0_grid)
    r1 = np.broadcast_to(r1_dep, shape + (3,)).reshape(-1, 3)
    v1p = np.broadcast_to(v1_dep, shape + (3,)).reshape(-1, 3)

    # Arrival: exact per distinct epoch, or cubic interpolation if that is
    # requested and actually cheaper.
    r2 = v2p = None
    if arrival_interp_step is not None:
        if arrival_interp_step <= 0:
            raise ValueError("arrival_interp_step must be positive (days)")
        span = float(t_arr_flat.max() - t_arr_flat.min())
        n_samples = max(np.ceil(span / arrival_interp_step) + 1, 4)
        if n_samples < np.unique(t_arr_flat).size:
            r2, v2p, _ = _eph_cubic(arrival_body, t_arr_flat, arrival_interp_step)
    if r2 is None:
        r2, v2p = _eph_unique(arrival_body, t_arr_flat)

    result = solve_lambert_batch(
        r1, r2, tof_flat * pk.DAY2SEC, mu, use_gpu=use_gpu,
    )

    v_dep_excess = (result.v1 - v1p) / 1000.0  # km/s
    v_arr_excess = (result.v2 - v2p) / 1000.0  # km/s

    c3 = np.sum(v_dep_excess**2, axis=1)
    v_inf_arr = np.linalg.norm(v_arr_excess, axis=1)
    total_dv = np.sqrt(c3) + v_inf_arr

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
