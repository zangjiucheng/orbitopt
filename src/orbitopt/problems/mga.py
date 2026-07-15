"""Multi-gravity-assist, one-deep-space-manoeuvre-per-leg transfer problem:
a thin, framework-conformant wrapper around pykep.trajopt.mga_1dsm.

This demonstrates that orbitopt is not limited to the toy single-leg problem
in orbitopt.problems.transfer_2body: pykep's own trajopt UDPs (mga, mga_1dsm,
pl2pl_N_impulses, ...) plug in directly. Their fitness() is pykep's compiled
C++ implementation, so there is no batch_fitness() to GPU-accelerate here --
the GPU payoff for this problem class instead comes from pre-screening the
launch-window search space with a batched porkchop grid (see
screen_departure_windows) before handing a narrowed t0 range to the
(CPU-bound, pygmo-driven) global optimizer.
"""
from __future__ import annotations

import numpy as np
import pykep as pk
from pykep.trajopt import mga_1dsm

from orbitopt.bodies import planet
from orbitopt.viz.porkchop import compute_porkchop


def build_mga_1dsm(
    sequence,
    t0_range,
    tof_range,
    vinf_range=(0.5, 4.5),
    add_vinf_dep=False,
    add_vinf_arr=True,
    multi_objective=False,
):
    """Build a pykep.trajopt.mga_1dsm UDP for a flyby ``sequence`` (list of
    body names, e.g. ["earth", "venus", "earth", "mars"]).

    Use directly with pygmo:
        prob = pg.problem(build_mga_1dsm(...))
        pop = pg.population(prob, size=...)
        pop = pg.algorithm(pg.sade(gen=...)).evolve(pop)
    """
    bodies = [planet(name) for name in sequence]
    return mga_1dsm(
        seq=bodies,
        t0=[pk.epoch(t0_range[0]), pk.epoch(t0_range[1])],
        tof=tof_range,
        vinf=list(vinf_range),
        add_vinf_dep=add_vinf_dep,
        add_vinf_arr=add_vinf_arr,
        multi_objective=multi_objective,
    )


def screen_departure_windows(
    departure_name,
    first_flyby_name,
    t0_range,
    tof_range,
    n_t0=400,
    n_tof=400,
    top_k=10,
    use_gpu=None,
):
    """GPU-batch-evaluate the first leg (departure -> first flyby body) of an
    MGA sequence over a dense (t0, tof) grid, and return the ``top_k``
    lowest-total-dv (t0, tof) cells as a narrowed search window.

    Intended use: call this before build_mga_1dsm to shrink t0_range to a
    handful of promising launch windows, so the (comparatively expensive,
    CPU-only) global optimizer over the full multi-leg problem starts from a
    much smaller, pre-validated basin instead of searching blind.
    """
    departure_body = planet(departure_name)
    flyby_body = planet(first_flyby_name)

    grid = compute_porkchop(
        departure_body, flyby_body, t0_range, tof_range,
        n_t0=n_t0, n_tof=n_tof, use_gpu=use_gpu,
    )

    dv = np.where(grid.converged, grid.total_dv, np.inf)
    flat_idx = np.argsort(dv.ravel())[:top_k]
    tof_idx, t0_idx = np.unravel_index(flat_idx, dv.shape)

    return [
        {
            "t0_mjd2000": float(grid.t0_grid[j]),
            "tof_days": float(grid.tof_grid[i]),
            "total_dv_km_s": float(grid.total_dv[i, j]),
        }
        for i, j in zip(tof_idx, t0_idx)
    ]
