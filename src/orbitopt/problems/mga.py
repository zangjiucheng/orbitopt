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
from orbitopt.optimize.multistart import find_diverse_basins
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
    diverse=True,
    min_separation=(30.0, 40.0),
):
    """GPU-batch-evaluate the first leg (departure -> first flyby body) of an
    MGA sequence over a dense (t0, tof) grid, and return ``top_k`` candidate
    launch windows (best first-leg total dv first).

    With ``diverse=True`` (default) the windows are ``top_k`` spatially
    distinct local minima of the grid (orbitopt.optimize.multistart.
    find_diverse_basins), i.e. genuinely different launch opportunities.
    ``diverse=False`` keeps the legacy behaviour of the ``top_k`` lowest
    *cells*, which on a dense grid are almost always neighbours inside one
    single valley -- ten copies of the same window.

    The first leg alone is only a screen: it says nothing about whether the
    downstream flybys line up. Rank the returned windows on the full
    multi-leg objective with rank_windows_by_full_mga before committing to
    one, instead of taking ``windows[0]``.
    """
    departure_body = planet(departure_name)
    flyby_body = planet(first_flyby_name)

    grid = compute_porkchop(
        departure_body, flyby_body, t0_range, tof_range,
        n_t0=n_t0, n_tof=n_tof, use_gpu=use_gpu,
    )

    dv = np.where(grid.converged, grid.total_dv, np.inf)
    if diverse:
        basins = find_diverse_basins(dv, grid.t0_grid, grid.tof_grid, max_basins=top_k, min_separation=min_separation)
        return [
            {"t0_mjd2000": b.x, "tof_days": b.y, "total_dv_km_s": b.value}
            for b in basins
        ]

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


def rank_windows_by_full_mga(
    sequence,
    windows,
    tof_range,
    t0_half_width=30.0,
    vinf_range=(0.5, 4.5),
    pop_size=40,
    generations=60,
    seed=0,
    algorithm_factory=None,
    **mga_kwargs,
):
    """Optimize the *full* MGA-1DSM trajectory independently around each
    screened launch window and rank the windows by the full objective.

    This is what keeps a launch window from being chosen or discarded on
    the first leg's dv alone: a window with a slightly worse
    departure->first-flyby leg routinely wins overall because its later
    flybys phase better. One independent pygmo run per window (seed
    ``seed + k``); ``algorithm_factory(generations, seed)`` returns a raw
    pygmo UDA (default ``pg.sade``).

    Returns a list of dicts (best full objective first) with the window's
    screening data plus ``full_objective``, ``champion_x``, ``first_leg_rank``.
    """
    import pygmo as pg

    if algorithm_factory is None:
        def algorithm_factory(gen, s):
            return pg.sade(gen=gen, seed=s)

    ranked = []
    for k, w in enumerate(windows):
        t0 = w["t0_mjd2000"]
        udp = build_mga_1dsm(
            sequence, (t0 - t0_half_width, t0 + t0_half_width), tof_range,
            vinf_range=vinf_range, **mga_kwargs,
        )
        run_seed = int(seed) + k
        pop = pg.population(pg.problem(udp), size=pop_size, seed=run_seed)
        pop = pg.algorithm(algorithm_factory(generations, run_seed)).evolve(pop)
        ranked.append({
            **w,
            "first_leg_rank": k,
            "full_objective": float(pop.champion_f[0]),
            "champion_x": np.asarray(pop.champion_x).tolist(),
        })
    ranked.sort(key=lambda r: r["full_objective"])
    return ranked
