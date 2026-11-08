"""Checks for orbitopt.optimize.multistart (multi-basin global search) on a
synthetic two-basin landscape -- no pykep/tudatpy needed.

The landscape is built to defeat the old single-basin recipe on purpose: a
wide, shallow valley owns the coarse grid's minimum, while a narrow, deeper
valley falls between grid nodes and so looks worse on the grid. Taking only
the grid minimum and shrinking the optimizer box to +-15/+-30 around it
(exactly what mars_transfer._search_transfer used to do) can never reach the
deeper valley; the multistart search must.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.core.problem import OrbitOptProblem
from orbitopt.optimize.multistart import (
    SINGLE_BASIN_HALF_WIDTH,
    Basin,
    Candidate,
    basin_bounds,
    find_diverse_basins,
    merge_candidates,
    multistart_optimize,
)

WIDE = (30.0, 50.0, 3.0, 20.0)  # (x, y, depth, width)
NARROW = (120.0, 150.0, 4.0, 3.0)
FULL_BOUNDS = ((0.0, 150.0), (0.0, 200.0))


def landscape(x, y):
    out = 10.0
    for cx, cy, depth, width in (WIDE, NARROW):
        out = out - depth * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / width ** 2))
    return out


def coarse_grid():
    xg = np.linspace(*FULL_BOUNDS[0], 16)
    yg = np.linspace(*FULL_BOUNDS[1], 16)
    xx, yy = np.meshgrid(xg, yg)
    return landscape(xx, yy), xg, yg


class Landscape(OrbitOptProblem):
    def __init__(self, x_bounds, y_bounds):
        super().__init__(use_gpu=False)
        self.x_bounds, self.y_bounds = x_bounds, y_bounds

    def get_bounds(self):
        return [self.x_bounds[0], self.y_bounds[0]], [self.x_bounds[1], self.y_bounds[1]]

    def fitness(self, dv):
        return [float(landscape(dv[0], dv[1]))]

    def batch_fitness(self, dvs):
        m = np.asarray(dvs, dtype=float).reshape(-1, 2)
        return landscape(m[:, 0], m[:, 1])


def test_find_diverse_basins_keeps_both_valleys_best_first():
    values, xg, yg = coarse_grid()
    basins = find_diverse_basins(values, xg, yg, max_basins=5, min_separation=(20.0, 40.0))
    assert len(basins) == 2
    # The grid ranks the wide valley first even though the narrow one is deeper.
    assert abs(basins[0].x - WIDE[0]) <= 10 and abs(basins[0].y - WIDE[1]) <= 14
    assert abs(basins[1].x - NARROW[0]) <= 10 and abs(basins[1].y - NARROW[1]) <= 14
    assert [b.rank for b in basins] == [0, 1]
    assert basins[0].value < basins[1].value


def test_find_diverse_basins_suppresses_plateau_and_neighbours():
    values = np.full((5, 5), 5.0)
    values[2, 1] = values[2, 2] = 1.0  # flat two-cell valley floor
    values[0, 4] = 3.0  # separate, worse valley in the corner
    grid = np.arange(5, dtype=float)
    basins = find_diverse_basins(values, grid, grid, max_basins=5, min_separation=(2.0, 2.0))
    assert [(b.x, b.y, b.value) for b in basins] == [(1.0, 2.0, 1.0), (4.0, 0.0, 3.0)]


def test_find_diverse_basins_ignores_infeasible_cells_and_ratio_cutoff():
    values = np.array([[np.inf, 4.0, 9.0], [np.inf, 9.0, 9.0], [1.0, 9.0, 2.5]])
    grid = np.arange(3, dtype=float)
    basins = find_diverse_basins(values, grid, grid, min_separation=(1.0, 1.0))
    assert [b.value for b in basins] == [1.0, 2.5, 4.0]
    basins = find_diverse_basins(values, grid, grid, min_separation=(1.0, 1.0), max_value_ratio=3.0)
    assert [b.value for b in basins] == [1.0, 2.5]
    with pytest.raises(ValueError):
        find_diverse_basins(np.full((2, 2), np.inf), [0, 1], [0, 1])
    with pytest.raises(ValueError):
        find_diverse_basins(np.zeros((2, 3)), [0, 1], [0, 1])


def test_basin_bounds_clips_to_full_box():
    b = Basin(x=5.0, y=190.0, value=0.0, rank=0)
    assert basin_bounds(b, (15.0, 30.0), FULL_BOUNDS) == ((0.0, 20.0), (160.0, 200.0))


def test_merge_candidates_collapses_runs_on_same_optimum():
    def c(x, y, v, rank):
        return Candidate(x=x, y=y, value=v, basin_rank=rank, bounds=FULL_BOUNDS, seed=rank)

    runs = [c(10.0, 10.0, 2.0, 0), c(11.0, 12.0, 1.9, None), c(50.0, 50.0, 1.0, 1), c(0.0, 0.0, np.inf, 2)]
    distinct = merge_candidates(runs, merge_distance=(3.0, 6.0))
    assert [(d.x, d.value, d.merged_runs) for d in distinct] == [(50.0, 1.0, 1), (11.0, 1.9, 2)]
    assert runs[1].merged_runs == 1  # inputs are not mutated


def _exact_local_optimizer(problem, pop_size, generations, seed):
    """Deterministic stand-in for PSO: dense brute force inside the box."""
    (xlo, ylo), (xhi, yhi) = problem.get_bounds()
    xs, ys = np.meshgrid(np.linspace(xlo, xhi, 301), np.linspace(ylo, yhi, 301))
    f = landscape(xs, ys)
    k = np.unravel_index(np.argmin(f), f.shape)
    return np.array([xs[k], ys[k]]), np.array([f[k]]), 0.0


def test_multistart_beats_single_basin_and_records_comparison():
    values, xg, yg = coarse_grid()
    basins = find_diverse_basins(values, xg, yg, max_basins=4, min_separation=(20.0, 40.0))
    report = multistart_optimize(
        Landscape, basins, FULL_BOUNDS, half_width=SINGLE_BASIN_HALF_WIDTH,
        seed=7, include_full_box_run=False, optimize_fn=_exact_local_optimizer,
    )
    # Single-basin baseline = run 0, the old recipe, stuck in the wide valley.
    assert report.baseline is report.runs[0]
    assert report.baseline.value == pytest.approx(10.0 - WIDE[2], abs=1e-3)
    # Multistart finds the deeper narrow valley and more candidates overall.
    assert report.best.value == pytest.approx(10.0 - NARROW[2], abs=1e-2)
    assert report.best.basin_rank == 1
    assert report.improvement == pytest.approx(NARROW[2] - WIDE[2], abs=1e-2)
    assert report.extra_candidates == 1
    assert [r.seed for r in report.runs] == [7, 8]
    assert any("single-basin" in n for n in report.notes)
    text = report.summary(unit="u")
    assert "single-basin baseline" in text and "2 distinct candidates" in text
    d = report.to_dict()
    assert d["improvement"] == pytest.approx(report.improvement)
    assert len(d["candidates"]) == 2


def test_multistart_with_real_pygmo_pso():
    pytest.importorskip("pygmo")
    values, xg, yg = coarse_grid()
    basins = find_diverse_basins(values, xg, yg, max_basins=4, min_separation=(20.0, 40.0))
    report = multistart_optimize(
        Landscape, basins, FULL_BOUNDS, pop_size=40, generations=80, seed=3,
    )
    assert len(report.runs) == len(basins) + 1  # + one full-box run
    assert report.baseline.value > 6.5  # single basin can't leave the wide valley
    assert report.best.value < 6.2
    assert report.improvement > 0.7
