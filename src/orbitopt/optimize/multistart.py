"""Multi-basin ("multistart / diverse basin") global search over a 2-D
(departure epoch, time of flight) landscape.

The single-basin recipe this replaces took the one global minimum of a coarse
porkchop grid and shrank the optimizer's bounds to a small box around it
(departure +-15 d, TOF +-30 d). That throws away every other trajectory
family before the optimizer ever sees it: a porkchop plot routinely has
several distinct valleys (type-I vs type-II transfers, adjacent launch
windows, long-way vs short-way arcs) whose grid-resolution minima differ by
less than the grid's own discretization error -- the "best cell" is not
reliably the best basin.

This module instead

1. finds every *local* minimum of the grid (``find_diverse_basins``), and
   keeps a spatially spread-out subset of them via greedy non-maximum
   suppression (a basin is only kept if it lies at least one separation
   ellipse away from every better basin already kept);
2. runs one independent optimizer (pygmo PSO by default; any raw pygmo UDA
   via ``algorithm_factory``, e.g. ``pg.cmaes``) per basin, each in its own
   bounded box around that basin and with its own seed, plus optionally one
   unrestricted run over the full search box (``multistart_optimize``);
3. merges runs that converged onto the same optimum, ranks the distinct
   candidates, and records how they compare against the single-basin
   baseline (``MultistartReport``) -- the run seeded from the global grid
   minimum with exactly the old +-15 d / +-30 d box *is* that baseline, so
   the comparison costs nothing extra.

The ranked candidate list is what downstream high-fidelity verification
(tudatpy, see orbitopt.viz.mars_transfer) walks through, instead of being
handed a single answer it can only accept.

Pure numpy here (pygmo is only imported by the default optimizer), so the
basin-detection logic is testable without the pykep/tudatpy compute stack.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace

import numpy as np

logger = logging.getLogger(__name__)

# The single-basin recipe's box half-widths (departure days, TOF days).
SINGLE_BASIN_HALF_WIDTH = (15.0, 30.0)


@dataclass
class Basin:
    """One local minimum of a gridded landscape."""

    x: float  # e.g. departure epoch (MJD2000)
    y: float  # e.g. time of flight (days)
    value: float
    rank: int  # 0 = lowest grid value among the kept basins


@dataclass
class Candidate:
    """One optimizer run's champion."""

    x: float
    y: float
    value: float
    basin_rank: int | None  # None for the unrestricted full-box run
    bounds: tuple  # ((x_lo, x_hi), (y_lo, y_hi)) the run searched
    seed: int | None
    elapsed_s: float = 0.0
    merged_runs: int = 1  # how many independent runs converged here


@dataclass
class MultistartReport:
    """Record of a multistart search and its comparison to the single-basin
    baseline (the run started from the global grid minimum)."""

    basins: list
    runs: list  # every run's raw champion, in run order
    candidates: list  # distinct optima, best first
    baseline: Candidate | None
    notes: list = field(default_factory=list)

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    @property
    def improvement(self) -> float:
        """Objective decrease of the multistart best vs the single-basin
        baseline (>= 0; 0 when the baseline already found the best basin)."""
        if self.baseline is None:
            return 0.0
        return float(self.baseline.value - self.best.value)

    @property
    def extra_candidates(self) -> int:
        """Distinct optima found beyond the single one the baseline yields."""
        return max(0, len(self.candidates) - 1)

    def summary(self, x_label="t0", y_label="tof", unit="") -> str:
        u = f" {unit}" if unit else ""
        lines = [
            f"multistart: {len(self.basins)} diverse basins, {len(self.runs)} independent runs, "
            f"{len(self.candidates)} distinct candidates"
        ]
        for i, c in enumerate(self.candidates):
            origin = "full-box" if c.basin_rank is None else f"basin {c.basin_rank}"
            lines.append(
                f"  #{i}: {x_label}={c.x:.2f} {y_label}={c.y:.2f} -> {c.value:.4f}{u} "
                f"({origin}, {c.merged_runs} run(s))"
            )
        if self.baseline is not None:
            lines.append(
                f"  single-basin baseline: {self.baseline.value:.4f}{u}; "
                f"multistart best: {self.best.value:.4f}{u} "
                f"(improvement {self.improvement:.4f}{u}, +{self.extra_candidates} extra candidates)"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "basins": [asdict(b) for b in self.basins],
            "runs": [asdict(r) for r in self.runs],
            "candidates": [asdict(c) for c in self.candidates],
            "baseline": None if self.baseline is None else asdict(self.baseline),
            "improvement": self.improvement,
            "extra_candidates": self.extra_candidates,
            "notes": list(self.notes),
        }


def _local_minima_mask(values):
    """True where a finite cell is <= all of its (up to 8) neighbours and
    strictly below at least one finite neighbour (so perfectly flat regions
    are not reported as minima; a flat valley *floor* still is, and greedy
    suppression in find_diverse_basins collapses it to one basin)."""
    v = np.asarray(values, dtype=float)
    padded = np.pad(v, 1, mode="constant", constant_values=np.nan)
    not_above = np.isfinite(v)
    below_some = np.zeros_like(not_above)
    rows, cols = v.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbour = padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]
            # Missing (off-grid) neighbours are ignored; infeasible (inf) ones
            # count as higher.
            not_above &= np.isnan(neighbour) | (v <= neighbour)
            below_some |= v < neighbour
    return not_above & below_some


def find_diverse_basins(
    values,
    x_grid,
    y_grid,
    max_basins=6,
    min_separation=(20.0, 40.0),
    max_value_ratio=None,
):
    """Return up to ``max_basins`` spatially distinct local minima of
    ``values`` (shape ``(len(y_grid), len(x_grid))``, the porkchop layout;
    non-finite cells are treated as infeasible), best first.

    A local minimum is kept only if it is at least one normalized separation
    away from every better kept basin, i.e.
    ``(dx / min_separation[0])**2 + (dy / min_separation[1])**2 >= 1``.
    This is greedy non-maximum suppression: plateaus and grid noise inside
    one valley collapse to the valley's lowest cell, while genuinely
    different valleys survive even when they are not the global minimum.

    ``max_value_ratio`` (optional) drops basins whose value exceeds
    ``ratio * global_min`` -- useful to skip hopeless valleys.
    """
    v = np.asarray(values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if v.shape != (len(y_grid), len(x_grid)):
        raise ValueError(f"values shape {v.shape} != (len(y_grid), len(x_grid)) = {(len(y_grid), len(x_grid))}")
    if not np.any(np.isfinite(v)):
        raise ValueError("no finite cells in the grid -- nothing to search")

    sx, sy = (float(s) for s in min_separation)
    iy, ix = np.nonzero(_local_minima_mask(v))
    if iy.size == 0:
        # No cell is strictly below a neighbour: a single-cell grid, or a
        # perfectly flat (finite part of the) grid. There is still exactly one
        # basin -- the best finite cell (first one, for a tie).
        flat = np.where(np.isfinite(v), v, np.inf)
        iy, ix = (np.atleast_1d(i) for i in np.unravel_index(int(np.argmin(flat)), v.shape))
    order = np.argsort(v[iy, ix], kind="stable")
    global_min = float(v[iy[order[0]], ix[order[0]]])

    kept = []
    for k in order:
        x, y, val = float(x_grid[ix[k]]), float(y_grid[iy[k]]), float(v[iy[k], ix[k]])
        if max_value_ratio is not None and global_min > 0 and val > max_value_ratio * global_min:
            break  # sorted ascending: every remaining basin is worse
        if all(((x - b.x) / sx) ** 2 + ((y - b.y) / sy) ** 2 >= 1.0 for b in kept):
            kept.append(Basin(x=x, y=y, value=val, rank=len(kept)))
            if len(kept) >= max_basins:
                break
    return kept


def basin_bounds(basin, half_width, full_bounds):
    """((x_lo, x_hi), (y_lo, y_hi)) box of +-half_width around ``basin``,
    clipped to ``full_bounds``."""
    (xlo, xhi), (ylo, yhi) = full_bounds
    hx, hy = half_width
    return (
        (max(xlo, basin.x - hx), min(xhi, basin.x + hx)),
        (max(ylo, basin.y - hy), min(yhi, basin.y + hy)),
    )


def merge_candidates(runs, merge_distance):
    """Collapse runs whose champions lie within one normalized
    ``merge_distance`` ellipse of a better champion; return the distinct
    optima, best first."""
    mx, my = merge_distance
    distinct = []
    for r in sorted(runs, key=lambda c: c.value):
        if not np.isfinite(r.value):
            continue
        for d in distinct:
            if ((r.x - d.x) / mx) ** 2 + ((r.y - d.y) / my) ** 2 < 1.0:
                d.merged_runs += 1
                break
        else:
            distinct.append(replace(r))
    return distinct


def _default_optimize(problem, pop_size, generations, seed, algorithm_factory):
    from orbitopt.optimize.runner import run_optimization

    algorithm = None if algorithm_factory is None else algorithm_factory(generations, seed)
    return run_optimization(
        problem, algorithm=algorithm, pop_size=pop_size, generations=generations, seed=seed, verbose=False,
    )


def multistart_optimize(
    problem_factory,
    basins,
    full_bounds,
    half_width=SINGLE_BASIN_HALF_WIDTH,
    pop_size=64,
    generations=60,
    seed=None,
    include_full_box_run=True,
    merge_distance=(3.0, 6.0),
    algorithm_factory=None,
    optimize_fn=None,
):
    """Run one independent optimizer per basin (plus, optionally, one over
    the whole ``full_bounds`` box) and compare them.

    ``problem_factory((x_lo, x_hi), (y_lo, y_hi))`` must build a pygmo UDP
    whose decision vector is ``[x, y]`` and whose first objective is to be
    minimized. ``algorithm_factory(generations, seed)``, if given, returns a
    raw pygmo UDA (e.g. ``lambda g, s: pg.cmaes(gen=g, seed=s, force_bounds=True)``);
    the default is run_optimization's PSO. ``optimize_fn(problem, pop_size,
    generations, seed)`` overrides the whole optimizer call (tests use this).

    Run k uses seed ``seed + k`` so the runs are independent but the whole
    search stays reproducible. ``basins[0]`` with the default ``half_width``
    reproduces the single-basin recipe exactly and is recorded as
    ``MultistartReport.baseline``.
    """
    if not basins:
        raise ValueError("multistart_optimize needs at least one basin")

    def run(bounds, run_seed):
        problem = problem_factory(*bounds)
        if optimize_fn is not None:
            x, f, elapsed = optimize_fn(problem, pop_size, generations, run_seed)
        else:
            x, f, elapsed = _default_optimize(problem, pop_size, generations, run_seed, algorithm_factory)
        return float(x[0]), float(x[1]), float(np.atleast_1d(f)[0]), float(elapsed)

    starts = [(b.rank, basin_bounds(b, half_width, full_bounds)) for b in basins]
    if include_full_box_run:
        starts.append((None, tuple(tuple(map(float, b)) for b in full_bounds)))

    runs = []
    for k, (rank, bounds) in enumerate(starts):
        run_seed = None if seed is None else int(seed) + k
        x, y, value, elapsed = run(bounds, run_seed)
        runs.append(Candidate(x=x, y=y, value=value, basin_rank=rank, bounds=bounds, seed=run_seed, elapsed_s=elapsed))

    candidates = merge_candidates(runs, merge_distance)
    if not candidates:
        raise RuntimeError("every multistart run returned a non-finite objective")
    baseline = runs[0] if tuple(half_width) == SINGLE_BASIN_HALF_WIDTH else None
    report = MultistartReport(basins=list(basins), runs=runs, candidates=candidates, baseline=baseline)
    if baseline is not None and report.improvement > 1e-9:
        report.notes.append(
            f"single-basin search (grid minimum +-{half_width[0]:g}/{half_width[1]:g}) would have "
            f"missed the best optimum by {report.improvement:.4f}"
        )
    logger.info("%s", report.summary())
    return report
