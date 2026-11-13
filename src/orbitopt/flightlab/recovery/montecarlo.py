"""FL-304: closed-loop Monte Carlo of the E3 recovery chain.

:func:`run_monte_carlo` flies ``n`` dispersed samples of one recovery: every
sample re-reads the ascent run, draws its own perturbation (thrust, Isp,
atmosphere density scale, drag coefficient, separation state), re-plans the
FL-302 boostback/entry script *for that perturbed world* and re-flies FL-303
under it -- closed loop at the plan level: guidance adapts, the acceptance
physics does not forgive. Samples are independent (one
``numpy.random`` stream per sample, ``[seed, idx]``), so the statistics are
deterministic per seed and independent of the worker layout; the process pool
only moves work, never numbers.

Honest dispersion accounting: thrust, Isp and Cd enter the FL-302 planner
(it flies the dispersed ``BoosterModel``) *and* the acceptance re-integration;
the density scale enters the acceptance re-integration (FL-301 propagator and
FL-303 landing burn) but NOT the FL-302 optimiser, whose rollout keeps the
nominal atmosphere -- a deliberate plan/reality mismatch, which is exactly
what an atmospheric dispersion is meant to probe. The separation-state
dispersion moves the state itself, so it enters everywhere.

Verdict bookkeeping: a sample is a success only when the FL-304 gate says
landed-in-limits; every failure carries its governing reason code (FL-302
shortfall, corridor miss, FL-303 limit break, ...), so the histogram explains
the whole population. Landing statistics (percentiles + the 1-sigma error
ellipse in the site ENU plane) are reported over the landed subset only, and
say so when there are too few touchdowns to fit one.

    from orbitopt.flightlab.recovery.montecarlo import run_monte_carlo, Dispersions
    mc = run_monte_carlo(target, vehicle_path, ascent_run,
                         dispersions=Dispersions(), seed=7, n=24, workers=4)
    mc.write("montecarlo.json")
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab.contract import FlightRun, FlightTarget, read_run, write_run
from orbitopt.flightlab.recovery.pipeline import (
    NOMINAL_PLAN_GUESS,
    dispersed_separation_state,
    fly_recovery,
    separation_state_ecef,
)
from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterPhysics

__all__ = ["Dispersions", "MonteCarloResult", "run_monte_carlo"]

#: The MC default plan options: every sample re-plans locally around the
#: accepted nominal script (a guidance computer would refine, not restart a
#: cold search) -- ``max_fevals`` below 40 keeps FL-302 where it started.
MC_PLAN_OPTIONS: dict[str, Any] = {"initial_guess": NOMINAL_PLAN_GUESS,
                                   "max_fevals": 30}


@dataclass(frozen=True)
class Dispersions:
    """One dispersion setup. ``*_pct`` are half-widths of symmetric uniform
    draws (5.0 = factor in U(0.95, 1.05)); the ``sep_*`` are 1-sigma sigmas
    of normal draws on the separation state (altitude / along-track /
    cross-track metres, speed m/s along the velocity vector, heading rotation
    of the horizontal velocity in degrees). Zero disables that knob; the
    default setup covers five classes (thrust, Isp, density, Cd,
    separation state)."""

    thrust_pct: float = 5.0
    isp_pct: float = 3.0
    density_pct: float = 10.0
    cd_pct: float = 8.0
    sep_altitude_m: float = 400.0
    sep_downrange_m: float = 800.0
    sep_crossrange_m: float = 800.0
    sep_speed_m_s: float = 4.0
    sep_heading_deg: float = 0.15

    def __post_init__(self) -> None:
        for name in ("thrust_pct", "isp_pct", "density_pct", "cd_pct"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("sep_altitude_m", "sep_downrange_m", "sep_crossrange_m",
                     "sep_speed_m_s", "sep_heading_deg"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        live = sum(1 for v in (self.thrust_pct, self.isp_pct, self.density_pct,
                               self.cd_pct) if v > 0.0) + (
            1 if max(self.sep_altitude_m, self.sep_downrange_m,
                     self.sep_crossrange_m, self.sep_speed_m_s,
                     self.sep_heading_deg) > 0.0 else 0)
        if live < 3:
            raise ValueError("a Monte Carlo setup needs at least 3 active "
                             f"dispersion classes, got {live}")

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in (
            "thrust_pct", "isp_pct", "density_pct", "cd_pct", "sep_altitude_m",
            "sep_downrange_m", "sep_crossrange_m", "sep_speed_m_s",
            "sep_heading_deg")}


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated Monte-Carlo verdict: success rate, the reason-code
    histogram that explains every sample, landing percentiles and the error
    ellipse over the landed subset, and the per-sample records (every number
    in the aggregates is recomputable from them)."""

    n: int
    seed: int
    dispersions: Mapping[str, float]
    success_rate: float
    reason_histogram: Mapping[str, int]
    landing_stats: Mapping[str, Any]
    ellipse: Mapping[str, Any]
    samples: tuple[Mapping[str, Any], ...]
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0",
            "kind": "flightlab-recovery-montecarlo",
            "n": int(self.n),
            "seed": int(self.seed),
            "dispersions": {k: float(v) for k, v in self.dispersions.items()},
            "success_rate": float(self.success_rate),
            "reason_histogram": {k: int(v) for k, v in self.reason_histogram.items()},
            "landing_stats": dict(self.landing_stats),
            "ellipse": dict(self.ellipse),
            "samples": [dict(s) for s in self.samples],
            "assumptions": list(self.assumptions),
        }

    def write(self, path: Path) -> None:
        """``montecarlo.json`` -- byte-identical for identical statistics
        (sorted keys, fixed float formatting through ``json``)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=1) + "\n",
            encoding="utf-8")


def _draw_factors(rng: np.random.Generator, disp: Dispersions) -> dict[str, float]:
    """Fixed draw order -- stream layout is the determinism contract."""
    def uniform(pct: float) -> float:
        return 1.0 + float(rng.uniform(-pct, pct)) / 100.0 if pct > 0.0 else 1.0

    def normal(sigma: float) -> float:
        return float(rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0

    return {
        "thrust_factor": uniform(disp.thrust_pct),
        "isp_factor": uniform(disp.isp_pct),
        "density_scale": uniform(disp.density_pct),
        "cd_factor": uniform(disp.cd_pct),
        "sep_altitude_m": normal(disp.sep_altitude_m),
        "sep_downrange_m": normal(disp.sep_downrange_m),
        "sep_crossrange_m": normal(disp.sep_crossrange_m),
        "sep_speed_m_s": normal(disp.sep_speed_m_s),
        "sep_heading_deg": normal(disp.sep_heading_deg),
    }


def _run_sample(job: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level pool worker: one dispersed, fully flown recovery (pickle-safe
    by necessity -- the job carries paths and a plain-data dispersion, the
    record returns plain data)."""
    idx = int(job["idx"])
    disp = Dispersions(**job["dispersions"])
    rng = np.random.default_rng([int(job["seed"]), idx])
    draw = _draw_factors(rng, disp)
    record: dict[str, Any] = {"idx": idx, **draw}
    try:
        ascent = read_run(Path(job["run_dir"]))
        model = BoosterModel.load(job["vehicle_path"])
        eng = model.engine
        model = replace(
            model,
            engine=replace(eng,
                           thrust_vacuum_n=eng.thrust_vacuum_n * draw["thrust_factor"],
                           isp_sea_level_s=eng.isp_sea_level_s * draw["isp_factor"],
                           isp_vacuum_s=eng.isp_vacuum_s * draw["isp_factor"]),
            cd_mach=tuple((m, c * draw["cd_factor"]) for m, c in model.cd_mach),
        )
        physics = BoosterPhysics(density_scale=draw["density_scale"])
        sep = separation_state_ecef(ascent)
        sep = dispersed_separation_state(
            sep, altitude_m=draw["sep_altitude_m"],
            downrange_m=draw["sep_downrange_m"], crossrange_m=draw["sep_crossrange_m"],
            speed_m_s=draw["sep_speed_m_s"], heading_deg=draw["sep_heading_deg"])
        outcome = fly_recovery(ascent, job["target"], job["vehicle_path"],
                               sep_state=sep, booster=model, physics=physics,
                               plan_options=dict(job["plan_options"]))
        record["outcome"] = "landed" if outcome.success else "failed"
        record["reason"] = "LANDED" if outcome.success else outcome.reason_codes[0]
        plan = outcome.plan
        if plan is not None and plan.corridor_report.reached:
            rep = plan.corridor_report
            record["corridor_miss_m"] = _num(rep.miss_m)
            if rep.propellant_available_kg is not None and rep.landing_need_kg is not None:
                record["corridor_fuel_margin_kg"] = _num(rep.propellant_available_kg
                                                         - rep.landing_need_kg
                                                         - rep.reserve_kg)
        landing = outcome.landing
        if landing is not None:
            record["position_error_m"] = _num(landing.position_error_m)
            record["vertical_speed_m_s"] = _num(landing.vertical_speed_m_s)
            record["lateral_speed_m_s"] = _num(landing.lateral_speed_m_s)
            if landing.touchdown_enu_m is not None:
                record["touchdown_east_m"] = float(landing.touchdown_enu_m[0])
                record["touchdown_north_m"] = float(landing.touchdown_enu_m[1])
            if getattr(landing, "actuator_margin", None) is not None:
                # R-05: the guidance's own actuator-margin accounting
                record["control_margin"] = landing.actuator_margin.to_dict()
    except Exception as exc:  # noqa: BLE001 -- a broken sample must never erase the run
        record["outcome"] = "error"
        record["reason"] = f"SAMPLE_ERROR:{type(exc).__name__}"
        record["error"] = str(exc)
    return record


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    p50, p95 = np.percentile(arr, [50.0, 95.0])
    return {"n": int(arr.size), "p50": float(p50), "p95": float(p95)}


def _ellipse(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """1-sigma error ellipse of the touchdowns in the site ENU plane:
    eigen-decomposition of the scatter (needs >= 3 touchdowns to mean
    anything; the state field says so honestly)."""
    pts = np.array([[s["touchdown_east_m"], s["touchdown_north_m"]]
                    for s in samples], dtype=np.float64)
    if len(pts) < 3:
        return {"state": "no_landings" if not len(pts) else "insufficient_samples",
                "n": int(len(pts))}
    cov = np.cov(pts.T)
    mean = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    major, minor = int(vals.argmax()), int(vals.argmin())
    vec = vecs[:, major]
    bearing = float(np.degrees(np.arctan2(vec[0], vec[1]))) % 180.0
    return {"state": "assessed", "n": int(len(pts)),
            "mean_east_m": float(mean[0]), "mean_north_m": float(mean[1]),
            "sigma_along_m": float(np.sqrt(max(vals[major], 0.0))),
            "sigma_cross_m": float(np.sqrt(max(vals[minor], 0.0))),
            "bearing_deg": bearing}


def run_monte_carlo(target: FlightTarget, vehicle_path: Path | str,
                    ascent_run: FlightRun, *,
                    dispersions: Dispersions | None = None,
                    seed: int = 1, n: int = 8, workers: int = 1,
                    plan_options: Mapping[str, Any] | None = None,
                    out_path: Path | str | None = None) -> MonteCarloResult:
    """Fly ``n`` dispersed recovery samples of ``ascent_run`` under ``target``
    (module docstring) and return the aggregated :class:`MonteCarloResult`;
    ``out_path`` additionally writes ``montecarlo.json``. Sampling is per
    ``[seed, idx]`` streams, so the same seed reproduces the statistics
    byte-for-byte regardless of ``workers`` (``workers > 1`` runs the samples
    through a ``ProcessPoolExecutor``; ``workers == 1`` stays in-process).

    ``dispersions=None`` (the default) takes the thrust / Isp / Cd half-widths
    from the vehicle file's FL-602 provenance uncertainty tags -- a Monte Carlo
    inherits the data's stated confidence bands unless the caller says
    otherwise; scenario-side knobs (density scale, separation-state sigma)
    keep their class defaults, and a vehicle file without uncertainty tags
    reproduces the plain :class:`Dispersions` defaults exactly. The
    substitution is recorded in the result's assumptions (FL-304/FL-602)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if int(seed) < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if dispersions is not None:
        disp, from_vehicle_notes = dispersions, []
    else:
        from orbitopt.flightlab.vehicle_catalog import vehicle_dispersions
        disp, from_vehicle_notes = vehicle_dispersions(vehicle_path)
    opts = dict(MC_PLAN_OPTIONS if plan_options is None else plan_options)

    assumptions = [
        "FL-304 Monte Carlo: guidance re-planned per sample on the dispersed "
        "vehicle/separation state (closed loop at plan level); density scale "
        "biases the acceptance atmosphere only, not the FL-302 optimiser model",
        "one numpy stream [seed, idx] per sample: statistics are exact for a "
        "seed, independent of the worker layout",
        "success = FL-304 recovery-landing gate: touched down inside every "
        "target landing limit; every failure carries its governing reason code",
        *from_vehicle_notes,
    ]

    jobs: list[dict[str, Any]]
    with tempfile.TemporaryDirectory(prefix="fl304mc-") as tmp:
        run_dir = Path(tmp) / "ascent"
        write_run(run_dir, ascent_run)
        jobs = [{"idx": i, "seed": int(seed), "run_dir": str(run_dir),
                 "target": target, "vehicle_path": str(vehicle_path),
                 "dispersions": disp.to_dict(), "plan_options": dict(opts)}
                for i in range(n)]
        if workers and workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                records = list(pool.map(_run_sample, jobs))
        else:
            records = [_run_sample(job) for job in jobs]

    histogram: dict[str, int] = {}
    for rec in records:
        histogram[rec["reason"]] = histogram.get(rec["reason"], 0) + 1
    landed = [r for r in records if r.get("outcome") == "landed"]
    n_ok = len(landed)

    def stats_of(key: str) -> dict[str, Any]:
        vals = [r[key] for r in landed if r.get(key) is not None]
        return _percentiles(vals) if vals else {"n": 0}

    def margin_stats() -> dict[str, Any]:
        """R-05 control-margin aggregation over the landed subset (the
        guidance's own accounting travels per sample; min/mean/max per
        metric here)."""
        vals = [r["control_margin"] for r in landed if r.get("control_margin")]
        if not vals:
            return {"n": 0}
        out: dict[str, Any] = {"n": len(vals)}
        for key in sorted({k for v in vals for k in v}):
            xs = [float(v[key]) for v in vals if v.get(key) is not None]
            out[key] = ({"n": len(xs), "min": min(xs), "mean": sum(xs) / len(xs),
                         "max": max(xs)} if xs else {"n": 0})
        return out

    result = MonteCarloResult(
        n=n, seed=int(seed), dispersions=disp.to_dict(),
        success_rate=n_ok / n,
        reason_histogram=dict(sorted(histogram.items())),
        landing_stats={
            "position_error_m": stats_of("position_error_m"),
            "vertical_speed_m_s": stats_of("vertical_speed_m_s"),
            "lateral_speed_m_s": stats_of("lateral_speed_m_s"),
            "control_margin": margin_stats(),
        },
        ellipse=_ellipse(landed),
        samples=tuple(records),
        assumptions=tuple(assumptions),
    )
    if out_path is not None:
        result.write(Path(out_path))
    return result
