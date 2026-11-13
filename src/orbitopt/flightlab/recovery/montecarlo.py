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
the density scale and the R-04 wind enter the acceptance re-integration (FL-301
propagator and FL-303 landing burn) but NOT the FL-302 optimiser, whose rollout
keeps the nominal atmosphere (and calm air) -- a deliberate plan/reality
mismatch, which is exactly what an atmospheric dispersion is meant to probe.
The separation-state dispersion moves the state itself, so it enters
everywhere.

R-04 (task #236) adds the declarative layer: pass ``uncertainty=`` an
:class:`~orbitopt.flightlab.recovery.uncertainty.UncertaintyModel` (or a
config mapping / YAML path) to declare every perturbation's distribution
and switch per source -- wind, density, thrust, Isp, drag, initial state.
The declaration is recorded verbatim in the report (``uncertainty`` section),
the draw stream stays a fixed-order per-[seed, idx] contract, and the legacy
flat :class:`Dispersions` path draws exactly what it always drew (wind
inactive), so pre-R-04 fixed-seed results remain reproducible.

Verdict bookkeeping: a sample is a success only when the FL-304 gate says
landed-in-limits; every failure carries its governing reason code (FL-302
shortfall, corridor miss, FL-303 limit break, ...), so the histogram explains
the whole population. Landing statistics (percentiles + the 1-sigma error
ellipse in the site ENU plane) are reported over the landed subset only, and
say so when there are too few touchdowns to fit one.

R-06 (task #238) adds the statistical reading of that population, in the
report's ``statistics`` section: the failure classes (planning infeasible /
execution failure / landing accuracy / insufficient propellant, with every
failure attributed to exactly one of them), a Wilson score interval on the
success rate, and per-source sensitivity. The joint-dispersion reading (how
each source's own draw associates with failure) is free and always reported;
``solo_sensitivity=True`` additionally re-flies every active source alone at
the same seed and sample count, which is what isolates one source from its
co-dispersed neighbours. Nothing in the statistics layer consumes RNG or
changes a flown sample, so a fixed seed still reproduces the report
byte-for-byte.

    from orbitopt.flightlab.recovery.montecarlo import run_monte_carlo, Dispersions
    mc = run_monte_carlo(target, vehicle_path, ascent_run,
                         dispersions=Dispersions(), seed=7, n=24, workers=4)
    mc.write("montecarlo.json")

    # R-04: the same setup declared per-source, with wind and per-source switches
    mc = run_monte_carlo(target, vehicle_path, ascent_run,
                         uncertainty="uncertainty.yaml", seed=7, n=24)

    # R-06: plus the one-at-a-time marginal sensitivity per source
    mc = run_monte_carlo(target, vehicle_path, ascent_run, uncertainty=unc,
                         seed=7, n=24, solo_sensitivity=True)
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
from orbitopt.flightlab.recovery.stats import (
    failure_breakdown,
    solo_models,
    statistics_summary,
    success_rate_interval,
)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel

__all__ = ["Dispersions", "MonteCarloResult", "run_monte_carlo"]#: The MC default plan options: every sample re-plans locally around the
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
    histogram that explains every sample, the R-06 statistics section
    (confidence interval, failure classes, sensitivity), landing percentiles
    and the error ellipse over the landed subset, and the per-sample records
    (every number in the aggregates is recomputable from them)."""

    n: int
    seed: int
    dispersions: Mapping[str, float]
    success_rate: float
    reason_histogram: Mapping[str, int]
    landing_stats: Mapping[str, Any]
    ellipse: Mapping[str, Any]
    samples: tuple[Mapping[str, Any], ...]
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    #: R-04: the declared perturbation model (source -> enabled flag,
    #: distribution kind and every parameter, see
    #: :mod:`orbitopt.flightlab.recovery.uncertainty`) -- the authoritative
    #: statement of what was dispersed; ``dispersions`` above is its legacy
    #: flat view.
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    #: R-06: the statistical reading of the sample population -- success-rate
    #: confidence interval, failure-class breakdown, perturbation sensitivity
    #: and (opt-in) the one-at-a-time marginal reading; a pure function of
    #: ``samples`` and ``uncertainty``, see
    #: :mod:`orbitopt.flightlab.recovery.stats`.
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.2",
            "kind": "flightlab-recovery-montecarlo",
            "n": int(self.n),
            "seed": int(self.seed),
            "dispersions": {k: float(v) for k, v in self.dispersions.items()},
            "uncertainty": dict(self.uncertainty),
            "success_rate": float(self.success_rate),
            "reason_histogram": {k: int(v) for k, v in self.reason_histogram.items()},
            "statistics": dict(self.statistics),
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
    """Fixed draw order -- stream layout is the determinism contract. The
    legacy flat entry point: delegates to the R-04
    :class:`~orbitopt.flightlab.recovery.uncertainty.UncertaintyModel`
    translation of ``disp`` (wind inactive), which consumes exactly the RNG
    calls FL-304 always consumed."""
    return UncertaintyModel.from_dispersions(disp).draw(rng)


def _run_sample(job: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level pool worker: one dispersed, fully flown recovery (pickle-safe
    by necessity -- the job carries paths and a plain-data uncertainty
    declaration, the record returns plain data)."""
    idx = int(job["idx"])
    model = UncertaintyModel.from_dict(job["uncertainty"])
    rng = np.random.default_rng([int(job["seed"]), idx])
    draw = model.draw(rng)
    record: dict[str, Any] = {"idx": idx, **draw}
    try:
        ascent = read_run(Path(job["run_dir"]))
        vehicle = BoosterModel.load(job["vehicle_path"])
        eng = vehicle.engine
        vehicle = replace(
            vehicle,
            engine=replace(eng,
                           thrust_vacuum_n=eng.thrust_vacuum_n * draw["thrust_factor"],
                           isp_sea_level_s=eng.isp_sea_level_s * draw["isp_factor"],
                           isp_vacuum_s=eng.isp_vacuum_s * draw["isp_factor"]),
            cd_mach=tuple((m, c * draw["cd_factor"]) for m, c in vehicle.cd_mach),
        )
        physics = BoosterPhysics(density_scale=draw["density_scale"],
                                 wind_east_m_s=draw["wind_east_m_s"],
                                 wind_north_m_s=draw["wind_north_m_s"],
                                 wind_up_m_s=draw["wind_up_m_s"])
        sep = separation_state_ecef(ascent)
        sep = dispersed_separation_state(
            sep, altitude_m=draw["sep_altitude_m"],
            downrange_m=draw["sep_downrange_m"], crossrange_m=draw["sep_crossrange_m"],
            speed_m_s=draw["sep_speed_m_s"], heading_deg=draw["sep_heading_deg"])
        outcome = fly_recovery(ascent, job["target"], job["vehicle_path"],
                               sep_state=sep, booster=vehicle, physics=physics,
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
    anything; the state field says so honestly). A landed record without
    touchdown coordinates contributes nothing to the scatter and is counted
    only in the shortfall."""
    pts = np.array([[s["touchdown_east_m"], s["touchdown_north_m"]]
                    for s in samples
                    if s.get("touchdown_east_m") is not None
                    and s.get("touchdown_north_m") is not None],
                   dtype=np.float64)
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
                    uncertainty: "UncertaintyModel | Mapping[str, Any] | str | Path | None" = None,
                    seed: int = 1, n: int = 8, workers: int = 1,
                    plan_options: Mapping[str, Any] | None = None,
                    solo_sensitivity: bool = False,
                    out_path: Path | str | None = None) -> MonteCarloResult:
    """Fly ``n`` dispersed recovery samples of ``ascent_run`` under ``target``
    (module docstring) and return the aggregated :class:`MonteCarloResult`;
    ``out_path`` additionally writes ``montecarlo.json``. Sampling is per
    ``[seed, idx]`` streams, so the same seed reproduces the statistics
    byte-for-byte regardless of ``workers`` (``workers > 1`` runs the samples
    through a ``ProcessPoolExecutor``; ``workers == 1`` stays in-process).

    R-04: ``uncertainty`` takes the declarative perturbation model -- a
    :class:`~orbitopt.flightlab.recovery.uncertainty.UncertaintyModel`, a
    config mapping or a YAML/JSON path (``UncertaintyModel.from_file`` form) --
    declaring each source's distribution, parameters and switch; it is
    recorded verbatim under ``uncertainty`` in the report and takes precedence
    over ``dispersions`` (passing both raises). ``dispersions=None`` (the default) takes the thrust / Isp / Cd half-widths
    from the vehicle file's FL-602 provenance uncertainty tags -- a Monte Carlo
    inherits the data's stated confidence bands unless the caller says
    otherwise; scenario-side knobs (density scale, separation-state sigma)
    keep their class defaults, and a vehicle file without uncertainty tags
    reproduces the plain :class:`Dispersions` defaults exactly. The
    substitution is recorded in the result's assumptions (FL-304/FL-602).

    R-06: the report's ``statistics`` section carries the statistical reading
    of the population (:mod:`orbitopt.flightlab.recovery.stats`) -- failure
    classes, a Wilson score interval on the success rate and a per-source
    sensitivity under joint dispersion, all pure functions of the flown
    samples. ``solo_sensitivity=True`` adds the marginal reading: every active
    source re-flown alone (all others neutral) at the same seed and sample
    count, i.e. ``1 + n_active`` passes in total -- a correspondingly more
    expensive, but far more explanatory, run."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if int(seed) < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if uncertainty is not None and dispersions is not None:
        raise ValueError("pass either uncertainty= or dispersions=, not both: "
                         "the declarative model already carries the full setup")
    from_vehicle_notes: list[str] = []
    if isinstance(uncertainty, (str, Path)):
        model = UncertaintyModel.from_file(uncertainty)
    elif isinstance(uncertainty, Mapping):
        model = UncertaintyModel.from_config(uncertainty)
    elif uncertainty is not None:
        model = uncertainty
    elif dispersions is not None:
        model = UncertaintyModel.from_dispersions(dispersions)
    else:
        from orbitopt.flightlab.vehicle_catalog import vehicle_dispersions
        disp, from_vehicle_notes = vehicle_dispersions(vehicle_path)
        model = UncertaintyModel.from_dispersions(disp)
    opts = dict(MC_PLAN_OPTIONS if plan_options is None else plan_options)

    assumptions = [
        "FL-304 Monte Carlo: guidance re-planned per sample on the dispersed "
        "vehicle/separation state (closed loop at plan level); density scale "
        "and R-04 wind bias the acceptance atmosphere only, not the FL-302 "
        "optimiser model",
        "one numpy stream [seed, idx] per sample: statistics are exact for a "
        "seed, independent of the worker layout",
        "success = FL-304 recovery-landing gate: touched down inside every "
        "target landing limit; every failure carries its governing reason code",
        "R-04 perturbation declaration (distribution kind, parameters, per-"
        "source switches) recorded under 'uncertainty'; 'dispersions' is the "
        "legacy flat view of it",
        "R-06 statistics: success rate reported with a Wilson score interval, "
        "every failure attributed to one R-06 failure class, and per-source "
        "sensitivity computed as a rank correlation of each source's own "
        "sigma-equivalent draw against failure under joint dispersion "
        "(orbitopt.flightlab.recovery.stats)",
        *from_vehicle_notes,
    ]
    if solo_sensitivity:
        assumptions.append(
            "R-06 one-at-a-time sensitivity: every active source re-flown "
            "alone at the same seed and sample count -- the marginal reading "
            "that separates a source from the sources dispersed with it")

    def _dispatch(m: UncertaintyModel) -> list[dict[str, Any]]:
        """Fly the ``n`` samples of one perturbation model over the shared
        ascent run (the pool moves work, never numbers)."""
        jobs = [{"idx": i, "seed": int(seed), "run_dir": str(run_dir),
                 "target": target, "vehicle_path": str(vehicle_path),
                 "uncertainty": m.to_dict(), "plan_options": dict(opts)}
                for i in range(n)]
        if workers and workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                return list(pool.map(_run_sample, jobs))
        return [_run_sample(job) for job in jobs]

    with tempfile.TemporaryDirectory(prefix="fl304mc-") as tmp:
        run_dir = Path(tmp) / "ascent"
        write_run(run_dir, ascent_run)
        records = _dispatch(model)
        solo: dict[str, Any] | None = None
        if solo_sensitivity:
            per_source: dict[str, Any] = {}
            for name, alone in solo_models(model).items():
                sub = _dispatch(alone)
                k = sum(1 for r in sub if r.get("outcome") == "landed")
                per_source[name] = {
                    "n": n, "success_rate": k / n,
                    "ci": success_rate_interval(k, n),
                    "failure_classes": failure_breakdown(sub)}
            solo = {"method": "one-at-a-time: each active source alone "
                              "(every other source neutral), same seed and n "
                              "as the joint run; the delta is marginal to "
                              "that source, not a variance decomposition",
                    "sources": per_source}

    histogram: dict[str, int] = {}
    for rec in records:
        histogram[rec["reason"]] = histogram.get(rec["reason"], 0) + 1
    landed = [r for r in records if r.get("outcome") == "landed"]
    n_ok = len(landed)
    if solo is not None:
        solo["baseline"] = {"success_rate": n_ok / n,
                            "ci": success_rate_interval(n_ok, n)}
        for entry in solo["sources"].values():
            entry["delta_vs_baseline"] = entry["success_rate"] - n_ok / n

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
        n=n, seed=int(seed), dispersions=model.dispersions_view(),
        uncertainty=model.to_dict(),
        success_rate=n_ok / n,
        reason_histogram=dict(sorted(histogram.items())),
        statistics=statistics_summary(successes=n_ok, n=n, samples=records,
                                      model=model, solo=solo),
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
