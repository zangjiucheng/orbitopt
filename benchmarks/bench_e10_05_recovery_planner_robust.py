#!/usr/bin/env python3
"""E10-05 (task #302): why did per-sample replanning LOSE to the fixed script on
DC7 (3/36 against 11/36, E5-08b), and does a single robustly-planned script do
better? Three arms on one hold-out, paired.

The DC7 hold-out (``scripts/e508_new_vehicle.py mc``, seeds 8/9/10 x 12) said:
the recorded script flown as-is lands 11/36, the same recipe with the stock
per-sample local refine (``max_fevals 40``) lands 3/36. Re-planning did not
help -- it hurt. This benchmark diagnoses that (flip tables, a search-budget
ladder, reason-code decomposition) and compares three planning policies on the
EXACT SAME 36 hold-out samples:

* ``fixed``   -- the recorded script flown as-is (``search="keep"``);
* ``replan40``-- the stock per-sample Nelder-Mead refine (``max_fevals=40``),
  the arm the sealed record scored 3/36;
* ``robust``  -- ONE script, chosen by a scenario-robust (1+lambda)-ES over a
  TRAINING scenario set disjoint from the hold-out (seeds 401/402, never
  overlapping the hold-out seeds 8/9/10 or the recorded search seed 287), then
  flown pinned on the hold-out -- optimize the perturbation set, not the sample.

Statistics follow the E7-05 convention: the seed is the exchangeable cluster
(cluster bootstrap CI on every rate), the samples are paired across arms
(exact McNemar per pair, Holm-corrected across the three pairwise family), and
the training/hold-out split with its seed manifest is written into the JSON.

Run (DC7 tree, ~1.5-2 h wall at --workers 6 on 8 cores):

    python benchmarks/bench_e10_05_recovery_planner_robust.py \
        --workers 6 \
        --fly-cache /tmp/e10_05_fly_cache.json \
        -o docs/benchmarks/e10_05_recovery_planner_robust.json \
        --md docs/benchmarks/e10_05_recovery_planner_robust.md

``--fly-cache`` persists every flown sample keyed by arm+seed+idx+script so an
interrupted run resumes and a re-run is free; the cache is not an artifact.
The sealed ``mc-summary*.json`` records are read, never written, by this bench.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# --------------------------------------------------------------------------
# DC7 context: the E5-08b variant of the E5-08 tooling (vehicle, scenario,
# reference tree). The env var must be set before the script module is loaded.
# --------------------------------------------------------------------------
os.environ["ORBITOPT_E508_VARIANT"] = "dc7"
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per fly worker

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e508_new_vehicle", REPO / "scripts" / "e508_new_vehicle.py")
E508 = importlib.util.module_from_spec(_spec)
sys.modules["e508_new_vehicle"] = E508
_spec.loader.exec_module(E508)

# --------------------------------------------------------------------------
# Seed manifest (acceptance: train/hold-out definitions + seeds in the bench).
# Everything here is disjoint from everything else: the recorded script was
# searched at 287; the hold-out reuses the recorded MC grid (8/9/10 x 12) so
# the fresh fixed/replan40 arms are directly comparable to the sealed 11/36
# and 3/36; the robust arm trains on 401/402 x 5, never flown into the
# hold-out, and its ES rng is its own stream (41171).
# --------------------------------------------------------------------------
RECORDED_SEARCH_SEED = 287          # the recorded DC7 script's search seed
HOLDOUT_SEEDS = [8, 9, 10]          # the sealed DC7 MC grid
N_PER_SEED = 12
TRAIN_SEEDS = [401, 402]            # robust-arm training scenarios (CRN set)
TRAIN_PER_SEED = 5                  # -> 10 training scenarios + 1 nominal
ES_SEED = 41171                     # the (1+lambda)-ES rng, training-only
BOOTSTRAP_SEED = 20261121           # every bootstrap in this bench
DIAG_N_FAILED = 6                   # both-failed control samples in the budget ladder

HOLDOUT_GRID = [(int(s), i) for s in HOLDOUT_SEEDS for i in range(N_PER_SEED)]
TRAIN_GRID = [(int(s), i) for s in TRAIN_SEEDS for i in range(TRAIN_PER_SEED)]
assert not (set(HOLDOUT_SEEDS) & set(TRAIN_SEEDS))
assert set(TRAIN_SEEDS) != {RECORDED_SEARCH_SEED} and set(HOLDOUT_SEEDS) != {RECORDED_SEARCH_SEED}

ARMS = ("fixed", "replan40", "robust")
BUDGETS = {10: "replan10", 120: "replan120"}  # diagnostic ladder arm names


def _grid_key(arm: str, seed: int, idx: int, script_tag: str) -> str:
    return f"{arm}|{seed}|{idx}|{script_tag}"


# --------------------------------------------------------------------------
# one flown sample, top level for the fork pool (mirrors
# orbitopt.flightlab.recovery.montecarlo._run_sample, with the DC7 canonical
# separation handoff and richer per-sample diagnostics)
# --------------------------------------------------------------------------
def _fly_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    from dataclasses import replace

    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.recovery.pipeline import dispersed_separation_state, fly_recovery
    from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterPhysics, _to_ecef_state
    from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff

    t0 = time.time()
    record: dict[str, Any] = {k: job[k] for k in ("arm", "seed", "idx", "script_tag")}
    try:
        ascent = read_run(Path(job["run_dir"]))
        vehicle = BoosterModel.load(job["vehicle_path"])
        eng = vehicle.engine
        if job["draw"] is None:  # nominal scenario: no dispersion at all
            draw = {}
        else:
            model = UncertaintyModel.from_dict(job["uncertainty"])
            rng = np.random.default_rng([int(job["draw"][0]), int(job["draw"][1])])
            draw = model.draw(rng)
            record["draw"] = {k2: float(v) for k2, v in draw.items()}
        vehicle = replace(
            vehicle,
            engine=replace(eng,
                           thrust_vacuum_n=eng.thrust_vacuum_n * draw.get("thrust_factor", 1.0),
                           isp_sea_level_s=eng.isp_sea_level_s * draw.get("isp_factor", 1.0),
                           isp_vacuum_s=eng.isp_vacuum_s * draw.get("isp_factor", 1.0)),
            cd_mach=tuple((m, c * draw.get("cd_factor", 1.0)) for m, c in vehicle.cd_mach))
        physics = BoosterPhysics(density_scale=draw.get("density_scale", 1.0),
                                 wind_east_m_s=draw.get("wind_east_m_s", 0.0),
                                 wind_north_m_s=draw.get("wind_north_m_s", 0.0),
                                 wind_up_m_s=draw.get("wind_up_m_s", 0.0))
        # the DC7 canonical handoff (E508._recovery_sep_state_ecef), spelled out
        # inline so the worker stays import-clean across pickling
        sep = booster_state_from_run(ascent, separation_handoff(ascent))
        if sep.frame != "ECEF":
            r_e, v_e = _to_ecef_state(sep)
            sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r_e),
                          velocity_m_s=tuple(float(c) for c in v_e))
        sep = dispersed_separation_state(
            sep, altitude_m=draw.get("sep_altitude_m", 0.0),
            downrange_m=draw.get("sep_downrange_m", 0.0),
            crossrange_m=draw.get("sep_crossrange_m", 0.0),
            speed_m_s=draw.get("sep_speed_m_s", 0.0),
            heading_deg=draw.get("sep_heading_deg", 0.0))
        outcome = fly_recovery(ascent, job["target"], job["vehicle_path"],
                               sep_state=sep, booster=vehicle, physics=physics,
                               plan_options=dict(job["options"]))
        record["outcome"] = "landed" if outcome.success else "failed"
        record["reason"] = "LANDED" if outcome.success else str(outcome.reason_codes[0])
        record["checks"] = {c.id: bool(c.passed) for c in outcome.checks}
        plan = outcome.plan
        if plan is not None:
            bb, eb = plan.boostback, plan.entry_burn
            record["script"] = [bb.start_delay_s, bb.duration_s, bb.throttle, bb.pitch_deg,
                                bb.yaw_deg, eb.trigger_altitude_m, eb.duration_s, eb.throttle]
            rep = plan.corridor_report
            if rep.reached:
                record["miss_m"] = _f(rep.miss_m)
                record["entry_q_pa"] = _f(rep.entry_max_q_pa)
                record["entry_g"] = _f(rep.entry_max_proper_g)
                record["entry_cut_h_m"] = _f(rep.entry_cutoff_altitude_m)
                if rep.propellant_available_kg is not None and rep.landing_need_kg is not None:
                    record["fuel_margin_kg"] = _f(rep.propellant_available_kg
                                                  - rep.landing_need_kg - rep.reserve_kg)
        landing = outcome.landing
        if landing is not None:
            record["touch_err_m"] = _f(landing.position_error_m)
            record["vert_m_s"] = _f(landing.vertical_speed_m_s)
            record["lat_m_s"] = _f(landing.lateral_speed_m_s)
        record["dur_s"] = round(time.time() - t0, 1)
    except Exception as exc:  # a broken sample must never erase the run
        record["outcome"] = "error"
        record["reason"] = f"SAMPLE_ERROR:{type(exc).__name__}"
        record["error"] = str(exc)
        record["dur_s"] = round(time.time() - t0, 1)
    return record


def _f(v: Any) -> float | None:
    return None if v is None else float(v)


# --------------------------------------------------------------------------
# flying: the fork pool inherits this module, so jobs carry paths/option dicts
# --------------------------------------------------------------------------
class FlyRig:
    def __init__(self, workers: int, cache_path: Path | None) -> None:
        self.workers = int(workers)
        self.cache_path = cache_path
        self.fly: dict[str, dict[str, Any]] = {}
        if cache_path is not None and cache_path.exists():
            self.fly = json.loads(cache_path.read_text())
        self.tmp = tempfile.TemporaryDirectory(prefix="e1005-")
        from orbitopt.flightlab.contract import load_target, read_run, write_run
        self.target = load_target(E508.SCENARIO)
        ascent = read_run(E508.REF_ROOT / "mission-optimized" / "launch" / "ascent")
        self.run_dir = Path(self.tmp.name) / "ascent"
        write_run(self.run_dir, ascent)
        from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel
        from orbitopt.flightlab.vehicle_catalog import vehicle_dispersions
        disp, _notes = vehicle_dispersions(E508.VEHICLE)
        self.uncertainty = UncertaintyModel.from_dispersions(disp).to_dict()
        self.pool: ProcessPoolExecutor | None = None
        if self.workers > 1:
            ctx = multiprocessing.get_context("fork")
            self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        self.tmp.cleanup()

    def save(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.fly) + "\n")

    def options(self, arm: str, script: Sequence[float] | None) -> dict[str, Any]:
        recipe = dict(json.loads(E508.SCRIPT_JSON.read_text())["recipe"])
        if arm in ("fixed", "robust"):
            recipe["search"] = "keep"
            recipe.pop("max_fevals", None)
            if script is not None:
                recipe["initial_guess"] = [float(x) for x in script]
        else:  # the per-sample refine arms: the stock search, budget by arm name
            recipe.pop("search", None)
            recipe["max_fevals"] = int(arm.replace("replan", ""))
        return recipe

    def jobs_for(self, arm: str, grid: Sequence[tuple[int, int]], script: Sequence[float] | None,
                 tag: str) -> list[dict[str, Any]]:
        opts = self.options(arm, script)
        out = []
        for seed, idx in grid:
            key = _grid_key(arm, seed, idx, tag)
            if key in self.fly:
                continue
            out.append({"arm": arm, "seed": int(seed), "idx": int(idx), "script_tag": tag,
                        "run_dir": str(self.run_dir), "target": self.target,
                        "vehicle_path": str(E508.VEHICLE), "uncertainty": self.uncertainty,
                        "options": opts, "draw": [int(seed), int(idx)]})
        return out

    def fly_jobs(self, jobs: Sequence[dict[str, Any]]) -> None:
        if not jobs:
            return
        if self.pool is not None:
            for rec in self.pool.map(_fly_worker, jobs, chunksize=1):
                self.fly[_grid_key(rec["arm"], rec["seed"], rec["idx"], rec["script_tag"])] = rec
        else:
            for job in jobs:
                rec = _fly_worker(job)
                self.fly[_grid_key(rec["arm"], rec["seed"], rec["idx"], rec["script_tag"])] = rec
        self.save()

    def fly_nominal(self, script: Sequence[float], tag: str) -> dict[str, Any]:
        key = _grid_key("nominal", 0, 0, tag)
        if key in self.fly:
            return self.fly[key]
        job = {"arm": "nominal", "seed": 0, "idx": 0, "script_tag": tag,
               "run_dir": str(self.run_dir), "target": self.target,
               "vehicle_path": str(E508.VEHICLE), "uncertainty": self.uncertainty,
               "options": self.options("fixed", script), "draw": None}
        rec = _fly_worker(job)
        self.fly[key] = rec
        return rec

    def arm_records(self, arm: str, grid: Sequence[tuple[int, int]], tag: str = "v0",
                    ) -> dict[tuple[int, int], dict[str, Any]]:
        return {(int(r["seed"]), int(r["idx"])): r
                for (seed, idx) in grid
                for key, r in self.fly.items()
                if (key == _grid_key(arm, seed, idx, tag) and r["arm"] == arm)}


# --------------------------------------------------------------------------
# the robust arm: scenario-robust (1+lambda)-ES on the recorded script
# --------------------------------------------------------------------------
def script_bounds(vehicle_path: Path) -> list[tuple[float, float]]:
    """The planner's own clamp box for the DC7 8-parameter script
    (boostback.py clamp(), DC7 modes: iip cutoff, speed cutoff, entry throttle
    on, DEFAULT_CORRIDOR as the pipeline builds it). The ES sigma and box
    follow the planner's own authority."""
    from orbitopt.flightlab.recovery.boostback import DEFAULT_CORRIDOR, ENTRY_CUT_SPEED_MAX_M_S
    from orbitopt.flightlab.recovery.propagator import BoosterModel

    eng = BoosterModel.load(vehicle_path).engine
    h0 = float(DEFAULT_CORRIDOR.h0_m)
    return [(0.0, 150.0), (-20.0, 80.0), (eng.throttle_min, eng.throttle_max),
            (0.0, 30.0), (-45.0, 45.0), (h0 + 2000.0, 35_000.0),
            (60.0, float(ENTRY_CUT_SPEED_MAX_M_S)), (eng.throttle_min, eng.throttle_max)]


def _train_score(rigs: FlyRig, x: Sequence[float], bounds, tag: str) -> dict[str, Any]:
    """One candidate = the pinned script flown on the nominal plus every
    training scenario (common random numbers: the grid never moves). Score =
    failure fraction - a median fuel-margin tiebreak clipped hard enough that
    no margin reading can buy back one lost landing."""
    jobs = [{"arm": "train", "seed": seed, "idx": idx, "script_tag": tag,
             "run_dir": str(rigs.run_dir), "target": rigs.target,
             "vehicle_path": str(E508.VEHICLE), "uncertainty": rigs.uncertainty,
             "options": rigs.options("fixed", x), "draw": [seed, idx]}
            for seed, idx in TRAIN_GRID
            if _grid_key("train", seed, idx, tag) not in rigs.fly]
    rigs.fly_jobs(jobs)
    recs = [rigs.fly[_grid_key("train", seed, idx, tag)] for seed, idx in TRAIN_GRID]
    nominal = rigs.fly_nominal(x, tag)
    rigs.save()
    landed = [1 if r.get("outcome") == "landed" else 0 for r in recs + [nominal]]
    margins = [r.get("fuel_margin_kg") for r in recs + [nominal]
               if r.get("fuel_margin_kg") is not None]
    med = float(np.median(margins)) if margins else -50_000.0
    med = max(-30_000.0, min(30_000.0, med))
    fail_frac = 1.0 - sum(landed) / len(landed)
    return {"x": [float(v) for v in x], "score": fail_frac - med / 1e6,
            "fail_frac": fail_frac, "landed_train": int(sum(landed)),
            "n_train": len(landed), "margin_med_kg": med,
            "per_scenario": "".join(str(v) for v in landed)}


def robust_train(rigs: FlyRig, x0: Sequence[float], bounds, lam: int = 8, gens: int = 4,
                 sigma_frac: float = 0.04, stage_tag: str = "") -> dict[str, Any]:
    """(1+lambda)-ES warm-started at the recorded script, sigma a trust region
    (a fraction of the clamp box): the whole point of this arm is a REGULARIZED
    search -- the E5-08 ensemble lesson is that a heavily-optimized script
    overfits (1-4/36 on the hold-out), so the budget is deliberately tiny.
    ``stage_tag`` namespaces candidate cache keys per trust-region stage."""
    rng = np.random.default_rng(ES_SEED)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    sigma = sigma_frac * (hi - lo)
    baseline = _train_score(rigs, x0, bounds, tag="gen0")
    best = baseline
    trace: list[dict[str, Any]] = [{"gen": 0, "child": -1, **{k: baseline[k] for k in
                                        ("score", "fail_frac", "landed_train", "margin_med_kg")}}]
    for gen in range(1, gens + 1):
        children = []
        for j in range(lam):
            x = np.clip(np.asarray(best["x"]) + sigma * rng.standard_normal(len(x0)), lo, hi)
            child = _train_score(rigs, x, bounds, tag=f"{stage_tag}gen{gen}_c{j}")
            children.append(child)
            trace.append({"gen": gen, "child": j, **{k: child[k] for k in
                          ("score", "fail_frac", "landed_train", "margin_med_kg")}})
        champ = min(children, key=lambda c: c["score"])
        if champ["score"] < best["score"]:
            best = champ
    return {"method": "(1+lambda)-ES, warm start = recorded script, lexicographic score = "
                      "training failure fraction - clipped median fuel-margin tiebreak, "
                      "CRN training grid",
            "lambda": lam, "generations": gens, "sigma_frac": sigma_frac,
            "train_seed": ES_SEED, "candidates_evaluated": len(trace),
            "baseline": {k: baseline[k] for k in ("score", "fail_frac", "landed_train",
                                                  "n_train", "margin_med_kg", "per_scenario")},
            "trace": trace, "best": {k: best[k] for k in
                                     ("x", "score", "fail_frac", "landed_train", "n_train",
                                      "margin_med_kg", "per_scenario")}}


def robust_select(rigs: FlyRig, x0: Sequence[float], bounds, lam: int, gens: int,
                  sigmas: Sequence[float]) -> dict[str, Any]:
    """Widen the trust region stage by stage, keep the incumbent unless a
    strictly better training score is found -- ties are broken toward the
    SMALLER trust region (regularization first)."""
    stages = []
    selected = None
    for si, sf in enumerate(sigmas):
        st = robust_train(rigs, x0, bounds, lam=lam, gens=gens, sigma_frac=sf,
                          stage_tag="" if si == 0 else f"w{int(round(sf * 100))}_")
        st["stage"] = si
        stages.append(st)
        if selected is None or st["best"]["score"] < selected["score"] - 1e-12:
            selected = {**st["best"], "stage": si, "sigma_frac": sf}
    return {"stages": [{k: st[k] for k in ("stage", "sigma_frac", "lambda", "generations",
                                           "candidates_evaluated", "baseline", "best", "trace")}
                       for st in stages],
            "selected": selected,
            "candidates_evaluated": sum(st["candidates_evaluated"] for st in stages)}


# --------------------------------------------------------------------------
# paired, cluster-aware statistics (E7-05 conventions)
# --------------------------------------------------------------------------
def cluster_counts(records: Mapping[tuple[int, int], Mapping[str, Any]]) -> list[list[int]]:
    by_seed: dict[int, list[int]] = {}
    for (seed, _idx), r in records.items():
        cell = by_seed.setdefault(int(seed), [0, 0])
        cell[1] += 1
        cell[0] += 1 if r.get("outcome") == "landed" else 0
    return [by_seed[s] for s in sorted(by_seed)]


def cluster_bootstrap_ci(counts: Sequence[tuple[int, int]], seed: int = BOOTSTRAP_SEED,
                        trials: int = 4000, level: float = 0.95) -> list[float] | None:
    from orbitopt.flightlab.recovery.reliability import cluster_bootstrap_rate_ci

    ci = cluster_bootstrap_rate_ci([(int(a), int(b)) for a, b in counts],
                                  trials=trials, seed=seed, level=level)
    return [round(float(v), 4) for v in ci] if ci else None


def paired_diff_cluster_ci(rec_a: Mapping[tuple[int, int], Mapping[str, Any]],
                           rec_b: Mapping[tuple[int, int], Mapping[str, Any]],
                           grid: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Percentile bootstrap CI of the PAIRED rate difference (A - B) under the
    whole-seed resample: the cluster is the resampled unit, within a cluster
    the paired difference is taken at its empirical mean."""
    by_seed: dict[int, list[float]] = {}
    for seed, idx in grid:
        a, b = rec_a.get((seed, idx)), rec_b.get((seed, idx))
        if a is None or b is None:
            continue
        d = (1.0 if a.get("outcome") == "landed" else 0.0) \
            - (1.0 if b.get("outcome") == "landed" else 0.0)
        by_seed.setdefault(int(seed), []).append(d)
    clusters = [sum(v) / len(v) for v in by_seed.values()]
    if not clusters:
        return {"diff": None, "ci95": None}
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(clusters)
    diffs = sorted(sum(clusters[rng.randrange(k)] for _ in range(k)) / k
                   for _ in range(4000))
    pooled = sum(sum(v) for v in by_seed.values()) / sum(len(v) for v in by_seed.values())
    return {"diff": round(pooled, 4),
            "ci95": [round(diffs[int(0.025 * len(diffs))], 4),
                     round(diffs[min(len(diffs) - 1, int(math.ceil(0.975 * len(diffs))) - 1)], 4)]}


def pair_stats(rig_a: str, rec_a, rig_b: str, rec_b, grid, alpha: float = 0.05
               ) -> dict[str, Any]:
    from orbitopt.flightlab.recovery.stats import mcnemar_test

    pairs = []
    table = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for key in grid:
        a, b = rec_a.get(key), rec_b.get(key)
        if a is None or b is None:
            continue
        la, lb = a.get("outcome") == "landed", b.get("outcome") == "landed"
        pairs.append((la, lb))
        table["both" if la and lb else "a_only" if la else "b_only" if lb else "neither"] += 1
    res = mcnemar_test(pairs)
    res["table"] = table
    res["arms"] = [rig_a, rig_b]
    res["paired_ci"] = paired_diff_cluster_ci(rec_a, rec_b, grid)
    return res


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
SCRIPT_PARAM_NAMES = ["boostback delay [s]", "boostback aim past pad [km]", "boostback throttle",
                      "boostback pitch [deg]", "boostback yaw trim [deg]",
                      "entry trigger alt [m]", "entry cutoff speed [m/s]", "entry throttle"]


def _script_delta(a: Sequence[float] | None, b: Sequence[float] | None,
                  bounds) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False}
    da = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    width = np.array([hi - lo for lo, hi in bounds], dtype=np.float64)
    return {"available": True,
            "delta": [round(float(v), 4) for v in da],
            "norm_box": round(float(np.sqrt(np.sum((da / width) ** 2))), 4),
            "max_param": SCRIPT_PARAM_NAMES[int(np.argmax(np.abs(da / width)))]}


def reason_mix(records: Mapping[tuple[int, int], Mapping[str, Any]]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for r in records.values():
        if r.get("outcome") != "landed":
            key = str(r.get("reason") or "unknown")
            mix[key] = mix.get(key, 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def _read_sealed() -> dict[str, Any]:
    out = {}
    for name in ("mc-summary-pinned.json", "mc-summary.json"):
        path = E508.REPORTS_ROOT / name
        if path.exists():
            doc = json.loads(path.read_text())
            out[name.removesuffix(".json")] = {
                "successes": doc.get("successes"), "n": doc.get("n"),
                "success_rate": doc.get("success_rate"),
                "per_seed": {s: {"successes": d.get("successes"), "n": d.get("n")}
                             for s, d in (doc.get("per_seed") or {}).items()}}
    return out


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
def _md(rep: Mapping[str, Any]) -> str:
    a = rep["arms_summary"]
    lines: list[str] = []
    add = lines.append
    add("# E10-05 recovery planner robust optimization (DC7)")
    add("")
    add("Provenance in the JSON (`provenance`, `seed_manifest`). Why did per-sample "
        "replanning score 3/36 where the fixed script scored 11/36, and can one "
        "scenario-robustly-planned script beat both? Paired on the sealed DC7 hold-out.")
    add("")
    add("## 1. Verdict")
    add("")
    add("| arm | landed | cluster CI95 | Wilson CI95 |")
    add("|---|---|---|---|")
    for arm in ARMS:
        s = a[arm]
        add(f"| `{arm}` | {s['successes']}/{s['n']} | {s['cluster_ci95']} | {s['wilson_ci95']} |")
    add("")
    add("Paired, Holm-corrected (family of 3):")
    add("")
    add("| pair | discordant | diff | p (Holm) |")
    add("|---|---|---|---|")
    for p in rep["pairwise"]:
        add(f"| {p['arms'][0]} vs {p['arms'][1]} | "
            f"{p['table']['a_only']}-{p['table']['b_only']} | {p['paired_ci']['diff']} | "
            f"{p['adjusted_p'] if p.get('adjusted_p') is not None else p['exact']['p_value']} |")
    add("")
    add("## 2. Why replan lost (diagnosis)")
    add("")
    add(f"2x2 (fixed x replan40): both {rep['diagnosis']['flip_table']['both']}, "
        f"fixed-only {rep['diagnosis']['flip_table']['a_only']}, "
        f"replan-only {rep['diagnosis']['flip_table']['b_only']}, "
        f"neither {rep['diagnosis']['flip_table']['neither']}.")
    add("")
    add("Reason mix per arm:")
    add("")
    for arm in ARMS:
        add(f"- `{arm}`: " + (", ".join(f"`{k}` {v}" for k, v in a[arm]["reasons"].items())
                              or "no failures"))
    add("")
    add("Budget ladder on the diagnostic subset (fixed-landed ∪ replan-landed ∪ first "
        f"{rep['diagnosis']['budget']['n_failed_controls']} both-failed controls, n = "
        f"{rep['diagnosis']['budget']['n_samples']}):")
    add("")
    add("| max_fevals | landed | fixed-landed kept | both-failed rescued |")
    add("|---|---|---|---|")
    for row in rep["diagnosis"]["budget"]["rows"]:
        add(f"| {row['max_fevals']} | {row['landed']} | {row['fixed_kept']} | {row['rescued']} |")
    add("")
    add("## 3. Robust arm")
    add("")
    r = rep["robust"]["training"]
    sel = r["selected"]
    add("(1+8)-ES trust-region stages, trained on seeds "
        f"{rep['seed_manifest']['train_seeds']} x {rep['seed_manifest']['train_per_seed']} "
        f"(+ nominal), {r['candidates_evaluated']} candidates total:")
    add("")
    add("| stage | sigma | candidates | best landed (train) | score |")
    add("|---|---|---|---|---|")
    for st in r["stages"]:
        add(f"| {st['stage']} | {st['sigma_frac']*100:.0f}% | {st['candidates_evaluated']} | "
            f"{st['best']['landed_train']}/{st['best']['n_train']} | {st['best']['score']:.4f} |")
    add("")
    add("Selected script = stage {} (sigma {}%); training landed {}/{}, hold-out landed "
        "{}/{}. A tie keeps the incumbent (smaller region): no local perturbation beat the "
        "recorded script on training.".format(
            sel["stage"], sel["sigma_frac"] * 100, sel["landed_train"], sel["n_train"],
            a["robust"]["successes"], a["robust"]["n"]))
    add("")
    add("Script movement from the recorded vector (robust arm, per parameter): "
        + ", ".join(f"{n} {d:+.2f}" for n, d in
                    zip(SCRIPT_PARAM_NAMES, rep["robust"]["script_delta"]["delta"])))
    add("")
    add("## 4. Seed manifest and split")
    add("")
    add("See `seed_manifest` in the JSON: hold-out {} x {} (the sealed DC7 grid), training "
        "{} x {}, recorded search seed {}, ES seed {}. Training and hold-out seeds are "
        "disjoint by construction (asserted in the bench).".format(
            rep["seed_manifest"]["holdout_seeds"], rep["seed_manifest"]["n_per_seed"],
            rep["seed_manifest"]["train_seeds"], rep["seed_manifest"]["train_per_seed"],
            rep["seed_manifest"]["recorded_search_seed"], rep["seed_manifest"]["es_seed"]))
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fly-cache", type=Path, default=None,
                    help="persist every flown sample (arm|seed|idx|script); resume/re-run")
    ap.add_argument("--lam", type=int, default=8, help="ES children per generation")
    ap.add_argument("--gens", type=int, default=4, help="ES generations")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.04, 0.10],
                    help="trust-region stages for the robust ES (fraction of the clamp box)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, required=True)
    args = ap.parse_args(argv)

    t_start = time.time()
    rig = FlyRig(args.workers, args.fly_cache)
    try:
        from orbitopt.flightlab.recovery.stats import holm_adjust, success_rate_interval

        recipe = dict(json.loads(E508.SCRIPT_JSON.read_text())["recipe"])
        x_recorded = [float(v) for v in recipe["initial_guess"]]
        bounds = script_bounds(E508.VEHICLE)

        # ---- robust arm training: widen the trust region stage by stage
        # (stage 0 = 4% matches the E5-08 overfit lesson; a wider stage only
        # takes over on a STRICT training-score improvement, so ties go to the
        # smaller region). The training grid never touches the hold-out seeds.
        best = robust_select(rig, x_recorded, bounds, args.lam, args.gens,
                             [s for s in args.sigmas if s > 0.0] or [0.04])
        if args.fly_cache is not None:
            Path(args.fly_cache).with_suffix(".robust.json").write_text(
                json.dumps(best, indent=1) + "\n")
        rig.save()
        robust_x = best["selected"]["x"]
        # when the robust search kept the incumbent, the robust arm IS the
        # fixed arm -- reuse its records rather than re-fly (bit-for-bit equal)
        rob_same = max(abs(a - b) for a, b in zip(robust_x, x_recorded)) < 1e-9
        rob_tag = "v0" if rob_same else "rob"

        # ---- the three hold-out arms on the identical grid
        for arm in ARMS:
            if arm == "robust" and rob_same:
                continue
            script = x_recorded if arm == "fixed" else robust_x if arm == "robust" else None
            rig.fly_jobs(rig.jobs_for(arm, HOLDOUT_GRID, script, rob_tag if arm == "robust" else "v0"))
        records = {arm: rig.arm_records(arm, HOLDOUT_GRID,
                                        rob_tag if arm == "robust" else "v0") for arm in ARMS}
        if rob_same:
            records["robust"] = dict(records["fixed"])

        # ---- diagnostic subset + budget ladder
        diag_keys: list[tuple[int, int]] = []
        failed_controls: list[tuple[int, int]] = []
        for key in HOLDOUT_GRID:
            fa = records["fixed"][key].get("outcome") == "landed"
            fb = records["replan40"][key].get("outcome") == "landed"
            if fa or fb:
                diag_keys.append(key)
            elif len(failed_controls) < DIAG_N_FAILED:
                failed_controls.append(key)
        diag_keys += failed_controls
        diag_keys.sort()
        diag_grid = [k for k in HOLDOUT_GRID if k in set(diag_keys)]
        for budget, arm in BUDGETS.items():
            rig.fly_jobs(rig.jobs_for(arm, diag_grid, None, "v0"))
        diag_records = {arm: rig.arm_records(arm, diag_grid) for arm in BUDGETS.values()}

        # ---- statistics
        arms_summary = {}
        for arm in ARMS:
            recs = records[arm]
            counts = cluster_counts(recs)
            succ = sum(c[0] for c in counts)
            n = sum(c[1] for c in counts)
            w = success_rate_interval(succ, n)
            arms_summary[arm] = {
                "successes": succ, "n": n, "rate": round(succ / n, 4),
                "per_seed": {str(s): dict(zip(("successes", "n"), c))
                             for s, c in zip(sorted({k[0] for k in recs}), counts)},
                "cluster_ci95": cluster_bootstrap_ci(counts),
                "wilson_ci95": [round(w["lower"], 4), round(w["upper"], 4)],
                "reasons": reason_mix(recs)}

        pairs = [
            pair_stats("fixed", records["fixed"], "replan40", records["replan40"], HOLDOUT_GRID),
            pair_stats("fixed", records["fixed"], "robust", records["robust"], HOLDOUT_GRID),
            pair_stats("replan40", records["replan40"], "robust", records["robust"],
                       HOLDOUT_GRID)]
        holm = holm_adjust([p["exact"]["p_value"] for p in pairs])
        for p, h in zip(pairs, holm):
            p["adjusted_p"] = h["adjusted_p"]
            p["holm_reject"] = h["reject"]

        flipped_down = [k for k in HOLDOUT_GRID
                        if records["fixed"][k].get("outcome") == "landed"
                        and records["replan40"][k].get("outcome") != "landed"]
        flipped_up = [k for k in HOLDOUT_GRID
                      if records["fixed"][k].get("outcome") != "landed"
                      and records["replan40"][k].get("outcome") == "landed"]
        flip_rows = []
        for k in flipped_down + flipped_up:
            f, r = records["fixed"][k], records["replan40"][k]
            flip_rows.append({
                "seed": k[0], "idx": k[1],
                "direction": "down" if k in flipped_down else "up",
                "fixed_reason": f.get("reason"), "replan_reason": r.get("reason"),
                "script_delta": _script_delta(f.get("script"), r.get("script"), bounds),
                "fixed_miss_m": f.get("miss_m"), "replan_miss_m": r.get("miss_m"),
                "fixed_fuel_margin_kg": f.get("fuel_margin_kg"),
                "replan_fuel_margin_kg": r.get("fuel_margin_kg"),
                "replan_checks": r.get("checks")})
        budget_rows = []
        for budget, arm in sorted(BUDGETS.items()):
            recs = diag_records[arm]
            landed = sum(1 for r in recs.values() if r.get("outcome") == "landed")
            kept = sum(1 for k in diag_grid
                       if records["fixed"][k].get("outcome") == "landed"
                       and recs[k].get("outcome") == "landed")
            rescued = sum(1 for k in diag_grid
                          if records["fixed"][k].get("outcome") != "landed"
                          and recs[k].get("outcome") == "landed")
            budget_rows.append({"max_fevals": budget, "arm": arm, "landed": landed,
                                "n_samples": len(recs), "fixed_kept": kept, "rescued": rescued})

        nominal_fixed = rig.fly_nominal(x_recorded, "gen0")
        rig.save()

        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
        from importlib.metadata import version as _pkg_version
        report = {
            "generated_by": "benchmarks/bench_e10_05_recovery_planner_robust.py",
            "provenance": {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_sha": git_sha, "python": sys.version.split()[0],
                "numpy": np.__version__, "gfold": _pkg_version("gfold"),
                "wall_s": round(time.time() - t_start, 1),
                "workers": args.workers,
                "sealed_reference_read_only": _read_sealed()},
            "seed_manifest": {
                "holdout_seeds": HOLDOUT_SEEDS, "n_per_seed": N_PER_SEED,
                "train_seeds": TRAIN_SEEDS, "train_per_seed": TRAIN_PER_SEED,
                "recorded_search_seed": RECORDED_SEARCH_SEED, "es_seed": ES_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "disjoint": "training seeds, hold-out seeds and the recorded search seed "
                            "are pairwise disjoint; the ES rng is its own stream",
                "uncertainty": "the vehicle file's FL-602 uncertainty tags via "
                               "vehicle_dispersions() -> UncertaintyModel, per-sample "
                               "numpy stream [seed, idx] (identical draws for every arm "
                               "and every ES candidate: common random numbers)"},
            "arms_summary": arms_summary,
            "pairwise": pairs,
            "diagnosis": {
                "flip_table": pairs[0]["table"],
                "flips": flip_rows,
                "budget": {"n_samples": len(diag_grid), "n_failed_controls": len(failed_controls),
                           "diag_grid": [[int(s), int(i)] for s, i in diag_grid],
                           "rows": budget_rows},
                "nominal_fixed": {"outcome": nominal_fixed.get("outcome"),
                                  "reason": nominal_fixed.get("reason")}},
            "robust": {
                "training": best,
                "script_delta": _script_delta(x_recorded, robust_x, bounds),
                "param_names": SCRIPT_PARAM_NAMES},
            "holdout_samples": {arm: {f"{k[0]}|{k[1]}": records[arm].get(k) for k in HOLDOUT_GRID}
                                for arm in ARMS},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_md(report), encoding="utf-8")
        print(f"wrote {args.out}\nwrote {args.md}")
        for arm in ARMS:
            print(f"  {arm}: {arms_summary[arm]['successes']}/{arms_summary[arm]['n']}")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    raise SystemExit(main())
