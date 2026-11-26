#!/usr/bin/env python3
"""E11R-01 (task #414): robust CANDIDATE SELECTION for the E11-06 continuous-burn search.

The E12-00 gate record archived the reproduction variance of the sealed E11-06 champion
plainly: the recovery search stops at the FIRST feasible design, so the stop point -- and
with it the corridor margin -- is seed dependent, and the five re-derived champions landed
between 6/36 and 33/36 on the FL-304 pinned set (``candidate-set.json`` in the gate
archive). This campaign changes the SELECTION RULE, not the search box, the vehicle, the
dispersion model or any graded limit:

* the DE search keeps running past the first feasible design (the callback never stops
  it) and HARVESTS every feasible design it evaluates -- a candidate pool, first-come;
* each pooled candidate is scored on fresh TRAINING Monte-Carlo seeds only (pinned
  script, FL-304 dispersion model): landing count first, then the mean corridor fuel
  margin, then the mean corridor miss -- robustness, not the first stop point, ranks
  the pool;
* the argmax of that training score is the seed's selection, flown PINNED on the sealed
  hold-out grid (seeds 8/9/10 x 12 -- the same 36 samples the archived 33/36 was graded
  on), sample-for-sample paired against the re-flown sealed champion and against the
  first-feasible stop point of the SAME search run.

Acceptance reads out of the report: the dispersion of the hold-out rate ACROSS SEARCH
SEEDS (selected arms vs the archived first-feasible table), the cluster-bootstrap CIs,
the McNemar (exact, Holm-corrected) pairings against the sealed 33/36 champion, and the
constraint audit: no graded limit is relaxed anywhere and the v3_pp grid-fin absolute
drag stays a DESIGN ASSUMPTION that every check inherits.

Run (remote-friendly; search + training MC + hold-out ~30-60 min at --workers 24,
resume via --cache):

    E11_ENGINE_STAGES=1 python benchmarks/bench_e11r_01_robust_candidate_selection.py \
        --workers 24 --cache /tmp/e11r01_cache.json \
        -o docs/benchmarks/e11r_01_robust_candidate_selection.json \
        --md docs/benchmarks/e11r_01_robust_candidate_selection.md

Smoke scale (plumbing check): ``--search-seeds 1 --maxiter 8 --pool-cap 2 --train-seeds
901 --train-n 2 --holdout-seeds 8 --n-per-seed 2``. ``E11_ENGINE_STAGES=1`` reproduces
the sealed E11-06 gate environment (13 -> 5 -> 3 engine shutdown inside the continuous
burn); the bench refuses to run without it. The sealed records are read, never written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmarks"))
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker

# --------------------------------------------------------------------------
# seed manifest (acceptance: training and hold-out seeds disjoint by construction)
# --------------------------------------------------------------------------
SEARCH_SEEDS = [1, 2, 3, 4, 5]          # the E12-00 archived candidate-set seeds
TRAIN_SEEDS = [901, 902, 903]           # this campaign's training MC (fresh)
TRAIN_PER_SEED = 4                      # -> 12 training scenarios per candidate
HOLDOUT_SEEDS = [8, 9, 10]              # the sealed FL-304 grid that graded 33/36
N_PER_SEED = 12
BOOTSTRAP_SEED = 20261126               # every bootstrap in this bench
PRIOR_STREAMS = {5, 6, 7, 8, 9, 10,     # sealed E5-08 / E11-06 gate grids
                 401, 402,              # E10-05 training
                 601, 602,              # E10-03 training
                 501, 502, 503, 504, 505, 506, 507, 508,   # E10-03/E11-03 hold-out
                 701, 702}              # E11-03 co-design training
assert not (set(TRAIN_SEEDS) & (set(HOLDOUT_SEEDS) | PRIOR_STREAMS)), \
    "training seeds must be disjoint from the hold-out and from every prior stream"
assert set(SEARCH_SEEDS).isdisjoint(set(HOLDOUT_SEEDS) | set(TRAIN_SEEDS))

GATE = REPO / "tests/data/mission/reference/e11-06-starbase-continuous-burn"
GATE_REPORTS = GATE.parent / "e11-06-starbase-continuous-burn-reports"
SEALED_CAND = GATE / "candidate.json"                 # the archived 33/36 champion
CAND_SET = GATE_REPORTS / "candidate-set.json"        # the archived 6/36..33/36 table
FINS = "v3_pp"                                        # DESIGN ASSUMPTION drag, unchanged
MAX_HOLDS = 36                                        # 3 sealed hold-out clusters


# --------------------------------------------------------------------------
# phase 1: the no-stop search with a feasibility harvest (runs in workers)
# --------------------------------------------------------------------------
_C = None
_U = None
_HARVEST: list[tuple] = []
_HARVEST_X: list[np.ndarray] = []
_STOP_X: list[np.ndarray] = []


def _worker_init():
    global _C, _U
    if _C is not None:
        return
    import e11_continuous_burn_search as C
    import e11_unpowered_search as U
    from orbitopt.flightlab.contract import load_target

    U._init(FINS, 1.0)
    C.U._CTX["recipe"] = {**C.RECIPE, "dt_opt": 0.5, "dt_final": 0.25}
    C.U._CTX["target"] = load_target(U.B.SCEN)
    _C, _U = C, U


def _key(x: np.ndarray) -> tuple:
    lo = np.array([b[0] for b in _C.BOUNDS])
    hi = np.array([b[1] for b in _C.BOUNDS])
    return tuple(np.round((np.asarray(x, dtype=float) - lo) / (hi - lo), 5))


def _harvest_objective(x) -> float:
    p = float(_C.objective(x))
    if p <= -0.5:                                   # feasible: the penalty floor is exactly -1
        k = _key(x)
        if k not in _HARVEST:
            _HARVEST.append(k)
            _HARVEST_X.append(np.array(x, dtype=float))
    return p


def search_worker(job: dict[str, Any]) -> dict[str, Any]:
    """One no-stop DE run: identical protocol to the sealed search (same bounds, box,
    population, rng seed, mutation/recombination, deferred updates, sobol init) except
    that nothing stops it at the first feasible design, and every feasible design it
    evaluates is harvested. In-process serial evaluation: same trial-vector order, no
    stop rule needed -- the callback only RECORDS the first generation whose champion
    (the old rule's stop point) is feasible, so the old arm is reconstructable."""
    from scipy.optimize import differential_evolution

    _worker_init()
    t0 = time.time()
    _HARVEST.clear()
    _HARVEST_X.clear()
    _STOP_X.clear()

    def cb(xk, convergence=None):
        if not _STOP_X and _C.objective(np.asarray(xk, dtype=float)) <= -0.5:
            _STOP_X.append(np.array(xk, dtype=float))
        return False

    res = differential_evolution(
        _harvest_objective, _C.BOUNDS, seed=int(job["seed"]), popsize=int(job["popsize"]),
        maxiter=int(job["maxiter"]), tol=0, mutation=(0.5, 1.0), recombination=0.8,
        polish=False, updating="deferred", init="sobol", callback=cb)
    return {"seed": int(job["seed"]),
            "pool": [[float(v) for v in x] for x in _HARVEST_X[:int(job["pool_cap"])]],
            "pool_total": len(_HARVEST_X),
            "first_x": [float(v) for v in _HARVEST_X[0]] if _HARVEST_X else None,
            "stop_x": [float(v) for v in _STOP_X[0]] if _STOP_X else None,
            "nfev": int(res.nfev), "runtime_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------
# phases 2/3: pinned FL-304 Monte Carlo (training scoring and hold-out arms)
# --------------------------------------------------------------------------
def mc_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Pinned-script MC of one candidate on one seed (the E11-06 gate mc script's
    method: disperse around the separation state of the sealed ascent the mission
    flies; workers=1 -> in-process, the vehicle yaml lives in this task's temp dir)."""
    import tempfile
    from dataclasses import replace

    import orbitopt.flightlab.recovery.montecarlo as mc
    import e11_continuous_burn_fly as F
    import e11_unpowered_search as U
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    def sep_ecef(run_):
        sep = booster_state_from_run(run_, separation_handoff(run_))
        if sep.frame == "ECEF":
            return sep
        r, v = _to_ecef_state(sep)
        return replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r),
                       velocity_m_s=tuple(float(c) for c in v))

    mc.separation_state_ecef = sep_ecef
    target = load_target(U.B.SCEN)
    run = read_run(U.B.ASCENT)
    x = np.asarray(job["x"][:13], dtype=float)
    with tempfile.TemporaryDirectory() as tmp, \
            freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        veh = U.B._vehicle_yaml(job["fins"], Path(tmp))
        r = mc.run_monte_carlo(target, veh, run, seed=int(job["seed"]), n=int(job["n"]),
                               workers=1, plan_options=F.plan_options(x, F.LANDING))
    return [{"idx": int(s["idx"]), "outcome": s.get("outcome"), "reason": s.get("reason"),
             "corridor_miss_m": s.get("corridor_miss_m"),
             "corridor_fuel_margin_kg": s.get("corridor_fuel_margin_kg")} for s in r.samples]


def run_mc_jobs(cache: dict, xs: Mapping[str, Sequence[float]], seeds: Sequence[int], n: int,
                workers: int) -> None:
    """Fly every (arm, seed) MC job not yet fully cached into cache['mc'] (resumable)."""
    todo = [{"arm": a, "seed": int(s), "x": list(xs[a]), "fins": FINS, "n": n}
            for a in sorted(xs) for s in seeds
            if not all(f"{a}|{s}|{i}" in cache["mc"] for i in range(n))]
    if not todo:
        return
    print(f"MC: {len(todo)} seed-jobs ({sum(t['n'] for t in todo)} samples) to fly", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, recs in zip(todo, pool.map(mc_worker, todo)):
            for r in recs:
                cache["mc"][f"{job['arm']}|{job['seed']}|{r['idx']}"] = r
            done += 1
            print(f"  mc {done}/{len(todo)} {job['arm']} seed {job['seed']}: "
                  f"{sum(1 for r in recs if r['outcome'] == 'landed')}/{len(recs)} "
                  f"landed ({time.time() - t0:.0f}s)", flush=True)
            save_cache(cache)


# --------------------------------------------------------------------------
# cache (resumability: the campaign is remote-scale compute)
# --------------------------------------------------------------------------
def save_cache(cache: dict) -> None:
    p = Path(cache["_path"])
    if str(p) == "/dev/null":
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({k: v for k, v in cache.items() if not k.startswith("_")}))
    tmp.replace(p)


def load_cache(path: Path) -> dict:
    c = {"search": {}, "mc": {}, "_path": str(path)}
    if path.is_file():
        c.update(json.loads(path.read_text()))
    return c


# --------------------------------------------------------------------------
# scoring / selection / statistics
# --------------------------------------------------------------------------
def train_score(cache: dict, arm: str) -> dict[str, Any]:
    recs = [cache["mc"][f"{arm}|{s}|{i}"] for s in TRAIN_SEEDS for i in range(TRAIN_PER_SEED)
            if f"{arm}|{s}|{i}" in cache["mc"]]
    fm = [float(r["corridor_fuel_margin_kg"]) for r in recs if r.get("corridor_fuel_margin_kg") is not None]
    ms = [float(r["corridor_miss_m"]) for r in recs if r.get("corridor_miss_m") is not None]
    return {"n": len(recs), "successes": sum(1 for r in recs if r["outcome"] == "landed"),
            "mean_fuel_margin_kg": round(float(np.mean(fm)), 1) if fm else None,
            "mean_miss_m": round(float(np.mean(ms)), 1) if ms else None}


def select(pool_arms: Sequence[str], scores: Mapping[str, Mapping[str, Any]]) -> str:
    """argmax of (training landings, mean corridor fuel margin, -mean corridor miss),
    ties broken by first-harvest order -- deterministic."""
    def rank(a: str) -> tuple:
        s = scores[a]
        return (s["successes"],
                s["mean_fuel_margin_kg"] if s["mean_fuel_margin_kg"] is not None else -1e18,
                -(s["mean_miss_m"] if s["mean_miss_m"] is not None else 1e18),
                -pool_arms.index(a))
    return max(pool_arms, key=rank)


def _holdout_grid() -> list[tuple[int, int]]:
    return [(s, i) for s in HOLDOUT_SEEDS for i in range(N_PER_SEED)]


def holdout_stats(cache: dict, arm: str) -> dict[str, Any]:
    from orbitopt.flightlab.recovery.reliability import cluster_bootstrap_rate_ci

    recs = {k: cache["mc"][f"{arm}|{k[0]}|{k[1]}"] for k in _holdout_grid()
            if f"{arm}|{k[0]}|{k[1]}" in cache["mc"]}
    by_seed: dict[int, list[int]] = {}
    for (seed, _i), r in recs.items():
        cell = by_seed.setdefault(int(seed), [0, 0])
        cell[1] += 1
        cell[0] += 1 if r["outcome"] == "landed" else 0
    counts = [by_seed[s] for s in sorted(by_seed)]
    ok, n = sum(c[0] for c in counts), sum(c[1] for c in counts)
    ci = cluster_bootstrap_rate_ci([(a, b) for a, b in counts], trials=4000, seed=BOOTSTRAP_SEED) \
        if counts else None
    reasons: dict[str, int] = {}
    for r in recs.values():
        if r["outcome"] != "landed":
            reasons[str(r["reason"])] = reasons.get(str(r["reason"]), 0) + 1
    return {"arm": arm, "successes": ok, "n": n, "rate": round(ok / n, 4) if n else None,
            "cluster_ci95": [round(float(v), 4) for v in ci] if ci else None,
            "per_seed": {str(s): {"successes": c[0], "n": c[1]} for s, c in zip(sorted(by_seed), counts)},
            "reasons": reasons}


def pair_vs(cache: dict, a: str, b: str) -> dict[str, Any]:
    """Exact McNemar on the shared hold-out grid, sample-for-sample paired."""
    from orbitopt.flightlab.recovery.stats import mcnemar_test

    pairs, table = [], {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for seed, idx in _holdout_grid():
        ka, kb = f"{a}|{seed}|{idx}", f"{b}|{seed}|{idx}"
        if ka not in cache["mc"] or kb not in cache["mc"]:
            continue
        la = cache["mc"][ka]["outcome"] == "landed"
        lb = cache["mc"][kb]["outcome"] == "landed"
        pairs.append((la, lb))
        table["both" if la and lb else "a_only" if la else "b_only" if lb else "neither"] += 1
    res = mcnemar_test(pairs)
    return {"arms": [a, b], "n_pairs": len(pairs), "table": table,
            "p_raw": float(res["exact"]["p_value"]),
            "diff": sum(1 for la, _ in pairs if la) - sum(1 for _, lb in pairs if lb)}


def pooled_selected_ci(cache: dict, selected: Mapping[int, str]) -> dict[str, Any]:
    """One rate over the union of the selected arms' hold-out grids, bootstrapped over
    the (arm, seed) clusters -- the script-to-script dispersion the acceptance asks
    for, not 180 pretend-i.i.d. draws."""
    from orbitopt.flightlab.recovery.reliability import cluster_bootstrap_rate_ci

    counts = []
    for s in sorted(selected):
        cell = [0, 0]
        for hs in HOLDOUT_SEEDS:
            for i in range(N_PER_SEED):
                r = cache["mc"].get(f"{selected[s]}|{hs}|{i}")
                if r is not None:
                    cell[1] += 1
                    cell[0] += 1 if r["outcome"] == "landed" else 0
        if cell[1]:
            counts.append(tuple(cell))
    ci = cluster_bootstrap_rate_ci(counts, trials=4000, seed=BOOTSTRAP_SEED) if counts else None
    ok, n = sum(c[0] for c in counts), sum(c[1] for c in counts)
    return {"successes": ok, "n": n, "rate": round(ok / n, 4) if n else None,
            "clusters": len(counts), "cluster_ci95": [round(float(v), 4) for v in ci] if ci else None}


def dispersion(rates: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(rates, dtype=float)
    return {"n": len(a), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "range": round(float(a.max() - a.min()), 4), "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else None}


# --------------------------------------------------------------------------
# constraint audit (acceptance: nothing relaxed; the fin assumption stays visible)
# --------------------------------------------------------------------------
def constraint_audit() -> dict[str, Any]:
    import yaml

    import bench_e11_06_starbase_grid_fin_scaling as B
    scen = yaml.safe_load((REPO / "src/orbitopt/data/flightlab/scenarios/"
                           "leo_starship_class_rtls_e508_starbase.yaml").read_text())
    veh_src = (REPO / "src/orbitopt/data/flightlab/vehicles/"
               "starship_class_rtls_starbase.yaml").read_text()
    graded = {k: float(scen["recovery"][k]) for k in
              ("max_touchdown_speed_m_s", "max_touchdown_lateral_speed_m_s",
               "min_propellant_reserve_kg")}
    graded.update(entry_q_max_pa=250000.0, entry_g_max=12.0)   # planner knobs, sealed values
    audit = {
        "graded_limits": graded,
        "limits_identical_to_sealed_record": (graded["max_touchdown_speed_m_s"] == 3.0
                                              and graded["max_touchdown_lateral_speed_m_s"] == 1.0
                                              and graded["min_propellant_reserve_kg"] == 100000.0),
        "fin_set": FINS,
        "fin_set_definition": B.FIN_SETS[FINS],
        "grid_fin_drag_is_design_assumption": True,
        "grid_fin_assumption_note": (
            "v3_pp grid-fin absolute drag: no public Super Heavy data -- the per-planform "
            "calibration (bench_e11_06 FIN_SETS/PP_CD) is a DESIGN ASSUMPTION, inherited "
            "unchanged here; every check graded in this campaign inherits it"),
        "sealed_vehicle_file_tagged": "DESIGN ASSUMPTION" in veh_src,
        "search_bounds_unchanged": [[float(a), float(b)] for a, b in _C.BOUNDS],
        "landing_law": "sealed E11-06 continuous burn (lit_at_start, 4.0 g cap, TVC 20 deg/s, "
                       "engine stages 13->5->3 under E11_ENGINE_STAGES=1)",
        "pinned_everywhere": "training and hold-out both fly the authored script pinned "
                             "(search=keep): the only object under test is the SELECTION rule",
    }
    assert audit["limits_identical_to_sealed_record"] and audit["sealed_vehicle_file_tagged"]
    return audit


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def write_md(rep: Mapping[str, Any], path: Path) -> None:
    L: list[str] = []
    a = L.append
    a("# E11R-01: robust candidate selection for the E11-06 continuous-burn search\n")
    a("Task #414, epic E11R. Provenance in the JSON (`provenance`, `seed_manifest`). The "
      "E12-00 record showed the sealed search's first-feasible stop leaves the pinned "
      "hold-out rate seed dependent (6/36 to 33/36); this campaign harvests the feasible "
      "pool, ranks it on fresh training seeds by landing count / corridor margin, and "
      "flies the argmax pinned on the sealed 36-sample hold-out, paired against the "
      "sealed champion and the same run's stop point.\n")
    v, d_sel, d_stp = rep["verdict"], rep["verdict"]["dispersion_selected"], rep["verdict"]["dispersion_stop"]
    a("**Verdict.** Hold-out rate across search seeds -- selected arms "
      f"{d_sel['min']:.2f}..{d_sel['max']:.2f} (range {d_sel['range']:.2f}, std {d_sel['std']}), "
      f"first-feasible stop points of the same runs {d_stp['min']:.2f}..{d_stp['max']:.2f} "
      f"(range {d_stp['range']:.2f}, std {d_stp['std']}); archived E12-00 first-feasible "
      f"table {v['archived_table']['min_rate']:.2f}..{v['archived_table']['max_rate']:.2f}. "
      f"Pooled selected {v['pooled_selected']['successes']}/{v['pooled_selected']['n']} "
      f"(cluster CI {v['pooled_selected']['cluster_ci95']}). Sealed champion re-flown: "
      f"{rep['sealed_reproduced']['successes']}/{rep['sealed_reproduced']['n']}.\n")
    a("## Search seeds\n")
    a("| seed | pool kept/total | stop-point train score | selected train score | stop k/36 | selected k/36 | selected cluster CI |")
    a("|---|---|---|---|---|---|---|")
    f = lambda h: "-" if not h or not h["n"] else f"{h['successes']}/{h['n']}"  # noqa: E731
    ci = lambda h: "-" if not h else str(h["cluster_ci95"])  # noqa: E731
    for row in rep["per_seed"]:
        ts = lambda t: "-" if not t else f"{t['successes']}/{t['n']} @ {t['mean_fuel_margin_kg']} kg"  # noqa: E731
        a(f"| {row['seed']} | {row['pool_kept']}/{row['pool_total']} | {ts(row['train_score_stop'])} | "
          f"{ts(row['train_score_selected'])} | {f(row['holdout_stop'])} | "
          f"{f(row['holdout_selected'])} | {ci(row['holdout_selected'])} |")
    a("")
    a("## Paired against the sealed 33/36 champion (exact McNemar, Holm-corrected)\n")
    a("| selected arm | both / a-only / b-only / neither | p raw | p Holm | sig @ 0.05 |")
    a("|---|---|---|---|---|")
    for c in rep["paired_vs_sealed"]:
        t = c["table"]
        a(f"| {c['arms'][0]} | {t['both']}/{t['a_only']}/{t['b_only']}/{t['neither']} | "
          f"{c['p_raw']:.4g} | {c.get('p_holm', float('nan')):.4g} | {c.get('significant')} |")
    a("")
    a("## Selected vs the same run's first-feasible stop (exact McNemar, uncorrected)\n")
    a("| pair | both / a-only / b-only / neither | diff | p raw |")
    a("|---|---|---|---|")
    for c in rep["paired_selected_vs_stop"]:
        t = c["table"]
        a(f"| {' vs '.join(c['arms'])} | {t['both']}/{t['a_only']}/{t['b_only']}/{t['neither']} | "
          f"{c['diff']:+d} | {c['p_raw']:.4g} |")
    a("")
    a("## Constraints (not relaxed)\n")
    ca = rep["constraint_audit"]
    a(f"- graded limits unchanged and identical to the sealed record ({ca['limits_identical_to_sealed_record']}): "
      f"{ca['graded_limits']}.")
    a(f"- grid-fin drag stays a **DESIGN ASSUMPTION**: {ca['grid_fin_assumption_note']}.")
    a(f"- {ca['pinned_everywhere']}; search box and landing law unchanged: {ca['landing_law']}.\n")
    a("## Reproduce\n")
    a("```")
    a(rep["provenance"]["command"])
    a("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    global TRAIN_SEEDS, TRAIN_PER_SEED, HOLDOUT_SEEDS, N_PER_SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--search-seeds", type=int, nargs="+", default=SEARCH_SEEDS)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=15)
    ap.add_argument("--pool-cap", type=int, default=10)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    ap.add_argument("--train-n", type=int, default=TRAIN_PER_SEED)
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=HOLDOUT_SEEDS)
    ap.add_argument("--n-per-seed", type=int, default=N_PER_SEED)
    args = ap.parse_args(argv)
    if not os.environ.get("E11_ENGINE_STAGES"):
        sys.exit("run with E11_ENGINE_STAGES=1 -- the sealed E11-06 gate environment; the "
                 "paired comparison is meaningless against a different landing law")
    TRAIN_SEEDS, TRAIN_PER_SEED = list(args.train_seeds), int(args.train_n)
    HOLDOUT_SEEDS, N_PER_SEED = list(args.holdout_seeds), int(args.n_per_seed)
    t0 = time.time()
    _worker_init()
    ca = constraint_audit()
    cache = load_cache(args.cache) if args.cache else {"search": {}, "mc": {}, "_path": "/dev/null"}

    # -- phase 1: no-stop searches, one DE per worker ------------------------
    todo = [{"seed": s, "popsize": args.popsize, "maxiter": args.maxiter, "pool_cap": args.pool_cap}
            for s in args.search_seeds if str(s) not in cache["search"]]
    if todo:
        print(f"search: {len(todo)} no-stop DE runs (pop {args.popsize}*13="
              f"{args.popsize * 13}, {args.maxiter} gens, no first-feasible stop), workers "
              f"{min(args.workers, len(todo))}", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
            for rec in pool.map(search_worker, todo):
                cache["search"][str(rec["seed"])] = rec
                save_cache(cache)
    for s in args.search_seeds:
        r = cache["search"][str(s)]
        print(f"seed {s}: pool {len(r['pool'])}/{r['pool_total']} feasible, "
              f"nfev {r['nfev']}, {r['runtime_s']}s", flush=True)

    # -- phase 2: training scoring of every kept candidate + stop point ------
    xs: dict[str, Sequence[float]] = {}
    pool_of: dict[int, list[str]] = {}
    for s in args.search_seeds:
        pool_of[s] = []
        for i, x in enumerate(cache["search"][str(s)]["pool"]):
            arm = f"cand-s{s}-{i}"
            xs[arm] = x
            pool_of[s].append(arm)
        if cache["search"][str(s)]["stop_x"] is not None:
            xs[f"stop-s{s}"] = cache["search"][str(s)]["stop_x"]
    run_mc_jobs(cache, xs, TRAIN_SEEDS, TRAIN_PER_SEED, args.workers)
    scores = {a: train_score(cache, a) for a in xs}

    selected: dict[int, str] = {}
    for s in args.search_seeds:
        cand = [a for a in pool_of[s] if scores[a]["n"] > 0]
        if cand:
            selected[s] = select(cand, scores)
            print(f"seed {s}: selected {selected[s]} "
                  f"{scores[selected[s]]} (stop would have flown cand-s{s}-0)", flush=True)
    sealed = json.loads(SEALED_CAND.read_text())
    xs["sealed"] = sealed["x"][:13]
    holdout_xs = {**{selected[s]: xs[selected[s]] for s in selected},
                  **{a: xs[a] for a in xs if a.startswith("stop-")}, "sealed": xs["sealed"]}
    run_mc_jobs(cache, holdout_xs, HOLDOUT_SEEDS, N_PER_SEED, args.workers)

    # -- statistics ----------------------------------------------------------
    hs = {a: holdout_stats(cache, a) for a in holdout_xs}
    fam = [pair_vs(cache, selected[s], "sealed") for s in sorted(selected)]
    if fam:
        from orbitopt.flightlab.recovery.stats import holm_adjust
        for c, row in zip(fam, holm_adjust([c["p_raw"] for c in fam])):
            c["p_holm"], c["significant"] = float(row["adjusted_p"]), bool(row["reject"])
    fam_stop = [pair_vs(cache, selected[s], f"stop-s{s}") for s in sorted(selected)
                if f"stop-s{s}" in hs and hs[selected[s]]["n"] and hs[f"stop-s{s}"]["n"]]
    archived = json.loads(CAND_SET.read_text())
    arch_rates = [c["pinned_script_successes"] / c["n"] for c in archived["candidates"]]
    sel_rates = [hs[selected[s]]["rate"] for s in sorted(selected) if hs[selected[s]]["n"]]
    stop_rates = [hs[f"stop-s{s}"]["rate"] for s in sorted(selected)
                  if f"stop-s{s}" in hs and hs[f"stop-s{s}"]["n"]]
    per_seed_rows = [{
        "seed": s, "pool_kept": len(cache["search"][str(s)]["pool"]),
        "pool_total": cache["search"][str(s)]["pool_total"], "nfev": cache["search"][str(s)]["nfev"],
        "search_runtime_s": cache["search"][str(s)]["runtime_s"],
        "train_score_stop": scores.get(f"stop-s{s}"), "selected_arm": selected.get(s),
        "train_score_selected": scores.get(selected[s]) if s in selected else None,
        "holdout_stop": hs.get(f"stop-s{s}"),
        "holdout_selected": hs.get(selected[s]) if s in selected else None} for s in args.search_seeds]
    sealed_stat = hs["sealed"]
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], check=False,
                         capture_output=True, text=True).stdout.strip()
    rep = {
        "task": "E11R-01 robust candidate selection (#414)",
        "provenance": {"git_sha": git, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "python": platform.python_version(), "host": platform.node(),
                       "env": {"E11_ENGINE_STAGES": os.environ.get("E11_ENGINE_STAGES"),
                               "E11_FINS_FIXED": os.environ.get("E11_FINS_FIXED"),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
                       "command": "E11_ENGINE_STAGES=1 python " + " ".join(sys.argv),
                       "sealed_records_read": [str(SEALED_CAND.relative_to(REPO)),
                                               str(CAND_SET.relative_to(REPO))],
                       "ascent": "sealed E5-08c Starbase ascent (read, not re-flown)", "fins": FINS},
        "seed_manifest": {"search_seeds": list(args.search_seeds), "train_seeds": TRAIN_SEEDS,
                          "train_per_seed": TRAIN_PER_SEED, "holdout_seeds": HOLDOUT_SEEDS,
                          "n_per_seed": N_PER_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
                          "disjoint_note": "training disjoint from the hold-out and from every "
                                           "prior stream (asserted at import); search seeds are "
                                           "the archived E12-00 candidate-set seeds"},
        "protocol": {"search": "E11-06b DE, no first-feasible stop (runs to --maxiter), every "
                               "feasible design harvested, pool capped first-come at --pool-cap",
                     "ranking": "pinned MC on training seeds only: landings, then mean corridor "
                                "fuel margin, then mean corridor miss, then harvest order",
                     "holdout": "pinned FL-304 on the sealed grid, paired sample-for-sample"},
        "constraint_audit": ca,
        "per_seed": per_seed_rows,
        "scores": {a: scores[a] for a in sorted(scores)},
        "holdout": hs,
        "paired_vs_sealed": fam,
        "paired_selected_vs_stop": fam_stop,
        "sealed_reproduced": {"successes": sealed_stat["successes"], "n": sealed_stat["n"],
                              "matches_record_33_36": (sealed_stat["successes"] == 33
                                                       and sealed_stat["n"] == 36)},
        "verdict": {"dispersion_selected": dispersion(sel_rates),
                    "dispersion_stop": dispersion(stop_rates),
                    "archived_table": {"successes": [c["pinned_script_successes"] for c in
                                                     archived["candidates"]],
                                       "min_rate": min(arch_rates), "max_rate": max(arch_rates)},
                    "pooled_selected": pooled_selected_ci(cache, selected) if selected else None},
        "runtime_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    if args.md:
        write_md(rep, args.md)
    print(json.dumps({k: rep["verdict"][k] for k in ("dispersion_selected", "dispersion_stop",
                                                     "pooled_selected")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
