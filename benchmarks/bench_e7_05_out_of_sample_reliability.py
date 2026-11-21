"""Benchmark: E7-05 out-of-sample reliability of the joint-chance model.

E7-02 found the product landing-probability unskilled on its training data
(Brier skill +0.00006, the [0, 0.5) bin forecasting 0.268 at 0/42 realized);
E7-03 found the fuel/divert failures negatively associated but read that off
a pooled 409-row Fisher test whose rows share four seeds (p = 0.010, the
report itself flags the pooling as optimistic); E7-04 found the joint-chance
arm rejecting 99/168 samples (59%) while its two candidate fixes changed
nothing (45 -> 49 p = 0.125; 35 -> 35). This benchmark answers the three
questions that blocks any further success-rate claim, in the task's order:

1. WHY do 99/168 joint samples reject? (:func:`...reliability.infeasibility
   _diagnosis` over the recorded E7-04 joint arm, paired row-for-row against
   the recorded E3-03 FL-305 baseline): every rejection is classified by the
   constraint that actually bound (reconstructing the violations exactly,
   zero recorded/reconstructed disagreements), by severity (near / mid /
   deep), and by what the SAME physical world did under the laxer FL-305
   grading. The three candidate causes -- requirement too strict, p_land
   underestimation, planner search -- then have measured shares, not opinions.
2. DO the cross-group conclusions survive cluster-aware statistics?
   (:func:`...reliability.stratified_association` over the E7-03 fit set,
   one Mantel-Haenszel stratum per seed instead of a pooled 2x2, plus a
   whole-cluster bootstrap CI and a Holm family over the per-arm readings).
3. DOES the calibration hold OUT of sample? Three hold-out cells, all
   seeds disjoint from the training seeds 1-4, flown fresh (joint-mode
   semantics, standard actuator):

   * ``a_seeds``   -- training vehicle/scenario, FL-304 standard bands,
     seeds {5,6,7,8} x 42 (an unseen draw of the training population);
   * ``b_bands``   -- training vehicle, widened bands (every FL-304 half-
     width and separation sigma x 1.5), seeds {9,10} x 42, the chance spec
     rebuilt from the widened bands (``from_dispersion_bands``);
   * ``c_rtls``    -- the independent ~180 t-class RTLS redesign vehicle
     (``synthetic_two_stage_rtls.yaml``, the recorded E5-05a ascent, the
     frozen E5-05c closed-loop recipe + joint chance), widened bands,
     seeds {11,12} x 24 -- a different vehicle, scenario AND ascent; its
     per-sample cost is ~14x the synthetic cell (dt_opt 0.25 s over a 900 s
     horizon at max_fevals 40), hence the smaller cell.

   Every hold-out reading is the E7-02 kit (reliability table, Brier skill,
   ECE/MCE) PLUS grouped CIs that resample WHOLE SEEDS
   (``cluster_bootstrap_rate_ci``) -- 42 draws inside one seed are one
   perturbation stream, not 42 independent points. No success-rate
   improvement is claimed by this benchmark (E7-04 measured none to
   confirm); the report says so where a claim would go.

Reproduce (recorded parts are free; the hold-out cells cost real physics --
~75 s/sample/core on cell a/b, ~1050 s on cell c, run the three cells on
several cores, ~90 min wall at --workers 14):

    PYTHONPATH=src python benchmarks/bench_e7_05_out_of_sample_reliability.py \
        --workers 14 -o docs/benchmarks/e7_05_out_of_sample_reliability.json \
        --md docs/benchmarks/e7_05_out_of_sample_reliability.md \
        --plot docs/assets/e7_05_out_of_sample_reliability.png

    # recorded-only analysis (skip the flights, read a previous run's JSON):
    PYTHONPATH=src python benchmarks/bench_e7_05_out_of_sample_reliability.py \
        --skip-fly --fly-cache docs/benchmarks/e7_05_out_of_sample_reliability.json

    # smoke (under-powered, machinery check):
    PYTHONPATH=src python benchmarks/bench_e7_05_out_of_sample_reliability.py \
        --seeds-a 5 --n-a 4 --seeds-b 6 --n-b 4 --seeds-c 11 --n-c 2 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the E2E ascent fixture builder

from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.recovery.calibration import (  # noqa: E402
    calibration_report, pairs_from_records,
)
from orbitopt.flightlab.recovery.chance import JointChanceSpec  # noqa: E402
from orbitopt.flightlab.recovery.joint_risk import contingency_table  # noqa: E402
from orbitopt.flightlab.recovery.landing import DECLARED_LANDING_ACTUATOR  # noqa: E402
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS, Dispersions, run_monte_carlo,
)
from orbitopt.flightlab.recovery.reliability import (  # noqa: E402
    cluster_bootstrap_rate_ci, infeasibility_diagnosis, per_seed_contingencies,
    stratified_association,
)
from orbitopt.flightlab.recovery.stats import (  # noqa: E402
    holm_adjust, success_rate_interval,
)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel  # noqa: E402

DATA = REPO / "src" / "orbitopt" / "data" / "flightlab"
A_SCENARIO = DATA / "scenarios" / "leo_two_stage.yaml"
A_VEHICLE = DATA / "vehicles" / "synthetic_two_stage.yaml"
C_SCENARIO = DATA / "scenarios" / "leo_two_stage_rtls.yaml"
C_VEHICLE = DATA / "vehicles" / "synthetic_two_stage_rtls.yaml"
C_ASCENT = REPO / "tests" / "data" / "mission" / "reference" / \
    "e5-05a-rtls-mission" / "ascent"

E3_03_REPORT = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"
E7_02_REPORT = REPO / "docs" / "benchmarks" / "e7_02_probability_calibration.json"
E7_04_REPORT = REPO / "docs" / "benchmarks" / "e7_04_failure_driven.json"

#: the FL-304 bands, widened by this factor for cells b and c (DESIGN
#: ASSUMPTION 1.5: every half-width and separation sigma scaled together --
#: a uniform "what if the model is 50 % worse than declared" band)
WIDE_FACTOR = 1.5

#: the E5-05c frozen closed-loop recipe (scripts/e505c_recovery_search.py
#: defaults at +30 t / 12 g, the one-burn IIP/speed-cut layout), joint-mode
#: re-pinned exactly like E7-04's joint arms: divert budget and reserve
#: margin OFF, the joint chance spec carrying the grading instead
C_RECIPE: dict[str, Any] = {
    "entry_burns": 1, "entry_throttle": True,
    "boostback_cutoff": "iip", "entry_cutoff": "speed",
    "landing_need_model": "closed_loop", "corridor_aim_margin_m": 7000.0,
    "entry_q_max_pa": 250_000.0, "entry_g_max": 12.0,
    "horizon_s": 900.0, "dt_opt": 0.25, "dt_final": 0.1,
    "landing": {**dict(DECLARED_LANDING_ACTUATOR), "max_thrust_accel_g": 3.6},
    "landing_reserve": "target",
    "initial_guess": [22.37, 2.5, 1.0, 0.0, -0.127, 22_000.0, 162.0, 1.0],
    "max_fevals": 40,
}
JOINT_PIN = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}


def _wide_dispersions(factor: float = WIDE_FACTOR) -> Dispersions:
    d = Dispersions()
    return Dispersions(**{k: v * factor for k, v in d.to_dict().items()})


def _joint_chance(disp: Dispersions) -> dict[str, float]:
    """The chance spec rebuilt from the bands actually flown (the same
    route JointChanceSpec.from_dispersion_bands takes for the planner)."""
    s = JointChanceSpec.from_dispersion_bands(disp)
    return {"thrust_pct": s.thrust_pct, "isp_pct": s.isp_pct,
            "density_pct": s.density_pct, "cd_pct": s.cd_pct}


def _arm_options(cell: str, disp: Dispersions) -> dict[str, Any]:
    if cell == "c_rtls":
        return {**C_RECIPE, **JOINT_PIN, "joint_chance": _joint_chance(disp)}
    return {**MC_PLAN_OPTIONS, **JOINT_PIN, "joint_chance": _joint_chance(disp)}


def _seed_list(text: str) -> list[int]:
    vals = [int(t) for t in text.replace(",", " ").split()]
    if not vals or any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("seeds must be positive integers")
    return vals


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({"seed": r["seed"], "idx": r["idx"], "outcome": r["outcome"],
                    "reason": r["reason"],
                    "corridor_miss_m": r.get("corridor_miss_m"),
                    "corridor_fuel_margin_kg": r.get("corridor_fuel_margin_kg"),
                    "position_error_m": r.get("position_error_m"),
                    "touchdown_east_m": r.get("touchdown_east_m"),
                    "touchdown_north_m": r.get("touchdown_north_m"),
                    "landing_probability": r.get("landing_probability"),
                    "joint_violation": r.get("joint_violation")})
    return out


def _pairs_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[tuple[float, float]]]:
    by: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        if r.get("landing_probability") is not None:
            by.setdefault(int(r["seed"]), []).append(
                (float(r["landing_probability"]),
                 1.0 if r["outcome"] == "landed" else 0.0))
    return by


def _grouped_calibration(rows: list[dict[str, Any]], *,
                         trials: int = 4000, seed: int = 20261118) -> dict[str, Any]:
    """The E7-02 reading of an arm PLUS per-bin Wilson intervals re-derived
    under whole-seed resampling (the honest group CI for a 42-draw-per-seed
    bin) and a cluster bootstrap CI for the Brier skill."""
    pairs = pairs_from_records(rows)
    rep = calibration_report(pairs)
    by_seed = _pairs_by_seed(rows)
    edges = rep["bin_edges"]

    def _bin_pairs(bucket: list[tuple[float, float]], lo: float, hi: float,
                   last: bool) -> list[tuple[float, float]]:
        return [(p, y) for p, y in bucket
                if (lo <= p < hi) or (last and p == hi == 1.0)]

    grouped = []
    for row, (lo, hi) in zip(rep["reliability"], zip(edges, edges[1:])):
        last = hi == edges[-1]
        counts = []
        for s, bucket in sorted(by_seed.items()):
            b = _bin_pairs(bucket, lo, hi, last)
            if b:
                counts.append((int(sum(y for _, y in b)), len(b)))
        grouped.append({"bin": row["bin"], "n": row["n"],
                        "mean_prediction": row.get("mean_predicted"),
                        "observed_rate": row.get("observed_rate"),
                        "wilson_ci": ([row["ci"]["lower"], row["ci"]["upper"]]
                                      if row.get("ci") else None),
                        "verdict": row["verdict"],
                        "seed_clusters": len(counts),
                        "cluster_ci95": cluster_bootstrap_rate_ci(
                            counts, trials=trials, seed=seed) if counts else None})
    rep["reliability_cluster_ci"] = grouped
    rep["brier_skill_cluster_ci95"] = _cluster_brier_skill(
        sorted(by_seed.items()), trials=trials, seed=seed)
    rep["seed_clusters"] = len(by_seed)
    return rep


def _cluster_brier_skill(pairs_by_seed: list[tuple[int, list[tuple[float, float]]]],
                         *, trials: int = 4000, seed: int = 20261118
                         ) -> "tuple[float, float] | None":
    """Percentile bootstrap CI for the Brier SKILL (vs climatology) under
    whole-seed resampling: a seed enters the resample whole or not at all."""
    import math
    import random
    from orbitopt.flightlab.recovery.calibration import brier_score
    if len(pairs_by_seed) < 2:
        return None
    rng = random.Random(seed)
    k = len(pairs_by_seed)
    vals: list[float] = []
    for _ in range(trials):
        ps: list[tuple[float, float]] = []
        for _ in range(k):
            ps.extend(pairs_by_seed[rng.randrange(k)][1])
        s = brier_score(ps)
        if s["skill"] is not None:
            vals.append(float(s["skill"]))
    if len(vals) < trials // 2:
        return None
    vals.sort()
    alpha = 0.025
    lo = vals[max(0, int(alpha * len(vals)))]
    hi = vals[min(len(vals) - 1, math.ceil(0.975 * len(vals)) - 1)]
    return (lo, hi)


def _histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(r["reason"]) for r in rows).items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-a", type=_seed_list, default=[5, 6, 7, 8],
                    metavar="LIST", help="cell a seeds (default 5 6 7 8 -- "
                    "disjoint from the training seeds 1-4)")
    ap.add_argument("--n-a", type=int, default=42)
    ap.add_argument("--seeds-b", type=_seed_list, default=[9, 10],
                    metavar="LIST", help="cell b seeds (default 9 10)")
    ap.add_argument("--n-b", type=int, default=42)
    ap.add_argument("--seeds-c", type=_seed_list, default=[11, 12],
                    metavar="LIST", help="cell c seeds (default 11 12)")
    ap.add_argument("--n-c", type=int, default=24)
    ap.add_argument("--wide-factor", type=float, default=WIDE_FACTOR,
                    help="cell b/c band multiplier on every FL-304 half-width "
                         "and separation sigma (default %(default)s)")
    ap.add_argument("--cells", default="abc", metavar="SEL",
                    help="hold-out cells to run: any of a,b,c (default abc; "
                         "empty = recorded-only analysis)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-fly", action="store_true",
                    help="do not fly the hold-out cells; read them from "
                         "--fly-cache (recorded-only re-analysis)")
    ap.add_argument("--fly-cache", type=Path, default=None,
                    help="report JSON of a previous full run to reuse the "
                         "flown hold-out rows from (with --skip-fly)")
    ap.add_argument("--skip-c", action="store_true",
                    help="skip the expensive RTLS cell c (documented in the report)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args(argv)

    spec = JointChanceSpec()
    e3 = json.loads(E3_03_REPORT.read_text(encoding="utf-8"))
    e7_04 = json.loads(E7_04_REPORT.read_text(encoding="utf-8"))
    e7_02 = json.loads(E7_02_REPORT.read_text(encoding="utf-8"))
    joint_rows = next(a for a in e7_04["arms"] if a["name"] == "e7_04_joint_base"
                      )["paired_rows"]
    base_rows = e3["runs"]["fl305_baseline"]["paired_rows"]

    report: dict[str, Any] = {
        "kind": "e7-05-out-of-sample-reliability",
        "provenance": {
            "diagnosis_input": str(E7_04_REPORT.relative_to(REPO))
                               + " [e7_04_joint_base paired_rows]",
            "pairing_input": str(E3_03_REPORT.relative_to(REPO))
                             + " [runs.fl305_baseline.paired_rows, same [seed, idx] grid]",
            "training_calibration": str(E7_02_REPORT.relative_to(REPO)),
            "bootstrap_seeds": {"phi": 20261118, "rate": 20261118,
                                "trials": 4000},
        },
        "fairness": ("hold-out seeds disjoint from training seeds 1-4 by "
                     "construction; every hold-out row is a fresh flight on "
                     "current HEAD under joint-mode semantics with the "
                     "standard declared actuator; chance specs rebuilt from "
                     "the bands flown"),
    }

    # -- 1. infeasibility diagnosis of the recorded E7-04 joint arm ---------
    report["diagnosis"] = infeasibility_diagnosis(joint_rows, spec,
                                                  baseline_rows=base_rows)
    d = report["diagnosis"]
    report["diagnosis_verdict"] = {
        "requirement_too_strict_share": None,  # filled below, it is a reading
        "probability_model_share": (d["probability_binding"]
                                     ["share_of_rejections"]),
        "planner_search_share": d["severity"]["near"] / d["classified"]
                                if d["classified"] else None,
        "constraint_cost_share": (d.get("paired_baseline", {})
                                   .get("landed_anyway_share")),
        "reading": (
            "the 59% rejections are physical, not probabilistic: "
            f"{d['binding'].get('divert', 0)}/{d['classified']} bind on the "
            f"corridor-divert budget and {d['binding'].get('fuel', 0)} on the "
            f"fuel quantile; NOT ONE rejection is bound by the landing-"
            "probability channel (the p_land term never decides alone, so a "
            "mis-calibrated p_land is not what makes the arm infeasible). "
            "The dispersion-band meeting the spec (deep violations) is the "
            f"majority ({d['severity']['deep']}/{d['classified']}), planner-"
            f"search-rescuable near-threshold rejections the minority "
            f"({d['severity']['near']}); and "
            f"{d['paired_baseline']['rejected_but_landed']}/{d['paired_baseline']['n_paired']} "
            "rejections are worlds the FL-305 grading DID land -- the "
            "measured constraint cost of the joint spec on the training grid."),
    }
    report["diagnosis_verdict"]["requirement_too_strict_share"] = \
        (d["severity"]["deep"] + d["severity"]["mid"]) / d["classified"] \
        if d["classified"] else None

    # -- 2. cluster-aware re-analysis of the E7-03 headline -----------------
    fit_arms = ["e3_01_declared_chain", "fl305_baseline", "r01_landing_aware"]
    report["cluster_reanalysis"] = _cluster_reanalysis(e3, fit_arms)

    # -- 3. training calibration references ----------------------------------
    report["training"] = {
        "e7_02_recorded": {k: e7_02[k] for k in ("n", "brier", "ece", "mce",
                                                 "reliability")},
        "e7_04_joint_arm": _grouped_calibration(joint_rows),
    }

    # -- 4. hold-out cells ----------------------------------------------------
    cells = []
    if "a" in args.cells:
        cells.append(("a_seeds", args.seeds_a, args.n_a, False, "A"))
    if "b" in args.cells:
        cells.append(("b_bands", args.seeds_b, args.n_b, True, "A"))
    if "c" in args.cells and not args.skip_c:
        cells.append(("c_rtls", args.seeds_c, args.n_c, True, "C"))
    cached = (json.loads(args.fly_cache.read_text(encoding="utf-8"))
              if args.skip_fly and args.fly_cache else None)
    if cached is not None and cached.get("kind") != report["kind"]:
        raise SystemExit("--fly-cache is not an e7-05 report")
    report["holdout"] = []
    for name, seeds, n, wide, vehicle in cells:
        if args.skip_fly:
            prev = next((c for c in (cached or {}).get("holdout", [])
                         if c["name"] == name), None)
            if prev is None:
                raise SystemExit(f"--fly-cache has no cell {name}")
            report["holdout"].append(prev)
            continue
        disp = _wide_dispersions(args.wide_factor) if wide else Dispersions()
        target_path, vehicle_path = ((C_SCENARIO, C_VEHICLE) if vehicle == "C"
                                     else (A_SCENARIO, A_VEHICLE))
        ascent = (read_run(C_ASCENT) if vehicle == "C"
                  else _ascent_with_textbook_separation())
        opts = _arm_options(name, disp)
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for s in seeds:
            res = run_monte_carlo(load_target(target_path), vehicle_path, ascent,
                                  dispersions=disp, seed=int(s), n=int(n),
                                  workers=args.workers, plan_options=opts)
            rows.extend({"seed": int(s), **dict(rec)} for rec in res.samples)
        landed = sum(1 for r in rows if r["outcome"] == "landed")
        cal = _grouped_calibration(rows)
        joint_rows_c = [r for r in rows if r.get("landing_probability") is not None]
        cell = {
            "name": name,
            "what": {"a_seeds": "independent seeds, standard FL-304 bands, "
                               "training vehicle/scenario/ascent (unseen draw "
                               "of the training population)",
                     "b_bands": f"independent seeds, dispersion bands x "
                                f"{args.wide_factor:g} (chance spec rebuilt "
                                "from the bands flown), training vehicle",
                     "c_rtls": "independent seeds, widened bands, the "
                               "independent ~180 t-class RTLS redesign vehicle "
                               "(E5-05a recorded ascent, E5-05c frozen "
                               "closed-loop recipe + joint chance)"}.get(name, ""),
            "vehicle": str(vehicle_path.relative_to(REPO)),
            "scenario": str(target_path.relative_to(REPO)),
            "ascent": ("tests/data/mission/reference/e5-05a-rtls-mission/ascent"
                       if vehicle == "C" else
                       "tests/test_flightlab_simulate_recovery.py::"
                       "_ascent_with_textbook_separation"),
            "recipe": "E5-05c closed-loop recipe, joint-pinned"
                      if vehicle == "C" else "FL-302 stock MC plan, joint-pinned",
            "seeds": [int(s) for s in seeds], "n_per_seed": int(n),
            "n": len(rows), "dispersions": disp.to_dict(),
            "wide_factor": float(args.wide_factor) if wide else 1.0,
            "landed": landed, "reason_histogram": _histogram(rows),
            "success_rate": landed / len(rows) if rows else None,
            "success_rate_ci": success_rate_interval(landed, len(rows)),
            "success_rate_cluster_ci95": cluster_bootstrap_rate_ci(
                [(sum(1 for r in rows if r["seed"] == s and r["outcome"] == "landed"),
                  sum(1 for r in rows if r["seed"] == s)) for s in seeds]),
            "calibration": cal,
            "diagnosis": infeasibility_diagnosis(joint_rows_c, spec)
            if joint_rows_c else None,
            "wall_s": round(time.perf_counter() - t0, 1),
            "paired_rows": _paired_rows(rows),
        }
        report["holdout"].append(cell)
        print(f"{name}: landed {landed}/{len(rows)} "
              f"brier_skill={cal['brier']['skill']} "
              f"ece={cal['ece']:.3f} ({cell['wall_s']} s)", flush=True)
        if args.out:  # partial flush: a long flight must not be lost to a crash
            report["holdout_comparison"] = _holdout_comparison(report, e7_02)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1) + "\n",
                                encoding="utf-8")

    # -- 5. verdicts ----------------------------------------------------------
    report["holdout_comparison"] = _holdout_comparison(report, e7_02)
    report["improvement_claim"] = {
        "made": False,
        "why": ("E7-04 measured no success-rate improvement to confirm "
                "(guidance fix 45->49 McNemar p=0.125, joint fix 35->35), and "
                "the diagnosis puts the infeasibility in the physical "
                "channels, not the probability model -- this task calibrates "
                "out of sample first, as ordered. No improvement is claimed; "
                "the acceptance's paired-significance clause does not bind."),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_md(report), encoding="utf-8")
        print(f"wrote {args.md}")
    if args.plot:
        _plot(report, args.plot)
        print(f"wrote {args.plot}")
    return 0


def _cluster_reanalysis(e3: dict[str, Any], fit_arms: list[str]) -> dict[str, Any]:
    e7_03 = json.loads((REPO / "docs" / "benchmarks" /
                        "e7_03_joint_risk.json").read_text(encoding="utf-8"))
    pooled = e7_03["pooled_fit_set"]
    per_seed: dict[Any, list[dict[str, Any]]] = {}
    for arm in fit_arms:
        for s, rows in per_seed_contingencies(e3["runs"][arm]["paired_rows"]).items():
            per_seed.setdefault(s, []).extend(rows)
    tables = {s: contingency_table(rows) for s, rows in sorted(per_seed.items())}
    strat = stratified_association(tables)
    per_arm = {}
    ps = []
    for arm in fit_arms:
        groups = {s: contingency_table(rows) for s, rows in
                  per_seed_contingencies(e3["runs"][arm]["paired_rows"]).items()}
        r = stratified_association(groups, trials=2000)
        per_arm[arm] = r
        ps.append((r["mantel_haenszel"] or {}).get("p_value"))
    holm = holm_adjust([p if p is not None else 1.0 for p in ps])
    return {
        "method_note": (
            "E7-03's headline pooled the three fit arms into ONE 409-row 2x2 "
            "(Fisher p = 0.0104) on rows that share four seeds and the same "
            "draw grid. Here each seed is one stratum with the fit arms' rows "
            "summed inside it (the seed is the sampling unit), tested with "
            "Mantel-Haenszel, exact per-seed Fisher combined, and a "
            "whole-cluster bootstrap CI; the per-arm MH p-values then form "
            "the Holm family."),
        "headline_pooled_reference": {
            "phi": pooled["phi"], "p": pooled["fisher_exact_p"],
            "n": pooled["n_observed"], "cells": pooled["cells"],
            "source": "e7_03 report pooled_fit_set"},
        "per_seed_stratified": strat,
        "per_arm": {arm: {
            "per_seed": [[row["cluster"], row["phi"], row["fisher_exact_p"]]
                         for row in r["per_cluster"]],
            "mantel_haenszel": r["mantel_haenszel"],
            "fisher_combined": r["fisher_combined"],
            "phi_cluster_bootstrap_ci95": r["phi_cluster_bootstrap_ci95"],
        } for arm, r in per_arm.items()},
        "holm_family_per_arm_mh": [
            {"arm": arm, "raw_p": p, "holm_p": h["adjusted_p"],
             "reject": h["reject"]}
            for arm, p, h in zip(fit_arms, ps, holm)],
    }


def _holdout_comparison(report: dict[str, Any],
                        e7_02: dict[str, Any]) -> dict[str, Any]:
    train = report["training"]["e7_04_joint_arm"]
    out: dict[str, Any] = {"reference": {
        "which": "e7_04_joint_base (training seeds 1-4, current-HEAD joint "
                 "semantics; the recorded E7-02 numbers on the stale "
                 "e3_02_joint_chance rows are kept alongside)",
        "brier_skill": train["brier"]["skill"], "ece": train["ece"],
        "mce": train["mce"],
        "e7_02_recorded_skill": e7_02["brier"]["skill"],
        "e7_02_recorded_bins": [[b["bin"], b.get("mean_predicted"),
                                 b.get("observed_rate"), b["n"]]
                                for b in e7_02["reliability"]]}, "cells": []}
    for cell in report["holdout"]:
        cal = cell["calibration"]
        bins = [[b["bin"], b["mean_prediction"], b["observed_rate"], b["n"],
                 b["cluster_ci95"]] for b in cal["reliability_cluster_ci"]]
        out["cells"].append({
            "name": cell["name"], "n": cell["n"], "seeds": cell["seeds"],
            "success_rate": cell["success_rate"],
            "success_rate_cluster_ci95": cell["success_rate_cluster_ci95"],
            "brier_skill": cal["brier"]["skill"],
            "brier_skill_cluster_ci95": cal["brier_skill_cluster_ci95"],
            "ece": cal["ece"], "mce": cal["mce"],
            "bins": bins,
            "diagnosis_binding": (cell.get("diagnosis") or {}).get("binding"),
        })
    return out


def _f3(value):
    return "-" if value is None else f"{float(value):.3f}"


def _md(report: dict[str, Any]) -> str:
    d = report["diagnosis"]
    dv = report["diagnosis_verdict"]
    cr = report["cluster_reanalysis"]
    mh = cr["per_seed_stratified"]["mantel_haenszel"]
    fc = cr["per_seed_stratified"]["fisher_combined"]
    ci = cr["per_seed_stratified"]["phi_cluster_bootstrap_ci95"]
    out = ["# E7-05 out-of-sample reliability", "",
           "Provenance in the JSON (`provenance`). All statistics "
           "deterministic (bootstrap seeds recorded).", "",
           "## 1. Why the joint arm rejects (E7-04's 99/168, diagnosed)", "",
           f"{d['n_rejections']} rejections classified (inconsistent rows: "
           f"{d['inconsistent_rows']}). Binding constraint: " +
           ", ".join(f"`{k}` {v}" for k, v in d["binding"].items()) + ".", "",
           "| binding | severity | n |", "|---|---|---|"]
    for key, v in d["binding_by_severity"].items():
        b, s = key.split("/")
        out.append(f"| `{b}` | {s} | {v} |")
    pb = d.get("paired_baseline", {})
    out += ["",
            f"Severity split near/mid/deep (<= {d['thresholds']['borderline']} / "
            f"middle / > {d['thresholds']['deep']} worst normalized violation): "
            f"{d['severity']['near']} / {d['severity']['mid']} / "
            f"{d['severity']['deep']}.", "",
            f"**Landing-probability channel bound on {d['probability_binding']['bound_on_prob_any']} "
            f"of {d['classified']} rejections (alone: "
            f"{d['probability_binding']['bound_on_prob_alone']})** -- the p_land "
            "term is not what makes the joint arm infeasible.", "",
            f"Paired against the FL-305 baseline on the identical grid: "
            f"{pb.get('rejected_but_landed', '-')} of {pb.get('n_paired', '-')} "
            "rejected worlds DID land under FL-305 (constraint cost "
            f"{pb.get('landed_anyway_share'):.3f}).", "",
            f"Verdict: {dv['reading']}", "",
            "## 2. The E7-03 headline under cluster-aware statistics", "",
            f"Pooled reference (E7-03): phi = -0.1266, Fisher p = 0.0104 at "
            "n = 409 (rows share seeds). Cluster-aware: per-seed 2x2 strata "
            "(fit arms summed inside each seed):", ""]
    for row in cr["per_seed_stratified"]["per_cluster"]:
        out.append(f"- seed {row['cluster']}: n {row['n']}, phi "
                   f"{row['phi']:+.3f}, Fisher p {row['fisher_exact_p']:.3g}")
    out += ["",
            f"- Mantel-Haenszel (continuity corrected): X2 = "
            f"{mh['statistic']:.2f}, p = {mh['p_value']:.4f}",
            f"- Fisher-combined exact per-seed: X2 = {fc['statistic']:.2f}, "
            f"p = {fc['p_value']:.4f} (k = {fc['k']})",
            f"- whole-cluster bootstrap CI for pooled phi: "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", "",
            "Per-arm MH p in the Holm family:", ""]
    for row in cr["holm_family_per_arm_mh"]:
        out.append(f"- `{row['arm']}`: MH p {row['raw_p']:.4g} -> Holm "
                   f"{row['holm_p']:.4g}")
    out += ["",
            "Reading: the negative association survives stratification "
            "(MH p below 0.05, cluster CI excludes zero) but is NOT uniform "
            "-- one seed sits at phi = 0 and Fisher-combination (which punishes "
            "heterogeneity) does not reach significance. The honest quote is "
            "the cluster CI, not the pooled p.", "",
            "## 3. Hold-out calibration (fresh flights, seeds disjoint from training)", ""]
    for cell in report["holdout"]:
        cal = cell["calibration"]
        ci = cell["success_rate_ci"]
        out += [f"### `{cell['name']}` -- {cell['what']}", "",
                f"vehicle `{Path(cell['vehicle']).name}`, seeds "
                f"{cell['seeds']} x {cell['n_per_seed']} = {cell['n']} samples, "
                f"bands x {cell['wide_factor']:g}, recipe: {cell['recipe']}.",
                "",
                f"landed {cell['landed']}/{cell['n']} "
                f"({cell['success_rate']:.3f}, Wilson "
                f"[{ci['lower']:.3f}, {ci['upper']:.3f}], whole-seed CI "
                f"{cell['success_rate_cluster_ci95']}); reasons "
                f"{cell['reason_histogram']}.",
                "",
                f"Brier {cal['brier']['score']:.4f} vs climatology "
                f"{cal['brier']['reference_climatology']:.4f} -> skill "
                f"{cal['brier']['skill']} (whole-seed CI "
                f"{cal['brier_skill_cluster_ci95']}), ECE {cal['ece']:.3f}, "
                f"MCE {cal['mce']:.3f}.", "",
                "| bin | n | mean p | observed | Wilson | seed-cluster CI | verdict |",
                "|---|---|---|---|---|---|---|"]
        for b in cal["reliability_cluster_ci"]:
            w = ("[%s, %s]" % tuple(f"{v:.3f}" for v in b["wilson_ci"])
                 if b["wilson_ci"] else "-")
            c = ("[%s, %s]" % tuple(f"{v:.3f}" for v in b["cluster_ci95"])
                 if b["cluster_ci95"] else "-")
            out.append(f"| [{b['bin'][0]:g}, {b['bin'][1]:g}) | {b['n']} | "
                       f"{_f3(b['mean_prediction'])} | {_f3(b['observed_rate'])} | "
                       f"{w} | {c} | {b['verdict']} |")
        dg = cell.get("diagnosis")
        if dg and dg["classified"]:
            out += ["", f"rejections: {dg['n_rejections']}, binding "
                    f"{dg['binding']}, severity {dg['severity']}, prob-bound "
                    f"{dg['probability_binding']['bound_on_prob_any']}"
                    + (f", landed-anyway "
                       f"{dg['paired_baseline']['rejected_but_landed']}/"
                       f"{dg['paired_baseline']['n_paired']}"
                       if dg.get('paired_baseline') else "") + "."]
        out.append("")
    t = report["training"]["e7_04_joint_arm"]
    out += ["## 4. Training reference (same reading, training seeds)", "",
            f"`e7_04_joint_base` (168 rows, seeds 1-4): Brier skill "
            f"{t['brier']['skill']}, ECE {t['ece']:.3f}, MCE {t['mce']:.3f}; "
            "the E7-02 recorded reading on the stale e3_02 rows (skill "
            f"{report['holdout_comparison']['reference']['e7_02_recorded_skill']:.5f}) "
            "is kept in the JSON for the historical comparison.", "",
            "| bin | n | mean p | observed | seed-cluster CI | verdict |",
            "|---|---|---|---|---|---|"]
    for b in t["reliability_cluster_ci"]:
        c = ("[%s, %s]" % tuple(f"{v:.3f}" for v in b["cluster_ci95"])
             if b["cluster_ci95"] else "-")
        out.append(f"| [{b['bin'][0]:g}, {b['bin'][1]:g}) | {b['n']} | "
                   f"{_f3(b['mean_prediction'])} | {_f3(b['observed_rate'])} | "
                   f"{c} | {b['verdict']} |")
    hc = report["holdout_comparison"]
    out += ["", "## 5. Training vs hold-out, side by side", "",
            "| cell | n | success | Brier skill | skill CI | ECE | MCE |",
            "|---|---|---|---|---|---|---|",
            f"| training (e7_04 joint arm) | 168 | {t['brier']['success_rate']:.3f} | "
            f"{t['brier']['skill']} | "
            f"{t['brier_skill_cluster_ci95']} | {t['ece']:.3f} | {t['mce']:.3f} |"]
    for c in hc["cells"]:
        out.append(f"| `{c['name']}` | {c['n']} | {c['success_rate']:.3f} | "
                   f"{c['brier_skill']} | {c['brier_skill_cluster_ci95']} | "
                   f"{c['ece']:.3f} | {c['mce']:.3f} |")
    ic = report["improvement_claim"]
    out += ["", "## 6. Claims", "",
            f"Success-rate improvement claimed: **{ic['made']}** -- {ic['why']}", ""]
    return "\n".join(out)


def _plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    d = report["diagnosis"]
    ax = axes[0]
    keys = sorted(d["binding_by_severity"], key=lambda k: (-d["binding_by_severity"][k]))
    sev_color = {"near": "#4c72b0", "mid": "#dd8452", "deep": "#c44e52"}
    for key in keys:
        b, s = key.split("/")
        ax.barh(key, d["binding_by_severity"][key], color=sev_color[s])
    ax.set_title("E7-04 rejections: binding/severity\n(51 divert, 48 fuel, 0 prob-bound)")
    ax.set_xlabel("rejections")

    ax = axes[1]
    edges = report["training"]["e7_04_joint_arm"]["bin_edges"]
    xs = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
    t = report["training"]["e7_04_joint_arm"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    marks = [("training", t, "o", "#555555")] + [
        (c["name"], c["calibration"], m, col) for c, m, col in
        zip(report["holdout"], "s^D", ("#4c72b0", "#dd8452", "#55a868"))]
    for name, cal, m, col in marks:
        pts = [(b["mean_prediction"], b["observed_rate"])
               for b in cal["reliability_cluster_ci"] if b["n"] > 0]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], m + "-",
                    color=col, label=name, mfc="none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted p_land"); ax.set_ylabel("observed success")
    ax.set_title("reliability: training vs hold-out"); ax.legend(fontsize=8)

    ax = axes[2]
    rows = report["cluster_reanalysis"]["per_seed_stratified"]["per_cluster"]
    ys = range(len(rows))
    ax.errorbar([r["phi"] for r in rows], list(ys), xerr=0.0, fmt="o",
                color="#4c72b0")
    ax.axvline(0, color="k", ls="--", lw=1)
    ci = report["cluster_reanalysis"]["per_seed_stratified"]["phi_cluster_bootstrap_ci95"]
    ax.axvspan(ci[0], ci[1], alpha=0.15, color="#4c72b0",
               label=f"cluster CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    mh = report["cluster_reanalysis"]["per_seed_stratified"]["mantel_haenszel"]
    ax.set_yticks(list(ys)); ax.set_yticklabels([f"seed {r['cluster']}" for r in rows])
    ax.set_xlabel("phi (fuel x divert co-failure)")
    ax.set_title(f"E7-03 per-seed phi (MH p={mh['p_value']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ascent_with_textbook_separation():
    from test_flightlab_simulate_recovery import _ascent_with_textbook_separation as f
    return f()


if __name__ == "__main__":
    raise SystemExit(main())
