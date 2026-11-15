"""Benchmark: E3-03 robust multi-seed Monte Carlo of the recovery arms.

Every recovery Monte Carlo merged so far flew one seed and n=32, and the
recorded success rates could not separate the modes: the 17/32 reading's 95 %
Wilson interval (36.4-69.1 %) swallows the 14/32 reading's (28.2-60.7 %)
almost completely. E3-03 fixes the statistics, not the physics:

* **fixed seed list, planned sample size** -- ``--seeds`` defaults to the
  frozen list ``1,2,3,4`` and the per-seed count is derived from the target
  Wilson interval width (``--ci-width-target``, worst-case planning rate
  0.5), so the pooled interval quoted per arm is as narrow as the benchmark
  says it is (``statistics.required_sample_size``);
* **four arms, one fleet** -- FL-305 (R-01 terms pinned off), the R-01
  landing-aware objective, E3-01's rate-limited declared-actuator landing
  chain on the FL-305 objective, and the E3-02 joint chance-constrained
  planner -- flown over *identical* perturbations (same ascent, same
  dispersions, same ``[seed, idx]`` draw streams, so sample k is the same
  physical world in every arm) and under *identical* declared actuator
  limits (20 deg/s slew in series with the 0.2 s lag in every arm; only the
  E3-01 arm compensates for the lag -- that is the law being graded);
* **paired statistics** -- McNemar's exact test on the paired landed/failed
  verdict for every arm pair (only discordant samples carry evidence), and
  an exact sign test on the touchdown-radius difference over the samples
  both arms landed (landed-conditional scatter, per the E3-02 decision:
  paired flips and landed-conditional scatter, not raw max radius);
* **failure classes and sensitivity per arm** -- the R-06 breakdown and
  joint-dispersion sensitivity recomputed over the *pooled* sample records.

The report keeps every arm's per-sample rows (seed, idx, reason, corridor,
touchdown) so the whole statistics section is recomputable from the JSON.
Real physics, ~10-15 s per sample per core:

    PYTHONPATH=src python benchmarks/bench_e3_03_robust_montecarlo.py \
        --workers 8 -o docs/benchmarks/e3_03_robust_montecarlo.json

    # quick look at the machinery (small fixed n, one seed):
    PYTHONPATH=src python benchmarks/bench_e3_03_robust_montecarlo.py \
        --n-per-seed 4 --seeds 1 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the E2E ascent fixture builder

from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.flightlab.recovery.landing import (  # noqa: E402
    DECLARED_LANDING_ACTUATOR,
)
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)
from orbitopt.flightlab.recovery.stats import (  # noqa: E402
    failure_breakdown,
    mcnemar_test,
    required_sample_size,
    sign_test,
    source_sensitivity,
    success_rate_interval,
)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel  # noqa: E402

#: FL-305's accepted objective: the R-01 soft terms pinned off (the same
#: pinning as every FL-305/R-07/E3-02 evidence run), and the E3-02 joint
#: chance mode on top of that pin (it replaces them and raises otherwise).
FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}
E302_CHANCE = {**FL305_OBJECTIVE, "joint_chance": {}}

#: The declared actuator PHYSICS every arm flies (E3-01's epic pin: the
#: JSBSim spike-stage 20 deg/s slew in series with the 0.2 s first-order
#: lag). This is the fairness pin -- every arm's landing burn faces the same
#: servo, and the only declared-set difference between the arms is the
#: E3-01 arm's command-side lag compensation (the law being graded), carried
#: by DECLARED_LANDING_ACTUATOR.
DECLARED_ACTUATOR = {"tvc_rate_limit_deg_s": 20.0, "tvc_lag_tau_s": 0.2}

CONFIGS: dict[str, dict[str, Any]] = {
    # FL-305 objective under the declared actuator physics (baseline arm)
    "fl305_baseline": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE,
                       "landing": dict(DECLARED_ACTUATOR)},
    # E3-01 alone: same objective, the declared chain with rate-aware divert
    # and command-side lag compensation (TERMINAL_GUIDANCE_E301)
    "e3_01_declared_chain": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE,
                             "landing": dict(DECLARED_LANDING_ACTUATOR)},
    # R-01: the landing-aware soft-term objective (current default knobs)
    "r01_landing_aware": {**MC_PLAN_OPTIONS,
                          "landing": dict(DECLARED_ACTUATOR)},
    # E3-02: joint chance constraints, DEFAULT JointChanceSpec (the E3-02
    # decision: E3-03 must not silently re-tune p_fuel_min=0.9,
    # min_landing_probability=0.8, scatter_budget_m=1000 @ q=0.9)
    "e3_02_joint_chance": {**MC_PLAN_OPTIONS, **E302_CHANCE,
                           "landing": dict(DECLARED_ACTUATOR)},
}

#: The paired-comparison reference every other arm is graded against.
BASELINE = "fl305_baseline"


def _percentiles(values: list[float]) -> dict[str, Any]:
    """min / p50 / p90 / max, exact-order (same convention as the E3-02
    benchmark; no numpy needed at summary time)."""
    if not values:
        return {"n": 0}
    vs = sorted(values)

    def q(frac: float) -> float:
        i = min(len(vs) - 1, max(0, round(frac * (len(vs) - 1))))
        return float(vs[i])

    return {"n": len(vs), "min": float(vs[0]), "p50": q(0.5), "p90": q(0.9),
            "max": float(vs[-1])}


def _radius(row: dict[str, Any]) -> float | None:
    e, n = row.get("touchdown_east_m"), row.get("touchdown_north_m")
    return None if e is None or n is None else (e * e + n * n) ** 0.5


def _landed(row: dict[str, Any]) -> bool:
    return row.get("outcome") == "landed"


def _scatter(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _percentiles([r for r in (_radius(row) for row in rows)
                         if r is not None])


def _corridor_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [r["corridor_miss_m"] for r in rows if "corridor_miss_m" in r]
    margin = [r["corridor_fuel_margin_kg"] for r in rows
              if "corridor_fuel_margin_kg" in r]
    p_land = [r["landing_probability"] for r in rows
              if r.get("landing_probability") is not None]
    viol = [r["joint_violation"] for r in rows
            if r.get("joint_violation") is not None]
    out = {"divert_m": _percentiles(miss), "fuel_margin_kg": _percentiles(margin)}
    if p_land:
        out["landing_probability"] = _percentiles(p_land)
        out["joint_violation"] = _percentiles(viol)
    return out


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The per-sample rows of the paired comparison: (seed, idx) aligns the
    arms (same [seed, idx] stream = the same physical dispersion), so the
    report itself can be re-analysed (McNemar, sign test) without re-flying."""
    keep = ("seed", "idx", "outcome", "reason", "corridor_miss_m",
            "corridor_fuel_margin_kg", "landing_probability",
            "joint_violation", "position_error_m", "touchdown_east_m",
            "touchdown_north_m")
    return [{k: r.get(k) for k in keep} for r in rows]


def _arm_summary(rows: list[dict[str, Any]], model: UncertaintyModel
                 ) -> dict[str, Any]:
    """The pooled statistical reading of one arm: Wilson CI on the pooled
    success rate, the R-06 failure-class distribution and joint-dispersion
    sensitivity over every flown sample, and the landed-conditional
    touchdown scatter."""
    landed = [r for r in rows if _landed(r)]
    return {
        "success_rate": len(landed) / len(rows) if rows else None,
        "success_rate_ci": success_rate_interval(len(landed), len(rows)),
        "failure_classes": failure_breakdown(rows),
        "sensitivity": source_sensitivity(rows, model),
        "touchdown_radius_m": _scatter(landed),
        "corridor_stats": _corridor_stats(rows),
    }


def _paired_analysis(runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Exact paired comparison for every arm pair over the common (seed, idx)
    samples: McNemar on the landed/failed verdict, sign test on the
    touchdown-radius difference over the samples both arms landed."""
    names = sorted(runs)
    index = {name: {(r["seed"], r["idx"]): r for r in runs[name]}
             for name in names}
    out: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(index[a]) & set(index[b]))
            mcn = mcnemar_test([(_landed(index[a][k]), _landed(index[b][k]))
                                for k in common])
            diffs = []
            for k in common:
                ra, rb = _radius(index[a][k]), _radius(index[b][k])
                if _landed(index[a][k]) and _landed(index[b][k]) \
                        and ra is not None and rb is not None:
                    diffs.append(ra - rb)
            sign = sign_test(diffs)
            flip_a_only = [list(k) for k in common
                           if _landed(index[a][k]) and not _landed(index[b][k])]
            flip_b_only = [list(k) for k in common
                           if not _landed(index[a][k]) and _landed(index[b][k])]
            out.append({"a": a, "b": b, "vs_baseline": a == BASELINE or b == BASELINE,
                        "n_pairs": len(common),
                        "mcnemar": mcn,
                        "flips_a_only": sorted(flip_a_only),
                        "flips_b_only": sorted(flip_b_only),
                        "touchdown_radius_sign_test": sign})
    return out


def _seed_list(text: str) -> list[int]:
    seeds = [int(t) for t in text.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError(
            f"seeds must be a unique non-negative list, got {text!r}")
    return seeds


def main() -> None:
    from test_flightlab_simulate_recovery import (  # noqa: E402
        TARGET,
        VEHICLE,
        _ascent_with_textbook_separation,
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=_seed_list, default=[1, 2, 3, 4],
                    metavar="LIST",
                    help="fixed seed list (default 1,2,3,4); sample i of seed "
                         "s draws from the stream [s, i] in every arm")
    ap.add_argument("--n-per-seed", type=int, default=None,
                    help="samples per seed and arm; default: derived from "
                         "--ci-width-target (evenly over the seed list)")
    ap.add_argument("--ci-width-target", type=float, default=0.15,
                    help="target 95%% Wilson interval width at the worst-case "
                         "planning rate 0.5 (default 0.15)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    target = load_target(TARGET)
    ascent = _ascent_with_textbook_separation()
    disp = Dispersions()
    model = UncertaintyModel.from_dispersions(disp)

    required = required_sample_size(args.ci_width_target)
    if args.n_per_seed is None:
        n_per_seed = -(-required // len(args.seeds))  # ceil, so pooled >= required
    else:
        n_per_seed = args.n_per_seed
    n_total = n_per_seed * len(args.seeds)
    sizing = {"method": "Wilson interval width at planning rate 0.5 "
                        "(orbitopt.flightlab.recovery.stats."
                        "required_sample_size)",
              "target_ci_width": float(args.ci_width_target),
              "planning_rate": 0.5, "required_n_pooled": required,
              "seeds": list(args.seeds), "n_per_seed": int(n_per_seed),
              "n_pooled": int(n_total),
              "planned_adequate": n_total >= required}

    summary: dict = {"kind": "e3-03-robust-recovery-montecarlo",
                    "seeds": list(args.seeds), "n_per_seed": int(n_per_seed),
                    "n_pooled": int(n_total),
                    "sizing": sizing,
                    "dispersions": disp.to_dict(),
                    "ascent": "tests/test_flightlab_simulate_recovery.py::"
                              "_ascent_with_textbook_separation",
                    "fairness": "identical ascent, dispersions and [seed, idx]"
                                " draw streams across arms (sample k is the "
                                "same physical world in every arm); identical "
                                "FL-302 warm start, referee and declared "
                                "actuator physics (20 deg/s slew + 0.2 s lag) "
                                "in every arm; the objective mode and (only "
                                "for e3_01_declared_chain) the command-side "
                                "lag compensation are the only differences",
                    "baseline_evidence": "docs/benchmarks/"
                                         "fl305_recovery_montecarlo.json, "
                                         "docs/benchmarks/r01_boostback_"
                                         "objective.json, docs/benchmarks/"
                                         "e3_02_joint_recovery.json, "
                                         "docs/benchmarks/"
                                         "r07_nominal_dispersed.json",
                    "runs": {}, "paired": []}

    per_arm_rows: dict[str, list[dict[str, Any]]] = {}
    for name, opts in CONFIGS.items():
        per_seed: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for seed in args.seeds:
            res = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp,
                                  seed=seed, n=n_per_seed,
                                  workers=args.workers, plan_options=opts)
            for rec in res.samples:
                rows.append({"seed": int(seed), **dict(rec)})
            per_seed[str(seed)] = {
                "n": int(res.n),
                "success_rate": res.success_rate,
                "success_rate_ci": dict(res.statistics["success_rate_ci"]),
                "reason_histogram": dict(sorted(res.reason_histogram.items())),
            }
        landed = sum(1 for r in rows if _landed(r))
        hist: dict[str, int] = {}
        for r in rows:
            hist[r["reason"]] = hist.get(r["reason"], 0) + 1
        per_arm_rows[name] = rows
        summary["runs"][name] = {
            "plan_options": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in opts.items()},
            "pooled": _arm_summary(rows, model),
            "per_seed": per_seed,
            "paired_rows": _paired_rows(rows),
        }
        print(f"{name}: success {landed / n_total:.3f} ({landed}/{n_total}) "
              f"{dict(sorted(hist.items()))} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)

    summary["paired"] = _paired_analysis(per_arm_rows)
    for cmp_ in summary["paired"]:
        mc = cmp_["mcnemar"]
        print(f"paired {cmp_['a']} vs {cmp_['b']}: "
              f"{mc['successes_a_only']}/{mc['successes_b_only']} discordant, "
              f"exact p={mc['exact']['p_value']:.4g}; "
              f"scatter sign p={cmp_['touchdown_radius_sign_test']['p_value']:.4g}",
              flush=True)

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
