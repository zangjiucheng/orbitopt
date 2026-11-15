"""Benchmark: E3-02 joint chance-constrained planning vs. FL-305 and R-01.

Paired (identical ascent, seed, dispersions, FL-302 warm start, referee, and
-- per the E3-01 declared actuator set -- the R-05 20 deg/s TVC slew preset
on the landing burn in *every* configuration) Monte Carlo of the three
objective modes, on the textbook RTLS ascent of the recovery E2E tests and
the FL-304/R-05/R-07 evidence runs:

* **fl305_baseline** -- the FL-305 objective (R-01 terms pinned off);
  reproduces the recorded ``fl305_recovery_montecarlo.json`` population;
* **r01_landing_aware** -- the R-01 soft-term objective (the current default
  and the R-07 stack column; the one whose landing scatter R-07 measured at
  a 833 m max radius against the baseline's 338 m);
* **e3_02_joint_chance** -- the joint chance constraints of
  :mod:`orbitopt.flightlab.recovery.chance` (ADR flightlab-0005): fuel
  margin as a quantile on the dispersion loss, divert feasibility derived
  from a 90 %-radius scatter requirement, a joint landing-probability
  bound, aggregated as a worst-violation exact penalty.

Per config the run records success rate (+CI), reason histogram, the
landed-subset scatter (max touchdown radius, error ellipse) and the corridor
handoff percentiles, and -- for the E3-03 paired comparison -- the
per-sample rows (sample index, reason, corridor miss / fuel margin /
predicted landing probability, touchdown offset), which are paired by
construction: the [seed, idx] streams make sample i the same physical
dispersion in every configuration. Real physics, ~20-30 s per sample:

    PYTHONPATH=src python benchmarks/bench_e3_02_joint_recovery.py \
        --n 32 --seed 1 --workers 8 -o docs/benchmarks/e3_02_joint_recovery.json
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
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)

#: FL-305's accepted objective: R-01 reweighted the FL-302 objective, so the
#: baseline pins those terms off (the same pinning as the FL-305/R-07
#: evidence, benchmarks/bench_fl305_recovery_montecarlo.py).
FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}

#: E3-01 consistency: every configuration lands under the declared actuator
#: set (the 20 deg/s TVC slew pin the epic declares -- the R-05 preset value
#: this branch carries, which the E3-01 rate-aware preset also pins), so the
#: paired comparison grades the planner under the same actuator limits.
DECLARED_ACTUATOR = {"landing": {"tvc_rate_limit_deg_s": 20.0}}

#: E3-02 joint chance mode: the DEFAULT JointChanceSpec (defaults documented
#: in orbitopt.flightlab.recovery.chance / ADR flightlab-0005) on top of the
#: pinned-off R-01 soft terms it replaces.
E302_CHANCE = {**FL305_OBJECTIVE, "joint_chance": {}}

CONFIGS = {
    "fl305_baseline": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE, **DECLARED_ACTUATOR},
    "r01_landing_aware": {**MC_PLAN_OPTIONS, **DECLARED_ACTUATOR},
    "e3_02_joint_chance": {**MC_PLAN_OPTIONS, **E302_CHANCE, **DECLARED_ACTUATOR},
}


def _percentiles(values: list[float]) -> dict[str, Any]:
    """min / p50 / p90 / max, sorted exact order (same convention as the R-01
    benchmark; no numpy needed at summary time)."""
    if not values:
        return {"n": 0}
    vs = sorted(values)

    def q(frac: float) -> float:
        i = min(len(vs) - 1, max(0, round(frac * (len(vs) - 1))))
        return float(vs[i])

    return {"n": len(vs), "min": float(vs[0]), "p50": q(0.5), "p90": q(0.9),
            "max": float(vs[-1])}


def _scatter(samples: list[dict]) -> dict[str, Any]:
    """Touchdown radius statistics over the landed samples (the acceptance
    metric: max radius + ellipse must not regress against the baseline)."""
    radii = [math_hyp(s["touchdown_east_m"], s["touchdown_north_m"])
             for s in samples
             if s.get("touchdown_east_m") is not None]
    return _percentiles(radii)


def math_hyp(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5


def _corridor_stats(samples: list[dict]) -> dict[str, Any]:
    miss = [s["corridor_miss_m"] for s in samples if "corridor_miss_m" in s]
    margin = [s["corridor_fuel_margin_kg"] for s in samples
              if "corridor_fuel_margin_kg" in s]
    p_land = [s["landing_probability"] for s in samples
              if s.get("landing_probability") is not None]
    viol = [s["joint_violation"] for s in samples
            if s.get("joint_violation") is not None]
    out = {"divert_m": _percentiles(miss), "fuel_margin_kg": _percentiles(margin)}
    if p_land:
        out["landing_probability"] = _percentiles(p_land)
        out["joint_violation"] = _percentiles(viol)
    return out


def _paired_rows(samples: list[dict]) -> list[dict[str, Any]]:
    """The per-sample rows of the paired comparison: sample idx aligns the
    configurations (same seed -> same [seed, idx] draw stream), so E3-03 can
    run per-sample paired statistics (McNemar on reason classes, paired
    bootstrap on scatter) without re-flying."""
    keep = ("idx", "reason", "corridor_miss_m", "corridor_fuel_margin_kg",
            "landing_probability", "joint_violation", "position_error_m",
            "touchdown_east_m", "touchdown_north_m")
    return [{k: s.get(k) for k in keep} for s in samples]


def main() -> None:
    from test_flightlab_simulate_recovery import (  # noqa: E402
        TARGET,
        VEHICLE,
        _ascent_with_textbook_separation,
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    target = load_target(TARGET)
    ascent = _ascent_with_textbook_separation()
    disp = Dispersions()
    summary: dict = {"kind": "e3-02-joint-chance-vs-fl305-and-r01",
                     "n": args.n, "seed": args.seed,
                     "dispersions": disp.to_dict(),
                     "ascent": "tests/test_flightlab_simulate_recovery.py::"
                               "_ascent_with_textbook_separation",
                     "fairness": "same ascent, seed, dispersions, FL-302 warm "
                                 "start, referee and declared 20 deg/s TVC "
                                 "actuator (E3-01 consistency); the objective "
                                 "mode is the only difference",
                     "baseline_evidence": "docs/benchmarks/"
                                          "fl305_recovery_montecarlo.json, "
                                          "docs/benchmarks/r01_boostback_"
                                          "objective.json, docs/benchmarks/"
                                          "r07_nominal_dispersed.json",
                     "runs": {}}
    for name, opts in CONFIGS.items():
        t0 = time.perf_counter()
        res = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp,
                              seed=args.seed, n=args.n, workers=args.workers,
                              plan_options=opts)
        landed = res.landing_stats
        summary["runs"][name] = {
            "plan_options": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in opts.items()},
            "success_rate": res.success_rate,
            "success_rate_ci": dict(res.statistics.get("success_rate_ci", {})),
            "reason_histogram": dict(sorted(res.reason_histogram.items())),
            "landed_stats": {k: landed[k] for k in sorted(landed)},
            "ellipse": dict(res.ellipse),
            "touchdown_radius_m": _scatter(list(res.samples)),
            "corridor_stats": _corridor_stats(list(res.samples)),
            "paired_rows": _paired_rows(list(res.samples)),
        }
        print(f"{name}: success {res.success_rate:.3f} "
              f"{dict(res.reason_histogram)} ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
