"""Benchmark: R-01 landing-aware Boostback objective vs. the FL-305 baseline.

Epic #217 R-01 moved the corridor miss and the corridor propellant inside
the FL-302 objective: the miss is soft-constrained against a landing-divert
budget (DEFAULT_LANDING_DIVERT_BUDGET_M) instead of only the 8 km corridor
radius, and the propellant must cover landing need + reserve + a minimum
margin (DEFAULT_LANDING_RESERVE_MARGIN_KG). This benchmark measures the
delta of that objective change under the *same* physical conditions as the
FL-305 baseline record (``fl305_recovery_montecarlo.json``, after_fl305):
the textbook RTLS ascent fixture of the recovery E2E tests, FL-304 default
dispersions, the FL-305 warm start, closed-loop landing, one seed, n=32:

* **fl305_baseline** -- R-01 terms off (``landing_divert_budget_m=None``,
  ``landing_reserve_margin_kg=0.0``): must reproduce the recorded
  after_fl305 success rate and reason histogram;
* **r01_landing_aware** -- R-01 defaults (1000 m divert budget, 1000 kg
  reserve margin).

Alongside the reason histogram each run reports the handoff the objective
steers at: corridor divert distance and corridor fuel margin (propellant at
the corridor minus landing need minus vehicle reserve) percentiles over all
samples that reached the corridor. ~20-30 s per sample per core, two runs:

    PYTHONPATH=src python benchmarks/bench_r01_boostback_objective.py \
        --n 32 --seed 1 --workers 8 -o docs/benchmarks/r01_boostback_objective.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the E2E ascent fixture builder

from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.flightlab.recovery.boostback import (  # noqa: E402
    DEFAULT_LANDING_DIVERT_BUDGET_M,
    DEFAULT_LANDING_RESERVE_MARGIN_KG,
)
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)

#: the exact FL-305 objective: R-01's terms switched off at the planner API
FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}

CONFIGS = {
    "fl305_baseline": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE},
    "r01_landing_aware": dict(MC_PLAN_OPTIONS),
}


def _percentiles(values: list[float]) -> dict[str, Any]:
    """min / p50 / p90 / max of one corridor metric (sorted exact order
    statistics, no numpy needed at summary time)."""
    if not values:
        return {"n": 0}
    vs = sorted(values)

    def q(frac: float) -> float:
        i = min(len(vs) - 1, max(0, round(frac * (len(vs) - 1))))
        return float(vs[i])

    return {"n": len(vs), "min": float(vs[0]), "p50": q(0.5), "p90": q(0.9),
            "max": float(vs[-1])}


def _corridor_stats(samples: list[dict]) -> dict[str, Any]:
    miss = [s["corridor_miss_m"] for s in samples if "corridor_miss_m" in s]
    margin = [s["corridor_fuel_margin_kg"] for s in samples
              if "corridor_fuel_margin_kg" in s]
    return {"divert_m": _percentiles(miss), "fuel_margin_kg": _percentiles(margin)}


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
    summary: dict = {"kind": "r01-boostback-objective-vs-fl305-baseline",
                     "n": args.n, "seed": args.seed,
                     "dispersions": disp.to_dict(),
                     "r01_defaults": {"landing_divert_budget_m":
                                      DEFAULT_LANDING_DIVERT_BUDGET_M,
                                      "landing_reserve_margin_kg":
                                      DEFAULT_LANDING_RESERVE_MARGIN_KG},
                     "ascent": "tests/test_flightlab_simulate_recovery.py::"
                               "_ascent_with_textbook_separation",
                     "baseline_reference": "docs/benchmarks/"
                                           "fl305_recovery_montecarlo.json",
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
            "reason_histogram": dict(sorted(res.reason_histogram.items())),
            "landed_stats": {k: landed[k] for k in sorted(landed)},
            "corridor_stats": _corridor_stats(list(res.samples)),
        }
        print(f"{name}: success {res.success_rate:.3f} "
              f"{dict(res.reason_histogram)} ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
