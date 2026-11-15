"""Benchmark: R-05 actuator-aware closed-loop guidance vs. the FL-305 baseline.

Runs the *same* recovery Monte Carlo (FL-304 harness, default dispersions,
one seed, one warm start, one referee -- the textbook-RTLS ascent of
``tests/test_flightlab_simulate_recovery.py``) four times:

* **baseline_fl305** -- FL-305's closed-loop defaults: tilt cone 30 deg in
  the loop, TVC direction slew unlimited, throttle range honoured only by
  the propagator's clamp;
* **r05_actuator** -- the same law with one actuator term added
  (``TERMINAL_GUIDANCE_R05``: a 20 deg/s TVC slew limit enforced *inside*
  the guidance, plus the R-05 margin accounting) -- the *naive* clip;
* **e301_lag_naive** -- the declared slew pin plus the declared 0.2 s lag
  flown *naively* (no compensation): what the actuator declaration alone
  costs the FL-305 law;
* **e301_declared** -- the E3-01 declared actuator set
  (``TERMINAL_GUIDANCE_E301``: that chain with the lag compensated at the
  command side so the realised gimbal tracks the law's demand). This is
  what the MC default now flies.

All three configurations carry the R-05 ``ActuatorMargin`` report per landed
sample and aggregated in ``landing_stats.control_margin`` (min/mean/max),
so the comparison covers success, failure mix *and* control margin -- the
FL-305 evidence baseline (docs/benchmarks/fl305_recovery_montecarlo.json)
stays untouched. Fair by construction: same seed, same dispersions, same
FL-302 warm start, same touchdown checks; the guidance differs by exactly
the actuator terms under test. E3-01 (task #258) additionally records every
FAILED sample (its governing reason, its dispersion draw and its touchdown
residuals) in each run's ``failure_records`` -- the acceptance asks that a
``LANDING_LATERAL_SPEED_EXCEEDED`` failure be *located*, not just counted.

Real physics, ~20-30 s per sample per core:

    PYTHONPATH=src python benchmarks/bench_r05_actuator_margins.py \
        --n 32 --seed 1 --workers 8 -o docs/benchmarks/r05_actuator_margins.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the E2E ascent fixture builder

from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.flightlab.recovery import (  # noqa: E402
    DECLARED_LANDING_ACTUATOR,
    TERMINAL_GUIDANCE_E301,
    TERMINAL_GUIDANCE_R05,
)
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)

#: The FL-305-era baseline is pinned to the unlimited-slew closed-loop
#: default EXPLICITLY ("landing": {}): the MC default itself now carries the
#: E3-01 declared actuator set, and this bench must keep measuring the law
#: it was written for, not whatever the default became.
CONFIGS = {
    "baseline_fl305": dict(MC_PLAN_OPTIONS, landing={}),
    "r05_actuator": dict(MC_PLAN_OPTIONS, landing={
        "tvc_rate_limit_deg_s": TERMINAL_GUIDANCE_R05.tvc_rate_limit_deg_s}),
    "e301_lag_naive": dict(MC_PLAN_OPTIONS, landing={
        "tvc_rate_limit_deg_s": TERMINAL_GUIDANCE_E301.tvc_rate_limit_deg_s,
        "tvc_lag_tau_s": TERMINAL_GUIDANCE_E301.tvc_lag_tau_s}),
    "e301_declared": dict(MC_PLAN_OPTIONS, landing=dict(
        DECLARED_LANDING_ACTUATOR)),
}

#: what one failure record carries: the governing reason, the sample's own
#: dispersion draw and its touchdown residuals -- everything needed to find
#: the sample again and to see WHY it failed (E3-01's "locate it" demand).
FAILURE_KEYS = ("idx", "outcome", "reason", "corridor_miss_m",
                "corridor_fuel_margin_kg", "position_error_m",
                "vertical_speed_m_s", "lateral_speed_m_s",
                "touchdown_east_m", "touchdown_north_m", "control_margin")


def _failure_records(res) -> list[dict]:
    """Every failed sample as a plain record, ordered by governing reason
    then index (a histogram counts failures; this localises them)."""
    out = []
    for rec in res.samples:
        if rec.get("outcome") == "landed":
            continue
        out.append({k: rec[k] for k in FAILURE_KEYS if k in rec}
                   | {"dispersion": {k: v for k, v in rec.items()
                                     if k not in FAILURE_KEYS
                                     and not isinstance(v, list)}})
    return sorted(out, key=lambda r: (r.get("reason", ""), r.get("idx", 0)))


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
    summary: dict = {"kind": "r05-actuator-margins-vs-fl305-baseline",
                     "n": args.n, "seed": args.seed,
                     "dispersions": disp.to_dict(),
                     "ascent": "tests/test_flightlab_simulate_recovery.py::"
                               "_ascent_with_textbook_separation",
                     "fairness": "same seed, dispersions, FL-302 warm start "
                                 "and referee; the guidance differs only by "
                                 "the actuator terms under test",
                     "runs": {}}
    for name, opts in CONFIGS.items():
        t0 = time.perf_counter()
        res = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp,
                              seed=args.seed, n=args.n, workers=args.workers,
                              plan_options=opts)
        summary["runs"][name] = {
            "plan_options": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in opts.items()},
            "success_rate": res.success_rate,
            "reason_histogram": dict(sorted(res.reason_histogram.items())),
            "landed_stats": {k: dict(res.landing_stats)[k]
                             for k in sorted(res.landing_stats)},
            "failure_records": _failure_records(res),
        }
        cm = res.landing_stats.get("control_margin", {})
        print(f"{name}: success {res.success_rate:.3f} "
              f"{dict(res.reason_histogram)} "
              f"(margin n={cm.get('n', 0)}, "
              f"slew max {cm.get('slew_deg_s_max', {}).get('max', float('nan')):.1f} "
              f"deg/s) [{time.perf_counter() - t0:.0f} s]", flush=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
