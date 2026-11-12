"""Benchmark: FL-304 recovery Monte Carlo before vs. after FL-305.

Flies the same dispersed recovery samples (FL-304 default dispersions, one
seed) of the textbook RTLS ascent used by the recovery E2E tests
(``tests/test_flightlab_simulate_recovery.py``: separation 8 km east of the
pad, 40 km up, 250 m/s, 65 t) twice:

* **before** -- FL-304 as merged: the old warm start (boostback 19.1 s) and
  FL-303's single open-loop G-FOLD plan (``landing.mode = "single"``);
* **after** -- FL-305 defaults: the re-pinned warm start (17.75 s) and the
  closed-loop terminal guidance.

Success rate and reason-code histogram of both land in one JSON (the per-run
``montecarlo.json`` documents are deterministic per seed; the summary is what
ADR flightlab-0002 quotes). Real physics, ~20-30 s per sample per core:

    PYTHONPATH=src python benchmarks/bench_fl305_recovery_montecarlo.py \
        --n 32 --seed 1 --workers 8 -o docs/benchmarks/fl305_recovery_montecarlo.json
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
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)

#: FL-304's merged warm start (before FL-305 re-pinned the boostback duration)
FL304_PLAN_GUESS = (18.4, 19.1, 1.0, 4.36, 0.0, 17456.0, 23.0)

#: R-01 changed the FL-302 objective (landing-aware miss/reserve terms), so
#: this recorded before/after comparison pins the objective the numbers were
#: taken under -- otherwise the "after_fl305" column would silently shift.
FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}

CONFIGS = {
    "before_fl304": {"initial_guess": FL304_PLAN_GUESS,
                     "max_fevals": MC_PLAN_OPTIONS["max_fevals"],
                     "landing": {"mode": "single"}, **FL305_OBJECTIVE},
    "after_fl305": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE},
}


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
    summary: dict = {"kind": "fl305-recovery-montecarlo-before-after",
                     "n": args.n, "seed": args.seed,
                     "dispersions": disp.to_dict(),
                     "ascent": "tests/test_flightlab_simulate_recovery.py::"
                               "_ascent_with_textbook_separation",
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
