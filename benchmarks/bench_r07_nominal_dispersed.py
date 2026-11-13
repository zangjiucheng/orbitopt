"""Benchmark: R-07 nominal / dispersed recovery trajectory comparison.

Answers the epic's "where and how does it land" question for the recovery
design stack (R-01 landing-aware objective + R-04/R-05/R-06 on top of
FL-305) against the **FL-305 baseline**, on the textbook RTLS ascent used by
the recovery E2E tests and the FL-304/R-05 evidence runs:

* **nominal** -- one unperturbed flight per configuration, written as the
  full verification report (gate + every check + corridor + landing + R-02
  handoff + booster track): ``r07_nominal_recovery.json`` and
  ``r07_nominal_baseline.json``;
* **dispersed** -- the same Monte Carlo (seed, dispersions, warm start,
  referee) flown twice, once per configuration, each sample recording its
  booster track (``capture_tracks``): the per-run envelope figures annotate
  every failure with its reason code and R-06 class, and the comparison
  figure overlays nominal tracks and touchdown ellipses;
* **fairness** -- identical ascent, seed, dispersion setup, FL-302 warm
  start and referee; the guidance/objective differs by exactly what the
  epic changed (the baseline pins the FL-305-era objective terms off,
  matching docs/benchmarks/fl305_recovery_montecarlo.json).

Real physics, ~20-30 s per sample per core:

    PYTHONPATH=src python benchmarks/bench_r07_nominal_dispersed.py \
        --n 32 --seed 1 --workers 8 \
        --out docs/benchmarks/r07_nominal_dispersed.json \
        --figures docs/benchmarks
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
from orbitopt.flightlab.recovery.envelope import (  # noqa: E402
    envelope_dataset, nominal_recovery, nominal_report, plot_comparison,
    plot_envelope, site_from_target, track_rows, write_nominal_report)
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)

#: FL-305's accepted objective: R-01 reweighted the FL-302 objective, so the
#: baseline pins those terms off -- the same pinning the recorded FL-305
#: before/after evidence uses (bench_fl305_recovery_montecarlo.py).
FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}

CONFIGS = {
    "baseline_fl305": {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE},
    "r07_stack": dict(MC_PLAN_OPTIONS),  # current defaults (R-01 on)
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
    ap.add_argument("--capture-rows", type=int, default=240)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="comparison summary JSON")
    ap.add_argument("--figures", type=Path, default=None,
                    help="directory for the envelope/comparison PNGs")
    ap.add_argument("--reports", type=Path, default=None,
                    help="directory for the nominal report JSONs "
                         "(default: the --out directory)")
    args = ap.parse_args()
    reports_dir = args.reports or (args.out.parent if args.out else args.figures)
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(TARGET)
    ascent = _ascent_with_textbook_separation()
    disp = Dispersions()
    site = site_from_target(target)
    fig_dir = args.figures
    summary: dict = {"kind": "r07-nominal-dispersed-vs-fl305-baseline",
                     "n": args.n, "seed": args.seed,
                     "dispersions": disp.to_dict(),
                     "ascent": "tests/test_flightlab_simulate_recovery.py::"
                               "_ascent_with_textbook_separation",
                     "fairness": "same ascent, seed, dispersions, FL-302 warm "
                                 "start and referee; the guidance/objective "
                                 "differs by exactly what epic #217 changed "
                                 "(baseline pins the R-01 objective terms off "
                                 "as the FL-305 evidence did)",
                     "baseline_evidence": "docs/benchmarks/"
                                          "fl305_recovery_montecarlo.json "
                                          "(same seed/n convention)",
                     "runs": {}}
    fig_inputs: list[dict] = []

    for name, opts in CONFIGS.items():
        t0 = time.perf_counter()
        nominal = nominal_recovery(target, VEHICLE, ascent, plan_options=opts)
        report = nominal_report(nominal, target)
        if reports_dir is not None:
            write_nominal_report(report, reports_dir / f"r07_nominal_{name}.json")
        res = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp,
                              seed=args.seed, n=args.n, workers=args.workers,
                              plan_options=opts,
                              capture_tracks=args.capture_rows)
        data = envelope_dataset(res.samples, site,
                                nominal=track_rows(nominal, site))
        if fig_dir is not None:
            plot_envelope(data, fig_dir / f"r07_envelope_{name}.png",
                          title=f"R-07 dispersion envelope — {name} "
                                f"(n={res.n}, seed={res.seed})")
        summary["runs"][name] = {
            "plan_options": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in opts.items()},
            "nominal": {"recovered": report["recovered"],
                        "reason_codes": report["reason_codes"],
                        "landing": report["landing"]},
            "success_rate": res.success_rate,
            "success_rate_ci": res.statistics.get("success_rate_ci", {}),
            "failure_classes": res.statistics.get("failure_classes", {}),
            "reason_histogram": dict(sorted(res.reason_histogram.items())),
            "landed_stats": {k: dict(res.landing_stats)[k]
                             for k in sorted(res.landing_stats)},
            "ellipse": dict(res.ellipse),
            "envelope": {"n_captured": data["n_captured"],
                         "max_touchdown_radius_m":
                             data["ground"]["max_touchdown_radius_m"],
                         "max_range_m": max((t["max_range_m"]
                                             for t in data["trajectories"]),
                                            default=0.0)},
        }
        print(f"{name}: nominal recovered={report['recovered']} "
              f"success {res.success_rate:.3f} "
              f"{dict(res.reason_histogram)} "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)
        fig_inputs.append({"label": name, "dataset": data,
                           "nominal": report["track"],
                           "success_rate": res.success_rate,
                           "ci": res.statistics.get("success_rate_ci")})

    if fig_dir is not None:
        plot_comparison(fig_inputs, fig_dir / "r07_comparison.png")

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
