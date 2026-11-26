"""Benchmark: robustness of the LEO-RDV-001 rendezvous under dispersion (E12-07).

One artifact holding everything the acceptance criteria ask for, measured on the
reference mission with the packaged campaign
(``orbitopt/data/leo_rdv_001_robustness.yaml``):

* the hold-out success rate -- arrival inside the acceptance tolerances with no
  safety violation anywhere on the flown trajectory -- with the Wilson interval
  and a whole-cluster bootstrap interval, the sample size derived from the
  declared CI width rather than picked by feel, and the train arm printed beside
  it so nobody can mistake the tuned arm for the measured one;
* the failure taxonomy, total by construction, with the governing reasons;
* the sensitivity reading of each disturbance source under joint dispersion
  (rank correlation of sigma-equivalents against failure, tail split), plus the
  one-source-at-a-time lineup that turns that association into marginal effects;
* the grid-convergence check on the campaign's sample interval;
* the independent spot check of the samples whose verdict a model difference
  could move: independent integration (DOP853 at the production tolerances
  against the production RK4), the independent CW formulation against E12-05's
  declared validity bands, and the TudatPy independent-model re-flight line --
  ``not_checked`` with REFLY_BACKEND_MISSING wherever this environment has no
  backend, recorded rather than passed, exactly as E12-05 does it.

    PYTHONPATH=src python benchmarks/bench_e12_07_rendezvous_robustness.py \
        -o docs/benchmarks/e12_07_rendezvous_robustness.json --jobs 8

``--quick`` runs a small layout for smoke purposes and marks the artifact
``quick_layout: true`` so a smoke artifact can never be read as the headline one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.rendezvous import (
    canonical_plan,
    load_reference_config,
    load_rendezvous_config,
    write_sealed_plan,
)
from orbitopt.rendezvous import robustness as rb
from orbitopt.verify.report import provenance, to_jsonable

ARTIFACT_KIND = "orbitopt.bench.e12-07-rendezvous-robustness"


def _arm_block(result) -> dict:
    return {**result.to_dict(),
            "per_check_min_margin_m": {
                k: min(r.margins[k] for r in result.records if k in r.margins)
                for k in rb.TRAJECTORY_CHECKS
                if any(k in r.margins for r in result.records)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default="docs/benchmarks/e12_07_rendezvous_robustness.json")
    ap.add_argument("--config", default=None, help="mission config (default: LEO-RDV-001)")
    ap.add_argument("--campaign", default=None,
                    help="campaign YAML (default: the packaged leo_rdv_001_robustness.yaml)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-solo", action="store_true",
                    help="skip the one-source-at-a-time lineup (4x the samples)")
    ap.add_argument("--no-grid", action="store_true", help="skip the grid-convergence check")
    ap.add_argument("--quick", action="store_true",
                    help="small layout for smoke runs; marked in the artifact")
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_rendezvous_config(config_path) if config_path else load_reference_config()
    campaign = rb.load_campaign(args.campaign) if args.campaign else rb.load_reference_campaign()
    if args.quick:
        import dataclasses
        campaign = dataclasses.replace(
            campaign,
            train=dataclasses.replace(campaign.train, n_clusters=2, samples_per_cluster=4),
            holdout=dataclasses.replace(campaign.holdout, n_clusters=2, samples_per_cluster=4),
            cluster_bootstrap_trials=400, spot_check_samples=3, grid_convergence_samples=2)

    # E12-08: the campaign flies the canonical sealed plan, and the sealed
    # document is written next to the artifact so a reviewer can re-verify
    # the ground_plan digests the campaign recorded.
    sealed = canonical_plan(cfg)
    write_sealed_plan(sealed, Path(args.out).parent / (Path(args.out).stem + ".sealed.json"))
    result = rb.run_campaign(cfg, campaign, sealed, jobs=max(1, args.jobs),
                             solo=not args.no_solo, grid_check=not args.no_grid,
                             config_path=config_path)

    ho, tr = result.holdout, result.train
    doc = {
        "kind": ARTIFACT_KIND, "schema_version": "1.0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "quick_layout": bool(args.quick),
        "provenance": provenance(),
        "mission_id": cfg.mission_id,
        "headline": {
            "holdout_success_rate": ho.summary["success_rate"],
            "wilson": ho.summary["wilson"],
            "cluster_bootstrap": ho.summary["cluster_bootstrap"],
            "per_cluster_rate": ho.summary["per_cluster_rate"],
            "meets_declared_ci_width": ho.summary["meets_target_ci_width"],
            "failure_classes": ho.failure["classes"],
            "n_samples": ho.n, "n_clusters": ho.n_clusters,
            "train_success_rate": tr.summary["success_rate"],
            "sensitivity_ranking": ho.sensitivity["ranking"],
            "independent_integration": result.cross_check["independent_integration"]["state"],
            "independent_formulation": result.cross_check["independent_formulation"]["state"],
            "independent_model": result.cross_check["independent_model"]["result"],
            "grid_convergence": result.grid_convergence["state"],
        },
        "ground_plan": result.ground_plan,
        "campaign": result.campaign,
        "arms": {k: _arm_block(v) for k, v in result.arms.items()},
        "solo": {k: _arm_block(v) for k, v in result.solo.items()},
        "grid_convergence": result.grid_convergence,
        "cross_check": result.cross_check,
        "notes": list(result.notes),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(doc), indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")

    s = doc["headline"]
    ci = s["wilson"]
    print(f"{cfg.mission_id}: hold-out success {s['holdout_success_rate']:.3f} "
          f"[{ci['lower']:.3f}, {ci['upper']:.3f}] Wilson, n={s['n_samples']} in "
          f"{s['n_clusters']} clusters; train arm {s['train_success_rate']:.3f}")
    boot = s["cluster_bootstrap"]
    if boot:
        print(f"  cluster bootstrap [{boot['lower']:.3f}, {boot['upper']:.3f}] "
              f"({boot['trials']} trials)")
    print("  failures: " + ", ".join(f"{k}={v}" for k, v in s["failure_classes"].items()))
    print(f"  sensitivity ranking (joint): {s['sensitivity_ranking']}")
    print(f"  cross-check: {s['independent_integration']} / {s['independent_formulation']} / "
          f"independent model {s['independent_model']}; grid {s['grid_convergence']}")
    print(f"wrote {out}")
    # The artifact is on disk and the summary is flushed. End here rather than
    # through the interpreter teardown: in this container pygmo's C++ teardown
    # (imported by pykep) aborts glibc at roughly one exit in five even for a
    # script that only runs ``import orbitopt.lambert``, and that would report a
    # finished, complete benchmark as a SIGABRT failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
