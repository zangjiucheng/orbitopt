"""Benchmark: independent robustness re-flight of LEO-RDV-001 (E12-10).

E12-07 characterized the rendezvous under dispersion -- but every number there
is orbitopt grading itself, and its own report marks the TudatPy line
``NOT_CHECKED``. This artifact closes that gap on the samples the verdicts rest
on, under the E12-10 grading criteria:

* the campaign is re-run with the **passive contingency checks inside the success
  definition** (``passive_contingency_in_success``), so the hold-out success rate
  and the failure taxonomy are reported under the criteria that actually decide
  qualification -- beside the E12-07 baseline (14/264) they replace;
* a deterministic, declared **selection** of samples -- every success, the
  passive-contingency-driven failures, one boundary sample per failure class,
  one planning-infeasible row -- is re-flown in **TudatPy**, each sample with its
  OWN perturbation (its executed burns, its J2, its drag factor and atmosphere),
  and verdicted by the mission's own contract applied to the independently flown
  trajectory;
* every re-flight is compared against the main model per sample: verdict
  agreement, and the residual on each acceptance quantity against the declared
  cross-model bands.

Until this re-flight is *checked* here, the campaign's conclusion is a
**Robustness Characterization** -- not a statement that the design is robust or safe. No
backend, no fake pass: the re-flight line is recorded ``not_checked`` with
``REFLY_BACKEND_MISSING``, exactly as E12-05 and E12-07 do it.

    PYTHONPATH=src ORBITOPT_TUDAT_PYTHON=/path/to/tudatpy/python \
        python benchmarks/bench_e12_10_independent_reflight.py \
        -o docs/benchmarks/e12_10_independent_reflight.json --jobs 8

``--quick`` runs a small layout for smoke purposes and marks the artifact
``quick_layout: true`` so a smoke artifact can never be read as the headline one.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from orbitopt.verify.rendezvous_refly import verify_sample_reflight
from orbitopt.verify.report import provenance, to_jsonable

ARTIFACT_KIND = "orbitopt.bench.e12-10-independent-reflight"
BASELINE_ARTIFACT = REPO / "docs" / "benchmarks" / "e12_07_rendezvous_robustness.json"


def _passive_view(records) -> dict:
    """How much of the hold-out failure taxonomy the passive-contingency checks
    carry under the new criteria: failures whose failed-check list contains a
    ``passive_*`` entry (a sample can fail both ways; it is counted here as
    passive-involved and remains a safety violation)."""
    involved = [r for r in records if r.failure_class == "safety_violation"
                and any(str(c).startswith(rb.PASSIVE_CHECK_PREFIX) for c in r.failed_checks)]
    only = [r for r in involved
            if all(str(c).startswith(rb.PASSIVE_CHECK_PREFIX) for c in r.failed_checks)]
    by_kind: dict[str, int] = {}
    for r in involved:
        for c in r.failed_checks:
            if str(c).startswith(rb.PASSIVE_CHECK_PREFIX):
                by_kind[str(c)] = by_kind.get(str(c), 0) + 1
    return {"n_safety_violations": sum(1 for r in records
                                       if r.failure_class == "safety_violation"),
            "n_passive_involved": len(involved), "n_passive_only": len(only),
            "by_passive_check": dict(sorted(by_kind.items()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default="docs/benchmarks/e12_10_independent_reflight.json")
    ap.add_argument("--config", default=None, help="mission config (default: LEO-RDV-001)")
    ap.add_argument("--campaign", default=None,
                    help="campaign YAML (default: the packaged leo_rdv_001_robustness.yaml)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="small layout for smoke runs; marked in the artifact")
    ap.add_argument("--no-reflight", action="store_true",
                    help="re-grade the campaign only, skip the TudatPy re-flight "
                         "(records the independent line not_checked)")
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_rendezvous_config(config_path) if config_path else load_reference_config()
    campaign = rb.load_campaign(args.campaign) if args.campaign else rb.load_reference_campaign()
    campaign = dataclasses.replace(campaign, passive_contingency_in_success=True)
    if args.quick:
        campaign = dataclasses.replace(
            campaign,
            train=dataclasses.replace(campaign.train, n_clusters=2, samples_per_cluster=4),
            holdout=dataclasses.replace(campaign.holdout, n_clusters=2, samples_per_cluster=4),
            cluster_bootstrap_trials=400, spot_check_samples=2, grid_convergence_samples=2)

    sealed = canonical_plan(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_sealed_plan(sealed, out.parent / (out.stem + ".sealed.json"))
    result = rb.run_campaign(cfg, campaign, sealed, jobs=max(1, args.jobs), solo=False,
                             grid_check=False, config_path=config_path)
    ho = result.holdout

    spec = rb.build_reflight_spec(cfg, campaign, ho.records, sealed.plan)
    # The campaign record itself does not carry the passive-release margins, so
    # re-fly the selected samples (<= 32 truth flights) to put the main model's
    # own passive margins next to the independent ones in the same table.
    by_key = {(int(r.cluster), int(r.index)): r for r in ho.records}
    for fl in spec["flights"]:
        rec = by_key[(int(fl["cluster"]), int(fl["index"]))]
        outcome = rb.fly_truth_sample(cfg, campaign, float(sealed.plan.tof_s), rec.draws)
        fl["main_passive"] = [p.to_dict() for p in outcome.passive]
    spec["spec_digest"] = rb.reflight_spec_digest(spec)
    run = ({"status": "not_checked", "backend": None, "evidence": None, "comparison": None,
            "reason": "--no-reflight: the re-flight was skipped by request"}
           if args.no_reflight else
           verify_sample_reflight(cfg, spec, config_path=config_path,
                                  out_dir=out.parent / (out.stem + "-crossenv")))

    comparison = run["comparison"]
    covered = (run["status"] == "checked" and comparison is not None
               and comparison["n_band_violations"] == 0 and comparison["n_verdict_flips"] == 0)
    baseline = {}
    if BASELINE_ARTIFACT.is_file():
        b = json.loads(BASELINE_ARTIFACT.read_text(encoding="utf-8"))["headline"]
        baseline = {"artifact": str(BASELINE_ARTIFACT.relative_to(REPO)),
                    "n_samples": b["n_samples"], "success_rate": b["holdout_success_rate"],
                    "wilson": b["wilson"], "cluster_bootstrap": b["cluster_bootstrap"],
                    "failure_classes": b["failure_classes"],
                    "grading": "arrival + safety checks, passive contingencies NOT in success"}

    measured: dict[str, dict] = {}
    if comparison is not None and comparison["samples"]:
        samples = comparison["samples"]
        for name in samples[0]["residuals"]:
            rs = [smp["residuals"][name] for smp in samples]
            measured[name] = {"max_abs_difference": max(abs(r["difference"]) for r in rs),
                              "tolerance": rs[0]["tolerance"],
                              "n_within": sum(bool(r["within_band"]) for r in rs)}
        ps = [p for smp in samples for p in smp["passive_comparison"]]
        if ps:
            measured["passive_keep_out_margin_m"] = {
                "max_abs_difference": max(abs(p["difference_m"]) for p in ps),
                "tolerance": ps[0]["tolerance_m"],
                "n_within": sum(bool(p["within_band"]) for p in ps), "n_releases": len(ps)}

    n_success = sum(1 for r in ho.records if r.success)
    n_success_reflown = spec["selection_rule"]["counts_by_reason"].get("success", 0)
    covered = covered and n_success_reflown == n_success
    sr = ho.summary["cluster_bootstrap"] or ho.summary["wilson"]
    label = "Independently Verified Robustness Characterization" if covered else "Robustness Characterization"

    doc = {
        "kind": ARTIFACT_KIND, "schema_version": "1.0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "quick_layout": bool(args.quick),
        "provenance": provenance(),
        "mission_id": cfg.mission_id,
        "classification": {
            "label": label,
            "independent_reflight": run["status"],
            "coverage": ("covered" if covered else
                         "divergences_found" if comparison and (comparison["n_verdict_flips"]
                                                                or comparison["n_band_violations"])
                         else "not_covered"),
            "wording_rule": ("this campaign may be reported as Robustness Characterization; "
                             "the wording Independently Verified Robustness Characterization is "
                             "only earned once the independent per-sample re-flight is CHECKED, "
                             "covers the samples the verdicts rest on, and every sample agrees "
                             "with the main model within the declared bands (task #412 "
                             "acceptance). It says the EVIDENCE is independent, not that the "
                             "design is robust or safe: see `outcome` for the success rate"),
            "outcome": (f"the evidence is independent; the design it characterizes "
                        f"succeeds on {n_success}/{ho.n} hold-out samples "
                        f"({ho.summary['success_rate']:.1%}, cluster CI "
                        f"{sr['lower']:.1%}-{sr['upper']:.1%}) under these criteria. No "
                        "success-rate requirement is declared, so this records the rate; it "
                        "does NOT show the design robust to the declared dispersions"),
        },
        "grading": {"passive_contingency_in_success": True,
                    "definition": ("success = arrival inside tolerances, no safety violation on "
                                   "the flown trajectory, AND every passive contingency release "
                                   "(no-burn at t0, gate abort at t1, hold-point coast) stays "
                                   "outside the keep-out corridor for its whole duration")},
        "baseline_e12_07": baseline,
        "regraded": {
            "n_samples": ho.n, "n_clusters": ho.n_clusters,
            "success_rate": ho.summary["success_rate"], "wilson": ho.summary["wilson"],
            "cluster_bootstrap": ho.summary["cluster_bootstrap"],
            "per_cluster_rate": ho.summary["per_cluster_rate"],
            "meets_declared_ci_width": ho.summary["meets_target_ci_width"],
            "failure_classes": ho.failure["classes"],
            "passive_view": _passive_view(ho.records)},
        "selection": {**spec["selection_rule"], "spec_digest": spec["spec_digest"],
                      "n_holdout_successes": n_success,
                      "all_successes_reflown": n_success_reflown == n_success},
        "independent": {
            "status": run["status"], "backend": run["backend"], "reason": run.get("reason"),
            "spec_digest": spec["spec_digest"], "plan_digest": spec["plan_digest"],
            "evidence_kind": (run["evidence"] or {}).get("kind"),
            "tolerances": (run["evidence"] or {}).get("tolerances"),
            "measured_residuals": measured,
            "n_reflied": (comparison or {}).get("n_reflied", 0),
            "n_verdict_within_bands": (comparison or {}).get("n_verdict_within_bands", 0),
            "n_verdict_flips": (comparison or {}).get("n_verdict_flips", 0),
            "verdict_flips": (comparison or {}).get("verdict_flips", []),
            "band_violations": (comparison or {}).get("band_violations", []),
        },
        "per_sample": (comparison or {}).get("samples", []),
        "skipped_rows": [{k: v for k, v in row.items() if k != "main_margins"}
                         for row in spec["skipped"]],
        "ground_plan": result.ground_plan,
        "campaign": result.campaign,
        "notes": list(result.notes),
    }

    out.write_text(json.dumps(to_jsonable(doc), indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")

    rg = doc["regraded"]
    ci = rg["wilson"]
    print(f"{cfg.mission_id}: re-graded hold-out success {rg['success_rate']:.3f} "
          f"[{ci['lower']:.3f}, {ci['upper']:.3f}] Wilson, n={rg['n_samples']} in "
          f"{rg['n_clusters']} clusters")
    boot = rg["cluster_bootstrap"]
    if boot:
        print(f"  cluster bootstrap [{boot['lower']:.3f}, {boot['upper']:.3f}] "
              f"({boot['trials']} trials); baseline E12-07 "
              f"{baseline.get('success_rate', 'n/a')!r} on {baseline.get('n_samples', '?')} "
              f"samples")
    print("  failures: " + ", ".join(f"{k}={v}" for k, v in rg["failure_classes"].items())
          + f"; passive-involved {rg['passive_view']['n_passive_involved']} "
          f"(only {rg['passive_view']['n_passive_only']})")
    sel = doc["selection"]
    print(f"  selection: {sel['counts_by_reason']} "
          f"(flyable={sel['n_flyable']}, skipped={sel['n_skipped']})")
    ind = doc["independent"]
    print(f"  independent re-flight: {ind['status']} (backend={ind['backend']}) "
          + (f"-- {ind['reason']}" if ind["reason"] else
             f"-- {ind['n_reflied']} samples, {ind['n_verdict_flips']} verdict flips, "
             f"{len(ind['band_violations'])} band violations"))
    print(f"  classification: {doc['classification']['label']} "
          f"(coverage {doc['classification']['coverage']})")
    print(f"wrote {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    # see bench_e12_07: pykep/pygmo's C++ teardown can abort glibc at exit
    os._exit(0 if run["status"] == "checked" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
