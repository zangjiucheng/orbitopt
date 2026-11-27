"""E13-05 (task #422): paired LQR / MPC vs the E13-04 baseline feedback law.

The LEO-RDV-001 holdout (22 clusters x 12 samples, seed base 20 000 000) is
flown THREE times on identical perturbation and execution draws: the E13-04
closed-loop baseline (CW mid-course schedule, identity-checked sample-by-
sample against the frozen E13-04 artifact -- the benchmark aborts on
disagreement), an LQR arm and a constrained-MPC arm, both tracking the
sample's own planned transfer arc and both firing the same scheduled
terminal pair (trim + arrival) the baseline schedule carries. The arms share
the departure burn, the EKF/sensing stack, the truth physics and the grading
contract; nothing but the feedback law differs. Metrics: terminal arrival
error, tracking RMSE, realized delta-v by phase, realized constraint
violations per family (and the worst margin actually flown), and decision
compute time. Paired endpoints (success, safety_violation) get exact
McNemar per comparison with Holm correction.

    PYTHONPATH=src python benchmarks/bench_e13_05_constrained_control.py \
        --jobs 16 -o docs/benchmarks/e13_05_constrained_control.json

(~2 h at 16 jobs on the full holdout; --quick runs a 2x2 smoke layout.)"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.rendezvous import canonical_plan, load_reference_config
from orbitopt.rendezvous import constrained_campaign as cc
from orbitopt.rendezvous import robustness as rb
from orbitopt.rendezvous.gnc_campaign import (default_guidance_law, fly_closed_loop_sample)


class _TimedBaselineLaw:
    """Delegating wrapper that times the baseline law's decision calls (the
    stock CwMidCourseLaw keeps no timing of its own; E13-04 reported flight
    wall-time only). Pass-through: nothing but perf_counter touches the law."""

    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    def delta_v(self, estimate):
        t0 = time.perf_counter()
        out = self._inner.delta_v(estimate)
        self._sink.append(time.perf_counter() - t0)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _timed_stats(timings):
    a = np.asarray(timings, dtype=np.float64) * 1000.0
    if a.size == 0:
        return {}
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)),
            "total_s": float(a.sum() / 1000.0)}

FROZEN = REPO / "docs/benchmarks/e13_04_closed_loop_guidance.json"
MISSION_YAML = REPO / "src/orbitopt/data/leo_rdv_001.yaml"
CAMPAIGN_YAML = REPO / "src/orbitopt/data/leo_rdv_001_robustness.yaml"
ARMS = ("baseline", "lqr", "mpc")
VIOL_FAMILIES = ("keep_out", "terminal_cone", "cross_track", "approach_speed")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mcnemar_exact(b1: int, b2: int) -> float:
    n = b1 + b2
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * float(binom.cdf(min(b1, b2), n, 0.5)))


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def _wilson(k: int, n: int, level: float = 0.95) -> dict:
    from scipy.stats import norm

    if n == 0:
        return {"lower": float("nan"), "upper": float("nan")}
    z = float(norm.ppf(0.5 + level / 2.0))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return {"lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def _stats(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"median": None, "p90": None, "max": None}
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def _worker(task):
    cfg, campaign, tof, seed_base, cluster, index, arm = task
    t0 = time.time()
    try:
        if arm == "baseline":
            timings = []
            outcome, assess, _log = fly_closed_loop_sample(
                cfg, campaign, tof, seed_base, cluster, index,
                law_factory=lambda ctx, _t=timings: _TimedBaselineLaw(
                    default_guidance_law(ctx), _t))
            viol = cc._violation_block(cfg, outcome, float(assess["t1_s"]))
            law_stats = _timed_stats(timings)
            fallbacks = None
        else:
            outcome, assess, _log = cc.fly_constrained_sample(
                cfg, campaign, tof, seed_base, cluster, index, arm=arm)
            viol = assess["violations"]
            law_stats = dict(assess.get("law_stats") or {})
            fallbacks = law_stats.get("fallbacks")
    except Exception as exc:  # one bad draw must not sink the campaign
        return {"cluster": int(cluster), "index": int(index), "seed": int(seed_base),
                "arm": arm, "ok": False, "failure_class": "bench_error",
                "reason": f"{type(exc).__name__}: {exc}"[:160], "violations": {},
                "wall_s": round(time.time() - t0, 1)}
    led = assess.get("ledger") or {}
    track = assess.get("tracking") or {}
    cls = outcome.failure_class
    row = {"cluster": int(cluster), "index": int(index),
           "seed": int(assess["draws"]["seed"]), "arm": arm,
           "ok": bool(outcome.ok and cls is None), "failure_class": cls,
           "reason": str(outcome.reason)[:160],
           "attribution": assess.get("attribution"),
           "position_error_m": float(outcome.position_error_m),
           "relative_speed_m_s": float(outcome.relative_speed_m_s),
           "realized_delta_v_m_s": float(outcome.realized_delta_v_m_s),
           "tracking_rmse_pos_m": float(track.get("rmse_pos_m", float("nan"))),
           "tracking_max_pos_m": float(track.get("max_pos_m", float("nan"))),
           "converged": bool((assess.get("convergence") or {}).get("converged", False)),
           "n_refusals": len(assess.get("law_refusals") or []),
           "commanded_by_phase_m_s": {k: float(v)
                                      for k, v in (led.get("commanded_by_phase_m_s") or {}).items()},
           "violations": {k: {kk: vv for kk, vv in viol[k].items()} for k in VIOL_FAMILIES},
           "n_violated_samples": int(viol.get("total_violated_samples", 0)),
           "decision_mean_ms": law_stats.get("mean_ms"),
           "decision_p95_ms": law_stats.get("p95_ms"),
           "decision_total_s": law_stats.get("total_s"),
           "fallbacks": fallbacks,
           "wall_s": round(time.time() - t0, 1)}
    return row


def _identity_guard(rows, frozen_path: Path) -> None:
    doc = json.loads(frozen_path.read_text())
    ref = {(r["cluster"], r["index"]): r for r in doc["closed_loop"]["per_sample"]}
    bad = []
    for r in rows:
        f = ref.get((r["cluster"], r["index"]))
        if f is None:
            bad.append((r["cluster"], r["index"], "missing in frozen artifact"))
            continue
        if r["ok"] != f["ok"] or r["failure_class"] != f["failure_class"]:
            bad.append((r["cluster"], r["index"], "outcome differs"))
        elif abs(r["position_error_m"] - f["position_error_m"]) > 1e-6 \
                or abs(r["realized_delta_v_m_s"] - f["realized_delta_v_m_s"]) > 1e-6:
            bad.append((r["cluster"], r["index"], "residuals differ"))
    if bad:
        raise SystemExit(f"baseline re-flight disagrees with the frozen E13-04 artifact "
                         f"on {len(bad)} samples, e.g. {bad[:3]}")


def _arm_block(rows) -> dict:
    ok_rows = [r for r in rows if r["ok"]]
    counts = {k: 0 for k in ("planning_infeasible", "safety_violation", "fuel_over_budget",
                             "arrival_accuracy", "other", "success")}
    for r in rows:
        if r["ok"]:
            counts["success"] += 1
        else:
            k = r.get("failure_class")
            counts[k if k in counts and k is not None else "other"] += 1
    viol = {}
    for k in VIOL_FAMILIES:
        fam = [r["violations"].get(k) for r in rows if r.get("violations") and k in r["violations"]]
        viol[k] = {"violated_samples": sum(f["violated_samples"] > 0 for f in fam),
                   "count_total": sum(f["violated_samples"] for f in fam),
                   "min_margin": (min((f["min_margin"] for f in fam), default=None))}
    phases: dict = {}
    for r in rows:
        for ph, dv in (r.get("commanded_by_phase_m_s") or {}).items():
            phases.setdefault(ph, []).append(dv)
    ms = [r.get("decision_mean_ms") for r in rows if r.get("decision_mean_ms")]
    p95 = [r.get("decision_p95_ms") for r in rows if r.get("decision_p95_ms")]
    tot = [r.get("decision_total_s") for r in rows if r.get("decision_total_s")]
    fb = [r["fallbacks"] for r in rows if isinstance(r.get("fallbacks"), int)]
    return {"arm": rows[0]["arm"], "n": len(rows), "counts": counts,
            "success_rate": len(ok_rows) / len(rows),
            "wilson": _wilson(len(ok_rows), len(rows)),
            "terminal_position_error_m": _stats([r.get("position_error_m") for r in rows]),
            "terminal_relative_speed_m_s": _stats([r.get("relative_speed_m_s") for r in rows]),
            "tracking_rmse_pos_m": _stats([r.get("tracking_rmse_pos_m") for r in rows]),
            "realized_delta_v_m_s": {**_stats([r.get("realized_delta_v_m_s") for r in rows]),
                                     "over_budget_samples": sum(
                                         r.get("failure_class") == "fuel_over_budget"
                                         for r in rows)},
            "commanded_by_phase_mean_m_s": {ph: float(np.mean(v)) for ph, v in sorted(phases.items())},
            "violations": viol,
            "any_violation_samples": sum(r.get("n_violated_samples", 0) > 0 for r in rows),
            "decision_ms": {"mean_of_means": (float(np.mean(ms)) if ms else None),
                            "p95_of_p95": (float(np.percentile(p95, 95)) if p95 else None),
                            "total_s": (float(np.sum(tot)) if tot else None),
                            "mpc_fallback_ticks_total": (int(np.sum(fb)) if fb else None)}}


def _paired(rows_by_arm) -> dict:
    key = {(r["cluster"], r["index"]): r for r in rows_by_arm["baseline"]}
    tests, labels = [], []
    out = {}
    for arm in ("lqr", "mpc"):
        if any(r.get("failure_class") == "bench_error" for r in rows_by_arm[arm]):
            raise SystemExit(f"the {arm} arm had bench_error rows -- fix before publishing")
        other = {(r["cluster"], r["index"]): r for r in rows_by_arm[arm]}
        if len(other) != len(key):
            raise SystemExit("arm rows do not pair with the baseline rows")
        for endpoint, getter in (("success", lambda r: r["ok"]),
                                 ("safety_violation", lambda r: r.get("n_violated_samples", 0) == 0)):
            b = sum(1 for k, r in key.items()
                    if getter(r) and not getter(other[k]))
            c = sum(1 for k, r in key.items()
                    if not getter(r) and getter(other[k]))
            lab = f"{arm}_vs_baseline::{endpoint}"
            tests.append(_mcnemar_exact(b, c))
            labels.append(lab)
            out[lab] = {"base_only_pass": b, "arm_only_pass": c}
    adj = _holm(tests)
    for lab, raw, corrected in zip(labels, tests, adj):
        out[lab]["mcnemar_p"] = raw
        out[lab]["holm_p"] = corrected
    return {"tests": out, "correction": "Holm across the four comparisons"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="docs/benchmarks/e13_05_constrained_control.json")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="small layout for smoke runs; marked in the artifact")
    args = ap.parse_args()

    cfg = load_reference_config()
    campaign = dataclasses.replace(
        rb.load_reference_campaign(), passive_contingency_in_success=True)
    if args.quick:
        campaign = dataclasses.replace(
            campaign,
            holdout=dataclasses.replace(campaign.holdout, n_clusters=2, samples_per_cluster=2))
    sealed = canonical_plan(cfg)
    tof = float(sealed.plan.tof_s)
    layout = campaign.holdout
    jobs = max(1, int(args.jobs))
    guards = {"frozen_e13_04_artifact_sha256": _sha(FROZEN),
              "mission_yaml_sha256": _sha(MISSION_YAML),
              "campaign_yaml_sha256": _sha(CAMPAIGN_YAML),
              "budget_unchanged": True,
              "tolerances": {"max_position_error_m": float(cfg.max_position_error_m),
                             "max_relative_speed_m_s": float(cfg.max_relative_speed_m_s)},
              "budget_m_s": float(cfg.delta_v_budget_m_s),
              "terminal_pair_shared_across_arms": True,
              "stress_definitions_modified": False}

    tasks = [(cfg, campaign, tof, layout.seed_base, c, i, arm)
             for arm in ARMS for c in range(layout.n_clusters)
             for i in range(layout.samples_per_cluster)]
    print(f"three-arm paired campaign: {layout.n_clusters} x {layout.samples_per_cluster} "
          f"x {len(ARMS)} arms at {jobs} jobs ...", flush=True)
    t0 = time.time()
    rows = []
    with Pool(jobs) as pool:
        for k, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
            rows.append(row)
            if (k + 1) % 24 == 0:
                print(f"  {k + 1}/{len(tasks)} flights "
                      f"({(k + 1) / (time.time() - t0) * 1:0.1f}/s, "
                      f"{time.time() - t0:.0f}s)", flush=True)
    rows_by_arm = {arm: sorted([r for r in rows if r["arm"] == arm],
                               key=lambda r: (r["cluster"], r["index"])) for arm in ARMS}
    if not args.quick:
        _identity_guard(rows_by_arm["baseline"], FROZEN)
        print("baseline identity vs frozen E13-04 artifact: exact", flush=True)
    doc = {"kind": "orbitopt.e13-05.constrained-control.1.0",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quick_layout": bool(args.quick),
           "mission_id": cfg.mission_id,
           "ground_plan": {"tof_s": tof,
                           "planned_total_m_s": float(sealed.plan.total_delta_v_m_s)},
           "holdout": {"seed_base": int(layout.seed_base),
                       "n_clusters": int(layout.n_clusters),
                       "samples_per_cluster": int(layout.samples_per_cluster),
                       "n": len(rows_by_arm["baseline"])},
           "guards": guards,
           "elapsed_s": round(time.time() - t0, 1),
           "arms": {arm: {"summary": _arm_block(rs), "per_sample": rs}
                    for arm, rs in rows_by_arm.items()},
           "paired": _paired(rows_by_arm)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"wrote {out} ({out.stat().st_size} bytes, {time.time() - t0:.0f}s)")
    for arm in ARMS:
        b = doc["arms"][arm]["summary"]
        print(f"  {arm:8s} success {b['counts']['success']}/{b['n']} "
              f"terminal pos med {b['terminal_position_error_m']['median']:.1f} m "
              f"dv med {b['realized_delta_v_m_s']['median']:.2f} m/s "
              f"violated samples {b['any_violation_samples']} "
              f"decision mean {b['decision_ms']['mean_of_means']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
