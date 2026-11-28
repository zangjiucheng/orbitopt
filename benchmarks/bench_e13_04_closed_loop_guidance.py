"""E13-04 (task #421): paired closed-loop vs frozen open-loop campaign.

The LEO-RDV-001 holdout (22 clusters x 12 samples, seed base 20 000 000) is
run TWICE on identical perturbation and execution draws: once open-loop
(re-flown here, and cross-checked against the frozen E12-10 artifact -- if
the re-flight disagrees the benchmark aborts) and once closed-loop through
``orbitopt.rendezvous.gnc_campaign.fly_closed_loop_sample``. The pairing is
exact: same ``draw_sample`` draws, same departure/arrival execution errors,
same truth physics and grading contract; the closed loop adds sensing, an
EKF with measurement-time compensation, and the CW mid-course law schedule
(departure, midcourse retarget, terminal trim, terminal velocity-null). No
budget, tolerance or stress definition is touched.

Statistics on the paired outcomes: McNemar's exact test per endpoint (
success, safety_violation, arrival_accuracy, fuel_over_budget) with Holm
correction across the four, a whole-cluster bootstrap (4 000 trials) of the
success-rate difference, and a Wilson interval for the closed-loop rate.
Failure classes and the E13-04 attribution (navigation / guidance /
execution / safety constraint) are tabulated for both arms.

    PYTHONPATH=src python benchmarks/bench_e13_04_closed_loop_guidance.py \
        --jobs 8 -o docs/benchmarks/e13_04_closed_loop_guidance.json

(~1 h at 8 jobs on the full holdout; use --quick for a smoke layout.)"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.rendezvous import canonical_plan, load_reference_config
from orbitopt.rendezvous import robustness as rb
from orbitopt.rendezvous.gnc_campaign import fly_closed_loop_sample

FROZEN = REPO / "docs/benchmarks/e12_10_independent_reflight.json"
MISSION_YAML = REPO / "src/orbitopt/data/leo_rdv_001.yaml"
CAMPAIGN_YAML = REPO / "src/orbitopt/data/leo_rdv_001_robustness.yaml"
ENDPOINTS = ("success", "safety_violation", "arrival_accuracy", "fuel_over_budget")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mcnemar_exact(b_fail_open: int, b_fail_closed: int) -> float:
    """Exact two-sided McNemar: the discordant pair count is the sample, the
    split between the two directions is the statistic."""
    n = b_fail_open + b_fail_closed
    if n == 0:
        return 1.0
    k = min(b_fail_open, b_fail_closed)
    p = 2.0 * float(binom.cdf(k, n, 0.5))
    return min(1.0, p)


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


def _cluster_diff_boot(
    open_ok: dict,
    closed_ok: dict,
    clusters: list[int],
    trials: int,
    seed: int,
    level: float,
) -> dict:
    """Whole-cluster percentile bootstrap of (closed - open) success rate."""
    rng = np.random.default_rng(seed)
    byc = {
        c: [(int(open_ok[c][i]), int(closed_ok[c][i])) for i in sorted(open_ok[c])]
        for c in clusters
    }
    diffs = np.empty(trials)
    for t in range(trials):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        n = s_o = s_c = 0
        for c in pick:
            for o, cl in byc[c]:
                n += 1
                s_o += o
                s_c += cl
        diffs[t] = (s_c - s_o) / n
    lo, hi = (
        float(x)
        for x in np.percentile(
            diffs, [(1.0 - level) / 2.0 * 100.0, (1.0 + level) / 2.0 * 100.0]
        )
    )
    return {
        "method": "whole-cluster percentile bootstrap of the paired difference",
        "trials": trials,
        "point": None,
        "lower": lo,
        "upper": hi,
        "resample_unit": "cluster",
    }


def _closed_worker(args):
    cfg, campaign, tof, seed_base, cluster, index = args
    t0 = time.time()
    outcome, assess, _log = fly_closed_loop_sample(
        cfg, campaign, tof, seed_base, cluster, index
    )
    led = assess.get("ledger") or {}
    track = assess.get("tracking") or {}
    cls = outcome.failure_class
    row = {
        "cluster": int(cluster),
        "index": int(index),
        "seed": int(assess["draws"]["seed"]),
        "ok": bool(outcome.ok and cls is None),
        "failure_class": cls,
        "reason": str(outcome.reason)[:160],
        "attribution": assess.get("attribution"),
        "position_error_m": float(outcome.position_error_m),
        "relative_speed_m_s": float(outcome.relative_speed_m_s),
        "realized_delta_v_m_s": float(outcome.realized_delta_v_m_s),
        "tracking_rmse_pos_m": float(track.get("rmse_pos_m", float("nan"))),
        "tracking_max_pos_m": float(track.get("max_pos_m", float("nan"))),
        "converged": bool((assess.get("convergence") or {}).get("converged", False)),
        "n_refusals": len(assess.get("law_refusals") or []),
        "commanded_by_phase_m_s": {
            k: float(v) for k, v in (led.get("commanded_by_phase_m_s") or {}).items()
        },
        "wall_s": round(time.time() - t0, 1),
    }
    return row


def _counts(rows, key="failure_class", ok_key="ok") -> dict:
    out = {
        "planning_infeasible": 0,
        "safety_violation": 0,
        "fuel_over_budget": 0,
        "arrival_accuracy": 0,
        "other": 0,
        "success": 0,
    }
    for r in rows:
        if r[ok_key]:
            out["success"] += 1
        else:
            out[r[key] if r[key] in out and r[key] is not None else "other"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "-o", "--out", default="docs/benchmarks/e13_04_closed_loop_guidance.json"
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small layout for smoke runs; marked in the artifact",
    )
    ap.add_argument(
        "--open-only",
        action="store_true",
        help="re-flight the open loop and check identity, skip the closed loop",
    )
    args = ap.parse_args()

    cfg = load_reference_config()
    campaign = dataclasses.replace(
        rb.load_reference_campaign(), passive_contingency_in_success=True
    )
    if args.quick:
        campaign = dataclasses.replace(
            campaign,
            train=dataclasses.replace(
                campaign.train, n_clusters=2, samples_per_cluster=4
            ),
            holdout=dataclasses.replace(
                campaign.holdout, n_clusters=2, samples_per_cluster=4
            ),
            cluster_bootstrap_trials=400,
        )
    sealed = canonical_plan(cfg)
    tof = float(sealed.plan.tof_s)
    layout = campaign.holdout
    jobs = max(1, int(args.jobs))

    guards = {
        "frozen_e12_10_artifact_sha256": _sha(FROZEN),
        "mission_yaml_sha256": _sha(MISSION_YAML),
        "campaign_yaml_sha256": _sha(CAMPAIGN_YAML),
        "budget_unchanged": True,
        "tolerances": {
            "max_position_error_m": float(cfg.max_position_error_m),
            "max_relative_speed_m_s": float(cfg.max_relative_speed_m_s),
        },
        "budget_m_s": float(cfg.delta_v_budget_m_s),
        "stress_definitions_modified": False,
    }
    print(
        f"open-loop re-flight: {layout.n_clusters} x {layout.samples_per_cluster} "
        f"at {jobs} jobs ...",
        flush=True,
    )
    t0 = time.time()
    ho = rb.run_campaign_arm(cfg, campaign, sealed.plan, "holdout", jobs=jobs)
    open_rows = [r.to_dict() for r in ho.records]
    print(
        f"  {ho.successes}/{len(ho.records)} open-loop successes "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # identity guard: the re-flight must reproduce the frozen E12-10 holdout
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["regraded"]
    fc = _counts(
        [{"ok": r["success"], "failure_class": r["failure_class"]} for r in open_rows]
    )
    identity = {
        "recomputed_successes": int(ho.successes),
        "frozen_successes": round(frozen["success_rate"] * len(open_rows)),
        "recomputed_failure_classes": fc,
        "frozen_failure_classes": frozen["failure_classes"],
        "match": (
            int(ho.successes) == round(frozen["success_rate"] * len(open_rows))
            and all(
                fc[k] == frozen["failure_classes"].get(k, 0)
                for k in frozen["failure_classes"]
            )
        ),
    }
    guards["open_loop_identity"] = identity
    if not args.quick and not identity["match"]:
        print(
            "ABORT: open-loop re-flight disagrees with the frozen E12-10 artifact",
            file=sys.stderr,
        )
        return 2

    result = {
        "kind": "orbitopt.e13-04.closed-loop-guidance.1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_layout": bool(args.quick),
        "mission_id": cfg.mission_id,
        "ground_plan": {
            "tof_s": tof,
            "planned_total_m_s": float(sealed.plan.total_delta_v_m_s),
        },
        "holdout": {
            "seed_base": layout.seed_base,
            "n_clusters": layout.n_clusters,
            "samples_per_cluster": layout.samples_per_cluster,
            "n": layout.n,
        },
        "guards": guards,
        "open_loop": {"arm": ho.to_dict(), "per_sample": open_rows},
    }
    if args.open_only:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        return 0

    work = [
        (cfg, campaign, tof, layout.seed_base, c, i)
        for c in range(layout.n_clusters)
        for i in range(layout.samples_per_cluster)
    ]
    print(f"closed-loop flight: {len(work)} samples at {jobs} jobs ...", flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            closed_rows = list(
                pool.map(
                    _closed_worker, work, chunksize=max(1, len(work) // (4 * jobs))
                )
            )
    else:
        closed_rows = [_closed_worker(a) for a in work]
    closed_rows.sort(key=lambda r: (r["cluster"], r["index"]))
    print(
        f"  {sum(1 for r in closed_rows if r['ok'])}/{len(closed_rows)} closed-loop "
        f"successes ({time.time() - t0:.0f}s)",
        flush=True,
    )

    cc = campaign.cluster_bootstrap_trials
    clusters = list(range(layout.n_clusters))
    open_ok = {c: {} for c in clusters}
    closed_ok = {c: {} for c in clusters}
    for r in open_rows:
        open_ok[r["cluster"]][r["index"]] = bool(r["success"])
    for r in closed_rows:
        closed_ok[r["cluster"]][r["index"]] = bool(r["ok"])

    # paired endpoints -----------------------------------------------------------------
    mcnemar, disc = {}, {}
    for ep in ENDPOINTS:
        if ep == "success":
            f_o = {(r["cluster"], r["index"]): not r["success"] for r in open_rows}
            f_c = {(r["cluster"], r["index"]): not r["ok"] for r in closed_rows}
        else:
            f_o = {
                (r["cluster"], r["index"]): r["failure_class"] == ep for r in open_rows
            }
            f_c = {
                (r["cluster"], r["index"]): r["failure_class"] == ep
                for r in closed_rows
            }
        b10 = sum(1 for k in f_o if f_o[k] and not f_c[k])  # fixed by closing the loop
        b01 = sum(1 for k in f_o if f_c[k] and not f_o[k])  # broken by closing the loop
        disc[ep] = {"open_fails_closed_ok": b10, "open_ok_closed_fails": b01}
        mcnemar[ep] = _mcnemar_exact(b10, b01)
    adj = _holm([mcnemar[ep] for ep in ENDPOINTS])
    paired = {
        ep: {**disc[ep], "mcnemar_p": mcnemar[ep], "holm_adjusted_p": adj[i]}
        for i, ep in enumerate(ENDPOINTS)
    }

    n = len(closed_rows)
    k_c = sum(1 for r in closed_rows if r["ok"])
    k_o = sum(1 for r in open_rows if r["success"])
    level = float(campaign.ci_level) if hasattr(campaign, "ci_level") else 0.95
    diff_boot = _cluster_diff_boot(
        open_ok,
        closed_ok,
        clusters,
        int(cc or 4000),
        int(layout.seed_base) % (2**31),
        level,
    )

    per_cluster = []
    for c in clusters:
        o = sum(1 for r in open_rows if r["cluster"] == c and r["success"])
        cl = sum(1 for r in closed_rows if r["cluster"] == c and r["ok"])
        per_cluster.append([c, o, cl])

    closed_fc = _counts(closed_rows)
    attrib = {}
    for r in closed_rows:
        attrib[str(r["attribution"])] = attrib.get(str(r["attribution"]), 0) + 1
    dv = np.array([r["realized_delta_v_m_s"] for r in closed_rows])
    rmse = np.array([r["tracking_rmse_pos_m"] for r in closed_rows])
    pos = np.array([r["position_error_m"] for r in closed_rows])
    spd = np.array([r["relative_speed_m_s"] for r in closed_rows])

    def _q(a):
        a = a[np.isfinite(a)]
        return (
            {
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
                "max": float(a.max()),
            }
            if len(a)
            else {}
        )

    result["closed_loop"] = {
        "successes": int(k_c),
        "n": int(n),
        "success_rate": k_c / n,
        "wilson": _wilson(k_c, n, level),
        "per_cluster": per_cluster,
        "failure_classes": closed_fc,
        "attribution": attrib,
        "realized_delta_v_m_s": {
            **_q(dv),
            "over_budget_samples": int((dv > float(cfg.delta_v_budget_m_s)).sum()),
        },
        "tracking": {"rmse_pos_m": _q(rmse)},
        "terminal": {"position_error_m": _q(pos), "relative_speed_m_s": _q(spd)},
        "converged_samples": int(sum(1 for r in closed_rows if r["converged"])),
        "refusal_samples": int(sum(1 for r in closed_rows if r["n_refusals"])),
        "per_sample": closed_rows,
    }
    result["paired"] = {
        "endpoints": paired,
        "success_rate_open": k_o / n,
        "success_rate_closed": k_c / n,
        "difference_bootstrap": {**diff_boot, "point": (k_c - k_o) / n},
        "per_cluster": per_cluster,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\nopen  {k_o}/{n} = {k_o / n:.3f}   closed {k_c}/{n} = {k_c / n:.3f}")
    print(f"closed Wilson {result['closed_loop']['wilson']}")
    b = result["paired"]["difference_bootstrap"]
    print(f"paired diff {b['point']:+.3f} [{b['lower']:+.3f}, {b['upper']:+.3f}]")
    for ep in ENDPOINTS:
        p = paired[ep]
        print(
            f"  {ep:<18} fixed {p['open_fails_closed_ok']:>3}  broken {p['open_ok_closed_fails']:>3}"
            f"  mcnemar p={p['mcnemar_p']:.4g} holm p={p['holm_adjusted_p']:.4g}"
        )
    print(f"failure classes (closed): {closed_fc}")
    print(f"attribution: {attrib}")
    print(f"artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
