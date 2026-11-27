"""E13-06 (task #423): supervised closed-loop campaign -- missed alarms and
false alarms, counted separately against truth.

The LEO-RDV-001 holdout (seed base 20 000 000) is flown closed-loop twice
per sample on identical draws: once with the E13-06 safety supervisor alone
(the specificity arm: a clean campaign run must not be touched, and any
truth violation that appears must have been alerted), and once with a
declared navigation fault injected through the observation hook (measurements
report the target 7.5 km closer along the line of sight from t = 60 s -- an
inertial LOS shrink the truth plant never sees). The fault arm measures both
error sides at once: alerts raised while the truth run never violated count
as false alarms, and alerts before fault-induced truth violations count as
detections -- the law steers on the corrupted estimate, so truth genuinely
deviates on some draws. Every arm's post-abort trajectory is a burn-free
coast that the E12-04 ``evaluate_trajectory_safety`` independently re-checks
against the keep-out sphere.

The supervisor is the E13-06 default tuning (3-sigma inflation, warn bands,
2-tick debounce, 180 s hold ceiling); the truth side is exclusively
``orbitopt.rendezvous.safety.evaluate_trajectory_safety`` (E12-04) on the
sealed outcome samples -- the supervisor's own state stream is never used as
evidence about truth.

    PYTHONPATH=src python benchmarks/bench_e13_06_safety_supervisor.py \
        --jobs 8 -o docs/benchmarks/e13_06_safety_supervisor.json

(~2 h at 8 jobs on the full holdout, both arms; use --quick for a smoke
layout.)"""

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.rendezvous import canonical_plan, load_reference_config
from orbitopt.rendezvous import robustness as rb
from orbitopt.rendezvous.gnc_campaign import fly_closed_loop_sample
from orbitopt.gnc.supervisor import SupervisorConfig

MISSION_YAML = REPO / "src/orbitopt/data/leo_rdv_001.yaml"
CAMPAIGN_YAML = REPO / "src/orbitopt/data/leo_rdv_001_robustness.yaml"
FAULT_D_M = 7500.0
FAULT_T_ONSET_S = 60.0


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _los_shrink_hook(d_m: float, t_onset_s: float):
    import dataclasses as dc

    def hook(obs):
        if obs.t_sample_s < t_onset_s:
            return obs
        r = np.asarray(obs.relative_state.r_rel_m, dtype=float)
        rng = float(np.linalg.norm(r))
        shrunk = r * max(rng - d_m, 10.0) / rng
        rs = dc.replace(obs.relative_state, r_rel_m=tuple(float(x) for x in shrunk))
        return dc.replace(obs, relative_state=rs)
    return hook


def _worker(args):
    cfg, campaign, tof, seed_base, cluster, index, faulted = args
    t0 = time.time()
    outcome, assess, _log = fly_closed_loop_sample(
        cfg, campaign, tof, seed_base, cluster, index,
        supervisor=SupervisorConfig(),
        observation_hook=_los_shrink_hook(FAULT_D_M, FAULT_T_ONSET_S) if faulted else None)
    rec = assess["supervision"]
    score = rec["score"]
    post = rec["post_abort"]
    events = rec["events"]
    triggers = {}
    for e in events:
        key = f"{e['state_from']}->{e['state_to']}:{e['trigger']}"
        triggers[key] = triggers.get(key, 0) + 1
    truth_violations = score["counts"]["violation_intervals"]
    return {
        "arm": "nav_fault" if faulted else "clean",
        "cluster": int(cluster),
        "index": int(index),
        "seed": int(assess["draws"]["seed"]),
        "ok": bool(outcome.ok and outcome.failure_class is None),
        "failure_class": outcome.failure_class,
        "final_state": rec["final_state"],
        "n_events": len(events),
        "transitions": triggers,
        "first_alert_s": (score["alert_intervals"][0]["start_s"]
                          if score["alert_intervals"] else None),
        "alerts": score["counts"]["alerts"],
        "truth_violation_intervals": truth_violations,
        "missed": score["counts"]["missed"],
        "false_alarms": score["counts"]["false_alarms"],
        "post_abort": {
            "aborted": bool(post["aborted"]),
            "t_abort_s": post["t_abort_s"],
            "window_s": post.get("window_s"),
            "contingency_s": post.get("contingency_s"),
            "covered": bool(post.get("covered", False)),
            "keep_out_min_margin_m": post.get("keep_out_min_margin_m"),
            "keep_out_violations": int(post.get("keep_out_violations", 0)),
            "keep_out_passed": bool(post.get("keep_out_passed", True)),
        },
        "realized_delta_v_m_s": float(outcome.realized_delta_v_m_s),
        "wall_s": round(time.time() - t0, 1),
    }


def _wilson(k: int, n: int, level: float = 0.95) -> dict:
    from scipy.stats import norm

    if n == 0:
        return {"lower": float("nan"), "upper": float("nan"), "n": 0}
    z = float(norm.ppf(0.5 + level / 2.0))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / d
    return {"n": n, "rate": p, "lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def _aggregate(rows) -> dict:
    n = len(rows)
    alerts = sum(r["alerts"] for r in rows)
    missed = sum(r["missed"] for r in rows)
    false = sum(r["false_alarms"] for r in rows)
    vio = sum(r["truth_violation_intervals"] for r in rows)
    aborted = [r for r in rows if r["post_abort"]["aborted"]]
    post_safe = [r for r in aborted if r["post_abort"]["keep_out_passed"]]
    covered = [r for r in aborted if r["post_abort"]["covered"]]
    return {
        "n": n,
        "supervisor_aborted_samples": len(aborted),
        "supervisor_held_or_nominal_samples": n - len(aborted),
        "alert_intervals": alerts,
        "truth_violation_intervals": vio,
        "missed": missed,
        "miss_rate": _wilson(missed, vio),
        "false_alarms": false,
        "false_alarm_rate_per_sample": _wilson(sum(1 for r in rows if r["false_alarms"]), n),
        "post_abort_keep_out_passed": len(post_safe),
        "post_abort_keep_out_failed": [
            {"cluster": r["cluster"], "index": r["index"],
             "min_margin_m": r["post_abort"]["keep_out_min_margin_m"]}
            for r in aborted if not r["post_abort"]["keep_out_passed"]],
        "post_abort_full_contingency_covered": len(covered),
        "final_states": {s: sum(1 for r in rows if r["final_state"] == s)
                         for s in sorted({r["final_state"] for r in rows})},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="docs/benchmarks/e13_06_safety_supervisor.json")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="small layout for smoke runs; marked in the artifact")
    args = ap.parse_args()

    cfg = load_reference_config()
    campaign = dataclasses.replace(rb.load_reference_campaign(),
                                   passive_contingency_in_success=True)
    if args.quick:
        campaign = dataclasses.replace(
            campaign,
            train=dataclasses.replace(campaign.train, n_clusters=2, samples_per_cluster=4),
            holdout=dataclasses.replace(campaign.holdout, n_clusters=2, samples_per_cluster=4))
    sealed = canonical_plan(cfg)
    tof = float(sealed.plan.tof_s)
    layout = campaign.holdout
    jobs = max(1, int(args.jobs))
    layout_desc = {"seed_base": layout.seed_base, "n_clusters": layout.n_clusters,
                   "samples_per_cluster": layout.samples_per_cluster, "n": layout.n}

    work = [(cfg, campaign, tof, layout.seed_base, c, i, faulted)
            for faulted in (False, True)
            for c in range(layout.n_clusters) for i in range(layout.samples_per_cluster)]
    print(f"supervised closed loop: {len(work)} flights "
          f"({layout.n_clusters} x {layout.samples_per_cluster} x 2 arms) at {jobs} jobs ...",
          flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_worker, work, chunksize=max(1, len(work) // (4 * jobs))))
    else:
        rows = [_worker(a) for a in work]
    rows.sort(key=lambda r: (r["arm"], r["cluster"], r["index"]))
    print(f"  {len(rows)} flights in {time.time() - t0:.0f}s", flush=True)

    clean = [r for r in rows if r["arm"] == "clean"]
    faulted = [r for r in rows if r["arm"] == "nav_fault"]
    sup = SupervisorConfig()
    result = {
        "kind": "orbitopt.e13-06.safety-supervisor.1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_layout": bool(args.quick),
        "mission_id": cfg.mission_id,
        "mission_yaml_sha256": _sha(MISSION_YAML),
        "campaign_yaml_sha256": _sha(CAMPAIGN_YAML),
        "holdout": layout_desc,
        "ground_plan": {"tof_s": tof,
                        "planned_total_m_s": float(sealed.plan.total_delta_v_m_s)},
        "supervisor_config": {
            "kind": "orbitopt.gnc.supervisor-config.1.0",
            "k_sigma": sup.k_sigma, "warn_keep_out_m": sup.warn_keep_out_m,
            "warn_cross_track_m": sup.warn_cross_track_m,
            "warn_cone_deg": sup.warn_cone_deg,
            "warn_approach_speed_m_s": sup.warn_approach_speed_m_s,
            "warn_passive_margin_m": sup.warn_passive_margin_m,
            "trip_ticks": sup.trip_ticks, "clear_ticks": sup.clear_ticks,
            "hold_max_s": sup.hold_max_s, "passive_step_s": sup.passive_step_s,
            "q_accel_m2_s3": sup.q_accel_m2_s3,
            "envelope": {"keep_out_radius_m": cfg.safety.keep_out_radius_m,
                         "passive_contingency_s": cfg.safety.passive_contingency_s},
            "provenance": "DESIGN ASSUMPTION (E13-06): default tuning"},
        "nav_fault": {"kind": "inertial_los_shrink", "delta_m": FAULT_D_M,
                      "t_onset_s": FAULT_T_ONSET_S,
                      "note": "measurements only; truth plant and log keep the truth"},
        "clean": {**_aggregate(clean), "per_sample": clean},
        "nav_fault": {**_aggregate(faulted), "per_sample": faulted},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    for name, agg in (("clean", result["clean"]), ("nav_fault", result["nav_fault"])):
        print(f"{name:>9}: n={agg['n']} alerts={agg['alert_intervals']} "
              f"truth_violations={agg['truth_violation_intervals']} "
              f"missed={agg['missed']} false_alarms={agg['false_alarms']} "
              f"aborted={agg['supervisor_aborted_samples']} "
              f"post_abort_keep_out_passed={agg['post_abort_keep_out_passed']}")
    print(f"artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
