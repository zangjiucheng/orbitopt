"""Benchmark: E11-04 counterfactual admission validation (real re-flight).

E10-06 (task #303) redesigned the joint-chance binary gate into the graded
admission rule of :mod:`orbitopt.flightlab.recovery.redesign` and validated
it OFFLINE: re-grading the recorded E7-04 joint arm, rejections fall 99 -> 84
and false rejections 20 -> 10, and of the 15 newly admitted training-grid
worlds "10 land, 5 fail" -- but that split is the PAIRED FL-305 baseline's
outcome, not the new rule's. A rejected world never flew the landing burn,
so what the admitted rule actually achieves on those exact worlds was
inferred, never measured (the E10-06 report says so under
``no_success_claim``; the E11 epic gates closure on "the effect of the new
admission rule is confirmed by measured re-flight"). This benchmark pays
that debt: every newly admitted world -- the 15 training-grid rows plus the
16 hold-out ``a_seeds`` rows, the 1 ``b_bands`` row and the 0 ``c_rtls``
rows of the E10-06 regrade -- is really re-flown under the shipped policy
(FL-302 with ``joint_admission=AdmissionPolicy().to_dict()``, every other
plan option, dispersion and [seed, idx] draw stream identical to the
recorded campaign), and the MEASURED landing rate replaces the inference.

Protocol per flipped world [seed, idx]: first re-fly WITHOUT the policy --
the current old-rule verdict on the identical physical world -- then WITH
the policy -- the counterfactual. Non-rejection control rows per cell
(landings and physical failures) are re-flown WITHOUT the policy as
backing evidence for the structural premise that the policy cannot touch
worlds the old gate already settled (its ``satisfies`` layer short-circuits
admitted rows, so the rule only ever un-rejects).

Era validity is measured, not assumed. The E7-05 hold-out cells were flown
AFTER the #287 closed-loop-guidance series and reproduce bit-for-bit on
current HEAD (rel/abs 1e-9 on outcome, reason and every corridor/fuel/
landing/joint read-out) -- there the replay-vs-recorded agreement is the
pairing proof. The E7-04 joint arm predates that series (the same staleness
E7-04 itself documented for the E3-02-era joint rows it replaced with a
fresh baseline), so its replays report per-row era drift instead of gating
on it, and the fresh no-policy replay serves as the old-rule baseline for
those rows. Era drift is itself a reported finding.

Readings (task #311 acceptance):

1. MEASURED landing rate of the newly admitted rows per cell (Wilson +
   whole-seed cluster CIs), against the FL-305 inference 10/15 = 0.667
   (exact binomial) and against the 0.208 climatology the floor equals;
2. the failure categories of admitted worlds that still fail (reason codes
   + R-06 failure classes) and the FL-305-vs-measured discordance (McNemar
   on the training rows both configurations "flew");
3. paired rule-effect statistics (fresh replay vs policy flight, same
   worlds): did the calibration improvement translate into a higher landing
   rate -- and the honest analysis when the measured rate sits below the
   FL-305 inference (does the admitted set still clear the admission floor
   -> keep the rule, else re-fit/update).

Cost: the flipped rows are flown twice (replay + admit) plus a few controls
per cell -- ~76 samples, ~10-75 s each per core (cells a/b cost more per
sample than the training grid; ``c_rtls`` has no flips and is
recorded-only):

    PYTHONPATH=src python benchmarks/bench_e11_04_counterfactual_admission.py \
        --workers 8 -o docs/benchmarks/e11_04_counterfactual_admission.json \
        --md docs/benchmarks/e11_04_counterfactual_admission.md

    # offline re-cut of a previous run's flights (no physics, seconds):
    PYTHONPATH=src python benchmarks/bench_e11_04_counterfactual_admission.py \
        --fly-cache docs/benchmarks/e11_04_counterfactual_admission.json \
        -o /tmp/recut.json --md /tmp/recut.md

    # smoke: machinery check, flips capped (claims then withheld):
    ... --max-flips-per-cell 2 --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import fsum
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from orbitopt.flightlab.contract import load_target, write_run  # noqa: E402
from orbitopt.flightlab.recovery.calibration import brier_score  # noqa: E402
from orbitopt.flightlab.recovery.chance import JointChanceSpec  # noqa: E402
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS, Dispersions, _run_sample)
from orbitopt.flightlab.recovery.redesign import (  # noqa: E402
    AdmissionPolicy, GATE_ADMITTED)
from orbitopt.flightlab.recovery.reliability import (  # noqa: E402
    cluster_bootstrap_rate_ci)
from orbitopt.flightlab.recovery.stats import (  # noqa: E402
    classify_failure, failure_breakdown, mcnemar_test, success_rate_interval)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel  # noqa: E402

E3_03_REPORT = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"
E7_04_REPORT = REPO / "docs" / "benchmarks" / "e7_04_failure_driven.json"
E7_05_REPORT = REPO / "docs" / "benchmarks" / "e7_05_out_of_sample_reliability.json"
E10_06_REPORT = REPO / "docs" / "benchmarks" / "e10_06_chance_constraint_redesign.json"

#: replay agreement tolerance: identical physics re-integrated; float repr
#: round-trips and cross-machine contraction make exact equality too strict
REPLAY_RTOL = 1e-9
REPLAY_ABS = 1e-9

#: control rows per cell re-flown without the policy: first (seed, idx) rows
#: the old gate did not reject (a mix of landings and physical failures)
DEFAULT_CONTROL_N = 4

#: the E10-06 regrade counts this re-flight must reproduce exactly
RECORDED_FLIPS = {"train": 15, "a_seeds": 16, "b_bands": 1, "c_rtls": 0}

#: cells whose recorded rows are expected to reproduce bit-for-bit on current
#: HEAD (flown after the #287 guidance series); violations gate the verdict.
#: The E7-04 training arm predates it -- there replay-vs-recorded agreement
#: is reported as an era-drift diagnostic and the fresh replay is the
#: old-rule baseline (E7-04 set this precedent for stale joint rows).
BIT_VALID_CELLS = ("a_seeds", "b_bands")

BOOT_SEED = 20261122
BOOT_TRIALS = 4000
CLAIM_ALPHA = 0.05

# the E7-05 hold-out options (the cells do not record plan_options, but the
# campaign script pins them): stock MC plan, joint-pinned, chance spec rebuilt
# from the bands flown. The training cell instead takes the RECORDED
# plan_options of the E7-04 joint_base arm.
_MC_JOINT = {**MC_PLAN_OPTIONS, "landing_divert_budget_m": None,
             "landing_reserve_margin_kg": 0.0}
_WIDE_FACTOR = 1.5


def _wide_dispersions() -> Dispersions:
    d = Dispersions()
    return Dispersions(**{k: v * _WIDE_FACTOR for k, v in d.to_dict().items()})


def _joint_chance(disp: Dispersions) -> dict[str, float]:
    s = JointChanceSpec.from_dispersion_bands(disp)
    return {"thrust_pct": s.thrust_pct, "isp_pct": s.isp_pct,
            "density_pct": s.density_pct, "cd_pct": s.cd_pct}


def _exact_binom_two_sided(k: int, n: int, p0: Fraction) -> float:
    """Two-sided exact binomial p against ``p0`` (minimum-likelihood rule:
    sum P(X = j) over j with P(X = j) <= P(X = k)) in exact rational
    arithmetic -- n <= 168 here, so this is trivially cheap."""
    probs = [Fraction(math.comb(n, j)) * p0 ** j * (1 - p0) ** (n - j)
             for j in range(n + 1)]
    pk = probs[k]
    return min(1.0, float(sum(q for q in probs if q <= pk)))


# ---------------------------------------------------------------------------
# configuration: cells, flips, controls, flight options
# ---------------------------------------------------------------------------

def _cell_setup(policy: AdmissionPolicy, control_n: int,
                max_flips: int | None) -> list[dict[str, Any]]:
    """Everything needed to re-fly each cell: recorded rows, the flipped
    (newly admitted) coordinates, controls, exact options and dispersions.
    Flips are identified exactly as E10-06 identified them (recorded
    ``JOINT_CHANCE_VIOLATED`` rows the policy admits); ``b_bands`` is
    cross-checked against the campaign's wide-band spec so the flight-time
    gate admits the same row."""
    from test_flightlab_simulate_recovery import TARGET, VEHICLE
    e704 = json.loads(E7_04_REPORT.read_text(encoding="utf-8"))
    arm = next(a for a in e704["arms"] if a["name"] == "e7_04_joint_base")
    e705 = json.loads(E7_05_REPORT.read_text(encoding="utf-8"))
    cells = {c["name"]: c for c in e705["holdout"]}
    std = JointChanceSpec()
    wide = JointChanceSpec.from_dispersion_bands(_wide_dispersions())

    def _flips(rows: Sequence[Mapping[str, Any]], spec: JointChanceSpec,
               flight_spec: JointChanceSpec | None = None) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            d = policy.decide(r, spec)
            if (d is None or d["worst"] <= spec.violation_tolerance
                    or r["reason"] != "JOINT_CHANCE_VIOLATED"
                    or not d["admitted"]):
                continue
            assert d["gate"] == GATE_ADMITTED
            if flight_spec is not None:
                w = policy.decide(r, flight_spec)
                assert w is not None and w["admitted"], (
                    "a flip the flight-time campaign spec would not admit")
            out.append({"seed": int(r["seed"]), "idx": int(r["idx"]),
                        "worst": d["worst"], "second_worst": d["second_worst"],
                        "p_raw": d["p_raw"], "p_hat": d["calibrated_probability"]})
        return out

    plan = [
        ("train", arm["paired_rows"], Dispersions(), dict(arm["plan_options"]),
         std, None),
        ("a_seeds", cells["a_seeds"]["paired_rows"], Dispersions(),
         {**_MC_JOINT, "joint_chance": _joint_chance(Dispersions())}, std, None),
        ("b_bands", cells["b_bands"]["paired_rows"], _wide_dispersions(),
         {**_MC_JOINT, "joint_chance": _joint_chance(_wide_dispersions())},
         std, wide),
    ]
    setup = []
    for name, rows, disp, opts, spec, flight_spec in plan:
        flips = _flips(rows, spec, flight_spec)
        if max_flips is None:
            assert len(flips) == RECORDED_FLIPS[name], (
                f"{name}: {len(flips)} flips != E10-06 recorded "
                f"{RECORDED_FLIPS[name]} -- regrade drifted")
        flip_keys = {(f["seed"], f["idx"]) for f in flips}
        controls = [{"seed": int(r["seed"]), "idx": int(r["idx"]),
                     "reason": str(r["reason"])}
                    for r in sorted(rows, key=lambda r: (r["seed"], r["idx"]))
                    if (r["seed"], r["idx"]) not in flip_keys
                    and r["reason"] != "JOINT_CHANCE_VIOLATED"][:control_n]
        setup.append({"name": name, "rows": rows,
                      "flips": flips[:max_flips] if max_flips else flips,
                      "controls": controls, "disp": disp, "options": opts,
                      "spec": spec, "target": TARGET, "vehicle": VEHICLE})
    # c_rtls: recorded-only (E10-06: 38 -> 38 rejections, no flips to re-fly)
    setup.append({"name": "c_rtls",
                  "rows": cells["c_rtls"]["paired_rows"], "flips": [],
                  "controls": [], "disp": None, "options": None,
                  "spec": wide, "target": None, "vehicle": None})
    return setup


# ---------------------------------------------------------------------------
# flights
# ---------------------------------------------------------------------------

def _fly_one(job: dict[str, Any]) -> dict[str, Any]:
    """Pool worker: fly one recorded campaign world once, with or without
    the admission policy appended to the recorded plan options."""
    rec = _run_sample(job)
    return {"cell": job["_cell"], "kind": job["_kind"],
            "seed": job["seed"], "idx": job["idx"], "record": rec}


def _build_jobs(setup: Sequence[Mapping[str, Any]], policy: AdmissionPolicy,
                run_dirs: Mapping[str, str]) -> list[dict[str, Any]]:
    policy_dict = json.loads(json.dumps(policy.to_dict()))
    jobs = []
    for cell in setup:
        if not cell["flips"] and not cell["controls"]:
            continue
        target = load_target(cell["target"])
        unc = UncertaintyModel.from_dispersions(cell["disp"]).to_dict()
        base_opts = dict(cell["options"])
        admit_opts = {**base_opts, "joint_admission": policy_dict}
        common = {"run_dir": run_dirs[cell["name"]], "target": target,
                  "vehicle_path": str(cell["vehicle"]), "uncertainty": unc,
                  "capture_tracks": None, "site": [0.0, 0.0, 0.0]}
        for kind, opts, coords in (
                ("replay", base_opts, cell["flips"]),
                ("admit", admit_opts, cell["flips"]),
                ("control", base_opts, cell["controls"])):
            for c in coords:
                jobs.append({"_cell": cell["name"], "_kind": kind,
                             "idx": c["idx"], "seed": c["seed"],
                             "plan_options": dict(opts), **common})
    return jobs


def _fly_all(jobs: Sequence[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            return list(pool.map(_fly_one, jobs))
    return [_fly_one(job) for job in jobs]


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=REPLAY_RTOL,
                        abs_tol=REPLAY_ABS)


def _row_agrees(recorded: Mapping[str, Any], flown: Mapping[str, Any]) -> list[str]:
    """Named disagreements between a recorded paired row and its replay."""
    bad = []
    for key in ("outcome", "reason"):
        if str(flown.get(key)) != str(recorded.get(key)):
            bad.append(f"{key} {recorded[key]} -> {flown.get(key)}")
    for key in ("corridor_miss_m", "corridor_fuel_margin_kg", "position_error_m",
                "touchdown_east_m", "touchdown_north_m", "landing_probability",
                "joint_violation"):
        if key in recorded and not _close(recorded.get(key), flown.get(key)):
            bad.append(f"{key} {recorded.get(key)!r} -> {flown.get(key)!r}")
    return bad


def _rate(k: int, n: int, per_seed: Mapping[int, tuple[int, int]]) -> dict[str, Any]:
    out: dict[str, Any] = {"landed": k, "n": n, "rate": k / n if n else None,
                           "wilson": success_rate_interval(k, n) if n else None}
    counts = [per_seed[s] for s in sorted(per_seed)]
    out["cluster_ci95"] = (list(cluster_bootstrap_rate_ci(
        counts, trials=BOOT_TRIALS, seed=BOOT_SEED)) if counts else None)
    return out


def _analyze(setup: Sequence[Mapping[str, Any]],
             flights: Sequence[Mapping[str, Any]], policy: AdmissionPolicy,
             max_flips: int | None) -> dict[str, Any]:
    flown = {(f["cell"], f["kind"], f["seed"], f["idx"]): f["record"]
             for f in flights}
    e303 = json.loads(E3_03_REPORT.read_text(encoding="utf-8"))
    fl305 = {(r["seed"], r["idx"]): r
             for r in e303["runs"]["fl305_baseline"]["paired_rows"]}
    e1006 = json.loads(E10_06_REPORT.read_text(encoding="utf-8"))
    inf_base = e1006["acceptance_2_rejection_rules"]["newly_admitted_by_paired_baseline"]
    inf_landed, inf_n = int(inf_base["landed"]), int(inf_base["landed"]) + int(
        inf_base["failed"])
    inferred_rate = Fraction(inf_base["landed"], inf_n)
    floor = policy.min_admission_probability

    cells_out, hard_violations, era_drift, pooled_pairs = [], [], [], []
    rule_effect_pairs: list[tuple[bool, bool]] = []
    pooled_per_seed: dict[int, tuple[int, int]] = {}
    struct_satisfied_rejected = {}
    for cell in setup:
        rows = {(int(r["seed"]), int(r["idx"])): r for r in cell["rows"]}
        # structural premise, offline and exact: the policy never rejects a
        # row the old gate accepted (the `satisfies` layer short-circuits)
        n_srej = sum(1 for r in cell["rows"]
                     if (d := policy.decide(r, cell["spec"])) is not None
                     and d["worst"] <= cell["spec"].violation_tolerance
                     and not d["admitted"])
        struct_satisfied_rejected[cell["name"]] = n_srej
        per_seed: dict[int, tuple[int, int]] = {}
        detail, admit_recs = [], []
        drift = reproduced = 0
        for f in cell["flips"]:
            rec = rows[(f["seed"], f["idx"])]
            entry = dict(f, recorded_reason=rec["reason"])
            if (f["seed"], f["idx"]) in fl305:
                entry["fl305_outcome"] = fl305[(f["seed"], f["idx"])]["outcome"]
                entry["fl305_reason"] = fl305[(f["seed"], f["idx"])]["reason"]
            replay = flown.get((cell["name"], "replay", f["seed"], f["idx"]))
            if replay is not None:
                diffs = _row_agrees(rec, replay)
                entry["replay_agrees"] = not diffs
                entry["replay_today"] = {"outcome": replay.get("outcome"),
                                         "reason": replay.get("reason")}
                if diffs:
                    drift += 1
                    era_drift.append({"cell": cell["name"], "seed": f["seed"],
                                      "idx": f["idx"], "diff": diffs[:6]})
                else:
                    reproduced += 1
                # the fresh replay IS the old-rule baseline on today's physics
                entry["old_rule_lands_today"] = replay.get("outcome") == "landed"
            admit = flown.get((cell["name"], "admit", f["seed"], f["idx"]))
            if admit is not None:
                landed = admit.get("outcome") == "landed"
                entry.update(measured_outcome=admit.get("outcome"),
                             measured_reason=admit.get("reason"),
                             landed=landed,
                             measured_failure_class=(
                                 None if landed else
                                 classify_failure(str(admit.get("reason") or ""))))
                if str(admit.get("reason") or "").startswith("SAMPLE_ERROR"):
                    hard_violations.append({
                        "cell": cell["name"], "kind": "admit", "seed": f["seed"],
                        "idx": f["idx"],
                        "diff": [f"SAMPLE_ERROR: {admit.get('reason')}"]})
                admit_recs.append(admit)
                prev = per_seed.get(f["seed"], (0, 0))
                per_seed[f["seed"]] = (prev[0] + (1 if landed else 0), prev[1] + 1)
                pooled_pairs.append((f["p_hat"], 1.0 if landed else 0.0))
            detail.append(entry)
        controls_ok, control_drift = 0, 0
        for c in cell["controls"]:
            rec = rows[(c["seed"], c["idx"])]
            ctl = flown.get((cell["name"], "control", c["seed"], c["idx"]))
            diffs = _row_agrees(rec, ctl) if ctl is not None else ["not flown"]
            ok = not diffs
            controls_ok += int(ok)
            control_drift += int(not ok)
            if not ok:
                (era_drift if cell["name"] not in BIT_VALID_CELLS
                 else hard_violations).append(
                    {"cell": cell["name"], "kind": "control",
                     "seed": c["seed"], "idx": c["idx"], "diff": diffs[:6]})
        # fix the paired rule-effect rows now that both verdicts are known
        for e in detail:
            if "measured_outcome" in e and "replay_agrees" in e:
                rule_effect_pairs.append((e["landed"],
                                         bool(e.get("old_rule_lands_today"))))
        landed_n = sum(1 for e in detail if e.get("landed"))
        flown_n = sum(1 for e in detail if "measured_outcome" in e)
        viol_today = sum(1 for e in detail if not e.get("old_rule_lands_today",
                                                        False)
                         and "replay_agrees" in e)
        viol_today_landed = sum(1 for e in detail if e.get("landed")
                                and not e.get("old_rule_lands_today", False))
        for s, (a, b) in per_seed.items():
            prev = pooled_per_seed.get(s, (0, 0))
            pooled_per_seed[s] = (prev[0] + a, prev[1] + b)
        rec_landed = sum(1 for r in cell["rows"] if r["outcome"] == "landed")
        n_rows = len(cell["rows"])
        cells_out.append({
            "name": cell["name"], "rows": n_rows,
            "recorded_joint_rejections": sum(
                1 for r in cell["rows"]
                if r["reason"] == "JOINT_CHANCE_VIOLATED"),
            "newly_admitted": len(detail),
            "recorded_newly_admitted": RECORDED_FLIPS[cell["name"]],
            "flights": {"replays": sum(1 for e in detail if "replay_agrees" in e),
                        "admitted": flown_n, "controls": len(cell["controls"])},
            "era_validity": {"flips_reproducing_recorded": reproduced,
                             "flips_drifted": drift,
                             "controls_reproducing": controls_ok,
                             "controls_drifted": control_drift,
                             "bit_valid_on_head": cell["name"] in BIT_VALID_CELLS},
            "newly_admitted_measured": _rate(landed_n, flown_n, per_seed),
            "mean_predicted_p_hat": (fsum(e["p_hat"] for e in detail)
                                     / len(detail)) if detail else None,
            "failure_categories": failure_breakdown(admit_recs) if admit_recs else None,
            "flips": detail,
            "satisfied_rows_newly_rejected": n_srej,
            "campaign": {
                "recorded_landed": rec_landed, "n": n_rows,
                "recorded_rate": rec_landed / n_rows,
                # the rule's gain on CURRENT physics: worlds whose fresh
                # replay still rejects AND the policy flight then lands
                "violation_today": viol_today,
                "admitted_after_violation_today_landed": viol_today_landed,
                "projected_new_rate": (rec_landed + viol_today_landed) / n_rows},
        })
    measured = sum(c["newly_admitted_measured"]["landed"] for c in cells_out)
    measured_n = sum(c["newly_admitted_measured"]["n"] for c in cells_out)
    train = next(c for c in cells_out if c["name"] == "train")
    train_landed = train["newly_admitted_measured"]["landed"]
    train_n = train["newly_admitted_measured"]["n"]
    # FL-305 vs measured, over training flips both configurations "flew":
    m_rows = [e for e in train["flips"]
              if "measured_outcome" in e and "fl305_outcome" in e]
    fl305_landed = sum(1 for e in m_rows if e["fl305_outcome"] == "landed")
    discordant = [{"seed": e["seed"], "idx": e["idx"], "p_hat": e["p_hat"],
                   "worst": e["worst"], "fl305": e["fl305_reason"],
                   "measured": e["measured_reason"],
                   "class": e["measured_failure_class"]}
                  for e in m_rows
                  if (e["fl305_outcome"] == "landed") != e["landed"]]
    explanation = None
    if train_n and train_landed / train_n < float(inferred_rate):
        worse = [d for d in discordant if d["fl305"] == "LANDED"
                 and d["measured"] != "LANDED"]
        better = [d for d in discordant if d["fl305"] != "LANDED"
                  and d["measured"] == "LANDED"]
        cls = Counter(d["class"] or "LANDED" for d in worse)
        explanation = {
            "train_measured": train_landed / train_n,
            "inferred_from_fl305": float(inferred_rate),
            "fl305_landed_but_measured_failed": worse,
            "fl305_failed_but_measured_landed": better,
            "failure_classes_of_lost_landings": dict(cls),
            "reading": ("FL-305 is a DIFFERENT planner's trajectory over the "
                        "same world -- an inference proxy, not a "
                        "counterfactual: a world FL-305 happened to land can "
                        "still miss its burn under the guidance this flight "
                        "implies (lost-landing classes above; part of the "
                        "gap is the E7-04-era physics the grid was recorded "
                        "under, see era_validity), and vice versa. The "
                        "measurement replaces the inference wherever they "
                        "disagree; per-row agreement is reported, not "
                        "assumed."),
        }
    rule_update_required = bool(measured_n and measured / measured_n < floor)
    return {
        "policy": {"deep_violation": policy.deep_violation,
                   "second_violation_max": policy.second_violation_max,
                   "min_admission_probability": floor},
        "cells": cells_out,
        "pooled": {
            "newly_admitted": {"measured": _rate(measured, measured_n,
                                                pooled_per_seed),
                               "brier_p_hat_vs_measured": brier_score(pooled_pairs)
                               if pooled_pairs else None,
                               "mean_predicted_p_hat": fsum(p for p, _ in
                                                            pooled_pairs)
                               / len(pooled_pairs) if pooled_pairs else None},
            "rule_effect_paired": mcnemar_test(rule_effect_pairs),
            "recorded_landed": sum(c["campaign"]["recorded_landed"]
                                   for c in cells_out),
            "rows": sum(c["campaign"]["n"] for c in cells_out),
            "recorded_rate": sum(c["campaign"]["recorded_landed"]
                                 for c in cells_out)
            / sum(c["campaign"]["n"] for c in cells_out),
            "projected_new_rate": (sum(c["campaign"]["recorded_landed"] +
                                       c["campaign"]["admitted_after_violation_today_landed"]
                                       for c in cells_out))
            / sum(c["campaign"]["n"] for c in cells_out),
            "violations_today": sum(c["campaign"]["violation_today"]
                                    for c in cells_out),
        },
        "training_vs_inference": {
            "inferred": {"landed": inf_landed, "n": inf_n,
                         "rate": float(inferred_rate),
                         "source": "E10-06 paired FL-305 baseline"},
            "measured": {"landed": train_landed, "n": train_n,
                         "rate": train_landed / train_n if train_n else None},
            "exact_binomial_vs_inferred_p": _exact_binom_two_sided(
                train_landed, train_n, inferred_rate) if train_n else None,
            "mcnemar_measured_vs_fl305": mcnemar_test(
                [(e["landed"], e["fl305_outcome"] == "landed")
                 for e in m_rows]) if m_rows else None,
            "fl305_landed": fl305_landed,
            "below_inference_explanation": explanation,
        },
        "structural_premise": {
            "satisfied_rows_newly_rejected": struct_satisfied_rejected,
            "note": ("AdmissionPolicy._gate short-circuits worst <= tolerance "
                     "to gate 'satisfies' (never rejected), so the campaign "
                     "landed-set can only grow; controls re-fly old-gate-"
                     "satisfied worlds WITHOUT the policy."),
        },
        "era_validity": {"drifted_rows": era_drift,
                         "bit_valid_cells": list(BIT_VALID_CELLS)},
        "hard_violations": hard_violations,
        "replay_tolerance": {"rel": REPLAY_RTOL, "abs": REPLAY_ABS},
    }


# ---------------------------------------------------------------------------
# verdict + report
# ---------------------------------------------------------------------------

def _verdict(analysis: Mapping[str, Any], max_flips: int | None) -> dict[str, Any]:
    a = analysis
    cells = a["cells"]
    flips_expected = sum(c["newly_admitted"] for c in cells)
    flips_flew = sum(c["flights"]["admitted"] for c in cells)
    replays_flew = sum(c["flights"]["replays"] for c in cells)
    claims_valid = max_flips is None
    # A1: bit-valid cells reproduce their recorded rows exactly (pairing
    # proof), and no admitted flight died with SAMPLE_ERROR anywhere
    bit_valid_ok = all(
        c["era_validity"]["flips_drifted"] == 0
        and c["era_validity"]["controls_drifted"] == 0
        for c in cells if c["name"] in BIT_VALID_CELLS)
    no_sample_error = not any("SAMPLE_ERROR" in str(m["diff"])
                              for m in a["hard_violations"])
    pairing_ok = (bit_valid_ok and no_sample_error
                  and flips_flew == replays_flew == flips_expected)
    pooled = a["pooled"]
    rate = pooled["newly_admitted"]["measured"]["rate"]
    floor = a["policy"]["min_admission_probability"]
    m = pooled["rule_effect_paired"]
    translation = {
        "supported": None, "measured_admitted_rate": rate, "floor": floor,
        "recorded_campaign_rate": pooled["recorded_rate"],
        "projected_new_rate": pooled["projected_new_rate"],
        "paired_p_value": m["exact"]["p_value"] if m["n_discordant"] else None,
        "b": m["successes_a_only"], "c": m["successes_b_only"],
    }
    if claims_valid and pairing_ok and m["n_discordant"]:
        translation["supported"] = bool(m["exact"]["p_value"] < CLAIM_ALPHA
                                        and m["paired_rate_difference"] > 0
                                        and rate is not None and rate >= floor)
    return {
        "A1_pairing_verified": {
            "pass": bool(pairing_ok and claims_valid),
            "bit_valid_cells_reproduce": bool(bit_valid_ok),
            "no_sample_errors": bool(no_sample_error),
            "era_drift_rows_reported": len(a["era_validity"]["drifted_rows"])},
        "A2_measured_not_inferred": {
            "pass": bool(flips_flew == flips_expected and claims_valid),
            "flips_flown": flips_flew, "flips_expected": flips_expected},
        "A3_failure_categories_reported": {
            "pass": all(c["failure_categories"] is not None
                        for c in cells if c["flights"]["admitted"]),
            "detail": "reason codes + R-06 classes per cell, and the "
                      "lost-landing class histogram in the inference diff"},
        "claim_calibration_translates_to_landings": translation,
        "measured_below_inference": bool(
            a["training_vs_inference"]["below_inference_explanation"]),
        "rule_update_required": bool(
            rate is not None and rate < floor and claims_valid),
        "smoke_mode": max_flips is not None,
        "overall_pass": bool(pairing_ok and claims_valid
                             and flips_flew == flips_expected),
    }


def _md(analysis: Mapping[str, Any], verdict: Mapping[str, Any],
        meta: Mapping[str, Any]) -> str:
    a = analysis
    L = ["# E11-04 — counterfactual admission validation (measured re-flight)\n"]
    L.append(f"Generated {meta['generated_utc']} (task #311). Re-flies **every "
             "newly admitted world** of the E10-06 regrade under the shipped "
             "graded admission rule — the measurement the E10-06 report could "
             "only infer from the paired FL-305 baseline. Inputs: "
             "`e7_04_failure_driven.json` (joint arm + plan options), "
             "`e7_05_out_of_sample_reliability.json` (hold-out cells), "
             "`e3_03_robust_montecarlo.json` (FL-305 pairing), "
             "`e10_06_chance_constraint_redesign.json` (inference under test).\n")
    if meta["max_flips_per_cell"]:
        L.append(f"> **SMOKE MODE** — flips capped at "
                 f"{meta['max_flips_per_cell']}/cell; claims withheld by "
                 "construction.\n")
    p = a["policy"]
    L.append(f"Rule (shipped default): reject a crossed handoff when worst > "
             f"{p['deep_violation']} or second > "
             f"{p['second_violation_max']} or P̂(land) < "
             f"{p['min_admission_probability']:.3f}; else admit and fly the "
             "burn. Protocol: each flipped world re-flown *without* the "
             "policy (old-rule baseline on today's physics) and then *with* "
             f"it; identical ascent, dispersion stream [seed, idx] and "
             f"options; replay-vs-recorded checked to rel/abs {REPLAY_RTOL:g} "
             "where the recorded campaign is bit-valid on current HEAD.\n")
    L.append("## Newly admitted worlds, measured\n")
    L.append("| cell | flips | landed (policy) | measured rate (95% Wilson) "
             "| mean P̂ | replay reproduces recorded | old rule lands today |")
    L.append("|---|---|---|---|---|---|---|")
    for c in a["cells"]:
        mz = c["newly_admitted_measured"]
        ev = c["era_validity"]
        ci = (f"{mz['rate']:.3f} ({mz['wilson']['lower']:.3f}–"
              f"{mz['wilson']['upper']:.3f})" if mz["n"] else "—")
        ph = (f"{c['mean_predicted_p_hat']:.3f}"
              if c["mean_predicted_p_hat"] is not None else "—")
        repro = (f"{ev['flips_reproducing_recorded']}/"
                 f"{ev['flips_reproducing_recorded'] + ev['flips_drifted']}")
        old_landed = sum(1 for e in c["flips"] if e.get("old_rule_lands_today"))
        L.append(f"| {c['name']} | {c['newly_admitted']} | {mz['landed']}/"
                 f"{mz['n']} | {ci} | {ph} | {repro} | {old_landed} |")
    po = a["pooled"]["newly_admitted"]["measured"]
    w = po["wilson"]
    rate_txt = f"{po['rate']:.3f}" if po["rate"] is not None else "n/a"
    ci_txt = ("—" if not w else
              f"Wilson {w['lower']:.3f}–{w['upper']:.3f}"
              + (", cluster {:.3f}–{:.3f}".format(*po["cluster_ci95"])
                 if po["cluster_ci95"] else ""))
    mph = a["pooled"]["newly_admitted"]["mean_predicted_p_hat"]
    mph_txt = f"{mph:.3f}" if mph is not None else "n/a"
    L.append(f"\n**Pooled measured landing rate of the newly admitted worlds: "
             f"{po['landed']}/{po['n']} = {rate_txt}** ({ci_txt}), mean "
             f"P̂ {mph_txt} — the floor the rule enforces is "
             f"{p['min_admission_probability']:.3f}.\n")
    ti = a["training_vs_inference"]
    L.append("## Measured vs the FL-305 inference (training grid)\n")
    if ti["measured"]["rate"] is not None:
        mm = ti["mcnemar_measured_vs_fl305"]
        L.append(f"measured {ti['measured']['landed']}/{ti['measured']['n']} = "
                 f"{ti['measured']['rate']:.3f} vs inferred "
                 f"{ti['inferred']['rate']:.3f}; exact binomial vs inference "
                 f"p = {ti['exact_binomial_vs_inferred_p']:.3f}; McNemar vs "
                 f"FL-305 b/c = {mm['successes_a_only']}/{mm['successes_b_only']} "
                 f"p = {mm['exact']['p_value']:.3f}.\n")
    if ti["below_inference_explanation"]:
        ex = ti["below_inference_explanation"]
        L.append(f"Measured **below** inference — why: {ex['reading']}\n")
        L.append("| seed | idx | P̂ | FL-305 | measured | class |")
        L.append("|---|---|---|---|---|---|")
        for d in ex["fl305_landed_but_measured_failed"]:
            L.append(f"| {d['seed']} | {d['idx']} | {d['p_hat']:.3f} "
                     f"| landed | {d['measured']} | {d['class']} |")
        for d in ex["fl305_failed_but_measured_landed"]:
            L.append(f"| {d['seed']} | {d['idx']} | {d['p_hat']:.3f} "
                     f"| {d['fl305']} | landed | — |")
        L.append("")
    L.append("## Failure categories of admitted worlds\n")
    any_fail = False
    for c in a["cells"]:
        fc = c["failure_categories"]
        if fc and fc["n_failures"]:
            any_fail = True
            L.append(f"- **{c['name']}** — " + ", ".join(
                f"{k} {v}" for k, v in sorted(fc["classes"].items()) if v))
            codes = sorted((n, k, v) for k, per in fc["codes_by_class"].items()
                           for n, v in per.items())
            L.append("  codes: " + ", ".join(f"{n} ({k})×{v}" for n, k, v in codes))
    if not any_fail:
        L.append("- none: every re-flown newly admitted world landed.")
    L.append("\n## Rule effect (paired on today's physics)\n")
    m = a["pooled"]["rule_effect_paired"]
    L.append(f"Over the {m['n_pairs']} re-flown flips the policy flight lands "
             f"where the fresh no-policy replay failed on {m['successes_a_only']} "
             f"worlds (b) and lost none of {m['successes_b_only']} (c = "
             "old-rule landings the policy could undo — impossible by "
             f"construction, verified: {sum(a['structural_premise']['satisfied_rows_newly_rejected'].values())} "
             "satisfied rows newly rejected); exact McNemar p = "
             f"{m['exact']['p_value'] if m['n_discordant'] else 1.0:.3g}. "
             f"Of {a['pooled']['violations_today']} flips whose fresh replay "
             f"STILL rejects on today's physics, the admitted worlds landing "
             "is the rule's live gain; the recorded campaigns project "
             f"{a['pooled']['recorded_rate']:.3f} → "
             f"{a['pooled']['projected_new_rate']:.3f} pooled landing rate "
             f"({a['pooled']['rows']} recorded worlds).\n")
    L.append("## Era validity of the recorded campaigns\n")
    L.append("E7-05 hold-out cells are bit-valid on current HEAD "
             "(replays reproduce recorded rows to rel/abs 1e-9) — the pairing "
             "proof. The E7-04 joint arm predates the #287 closed-loop-guidance "
             "series; its rows are re-flown on today's physics with the fresh "
             "replay as old-rule baseline (the same precedent E7-04 itself set "
             "for the stale E3-02-era joint rows):\n")
    for c in a["cells"]:
        ev = c["era_validity"]
        tot = ev["flips_reproducing_recorded"] + ev["flips_drifted"]
        L.append(f"- {c['name']}: {ev['flips_reproducing_recorded']}/{tot} flip "
                 f"replays reproduce, controls "
                 f"{ev['controls_reproducing']}/"
                 f"{ev['controls_reproducing'] + ev['controls_drifted']}"
                 + (" — bit-valid cell" if ev["bit_valid_on_head"] else ""))
    L.append("\n## Verdict\n")
    v = verdict
    L.append(f"- **A1 pairing verified**: "
             f"{'pass' if v['A1_pairing_verified']['pass'] else 'FAIL'} "
             f"(bit-valid cells reproduce: "
             f"{v['A1_pairing_verified']['bit_valid_cells_reproduce']}; no "
             f"SAMPLE_ERROR: {v['A1_pairing_verified']['no_sample_errors']}; "
             f"{v['A1_pairing_verified']['era_drift_rows_reported']} era-drift "
             "rows reported).")
    L.append(f"- **A2 measured, not inferred**: "
             f"{'pass' if v['A2_measured_not_inferred']['pass'] else 'FAIL'} "
             f"({v['A2_measured_not_inferred']['flips_flown']}/"
             f"{v['A2_measured_not_inferred']['flips_expected']} flips flown).")
    L.append(f"- **A3 failure categories**: "
             f"{'pass' if v['A3_failure_categories_reported']['pass'] else 'FAIL'}.")
    t = v["claim_calibration_translates_to_landings"]
    if t["supported"] is None:
        L.append(f"- **calibration → landings?** not claimed "
                 f"({'smoke mode' if v['smoke_mode'] else 'pairing failed'}).")
    else:
        L.append(f"- **calibration → landings?** "
                 f"{'SUPPORTED' if t['supported'] else 'not supported'}: "
                 f"paired b/c = {t['b']}/{t['c']} p = "
                 f"{t['paired_p_value']:.3g}; measured admitted rate "
                 f"{t['measured_admitted_rate'] if t['measured_admitted_rate'] is not None else 'n/a'} "
                 f"vs floor {t['floor']:.3f}; campaigns project "
                 f"{t['recorded_campaign_rate']:.3f} → "
                 f"{t['projected_new_rate']:.3f}.")
    if v["measured_below_inference"]:
        L.append("- measured came in **below** the FL-305 inference; the "
                 "discrepancy is analysed above (FL-305 ≠ counterfactual + "
                 "era drift) and against the floor; rule re-fit "
                 + ("**required**" if v["rule_update_required"] else "not required")
                 + ".")
    L.append(f"- **overall**: {'PASS' if v['overall_pass'] else 'FAIL'}.")
    L.append(f"\n*Machinery: {meta['flights_flown']} flights, "
             f"{meta['wall_seconds']:.0f} s wall, workers={meta['workers']}.*\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--control-n", type=int, default=DEFAULT_CONTROL_N)
    ap.add_argument("--max-flips-per-cell", type=int, default=None,
                    help="smoke mode: cap flips per cell (claims withheld)")
    ap.add_argument("--fly-cache", default=None,
                    help="reuse flights from a previous report JSON "
                         "(offline re-cut, no physics)")
    args = ap.parse_args(argv)

    t0 = time.time()
    policy = AdmissionPolicy()
    setup = _cell_setup(policy, args.control_n, args.max_flips_per_cell)
    if args.fly_cache:
        prev = json.loads(Path(args.fly_cache).read_text(encoding="utf-8"))
        flights = prev["flights"]
        wall = float(prev["meta"]["wall_seconds"])
        print(f"fly-cache: {len(flights)} flights reused, no physics")
    else:
        with tempfile.TemporaryDirectory(prefix="e11-04-") as tmp:
            run_dirs = {}
            from test_flightlab_simulate_recovery import _ascent_with_textbook_separation
            ascent = _ascent_with_textbook_separation()
            for cell in setup:
                if cell["target"] is None:
                    continue
                d = Path(tmp) / cell["name"] / "ascent"
                write_run(d, ascent)
                run_dirs[cell["name"]] = str(d)
            jobs = _build_jobs(setup, policy, run_dirs)
            print(f"flying {len(jobs)} samples "
                  f"(replay+admit flips + controls), workers={args.workers}")
            flights = _fly_all(jobs, args.workers)
        wall = time.time() - t0

    analysis = _analyze(setup, flights, policy, args.max_flips_per_cell)
    verdict = _verdict(analysis, args.max_flips_per_cell)
    meta = {"kind": "orbitopt.e11-04.counterfactual-admission-validation",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "benchmarks/bench_e11_04_counterfactual_admission.py",
            "workers": args.workers, "flights_flown": len(flights),
            "wall_seconds": wall,
            "max_flips_per_cell": args.max_flips_per_cell,
            "control_n": args.control_n,
            "inputs": [str(p.relative_to(REPO)) for p in
                       (E3_03_REPORT, E7_04_REPORT, E7_05_REPORT, E10_06_REPORT)]}
    report = {"meta": meta, "analysis": analysis, "verdict": verdict,
              "flights": flights}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=list) + "\n",
                   encoding="utf-8")
    Path(args.md).write_text(_md(analysis, verdict, meta), encoding="utf-8")
    print(f"wrote {out} and {args.md}")
    print("verdict:", json.dumps(verdict["A1_pairing_verified"], default=list),
          "| measured pooled:", json.dumps(
              analysis["pooled"]["newly_admitted"]["measured"], default=list))
    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
