"""E14-10 campaign analysis: collect and verify shard files, the paired statistics, and the report tables.

Every statistic comes from ``benchmarks/e14_spec/metrics.py``, the pre-registered implementation; this module only
arranges the data: seeds aggregate within an instance first, instances are the units, families are the bootstrap
strata. It is not part of the code digest a scored shard records, because nothing here changes a cell's result:
``analyze`` instead refuses shards whose own digests differ.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import e14_common as C

P, A, M, F = C.P, C.A, C.M, C.F
SMOKE = "pipeline-smoke"
ENDPOINTS = ("success", "evaluations", "wall_clock")
SECONDARY_ENDPOINT = "gap"
#: the spec's arm ids for the five primary comparisons; E-rule stands for no pre-registered arm
PRIMARY = tuple(tuple("E-LLM" if a == "E" else a for a in pair) for pair in A.PRIMARY_COMPARISONS)
SECONDARY = (("E-rule", "A"), ("E-rule", "B"))



class MixedDigestError(RuntimeError):
    """Rows from runs under different frozen digests cannot be analysed together."""



# --------------------------------------------------------------------------------------------------- collect
DIGEST_KEYS = ("spec_digest", "protocol_digest", "code_digest", "freeze_digest")


def load_shards(out: Path) -> list[dict]:
    docs = []
    for path in sorted((out / "shards").glob("*.json")):
        doc = json.loads(path.read_text())
        if not C.seal_ok(doc):
            raise RuntimeError(f"{path}: content digest does not match; rerun that shard")
        doc["_file"] = path.name
        docs.append(doc)
    return docs


def require_one_digest(docs: list[dict], keys=DIGEST_KEYS) -> dict:
    """The digests every document shares, or MixedDigestError naming the documents that differ."""
    seen: dict[str, set] = {k: {d.get(k) for d in docs} for k in keys}
    mixed = {k: sorted(map(str, v)) for k, v in seen.items() if len(v) > 1}
    if mixed:
        raise MixedDigestError(f"shards were run under different digests, refusing to analyse them together: "
                               f"{mixed}")
    return {k: next(iter(v)) for k, v in seen.items()}


def collect(out: Path, shards: list[dict]) -> dict:
    docs = load_shards(out)
    scored = [d for d in docs if d["shard"]["split"] == "scoring"]
    tuning = [d for d in docs if d["shard"]["split"] == "tuning"]
    digests = require_one_digest(scored) if scored else {}
    have = {d["shard"]["id"] for d in docs}
    missing = [s["id"] for s in shards if s["id"] not in have]
    manifest = {"schema": "orbitopt.e14-campaign.manifest/1", "digests": digests,
                "files": {f"shards/{d['_file']}": {"sha256": C.file_sha256(out / 'shards' / d['_file']),
                                                    "content_digest": d["content_digest"],
                                                    "split": d["shard"]["split"], "mode": d["mode"],
                                                    "rows": len(d["rows"]), "faults": len(d["faults"]),
                                                    "wall_s": d.get("wall_s"), "host": d["provenance"]["host"]}
                          for d in docs},
                "missing": missing}
    rows = [r for d in docs for r in d["rows"]]
    write_csv(out / "cells.csv", rows)
    manifest["files"]["cells.csv"] = {"sha256": C.file_sha256(out / "cells.csv")}
    C.write_json_atomic(out / "manifest.json", C.sealed(manifest))
    return {"scored_shards": len(scored), "tuning_shards": len(tuning), "missing": missing, "rows": len(rows)}


CSV_FIELDS = ("arm_id", "instance_id", "family", "problem_class", "group", "split", "seed", "mode", "status", "stop",
              "success", "feasible", "gap", "gap_norm", "objective", "hypervolume", "max_violation", "n_evaluations",
              "n_evaluations_independent", "budget_evaluations", "within_budget", "budget_typed", "wall_clock_s",
              "tokens_in", "tokens_out", "cost_usd", "run_status", "verdict", "optimality_certified", "replay_equal",
              "row_digest", "not_run")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            ver = r.get("verification") or {}
            w.writerow({**r, "run_status": ver.get("run_status"), "verdict": ver.get("verdict"),
                        "optimality_certified": (ver.get("statuses") or {}).get("optimality_certified"),
                        "replay_equal": (r.get("replay") or {}).get("equal")})


def verify(out: Path) -> list[str]:
    """Re-hash every file the manifest names; the copy-back step runs this after rsync."""
    manifest = json.loads((out / "manifest.json").read_text())
    problems = [] if C.seal_ok(manifest) else ["manifest.json: content digest does not match"]
    for rel, meta in manifest["files"].items():
        path = out / rel
        if not path.exists():
            problems.append(f"{rel}: missing")
        elif C.file_sha256(path) != meta["sha256"]:
            problems.append(f"{rel}: sha256 differs")
        elif rel.startswith("shards/") and not C.seal_ok(json.loads(path.read_text())):
            problems.append(f"{rel}: content digest does not match")
    return problems


# --------------------------------------------------------------------------------------------------- analysis
def _rows_for_analysis(docs: list[dict], allow_smoke: bool) -> tuple[list[dict], dict, str]:
    scored = [d for d in docs if d["shard"]["split"] == "scoring" and d["shard"]["kind"] in ("cells", "faults",
                                                                                              "formulation")]
    digests = require_one_digest(scored)
    modes = {d["mode"] for d in scored}
    if SMOKE in modes and not allow_smoke:
        raise RuntimeError("pipeline-smoke shards are excluded from every statistic; pass --allow-smoke only to "
                           "test the analysis code, and its output is labelled as such")
    if len(modes) > 1:
        raise MixedDigestError(f"mixed modes {sorted(modes)}")
    return scored, digests, next(iter(modes))


def per_instance(rows: list[dict], arm: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["arm_id"] == arm and r["status"] != "not_run":
            out[r["instance_id"]][r["seed"]] = r
    return out


def ert(cells: dict[int, dict], field: str) -> float:
    succ = sum(1 for r in cells.values() if r["success"])
    total = sum(float(r[field]) for r in cells.values())
    return total / succ if succ else math.inf


def _log_ratio(t: float, c: float) -> float:
    if math.isinf(t) and math.isinf(c):
        return 0.0
    if math.isinf(t):
        return math.inf
    if math.isinf(c):
        return -math.inf
    return math.log(t / c)


def _median_gap(cells: dict[int, dict]) -> float:
    vals = [max(C.num(r["gap_norm"]), 1e-12) for r in cells.values()]
    return statistics.median(vals)


def compare(rows: list[dict], treat: str, ctrl: str, instance_ids: list[str], strata: dict) -> dict:
    """Every endpoint of one paired comparison, clustered by instance; p raw (Holm is applied by the caller)."""
    T, Cc = per_instance(rows, treat), per_instance(rows, ctrl)
    ids = [i for i in instance_ids if i in T and i in Cc and set(T[i]) == set(Cc[i])]
    res: dict = {"treatment": treat, "control": ctrl, "instances": len(ids)}
    if not ids:
        return {**res, "not_run": "no paired cells"}
    # success
    disc, diff = [], {}
    a0 = b0 = pooled_a = pooled_b = 0
    for i in ids:
        seeds = sorted(T[i])
        t = [bool(T[i][s]["success"]) for s in seeds]
        c = [bool(Cc[i][s]["success"]) for s in seeds]
        disc.append(sum(x and not y for x, y in zip(t, c)) - sum(y and not x for x, y in zip(t, c)))
        diff[i] = (sum(t) - sum(c)) / len(seeds)
        pooled_a += sum(x and not y for x, y in zip(t, c))
        pooled_b += sum(y and not x for x, y in zip(t, c))
        s0 = seeds[0]
        a0 += bool(T[i][s0]["success"]) and not bool(Cc[i][s0]["success"])
        b0 += bool(Cc[i][s0]["success"]) and not bool(T[i][s0]["success"])
    ci = M.cluster_bootstrap_ci(diff, {i: strata[i] for i in ids})
    res["success"] = {"p": M.cluster_sign_flip_p(disc), "ci": list(ci), "effect": sum(diff.values()) / len(ids),
                      "informative": sum(1 for d in disc if d != 0),
                      "rate_treatment": sum(bool(T[i][s]["success"]) for i in ids for s in T[i]) / sum(
                          len(T[i]) for i in ids),
                      "rate_control": sum(bool(Cc[i][s]["success"]) for i in ids for s in Cc[i]) / sum(
                          len(Cc[i]) for i in ids),
                      "mcnemar_seed0": {"treatment_only": a0, "control_only": b0, "p": M.exact_mcnemar_p(a0, b0)},
                      "pooled_mcnemar_ignores_clustering": {"treatment_only": pooled_a, "control_only": pooled_b,
                                                            "p": M.exact_mcnemar_p(pooled_a, pooled_b)}}
    for endpoint, field in (("evaluations", "n_evaluations"), ("wall_clock", "wall_clock_s")):
        d = {i: _log_ratio(ert(T[i], field), ert(Cc[i], field)) for i in ids}
        finite = {i: -v for i, v in d.items() if math.isfinite(v)}  # positive = treatment cheaper
        ci = M.cluster_bootstrap_ci(finite, {i: strata[i] for i in finite}) if finite else (math.nan, math.nan)
        res[endpoint] = {"p": M.wilcoxon_signed_rank_p(list(d.values())), "ci": list(ci),
                         "effect": math.exp(-sum(finite.values()) / len(finite)) if finite else math.nan,
                         "ci_effect": [math.exp(-ci[1]), math.exp(-ci[0])],
                         "effect_unit": "geometric mean ERT ratio treatment/control over instances both solved",
                         "informative": sum(1 for v in d.values() if v != 0),
                         "finite": len(finite), "only_treatment_solved": sum(1 for v in d.values() if v == -math.inf),
                         "only_control_solved": sum(1 for v in d.values() if v == math.inf),
                         "neither_solved": sum(1 for i in ids if math.isinf(ert(T[i], field))
                                               and math.isinf(ert(Cc[i], field)))}
    gd = {i: _log_ratio(_median_gap(T[i]), _median_gap(Cc[i])) / math.log(10) for i in ids}
    finite = {i: -v for i, v in gd.items() if math.isfinite(v)}
    ci = M.cluster_bootstrap_ci(finite, {i: strata[i] for i in finite}) if finite else (math.nan, math.nan)
    res["gap"] = {"p": M.wilcoxon_signed_rank_p(list(gd.values())), "ci": list(ci),
                  "effect": -sum(finite.values()) / len(finite) if finite else math.nan,
                  "ci_effect": [-ci[1], -ci[0]],
                  "effect_unit": "mean log10(median gap treatment / median gap control), gaps floored at 1e-12",
                  "informative": sum(1 for v in gd.values() if v != 0), "finite": len(finite)}
    return res


def _holm_into(comparisons: list[dict], keys: list[tuple[int, str]], extra_untested: int = 0) -> None:
    pv = {f"{n}:{e}": comparisons[n][e]["p"] for n, e in keys}
    for k in range(extra_untested):
        pv[f"untested:{k}"] = 1.0
    adj = M.holm(pv)
    for n, e in keys:
        c = comparisons[n][e]
        c["p_holm"] = adj[f"{n}:{e}"]
        c["holm_family_size"] = len(pv)
        c["verdict"] = M.verdict(c["p_holm"], tuple(c["ci"]) if not any(map(math.isnan, c["ci"])) else (0.0, 0.0),
                                 c["informative"])


ALARM_RULE = ("spec section 2: a gap below minus the tolerance is not a win but an alarm; such a cell counts as no "
              "success here (its scorer verdict is kept as success_scorer), and its gap enters the gap endpoint as "
              "its distance from the optimum, |gap|")


def apply_alarm_rule(row: dict) -> dict:
    out = dict(row, success_scorer=row.get("success"))
    if row.get("alarm"):
        out["success"] = False
        out["gap_norm"] = abs(C.num(row["gap_norm"]))
    return out


def comparisons(rows: list[dict], ran: list[str], scoring_ids: list[str], strata: dict) -> list[dict]:
    """Every comparison overall and per analysis group, with Holm applied family by family."""
    scopes = {"overall": scoring_ids}
    for group, classes in M.ANALYSIS_GROUPS.items():
        scopes[group] = [i for i in scoring_ids if P.by_id(i).problem_class in classes]
    comps = []
    for scope, ids in scopes.items():
        for treat, ctrl in PRIMARY + SECONDARY:
            c = compare(rows, treat, ctrl, ids, strata) if treat in ran and ctrl in ran else {
                "treatment": treat, "control": ctrl, "not_run": "an arm of this comparison was not run"}
            c["scope"] = scope
            c["family"] = "primary" if (treat, ctrl) in PRIMARY else "secondary (E-rule is not arm E)"
            comps.append(c)
    tested = [n for n, c in enumerate(comps) if "not_run" not in c]
    over = [n for n in tested if comps[n]["scope"] == "overall"]
    prim = [n for n in over if comps[n]["family"] == "primary"]
    _holm_into(comps, [(n, e) for n in prim for e in ENDPOINTS],
               extra_untested=len(PRIMARY) * len(ENDPOINTS) - len(prim) * len(ENDPOINTS))
    sec = [n for n in over if comps[n]["family"] != "primary"]
    _holm_into(comps, [(n, e) for n in sec for e in ENDPOINTS])
    _holm_into(comps, [(n, SECONDARY_ENDPOINT) for n in over])
    for group in M.ANALYSIS_GROUPS:
        g = [n for n in tested if comps[n]["scope"] == group]
        _holm_into(comps, [(n, e) for n in g for e in ENDPOINTS])
        _holm_into(comps, [(n, SECONDARY_ENDPOINT) for n in g])
    return comps


def analyze(out: Path, allow_smoke: bool = False) -> dict:
    docs = load_shards(out)
    scored, digests, mode = _rows_for_analysis(docs, allow_smoke)
    rows = [apply_alarm_rule(r) for d in scored for r in d["rows"]]
    faults = [f for d in scored for f in d["faults"]]
    formulation = [d["formulation"] for d in scored if d["formulation"]]
    strata = {i.id: i.family for i in P.INSTANCES}
    ran = sorted({r["arm_id"] for r in rows if r["status"] != "not_run"})
    scoring_ids = [i.id for i in P.INSTANCES if i.split == "scoring" and any(r["instance_id"] == i.id for r in rows)]
    analysis: dict = {"mode": mode, "label": "PIPELINE SMOKE: not evidence" if mode == SMOKE else "scored",
                      **digests, "arms_run": ran,
                      "arms_not_run": sorted({r["arm_id"] for r in rows if r["status"] == "not_run"}),
                      "cells": dict(Counter(r["arm_id"] for r in rows)), "scoring_instances": len(scoring_ids)}
    analysis["alarm_rule"] = ALARM_RULE
    analysis["comparisons"] = comparisons(rows, ran, scoring_ids, strata)
    analysis["sensitivity_scorer_success"] = {
        "why": "the frozen scorer counts an alarm cell as a success; this repeats every comparison that way",
        "comparisons": comparisons([dict(r, success=r["success_scorer"]) for r in rows], ran, scoring_ids, strata)}
    analysis["per_class"] = per_class_table(rows)
    analysis["safety"] = safety(rows, faults)
    analysis["verification"] = verification_distribution(rows)
    analysis["formulation"] = [{k: v for k, v in f.items() if k != "rows"} for f in formulation]
    analysis["alarms"] = [{k: r[k] for k in ("arm_id", "instance_id", "seed", "gap", "max_violation")}
                          for r in rows if r.get("alarm")]
    analysis["failures"] = failures(rows)
    analysis["tuning"] = per_class_table([apply_alarm_rule(r) for d in docs if d["shard"]["split"] == "tuning"
                                            for r in d["rows"]])
    C.write_json_atomic(out / "analysis.json", C.sealed(C.jsonable(analysis)))
    return analysis


def per_class_table(rows: list[dict]) -> dict:
    table: dict = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        if r["status"] != "not_run":
            by[(r["problem_class"], r["arm_id"])].append(r)
    for (klass, arm), rs in sorted(by.items()):
        gaps = [C.num(r["gap_norm"]) for r in rs]
        finite = [g for g in gaps if math.isfinite(g)]
        table[klass][arm] = {
            "cells": len(rs), "instances": len({r["instance_id"] for r in rs}),
            "success": sum(bool(r["success"]) for r in rs), "feasible": sum(bool(r["feasible"]) for r in rs),
            "alarms": sum(bool(r.get("alarm")) for r in rs),
            "median_gap_norm": statistics.median(finite) if finite else math.inf,
            "max_gap_norm": max(gaps) if gaps else math.nan, "infeasible_cells": len(gaps) - len(finite),
            "median_evaluations": statistics.median(r["n_evaluations"] for r in rs),
            "budget_exhausted": sum(1 for r in rs if str(r["stop"]).startswith("budget_exhausted")),
            "median_wall_s": statistics.median(float(r["wall_clock_s"]) for r in rs),
            "tokens": sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in rs),
            "statuses": dict(Counter(r["status"] for r in rs))}
    return table


def _upper95(n: int) -> float:
    return 1.0 - 0.05 ** (1.0 / n) if n else math.nan


def safety(rows: list[dict], faults: list[dict]) -> dict:
    run = [r for r in rows if r["status"] != "not_run"]
    by_arm = defaultdict(list)
    for r in run:
        by_arm[r["arm_id"]].append(r)
    out: dict = {"budget_enforcement": {}, "experiment_replay": {}, "hard_constraint_preservation": {},
                 "scored_false_pass": {}, "token_cost": {}}
    for arm, rs in sorted(by_arm.items()):
        assessed = [r for r in rs if r["within_budget"] is not None]
        within = sum(bool(r["within_budget"]) and bool(r["budget_typed"]) for r in assessed)
        out["budget_enforcement"][arm] = {"cells": len(rs), "assessed": len(assessed), "pass": within,
                                          "fraction": within / len(assessed) if assessed else math.nan,
                                          "independent_equals_reported": sum(
                                              r["n_evaluations"] == r["n_evaluations_independent"] for r in rs),
                                          "max_used_over_budget": max(r["n_evaluations_independent"] /
                                                                      r["budget_evaluations"] for r in rs),
                                          "budget_exhausted": sum(str(r["stop"]).startswith("budget_exhausted")
                                                                  for r in rs)}
        eq = sum(bool((r.get("replay") or {}).get("equal")) for r in rs)
        out["experiment_replay"][arm] = {"cells": len(rs), "equal": eq, "fraction": eq / len(rs),
                                         "kind": (rs[0].get("replay") or {}).get("kind")}
        hc = [r["hard_constraints"] for r in rs if r.get("hard_constraints")]
        if hc:
            items = sum(h["items"] * h["formulations"] for h in hc)
            out["hard_constraint_preservation"][arm] = {"items": items, "preserved": sum(h["preserved"] for h in hc),
                                                        "fraction": sum(h["preserved"] for h in hc) / items}
        verified = [r for r in rs if (r.get("verification") or {}).get("verdict") == "passed"]
        out["scored_false_pass"][arm] = {
            "scored_runs": len(rs), "guard_passed": len(verified),
            "false_pass": sum(1 for r in verified if not r["feasible"]),
            "passed_but_not_within_tolerance": sum(1 for r in verified if r["feasible"] and not r["success"]),
            "upper95": _upper95(len(rs))}
        out["token_cost"][arm] = {"tokens": sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in rs),
                                  "usd": sum(float(r["cost_usd"]) for r in rs)}
    fv: dict = {"artifacts": 0, "by_path": {}}
    for path in sorted({f["path"] for f in faults}):
        fs = [f for f in faults if f["path"] == path]
        informative = [f for f in fs if f["honest_verdict"] == "passed"]
        fv["by_path"][path] = {
            "graded": len(fs), "false_pass": sum(f["verdict"] == "passed" for f in fs), "upper95": _upper95(len(fs)),
            "reason_named": sum(f["reason_named"] for f in fs), "honest_host_passed": len(informative),
            "false_pass_on_informative": sum(f["verdict"] == "passed" for f in informative),
            "reason_named_on_informative": sum(f["reason_named"] for f in informative),
            "upper95_informative": _upper95(len(informative)),
            "by_fault": {fid: {"graded": sum(1 for f in fs if f["fault"] == fid),
                               "false_pass": sum(1 for f in fs if f["fault"] == fid and f["verdict"] == "passed"),
                               "reason_named": sum(1 for f in fs if f["fault"] == fid and f["reason_named"])}
                         for fid in F.FAULT_IDS},
            "by_host": {h: {"graded": sum(1 for f in fs if f["host"] == h),
                            "honest_passed": sum(1 for f in fs if f["host"] == h and f["honest_verdict"] == "passed"),
                            "false_pass": sum(1 for f in fs if f["host"] == h and f["verdict"] == "passed")}
                        for h in sorted({f["host"] for f in fs})}}
        fv["artifacts"] = max(fv["artifacts"], len(fs))
    out["false_verification_pass_faults"] = fv
    return out


def verification_distribution(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in sorted({r["arm_id"] for r in rows if r["status"] != "not_run"}):
        rs = [r for r in rows if r["arm_id"] == arm and r["status"] != "not_run" and r.get("verification")]
        statuses = defaultdict(Counter)
        for r in rs:
            for k, v in (r["verification"].get("statuses") or {}).items():
                statuses[k][v] += 1
        out[arm] = {"run_status": dict(Counter(str(r["verification"].get("run_status")) for r in rs)),
                    "verdict": dict(Counter(str(r["verification"].get("verdict")) for r in rs)),
                    "statuses": {k: dict(v) for k, v in statuses.items()},
                    "by_class": {k: dict(Counter(str(r["verification"].get("run_status")) for r in rs
                                                 if r["problem_class"] == k))
                                 for k in sorted({r["problem_class"] for r in rs})}}
    return out


def failures(rows: list[dict]) -> list[dict]:
    """Per arm and instance, every scored instance that did not succeed on all its seeds: a negative is a result."""
    agg = defaultdict(list)
    for r in rows:
        if r["status"] != "not_run":
            agg[(r["arm_id"], r["instance_id"])].append(r)
    out = []
    for (arm, inst), rs in sorted(agg.items()):
        succ = sum(bool(r["success"]) for r in rs)
        if succ < len(rs):
            gaps = [C.num(r["gap_norm"]) for r in rs]
            out.append({"arm": arm, "instance": inst, "success": succ, "cells": len(rs),
                        "statuses": dict(Counter(r["status"] for r in rs)),
                        "stops": dict(Counter(str(r["stop"]) for r in rs)),
                        "median_gap_norm": statistics.median(gaps), "worst_gap_norm": max(gaps)})
    return out


# ----------------------------------------------------------------------------------------------------- report
def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{digits}g}"


def render_tables(an: dict) -> str:
    """Markdown tables of the analysis; the report quotes them."""
    lines = [f"<!-- generated by bench_e14_end_to_end.py report from analysis.json {an['content_digest']} -->", ""]
    if an["mode"] == SMOKE:
        lines += ["**PIPELINE SMOKE: every number below comes from mock providers and is not evidence.**", ""]
    lines += ["### Per class", "", "| class | arm | cells | success | feasible | alarms | median gap | "
              "budget exhausted | median evaluations | median wall s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for klass, arms in an["per_class"].items():
        for arm, t in arms.items():
            lines.append(f"| {klass} | {arm} | {t['cells']} | {t['success']} | {t['feasible']} | {t['alarms']} | "
                         f"{fmt(C.num(t['median_gap_norm']))} | {t['budget_exhausted']} | "
                         f"{fmt(C.num(t['median_evaluations']))} | {fmt(C.num(t['median_wall_s']))} |")
    lines += ["", "### Comparisons", "", "| scope | treatment vs control | family | endpoint | effect | 95% CI | "
              "informative | p | p Holm (m) | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in an["comparisons"]:
        name = f"{c['treatment']} vs {c['control']}"
        if "not_run" in c:
            lines.append(f"| {c['scope']} | {name} | {c['family']} | all | - | - | - | - | - | not run: "
                         f"{c['not_run']} |")
            continue
        for e in ENDPOINTS + (SECONDARY_ENDPOINT,):
            s = c[e]
            ci = s.get("ci_effect", s["ci"])
            label = {"success": "success-rate difference", "evaluations": "ERT ratio t/c",
                     "wall_clock": "ERT-time ratio t/c", "gap": "log10 gap ratio t/c"}[e]
            lines.append(f"| {c['scope']} | {name} | {c['family']} | {e} ({label}) | {fmt(C.num(s['effect']))} | "
                         f"[{fmt(C.num(ci[0]))}, {fmt(C.num(ci[1]))}] | {s['informative']} | "
                         f"{fmt(C.num(s['p']))} | {fmt(C.num(s.get('p_holm')))} ({s.get('holm_family_size', '-')}) | "
                         f"{s.get('verdict', '-')} |")
    lines += ["", "### Sensitivity: success as the frozen scorer counts it (alarm cells as successes)", "",
              "| scope | treatment vs control | effect | 95% CI | informative | p | p Holm (m) | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for c in an["sensitivity_scorer_success"]["comparisons"]:
        if "not_run" not in c:
            s_ = c["success"]
            lines.append(f"| {c['scope']} | {c['treatment']} vs {c['control']} | {fmt(C.num(s_['effect']))} | "
                         f"[{fmt(C.num(s_['ci'][0]))}, {fmt(C.num(s_['ci'][1]))}] | {s_['informative']} | "
                         f"{fmt(C.num(s_['p']))} | {fmt(C.num(s_.get('p_holm')))} "
                         f"({s_.get('holm_family_size', '-')}) | {s_.get('verdict', '-')} |")
    sf = an["safety"]
    lines += ["", "### Safety metrics", "", "| arm | budget enforcement | independent = reported | replay EQUAL | "
              "hard constraints preserved | guard passed | false pass (scored) | passed, outside tolerance | tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm, b in sf["budget_enforcement"].items():
        rp, fp = sf["experiment_replay"][arm], sf["scored_false_pass"][arm]
        hc = sf["hard_constraint_preservation"].get(arm)
        lines.append(f"| {arm} | {b['pass']}/{b['cells']} | {b['independent_equals_reported']}/{b['cells']} | "
                     f"{rp['equal']}/{rp['cells']} | {'-' if not hc else str(hc['preserved']) + '/' + str(hc['items'])}"
                     f" | {fp['guard_passed']} | {fp['false_pass']} | {fp['passed_but_not_within_tolerance']} | "
                     f"{sf['token_cost'][arm]['tokens']} |")
    fv = sf["false_verification_pass_faults"]["by_path"]
    lines += ["", "### Fault injection", "", "| path | graded | false pass | 95% upper bound | reason named | "
              "honest host passed | false pass on those | reason named on those | 95% upper bound on those |",
              "|---|---|---|---|---|---|---|---|---|"]
    for path, v in fv.items():
        lines.append(f"| {path} | {v['graded']} | {v['false_pass']} | {fmt(C.num(v['upper95']))} | "
                     f"{v['reason_named']} | {v['honest_host_passed']} | {v['false_pass_on_informative']} | "
                     f"{v['reason_named_on_informative']} | {fmt(C.num(v['upper95_informative']))} |")
    lines += ["", "### Verification status (rule arms)", "", "| arm | run status | guard verdict | "
              "optimality_certified |", "|---|---|---|---|"]
    for arm, v in an["verification"].items():
        lines.append(f"| {arm} | {v['run_status']} | {v['verdict']} | "
                     f"{v['statuses'].get('optimality_certified', {})} |")
    lines += ["", "### Instances not solved on every seed", "", "| arm | instance | success | statuses | stops | "
              "median gap | worst gap |", "|---|---|---|---|---|---|---|"]
    for f in an["failures"]:
        lines.append(f"| {f['arm']} | {f['instance']} | {f['success']}/{f['cells']} | {f['statuses']} | {f['stops']} | "
                     f"{fmt(C.num(f['median_gap_norm']))} | {fmt(C.num(f['worst_gap_norm']))} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- post-hoc, descriptive only
def evaluation_split(out: Path) -> list[dict]:
    """Per instance, the median E-rule evaluations spent in the planner loop and in the final verification.

    Read from the saved run directories of the scored E-rule shards (they stay on the server). Descriptive only:
    added after the scored run to explain the E-rule evaluation counts, and not part of any test."""
    split: dict = defaultdict(lambda: defaultdict(list))
    for trace_path in sorted(Path(out).glob("runs/cells.scoring.E-rule.*/E-rule/*/seed-*/trace.json")):
        trace = json.loads(trace_path.read_text())
        s = split[trace_path.parts[-3]]
        s["loop"].append(sum(it["evaluations"] for it in trace["iterations"]))
        s["verification"].append(sum(v["evaluations"] for v in trace.get("verifications", [])))
        s["total"].append(trace["usage"]["evaluations"])
        s["solves"].append(sum(1 for it in trace["iterations"] if it.get("kind") == "solve"))
    return [{"instance": inst, "runs": len(s["loop"]), "median_loop": statistics.median(s["loop"]),
             "median_verification": statistics.median(s["verification"]),
             "median_solves": statistics.median(s["solves"]), "median_total": statistics.median(s["total"])}
            for inst, s in sorted(split.items())]


if __name__ == "__main__":
    import sys
    for line in evaluation_split(Path(sys.argv[1])):
        print(json.dumps(line))


