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
#: added after the v3 freeze: the LLM planner against the rule planner, a Holm family of its own so that no
#: pre-registered family changes size
EXPLORATORY = (("E-LLM", "E-rule"),)
FAMILY_PRIMARY, FAMILY_SECONDARY = "primary", "secondary (E-rule is not arm E)"
FAMILY_EXPLORATORY = "exploratory (added after the freeze)"


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
        for treat, ctrl in PRIMARY + SECONDARY + EXPLORATORY:
            c = compare(rows, treat, ctrl, ids, strata) if treat in ran and ctrl in ran else {
                "treatment": treat, "control": ctrl, "not_run": "an arm of this comparison was not run"}
            c["scope"] = scope
            c["family"] = (FAMILY_PRIMARY if (treat, ctrl) in PRIMARY else
                           FAMILY_SECONDARY if (treat, ctrl) in SECONDARY else FAMILY_EXPLORATORY)
            comps.append(c)
    tested = [n for n, c in enumerate(comps) if "not_run" not in c]
    over = [n for n in tested if comps[n]["scope"] == "overall"]
    prim = [n for n in over if comps[n]["family"] == FAMILY_PRIMARY]
    _holm_into(comps, [(n, e) for n in prim for e in ENDPOINTS],
               extra_untested=len(PRIMARY) * len(ENDPOINTS) - len(prim) * len(ENDPOINTS))
    sec = [n for n in over if comps[n]["family"] == FAMILY_SECONDARY]
    _holm_into(comps, [(n, e) for n in sec for e in ENDPOINTS])
    _holm_into(comps, [(n, SECONDARY_ENDPOINT) for n in over if comps[n]["family"] != FAMILY_EXPLORATORY])
    for group in M.ANALYSIS_GROUPS:
        g = [n for n in tested if comps[n]["scope"] == group and comps[n]["family"] != FAMILY_EXPLORATORY]
        _holm_into(comps, [(n, e) for n in g for e in ENDPOINTS])
        _holm_into(comps, [(n, SECONDARY_ENDPOINT) for n in g])
    for scope in scopes:
        x = [n for n in tested if comps[n]["scope"] == scope and comps[n]["family"] == FAMILY_EXPLORATORY]
        _holm_into(comps, [(n, e) for n in x for e in ENDPOINTS + (SECONDARY_ENDPOINT,)])
    return comps


def analyze(out: Path, allow_smoke: bool = False) -> dict:
    docs = load_shards(out)
    scored, digests, mode = _rows_for_analysis(docs, allow_smoke)
    latency = load_llm_latency(out)
    rows = [with_spec_wall_clock(apply_alarm_rule(r), latency) for d in scored for r in d["rows"]]
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
    if any(r.get("alarm") and r["success_scorer"] for r in rows):  # only a v1 scorer counted an alarm a success
        analysis["sensitivity_scorer_success"] = {
            "why": "the frozen scorer counts an alarm cell as a success; this repeats every comparison that way",
            "comparisons": comparisons([dict(r, success=r["success_scorer"]) for r in rows], ran, scoring_ids,
                                       strata)}
    analysis["per_class"] = per_class_table(rows)
    analysis["safety"] = safety(rows, faults)
    analysis["verification"] = verification_distribution(rows)
    analysis["formulation"] = [{k: v for k, v in f.items() if k != "rows"} for f in formulation]
    form_rows = [r for f in formulation for r in f.get("rows", [])]
    analysis["formulation_breakdown"] = formulation_breakdown(form_rows) if form_rows else None
    analysis["ir_schema_validity"] = ir_schema_validity(rows, form_rows)
    analysis["llm_accounting"] = llm_accounting(rows, form_rows)
    analysis["wall_clock_rule"] = WALL_CLOCK_RULE
    analysis["llm_latency_source"] = "llm_latency.jsonl" if latency else None
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


# ------------------------------------------------------------------- v3 additions, outside the frozen analysis
# Everything in this section was added after the v3 freeze. None of it changes a row; it reads the rows, and the
# cassettes through llm_latency.jsonl.

WALL_CLOCK_RULE = ("spec: wall clock includes LLM latency and compilation. An E-LLM row's wall_clock_s times the "
                   "runtime only, so the latency of its compile calls (from the cell's cassette, llm_latency.jsonl) "
                   "is added; the row's own value is kept as wall_clock_s_row. A cell resumed from its cassette after "
                   "a stop served the recorded calls without waiting, so its wall clock under-states by their "
                   "latency; such cells are counted in llm_accounting.resumed_cells.")


def llm_latency_records(out: Path) -> list[dict]:
    """Per provider cell, the recorded latency of every call in its cassette, split into compile calls (the
    compiler's response schema) and the rest. Reads ``runs/`` (the cassettes stay on the server)."""
    from orbitopt.agent.compiler.schema import COMPILER_OUTPUT_SCHEMA
    compiler = C.canonical(COMPILER_OUTPUT_SCHEMA)
    out_rows = []
    for cas in sorted(Path(out).glob("runs/cells.scoring.*/*/*/seed-*.cassette.jsonl")):
        arm, inst, seed = cas.parts[-3], cas.parts[-2], int(cas.name.split(".")[0].removeprefix("seed-"))
        compile_s = other_s = 0.0
        calls = 0
        models: Counter = Counter()
        for line in cas.read_text().splitlines()[1:]:
            rec = json.loads(line)
            lat = float(rec["response"].get("latency_s") or 0.0)
            calls += 1
            models[str(rec["response"].get("model"))] += 1
            if C.canonical(rec["request"].get("response_schema")) == compiler:
                compile_s += lat
            else:
                other_s += lat
        out_rows.append({"arm_id": arm, "instance_id": inst, "seed": seed, "calls": calls,
                         "compile_latency_s": compile_s, "other_latency_s": other_s, "models": dict(models)})
    return out_rows


def load_llm_latency(out: Path) -> dict:
    path = Path(out) / "llm_latency.jsonl"
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["arm_id"], r["instance_id"], r["seed"]): r for r in recs}


def with_spec_wall_clock(row: dict, latency: dict) -> dict:
    rec = latency.get((row["arm_id"], row["instance_id"], row["seed"]))
    if rec is None or row["status"] == "not_run":
        return row
    out = dict(row, wall_clock_s_row=row["wall_clock_s"],
               llm_latency_s=rec["compile_latency_s"] + rec["other_latency_s"])
    if row["arm_id"] == "E-LLM":
        out["wall_clock_s"] = float(row["wall_clock_s"]) + rec["compile_latency_s"]
    live = {"C": float(row["wall_clock_s"]), "D": float(row.get("llm_wall_clock_s") or 0.0),
            "E-LLM": float(row["wall_clock_s"])}.get(row["arm_id"])
    timed = {"C": out["llm_latency_s"], "D": rec["compile_latency_s"], "E-LLM": rec["other_latency_s"]}.get(
        row["arm_id"])
    out["resumed"] = live is not None and timed is not None and live + 0.05 < timed
    return out


#: Formulation cases whose expected outcome the live harness cannot produce from the prompt alone, classified from
#: each case's notes BEFORE any v3 formulation outcome was read. They stay in the frozen metric; the breakdown also
#: reports the metric without them.
REPLY_DETERMINED = {
    "FC-044": "compile_failed only because the scripted replies are malformed JSON twice",
    "FC-045": "compile_failed only because the scripted replies try unsafe expressions",
    "FC-046": "compile_failed only because every scripted reply drops a constraint",
    "FC-047": "compile_failed only because the scripted replies are IR-invalid three times",
    "FC-048": "compile_failed only because the scripted reply relaxes a constraint silently",
    "FC-031": "a revision: the scripted harness supplies the base problem; compile_statement sends the prompt only",
    "FC-041": "a revision of a base problem the live harness does not send",
    "FC-042": "a revision of a base problem the live harness does not send",
    "FC-043": "a revision of a base problem and an attachment the live harness does not send",
}


def _case_kind(row: dict) -> str:
    return {"compile": "supported", "unsupported": "unsupported", "needs_clarification": "ambiguous"}.get(
        row["expected_outcome"], row["expected_outcome"])


def _accuracy(rows: list[dict], strata_of) -> dict:
    per_case: dict = defaultdict(list)
    for r in rows:
        per_case[r["case_id"]].append(bool(r["correct"]))
    means = {c: sum(v) / len(v) for c, v in per_case.items()}
    strata = {c: strata_of(c) for c in means}
    lo, hi = M.cluster_bootstrap_ci(means, strata) if means else (math.nan, math.nan)
    n = len(rows)
    return {"compiles": n, "cases": len(means), "correct": sum(bool(r["correct"]) for r in rows),
            "accuracy": sum(bool(r["correct"]) for r in rows) / n if n else math.nan, "ci": [lo, hi],
            "ci_rule": "cluster bootstrap over cases (seeds of a case are one cluster), strata = language x "
                       "expected outcome", "meets_0.95": None if not n else (lo >= 0.95)}


def formulation_breakdown(rows: list[dict]) -> dict:
    """Formulation accuracy overall and by case type, with case-clustered CIs, and the outcome confusion."""
    kinds = {r["case_id"]: (r.get("language"), r["expected_outcome"]) for r in rows}
    strata_of = lambda c: "/".join(map(str, kinds[c]))  # noqa: E731
    out: dict = {"overall": _accuracy(rows, strata_of)}
    out["by_language"] = {lang: _accuracy([r for r in rows if r.get("language") == lang], strata_of)
                          for lang in sorted({str(r.get("language")) for r in rows})}
    out["by_expected_outcome"] = {e: _accuracy([r for r in rows if r["expected_outcome"] == e], strata_of)
                                  for e in sorted({r["expected_outcome"] for r in rows})}
    out["by_kind"] = {k: _accuracy([r for r in rows if _case_kind(r) == k], strata_of)
                      for k in sorted({_case_kind(r) for r in rows})}
    out["by_language_and_kind"] = {f"{lang}/{k}": _accuracy(
        [r for r in rows if str(r.get("language")) == lang and _case_kind(r) == k], strata_of)
        for lang in sorted({str(r.get("language")) for r in rows}) for k in sorted({_case_kind(r) for r in rows})
        if any(str(r.get("language")) == lang and _case_kind(r) == k for r in rows)}
    prompt_rows = [r for r in rows if r["case_id"] not in REPLY_DETERMINED]
    out["without_reply_determined"] = {**_accuracy(prompt_rows, strata_of),
                                       "excluded": REPLY_DETERMINED, "label": "post hoc sensitivity"}
    out["confusion"] = {f"{e} -> {o}": n for (e, o), n in sorted(Counter(
        (r["expected_outcome"], r["outcome"]) for r in rows).items())}
    compiled = [r for r in rows if r["outcome"] == "compile" and r["equivalent"] is not None]
    out["compiled_with_reference"] = {"compiles": len(compiled),
                                      "equivalent": sum(bool(r["equivalent"]) for r in compiled),
                                      "not_equivalent": sum(r["equivalent"] is False for r in compiled)}
    out["replay"] = {"compiles": len(rows), "equal": sum(bool((r.get("replay") or {}).get("equal")) for r in rows)}
    per_case = defaultdict(list)
    for r in rows:
        per_case[r["case_id"]].append(r)
    out["per_case"] = {c: {"language": rs[0].get("language"), "expected": rs[0]["expected_outcome"],
                           "correct": sum(bool(r["correct"]) for r in rs), "seeds": len(rs),
                           "outcomes": dict(Counter(r["outcome"] for r in rs)),
                           "not_equivalent": sum(r["equivalent"] is False for r in rs),
                           "reasons": sorted({x for r in rs for x in r.get("reasons") or []})[:5]}
                       for c, rs in sorted(per_case.items())}
    return out


def ir_schema_validity(rows: list[dict], form_rows: list[dict]) -> dict:
    """The spec's denominator: (formulation cases + scoring instances) x scoring seeds. Arm D's cell rows carry
    ``schema_valid`` (a budget-cut compile, None there, counts as not valid); an E-LLM row's compile is valid unless
    its outcome is compile_failed or the budget cut it."""
    d_cells = [r for r in rows if r["arm_id"] == "D" and r["status"] != "not_run"]
    e_cells = [r for r in rows if r["arm_id"] == "E-LLM" and r["status"] != "not_run"]
    f_valid = sum(bool(r["schema_valid"]) for r in form_rows)
    d_valid = sum(bool(r.get("schema_valid")) for r in d_cells)
    e_valid = sum(r.get("compile_outcome") not in (None, "compile_failed", "budget_exhausted") for r in e_cells)
    n_d = len(form_rows) + len(d_cells)
    return {"D": {"formulation": [f_valid, len(form_rows)], "cells": [d_valid, len(d_cells)],
                  "valid": f_valid + d_valid, "invocations": n_d,
                  "fraction": (f_valid + d_valid) / n_d if n_d else math.nan},
            "E-LLM": {"valid": e_valid, "invocations": len(e_cells),
                      "fraction": e_valid / len(e_cells) if e_cells else math.nan},
            "note": "added after the freeze: v3's frozen analysis counted the formulation suite only"}


def llm_accounting(rows: list[dict], form_rows: list[dict]) -> dict:
    """Calls, tokens, cache share, list-price USD, models returned, caps hit, and cost per solution, per arm."""
    out: dict = {}
    groups = {arm: [r for r in rows if r["arm_id"] == arm and r["status"] != "not_run"] for arm in C.PROVIDER_ARMS}
    groups["formulation"] = form_rows
    for arm, rs in groups.items():
        if not rs:
            continue
        hit = sum(int(r.get("cache_hit_tokens") or 0) for r in rs)
        miss = sum(int(r.get("cache_miss_tokens") or 0) for r in rs)
        usd = math.fsum(float(r.get("cost_usd") or 0.0) for r in rs)
        succ = sum(bool(r.get("success")) for r in rs) if arm != "formulation" else sum(bool(r["correct"]) for r in rs)
        verified = sum(1 for r in rs if (r.get("verification") or {}).get("verdict") == "passed"
                       and r.get("success"))
        lat = [float(r["llm_latency_s"]) for r in rs if r.get("llm_latency_s") is not None]
        out[arm] = {
            "cells": len(rs), "calls": sum(int(r.get("llm_calls") or 0) for r in rs),
            "max_calls_in_a_cell": max(int(r.get("llm_calls") or 0) for r in rs),
            "tokens_in": sum(int(r.get("tokens_in") or 0) for r in rs),
            "tokens_out": sum(int(r.get("tokens_out") or 0) for r in rs),
            "cache_hit_tokens": hit, "cache_miss_tokens": miss,
            "cache_hit_share": hit / (hit + miss) if hit + miss else math.nan,
            "usd_list_price": usd, "usd_per_cell": usd / len(rs),
            "models": dict(Counter(m for r in rs for m in r.get("llm_models") or [])),
            "llm_cut": dict(Counter(str(r.get("llm_cut")) for r in rs if r.get("llm_cut"))),
            "over_token_budget": sum(bool(r.get("llm_over_budget")) for r in rs),
            "errors": sum(r.get("status") == "error" for r in rs),
            "successes": succ, "usd_per_success": usd / succ if succ else math.inf,
            "verified_successes": verified,
            "usd_per_verified_success": (usd / verified if verified else math.inf) if arm == "E-LLM" else None,
            "median_llm_latency_s": statistics.median(lat) if lat else None,
            "total_llm_latency_s": math.fsum(lat) if lat else None,
            "median_wall_clock_s": statistics.median(float(r["wall_clock_s"]) for r in rs),
            "resumed_cells": sum(bool(r.get("resumed")) for r in rs),
        }
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
    sensitivity = an.get("sensitivity_scorer_success", {"comparisons": []})["comparisons"]
    if sensitivity:
        lines += ["", "### Sensitivity: success as the frozen scorer counts it (alarm cells as successes)", "",
                  "| scope | treatment vs control | effect | 95% CI | informative | p | p Holm (m) | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
    for c in sensitivity:
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
    lines += ["", "### Verification status", "", "| arm | run status | guard verdict | "
              "optimality_certified |", "|---|---|---|---|"]
    for arm, v in an["verification"].items():
        lines.append(f"| {arm} | {v['run_status']} | {v['verdict']} | "
                     f"{v['statuses'].get('optimality_certified', {})} |")
    acc = an.get("llm_accounting") or {}
    if acc:
        lines += ["", "### LLM accounting (list price)", "", "| arm | cells | calls | max calls/cell | tokens in | "
                  "tokens out | cache hit share | USD | USD/cell | successes | USD/success | verified successes | "
                  "USD/verified success | models | caps hit | errors | median LLM s | median wall s | resumed |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, a in acc.items():
            lines.append(f"| {arm} | {a['cells']} | {a['calls']} | {a['max_calls_in_a_cell']} | {a['tokens_in']} | "
                         f"{a['tokens_out']} | {fmt(C.num(a['cache_hit_share']))} | {a['usd_list_price']:.4f} | "
                         f"{fmt(C.num(a['usd_per_cell']))} | {a['successes']} | {fmt(C.num(a['usd_per_success']))} | "
                         f"{a['verified_successes']} | {fmt(a['usd_per_verified_success'])} | {a['models']} | "
                         f"{a['llm_cut']} | {a['errors']} | {fmt(a['median_llm_latency_s'])} | "
                         f"{fmt(C.num(a['median_wall_clock_s']))} | {a['resumed_cells']} |")
    sv = an.get("ir_schema_validity")
    if sv:
        lines += ["", "### IR schema validity", "", "| arm | valid | invocations | fraction |", "|---|---|---|---|",
                  f"| D (formulation {sv['D']['formulation'][0]}/{sv['D']['formulation'][1]}, cells "
                  f"{sv['D']['cells'][0]}/{sv['D']['cells'][1]}) | {sv['D']['valid']} | {sv['D']['invocations']} | "
                  f"{fmt(C.num(sv['D']['fraction']))} |",
                  f"| E-LLM | {sv['E-LLM']['valid']} | {sv['E-LLM']['invocations']} | "
                  f"{fmt(C.num(sv['E-LLM']['fraction']))} |"]
    fb = an.get("formulation_breakdown")
    if fb:
        lines += ["", "### Formulation accuracy", "", "| slice | compiles | cases | correct | accuracy | 95% CI |",
                  "|---|---|---|---|---|---|"]
        slices = [("overall", fb["overall"])] + [(f"language {k}", v) for k, v in fb["by_language"].items()] + [
            (f"kind {k}", v) for k, v in fb["by_kind"].items()] + [
            (f"expected {k}", v) for k, v in fb["by_expected_outcome"].items()] + [
            (k, v) for k, v in fb["by_language_and_kind"].items()] + [
            ("without reply-determined cases (post hoc)", fb["without_reply_determined"])]
        for name, s in slices:
            lines.append(f"| {name} | {s['compiles']} | {s['cases']} | {s['correct']} | {fmt(C.num(s['accuracy']))} "
                         f"| [{fmt(C.num(s['ci'][0]))}, {fmt(C.num(s['ci'][1]))}] |")
        lines += ["", "| expected -> outcome | compiles |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in fb["confusion"].items()]
        lines += ["", "| case | language | expected | correct | outcomes | not equivalent | reasons |",
                  "|---|---|---|---|---|---|---|"]
        for cid, p in fb["per_case"].items():
            if p["correct"] < p["seeds"]:
                lines.append(f"| {cid} | {p['language']} | {p['expected']} | {p['correct']}/{p['seeds']} | "
                             f"{p['outcomes']} | {p['not_equivalent']} | {'; '.join(p['reasons'])[:160]} |")
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
    if sys.argv[1] == "--llm-latency":  # python e14_analysis.py --llm-latency OUT > OUT/llm_latency.jsonl
        for line in llm_latency_records(Path(sys.argv[2])):
            print(json.dumps(line, sort_keys=True))
        sys.exit(0)
    for line in evaluation_split(Path(sys.argv[1])):
        print(json.dumps(line))


