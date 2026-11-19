"""E7-05 (task #293): out-of-sample reliability of the joint-chance grading.

Three reading tools, all pure functions over recorded Monte Carlo rows, in
the order the task asks for them (diagnose first, then hold out, then test
across clusters):

* :func:`infeasibility_diagnosis` -- decompose every ``JOINT_CHANCE_VIOLATED``
  row of a joint-mode arm into the constraint that actually bound (fuel /
  divert / landing-probability), its severity, and -- when the arm was flown
  on a grid a FL-305-mode baseline also covered -- whether the *same physical
  world* landed under the laxer grading. This answers the E7-04 question
  (why did 99/168 joint samples reject?) by counting, not by arguing: a
  rejection whose paired baseline landed is a measured constraint cost, a
  rejection within :data:`DEFAULT_BORDERLINE_VIOLATION` of the feasible set
  is planner-search territory, a deep violation is the spec meeting the
  dispersion band, and a row bound only on the landing-probability channel
  is a probability-model problem.
* :func:`stratified_association` / :func:`cluster_bootstrap_phi_ci` -- the
  cluster-aware replacement for the E7-03 headline. The E7-03 report pooled
  three arms sharing one seed/grid into a 409-row 2x2 (Fisher p = 0.010) and
  flagged the pooling itself as optimistic; here the per-cluster tables are
  kept separate (Mantel-Haenszel with a continuity correction, exact Fisher
  per cluster, Fisher-combined over clusters) and the association's CI
  resamples *whole clusters*, which is the honest sampling unit when arms
  share seeds.
* :func:`cluster_bootstrap_rate_ci` -- the same whole-cluster resampling for
  a rate (a reliability-bin success rate, a success count), so hold-out bin
  CIs do not pretend 42 draws inside one seed are four independent points.

The violation reconstruction in :func:`constraint_violations` re-derives the
three normalized violations of
:meth:`~orbitopt.flightlab.recovery.chance.evaluate_joint_chance` from what
the row *recorded* (product ``landing_probability``, ``corridor_miss_m``,
``corridor_fuel_margin_kg``): ``p_divert`` from the declared model, ``p_fuel``
by the same product back-out
:func:`~orbitopt.flightlab.recovery.joint_risk.back_out_component_probabilities`
performs, and the fuel violation through the exact identity
``g_fuel = max(0, phi_inv(p_fuel_min) - phi_inv(p_fuel))`` (the exposure
``sigma`` cancels -- ``p_fuel = phi(margin / sigma)`` inverts to
``margin / sigma = phi_inv(p_fuel)``). A rejected joint row is inconsistent
(``consistent = False``) when the reconstruction disagrees with the recorded
``joint_violation`` beyond tolerance -- the diagnosis reports it, it does not
silently classify on it.

Deterministic throughout: every bootstrap takes an explicit ``seed``.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .chance import JointChanceSpec, phi_inv
from .joint_risk import (back_out_component_probabilities,
                        fisher_exact_two_sided_p, phi_coefficient)

__all__ = [
    "CONSTRAINT_CHANNELS",
    "DEFAULT_BORDERLINE_VIOLATION",
    "DEFAULT_DEEP_VIOLATION",
    "RECONSTRUCTION_TOLERANCE",
    "channels_bound",
    "cluster_bootstrap_phi_ci",
    "cluster_bootstrap_rate_ci",
    "constraint_violations",
    "infeasibility_diagnosis",
    "per_seed_contingencies",
    "stratified_association",
]

#: the three joint constraints, in ``JointVerdict.violations`` order
CONSTRAINT_CHANNELS = ("fuel", "divert", "prob")

#: a rejection counts as near-threshold (planner-search evidence: a script a
#: little better would have passed) at or below this worst normalized
#: violation, and as deep (the spec meeting the dispersion band, not the
#: search) above this one; the band between is "mid". DESIGN ASSUMPTION
#: anchored on the recorded E7-04 distribution (quantiles of the 99
#: rejections: p10 0.067, p50 0.38, p90 1.06): 0.15 sits inside the observed
#: near-threshold mode, 0.5 at the distribution's lower median.
DEFAULT_BORDERLINE_VIOLATION = 0.15
DEFAULT_DEEP_VIOLATION = 0.5

#: relative slack on the reconstructed-vs-recorded worst violation (floats
#: through the JSON round-trip; the identity is exact in exact arithmetic)
RECONSTRUCTION_TOLERANCE = 1e-6

_INV_EPS = 1e-15  # domain guard on phi_inv for back-out probs at the ends


def constraint_violations(record: Mapping[str, Any],
                          spec: JointChanceSpec,
                          ) -> Optional[dict[str, Any]]:
    """Reconstruct the three normalized joint violations of one recorded
    joint-mode row from the fields it carries, or ``None`` when the row is
    not a joint-mode row (no ``landing_probability``, or no consistent
    back-out). ``violations`` is ordered ``(fuel, divert, prob)``;
    ``consistent`` compares the reconstructed worst violation against the
    recorded ``joint_violation`` within :data:`RECONSTRUCTION_TOLERANCE`."""
    if record.get("landing_probability") is None or record.get("corridor_miss_m") is None:
        return None
    comp = back_out_component_probabilities(record, spec)
    if comp is None:
        return None
    p_fuel, p_divert = comp
    p_land = float(record["landing_probability"])
    miss = float(record["corridor_miss_m"])
    z = phi_inv(spec.p_fuel_min)
    g_fuel = max(0.0, z - phi_inv(min(max(p_fuel, _INV_EPS), 1.0 - _INV_EPS)))
    g_div = max(0.0, (miss - spec.divert_budget_m()) / spec.scatter_budget_m)
    g_prob = max(0.0, (spec.min_landing_probability - p_land)
                 / spec.min_landing_probability)
    worst = max(g_fuel, g_div, g_prob)
    recorded = record.get("joint_violation")
    consistent = recorded is not None and math.isclose(
        float(recorded), worst, rel_tol=RECONSTRUCTION_TOLERANCE, abs_tol=1e-9)
    return {"p_fuel": p_fuel, "p_divert": p_divert,
            "violations": (g_fuel, g_div, g_prob), "worst": worst,
            "consistent": bool(consistent)}


def channels_bound(record: Mapping[str, Any], spec: JointChanceSpec,
                   ) -> Optional[Tuple[str, ...]]:
    """Names of the constraint(s) attaining the reconstructed worst violation
    of a joint row (tied constraints all named -- a tie is a fact, not noise
    to break), ``()`` for an accepted row (worst violation zero, nothing
    bound), ``None`` for a non-joint row."""
    v = constraint_violations(record, spec)
    if v is None:
        return None
    w = v["worst"]
    if w <= 0.0:
        return ()
    return tuple(name for name, g in zip(CONSTRAINT_CHANNELS, v["violations"])
                 if g >= w * (1.0 - 10.0 * RECONSTRUCTION_TOLERANCE) and g > 0.0)


def _severity(worst: float, borderline: float, deep: float) -> str:
    return "near" if worst <= borderline else ("deep" if worst > deep else "mid")


def infeasibility_diagnosis(joint_rows: Iterable[Mapping[str, Any]],
                            spec: JointChanceSpec, *,
                            baseline_rows: Optional[Iterable[Mapping[str, Any]]] = None,
                            borderline: float = DEFAULT_BORDERLINE_VIOLATION,
                            deep: float = DEFAULT_DEEP_VIOLATION,
                            ) -> dict[str, Any]:
    """Classify every ``JOINT_CHANCE_VIOLATED`` row of one joint-mode arm.

    ``joint_rows`` are recorded rows carrying ``seed`` / ``idx`` /
    ``corridor_miss_m`` / ``corridor_fuel_margin_kg`` / ``landing_probability``
    / ``joint_violation`` (the E7-04 paired-row format); ``baseline_rows``
    are rows of an arm flown on the SAME ``[seed, idx]`` grid under a laxer
    grading (the FL-305 baseline) -- when given, every rejection is also
    read against its paired world: did that world actually land?

    The returned dict counts: ``binding`` (which constraint bound, ties
    named jointly), ``severity`` (near / mid / deep), the cross-tab
    ``binding_by_severity``, ``probability_binding`` (rows bound on the
    landing-probability channel at all / on it alone), the worst-violation
    quantiles, the reconstruction integrity count, and -- when pairing was
    possible -- ``paired_baseline`` with the landed-anyway counts overall and
    per binding channel (the measured constraint cost). Rows the
    reconstruction calls inconsistent are counted under
    ``inconsistent_rows`` and excluded from the classification."""
    joint_rows = list(joint_rows)
    base = {(int(r["seed"]), int(r["idx"])): r for r in (baseline_rows or [])}
    jv = [r for r in joint_rows if r.get("reason") == "JOINT_CHANCE_VIOLATED"]
    binding: Counter[str] = Counter()
    severity: Counter[str] = Counter()
    cross: Counter[str] = Counter()
    paired_state: Counter[str] = Counter()
    paired_by_binding: Counter[str] = Counter()
    prob_any = prob_only = inconsistent = 0
    misses = [float(r["corridor_miss_m"]) for r in jv
              if r.get("corridor_miss_m") is not None]
    worsts: list[float] = []
    for r in jv:
        v = constraint_violations(r, spec)
        if v is None or not v["consistent"]:
            inconsistent += 1
            continue
        bound = channels_bound(r, spec) or ()
        key = "+".join(bound) if bound else "none"
        sev = _severity(v["worst"], borderline, deep)
        binding[key] += 1
        severity[sev] += 1
        cross[f"{key}/{sev}"] += 1
        worsts.append(v["worst"])
        if "prob" in bound:
            prob_any += 1
            if bound == ("prob",):
                prob_only += 1
        b = base.get((int(r["seed"]), int(r["idx"])))
        if b is not None:
            landed = str(b.get("outcome")) == "landed"
            paired_state["rejected_but_landed" if landed else "rejected_and_failed"] += 1
            paired_by_binding[f"{key}/{'landed' if landed else 'failed'}"] += 1
    n = sum(binding.values())
    worst_q = _quantiles(worsts, (0.1, 0.25, 0.5, 0.75, 0.9))
    out: dict[str, Any] = {
        "n_joint_rows": len(joint_rows),
        "n_rejections": len(jv),
        "inconsistent_rows": inconsistent,
        "classified": n,
        "binding": dict(sorted(binding.items())),
        "severity": {k: severity.get(k, 0) for k in ("near", "mid", "deep")},
        "binding_by_severity": dict(sorted(cross.items())),
        "probability_binding": {"bound_on_prob_any": prob_any,
                                "bound_on_prob_alone": prob_only,
                                "share_of_rejections": prob_any / n if n else None},
        "worst_violation_quantiles": worst_q,
        "corridor_miss_quantiles_m": _quantiles(misses, (0.5, 0.9, 0.99, 1.0)),
        "thresholds": {"borderline": float(borderline), "deep": float(deep)},
    }
    if baseline_rows is not None:
        tot = sum(paired_state.values())
        out["paired_baseline"] = {
            "n_paired": tot,
            "rejected_but_landed": paired_state.get("rejected_but_landed", 0),
            "rejected_and_failed": paired_state.get("rejected_and_failed", 0),
            "landed_anyway_share": (paired_state.get("rejected_but_landed", 0) / tot
                                    if tot else None),
            "by_binding": dict(sorted(paired_by_binding.items())),
            "note": ("a rejection whose paired baseline world LANDED is a "
                     "measured constraint cost (the chance model refused a "
                     "world that would have flown); one whose baseline also "
                     "failed is a rejection that agreed with an independent "
                     "grading"),
        }
    return out


def _quantiles(values: Sequence[float], qs: Sequence[float]) -> dict[str, Optional[float]]:
    """Nearest-rank quantiles (deterministic, no interpolation), keyed by the
    requested q; ``{}``-safe on empty input."""
    xs = sorted(float(v) for v in values)
    out: dict[str, Optional[float]] = {}
    for q in qs:
        out[f"q{q:g}"] = (xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))]
                          if xs else None)
    return out


def per_seed_contingencies(rows: Iterable[Mapping[str, Any]], *,
                           key: str = "seed") -> dict[Any, list[Mapping[str, Any]]]:
    """Group recorded rows into clusters by a row field (default ``seed``),
    a thin helper for callers that build one
    :func:`~orbitopt.flightlab.recovery.joint_risk.contingency_table` per
    cluster -- the channel rules stay in
    :mod:`~orbitopt.flightlab.recovery.joint_risk`."""
    out: dict[Any, list[Mapping[str, Any]]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def stratified_association(tables_by_cluster: Mapping[Any, Mapping[str, Any]], *,
                           trials: int = 4000, seed: int = 20261118,
                           ) -> dict[str, Any]:
    """Cluster-aware test of channel independence over 2x2 contingency
    tables kept separate per cluster (one per seed, or one per (seed, arm)
    cell -- the caller decides what a cluster is).

    Replaces the E7-03 pooled reading. Reports, per cluster: n, phi and the
    exact Fisher p; pooled: the Mantel-Haenszel chi-squared with continuity
    correction (exact for the shared-margin alternative, the standard
    stratified 2x2 test), Fisher's method combining the per-cluster exact
    p-values (a combination of exact tests, not of pooled rows), and a
    bootstrap CI for the pooled phi that resamples WHOLE clusters
    (:func:`cluster_bootstrap_phi_ci`). A degenerate cluster (a zero margin,
    no rows) is listed under ``skipped`` with its reason and excluded, never
    counted as phi = 0.
    """
    per: list[dict[str, Any]] = []
    skipped: dict[Any, str] = {}
    tables: list[dict[str, Any]] = []
    for cluster, t in sorted(tables_by_cluster.items(), key=lambda kv: str(kv[0])):
        cells = (int(t["both"]), int(t["fuel_only"]), int(t["divert_only"]),
                 int(t["neither"]))
        n = sum(cells)
        r = phi_coefficient(*cells)
        if n == 0 or r is None:
            skipped[cluster] = "no rows" if n == 0 else "degenerate margin"
            continue
        per.append({"cluster": cluster, "cells": list(cells), "n": n,
                    "phi": r,
                    "fisher_exact_p": fisher_exact_two_sided_p(*cells)})
        tables.append({"a": cells[0], "b": cells[1], "c": cells[2], "d": cells[3], "n": n})
    mh = _mantel_haenszel(tables)
    fc = _fisher_combined([row["fisher_exact_p"] for row in per])
    return {"method": ("Mantel-Haenszel (continuity corrected) + Fisher-combined "
                       "exact per-cluster p + whole-cluster bootstrap phi; "
                       "clusters are the resampling unit"),
            "n_clusters": len(per), "clusters_skipped": skipped,
            "per_cluster": per, "mantel_haenszel": mh,
            "fisher_combined": fc,
            "phi_cluster_bootstrap_ci95": cluster_bootstrap_phi_ci(
                [{"both": t["a"], "fuel_only": t["b"], "divert_only": t["c"],
                  "neither": t["d"]} for t in tables], trials=trials, seed=seed),
            "pooled_naive_phi": _naive_pooled(tables),
            "note": ("pooled_naive_phi is the E7-03-style pooled reading kept "
                     "for the comparison only; its significance is the MH / "
                     "combined-exact numbers, not the pooled 2x2 Fisher")}


def _mantel_haenszel(tables: Sequence[Mapping[str, int]]) -> Optional[dict[str, Any]]:
    """MH chi-squared (df 1) with continuity correction over 2x2 tables
    keyed a/b/c/d/n; p via the exact df-1 tail ``erfc(sqrt(x/2))`` (the same
    route ``stats.mcnemar_test`` takes for its chi-squared alongside)."""
    if len(tables) < 2:
        return None
    num = 0.0
    den = 0.0
    ok = 0
    for t in tables:
        n = t["n"]
        if n < 2:
            continue
        a, b, c, d = t["a"], t["b"], t["c"], t["d"]
        row1, row0 = a + b, c + d
        col1, col0 = a + c, b + d
        e = row1 * col1 / n
        v = row1 * row0 * col1 * col0 / (n * n * (n - 1.0))
        if v <= 0.0:
            continue
        num += a - e
        den += v
        ok += 1
    if ok < 2 or den <= 0.0:
        return None
    chi2 = (abs(num) - 0.5) ** 2 / den
    return {"statistic": chi2, "p_value": math.erfc(math.sqrt(chi2 / 2.0)),
            "n_tables": ok,
            "note": "continuity correction -0.5 on the pooled numerator; "
                    "df = 1, tail exact via erfc"}


def _fisher_combined(p_values: Sequence[float]) -> Optional[dict[str, Any]]:
    """Fisher's method over exact per-cluster p-values: X^2 = -2 sum ln p ~
    chi-square 2k; the df-2k tail is the Poisson sum e^-x sum_{j<k} x^j / j!
    with x = X^2 / 2 (evaluated by accumulation, no approximation)."""
    ps = [max(float(p), 1e-300) for p in p_values if p is not None]
    k = len(ps)
    if k < 2:
        return None
    x = -sum(math.log(p) for p in ps)  # x = X^2 / 2
    term = 1.0
    acc = 1.0
    for j in range(1, k):
        term = term * x / j
        acc += term
    return {"statistic": 2.0 * x, "p_value": math.exp(-x) * acc, "k": k}


def _naive_pooled(tables: Sequence[Mapping[str, int]]) -> Optional[dict[str, Any]]:
    tot = {k: sum(t[k] for t in tables) for k in ("a", "b", "c", "d")}
    n = tot["a"] + tot["b"] + tot["c"] + tot["d"]
    r = phi_coefficient(tot["a"], tot["b"], tot["c"], tot["d"])
    return {"cells": [tot["a"], tot["b"], tot["c"], tot["d"]], "n": n,
            "phi": r, "fisher_exact_p": fisher_exact_two_sided_p(
                tot["a"], tot["b"], tot["c"], tot["d"]) if n else None,
            "warning": "pools cluster-correlated rows as iid -- optimistic"}


def cluster_bootstrap_phi_ci(tables: Sequence[Mapping[str, int]], *,
                             trials: int = 4000, seed: int = 20261118,
                             level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for the pooled phi under the CLUSTER resample:
    draw ``len(tables)`` tables with replacement (a cluster enters whole or
    not at all), pool their counts, recompute phi. Deterministic on
    ``seed``. ``None`` when fewer than half the resamples have a defined phi
    (the same guard ``joint_risk.bootstrap_phi_ci`` uses)."""
    if not tables:
        return None
    rng = random.Random(seed)
    k = len(tables)
    phis: list[float] = []
    for _ in range(trials):
        tot = [0, 0, 0, 0]
        for _ in range(k):
            t = tables[rng.randrange(k)]
            tot[0] += t["both"]
            tot[1] += t["fuel_only"]
            tot[2] += t["divert_only"]
            tot[3] += t["neither"]
        r = phi_coefficient(*tot)
        if r is not None:
            phis.append(r)
    if len(phis) < trials // 2:
        return None
    phis.sort()
    alpha = (1.0 - level) / 2.0
    lo = phis[max(0, int(alpha * len(phis)))]
    hi = phis[min(len(phis) - 1, int(math.ceil((1.0 - alpha) * len(phis))) - 1)]
    return (lo, hi)


def cluster_bootstrap_rate_ci(counts: Sequence[Tuple[int, int]], *,
                              trials: int = 4000, seed: int = 20261118,
                              level: float = 0.95
                              ) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for a pooled rate under the cluster resample,
    ``counts`` one ``(successes, n)`` pair per cluster (e.g. per seed: the
    42 draws inside a seed share the seed's perturbation stream and are
    correlated, so the seed -- not the draw -- is the exchangeable unit).
    Empty clusters are skipped; ``None`` when every resample divides by
    zero."""
    pairs = [(int(s), int(n)) for s, n in counts if int(n) > 0]
    if not pairs:
        return None
    rng = random.Random(seed)
    k = len(pairs)
    rates: list[float] = []
    for _ in range(trials):
        s = n = 0
        for _ in range(k):
            a, b = pairs[rng.randrange(k)]
            s += a
            n += b
        if n > 0:
            rates.append(s / n)
    if not rates or len(rates) < trials // 2:
        return None
    rates.sort()
    alpha = (1.0 - level) / 2.0
    lo = rates[max(0, int(alpha * len(rates)))]
    hi = rates[min(len(rates) - 1, int(math.ceil((1.0 - alpha) * len(rates))) - 1)]
    return (lo, hi)
