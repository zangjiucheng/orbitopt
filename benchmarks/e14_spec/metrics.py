"""E14-10 metrics, statistical protocol constants, and the tests the later runner must use.

Every threshold below is PROPOSED; ADR optimization-0001 (task E14-01) freezes it, and no threshold may change after
the first scored run. The statistics are implemented here, not only described, so the analysis is fixed before any
data exists: the runner imports these functions instead of choosing its own.

Lesson from E11R-01 (docs/benchmarks, tests/data/mission/reference/e11r-01-robust-candidate-selection): five
selected arms re-flown on the SAME 36 hold-out samples were pooled to 179/180 with a tight CI, which only measured
search-seed variation, and none of the arms was significantly better than the champion (Holm p = 1). Here the cluster
is the instance: seeds are nested inside it, every CI resamples instances, and a superiority claim needs the paired
cluster test, after Holm, to agree with the CI.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from math import comb
from typing import Mapping, Sequence

THRESHOLD_STATUS = "proposed; frozen in ADR optimization-0001 by E14-01"

SCORING_SEEDS: tuple[int, ...] = tuple(range(10))
TUNING_SEEDS: tuple[int, ...] = tuple(range(1000, 1005))
ALPHA = 0.05
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1410


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    definition: str
    unit: str
    denominator: str
    arms: tuple[str, ...]
    target: str
    threshold_status: str = THRESHOLD_STATUS


METRICS: tuple[Metric, ...] = (
    Metric("ir_schema_validity", "IR schema validity",
           "Share of compiler outputs that either validate against the IR schema or are an explicit typed rejection "
           "(needs_clarification, unsupported, rejected_unauthorized_change). Free text, partial JSON or a "
           "schema-invalid problem counts as a failure.",
           "fraction", "compiler invocations: (formulation cases + scoring instances) x scoring seeds",
           ("D", "E"), "= 1.00"),
    Metric("formulation_accuracy", "Formulation accuracy",
           "Share of formulation cases whose outcome matches expected_outcome and, for 'compile' cases, whose "
           "compiled problem is equivalent to the reference: same optimum within the instance tolerance AND the "
           "same feasibility verdict as the reference at 1,000 seeded probe points in the bounds.",
           "fraction", "formulation cases (E14-02 suite, >= 30) x scoring seeds", ("D", "E"), ">= 0.95"),
    Metric("hard_constraint_preservation", "Hard-constraint preservation",
           "Share of the reference problem's hard constraints and bounds that are present and unrelaxed in every "
           "formulation the arm solved (initial compile and every agent round), checked at the probe points.",
           "fraction", "hard constraints and bounds of the reference x compiled formulations x scoring seeds",
           ("D", "E"), "= 1.00"),
    Metric("false_verification_pass", "False verification pass",
           "Number of artifacts the verifier marked 'verified' although they are faulty: every injected fault "
           "(faults.FAULTS applied to honest results) plus every scored run whose recomputation with the reference "
           "callables fails. Reported with its denominator and the one-sided 95% Clopper-Pearson upper bound.",
           "count", "fault-injected artifacts (9 faults x scoring instances x arms) + scored runs",
           ("A", "B", "C", "D", "E"), "= 0"),
    Metric("budget_enforcement", "Budget enforcement",
           "Share of cells that stayed within every budget (evaluations, wall clock, tokens, LLM calls) and, when a "
           "budget ran out, stopped with a typed budget_exhausted status instead of overrunning or a silent "
           "truncation.",
           "fraction", "cells: scoring instances x scoring seeds, per arm", ("A", "B", "C", "D", "E"), "= 1.00"),
    Metric("experiment_replay", "Experiment replay",
           "Share of scored cells whose replay from the artifact (recorded seeds, recorded LLM transcripts, no "
           "network) reproduces the per-run row exactly, timing fields excluded.",
           "fraction", "scored cells, per arm", ("A", "B", "C", "D", "E"), "= 1.00"),
    Metric("optimality_gap", "Optimality gap per class",
           "Gap of the best feasible returned point to the reference, oriented so 0 is optimal (minimize: f - f*; "
           "maximize: f* - f; multi-objective: HV* - HV), divided by max(1, abs(f*)); a cell with no feasible point "
           "has gap = inf. Success = feasible and gap within the instance tolerance. Reported per problem class.",
           "dimensionless (normalised gap); success as fraction", "cells of the class, per arm",
           ("A", "B", "C", "D", "E"),
           "A, B, D, E: success = 1.00 on lp, convex_qp, milp (exact solvers exist; a miss is a defect). "
           "Other classes: no absolute target, paired comparison only."),
    Metric("function_evaluations", "Function evaluations vs baseline",
           "Expected running time ERT = (evaluations summed over the instance's seeds) / (successful seeds), inf if "
           "none succeeded; compared per instance as log(ERT_arm / ERT_control).",
           "evaluations (ratio to control)", "scoring instances, per compared pair", ("B", "C", "D", "E"),
           "comparative only: paired Wilcoxon vs control"),
    Metric("wall_clock", "Wall clock including LLM time",
           "Seconds from cell start to the verifier's verdict, including LLM latency, compilation, solving and "
           "verification; LLM time is also reported on its own. Live runs only (replays are not timed).",
           "seconds", "cells, per arm", ("A", "B", "C", "D", "E"), "comparative only: paired Wilcoxon vs control"),
    Metric("token_cost", "Token cost",
           "Input and output tokens reported by the provider per cell, and USD at the price table pinned in the "
           "artifact provenance. Arms A and B are 0 by construction.",
           "tokens and USD", "cells, per arm", ("C", "D", "E"), "report only; no target"),
)

METRIC_IDS: tuple[str, ...] = tuple(m.id for m in METRICS)


# ---------------------------------------------------------------------------------------------------------- scoring
def optimality_gap(reference: float, achieved: float, sense: str, indicator: str = "objective") -> float:
    """Normalised gap, 0 = optimal; positive = worse. ``achieved`` is the objective, or the hypervolume."""
    if indicator == "hypervolume" or sense == "maximize":
        raw = reference - achieved
    else:
        raw = achieved - reference
    return raw / max(1.0, abs(reference))


# ------------------------------------------------------------------------------------------------------- statistics
def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def cluster_sign_flip_p(discordance: Sequence[int]) -> float:
    """Exact two-sided paired test for clustered binary outcomes.

    ``discordance[k]`` = (seeds where only the treatment succeeded) - (seeds where only the control succeeded) on
    instance k. Under H0 each instance's difference is symmetric about 0, so its sign is a fair coin; the p-value is
    P(|sum of randomly signed d_k| >= |sum d_k|), computed exactly. With one pair per instance this is the exact
    McNemar test; with ten seeds per instance it counts the instance, not the seed, as the unit.
    """
    d = [abs(int(v)) for v in discordance if int(v) != 0]
    observed = abs(sum(int(v) for v in discordance))
    dist: dict[int, int] = {0: 1}
    for v in d:
        nxt: dict[int, int] = {}
        for s, c in dist.items():
            nxt[s + v] = nxt.get(s + v, 0) + c
            nxt[s - v] = nxt.get(s - v, 0) + c
        dist = nxt
    total = 2 ** len(d)
    return sum(c for s, c in dist.items() if abs(s) >= observed) / total


def wilcoxon_signed_rank_p(diffs: Sequence[float]) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value over per-instance paired differences.

    Zeros are dropped (Wilcoxon's convention); ties get average ranks, handled exactly by working in doubled ranks.
    ``inf`` differences (one arm never succeeded) rank above every finite one; inf - inf counts as a zero.
    """
    vals = [v for v in diffs if not (v == 0 or (isinstance(v, float) and math.isnan(v)))]
    if not vals:
        return 1.0
    order = sorted(range(len(vals)), key=lambda i: abs(vals[i]))
    ranks2 = [0] * len(vals)  # doubled average ranks, integers
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(vals[order[j + 1]]) == abs(vals[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2  # 2 * mean(i+1 .. j+1)
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks2, vals) if v > 0)
    total = sum(ranks2)
    dist: dict[int, int] = {0: 1}
    for r in ranks2:
        nxt = dict(dist)
        for s, c in dist.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        dist = nxt
    centre_dev = abs(2 * w_plus - total)
    return sum(c for s, c in dist.items() if abs(2 * s - total) >= centre_dev) / 2 ** len(vals)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(names), 0.0, {}
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def cluster_bootstrap_ci(per_instance: Mapping[str, float], strata: Mapping[str, str],
                         resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                         confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Percentile CI for the mean over instances, resampling INSTANCES within family strata.

    ``per_instance`` holds one already seed-aggregated value per instance (e.g. the treatment-minus-control success
    rate over its seeds). Seeds never enter as independent draws, which is what made the E11R-01 pooled CI too
    tight.
    """
    rng = random.Random(seed)
    groups: dict[str, list[float]] = {}
    for inst, value in sorted(per_instance.items()):
        groups.setdefault(strata[inst], []).append(value)
    n = sum(len(g) for g in groups.values())
    stats = []
    for _ in range(resamples):
        total = 0.0
        for values in groups.values():
            total += sum(values[rng.randrange(len(values))] for _ in values)
        stats.append(total / n)
    stats.sort()
    lo = stats[int(math.floor((1.0 - confidence) / 2.0 * resamples))]
    hi = stats[int(math.ceil((1.0 + confidence) / 2.0 * resamples)) - 1]
    return lo, hi


def _min_clusters_for_claim(alpha: float = ALPHA) -> int:
    k = 1
    while cluster_sign_flip_p([1] * k) >= alpha:
        k += 1
    return k


#: fewest informative (discordant) instances for which the exact cluster test can reach p < ALPHA at all (= 6);
#: a class with fewer scoring instances is reported descriptively as "not testable"
MIN_CLUSTERS_FOR_CLAIM = _min_clusters_for_claim()

#: per-class claims are made per ANALYSIS GROUP: no single problem class has enough scoring instances for the exact
#: cluster test, so classes that share a solver technology are pooled. Fixed before any run; a group with fewer than
#: MIN_CLUSTERS_FOR_CLAIM scoring instances is reported descriptively only.
ANALYSIS_GROUPS: dict[str, tuple[str, ...]] = {
    "algebraic": ("lp", "convex_qp", "milp"),
    "continuous_nonconvex": ("nlp", "global"),
    "black_box": ("black_box",),
    "multi_objective": ("multi_objective",),
}

VERDICTS = ("superior", "inferior", "not shown different", "not testable")


def verdict(p_holm: float, ci: tuple[float, float], informative_clusters: int, alpha: float = ALPHA) -> str:
    """The only wording a report may use for a comparison. ``ci`` is oriented so positive = treatment better.

    "superior" needs ALL of: enough informative instances, Holm-adjusted p < alpha, and a CI that excludes 0 on the
    favourable side. A better point estimate alone is "not shown different" (E11R-01: selected arms 36/36 vs the
    champion's 33/36 were Holm p = 1).
    """
    if informative_clusters < MIN_CLUSTERS_FOR_CLAIM:
        return "not testable"
    if p_holm < alpha and ci[0] > 0.0:
        return "superior"
    if p_holm < alpha and ci[1] < 0.0:
        return "inferior"
    return "not shown different"
