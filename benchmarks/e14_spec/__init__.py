"""E14-10 benchmark specification: the pre-registered design for judging the LLM compiler and agent (epic E14).

This package is DATA and scoring rules, not the runner. ``spec_digest()`` hashes everything a scored run depends on;
docs/benchmarks/e14_benchmark_spec.md quotes it, and the later runner must refuse to score when the digest it loads
differs from the one frozen in ADR optimization-0001.
"""
from __future__ import annotations

from dataclasses import asdict

from . import arms, faults, formulation, metrics, problems

SPEC_VERSION = "e14-10/0.1-proposed"


def spec_document() -> dict:
    """Every pre-registered choice, as JSON-able data (callables excluded)."""
    return {
        "version": SPEC_VERSION,
        "instances": [dict(problems.neutral_form(i), reference=asdict(i.reference), tolerance=asdict(i.tolerance),
                           split=i.split, well_known=i.well_known, statement=i.statement)
                      for i in problems.INSTANCES],
        "external": [asdict(e) for e in problems.EXTERNAL],
        "arms": [asdict(a) for a in arms.ARMS],
        "budget": asdict(arms.BUDGET),
        "fixed_solver": arms.FIXED_SOLVER,
        "rule_table": arms.RULE_TABLE,
        "primary_comparisons": [list(p) for p in arms.PRIMARY_COMPARISONS],
        "metrics": [asdict(m) for m in metrics.METRICS],
        "protocol": {"scoring_seeds": list(metrics.SCORING_SEEDS), "tuning_seeds": list(metrics.TUNING_SEEDS),
                     "alpha": metrics.ALPHA, "confidence": metrics.CONFIDENCE,
                     "bootstrap_resamples": metrics.BOOTSTRAP_RESAMPLES, "bootstrap_seed": metrics.BOOTSTRAP_SEED,
                     "min_clusters_for_claim": metrics.MIN_CLUSTERS_FOR_CLAIM, "mo_max_points": problems.MO_MAX_POINTS,
                     "analysis_groups": metrics.ANALYSIS_GROUPS},
        "faults": [{k: v for k, v in asdict(f).items() if k != "inject"} | {"changes": sorted(f.changes)}
                   for f in faults.FAULTS],
        "formulation_examples": [asdict(c) for c in formulation.EXAMPLES],
    }


def spec_digest() -> str:
    return faults.digest(spec_document())
