"""E7-04 (task #280) failure-driven optimization: the *paired comparison*
machinery a failure-improvement claim has to clear.

E3-03 put a finger on which failures a recovery mode actually owns
(``initial_state`` ranked first in every arm, Spearman 0.38-0.47;
``CORRIDOR_MISSED`` 5 -> 22 and ``LANDING_LATERAL_SPEED_EXCEEDED``
24 -> 32 as the joint-chance objective regressed them,
docs/benchmarks/e3_03_robust_montecarlo.md) -- and E7-04's acceptance is
the other half of the scientific method: *fly the fix over the identical
perturbation samples* and grade the failure categories it claims to target
against every category it might have made worse, with the multiple
comparisons corrected (the raw p = 0.019 E3-02-vs-FL-305 number did not
survive correction). This module is the deterministic, library-grade
reading of that comparison; the numbers are assembled from two lists of
paired Monte Carlo rows (each row a dict with ``seed``, ``idx``,
``outcome``, ``reason`` -- exactly the ``paired_rows`` list every
``run_monte_carlo`` report carries, so a recorded E3-03 arm can be
compared against a newly flown arm without re-flying it).

The comparison is deliberately per-category, not per-outcome: "the landed
rate went up" is not the claim ("target failure category counts fall
*without other categories significantly rising*, or say what was traded").
:func:`category_pair_table` runs a paired exact McNemar test per reason
code (success side = "this sample's governing reason is not this code"),
:func:`holm_family` corrects the whole family with
:func:`orbitopt.flightlab.recovery.stats.holm_adjust` step-down, and
:func:`sensitivity_shift` reports how the initial-state sensitivity rank
moved between the two arms' ``source_sensitivity`` dicts. Every entry
point is a pure function of its inputs (no RNG, no I/O).

    from orbitopt.flightlab.recovery.failure_driven import category_pair_table
    category_pair_table(new_rows, base_rows,
                        codes=["CORRIDOR_MISSED",
                               "LANDING_LATERAL_SPEED_EXCEEDED"])
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from orbitopt.flightlab.recovery.stats import holm_adjust, mcnemar_test

__all__ = [
    "category_pair_table",
    "holm_family",
    "index_paired_rows",
    "sensitivity_shift",
    "verdict_summary",
]

Row = Mapping[str, Any]


def index_paired_rows(rows: Sequence[Row]) -> dict[tuple[int, int], Row]:
    """Index paired Monte Carlo rows by their perturbation coordinate
    ``(seed, idx)`` -- the coordinate the draw stream is keyed on, so two
    arms' rows with the same coordinate saw the *same* physical world.
    Duplicate coordinates are an authoring error (one draw, one row) and
    raise."""
    out: dict[tuple[int, int], Row] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["idx"]))
        if key in out:
            raise ValueError(f"duplicate paired row at seed={key[0]} idx={key[1]}")
        out[key] = row
    return out


def category_pair_table(new_rows: Sequence[Row], base_rows: Sequence[Row], *,
                        codes: Sequence[str] | None = None,
                        alpha: float = 0.05) -> dict[str, Any]:
    """Paired per-reason-code McNemar table of a candidate arm against a
    baseline arm (E7-04's acceptance instrument).

    Rows are matched by ``(seed, idx)`` -- only coordinates present in both
    arms are compared (the intersection is reported, and an asymmetric one
    is surfaced loudly, because a silent subset would move the family the
    correction is charged for). For each governing reason ``code`` the
    paired binary is "this sample did not fail with ``code``": ``b`` counts
    samples the *new* arm spared from the category but the baseline lost to
    it, ``c`` the reverse; the p-value is the same exact binomial as
    :func:`orbitopt.flightlab.recovery.stats.mcnemar_test`. ``codes``
    defaults to every code governing a failure in either arm (LANDED never
    enters the family -- the landed/failed pair is the outcome comparison,
    graded separately). Each row carries the raw counts, the exact McNemar
    result, and a one-line ``direction`` verdict read at ``alpha``."""
    new_ix, base_ix = index_paired_rows(new_rows), index_paired_rows(base_rows)
    shared = sorted(set(new_ix) & set(base_ix))
    if codes is None:
        codes = sorted({r["reason"] for k in shared
                        for r in (new_ix[k], base_ix[k])
                        if r["reason"] != "LANDED"})
    landed = mcnemar_test([(new_ix[k]["outcome"] == "landed",
                            base_ix[k]["outcome"] == "landed") for k in shared])
    table: list[dict[str, Any]] = []
    for code in codes:
        pairs = [(new_ix[k]["reason"] != code, base_ix[k]["reason"] != code)
                 for k in shared]
        res = mcnemar_test(pairs)
        new_n = sum(1 for k in shared if new_ix[k]["reason"] == code)
        base_n = sum(1 for k in shared if base_ix[k]["reason"] == code)
        if res["exact"]["p_value"] > alpha:
            direction = "no significant change"
        elif new_n < base_n:
            direction = "improved"
        elif new_n > base_n:
            direction = "regressed"
        else:
            direction = "significantly re-paired, counts equal"
        table.append({"code": code, "baseline_count": base_n,
                      "candidate_count": new_n, "delta": new_n - base_n,
                      "direction": direction, "mcnemar": res})
    return {
        "method": "paired exact McNemar per governing reason code on the "
                  "(seed, idx) intersection of the two arms; success side = "
                  "'governing reason is not this code'",
        "n_pairs": len(shared),
        "n_new_only": len(new_ix) - len(shared), "n_base_only": len(base_ix) - len(shared),
        "alpha": float(alpha),
        "landed_pair": landed,
        "categories": table,
    }


def holm_family(table: Mapping[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Correct the per-category family of a :func:`category_pair_table`
    with the Holm-Bonferroni step-down (the E3-03 lesson: raw per-category
    p-values are not a verdict -- the E3-02 p = 0.019 headline died under
    correction). Attaches ``adjusted_p`` / ``reject`` / Holm rank to every
    category row and re-reads ``direction`` under the corrected bar (a
    marginal raw rejection that Holm does not survive is downgraded to
    "suggestive, not corrected"; the raw verdict stays in the nested
    ``mcnemar`` dict). The landed pair is reported alongside, corrected as
    its own single-member family (rank 1, i.e. unchanged)."""
    rows = [dict(row) for row in table["categories"]]
    if rows:
        adj = holm_adjust([row["mcnemar"]["exact"]["p_value"] for row in rows],
                          alpha=alpha)
        for row, a in zip(rows, adj):
            row["holm"] = a
            if row["direction"] in ("improved", "regressed") and not a["reject"]:
                row["direction"] = f"{row['direction']} (suggestive; not Holm-significant)"
            elif row["direction"] == "no significant change" and a["reject"]:
                row["direction"] = "significant after Holm"
    landed = dict(table["landed_pair"])
    single = holm_adjust([_landed_p(landed)], alpha=alpha)[0]
    return {"method": "Holm-Bonferroni step-down over the per-category family "
                      "(landed pair corrected alone)",
            "alpha": float(alpha),
            "categories": rows,
            "landed_pair": {**landed, "holm": single}}


def _landed_p(landed: Mapping[str, Any]) -> float:
    return float(landed["exact"]["p_value"])


def sensitivity_shift(candidate: Mapping[str, Any], baseline: Mapping[str, Any], *,
                      source: str = "initial_state") -> dict[str, Any]:
    """How the sensitivity of one uncertainty ``source`` moved between a
    baseline arm's ``source_sensitivity`` reading and a candidate arm's
    (E7-04's "report the initial_state sensitivity change" requirement).
    Both inputs are the sensitivity dicts as recorded (either the
    :func:`orbitopt.flightlab.recovery.stats.source_sensitivity` output or
    its JSON mirror); the reading compares the Spearman
    ``rank_correlation`` of |z| against failure, the rank *position* of the
    source in each arm's ranking, and the landed-rate gap between the
    low/high |z| tails -- the diagnostic that shows whether the fix bent
    the response curve or just moved the base rate. ``abs_rank_correlation``
    is the number E7-04 quotes: the task claims a *less* initial-state-
    sensitive plan, so the claim lives in |rho| falling, not its sign."""
    def read(arm: Mapping[str, Any]) -> dict[str, Any]:
        sources = arm["sources"]
        if source not in sources:
            raise ValueError(f"sensitivity reading has no source {source!r}: "
                             f"{sorted(sources)}")
        src = sources[source]
        ranking = list(arm["ranking"])
        rho = float(src["rank_correlation"])
        tail = src.get("tail") or {}
        gap = (float(tail["high"]["failure_rate"]) - float(tail["low"]["failure_rate"])
               if tail.get("high") and tail.get("low") else None)
        return {"rank_correlation": rho, "abs_rank_correlation": abs(rho),
                "signed_rank_correlation": (None if "signed_rank_correlation" not in src
                                            else float(src["signed_rank_correlation"])),
                "rank_position": ranking.index(source) + 1 if source in ranking else None,
                "ranking": ranking, "tail_failure_gap": gap}
    c, b = read(candidate), read(baseline)
    return {
        "source": source,
        "baseline": b, "candidate": c,
        "abs_rank_correlation_delta": c["abs_rank_correlation"]
                                      - b["abs_rank_correlation"],
        "rank_position_delta": (None if c["rank_position"] is None or b["rank_position"] is None
                                else c["rank_position"] - b["rank_position"]),
        "tail_failure_gap_delta": (None if c["tail_failure_gap"] is None
                                   or b["tail_failure_gap"] is None
                                   else c["tail_failure_gap"] - b["tail_failure_gap"]),
    }


def verdict_summary(table: Mapping[str, Any], *, targets: Sequence[str],
                    alpha: float = 0.05) -> dict[str, Any]:
    """The one-paragraph verdict the E7-04 acceptance asks for, read off a
    Holm-corrected family: every *target* category's corrected direction,
    every *non-target* category that regressed under correction (the
    "without other categories significantly rising, or explain the trade"
    clause -- an unexplained non-target regression is a failed run), and
    the landed-pair verdict. Deterministic string-free structure; the bench
    narrates it.

    The acceptance has two tiers, and this reads both: the *strict* one --
    a target category Holm-survives as improved -- and the *literal* one,
    which grades counts: ``target failure category counts fall without
    other categories significantly rising``. At E3-03's sample size a
    6-comparison family can show a real 24 -> 18 count fall (McNemar raw
    p = 0.070, 7/1 discordants) that Holm cannot certify -- so the verdict
    distinguishes ``improved`` (corrected), ``counts fell`` (the literal
    clause, raw direction only) and plain ``no change``, and always lists
    the raw non-target count increases so a trade is visible even when
    correction silences it."""
    rows = [dict(r) for r in table["categories"]]
    targets = list(targets)
    def read(code: str) -> dict[str, Any]:
        row = next((r for r in rows if r["code"] == code), None)
        if row is None:
            return {"code": code, "present": False, "direction": "not observed",
                    "baseline_count": 0, "candidate_count": 0, "delta": 0}
        return {"code": code, "present": True, "direction": row["direction"],
                "baseline_count": row["baseline_count"],
                "candidate_count": row["candidate_count"], "delta": row["delta"]}
    target_rows = [read(c) for c in targets]
    side_rows = [read(r["code"]) for r in rows if r["code"] not in targets]
    regressed = [r for r in side_rows if "regressed" in r["direction"]]
    side_increases = [r for r in side_rows if r["delta"] > 0]
    improved_targets = [r for r in target_rows
                        if r["direction"].startswith("improved")]
    decreased_targets = [r for r in target_rows if r["delta"] < 0]
    if improved_targets and not regressed:
        acceptance = ("target categories improved under correction, no "
                      "non-target regression")
    elif improved_targets:
        acceptance = "trade-off: see non_target_regressions"
    elif decreased_targets and not regressed:
        acceptance = ("target category counts fell (literal clause; not "
                      "Holm-certified at this family size), no non-target "
                      "regression under correction")
    elif decreased_targets:
        acceptance = ("trade-off: target counts fell but a non-target "
                      "category rose under correction; see "
                      "non_target_regressions")
    else:
        acceptance = "no target category count decreased"
    return {
        "method": "Holm-corrected per-category directions, split into target "
                  "and non-target readings (E7-04 acceptance)",
        "alpha": float(alpha),
        "targets": target_rows,
        "non_targets": side_rows,
        "targets_improved": [r["code"] for r in improved_targets],
        "targets_count_decreased": [r["code"] for r in decreased_targets],
        "non_target_regressions": [r["code"] for r in regressed],
        "non_target_count_increases": [r["code"] for r in side_increases],
        "acceptance": acceptance,
    }
