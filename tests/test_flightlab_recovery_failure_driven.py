"""Checks for orbitopt.flightlab.recovery.failure_driven (E7-04, task #280).

The paired comparison machinery behind the failure-driven acceptance:

* rows index by the (seed, idx) perturbation coordinate, duplicates are an
  authoring error;
* the per-category paired McNemar table counts b/c by hand on a synthetic
  2x2 and reports the (seed, idx) intersection honestly;
* the Holm family downgrades marginal raw rejections to "suggestive";
* the sensitivity reading moves rank and tail gap between two arms;
* the verdict summary splits targets from non-targets (the "or explain the
  trade" clause);
* and the machinery *reproduces the recorded E3-03 reading*: the pooled
  E3-02-vs-FL-305 table it computes from the committed report matches the
  numbers in the task background to the last code count, including the
  p = 0.019 landed comparison the correction is charged with policing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitopt.flightlab.recovery.failure_driven import (
    category_pair_table,
    holm_family,
    index_paired_rows,
    sensitivity_shift,
    verdict_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
E3_03_REPORT = REPO_ROOT / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"


def _row(seed, idx, outcome, reason, **kw):
    return {"seed": seed, "idx": idx, "outcome": outcome, "reason": reason, **kw}


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------

def test_index_is_by_the_perturbation_coordinate():
    rows = [_row(1, 0, "landed", "LANDED"), _row(1, 1, "failed", "CORRIDOR_MISSED"),
            _row(2, 0, "landed", "LANDED")]
    ix = index_paired_rows(rows)
    assert set(ix) == {(1, 0), (1, 1), (2, 0)}
    with pytest.raises(ValueError):
        index_paired_rows([_row(1, 0, "landed", "LANDED"), _row(1, 0, "failed", "X")])


# ---------------------------------------------------------------------------
# the per-category paired table, on a hand-computable 2x2
# ---------------------------------------------------------------------------

def test_category_pair_table_counts_the_discordants_by_hand():
    # code C: base lost 3 samples to C, candidate lost 2. Three discordants:
    # idx 0 and idx 2 were spared (new ok, base C -> A-only), idx 3 was
    # added (new C, base ok -> B-only).
    base = [_row(1, 0, "failed", "C"), _row(1, 1, "failed", "C"),
            _row(1, 2, "failed", "C"), _row(1, 3, "landed", "LANDED"),
            _row(1, 4, "landed", "LANDED")]
    new = [_row(1, 0, "landed", "LANDED"), _row(1, 1, "failed", "C"),
           _row(1, 2, "landed", "LANDED"), _row(1, 3, "failed", "C"),
           _row(1, 4, "landed", "LANDED")]
    table = category_pair_table(new, base, codes=["C"])
    row, = table["categories"]
    assert (row["baseline_count"], row["candidate_count"], row["delta"]) == (3, 2, -1)
    assert row["mcnemar"]["successes_a_only"] == 2   # idx 0 + idx 2: spared
    assert row["mcnemar"]["successes_b_only"] == 1   # idx 3: candidate gained
    # b=2, c=1 -> m=3, exact p = 2 * P(Bin(3,1/2) <= 1) = 1.0
    assert row["mcnemar"]["exact"]["p_value"] == pytest.approx(1.0)
    assert row["direction"] == "no significant change"
    # landed-side pair: same shape -- two spared, one gained
    assert table["landed_pair"]["successes_a_only"] == 2
    assert table["landed_pair"]["successes_b_only"] == 1


def test_category_pair_table_intersects_and_flags_asymmetry():
    base = [_row(1, 0, "failed", "C"), _row(1, 1, "landed", "LANDED")]
    new = [_row(1, 0, "failed", "C"), _row(1, 1, "landed", "LANDED"),
           _row(1, 2, "failed", "C")]
    table = category_pair_table(new, base, codes=["C"])
    assert table["n_pairs"] == 2 and table["n_new_only"] == 1 and table["n_base_only"] == 0


def test_category_pair_table_codes_default_to_governing_reasons():
    base = [_row(1, 0, "failed", "A"), _row(1, 1, "landed", "LANDED")]
    new = [_row(1, 0, "failed", "A"), _row(1, 1, "failed", "B")]
    table = category_pair_table(new, base)
    assert [r["code"] for r in table["categories"]] == ["A", "B"]


# ---------------------------------------------------------------------------
# Holm family over the table
# ---------------------------------------------------------------------------

def test_holm_family_downgrades_marginal_rejections():
    # A one-member family is Holm-equivalent to the raw test (rank 1, bar
    # alpha), and the raw mcnemar dict stays nested untouched beside it.
    # Six samples the candidate lost to REG, none in the baseline: b=0,
    # c=6, exact p = 2/64 = 0.03125 -- raw-significant regression, and
    # Holm alone cannot talk it out of.
    base = [_row(1, i, "landed", "LANDED") for i in range(6)]
    new = [_row(1, i, "failed", "REG") for i in range(6)]
    table = category_pair_table(new, base, codes=["REG"])
    fam = holm_family(table)
    reg, = fam["categories"]
    assert reg["baseline_count"] == 0 and reg["candidate_count"] == 6
    assert reg["mcnemar"]["exact"]["p_value"] == pytest.approx(0.03125)
    assert reg["holm"]["rank"] == 1 and reg["holm"]["threshold"] == 0.05
    assert reg["holm"]["reject"] and reg["direction"] == "regressed"
    assert "adjusted_p" in reg["holm"]


def test_holm_family_suggestive_downgrade_on_a_live_pair():
    # SUGGESTIVE: six samples the candidate spared (b=6, c=0), exact
    # p = 2/64 = 0.03125 -- raw-significant improvement. But the family has
    # six members, and the rank-1 step-down bar is 0.05/6 = 0.00833: Holm
    # will not carry it. The five concordant dummy categories (11/11,
    # p = 1) keep the family size honest.
    base = [_row(1, i, "failed", "SUGGESTIVE") for i in range(6)]
    new = [_row(1, i, "landed", "LANDED") for i in range(6)]
    for j, code in enumerate("ABCDE"):
        base.append(_row(1, 100 + j, "failed", code))
        new.append(_row(1, 100 + j, "failed", code))
    table = category_pair_table(new, base)
    fam = holm_family(table, alpha=0.05)
    sug = next(r for r in fam["categories"] if r["code"] == "SUGGESTIVE")
    assert sug["mcnemar"]["exact"]["p_value"] == pytest.approx(0.03125)
    assert sug["direction"] == "improved (suggestive; not Holm-significant)"
    assert sug["holm"]["rank"] == 1 and sug["holm"]["reject"] is False
    assert sug["holm"]["adjusted_p"] == pytest.approx(0.03125 * 6)
    # the landed pair is corrected as its own family: same six discordants
    # (the whole category improvement IS the landed-rate improvement here)
    assert fam["landed_pair"]["holm"]["adjusted_p"] == pytest.approx(0.03125)


# ---------------------------------------------------------------------------
# sensitivity reading
# ---------------------------------------------------------------------------

def _sens(rho, ranking, gap=None):
    tail = {"low": {"failure_rate": 0.5 - gap / 2, "n": 50},
            "high": {"failure_rate": 0.5 + gap / 2, "n": 50}} if gap is not None else None
    return {"method": "x", "ranking": ranking,
            "sources": {"initial_state": {"rank_correlation": rho,
                                          "signed_rank_correlation": rho,
                                          "components": 6, "state": "assessed",
                                          "tail": tail}}}


def test_sensitivity_shift_reports_rank_and_magnitude_move():
    shift = sensitivity_shift(_sens(0.25, ["drag", "initial_state", "thrust"], 0.10),
                              _sens(0.47, ["initial_state", "drag", "thrust"], 0.34))
    assert shift["baseline"]["rank_position"] == 1
    assert shift["candidate"]["rank_position"] == 2
    assert shift["rank_position_delta"] == 1
    assert shift["abs_rank_correlation_delta"] == pytest.approx(0.25 - 0.47)
    assert shift["tail_failure_gap_delta"] == pytest.approx(0.10 - 0.34)
    with pytest.raises(ValueError):
        sensitivity_shift({"method": "x", "ranking": ["wind"], "sources": {}},
                          _sens(0.1, ["initial_state"]))


# ---------------------------------------------------------------------------
# verdict summary
# ---------------------------------------------------------------------------

def test_verdict_summary_splits_targets_from_trades():
    fake = {"categories": [
        {"code": "CORRIDOR_MISSED", "direction": "improved",
         "baseline_count": 5, "candidate_count": 2, "delta": -3},
        {"code": "LANDING_POSITION_OUT_OF_TOLERANCE", "direction": "regressed",
         "baseline_count": 40, "candidate_count": 46, "delta": 6},
        {"code": "INSUFFICIENT_PROPELLANT_RESERVE", "direction": "no significant change",
         "baseline_count": 30, "candidate_count": 28, "delta": -2}]}
    clean = verdict_summary(fake, targets=["CORRIDOR_MISSED"])
    assert clean["targets_improved"] == ["CORRIDOR_MISSED"]
    assert clean["targets_count_decreased"] == ["CORRIDOR_MISSED"]
    assert clean["non_target_regressions"] == ["LANDING_POSITION_OUT_OF_TOLERANCE"]
    assert clean["non_target_count_increases"] == ["LANDING_POSITION_OUT_OF_TOLERANCE"]
    assert clean["acceptance"].startswith("trade-off")
    nofix = verdict_summary(fake, targets=["LANDING_POSITION_OUT_OF_TOLERANCE"])
    assert nofix["targets_improved"] == []
    assert nofix["targets_count_decreased"] == []
    assert nofix["acceptance"].startswith("no target category count decreased")
    missing = verdict_summary(fake, targets=["SOMETHING_UNOBSERVED"])
    assert missing["targets"][0]["present"] is False


# ---------------------------------------------------------------------------
# reproduction: the machinery reads the recorded E3-03 report exactly as
# the task background states it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arms", [("fl305_baseline", "e3_02_joint_chance")])
def test_reproduces_the_recorded_e3_03_reading(arms):
    d = json.loads(E3_03_REPORT.read_text(encoding="utf-8"))
    base_rows = d["runs"][arms[0]]["paired_rows"]
    new_rows = d["runs"][arms[1]]["paired_rows"]
    raw = category_pair_table(new_rows, base_rows)
    table = holm_family(raw)
    by = {r["code"]: r for r in table["categories"]}
    assert by["CORRIDOR_MISSED"]["baseline_count"] == 5
    assert by["CORRIDOR_MISSED"]["candidate_count"] == 22
    assert by["CORRIDOR_MISSED"]["direction"] == "regressed"
    assert by["LANDING_LATERAL_SPEED_EXCEEDED"]["baseline_count"] == 24
    assert by["LANDING_LATERAL_SPEED_EXCEEDED"]["candidate_count"] == 32
    assert by["LANDING_POSITION_OUT_OF_TOLERANCE"]["baseline_count"] == 42
    assert by["LANDING_POSITION_OUT_OF_TOLERANCE"]["candidate_count"] == 34
    assert by["INSUFFICIENT_PROPELLANT_RESERVE"]["baseline_count"] == 38
    assert by["INSUFFICIENT_PROPELLANT_RESERVE"]["candidate_count"] == 12
    assert raw["n_pairs"] == 168
    # the raw p = 0.019 of the task background: the landed-pair comparison,
    # significant alone, and the family's *category* story is the regression
    landed_p = table["landed_pair"]["exact"]["p_value"]
    assert landed_p == pytest.approx(0.0192108154296875)
    verdict = verdict_summary(table, targets=["CORRIDOR_MISSED",
                                              "LANDING_LATERAL_SPEED_EXCEEDED"])
    assert verdict["targets_improved"] == []  # the honest "before" picture
    assert "CORRIDOR_MISSED" in verdict["non_target_regressions"] \
        or by["CORRIDOR_MISSED"]["direction"] == "regressed"
    shift = sensitivity_shift(d["runs"]["e3_02_joint_chance"]["pooled"]["sensitivity"],
                              d["runs"]["fl305_baseline"]["pooled"]["sensitivity"])
    assert shift["baseline"]["rank_correlation"] == pytest.approx(0.40340607335390777)
    assert shift["candidate"]["rank_correlation"] == pytest.approx(0.47386801770492254)
    assert shift["abs_rank_correlation_delta"] > 0.0  # the regression to undo


def test_verdict_grades_the_recorded_compensation_pair_on_both_tiers():
    # the guidance fix's core, already recorded by E3-03 on the identical
    # grid: the lateral count falls 24 -> 18 (7/1 discordants, raw p 0.070)
    # -- the literal acceptance clause is met, Holm is not; the verdict
    # must say exactly that, not hide either tier
    d = json.loads(E3_03_REPORT.read_text(encoding="utf-8"))
    table = holm_family(category_pair_table(
        d["runs"]["e3_01_declared_chain"]["paired_rows"],
        d["runs"]["fl305_baseline"]["paired_rows"]))
    verdict = verdict_summary(table, targets=["CORRIDOR_MISSED",
                                              "LANDING_LATERAL_SPEED_EXCEEDED"])
    assert verdict["targets_improved"] == []  # Holm cannot certify it
    assert verdict["targets_count_decreased"] == ["LANDING_LATERAL_SPEED_EXCEEDED"]
    assert verdict["non_target_regressions"] == []  # nothing rose under correction
    assert "LANDING_PROPELLANT_RESERVE_LOW" in verdict["non_target_count_increases"]
    assert verdict["acceptance"].startswith("target category counts fell")
